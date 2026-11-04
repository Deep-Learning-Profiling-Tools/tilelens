"""The compiled sanitizer's per-launch checks over an AccessGraph.

Given a kernel's ``AccessGraph`` (``tilelens.ir.ttir_reader``) and one
launch's ``LaunchBinding``, each access becomes a few Z3 queries over the
launch's free variables (program ids, the access's lane positions, the
loop's iteration index), with its scalar arguments and grid substituted as
constants:

* ``out-of-bounds``: the access executes (its loop runs, its path and
  mask hold) at an element offset outside the view footprint of its
  tensor, the eager sanitizer's legal set: the offsets
  ``sum(i_d * stride_d)`` with ``0 <= i_d < size_d``, a stride-0 dim
  counting once, relative to the view's ``data_ptr``;
* ``integer-overflow``: a width obligation the access depends on can
  fail, so the IR's fixed-width arithmetic is not the unbounded reading;
* ``division-by-zero``: a ``//`` or ``%`` the access reaches (from its
  offset, mask or path, or from its loop's bounds) can divide by zero.

The unbounded reading of a term is exact only where the obligations below
it hold and its divisors are non-zero. The reader's roles order that
dependence: the loop's bounds, then its increment (only when the loop
runs), then the access's path, its mask (under the path) and its offset
(under path and mask). Each role is discharged by ONE query, "some
obligation or division of this role fails", under the safety of the roles
before it and never under its own (a wrap can decide a sibling divisor, and
the reverse); the out-of-bounds query assumes every role's safety. A
wrapped offset is therefore an integer overflow, never an out-of-bounds
witness that only the unbounded reading has.

A value in an arm of a ``Select`` (``tl.where``) matters only where the
select picks that arm, so the width obligations of the access's path, mask
and offset are discharged under the conditions of the arms their terms are
read through: a wrap the select discards is harmless (``arith`` ops wrap,
with defined results). Divisions are not: the IR computes both arms, and a
zero divisor (or ``INT_MIN // -1``) in the discarded one is still
undefined, so they are checked wherever the access evaluates them. (The
loop's own obligations are always checked unguarded.)

Lanes: every tensor one access's terms combine has the access's shape
(TTIR broadcasts explicitly, and only size-1 dims), so all aranges along
one dim index the same position there: an arange's value is its start plus
the lane position of its dim (one position per dim and extent; an extent-1
arange broadcast along a longer dim stays at position 0).

Findings are exact: a SAT model is a concrete launch state, and the op a
role reports is the innermost one failing in the model (so one wrap is not
reported at the ops computed from it). A model that may be unreachable is
withheld and the access *abstains* instead: a
branch condition the reader could not model (``guarded``) or one reading an
atomic observation, a mask it dropped (``mask_dropped``) or one reading an
observation. Z3's ``unknown`` abstains too, each query on a fresh
Solver with its own timeout, so whether a hard query ends in
``solver-unknown`` can depend on the machine's load. An access the model
cannot check at all is refused (an unbound argument, a non-positive loop
step, an address that depends on an atomic observation). Abstentions never
suppress another access's findings: a ``CheckResult`` carries both, and
only one with neither is a proof for the launch.

Each check builds its Z3 terms in a Z3 context of its own, so checks may
run on several host threads at once (one Z3 context is not thread-safe). A
Ctrl+C that Z3 caught during a query is raised as ``KeyboardInterrupt``,
never taken for an unknown.

Terms can be deeper than Python's recursion limit and their generated
``==`` / ``hash`` recurse, so lowering is iterative and memoized by term
identity.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

from z3 import (
    And,
    ArithRef,
    BoolRef,
    BoolVal,
    Context,
    Exists,
    If,
    Implies,
    Int,
    IntVal,
    ModelRef,
    Or,
    Solver,
    Sum,
    is_bool,
    is_false,
    is_int_value,
    sat,
    simplify,
    unknown,
    unsat,
)
from z3 import Not as Z3Not

from ....ir.launch import LaunchBinding, TensorFacts
from ....ir.ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    Bin,
    BoolBin,
    Cmp,
    Const,
    DataDep,
    IntCast,
    IterArgOffset,
    LoopInfo,
    LoopVar,
    Not,
    NumPrograms,
    Observed,
    Param,
    Pid,
    Select,
    WidthObligation,
    mentions_observed,
    width_obligations,
)
from ....ir.verdict import Refusal, SourceLocation
from ..data import CompiledFindingKind


class SanitizerKind(str, Enum):
    """Why the compiled sanitizer left (part of) a launch undecided. The
    sanitizer's own kinds, apart from the reader's ``TTIRKind``."""

    # A kernel argument the access (or its loop) reads has no launch
    # binding, or the launch grid is unknown (never left unconstrained).
    MISSING_BINDING = "missing-binding"
    # The loop step is not positive (or may not be) for this launch.
    NON_POSITIVE_STEP = "non-positive-step"
    # Z3 answered unknown (a timeout included).
    SOLVER_UNKNOWN = "solver-unknown"
    # A witness sits under a branch condition that is not modeled (loaded
    # data, or an atomic observation read as a free value).
    UNMODELABLE_CONDITION = "unmodelable-condition"
    # A witness sits behind a mask that is not modeled (dropped as loaded
    # data, or reading an atomic observation).
    DATA_DEPENDENT_MASK = "data-dependent-mask"
    # The address depends on the value an atomic observed.
    OBSERVATION_IN_ADDRESS = "observation-in-address"
    # A term the reader marked unmodelable (DataDep) reached a lowered
    # address, mask, path or loop bound; the reader never lets it.
    UNMODELED_VALUE = "unmodeled-value"
    # The client's own (compiled/client.py): the launch compiled nothing to
    # read (no JITFunction, or none of its configs was delivered); a config
    # failed to compile for the IR target with an error that need not hold
    # for another target, or its call does not bind the kernel's parameters,
    # or no config compiled at all; a compiled kernel holds no TTIR; the host
    # compile could not run for a config; the analysis itself failed (a bug,
    # contained).
    NO_COMPILED_KERNEL = "no-compiled-kernel"
    COMPILE_FAILED = "compile-failed"
    NO_TTIR = "no-ttir"
    HOST_COMPILE_UNAVAILABLE = "host-compile-unavailable"
    INTERNAL_ERROR = "internal-error"

    def __str__(self) -> str:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)


@dataclass(frozen=True)
class Finding:
    """One exact finding: its witness is a concrete state of the launch."""

    kind: CompiledFindingKind
    access_index: int  # into graph.accesses
    access_kind: str  # "load" | "store" | "atomic_rmw" | "atomic_cas"
    base_param: str
    # The op the finding is about: the access (out-of-bounds), the op or
    # cast whose width does not hold (integer-overflow), the division
    # (division-by-zero); the access's when that op's site is unknown.
    line_no: int | None
    loc: SourceLocation | None
    # The free variables by name: pid_<axis>; arange_<start>_<end>, the
    # value of that tl.arange at the witness lane (with _d<dim> for a dim
    # of a 2-D or wider tile); iter_loop, the loop's 0-based iteration. An
    # integer-overflow adds "value", the out-of-range value.
    witness: Mapping[str, int]
    # out-of-bounds only: the element offset (in the accessed pointer's
    # elements) from the view's data_ptr, and its byte address.
    violation_offset: int | None = None
    violation_address: int | None = None
    detail: str = ""

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "witness", dict(self.witness))


@dataclass(frozen=True)
class CheckResult:
    """What ``check_graph`` decided for one launch: every exact finding,
    and every undecided (access index, kind), whose first one is
    ``refusal``. Only a result with neither proves the launch in bounds."""

    findings: tuple[Finding, ...]
    refusal: Refusal | None
    abstained: tuple[tuple[int, SanitizerKind], ...]

    __hash__ = None  # type: ignore[assignment]


def check_graph(
    graph: AccessGraph, binding: LaunchBinding, *, timeout_ms: int = 10_000
) -> CheckResult:
    """Check every access of ``graph`` under ``binding`` (see the module
    docstring); ``timeout_ms`` (a positive int) bounds each Z3 query.

    A limit of the model is returned as an abstention, never raised; an
    exception means a bug (e.g. a graph that breaks the reader's
    invariants), or is the KeyboardInterrupt of a Ctrl+C."""
    _check_timeout(timeout_ms)
    return _Check(graph, binding, timeout_ms).run()


def launch_key(binding: LaunchBinding) -> Hashable:
    """Everything ``check_graph`` reads of ``binding`` but the tensors'
    addresses: for one graph (and timeout), bindings with one key get the
    same CheckResult up to the findings' ``violation_address`` (see
    ``readdressed``); no message holds an address."""
    return (
        tuple(sorted(binding.params.items())),
        binding.grid,
        tuple(
            sorted(
                (name, (f.elem_size, f.numel, f.shape, f.strides, f.contiguous))
                for name, f in binding.tensors.items()
            )
        ),
        binding.error,
    )


def readdressed(
    result: CheckResult, graph: AccessGraph, binding: LaunchBinding
) -> CheckResult:
    """``result``, the check of ``graph`` under a binding with the
    ``launch_key`` of ``binding``, with its findings' byte addresses in
    ``binding``'s tensors."""
    findings = tuple(
        finding
        if finding.violation_offset is None
        else replace(
            finding,
            violation_address=binding.tensors[finding.base_param].data_ptr
            + finding.violation_offset
            * _pointee_bytes(graph.accesses[finding.access_index].elem_bits),
        )
        for finding in result.findings
    )
    return CheckResult(findings, result.refusal, result.abstained)


def describe_site(line_no: int | None, loc: Any) -> str:
    """Where a refused or reported op is: ``file:line`` of its user-source
    location, else its TTIR line."""
    if loc is not None:
        return f"{loc.file}:{loc.line}"
    return f"TTIR line {line_no}" if line_no is not None else "the kernel"


def _check_timeout(timeout_ms: Any) -> None:
    # Z3 reads a timeout of 0 or below as none at all.
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or timeout_ms <= 0
    ):
        raise ValueError(f"timeout_ms must be a positive int, not {timeout_ms!r}")


# ─────────────────────────── lowering ───────────────────────────


class _Refused(Exception):
    """An access (or the loop) the model cannot check: becomes an
    abstention; never escapes check_graph."""

    def __init__(
        self,
        kind: SanitizerKind,
        message: str,
        line_no: int | None = None,
        loc: Any = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.line_no = line_no
        self.loc = loc

    def at(self, line_no: int | None, loc: Any) -> _Refused:
        if self.line_no is None and self.loc is None:
            self.line_no, self.loc = line_no, loc
        return self


def _as_bool(e: Any) -> BoolRef:
    """An i1 value in a boolean position: i1 constants (e.g. the dense<true>
    mask of an unmasked atomic, Const(1)) lower to Int."""
    return e if is_bool(e) else e != 0


def _as_int(e: Any) -> ArithRef:
    """An i1 value in an integer position (an extui of a compare, ...)."""
    return If(e, IntVal(1, e.ctx), IntVal(0, e.ctx)) if is_bool(e) else e


def _trunc_div(a: ArithRef, b: ArithRef) -> ArithRef:
    """arith.divsi rounds toward zero, but Z3's Int ``/`` is Euclidean
    (floor for a positive divisor): they disagree on negative dividends.
    Divide the magnitudes, where the two agree, and re-apply the sign."""
    aa = If(a >= 0, a, -a)
    ab = If(b >= 0, b, -b)
    q = aa / ab
    return If((a >= 0) == (b >= 0), q, -q)


def _bin(op: str, a: ArithRef, b: ArithRef) -> ArithRef:
    # The unsigned ops read their operands unsigned; their width
    # obligations make both non-negative, where they equal the signed ones.
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op in ("//", "u//"):
        return _trunc_div(a, b)
    if op in ("%", "u%"):
        # arith.remsi: the remainder carries the dividend's sign
        return a - b * _trunc_div(a, b)
    if op in ("min", "umin"):
        return If(a <= b, a, b)
    if op in ("max", "umax"):
        return If(a >= b, a, b)
    raise ValueError(f"unknown integer op {op!r}")


# Unsigned predicates read their operands unsigned; their width obligations
# make both non-negative, where they equal the signed ones.
_SIGNED_PRED = {"ult": "slt", "ule": "sle", "ugt": "sgt", "uge": "sge"}


def _cmp(pred: str, a: Any, b: Any) -> BoolRef:
    if is_bool(a) or is_bool(b):  # i1 operands, as 0/1
        a, b = _as_int(a), _as_int(b)
    table = {
        "slt": a < b, "sle": a <= b, "sgt": a > b,
        "sge": a >= b, "eq": a == b, "ne": a != b,
    }  # fmt: skip
    try:
        return table[_SIGNED_PRED.get(pred, pred)]
    except KeyError:
        raise ValueError(f"unknown cmpi predicate {pred!r}") from None


def _kids(t: object, graph: AccessGraph) -> tuple:
    """The terms ``t`` is computed from; a loop-carried pointer's offset
    is computed from its IterArgInfo's ``offset0`` and ``delta``."""
    if isinstance(t, (Bin, Cmp, BoolBin)):
        return (t.a, t.b)
    if isinstance(t, Select):
        return (t.cond, t.t, t.f)
    if isinstance(t, Not):
        return (t.a,)
    if isinstance(t, IntCast):
        return (t.x,)
    if isinstance(t, IterArgOffset):
        info = graph.iter_args[t.arg_id]
        return (info.offset0, info.delta)
    return ()


def _walk(
    roots: Iterable[object], graph: AccessGraph, seen: set[int]
) -> Iterator[object]:
    """The nodes reachable from ``roots`` in pre-order, skipping (and
    adding to) ``seen`` by identity. Iterative."""
    stack = [r for r in reversed(list(roots)) if r is not None]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        yield t
        stack.extend(reversed(_kids(t, graph)))


_DIVISIONS = frozenset({"//", "%", "u//", "u%"})


def _divisions(
    roots: Iterable[object], graph: AccessGraph, seen: set[int]
) -> list[Bin]:
    return [
        n
        for n in _walk(roots, graph, seen)
        if isinstance(n, Bin) and n.op in _DIVISIONS
    ]


def _signed(value: int, bits: int) -> int:
    """The IR's signed reading of an argument of ``bits`` (an i1 stays 0/1,
    the boolean model)."""
    if bits <= 1:
        return value
    half = 1 << (bits - 1)
    return (value + half) % (1 << bits) - half


def _lane_name(t: Arange) -> str:
    # Named like the eager engine's arange variables; a tile's dim added
    # (a 1-D range has dim -1).
    name = f"arange_{t.start}_{t.end}"
    return name if t.dim < 0 else f"{name}_d{t.dim}"


class _Env:
    """The Z3 variables of one family of queries (one access, or the
    loop's own checks), their range premises, and the lowering memo."""

    def __init__(
        self,
        graph: AccessGraph,
        binding: LaunchBinding,
        grid: tuple[int, int, int],
        ctx: Context,
    ) -> None:
        self.graph = graph
        self.binding = binding
        self.grid = grid
        self.ctx = ctx
        # Range premises of the variables created so far.
        self.premises: list[BoolRef] = []
        self.pids = tuple(Int(f"pid_{axis}", ctx) for axis in range(3))
        for pid, size in zip(self.pids, grid):
            self.premises += [pid >= 0, pid < size]
        # (dim, extent) -> the lane's position along that dim (see lane).
        self.positions: dict[tuple[int, int], ArithRef] = {}
        # witness name -> the arange's value at the lane
        self.lanes: dict[str, ArithRef] = {}
        self.iteration: ArithRef | None = None
        self._observed: dict[int, ArithRef] = {}
        # id(term) -> (term, lowered): holding the term keeps its id unique.
        self._memo: dict[int, tuple[object, Any]] = {}

    # ── leaves ──

    def param(self, name: str) -> int:
        try:
            value = self.binding.params[name]
        except KeyError:
            raise _Refused(
                SanitizerKind.MISSING_BINDING,
                f"scalar argument {name!r} has no launch binding"
                + _binding_error(self.binding),
            ) from None
        arg = self.graph.arg(name)
        return _signed(value, arg.int_bits if arg is not None else 0)

    def lane(self, t: Arange) -> ArithRef:
        """``t``'s value at the access's lane: its start plus the lane's
        position along its dim, one position per dim and extent (see the
        module docstring)."""
        extent = t.end - t.start
        key = (t.dim, extent)
        pos = self.positions.get(key)
        if pos is None:
            pos = Int(f"lane_d{t.dim}_n{extent}", self.ctx)
            self.positions[key] = pos
            self.premises += [pos >= 0, pos < extent]
        value = pos + t.start
        self.lanes.setdefault(_lane_name(t), value)
        return value

    def observed(self, index: int) -> ArithRef:
        """An atomic observation: a free value of the atomic's width."""
        v = self._observed.get(index)
        if v is None:
            v = Int(f"observed_{index}", self.ctx)
            self._observed[index] = v
            bits = self.graph.accesses[index].elem_bits
            if bits > 1:
                half = 1 << (bits - 1)
                self.premises += [v >= -half, v < half]
        return v

    def bounds(self) -> tuple[ArithRef, ArithRef, ArithRef]:
        loop = self._loop()
        return self.value(loop.lower), self.value(loop.upper), self.value(loop.step)

    def loop_iteration(self) -> ArithRef:
        """The loop's 0-based iteration index ``k``, with its premise ``k >=
        0 and lower + k*step < upper``: only iterations that run, and none
        when the launch's trip count is zero."""
        if self.iteration is None:
            loop = self._loop()
            lower, upper, step = self.bounds()
            k = Int(f"iter_{loop.loop_ssa.strip('%')}", self.ctx)
            self.premises += [k >= 0, lower + k * step < upper]
            self.iteration = k
        return self.iteration

    def _loop(self) -> LoopInfo:
        loop = self.graph.loop
        if loop is None:
            raise ValueError(
                f"kernel {self.graph.kernel_name!r}: a loop term without a loop"
            )
        return loop

    # ── terms ──

    def value(self, term: object) -> ArithRef:
        return _as_int(self.lower(term))

    def cond(self, term: object) -> BoolRef:
        return _as_bool(self.lower(term))

    def lower(self, root: object) -> Any:
        """``root`` as a Z3 expression (Int, or Bool for a compare)."""
        memo = self._memo
        stack: list[tuple[object, bool]] = [(root, False)]
        while stack:
            t, ready = stack.pop()
            if id(t) in memo:
                continue
            kids = _kids(t, self.graph)
            if kids and not ready:
                stack.append((t, True))
                stack.extend((k, False) for k in reversed(kids) if id(k) not in memo)
                continue
            memo[id(t)] = (t, self._apply(t, [memo[id(k)][1] for k in kids]))
        return memo[id(root)][1]

    def _apply(self, t: object, kids: Sequence[Any]) -> Any:
        if isinstance(t, Const):
            return IntVal(t.value, self.ctx)
        if isinstance(t, Param):
            return IntVal(self.param(t.name), self.ctx)
        if isinstance(t, Pid):
            return self.pids[t.axis]
        if isinstance(t, NumPrograms):
            return IntVal(self.grid[t.axis], self.ctx)
        if isinstance(t, Arange):
            return self.lane(t)
        if isinstance(t, LoopVar):
            lower, _upper, step = self.bounds()
            return lower + self.loop_iteration() * step
        if isinstance(t, IterArgOffset):
            return _as_int(kids[0]) + self.loop_iteration() * _as_int(kids[1])
        if isinstance(t, Bin):
            return _bin(t.op, _as_int(kids[0]), _as_int(kids[1]))
        if isinstance(t, Cmp):
            return _cmp(t.pred, kids[0], kids[1])
        if isinstance(t, BoolBin):
            a, b = _as_bool(kids[0]), _as_bool(kids[1])
            return And(a, b) if t.op == "and" else Or(a, b)
        if isinstance(t, Select):
            a, b = kids[1], kids[2]
            if is_bool(a) != is_bool(b):
                a, b = _as_int(a), _as_int(b)
            return If(_as_bool(kids[0]), a, b)
        if isinstance(t, Not):
            return Z3Not(_as_bool(kids[0]))
        if isinstance(t, IntCast):
            # Its value is the operand's while its width obligation holds.
            return _as_int(kids[0])
        if isinstance(t, Observed):
            return self.observed(t.access_index)
        if isinstance(t, DataDep):
            raise _Refused(
                SanitizerKind.UNMODELED_VALUE, f"an unmodeled value ({t.why})"
            )
        raise TypeError(f"unknown term {type(t).__name__}")


class _Dag:
    """The nodes one family of queries reads from ``roots``: each one's
    rank (children before parents) and, when built with the family's env
    (whose memo holds the lowered roots), its guard: the condition under
    which its value reaches a root through the arms of Selects (a node
    without one reaches a root directly). Iterative, keyed by identity."""

    def __init__(
        self, roots: Sequence[object], graph: AccessGraph, env: _Env | None = None
    ) -> None:
        self.graph = graph
        self.order: dict[int, int] = {}
        nodes: list[object] = []
        stack = [(r, False) for r in reversed(roots) if r is not None]
        while stack:
            t, done = stack.pop()
            if id(t) in self.order:
                continue
            if done:
                self.order[id(t)] = len(nodes)
                nodes.append(t)
                continue
            stack.append((t, True))
            stack.extend(
                (k, False) for k in reversed(_kids(t, graph)) if id(k) not in self.order
            )
        self.guards: dict[int, BoolRef] = {}
        if env is None:
            return
        direct = {id(r) for r in roots if r is not None}
        shares: dict[int, list[BoolRef]] = {}
        for t in reversed(nodes):  # every parent before its children
            parts = shares.pop(id(t), [])
            guard = None
            if id(t) not in direct:
                guard = parts[0] if len(parts) == 1 else Or(parts)
                self.guards[id(t)] = guard
            edges: list[tuple[object, BoolRef | None]]
            if isinstance(t, Select):
                c = env.cond(t.cond)
                edges = [(t.cond, guard)]
                for arm, taken in ((t.t, c), (t.f, Z3Not(c))):
                    edges.append((arm, taken if guard is None else And(guard, taken)))
            else:
                edges = [(k, guard) for k in _kids(t, graph)]
            for kid, kid_guard in edges:
                if kid_guard is None:
                    direct.add(id(kid))
                elif id(kid) not in direct:
                    shares.setdefault(id(kid), []).append(kid_guard)

    def rank(self, term: object) -> float:
        """Children first, and a Select's condition before its arms; a term
        the walk does not hold (an obligation's own, e.g. a remainder's
        quotient) right after its children."""
        index = self.order.get(id(term))
        if index is not None:
            return index
        kids = [
            self.order[id(k)] for k in _kids(term, self.graph) if id(k) in self.order
        ]
        return max(kids) + 0.5 if kids else len(self.order)


# ─────────────────────────── view footprint ───────────────────────────


def _in_view(e: ArithRef, facts: TensorFacts) -> BoolRef:
    """``e`` (an element index from the view's data_ptr) is the offset of one
    of the view's elements: ``sum(i_d * stride_d)``, ``0 <= i_d < size_d``."""
    if facts.numel == 0:
        return BoolVal(False, e.ctx)
    if facts.contiguous:
        return And(e >= 0, e < facts.numel)
    # Size-1 dims add nothing and stride-0 dims alias one element: drop both.
    # (stride, size), largest stride first
    dims = sorted(
        ((st, sz) for sz, st in zip(facts.shape, facts.strides) if sz != 1 and st != 0),
        reverse=True,
    )
    if any(st < 0 for st, _ in dims):
        return _any_index(e, dims)
    extent = 1  # dense (a permutation of a contiguous layout): an interval
    for st, sz in reversed(dims):
        if st != extent:
            break
        extent *= sz
    else:
        return And(e >= 0, e < extent)
    # Without overlap (each stride exceeds the reach of the smaller ones),
    # the indices are the greedy quotients: quantifier-free and exact.
    reach = 0
    for st, sz in reversed(dims):
        if st <= reach:
            return _any_index(e, dims)
        reach += (sz - 1) * st
    conds = [e >= 0]
    rest = e
    for st, sz in dims:
        conds.append(rest / st < sz)
        rest = rest % st
    conds.append(rest == 0)
    return And(conds)


def _any_index(e: ArithRef, dims: list[tuple[int, int]]) -> BoolRef:
    """The stride equation itself, for overlapping or negative strides (as
    a quantifier, Z3 may answer unknown)."""
    idx = [Int(f"view_index_{d}", e.ctx) for d in range(len(dims))]
    body = [i >= 0 for i in idx] + [i < sz for i, (_, sz) in zip(idx, dims)]
    body.append(e == Sum([i * st for i, (st, _) in zip(idx, dims)]))
    return Exists(idx, And(body))


def _legal(offset: ArithRef, facts: TensorFacts, width: int) -> BoolRef:
    """An access of ``width`` bytes at element ``offset`` touches only bytes
    of the view's elements."""
    if width == facts.elem_size:
        return _in_view(offset, facts)
    # A pointer whose element width differs from the tensor's (a
    # reinterpreting view): every byte it touches must be in an element.
    first = offset * width
    return And(
        [
            And(first + j >= 0, _in_view((first + j) / facts.elem_size, facts))
            for j in range(width)
        ]
    )


def _footprint(facts: TensorFacts) -> str:
    if facts.numel == 0:
        return "the empty tensor"
    if facts.contiguous:
        return f"the tensor's {facts.numel} elements"
    return f"the view of shape {facts.shape} and strides {facts.strides}"


def _unusable(facts: TensorFacts) -> str | None:
    if facts.elem_size <= 0 or facts.numel < 0:
        return f"element size {facts.elem_size}, numel {facts.numel}"
    if len(facts.shape) != len(facts.strides):
        return f"shape {facts.shape} with strides {facts.strides}"
    return None


# ─────────────────────────── the checks ───────────────────────────


def _fits(value: ArithRef, bits: int, signed: bool) -> BoolRef:
    if signed:
        half = 1 << (bits - 1)
        return And(value >= -half, value < half)
    return And(value >= 0, value < (1 << bits))


def _undefined_when_wide(ob: WidthObligation) -> bool:
    """A signed quotient that does not fit (``INT_MIN // -1``, a
    remainder's quotient included) is undefined in the IR, not a wrap: like
    a zero divisor, it counts in either arm of a Select."""
    return isinstance(ob.term, Bin) and ob.term.op == "//"


def _is_increment(ob: WidthObligation, loop: LoopInfo, bound_ids: set[int]) -> bool:
    """The loop's increment obligation (``upper - 1 + step``, which
    width_obligations builds from the bound nodes themselves): the one loop
    obligation that holds only when the loop runs."""
    t = ob.term
    return (
        id(t) not in bound_ids
        and isinstance(t, Bin)
        and t.op == "+"
        and t.b is loop.step
        and isinstance(t.a, Bin)
        and t.a.op == "-"
        and t.a.a is loop.upper
    )


def _location(loc: Any) -> SourceLocation | None:
    if loc is None or isinstance(loc, SourceLocation):
        return loc
    return SourceLocation(loc.file, loc.line, getattr(loc, "col", None))


def _binding_error(binding: LaunchBinding) -> str:
    return f" (unreadable: {binding.error})" if binding.error else ""


def _pointee_bytes(elem_bits: int) -> int:
    return max(1, (elem_bits + 7) // 8)


_WITHHELD = {
    SanitizerKind.UNMODELABLE_CONDITION: "under a branch condition that is not "
    "modeled (loaded data, or an atomic observation read as a free value)",
    SanitizerKind.DATA_DEPENDENT_MASK: "behind a mask that is not modeled "
    "(loaded data, or an atomic observation read as a free value)",
}

# What Z3 answers for a query its Ctrl+C handler cancelled (Python's own
# handler does not run during the query).
_INTERRUPTED = "interrupted from keyboard"


@dataclass(frozen=True, eq=False)
class _Condition:
    """One width obligation or division of a role: ``safe`` holds where it
    does not fail; the lowest ``rank`` is the innermost failure."""

    kind: Literal["integer-overflow", "division-by-zero"]
    site: WidthObligation | Bin
    safe: BoolRef
    rank: tuple


class _Check:
    def __init__(
        self, graph: AccessGraph, binding: LaunchBinding, timeout_ms: int
    ) -> None:
        self.graph = graph
        self.binding = binding
        self.timeout_ms = timeout_ms
        # This check's own Z3 context (see the module docstring).
        self.ctx = Context()
        self.findings: list[Finding] = []
        self.abstained: list[tuple[int, SanitizerKind]] = []
        self.refusal: Refusal | None = None
        # (finding kind, site) already reported: one finding per op site.
        self.reported: set[tuple[str, object]] = set()
        self._alive: list[object] = []  # reported sites keyed by id
        # The loop's obligations and divisions, shared by its accesses.
        self.loop_obs: tuple[WidthObligation, ...] = ()
        self.loop_divs: tuple[Bin, ...] = ()
        self.loop_ids: set[int] = set()

    def run(self) -> CheckResult:
        graph, grid = self.graph, self.binding.grid
        if grid is None:
            for i, access in enumerate(graph.accesses):
                self.abstain(
                    i,
                    _Refused(
                        SanitizerKind.MISSING_BINDING,
                        "the launch grid is unknown" + _binding_error(self.binding),
                        access.line_no,
                        access.loc,
                    ),
                )
            return self.result()
        in_loop = [i for i, a in enumerate(graph.accesses) if a.in_loop]
        loop_refusal = None
        if graph.loop is not None and in_loop:
            loop_refusal = self.check_loop(in_loop[0], grid)
        for i, access in enumerate(graph.accesses):
            if access.in_loop and loop_refusal is not None:
                self.abstain(i, loop_refusal)
                continue
            try:
                self.check_access(i, access, grid)
            except _Refused as r:
                self.abstain(i, r.at(access.line_no, access.loc))
        return self.result()

    def result(self) -> CheckResult:
        return CheckResult(tuple(self.findings), self.refusal, tuple(self.abstained))

    # ── the loop, once for all of its accesses ──

    def check_loop(self, first: int, grid: tuple[int, int, int]) -> _Refused | None:
        """Check the loop's step, bounds and increment; findings go to the
        loop's ``first`` access. A refusal refuses every access in the loop.
        No Select guards here: the loop's accesses assume its obligations
        unguarded."""
        graph = self.graph
        loop = graph.loop
        assert loop is not None
        env = _Env(graph, self.binding, grid, self.ctx)
        try:
            _lower, _upper, step = env.bounds()
            refusal = self.step_refusal(env, step)
        except _Refused as r:
            refusal = r
        if refusal is not None:
            return refusal.at(loop.line_no, loop.loc)
        roots = (loop.lower, loop.upper, loop.step)
        dag = _Dag(roots, graph)
        self.loop_ids = {id(n) for n in _walk(roots, graph, set())}
        self.loop_divs = tuple(_divisions(roots, graph, set()))
        self.loop_obs = tuple(
            ob
            for ob in width_obligations(graph, graph.accesses[first])
            if ob.role == "loop"
        )
        increment = [
            ob for ob in self.loop_obs if _is_increment(ob, loop, self.loop_ids)
        ]
        bounds = [
            ob for ob in self.loop_obs if not _is_increment(ob, loop, self.loop_ids)
        ]
        # The bounds are computed whether or not the loop runs; the
        # increment only matters when it does.
        assumed = self.role(env, first, bounds, self.loop_divs, [], None, dag)
        env.loop_iteration()
        self.role(env, first, increment, (), assumed, None, dag)
        return None

    def step_refusal(self, env: _Env, step: ArithRef) -> _Refused | None:
        s = simplify(step)
        if is_int_value(s):
            if s.as_long() > 0:
                return None
            return _Refused(
                SanitizerKind.NON_POSITIVE_STEP,
                f"the loop step is {s.as_long()}; only positive steps are modeled",
            )
        status, model, reason = self.solve([*env.premises, step <= 0])
        if status == unsat:
            return None
        if status == sat:
            assert model is not None
            return _Refused(
                SanitizerKind.NON_POSITIVE_STEP,
                f"the loop step can be {model.eval(step, model_completion=True)} "
                f"(at {self.witness(env, model)}); only positive steps are modeled",
            )
        return _Refused(
            SanitizerKind.SOLVER_UNKNOWN,
            f"Z3 could not decide whether the loop step is positive ({reason})",
        )

    # ── one access ──

    def check_access(
        self, index: int, access: AccessEvent, grid: tuple[int, int, int]
    ) -> None:
        graph = self.graph
        if mentions_observed(access.offset, graph):
            # A free observation would make any address reachable.
            raise _Refused(
                SanitizerKind.OBSERVATION_IN_ADDRESS,
                "the address depends on the value an atomic observed",
            )
        facts = self.binding.tensors.get(access.base_param)
        if facts is None:
            raise _Refused(
                SanitizerKind.MISSING_BINDING,
                f"pointer argument {access.base_param!r} has no tensor binding"
                + _binding_error(self.binding),
            )
        problem = _unusable(facts)
        if problem is not None:
            raise _Refused(
                SanitizerKind.MISSING_BINDING,
                f"the tensor facts of {access.base_param!r} are unusable ({problem})",
            )
        env = _Env(graph, self.binding, grid, self.ctx)
        assumed: list[BoolRef] = []
        if access.in_loop:
            # Even an offset without the induction variable executes only
            # on iterations that run: none for a zero-trip loop.
            env.loop_iteration()
            assumed += [
                _fits(env.value(ob.term), ob.bits, ob.signed) for ob in self.loop_obs
            ]
            assumed += [env.value(d.b) != 0 for d in self.loop_divs]
        # Lower every root first: a refusal leaves no partial result.
        offset = env.value(access.offset)
        path = env.cond(access.path) if access.path is not None else None
        mask = env.cond(access.mask) if access.mask is not None else None
        # One DAG over the three roles: a node's guard covers every root it
        # reaches, so a node one role reads through a Select arm and another
        # directly is checked wherever it is read.
        dag = _Dag((access.path, access.mask, access.offset), graph, env)
        obs: dict[str, list[WidthObligation]] = {"path": [], "mask": [], "offset": []}
        for ob in width_obligations(graph, access):
            if ob.role != "loop":
                obs[ob.role].append(ob)
        seen = set(self.loop_ids) if access.in_loop else set()
        divs = {
            role: _divisions((root,), graph, seen)
            for role, root in (
                ("path", access.path),
                ("mask", access.mask),
                ("offset", access.offset),
            )
        }
        observed_path = access.path is not None and mentions_observed(
            access.path, graph
        )
        observed_mask = access.mask is not None and mentions_observed(
            access.mask, graph
        )

        def uncertain(role: str) -> SanitizerKind | None:
            if access.guarded or observed_path:
                return SanitizerKind.UNMODELABLE_CONDITION
            if role != "path" and observed_mask:
                return SanitizerKind.DATA_DEPENDENT_MASK
            if role == "offset" and access.mask_dropped:
                return SanitizerKind.DATA_DEPENDENT_MASK
            return None

        for role, condition in (("path", path), ("mask", mask), ("offset", None)):
            assumed = self.role(
                env, index, obs[role], divs[role], assumed, uncertain(role), dag
            )
            if condition is not None:
                assumed.append(condition)
        self.out_of_bounds(
            env, index, access, facts, offset, assumed, uncertain("offset")
        )

    def role(
        self,
        env: _Env,
        index: int,
        obs: Sequence[WidthObligation],
        divs: Sequence[Bin],
        assumed: list[BoolRef],
        uncertain: SanitizerKind | None,
        dag: _Dag,
    ) -> list[BoolRef]:
        """Discharge one role's width obligations and divisions under
        ``assumed`` (the earlier roles' safety) in one joint query: some
        condition of the role fails. None of the role's own conditions is
        assumed while they are checked, only the sites reported already (so
        a wrap another access reported does not resurface at the ops
        computed from it). At most one finding of each kind: the model's
        innermost failure, then a query for the other kind with that site
        assumed. Returns ``assumed`` plus the role's safety."""
        conds: list[_Condition] = []
        for i, ob in enumerate(obs):
            fits = _fits(env.value(ob.term), ob.bits, ob.signed)
            guard = None if _undefined_when_wide(ob) else dag.guards.get(id(ob.term))
            conds.append(
                _Condition(
                    "integer-overflow",
                    ob,
                    fits if guard is None else Implies(guard, fits),
                    # Children first (a select's condition before its arms),
                    # and on one term the op's own result before the reads
                    # of the ops that read it.
                    (dag.rank(ob.term), -i),
                )
            )
        for d in divs:
            nonzero = env.value(d.b) != 0
            conds.append(_Condition("division-by-zero", d, nonzero, (dag.rank(d), 0)))
        held = [c.safe for c in conds if self.is_reported(c.kind, c.site)]
        pending = [c for c in conds if not self.is_reported(c.kind, c.site)]
        while pending:
            status, model, reason = self.solve(
                [
                    *env.premises,
                    *assumed,
                    *held,
                    Or([Z3Not(c.safe) for c in pending]),
                ]
            )
            if status == unknown:
                self.unknown(index, "an integer-overflow or division-by-zero", reason)
            if status != sat:
                break
            assert model is not None
            failed = [c for c in pending if is_false(model.eval(c.safe, True))]
            chosen = min(failed, key=lambda c: c.rank) if failed else pending[0]
            self.found(chosen, env, model, index, uncertain)
            held.append(chosen.safe)
            pending = [c for c in pending if c.kind != chosen.kind]
        return [*assumed, *(c.safe for c in conds)]

    def out_of_bounds(
        self,
        env: _Env,
        index: int,
        access: AccessEvent,
        facts: TensorFacts,
        offset: ArithRef,
        assumed: list[BoolRef],
        uncertain: SanitizerKind | None,
    ) -> None:
        width = _pointee_bytes(access.elem_bits)
        status, model, reason = self.solve(
            [*env.premises, *assumed, Z3Not(_legal(offset, facts, width))]
        )
        if status == unknown:
            self.unknown(index, "the out-of-bounds", reason)
        if status != sat:
            return
        assert model is not None
        off = model.eval(offset, True).as_long()
        # No address in the text: it is the same for every launch with the
        # same launch_key (the address is violation_address).
        detail = (
            f"{access.kind} of {access.base_param!r} at element offset {off} "
            f"is outside {_footprint(facts)}"
        )
        if uncertain is not None:
            self.withhold(index, uncertain, "out-of-bounds", detail)
            return
        self.findings.append(
            Finding(
                kind="out-of-bounds",
                access_index=index,
                access_kind=access.kind,
                base_param=access.base_param,
                line_no=access.line_no,
                loc=_location(access.loc),
                witness=self.witness(env, model),
                violation_offset=off,
                violation_address=facts.data_ptr + off * width,
                detail=detail,
            )
        )

    # ── results ──

    def found(
        self,
        cond: _Condition,
        env: _Env,
        model: ModelRef,
        index: int,
        uncertain: SanitizerKind | None,
    ) -> None:
        site = cond.site
        extra: dict[str, int] = {}
        if isinstance(site, WidthObligation):
            value = model.eval(env.value(site.term), True).as_long()
            detail = self.overflow_detail(site, value)
            extra["value"] = value
        else:
            what = "remainder" if site.op in ("%", "u%") else "division"
            detail = f"the divisor of this {what} ({site.op!r}) can be 0"
        if uncertain is not None:
            self.withhold(index, uncertain, cond.kind, detail)
            return
        self.reported.add(self.site_key(cond.kind, site))
        self._alive.append(site)
        access = self.graph.accesses[index]
        line_no, loc = site.line_no, site.loc
        if line_no is None:
            line_no, loc = access.line_no, access.loc
        self.findings.append(
            Finding(
                kind=cond.kind,
                access_index=index,
                access_kind=access.kind,
                base_param=access.base_param,
                line_no=line_no,
                loc=_location(loc),
                witness={**self.witness(env, model), **extra},
                detail=detail,
            )
        )

    @staticmethod
    def site_key(kind: str, site: WidthObligation | Bin) -> tuple[str, object]:
        """One finding per op: by its TTIR line, else by the term's identity
        (``found`` keeps a reported term alive, so its id stays unique)."""
        if site.line_no is not None:
            return kind, site.line_no
        return kind, id(site.term if isinstance(site, WidthObligation) else site)

    def is_reported(self, kind: str, site: WidthObligation | Bin) -> bool:
        return self.site_key(kind, site) in self.reported

    def withhold(
        self, index: int, kind: SanitizerKind, finding: str, detail: str
    ) -> None:
        self.abstain(
            index,
            _Refused(
                kind,
                f"possible {finding} {_WITHHELD[kind]}, so the witness may be "
                f"unreachable: {detail}",
            ),
        )

    def unknown(self, index: int, query: str, reason: str | None) -> None:
        self.abstain(
            index,
            _Refused(
                SanitizerKind.SOLVER_UNKNOWN,
                f"Z3 could not decide {query} query ({reason})",
            ),
        )

    def abstain(self, index: int, refused: _Refused) -> None:
        access = self.graph.accesses[index]
        refused.at(access.line_no, access.loc)
        entry = (index, refused.kind)
        if entry not in self.abstained:
            self.abstained.append(entry)
        if self.refusal is None:
            self.refusal = Refusal(
                kind=refused.kind.value,
                message=f"{describe_site(refused.line_no, refused.loc)}: "
                f"{refused.message}",
                line_no=refused.line_no,
                loc=_location(refused.loc),
            )

    def overflow_detail(self, ob: WidthObligation, value: int) -> str:
        if ob.signed:
            half = 1 << (ob.bits - 1)
            bounds = f"i{ob.bits} range [{-half}, {half})"
        else:
            bounds = f"unsigned i{ob.bits} range [0, {1 << ob.bits})"
        # The obligation's origin (see width_obligations): an op's result, a
        # trunci operand (the other signed ones), or an unsigned read.
        loop = self.graph.loop
        if loop is not None and _is_increment(ob, loop, self.loop_ids):
            what = "the loop's induction-variable increment"
        elif isinstance(ob.term, Bin) and ob.signed and ob.term.bits == ob.bits:
            what = f"the result of {ob.term.op!r}"
        elif ob.signed:
            what = f"the operand of a truncation to i{ob.bits}"
        else:
            what = "a value read as unsigned"
        return (
            f"{what} can be {value}, outside the {bounds}: the IR's "
            "fixed-width arithmetic differs from the unbounded reading"
        )

    def witness(self, env: _Env, model: ModelRef) -> dict[str, int]:
        def val(v: ArithRef) -> int:
            return model.eval(v, model_completion=True).as_long()

        out = {f"pid_{axis}": val(pid) for axis, pid in enumerate(env.pids)}
        for name, lane in env.lanes.items():
            out[name] = val(lane)
        if env.iteration is not None:
            out[str(env.iteration)] = val(env.iteration)
        return out

    def solve(self, formulas: list[Any]) -> tuple[Any, ModelRef | None, str | None]:
        """One query on a fresh Solver with its own timeout (never the
        process-global z3.set_param). A query Z3 cancelled for a Ctrl+C
        raises KeyboardInterrupt."""
        solver = Solver(ctx=self.ctx)
        solver.set(timeout=self.timeout_ms)
        solver.add(*formulas)
        status = solver.check()
        if status == sat:
            return status, solver.model(), None
        if status == unknown:
            reason = solver.reason_unknown()
            if reason == _INTERRUPTED:
                raise KeyboardInterrupt(f"Z3 query cancelled ({reason})")
            return status, None, reason
        return status, None, None
