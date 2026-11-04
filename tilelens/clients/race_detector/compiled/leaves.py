"""The race detector's meaning of the TTIR reader's leaves.

``RaceLeaves`` is the race detector's ``TermLeaves``: it lowers the
reader's terms in Z3's main context (``ctx`` None), where the two-copy
solver renames each program copy's variables with ``z3.substitute``:

* a program id is ``tilelens.clients.pids.pid_var(axis)`` and the grid size
  ``grid_var(axis)``, the names the solver renames and bounds;
* an arange is ``start + lane``, one lane variable per ``(dim, extent)``
  (the reader's contract, see ``Arange``), registered in the solver's
  arange dict so each copy gets its own lane in ``[0, extent)``;
* a loop's iteration index is the variable its binding gives
  (``iterations``), copy-local in the records;
* an atomic observation is one interned variable per access;
* a scalar argument is a shared free Int when ``params`` is None (no
  launch: the any-params rung) and the launch's value, read signed at the
  argument's width, otherwise;
* a loaded value is a ``Select`` over the launch's pre-launch contents of
  its tensor (a snapshot), read signed at the load's own element width (an
  unsigned table with a value the signed reading makes negative has no
  usable snapshot); without a usable snapshot it is free (a per-load array,
  copy-local), and the access that reads it is over-approximated;
* an unmodeled value refuses;
* ``//`` and ``%`` are Z3's Euclidean ones where the operands' signs make
  them the IR's truncating ones (``BoundedDivRem``), else the lowering's
  default.

What cannot be modeled raises :class:`RaceRefusal` with a ``RaceKind``.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from z3 import (
    And,
    ArithRef,
    Array,
    BoolRef,
    If,
    Implies,
    Int,
    IntSort,
    IntVal,
    Select,
    simplify,
)

from ....ir import failures
from ....ir.launch import TensorFacts
from ....ir.lowering import Lowerer, children, fold, truncating_divrem
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
    Loaded,
    LoopVar,
    Not,
    NumPrograms,
    Observed,
    Param,
    Pid,
)
from ....ir.ttir_reader import Select as TSelect
from ... import pids


class RaceKind(str, Enum):
    """Why the compiled race detector left a kernel undecided (its own
    kinds, apart from the reader's ``TTIRKind``)."""

    # An scf.while that is not the await shape.
    SPIN_SHAPE = "spin-shape"
    # A CAS whose operands are outside the integer value model.
    CAS_VALUE = "cas-value"
    # A CAS inside a loop (one observation cannot stand for one per
    # iteration).
    CONTROL_FLOW = "control-flow"
    # An address built on an observation or a loaded value that is not
    # value-modeled.
    INDIRECT_ADDRESS = "indirect-address"
    # An address built on a loaded value whose table is too large to read.
    SNAPSHOT_BOUND = "snapshot-bound"
    # A value the reader marked unmodelable reached a modeled term.
    UNMODELED_VALUE = "unmodeled-value"
    # A kernel argument, tensor fact or the grid the analysis needs is not
    # bound (or unusable) for the launch.
    MISSING_BINDING = "missing-binding"
    # A loop whose bounds or step the model cannot iterate.
    LOOP_BOUND = "loop-bound"
    # A width obligation can fail at the launch, where the proof needs it.
    INTEGER_OVERFLOW = "integer-overflow"
    # Z3 answered unknown: a timeout, or another reason.
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    # Every race found is between over-approximated records (a dropped
    # mask, a free loaded value): a possible race, not a witness.
    POTENTIAL_RACE = "potential-race"
    # The premises admit no execution, so a proof would be vacuous.
    VACUOUS = "vacuous"
    OTHER = "other"
    # The client's (compiled/client.py, which classifies a failed compile
    # with tilelens.ir.failures, whose two kinds these are): the launch
    # compiled nothing to read; a config failed to compile for the IR
    # target, or none compiled; a compiled kernel holds no TTIR; the host
    # compile could not run; the analysis itself failed (a bug, contained).
    NO_COMPILED_KERNEL = "no-compiled-kernel"
    COMPILE_FAILED = failures.COMPILE_FAILED
    NO_TTIR = "no-ttir"
    HOST_COMPILE_UNAVAILABLE = failures.HOST_COMPILE_UNAVAILABLE
    INTERNAL_ERROR = "internal-error"

    def __str__(self) -> str:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)


class RaceRefusal(Exception):
    """The compiled race detector cannot decide a kernel: ``kind`` (a
    ``RaceKind``), ``message``, and the TTIR ``line_no`` / user ``loc`` of
    the access it refused at (None when not tied to one)."""

    def __init__(
        self,
        kind: RaceKind | str,
        message: str,
        *,
        line_no: int | None = None,
        loc: Any = None,
    ) -> None:
        super().__init__(message)
        self.kind = RaceKind(kind)
        self.message = message
        self.line_no = line_no
        self.loc = loc

    def at(self, line_no: int | None, loc: Any) -> RaceRefusal:
        """Tie the refusal to an access when it is tied to none yet."""
        if self.line_no is None and self.loc is None:
            self.line_no, self.loc = line_no, loc
        return self

    def __reduce__(self):
        return (
            type(self),
            (self.kind, self.message),
            {"line_no": self.line_no, "loc": self.loc},
        )


@dataclass(frozen=True)
class LaunchTensor:
    """What the analysis knows of one tensor argument at the launch: its
    facts and, for a small integer tensor, its pre-launch contents
    (``values``, flat in the view's element order, as the dtype's values;
    None with ``why`` saying why not)."""

    facts: TensorFacts
    values: tuple[int, ...] | None = None
    why: str = ""


def signed(value: int, bits: int) -> int:
    """The IR's signed reading of a ``bits``-wide value (an i1 stays 0/1)."""
    if bits <= 1:
        return value
    half = 1 << (bits - 1)
    return (value + half) % (1 << bits) - half


def param_value(graph: AccessGraph, params: Mapping[str, int], name: str) -> int:
    """A scalar argument's launch value, read signed at its width."""
    arg = graph.arg(name)
    return signed(int(params[name]), arg.int_bits if arg else 0)


Bounds = tuple[int | None, int | None]


class BoundedDivRem:
    """The race detector's ``//`` and ``%`` (the Lowerer's ``divrem``):
    Z3's Euclidean ``/`` and ``%`` where the operands' signs make them the
    IR's truncating ones, which saves the lowering's sign branches (a
    grouped program-id swizzle, ``pid // width`` and ``pid % group``, is
    several times faster to solve). The signs come from inclusive bounds
    over the domains every query assumes (``bounds``); otherwise the
    lowering's default:

    * a dividend ``>= 0``: ``a / b`` and ``a % b`` (Euclidean and truncating
      agree for either sign of ``b``; ``%`` is guarded at ``b == 0``, where
      the truncating remainder is ``a``, unless ``b``'s sign is known);
    * a dividend ``<= 0`` with a divisor of known sign: the same on ``-a``,
      the sign re-applied.
    """

    def __init__(self, graph: AccessGraph, params: Mapping[str, int] | None) -> None:
        self.graph = graph
        self.params = params
        self._memo: dict[int, tuple[object, Any]] = {}

    def bounds(self, term: object) -> Bounds:
        """Inclusive bounds on ``term``'s lowered value (None: unknown),
        from the domains every query assumes and nothing else: constants,
        the launch's parameters (none at T0), program ids ``>= 0``, the grid
        ``>= 1``, lanes in ``[0, extent)``, and the arithmetic over them.
        Never a mask, a path, a launch pin or a loop's range (a loop's
        iteration range holds only in its own accesses' records)."""
        return fold(term, self.graph, self._bounds, self._memo)

    def _bounds(self, t: object, kids: Sequence[Bounds]) -> Bounds:
        if isinstance(t, Const):
            return t.value, t.value
        if isinstance(t, Param):
            if self.params is None or t.name not in self.params:
                return None, None
            v = param_value(self.graph, self.params, t.name)
            return v, v
        if isinstance(t, Pid):
            return 0, None
        if isinstance(t, NumPrograms):
            return 1, None
        if isinstance(t, Arange):
            return (t.start, t.end - 1) if t.start < t.end else (None, None)
        if isinstance(t, (Cmp, BoolBin, Not)):
            return 0, 1  # a Bool, read as 0/1 in an integer position
        if isinstance(t, IntCast):
            return kids[0]  # lowered as its operand
        if isinstance(t, TSelect):
            (lo_a, hi_a), (lo_b, hi_b) = kids[1], kids[2]
            return (
                None if lo_a is None or lo_b is None else min(lo_a, lo_b),
                None if hi_a is None or hi_b is None else max(hi_a, hi_b),
            )
        if isinstance(t, Bin):
            return _bin_bounds(t.op.lstrip("u"), *kids[0], *kids[1])
        return None, None

    def __call__(self, t: Bin, a: ArithRef, b: ArithRef) -> ArithRef:
        lo_a, hi_a = self.bounds(t.a)
        lo_b, hi_b = self.bounds(t.b)
        quotient = t.op.lstrip("u") == "//"
        positive_b = lo_b is not None and lo_b > 0
        negative_b = hi_b is not None and hi_b < 0
        if lo_a is not None and lo_a >= 0:
            if quotient:
                return a / b
            if positive_b or negative_b:
                return a % b
            return If(b == 0, a, a % b)
        if (positive_b or negative_b) and hi_a is not None and hi_a <= 0:
            d = b if positive_b else -b
            if not quotient:
                return -((-a) % d)
            return -((-a) / d) if positive_b else (-a) / d
        return truncating_divrem(t, a, b)


def _bin_bounds(op: str, lo_a: Any, hi_a: Any, lo_b: Any, hi_b: Any) -> Bounds:
    """Bounds on ``a op b`` (``op`` signed; the unsigned ops lower as their
    signed twins) from the operands'."""
    known = lo_a is not None and hi_a is not None and lo_b is not None
    known = known and hi_b is not None
    if op == "+":
        return (
            None if lo_a is None or lo_b is None else lo_a + lo_b,
            None if hi_a is None or hi_b is None else hi_a + hi_b,
        )
    if op == "-":
        return (
            None if lo_a is None or hi_b is None else lo_a - hi_b,
            None if hi_a is None or lo_b is None else hi_a - lo_b,
        )
    if op == "min":
        highs = [v for v in (hi_a, hi_b) if v is not None]
        return (
            None if lo_a is None or lo_b is None else min(lo_a, lo_b),
            min(highs) if highs else None,
        )
    if op == "max":
        lows = [v for v in (lo_a, lo_b) if v is not None]
        return (
            max(lows) if lows else None,
            None if hi_a is None or hi_b is None else max(hi_a, hi_b),
        )
    if op == "*":
        if known:
            products = [x * y for x in (lo_a, hi_a) for y in (lo_b, hi_b)]
            return min(products), max(products)
        if lo_a == hi_a == 0 or lo_b == hi_b == 0:
            return 0, 0
        nonneg_a, nonneg_b = (
            lo_a is not None and lo_a >= 0,
            lo_b is not None and lo_b >= 0,
        )
        nonpos_a, nonpos_b = (
            hi_a is not None and hi_a <= 0,
            hi_b is not None and hi_b <= 0,
        )
        if nonneg_a and nonneg_b:
            return lo_a * lo_b, None
        if nonpos_a and nonpos_b:
            return hi_a * hi_b, None
        if nonneg_a and nonpos_b:
            return None, lo_a * hi_b
        if nonpos_a and nonneg_b:
            return None, hi_a * lo_b
        return None, None
    if op == "//":
        if lo_a is not None and lo_a >= 0 and lo_b is not None and lo_b > 0:
            return (
                0 if hi_b is None else lo_a // hi_b,
                None if hi_a is None else hi_a // lo_b,
            )
        return None, None
    if op == "%":
        if lo_b is not None and hi_b is not None and lo_b > 0:
            return (
                0 if lo_a is not None and lo_a >= 0 else 1 - hi_b,
                0 if hi_a is not None and hi_a <= 0 else hi_b - 1,
            )
        if lo_a is not None and lo_a >= 0:
            # the truncating remainder has the dividend's sign (a at b == 0)
            return 0, None
        return None, None
    return None, None


# Representation only: a non-affine table up to this many elements is read
# as an If-chain, a larger one through the array alone (same values).
_SMALL_SNAPSHOT_ELEMENTS = 32


def snapshot_lookup(values: Sequence[int], index: ArithRef, fallback: Any) -> Any:
    """``values[index]`` as arithmetic or a small finite choice where that
    is exact, else ``fallback`` (the Select over the snapshot array, whose
    equalities the solver asserts). Outside the table the value stays the
    fallback: no value is invented for an element that was not read."""
    if not values:
        return fallback
    index = simplify(index)
    try:
        concrete = index.as_long()
    except AttributeError:
        concrete = None
    if concrete is not None:
        return IntVal(values[concrete]) if 0 <= concrete < len(values) else fallback
    step = values[1] - values[0] if len(values) > 1 else 0
    if all(v == values[0] + i * step for i, v in enumerate(values)):
        return If(
            And(index >= 0, index < len(values)),
            IntVal(values[0]) + index * step,
            fallback,
        )
    if len(values) > _SMALL_SNAPSHOT_ELEMENTS:
        return fallback
    value = fallback
    for i in reversed(range(len(values))):
        value = If(index == i, IntVal(values[i]), value)
    return value


def terms_below(
    roots: Sequence[object], graph: AccessGraph, *, loops: bool = True
) -> Iterator[object]:
    """Every term reachable from ``roots`` (the lowering's ``children``, and
    a Loaded's offset, mask and other), each once by identity, pre-order;
    without ``loops``, not into a loop variable's bounds. Iterative."""
    seen: set[int] = set()
    stack = [r for r in reversed(roots) if r is not None]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        yield t
        if isinstance(t, Loaded):
            kids: tuple = tuple(k for k in (t.offset, t.mask, t.other) if k is not None)
        elif isinstance(t, LoopVar) and not loops:
            kids = ()
        else:
            kids = children(t, graph)
        stack.extend(reversed(kids))


def loaded_leaves(
    roots: Sequence[object], graph: AccessGraph, *, loops: bool = True
) -> list[Loaded]:
    """The Loaded leaves ``roots`` read (each placement once)."""
    return [t for t in terms_below(roots, graph, loops=loops) if isinstance(t, Loaded)]


def observed_leaves(roots: Sequence[object], graph: AccessGraph) -> set[int]:
    """The access indices of the observations ``roots`` read."""
    return {
        t.access_index for t in terms_below(roots, graph) if isinstance(t, Observed)
    }


class RaceLeaves:
    """The leaves of one encoding (see the module docstring): one launch
    reading (``params`` None: any parameters), one Lowerer, and the state
    the records read back (the arange dict, the snapshot equalities, the
    loads left free)."""

    ctx = None

    def __init__(
        self,
        graph: AccessGraph,
        params: Mapping[str, int] | None,
        tensors: Mapping[str, LaunchTensor],
        *,
        blind: bool = False,
        unusable: Mapping[str, str] | None = None,
    ) -> None:
        self.graph = graph
        self.params = params
        self.tensors = tensors
        # The content-free reading: no snapshot is used.
        self.blind = blind
        # Sources whose contents may change during the launch (see
        # global_records.written_sources).
        self.unusable = dict(unusable or {})
        self.arange_dict: dict[Any, Any] = {}
        self.iterations: dict[str, ArithRef] = {}
        self._lanes: dict[tuple[int, int], ArithRef] = {}
        self._observed: dict[int, ArithRef] = {}
        self._params: dict[str, ArithRef] = {}
        # Snapshot arrays by (base, width), their equalities, and per-load
        # free arrays (copy-local: masked-off or unread lanes).
        self._snapshots: dict[tuple[str, int], tuple[Any, tuple[int, ...]]] = {}
        self.assumptions: list[BoolRef] = []
        self._pads: dict[int, Any] = {}
        self.used_snapshot = False
        # Loads whose value is free, with why.
        self.free_loaded: dict[int, str] = {}
        # The lowering reaches its leaves through a proxy, so the leaves and
        # their terms are freed without the cyclic GC.
        self.lowering = Lowerer(
            graph, weakref.proxy(self), divrem=BoundedDivRem(graph, params)
        )

    # ── lowering ──

    def value(self, term: object) -> ArithRef:
        return self.lowering.value(term)

    def cond(self, term: object) -> BoolRef:
        return self.lowering.cond(term)

    @property
    def lanes(self) -> Mapping[tuple[int, int], ArithRef]:
        """The lane variables made so far, by ``(dim, extent)``."""
        return self._lanes

    @property
    def pads(self) -> tuple[Any, ...]:
        return tuple(self._pads.values())

    def observation(self, index: int) -> ArithRef:
        v = self._observed.get(index)
        if v is None:
            v = self._observed[index] = Int(f"ttir_obs_{index}")
        return v

    def lane(self, dim: int, extent: int) -> ArithRef:
        key = (dim, extent)
        v = self._lanes.get(key)
        if v is None:
            v = self._lanes[key] = Int(f"ttir_lane_d{dim}_n{extent}")
            # The solver's arange dict: key[0:2] is the lane's range, which
            # it rebuilds per copy; the value's first item the variable.
            self.arange_dict[(0, extent, "ttir", dim)] = (v, None)
        return v

    # ── TermLeaves ──

    def param(self, t: Param) -> ArithRef:
        if self.params is None:
            v = self._params.get(t.name)
            if v is None:
                v = self._params[t.name] = Int(f"ttir_param_{t.name}")
            return v
        if t.name not in self.params:
            raise RaceRefusal(
                RaceKind.MISSING_BINDING,
                f"scalar argument {t.name!r} has no launch binding",
            )
        return IntVal(param_value(self.graph, self.params, t.name))

    def pid(self, t: Pid) -> ArithRef:
        return pids.pid_var(t.axis)

    def num_programs(self, t: NumPrograms) -> ArithRef:
        return pids.grid_var(t.axis)

    def arange(self, t: Arange) -> ArithRef:
        return IntVal(t.start) + self.lane(t.dim, t.end - t.start)

    def iteration(self, loop_ssa: str) -> ArithRef:
        k = self.iterations.get(loop_ssa)
        if k is None:
            raise ValueError(
                f"kernel {self.graph.kernel_name!r}: loop {loop_ssa!r} is not bound"
            )
        return k

    def observed(self, t: Observed) -> ArithRef:
        return self.observation(t.access_index)

    def data_dep(self, t: DataDep) -> ArithRef:
        raise RaceRefusal(RaceKind.UNMODELED_VALUE, f"an unmodeled value ({t.why})")

    def loaded(self, t: Loaded) -> ArithRef:
        """``If(mask, snapshot[offset], other)`` at the lane: a masked-off
        lane holds ``other``, or, without one, a free value. The offset's
        domain is not guarded here: each record that reads the value
        carries ``mask -> 0 <= offset < numel`` (``domain_premises``)."""
        index = t.access_index
        off = self.value(t.offset)
        access = self.graph.accesses[index]
        problem = self.snapshot_problem(access)
        if problem is not None:
            self.free_loaded.setdefault(index, problem)
            return Select(self.pad(index), off)
        arr, values = self.snapshot(access)
        self.used_snapshot = True
        value = snapshot_lookup(values, off, Select(arr, off))
        if t.mask is None:
            return value
        other = (
            self.value(t.other) if t.other is not None else Select(self.pad(index), off)
        )
        return If(self.cond(t.mask), value, other)

    # ── snapshots ──

    def pad(self, index: int) -> Any:
        pad = self._pads.get(index)
        if pad is None:
            pad = self._pads[index] = Array(f"ttir_pad_{index}", IntSort(), IntSort())
        return pad

    def snapshot_problem(self, access: AccessEvent) -> str | None:
        """Why the load ``access`` has no usable snapshot, or None. A
        snapshot is read only at the load's own width (an i1 load of a bool
        tensor included): a load through a pointer cast to another width
        would read the bytes another way."""
        if self.params is None:
            return "no launch at the any-parameters rung"
        if self.blind:
            return "content-free attempt"
        base = access.base_param
        tensor = self.tensors.get(base)
        if tensor is None:
            return "not captured"
        if base in self.unusable:
            return self.unusable[base]
        facts = tensor.facts
        if not facts.contiguous:
            return "non-contiguous"
        if tensor.values is None:
            return tensor.why or "no snapshot"
        if facts.allocation_interval() != (
            facts.data_ptr,
            facts.data_ptr + facts.numel * facts.elem_size,
        ):
            # Consumers assume 0 <= offset < numel; a view of a larger
            # allocation can legally read outside the captured values.
            return "the snapshot covers only a view of its allocation"
        bits = facts.elem_size * 8
        if access.elem_bits != bits and not (
            facts.dtype == "torch.bool" and access.elem_bits == 1
        ):
            return f"read as i{access.elem_bits}, but the tensor holds {facts.dtype}"
        if facts.dtype.startswith("torch.uint") and any(
            v >= 1 << (bits - 1) for v in tensor.values
        ):
            # The IR's signless integer reads them negative: an extui of one
            # would fail its width obligation, though the program computes
            # it exactly.
            return (
                f"{facts.dtype} values of 2**{bits - 1} or more, which the "
                f"signed i{bits} reading does not keep"
            )
        return None

    def snapshot(self, access: AccessEvent) -> tuple[Any, tuple[int, ...]]:
        """The snapshot array of the load's tensor at its width, with its
        values read signed at that width (their equalities join
        ``assumptions`` once)."""
        bits = access.elem_bits
        key = (access.base_param, bits)
        found = self._snapshots.get(key)
        if found is not None:
            return found
        tensor = self.tensors[access.base_param]
        assert tensor.values is not None
        values = tuple(signed(int(v), bits) for v in tensor.values)
        arr = Array(f"ttir_snap_{access.base_param}_i{bits}", IntSort(), IntSort())
        self.assumptions += [
            Select(arr, IntVal(i)) == IntVal(v) for i, v in enumerate(values)
        ]
        found = self._snapshots[key] = (arr, values)
        return found

    def domain_premises(self, leaves: Sequence[Loaded]) -> tuple[BoolRef, ...]:
        """For each snapshotted load among ``leaves``: where its mask holds,
        its offset is inside the table (an instance whose load reads outside
        its source is outside the model, as its own record says). A free
        load needs none."""
        out: list[BoolRef] = []
        for leaf in leaves:
            access = self.graph.accesses[leaf.access_index]
            if self.snapshot_problem(access) is not None:
                continue
            numel = self.tensors[access.base_param].facts.numel
            off = self.value(leaf.offset)
            inside = And(off >= 0, off < numel)
            out.append(
                inside if leaf.mask is None else Implies(self.cond(leaf.mask), inside)
            )
        return tuple(out)
