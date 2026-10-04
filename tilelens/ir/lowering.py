"""Term -> Z3 lowering shared by the compiled-mode clients.

Mechanism only: the operator semantics live here,
every leaf's meaning comes from the client. A client's :class:`TermLeaves`
says what a scalar argument, a program id, the grid, a lane, a loop's
iteration, an atomic observation and an unmodeled value are, and a client
that cannot model a leaf raises its own exception from it, which propagates
unchanged. The lowering itself raises only on a malformed graph, a bug and
never a limit of the model: ``TypeError`` or ``ValueError`` for a term or op
outside the algebra, or for a ``LoopVar`` (or an ``IterArgOffset`` whose
IterArgInfo names no loop) in a graph without a loop, and ``IndexError`` for
an ``arg_id`` outside ``iter_args``.

Operator semantics, the reader's integer model (see ``ttir_reader``):

* ``+ - *`` are unbounded Int arithmetic; ``//`` and ``%`` truncate toward
  zero (``arith.divsi`` / ``remsi``: the remainder takes the dividend's
  sign); ``min`` / ``max`` pick an operand;
* the unsigned ops (``u//``, ``u%``, ``umin``, ``umax``) and predicates
  (``ult``, ...) read as their signed twins, and an ``IntCast`` as its
  operand: exact only where the access's ``width_obligations`` hold, which
  the client discharges (a zero divisor is the client's concern too);
* an i1 value is a Bool in a boolean position (a mask or path, a Select's
  condition, ``and`` / ``or`` / negation) and 0/1 in an integer one, and a
  Select whose arms differ in sort is an Int;
* ``LoopVar`` is ``lower + k * step`` and ``IterArgOffset`` is ``offset0 +
  k * delta``, with ``k`` the client's iteration of that loop.

Z3 context: the lowering never creates one. Constants are made in
``leaves.ctx`` (None: Z3's main context) and every other term in its
operands' context, so all of a client's terms share the context of its
leaves (the compiled sanitizer's is a Context per check).

Terms can be deeper than Python's recursion limit and their generated
``==`` / ``hash`` recurse, so the walk is iterative and memoized by term
identity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from z3 import And, ArithRef, BoolRef, Context, If, IntVal, Or, is_bool
from z3 import Not as Z3Not

from .ttir_reader import (
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
)


class TermLeaves(Protocol):
    """A client's meaning of every leaf of the algebra (there are no
    defaults): each method returns a Z3 Int in ``ctx``, or raises the
    client's own refusal."""

    @property
    def ctx(self) -> Context | None:
        """The Z3 context of the client's terms (None: the main one)."""

    def param(self, t: Param) -> ArithRef:
        """A scalar kernel argument."""

    def pid(self, t: Pid) -> ArithRef:
        """The program id along ``t.axis``."""

    def num_programs(self, t: NumPrograms) -> ArithRef:
        """The grid size along ``t.axis``."""

    def arange(self, t: Arange) -> ArithRef:
        """``t``'s value at the lane a query reads (the reader's contract:
        key a lane by ``(dim, end - start)``, see ``Arange``)."""

    def iteration(self, loop_ssa: str) -> ArithRef:
        """The iteration index ``k`` of the loop ``loop_ssa``."""

    def observed(self, t: Observed) -> ArithRef:
        """The old value the atomic at ``t.access_index`` observed."""

    def data_dep(self, t: DataDep) -> ArithRef:
        """A value the reader could not model (``t.why`` says which)."""


def as_bool(e: Any) -> BoolRef:
    """An i1 value in a boolean position: i1 constants (e.g. the dense<true>
    mask of an unmasked atomic, Const(1)) lower to Int."""
    return e if is_bool(e) else e != 0


def as_int(e: Any) -> ArithRef:
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


def _divrem_ir(op: str, a: ArithRef, b: ArithRef) -> ArithRef:
    """The IR's quotient (``op`` ``//`` or ``u//``) or remainder (``%`` or
    ``u%``): truncating, so the remainder has the dividend's sign."""
    q = _trunc_div(a, b)
    return q if op in ("//", "u//") else a - b * q


def _bin(op: str, a: ArithRef, b: ArithRef) -> ArithRef:
    # The unsigned ops read their operands unsigned; their width
    # obligations make both non-negative, where they equal the signed ones.
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op in ("//", "u//", "%", "u%"):
        return _divrem_ir(op, a, b)
    if op in ("min", "umin"):
        return If(a <= b, a, b)
    if op in ("max", "umax"):
        return If(a >= b, a, b)
    raise ValueError(f"unknown integer op {op!r}")


# Unsigned predicates read their operands unsigned; their width obligations
# make both non-negative, where they equal the signed ones.
_SIGNED_TWIN = {"ult": "slt", "ule": "sle", "ugt": "sgt", "uge": "sge"}


def _cmp(pred: str, a: Any, b: Any) -> BoolRef:
    if is_bool(a) or is_bool(b):  # i1 operands, as 0/1
        a, b = as_int(a), as_int(b)
    p = _SIGNED_TWIN.get(pred, pred)
    if p == "slt":
        return a < b
    if p == "sle":
        return a <= b
    if p == "sgt":
        return a > b
    if p == "sge":
        return a >= b
    if p == "eq":
        return a == b
    if p == "ne":
        return a != b
    raise ValueError(f"unknown cmpi predicate {pred!r}")


def _loop(graph: AccessGraph, t: object) -> LoopInfo:
    if graph.loop is None:
        raise ValueError(
            f"kernel {graph.kernel_name!r}: a loop term ({type(t).__name__}) "
            "without a loop"
        )
    return graph.loop


def children(t: object, graph: AccessGraph) -> tuple:
    """The terms ``t``'s value is computed from: its operands; for a
    loop-carried pointer's offset, its IterArgInfo's ``offset0`` and
    ``delta``; for the induction variable, the loop's ``lower`` and
    ``step``. A leaf (a DataDep included) has none."""
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
    if isinstance(t, LoopVar):
        loop = _loop(graph, t)
        return (loop.lower, loop.step)
    return ()


def fold(
    root: object,
    graph: AccessGraph,
    apply: Callable[[object, Sequence[Any]], Any],
    memo: dict[int, tuple[object, Any]],
) -> Any:
    """``apply(t, the values of children(t))`` at ``root``, children first,
    each term once for ``memo`` (id(term) -> (term, value): holding the
    term keeps its id unique). Iterative post-order."""
    stack: list[tuple[object, bool]] = [(root, False)]
    while stack:
        t, ready = stack.pop()
        if id(t) in memo:
            continue
        kids = children(t, graph)
        if kids and not ready:
            stack.append((t, True))
            stack.extend((k, False) for k in reversed(kids) if id(k) not in memo)
            continue
        memo[id(t)] = (t, apply(t, [memo[id(k)][1] for k in kids]))
    return memo[id(root)][1]


class Lowerer:
    """The lowering of one family of a client's queries: bound to one graph
    and one leaves object (so one Z3 context), with one memo."""

    def __init__(self, graph: AccessGraph, leaves: TermLeaves) -> None:
        self.graph = graph
        self.leaves = leaves
        self._memo: dict[int, tuple[object, Any]] = {}

    def lower(self, term: object) -> Any:
        """``term`` as a Z3 expression: an Int, or a Bool for a compare,
        ``and`` / ``or`` / negation, and a Select of Bools."""
        return fold(term, self.graph, self._apply, self._memo)

    def value(self, term: object) -> ArithRef:
        return as_int(self.lower(term))

    def cond(self, term: object) -> BoolRef:
        return as_bool(self.lower(term))

    def _apply(self, t: object, kids: Sequence[Any]) -> Any:
        leaves = self.leaves
        if isinstance(t, Const):
            return IntVal(t.value, leaves.ctx)
        if isinstance(t, Param):
            return leaves.param(t)
        if isinstance(t, Pid):
            return leaves.pid(t)
        if isinstance(t, NumPrograms):
            return leaves.num_programs(t)
        if isinstance(t, Arange):
            return leaves.arange(t)
        if isinstance(t, LoopVar):
            k = leaves.iteration(t.loop_ssa)
            return as_int(kids[0]) + k * as_int(kids[1])
        if isinstance(t, IterArgOffset):
            loop_ssa = self.graph.iter_args[t.arg_id].loop_ssa
            k = leaves.iteration(loop_ssa or _loop(self.graph, t).loop_ssa)
            return as_int(kids[0]) + k * as_int(kids[1])
        if isinstance(t, Bin):
            return _bin(t.op, as_int(kids[0]), as_int(kids[1]))
        if isinstance(t, Cmp):
            return _cmp(t.pred, kids[0], kids[1])
        if isinstance(t, BoolBin):
            a, b = as_bool(kids[0]), as_bool(kids[1])
            if t.op == "and":
                return And(a, b)
            if t.op == "or":
                return Or(a, b)
            raise ValueError(f"unknown boolean op {t.op!r}")
        if isinstance(t, Select):
            a, b = kids[1], kids[2]
            if is_bool(a) != is_bool(b):
                a, b = as_int(a), as_int(b)
            return If(as_bool(kids[0]), a, b)
        if isinstance(t, Not):
            return Z3Not(as_bool(kids[0]))
        if isinstance(t, IntCast):
            # Its value is the operand's while its width obligation holds.
            return as_int(kids[0])
        if isinstance(t, Observed):
            return leaves.observed(t)
        if isinstance(t, DataDep):
            return leaves.data_dep(t)
        raise TypeError(f"unknown term {type(t).__name__}")
