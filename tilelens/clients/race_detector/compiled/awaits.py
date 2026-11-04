"""The await abstraction: which ``scf.while`` loops the race detector reads
as a spin on one location.

The reader reports each ``scf.while`` as a neutral ``WhileRegion``. A spin
``while load(flag) != 1: pass`` (or on a CAS or an identity RMW) is read as
ONE poll that observed its exit value: the loop terminates, and the last
read saw ``Not(cond)``. The failed iterations are dropped, and a pre-exit
representative of the poll (``global_records``) keeps the races they could
have. Anything else refuses as ``spin-shape``, naming the rule it breaks
(an ``scf.condition`` outside a while's condition region is the TTIR
reader's refusal, and a condition it cannot read breaks rule 8):

1. the loop carries no values (no iter args or results);
2. it is not nested in another while (inside an scf.for it may be);
3. its body is ``scf.yield`` only;
4. its condition region writes nothing (no store);
5. its condition region reads exactly one location;
6. that location is not float-typed (observations are integers);
7. an RMW poll writes back what it read (``add`` / ``or`` / ``xor`` of 0);
8. the condition is a comparison;
9. of the poll's observation against a value that does not depend on it.
"""

from __future__ import annotations

from ....ir.ttir_reader import (
    AccessGraph,
    Cmp,
    Const,
    Not,
    Observed,
    Term,
    WhileRegion,
    observed_indices,
)
from .leaves import RaceKind, RaceRefusal

_IDENTITY_RMW = ("add", "or", "xor")


def awaited_accesses(graph: AccessGraph) -> dict[int, Term]:
    """The awaited polls of ``graph``: access index -> exit predicate.
    Raises a ``spin-shape`` RaceRefusal for a while that is not a spin."""
    for w in graph.whiles:  # outer first, as the loops open
        if w.carries:
            raise _refuse(w, "the loop carries values (iter args or results)")
        if w.parent is not None:
            raise _refuse(
                w, "a spin nested in another scf.while is not the await shape"
            )
    return dict(_spin(graph, w) for w in graph.whiles)


def _refuse(w: WhileRegion, why: str) -> RaceRefusal:
    return RaceRefusal(
        RaceKind.SPIN_SHAPE,
        f"line {w.line_no} (scf.while): {why}",
        line_no=w.line_no,
        loc=w.loc,
    )


def _spin(graph: AccessGraph, w: WhileRegion) -> tuple[int, Term]:
    def refuse(why: str) -> RaceRefusal:
        return _refuse(w, why)

    extra = [name for name in w.after_ops if name != "scf.yield"]
    if extra:
        raise refuse(f"the loop body must be scf.yield only, found {extra[0]}")
    reads = [graph.accesses[i] for i in w.before]
    if any(a.kind == "store" for a in reads):
        raise refuse("the spin condition stores")
    if len(reads) != 1:
        raise refuse(
            "the spin condition must read exactly one location "
            f"(found {len(reads)} accesses)"
        )
    index, poll = w.before[0], reads[0]
    if poll.elem_float:
        raise refuse("the awaited location is float-typed")
    if poll.kind == "atomic_rmw":
        op = poll.atomic.rmw_op if poll.atomic is not None else None
        if op not in _IDENTITY_RMW or poll.atomic_val != Const(0):
            # A mutating poll writes on every dropped iteration: the loop
            # could end by observing its own writes.
            raise refuse(
                f"the spin re-read writes the awaited location (atomic {op} "
                "with a nonzero operand)"
            )
    cond = w.cond
    if not isinstance(cond, Cmp):
        raise refuse("the spin condition is not a comparison over the awaited read")
    own = Observed(index)
    a_is, b_is = cond.a == own, cond.b == own
    expected = cond.b if a_is else cond.a
    if a_is == b_is or index in observed_indices(expected, graph):
        raise refuse(
            "the spin condition must compare the awaited read against a value "
            "that does not depend on it"
        )
    return index, Not(cond)
