"""Shared HB primitives for the race detector's two-copy solver."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

from z3 import And, BoolSort, BoolVal, Not, Or, is_false, is_true, simplify, substitute
from z3.z3 import BoolRef, IntNumRef


class UnsupportedSymbolicRaceQuery(Exception):
    """Raised when a record cannot be lowered to Z3 or a query is undecided.

    Either capture-side normalisation or solver-side lowering may raise this,
    and so may any solver query that Z3 cannot decide. The caller must report
    the launch as unsupported, never as race-free.
    """


class CountingScopeUnsupported(UnsupportedSymbolicRaceQuery):
    """An atomic address needs a counter whose symbolic-grid bound may wrap.

    A caller may retry a smaller declared launch domain with every guard
    intact. This does not certify that the counter is valid in that domain.
    """


def as_bool(value: Any) -> BoolRef:
    """Coerce ``value`` to a Z3 ``BoolRef``.

    Accepts Python bool/int/float, Z3 ``IntNumRef``, and non-Bool-sort Z3
    expressions (returning ``value != 0``). ``None`` is intentionally NOT
    handled — passing ``None`` indicates a caller-side bug and should
    surface as a Z3 type error rather than silently inactivating the
    event.
    """
    if isinstance(value, bool):
        return BoolVal(value)
    if isinstance(value, IntNumRef):
        return BoolVal(value.as_long() != 0)
    if isinstance(value, (int, float)):
        return BoolVal(value != 0)
    if hasattr(value, "sort") and value.sort() != BoolSort():
        return value != 0
    return value


def iter_constraints(value: Any) -> Iterator[Any]:
    """Flatten ``None`` / scalar / list / tuple into individual constraints."""
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_constraints(item)
        return
    yield value


def to_lanes(value: Any) -> tuple[Any, ...]:
    """Return ``value`` as a tuple of per-lane components."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def lane_value(value: Any, lane: int, n_lanes: int) -> Any:
    """Return the component of ``value`` at ``lane``.

    - ``None`` → ``None``.
    - Python scalar / Z3 expression → broadcast to every lane.
    - list/tuple of length ``n_lanes`` → indexed lookup.
    - list/tuple of length 1 → broadcast.
    - any other shape → :class:`UnsupportedSymbolicRaceQuery`.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == n_lanes:
            return value[lane]
        if len(value) == 1:
            return value[0]
        raise UnsupportedSymbolicRaceQuery(
            f"lane_value: vector of length {len(value)} cannot be aligned with "
            f"n_lanes={n_lanes}"
        )
    return value


def apply_sub(value: Any, substitutions: tuple[tuple[Any, Any], ...]) -> Any:
    """Recursively apply Z3 substitutions through ``None`` / list / tuple / scalar."""
    if value is None:
        return None
    if isinstance(value, list):
        return [apply_sub(v, substitutions) for v in value]
    if isinstance(value, tuple):
        return tuple(apply_sub(v, substitutions) for v in value)
    if isinstance(value, (bool, int)):
        return value
    if not substitutions:
        return value
    return substitute(value, *substitutions)


def is_release_sem(sem: Any) -> bool:
    return sem in ("release", "acq_rel")


def is_acquire_sem(sem: Any) -> bool:
    return sem in ("acquire", "acq_rel")


def build_transitive_hb(
    events: list[Any],
    edge_fn: Callable[[Any, Any], BoolRef],
) -> list[list[BoolRef]]:
    """Floyd-Warshall closure, omitting only identically false paths.

    Simplification uses no query assumptions: every conditional edge, including
    cycles and self-edges, retains its original meaning.  Each layer combines
    only the non-false incoming and outgoing entries for its intermediate node,
    instead of constructing expressions for all pairs.  The initial matrix and
    row/column scans cost O(n**2); path construction costs the sum of the layers'
    incoming-count times outgoing-count, with O(n**3) worst-case work.
    """
    n = len(events)
    reach = [
        [simplify(edge_fn(events[i], events[j])) for j in range(n)] for i in range(n)
    ]
    for k in range(n):
        # Snapshot both vectors before updating the matrix.  Even for a cyclic
        # graph, every right-hand side then refers to the previous FW layer.
        incoming = [(i, reach[i][k]) for i in range(n) if not is_false(reach[i][k])]
        outgoing = [(j, edge) for j, edge in enumerate(reach[k]) if not is_false(edge)]
        for i, left in incoming:
            for j, right in outgoing:
                previous = reach[i][j]
                if is_true(previous):
                    continue
                if is_true(left):
                    through = right
                elif is_true(right) or left.eq(right):
                    through = left
                else:
                    through = And(left, right)
                if is_false(previous) or is_true(through):
                    reach[i][j] = through
                elif not previous.eq(through):
                    reach[i][j] = Or(previous, through)
    return reach


def conflicting_access_modes(first: Any, second: Any) -> BoolRef:
    """``(write,read|write)`` conflict, minus morally strong pairs.

    An atomic-vs-atomic pair is race-free only when the two operations are
    actually atomic with respect to EACH OTHER: mutually inclusive scopes,
    same access width, and the exact same address. Scope inclusion is
    symbolic: a pair involving a ``"cta"`` scope is mutually inclusive
    exactly when the two events share a block (PTX ``.cta`` scope
    guarantees atomicity within one CTA only), which cross-copy queries
    falsify via their different-blocks assertion while same-instance
    queries satisfy. Byte-overlapping atomics at different addresses or
    widths are torn accesses, which race like plain writes.

    Built with explicit ``And``/``Or`` to avoid Python operator-precedence
    pitfalls between Z3 expressions and Python booleans.
    """
    access_conflict = Or(
        And(first.writes, Or(second.reads, second.writes)),
        And(second.writes, Or(first.reads, first.writes)),
    )
    if not (first.is_atomic and second.is_atomic):
        return access_conflict

    if first.elem_size != second.elem_size:
        return access_conflict

    if first.scope != "cta" and second.scope != "cta":
        return And(access_conflict, Not(first.addr == second.addr))
    scope_ok = And(
        first.pid[0] == second.pid[0],
        first.pid[1] == second.pid[1],
        first.pid[2] == second.pid[2],
    )
    return And(access_conflict, Not(And(scope_ok, first.addr == second.addr)))


def modeled_atomic_read_from(
    writer: Any,
    reader: Any,
    *,
    same_atomic_addr_fn: Callable[[Any, Any], BoolRef],
) -> BoolRef:
    """Read-from predicate between two VALUE-MODELED atomic events.

    Membership is field-presence-based: the writer must have a modeled
    written value and the reader a modeled observation (CAS always has
    both; an RMW has them exactly when its return value is modeled). An
    atomic without value modeling never gets an rf edge — its write stays
    in the unmodeled-writer set that opens the ``rf_unknown`` escape.
    """
    if not writer.is_atomic or not reader.is_atomic:
        return BoolVal(False)
    if writer.written_value is None or reader.old_value is None:
        return BoolVal(False)
    return And(
        writer.writes,
        reader.reads,
        same_atomic_addr_fn(writer, reader),
        writer.written_value == reader.old_value,
    )


def normalize_copy_local_vars(values: Iterable[Any]) -> tuple[Any, ...]:
    """Flatten and dedup copy-local Z3 vars.

    Dedup key is ``(v.hash(), str(v.sort()), v.decl().name())`` rather than
    ``v.hash()`` alone, so distinct vars that happen to share a hash are not
    collapsed. ``None`` and Python scalars are skipped — only Z3 vars are
    retained.
    """
    out: list[Any] = []
    seen: set[tuple[int, str, str]] = set()

    def visit(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
            return
        if isinstance(v, (bool, int)):
            return
        try:
            key = (v.hash(), str(v.sort()), v.decl().name())
        except Exception:
            return
        if key in seen:
            return
        seen.add(key)
        out.append(v)

    for value in values:
        visit(value)
    return tuple(out)


__all__ = [
    "CountingScopeUnsupported",
    "UnsupportedSymbolicRaceQuery",
    "apply_sub",
    "as_bool",
    "build_transitive_hb",
    "conflicting_access_modes",
    "is_acquire_sem",
    "is_release_sem",
    "iter_constraints",
    "lane_value",
    "modeled_atomic_read_from",
    "normalize_copy_local_vars",
    "to_lanes",
]
