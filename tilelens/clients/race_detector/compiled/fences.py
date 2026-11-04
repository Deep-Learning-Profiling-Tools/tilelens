"""The compiled race detector's fence rule, the two-copy solver's
``fence_between`` and ``fence_epoch`` hooks over the reader's Fence facts.

A ``ttg.barrier`` (``tl.debug_barrier()``) orders global memory only when
its ``addr_space`` has both the global_read (2) and global_write (4) bits.
Such a fence ``f`` orders two accesses ``a < f < b`` of one program
instance when its block path is a prefix of both accesses' block paths:
then it runs whenever both do (it is not under a branch either one escapes,
and not in a loop either one is outside of). In a loop, ``f`` orders the
two accesses of one iteration; ``fence_between`` (the solver asks it for
every pair of executions) holds only when every pair of executions is
ordered, so a fence in a loop also needs one on the back edge: another
fence of the loop's body outside ``[a, b]``. The pairs of different
iterations that a single fence leaves unordered the solver queries at
equal fence epochs (``epoch``): the number of ordering fences the instance
has run before the execution, exact where every fence that may run before
it is unconditional and the loop trip counts it depends on are constants,
else unconstrained.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from z3 import ArithRef, Int, IntVal

from ....ir.ttir_reader import AccessEvent, AccessGraph, Fence
from ..data import AccessEventRecord

_GLOBAL = 2 | 4  # global_read | global_write


def _prefix(p: tuple[int, ...], q: tuple[int, ...]) -> bool:
    return len(p) <= len(q) and q[: len(p)] == p


def _common(p: tuple[str, ...], q: tuple[str, ...]) -> tuple[str, ...]:
    n = 0
    while n < len(p) and n < len(q) and p[n] == q[n]:
        n += 1
    return p[:n]


class FenceRule:
    """The rule for one graph. ``loops`` maps a loop name to its
    iteration-index variable and its trip count (None when not a constant
    for the encoding)."""

    def __init__(
        self, graph: AccessGraph, loops: Mapping[str, tuple[ArithRef, int | None]]
    ) -> None:
        self.graph = graph
        self.loops = loops
        self.fences = [f for f in graph.fences if f.addr_space & _GLOBAL == _GLOBAL]
        # Fences inside an scf.while run a data-dependent number of times.
        self._in_while = [
            any(
                _prefix(w.block_path + (block,), f.block_path)
                for w in graph.whiles
                for block in w.blocks
            )
            for f in self.fences
        ]
        self._epochs: dict[int, Any] = {}

    def _access(self, record: AccessEventRecord) -> AccessEvent:
        return self.graph.accesses[record.program_seq]

    def between(self, first: AccessEventRecord, second: AccessEventRecord) -> bool:
        """Every execution of ``first`` is ordered with every execution of
        ``second`` by a fence between them (``first`` comes first in program
        order; in a loop, a later iteration's ``first`` comes after an
        earlier one's ``second``)."""
        a, b = self._access(first), self._access(second)
        lo, hi = first.program_seq, second.program_seq
        shared = [
            f
            for f in self.fences
            if _prefix(f.block_path, a.block_path)
            and _prefix(f.block_path, b.block_path)
        ]
        for f in shared:
            if not lo < f.position < hi:
                continue
            if not f.loops:
                return True
            # In a loop: a back-edge fence of the same loop body orders the
            # later iterations' executions of ``first`` after ``second``.
            if any(
                g is not f
                and g.loops[: len(f.loops)] == f.loops
                and not lo < g.position < hi
                for g in shared
            ):
                return True
        return False

    def epoch(self, record: AccessEventRecord) -> Any:
        """The number of ordering fences the instance ran before this
        execution of ``record``: an int, a term over its loops' iteration
        indices, or an unconstrained value where that is not exact."""
        seq = record.program_seq
        if seq not in self._epochs:
            self._epochs[seq] = self._epoch(seq)
        return self._epochs[seq]

    def _epoch(self, seq: int) -> Any:
        access = self.graph.accesses[seq]
        own = access.loops
        total: Any = 0
        for f, in_while in zip(self.fences, self._in_while):
            count = self._count(f, in_while, seq, own)
            if count is None:
                return Int(f"fence_epoch_unknown_{seq}")
            total = count if isinstance(total, int) and total == 0 else total + count
        return total

    def _count(
        self, f: Fence, in_while: bool, seq: int, own: tuple[str, ...]
    ) -> Any | None:
        """How many times ``f`` ran before the execution of access ``seq``
        (None: not exact)."""
        common = _common(f.loops, own)
        before = f.position < seq
        if not common and not before:
            return 0  # runs only after the access, in no loop they share
        if in_while or f.guarded or f.path is not None:
            return None
        # Executions of f per iteration of the innermost common loop.
        per = 1
        for name in f.loops[len(common) :]:
            trips = self._trips(name)
            if trips is None:
                return None
            per *= trips
        # Complete iterations of the innermost common body before this one.
        earlier: Any = 0
        for i, name in enumerate(common):
            weight = 1
            for inner in common[i + 1 :]:
                trips = self._trips(inner)
                if trips is None:
                    return None
                weight *= trips
            k = self.loops[name][0]
            earlier = k * weight if i == 0 else earlier + k * weight
        current = per if before else 0
        if isinstance(earlier, int):
            return earlier * per + current
        return earlier * IntVal(per) + current if per != 1 else earlier + current

    def _trips(self, name: str) -> int | None:
        found = self.loops.get(name)
        return None if found is None else found[1]
