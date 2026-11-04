"""The proof rung of one compiled kernel: the least concretization that
decides it.

* T0, any parameters and any grid along the axes the kernel reads: tried
  when the launch's tensors are pairwise disjoint (one encoding per tensor
  stands for non-aliased arguments) and every term is linear with the
  params symbolic; any SAT, refusal or undecided query falls through.
* T1, this launch's parameters, any grid: first with every loaded value
  free (the content-free attempt, when a term reads one: its UNSAT is a
  proof for any contents, its SAT is no report), then with the launch's
  snapshots (a proof that read one holds for these contents).
* T1-launch, this launch's parameters and grid: when the any-grid query
  finds a race, the race pairs are asked again with the grid pinned; UNSAT
  there proves the kernel as launched, and the any-grid reports are kept
  as grid-fragility evidence, never as races. When Z3 does not decide that
  requery, the config is undecided (``timeout`` / ``unknown``), with the
  any-grid reports as evidence only.

A proof needs a feasible base (some execution satisfies the premises), or
it would be vacuous. An any-grid proof assumes the width obligations hold
(``Outcome.overflow`` names the widths when some can fail in its domain);
they are discharged exactly at the launch (its parameters and grid), and a
proof whose obligations can fail there refuses as ``integer-overflow``,
whatever its rung: the unbounded reading it proved is not what this launch
computes.

Z3 context: every term lives in Z3's main context, which is not
thread-safe; the caller serializes analyses (``MAIN_CONTEXT_LOCK``) and
drops the outcome's terms before releasing it. A Ctrl+C that cancels a
query raises KeyboardInterrupt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from z3 import Or, Solver, is_true, sat, unknown

from ... import pids
from ....ir.ttir_reader import AccessGraph, Term
from ..hb_common import CountingScopeUnsupported, UnsupportedSymbolicRaceQuery
from ..two_copy_symbolic_hb_solver import TwoCopySymbolicHBSolver
from .awaits import awaited_accesses
from .global_records import (
    Encoder,
    Encoding,
    content_free_view,
    graph_mentions_loaded,
    grid_pins,
    symbolic_grid,
    t0_linearity_gate,
)
from .leaves import LaunchTensor, RaceKind, RaceRefusal

T0_TIMEOUT_MS = 10_000
T1_TIMEOUT_MS = 120_000
# What Z3 answers for a query its Ctrl+C handler cancelled (Python's own
# handler does not run during the query).
_INTERRUPTED = "interrupted from keyboard"


@dataclass
class Outcome:
    """What one kernel's analysis decided. ``status`` is "proved", "races"
    (``reports`` exact, ``widened`` from over-approximated records),
    "vacuous" (no execution satisfies the premises) or "unsupported"
    (``refusal``)."""

    status: str
    rung: str | None = None  # "T0" | "T1" | "T1-launch" for a proof
    # The verdict read the launch's tensor contents (a snapshot).
    content: bool = False
    # The kernel awaits: the verdict assumes its spin loops terminate.
    assumes_termination: bool = False
    # The widths whose obligations an any-grid proof assumes (some can fail
    # in its domain).
    overflow: tuple[int, ...] = ()
    reports: list[Any] = field(default_factory=list)
    widened: list[Any] = field(default_factory=list)
    # The any-grid reports behind a T1-launch proof.
    grid_fragile: list[Any] = field(default_factory=list)
    refusal: RaceRefusal | None = None


def _unsupported(r: RaceRefusal, termination: bool = False) -> Outcome:
    # A caught refusal's traceback holds frames full of Z3 terms.
    return Outcome(
        "unsupported", refusal=r.with_traceback(None), assumes_termination=termination
    )


def _check_interrupt(reason: str | None) -> None:
    if reason == _INTERRUPTED:
        raise KeyboardInterrupt(f"Z3 query cancelled ({reason})")


def _undecided(e: UnsupportedSymbolicRaceQuery) -> RaceRefusal:
    _check_interrupt(e.reason_unknown)
    kind = RaceKind.TIMEOUT if e.reason_unknown == "timeout" else RaceKind.UNKNOWN
    return RaceRefusal(kind, f"solver: {e}")


def disjoint_tensors(graph: AccessGraph, tensors: Mapping[str, LaunchTensor]) -> bool:
    """Every accessed tensor has a verified allocation, and they are pairwise
    disjoint: the non-aliasing the per-tensor T0 encoding assumes."""
    intervals = []
    for name in {a.base_param for a in graph.accesses}:
        t = tensors.get(name)
        interval = None if t is None else t.facts.allocation_interval()
        if interval is None:
            return False
        intervals.append(interval)
    intervals.sort()
    return all(s2 >= e1 for (_, e1), (s2, _) in zip(intervals, intervals[1:]))


class _Analysis:
    def __init__(
        self,
        graph: AccessGraph,
        params: Mapping[str, int],
        tensors: Mapping[str, LaunchTensor],
        launch_grid: tuple[int, int, int] | None,
        awaits: Mapping[int, Term],
        t0_timeout_ms: int,
        t1_timeout_ms: int,
    ) -> None:
        self.graph = graph
        self.params = params
        self.tensors = tensors
        self.grid = launch_grid
        self.awaits = awaits
        self.t0_ms = t0_timeout_ms
        self.t1_ms = t1_timeout_ms
        self.termination = bool(awaits)

    def solver(self, enc: Encoding, timeout_ms: int, **kw: Any) -> Any:
        kw.setdefault("extra_assumptions", enc.assumptions)
        return TwoCopySymbolicHBSolver(
            enc.records,
            arange_dict=enc.arange_dict,
            fence_between=enc.fences.between,
            fence_epoch=enc.fences.epoch,
            timeout_ms=timeout_ms,
            **kw,
        )

    def proved(self, rung: str, enc: Encoding | None, overflow=()) -> Outcome:
        return Outcome(
            "proved",
            rung=rung,
            content=bool(enc and enc.content_qualified),
            assumes_termination=self.termination,
            overflow=tuple(overflow),
        )

    # ── width obligations ──

    def overflow(
        self, encoder: Encoder, grid: tuple[Any, Any, Any], timeout_ms: int
    ) -> tuple[int, ...] | None:
        """The widths of the obligations that can fail in the domain of
        ``grid`` (its symbolic dims free): () when none can, None when Z3
        cannot tell. Each width is asked until none is left that can fail
        (one model need not show them all)."""
        failing = encoder.violations
        found: set[int] = set()
        s = Solver()
        s.set("timeout", timeout_ms)
        s.add(*encoder.domain())
        s.add(
            *(pids.grid_var(i) == d for i, d in enumerate(grid) if isinstance(d, int))
        )
        while True:
            rest = [(bits, f) for bits, f in failing if bits not in found]
            if not rest:
                break
            s.push()
            s.add(Or(*(f for _, f in rest)))
            result = s.check()
            if result == unknown:
                _check_interrupt(s.reason_unknown())
                return None
            if result != sat:
                break
            model = s.model()
            found |= {bits for bits, f in rest if is_true(model.eval(f, True))}
            s.pop()
        return tuple(sorted(found))

    def any_grid_proof(
        self, rung: str, encoder: Encoder, enc: Encoding, grid: Any, ms: int
    ) -> Outcome | None:
        """The proof at ``rung``, qualified by the widths whose obligations
        can fail in its domain; those must hold at the launch (a T0 proof
        is checked there with a launch encoding: None when it cannot be
        built, and the next rung decides)."""
        widths = self.overflow(encoder, grid, ms)
        if widths is None:  # undecided: assume every width the proof uses
            widths = tuple(sorted({bits for bits, _ in encoder.violations}))
        if widths and self.grid is not None:
            at_launch = encoder
            if rung == "T0":
                try:
                    at_launch = Encoder(
                        self.graph, self.params, self.tensors, self.awaits
                    )
                    at_launch.encode()
                except RaceRefusal:
                    return None
            refused = self.launch_widths(at_launch)
            if refused is not None:
                return refused
        return self.proved(rung, enc, widths)

    def launch_widths(self, encoder: Encoder) -> Outcome | None:
        """The refusal when a width obligation of ``encoder``'s records can
        fail at the launch (its parameters and grid), or Z3 cannot tell;
        None when they all hold."""
        assert self.grid is not None
        widths = self.overflow(encoder, self.grid, self.t1_ms)
        if widths is None:
            return _unsupported(
                RaceRefusal(
                    RaceKind.TIMEOUT,
                    "Z3 could not decide the width obligations at the launch",
                ),
                self.termination,
            )
        if widths:
            bits = "/".join(f"i{b}" for b in widths)
            return _unsupported(
                RaceRefusal(
                    RaceKind.INTEGER_OVERFLOW,
                    f"an {bits} value can overflow at this launch, where the "
                    "race-freedom proof needs the unbounded reading",
                ),
                self.termination,
            )
        return None

    def launch_proof(self, encoder: Encoder, enc: Encoding, **kw: Any) -> Outcome:
        """A proof at the launch's grid: its width obligations must hold
        there exactly."""
        refused = self.launch_widths(encoder)
        if refused is not None:
            return refused
        out = self.proved("T1-launch", enc)
        out.grid_fragile = list(kw.get("grid_fragile", ()))
        return out

    # ── rungs ──

    def run(self) -> Outcome:
        if disjoint_tensors(self.graph, self.tensors) and t0_linearity_gate(
            self.graph, self.awaits
        ):
            t0 = self.t0()
            if t0 is not None:
                return t0
        if graph_mentions_loaded(self.graph):
            free = self.t1(blind=True)
            if free is not None:
                return free
        out = self.t1(blind=False)
        assert out is not None
        return out

    def t0(self) -> Outcome | None:
        """Every per-tensor group UNSAT over a feasible base, or None."""
        try:
            encoder = Encoder(self.graph, None, {}, self.awaits)
            groups: dict[str, list] = {}
            for seq, access in enumerate(self.graph.accesses):
                groups.setdefault(access.base_param, []).append(
                    (seq, content_free_view(access, self.graph))
                )
            last = None
            for items in groups.values():
                if all(a.kind == "load" for _, a in items):
                    continue  # reads never conflict
                enc = encoder.encode(items)
                solver = self.solver(
                    enc, self.t0_ms, grid=symbolic_grid(enc, self.grid, t0=True)
                )
                if solver.find_races() or not solver.check_feasibility():
                    return None
                last = enc
        except RaceRefusal:
            return None
        except UnsupportedSymbolicRaceQuery as e:
            _check_interrupt(e.reason_unknown)
            return None
        if last is None:
            return self.proved("T0", None)
        return self.any_grid_proof(
            "T0", encoder, last, symbolic_grid(last, self.grid, t0=True), self.t0_ms
        )

    def t1(self, blind: bool) -> Outcome | None:
        """The T1 rungs; with ``blind`` (the content-free attempt) only a
        proof, else None."""
        try:
            encoder = Encoder(
                self.graph, self.params, self.tensors, self.awaits, blind=blind
            )
            enc = encoder.encode()
            grid = symbolic_grid(enc, self.grid)
            scoped = False
            try:
                solver = self.solver(
                    enc, self.t1_ms, grid=grid, enum_fallback_grid=self.grid
                )
            except CountingScopeUnsupported:
                if self.grid is None:
                    raise
                # Retry every query at this launch, every counting guard kept.
                solver = self.solver(
                    enc,
                    self.t1_ms,
                    grid=self.grid,
                    extra_assumptions=grid_pins(grid, self.grid) + enc.assumptions,
                    enum_fallback_grid=self.grid,
                    launch_ceiling=True,
                )
                scoped = True
            found = solver.find_races()
            if found:
                if blind:
                    return None
                return self.races(encoder, enc, grid, found, scoped)
            extra = ()
            if solver.enum_used:
                assert self.grid is not None  # the fallback needs a launch grid
                extra = grid_pins(grid, self.grid)
            feasible = solver.check_feasibility(extra=extra)
        except RaceRefusal as r:
            return None if blind else _unsupported(r, self.termination)
        except UnsupportedSymbolicRaceQuery as e:
            refusal = _undecided(e)  # a Ctrl+C raises, blind or not
            return None if blind else _unsupported(refusal, self.termination)
        if not feasible:
            if blind:
                return None
            return Outcome(
                "vacuous",
                refusal=RaceRefusal(
                    RaceKind.VACUOUS,
                    "the premises admit no execution, so the race-freedom "
                    "proof would be vacuous",
                ),
                assumes_termination=self.termination,
            )
        out: Outcome | None
        if scoped or solver.enum_used:
            out = self.launch_proof(encoder, enc)
        else:
            out = self.any_grid_proof("T1", encoder, enc, grid, self.t1_ms)
        assert out is not None  # a T1 encoder is a launch encoding
        # A refusal of the content-free attempt leaves the snapshots to try.
        return None if blind and out.status != "proved" else out

    def races(
        self, encoder: Encoder, enc: Encoding, grid: Any, found: list, scoped: bool
    ) -> Outcome:
        """Any-grid reports: asked again at the launch's grid, unless the
        query already ran there. When that requery is not decided, the
        config is undecided: the any-grid reports may lie outside the
        launch's grid, so they are kept as evidence only (``grid_fragile``),
        never as races of this launch."""
        if not scoped and self.grid is not None:
            pinned = self.requery(enc, grid, found)
            if isinstance(pinned, RaceRefusal):
                out = _unsupported(pinned, self.termination)
                out.grid_fragile = list(found)
                return out
            if not pinned:
                return self.launch_proof(encoder, enc, grid_fragile=found)
            found = pinned
        out = Outcome(
            "races", content=enc.content_qualified, assumes_termination=self.termination
        )
        for rep in found:
            ids = {rep.first.event_id, rep.second.event_id}
            (out.widened if ids & enc.uncertain else out.reports).append(rep)
        return out

    def requery(self, enc: Encoding, grid: Any, prior: list) -> list | RaceRefusal:
        """The pairs behind ``prior`` at the launch's grid (a pair with no
        any-grid report stays UNSAT under the pins); the refusal saying why
        when Z3 does not decide it."""
        assert self.grid is not None
        pairs = frozenset(
            (
                min(r.first.event_id, r.second.event_id),
                max(r.first.event_id, r.second.event_id),
            )
            for r in prior
        )
        try:
            return self.solver(
                enc,
                self.t1_ms,
                grid=grid,
                extra_assumptions=grid_pins(grid, self.grid) + enc.assumptions,
                only_pairs=pairs,
                enum_fallback_grid=self.grid,
                launch_ceiling=True,
            ).find_races()
        except UnsupportedSymbolicRaceQuery as e:
            undecided = _undecided(e)  # a Ctrl+C raises
            return RaceRefusal(
                undecided.kind,
                f"the race query at the launch's grid was not decided "
                f"({undecided.message}); the {len(prior)} race(s) found at "
                "another grid are evidence only",
            )


def analyze_graph(
    graph: AccessGraph,
    params: Mapping[str, int],
    tensors: Mapping[str, LaunchTensor],
    launch_grid: tuple[int, int, int] | None,
    *,
    t0_timeout_ms: int = T0_TIMEOUT_MS,
    t1_timeout_ms: int = T1_TIMEOUT_MS,
) -> Outcome:
    """Decide one compiled kernel (a graph read with ``multipath=True`` and
    ``while_regions=True``) for a launch's ``params``, ``tensors`` and
    ``launch_grid`` (None: no launch rung)."""
    try:
        awaits = awaited_accesses(graph)
    except RaceRefusal as r:
        return _unsupported(r)
    return _Analysis(
        graph, params, tensors, launch_grid, awaits, t0_timeout_ms, t1_timeout_ms
    ).run()


__all__ = ["Outcome", "analyze_graph", "disjoint_tensors"]
