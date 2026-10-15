"""TwoCopySymbolicHBSolver on hand-built records: race found or proved, race
types, atomics, fences through the caller's hook, intra-instance lanes, the
enumeration fallback, the compiled-only restrictions and the Z3 budget."""

from __future__ import annotations

import threading

import pytest
import torch
import z3
from z3 import And, BoolVal, Int, IntVal

import tilelens.clients.race_detector.two_copy_symbolic_hb_solver as tc
from tilelens.clients.race_detector.data import AccessEventRecord, RaceType
from tilelens.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from tilelens.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from tilelens.clients.symbolic_engine import SymbolicExpr
from tilelens.core.data import Store

from .solver_records import (
    cas,
    every_fence,
    fences_at,
    load,
    report_pairs,
    rmw,
    solve,
    store,
)

PID0 = SymbolicExpr.PID0
DATA = 1 << 21


# ──────────────────────── two copies ────────────────────────


def test_pid_alpha_renaming_reports_cross_instance_waw():
    """elem_size 4 at stride 1: adjacent pids overlap by 3 bytes."""
    reports = solve([store(IntVal(DATA) + PID0, event_id=0)]).find_races()
    assert len(reports) == 1
    assert reports[0].race_type is RaceType.WAW
    assert reports[0].witness_grid_a != reports[0].witness_grid_b
    assert reports[0].witness_addr in {DATA + 1, DATA + 2, DATA + 3}


def test_arange_lanes_are_renamed_per_copy():
    lane = Int("lane")
    arange = {(0, 8): (lane, And(lane >= 0, lane < 8))}
    aliasing = store(IntVal(DATA) + 28 * PID0 + 4 * lane, event_id=0)
    disjoint = store(IntVal(DATA) + 32 * PID0 + 4 * lane, event_id=0)
    assert solve([aliasing], grid=(4, 1, 1), arange_dict=arange).find_races()
    assert solve([disjoint], grid=(4, 1, 1), arange_dict=arange).find_races() == []


def test_masked_off_access_never_races():
    rec = store(IntVal(DATA), event_id=0, mask=BoolVal(False))
    assert solve([rec]).find_races() == []


# ──────────────────────── fences via the caller's hook ────────────────────────


def _two_operations(kind):
    makers = {"RAW": (store, load), "WAR": (load, store), "WAW": (store, store)}
    records = [
        make(IntVal(100), event_id=i, program_seq=i)
        for i, make in enumerate(makers[kind])
    ]
    for i, record in enumerate(records):
        record.source_location = ("kernel.py", 10 + i, "kernel")
    return records


@pytest.mark.parametrize("kind", ["RAW", "WAR", "WAW"])
@pytest.mark.parametrize("fences", [None, (0.5,), (1.5,)])
def test_unfenced_operations_of_one_instance_race(kind, fences):
    hook = None if fences is None else fences_at(*fences)
    solver = solve(_two_operations(kind), grid=(1, 1, 1), fence_between=hook)
    reports = solver.find_races()
    if fences == (0.5,):
        assert reports == []
        return
    assert len(reports) == 1
    report = reports[0]
    assert report.race_type.name == kind
    assert report.witness_addr == 100
    assert report.witness_grid_a == report.witness_grid_b
    assert report.reason.startswith(solver._INTRA_INSTANCE_REASON)
    assert "Missing source fence:" in report.reason
    assert "kernel.py:10" in report.reason and "kernel.py:11" in report.reason


def test_fence_hook_is_asked_in_program_order_with_the_records():
    records = _two_operations("WAR")
    records[0].event_id, records[1].event_id = 1, 0
    calls = []

    def hook(first, second):
        calls.append((first, second))
        return False

    report = solve(records, grid=(1, 1, 1), fence_between=hook).find_races()[0]
    assert calls
    assert all(first is records[0] and second is records[1] for first, second in calls)
    # The report keeps its event-id order; the explanation uses source order.
    assert (report.first.event_id, report.second.event_id) == (0, 1)
    assert report.race_type is RaceType.RAW
    assert "read at kernel.py:10 and the write at kernel.py:11" in report.reason


def test_fence_between_every_operation_orders_one_instance():
    records = _two_operations("RAW")
    assert solve(records, grid=(1, 1, 1), fence_between=every_fence).find_races() == []


def test_fence_hook_answers_each_pair_on_its_own():
    """A fence at 1.5 orders (0, 2) and (1, 2) but not (0, 1)."""
    records = [store(IntVal(100), event_id=i, program_seq=i) for i in range(3)]
    solver = solve(records, grid=(1, 1, 1), fence_between=fences_at(1.5))
    assert report_pairs(solver.find_races()) == {(0, 1)}


@pytest.mark.parametrize("case", ["unknown_seq", "equal_seq", "duplicate", "cross"])
def test_missing_fence_diagnostic_needs_two_ordered_operations(case):
    records, grid = _two_operations("RAW"), (1, 1, 1)
    if case == "unknown_seq":
        records[0].program_seq = -1
    elif case == "equal_seq":
        records[1].program_seq = 0
    elif case == "duplicate":
        records = records[:1]
        records[0].addr_expr = [IntVal(100), IntVal(100)]
    else:
        grid = (2, 1, 1)
        records[0].active = PID0 == 0
        records[1].active = PID0 == 1
    reports = solve(records, grid=grid).find_races()
    assert reports
    assert all("Missing source fence:" not in r.reason for r in reports)


def test_dependency_orders_only_the_same_position():
    """A store consuming a load's value is ordered after the load at the same
    lane, but another lane of the store can still race the load."""
    lane = Int("dep_lane")
    records = _two_operations("WAR")
    records[0].addr_expr = IntVal(100) + 4 * lane
    records[1].addr_expr = IntVal(104) + 4 * lane
    records[1].dep_loads = (records[0].event_id,)
    arange = {(0, 2): (lane, And(lane >= 0, lane < 2))}
    report = solve(records, grid=(1, 1, 1), arange_dict=arange).find_races()[0]
    assert report.witness_addr == 104
    assert "Missing source fence:" in report.reason
    records[1].addr_expr = IntVal(100) + 4 * lane
    assert solve(records, grid=(1, 1, 1), arange_dict=arange).find_races() == []


# ──────────────────────── atomics ────────────────────────


def test_competing_same_location_atomics_do_not_race():
    pair = [
        cas(IntVal(500), IntVal(0), IntVal(1), Int(f"c{i}"), event_id=i)
        for i in range(2)
    ]
    assert solve(pair).find_races() == []
    assert solve([rmw(IntVal(900), event_id=0)]).find_races() == []


@pytest.mark.parametrize(
    "first,second,races",
    [((0, 4), (0, 8), True), ((0, 4), (0, 4), False), ((0, 4), (2, 4), True)],
)
def test_only_same_address_same_width_atomics_are_mutually_atomic(first, second, races):
    pair = [
        rmw(IntVal(DATA + offset), event_id=i, program_seq=i, elem_size=width)
        for i, (offset, width) in enumerate((first, second))
    ]
    assert bool(solve(pair).find_races()) is races


@pytest.mark.parametrize(
    "scopes,races", [(("gpu", "gpu"), False), (("cta", "gpu"), True)]
)
def test_cta_scoped_atomics_race_across_ctas(scopes, races):
    pair = [
        rmw(IntVal(DATA), event_id=i, program_seq=i, scope=scope)
        for i, scope in enumerate(scopes)
    ]
    assert bool(solve(pair).find_races()) is races


def test_cas_trylock_single_winner_suppresses_guarded_waw():
    """Two CAS(0 -> 1) on one flag cannot both read 0: atomic coherence
    leaves one winner, so the guarded stores never overlap."""
    flag = torch.zeros(1, dtype=torch.int32)
    old = Int("trylock_old")
    records = [
        cas(
            IntVal(flag.data_ptr()), IntVal(0), IntVal(1), old, event_id=0, tensor=flag
        ),
        store(IntVal(DATA), event_id=1, program_seq=1, mask=old == 0),
    ]
    solver = solve(records)
    assert next(e.idx for e in solver.events if e.copy == "a") in solver.rf_init_source
    assert solver.find_races() == []


@pytest.mark.parametrize("flag_kind", ["float", "large"])
def test_unidentifiable_initial_flag_falls_back_to_rf_unknown(flag_kind):
    """A float flag (int(0.7) == 0 would fabricate a lock winner) or one
    larger than the initial-source cap has no closed-world initial value:
    rf_unknown opens, and the guarded WAW is reported."""
    if flag_kind == "float":
        flag = torch.full((1,), 0.7, dtype=torch.float32)
    else:
        size = TwoCopySymbolicHBSolver._MAX_INITIAL_ATOMIC_ELEMENTS + 1
        flag = torch.zeros(size, dtype=torch.int32)
    old = Int(f"{flag_kind}_flag_old")
    records = [
        cas(
            IntVal(flag.data_ptr()), IntVal(0), IntVal(1), old, event_id=0, tensor=flag
        ),
        store(IntVal(DATA), event_id=1, program_seq=1, mask=old != 1),
    ]
    solver = solve(records)
    reader = next(e for e in solver.events if e.copy == "a" and e.atomic_kind == "cas")
    assert reader.idx in solver.rf_unknown_source
    assert reader.idx not in solver.rf_init_source
    assert solver.find_races()


@pytest.mark.parametrize("offset,races", [(0, False), (1, True)])
def test_release_acquire_cas_orders_only_through_the_exact_location(offset, races):
    """pid 0 stores data, then CAS-releases flag 0 -> 1; the other pid
    CAS-acquires the flag and reads data once it observes 1. An acquire at a
    byte-overlapping but different address is not the same location: no
    synchronizes-with edge, so the data race stays."""
    flag = torch.zeros(1, dtype=torch.int32)
    base = int(flag.data_ptr())
    seen = Int(f"acquire_old_{offset}")
    producer, consumer = PID0 == 0, PID0 != 0
    records = [
        store(IntVal(DATA), event_id=0, active=producer),
        cas(
            IntVal(base),
            IntVal(0),
            IntVal(1),
            Int(f"release_old_{offset}"),
            event_id=1,
            program_seq=1,
            sem="release",
            tensor=flag,
            active=producer,
        ),
        cas(
            IntVal(base + offset),
            IntVal(1),
            IntVal(1),
            seen,
            event_id=2,
            program_seq=2,
            sem="acquire",
            tensor=flag,
            active=consumer,
        ),
        load(IntVal(DATA), event_id=3, program_seq=3, active=consumer, mask=seen == 1),
    ]
    pairs = report_pairs(solve(records, fence_between=every_fence).find_races())
    # The misaligned CAS pair (1, 2) is itself torn.
    assert pairs == ({(0, 3), (1, 2)} if races else set())
    # Without a fence, program order carries no happens-before edge.
    assert (0, 3) in report_pairs(solve(records).find_races())


def test_same_copy_later_writer_is_no_read_from_source():
    earlier = cas(IntVal(5100), IntVal(0), IntVal(1), Int("earlier"), event_id=0)
    later = cas(
        IntVal(5100), IntVal(0), IntVal(1), Int("later"), event_id=1, program_seq=1
    )
    solver = solve([earlier, later])
    a = {e.record.event_id: e.idx for e in solver.events if e.copy == "a"}
    assert (a[1], a[0]) not in solver.rf_source
    assert (a[0], a[1]) in solver.rf_source


# ──────────────────────── one instance ────────────────────────


def _lane_store(addr_of, *, mask=None):
    lane = Int("intra_lane")
    arange = {(0, 4): (lane, And(lane >= 0, lane < 4))}
    m = None if mask is None else mask(lane)
    return store(addr_of(lane), event_id=0, mask=m), arange


@pytest.mark.parametrize(
    "addr_of,mask,races",
    [
        (lambda lane: IntVal(DATA) + 4 * (lane % 2), None, True),
        (lambda lane: IntVal(DATA) + 4 * lane, None, False),
        (lambda lane: IntVal(DATA) + 0 * lane, lambda lane: lane == 3, False),
        (lambda lane: IntVal(DATA) + 0 * lane, lambda lane: lane < 2, True),
    ],
)
def test_duplicate_lanes_of_one_store_race_in_one_instance(addr_of, mask, races):
    rec, arange = _lane_store(addr_of, mask=mask)
    reports = solve([rec], grid=(1, 1, 1), arange_dict=arange).find_races()
    assert bool(reports) is races
    if races:
        assert reports[0].race_type is RaceType.WAW
        assert reports[0].witness_grid_a == reports[0].witness_grid_b


def test_lanes_of_one_atomic_and_a_scalar_store_do_not_self_race():
    lanes = rmw([IntVal(DATA), IntVal(DATA)], event_id=0)
    assert solve([lanes], grid=(1, 1, 1)).find_races() == []
    scalar = store(IntVal(DATA) + 4 * PID0, event_id=0)
    assert solve([scalar], grid=(1, 1, 1)).find_races() == []


# ──────────────────────── feasibility and restrictions ────────────────────────


def test_unsatisfiable_premises_fail_feasibility():
    rec = store(IntVal(DATA), event_id=0)
    assert solve([rec]).check_feasibility()
    rec.premises = (PID0 < 0,)
    solver = solve([rec])
    assert solver.find_races() == []
    assert not solver.check_feasibility()


def test_only_pairs_queries_just_the_listed_record_pairs():
    records = [store(IntVal(DATA + 100 * i), event_id=i) for i in range(2)]
    assert report_pairs(solve(records).find_races()) == {(0, 0), (1, 1)}
    only = frozenset({(1, 1)})
    assert report_pairs(solve(records, only_pairs=only).find_races()) == {(1, 1)}


def test_extra_assumptions_constrain_a_symbolic_grid():
    g0 = Int("grid_0")
    rec = store(IntVal(DATA), event_id=0)
    assert solve([rec], grid=(g0, 1, 1)).find_races()
    assert (
        solve([rec], grid=(g0, 1, 1), extra_assumptions=(g0 == 1,)).find_races() == []
    )


# ──────────────────────── enumeration fallback ────────────────────────


def _force_unknown(monkeypatch):
    def undecided(solver, a, b):
        raise UnsupportedSymbolicRaceQuery(f"forced unknown for {a.name} vs {b.name}")

    monkeypatch.setattr(
        TwoCopySymbolicHBSolver, "_race_query_is_sat", staticmethod(undecided)
    )


def test_enumeration_decides_an_undecided_query_at_the_extent(monkeypatch):
    _force_unknown(monkeypatch)
    disjoint = [store(IntVal(DATA) + 4 * PID0, event_id=0)]
    solver = solve(disjoint, grid=(4, 1, 1), enum_fallback_grid=(4, 1, 1))
    assert solver.find_races() == []
    assert solver.enum_used
    colliding = [store(IntVal(DATA), event_id=0)]
    solver = solve(colliding, grid=(4, 1, 1), enum_fallback_grid=(4, 1, 1))
    reports = solver.find_races()
    assert solver.enum_used and reports
    for pid in (reports[0].witness_grid_a, reports[0].witness_grid_b):
        assert 0 <= pid[0] < 4 and pid[1:] == (0, 0)


@pytest.mark.parametrize("fallback", [None, (100, 1, 1)])
def test_enumeration_refusal_reraises_the_unknown(monkeypatch, fallback):
    """No fallback grid, or more cases than ENUM_MAX_CASES."""
    _force_unknown(monkeypatch)
    solver = solve(
        [store(IntVal(DATA) + 4 * PID0, event_id=0)],
        grid=(100, 1, 1),
        enum_fallback_grid=fallback,
    )
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="forced unknown"):
        solver.find_races()


def test_undecided_enumeration_case_reraises_the_unknown(monkeypatch):
    """An undecided case must not count as UNSAT: that would be a false
    launch-scope proof. A budget below the per-case limit caps each case."""
    _force_unknown(monkeypatch)
    monkeypatch.setattr(_RecordingSolver, "made", [])
    monkeypatch.setattr(tc, "Solver", _RecordingSolver)
    solver = solve(
        [store(IntVal(DATA) + 4 * PID0, event_id=0)],
        grid=(4, 1, 1),
        enum_fallback_grid=(4, 1, 1),
        timeout_ms=1234,
    )
    monkeypatch.setattr(z3.Solver, "check", lambda self, *a: z3.unknown)
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="forced unknown"):
        solver.find_races()
    assert solver.enum_used
    assert {s.timeout for s in _RecordingSolver.made} == {1234}


def test_decided_query_never_enumerates():
    rec = store(IntVal(DATA) + 4 * PID0, event_id=0)
    solver = solve([rec], grid=(4, 1, 1), enum_fallback_grid=(4, 1, 1))
    assert solver.find_races() == [] and not solver.enum_used


# ──────────────────────── Z3 budget and unknown ────────────────────────


class _RecordingSolver(z3.Solver):
    made: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = None
        type(self).made.append(self)

    def set(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout", self.timeout)
        super().set(*args, **kwargs)


@pytest.mark.parametrize(
    "timeout_ms,launch_ceiling,expected",
    [(1234, False, {1234}), (1234, True, {1234}), (60_000, True, {60_000, 10_000})],
)
def test_every_solver_gets_its_own_timeout(
    monkeypatch, timeout_ms, launch_ceiling, expected
):
    """Each Solver carries the budget; the retry cap only shortens it."""
    monkeypatch.setattr(_RecordingSolver, "made", [])
    monkeypatch.setattr(tc, "Solver", _RecordingSolver)
    flag = torch.zeros(1, dtype=torch.int32)
    records = [
        cas(
            IntVal(flag.data_ptr()),
            IntVal(0),
            IntVal(1),
            Int("t"),
            event_id=0,
            tensor=flag,
        ),
        store(IntVal(DATA), event_id=1, program_seq=1),
    ]
    before = z3.get_param("timeout")
    solver = solve(
        records,
        timeout_ms=timeout_ms,
        launch_ceiling=launch_ceiling,
        enum_fallback_grid=(2, 1, 1),
    )
    assert solver.find_races() and solver.check_feasibility()
    assert z3.get_param("timeout") == before
    assert _RecordingSolver.made
    assert {s.timeout for s in _RecordingSolver.made} == expected
    assert TwoCopySymbolicHBSolver.TIMEOUT_MS == 180_000
    assert solve(records).timeout_ms == 180_000
    with pytest.raises(ValueError, match="positive"):
        solve(records, timeout_ms=0)


def test_tiny_timeout_makes_a_hard_query_unsupported():
    """x^3 + y^3 == z^3 over positive pids (no solution, which Z3 cannot
    prove in time) gates a colliding store: unknown, never a proof."""
    p = (SymbolicExpr.PID0, SymbolicExpr.PID1, SymbolicExpr.PID2)
    fermat = And(
        *(v > 0 for v in p), sum(v * v * v for v in p[:2]) == p[2] * p[2] * p[2]
    )
    grid = tuple(Int(f"fermat_grid_{i}") for i in range(3))
    before = z3.get_param("timeout")
    solver = solve(
        [store(IntVal(DATA), event_id=0, mask=fermat)], grid=grid, timeout_ms=50
    )
    outcome = []

    def run():
        try:
            solver.find_races()
        except UnsupportedSymbolicRaceQuery as exc:
            outcome.append(exc)

    # Fail instead of hanging the suite if the budget stops being applied.
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive(), "the 50 ms budget was not applied"
    assert outcome and "could not decide" in str(outcome[0])
    assert z3.get_param("timeout") == before


class _UnknownSolver:
    """Stands in for z3.Solver: every query is undecided."""

    def __init__(self, *args, **kwargs):
        pass

    def add(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass

    def push(self):
        pass

    def pop(self):
        pass

    def check(self):
        return z3.unknown

    def reason_unknown(self):
        return "stubbed unknown"


def test_unknown_overlap_check_opens_rf_unknown_escape(monkeypatch):
    """Keeping the closed world on an undecided overlap would pin the
    reader's old value and hide every conflict gated on it."""
    flag = torch.zeros(1, dtype=torch.int32)
    old = Int("escape_old")
    records = [
        cas(
            IntVal(flag.data_ptr()), IntVal(0), IntVal(1), old, event_id=0, tensor=flag
        ),
        store(IntVal(DATA), event_id=1, program_seq=1),
    ]
    monkeypatch.setattr(tc, "Solver", _UnknownSolver)
    solver = solve(records)
    reader = next(e for e in solver.events if e.copy == "a" and e.atomic_kind == "cas")
    assert reader.idx in solver.rf_unknown_source
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="stubbed unknown"):
        solver.check_feasibility()


def test_records_need_byte_addresses():
    rec = AccessEventRecord(op_type=Store, access_mode="write")
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="addr_expr"):
        solve([rec])
