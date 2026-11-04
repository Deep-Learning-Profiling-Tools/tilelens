"""Value-modeled RMW returns: reads-through release sequences, the counting
axiom and its guards, and the observation-in-address gate. Most proofs have
a mutation twin that must flip to a report."""

from __future__ import annotations

import pytest
import torch
import z3
from z3 import Int, IntVal

from tilelens.clients.race_detector.hb_common import (
    CountingScopeUnsupported,
    UnsupportedSymbolicRaceQuery,
)
from tilelens.clients.race_detector.two_copy_symbolic_hb_solver import (
    _rmw_written_value,
)
from tilelens.clients.symbolic_engine import SymbolicExpr
from tilelens.core.data import Load, Store

from .solver_records import (
    cas,
    counter,
    every_fence,
    load,
    report_pairs,
    rmw,
    solve,
    store,
)

PID0 = SymbolicExpr.PID0
PARTIAL = 1 << 20
DATA = 1 << 21
BUF = 1 << 22


def test_rmw_written_value_table():
    old, v = Int("o"), Int("v")
    for op in ("add", "max", "min", "xchg"):
        assert _rmw_written_value(op, old, v) is not None
    for op in ("and", "or", "xor", "umax", "umin", "fadd", None):
        assert _rmw_written_value(op, old, v) is None
    assert _rmw_written_value("or", old, IntVal(0)) is old
    assert _rmw_written_value("add", None, v) is None


@pytest.mark.parametrize("op,operand", [("max", 7), ("min", -3)])
def test_written_value_reaches_a_relaxed_reader(op, operand):
    """pid 0 stores data, then a relaxed max/min writes ``operand`` to a
    zero flag; another pid polls the flag and reads data on seeing it. The
    relaxed write publishes nothing, so the data race is reported only if
    the write part carries the right value."""
    flag = counter(0)
    obs = Int(f"written_obs_{op}")
    producer, consumer = PID0 == 0, PID0 != 0
    records = [
        store(IntVal(DATA), event_id=0, active=producer),
        rmw(
            IntVal(flag.data_ptr()),
            Int(f"written_old_{op}"),
            op=op,
            operand=IntVal(operand),
            event_id=1,
            program_seq=1,
            sem="relaxed",
            tensor=flag,
            active=producer,
        ),
        _add(
            flag,
            obs,
            event_id=2,
            program_seq=2,
            operand=IntVal(0),
            sem="acquire",
            active=consumer,
        ),
        load(
            IntVal(DATA),
            event_id=3,
            program_seq=3,
            active=consumer,
            mask=obs == operand,
        ),
    ]
    pairs = report_pairs(solve(records, fence_between=every_fence).find_races())
    assert (0, 3) in pairs


def _add(tensor, old, **kwargs):
    kwargs.setdefault("sem", "relaxed")
    return rmw(
        IntVal(tensor.data_ptr()),
        old,
        op="add",
        operand=kwargs.pop("operand", IntVal(1)),
        tensor=tensor,
        **kwargs,
    )


# ──────────────────── last block done ────────────────────


def _last_block(sem, *, tensor=True):
    """partial[pid] = ...; old = atomic_add(counter, 1, sem); if old == 3:
    read partial[0:4], on a grid of 4."""
    c = counter(0)
    old, lane = Int("lbd_old"), Int("lbd_lane")
    records = [
        store(IntVal(PARTIAL) + PID0 * 4, event_id=0),
        _add(c, old, event_id=1, program_seq=1, sem=sem),
        load(IntVal(PARTIAL) + lane * 4, event_id=2, program_seq=2, mask=old == 3),
    ]
    if not tensor:
        records[1].tensor = None
    return solve(
        records,
        grid=(4, 1, 1),
        arange_dict={(0, 4): (lane, None)},
        fence_between=every_fence,
    )


def test_last_block_done_acq_rel_is_proved():
    solver = _last_block("acq_rel")
    assert solver._counting
    assert solver.find_races() == []


def test_last_block_done_relaxed_flips_to_race():
    reports = _last_block("relaxed").find_races()
    kinds = {(r.first_record.op_type, r.second_record.op_type) for r in reports}
    assert (Store, Load) in kinds or (Load, Store) in kinds


def test_unknown_initial_value_omits_counting_and_over_reports():
    solver = _last_block("acq_rel", tensor=False)
    assert not solver._counting
    assert solver.find_races()


# ──────────────────── counting distinctness ────────────────────


def test_single_winner_store_is_proved():
    c, old = counter(0), Int("winner_old")
    records = [
        _add(c, old, event_id=0),
        store(IntVal(DATA), event_id=1, program_seq=1, mask=old == 0),
    ]
    solver = solve(records, grid=(4, 1, 1))
    assert solver._counting and solver.find_races() == []
    records[0].tensor = None
    solver = solve(records, grid=(4, 1, 1))
    assert not solver._counting and solver.find_races()


@pytest.mark.parametrize("stride,races", [(4, False), (2, True)])
def test_work_queue_slots_are_disjoint_by_rank(stride, races):
    """idx = atomic_add(head, 1); store buf[idx]: the observation feeds the
    address, allowed exactly because the counting axiom fired."""
    head, old = counter(0), Int(f"wq_old_{stride}")
    records = [
        _add(head, old, event_id=0),
        store(IntVal(BUF) + old * stride, event_id=1, program_seq=1),
    ]
    solver = solve(records, grid=(4, 1, 1))
    assert solver._counting
    assert bool(solver.find_races()) is races


def test_uncounted_observation_in_an_address_is_unsupported():
    head, old = counter(0), Int("xchg_old")
    records = [
        rmw(
            IntVal(head.data_ptr()),
            old,
            op="xchg",
            operand=IntVal(7),
            event_id=0,
            sem="relaxed",
            tensor=head,
        ),
        store(IntVal(BUF) + old * 4, event_id=1, program_seq=1),
    ]
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="counting axiom"):
        solve(records, grid=(4, 1, 1))


def test_counter_that_may_wrap_on_a_symbolic_grid_asks_for_a_smaller_scope():
    """A work-queue counter whose capped maximum overflows int32 on a
    symbolic grid raises the retryable CountingScopeUnsupported; at a
    concrete grid that overflows, no smaller scope helps."""
    head, old = counter(2**31 - 8), Int("scope_old")
    records = [
        _add(head, old, event_id=0),
        store(IntVal(BUF) + old * 4, event_id=1, program_seq=1),
    ]
    with pytest.raises(CountingScopeUnsupported):
        solve(records, grid=(Int("scope_g"), 1, 1))
    assert solve(records, grid=(4, 1, 1))._counting
    with pytest.raises(UnsupportedSymbolicRaceQuery) as info:
        solve(records, grid=(16, 1, 1))
    assert not isinstance(info.value, CountingScopeUnsupported)


def test_rf_chain_keeps_a_non_adjacent_rank_race():
    """rank 0 stores X, rank 2 reads X, relaxed: a real race the closed world
    (init + the peer copy's write) alone could not justify."""
    c, old = counter(0), Int("chain_old")
    records = [
        _add(c, old, event_id=0),
        store(IntVal(DATA), event_id=1, program_seq=1, mask=old == 0),
        load(IntVal(DATA), event_id=2, program_seq=2, mask=old == 2),
    ]
    solver = solve(records, grid=(4, 1, 1))
    assert solver._counting and solver.find_races()


def test_rank0_winner_duplicate_lanes_still_race():
    """The two copies of the one dynamic RMW in one instance are one
    operation: the winner's own colliding lanes must stay reportable."""
    c, old = counter(0), Int("rank0_old")
    records = [
        _add(c, old, event_id=0),
        store([IntVal(DATA), IntVal(DATA)], event_id=1, program_seq=1, mask=old == 0),
    ]
    solver = solve(records, grid=(4, 1, 1))
    assert solver._counting and solver.find_races()


# ──────────────────── counting guards ────────────────────


def test_counting_guards_omit_the_axiom():
    c = counter(0)
    masked = _add(c, Int("masked_old"), event_id=0, active=PID0 == 0)
    nonconstant = _add(c, Int("nc_old"), event_id=0, operand=PID0 + 1)
    overlapped = [
        _add(c, Int("ovl_old"), event_id=0),
        store(IntVal(c.data_ptr()), event_id=1, program_seq=1),
    ]
    for records in ([masked], [nonconstant], overlapped):
        assert not solve(records, grid=(4, 1, 1))._counting


def test_possible_wraparound_omits_counting():
    """An INT32_MAX counter wraps on hardware; the unbounded-Int model does
    not, so the axiom is omitted and the wrap-gated store races."""
    c, old = counter(2**31 - 1), Int("wrap_old")
    records = [
        _add(c, old, event_id=0),
        store(IntVal(DATA), event_id=1, program_seq=1),
        store(IntVal(DATA), event_id=2, program_seq=2, mask=old == -(2**31)),
    ]
    solver = solve(records, grid=(4, 1, 1))
    assert not solver._counting
    assert {(1, 2), (2, 2)} <= report_pairs(solver.find_races())
    sane = solve([_add(counter(0), Int("sane_old"), event_id=0)], grid=(Int("g"), 1, 1))
    assert sane._counting


# ──────────────────── release sequences ────────────────────


def _release_chain(producer_sem, *, modeled_middle=True):
    """pid 0: store data; xchg(flag, 10, producer_sem). Other pids:
    cas(flag, 10, 15, relaxed) as the middle link; fetch = add(flag, 0,
    acquire); if fetch == 15: read data. The sw edge reaches the reader only
    through the relaxed middle link."""
    flag = counter(0)
    base = IntVal(flag.data_ptr())
    producer, consumer = PID0 == 0, PID0 != 0
    fetch = Int("rc_fetch")
    if modeled_middle:
        middle = cas(
            base,
            IntVal(10),
            IntVal(15),
            Int("rc_mid"),
            event_id=2,
            program_seq=2,
            sem="relaxed",
            tensor=flag,
            active=consumer,
        )
    else:
        middle = rmw(
            base,
            Int("rc_mid"),
            op="or",
            operand=IntVal(5),
            event_id=2,
            program_seq=2,
            sem="relaxed",
            tensor=flag,
            active=consumer,
        )
    return [
        store(IntVal(DATA), event_id=0, active=producer),
        rmw(
            base,
            Int("rc_x"),
            op="xchg",
            operand=IntVal(10),
            event_id=1,
            program_seq=1,
            sem=producer_sem,
            tensor=flag,
            active=producer,
        ),
        middle,
        rmw(
            base,
            fetch,
            op="add",
            operand=IntVal(0),
            event_id=3,
            program_seq=3,
            sem="acquire",
            tensor=flag,
            active=consumer,
        ),
        load(
            IntVal(DATA), event_id=4, program_seq=4, active=consumer, mask=fetch == 15
        ),
    ]


def test_release_sequence_through_an_rmw_chain_is_proved():
    records = _release_chain("release")
    assert solve(records, fence_between=every_fence).find_races() == []
    # Without fences the producer's store is not ordered before its release.
    assert solve(records).find_races()


@pytest.mark.parametrize("sem,modeled_middle", [("relaxed", True), ("release", False)])
def test_broken_release_sequence_flips_to_race(sem, modeled_middle):
    """A relaxed publisher heads no release sequence; an unmodeled middle
    link opens rf_unknown instead of extending the chain."""
    records = _release_chain(sem, modeled_middle=modeled_middle)
    assert solve(records, fence_between=every_fence).find_races()


def test_torn_width_modeled_writer_opens_the_escape():
    """An 8-byte modeled xchg over a 4-byte modeled reader is rf-incompatible:
    it counts as an unmodeled overlapping writer."""
    flag = counter(0)
    base = IntVal(flag.data_ptr())
    seen = Int("torn_seen")
    records = [
        rmw(
            base,
            Int("torn_w"),
            op="xchg",
            operand=IntVal(10),
            event_id=0,
            sem="relaxed",
            elem_size=8,
        ),
        rmw(
            base,
            seen,
            op="add",
            operand=IntVal(0),
            event_id=1,
            program_seq=1,
            sem="acquire",
            tensor=flag,
        ),
        store(IntVal(DATA), event_id=2, program_seq=2, mask=seen == 10),
        store(IntVal(DATA), event_id=3, program_seq=3),
    ]
    solver = solve(records, grid=(4, 1, 1))
    assert any(
        e.idx in solver.rf_unknown_source
        for e in solver.events
        if e.record is records[1]
    )
    assert solver.find_races()


# ──────────────────── awaits ────────────────────


@pytest.mark.parametrize("initial", [0, 1])
def test_await_exit_controls_the_later_publication(initial):
    """Each instance waits for its own flag to be 1, then sets the peer's
    flag. With zero flags only a causal cycle (each publication justifying
    the other's exit) could terminate: the launch premises are infeasible."""
    flags = torch.full((2,), initial, dtype=torch.int32)
    base = IntVal(flags.data_ptr())
    poll = Int(f"await_poll_{initial}")
    records = [
        rmw(
            base + 4 * PID0,
            poll,
            op="add",
            operand=IntVal(0),
            event_id=0,
            sem="relaxed",
            tensor=flags,
        ),
        rmw(
            base + 4 * (1 - PID0),
            Int(f"await_pub_{initial}"),
            op="xchg",
            operand=IntVal(1),
            event_id=1,
            program_seq=1,
            sem="relaxed",
            tensor=flags,
        ),
    ]
    records[1].premises = (poll == 1,)  # the await's exit
    records[1].causal_constraints = ()  # an exit premise is no value dependency
    records[1].await_dependencies = (0,)
    solver = solve(records)
    assert solver._base_solver().check() == z3.sat
    assert solver.check_feasibility() is bool(initial)
    # Without the control edge the cycle would certify termination.
    records[1].await_dependencies = ()
    assert solve(records).check_feasibility()


@pytest.mark.parametrize("pre_exit", [True, False])
def test_pre_exit_representative_keeps_its_poll_closed_world(pre_exit):
    """The value-model-free twin of an identity-RMW poll writes at the polled
    location; as a pre-exit representative it opens no rf_unknown escape for
    its own poll and is not a same-instance race partner of it."""
    flag = counter(0)
    base = IntVal(flag.data_ptr())
    poll = rmw(
        base,
        Int(f"rep_poll_{pre_exit}"),
        op="add",
        operand=IntVal(0),
        event_id=0,
        sem="acquire",
        tensor=flag,
    )
    rep = rmw(base, event_id=1, sem="acquire", tensor=flag)
    rep.pre_exit = pre_exit
    solver = solve([poll, rep], grid=(1, 1, 1))
    assert bool(solver.rf_unknown_source) is not pre_exit
    assert solver.find_races() == []
