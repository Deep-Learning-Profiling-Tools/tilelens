"""The compiled race detector's analysis of host-compiled TTIR: one race
per class, a proof per rung, fences, awaits, CAS operands, dependency order,
multipath graphs and the width obligations (``analysis.analyze_graph`` over
``compiled_kernels``, compiled at test time)."""

from __future__ import annotations

import pytest
import torch

from tilelens.clients.race_detector.compiled.analysis import analyze_graph
from tilelens.clients.race_detector.compiled.leaves import LaunchTensor
from tilelens.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from tilelens.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from tilelens.ir.launch import tensor_facts

from . import compiled_kernels as K

B = K.B


def i32(n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.int32)


def f32(n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.float32)


def run(name, tensors, params=None, grid=(2, 1, 1), snap=(), timeout_ms=None, **cx):
    """Analyze kernel ``name`` for a launch: ``tensors`` (the caller keeps
    them alive, so no two share an address), ``snap`` the ones whose
    pre-launch contents are captured."""
    launch = {
        k: LaunchTensor(
            tensor_facts(t), tuple(t.reshape(-1).tolist()) if k in snap else None
        )
        for k, t in tensors.items()
    }
    kw = (
        {}
        if timeout_ms is None
        else dict(t0_timeout_ms=timeout_ms, t1_timeout_ms=timeout_ms)
    )
    return analyze_graph(K.graph(name, **cx), params or {}, launch, grid, **kw)


def races(out) -> set[tuple[str, tuple[int, int], tuple[str, str]]]:
    return {
        (
            r.race_type.value,
            tuple(sorted((r.witness_grid_a[0], r.witness_grid_b[0]))),
            (r.first_record.tensor_name, r.second_record.tensor_name),
        )
        for r in out.reports
    }


@pytest.mark.parametrize(
    "name, race",
    [
        ("waw", ("WAW", "out", "out")),
        ("raw", ("RAW", "x", "x")),
        ("war", ("WAR", "x", "x")),
        ("atomic_vs_load", ("RAW", "x", "x")),
    ],
)
def test_each_race_class_between_two_programs(name, race):
    tensors = {"x": i32(B), "out": i32(4 * B)}
    out = run(name, {k: v for k, v in tensors.items() if k in K.SPECS[name][0]})
    assert out.status == "races" and not out.widened
    kind, a, b = race
    assert {(t, pids, names) for t, pids, names in races(out)} == {
        (kind, (0, 1), (a, b))
    }
    # the witness lines are the kernel's source lines
    lines = {r.first_record.source_location[1] for r in out.reports}
    assert all(line > 0 for line in lines)


def test_rungs():
    # T0: any parameters and grid, with the width assumption it needs
    out = run("add", {"x": f32(1024), "y": f32(1024), "out": f32(1024)}, {"n": 1000})
    assert (out.status, out.rung, out.overflow) == ("proved", "T0", (32,))
    out = run("atomic_only", {"x": i32(B)}, grid=(4, 1, 1))
    assert (out.status, out.rung, out.overflow) == ("proved", "T0", ())
    # pid * stride is nonlinear with the params symbolic: T1
    tensors = {"x": f32(64 * B), "out": f32(64 * B)}
    out = run("strided", tensors, {"stride": B}, grid=(4, 1, 1))
    assert (out.status, out.rung) == ("proved", "T1")
    out = run("strided", tensors, {"stride": B // 2}, grid=(4, 1, 1))
    assert ("WAW", (0, 1), ("out", "out")) in races(out)
    # race-free only up to two programs: the launch rung, with the
    # any-grid reports kept as evidence only
    out = run("two_slots", {"out": i32(2 * B)})
    assert (out.status, out.rung, out.reports) == ("proved", "T1-launch", [])
    assert out.grid_fragile
    out = run("two_slots", {"out": i32(2 * B)}, grid=(3, 1, 1))
    assert races(out) == {("WAW", (0, 2), ("out", "out"))}
    # T0's one encoding per tensor needs disjoint tensors: aliased views
    # race through the T1 byte addresses
    x = f32(4 * B + 1)
    tensors = {"x": x[: 4 * B], "y": f32(4 * B), "out": x[1:]}
    out = run("add", tensors, {"n": 4 * B}, grid=(4, 1, 1))
    assert ("WAR", (0, 1), ("x", "out")) in races(out)
    # a mask on loaded flags, no T0: the content-free attempt proves it for
    # any contents, so the proof does not claim to read them
    tensors = {"flags": torch.ones(4 * B, dtype=torch.int32), "out": i32(4 * B)}
    out = run("masked_table", tensors, {"stride": B}, grid=(4, 1, 1), snap=("flags",))
    assert (out.status, out.rung, out.content) == ("proved", "T1", False)
    # with no snapshot the mask is free: a race over it is only possible
    out = run("table_mask", {"flags": f32(4 * B), "out": i32(B)}, grid=(4, 1, 1))
    assert (out.status, out.reports) == ("races", []) and out.widened


def test_loaded_index_reads_the_launch_snapshot():
    perm = torch.randperm(4 * B, generator=torch.Generator().manual_seed(0))
    tensors = {"idx": perm.to(torch.int32), "out": i32(4 * B)}
    out = run("scatter", tensors, grid=(4, 1, 1), snap=("idx",))
    assert (out.status, out.rung, out.content) == ("proved", "T1", True)
    dup = dict(tensors, idx=tensors["idx"].clone())
    dup["idx"][5] = dup["idx"][40]  # programs 0 and 2 store one element
    out = run("scatter", dup, grid=(4, 1, 1), snap=("idx",))
    assert out.content and races(out) == {("WAW", (0, 2), ("out", "out"))}
    out = run("scatter", tensors, grid=(4, 1, 1))
    assert out.status == "unsupported"
    assert out.refusal.kind == "indirect-address"
    assert "no usable snapshot" in out.refusal.message
    # a view of a larger allocation may legally read outside its values
    buf = i32(8 * B)
    buf[: 4 * B] = tensors["idx"]
    out = run(
        "scatter",
        {"idx": buf[: 4 * B], "out": i32(4 * B)},
        grid=(4, 1, 1),
        snap=("idx",),
    )
    assert out.refusal.kind == "indirect-address" and "view" in out.refusal.message
    # a table the kernel writes is no snapshot of what it reads (program 1
    # stores out[5] after program 0 published idx[0] = 5)
    tensors = {"flag": i32(1), "idx": i32(1), "out": i32(8)}
    out = run("handoff_index", tensors, snap=("flag", "idx"))
    assert (
        out.refusal.kind == "indirect-address" and "writes 'idx'" in out.refusal.message
    )
    # masked-off lanes read `other`: lanes >= 8 of every program store out[0]
    idx = torch.arange(1, 4 * B + 1, dtype=torch.int32)
    out = run(
        "masked_gather_other",
        {"idx": idx, "out": i32(5 * B)},
        grid=(4, 1, 1),
        snap=("idx",),
    )
    assert "WAW" in {t for t, _, _ in races(out)}


@pytest.mark.parametrize(
    "name, params, tensors, expected",
    [
        # one program: a store and a reload of another lane's element
        ("store_then_load", {}, ("hist", "out"), "races"),
        # ordered by a global barrier (any grid races: the launch rung)
        ("store_fence_load", {}, ("hist", "out"), "proved"),
        # a conditional barrier orders nothing
        ("store_flag_fence_load", {"flag": 1}, ("hist", "out"), "races"),
        # in a loop: the barrier orders one iteration's store and reload ...
        ("loop_rows_fenced", {"n": 4}, ("t", "u"), "proved"),
        # ... and not one iteration's reload against the next one's store
        ("loop_buffer_fenced", {"n": 4}, ("t", "u"), "races"),
        ("loop_buffer_two_fences", {"n": 4}, ("t", "u"), "proved"),
    ],
)
def test_fences_order_one_program(name, params, tensors, expected):
    out = run(name, {k: i32(4 * B) for k in tensors}, params, grid=(1, 1, 1))
    assert out.status == expected
    if expected == "races":
        assert races(out) == {("RAW", (0, 0), (tensors[0], tensors[0]))}


def test_dependency_orders_a_read_modify_write_at_one_lane():
    assert run("inplace", {"x": i32(4 * B)}, grid=(4, 1, 1)).status == "proved"
    out = run("inplace_shift", {"x": i32(4 * B + 1)}, grid=(4, 1, 1))
    assert races(out) == {("WAR", (0, 1), ("x", "x"))}
    # through an atomic that runs wherever the store does, the order composes
    tensors = {"x": f32(4 * B), "out": f32(4 * B)}
    assert run("atomic_chain", tensors, {"n": 4 * B}, grid=(4, 1, 1)).status == "proved"
    # where the atomic is masked off, the stored value is not the load's
    out = run("masked_chain", {"x": i32(B), "y": i32(B)}, {"flag": 0}, grid=(1, 1, 1))
    assert races(out) == {("WAR", (0, 0), ("x", "x"))}


@pytest.mark.parametrize(
    "name, grid, race",
    [
        # B lanes store B values to out[pid]
        ("one_address", 2, True),
        ("one_address_masked", 1, False),
        # the columns of a broadcast row pointer store to one element
        ("broadcast_rows", 1, True),
        ("tile2d", 1, False),
    ],
)
def test_lanes_not_named_by_a_variable_collide(name, grid, race):
    out = run(name, {"out": i32(B * B)}, grid=(grid, 1, 1))
    assert races(out) == ({("WAW", (0, 0), ("out", "out"))} if race else set())
    assert race or out.status == "proved"


@pytest.mark.parametrize(
    "cx, payload_race",
    [
        ({}, False),
        ({"SEM": "relaxed"}, True),  # the poll acquires nothing
        ({"EXPECT": 0}, True),  # it exits on the initial value
        ({"ALL_POLL": False}, True),  # program 2 never polls
    ],
)
def test_an_awaited_flag_orders_the_payload(cx, payload_race):
    tensors = {"sem": i32(1), "payload": f32(B), "out": f32(2 * B)}
    out = run("comm_comp", tensors, grid=(3, 1, 1), snap=("sem",), **cx)
    assert out.assumes_termination
    if payload_race:
        assert ("RAW", "payload") in {(t, n[0]) for t, _, n in races(out)}
    else:
        assert out.status == "proved"


def test_a_wrapping_exit_value_drops_the_await():
    # program 4 expects (4 * 2**30) // 2**30, which wraps to 0: it does not
    # wait for program 0's publication
    tensors = {"flag": i32(1), "data": i32(B), "out": i32(5 * B)}
    out = run("exit_wrap", tensors, {"big": 2**30}, grid=(5, 1, 1), snap=("flag",))
    assert ("RAW", (0, 4), ("data", "data")) in races(out)
    out = run("exit_wrap", tensors, {"big": 2**30}, grid=(2, 1, 1), snap=("flag",))
    assert (out.status, out.rung) == ("proved", "T1-launch")


def test_a_while_that_is_no_spin_refuses():
    out = run("counted_while", {"x": i32(B)}, {"n": 4}, grid=(1, 1, 1))
    assert out.status == "unsupported" and out.refusal.kind == "spin-shape"
    assert "carries values" in out.refusal.message


def test_cas_lock_and_operands():
    tensors = {"lock": i32(1), "data": i32(B)}
    out = run("cas_lock", tensors, snap=("lock",))
    assert (out.status, out.assumes_termination) == ("proved", True)
    out = run("cas_lock", tensors, snap=("lock",), SEM="relaxed")
    assert ("WAW", (0, 1), ("data", "data")) in races(out)
    # (pid == 0) as an int32 operand: the zero extension of a comparison
    out = run("cas_bool_operand", {"flag": i32(1), "out": i32(4)}, snap=("flag",))
    assert out.status == "proved"
    for name, why in (
        ("cas_cast_operand", "changes an integer's width"),
        ("cas_arith_operand", "arithmetic"),
    ):
        out = run(name, {"flag": i32(1), "data": i32(1)}, snap=("flag", "data"))
        assert out.status == "unsupported" and out.refusal.kind == "cas-value"
        assert why in out.refusal.message
    out = run("cas_in_loop", {"flag": i32(1)}, {"n": 2}, snap=("flag",))
    assert out.refusal.kind == "control-flow"


@pytest.mark.parametrize(
    "cx, data_race",
    [
        ({}, False),
        ({"PSEM": "relaxed"}, True),
        ({"CSEM": "relaxed"}, True),
        ({"SCOPE": "cta"}, True),
    ],
)
def test_a_failed_cas_acquires_a_cas_publication(cx, data_race):
    tensors = {"flag": i32(1), "data": i32(1), "out": i32(4)}
    out = run("cas_handoff", tensors, snap=("flag",), **cx)
    assert (
        ("RAW", (0, 1), ("data", "data")) in races(out)
        if data_race
        else out.status == "proved"
    )


def test_multipath_graphs():
    out = run("early_return", {"x": f32(4 * B), "out": f32(4 * B)}, {"T": 4 * B})
    assert (out.status, out.rung) == ("proved", "T0")
    tensors = {"x": f32(B), "out": f32(32 * B)}
    out = run("sequential_loops", tensors, {"n": 2}, grid=(4, 1, 1))
    assert (out.status, out.rung) == ("proved", "T1")
    out = run("sequential_loops", tensors, {"n": 2}, grid=(4, 1, 1), SHIFT=2)
    assert ("WAW", "out") in {(t, n[0]) for t, _, n in races(out)}
    tensors = {"x": f32(64 * B), "out": f32(64 * B)}
    out = run("nested_loops", tensors, {"n": 2, "m": 3}, grid=(4, 1, 1))
    assert out.status == "proved"
    # an in-place update beside a loop it is not in: one execution, whose
    # load comes before its store (the other loop's iteration is no copy's)
    for in_loop in (True, False):
        tensors = {"x": f32(8 * B), "y": f32(4 * B)}
        out = run(
            "read_loop_then_inplace", tensors, {"n": 3}, grid=(4, 1, 1), IN_LOOP=in_loop
        )
        assert (out.status, out.rung) == ("proved", "T0")


def test_width_obligations():
    # A loop under a branch: its bounds wrap only where it is not reached.
    out = run("loop_under_guard", {"out": i32(8 * B)}, grid=(8, 1, 1))
    assert (out.status, out.rung, out.overflow) == ("proved", "T0", ())
    # A width premise holds only where a Select arm reads its term: pid *
    # 2**30 wraps at pid 2, where the other arm stores out[0] as pid 0 does.
    out = run("where_wrap", {"out": i32(4)}, grid=(3, 1, 1))
    assert races(out) == {("WAW", (0, 2), ("out", "out"))}
    # Assuming no overflow, two programs at most store: an any-grid proof,
    # qualified, that holds at a launch where nothing wraps ...
    out = run("wrapped_slots", {"out": i32(3)}, grid=(2, 1, 1))
    assert (out.status, out.rung, out.overflow) == ("proved", "T0", (32,))
    # ... and refuses at one where program 2 wraps to element 0.
    out = run("wrapped_slots", {"out": i32(3)}, grid=(3, 1, 1))
    assert (out.status, out.refusal.kind) == ("unsupported", "integer-overflow")
    # The qualifier names every width that can wrap, though no one
    # execution shows both.
    out = run("two_widths", {"out": i32(8)}, grid=(2, 1, 1))
    assert (out.status, out.rung, out.overflow) == ("proved", "T0", (32, 64))
    # The launch rung discharges them: pid * 2**31 wraps at this launch.
    tensors = {"x": i32(2), "out": i32(2 * B)}
    assert run("launch_wrap", tensors, {"big": 2**29}).rung == "T1-launch"
    out = run("launch_wrap", tensors, {"big": 2**30})
    assert out.status == "unsupported"
    assert out.refusal.kind == "integer-overflow"


def test_an_undecided_query_is_a_timeout():
    perm = torch.arange(4 * B, dtype=torch.int32)
    perm[5] = 40
    tensors = {"idx": perm, "out": i32(4 * B)}
    out = run("scatter", tensors, grid=(4, 1, 1), snap=("idx",), timeout_ms=1)
    assert out.status == "unsupported" and out.refusal.kind == "timeout"


def test_an_undecided_launch_grid_requery_is_a_timeout(monkeypatch):
    """When the requery at the launch's grid is not decided, the any-grid
    races (which may lie outside the launch's grid) are evidence only."""
    found = TwoCopySymbolicHBSolver.find_races

    def requery_times_out(self, **kw):
        if self.only_pairs is not None:
            raise UnsupportedSymbolicRaceQuery(
                "Z3 could not decide", reason_unknown="timeout"
            )
        return found(self, **kw)

    monkeypatch.setattr(TwoCopySymbolicHBSolver, "find_races", requery_times_out)
    out = run("two_slots", {"out": i32(2 * B)})
    assert (out.status, out.refusal.kind, out.reports) == ("unsupported", "timeout", [])
    assert "launch's grid was not decided" in out.refusal.message and out.grid_fragile


def test_a_cancelled_query_is_a_keyboard_interrupt(monkeypatch):
    """Z3's Ctrl+C handler cancels a query as unknown; the analysis raises
    instead of reading it as undecided and asking the next rung."""

    def interrupted(self, **kw):
        raise UnsupportedSymbolicRaceQuery(
            "Z3 could not decide", reason_unknown="interrupted from keyboard"
        )

    monkeypatch.setattr(TwoCopySymbolicHBSolver, "find_races", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run("waw", {"out": i32(B)})
