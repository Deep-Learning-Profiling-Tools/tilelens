"""The compiled race detector's leaves, await rules and fence rule."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from z3 import And, Solver, simplify, substitute, unsat

from tilelens.clients.race_detector.compiled.analysis import analyze_graph
from tilelens.clients.race_detector.compiled.awaits import awaited_accesses
from tilelens.clients.race_detector.compiled.global_records import Encoder
from tilelens.clients.race_detector.compiled.leaves import (
    LaunchTensor,
    RaceLeaves,
    RaceRefusal,
)
from tilelens.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from tilelens.clients import pids
from tilelens.ir.launch import tensor_facts
from tilelens.ir.lowering import truncating_divrem
from tilelens.ir.ttir_reader import Arange, Bin, Cmp, Const, DataDep, Param, Pid
from tilelens.ir.ttir_reader import Select as TSelect
from tilelens.ir.ttir_reader import parse_ttir

from . import compiled_kernels as K


def _launch(**tensors) -> dict[str, LaunchTensor]:
    return {
        k: LaunchTensor(tensor_facts(t), tuple(t.tolist())) for k, t in tensors.items()
    }


def test_the_solver_renames_a_lowered_pid_per_copy():
    """RaceLeaves' pids are the variables the two-copy solver renames: a
    pid-dependent address must change under each copy's rename, or both
    copies would be one program and every cross-program race missed."""
    out = torch.zeros(4 * K.B, dtype=torch.int32)
    enc = Encoder(K.graph("raw"), {}, _launch(x=out[: K.B], out=out), {}).encode()
    record = next(r for r in enc.records if "pid_0" in str(r.addr_expr))
    solver = TwoCopySymbolicHBSolver(
        enc.records, grid=(2, 1, 1), arange_dict=enc.arange_dict
    )
    for ctx in (solver.ctx_a, solver.ctx_b):
        renamed = substitute(record.addr_expr, *ctx.pid_substitutions)
        assert not renamed.eq(record.addr_expr)
        assert f"pid_{ctx.label}_0" in str(renamed)


def test_lanes_are_keyed_by_dim_and_extent():
    graph = K.graph("tile2d")
    leaves = RaceLeaves(graph, {}, {})
    leaves.value(graph.accesses[0].offset)
    # one make_range read along two dims: two lanes, each in [0, B)
    assert sorted(leaves.arange_dict) == [(0, K.B, "ttir", 0), (0, K.B, "ttir", 1)]
    # the reader's tile shape; each dimension named by its lane
    assert graph.accesses[0].shape == (K.B, K.B)
    position = Encoder(graph, {}, {}, {}).lane_position(0)
    assert [str(v) for v in position] == [f"ttir_lane_d{d}_n{K.B}" for d in (0, 1)]


def test_params_snapshots_and_unmodeled_values():
    graph = K.graph("scatter")
    leaves = RaceLeaves(replace(graph), {"idx": 0}, {})
    # a launch value is read signed at the argument's width
    strided = K.graph("strided")
    assert (
        RaceLeaves(strided, {"stride": 2**32 - 1}, {}).param(Param("stride")).as_long()
        == -1
    )
    assert (
        str(RaceLeaves(strided, None, {}).param(Param("stride"))) == "ttir_param_stride"
    )
    with pytest.raises(RaceRefusal) as exc:
        leaves.data_dep(DataDep("loaded value"))
    assert exc.value.kind == "unmodeled-value"
    # a snapshot is read at the load's width, signed, or not at all
    load = graph.accesses[0]
    for dtype, bits, first, problem in (
        (torch.int64, 32, 1, "read as i32"),
        (torch.int8, 8, -1, None),
        (torch.uint8, 8, 127, None),
        (torch.uint8, 8, 128, "signed i8 reading"),
        (torch.bool, 1, 1, None),
    ):
        t = torch.tensor([first, 0], dtype=dtype)
        leaves = RaceLeaves(graph, {}, _launch(idx=t))
        access = replace(load, elem_bits=bits)
        found = leaves.snapshot_problem(access)
        assert (found is None) if problem is None else problem in found
        if problem is None:
            assert leaves.snapshot(access)[1][0] == first


_P, _NEG = Pid(0), Bin("-", Const(0), Pid(0))


@pytest.mark.parametrize(
    "term, stride, branches",
    [
        (Bin("//", _P, Const(4)), 1, False),
        (Bin("u%", _P, Const(-3)), 1, False),
        (Bin("//", Arange("r", 0, 16, 0), Const(4)), 1, False),
        (Bin("%", _P, Param("stride")), 0, True),  # the remainder at b == 0
        (Bin("//", _NEG, Const(3)), 1, False),
        (Bin("//", _NEG, Param("stride")), -3, False),
        (Bin("%", _NEG, Const(-3)), 1, False),
        (Bin("%", Bin("-", _P, Const(5)), Const(4)), 1, True),  # sign unknown
        (
            Bin("//", TSelect(Cmp("slt", _P, Const(2)), Const(3), _P), Const(2)),
            1,
            False,
        ),
    ],
)
def test_division_drops_sign_branches_only_where_exact(term, stride, branches):
    """RaceLeaves' ``//`` / ``%`` are Euclidean where the operands' bounds
    make them the IR's truncating ones, and equal to the lowering's default
    under the domains every query assumes."""
    leaves = RaceLeaves(K.graph("strided"), {"stride": stride}, {})
    bounded = leaves.value(term)
    a, b = leaves.value(term.a), leaves.value(term.b)
    exact = truncating_divrem(term, a, b)
    # a branch beyond the operands' own
    assert (
        str(bounded).count("If") > str(a).count("If") + str(b).count("If")
    ) == branches
    s = Solver()
    s.add(pids.pid_var(0) >= 0, bounded != exact)
    for (start, end, *_), (var, _) in leaves.arange_dict.items():
        s.add(And(var >= start, var < end))
    assert s.check() == unsat


def _module(*body: str, flag: str = "i32") -> str:
    return (
        "module {\n  tt.func public @k(%flag: !tt.ptr<"
        + flag
        + '> loc("flag"("k.py":1:1)), '
        '%out: !tt.ptr<i32> loc("out"("k.py":1:1))) attributes {noinline = false} {\n    '
        "%c0 = arith.constant 0 : i32\n    %c1 = arith.constant 1 : i32\n    "
        "%t = arith.constant true\n    "
        + "\n    ".join(body)
        + "\n    tt.return\n  }\n}\n"
    )


def _spin(*cond: str, body: str = "scf.yield", flag: str = "i32") -> str:
    return _module("scf.while : () -> () {", *cond, "} do {", body, "}", flag=flag)


_POLL = "%o = tt.load %flag {isVolatile = true} : !tt.ptr<i32>"
_NE = "%c = arith.cmpi ne, %o, %c1 : i32"
_RMW = "%o = tt.atomic_rmw {op}, acquire, gpu, %flag, {v}, %t : (!tt.ptr<i32>, i32, i1) -> i32"


@pytest.mark.parametrize(
    "text",
    [
        _spin(_POLL, _NE, "scf.condition(%c)"),
        _spin(_RMW.format(op="add", v="%c0"), _NE, "scf.condition(%c)"),
        _spin(_RMW.format(op="xor", v="%c0"), _NE, "scf.condition(%c)"),
        _spin(
            "%o = tt.atomic_cas acquire, gpu, %flag, %c0, %c1 : (!tt.ptr<i32>, i32, i32) -> i32",
            "%c = arith.cmpi ne, %o, %c0 : i32",
            "scf.condition(%c)",
        ),
    ],
)
def test_a_spin_awaits_its_exit_value(text):
    graph = parse_ttir(text, multipath=True, while_regions=True)
    ((index, exit_pred),) = awaited_accesses(graph).items()
    assert index == 0 and type(exit_pred).__name__ == "Not"


@pytest.mark.parametrize(
    "text, why",
    [
        (
            _module(
                "%r = scf.while (%a = %c0) : (i32) -> i32 {",
                _POLL,
                _NE,
                "scf.condition(%c) %o : i32",
                "} do {",
                "^bb0(%b: i32):",
                "scf.yield %b : i32",
                "}",
            ),
            "carries values",
        ),
        (
            _spin(
                "scf.while : () -> () {",
                _POLL,
                _NE,
                "scf.condition(%c)",
                "} do {",
                "scf.yield",
                "}",
                "scf.condition(%t)",
            ),
            "nested",
        ),
        (
            _spin(
                _POLL,
                _NE,
                "scf.condition(%c)",
                body="tt.store %out, %c1 : !tt.ptr<i32>\n scf.yield",
            ),
            "scf.yield only",
        ),
        (
            _spin(_POLL, "tt.store %out, %c1 : !tt.ptr<i32>", _NE, "scf.condition(%c)"),
            "stores",
        ),
        (
            _spin(_POLL, "%p = tt.load %out : !tt.ptr<i32>", _NE, "scf.condition(%c)"),
            "exactly one",
        ),
        (
            _spin(
                "%o = tt.load %flag : !tt.ptr<f32>",
                "%one = arith.constant 1.0 : f32",
                "%c = arith.cmpf one, %o, %one : f32",
                "scf.condition(%c)",
                flag="f32",
            ),
            "float",
        ),
        (
            _spin(_RMW.format(op="add", v="%c1"), _NE, "scf.condition(%c)"),
            "writes the awaited",
        ),
        (
            _spin(_RMW.format(op="exch", v="%c0"), _NE, "scf.condition(%c)"),
            "writes the awaited",
        ),
        (
            _spin(_POLL, "%c = arith.trunci %o : i32 to i1", "scf.condition(%c)"),
            "not a comparison",
        ),
        (
            _spin(
                _POLL,
                "%e = arith.addi %o, %c1 : i32",
                "%c = arith.cmpi ne, %o, %e : i32",
                "scf.condition(%c)",
            ),
            "does not depend on it",
        ),
    ],
)
def test_a_while_that_is_no_spin_is_spin_shape(text, why):
    graph = parse_ttir(text, multipath=True, while_regions=True)
    with pytest.raises(RaceRefusal) as exc:
        awaited_accesses(graph)
    assert exc.value.kind == "spin-shape" and why in exc.value.message


@pytest.mark.parametrize(
    "reset, race",
    [
        ("", None),
        # a cta-scope reset of the flag races the failed polls (corner A)
        ("%r = tt.atomic_rmw exch, relaxed, cta, %flag, %c0, %t", "WAW"),
        ("%r = tt.atomic_rmw exch, relaxed, gpu, %flag, %c0, %t", None),
        # a plain read of the flag races them too (corner B)
        ("%f = tt.load %flag : !tt.ptr<i32>", "WAR"),
    ],
)
def test_the_failed_polls_of_an_await_keep_their_races(reset, race):
    """Program 0 writes, then releases a flag that program 1 awaits; what
    program 0 does to the flag before the release meets the failed polls,
    which only the pre-exit representative stands for."""
    if reset.startswith("%r"):
        reset += " : (!tt.ptr<i32>, i32, i1) -> i32"
    text = _module(
        "%pid = tt.get_program_id x : i32",
        "%isp = arith.cmpi eq, %pid, %c0 : i32",
        "scf.if %isp {",
        "tt.store %out, %c1 : !tt.ptr<i32>",
        reset,
        "ttg.barrier all",
        "%x = tt.atomic_rmw exch, release, gpu, %flag, %c1, %t"
        " : (!tt.ptr<i32>, i32, i1) -> i32",
        "} else {",
        "scf.while : () -> () {",
        _RMW.format(op="add", v="%c0"),
        _NE,
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
        "ttg.barrier all",
        "%v = tt.load %out : !tt.ptr<i32>",
        "}",
    )
    graph = parse_ttir(text, multipath=True, while_regions=True)
    zero = torch.zeros(1, dtype=torch.int32)
    out = analyze_graph(graph, {}, _launch(flag=zero, out=zero.clone()), (2, 1, 1))
    if race is None:
        assert out.status == "proved"
        return
    (report,) = out.reports
    assert report.race_type.value == race
    assert report.second_record.debug_name.endswith(":pre-exit")


@pytest.mark.parametrize(
    "space, ordered",
    [
        ("all", True),
        ("global_read|global_write", True),
        ("local", False),
        ("global_read", False),
    ],
)
def test_only_a_global_barrier_orders_global_memory(space, ordered):
    text = _module(
        "%r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>",
        "%s = tt.splat %out : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>",
        "%p = tt.addptr %s, %r : tensor<16x!tt.ptr<i32>>, tensor<16xi32>",
        "tt.store %p, %r : tensor<16x!tt.ptr<i32>>",
        f"ttg.barrier {space}",
        "%q = tt.addptr %out, %c1 : !tt.ptr<i32>, i32",
        "%v = tt.load %q : !tt.ptr<i32>",
    )
    graph = parse_ttir(text, multipath=True, while_regions=True)
    out = torch.zeros(16, dtype=torch.int32)
    flag = torch.zeros(1, dtype=torch.int32)
    outcome = analyze_graph(graph, {}, _launch(out=out, flag=flag), (1, 1, 1))
    assert (outcome.status == "proved") is ordered


def test_fence_epochs_count_the_barriers_before_an_execution():
    def epochs(name, params, **tensors):
        encoder = Encoder(K.graph(name), params, _launch(**tensors), {})
        enc = encoder.encode()
        out = [enc.fences.epoch(r) for r in enc.records]
        return [e if isinstance(e, int) else simplify(e) for e in out]

    t = torch.zeros(4 * K.B, dtype=torch.int32)
    store, load, _ = epochs("loop_buffer_fenced", {"n": 4}, t=t, u=t.clone())
    assert str(store) == "ttir_loop_k" and str(load) == "1 + ttir_loop_k"
    before, after, _ = epochs(
        "store_flag_fence_load", {"flag": 1}, hist=t, out=t.clone()
    )
    assert before == 0
    assert "fence_epoch_unknown" in str(after)  # behind a conditional barrier
