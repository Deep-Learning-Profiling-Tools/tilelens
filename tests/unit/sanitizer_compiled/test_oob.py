"""tilelens.clients.sanitizer.compiled.oob: the compiled sanitizer's checks.

CPU only: graphs come from the TTIR reader on the IR tests' corpus
(tests/unit/ir/ttir_corpus.py: ttir/ and reader_ttir/, compiled at test
time) or on small TTIR texts, or are built by hand;
launches are synthetic LaunchBindings, with TensorFacts read from CPU
tensors where a view's layout matters.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
import z3

from tilelens.clients.sanitizer.compiled import oob
from tilelens.clients.sanitizer.compiled.oob import (
    CheckResult,
    Finding,
    SanitizerKind,
    _in_view,
    check_graph,
    launch_key,
    readdressed,
)
from tilelens.clients.symbolic_engine import SymbolicClient
from tilelens.ir.launch import LaunchBinding, TensorFacts, tensor_facts
from tilelens.ir.ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    Bin,
    BoolBin,
    Cmp,
    Const,
    DataDep,
    FuncArg,
    LoopInfo,
    LoopVar,
    Observed,
    Param,
    Pid,
    TTIRKind,
    parse_ttir,
)
from tilelens.ir.verdict import Refusal, SourceLocation

from ..ir import ttir_corpus

K = SanitizerKind
REPO = Path(__file__).resolve().parents[3]
ADD = "ttir/golden_add_sm80.ttir"
MATMUL = "ttir/golden_matmul_s3_sm80.ttir"
TILE2D = "ttir/golden_tile2d_sm80.ttir"
PTR = 0x1000


def _text(name: str) -> str:
    return ttir_corpus.text(name)


def _graph(name: str) -> AccessGraph:
    return parse_ttir(_text(name))


def _module(
    body: str, args: str = "%p: !tt.ptr<i32>, %n: i32"
) -> tuple[AccessGraph, str]:
    """A minimal TTIR module (no locs, so its parameters read as arg0,
    arg1, ...) around ``body``'s op lines, and its text."""
    lines = "\n    ".join(line.strip() for line in body.strip().splitlines())
    text = (
        f"module {{\n  tt.func public @k({args}) attributes {{noinline = false}} {{\n"
        f"    {lines}\n    tt.return\n  }}\n}}\n"
    )
    return parse_ttir(text), text


def _line(text: str, needle: str) -> int:
    (line,) = [i for i, t in enumerate(text.splitlines(), 1) if needle in t]
    return line


def _facts(numel: int, elem_size: int = 4, data_ptr: int = PTR) -> TensorFacts:
    """A contiguous 1D tensor."""
    return TensorFacts(
        data_ptr, elem_size, numel, (numel,), (1,), "torch.float32", True
    )


def _bind(grid=(1, 1, 1), params=None, **tensors) -> LaunchBinding:
    return LaunchBinding(
        params=MappingProxyType(dict(params or {})),
        tensors=MappingProxyType(tensors),
        constexprs=MappingProxyType({}),
        raw_grid=grid,
        grid=grid,
        config=MappingProxyType({}),
    )


def _add(grid, n, **tensors) -> CheckResult:
    return check_graph(_graph(ADD), _bind(grid, {"n_elements": n}, **tensors))


def _all(numel: int, *names: str, elem_size: int = 4) -> dict[str, TensorFacts]:
    return {name: _facts(numel, elem_size) for name in names}


ADD_TENSORS = ("x_ptr", "y_ptr", "out_ptr")


def _clean(result: CheckResult) -> None:
    assert result.findings == ()
    assert result.refusal is None and result.abstained == ()


def _found(result: CheckResult) -> list[tuple[str, int]]:
    return [(f.kind, f.access_index) for f in result.findings]


def _synthetic(*accesses: AccessEvent, loop: LoopInfo | None = None, args=("p",)):
    return AccessGraph(
        kernel_name="synthetic",
        func_args=[FuncArg(a, True, 32) for a in args],
        accesses=accesses,
        loop=loop,
    )


def _access(offset, *, kind="load", base="p", mask=None, line=1, **kw) -> AccessEvent:
    return AccessEvent(kind, base, offset, mask, 32, None, line, **kw)


# ─────────────────────────── add (1D, masked) ───────────────────────────


def test_add_in_bounds_is_clean():
    _clean(_add((4, 1, 1), 4096, **_all(4096, *ADD_TENSORS)))


def test_add_unmasked_tail_is_out_of_bounds():
    """A mask bound (n_elements) past the tensor leaves the last block's
    tail unguarded."""
    r = _add((5, 1, 1), 10**9, **_all(4096, *ADD_TENSORS))
    assert r.refusal is None and r.abstained == ()
    assert _found(r) == [
        ("out-of-bounds", 0),
        ("out-of-bounds", 1),
        ("out-of-bounds", 2),
    ]
    text = _text(ADD).splitlines()
    for f, kind in zip(r.findings, ("load", "load", "store")):
        assert f.access_kind == kind and f.base_param == ADD_TENSORS[f.access_index]
        assert f.violation_offset >= 4096
        assert f.violation_address == PTR + f.violation_offset * 4
        # the witness is the state that reaches the offset
        lane = next(v for k, v in f.witness.items() if k.startswith("arange_"))
        assert f.witness["pid_0"] * 1024 + lane == f.violation_offset
        assert f"tt.{kind}" in text[f.line_no - 1]
        assert isinstance(f.loc, SourceLocation) and f.loc.line > 0
        assert "outside the tensor's 4096 elements" in f.detail


# ─────────────────────────── matmul (loop, 2D) ───────────────────────────

_MATMUL_PARAMS = {
    "M": 128,
    "N": 128,
    "K": 128,
    "stride_am": 128,
    "stride_bk": 128,
    "stride_cm": 128,
}


def test_matmul_in_bounds_and_oversized_grid():
    tensors = _all(128 * 128, "a_ptr", "b_ptr", "c_ptr", elem_size=2)
    _clean(check_graph(_graph(MATMUL), _bind((2, 2, 1), _MATMUL_PARAMS, **tensors)))
    # The A load has no row mask (only a K mask): too many row blocks
    # (pid_m = 2 with M = 128, BLOCK_M = 64) read rows past M.
    r = check_graph(_graph(MATMUL), _bind((3, 2, 1), _MATMUL_PARAMS, **tensors))
    assert ("out-of-bounds", 0) in _found(r) and r.refusal is None
    a = r.findings[0]
    assert a.base_param == "a_ptr" and a.witness["pid_0"] == 2
    assert 0 <= a.witness["iter_loop"] < 4  # K / BLOCK_K iterations run
    assert a.violation_address == PTR + a.violation_offset * 2


def test_modeled_branch_path_constrains_the_witness():
    """Only program 0 stores (``if pid == 0``): the path is modeled, so a
    later program's out-of-range offsets are no witness."""
    g = _graph("ttir/golden_pid_branch_sm80.ttir")
    b = _bind((4, 1, 1), {"n_elements": 1024}, x_ptr=_facts(1024), out_ptr=_facts(256))
    _clean(check_graph(g, b))
    b = _bind((4, 1, 1), {"n_elements": 1024}, x_ptr=_facts(1024), out_ptr=_facts(255))
    ((f,),) = [check_graph(g, b).findings]
    assert f.access_index == 1 and f.witness["pid_0"] == 0 and f.violation_offset == 255


def test_grid_stride_loop_with_a_pid_lower_bound():
    """``for row in range(pid, n_rows, 4)``: a loop bound that depends on
    the program id stays symbolic."""
    g = _graph("ttir/golden_grid_stride_sm80.ttir")
    params = {"n_rows": 8, "stride": 64}
    _clean(check_graph(g, _bind((4, 1, 1), params, **_all(8 * 64, "x_ptr", "out_ptr"))))
    r = check_graph(
        g, _bind((4, 1, 1), params, x_ptr=_facts(7 * 64), out_ptr=_facts(8 * 64))
    )
    ((f,),) = [r.findings]
    assert f.base_param == "x_ptr" and f.violation_offset >= 7 * 64
    # row = pid + 4 * iter reaches row 7 only
    assert f.witness["pid_0"] + 4 * f.witness["iter_loop"] == 7


def test_expanded_loop_carried_tiles_keep_their_lanes():
    """q[i, j, l] = x + j*N + l - i*N + k (N = 4, k the iteration): the
    reader's expanded iter_arg tile, one make_range on three dims."""
    g = _graph("reader_ttir/expand_iterarg_3d.ttir")
    r = check_graph(
        g, _bind((1, 1, 1), {"n": 2}, x_ptr=_facts(1000), out_ptr=_facts(64))
    )
    ((f,),) = [r.findings]
    assert f.access_index == 0 and f.violation_offset < 0
    lane = {int(k[-1]): v for k, v in f.witness.items() if k.startswith("arange_")}
    assert sorted(lane) == [0, 1, 2]
    k = f.witness["iter_loop"]
    assert 0 <= k < 2
    assert lane[1] * 4 + lane[2] - lane[0] * 4 + k == f.violation_offset


def test_select_of_pointers():
    """``tl.where(offs < n, x + offs, x + 100)`` over 16 lanes."""
    g = _graph("reader_ttir/where_pointer.ttir")
    _clean(check_graph(g, _bind(params={"n": 8}, x_ptr=_facts(101))))
    _clean(check_graph(g, _bind(params={"n": 16}, x_ptr=_facts(16))))
    ((f,),) = [check_graph(g, _bind(params={"n": 8}, x_ptr=_facts(100))).findings]
    assert f.violation_offset == 100


def test_aranges_along_one_dim_share_the_lane():
    """tl.arange(1, 65) - tl.arange(0, 64) is 1 on every lane: two
    make_range sites along one dim index one lane position, so positions 1
    and 0 of the two are no witness (the real kernel reads x[1] only)."""
    g, _ = _module(
        """
        %r1 = tt.make_range {end = 65 : i32, start = 1 : i32} : tensor<64xi32>
        %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %d = arith.subi %r1, %r0 : tensor<64xi32>
        %ps = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
        %a = tt.addptr %ps, %d : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
        %v = tt.load %a : tensor<64x!tt.ptr<i32>>"""
    )
    _clean(check_graph(g, _bind(arg0=_facts(2))))
    ((f,),) = [check_graph(g, _bind(arg0=_facts(1))).findings]
    assert f.violation_offset == 1
    # the witness names each arange by its range, at the one lane
    assert f.witness["arange_1_65"] == f.witness["arange_0_64"] + 1


def test_a_broadcast_extent_one_arange_keeps_its_position():
    """tl.arange(5, 6) broadcast to 64 lanes plus tl.arange(0, 64): the
    extent-1 range stays at its one position, every lane of the other
    range stays free."""
    g, _ = _module(
        """
        %r5 = tt.make_range {end = 6 : i32, start = 5 : i32} : tensor<1xi32>
        %b = tt.broadcast %r5 : tensor<1xi32> -> tensor<64xi32>
        %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
        %o = arith.addi %b, %r0 : tensor<64xi32>
        %ps = tt.splat %p : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
        %a = tt.addptr %ps, %o : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
        %v = tt.load %a : tensor<64x!tt.ptr<i32>>"""
    )
    _clean(check_graph(g, _bind(arg0=_facts(69))))
    ((f,),) = [check_graph(g, _bind(arg0=_facts(68))).findings]
    assert f.violation_offset == 68
    assert (f.witness["arange_5_6"], f.witness["arange_0_64"]) == (5, 63)


# ─────────────────── synthetic graphs ───────────────────


def test_reused_arange_rows_and_cols_are_independent():
    """One make_range on the row and the column dim is two variables: with
    offset = row - col, collapsing them makes the offset 0 (never OOB)."""
    r = "%shared_range"
    offset = Bin("-", Arange(r, 0, 4, dim=0), Arange(r, 0, 4, dim=1))
    ((f,),) = [check_graph(_synthetic(_access(offset)), _bind(p=_facts(64))).findings]
    assert f.violation_offset < 0


def _store_loop(lower, step, upper, offset):
    return _synthetic(
        _access(offset, kind="store", base="out", in_loop=True),
        loop=LoopInfo("%loop", lower=lower, upper=upper, step=step),
        args=("out",),
    )


def test_loop_nonzero_lower_no_false_positive():
    """for k in range(1, n): store(out + (k - 1)) writes 0..n-2: the
    iteration model never runs k = 0."""
    g = _store_loop(
        Const(1), Const(1), Param("n"), Bin("-", LoopVar("%loop"), Const(1))
    )
    _clean(check_graph(g, _bind(params={"n": 8}, out=_facts(7))))


def test_loop_step_skips_unrun_iterations():
    """for k in range(0, n, 2): store(out + k) writes only even offsets."""
    g = _store_loop(Const(0), Const(2), Param("n"), LoopVar("%loop"))
    _clean(check_graph(g, _bind(params={"n": 8}, out=_facts(7))))
    ((f,),) = [check_graph(g, _bind(params={"n": 8}, out=_facts(6))).findings]
    assert f.violation_offset == 6 and f.witness["iter_loop"] == 3


def test_descending_loop_refuses_non_positive_step():
    g = _store_loop(Const(10), Const(-1), Const(0), LoopVar("%loop"))
    r = check_graph(g, _bind(out=_facts(16)))
    assert r.findings == () and r.abstained == ((0, K.NON_POSITIVE_STEP),)
    assert r.refusal.kind == "non-positive-step" and "step is -1" in r.refusal.message


def test_step_that_depends_on_the_program_id():
    lower, upper = Const(0), Const(8)
    ok = _store_loop(lower, Bin("+", Pid(0), Const(1)), upper, LoopVar("%loop"))
    _clean(check_graph(ok, _bind((4, 1, 1), out=_facts(8))))
    bad = _store_loop(lower, Bin("-", Pid(0), Const(1)), upper, LoopVar("%loop"))
    r = check_graph(bad, _bind((4, 1, 1), out=_facts(8)))
    assert r.abstained == ((0, K.NON_POSITIVE_STEP),)
    assert "step can be" in r.refusal.message


def _flat_load(offset, numel, grid_x):
    return check_graph(
        _synthetic(_access(offset)), _bind((grid_x, 1, 1), p=_facts(numel))
    )


@pytest.mark.parametrize(
    "offset, numel, grid_x, oob",
    [
        # pid % 8 stays in [0, 8): the grouped-swizzle `pid % group_size_m`
        (Bin("%", Pid(0), Const(8)), 8, 64, None),
        (Bin("%", Pid(0), Const(8)), 7, 64, 7),
        # min(pid, 9) never leaves [0, 10)
        (Bin("min", Pid(0), Const(9)), 10, 1000, None),
        (Bin("min", Pid(0), Const(9)), 9, 1000, 9),
        # max(pid, 5) pins the floor at 5
        (Bin("max", Pid(0), Const(5)), 6, 4, None),
        (Bin("max", Pid(0), Const(5)), 5, 4, 5),
        # divsi truncates: (0 - 1) // 2 == 0, not Euclidean -1
        (Bin("//", Bin("-", Const(0), Pid(0)), Const(2)), 4, 2, None),
        # remsi keeps the dividend's sign: (0 - 1) % 2 == -1
        (Bin("%", Bin("-", Const(0), Pid(0)), Const(2)), 2, 2, -1),
    ],
)
def test_integer_semantics(offset, numel, grid_x, oob):
    r = _flat_load(offset, numel, grid_x)
    assert r.refusal is None
    assert [f.violation_offset for f in r.findings] == ([] if oob is None else [oob])


# ─────────────────── uncertainty: guarded, dropped, observed ───────────────────


def test_guarded_access_unsat_is_still_a_proof():
    g = _synthetic(_access(Pid(0), guarded=True))
    _clean(check_graph(g, _bind((4, 1, 1), p=_facts(4))))


def test_guarded_access_sat_abstains():
    """SAT under a branch condition the reader could not model may be a
    branch the launch never takes: abstain, never a witness."""
    g = _synthetic(_access(Bin("-", Pid(0), Const(1)), guarded=True))
    r = check_graph(g, _bind((4, 1, 1), p=_facts(4)))
    assert r.findings == () and r.abstained == ((0, K.UNMODELABLE_CONDITION),)
    assert r.refusal.kind == "unmodelable-condition"
    assert "possible out-of-bounds" in r.refusal.message
    assert r.refusal.message.startswith("TTIR line 1: ")


def test_exact_finding_kept_alongside_guarded_abstention():
    g = _synthetic(
        _access(Bin("-", Pid(0), Const(1)), guarded=True),
        _access(Bin("+", Pid(0), Const(100)), line=2),
    )
    r = check_graph(g, _bind((4, 1, 1), p=_facts(4)))
    assert _found(r) == [("out-of-bounds", 1)] and r.findings[0].line_no == 2
    assert r.findings[0].violation_offset >= 100
    assert r.abstained == ((0, K.UNMODELABLE_CONDITION),)


def test_mask_dropped_abstains_and_exact_findings_stay():
    """atomic_fmax's atomics are masked by loaded data (dropped as free)."""
    g = _graph("ttir/golden_atomic_fmax_sm80.ttir")
    assert [a.mask_dropped for a in g.accesses] == [False, True, True]
    r = check_graph(
        g,
        _bind((4, 1, 1), {"n_elements": 1024}, x_ptr=_facts(2048), out_ptr=_facts(10)),
    )
    assert r.findings == ()
    assert r.abstained == ((1, K.DATA_DEPENDENT_MASK), (2, K.DATA_DEPENDENT_MASK))
    assert (
        r.refusal.kind == "data-dependent-mask"
        and r.refusal.line_no == g.accesses[1].line_no
    )
    # the exact load's finding is kept
    r = check_graph(
        g, _bind((4, 1, 1), {"n_elements": 1024}, x_ptr=_facts(10), out_ptr=_facts(10))
    )
    assert _found(r) == [("out-of-bounds", 0)]
    assert r.abstained == ((1, K.DATA_DEPENDENT_MASK), (2, K.DATA_DEPENDENT_MASK))
    # UNSAT behind a dropped mask is still a proof
    _clean(
        check_graph(
            g,
            _bind(
                (4, 1, 1),
                {"n_elements": 1024},
                x_ptr=_facts(1024),
                out_ptr=_facts(1024),
            ),
        )
    )


def test_observation_gated_mask_and_path_abstain():
    atomic = AccessEvent("atomic_rmw", "c", Const(0), None, 32, None, 1)
    gate = Cmp("slt", Observed(0), Const(4), 32)
    g = AccessGraph(
        "k",
        [FuncArg("c", True, 32), FuncArg("p", True, 32)],
        [
            atomic,
            _access(Const(10), mask=gate, line=2),
            _access(Const(10), path=gate, line=3),
            _access(Const(0), mask=gate, line=4),  # in bounds: a proof
        ],
        None,
    )
    r = check_graph(g, _bind(c=_facts(1), p=_facts(4)))
    assert r.findings == ()
    assert r.abstained == ((1, K.DATA_DEPENDENT_MASK), (2, K.UNMODELABLE_CONDITION))


@pytest.mark.parametrize(
    "name", ["p4_observed_direct", "p4_observed_loop", "p4_observed_delta"]
)
def test_observation_in_address_refuses(name):
    """An atomic's old value in an address (directly, in a loop-carried
    pointer's offset0 or in its delta) would make any address reachable."""
    g = _graph(f"reader_ttir/{name}.ttir")
    r = check_graph(g, _bind((4, 1, 1), {"n": 8}, cnt_ptr=_facts(1), x_ptr=_facts(8)))
    assert r.findings == ()
    assert r.abstained == ((1, K.OBSERVATION_IN_ADDRESS), (2, K.OBSERVATION_IN_ADDRESS))
    assert r.refusal.kind == "observation-in-address"
    assert r.refusal.line_no == g.accesses[1].line_no
    assert r.refusal.message.endswith(
        "the address depends on the value an atomic observed"
    )


# ─────────────────────────── bindings and loops ───────────────────────────


def test_missing_bindings_refuse():
    tensors = _all(4096, *ADD_TENSORS)
    r = check_graph(_graph(ADD), _bind((4, 1, 1), **tensors))  # no n_elements
    assert r.findings == () and r.abstained == tuple(
        (i, K.MISSING_BINDING) for i in range(3)
    )
    assert "scalar argument 'n_elements' has no launch binding" in r.refusal.message
    b = _bind((4, 1, 1), {"n_elements": 4096}, **tensors)
    b = LaunchBinding(
        b.params, b.tensors, b.constexprs, None, None, b.config, "grid: boom"
    )
    r = check_graph(_graph(ADD), b)
    assert r.abstained == tuple((i, K.MISSING_BINDING) for i in range(3))
    assert "grid is unknown (unreadable: grid: boom)" in r.refusal.message


def test_an_unbound_loop_bound_refuses_the_loop():
    g = _graph("reader_ttir/iv_wrap.ttir")
    r = check_graph(g, _bind(params={"lo": 0}, x_ptr=_facts(10)))
    assert r.findings == () and r.abstained == ((0, K.MISSING_BINDING),)
    assert r.refusal.line_no == g.loop.line_no
    assert "scalar argument 'n'" in r.refusal.message


def test_unmodeled_values_and_unusable_facts_refuse():
    """Defensive: the reader never lets a DataDep into an address, and
    bind_launch never builds such facts."""
    g = _synthetic(_access(Bin("+", Const(0), DataDep("loaded value"))))
    r = check_graph(g, _bind(p=_facts(4)))
    assert r.abstained == ((0, K.UNMODELED_VALUE),)
    assert "an unmodeled value (loaded value)" in r.refusal.message
    bad = TensorFacts(PTR, 4, 4, (4,), (), "torch.float32", False)
    r = check_graph(_synthetic(_access(Const(0))), _bind(p=bad))
    assert r.abstained == ((0, K.MISSING_BINDING),)
    assert "unusable" in r.refusal.message


def test_exact_finding_kept_alongside_a_refusal():
    """A tensor without a binding refuses its access only: the other
    accesses' exact findings are returned with it."""
    r = _add((5, 1, 1), 10**9, **_all(4096, "x_ptr", "out_ptr"))
    assert _found(r) == [("out-of-bounds", 0), ("out-of-bounds", 2)]
    assert r.abstained == ((1, K.MISSING_BINDING),)
    assert r.refusal == Refusal(
        kind="missing-binding",
        message=r.refusal.message,
        line_no=r.findings[0].line_no + 3,
        loc=r.refusal.loc,
    )
    assert "pointer argument 'y_ptr' has no tensor binding" in r.refusal.message


_LOOP_THEN_TAIL = """
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c100 = arith.constant 100 : i32
    scf.for %i = %c0 to %c10 step %n : i32 {
      %a = tt.addptr %p, %i : !tt.ptr<i32>, i32
      tt.store %a, %c0 : !tt.ptr<i32>
    }
    %b = tt.addptr %p, %c100 : !tt.ptr<i32>, i32
    tt.store %b, %c0 : !tt.ptr<i32>
"""


def test_non_positive_step_refuses_only_the_loop():
    g, text = _module(_LOOP_THEN_TAIL)
    r = check_graph(g, _bind(params={"arg1": 0}, arg0=_facts(10)))
    assert r.abstained == ((0, K.NON_POSITIVE_STEP),)
    assert r.refusal.line_no == _line(text, "scf.for")
    assert _found(r) == [("out-of-bounds", 1)] and r.findings[0].violation_offset == 100
    r = check_graph(g, _bind(params={"arg1": 2}, arg0=_facts(10)))
    assert _found(r) == [("out-of-bounds", 1)] and r.abstained == ()


def test_zero_trip_loop_has_no_footprint():
    """An access in the loop whose offset does not mention the induction
    variable still runs only on iterations that run."""
    g, _ = _module(
        """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c100 = arith.constant 100 : i32
        scf.for %i = %c0 to %n step %c1 : i32 {
          %a = tt.addptr %p, %c100 : !tt.ptr<i32>, i32
          tt.store %a, %c0 : !tt.ptr<i32>
        }"""
    )
    assert g.accesses[0].in_loop
    _clean(check_graph(g, _bind(params={"arg1": 0}, arg0=_facts(10))))
    _clean(check_graph(g, _bind(params={"arg1": -5}, arg0=_facts(10))))
    ((f,),) = [check_graph(g, _bind(params={"arg1": 1}, arg0=_facts(10))).findings]
    assert f.violation_offset == 100 and f.witness["iter_loop"] == 0


def test_solver_unknown_abstains():
    """x^3 + y^3 == z^3 with positive x, y, z (no solution, which Z3 cannot
    prove in time) gates an always-OOB offset: unknown, never a proof."""
    x, y, z = Pid(0), Pid(1), Pid(2)

    def cube(t):
        return Bin("*", Bin("*", t, t), t)

    def pos(t):
        return Cmp("sgt", t, Const(0))

    fermat = BoolBin(
        "and",
        BoolBin("and", pos(x), pos(y)),
        BoolBin("and", pos(z), Cmp("eq", Bin("+", cube(x), cube(y)), cube(z))),
    )
    g = _synthetic(_access(Const(-1), mask=fermat))
    before = z3.get_param("timeout")
    r = check_graph(g, _bind((1 << 20,) * 3, p=_facts(4)), timeout_ms=50)
    assert z3.get_param("timeout") == before  # a per-Solver timeout only
    assert r.findings == () and r.abstained == ((0, K.SOLVER_UNKNOWN),)
    assert r.refusal.kind == "solver-unknown"
    assert "could not decide the out-of-bounds query" in r.refusal.message


# ─────────────────────────── view footprint ───────────────────────────


def test_strided_view_gap_is_out_of_bounds():
    """x[::2] has 4096 elements two apart: the odd offsets between them are
    outside the view, although every offset is below numel."""
    x = torch.empty(8192)[::2]
    tensors = {"x_ptr": tensor_facts(x), **_all(4096, "y_ptr", "out_ptr")}
    r = _add((4, 1, 1), 4096, **tensors)
    ((f,),) = [r.findings]
    assert (f.kind, f.access_index) == ("out-of-bounds", 0) and r.refusal is None
    assert f.violation_offset % 2 == 1 and 0 < f.violation_offset < 4096
    assert f.violation_address == x.data_ptr() + f.violation_offset * 4
    assert "shape (4096,) and strides (2,)" in f.detail


def _tile2d(stride_m: int, stride_n: int, inp, out) -> CheckResult:
    params = {"M": 64, "N": 64, "stride_m": stride_m, "stride_n": stride_n}
    return check_graph(
        _graph(TILE2D),
        _bind((2, 2, 1), params, in_ptr=tensor_facts(inp), out_ptr=tensor_facts(out)),
    )


def test_strided_views_indexed_by_their_strides_are_clean():
    # a column slice (strides (128, 1)) and a transpose (a dense permutation)
    sliced = torch.empty(64, 128)[:, :64]
    _clean(_tile2d(128, 1, sliced, torch.empty(64, 128)[:, :64]))
    t = torch.empty(64, 64).t()
    _clean(_tile2d(1, 64, t, torch.empty(64, 64).t()))
    # indexing the slice as if it were dense reads the gaps
    r = _tile2d(64, 1, sliced, torch.empty(64, 64))
    ((f,),) = [r.findings]
    assert f.access_index == 0 and 64 <= f.violation_offset % 128 < 128


def test_broadcast_stride0_tensor():
    """A row expanded to 64x64 (strides (0, 1)) has 64 distinct elements:
    offsets past them are out of bounds although below numel (4096)."""
    row = torch.empty(64).expand(64, 64)
    assert tensor_facts(row).numel == 4096
    _clean(_tile2d(0, 1, row, torch.empty(64, 64)))
    r = _tile2d(64, 1, row, torch.empty(64, 64))
    ((f,),) = [r.findings]
    assert f.access_index == 0 and 64 <= f.violation_offset < 4096


def _views():
    base = torch.empty(64)
    return {
        "contiguous": base,
        "step2": base[::2],
        "offset_step3": base[3:40:3],
        "column_slice": base.view(8, 8)[:, :5],
        "transpose": base.view(8, 8).t(),
        "grid_slice": base.view(8, 8)[::2, 1::3],
        "broadcast_rows": torch.empty(5).expand(3, 5),
        "broadcast_cols": torch.empty(4, 1).expand(4, 6),
        "gappy": base.as_strided((3, 3), (5, 1)),
        "overlapping": base.as_strided((3, 4), (2, 1)),
        "scalar": torch.empty(()),
        "empty": torch.empty(0),
        "empty_2d": torch.empty(3, 0),
        "channels_last": torch.empty(2, 3, 2, 2).to(memory_format=torch.channels_last),
        "expand_to_0": torch.empty(1).expand(0),
        "size1_odd_stride": base.as_strided((1, 4), (100, 1)),
        "mixed_broadcast": base.as_strided((3, 4, 5), (0, 10, 2)),
        "duplicate_strides": base.as_strided((3, 3), (3, 3)),
    }


def _eager_offsets(t: torch.Tensor) -> set[int]:
    """The element offsets the eager sanitizer admits for ``t``: the byte
    segments SymbolicClient._tensor_physical_addresses dispatches to
    (contiguous, storage-contiguous, inner-stride-1 slices, per element)."""
    if not t.numel():
        return set()
    base, item = t.data_ptr(), t.element_size()
    return {
        (a - base) // item
        for start, end, _ in SymbolicClient._tensor_physical_addresses(None, "t", t)
        for a in range(start, end + 1)
        if (a - base) % item == 0
    }


@pytest.mark.parametrize("name", list(_views()))
def test_view_footprint_matches_the_eager_legal_set(name):
    """The element offsets _in_view admits are exactly the ones the eager
    sanitizer's own dispatch admits."""
    t = _views()[name]
    facts = tensor_facts(t)
    eager = _eager_offsets(t)
    for e in range(-3, max([70, *eager]) + 8):
        solver = z3.Solver()
        solver.add(_in_view(z3.IntVal(e), facts))
        assert (solver.check() == z3.sat) == (e in eager), (name, e)


def test_empty_tensor():
    """No element is legal: every executed access is out of bounds, and an
    access that never executes is none."""
    g = _graph("ttir/golden_cas_sm80.ttir")
    empty = TensorFacts(PTR, 4, 0, (0,), (1,), "torch.int32", True)
    r = check_graph(g, _bind(lock_ptr=empty, out_ptr=_facts(1)))
    ((f,),) = [r.findings]
    assert (f.access_index, f.access_kind, f.violation_offset) == (0, "atomic_cas", 0)
    assert "the empty tensor" in f.detail
    _clean(_add((4, 1, 1), 0, **{n: empty for n in ADD_TENSORS}))


def test_reinterpreted_width_checks_every_byte():
    """An f32 pointer over a byte tensor (a width-changing reinterpret):
    offsets count f32 elements, the view counts bytes."""
    g = _graph(ADD)
    bytes_ = {n: _facts(4096, elem_size=1) for n in ADD_TENSORS}
    _clean(check_graph(g, _bind((1, 1, 1), {"n_elements": 1024}, **bytes_)))
    short = {**bytes_, "x_ptr": _facts(4095, elem_size=1)}
    r = check_graph(g, _bind((1, 1, 1), {"n_elements": 1024}, **short))
    ((f,),) = [r.findings]
    assert f.access_index == 0 and f.violation_offset == 1023
    assert f.violation_address == PTR + 1023 * 4


# ─────────────────────────── integer widths ───────────────────────────


def test_i32_wrap_is_integer_overflow_not_out_of_bounds():
    """(pid * S) * S wraps to 0 in i32 for S = 65536: every program stores
    x[0]. The unbounded reading's offsets are an overflow, not OOB."""
    g = _graph("reader_ttir/rv_i32_wrap.ttir")
    r = check_graph(g, _bind((4, 1, 1), {"S": 65536}, x_ptr=_facts(1)))
    ((f,),) = [r.findings]
    assert (f.kind, f.access_index) == ("integer-overflow", 0) and r.refusal is None
    assert f.violation_offset is None and f.violation_address is None
    assert not -(1 << 31) <= f.witness["value"] < 1 << 31
    assert (
        "arith.muli"
        in _text("reader_ttir/rv_i32_wrap.ttir").splitlines()[f.line_no - 1]
    )
    assert "i32 range" in f.detail
    # no wrap for a small S: a clean launch
    _clean(check_graph(g, _bind((4, 1, 1), {"S": 2}, x_ptr=_facts(16))))


def test_trunci_is_integer_overflow_not_out_of_bounds():
    """trunc_i32(pid_i64 * 2**32) is 0 for every pid."""
    g = _graph("reader_ttir/rv_trunci_alias.ttir")
    r = check_graph(g, _bind((2, 1, 1), x_ptr=_facts(1)))
    ((f,),) = [r.findings]
    assert (f.kind, f.witness["pid_0"], f.witness["value"]) == (
        "integer-overflow",
        1,
        1 << 32,
    )
    assert (
        "arith.trunci"
        in _text("reader_ttir/rv_trunci_alias.ttir").splitlines()[f.line_no - 1]
    )
    assert "truncation to i32" in f.detail
    _clean(check_graph(g, _bind((1, 1, 1), x_ptr=_facts(1))))


def test_loop_increment_overflow():
    """range(lo, n, 1 << 20) wraps its induction variable near INT32_MAX."""
    g = _graph("reader_ttir/iv_wrap.ttir")
    n = (1 << 31) - (1 << 19)
    r = check_graph(g, _bind(params={"lo": 0, "n": n}, x_ptr=_facts(1 << 31)))
    ((f,),) = [r.findings]
    assert (f.kind, f.access_index) == ("integer-overflow", 0)
    assert "scf.for" in _text("reader_ttir/iv_wrap.ttir").splitlines()[f.line_no - 1]
    assert "induction-variable increment" in f.detail
    _clean(check_graph(g, _bind(params={"lo": 0, "n": 1000}, x_ptr=_facts(1000))))
    # a zero-trip loop never increments
    _clean(check_graph(g, _bind(params={"lo": n, "n": n}, x_ptr=_facts(1))))


def test_unsigned_read_of_a_negative_param():
    """``pid < n.to(tl.uint32)`` reads n unsigned; the model reads the i32
    argument signed (2**32 - 1 and -1 are one i32)."""
    g = _graph("reader_ttir/unsigned_index.ttir")
    _clean(check_graph(g, _bind((8, 1, 1), {"n": 8}, x_ptr=_facts(3))))
    for n in (-1, (1 << 32) - 1):
        r = check_graph(g, _bind((8, 1, 1), {"n": n}, x_ptr=_facts(3)))
        ((f,),) = [r.findings]
        assert f.kind == "integer-overflow" and f.witness["value"] == -1
        assert (
            "cmpi ult"
            in _text("reader_ttir/unsigned_index.ttir").splitlines()[f.line_no - 1]
        )


def test_one_finding_per_op_site():
    """The add kernel's three accesses share one mask (pid * 1024 + lane):
    its wrap is reported once."""
    r = _add((1 << 22, 1, 1), (1 << 31) - 1, **_all(1 << 32, *ADD_TENSORS))
    assert _found(r) == [("integer-overflow", 0)] and r.refusal is None


# A wrap that decides its own role's divisor (with BIG = 2**31 - 1):
# t = pid + BIG, d = where(t > BIG, 0, -3). At pid 1,
# t wraps in i32, so the kernel divides by -3 and r = t // d - BIG // -3 is
# 1431655764 (0 at pid 0); in the unbounded reading d is 0 there. Checking
# the wrap assuming the divisor non-zero, and the divisor assuming no wrap,
# neither query has a model: that was a false proof in every role.
_CIRCULAR = """
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %cm3 = arith.constant -3 : i32
    %pid = tt.get_program_id x : i32
    %t = arith.addi %pid, %n : i32
    %gt = arith.cmpi sgt, %t, %n : i32
    %d = arith.select %gt, %c0, %cm3 : i32
    %q = arith.divsi %t, %d : i32
    %b3 = arith.divsi %n, %cm3 : i32
    %r = arith.subi %q, %b3 : i32
"""
_CIRCULAR_ACCESS = {
    # x + r
    "offset": """
        %a = tt.addptr %p, %r : !tt.ptr<i32>, i32
        tt.store %a, %c0 : !tt.ptr<i32>""",
    # if r > 0: x[1]
    "path": """
        %pos = arith.cmpi sgt, %r, %c0 : i32
        scf.if %pos {
          %a = tt.addptr %p, %c1 : !tt.ptr<i32>, i32
          tt.store %a, %c0 : !tt.ptr<i32>
        }""",
    # x + arange(16), masked to lanes < r + 1
    "mask": """
        %lim = arith.addi %r, %c1 : i32
        %offs = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        %sl = tt.splat %lim : i32 -> tensor<16xi32>
        %m = arith.cmpi slt, %offs, %sl : tensor<16xi32>
        %ps = tt.splat %p : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
        %a = tt.addptr %ps, %offs : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
        %z = arith.constant dense<0> : tensor<16xi32>
        tt.store %a, %z, %m : tensor<16x!tt.ptr<i32>>""",
    # for i in range(min(r + 1, 4)): x[i]
    "loop": """
        %lim = arith.addi %r, %c1 : i32
        %hi = arith.minsi %lim, %c4 : i32
        scf.for %i = %c0 to %hi step %c1 : i32 {
          %a = tt.addptr %p, %i : !tt.ptr<i32>, i32
          tt.store %a, %c0 : !tt.ptr<i32>
        }""",
}


@pytest.mark.parametrize("role", list(_CIRCULAR_ACCESS))
def test_a_role_never_assumes_its_own_conditions(role):
    """Each role is one joint query: the wrap at pid 1 is found, as the
    innermost failure (the divisor's zero is only the unbounded reading's)."""
    g, text = _module(_CIRCULAR + _CIRCULAR_ACCESS[role])
    big = (1 << 31) - 1
    r = check_graph(g, _bind((2, 1, 1), {"arg1": big}, arg0=_facts(1)))
    assert _found(r) == [("integer-overflow", 0)] and r.refusal is None
    (f,) = r.findings
    assert f.line_no == _line(text, "arith.addi %pid")
    assert (f.witness["pid_0"], f.witness["value"]) == (1, 1 << 31)
    # pid 0 alone is in bounds, and proved so
    _clean(check_graph(g, _bind((1, 1, 1), {"arg1": big}, arg0=_facts(1))))


def test_a_divisor_zero_where_a_sibling_wraps_is_found():
    """The reverse dependence: pid 1 both wraps (pid + BIG) and divides by
    zero (d = 1 - pid); either is a finding, never a proof."""
    g, _ = _module(
        """
        %c1 = arith.constant 1 : i32
        %pid = tt.get_program_id x : i32
        %t = arith.addi %pid, %n : i32
        %d = arith.subi %c1, %pid : i32
        %q = arith.divsi %pid, %d : i32
        %o = arith.subi %t, %n : i32
        %s = arith.addi %o, %q : i32
        %a = tt.addptr %p, %s : !tt.ptr<i32>, i32
        %v = tt.load %a : !tt.ptr<i32>"""
    )
    r = check_graph(g, _bind((2, 1, 1), {"arg1": (1 << 31) - 1}, arg0=_facts(1)))
    (f,) = r.findings
    assert f.kind in ("integer-overflow", "division-by-zero")
    assert f.witness["pid_0"] == 1 and r.refusal is None


_SELECT_ARM = """
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %big = arith.constant 1073741824 : i32
    %pid = tt.get_program_id x : i32
    %is0 = arith.cmpi eq, %pid, %c0 : i32
    %m = arith.muli %pid, %big : i32
"""


def test_a_wrap_in_the_arm_a_select_discards_is_no_finding():
    """off = where(pid == 0, pid * 2**30, pid): pid * 2**30 wraps from pid 2
    on, but only pid 0 reads it."""
    g, _ = _module(
        _SELECT_ARM
        + """
        %o = arith.select %is0, %m, %pid : i32
        %a = tt.addptr %p, %o : !tt.ptr<i32>, i32
        %v = tt.load %a : !tt.ptr<i32>"""
    )
    _clean(check_graph(g, _bind((4, 1, 1), arg0=_facts(4))))


def test_a_wrap_in_the_arm_a_select_takes_is_a_finding():
    g, text = _module(
        _SELECT_ARM
        + """
        %o = arith.select %is0, %pid, %m : i32
        %a = tt.addptr %p, %o : !tt.ptr<i32>, i32
        %v = tt.load %a : !tt.ptr<i32>"""
    )
    (f,) = check_graph(g, _bind((4, 1, 1), arg0=_facts(1 << 32))).findings
    assert (f.kind, f.line_no) == ("integer-overflow", _line(text, "arith.muli"))
    assert f.witness["pid_0"] in (2, 3)


def test_a_value_also_read_directly_is_checked_unguarded():
    """pid * 2**30 sits in the select's discarded arm of the offset, but the
    mask reads it directly: its wrap matters where the mask is computed."""
    g, text = _module(
        _SELECT_ARM
        + """
        %lt = arith.cmpi slt, %m, %c10 : i32
        %o = arith.select %is0, %m, %pid : i32
        %a = tt.addptr %p, %o : !tt.ptr<i32>, i32
        %v = tt.load %a, %lt : !tt.ptr<i32>"""
    )
    (f,) = check_graph(g, _bind((4, 1, 1), arg0=_facts(4))).findings
    assert (f.kind, f.line_no) == ("integer-overflow", _line(text, "arith.muli"))


def test_undefined_divisions_count_in_either_arm():
    """where(n != 0, pid // n, 0) divides by zero in the discarded arm too
    (the GPU run faults), and so does INT_MIN // -1."""
    g, text = _module(
        """
        %c0 = arith.constant 0 : i32
        %cm1 = arith.constant -1 : i32
        %pid = tt.get_program_id x : i32
        %nz = arith.cmpi ne, %n, %c0 : i32
        %q = arith.divsi %pid, %n : i32
        %o = arith.select %nz, %q, %c0 : i32
        %a = tt.addptr %p, %o : !tt.ptr<i32>, i32
        %v = tt.load %a : !tt.ptr<i32>"""
    )
    r = check_graph(g, _bind((4, 1, 1), {"arg1": 0}, arg0=_facts(4)))
    assert _found(r) == [("division-by-zero", 0)]
    _clean(check_graph(g, _bind((4, 1, 1), {"arg1": 2}, arg0=_facts(4))))
    g, text = _module(
        """
        %c0 = arith.constant 0 : i32
        %c5 = arith.constant 5 : i32
        %cm1 = arith.constant -1 : i32
        %pid = tt.get_program_id x : i32
        %never = arith.cmpi eq, %pid, %c5 : i32
        %q = arith.divsi %n, %cm1 : i32
        %o = arith.select %never, %q, %c0 : i32
        %a = tt.addptr %p, %o : !tt.ptr<i32>, i32
        %v = tt.load %a : !tt.ptr<i32>"""
    )
    (f,) = check_graph(g, _bind(params={"arg1": -(1 << 31)}, arg0=_facts(1))).findings
    assert (f.kind, f.line_no) == ("integer-overflow", _line(text, "arith.divsi"))
    assert f.witness["value"] == 1 << 31 and "the result of '//'" in f.detail


# ─────────────────────────── division by zero ───────────────────────────


def test_division_by_zero_in_an_offset():
    g, text = _module(
        """
        %pid = tt.get_program_id x : i32
        %q = arith.divsi %pid, %n : i32
        %a = tt.addptr %p, %q : !tt.ptr<i32>, i32
        %v = tt.load %a : !tt.ptr<i32>"""
    )
    r = check_graph(g, _bind((4, 1, 1), {"arg1": 0}, arg0=_facts(2)))
    assert _found(r) == [("division-by-zero", 0)] and r.refusal is None
    assert r.findings[0].line_no == _line(text, "arith.divsi")
    assert "divisor of this division" in r.findings[0].detail
    _clean(check_graph(g, _bind((4, 1, 1), {"arg1": 2}, arg0=_facts(2))))


def test_division_by_zero_in_a_mask():
    g, text = _module(
        """
        %pid = tt.get_program_id x : i32
        %c1 = arith.constant 1 : i32
        %r = arith.remsi %pid, %n : i32
        %m = arith.cmpi slt, %r, %c1 : i32
        %v = tt.load %p, %m : !tt.ptr<i32>"""
    )
    r = check_graph(g, _bind((4, 1, 1), {"arg1": 0}, arg0=_facts(1)))
    assert _found(r) == [("division-by-zero", 0)]
    assert r.findings[0].line_no == _line(text, "arith.remsi")
    assert "remainder" in r.findings[0].detail
    _clean(check_graph(g, _bind((4, 1, 1), {"arg1": 1}, arg0=_facts(1))))


def test_division_by_zero_in_a_loop_bound():
    g, text = _module(
        """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c8 = arith.constant 8 : i32
        %u = arith.divsi %c8, %n : i32
        scf.for %i = %c0 to %u step %c1 : i32 {
          %a = tt.addptr %p, %i : !tt.ptr<i32>, i32
          tt.store %a, %c0 : !tt.ptr<i32>
        }"""
    )
    r = check_graph(g, _bind(params={"arg1": 0}, arg0=_facts(4)))
    assert _found(r) == [("division-by-zero", 0)]
    assert r.findings[0].line_no == _line(text, "arith.divsi")
    _clean(check_graph(g, _bind(params={"arg1": 2}, arg0=_facts(4))))
    ((f,),) = [check_graph(g, _bind(params={"arg1": 2}, arg0=_facts(3))).findings]
    assert f.kind == "out-of-bounds" and f.violation_offset == 3


# ─────────────────────────── robustness ───────────────────────────


def test_deep_terms_are_lowered_iteratively():
    """kernel_deep_chain's offset nests more than 1000 levels: at Python's
    default recursion limit the generated == / hash would raise."""
    g = _graph("ttir/kernel_deep_chain.ttir")
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        clean = check_graph(g, _bind((4, 1, 1), {"s": 1}, out_ptr=_facts(1 << 20)))
        wrap = check_graph(g, _bind((4, 1, 1), {"s": 3}, out_ptr=_facts(1 << 20)))
    finally:
        sys.setrecursionlimit(limit)
    _clean(clean)
    assert _found(wrap) == [("integer-overflow", 0)]


def test_a_launch_key_ignores_only_the_addresses():
    """Launches whose bindings share a launch_key get one result, the
    findings' addresses moved to each launch's tensors; nothing else in a
    result (a withheld finding's message included) holds an address."""
    g = _graph(ADD)

    def at(ptr):
        return _bind(
            (5, 1, 1),
            {"n_elements": 10**9},
            **{n: _facts(4096, data_ptr=ptr) for n in ADD_TENSORS},
        )

    first, moved = at(PTR), at(PTR + 0x10000)
    assert launch_key(first) == launch_key(moved)
    result = check_graph(g, first)
    again = readdressed(result, g, moved)
    assert again == check_graph(g, moved) != result
    assert [f.violation_address for f in again.findings] == [
        PTR + 0x10000 + f.violation_offset * 4 for f in result.findings
    ]
    assert launch_key(at(PTR)) != launch_key(_bind((5, 1, 1), {"n_elements": 1}))
    guarded = _synthetic(_access(Bin("-", Pid(0), Const(1)), guarded=True))
    messages = {
        check_graph(guarded, _bind((4, 1, 1), p=_facts(4, data_ptr=ptr))).refusal
        for ptr in (PTR, PTR + 0x10000)
    }
    assert len(messages) == 1 and "0x" not in messages.pop().message


@pytest.mark.parametrize("timeout_ms", [0, 2.5, True])
def test_a_timeout_must_be_positive(timeout_ms):
    # Z3 reads 0 and below as no timeout at all.
    with pytest.raises(ValueError, match="positive int"):
        check_graph(_graph(ADD), _bind(), timeout_ms=timeout_ms)


class _InterruptedSolver:
    """A Solver whose query Z3's Ctrl+C handler cancelled."""

    def __init__(self, ctx=None):
        pass

    def set(self, **kwargs):
        pass

    def add(self, *formulas):
        pass

    def check(self):
        return z3.unknown

    def reason_unknown(self):
        return "interrupted from keyboard"


def test_a_ctrl_c_during_a_query_is_raised(monkeypatch):
    """Z3 turns a Ctrl+C into an unknown; it is the user's interrupt, never
    a solver-unknown abstention."""
    monkeypatch.setattr(oob, "Solver", _InterruptedSolver)
    with pytest.raises(KeyboardInterrupt):
        _add((4, 1, 1), 4096, **_all(4096, *ADD_TENSORS))


_SIGINT_SCRIPT = """
import os, signal, threading, time
from types import MappingProxyType
from tilelens.clients.sanitizer.compiled.oob import check_graph
from tilelens.ir.launch import LaunchBinding, TensorFacts
from tilelens.ir.ttir_reader import (
    AccessEvent, AccessGraph, Bin, BoolBin, Cmp, Const, FuncArg, Pid,
)

def cube(t):
    return Bin("*", Bin("*", t, t), t)

x, y, z = Pid(0), Pid(1), Pid(2)
pos = [Cmp("sgt", t, Const(0)) for t in (x, y, z)]
fermat = BoolBin("and", BoolBin("and", pos[0], pos[1]), BoolBin(
    "and", pos[2], Cmp("eq", Bin("+", cube(x), cube(y)), cube(z))))
graph = AccessGraph("k", [FuncArg("p", True, 32)],
                    [AccessEvent("load", "p", Const(-1), fermat, 32, None, 1)], None)
facts = TensorFacts(4096, 4, 4, (4,), (1,), "torch.float32", True)
grid = (1 << 20,) * 3
binding = LaunchBinding(MappingProxyType({}), MappingProxyType({"p": facts}),
                        MappingProxyType({}), grid, grid, MappingProxyType({}))
threading.Timer(1.5, os.kill, (os.getpid(), signal.SIGINT)).start()
start = time.monotonic()
try:
    result = check_graph(graph, binding, timeout_ms=60_000)
except KeyboardInterrupt as exc:
    print("interrupted", repr(exc), round(time.monotonic() - start))
else:
    print("returned", result.abstained)
"""


def test_a_real_ctrl_c_stops_a_hard_query():
    """A SIGINT 1.5 s into a query Z3 cannot decide in a minute (the Fermat
    mask of test_solver_unknown_abstains): Z3 cancels it, and the check
    raises KeyboardInterrupt at once instead of abstaining and going on."""
    proc = subprocess.run(
        [sys.executable, "-c", _SIGINT_SCRIPT],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith(
        "interrupted KeyboardInterrupt('Z3 query"
    ), proc.stdout
    assert int(proc.stdout.split()[-1]) < 30


_THREADS_SCRIPT = """
import sys, threading
from pathlib import Path
from types import MappingProxyType
from tilelens.clients.sanitizer.compiled.oob import check_graph
from tilelens.ir.launch import LaunchBinding, TensorFacts
from tilelens.ir.ttir_reader import parse_ttir

graph = parse_ttir(Path(sys.argv[1]).read_text())
facts = TensorFacts(4096, 4, 4096, (4096,), (1,), "torch.float32", True)
grid = (5, 1, 1)
binding = LaunchBinding(
    MappingProxyType({"n_elements": 10**6}),
    MappingProxyType(dict.fromkeys(("x_ptr", "y_ptr", "out_ptr"), facts)),
    MappingProxyType({}), grid, grid, MappingProxyType({}))
errors, kinds = [], set()

def worker():
    try:
        for _ in range(25):
            kinds.add(tuple(f.kind for f in check_graph(graph, binding).findings))
    except Exception as exc:
        errors.append(repr(exc))

threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(errors[:3], sorted(kinds))
"""


def _corpus_file(tmp_path: Path, name: str) -> Path:
    """Corpus text ``name`` in a file under ``tmp_path``, for a subprocess."""
    path = tmp_path / Path(name).name
    path.write_text(_text(name), encoding="utf-8")
    return path


def test_checks_on_several_host_threads_at_once(tmp_path):
    """Two traced kernels finalized on different host threads check at the
    same time: each check has a Z3 context of its own. In a subprocess, so a
    crash is a failure here and not the test run's end."""
    proc = subprocess.run(
        [sys.executable, "-c", _THREADS_SCRIPT, str(_corpus_file(tmp_path, ADD))],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    oob_kinds = ("out-of-bounds",) * 3
    assert proc.stdout.strip() == f"[] [{oob_kinds!r}]"


def test_results_are_plain_data():
    r = _add((5, 1, 1), 10**9, **_all(4096, "x_ptr", "out_ptr"))
    again = pickle.loads(pickle.dumps(r))
    assert again == r and isinstance(again.findings[0], Finding)
    # a Refusal holds its kind as a plain str; the abstentions keep the enum
    assert not isinstance(r.refusal.kind, SanitizerKind)
    assert (
        r.refusal.kind == "missing-binding" and r.abstained[0][1] is K.MISSING_BINDING
    )
    with pytest.raises(TypeError):
        hash(r.findings[0])
    # the sanitizer's kinds are its own, apart from the reader's
    assert not {k.value for k in SanitizerKind} & {k.value for k in TTIRKind}
    assert f"{K.SOLVER_UNKNOWN}" == str(K.SOLVER_UNKNOWN) == "solver-unknown"
