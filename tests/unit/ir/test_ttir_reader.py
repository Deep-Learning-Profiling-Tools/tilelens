"""tilelens.ir.ttir_reader: the TTIR -> AccessGraph reader on top of _mlir_walk.

The corpus is tests/unit/ir/ttir_corpus.py, compiled at test time with
Triton 3.8: ttir/ (the walk layer's corpus) and reader_ttir/ (the kernels
of tests/unit/ir/reader_kernels.py).
"""

from __future__ import annotations

import itertools
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from tilelens.ir import ParseCache, Refusal, SourceLocation
from tilelens.ir import _mlir_walk as W
from tilelens.ir import ttir_reader as R
from tilelens.ir.ttir_reader import (
    AccessGraph,
    Arange,
    AtomicInfo,
    Bin,
    BoolBin,
    Cmp,
    Const,
    IntCast,
    IterArgOffset,
    Loaded,
    LoopVar,
    Not,
    Observed,
    Param,
    Pid,
    Select,
    TTIRKind,
    UnsupportedTTIR,
    mentions_observed,
    observed_indices,
    parse_ttir,
    width_obligations,
)

from . import ttir_corpus

KERNELS = Path(__file__).resolve().parent / "reader_kernels.py"
FILES = ttir_corpus.NAMES


def _text(name: str) -> str:
    return ttir_corpus.text(name)


def _graph(name: str) -> AccessGraph:
    return parse_ttir(_text(name))


def _refusal(name_or_text: str) -> UnsupportedTTIR:
    text = _text(name_or_text) if name_or_text.endswith(".ttir") else name_or_text
    with pytest.raises(UnsupportedTTIR) as info:
        parse_ttir(text)
    return info.value


def _module(body: str, args: str = "%p: !tt.ptr<f32>, %n: i32", extra: str = "") -> str:
    """A minimal TTIR module (no locs) around ``body``'s op lines."""
    lines = "\n    ".join(line.strip() for line in body.strip().splitlines())
    return (
        f"module {{\n  tt.func public @k({args}) attributes {{noinline = false}} {{\n"
        f"    {lines}\n    tt.return\n  }}\n{extra}}}\n"
    )


# Where a refusal's loc points: the frontend locates an op at its own AST
# node (scf.yield at the loop header, a call spanning lines at its first line).
_REFUSAL_SITE = {
    "p1_variant_delta": "for k in range(-8, n):",
    "loop_observed_advance": "for i in range(0, n):",
    "rv_inline_asm_store": "tl.inline_asm_elementwise(",
}


def _source_line(loc) -> str:
    """The reader_kernels.py source line a probe text's loc points at."""
    assert loc is not None and loc.file.endswith("reader_kernels.py"), loc
    return KERNELS.read_text(encoding="utf-8").splitlines()[loc.line - 1]


@pytest.fixture(autouse=True)
def _fresh_cache():
    W._CACHE.clear()
    yield
    W._CACHE.clear()


# ─────────────────────────── refusal form ───────────────────────────


def test_kinds_are_the_representational_ones():
    assert {k.value for k in TTIRKind} == {
        "indirect-address",
        "data-dependent-bound",
        "nested-loop",
        "control-flow",
        "out-of-vocabulary",
        "call",
        "loop-variant-advance",
        "inline-asm",
        "reader-misalignment",
        "unparsable",
        "other",
    }
    # a kind is its string, also when formatted
    assert (
        TTIRKind.CALL == "call" and f"{TTIRKind.CALL}" == str(TTIRKind.CALL) == "call"
    )


def test_unsupported_ttir_carries_structured_fields():
    loc = W.SourceLoc("k.py", 3, 4)
    e = UnsupportedTTIR("control-flow", "scf.while", line_no=7, loc=loc)
    assert e.kind is TTIRKind.CONTROL_FLOW
    assert (e.message, e.line_no, e.loc, str(e)) == ("scf.while", 7, loc, "scf.while")
    back = pickle.loads(pickle.dumps(e))
    assert (back.kind, back.message, back.line_no, back.loc) == (
        e.kind,
        e.message,
        7,
        loc,
    )
    # client-owned kinds are not reader kinds
    for kind in ("spin-shape", "cas-value", "data-dependent-mask"):
        with pytest.raises(ValueError):
            UnsupportedTTIR(kind, "no")
    # the verdict record copies the fields, no string round-trip; the
    # private walk loc becomes the public SourceLocation (a saved trace holds it)
    r = Refusal.from_exception(e)
    assert (r.kind, r.message, r.line_no, r.loc) == (
        "control-flow",
        "scf.while",
        7,
        SourceLocation("k.py", 3, 4),
    )
    assert type(r.loc) is SourceLocation


def test_parse_cache_reads_with_this_reader():
    cache = ParseCache()
    parsed = cache.get(_text("ttir/golden_add_sm80.ttir"))
    assert isinstance(parsed.graph, AccessGraph) and parsed.error is None
    refused = cache.get(_text("reader_ttir/p2_swap.ttir"))
    assert isinstance(refused.refusal, UnsupportedTTIR) and refused.error is None
    assert refused.refusal.kind is TTIRKind.LOOP_VARIANT_ADVANCE


def test_walk_failures_map_to_kinds():
    text = _text("ttir/golden_add_sm80.ttir")
    commented = text.replace(" loc(#loc1)", " loc(#loc1) // c", 1)
    e = _refusal(commented)
    assert e.kind is TTIRKind.READER_MISALIGNMENT and "unexpected comment" in e.message
    assert "// c" in commented.splitlines()[e.line_no - 1]
    e = _refusal("module {\n  tt.func public @k( {\n}\n")
    assert e.kind is TTIRKind.UNPARSABLE
    assert e.line_no == 2 and "<ttir>" in e.message


def test_refusals_carry_the_op_line_and_source_loc():
    e = _refusal("reader_ttir/p3_call_offset.ttir")
    assert e.kind is TTIRKind.CALL and "_p3_helper" in e.message
    assert (
        "tt.call"
        in _text("reader_ttir/p3_call_offset.ttir").splitlines()[e.line_no - 1]
    )
    assert "_p3_helper(" in _source_line(e.loc)


# ─────────────────────────── every corpus text's outcome ───────────────────────────

# The reader's refusal kind for every refused corpus text (the rest parse).
REFUSED = {
    "ttir/adv_descs.ttir": "out-of-vocabulary",
    "ttir/adv_views.ttir": "other",  # tt.reshape of a pointer tile
    "ttir/crafted_attr_dicts.ttir": "other",
    "ttir/crafted_same_dest.ttir": "control-flow",
    "ttir/crafted_symbols_strings.ttir": "call",
    "ttir/golden_gather_sm80.ttir": "indirect-address",
    "ttir/golden_nested_guard_merge_sm80.ttir": "control-flow",
    "ttir/spike_inline_asm.ttir": "inline-asm",
    "ttir/spike_spin_while.ttir": "control-flow",
    "reader_ttir/p1_variant_delta.ttir": "loop-variant-advance",
    "reader_ttir/loop_observed_advance.ttir": "loop-variant-advance",
    "reader_ttir/p2_swap.ttir": "loop-variant-advance",
    "reader_ttir/p3_call_guarded.ttir": "call",
    "reader_ttir/p3_call_offset.ttir": "call",
    "reader_ttir/p3_call_formals.ttir": "call",
    "reader_ttir/rv_inline_asm_store.ttir": "inline-asm",
    "reader_ttir/int_iterarg_offset.ttir": "loop-variant-advance",
    "reader_ttir/pure_asm_int_addr.ttir": "inline-asm",
    "reader_ttir/observed_lanes.ttir": "indirect-address",
    "reader_ttir/poll_while.ttir": "control-flow",
    "reader_ttir/mp_loops.ttir": "nested-loop",
    "reader_ttir/mp_loaded.ttir": "indirect-address",
    "reader_ttir/mp_early_return.ttir": "control-flow",
    "reader_ttir/mp_loaded_advance.ttir": "indirect-address",
    "reader_ttir/mp_for_in_while.ttir": "control-flow",
}


@pytest.mark.parametrize("name", FILES)
def test_each_text_reads_as_pinned(name):
    try:
        parse_ttir(_text(name))
    except UnsupportedTTIR as e:
        assert e.kind == REFUSED.get(name), e.message
    else:
        assert name not in REFUSED


# ─────────────────────────── soundness edge cases ───────────────────────────


def test_p1_loop_variant_advance_refuses():
    e = _refusal("reader_ttir/p1_variant_delta.ttir")
    assert e.kind is TTIRKind.LOOP_VARIANT_ADVANCE and "loop-variant" in e.message
    assert _REFUSAL_SITE["p1_variant_delta"] in _source_line(e.loc)


def test_p2_swapped_iter_args_refuse():
    e = _refusal("reader_ttir/p2_swap.ttir")
    assert (
        e.kind is TTIRKind.LOOP_VARIANT_ADVANCE
        and "not advanced from itself" in e.message
    )


def test_observation_inside_the_loop_is_a_variant_advance():
    e = _refusal("reader_ttir/loop_observed_advance.ttir")
    assert e.kind is TTIRKind.LOOP_VARIANT_ADVANCE
    assert _REFUSAL_SITE["loop_observed_advance"] in _source_line(e.loc)


def test_quoted_callee_and_a_second_function_refuse_as_call():
    e = _refusal("ttir/crafted_symbols_strings.ttir")
    assert e.kind is TTIRKind.CALL and "'f{%x} \"q\" (a)'" in e.message
    # a second tt.func refuses even when nothing calls it
    extra = (
        "  tt.func private @h(%q: !tt.ptr<f32>) attributes {noinline = true} {\n"
        "    tt.return\n  }\n"
    )
    e = _refusal(_module("", extra=extra))
    assert e.kind is TTIRKind.CALL and "'h'" in e.message


def test_p4_walkers_resolve_loop_carried_pointers():
    direct = _graph("reader_ttir/p4_observed_direct.ttir")
    loop = _graph("reader_ttir/p4_observed_loop.ttir")
    delta = _graph("reader_ttir/p4_observed_delta.ttir")
    for g in (direct, loop, delta):
        load = next(a for a in g.accesses if a.kind == "load")
        assert mentions_observed(load.offset, g)
    # the address reaches the observation only through the iter_arg (its
    # offset0, resp. its delta), which a term-local walker misses
    for g, part in ((loop, "offset0"), (delta, "delta")):
        load = next(a for a in g.accesses if a.kind == "load")
        assert isinstance(load.offset, IterArgOffset)
        assert mentions_observed(getattr(g.iter_args[0], part), g)


def test_rv_inline_asm_refuses():
    e = _refusal("reader_ttir/rv_inline_asm_store.ttir")
    assert e.kind is TTIRKind.INLINE_ASM and "side effects" in e.message
    assert _REFUSAL_SITE["rv_inline_asm_store"] in _source_line(e.loc)
    # a pure asm with a pointer operand is still an address handed to asm
    text = _text("ttir/spike_inline_asm.ttir").replace("pure = false", "pure = true")
    e = _refusal(text)
    assert e.kind is TTIRKind.INLINE_ASM and "pointer operand" in e.message
    # a pure asm over data is plain data
    text = "\n".join(
        line
        for line in _text("ttir/spike_inline_asm.ttir").splitlines()
        if "pure = false" not in line
    )
    assert [a.kind for a in parse_ttir(text).accesses] == ["load"]


def _footprint(g, access, params: dict, iters: int) -> set[int]:
    """The modeled element offsets of ``access`` over iterations
    ``0 .. iters-1`` with every arange lane free, the lanes keyed by
    (make_range, dim) as a consumer keys them; mask applied."""
    lanes = sorted(
        {
            (n.ssa, n.dim, n.start, n.end)
            for n in R._nodes((access.offset, access.mask), g.iter_args)
            if isinstance(n, Arange)
        }
    )

    def ev(t, env):
        if isinstance(t, Const):
            return t.value
        if isinstance(t, Param):
            return env[t.name]
        if isinstance(t, Arange):
            return env[(t.ssa, t.dim)]
        if isinstance(t, IterArgOffset):
            info = g.iter_args[t.arg_id]
            return ev(info.offset0, env) + env["k"] * ev(info.delta, env)
        if isinstance(t, IntCast):
            return ev(t.x, env)
        a, b = ev(t.a, env), ev(t.b, env)
        if isinstance(t, Cmp):
            return int({"slt": a < b}[t.pred])
        return {"+": a + b, "-": a - b, "*": a * b}[t.op]

    out = set()
    for k in range(iters):
        for values in itertools.product(*(range(s, e) for _, _, s, e in lanes)):
            env = dict(params, k=k)
            env.update({(ssa, dim): v for (ssa, dim, _, _), v in zip(lanes, values)})
            if access.mask is None or ev(access.mask, env):
                out.add(ev(access.offset, env))
    return out


def test_expanded_loop_carried_tiles_keep_their_lanes():
    # [N, N] pointer tile expanded to 3D in the loop, next to the same
    # make_range at dim 0
    g = _graph("reader_ttir/expand_iterarg_3d.ttir")
    load = g.accesses[0]
    assert load.kind == "load" and load.in_loop
    (tile,) = [
        n for n in R._nodes((load.offset,), None) if isinstance(n, IterArgOffset)
    ]
    expanded = g.iter_args[tile.arg_id]
    assert expanded.base_param == "x_ptr" and expanded.delta == Const(1)
    n = 4
    want = {
        j * n + m - i * n + k
        for i, j, m in itertools.product(range(n), repeat=3)
        for k in range(2)
    }
    assert _footprint(g, load, {}, 2) == want  # [-12, 16], every OOB offset kept
    # 1D pointer expanded to 2D, masked at its own lane (dim 1)
    g = _graph("reader_ttir/expand_iterarg_mask.ttir")
    load = g.accesses[0]
    assert load.offset == IterArgOffset(1) and g.iter_args[1].delta == Const(4)
    assert _footprint(g, load, {"M": 2}, 2) == {0, 1, 4, 5}


def test_tensor_observations_do_not_meet_across_lanes():
    # c[:, None] - c[None, :] over one tensor atomic's old values
    e = _refusal("reader_ttir/observed_lanes.ttir")
    assert e.kind is TTIRKind.INDIRECT_ADDRESS and "across lanes" in e.message
    assert "c[:, None] - c[None, :]" in _source_line(e.loc)
    # an expanded tile of a pointer that advances by per-lane observations
    body = """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
        %ps = tt.splat %p : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
        %q = tt.addptr %ps, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        %one = arith.constant dense<1> : tensor<4xi32>
        %old = tt.atomic_rmw add, acq_rel, gpu, %q, %one : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
        %res = scf.for %k = %c0 to %n step %c1 iter_args(%a = %q) -> (tensor<4x!tt.ptr<i32>>)  : i32 {
          %e = tt.expand_dims %a {axis = 0 : i32} : tensor<4x!tt.ptr<i32>> -> tensor<1x4x!tt.ptr<i32>>
          %v = tt.load %e : tensor<1x4x!tt.ptr<i32>>
          %a2 = tt.addptr %a, %old : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
          scf.yield %a2 : tensor<4x!tt.ptr<i32>>
        }"""
    e = _refusal(_module(body, args="%p: !tt.ptr<i32>, %n: i32"))
    assert e.kind is TTIRKind.OTHER and "per-lane atomic result" in e.message


def test_loop_carried_integers_make_addresses_loop_variant():
    # offs += B carried by the loop
    e = _refusal("reader_ttir/int_iterarg_offset.ttir")
    assert (
        e.kind is TTIRKind.LOOP_VARIANT_ADVANCE and "carried by the loop" in e.message
    )
    # a pointer advanced by a loop-carried integer
    body = """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%a = %p, %s = %c0) -> (!tt.ptr<f32>, i32)  : i32 {
          %v = tt.load %a : !tt.ptr<f32>
          %a2 = tt.addptr %a, %s : !tt.ptr<f32>, i32
          %s2 = arith.addi %s, %c1 : i32
          scf.yield %a2, %s2 : !tt.ptr<f32>, i32
        }"""
    assert _refusal(_module(body)).kind is TTIRKind.LOOP_VARIANT_ADVANCE


def test_an_address_handed_to_opaque_ops_as_an_integer_refuses():
    # tl.inline_asm_elementwise(..., is_pure=True) given x_ptr.to(tl.int64)
    e = _refusal("reader_ttir/pure_asm_int_addr.ttir")
    assert e.kind is TTIRKind.INLINE_ASM and "tt.ptr_to_int" in e.message
    # through arithmetic with loaded data, and through a loop-carried value
    args = "%p: !tt.ptr<i64>, %n: i32"
    head = """
        %x = tt.load %p : !tt.ptr<i64>
        %i = tt.ptr_to_int %p : !tt.ptr<i64> -> i64
        %a = arith.addi %i, %x : i64"""
    asm = (
        '%y = tt.elementwise_inline_asm "mov.b64 $0, $1;" {constraints = "=l,l", '
        "packed_element = 1 : i32, pure = true} %a : i64 -> i64"
    )
    assert _refusal(_module(head + "\n" + asm, args=args)).kind is TTIRKind.INLINE_ASM
    loop = """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %z = arith.constant 0 : i64
        %i = tt.ptr_to_int %p : !tt.ptr<i64> -> i64
        %a = scf.for %k = %c0 to %n step %c1 iter_args(%s = %z) -> (i64)  : i32 {
          %s2 = arith.addi %s, %i : i64
          scf.yield %s2 : i64
        }"""
    assert _refusal(_module(loop + "\n" + asm, args=args)).kind is TTIRKind.INLINE_ASM
    extern = (
        '%y = tt.extern_elementwise %a {libname = "", libpath = "", pure = true, '
        'symbol = "f"} : (i64) -> i64'
    )
    e = _refusal(_module(head + "\n" + extern, args=args))
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "tt.ptr_to_int" in e.message
    # loaded data carries no address: an asm over it stays plain data
    data = head.replace("%i, %x", "%x, %x") + "\n" + asm
    assert [a.kind for a in parse_ttir(_module(data, args=args)).accesses] == ["load"]


def test_llvm_and_gpu_ops_outside_the_inert_ones_refuse():
    # memory effects through ops of the llvm dialect
    cases = {
        "llvm.inttoptr": """
            %i = tt.ptr_to_int %p : !tt.ptr<i32> -> i64
            %lp = llvm.inttoptr %i : i64 to !llvm.ptr<1>
            %c = arith.constant 7 : i32
            %old = llvm.atomicrmw add %lp, %c monotonic : !llvm.ptr<1>, i32""",
        "llvm.inline_asm": """
            %i = tt.ptr_to_int %p : !tt.ptr<i32> -> i64
            %c = arith.constant 7 : i32
            %r = llvm.inline_asm has_side_effects "st.global.b32 [$1], $2; mov.b32 $0, 0;", "=r,l,r" %i, %c : (i64, i32) -> i32""",
    }
    for name, body in cases.items():
        e = _refusal(_module(body, args="%p: !tt.ptr<i32>, %n: i32"))
        assert e.kind is TTIRKind.OUT_OF_VOCABULARY and name in e.message
    # inside a combine region too
    body = """
        %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
        %s = "tt.reduce"(%r) <{axis = 0 : i32}> ({
        ^bb0(%a: i32, %b: i32):
          %i = tt.ptr_to_int %p : !tt.ptr<i32> -> i64
          %lp = llvm.inttoptr %i : i64 to !llvm.ptr<1>
          %y = arith.addi %a, %b : i32
          tt.reduce.return %y : i32
        }) : (tensor<4xi32>) -> i32"""
    e = _refusal(_module(body, args="%p: !tt.ptr<i32>, %n: i32"))
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "llvm.inttoptr" in e.message
    # ttg.barrier is the one ttg op accepted (a Fence, see below);
    # gpu.barrier still parses, but the frontend never emits it: refused
    g = parse_ttir(_module("ttg.barrier all\n%v = tt.load %p : !tt.ptr<f32>"))
    assert [a.kind for a in g.accesses] == ["load"]
    e = _refusal(_module("gpu.barrier\n%v = tt.load %p : !tt.ptr<f32>"))
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY, e.message
    assert "op gpu.barrier is not TTIR" in e.message
    e = _refusal(_module("ttg.local_barrier\n%v = tt.load %p : !tt.ptr<f32>"))
    assert e.kind in (TTIRKind.UNPARSABLE, TTIRKind.OUT_OF_VOCABULARY)


def test_pointers_outside_global_memory_refuse():
    # a shared-memory (address space 3) pointer argument
    e = _refusal(
        _module(
            "%c1 = arith.constant 1 : i32\ntt.store %p, %c1 : !tt.ptr<i32, 3>",
            args="%p: !tt.ptr<i32, 3>, %n: i32",
        )
    )
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "address space" in e.message
    # a pointer made into another address space
    body = """
        %c = arith.constant 0 : i64
        %q = tt.int_to_ptr %c : i64 -> !tt.ptr<f32, 3>
        %v = tt.load %q : !tt.ptr<f32, 3>"""
    e = _refusal(_module(body))
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "address space" in e.message
    assert e.line_no == 4  # the tt.int_to_ptr


def test_generic_load_reads_its_mask_by_operand_segment():
    # generic tt.load with `other` and no mask; an i1 `other` is no mask
    load = (
        '%v = "tt.load"(%p, %f) <{boundaryCheck = array<i32>, cache = 1 : i32, '
        "evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: SEG>}> "
        ": (!tt.ptr<i1>, i1) -> i1"
    )
    for segments, mask in (("1, 0, 1", None), ("1, 1, 0", Const(0))):
        g = parse_ttir(
            _module(
                "%f = arith.constant false\n" + load.replace("SEG", segments),
                args="%p: !tt.ptr<i1>, %n: i32",
            )
        )
        (access,) = g.accesses
        assert (access.mask, access.mask_dropped) == (mask, False), segments


# ─────────────────────────── modeled shapes ───────────────────────────


def test_two_step_advance_is_the_sum():
    g = _graph("reader_ttir/loop_two_step_advance.ttir")
    (info,) = g.iter_args
    assert (info.base_param, info.offset0, info.delta) == (
        "x_ptr",
        Const(0),
        Bin("+", Param("s"), Const(2)),
    )
    assert g.accesses[0].offset == IterArgOffset(0) and g.accesses[0].in_loop


def test_same_base_pointer_selects():
    g = _graph("reader_ttir/where_pointer.ttir")
    (store,) = g.accesses
    assert store.base_param == "x_ptr" and isinstance(store.offset, Select)
    g = _graph("ttir/spike_if_yield.ttir")
    store = g.accesses[-1]
    assert store.kind == "store" and store.base_param == "out_ptr"
    assert isinstance(store.offset, Select) and isinstance(store.offset.cond, Cmp)


def test_three_dims_of_one_make_range_are_three_lanes():
    (store,) = _graph("reader_ttir/tile3d_shared_arange.ttir").accesses
    ranges = [n for n in R._nodes((store.offset,), None) if isinstance(n, Arange)]
    assert len({r.ssa for r in ranges}) == 1
    assert sorted(r.dim for r in ranges) == [0, 1, 2]


def test_same_width_pointer_bitcast_keeps_the_base():
    g = _graph("ttir/golden_atomic_fmax_sm80.ttir")
    atomics = [a for a in g.accesses if a.kind == "atomic_rmw"]
    assert len(atomics) == 2
    assert all(a.base_param == "out_ptr" and a.elem_bits == 32 for a in atomics)
    # the atomics' mask reads loaded data
    assert all(a.mask_dropped and a.mask is None for a in atomics)
    # a bitcast to another element width changes what an element offset means
    body = """
        %q = tt.bitcast %p : !tt.ptr<f32> -> !tt.ptr<f16>
        %v = tt.load %q : !tt.ptr<f16>"""
    e = _refusal(_module(body))
    assert e.kind is TTIRKind.OTHER and "element width" in e.message


def test_other_refusals():
    cases = {
        # a block pointer: the dialect has none (its parser aborts on the
        # type), so the walk refuses the text before parsing
        "unparsable": """
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i64
            %c32 = arith.constant 32 : i64
            %bp = tt.make_tensor_ptr %p, [%c32, %c32], [%c32, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<32x32xf32>>
            %v = tt.load %bp : !tt.ptr<tensor<32x32xf32>>""",
        "out-of-vocabulary": """
            %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
            %v = arith.sitofp %r : tensor<64xi32> to tensor<64xf32>
            %s = "tt.reduce"(%v) <{axis = 0 : i32}> ({
            ^bb0(%a: f32, %b: f32):
              %x = tt.load %p : !tt.ptr<f32>
              %y = arith.addf %a, %x : f32
              tt.reduce.return %y : f32
            }) : (tensor<64xf32>) -> f32""",
        "control-flow": "%c0 = arith.constant 0 : i32\n"
        "%b = arith.cmpi slt, %n, %c0 : i32\n"
        'cf.assert %b, "neg"',
        "nested-loop": """
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            scf.for %i = %c0 to %n step %c1  : i32 {
              scf.for %j = %i to %n step %c1  : i32 {
                %q = tt.addptr %p, %j : !tt.ptr<f32>, i32
                %v = tt.load %q : !tt.ptr<f32>
              }
            }""",
    }
    for kind, body in cases.items():
        assert _refusal(_module(body)).kind == kind, kind
    e = _refusal(_module(cases["unparsable"]))
    assert "no block pointers" in e.message and e.line_no == 7
    e = _refusal(
        _module(
            "%x = tt.load %p : !tt.ptr<f32>\n%y = tt.extern_elementwise %x "
            '{libname = "", libpath = "", pure = false, symbol = "foo"} : (f32) -> f32'
        )
    )
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "'foo'" in e.message
    # deep structured nesting refuses instead of exhausting the recursion limit
    depth = 250
    nest = (
        "".join("scf.if %c {\n" for _ in range(depth))
        + "tt.store %p, %f : !tt.ptr<f32>\n"
        + "}\n" * depth
    )
    e = _refusal(
        _module(
            "%f = arith.constant 1.0 : f32\n" + nest, args="%p: !tt.ptr<f32>, %c: i1"
        )
    )
    assert e.kind is TTIRKind.OTHER and "nested deeper" in e.message


def test_pure_extern_and_signed_i1_compare_are_data():
    g = parse_ttir(
        _module(
            "%x = tt.load %p : !tt.ptr<f32>\n%y = tt.extern_elementwise %x "
            '{libname = "", libpath = "", pure = true, symbol = "__nv_expf"} : (f32) -> f32\n'
            "tt.store %p, %y : !tt.ptr<f32>"
        )
    )
    assert [a.kind for a in g.accesses] == ["load", "store"]
    # a signed compare of i1 reads true as -1: not the boolean model, so the
    # mask is dropped (widened), never misread
    g = parse_ttir(
        _module(
            """
        %c0 = arith.constant 0 : i32
        %b = arith.cmpi slt, %n, %c0 : i32
        %t = arith.constant true
        %c = arith.cmpi slt, %b, %t : i1
        %v = tt.load %p, %c : !tt.ptr<f32>"""
        )
    )
    (load,) = g.accesses
    assert load.mask is None and load.mask_dropped


def test_matmul_loop_iter_args_and_params():
    g = _graph("ttir/golden_matmul_s3_sm80.ttir")
    assert g.kernel_name == "matmul_kernel" and g.loop is not None
    assert [i.base_param for i in g.iter_args] == ["a_ptr", "b_ptr"]
    assert all(
        i.arg_id == k and i.loop_ssa == g.loop.loop_ssa
        for k, i in enumerate(g.iter_args)
    )
    assert Param("K") in list(R._nodes((g.loop.upper,), None))
    loads = [a for a in g.accesses if a.kind == "load"]
    assert [a.offset for a in loads] == [IterArgOffset(0), IterArgOffset(1)]
    assert all(a.in_loop for a in loads) and not g.accesses[-1].in_loop
    assert g.loop.bits == 32 and not g.loop.unsigned and g.loop.line_no is not None
    assert g.arg("K").int_bits == 32 and g.arg("a_ptr").elem_bits == 16


# ─────────────────────────── integer widths ───────────────────────────


def _value(t, env: dict) -> int:
    """The unbounded-integer reading of a (shallow) term: casts as identity."""
    if isinstance(t, Const):
        return t.value
    if isinstance(t, Pid):
        return env["pid"]
    if isinstance(t, Param):
        return env[t.name]
    if isinstance(t, IntCast):
        return _value(t.x, env)
    if isinstance(t, Cmp):
        a, b = _value(t.a, env), _value(t.b, env)
        return int({"slt": a < b, "ult": a < b, "eq": a == b}[t.pred])
    if isinstance(t, Bin):
        a, b = _value(t.a, env), _value(t.b, env)
        ops = {"+": a + b, "*": a * b, "-": a - b}
        if t.op in ("u//", "//"):
            return int(a / b)
        if t.op == "%":
            return a - b * int(a / b)
        return ops[t.op]
    raise AssertionError(type(t).__name__)


def _holds(ob, env: dict) -> bool:
    v = _value(ob.term, env)
    if ob.signed:
        return -(1 << (ob.bits - 1)) <= v < (1 << (ob.bits - 1))
    return 0 <= v < (1 << ob.bits)


def _integer_nodes(g):
    for a in g.accesses:
        yield from R._nodes((a.offset, a.mask, a.path), g.iter_args)
    if g.loop is not None:
        yield from R._nodes((g.loop.lower, g.loop.upper, g.loop.step), None)


@pytest.mark.parametrize("name", [f for f in FILES if f not in REFUSED])
def test_every_integer_op_carries_its_width(name):
    for n in _integer_nodes(_graph(name)):
        if isinstance(n, Bin):
            # bits=None only for the element-offset sums of tt.addptr
            assert isinstance(n.bits, int) or (n.bits is None and n.op == "+")
        elif isinstance(n, Cmp):
            assert isinstance(n.bits, int) and n.bits >= 1
        elif isinstance(n, IntCast):
            assert n.kind in ("trunci", "extsi", "extui") and n.src_bits != n.dst_bits


def test_i32_wrap_obligations():
    g = _graph("reader_ttir/rv_i32_wrap.ttir")
    (store,) = g.accesses
    inner = Bin("*", Pid(0), Param("S"), 32)
    outer = Bin("*", inner, Param("S"), 32)
    assert store.offset == Bin("+", Const(0), outer)
    obs = width_obligations(g, store)
    assert [(o.term, o.bits, o.signed) for o in obs] == [
        (outer, 32, True),
        (inner, 32, True),
    ]
    assert all("pid * S" in _source_line(o.loc) for o in obs)
    text = _text("reader_ttir/rv_i32_wrap.ttir").splitlines()
    assert all("arith.muli" in text[o.line_no - 1] for o in obs)
    # (pid * 65536) * 65536 is 0 in i32: the unbounded model is exact only for pid 0
    assert all(_holds(o, {"pid": 0, "S": 65536}) for o in obs)
    assert [_holds(o, {"pid": 1, "S": 65536}) for o in obs] == [False, True]


def test_trunci_obligation():
    g = _graph("reader_ttir/rv_trunci_alias.ttir")
    (store,) = g.accesses
    wide = Bin("*", IntCast("extsi", 32, 64, Pid(0)), Const(1 << 32), 64)
    assert store.offset == Bin("+", Const(0), IntCast("trunci", 64, 32, wide))
    obs = width_obligations(g, store)
    assert [(o.term, o.bits, o.signed) for o in obs] == [
        (wide, 32, True),
        (wide, 64, True),
    ]
    trunc = obs[0]
    assert (
        "arith.trunci"
        in _text("reader_ttir/rv_trunci_alias.ttir").splitlines()[trunc.line_no - 1]
    )
    assert ".to(tl.int32)" in _source_line(trunc.loc)
    # trunc_i32(pid * 2**32) is 0 for every pid: exact only for pid 0
    assert _holds(trunc, {"pid": 0}) and not _holds(trunc, {"pid": 1})
    assert _holds(obs[1], {"pid": 1})


def test_unsigned_reads_need_non_negative_operands():
    g = _graph("reader_ttir/unsigned_index.ttir")
    (store,) = g.accesses
    quotient = Bin("u//", Pid(0), Const(3), 32)
    assert store.offset == Bin("+", Const(0), IntCast("extui", 32, 64, quotient))
    assert store.mask == Cmp("ult", Pid(0), Param("n"), 32)
    obs = {(o.term, o.bits, o.signed) for o in width_obligations(g, store)}
    assert obs == {
        (quotient, 32, False),  # the extui operand
        (quotient, 32, True),  # the divui result
        (Pid(0), 32, False),
        (Const(3), 32, False),
        (Param("n"), 32, False),  # cmpi ult reads n unsigned
    }
    obs = width_obligations(g, store)
    assert all(_holds(o, {"pid": 5, "n": 8}) for o in obs)
    # n = -1 reads as 2**32 - 1 under ult: the unbounded model is not exact
    assert not all(_holds(o, {"pid": 5, "n": -1}) for o in obs)


def test_narrowing_casts_and_extui_in_spike_casts():
    g = _graph("ttir/spike_casts.ttir")
    load = g.accesses[0]
    casts = [n for n in R._nodes((load.offset,), None) if isinstance(n, IntCast)]
    assert {(c.kind, c.src_bits, c.dst_bits) for c in casts} == {
        ("trunci", 32, 16),
        ("extsi", 16, 64),
        ("trunci", 32, 8),
        ("extui", 8, 64),
    }
    obs = [
        (o.bits, o.signed) for o in width_obligations(g, load) if o.line_no is not None
    ]
    assert (16, True) in obs and (8, True) in obs and (8, False) in obs


def test_i1_casts_and_unsigned_loops():
    # extsi of an i1 maps true to -1: read as 0 - extui(b)
    g = parse_ttir(
        _module(
            """
        %c0 = arith.constant 0 : i32
        %b = arith.cmpi slt, %n, %c0 : i32
        %e = arith.extsi %b : i1 to i32
        %q = tt.addptr %p, %e : !tt.ptr<f32>, i32
        %v = tt.load %q : !tt.ptr<f32>"""
        )
    )
    b = Cmp("slt", Param("arg1"), Const(0), 32)
    assert g.accesses[0].offset == Bin(
        "+", Const(0), Bin("-", Const(0), IntCast("extui", 1, 32, b), 32)
    )
    # trunci to i1 keeps only bit 0: exact for 0 / 1
    g = parse_ttir(
        _module("%b = arith.trunci %n : i32 to i1\n%v = tt.load %p, %b : !tt.ptr<f32>")
    )
    ((ob,),) = [width_obligations(g, a) for a in g.accesses]
    assert (ob.term, ob.bits, ob.signed) == (Param("arg1"), 1, False)
    # an unsigned loop compare needs non-negative bounds
    g = parse_ttir(
        _module(
            """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        scf.for unsigned %i = %c0 to %n step %c1  : i32 {
          %q = tt.addptr %p, %i : !tt.ptr<f32>, i32
          %v = tt.load %q : !tt.ptr<f32>
        }"""
        )
    )
    assert g.loop.unsigned and g.loop.bits == 32
    obs = [
        (o.term, o.bits, o.signed, o.line_no)
        for o in width_obligations(g, g.accesses[0])
    ]
    latch = Bin("+", Bin("-", Param("arg1"), Const(1), 32), Const(1), 32)
    assert obs == [
        (Const(0), 32, False, 5),
        (Param("arg1"), 32, False, 5),
        (Const(1), 32, False, 5),
        (latch, 32, False, 5),  # the increment does not wrap, read unsigned
    ]


def test_loop_increment_obligation():
    # for k in range(lo, n, 1 << 20) wraps its induction variable when
    # n is near INT32_MAX (the GPU then runs iterations at negative k)
    g = _graph("reader_ttir/iv_wrap.ttir")
    (store,) = g.accesses
    latch = Bin("+", Bin("-", Param("n"), Const(1), 32), Const(1 << 20), 32)
    obs = width_obligations(g, store)
    assert [(o.term, o.bits, o.signed, o.role) for o in obs] == [
        (latch, 32, True, "loop")
    ]
    assert "range(lo, n, 1 << 20)" in _source_line(obs[0].loc)
    assert (
        "scf.for" in _text("reader_ttir/iv_wrap.ttir").splitlines()[obs[0].line_no - 1]
    )
    assert _holds(obs[0], {"n": 1000})
    assert not _holds(obs[0], {"n": (1 << 31) - (1 << 19)})
    # an access outside the loop gets no loop obligations
    g = _graph("ttir/golden_matmul_s3_sm80.ttir")
    assert not any(o.role == "loop" for o in width_obligations(g, g.accesses[-1]))
    assert any(o.role == "loop" for o in width_obligations(g, g.accesses[0]))


def test_signed_remainder_needs_its_quotient_to_fit():
    # remsi(INT_MIN, -1) is undefined, though its value 0 fits
    body = """
        %cm = arith.constant -2147483648 : i32
        %cn = arith.constant -1 : i32
        %r = arith.remsi %cm, %cn : i32
        %q = tt.addptr %p, %r : !tt.ptr<f32>, i32
        %v = tt.load %q : !tt.ptr<f32>"""
    g = parse_ttir(_module(body))
    (load,) = g.accesses
    lo = Const(-(1 << 31))
    obs = width_obligations(g, load)
    assert [(o.term, o.bits, o.signed, o.line_no) for o in obs] == [
        (Bin("%", lo, Const(-1), 32), 32, True, 5),
        (Bin("//", lo, Const(-1), 32), 32, True, 5),
    ]
    assert _holds(obs[0], {}) and not _holds(obs[1], {})


def test_obligation_roles():
    # a node the path, mask and offset share is listed once, under the most
    # restrictive role (path, then mask, then offset)
    body = """
        %c4 = arith.constant 4 : i32
        %pid = tt.get_program_id x : i32
        %o = arith.muli %pid, %c4 : i32
        %s = arith.subi %n, %c4 : i32
        %b = arith.cmpi slt, %o, %s : i32
        scf.if %b {
          %t = arith.addi %o, %c4 : i32
          %m = arith.cmpi slt, %t, %n : i32
          %o2 = arith.addi %t, %c4 : i32
          %q = tt.addptr %p, %o2 : !tt.ptr<f32>, i32
          %v = tt.load %q, %m : !tt.ptr<f32>
        }"""
    g = parse_ttir(_module(body))
    (load,) = g.accesses
    o = Bin("*", Pid(0), Const(4), 32)
    s = Bin("-", Param("arg1"), Const(4), 32)
    t = Bin("+", o, Const(4), 32)
    o2 = Bin("+", t, Const(4), 32)
    assert [(ob.term, ob.role) for ob in width_obligations(g, load)] == [
        (o, "path"),
        (s, "path"),
        (t, "mask"),
        (o2, "offset"),
    ]


def test_obligation_sites_do_not_affect_term_equality():
    a = Bin("+", Pid(0), Const(1), 32, line_no=3, loc=W.SourceLoc("a.py", 1, 1))
    b = Bin("+", Pid(0), Const(1), 32, line_no=9, loc=None)
    assert a == b and hash(a) == hash(b) and repr(a) == repr(b)
    assert Bin("+", Pid(0), Const(1), 64) != a  # the width is part of the value


# ─────────────────────────── facts for the race detector ───────────────────────────


def test_atomics_carry_their_attributes_and_operands():
    g = _graph("reader_ttir/atomic_attrs.ttir")
    atomics = [a for a in g.accesses if a.atomic is not None]
    assert [(a.kind, a.atomic) for a in atomics] == [
        ("atomic_rmw", AtomicInfo(op, sem, scope))
        for op, sem, scope in [
            ("and", "relaxed", "gpu"),
            ("or", "acquire", "cta"),
            ("xor", "release", "sys"),
            ("add", "acq_rel", "gpu"),
            ("fadd", "acq_rel", "gpu"),
            ("max", "acq_rel", "gpu"),
            ("min", "acq_rel", "gpu"),
            ("umax", "acq_rel", "gpu"),
            ("umin", "acq_rel", "gpu"),
            ("exch", "acq_rel", "gpu"),
        ]
    ] + [("atomic_cas", AtomicInfo(None, "release", "sys"))]
    # the value operand as a Term, None for loaded data (and floats)
    vals = [a.atomic_val for a in atomics]
    assert vals[3] == Pid(0) and vals[0] == Const(1)
    assert vals[4] is None and vals[9] is None  # 1.0, a loaded value
    cas = atomics[-1]
    assert (cas.atomic_cmp, cas.atomic_val) == (Pid(0), Const(1))
    assert all(a.atomic_cmp is None for a in atomics[:-1])
    (load,) = [a for a in g.accesses if a.kind == "load"]
    assert load.atomic is None and load.atomic_val is None
    # elem_float is the accessed pointer's pointee: the fmax atomics of
    # golden_atomic_fmax go through i32 views of the f32 out_ptr
    assert [a.elem_float for a in atomics] == [i == 4 for i in range(11)]
    g = _graph("ttir/golden_atomic_fmax_sm80.ttir")
    assert [(a.kind, a.elem_float) for a in g.accesses] == [
        ("load", True),
        ("atomic_rmw", False),
        ("atomic_rmw", False),
    ]


def test_observed_indices_reach_through_loop_carried_pointers():
    for name in ("p4_observed_direct", "p4_observed_loop", "p4_observed_delta"):
        g = _graph(f"reader_ttir/{name}.ttir")
        load = next(a for a in g.accesses if a.kind == "load")
        assert observed_indices(load.offset, g) == {0}, name
        assert observed_indices(load.mask, g) == frozenset()
    # the iter_arg's offset0 / delta hold the observation, not the offset
    assert not any(isinstance(n, Observed) for n in R._nodes((load.offset,), None))


def test_pid_axes_are_every_axis_read():
    # axes 1 and 2 reach only a stored value, never an address
    assert _graph("reader_ttir/fences.ttir").pid_axes == {0, 1, 2}
    assert _graph("ttir/spike_atomics.ttir").pid_axes == frozenset()
    # a combine region's program id counts; its barrier is no Fence
    g = _graph("reader_ttir/pid_in_combine.ttir")
    assert (g.pid_axes, g.fences) == ({0, 1}, ())


def test_barriers_are_fences_among_the_accesses():
    text = _text("reader_ttir/fences.ttir")
    g = parse_ttir(text)
    store0, load, store1, store_loop, store2 = g.accesses
    top, then, data, loop = g.fences
    assert [f.position for f in g.fences] == [0.5, 1.5, 2.5, 2.5]
    assert all(f.op == "ttg.barrier" and f.addr_space == 31 for f in g.fences)
    for f in g.fences:
        assert "ttg.barrier all" in text.splitlines()[f.line_no - 1]
        assert "tl.debug_barrier()" in _source_line(f.loc)
    # path, guarded and in_loop read as an access's
    assert then.path == Cmp("slt", Pid(0), Param("n"), 32) == store1.path
    assert (data.guarded, data.path) == (True, None)
    assert (loop.in_loop, store_loop.in_loop) == (True, True)
    assert not any(f.in_loop or f.guarded for f in (top, then))

    # block paths: a fence's is a prefix of exactly the accesses in its block
    def under(f):
        return [a.block_path[: len(f.block_path)] == f.block_path for a in g.accesses]

    assert under(top) == [True] * 5
    assert under(then) == [False, False, True, False, False]
    assert under(data) == [False] * 5
    assert under(loop) == [False, False, False, True, False]
    # addrSpace is a bit set
    for kw, bits in (("local", 1), ("global_read|global_write", 6)):
        g = parse_ttir(_module(f"ttg.barrier {kw}\n%v = tt.load %p : !tt.ptr<f32>"))
        assert [(f.addr_space, f.position) for f in g.fences] == [(bits, -0.5)]


def _deps(name: str) -> list[tuple[str, tuple[int, ...]]]:
    return [(a.kind, a.deps) for a in _graph(f"reader_ttir/{name}.ttir").accesses]


def test_deps_are_the_loads_a_value_reads_lane_for_lane():
    g = _graph("reader_ttir/deps_rules.ttir")
    assert all(a.elem_float for a in g.accesses if a.kind == "store")  # f32 out
    assert _deps("deps_rules") == [
        ("load", ()),
        ("load", ()),
        ("load", ()),
        ("store", (0, 2)),  # x * 2 + 1 under mask m > 0
        ("store", (2,)),  # where(m > 0, x, y): the condition's
        ("store", (0,)),  # where(offs < n, x, x + 1): both arms'
        ("store", ()),  # where(offs < n, x, y)
        ("store", (1,)),  # sum(x) + y: a splat of a scalar sum drops x
        ("store", ()),  # an scf.if result
        ("store", (0,)),  # exp, bitcast
        ("load", ()),
        ("store", ()),  # tt.reduce
        ("load", ()),  # masked by m
        ("store", (12,)),
        ("store", (0,)),  # x under a mask of sum(x): the operands unite
        ("store", ()),  # the loop-carried acc
    ]
    # tt.dot: the accumulator, not the operands
    assert _deps("deps_dot")[-1] == ("store", (2,))
    # an atomic_rmw's val and mask, an atomic_cas's cmp and val, and an
    # atomic's result
    assert _deps("deps_atomics") == [
        ("load", ()),
        ("load", ()),
        ("atomic_rmw", (0, 1)),
        ("load", ()),
        ("load", ()),
        ("atomic_cas", (3, 4)),
        ("store", (2,)),
    ]
    # a shape change keeps no lane: a splat of a loaded scalar
    body = """
        %x = tt.load %p : !tt.ptr<f32>
        %s = tt.splat %x : f32 -> tensor<4xf32>
        %ps = tt.splat %p : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
        tt.store %ps, %s : tensor<4x!tt.ptr<f32>>
        tt.store %p, %x : !tt.ptr<f32>"""
    assert [a.deps for a in parse_ttir(_module(body)).accesses] == [(), (), (0,)]


def test_while_regions_are_read_only_on_request():
    text = _text("reader_ttir/poll_while.ttir")
    e = _refusal(text)
    assert e.kind is TTIRKind.CONTROL_FLOW and "scf.while" in e.message
    g = parse_ttir(text, while_regions=True)
    first, guarded, outer, inner = g.whiles
    assert [w.before for w in g.whiles] == [(0,), (1,), (2,), (3,)]
    assert [w.after for w in g.whiles] == [(), (), (3,), ()]
    assert [w.parent for w in g.whiles] == [None, None, None, 2]
    assert outer.after_ops == ("scf.while", "scf.yield")
    assert first.after_ops == inner.after_ops == ("scf.yield",)
    assert not any(w.carries for w in g.whiles)
    # a poll's result is an observation, like an atomic's
    assert first.cond == Cmp("eq", Observed(0), Const(0), 32)
    assert guarded.cond == Cmp("ne", Observed(1), Param("n"), 32)
    assert g.accesses[1].path == Cmp("slt", Pid(0), Param("n"), 32)
    assert guarded.block_path == g.accesses[1].block_path[:-1]
    assert guarded.blocks[0] == g.accesses[1].block_path[-1]
    for w in g.whiles:
        assert "scf.while" in text.splitlines()[w.line_no - 1]
        assert "while tl." in _source_line(w.loc)
    # a carried loop value and an atomic poll (the walk-layer corpus)
    g = parse_ttir(_text("ttir/spike_spin_while.ttir"), while_regions=True)
    cas, carried = g.whiles
    assert cas.cond == Cmp("eq", Observed(0), Const(1), 32) and not cas.carries
    assert (carried.carries, carried.cond, carried.after) == (True, None, (2,))
    assert carried.after_ops == ("tt.load", "scf.yield")
    assert g.accesses[3].deps == ()  # the store of the loop's result
    # without an scf.while the option changes nothing
    for name in ("ttir/golden_matmul_s3_sm80.ttir", "reader_ttir/fences.ttir"):
        assert parse_ttir(_text(name), while_regions=True) == _graph(name)
    outcome = ParseCache().get(text, while_regions=True)
    assert len(outcome.graph.whiles) == 4
    # with multipath too, a poll's load stays an observation
    for name in ("reader_ttir/poll_while.ttir", "ttir/spike_spin_while.ttir"):
        both = parse_ttir(_text(name), while_regions=True, multipath=True)
        assert pickle.dumps(both) == pickle.dumps(
            parse_ttir(_text(name), while_regions=True)
        )


# ─────────────────────────── multipath ───────────────────────────


def _mp(name: str) -> AccessGraph:
    return parse_ttir(_text(name), multipath=True)


@pytest.mark.parametrize("name", FILES)
def test_multipath_reads_a_default_text_alike_but_for_loaded_values(name):
    try:
        g = _graph(name)
    except UnsupportedTTIR:
        return
    # the default's loop facts: its one loop
    assert g.loops == ((g.loop,) if g.loop else ())
    for x in g.accesses + g.fences:
        assert x.loops == (("%loop",) if x.in_loop else ())
    m = _mp(name)
    roots = [f.path for f in m.fences] + [
        t for a in m.accesses for t in (a.offset, a.mask, a.path, a.atomic_val)
    ]
    keep = any(a.mask_dropped and a.mask is not None for a in m.accesses)
    if not keep and not any(R.loaded_indices(t, m) for t in roots):
        # pickle: byte for byte, and iterative (kernel_deep_chain)
        assert pickle.dumps(m) == pickle.dumps(g)


def test_several_loops_each_with_their_own_variable():
    assert _refusal("reader_ttir/mp_loops.ttir").kind is TTIRKind.NESTED_LOOP
    g = _mp("reader_ttir/mp_loops.ttir")
    outer, inner, under_if = g.loops
    # a graph built with one of loop and loops gets the other
    assert AccessGraph("k", (), (), outer).loops == (outer,)
    assert AccessGraph("k", (), (), None, loops=(outer,)).loop is outer
    assert g.loop is None and [lp.loop_ssa for lp in g.loops] == [
        "%loop",
        "%loop1",
        "%loop2",
    ]
    assert (inner.upper, inner.step, under_if.upper) == (
        Param("m"),
        Const(2),
        Param("n"),
    )
    nested, seq = g.accesses
    assert (nested.loops, seq.loops, seq.in_loop) == (
        ("%loop", "%loop1"),
        ("%loop2",),
        True,
    )
    assert nested.offset == Bin("+", IterArgOffset(0), LoopVar("%loop1"))
    assert seq.path == Cmp("slt", Pid(0), Param("n"), 32)
    # the inner pointer advances by the outer variable: invariant inside
    q, p = g.iter_args
    assert (q.loop_ssa, q.delta, p.loop_ssa) == ("%loop1", LoopVar("%loop"), "%loop2")
    # one increment obligation per enclosing loop, outermost first
    latches = [ob.term for ob in width_obligations(g, nested) if ob.role == "loop"]
    assert [t.b for t in latches] == [outer.step, inner.step]
    assert [ob.term.b for ob in width_obligations(g, seq)] == [under_if.step]
    # an advance by the loop's own variable, or by a value loaded in the
    # loop, is still variant; a loop in an scf.while still refuses
    for name in ("p1_variant_delta", "mp_loaded_advance"):
        with pytest.raises(UnsupportedTTIR, match="loop-variant amount") as e:
            _mp(f"reader_ttir/{name}.ttir")
        assert e.value.kind is TTIRKind.LOOP_VARIANT_ADVANCE
    text = _text("reader_ttir/mp_for_in_while.ttir")
    with pytest.raises(UnsupportedTTIR) as e:
        parse_ttir(text, while_regions=True, multipath=True)
    assert e.value.kind is TTIRKind.NESTED_LOOP


def test_integer_loads_read_as_loaded_values():
    assert _refusal("reader_ttir/mp_loaded.ttir").kind is TTIRKind.INDIRECT_ADDRESS
    g = _mp("reader_ttir/mp_loaded.ttir")
    j, gather, _, by_j, _, tile, w, kept, _, under, either, kept2, add = g.accesses
    lane = Arange(j.offset.b.ssa, 0, 8)
    loaded = Loaded(0, j.offset, Cmp("slt", lane, Param("n"), 32), Const(0))
    # in an address, a mask and a branch condition
    assert gather.offset == Bin("+", Const(0), loaded)
    assert by_j.mask == Cmp("sgt", loaded, Const(0), 32) and not by_j.mask_dropped
    assert under.path == Cmp("sgt", Loaded(8, Const(0), None, None), Const(0), 32)
    assert R.loaded_indices(under.path, g) == {8}
    # an expanded tile is re-placed along the new dimension, its mask too
    (row,) = [n for n in R._nodes((tile.offset,), None) if isinstance(n, Loaded)]
    row_lane = replace(lane, dim=0)
    assert row == Loaded(
        4,
        Bin("+", Const(0), row_lane),
        Cmp("slt", row_lane, Param("n"), 32),
        Const(0),
    )
    # a load whose mask is dropped stays DataDep; the modelable conjunct of
    # a dropped mask is kept (re-placed by an expand_dims), a disjunct not
    assert (w.mask_dropped, kept.mask_dropped, kept2.mask_dropped) == (True,) * 3
    assert kept.mask == Cmp("slt", lane, Param("n"), 32)
    assert kept2.mask == Cmp("slt", row_lane, Param("n"), 32)
    assert (either.mask, either.mask_dropped) == (None, True)
    # a float load never reaches an integer term (an atomic operand here)
    assert add.atomic is not None and add.atomic_val is None
    # loaded_indices follows a loop variable into its loop's bounds
    body = """%c1 = arith.constant 1 : i32
        %lo = tt.load %q : !tt.ptr<i32>
        scf.for %k = %lo to %n step %c1 : i32 {
        %a = tt.addptr %q, %k : !tt.ptr<i32>, i32
        tt.store %a, %c1 : !tt.ptr<i32>
        }"""
    csr = parse_ttir(_module(body, "%q: !tt.ptr<i32>, %n: i32"), multipath=True)
    assert csr.accesses[1].offset == Bin("+", Const(0), LoopVar("%loop"))
    assert R.loaded_indices(csr.accesses[1].offset, csr) == {0}


def test_cf_blocks_read_with_block_predicates_and_merged_arguments():
    g = _mp("ttir/golden_nested_guard_merge_sm80.ttir")
    pid, n, T = Pid(0), Param("n"), Param("T")
    go_on = Not(Cmp("sge", pid, T, 32))
    first = BoolBin(
        "and",
        BoolBin("and", go_on, Cmp("eq", pid, Const(0), 32)),
        Not(Cmp("slt", n, Const(0), 32)),
    )
    other = BoolBin("and", go_on, Not(Cmp("eq", pid, Const(0), 32)))
    load, store = g.accesses
    # ^bb5's predicate, its two incoming edges, is the path; its argument
    # (`base`) a Select over what they pass
    assert load.path == store.path == BoolBin("or", first, other)
    base = Select(first, Const(0), Bin("+", Bin("*", pid, Const(64), 32), n, 32))
    assert load.offset == Bin(
        "+", Const(0), Bin("+", base, Arange(load.offset.b.b.ssa, 0, 64), 32)
    )
    # both arms of one cond_br into one block
    (store,) = _mp("ttir/crafted_same_dest.ttir").accesses
    c, a, b = Cmp("slt", Param("arg1"), Param("arg2"), 32), Param("arg1"), Param("arg2")
    assert store.path == BoolBin("and", BoolBin("or", c, Not(c)), c)
    assert store.offset == Bin("+", Const(0), Bin("+", Select(c, a, b), a, 32))
    # an edge under an unmodeled condition leaves the block guarded
    _, store = _mp("reader_ttir/mp_early_return.ttir").accesses
    assert (store.guarded, store.path) == (True, None)
    # an address handed to inline asm through a block argument
    asm = (
        "%i = tt.ptr_to_int %p : !tt.ptr<f32> -> i64\ncf.br ^bb1(%i : i64)\n"
        '^bb1(%x: i64):\n%y = tt.elementwise_inline_asm "mov.b64 $0, $1;" '
        '{constraints = "=l,l", packed_element = 1 : i32, pure = true} %x : i64 -> i64'
    )
    with pytest.raises(UnsupportedTTIR, match="computed from a pointer"):
        parse_ttir(_module(asm), multipath=True)
    # but not one whose block argument only the branch condition's is
    clean = asm.replace(
        "cf.br ^bb1(%i : i64)",
        "%z = arith.constant 0 : i64\n%c = arith.cmpi ne, %i, %z : i64\n"
        "cf.cond_br %c, ^bb1(%z : i64), ^bb1(%z : i64)",
    )
    parse_ttir(_module(clean), multipath=True)


@pytest.mark.parametrize(
    "blocks",
    [
        # a block entered before its predecessor holds an access
        "cf.cond_br %c, ^bb2, ^bb1\n^bb1:\ntt.store %p, %n : !tt.ptr<i32>\n"
        "tt.return\n^bb2:\ncf.cond_br %c, ^bb1, ^bb3\n^bb3:",
        # a block branching back to itself
        "cf.cond_br %c, ^bb1, ^bb2\n^bb1:\ncf.br ^bb1\n^bb2:",
        # an address made from the argument of a block entered early
        "cf.cond_br %c, ^bb2, ^bb1(%n : i32)\n^bb1(%x: i32):\n"
        "%q = tt.addptr %p, %x : !tt.ptr<i32>, i32\ntt.return\n"
        "^bb2:\ncf.br ^bb1(%n : i32)\n^bb3:",
    ],
)
def test_cf_blocks_out_of_order_or_cyclic_refuse(blocks):
    text = _module(blocks, "%p: !tt.ptr<i32>, %n: i32, %c: i1")
    with pytest.raises(UnsupportedTTIR, match="entered") as e:
        parse_ttir(text, multipath=True)
    assert e.value.kind is TTIRKind.CONTROL_FLOW


_STORE_AT_X = "^bb1(%x: i32):\n%q = tt.addptr %p, %x : !tt.ptr<i32>, i32\n"
_STORE_AT_X += "tt.store %q, %n : !tt.ptr<i32>"


@pytest.mark.parametrize(
    "blocks, match",
    [
        # an edge under an unmodeled condition: a merged value is unknown
        (
            "%v = tt.load %f : !tt.ptr<f32>\n%z = arith.constant 0.0 : f32\n"
            "%c = arith.cmpf ogt, %v, %z : f32\ncf.cond_br %c, ^bb1(%n : i32), "
            "^bb1(%n2 : i32)\n" + _STORE_AT_X,
            "block argument",
        ),
        # an unmodelable incoming value
        (
            "%v = tt.load %f : !tt.ptr<f32>\n%i = arith.fptosi %v : f32 to i32\n"
            "%c = arith.cmpi slt, %n, %n2 : i32\ncf.cond_br %c, ^bb1(%i : i32), "
            "^bb1(%n : i32)\n" + _STORE_AT_X,
            "block argument",
        ),
        # pointers of two bases
        (
            "%c = arith.cmpi slt, %n, %n2 : i32\ncf.cond_br %c, ^bb1(%p : "
            "!tt.ptr<i32>), ^bb1(%p2 : !tt.ptr<i32>)\n^bb1(%x: !tt.ptr<i32>):\n"
            "tt.store %x, %n : !tt.ptr<i32>",
            "block argument",
        ),
        # a branch in generic form names no successors the reader reads
        ('"cf.br"()[^bb1] : () -> ()\n^bb1:', "control flow cf.br"),
    ],
)
def test_cf_block_arguments_merge_only_modeled_values(blocks, match):
    args = "%p: !tt.ptr<i32>, %p2: !tt.ptr<i32>, %f: !tt.ptr<f32>, %n: i32, %n2: i32"
    with pytest.raises(UnsupportedTTIR, match=match):
        parse_ttir(_module(blocks, args), multipath=True)


@pytest.mark.parametrize(
    "body, kind",
    [
        # a bitwise op the reader does not model over an integer load
        (
            "%j = tt.load %q : !tt.ptr<i32>\n%c1 = arith.constant 1 : i32\n"
            "%x = arith.xori %j, %c1 : i32\n"
            "%a = tt.addptr %q, %x : !tt.ptr<i32>, i32\n"
            "tt.store %a, %c1 : !tt.ptr<i32>",
            TTIRKind.INDIRECT_ADDRESS,
        ),
        # casts of an integer load, in a loop bound
        (
            "%j = tt.load %q : !tt.ptr<i32>\n%f = arith.sitofp %j : i32 to f32\n"
            "%u = arith.fptosi %f : f32 to i32\n%c0 = arith.constant 0 : i32\n"
            "%c1 = arith.constant 1 : i32\n"
            "scf.for %i = %c0 to %u step %c1 : i32 {\n"
            "tt.store %q, %c0 : !tt.ptr<i32>\n}",
            TTIRKind.DATA_DEPENDENT_BOUND,
        ),
        # a float load cast to an integer, in an address
        (
            "%v = tt.load %p : !tt.ptr<f32>\n%i = arith.fptosi %v : f32 to i32\n"
            "%a = tt.addptr %q, %i : !tt.ptr<i32>, i32\n"
            "tt.store %a, %i : !tt.ptr<i32>",
            TTIRKind.INDIRECT_ADDRESS,
        ),
        # i1 arithmetic over a load
        (
            "%b = tt.load %r : !tt.ptr<i1>\n%s = arith.addi %b, %b : i1\n"
            "%e = arith.extui %s : i1 to i32\n"
            "%a = tt.addptr %q, %e : !tt.ptr<i32>, i32\n"
            "tt.store %a, %e : !tt.ptr<i32>",
            TTIRKind.INDIRECT_ADDRESS,
        ),
        # a loop-carried integer that starts as a load
        (
            "%j = tt.load %q : !tt.ptr<i32>\n%c0 = arith.constant 0 : i32\n"
            "%c1 = arith.constant 1 : i32\n%o = scf.for %i = %c0 to %c1 step %c1 "
            "iter_args(%x = %j) -> (i32) : i32 {\n"
            "%a = tt.addptr %q, %x : !tt.ptr<i32>, i32\n"
            "tt.store %a, %c0 : !tt.ptr<i32>\nscf.yield %x : i32\n}",
            TTIRKind.INDIRECT_ADDRESS,
        ),
    ],
)
def test_a_value_made_from_a_load_refuses_alike_in_both_modes(body, kind):
    text = _module(body, "%p: !tt.ptr<f32>, %q: !tt.ptr<i32>, %r: !tt.ptr<i1>")
    for options in ({}, {"multipath": True}):
        with pytest.raises(UnsupportedTTIR) as e:
            parse_ttir(text, **options)
        assert e.value.kind is kind
