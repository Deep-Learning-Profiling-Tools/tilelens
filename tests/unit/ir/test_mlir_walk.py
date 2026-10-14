"""tilelens.ir._mlir_walk: the bindings-first walk under the TTIR reader.

The corpus is tests/unit/ir/ttir_corpus.py: the TTIR Triton 3.8 prints for
tests/unit/ir/ttir_kernels.py's kernels, compiled at test time, and
hand-written texts.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tilelens.ir import _mlir_walk as W

from . import ttir_corpus

REPO = Path(__file__).resolve().parents[3]
FILES = [n.removeprefix("ttir/") for n in ttir_corpus.NAMES if n.startswith("ttir/")]


def _text(name: str) -> str:
    return ttir_corpus.text(f"ttir/{name}")


def _ops(m: W.Module, name: str) -> list[W.Op]:
    return [op for op in m.ops if op.name == name]


@pytest.fixture(autouse=True)
def _fresh_cache():
    W._CACHE.clear()
    yield
    W._CACHE.clear()


# ─────────────────────────── the corpus ───────────────────────────


def test_generic_form_reads_like_the_custom_form():
    # the bindings read the attributes, whatever form the text prints
    m = W._walk(_text("crafted_generic_form.ttir"))
    assert _ops(m, "arith.cmpi")[0].attrs["predicate"] == "slt"
    assert _ops(m, "tt.get_program_id")[0].attrs["axis"] == 1
    (rmw,) = _ops(m, "tt.atomic_rmw")
    assert dict(rmw.attrs) == {"rmw_op": "fadd", "sem": "acq_rel", "scope": "gpu"}


def test_structure_records_are_consistent():
    for name in FILES:
        m = W._walk(_text(name))
        assert m.ops[0].name == "builtin.module" and m.ops[0].path == ()
        assert [op.index for op in m.ops] == list(range(len(m.ops)))
        assert [b.index for b in m.blocks] == list(range(len(m.blocks)))
        assert [v.index for v in m.values] == list(range(len(m.values)))
        for op in m.ops:
            for v in op.results:
                assert m.values[v].op == op.index
            for k, blocks in enumerate(op.regions):
                for pos, b in enumerate(blocks):
                    blk = m.blocks[b]
                    assert (blk.op, blk.region, blk.position) == (op.index, k, pos)
                    for i in blk.ops:
                        assert m.ops[i].path == op.path + (b,)
            if op.path:
                assert m.blocks[op.block].ops[op.position] == op.index
        for v in m.values:
            assert (v.op is None) != (v.block is None)
        # pre-order: an op's regions come after it, its uses are in-module values
        for op in m.ops:
            assert all(0 <= v < len(m.values) for v in op.operands)
            assert op.operand_types == tuple(m.values[v].type for v in op.operands)


# ─────────────────────────── a misread fails closed ───────────────────────────
# The bindings parse the original text while the text scan reads an edited
# copy: a scan that misreads the module must never align silently.


@pytest.mark.parametrize(
    "misread, match",
    [
        # the op name the line prints is not the op the bindings walked
        (lambda ln: ln.replace("arith.cmpi", "arith.subi"), "text op 'arith.subi'"),
        # the line's loc, the anchor to the bindings, names another site
        (
            lambda ln: re.sub(r"loc\(#loc\d+\)$", 'loc("k.py":1:1)', ln),
            "result 0 loc differs",
        ),
    ],
    ids=["op-name", "result-loc"],
)
def test_a_misread_reports_its_line(misread, match):
    text = _text("golden_add_sm80.ttir")
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if "arith.cmpi slt" in ln)
    lines[i] = misread(lines[i])
    with pytest.raises(W.MisalignedModule, match=match) as e:
        W._walk(text, scan_text="\n".join(lines))
    assert e.value.line_no == i + 1


# a zero-result op's loc has no second source: a garbled trailer must not
# read as "no loc" while the rest of the module prints locs
@pytest.mark.parametrize(
    "old, new", [("tt.return loc(", "tt.return oc("), ("} loc(#loc)", "} loc#(#loc)")]
)
def test_a_garbled_loc_is_not_read_as_no_loc(old, new):
    text = _text("golden_add_sm80.ttir")
    assert old in text
    with pytest.raises(W.MisalignedModule, match="prints no loc"):
        W._walk(text, scan_text=text.replace(old, new, 1))


# ─────────────────────────── what the walk reads ───────────────────────────


def test_unicode_names_and_locs():
    m = W._walk(_text("crafted_unicode_strings.ttir"))
    assert [f.sym_name for f in m.funcs] == ["核"]
    assert [a.name for a in m.funcs[0].args] == ["π_ptr", "数"]
    assert _ops(m, "tt.load")[0].loc_name == "值"
    assert {op.loc.file for op in m.ops if op.loc} == {"/tmp/内核/k.py"}


@pytest.mark.parametrize(
    "body, value",
    [
        ("\\E9\\94\\99", "错"),
        ("a\\22b\\\\c", 'a"b\\c'),
        ("tab\\09nl\\0A", "tab\tnl\n"),
        ("\\n\\t", "\n\t"),
        ("plain", "plain"),
    ],
)
def test_unescape_decodes_utf8_bytes(body, value):
    assert W._unescape(body) == value


@pytest.mark.parametrize("body", ["\\E9\\94", "\\q", "\\"])
def test_unescape_rejects_malformed(body):
    with pytest.raises(ValueError):
        W._unescape(body)


def test_constant_kinds():
    m = W._walk(_text("adv_consts.ttir"))
    values = [op.attrs["value"] for op in _ops(m, "arith.constant")]
    assert -9223372036854775807 in values and ("float", None) in values
    text = _text("crafted_attr_dicts.ttir")
    assert [op.attrs["value"] for op in _ops(W._walk(text), "arith.constant")] == [
        16,
        -3,
    ]
    dense = "%ds = arith.constant dense<[1, 2]> : tensor<2xi32>\n    %r = tt.make_range"
    m = W._walk(text.replace("%r = tt.make_range", dense, 1))
    assert ("dense", None) in [op.attrs["value"] for op in _ops(m, "arith.constant")]


@pytest.mark.parametrize(
    "name, shapes",
    [
        ("crafted_empty_else.ttir", {"scf.if": [(1, 1)]}),
        ("crafted_empty_for.ttir", {"scf.for": [(1,)]}),
        (
            "crafted_empty_bodies.ttir",
            {"scf.if": [(1, 1), (1, 1), (1, 0)], "scf.for": [(1,)]},
        ),
    ],
)
def test_empty_region_bodies(name, shapes):
    m = W._walk(_text(name))
    for opname, want in shapes.items():
        assert [tuple(len(r) for r in op.regions) for op in _ops(m, opname)] == want
    for blk in m.blocks:
        if m.ops[blk.op].name in ("scf.if", "scf.for"):
            last = m.ops[blk.ops[-1]]
            assert last.name == "scf.yield"
            if last.implicit:
                assert last.line_no is None and last.operands == () and last.loc is None


def test_empty_for_body_keeps_its_induction_variable():
    m = W._walk(_text("crafted_empty_for.ttir"))
    (loop,) = _ops(m, "scf.for")
    (body,) = loop.regions[0]
    assert len(m.blocks[body].args) == 1 and m.blocks[body].arg_types == ("i32",)
    assert [m.ops[i].implicit for i in m.blocks[body].ops] == [True]


def test_descriptor_operand_order():
    m = W._walk(_text("adv_descs.ttir"))
    tile, row = "!tt.tensordesc<32x32xf16>", "!tt.tensordesc<1x32xf16>"
    (store,) = _ops(m, "tt.descriptor_store")
    (red,) = _ops(m, "tt.descriptor_reduce")
    (scatter,) = _ops(m, "tt.descriptor_scatter")
    types = lambda op: [m.values[v].type for v in op.operands]  # noqa: E731
    # ODS order (desc, src, indices...), printed `%desc[%i, %j], %src`
    assert types(store) == [tile, "tensor<32x32xf16>", "i32", "i32"]
    assert types(red) == types(store)
    # descriptor_scatter prints in ODS order (desc, x_offsets, y_offset, src)
    assert types(scatter) == [row, "tensor<32xi32>", "i32", "tensor<32x32xf16>"]


def test_dot_scaled_operand_order():
    m = W._walk(_text("kernel_dot_scaled.ttir"))
    (d,) = _ops(m, "tt.dot_scaled")
    # ODS (a, b, c, a_scale, b_scale), printed `%a scale %as, %b scale %bs, %c`
    assert [m.values[v].type for v in d.operands] == [
        "tensor<128x64xf8E4M3FN>",
        "tensor<64x128xf8E4M3FN>",
        "tensor<128x128xf32>",
        "tensor<128x2xi8>",
        "tensor<128x2xi8>",
    ]
    assert [m.values[v].name for v in d.operands[3:]] == ["a_scale", "b_scale"]


def test_cf_blocks_align():
    m = W._walk(_text("golden_nested_guard_merge_sm80.ttir"))
    (func,) = _ops(m, "tt.func")
    labels = [m.blocks[b].label for b in func.regions[0]]
    assert labels[0] is None and "^bb5" in labels and len(labels) == 6
    assert all(op.loc is not None for op in _ops(m, "cf.br") + _ops(m, "cf.cond_br"))


def test_parser_invented_locs_are_absent():
    m = W._walk(_text("spike_spin_while.ttir"))
    whiles = _ops(m, "scf.while")
    before = m.blocks[whiles[1].regions[0][0]]
    assert before.arg_names == (None,)  # `scf.while (%v_1 = %v)` prints no arg loc
    for op in m.ops:
        for loc in (op.loc, *op.callers):
            assert loc is None or not loc.file.startswith(
                ("/proc/self/fd", tempfile.gettempdir())
            )


def test_callsite_and_fused_locs():
    m = W._walk(_text("crafted_locs.ttir"))
    b = [op for op in m.ops if op.name == "arith.addi"][1]
    assert b.loc == W.SourceLoc("y.py", 3, 4) and b.loc_name == "inner"
    assert b.callers == (W.SourceLoc("k.py", 2, 5), W.SourceLoc("k.py", 3, 6))
    (store,) = _ops(m, "tt.store")
    assert store.loc == W.SourceLoc("k.py", 2, 5) and store.callers == (
        W.SourceLoc("k.py", 2, 5),
    )
    (ret,) = _ops(m, "tt.return")
    assert ret.loc == W.SourceLoc("k.py", 12, 1)  # alias defined after its use


def test_quoted_symbols_and_strings_with_syntax_chars():
    m = W._walk(_text("crafted_symbols_strings.ttir"))
    assert [(f.sym_name, f.visibility) for f in m.funcs] == [
        ('f{%x} "q" (a)', "private"),
        ("k", "public"),
    ]
    (call,) = _ops(m, "tt.call")
    assert call.attrs["callee"] == 'f{%x} "q" (a)' and len(call.results) == 2
    (asm,) = _ops(m, "tt.elementwise_inline_asm")
    assert asm.attrs["pure"] is True


def test_program_id_axes():
    m = W._walk(_text("golden_tile2d_sm80.ttir"))
    assert sorted(op.attrs["axis"] for op in _ops(m, "tt.get_program_id")) == [0, 1]


def _const_module(lit: str, ty: str) -> str:
    return (
        "module {\n"
        "  tt.func public @k() attributes {noinline = false} {\n"
        f"    %c = arith.constant {lit} : {ty}\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


@pytest.mark.parametrize(
    "lit, ty, value",
    [
        # the parser accepts these and MLIR holds the bits: read as the
        # printer prints them (signed; i1 as a bool)
        ("4294967295", "i32", -1),
        ("2147483648", "i32", -(2**31)),
        ("255", "i8", -1),
        ("dense<4294967295>", "tensor<4xi32>", -1),
        ("18446744073709551615", "i64", -1),
        ("dense<1>", "tensor<4xi1>", True),
    ],
)
def test_constant_outside_the_printed_signed_range(lit, ty, value):
    (c,) = _ops(W._walk(_const_module(lit, ty)), "arith.constant")
    assert c.attrs["value"] == value


@pytest.mark.parametrize(
    "lit, ty, value",
    [
        ("-1", "i32", -1),
        ("2147483647", "i32", 2**31 - 1),
        ("-2147483648", "i32", -(2**31)),
        ("-128", "i8", -128),
        ("dense<-1>", "tensor<4xi32>", -1),
        ("-9223372036854775808", "i64", -(2**63)),
        ("9223372036854775807", "index", 2**63 - 1),
    ],
)
def test_constant_in_the_printed_signed_range(lit, ty, value):
    (c,) = _ops(W._walk(_const_module(lit, ty)), "arith.constant")
    assert c.attrs["value"] == value


_READ_CONSTANTS = r"""
import sys
sys.path.insert(0, sys.argv[1])
from tilelens.ir import _mlir_walk as W
for text in sys.argv[2:]:
    (c,) = [op for op in W.walk_module(text).ops if op.name == "arith.constant"]
    print(c.attrs["value"])
"""


def test_a_constant_wider_than_64_bits_does_not_abort_the_process():
    # get_constant_value asserts on one (APInt::getSExtValue) and the
    # assertion aborts the process; in a child, so a regression fails here
    # instead of killing the test run
    texts = [
        _const_module(str(2**127 - 1), "i128"),
        _const_module(f"dense<{-(2**126)}>", "tensor<4xi128>"),
    ]
    r = subprocess.run(
        [sys.executable, "-c", _READ_CONSTANTS, str(REPO), *texts],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
    assert r.stdout.splitlines() == ["('dense', None)", "('dense', None)"]


_FOR = """module {
  tt.func public @k(%p: !tt.ptr<i32>, %n: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for KW%i = %c0 to %n step %c1  : i32 {
      %q = tt.addptr %p, %i : !tt.ptr<i32>, i32
      tt.store %q, %i : !tt.ptr<i32>
    }
    tt.return
  }
}
"""


def test_scf_for_unsigned_compare_is_recorded():
    signed, unsigned = _FOR.replace("KW", ""), _FOR.replace("KW", "unsigned ")
    (loop,) = _ops(W._walk(signed), "scf.for")
    assert loop.attrs["unsignedCmp"] is False
    (loop,) = _ops(W._walk(unsigned), "scf.for")
    assert loop.attrs["unsignedCmp"] is True
    # an unknown header keyword is a misread, never an ignored word
    with pytest.raises(W.MisalignedModule, match="scf.for header"):
        W._walk(unsigned, scan_text=_FOR.replace("KW", "signless "))


def test_ttgir_is_refused():
    text = "#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\n"
    text += _text("golden_add_sm80.ttir")
    with pytest.raises((W.MisalignedModule, W.ModuleParseError)):
        W._walk(text)


# ─────────────────────────── parse errors ───────────────────────────


def test_parse_error_carries_the_diagnostic(capfd):
    text = _text("golden_add_sm80.ttir")
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if "tt.make_range {" in ln)
    lines[i] = lines[i].replace("tt.make_range {", "tt.make_range_v2 {")
    with pytest.raises(W.ModuleParseError) as e:
        W.walk_module("\n".join(lines))
    assert (
        "tt.make_range_v2" in e.value.diagnostic and "is unknown" in e.value.diagnostic
    )
    assert "<ttir>" in e.value.diagnostic and "/proc/self/fd" not in e.value.diagnostic
    assert e.value.line_no == i + 1
    out, err = capfd.readouterr()
    assert "make_range_v2" not in err  # captured, not leaked onto the process stderr


# ─────────────────────────── the release, barriers, block pointers ───────────────────────────


def _barrier_module(barrier: str) -> str:
    return (
        "module {\n"
        "  tt.func public @k(%p: !tt.ptr<f32>) attributes {noinline = false} {\n"
        f"    {barrier}\n"
        "    %v = tt.load %p : !tt.ptr<f32>\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


def test_the_barrier_aligns():
    m = W._walk(_barrier_module("ttg.barrier all"))
    (barrier,) = _ops(m, "ttg.barrier")
    assert not barrier.results and barrier.loc is None
    # a combination of address spaces prints as `a|b`; the text never reads it
    assert _ops(
        W._walk(_barrier_module("ttg.barrier local|global_read")), "ttg.barrier"
    )
    # gpu.barrier still parses (the reader refuses it, see test_ttir_reader)
    assert _ops(W._walk(_barrier_module("gpu.barrier")), "gpu.barrier")


def test_another_triton_release_is_refused_before_any_parse(monkeypatch):
    import triton

    monkeypatch.setattr(W, "_walk", lambda *a: pytest.fail("walked"))
    monkeypatch.setattr(triton, "__version__", "3.6.0")
    with pytest.raises(W.ModuleParseError, match="Triton 3.8.x only.*is 3.6.0"):
        W.walk_module(_barrier_module(""))


_SPLIT_NEWLINE = "!tt.ptr<\n tensor<32xf32>>"
_SPLIT_COMMENT = "!tt.ptr< // c\n tensor<32xf32>>"

_BLOCK_PTR = """module {
  tt.func public @k(%p: !tt.ptr<f32>, %b: TYPE) attributes {noinline = false} {
    tt.return
  }
}
"""


@pytest.mark.parametrize(
    "text, line",
    [
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr<tensor<32x32xf32>>"), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr<tensor<32xf32>, 1>"), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr< tensor <4xf32>>"), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt<ptr<tensor<4xf32>>>"), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr<!tt.ptr<tensor<4xf32>>>"), 2),
        # split over lines, or around a comment: no single line holds it
        (_BLOCK_PTR.replace("TYPE", _SPLIT_NEWLINE), 2),
        (_BLOCK_PTR.replace("TYPE", _SPLIT_COMMENT), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr\n<tensor<4xf32>>"), 2),
        (_BLOCK_PTR.replace("TYPE", '!tt.ptr< // "q>\n // b\n tensor<4xf32>>'), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt<ptr // c\n<tensor<4xf32>>>"), 2),
        # a type alias may hide the pointee
        ("!t = tensor<4xf32>\n" + _BLOCK_PTR.replace("TYPE", "!tt.ptr<!t>"), 1),
        ("!t // c\n = tensor<4xf32>\n" + _BLOCK_PTR.replace("TYPE", "!tt.ptr<!t>"), 1),
        (
            "// c\n\n  !t\n = tensor<4xf32>\n"
            + _BLOCK_PTR.replace("TYPE", "!tt.ptr<!t>"),
            3,
        ),
    ],
)
def test_block_pointer_types_are_refused_before_the_parser(monkeypatch, text, line):
    """The parser aborts the process on a block-pointer type."""

    def no_parse(_data):
        raise AssertionError("the bindings got a block-pointer type")

    monkeypatch.setattr(W, "_bind_walk", no_parse)
    with pytest.raises(W.ModuleParseError, match="no block pointers") as e:
        W._walk(text)
    assert e.value.line_no == line and f"line {line}:" in e.value.diagnostic


def test_the_block_pointer_screen_reads_types_not_strings():
    text = _module_with_print('"ptr<tensor<"')
    W._screen(text)
    # nor comments (the parser skips them), nor a `//` inside a string
    W._screen(text.replace("tt.return", "tt.return // !tt.ptr<tensor<4xf32>>"))
    W._screen(_module_with_print('"ptr< // "'))
    in_string = _BLOCK_PTR.replace("TYPE", "i32").replace(
        "{noinline = false}",
        '{noinline = false, s = "a // b", t = !tt.ptr<\n tensor<4xf32>>}',
    )
    with pytest.raises(W.ModuleParseError, match="line 2: a block-pointer type"):
        W._screen(in_string)
    unterminated = text.replace('"ptr<tensor<"', '"ptr<tensor<')
    with pytest.raises(W.ModuleParseError, match="block-pointer type"):
        W._screen(unterminated)


def _module_with_print(prefix: str) -> str:
    return (
        "module {\n"
        "  tt.func public @k(%x: i32) attributes {noinline = false} {\n"
        f"    tt.print {prefix} {{hex = false, isSigned = array<i32: 1>}} : %x : i32\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


# ─────────────────────────── the cache ───────────────────────────


def test_cache_returns_the_same_module_and_caches_misalignment(monkeypatch):
    text = _text("golden_add_sm80.ttir")
    assert W.walk_module(text) is W.walk_module(text)
    # a comment on an op line is a text the scan does not read
    bad = text.replace(" loc(#loc1)", " loc(#loc1) // c", 1)
    with pytest.raises(W.MisalignedModule) as first:
        W.walk_module(bad)

    def no_parse(_data):
        raise AssertionError("parsed twice")

    monkeypatch.setattr(W, "_bind_walk", no_parse)
    with pytest.raises(W.MisalignedModule) as again:
        W.walk_module(bad)
    assert (
        again.value.problems == first.value.problems and again.value is not first.value
    )
    W.walk_module(text)
