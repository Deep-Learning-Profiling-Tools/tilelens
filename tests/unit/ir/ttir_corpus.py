"""The TTIR the IR tests read, compiled from kernels at test time.

``text("ttir/<name>.ttir")`` is the TTIR Triton 3.8 prints for spec
``<name>`` of ``ttir_kernels.py``, and ``text("reader_ttir/<name>.ttir")``
the TTIR of kernel ``<name>`` of ``reader_kernels.py``: host-compiled up to
the TTIR stage only (no GPU, no Triton cache; the whole corpus takes well
under a second), each loc's path cut back to ``tests/`` or ``triton/``,
once per process. The ``crafted_*`` texts are hand-written TTIR for shapes
no kernel prints reliably (empty region bodies, generic form, quoted
symbols, loc forms, cf edge cases).

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

KERNELS = Path(__file__).resolve().parent

CRAFTED = {
    "ttir/crafted_attr_dicts.ttir": r"""module {
  tt.func public @k(%p: !tt.ptr<i32>) attributes {noinline = false} {
    %c = arith.constant {tt.divisibility = dense<16> : tensor<1xi32>} 16 : i32
    %d = arith.constant {axis = 5 : i32} -3 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %s = "tt.reduce"(%r) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %t = arith.addi %a, %b : i32
      tt.reduce.return %t : i32
    }) {tt.divisibility = dense<16> : tensor<1xi32>, axis_note = 3 : i32} : (tensor<64xi32>) -> i32
    %e = tt.expand_dims %r {axis = 0 : i32, tt.note = "axis = 1"} : tensor<64xi32> -> tensor<1x64xi32>
    %q = tt.addptr %p, %s : !tt.ptr<i32>, i32
    %q2 = tt.addptr %q, %c : !tt.ptr<i32>, i32
    %q3 = tt.addptr %q2, %d : !tt.ptr<i32>, i32
    tt.store %q3, %s : !tt.ptr<i32>
    tt.return
  }
}
""",
    "ttir/crafted_empty_bodies.ttir": r"""#loc = loc("k.py":1:0)
module {
  tt.func public @k(%p: !tt.ptr<i32> loc("p"(#loc)), %c: i1 loc("c"(#loc)), %n: i32 loc("n"(#loc))) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32 loc(#loc1)
    %c1 = arith.constant 1 : i32 loc(#loc1)
    scf.if %c {
    } else {
      tt.store %p, %c0 : !tt.ptr<i32> loc(#loc2)
    } loc(#loc1)
    scf.if %c {
      tt.store %p, %c1 : !tt.ptr<i32> loc(#loc2)
    } else {
    } loc(#loc1)
    scf.for %i = %c0 to %n step %c1  : i32 {
    } loc(#loc3)
    scf.if %c {
    } loc(#loc1)
    tt.return loc(#loc4)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("k.py":2:4)
#loc2 = loc("k.py":3:8)
#loc3 = loc("k.py":4:4)
#loc4 = loc("k.py":5:4)
""",
    "ttir/crafted_empty_else.ttir": r"""module {
  tt.func public @k(%p: !tt.ptr<i32>, %c: i1) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    scf.if %c {
      tt.store %p, %c0 : !tt.ptr<i32>
    } else {
    }
    tt.return
  }
}
""",
    "ttir/crafted_empty_for.ttir": r"""module {
  tt.func public @k(%p: !tt.ptr<i32>, %n: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
    }
    tt.store %p, %c0 : !tt.ptr<i32>
    tt.return
  }
}
""",
    "ttir/crafted_generic_form.ttir": r"""module {
  tt.func public @k(%p: !tt.ptr<i32>, %a: i32, %b: i32) attributes {noinline = false} {
    %c = "arith.cmpi"(%a, %b) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %x = "arith.select"(%c, %a, %b) : (i1, i32, i32) -> i32
    %o = "tt.atomic_rmw"(%p, %x) <{atomic_rmw_op = 5 : i32, scope = 1 : i32, sem = 4 : i32}> : (!tt.ptr<i32>, i32) -> i32
    %pid = "tt.get_program_id"() <{axis = 1 : i32}> : () -> i32
    %q = tt.addptr %p, %pid : !tt.ptr<i32>, i32
    tt.store %q, %o : !tt.ptr<i32>
    tt.return
  }
}
""",
    "ttir/crafted_locs.ttir": r"""#loc = loc("k.py":1:0)
#loc1 = loc("k.py":2:5)
#loc2 = loc("k.py":3:6)
#loc9 = loc(fused<"meta">[#loc1, #loc2])
#loc10 = loc(callsite(#loc1 at #loc9))
module {
  tt.func public @k(%p: !tt.ptr<i32> loc("p"(#loc)), %n: i32 loc(unknown)) attributes {noinline = false} {
    %c1 = arith.constant 1 : i32 loc(fused[#loc1, "x.py":7:3])
    %a = arith.addi %n, %c1 : i32 loc(#loc9)
    %b = arith.addi %a, %c1 : i32 loc(callsite("inner"("y.py":3:4) at callsite(#loc1 at #loc2)))
    %q = tt.addptr %p, %b : !tt.ptr<i32>, i32 loc("q.py":9:9)
    tt.store %q, %a : !tt.ptr<i32> loc(#loc10)
    tt.return loc(#loc11)
  } loc(#loc)
} loc(#loc)
#loc11 = loc("k.py":12:1)
""",
    "ttir/crafted_same_dest.ttir": r"""module {
  tt.func public @k(%p: !tt.ptr<i32>, %a: i32, %b: i32) attributes {noinline = false} {
    %c = arith.cmpi slt, %a, %b : i32
    cf.cond_br %c, ^bb1(%a : i32), ^bb1(%b : i32)
  ^bb1(%x: i32):
    %y = arith.addi %x, %a : i32
    cf.cond_br %c, ^bb2(%y, %x : i32, i32), ^bb3
  ^bb2(%u: i32, %v: i32):
    %q = tt.addptr %p, %u : !tt.ptr<i32>, i32
    tt.store %q, %v : !tt.ptr<i32>
    cf.br ^bb3
  ^bb3:
    tt.return
  }
}
""",
    "ttir/crafted_symbols_strings.ttir": r"""module {
  tt.func private @"f{%x} \22q\22 (a)"(%a: i32, %b: i32) -> (i32, i32) attributes {noinline = true} {
    %s = arith.addi %a, %b : i32
    tt.return %s, %a : i32, i32
  }
  tt.func public @k(%p: !tt.ptr<i32>, %n: i32) attributes {noinline = false} {
    %r:2 = tt.call @"f{%x} \22q\22 (a)"(%n, %n) : (i32, i32) -> (i32, i32)
    %y = tt.elementwise_inline_asm "{ mov.u32 $0, %tid.x; } // loc(\22x\22) }" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %r#0 : i32 -> i32
    %c = arith.cmpi sgt, %y, %r#1 : i32
    tt.assert %c, "bad { } loc( \22 %x" : i1
    tt.print " p={%d} " {hex = false, isSigned = array<i32: 1>} : %y : i32
    %q = tt.addptr %p, %y : !tt.ptr<i32>, i32
    tt.store %q, %r#1 : !tt.ptr<i32>
    tt.return
  }
}
""",
    "ttir/crafted_unicode_strings.ttir": r"""#loc = loc("/tmp/\E5\86\85\E6\A0\B8/k.py":1:0)
#loc1 = loc("/tmp/\E5\86\85\E6\A0\B8/k.py":2:4)
module {
  tt.func public @"\E6\A0\B8"(%p: !tt.ptr<f32> loc("\CF\80_ptr"(#loc)), %c: i1 loc("\E6\95\B0"(#loc))) attributes {noinline = false} {
    %v = tt.load %p : !tt.ptr<f32> loc("\E5\80\BC"(#loc1))
    tt.assert %c, "\E9\94\99\E8\AF\AF \22quoted\22 \\ back" : i1 loc(#loc1)
    tt.print "\CF\80=" {hex = false, isSigned = array<i32: 0>} : %v : f32 loc(#loc1)
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
""",
}

# A loc's path up to this repository's tests/ or Triton's package directory:
# where the checkout and Triton happen to be on the machine that compiles.
_REPO_LOC = r'loc\("[^"]*/tests/(?=unit/ir/)'
_TRITON_LOC = r'loc\("[^"]*/triton/'


def portable(text: str) -> str:
    """``text`` with each loc naming a file of this repository from
    ``tests/`` and one of Triton's own sources (tl.cdiv, ...) from
    ``triton/``, so a text reads the same on every machine."""
    return re.sub(_TRITON_LOC, 'loc("triton/', re.sub(_REPO_LOC, 'loc("tests/', text))


def _load(name: str) -> Any:
    """Kernel module ``name`` of this directory: its @triton.jit functions
    built as real JITFunctions even after tests/unit/test_multithreading.py
    set TRITON_INTERPRET=1 at import time."""
    from triton import knobs

    spec = importlib.util.spec_from_file_location(name, KERNELS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Triton reads the jit fn's module
    missing = object()
    previous = knobs.runtime.__dict__.get("interpret", missing)
    knobs.runtime.__dict__["interpret"] = False
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is missing:
            knobs.runtime.__dict__.pop("interpret", None)
        else:
            knobs.runtime.__dict__["interpret"] = previous
    return module


@functools.cache
def _kernels() -> tuple[Any, Any]:
    return _load("ttir_kernels"), _load("reader_kernels")


NAMES = sorted(
    [f"ttir/{n}.ttir" for n in _kernels()[0].SPECS]
    + [f"reader_ttir/{n}.ttir" for n in _kernels()[1].SPECS]
    + list(CRAFTED)
)


def _real_compiles_available() -> bool:
    # Triton imported under TRITON_INTERPRET=1 builds its own standard library
    # as InterpretedFunctions, so nothing can compile for real in-process.
    import triton.language.standard as tl_standard
    from triton.runtime.jit import JITFunction

    return isinstance(tl_standard.cdiv, JITFunction)


def _ttir(fn, sig, consts, cc=80, options=None, attrs=None) -> str:
    """triton.compile's front half for a ``cuda:<cc>`` target, stopped
    after the TTIR stage."""
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import make_backend

    src = ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs or {})
    target = GPUTarget("cuda", cc, 32)
    backend = make_backend(target)
    opts = backend.parse_options({"num_warps": 4, **(options or {})})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    module = src.make_ir(
        target,
        opts,
        backend.get_codegen_implementation(opts),
        backend.get_module_map(),
        ctx,
    )
    stages: dict[str, Any] = {}
    backend.add_stages(stages, opts, src.language)
    return str(stages["ttir"](module, {"hash": "", "target": target, **opts.__dict__}))


@functools.cache
def text(name: str) -> str:
    """The TTIR ``name`` (``ttir/<n>.ttir`` or ``reader_ttir/<n>.ttir``, one
    of NAMES)."""
    if name in CRAFTED:
        return CRAFTED[name]
    if not _real_compiles_available():
        pytest.skip(
            "Triton was imported under TRITON_INTERPRET=1: nothing compiles",
            allow_module_level=True,  # a test module may read a text at import
        )
    walk, reader = _kernels()
    kind, stem = name.removesuffix(".ttir").split("/")
    if kind == "ttir":
        return portable(_ttir(*walk.SPECS[stem]))
    sig, consts = reader.SPECS[stem]
    return portable(_ttir(getattr(reader, stem), sig, consts))
