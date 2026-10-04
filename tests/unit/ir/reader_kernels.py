"""Kernels whose TTIR the reader tests read as ``reader_ttir/<name>.ttir``.

``ttir_corpus.py`` compiles them at test time; ``test_ttir_reader.py``
reads the result, and checks that a refusal's loc names the right line of
this file.

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl


# ── p1: a loop-carried pointer advanced by a loop-VARIANT amount ──
@triton.jit
def p1_variant_delta(x_ptr, out_ptr, n):
    p = x_ptr + 20
    for k in range(-8, n):  # n is a runtime scalar
        v = tl.load(p)
        tl.store(out_ptr, v)
        p += k  # the advance depends on the induction variable


# ── p2: two pointer iter_args swapped in the scf.yield ──
@triton.jit
def p2_swap(a_ptr, b_ptr, n):
    p = a_ptr
    q = b_ptr + 100
    for i in range(0, n):
        tl.store(p, 1.0)
        p, q = q, p  # ping-pong between two buffers


# ── p3: a noinline tt.call (the callee body is another tt.func) ──
@triton.jit(noinline=True)
def _p3_helper(x_ptr, pid):
    tl.store(x_ptr + pid, 1.0)


@triton.jit
def p3_call_guarded(x_ptr, n):
    # callee formal names match caller names; the call sits under `if pid < n`
    pid = tl.program_id(0)
    if pid < n:
        _p3_helper(x_ptr, pid)


@triton.jit
def p3_call_offset(x_ptr, n):
    # the ACTUAL argument differs from the caller value sharing the formal's name
    pid = tl.program_id(0)
    _p3_helper(x_ptr, pid + 100)


@triton.jit(noinline=True)
def _p3_helper_c(dst, i):
    tl.store(dst + i, 1.0)


@triton.jit
def p3_call_formals(x_ptr, n):
    # callee formals match no caller name
    pid = tl.program_id(0)
    _p3_helper_c(x_ptr, pid + 100)


# ── p4: an atomic observation reaching an address ──
@triton.jit
def p4_observed_direct(cnt_ptr, x_ptr, n):
    old = tl.atomic_add(cnt_ptr, 1)
    p = x_ptr + old % n
    v = tl.load(p)
    tl.store(p, v + 1.0)


@triton.jit
def p4_observed_loop(cnt_ptr, x_ptr, n):
    # the same address, carried through an scf.for iter_arg (offset0)
    old = tl.atomic_add(cnt_ptr, 1)
    p = x_ptr + old % n
    for i in range(0, 4):
        v = tl.load(p)
        tl.store(p, v + 1.0)
        p += 1


@triton.jit
def p4_observed_delta(cnt_ptr, x_ptr, n):
    # the observation in the per-iteration DELTA instead of offset0
    old = tl.atomic_add(cnt_ptr, 1)
    step = old % n
    p = x_ptr
    for i in range(0, 4):
        v = tl.load(p)
        tl.store(p, v + 1.0)
        p += step


# ── integer widths ──
@triton.jit
def rv_trunci_alias(x_ptr):
    # trunc_i32(pid_i64 * 2**32) is 0 for every pid: every program stores x[0]
    pid = tl.program_id(0).to(tl.int64)
    off = (pid * 4294967296).to(tl.int32)
    tl.store(x_ptr + off, 1)


@triton.jit
def rv_i32_wrap(x_ptr, S):
    # (pid * S) * S wraps to 0 in i32 for S = 65536
    pid = tl.program_id(0)
    off = (pid * S) * S
    tl.store(x_ptr + off, 1)


@triton.jit
def unsigned_index(x_ptr, n):
    # divui, cmpi ult and extui read their operands unsigned
    pid = tl.program_id(0).to(tl.uint32)
    q = pid // 3
    m = pid < n.to(tl.uint32)
    tl.store(x_ptr + q, 1.0, mask=m)


# ── inline asm ──
@triton.jit
def rv_inline_asm_store(x_ptr, OFF: tl.constexpr):
    # an impure asm st.global through a pointer cast to i64: a memory access
    # the graph cannot see
    p = (x_ptr + OFF).to(tl.int64, bitcast=True)
    v = tl.full((), 7, tl.int32)
    tl.inline_asm_elementwise(
        "st.global.b32 [$1], $2; mov.b32 $0, 0;",
        "=r,l,r",
        [p, v],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


# ── shapes the reader models ──
@triton.jit
def loop_two_step_advance(x_ptr, n, s):
    # two addptrs per iteration: the advance is their (loop-invariant) sum
    p = x_ptr
    for i in range(0, n):
        tl.store(p, 1.0)
        p += s
        p += 2


@triton.jit
def loop_observed_advance(cnt_ptr, x_ptr, n):
    # the advance is an atomic observed inside the loop: loop-variant
    p = x_ptr
    for i in range(0, n):
        tl.store(p, 1.0)
        p += tl.atomic_add(cnt_ptr, 1)


@triton.jit
def where_pointer(x_ptr, n):
    # arith.select over two pointers of one base
    offs = tl.arange(0, 16)
    p = tl.where(offs < n, x_ptr + offs, x_ptr + 100)
    tl.store(p, 1.0)


@triton.jit
def tile3d_shared_arange(x_ptr, N: tl.constexpr):
    # one make_range on all three dims of a tile: three lane variables
    r = tl.arange(0, N)
    off = r[:, None, None] * (N * N) + r[None, :, None] * N + r[None, None, :]
    tl.store(x_ptr + off, 1.0)


@triton.jit
def expand_iterarg_3d(x_ptr, out_ptr, n, N: tl.constexpr):
    # a loop-carried [N, N] pointer tile expanded to 3D inside the loop, next
    # to the same make_range at dim 0: the load reads j*N + l - i*N
    r = tl.arange(0, N)
    p = x_ptr + r[:, None] * N + r[None, :]
    for k in range(0, n):
        q = p[None, :, :] - r[:, None, None] * N
        v = tl.load(q)
        o = out_ptr + r[:, None, None] * N * N + r[None, :, None] * N + r[None, None, :]
        tl.store(o, v)
        p += 1


@triton.jit
def expand_iterarg_mask(x_ptr, out_ptr, n, M, N: tl.constexpr):
    # a loop-carried 1D pointer expanded to 2D inside the loop, masked by the
    # same make_range at the pointer's lane (dim 1)
    r = tl.arange(0, N)
    p = x_ptr + r
    for k in range(0, n):
        q = p[None, :] + r[:, None] * 0
        v = tl.load(q, mask=r[None, :] < M)
        tl.store(out_ptr + r[:, None] * N + r[None, :], v)
        p += N


@triton.jit
def int_iterarg_offset(x_ptr, n, B: tl.constexpr):
    # an integer offset carried by the loop: a loop-variant address
    offs = tl.arange(0, B)
    for k in range(0, n):
        tl.store(x_ptr + offs, 1.0)
        offs += B


@triton.jit
def iv_wrap(x_ptr, lo, n):
    # the induction variable's increment wraps in i32 when n is near INT32_MAX
    for k in range(lo, n, 1 << 20):
        tl.store(x_ptr + k, 1.0)


@triton.jit
def pure_asm_int_addr(x_ptr):
    # a "pure" asm handed the address as an integer stores through it
    a = x_ptr.to(tl.int64, bitcast=False)
    r = tl.inline_asm_elementwise(
        "st.global.b32 [$1], $2; mov.b32 $0, 0;",
        "=r,l,r",
        [a, tl.full([], 7, tl.int32)],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(x_ptr + 4096, r)


@triton.jit
def observed_lanes(cnt_ptr, x_ptr, N: tl.constexpr):
    # two lanes of one tensor atomic's old values in one address
    r = tl.arange(0, N)
    old = tl.atomic_add(cnt_ptr + r, 1)
    c = tl.minimum(tl.maximum(old, 0), 10)
    tl.store(x_ptr + c[:, None] - c[None, :], 1.0)


_F32 = "*fp32"
# corpus name (reader_ttir/<name>.ttir, the kernel of that name) ->
# (signature, constexprs), compiled for sm80 with the default options
SPECS: dict[str, tuple[dict[str, str], dict[str, Any]]] = {
    "p1_variant_delta": ({"x_ptr": _F32, "out_ptr": _F32, "n": "i32"}, {}),
    "p2_swap": ({"a_ptr": _F32, "b_ptr": _F32, "n": "i32"}, {}),
    "p3_call_guarded": ({"x_ptr": _F32, "n": "i32"}, {}),
    "p3_call_offset": ({"x_ptr": _F32, "n": "i32"}, {}),
    "p3_call_formals": ({"x_ptr": _F32, "n": "i32"}, {}),
    "p4_observed_direct": ({"cnt_ptr": "*i32", "x_ptr": _F32, "n": "i32"}, {}),
    "p4_observed_loop": ({"cnt_ptr": "*i32", "x_ptr": _F32, "n": "i32"}, {}),
    "p4_observed_delta": ({"cnt_ptr": "*i32", "x_ptr": _F32, "n": "i32"}, {}),
    "rv_trunci_alias": ({"x_ptr": "*i32"}, {}),
    "rv_i32_wrap": ({"x_ptr": "*i32", "S": "i32"}, {}),
    "unsigned_index": ({"x_ptr": _F32, "n": "i32"}, {}),
    "rv_inline_asm_store": ({"x_ptr": "*i32", "OFF": "constexpr"}, {"OFF": 4096}),
    "loop_two_step_advance": ({"x_ptr": _F32, "n": "i32", "s": "i32"}, {}),
    "loop_observed_advance": ({"cnt_ptr": "*i32", "x_ptr": _F32, "n": "i32"}, {}),
    "where_pointer": ({"x_ptr": _F32, "n": "i32"}, {}),
    "tile3d_shared_arange": ({"x_ptr": _F32, "N": "constexpr"}, {"N": 4}),
    "expand_iterarg_3d": (
        {"x_ptr": _F32, "out_ptr": _F32, "n": "i32", "N": "constexpr"},
        {"N": 4},
    ),
    "expand_iterarg_mask": (
        {"x_ptr": _F32, "out_ptr": _F32, "n": "i32", "M": "i32", "N": "constexpr"},
        {"N": 4},
    ),
    "int_iterarg_offset": ({"x_ptr": _F32, "n": "i32", "B": "constexpr"}, {"B": 8}),
    "iv_wrap": ({"x_ptr": _F32, "lo": "i32", "n": "i32"}, {}),
    "pure_asm_int_addr": ({"x_ptr": "*i32"}, {}),
    "observed_lanes": ({"cnt_ptr": "*i32", "x_ptr": _F32, "N": "constexpr"}, {"N": 4}),
}
