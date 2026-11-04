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


# ── facts for the race detector ──
@triton.jit
def atomic_attrs(p_ptr, u_ptr, f_ptr, lock_ptr):
    # every tt.atomic_rmw op, sem and scope, and a tt.atomic_cas
    pid = tl.program_id(0)
    tl.atomic_and(p_ptr, 1, sem="relaxed", scope="gpu")
    tl.atomic_or(p_ptr, 1, sem="acquire", scope="cta")
    tl.atomic_xor(p_ptr, 1, sem="release", scope="sys")
    tl.atomic_add(p_ptr, pid, sem="acq_rel")
    tl.atomic_add(f_ptr, 1.0)
    tl.atomic_max(p_ptr, 1)
    tl.atomic_min(p_ptr, 1)
    tl.atomic_max(u_ptr, 1)
    tl.atomic_min(u_ptr, 1)
    tl.atomic_xchg(p_ptr, tl.load(p_ptr + 1))
    tl.atomic_cas(lock_ptr, pid, 1, sem="release", scope="sys")


@triton.jit
def deps_rules(x_ptr, y_ptr, m_ptr, out_ptr, n, B: tl.constexpr):
    # which loads reach each store lane for lane
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    m = tl.load(m_ptr + offs)
    tl.store(out_ptr + offs, x * 2 + 1, mask=m > 0)  # element-wise: x, m
    tl.store(out_ptr + offs, tl.where(m > 0, x, y))  # the condition's: m
    tl.store(out_ptr + offs, tl.where(offs < n, x, x + 1))  # both arms': x
    tl.store(out_ptr + offs, tl.where(offs < n, x, y))  # one arm each: none
    tl.store(out_ptr + offs, tl.sum(x, 0) + y)  # a splat of a scalar sum drops x
    v = x
    if n > 0:
        v = x + 1
    tl.store(out_ptr + offs, v)  # an scf.if result: none
    e = tl.exp(x).to(tl.int32, bitcast=True).to(tl.float32, bitcast=True)
    tl.store(out_ptr + offs, e)  # math.exp, tt.bitcast: x
    x2 = tl.load(x_ptr + offs[:, None] * B + offs[None, :])
    tl.store(out_ptr + offs, tl.sum(x2, 1))  # a reduction: none
    z = tl.load(y_ptr + offs, mask=m > 0)  # a load: none, though masked by m
    tl.store(out_ptr + offs, z)
    tl.store(out_ptr + offs, x, mask=tl.sum(x, 0) > 0)  # operands unite: x
    acc = x
    for i in range(0, n):
        acc = acc + 1
        tl.store(out_ptr + offs, acc)  # a loop-carried value: none


@triton.jit
def deps_dot(a_ptr, b_ptr, c_ptr, out_ptr):
    # tt.dot keeps only its accumulator's lanes
    r = tl.arange(0, 16)
    t = r[:, None] * 16 + r[None, :]
    a = tl.load(a_ptr + t)
    b = tl.load(b_ptr + t)
    c = tl.load(c_ptr + t)
    tl.store(out_ptr + t, tl.dot(a, b, c))


@triton.jit
def deps_atomics(x_ptr, m_ptr, cnt_ptr, lock_ptr, B: tl.constexpr):
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + offs)
    m = tl.load(m_ptr + offs)
    old = tl.atomic_add(cnt_ptr + offs, x, mask=m > 0)  # val and mask
    c = tl.load(x_ptr)
    v = tl.load(m_ptr)
    tl.atomic_cas(lock_ptr, c, v)  # cmp and val
    tl.store(x_ptr + offs, old)  # an atomic's result


@triton.jit
def fences(x_ptr, n):
    pid = tl.program_id(0)
    tl.store(x_ptr + pid, 1)
    tl.debug_barrier()
    v = tl.load(x_ptr)
    if pid < n:
        tl.debug_barrier()
        tl.store(x_ptr + 1, v)
    if v > 0:
        tl.debug_barrier()
    for i in range(0, n):
        tl.debug_barrier()
        tl.store(x_ptr + i, 2)
    tl.store(x_ptr + 2, tl.program_id(1) + tl.num_programs(2))


@triton.jit
def _pid_combine(a, b):
    tl.debug_barrier()
    return a + b * tl.program_id(1)


@triton.jit
def pid_in_combine(x_ptr, o_ptr, B: tl.constexpr):
    # a program id and a barrier only inside a combine region
    s = tl.reduce(tl.load(x_ptr + tl.arange(0, B)), 0, _pid_combine)
    tl.store(o_ptr + tl.program_id(0), s)


@triton.jit
def poll_while(flag_ptr, out_ptr, n):
    # a volatile poll, a poll under a modeled branch, and one nested in another
    while tl.load(flag_ptr, volatile=True) == 0:
        pass
    pid = tl.program_id(0)
    if pid < n:
        while tl.load(flag_ptr + pid, volatile=True) != n:
            pass
    while tl.atomic_add(flag_ptr + 1, 0) == 0:
        while tl.load(flag_ptr + 2, volatile=True) == 0:
            pass
    tl.store(out_ptr + pid, 1)


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
    "atomic_attrs": (
        {"p_ptr": "*i32", "u_ptr": "*u32", "f_ptr": _F32, "lock_ptr": "*i32"},
        {},
    ),
    "deps_rules": (
        {
            "x_ptr": _F32,
            "y_ptr": _F32,
            "m_ptr": "*i32",
            "out_ptr": _F32,
            "n": "i32",
            "B": "constexpr",
        },
        {"B": 8},
    ),
    "deps_dot": (
        {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": _F32, "out_ptr": _F32},
        {},
    ),
    "deps_atomics": (
        {
            "x_ptr": "*i32",
            "m_ptr": "*i32",
            "cnt_ptr": "*i32",
            "lock_ptr": "*i32",
            "B": "constexpr",
        },
        {"B": 8},
    ),
    "fences": ({"x_ptr": "*i32", "n": "i32"}, {}),
    "pid_in_combine": ({"x_ptr": "*i32", "o_ptr": "*i32", "B": "constexpr"}, {"B": 8}),
    "poll_while": ({"flag_ptr": "*i32", "out_ptr": "*i32", "n": "i32"}, {}),
}
