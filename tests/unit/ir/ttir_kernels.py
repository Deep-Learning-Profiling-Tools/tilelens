"""Kernels whose TTIR the walk and reader tests read as ``ttir/<name>.ttir``.

ttir_corpus.py compiles each spec below to TTIR at test time.

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl


@triton.jit
def dot_precisions(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    # fp32 inputs: `ieee` is the printer-elided default, tf32x3 prints
    offs = tl.arange(0, BLOCK)
    idx = offs[:, None] * BLOCK + offs[None, :]
    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)
    c = tl.dot(a, b, input_precision="ieee")
    d = tl.dot(a, b, input_precision="tf32x3")
    tl.store(c_ptr + idx, c + d)


@triton.jit
def eps_consts(x_ptr, s_ptr, out_ptr, BLOCK: tl.constexpr):
    # uppercase-E float literals: a scalar constant and a dense splat
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.load(s_ptr) + 1e-6
    tl.store(out_ptr + offs, x * s + 1e-12)


@triton.jit
def unicode_msgs(x_ptr, BLOCK: tl.constexpr):
    # a non-ASCII tt.assert message (debug=True); device_print prefixes must be ASCII
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.device_assert(x > 0, "错误: π must be > 0")
    tl.device_print("x=", x)
    tl.store(x_ptr + offs, x + 1)


@triton.jit
def deep_chain(out_ptr, s, N: tl.constexpr):
    # a def chain deeper than Python's default recursion limit
    pid = tl.program_id(0)
    off = pid
    for _ in tl.static_range(N):
        off = off * s + pid
    tl.store(out_ptr + off, 1.0)


@triton.jit
def dot_scaled_k(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):  # fmt: skip
    # tt.dot_scaled prints `%a scale %as, %b scale %bs, %c`: ODS (a, b, c, as, bs)
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    rs = tl.arange(0, K // 32)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    a_scale = tl.load(as_ptr + rm[:, None] * (K // 32) + rs[None, :])
    b_scale = tl.load(bs_ptr + rn[:, None] * (K // 32) + rs[None, :])
    c = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3")
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


@triton.jit
def descs(a_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # device-side tensor descriptors, descriptor_load / store / reduce / gather
    # / scatter
    d = tl.make_tensor_descriptor(a_ptr, [M, N], [N, 1], [BM, BN])
    x = d.load([0, BN])
    d.store([BM, 0], x)
    d.atomic_add([BM, BN], x)
    d1 = tl.make_tensor_descriptor(a_ptr, [M, N], [N, 1], [1, BN])
    rows = tl.arange(0, BM)
    g = d1.gather(rows, 0)
    d1.scatter(g, rows, BN)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    stride_cm,  # inner strides are 1 (row-major)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def tile2d_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D tile copy: independent row/col arange instances, per-axis masks ANDed
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ptrs = in_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    vals = tl.load(ptrs, mask=mask, other=0.0)
    optrs = out_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    tl.store(optrs, vals * 2.0, mask=mask)


@triton.jit
def atomic_fmax_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # float tl.atomic_max lowers to a sign-trick dance: the pointer is
    # tt.bitcast to i32 and the two RMWs' masks derive from the loaded value
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.atomic_max(out_ptr + offs, v, mask=mask)


@triton.jit
def nested_guard_merge_kernel(x_ptr, out_ptr, n, T, BLOCK: tl.constexpr):
    # early returns under nested guards: cf.cond_br blocks that merge
    pid = tl.program_id(0)
    base = pid * BLOCK
    if pid >= T:
        return
    if pid == 0:
        base = 0
        if n < 0:
            return
    else:
        base = base + n
    offs = base + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, v)


@triton.jit
def pid_branch_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # a pid-dependent branch without results: only program 0 stores (an
    # scf.if with a then-region only)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(x_ptr + offs, mask=mask)
    if pid == 0:
        tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def grid_stride_kernel(
    x_ptr, out_ptr, n_rows, stride, NUM_PRGMS: tl.constexpr, BLOCK: tl.constexpr
):
    # a grid-stride loop: its lower bound is the program id
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    for row in range(pid, n_rows, NUM_PRGMS):
        v = tl.load(x_ptr + row * stride + cols)
        tl.store(out_ptr + row * stride + cols, v)


@triton.jit
def cas_kernel(lock_ptr, out_ptr):
    # a scalar tt.atomic_cas (no mask operand) on a pointer argument
    old = tl.atomic_cas(lock_ptr, 0, 1)
    tl.store(out_ptr, old)


@triton.jit
def gather_kernel(idx_ptr, src_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # an indirect load: a loaded value feeds the second load's address
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    vals = tl.load(src_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def if_yield(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # scf.if with yields (the loaded value and the pointer both flow out)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if pid % 2 == 0:
        v = tl.load(x_ptr + offs, mask=m)
        dst = out_ptr + offs
    else:
        v = tl.load(y_ptr + offs, mask=m) * 2.0
        dst = out_ptr + n + offs
    tl.store(dst, v, mask=m)


@triton.jit
def spin_while(lock_ptr, flag_ptr, out_ptr):
    # scf.while spin loop (ticket lock style): atomic_cas poll + volatile flag poll
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="gpu") == 1:
        pass
    v = tl.load(flag_ptr, volatile=True)
    while v == 0:
        v = tl.load(flag_ptr, volatile=True)
    tl.store(out_ptr, v)
    tl.atomic_xchg(lock_ptr, 0, sem="release", scope="gpu")


@triton.jit
def atomics(p_ptr, q_ptr, n, BLOCK: tl.constexpr):
    # atomic_rmw / atomic_cas with every sem / scope spelling
    offs = tl.arange(0, BLOCK)
    m = offs < n
    tl.atomic_add(p_ptr + offs, 1.0, mask=m, sem="relaxed", scope="cta")
    tl.atomic_max(q_ptr + offs, 7, mask=m, sem="release", scope="sys")
    tl.atomic_min(q_ptr + offs, -3, mask=m, sem="acquire", scope="gpu")
    tl.atomic_and(q_ptr + offs, 0xF0, mask=m)
    tl.atomic_or(q_ptr + offs, 1, mask=m, sem="acq_rel")
    tl.atomic_xor(q_ptr + offs, 2, mask=m)
    old = tl.atomic_xchg(q_ptr, 5, sem="relaxed", scope="sys")
    tl.atomic_cas(q_ptr + 1, old, 9, sem="acq_rel", scope="cta")
    tl.atomic_add(q_ptr + 2, 3)  # scalar, default sem/scope


@triton.jit
def _combine_max_abs(a, b):
    return tl.maximum(tl.abs(a), tl.abs(b))


@triton.jit
def reduce_scan(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # tt.reduce with a combine region (builtin sum, custom combine, argmax = 2
    # results), tt.scan (cumsum)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr, tl.sum(x, axis=0))
    tl.store(out_ptr + 1, tl.reduce(x, 0, _combine_max_abs))
    tl.store(out_ptr + 2, tl.argmax(x, axis=0).to(tl.float32))
    tl.store(out_ptr + 3 + offs, tl.cumsum(x, axis=0))


@triton.jit
def inline_asm(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # tl.inline_asm_elementwise: pure, impure (a store), braces / % in the asm
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.inline_asm_elementwise(
        "{ .reg .u32 t; mov.u32 t, %tid.x; shl.b32 $0, $1, 3; }",
        "=r,r",
        [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.inline_asm_elementwise(
        "st.global.b32 [$1], $2; mov.u32 $0, 0;",
        "=r,l,r",
        [out_ptr + offs, y],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def casts(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # trunci / extsi / extui casts on the address path and on values
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    o16 = offs.to(tl.int16)
    o64 = o16.to(tl.int64)
    u8 = offs.to(tl.uint8)
    idx = o64 + u8.to(tl.int32)
    v = tl.load(x_ptr + idx, mask=idx < n)
    tl.store(out_ptr + offs.to(tl.int64), v, mask=offs < n)


@triton.jit
def consts(x_ptr, i8_ptr, i64_ptr, u32_ptr, BLOCK: tl.constexpr):
    # constants: negative / large / min-int / unsigned / narrow ints, special
    # floats, dense splats of negative ints and bools
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    neg = tl.full([BLOCK], -7, tl.int32)
    mn = tl.full([BLOCK], -2147483648, tl.int32)
    tl.store(x_ptr + offs, v + neg + mn.to(tl.float32))
    tl.store(x_ptr + offs + BLOCK, tl.where(v != v, float("nan"), float("-inf")))
    tl.store(x_ptr + offs + 2 * BLOCK, tl.full([BLOCK], -0.0, tl.float32) + 1e-30)
    tl.store(
        i8_ptr + offs, tl.full([BLOCK], -128, tl.int8) + tl.full([BLOCK], 127, tl.int8)
    )
    big = tl.full([BLOCK], 9223372036854775807, tl.int64)
    small = tl.full([BLOCK], -9223372036854775808, tl.int64)
    tl.store(i64_ptr + offs, big + small + 4294967296)
    tl.store(i64_ptr - 9223372036854775807 + offs, big)
    tl.store(u32_ptr + offs, tl.full([BLOCK], 4294967295, tl.uint32))
    b = tl.full([BLOCK], True, tl.int1)
    tl.store(u32_ptr + offs + BLOCK, tl.full([BLOCK], 1, tl.uint32), mask=b)
    tl.store(u32_ptr + offs + 2 * BLOCK, tl.full([BLOCK], 1, tl.uint32), mask=offs < -1)
    h = tl.full([BLOCK], 1.5, tl.float16) * tl.full([BLOCK], -2.0, tl.bfloat16).to(
        tl.float16
    )
    tl.store(x_ptr + offs + 3 * BLOCK, h.to(tl.float32))


@triton.jit
def views(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # reshape with can_reorder (tt.reshape allow_reorder), trans, broadcast, gather
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs
    p2 = tl.reshape(p, [BLOCK // 4, 4], can_reorder=True)
    m2 = tl.reshape(offs < 50, [BLOCK // 4, 4], can_reorder=True)
    v = tl.load(p2, mask=m2)
    t = tl.trans(v)
    q = (
        out_ptr
        + tl.arange(0, 4)[:, None] * (BLOCK // 4)
        + tl.arange(0, BLOCK // 4)[None, :]
    )
    tl.store(q, t)
    g = tl.gather(offs, (BLOCK - 1) - offs, 0)
    tl.store(out_ptr + BLOCK + g, tl.load(x_ptr + offs))


# (kernel, signature, constexprs, compute capability, compile options,
# ASTSource attrs)
_Spec = tuple[Any, dict[str, str], dict[str, int], int, dict[str, Any], dict]


def _divisible_by_16(count: int) -> dict:
    """ASTSource attrs: divisibility 16 on the first ``count`` arguments, as
    the JIT specializes well-aligned pointers and sizes."""
    return {(i,): [["tt.divisibility", 16]] for i in range(count)}


# corpus name (ttir/<name>.ttir) -> spec
SPECS: dict[str, _Spec] = {
    "kernel_dot_precisions": (
        dot_precisions,
        {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 16},
        80,
        {},
        {},
    ),
    "kernel_eps_consts": (
        eps_consts,
        {"x_ptr": "*fp32", "s_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {},
        {},
    ),
    "kernel_unicode_msgs": (
        unicode_msgs,
        {"x_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {"debug": True},
        {},
    ),
    "kernel_deep_chain": (
        deep_chain,
        {"out_ptr": "*fp32", "s": "i32", "N": "constexpr"},
        {"N": 600},
        80,
        {},
        {},
    ),
    "kernel_dot_scaled": (
        dot_scaled_k,
        {
            "a_ptr": "*fp8e4nv",
            "as_ptr": "*u8",
            "b_ptr": "*fp8e4nv",
            "bs_ptr": "*u8",
            "c_ptr": "*fp32",
            "M": "constexpr",
            "N": "constexpr",
            "K": "constexpr",
        },
        {"M": 128, "N": 128, "K": 64},
        100,
        {},
        {},
    ),
    "adv_descs": (
        descs,
        {
            "a_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "BM": "constexpr",
            "BN": "constexpr",
        },
        {"BM": 32, "BN": 32},
        100,
        {},
        {},
    ),
    "golden_add_sm80": (
        add_kernel,
        {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        {"BLOCK_SIZE": 1024},
        80,
        {"num_stages": 3},
        _divisible_by_16(4),
    ),
    "golden_matmul_s3_sm80": (
        matmul_kernel,
        {
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*fp16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i32",
            "stride_bk": "i32",
            "stride_cm": "i32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
        },
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        80,
        {"num_stages": 3},
        _divisible_by_16(9),
    ),
    "golden_tile2d_sm80": (
        tile2d_kernel,
        {
            "in_ptr": "*fp32",
            "out_ptr": "*fp32",
            "M": "i32",
            "N": "i32",
            "stride_m": "i32",
            "stride_n": "i32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
        },
        {"BLOCK_M": 32, "BLOCK_N": 32},
        80,
        {"num_stages": 1},
        _divisible_by_16(6),
    ),
    "golden_atomic_fmax_sm80": (
        atomic_fmax_kernel,
        {
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": 256},
        80,
        {"num_stages": 1},
        _divisible_by_16(3),
    ),
    "golden_nested_guard_merge_sm80": (
        nested_guard_merge_kernel,
        {
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n": "i32",
            "T": "i32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": 64},
        80,
        {"num_stages": 1},
        {},
    ),
    "golden_pid_branch_sm80": (
        pid_branch_kernel,
        {
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": 256},
        80,
        {"num_stages": 1},
        _divisible_by_16(3),
    ),
    "golden_grid_stride_sm80": (
        grid_stride_kernel,
        {
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_rows": "i32",
            "stride": "i32",
            "NUM_PRGMS": "constexpr",
            "BLOCK": "constexpr",
        },
        {"NUM_PRGMS": 4, "BLOCK": 64},
        80,
        {"num_stages": 1},
        {},
    ),
    "golden_cas_sm80": (
        cas_kernel,
        {"lock_ptr": "*i32", "out_ptr": "*i32"},
        {},
        80,
        {"num_stages": 1},
        _divisible_by_16(2),
    ),
    "golden_gather_sm80": (
        gather_kernel,
        {
            "idx_ptr": "*i32",
            "src_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": 256},
        80,
        {"num_stages": 1},
        _divisible_by_16(4),
    ),
    "spike_if_yield": (
        if_yield,
        {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n": "i32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": 64},
        80,
        {"num_stages": 1},
        {},
    ),
    "spike_spin_while": (
        spin_while,
        {"lock_ptr": "*i32", "flag_ptr": "*i32", "out_ptr": "*i32"},
        {},
        80,
        {"num_stages": 1},
        {},
    ),
    "spike_atomics": (
        atomics,
        {"p_ptr": "*fp32", "q_ptr": "*i32", "n": "i32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {"num_stages": 1},
        {},
    ),
    "spike_reduce_scan": (
        reduce_scan,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {"num_stages": 1},
        {},
    ),
    "spike_inline_asm": (
        inline_asm,
        {"x_ptr": "*i32", "out_ptr": "*i32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {"num_stages": 1},
        {},
    ),
    "spike_casts": (
        casts,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {"num_stages": 1},
        {},
    ),
    "adv_consts": (
        consts,
        {
            "x_ptr": "*fp32",
            "i8_ptr": "*i8",
            "i64_ptr": "*i64",
            "u32_ptr": "*u32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": 64},
        80,
        {},
        {},
    ),
    "adv_views": (
        views,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
        80,
        {},
        {},
    ),
}
