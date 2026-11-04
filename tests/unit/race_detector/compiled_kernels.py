"""Kernels whose TTIR the compiled race detector's unit tests analyze.

``ttir(name)`` host-compiles kernel ``name`` (with its ``SPECS`` entry: the
signature and constexprs) up to the TTIR stage at test time, through
``tests/unit/ir/ttir_corpus.py``.

Not a test module: pytest imports it (python_files = *.py) and finds nothing.
"""

from __future__ import annotations

import functools

import triton
import triton.language as tl
from triton import knobs

from tilelens.ir.ttir_reader import AccessGraph, parse_ttir

B = 16


def real_jit(fn):
    """``triton.jit``, building a JITFunction whatever TRITON_INTERPRET says
    (tests/unit/test_multithreading.py sets it when imported, and a module
    collected after it would get InterpretedFunctions); the knob is put back
    as it was."""
    missing = object()
    previous = knobs.runtime.__dict__.get("interpret", missing)
    knobs.runtime.__dict__["interpret"] = False
    try:
        return triton.jit(fn)
    finally:
        if previous is missing:
            knobs.runtime.__dict__.pop("interpret", None)
        else:
            knobs.runtime.__dict__["interpret"] = previous


# ── one kernel per race class ──


@real_jit
def add(x, y, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=m) + tl.load(y + offs, mask=m), mask=m)


@real_jit
def waw(out, BLOCK: tl.constexpr):
    tl.store(out + tl.arange(0, BLOCK), tl.program_id(0))


@real_jit
def raw(x, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    tl.store(x + offs, 1, mask=pid == 0)
    tl.store(out + pid * BLOCK + offs, tl.load(x + offs, mask=pid == 1), mask=pid == 1)


@real_jit
def war(x, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    tl.store(out + pid * BLOCK + offs, tl.load(x + offs, mask=pid == 0), mask=pid == 0)
    tl.store(x + offs, 2, mask=pid == 1)


@real_jit
def atomic_vs_load(x, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    tl.atomic_add(x + offs, 1, mask=pid == 0)
    tl.store(out + pid * BLOCK + offs, tl.load(x + offs), mask=pid == 1)


@real_jit
def atomic_only(x, BLOCK: tl.constexpr):
    tl.atomic_add(x + tl.arange(0, BLOCK), 1)


@real_jit
def tile2d(out, BLOCK: tl.constexpr):
    # one make_range for both dimensions of the tile
    r = tl.arange(0, BLOCK)
    tl.store(out + r[:, None] * BLOCK + r[None, :], 1)


# ── rungs ──


@real_jit
def strided(x, out, stride, BLOCK: tl.constexpr):
    # pid * stride is nonlinear with the params symbolic: no T0
    offs = tl.program_id(0) * stride + tl.arange(0, BLOCK)
    tl.store(out + offs, tl.load(x + offs))


@real_jit
def two_slots(out, BLOCK: tl.constexpr):
    # race-free for at most two programs only
    tl.store(out + (tl.program_id(0) % 2) * BLOCK + tl.arange(0, BLOCK), 1)


@real_jit
def scatter(idx, out, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + tl.load(idx + offs), 1)


@real_jit
def launch_wrap(x, out, big, BLOCK: tl.constexpr):
    # race-free for two programs; with big = 2**30, pid * big * 2 wraps
    # at program 1
    pid = tl.program_id(0)
    v = tl.load(x + (pid * big * 2) // big // 2)
    tl.store(out + (pid % 2) * BLOCK + tl.arange(0, BLOCK), v)


@real_jit
def wrapped_slots(out):
    # (pid * 2**30) // 2**30 is pid until the product wraps, at pid 2
    pid = tl.program_id(0)
    tl.store(out + ((pid * 1073741824) // 1073741824) % 3, 1)


@real_jit
def two_widths(out):
    # an i32 product that wraps at pids 2 and 3, an i64 one from pid 4 on,
    # on disjoint paths: no one program shows both widths wrapping
    pid = tl.program_id(0)
    if pid < 4:
        tl.store(out + (pid * 1073741824) // 1073741824, 1)
    else:
        tl.store(out + (pid.to(tl.int64) * 2**61) // 2**61, 1)


@real_jit
def where_wrap(out):
    # pid * 2**30 wraps from pid 2 on, where the other arm is taken: pids
    # >= 2 store out[0], as pid 0 does
    pid = tl.program_id(0)
    v = pid * 1073741824
    tl.store(out + tl.where(pid >= 2, 0, v // 1073741824), 1)


@real_jit
def masked_table(flags, out, stride, BLOCK: tl.constexpr):
    # pid * stride: no T0; race-free whatever the table holds
    offs = tl.program_id(0) * stride + tl.arange(0, BLOCK)
    tl.store(out + offs, 1, mask=tl.load(flags + offs) != 0)


@real_jit
def table_mask(flags, out, BLOCK: tl.constexpr):
    # every program stores out[0:BLOCK] where its flags are set
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + tl.arange(0, BLOCK), 1, mask=tl.load(flags + offs) != 0)


@real_jit
def table_mask_and_waw(flags, out, both, BLOCK: tl.constexpr):
    d = tl.arange(0, BLOCK)
    tl.store(out + d, 1, mask=tl.load(flags + tl.program_id(0) * BLOCK + d) != 0)
    tl.store(both + d, tl.program_id(0))


@real_jit
def masked_gather_other(idx, out, BLOCK: tl.constexpr):
    # lanes >= 8 load nothing and store out[0], in every program
    d = tl.arange(0, BLOCK)
    i = tl.load(idx + tl.program_id(0) * BLOCK + d, mask=d < 8, other=0)
    tl.store(out + i, 1)


@real_jit
def handoff_index(flag, idx, out):
    # program 0 publishes idx[0] = 5 and stores out[5] after its release;
    # program 1 acquires, reads idx[0] and stores out[idx[0]]: a WAW
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(idx, 5)
        tl.debug_barrier()
        tl.atomic_xchg(flag, 1, sem="release")
        tl.store(out + 5, 1)
    else:
        while tl.atomic_add(flag, 0, sem="acquire") != 1:
            pass
        tl.debug_barrier()
        tl.store(out + tl.load(idx), 2)


# ── fences ──


@real_jit
def store_then_load(hist, out, N: tl.constexpr):
    p = tl.program_id(0)
    tl.store(hist + tl.arange(0, N), tl.arange(0, N))
    tl.store(out + p, tl.load(hist + p))


@real_jit
def store_fence_load(hist, out, N: tl.constexpr):
    p = tl.program_id(0)
    tl.store(hist + tl.arange(0, N), tl.arange(0, N))
    tl.debug_barrier()
    tl.store(out + p, tl.load(hist + p))


@real_jit
def store_flag_fence_load(hist, out, flag, N: tl.constexpr):
    p = tl.program_id(0)
    tl.store(hist + tl.arange(0, N), tl.arange(0, N))
    if flag:
        tl.debug_barrier()
    tl.store(out + p, tl.load(hist + p))


@real_jit
def loop_rows_fenced(t, u, n, BLOCK: tl.constexpr):
    # each iteration writes its own row, then reads it back reversed
    d = tl.arange(0, BLOCK)
    for i in range(n):
        tl.store(t + i * BLOCK + d, d)
        tl.debug_barrier()
        tl.store(u + i * BLOCK + d, tl.load(t + i * BLOCK + (BLOCK - 1 - d)))


@real_jit
def loop_buffer_fenced(t, u, n, BLOCK: tl.constexpr):
    # one buffer for every iteration: nothing orders iteration i's reversed
    # read before iteration i + 1's write
    d = tl.arange(0, BLOCK)
    for i in range(n):
        tl.store(t + d, d + i)
        tl.debug_barrier()
        tl.store(u + d, tl.load(t + (BLOCK - 1 - d)))


@real_jit
def loop_buffer_two_fences(t, u, n, BLOCK: tl.constexpr):
    d = tl.arange(0, BLOCK)
    for i in range(n):
        tl.store(t + d, d + i)
        tl.debug_barrier()
        tl.store(u + d, tl.load(t + (BLOCK - 1 - d)))
        tl.debug_barrier()


# ── dependencies ──


@real_jit
def inplace(x, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x + offs, tl.load(x + offs) + 1)


@real_jit
def inplace_shift(x, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x + offs, tl.load(x + offs + 1))


@real_jit
def atomic_chain(x, out, n, BLOCK: tl.constexpr):
    # the store's value comes from the load through an unmasked exchange
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x + offs, mask=m)
    tl.atomic_add(out + offs, v, mask=m)
    tl.store(x + offs, tl.atomic_xchg(out + offs, v), mask=m)


@real_jit
def masked_chain(x, y, flag, BLOCK: tl.constexpr):
    # where the atomic is masked off, the store's value is not the load's
    offs = tl.arange(0, BLOCK)
    old = tl.atomic_add(y + offs, tl.load(x + offs), mask=offs < flag)
    tl.store(x + offs, old)


# ── tile shapes ──


@real_jit
def one_address(out, BLOCK: tl.constexpr):
    # every lane stores its own value to out[pid]
    tl.store(
        out + tl.program_id(0) + tl.zeros([BLOCK], dtype=tl.int32), tl.arange(0, BLOCK)
    )


@real_jit
def one_address_masked(out, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    tl.store(out + tl.zeros([BLOCK], dtype=tl.int32), r, mask=r == 0)


@real_jit
def broadcast_rows(out, BLOCK: tl.constexpr):
    # the columns of a row store different values to out[row]
    r = tl.arange(0, BLOCK)
    p = tl.broadcast_to(out + r[:, None], (BLOCK, BLOCK))
    tl.store(p, r[:, None] * BLOCK + r[None, :])


# ── awaits ──


@real_jit
def comm_comp(
    sem,
    payload,
    out,
    BLOCK: tl.constexpr,
    SEM: tl.constexpr,
    EXPECT: tl.constexpr,
    ALL_POLL: tl.constexpr,
):
    # program 0 publishes a payload and arrives on a semaphore; the others
    # await it, then read the payload
    pid = tl.program_id(0)
    if pid < 1:
        offs = tl.arange(0, BLOCK)
        tl.store(payload + offs, (offs + 1).to(tl.float32))
        tl.debug_barrier()
        tl.atomic_xchg(sem, 1, sem="release")
    else:
        if ALL_POLL or pid == 1:
            while tl.atomic_add(sem, 0, sem=SEM) != EXPECT:
                pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        tl.store(out + (pid - 1) * BLOCK + offs, tl.load(payload + offs))


@real_jit
def exit_wrap(flag, data, out, big, BLOCK: tl.constexpr):
    # program 4's expected flag value wraps to 0 in i32, the initial value:
    # it reads data while program 0 writes it
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        tl.store(data + offs, offs + 1)
        tl.debug_barrier()
        tl.atomic_xchg(flag, 1, sem="release")
    elif (pid == 1) | (pid == 4):
        while tl.atomic_add(flag, 0, sem="acquire") != (pid * big) // big:
            pass
        tl.debug_barrier()
        tl.store(out + pid * BLOCK + offs, tl.load(data + offs))


@real_jit
def counted_while(x, n):
    i = 0
    while i < n:
        i += 1
    tl.store(x, i)


@real_jit
def cas_lock(lock, data, BLOCK: tl.constexpr, SEM: tl.constexpr):
    while tl.atomic_cas(lock, 0, 1, sem="acquire") != 0:
        pass
    tl.debug_barrier()
    offs = tl.arange(0, BLOCK)
    tl.store(data + offs, tl.load(data + offs) + 1)
    tl.debug_barrier()
    tl.atomic_xchg(lock, 0, sem=SEM)


# ── CAS operands ──


@real_jit
def cas_handoff(
    flag, data, out, PSEM: tl.constexpr, CSEM: tl.constexpr, SCOPE: tl.constexpr
):
    # program 0 publishes data with a CAS; the others read it once their
    # (failing) CAS observes the publication
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(data, 1)
        tl.debug_barrier()
        tl.atomic_cas(flag, 0, 1, sem=PSEM, scope=SCOPE)
    else:
        old = tl.atomic_cas(flag, 0, 0, sem=CSEM, scope=SCOPE)
        tl.debug_barrier()
        tl.store(out + pid, tl.load(data, mask=old == 1), mask=old == 1)


@real_jit
def cas_bool_operand(flag, out):
    pid = tl.program_id(0)
    old = tl.atomic_cas(flag, (pid == 0).to(tl.int32), 1)
    tl.store(out + pid, old)


@real_jit
def cas_cast_operand(flag, data):
    prior = tl.atomic_cas(data, 0, 1)
    tl.atomic_cas(flag, prior.to(tl.int8).to(tl.int32), 1)


@real_jit
def cas_arith_operand(flag, data):
    prior = tl.atomic_cas(data, 0, 1)
    tl.atomic_cas(flag, prior + 1, 1)


@real_jit
def cas_in_loop(flag, n):
    for _ in range(n):
        tl.atomic_cas(flag, 0, 1)


# ── multipath ──


@real_jit
def early_return(x, out, T, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid * BLOCK >= T:
        return
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + offs, tl.load(x + offs))


@real_jit
def sequential_loops(x, out, n, BLOCK: tl.constexpr, SHIFT: tl.constexpr):
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    for i in range(n):
        tl.store(out + (pid * 2 * n + i) * BLOCK + d, tl.load(x + d))
    for j in range(n):
        tl.store(out + ((pid * 2 + SHIFT) * n + j) * BLOCK + d, j)


@real_jit
def read_loop_then_inplace(x, y, n, BLOCK: tl.constexpr, IN_LOOP: tl.constexpr):
    # a loop that only reads; then x updated in place, in a second loop or
    # in no loop
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(n):
        acc += tl.load(y + pid * BLOCK + d)
    if IN_LOOP:
        for j in range(2):
            p = x + (pid * 2 + j) * BLOCK + d
            tl.store(p, tl.load(p) + acc)
    else:
        p = x + pid * BLOCK + d
        tl.store(p, tl.load(p) + acc)


@real_jit
def grouped_swizzle(out, n_m, n_n, GROUP: tl.constexpr):
    # the grouped program-id order of a tiled matmul: one store per tile
    pid = tl.program_id(0)
    width = GROUP * n_n
    first_m = (pid // width) * GROUP
    group = tl.minimum(n_m - first_m, GROUP)
    pid_m = first_m + (pid % width) % group
    pid_n = (pid % width) // group
    tl.store(out + pid_m * n_n + pid_n, pid)


@real_jit
def nested_loops(x, out, n, m, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    for i in range(n):
        for j in range(m):
            row = (pid * n + i) * m + j
            tl.store(out + row * BLOCK + d, tl.load(x + row * BLOCK + d))


@real_jit
def loop_under_guard(out, BLOCK: tl.constexpr):
    # the loop's bounds wrap for pid >= 8 only, where it is not reached
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    if pid < 4:
        for i in range(pid * 268435456, pid * 268435456 + 2):
            tl.store(out + (pid * 2 + i - pid * 268435456) * BLOCK + d, 1)


SPECS: dict[str, tuple[dict, dict]] = {
    "add": ({"x": "*fp32", "y": "*fp32", "out": "*fp32", "n": "i32"}, {"BLOCK": B}),
    "waw": ({"out": "*i32"}, {"BLOCK": B}),
    "raw": ({"x": "*i32", "out": "*i32"}, {"BLOCK": B}),
    "war": ({"x": "*i32", "out": "*i32"}, {"BLOCK": B}),
    "atomic_vs_load": ({"x": "*i32", "out": "*i32"}, {"BLOCK": B}),
    "atomic_only": ({"x": "*i32"}, {"BLOCK": B}),
    "tile2d": ({"out": "*i32"}, {"BLOCK": B}),
    "strided": ({"x": "*fp32", "out": "*fp32", "stride": "i32"}, {"BLOCK": B}),
    "two_slots": ({"out": "*i32"}, {"BLOCK": B}),
    "wrapped_slots": ({"out": "*i32"}, {}),
    "where_wrap": ({"out": "*i32"}, {}),
    "two_widths": ({"out": "*i32"}, {}),
    "masked_table": ({"flags": "*i32", "out": "*i32", "stride": "i32"}, {"BLOCK": B}),
    "table_mask": ({"flags": "*fp32", "out": "*i32"}, {"BLOCK": B}),
    "table_mask_and_waw": (
        {"flags": "*fp32", "out": "*i32", "both": "*i32"},
        {"BLOCK": B},
    ),
    "masked_gather_other": ({"idx": "*i32", "out": "*i32"}, {"BLOCK": B}),
    "handoff_index": ({"flag": "*i32", "idx": "*i32", "out": "*i32"}, {}),
    "atomic_chain": ({"x": "*fp32", "out": "*fp32", "n": "i32"}, {"BLOCK": B}),
    "masked_chain": ({"x": "*i32", "y": "*i32", "flag": "i32"}, {"BLOCK": B}),
    "one_address": ({"out": "*i32"}, {"BLOCK": B}),
    "one_address_masked": ({"out": "*i32"}, {"BLOCK": B}),
    "broadcast_rows": ({"out": "*i32"}, {"BLOCK": B}),
    "exit_wrap": (
        {"flag": "*i32", "data": "*i32", "out": "*i32", "big": "i32"},
        {"BLOCK": B},
    ),
    "cas_in_loop": ({"flag": "*i32", "n": "i32"}, {}),
    "launch_wrap": ({"x": "*i32", "out": "*i32", "big": "i32"}, {"BLOCK": B}),
    "cas_cast_operand": ({"flag": "*i32", "data": "*i32"}, {}),
    "cas_arith_operand": ({"flag": "*i32", "data": "*i32"}, {}),
    "loop_under_guard": ({"out": "*i32"}, {"BLOCK": B}),
    "scatter": ({"idx": "*i32", "out": "*i32"}, {"BLOCK": B}),
    "store_then_load": ({"hist": "*i32", "out": "*i32"}, {"N": B}),
    "store_fence_load": ({"hist": "*i32", "out": "*i32"}, {"N": B}),
    "store_flag_fence_load": (
        {"hist": "*i32", "out": "*i32", "flag": "i32"},
        {"N": B},
    ),
    "loop_rows_fenced": ({"t": "*i32", "u": "*i32", "n": "i32"}, {"BLOCK": B}),
    "loop_buffer_fenced": ({"t": "*i32", "u": "*i32", "n": "i32"}, {"BLOCK": B}),
    "loop_buffer_two_fences": ({"t": "*i32", "u": "*i32", "n": "i32"}, {"BLOCK": B}),
    "inplace": ({"x": "*i32"}, {"BLOCK": B}),
    "inplace_shift": ({"x": "*i32"}, {"BLOCK": B}),
    "comm_comp": (
        {"sem": "*i32", "payload": "*fp32", "out": "*fp32"},
        {"BLOCK": B, "SEM": "acquire", "EXPECT": 1, "ALL_POLL": True},
    ),
    "counted_while": ({"x": "*i32", "n": "i32"}, {}),
    "cas_lock": ({"lock": "*i32", "data": "*i32"}, {"BLOCK": B, "SEM": "release"}),
    "cas_handoff": (
        {"flag": "*i32", "data": "*i32", "out": "*i32"},
        {"PSEM": "release", "CSEM": "acquire", "SCOPE": "gpu"},
    ),
    "cas_bool_operand": ({"flag": "*i32", "out": "*i32"}, {}),
    "early_return": ({"x": "*fp32", "out": "*fp32", "T": "i32"}, {"BLOCK": B}),
    "sequential_loops": (
        {"x": "*fp32", "out": "*fp32", "n": "i32"},
        {"BLOCK": B, "SHIFT": 1},
    ),
    "read_loop_then_inplace": (
        {"x": "*fp32", "y": "*fp32", "n": "i32"},
        {"BLOCK": B, "IN_LOOP": True},
    ),
    "grouped_swizzle": ({"out": "*i32", "n_m": "i32", "n_n": "i32"}, {"GROUP": 4}),
    "nested_loops": (
        {"x": "*fp32", "out": "*fp32", "n": "i32", "m": "i32"},
        {"BLOCK": B},
    ),
}


@functools.cache
def ttir(name: str, **constexprs: object) -> str:
    """The TTIR of kernel ``name``, its SPECS constexprs updated by
    ``constexprs``."""
    from ..ir import ttir_corpus

    sig, consts = SPECS[name]
    consts = {**consts, **constexprs}
    full = {**sig, **{k: "constexpr" for k in consts}}
    return ttir_corpus._ttir(globals()[name], full, consts, cc=89)


def graph(name: str, **constexprs: object) -> AccessGraph:
    """Kernel ``name`` read as the compiled race detector reads it."""
    return parse_ttir(ttir(name, **constexprs), multipath=True, while_regions=True)
