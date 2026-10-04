"""Sanitizer(compile=True) on real kernels: each launch is compiled on the host
for the client's target, checked against its TTIR and not run, on
CPU tensors and with Triton's driver unreachable: no GPU is needed.
Counterparts on fake launches live in
tests/unit/sanitizer_compiled/test_client.py.
"""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

import tilelens
from tilelens.clients import Sanitizer, Tracer
from tilelens.clients.sanitizer.compiled import CompiledSanitizer
from tilelens.clients.sanitizer.data import CompiledSanitizerRecord
from tilelens.core.config import DEFAULT_IR_TARGET, Config
from tilelens.core.data import Load, Store
from tilelens.ir import IRVerdict
from tilelens.ir.verdict import SourceLocation

trace_module = importlib.import_module("tilelens.core.trace")
config_module = importlib.import_module("tilelens.core.config")
REPO = Path(__file__).resolve().parents[2]


def _real_compiles_available() -> bool:
    # Triton imported under TRITON_INTERPRET=1 builds its own standard library
    # as InterpretedFunctions, so nothing can compile for real in-process. No
    # GPU is needed: IR mode compiles on the host.
    import triton.language.standard as tl_standard
    from triton.runtime.jit import JITFunction

    return isinstance(tl_standard.cdiv, JITFunction)


pytestmark = pytest.mark.skipif(
    not _real_compiles_available(),
    reason="Triton was imported under TRITON_INTERPRET=1: nothing compiles in-process",
)


@pytest.fixture(autouse=True)
def _no_driver(unreachable_driver):
    """IR mode needs no GPU: Triton's driver is unreachable here, as on
    a machine without one (where it raises "0 active drivers")."""
    unreachable_driver("IR mode queried Triton's driver")


@pytest.fixture(autouse=True)
def _default_ir_target(monkeypatch):
    """The default IR target, whatever TILELENS_IR_TARGET the caller
    set: in the process config, and in any Config read from the environment
    (configured_target below sets its own)."""
    for name in ("TILELENS_IR_TARGET", "TRITON_VIZ_IR_TARGET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module.config, "ir_target", DEFAULT_IR_TARGET)


@pytest.fixture(autouse=True)
def _real_jit(monkeypatch):
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import time,
    # and a traced launch's patch scope restores knobs.runtime.interpret as an
    # explicit override. These tests need @triton.jit to build real
    # JITFunctions, so pin the knob off and put back exactly what was there.
    from triton import knobs

    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    missing = object()
    previous = knobs.runtime.__dict__.get("interpret", missing)
    knobs.runtime.__dict__["interpret"] = False
    yield
    if previous is missing:
        knobs.runtime.__dict__.pop("interpret", None)
    else:
        knobs.runtime.__dict__["interpret"] = previous


def _sanitizer() -> CompiledSanitizer:
    det = Sanitizer(compile=True, abort_on_error=False)
    assert isinstance(det, CompiledSanitizer)
    return det


def _line(kernel, needle: str) -> int:
    """The source line of ``kernel`` (a JITFunction) holding ``needle``."""
    lines, start = inspect.getsourcelines(kernel.fn)
    (index,) = [i for i, line in enumerate(lines) if needle in line]
    return start + index


def _lane(record: CompiledSanitizerRecord) -> int:
    (lane,) = [v for k, v in record.witness.items() if k.startswith("arange_")]
    return lane


def _make_add():
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    return add_kernel


def _make_add_nomask():
    @triton.jit
    def add_nomask(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)  # no mask: OOB on a ragged tail
        tl.store(out_ptr + offs, x)

    return add_nomask


# ======== proofs and findings =========


def test_an_in_bounds_launch_is_proved_and_nothing_runs():
    det = _sanitizer()
    traced = tilelens.trace(det)(_make_add())
    n = 3000  # a ragged tail, masked
    x, y = torch.randn(n), torch.randn(n)
    out = torch.zeros(n)

    kernel = traced[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)

    assert det.last_status == "ok" and det.records == []
    verdict = det.last_verdict
    assert trace_module.launches[-1].records == [verdict]
    (config,) = verdict.per_config
    assert (config.specialization, config.status) == (kernel.hash, "ok")
    assert verdict.refusal is None and verdict.notes == ()
    # LAUNCH="skip": compiled and checked, never run.
    assert torch.count_nonzero(out) == 0


def test_an_unmasked_tail_is_reported_at_its_line_and_device_address():
    det = _sanitizer()
    add_nomask = _make_add_nomask()
    traced = tilelens.trace(det)(add_nomask)
    n = 3000
    x, out = torch.randn(n), torch.zeros(n)

    traced[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)

    assert det.last_status == "violations"
    load, store = det.records
    assert trace_module.launches[-1].records == [load, store, det.last_verdict]
    assert [(r.kind, r.op_type, r.tensor_name) for r in (load, store)] == [
        ("out-of-bounds", Load, "x_ptr"),
        ("out-of-bounds", Store, "out_ptr"),
    ]
    for record, tensor, needle in ((load, x, "tl.load"), (store, out, "tl.store")):
        offset = record.violation_offset
        assert n <= offset < 3 * 1024
        assert record.witness["pid_0"] == 2
        assert 2 * 1024 + _lane(record) == offset
        assert record.violation_address == tensor.data_ptr() + offset * 4
        assert record.tensor_facts.data_ptr == tensor.data_ptr()
        (tb,) = record.user_code_tracebacks
        assert (tb.filename, tb.lineno, tb.func_name) == (
            __file__,
            _line(add_nomask, needle),
            "add_nomask",
        )
        assert needle in tb.line_of_code
    assert torch.count_nonzero(out) == 0


def test_a_loop_advancing_a_pointer_is_checked_per_iteration():
    @triton.jit
    def loop_sum(x_ptr, out_ptr, n_iters, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        ptrs = x_ptr + offs
        acc = tl.zeros((BLOCK,), tl.float32)
        for _ in range(n_iters):
            acc += tl.load(ptrs)
            ptrs += BLOCK
        tl.store(out_ptr + offs, acc)

    det = _sanitizer()
    traced = tilelens.trace(det)(loop_sum)
    x, out = torch.randn(64), torch.zeros(16)

    traced[(1,)](x, out, 4, BLOCK=16)  # 4 * 16 == 64
    assert (det.last_status, det.records) == ("ok", [])
    traced[(1,)](x, out, 0, BLOCK=16)  # a zero-trip loop reads nothing
    assert (det.last_status, det.records) == ("ok", [])

    traced[(1,)](x, out, 5, BLOCK=16)
    (record,) = det.records
    assert (record.kind, record.op_type, record.tensor_name) == (
        "out-of-bounds",
        Load,
        "x_ptr",
    )
    assert record.witness["iter_loop"] == 4
    assert record.violation_offset == 4 * 16 + _lane(record)
    assert record.user_code_tracebacks[0].lineno == _line(loop_sum, "tl.load")


def test_a_store_loop_without_an_accumulator_is_checked():
    @triton.jit
    def store_loop(out_ptr, iters, BLOCK: tl.constexpr):
        for i in range(0, iters):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, tl.full((BLOCK,), 1.0, tl.float32))

    det = _sanitizer()
    traced = tilelens.trace(det)(store_loop)
    out = torch.zeros(16)
    traced[(1,)](out, 4, BLOCK=4)  # 4 * 4 == 16, exactly fits
    assert (det.last_status, det.records) == ("ok", [])
    traced[(1,)](out, 6, BLOCK=4)  # 6 * 4 == 24 > 16
    assert det.last_status == "violations"
    assert {r.op_type for r in det.records} == {Store}
    assert torch.count_nonzero(out) == 0


def test_a_strided_view_is_checked_against_its_elements():
    @triton.jit
    def strided(x_ptr, out_ptr, n, stride, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs * stride, mask=mask), mask=mask)

    @triton.jit
    def ignores_stride(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    x = torch.randn(64, 2)[:, 0]  # stride 2: every other float
    out = torch.zeros(64)
    assert not x.is_contiguous()

    det = _sanitizer()
    tilelens.trace(det)(strided)[(1,)](x, out, 64, x.stride(0), BLOCK=64)
    assert (det.last_status, det.records) == ("ok", [])

    # Offsets 0..63 land in the view's gaps, which are out of bounds.
    tilelens.trace(det)(ignores_stride)[(1,)](x, out, 64, BLOCK=64)
    (record,) = det.records
    assert (record.kind, record.op_type) == ("out-of-bounds", Load)
    assert record.violation_offset % 2 == 1
    assert (record.tensor_facts.strides, record.tensor_facts.contiguous) == (
        (2,),
        False,
    )


def test_an_i32_wrap_is_an_integer_overflow():
    @triton.jit
    def i32_wrap(x_ptr, S):
        pid = tl.program_id(0)
        off = (pid * S) * S  # wraps in i32 for S = 65536
        tl.store(x_ptr + off, 1)

    det = _sanitizer()
    traced = tilelens.trace(det)(i32_wrap)
    x = torch.zeros(16, dtype=torch.int32)

    traced[(4,)](x, 2)
    assert (det.last_status, det.records) == ("ok", [])

    traced[(4,)](x, 65536)
    (record,) = det.records
    assert (record.kind, record.op_type, record.tensor_name) == (
        "integer-overflow",
        Store,
        "x_ptr",
    )
    assert (record.violation_offset, record.violation_address) == (None, None)
    assert not -(1 << 31) <= record.witness["value"] < 1 << 31
    assert record.user_code_tracebacks[0].lineno == _line(i32_wrap, "off = ")
    assert torch.count_nonzero(x) == 0


def test_a_division_by_a_zero_argument_is_reported():
    @triton.jit
    def divide(x_ptr, d):
        pid = tl.program_id(0)
        q = pid // d
        tl.store(x_ptr + q, 1)

    det = _sanitizer()
    traced = tilelens.trace(det)(divide)
    x = torch.zeros(2, dtype=torch.int32)

    traced[(4,)](x, 2)
    assert (det.last_status, det.records) == ("ok", [])

    traced[(4,)](x, 0)
    (record,) = det.records
    assert (record.kind, record.op_type) == ("division-by-zero", Store)
    (tb,) = record.user_code_tracebacks
    assert (tb.lineno, tb.line_of_code.strip()) == (
        _line(divide, "q = "),
        "q = pid // d",
    )


BIG32 = (1 << 31) - 1


def _make_circular():
    """At pid 1, t = pid + BIG wraps in i32, so the kernel divides by 3 (-3)
    where the unbounded reading divides by 0."""

    @triton.jit
    def circ_offset(x_ptr, BIG):
        pid = tl.program_id(0)
        t = pid + BIG
        d = tl.where(t > BIG, 0, 3)
        off = (t // d).to(tl.int64) - BIG // 3
        tl.store(x_ptr + off, 1.0)  # pid 1: a wild store

    @triton.jit
    def circ_loop(x_ptr, BIG):
        pid = tl.program_id(0)
        t = pid + BIG
        d = tl.where(t > BIG, 0, -3)
        hi = tl.minimum((t // d) - BIG // -3 + 1, 4)
        for i in range(0, hi):
            tl.store(x_ptr + i, 1.0)  # pid 1: x[1..3]

    return circ_offset, circ_loop


@pytest.mark.parametrize("which", [0, 1], ids=["offset", "loop"])
def test_a_wrap_that_decides_its_own_divisor_is_never_a_proof(which):
    kernel = _make_circular()[which]
    det = _sanitizer()
    traced = tilelens.trace(det)(kernel)
    x = torch.zeros(1)

    traced[(1,)](x, BIG32)  # pid 0 alone: offset 0, one iteration
    assert (det.last_status, det.records) == ("ok", [])

    traced[(2,)](x, BIG32)
    assert det.last_status == "violations"
    (record,) = det.records
    assert (record.kind, record.witness["pid_0"]) == ("integer-overflow", 1)
    assert record.user_code_tracebacks[0].lineno == _line(kernel, "t = pid + BIG")


def test_a_wrap_a_where_discards_is_no_finding():
    """i * S wraps on the lanes the where discards."""

    @triton.jit
    def guarded_scale(x_ptr, n, S, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        off = tl.where(i < n, (i * S) // S, 0)
        tl.store(x_ptr + off, 1.0)

    det = _sanitizer()
    traced = tilelens.trace(det)(guarded_scale)
    x = torch.zeros(256)
    traced[(1,)](x, 100, 1 << 24, BLOCK=256)
    assert (det.last_status, det.records) == ("ok", [])
    traced[(1,)](x, 200, 1 << 24, BLOCK=256)  # lanes 128..199 read i * S: wraps
    assert [r.kind for r in det.records] == ["integer-overflow"]


def test_launches_on_two_host_threads_check_at_the_same_time():
    """Two traced kernels, each with its own sanitizer, launched from two
    host threads at once: their Z3 checks overlap."""
    kernels = [_make_add_nomask(), _make_add_nomask()]
    dets = [_sanitizer(), _sanitizer()]
    traced = [tilelens.trace(d)(k) for d, k in zip(dets, kernels)]
    tensors = [(torch.randn(4096), torch.zeros(4096)) for _ in kernels]
    for t, (x, out) in zip(traced, tensors):  # compile first, one at a time
        t[(3,)](x, out, 3000, BLOCK=1024)
    barrier = threading.Barrier(len(kernels))
    errors: list[BaseException] = []
    statuses: list[list] = [[] for _ in kernels]

    def launch(i):
        try:
            barrier.wait()
            for k in range(6):
                # another n each time: a check of its own, not a remembered one
                traced[i][(5,)](*tensors[i], 4097 + k, BLOCK=1024)
                statuses[i].append(dets[i].last_status)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=launch, args=(i,)) for i in range(len(kernels))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert statuses == [["violations"] * 6] * len(kernels)


def test_a_grouped_swizzle_matmul_is_modeled():
    """The tutorial-03 grouped swizzle (``//``, ``%`` and ``min`` over launch
    quantities) with the ``% M`` / ``% N`` row clamps removed (TritonBench's
    matmul_triton2): a proof when M, N, K cover the blocks, the real OOB
    otherwise."""

    @triton.jit
    def swizzle_matmul(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):  # fmt: skip
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # no `% M` clamp
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # no `% N` clamp
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)

    def launch(m, n, k):
        det = _sanitizer()
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        c = torch.empty(m, n)
        grid = (triton.cdiv(m, 32) * triton.cdiv(n, 32),)
        tilelens.trace(det)(swizzle_matmul)[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, GROUP_M=8,
        )  # fmt: skip
        return det

    clean = launch(64, 64, 64)
    assert (clean.last_status, clean.records) == ("ok", []), clean.last_verdict
    # M = N = K = 16 < BLOCK: the K-only masks leave rows 16..31 of A (and
    # cols 16..31 of B) unguarded.
    buggy = launch(16, 16, 16)
    assert buggy.last_status == "violations", buggy.last_verdict
    assert {r.tensor_name for r in buggy.records} >= {"a_ptr", "b_ptr"}


# ======== branches: path conditions and abstentions =========


def test_a_modeled_branch_condition_leaves_no_false_witness():
    """``if t > 0: load(p + t * n_cols + offs - n_cols)`` never reads offset
    -n_cols: the t == 0 iteration takes the other branch."""

    @triton.jit
    def guarded_scan(x_ptr, out_ptr, n_steps, n_cols, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        acc = tl.zeros((BLOCK,), tl.float32)
        for i in range(n_steps):
            t = n_steps - 1 - i
            if t > 0:
                prev = tl.load(x_ptr + t * n_cols + offs - n_cols, mask=mask, other=0)
            else:
                prev = tl.zeros((BLOCK,), tl.float32)
            acc += prev
        tl.store(out_ptr + offs, acc, mask=mask)

    det = _sanitizer()
    x, out = torch.randn(5 * 8), torch.zeros(8)
    tilelens.trace(det)(guarded_scan)[(1,)](x, out, 5, 8, BLOCK=8)
    assert (det.last_status, det.records) == ("ok", []), det.last_verdict


def test_a_data_dependent_branch_abstains():
    """A possible OOB under a branch on loaded data is unsupported: never a
    witness from a branch that may not run, never "ok"."""

    @triton.jit
    def flag_gated(flag_ptr, x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        flag = tl.load(flag_ptr)
        acc = tl.zeros((BLOCK,), tl.float32)
        if flag > 0:
            acc = tl.load(x_ptr + offs - n_cols, mask=mask, other=0)
        tl.store(out_ptr + offs, acc, mask=mask)

    det = _sanitizer()
    flag = torch.zeros(1, dtype=torch.int32)
    x, out = torch.randn(8), torch.zeros(8)
    tilelens.trace(det)(flag_gated)[(1,)](flag, x, out, 8, BLOCK=8)
    assert (det.last_status, det.records) == ("unsupported", [])
    refusal = det.last_verdict.refusal
    assert refusal.kind == "unmodelable-condition"
    assert refusal.loc.line == _line(flag_gated, "offs - n_cols")


def test_an_unguarded_oob_is_reported_beside_a_branch():
    @triton.jit
    def mixed(x_ptr, out_ptr, n, flag, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)  # unguarded: OOB when numel < BLOCK
        if flag > 0:
            x += tl.load(x_ptr + offs, mask=offs < n, other=0)  # guarded, safe
        tl.store(out_ptr + offs, x, mask=offs < n)

    det = _sanitizer()
    x, out = torch.randn(8), torch.zeros(8)
    tilelens.trace(det)(mixed)[(1,)](x, out, 8, 1, BLOCK=16)
    assert det.last_status == "violations", det.last_verdict
    (record,) = det.records
    assert record.op_type is Load
    assert record.user_code_tracebacks[0].lineno == _line(mixed, "unguarded")


# ======== what cannot be checked =========


def test_a_gather_is_unsupported_at_its_line():
    @triton.jit
    def gather(idx_ptr, src_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        idx = tl.load(idx_ptr + offs, mask=mask)
        vals = tl.load(src_ptr + idx, mask=mask)  # data-dependent address
        tl.store(out_ptr + offs, vals, mask=mask)

    det = Sanitizer(compile=True)  # abort_on_error: unsupported never exits
    n = 1024
    idx = torch.zeros(n, dtype=torch.int32)
    src, out = torch.randn(n), torch.zeros(n)
    tilelens.trace(det)(gather)[(4,)](idx, src, out, n, BLOCK=256)

    assert (det.last_status, det.records) == ("unsupported", [])
    refusal = det.last_verdict.refusal
    assert refusal.kind == "indirect-address"
    assert (refusal.loc.file, refusal.loc.line) == (
        __file__,
        _line(gather, "src_ptr + idx"),
    )
    assert refusal.message.startswith(f"{__file__}:{refusal.loc.line}: ")
    assert det.last_verdict.per_config[0].refusal == refusal


def test_nested_loops_are_unsupported():
    @triton.jit
    def nested(in_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
        for i in range(0, M):
            for j in range(0, N):
                offs = (i * N + j) * BLOCK + tl.arange(0, BLOCK)
                tl.store(out_ptr + offs, tl.load(in_ptr + offs))

    det = _sanitizer()
    inp, out = torch.randn(20), torch.zeros(20)
    tilelens.trace(det)(nested)[(1,)](inp, out, 8, 2, BLOCK=4)
    assert (det.last_status, det.records) == ("unsupported", [])
    assert det.last_verdict.refusal.kind == "nested-loop"


# How a kernel taking a tuple, or a host TensorDescriptor, is refused: Triton
# names each TTIR argument the parameter flattens to by its path in the tuple
# (``ptrs.0``), which reads, and binds to no launch argument (a host
# descriptor's leaves still repeat a name, ``d.shape.0``). Never checked,
# never a finding either way.
_AGGREGATE_REFUSALS = {
    "tuple of pointers": "missing-binding",
    "tuple of ints": "missing-binding",
    "descriptor": "other",
}


def _tuple_of_pointers():
    @triton.jit
    def tuple_of_pointers(ptrs, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        vals = tl.load(ptrs[0] + offs)  # unmasked: out of bounds of 32 elements
        tl.store(ptrs[1] + offs, vals, mask=offs < n)

    return tuple_of_pointers, ((torch.zeros(32), torch.zeros(64)), 64), {"BLOCK": 64}


def _tuple_of_ints():
    @triton.jit
    def tuple_of_ints(x_ptr, bounds, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK) + bounds[0]
        tl.store(x_ptr + offs, 1.0, mask=offs < bounds[1])  # out of bounds from 8

    return tuple_of_ints, (torch.zeros(64), (8, 72)), {"BLOCK": 64}


def _host_descriptor():
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def descriptor_copy(desc, out_ptr, BM: tl.constexpr, BN: tl.constexpr):
        offs = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        tl.store(out_ptr + offs, desc.load([0, 0]))

    desc = TensorDescriptor.from_tensor(torch.zeros(64, 64), [32, 32])
    return descriptor_copy, (desc, torch.zeros(16 * 16)), {"BM": 32, "BN": 32}


@pytest.mark.parametrize(
    "case, make",
    [
        ("tuple of pointers", _tuple_of_pointers),
        ("tuple of ints", _tuple_of_ints),
        ("descriptor", _host_descriptor),
    ],
)
def test_a_tuple_or_host_descriptor_parameter_is_unsupported(case, make):
    kernel, args, kwargs = make()
    det = _sanitizer()
    tilelens.trace(det)(kernel)[(1,)](*args, **kwargs)
    assert (det.last_status, det.records) == ("unsupported", []), det.last_verdict
    refusal = det.last_verdict.refusal
    assert refusal.kind == _AGGREGATE_REFUSALS[case], refusal


# ======== configs =========


def test_autotune_reports_the_one_config_that_goes_out_of_bounds():
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_warps=1),
            triton.Config({"BLOCK": 128}, num_warps=1),
        ],
        key=["n"],
    )
    @triton.jit
    def copy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs))  # no mask

    det = _sanitizer()
    x, out = torch.randn(64), torch.zeros(64)
    ret = tilelens.trace(det)(copy)[lambda meta: (triton.cdiv(64, meta["BLOCK"]),)](
        x, out, 64
    )

    assert ret is None  # a skipped autotuned launch picks no config
    verdict = det.last_verdict
    assert verdict.status == "violations"
    assert [(c.config["BLOCK"], c.status) for c in verdict.per_config] == [
        (16, "ok"),
        (128, "violations"),
    ]
    assert {r.config["BLOCK"] for r in det.records} == {128}
    assert verdict.per_config[1].n_reports == len(det.records) == 2
    assert torch.count_nonzero(out) == 0


def test_configs_compiling_to_one_kernel_are_each_checked():
    """S is a runtime int that 2 and 3
    specialize alike, so both configs compile to one kernel; only S=3, on
    its larger grid, goes out of bounds. Deduplicating events by kernel
    alone would check S=2's binding only and prove the launch."""

    @triton.autotune(
        configs=[
            triton.Config({"S": 2}, num_warps=1),
            triton.Config({"S": 3}, num_warps=1),
        ],
        key=["n"],
    )
    @triton.jit
    def strided_copy(x_ptr, out_ptr, n, S, BLOCK: tl.constexpr):
        # Program p covers [p * BLOCK * S, p * BLOCK * S + BLOCK).
        offs = tl.program_id(0) * BLOCK * S + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs))

    det = _sanitizer()
    x, out = torch.randn(64), torch.zeros(64)
    # S=2: 2 programs, up to element 47; S=3: 4 programs, up to 159.
    tilelens.trace(det)(strided_copy)[lambda meta: (4 if meta["S"] == 3 else 2,)](
        x, out, 64, BLOCK=16
    )

    verdict = det.last_verdict
    assert verdict.status == "violations"
    assert [(c.config["S"], c.status) for c in verdict.per_config] == [
        (2, "ok"),
        (3, "violations"),
    ]
    assert len({c.specialization for c in verdict.per_config}) == 1
    assert {(r.tensor_name, r.config["S"]) for r in det.records} == {
        ("x_ptr", 3),
        ("out_ptr", 3),
    }
    assert torch.count_nonzero(out) == 0


def test_a_config_that_fails_to_compile_is_a_note():
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_warps=1),
            triton.Config({"BLOCK": 64}, num_warps=1),
        ],
        key=["n"],
    )
    @triton.jit
    def add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        tl.static_assert(BLOCK <= 32)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    det = _sanitizer()
    x, out = torch.randn(64), torch.zeros(64)
    tilelens.trace(det)(add_one)[lambda meta: (triton.cdiv(64, meta["BLOCK"]),)](
        x, out, 64
    )

    verdict = det.last_verdict
    assert verdict.status == "ok"
    (config,) = verdict.per_config
    assert config.config["BLOCK"] == 16
    (note,) = verdict.notes
    assert "'BLOCK': 64" in note and "CompileTimeAssertionFailure" in note


# ======== composition and the process exit =========


def test_a_mixed_trace_with_the_eager_tracer():
    """The compiled sanitizer checks the compiled kernel; the tracer
    gets the interpreted run, which writes the outputs."""
    det, tracer = _sanitizer(), Tracer()
    traced = tilelens.trace(tracer)(tilelens.trace(det)(_make_add()))
    n = 100
    x, y = torch.randn(n), torch.randn(n)
    out = torch.zeros(n)

    traced[(1,)](x, y, out, n, BLOCK=128)

    assert det.last_status == "ok"
    records = trace_module.launches[-1].records
    assert det.last_verdict in records
    assert {type(r) for r in records if not isinstance(r, IRVerdict)} >= {Load, Store}
    torch.testing.assert_close(out, x + y)


# Triton's on-disk cache keys a kernel by its source and first line, not its
# file, and a cached TTIR keeps the locs of the file first compiled: the
# comment makes each script's kernel its own (the TTIR-only host compile
# never reaches that cache; the JIT's own compile does).
_OOB_SCRIPT = """\
import torch, triton, triton.language as tl
{prelude}

{decorator}
@triton.jit
def add_nomask(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # {tag}
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs))

n = {n}
x = torch.randn(n)
out = torch.zeros(n)
add_nomask[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
print("launch returned")
"""


def _run(script: Path, *argv: str, **env_vars: str) -> subprocess.CompletedProcess:
    # No GPU in the child either, and no IR target but env_vars'. This
    # checkout first on the path, then whatever the caller put there (e.g.
    # another Triton release).
    unset = ("TRITON_INTERPRET", "TILELENS_IR_TARGET", "TRITON_VIZ_IR_TARGET")
    env = {k: v for k, v in os.environ.items() if k not in unset}
    path = os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])])
    env.update(PYTHONPATH=path, CUDA_VISIBLE_DEVICES="", **env_vars)
    return subprocess.run(
        [sys.executable, *argv], capture_output=True, text=True, env=env, cwd=REPO
    )


def test_abort_on_error_exits_after_reporting(tmp_path):
    script = tmp_path / "oob.py"
    script.write_text(
        _OOB_SCRIPT.format(
            prelude="import tilelens\nfrom tilelens.clients import Sanitizer",
            decorator="@tilelens.trace(Sanitizer(compile=True))",
            n=3000,
            tag=script,
        )
    )
    proc = _run(script, str(script))
    assert proc.returncode == 1, proc.stderr
    assert proc.stdout.count("Out-Of-Bounds Access Detected") == 2
    lines = script.read_text().splitlines()
    (line,) = [i for i, text in enumerate(lines, 1) if "tl.store(" in text]
    assert f"File: {script}, Line: {line}, in add_nomask" in proc.stdout
    assert "launch returned" not in proc.stdout


def test_the_cli_compile_flag_runs_the_compiled_sanitizer(tmp_path):
    script = tmp_path / "oob.py"
    script.write_text(_OOB_SCRIPT.format(prelude="", decorator="", n=3000, tag=script))
    cli = (
        f"import sys; sys.argv = ['tile-sanitizer', '--compile', {str(script)!r}]; "
        "from tilelens.wrapper import apply_sanitizer; apply_sanitizer()"
    )
    proc = _run(script, "-c", cli)
    assert proc.returncode == 1, proc.stderr
    assert proc.stdout.count("(compiled sanitizer)") == 2
    assert f"File: {script}, " in proc.stdout
    assert "launch returned" not in proc.stdout


# ======== the target =========


def _make_two_cta_configs():
    # num_ctas=2 is an sm90+ option: a compile for the default cuda:89 (or
    # any target below sm90) rejects the config.
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_ctas=1),
            triton.Config({"BLOCK": 16}, num_ctas=2),
        ],
        key=["n"],
    )
    @triton.jit
    def copy_ctas(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs))

    return copy_ctas


def _assert_compile_failed(refusal, target, error):
    """A compile-failed refusal: the config failed to compile for ``target``, with
    ``error``, and the refusal says how to check it for another target. It
    is one line, starting where in the kernel's source file it failed."""
    assert refusal.kind == "compile-failed"
    assert f"failed to compile for {target} (" in refusal.message
    assert error in refusal.message
    assert "Sanitizer(compile=True, target=...)" in refusal.message
    assert "TILELENS_IR_TARGET" in refusal.message
    assert "\n" not in refusal.message
    loc = refusal.loc
    assert loc is not None and refusal.message.startswith(f"{loc.file}:{loc.line}: ")


def test_a_target_passed_to_the_sanitizer_is_the_one_checked():
    x, out = torch.zeros(64), torch.zeros(64)
    # For the default cuda:89 the sm90 config fails to compile: unchecked,
    # and it may run on an sm90 GPU, so the launch is unsupported.
    det = _sanitizer()
    tilelens.trace(det)(_make_two_cta_configs())[(4,)](x, out, 64)
    assert det.last_status == "unsupported"
    one, two = det.last_verdict.per_config
    assert (one.config["num_ctas"], one.status) == (1, "ok")
    assert (two.config["num_ctas"], two.status) == (2, "unsupported")

    for target in ("cuda:90", GPUTarget("cuda", 90, 32)):
        det = Sanitizer(compile=True, abort_on_error=False, target=target)
        assert det.ir_target == GPUTarget("cuda", 90, 32)
        tilelens.trace(det)(_make_two_cta_configs())[(4,)](x, out, 64)
        assert det.last_status == "ok", det.last_verdict
        per_config = det.last_verdict.per_config
        assert [c.config["num_ctas"] for c in per_config] == [1, 2]


# ======== a kernel that fails to compile for the target =========


def test_a_config_that_needs_two_ctas_is_unsupported_and_the_program_goes_on(capsys):
    """An autotuned kernel one config of which needs num_ctas=2 (sm90+): that
    config is unsupported, kind compile-failed, naming the target and how to
    name another; the other is checked; the launch returns (None, as for any
    skipped autotuned launch) even with abort_on_error, and the program
    goes on to its next launch."""
    det = Sanitizer(compile=True)  # abort_on_error=True
    kernel = _make_two_cta_configs()
    traced = tilelens.trace(det)(kernel)
    x, out = torch.zeros(64), torch.zeros(64)

    assert traced[(4,)](x, out, 64) is None

    verdict = det.last_verdict
    assert (verdict.status, verdict.scope, verdict.notes) == ("unsupported", None, ())
    ok, failed = verdict.per_config
    assert (ok.config["num_ctas"], ok.status) == (1, "ok")
    assert (failed.config["num_ctas"], failed.status) == (2, "unsupported")
    assert failed.specialization is None and verdict.refusal == failed.refusal
    _assert_compile_failed(
        failed.refusal, "cuda:89", "num_ctas > 1 requires NVIDIA SM90"
    )
    assert det.records == []
    # No source position for an option error: the kernel's def line.
    assert failed.refusal.loc == SourceLocation(
        kernel.fn.fn.__code__.co_filename, _line(kernel.fn, "def copy_ctas(")
    )
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith("[CompiledSanitizer] not checked (config {")
    assert (
        f": compile-failed: {failed.refusal.loc.file}:{failed.refusal.loc.line}: "
        "it failed to compile for cuda:89 (ValueError: num_ctas > 1 requires"
    ) in line

    # The program goes on: its next launch is checked like any other.
    traced[(4,)](x, out, 64)
    assert det.last_status == "unsupported"
    assert [c.status for c in det.last_verdict.per_config] == ["ok", "unsupported"]


def test_a_plain_kernel_launched_with_two_ctas_is_unsupported():
    """A plain kernel whose only config fails to compile for the target: the
    launch is unsupported compile-failed and returns None (no kernel to
    return), and the next launch compiles as usual."""
    det = Sanitizer(compile=True)  # abort_on_error=True
    traced = tilelens.trace(det)(_make_add_nomask())
    x, out = torch.zeros(4096), torch.zeros(4096)

    assert traced[(4,)](x, out, 4096, BLOCK=1024, num_ctas=2) is None

    verdict = det.last_verdict
    assert verdict.status == "unsupported" and verdict.notes == ()
    (failed,) = verdict.per_config
    assert (failed.specialization, failed.config, failed.status) == (
        None,
        {},
        "unsupported",
    )
    assert verdict.refusal == failed.refusal
    _assert_compile_failed(
        failed.refusal, "cuda:89", "num_ctas > 1 requires NVIDIA SM90"
    )

    kernel = traced[(4,)](x, out, 4096, BLOCK=1024)
    assert kernel.target == GPUTarget("cuda", 89, 32)
    assert det.last_status == "ok"


def _make_block_asserting_configs(*blocks):
    @triton.autotune(
        configs=[triton.Config({"BLOCK": block}, num_warps=1) for block in blocks],
        key=["n"],
    )
    @triton.jit
    def add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        tl.static_assert(BLOCK <= 32)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return add_one


def test_a_launch_none_of_whose_configs_compiled_is_unsupported():
    """Every config failed: unsupported compile-failed, whether each could
    compile for another target (num_ctas=2 here: its refusal) or for none
    (a failing tl.static_assert: noted), and the launch returns."""
    x, out = torch.zeros(64), torch.zeros(64)

    def grid(meta):
        return (triton.cdiv(64, meta["BLOCK"]),)

    det = Sanitizer(compile=True)
    kernel = _make_block_asserting_configs(64, 128)
    assert tilelens.trace(det)(kernel)[grid](x, out, 64) is None
    verdict = det.last_verdict
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", "compile-failed")
    assert verdict.per_config == ()
    first, second = verdict.notes
    assert first.startswith("config {'BLOCK': 64, ")
    assert second.startswith("config {'BLOCK': 128, ")
    # Where the assertion is, and the error in one line.
    path = kernel.fn.fn.__code__.co_filename
    at = f"{path}:{_line(kernel.fn, 'tl.static_assert(')}: "
    assert all(
        f"was not checked: {at}it failed to compile for cuda:89 "
        "(CompileTimeAssertionFailure), " in note
        for note in verdict.notes
    )
    assert verdict.refusal.message == (
        f"{path}:{_line(kernel.fn, 'def add_one(')}: no config of the launch "
        "compiled for cuda:89, so nothing was checked: each failed with an "
        "error of its own code whatever the target (see the notes)"
    )

    @triton.autotune(configs=[triton.Config({"BLOCK": 16}, num_ctas=2)], key=["n"])
    @triton.jit
    def copy_two_ctas(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs))

    det = Sanitizer(compile=True)
    assert tilelens.trace(det)(copy_two_ctas)[(4,)](x, out, 64) is None
    verdict = det.last_verdict
    assert verdict.status == "unsupported" and verdict.notes == ()
    (failed,) = verdict.per_config
    assert verdict.refusal == failed.refusal
    _assert_compile_failed(failed.refusal, "cuda:89", "num_ctas > 1")


def test_a_static_assert_on_the_target_is_the_targets():
    """A tl.static_assert that asks for the target (tl.target_info) fails
    for this target only: the config may run on the user's GPU, so it is
    unsupported, not a note; a target it compiles for checks it."""

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16, "HOPPER": False}),
            triton.Config({"BLOCK": 16, "HOPPER": True}),
        ],
        key=["n"],
    )
    @triton.jit
    def maybe_hopper(x_ptr, n, BLOCK: tl.constexpr, HOPPER: tl.constexpr):
        tl.static_assert(not HOPPER or tl.target_info.cuda_capability_geq(9, 0))
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, 1.0, mask=offs < n)

    x = torch.zeros(64)
    det = _sanitizer()
    tilelens.trace(det)(maybe_hopper)[(4,)](x, 64)
    verdict = det.last_verdict
    assert verdict.status == "unsupported" and verdict.notes == ()
    ok, failed = verdict.per_config
    assert (ok.config["HOPPER"], ok.status) == (False, "ok")
    assert (failed.config["HOPPER"], failed.status) == (True, "unsupported")
    _assert_compile_failed(failed.refusal, "cuda:89", "CompileTimeAssertionFailure")

    det = Sanitizer(compile=True, abort_on_error=False, target="cuda:90")
    tilelens.trace(det)(maybe_hopper)[(4,)](x, 64)
    assert det.last_status == "ok"
    assert [c.status for c in det.last_verdict.per_config] == ["ok", "ok"]


def _make_device_asserting_configs(asks):
    # Two configs, both calling ``asks()`` (the first compiles first); the
    # second, which goes out of bounds (64 lanes, no mask, over 16
    # elements), holds only where ``asks()`` answers yes.
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16, "BIG": False}),
            triton.Config({"BLOCK": 64, "BIG": True}),
        ],
        key=["n"],
    )
    @triton.jit
    def big_only(x_ptr, n, BLOCK: tl.constexpr, BIG: tl.constexpr):
        yes: tl.constexpr = asks()
        tl.static_assert(not BIG or yes)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, 1.0)

    return big_only


def test_a_static_assert_on_a_caught_device_query_is_the_devices():
    """The host compile refuses a device query; a kernel that catches that
    and falls back to an answer of its own ("no big shared memory") fails
    a static_assert a GPU with more might pass (an H100 would run the out
    of bounds config): that config was not checked, so the launch is never
    "ok", and it is no note."""
    from triton.runtime.jit import constexpr_function

    @constexpr_function
    def has_big_smem():
        from triton.runtime import driver

        try:
            properties = driver.active.utils.get_device_properties(0)
        except Exception:
            return False
        return properties["max_shared_mem"] >= 200_000

    kernel = _make_device_asserting_configs(has_big_smem)
    det = _sanitizer()
    tilelens.trace(det)(kernel)[(1,)](torch.zeros(16), 16)
    verdict = det.last_verdict
    assert verdict.status == "unsupported" and verdict.notes == ()
    ok, failed = verdict.per_config
    assert (ok.config["BIG"], ok.status) == (False, "ok")
    assert (failed.config["BIG"], failed.status) == (True, "unsupported")
    _assert_compile_failed(failed.refusal, "cuda:89", "CompileTimeAssertionFailure")
    assert failed.refusal.loc.line == _line(kernel.fn, "tl.static_assert(")


def test_a_static_assert_on_a_kept_target_answer_is_the_targets():
    """A kernel's code may keep the target answer after its first compile
    asked (a memo): a later config failing a static_assert on the kept
    answer, without asking again, is the target's all the same. At a
    target it holds for, the config is checked (and goes out of bounds)."""
    from triton.runtime.jit import constexpr_function

    @constexpr_function
    def is_hopper(_memo={}):  # noqa: B006  the kernel's own memo
        if "arch" not in _memo:
            from triton.runtime import driver

            _memo["arch"] = driver.active.get_current_target().arch
        return _memo["arch"] >= 90

    kernel = _make_device_asserting_configs(is_hopper)
    det = _sanitizer()
    tilelens.trace(det)(kernel)[(1,)](torch.zeros(16), 16)
    assert is_hopper.fn.__defaults__[0] == {"arch": 89}
    verdict = det.last_verdict
    assert verdict.status == "unsupported" and verdict.notes == ()
    ok, failed = verdict.per_config
    assert (ok.config["BIG"], ok.status) == (False, "ok")
    assert (failed.config["BIG"], failed.status) == (True, "unsupported")
    _assert_compile_failed(failed.refusal, "cuda:89", "CompileTimeAssertionFailure")

    is_hopper.fn.__defaults__[0].clear()
    det = Sanitizer(compile=True, abort_on_error=False, target="cuda:90")
    tilelens.trace(det)(kernel)[(1,)](torch.zeros(16), 16)
    assert det.last_status == "violations"
    assert {r.config["BIG"] for r in det.records} == {True}


# ======== calls that do not bind the kernel's parameters =========


def _make_two_args():
    @triton.jit
    def two_args(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, 1.0, mask=offs < n)

    return two_args


# Calls of two_args(x_ptr, n, BLOCK) that do not bind: the arguments after
# x_ptr, and the keyword arguments.
_UNBOUND_CALLS = {
    "missing-argument": ((), {"BLOCK": 64}),
    "extra-argument": ((64, 64, 7), {}),
    "misnamed-keyword": ((), {"N": 64, "BLOCK": 64}),
    "repeated-keyword": ((64,), {"n": 64, "BLOCK": 64}),
    # binds, but the JIT cannot key the call (compute_cache_key), on any GPU
    "unhashable-constexpr": ((64,), {"BLOCK": [64]}),
}


class _StandInDriver:
    """What JITFunction.run asks Triton's driver for before it binds a call,
    on a machine with a GPU of the default IR target."""

    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return GPUTarget("cuda", 89, 32)


def _untraced_error(kernel, args, kwargs) -> BaseException:
    """What the untraced JIT raises for ``kernel[grid](*args, **kwargs)``,
    for a call that does not bind: JITFunction.run's own binder, on a
    stand-in driver (no GPU: the call fails before anything is compiled or
    launched)."""
    from triton.runtime.driver import driver

    owner = type(driver)
    saved = owner.__dict__["active"]
    stand_in = _StandInDriver()
    owner.active = property(lambda self: stand_in)
    try:
        kernel.warmup(*args, grid=(1,), **kwargs)
    except Exception as exc:
        return exc
    finally:
        owner.active = saved
    raise AssertionError("the untraced call bound")


@pytest.mark.parametrize("case", list(_UNBOUND_CALLS))
def test_a_call_that_does_not_bind_raises_as_untraced(case, capsys):
    """A call that does not match the kernel's signature is a bug in
    the call, whatever the target: the launch raises the untraced JIT's own
    error (the same type and message) instead of reporting the launch as
    not checked. Nothing is printed or recorded, and the trace goes on."""
    kernel = _make_two_args()
    x = torch.zeros(64)
    rest, kwargs = _UNBOUND_CALLS[case]
    expected = _untraced_error(kernel, (x, *rest), kwargs)
    assert type(expected) is TypeError
    det = Sanitizer(compile=True)  # abort_on_error=True
    traced = tilelens.trace(det)(kernel)
    launches = len(trace_module.launches)

    with pytest.raises(TypeError) as raised:
        traced[(1,)](x, *rest, **kwargs)

    assert (type(raised.value), str(raised.value)) == (type(expected), str(expected))
    assert det.last_verdict is None and det.records == []
    assert len(trace_module.launches) == launches
    assert capsys.readouterr().out == ""
    # The trace is not left mid-launch: a call that binds is checked.
    traced[(1,)](x, 64, BLOCK=64)
    assert det.last_status == "ok"


def test_an_autotuned_call_that_does_not_bind_raises_as_untraced():
    """Through the autotuner too: the first config's compile raises."""

    @triton.autotune(
        configs=[triton.Config({"BLOCK": 16}), triton.Config({"BLOCK": 32})],
        key=["n"],
    )
    @triton.jit
    def tuned(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, 1.0, mask=offs < n)

    x = torch.zeros(64)
    # The JIT's binder, as the autotuner calls it for its first config.
    expected = _untraced_error(tuned.fn, (x,), {"BLOCK": 16})
    det = _sanitizer()

    with pytest.raises(TypeError) as raised:
        tilelens.trace(det)(tuned)[(1,)](x)

    assert str(raised.value) == str(expected)
    assert det.last_verdict is None


def _make_tuned():
    @triton.autotune(
        configs=[triton.Config({"BLOCK": 16}), triton.Config({"BLOCK": 32})],
        key=["n"],
    )
    @triton.jit
    def tuned(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, 1.0, mask=offs < n)

    return tuned


def test_an_autotuned_call_passing_a_tuned_parameter_raises_as_untraced():
    """A call that passes an autotuned meta-parameter itself: the untraced
    launch's autotuning refuses it before benchmarking (Autotuner._bench's
    ValueError; no device involved), and so does a traced launch, IR-only
    or mixed, not a TypeError naming a tilelens internal; the trace goes
    on."""
    x = torch.zeros(64)
    with pytest.raises(ValueError) as untraced:
        _make_tuned()[(1,)](x, 64, BLOCK=64)
    assert "Conflicting meta-parameters: BLOCK" in str(untraced.value)
    det = _sanitizer()
    traced = tilelens.trace(det)(_make_tuned())

    with pytest.raises(ValueError) as raised:
        traced[(1,)](x, 64, BLOCK=64)

    assert str(raised.value) == str(untraced.value)
    assert det.last_verdict is None
    traced[(1,)](x, 64)
    assert det.last_status == "ok"
    mixed = tilelens.trace(Tracer())(tilelens.trace(_sanitizer())(_make_tuned()))
    with pytest.raises(ValueError) as raised:
        mixed[(1,)](x, 64, BLOCK=64)
    assert str(raised.value) == str(untraced.value)
    assert torch.equal(x, torch.zeros(64))


def test_a_mixed_trace_raises_the_untraced_error_before_interpreting():
    """In a mixed trace, the call raises the untraced JIT's error
    from the compile pass, before the interpreter runs, which would fail on
    the same call too (with Python's own TypeError for the kernel's
    function), so no output is written either way."""
    kernel = _make_two_args()
    x = torch.zeros(64)
    expected = _untraced_error(kernel, (x,), {"BLOCK": 64})
    traced = tilelens.trace(Tracer())(tilelens.trace(_sanitizer())(kernel))

    with pytest.raises(TypeError) as raised:
        traced[(1,)](x, BLOCK=64)

    assert str(raised.value) == str(expected)
    assert torch.equal(x, torch.zeros(64))
    with pytest.raises(TypeError, match="'n'"):
        tilelens.trace(Tracer())(_make_two_args())[(1,)](x, BLOCK=64)
    assert torch.equal(x, torch.zeros(64))


def test_an_option_the_target_does_not_know_is_a_compile_failure():
    """A keyword naming no parameter is a compile option, which the
    target's backend may not know (waves_per_eu is a HIP option): no bind
    failure, but the target's compile failure, so the launch is
    unsupported, the program goes on, and another target checks it."""
    x = torch.zeros(64)
    det = _sanitizer()
    call = dict(BLOCK=64, waves_per_eu=2)

    assert tilelens.trace(det)(_make_two_args())[(1,)](x, 64, **call) is None

    refusal = det.last_verdict.refusal
    assert (det.last_status, refusal.kind) == ("unsupported", "compile-failed")
    assert "failed to compile for cuda:89 (KeyError:" in refusal.message
    assert "waves_per_eu" in refusal.message and "TILELENS_IR_TARGET" in refusal.message
    hip = Sanitizer(compile=True, abort_on_error=False, target="hip:gfx942")
    tilelens.trace(hip)(_make_two_args())[(1,)](x, 64, **call)
    assert hip.last_status == "ok"


@pytest.mark.parametrize("target", ["cuda:89", "hip:gfx942"])
def test_an_unknown_option_is_named_not_blamed_on_the_kernel(target):
    """A keyword no backend knows (a misspelled option) fails for every
    target: the refusal says the call passes an option the target lacks,
    that a misspelled one fails on every GPU, and that only a target of a
    backend with that option can check it; it does not claim the kernel may
    compile for another target."""
    det = Sanitizer(compile=True, abort_on_error=False, target=target)

    assert (
        tilelens.trace(det)(_make_two_args())[(1,)](
            torch.zeros(64), 64, BLOCK=64, bogus=1
        )
        is None
    )

    refusal = det.last_verdict.refusal
    assert (det.last_status, refusal.kind) == ("unsupported", "compile-failed")
    assert (
        f"the call passes 'bogus', neither a parameter of the kernel nor a "
        f"compile option for {target}, so it was not checked" in refusal.message
    )
    assert "a misspelled option: on every GPU" in refusal.message
    assert "name a target of that backend" in refusal.message
    assert "a kernel can compile for one target and fail for another" not in (
        refusal.message
    )


_UNBOUND_SCRIPT = """\
import torch, triton, triton.language as tl


@triton.jit
def two_args(x_ptr, n, BLOCK: tl.constexpr):
    # {tag}
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, 1.0, mask=offs < n)


x = torch.zeros(64)
print("before the launch")
two_args[(1,)]({call})  # the launch
print("launch returned")
"""


def test_the_cli_raises_a_call_that_does_not_bind(tmp_path):
    """tile-sanitizer --compile: the script fails at the call as it does
    untraced, with a non-zero exit status and the traceback, whose last
    line is the untraced JIT's error."""
    rest, kwargs = _UNBOUND_CALLS["missing-argument"]
    expected = _untraced_error(_make_two_args(), (torch.zeros(64), *rest), kwargs)
    call = ", ".join(
        ["x", *map(repr, rest), *(f"{k}={v!r}" for k, v in kwargs.items())]
    )
    script = tmp_path / "unbound.py"
    script.write_text(_UNBOUND_SCRIPT.format(tag=script, call=call))
    cli = (
        f"import sys; sys.argv = ['tile-sanitizer', '--compile', {str(script)!r}]; "
        "from tilelens.wrapper import apply_sanitizer; apply_sanitizer()"
    )

    proc = _run(script, "-c", cli)

    assert proc.returncode == 1, proc.stderr
    assert proc.stdout.splitlines() == ["before the launch"]
    lines = script.read_text().splitlines()
    (line,) = [i for i, text in enumerate(lines, 1) if "# the launch" in text]
    assert "Traceback (most recent call last):" in proc.stderr
    assert f'File "{script}", line {line}, in <module>' in proc.stderr
    assert proc.stderr.rstrip().splitlines()[-1] == f"TypeError: {expected}"


def test_a_mixed_trace_still_interprets_after_a_compile_failure():
    """The interpreted peer runs (and writes the outputs) while
    the compiled sanitizer reports the config it could not compile."""
    det, tracer = _sanitizer(), Tracer()
    traced = tilelens.trace(tracer)(tilelens.trace(det)(_make_add_nomask()))
    x, out = torch.randn(64), torch.zeros(64)

    traced[(1,)](x, out, 64, BLOCK=64, num_ctas=2)

    torch.testing.assert_close(out, x)
    verdict = det.last_verdict
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", "compile-failed")
    assert any(isinstance(r, Store) for r in trace_module.launches[-1].records)


def _make_unmasked_on_cuda():
    @triton.jit
    def unmasked_on_cuda(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if tl.target_info.is_cuda():
            tl.store(x_ptr + offs, 5.0)  # what every CUDA device runs
        else:
            tl.store(x_ptr + offs, 6.0, mask=offs < n)

    return unmasked_on_cuda


def _make_unmasked_from_sm89():
    @triton.jit
    def unmasked_from_sm89(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if tl.target_info.cuda_capability_geq(8, 9):
            tl.store(x_ptr + offs, 1.0)
        else:
            tl.store(x_ptr + offs, 2.0, mask=offs < n)

    return unmasked_from_sm89


def _make_unmasked_on_hip():
    @triton.jit
    def unmasked_on_hip(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if tl.target_info.is_hip():
            tl.store(x_ptr + offs, 3.0)
        else:
            tl.store(x_ptr + offs, 4.0, mask=offs < n)

    return unmasked_on_hip


# (kernel, target, status): 128 lanes over 64 elements, so the unmasked
# branch goes out of bounds.
TARGET_BRANCHES = [
    (_make_unmasked_on_cuda, None, "violations"),
    (_make_unmasked_on_cuda, "hip:gfx942", "ok"),
    (_make_unmasked_from_sm89, None, "violations"),
    (_make_unmasked_from_sm89, "cuda:80", "ok"),
    (_make_unmasked_from_sm89, "cuda:90", "violations"),
    (_make_unmasked_on_hip, None, "ok"),
    (_make_unmasked_on_hip, "hip:gfx942", "violations"),
]


def _check_branches(make, target):
    det = Sanitizer(compile=True, abort_on_error=False, target=target)
    tilelens.trace(det)(make())[(8,)](torch.zeros(64), 64, BLOCK=16)
    return det.last_status, sorted(
        (r.kind, r.tensor_name, r.violation_offset, tuple(sorted(r.witness.items())))
        for r in det.records
    )


@pytest.mark.parametrize(
    "make, target, status",
    TARGET_BRANCHES,
    ids=[f"{m.__name__[6:]}-{t or 'default'}" for m, t, _ in TARGET_BRANCHES],
)
def test_target_dependent_branches_are_the_ir_targets(make, target, status):
    assert _check_branches(make, target)[0] == status


class _Machine:
    """A stand-in for Triton's active driver on a machine with a GPU of
    ``target``; it must never be asked during IR mode's compile."""

    def __init__(self, target):
        self.target = target

    def get_current_target(self):
        raise AssertionError("IR mode asked the machine's driver for its target")


def test_a_verdict_does_not_depend_on_the_machine(monkeypatch):
    """Every verdict and finding above is the same on a GPU of any kind as
    without one (this module's default: the driver is unreachable)."""
    without_gpu = [_check_branches(make, target) for make, target, _ in TARGET_BRANCHES]
    from triton.runtime.driver import driver

    stand_in = _Machine(GPUTarget("cuda", 90, 32))
    monkeypatch.setattr(type(driver), "active", property(lambda self: stand_in))
    on_gpu = [_check_branches(make, target) for make, target, _ in TARGET_BRANCHES]
    assert on_gpu == without_gpu
    assert [status for status, _ in on_gpu] == [s for _, _, s in TARGET_BRANCHES]


def _make_to_fp8e4nv():
    @triton.jit
    def to_fp8e4nv(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x.to(tl.float8e4nv).to(tl.float32), mask=mask)

    return to_fp8e4nv


def _make_load_fp8e4nv():
    @triton.jit
    def load_fp8e4nv(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(
            out_ptr + offs, tl.load(x_ptr + offs, mask=mask).to(tl.float32), mask=mask
        )

    return load_fp8e4nv


FP8E4NV_KERNELS = pytest.mark.parametrize(
    "make, dtype",
    [(_make_to_fp8e4nv, torch.float32), (_make_load_fp8e4nv, torch.float8_e4m3fn)],
    ids=["cast", "tensor"],
)


@FP8E4NV_KERNELS
def test_fp8e4nv_compiles_under_the_default_target(make, dtype):
    """The default target, cuda:89, is the first with fp8e4nv,
    so such kernels are checked without naming a target."""
    x, out = torch.zeros(64).to(dtype), torch.zeros(64)
    det = _sanitizer()
    kernel = tilelens.trace(det)(make())[(4,)](x, out, 64, BLOCK=16)
    assert kernel.target == GPUTarget("cuda", 89, 32)
    assert det.last_status == "ok", det.last_verdict


@FP8E4NV_KERNELS
def test_an_explicit_cuda80_refuses_fp8e4nv_as_unsupported(make, dtype):
    """cuda:80 has no fp8e4nv: the launch fails to compile there (whatever
    this machine's GPU), which is unsupported compile-failed naming cuda:80
    and how to name another target, never an exception."""
    x, out = torch.zeros(64).to(dtype), torch.zeros(64)
    det = Sanitizer(compile=True, target="cuda:80")  # abort_on_error=True
    assert tilelens.trace(det)(make())[(4,)](x, out, 64, BLOCK=16) is None
    verdict = det.last_verdict
    assert verdict.status == "unsupported" and verdict.notes == ()
    (failed,) = verdict.per_config
    assert verdict.refusal == failed.refusal
    _assert_compile_failed(failed.refusal, "cuda:80", "fp8e4nv not supported")


def _device_is_zero():
    # A host function a kernel's constexpr function calls (marked like
    # tl.target_info.current_target), asking Triton's driver for a device.
    return triton.runtime.driver.active.get_current_device() == 0


_device_is_zero.__triton_builtin__ = True  # type: ignore[attr-defined]


def test_a_host_compile_that_cannot_run_is_unsupported_not_raised():
    """A compile that asks for a device cannot run on the host: no error of
    the kernel's (a GPU would compile it), so the launch is unsupported and
    goes on, never failed."""
    from triton.runtime.jit import constexpr_function

    @constexpr_function
    def on_device_zero():
        return _device_is_zero()

    @triton.jit
    def device_dependent(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if on_device_zero():
            tl.store(x_ptr + offs, 1.0, mask=offs < n)

    det = _sanitizer()
    tilelens.trace(det)(device_dependent)[(4,)](torch.zeros(64), 64, BLOCK=16)
    verdict = det.last_verdict
    assert (verdict.status, verdict.refusal.kind) == (
        "unsupported",
        "host-compile-unavailable",
    )
    assert "'get_current_device'" in verdict.refusal.message
    assert verdict.notes == ()


def test_another_triton_release_is_unsupported_and_the_program_goes_on(monkeypatch):
    """IR mode runs on Triton 3.8 only: on any other release the host
    compile is unavailable, so every launch is unsupported, nothing is
    compiled or run, and the program goes on. A call that does not bind is
    one more such launch: the JIT's binder is never reached."""
    from tilelens.core.host_compile import triton_api

    triton_api.cache_clear()  # a refusal is not cached
    monkeypatch.setattr(triton, "__version__", "3.6.0")
    det = _sanitizer()
    traced = tilelens.trace(det)(_make_add())
    x, y, out = torch.ones(64), torch.ones(64), torch.zeros(64)

    assert traced[(4,)](x, y, out, 64, BLOCK=16) is None

    verdict = det.last_verdict
    assert (verdict.status, verdict.refusal.kind) == (
        "unsupported",
        "host-compile-unavailable",
    )
    assert verdict.refusal.message.endswith(
        "IR mode supports Triton 3.8.x only; the installed Triton is 3.6.0"
    )
    assert det.records == [] and torch.equal(out, torch.zeros(64))
    assert traced[(4,)](x, y, out, BLOCK=16) is None  # n is missing
    assert det.last_verdict.refusal.kind == "host-compile-unavailable"


@pytest.fixture
def configured_target(monkeypatch):
    """Set TILELENS_IR_TARGET and reload the process config from it."""

    def set_target(value):
        monkeypatch.setenv("TILELENS_IR_TARGET", value)
        monkeypatch.setattr(config_module, "config", Config())

    return set_target


def test_the_environment_sets_the_default_target(configured_target):
    configured_target("cuda:90")
    x, out = torch.zeros(64), torch.zeros(64)
    det = _sanitizer()
    tilelens.trace(det)(_make_two_cta_configs())[(4,)](x, out, 64)
    assert [c.config["num_ctas"] for c in det.last_verdict.per_config] == [1, 2]

    # A target the client names wins over the environment's.
    det = Sanitizer(compile=True, abort_on_error=False, target="cuda:80")
    tilelens.trace(det)(_make_two_cta_configs())[(4,)](x, out, 64)
    assert det.last_status == "unsupported"
    _assert_compile_failed(det.last_verdict.refusal, "cuda:80", "num_ctas > 1")


def test_a_spec_that_names_no_target_is_refused_up_front():
    with pytest.raises(ValueError, match="invalid IR target 'sm90'"):
        Sanitizer(compile=True, target="sm90")


def test_a_configured_spec_that_names_no_target_fails_the_launch(configured_target):
    configured_target("gfx942")
    det = _sanitizer()
    x, out = torch.zeros(64), torch.zeros(64)
    with pytest.raises(ValueError, match=r"TILELENS_IR_TARGET\) is 'gfx942'"):
        tilelens.trace(det)(_make_add_nomask())[(1,)](x, out, 64, BLOCK=64)
    assert det.last_verdict is None


_TARGET_SCRIPT = """\
import torch, triton, triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({{"BLOCK": 16}}, num_ctas=1),
        triton.Config({{"BLOCK": 16}}, num_ctas=2),
    ],
    key=["n"],
)
@triton.jit
def copy_ctas(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # {tag}
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs))

x, out = torch.zeros(64), torch.zeros(64)
copy_ctas[(4,)](x, out, 64)
print("launch returned")
"""


@pytest.mark.parametrize(
    "target, returncode, output",
    [
        ("cuda:90", 0, "launch returned"),
        ("sm90", 1, "TILELENS_IR_TARGET) is 'sm90'"),
    ],
)
def test_the_cli_reads_the_target_from_the_environment(
    tmp_path, target, returncode, output
):
    script = tmp_path / "ctas.py"
    script.write_text(_TARGET_SCRIPT.format(tag=script))
    cli = (
        f"import sys; sys.argv = ['tile-sanitizer', '--compile', {str(script)!r}]; "
        "from tilelens.wrapper import apply_sanitizer; apply_sanitizer()"
    )
    proc = _run(script, "-c", cli, TILELENS_IR_TARGET=target)
    assert proc.returncode == returncode, proc.stderr
    assert output in proc.stdout + proc.stderr
    assert ("launch returned" in proc.stdout) == (returncode == 0)


_UNCOMPILABLE_SCRIPT = """\
import torch, triton, triton.language as tl

@triton.jit
def copy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # {tag}
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)

@triton.autotune(configs=[triton.Config({{"BLOCK": 64}})], key=["n"])
@triton.jit
def bounded(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    tl.static_assert(BLOCK <= 32)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)

x, out = torch.zeros(64), torch.zeros(64)
copy[(1,)](x, out, 64, BLOCK=64, num_ctas=2)
print("first launch returned")
bounded[(1,)](x, out, 64)
print("second launch returned")
"""


def test_the_cli_goes_on_past_kernels_that_fail_to_compile(tmp_path):
    """tile-sanitizer --compile on kernels no config of which compiles for
    the target (num_ctas=2 below sm90; a failing tl.static_assert): no
    finding, so exit status 0, each launch reported as not checked, in one
    line naming where the kernel failed. A launch whose configs all failed
    as notes prints the notes too: they say why."""
    script = tmp_path / "uncompilable.py"
    script.write_text(_UNCOMPILABLE_SCRIPT.format(tag=script))
    cli = (
        f"import sys; sys.argv = ['tile-sanitizer', '--compile', {str(script)!r}]; "
        "from tilelens.wrapper import apply_sanitizer; apply_sanitizer()"
    )
    proc = _run(script, "-c", cli)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert "first launch returned" in lines and "second launch returned" in lines
    source = script.read_text().splitlines()

    def at(needle):
        (line,) = [i for i, text in enumerate(source, 1) if needle in text]
        return f"{script}:{line}: "

    printed = [line for line in lines if line.startswith("[CompiledSanitizer]")]
    first, *unchecked = printed
    assert first.startswith(
        "[CompiledSanitizer] not checked: compile-failed: "
        f"{at('def copy(')}it failed to compile for cuda:89 (ValueError: num_ctas "
        "> 1 requires NVIDIA SM90"
    )
    assert unchecked == [
        "[CompiledSanitizer] not checked: compile-failed: "
        f"{at('def bounded(')}no config of the launch compiled for cuda:89, so "
        "nothing was checked: each failed with an error of its own code whatever "
        "the target (see the notes)",
        "[CompiledSanitizer] note: config {'BLOCK': 64, 'num_warps': 4, "
        "'num_ctas': 1, 'num_stages': 3} was not checked: "
        f"{at('tl.static_assert(')}it failed to compile for cuda:89 "
        "(CompileTimeAssertionFailure), an error of its own code whatever the "
        "target, so it never launches",
    ]
    assert "Traceback" not in proc.stderr
