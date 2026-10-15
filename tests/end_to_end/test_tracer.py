import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import tilelens
from tilelens.clients import Tracer, Sanitizer
from tilelens.core.data import Grid, Load, Store, ReduceSum, Dot
from tilelens.core.trace import launches


def test_tracer_records_masked_load_store():
    tilelens.clear()

    @tilelens.trace(client=Tracer())
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        y = tl.load(y_ptr + offs, mask=mask, other=0)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n_elements = 6
    block_size = 4
    x = torch.arange(n_elements, dtype=torch.float32)
    y = torch.arange(n_elements, dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)

    records = launches[-1].records

    record_types = [type(r) for r in records]
    assert record_types == [Grid, Load, Load, Store] * grid[0]

    load_records = [r for r in records if isinstance(r, Load)]
    store_records = [r for r in records if isinstance(r, Store)]
    all_records = load_records + store_records
    input_ptrs = {x.data_ptr(), y.data_ptr()}

    assert any(not r.masks.all() for r in all_records)
    assert all(r.offsets.shape == r.masks.shape for r in all_records)
    assert all(r.ptr in input_ptrs for r in load_records)
    assert all(r.ptr == out.data_ptr() for r in store_records)


@triton.jit
def copy_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x)


def test_tracer_grid_idx_sampling():
    tilelens.clear()

    traced = tilelens.trace(client=Tracer(grid_idx=1))(copy_kernel)

    block_size = 4
    n_elements = 12
    x = torch.arange(n_elements, dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, block_size),)
    traced[grid](x, out, BLOCK_SIZE=block_size)

    records = launches[-1].records

    record_types = [type(r) for r in records]

    # first and third blocks skipped upon seeing Grid record hence [Grid]
    assert record_types == [Grid] + [Grid, Load, Store] + [Grid]

    grid_records = [r for r in records if isinstance(r, Grid)]
    assert all([r.idx == (grid_idx, 0, 0) for grid_idx, r in enumerate(grid_records)])


def test_tracer_records_reduce_sum():
    tilelens.clear()

    @tilelens.trace(client=Tracer())
    @triton.jit
    def reduce_sum_kernel(
        x_ptr,
        out_ptr,
        stride_xm,
        stride_xn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        x = tl.load(x_ptrs)
        s = tl.sum(x, axis=1)
        tl.store(out_ptr + offs_m, s)

    block_m = 4
    block_n = 8
    x = torch.arange(block_m * block_n, dtype=torch.float32).reshape(block_m, block_n)
    out = torch.empty(block_m, dtype=torch.float32)

    grid = (1,)
    reduce_sum_kernel[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )

    records = launches[-1].records
    reduce_records = [r for r in records if isinstance(r, ReduceSum)]

    assert len(reduce_records) == 1
    record = reduce_records[0]
    assert record.input_shape == (block_m, block_n)
    assert record.index == 1
    assert record.keep_dims is False
    assert record.output_shape == (block_m,)


def test_tracer_records_dot():
    tilelens.clear()

    @tilelens.trace(client=Tracer())
    @triton.jit
    def dot_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_M)[:, None]
        offs_n = tl.arange(0, BLOCK_N)[None, :]
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n * stride_bn
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b)
        c_ptrs = out_ptr + offs_m * stride_cm + offs_n * stride_cn
        tl.store(c_ptrs, c)

    block_m = 2
    block_n = 2
    block_k = 4
    a = torch.arange(block_m * block_k, dtype=torch.float16).reshape(block_m, block_k)
    b = torch.arange(block_k * block_n, dtype=torch.float16).reshape(block_k, block_n)
    out = torch.empty((block_m, block_n), dtype=torch.float16)

    grid = (1,)
    dot_kernel[grid](
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    records = launches[-1].records
    dot_records = [r for r in records if isinstance(r, Dot)]

    assert len(dot_records) == 1
    record = dot_records[0]
    assert record.input_shape == (block_m, block_k)
    assert record.other_shape == (block_k, block_n)
    assert record.output_shape == (block_m, block_n)


def test_kernel_cache_autotune_with_dummy_benchmarker():
    """
    Test that autotuned kernels install dummy_benchmarker.
    """

    # Create a fresh autotuned kernel inside the test to avoid state corruption
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def autotune_add_kernel_cache_on(
        x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    traced_kernel = tilelens.trace(client=Sanitizer())(autotune_add_kernel_cache_on)

    # Verify dummy benchmarker is installed
    if hasattr(traced_kernel, "runner") and hasattr(traced_kernel.runner, "_do_bench"):
        bench_fn = traced_kernel.runner._do_bench
        assert (
            bench_fn is not None and bench_fn.__name__ == "dummy_benchmarker"
        ), f"Expected dummy_benchmarker, got: {bench_fn}"


def _fenced(values, dtype=np.float32, pad=64):
    """
    Return a tensor whose storage holds exactly `values`, plus the sentinel
    memory on both sides of it. Out-of-bounds writes would land in the
    sentinels instead of the heap, so tests can check them deterministically.
    """
    buf = np.full(len(values) + 2 * pad, -7, dtype=dtype)
    buf[pad:-pad] = values
    return torch.from_numpy(buf[pad:-pad]), (buf[:pad], buf[-pad:])


def _untouched(fences):
    return all((fence == -7).all() for fence in fences)


@triton.jit
def unmasked_store_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0)
    tl.store(out_ptr + offs, x)


@pytest.mark.parametrize("grid_idx", [None, 0])
def test_tracer_refuses_out_of_bounds_load(grid_idx):
    traced = tilelens.trace(client=Tracer(grid_idx=grid_idx))(copy_kernel)
    x, x_fences = _fenced(np.arange(10))
    out, out_fences = _fenced(np.zeros(10))

    with pytest.raises(IndexError, match=r"(?s)out-of-bounds load .*`x_ptr`"):
        traced[(3,)](x, out, BLOCK_SIZE=4)
    assert _untouched(x_fences) and _untouched(out_fences)


@pytest.mark.parametrize("num_sms", [1, 4])
def test_tracer_refuses_out_of_bounds_store(monkeypatch, num_sms):
    monkeypatch.setattr(tilelens.config, "num_sms", num_sms)
    traced = tilelens.trace(client=Tracer())(unmasked_store_kernel)
    x = torch.arange(10, dtype=torch.float32)
    out, fences = _fenced(np.zeros(10))

    with pytest.raises(IndexError, match=r"(?s)out-of-bounds store .*`out_ptr`"):
        traced[(3,)](x, out, x.numel(), BLOCK_SIZE=4)
    assert _untouched(fences)


@triton.jit
def unmasked_atomic_add_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.atomic_add(out_ptr + offs, 1)


@triton.jit
def unmasked_atomic_cas_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.atomic_cas(out_ptr + offs, tl.zeros_like(offs), tl.full(offs.shape, 1, tl.int32))


@pytest.mark.parametrize(
    "kernel, op_name",
    [
        (unmasked_atomic_add_kernel, "atomic"),
        (unmasked_atomic_cas_kernel, "atomic_cas"),
    ],
)
def test_tracer_refuses_out_of_bounds_atomic(kernel, op_name):
    traced = tilelens.trace(client=Tracer())(kernel)
    out, fences = _fenced(np.zeros(10), dtype=np.int32)

    with pytest.raises(IndexError, match=f"out-of-bounds {op_name} "):
        traced[(3,)](out, BLOCK_SIZE=4)
    assert _untouched(fences)


@triton.jit
def unmasked_atomic_max_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs, mask=offs < n_elements, other=-1.0)
    tl.atomic_max(out_ptr + offs, x)


def test_tracer_refuses_sign_split_float_atomic_max():
    # Triton issues float atomic_max as two calls masked by sign; here the call
    # for the negative lanes holds only the out-of-bounds ones
    traced = tilelens.trace(client=Tracer())(unmasked_atomic_max_kernel)
    x = torch.arange(1, 9, dtype=torch.float32)
    out, fences = _fenced(np.zeros(8))

    with pytest.raises(IndexError, match="out-of-bounds atomic "):
        traced[(1,)](x, out, x.numel(), BLOCK_SIZE=16)
    assert _untouched(fences)


@triton.jit
def packed_word_store_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    words = x_ptr.to(tl.pointer_type(tl.int32))
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(words + offs, 1)


def test_tracer_refuses_word_overlapping_storage_end():
    # x holds 10 bytes, so the int32 word at byte 8 overlaps its end
    traced = tilelens.trace(client=Tracer())(packed_word_store_kernel)
    x, fences = _fenced(np.zeros(10), dtype=np.int8)

    with pytest.raises(IndexError, match=r"(?s)out-of-bounds store .*`x_ptr`"):
        traced[(2,)](x, BLOCK_SIZE=2)
    assert _untouched(fences)


def test_tracer_refuses_out_of_bounds_store_through_reinterpret():
    traced = tilelens.trace(client=Tracer())(unmasked_store_kernel)
    x = torch.arange(10, dtype=torch.float16)
    out, fences = _fenced(np.zeros(10), dtype=np.int16)

    with pytest.raises(IndexError, match=r"(?s)out-of-bounds store .*`out_ptr`"):
        traced[(3,)](x, triton.reinterpret(out, tl.float16), x.numel(), BLOCK_SIZE=4)
    assert _untouched(fences)


@triton.jit
def tuple_store_kernel(x_ptr, ptrs, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(ptrs[1] + offs, tl.load(x_ptr + offs))


def test_tracer_refuses_out_of_bounds_store_through_tuple_arg():
    traced = tilelens.trace(client=Tracer())(tuple_store_kernel)
    x = torch.zeros(16)
    out, fences = _fenced(np.zeros(10))

    with pytest.raises(IndexError, match=r"(?s)out-of-bounds store .*`ptrs\[1\]`"):
        traced[(3,)](x, (x, out), BLOCK_SIZE=4)
    assert _untouched(fences)


@triton.jit
def shifted_copy_kernel(x_ptr, out_ptr, SHIFT: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs - SHIFT))


def test_tracer_allows_view_access_within_base_storage():
    base = torch.arange(16, dtype=torch.float32)
    out = torch.empty(8)
    traced = tilelens.trace(client=Tracer())(shifted_copy_kernel)

    # reads base[0:8] through a view that starts at base[4]
    traced[(1,)](base[4:8], out, SHIFT=4, BLOCK_SIZE=8)

    torch.testing.assert_close(out, base[:8])


@triton.jit
def pointer_table_kernel(table_ptr, arg_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    rows = tl.arange(0, 2)
    cols = tl.arange(0, BLOCK_SIZE)
    row_ptrs = tl.load(table_ptr + rows).to(tl.pointer_type(tl.float32))
    values = tl.load(row_ptrs[:, None] + cols[None, :])
    tl.store(out_ptr + rows[:, None] * BLOCK_SIZE + cols[None, :], values)


def test_tracer_allows_pointer_table_gather():
    # one row is a kernel arg and one is not, so a single gather touches known
    # and unknown storage; pointers built from integers are not judged
    arg_row = torch.arange(4, dtype=torch.float32)
    other_row = torch.arange(4, 8, dtype=torch.float32)
    table = torch.tensor([arg_row.data_ptr(), other_row.data_ptr()])
    out = torch.empty(8)
    traced = tilelens.trace(client=Tracer())(pointer_table_kernel)

    traced[(1,)](table, arg_row, out, BLOCK_SIZE=4)

    torch.testing.assert_close(out, torch.arange(8, dtype=torch.float32))
