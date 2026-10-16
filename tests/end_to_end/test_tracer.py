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
    if hasattr(traced_kernel, "runner") and hasattr(traced_kernel.runner, "do_bench"):
        bench_fn = traced_kernel.runner.do_bench
        assert (
            bench_fn is not None and bench_fn.__name__ == "dummy_benchmarker"
        ), f"Expected dummy_benchmarker, got: {bench_fn}"
