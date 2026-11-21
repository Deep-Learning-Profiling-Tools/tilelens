import numpy as np
import pytest
from pathlib import Path

import tilelens
from tilelens.clients import Tracer
from tilelens.core.data import Grid, Load, Store, ReduceSum, Dot
from tilelens.core.trace import launches
import math

try:
    import neuronxcc.nki.language as nl
    from tilelens.core.simulation.nki import NDArray
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install tilelens[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki  # only run at "pytest -m nki"


def div_ceil(n, d):
    return (n + d - 1) // d


def test_tracer_records_masked_load_store():
    tilelens.clear()

    @tilelens.trace(client=Tracer(), frontend="nki")
    def add_kernel(x_ptr, y_ptr, out_ptr):
        block_size = 4
        pid = nl.program_id(axis=0)
        offs = pid * block_size + nl.arange(block_size)
        mask = offs < x_ptr.shape[0]
        x = nl.load(x_ptr[offs], mask=mask)
        y = nl.load(y_ptr[offs], mask=mask)
        nl.store(out_ptr[offs], x + y, mask=mask)

    n_elements = 6
    x = NDArray(value=np.arange(n_elements, dtype=np.float32))
    y = NDArray(value=np.arange(n_elements, dtype=np.float32))
    out = NDArray(value=np.empty_like(x.data))

    grid = (div_ceil(n_elements, 4),)
    add_kernel[grid](x, y, out)

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


def copy_kernel(x_ptr, out_ptr):
    block_size = 4
    pid = nl.program_id(axis=0)
    offs = pid * block_size + nl.arange(block_size)
    mask = offs < x_ptr.shape[0]
    x = nl.load(x_ptr[offs], mask=mask)
    nl.store(out_ptr[offs], x, mask=mask)


def test_tracer_grid_idx_sampling():
    tilelens.clear()

    traced = tilelens.trace(client=Tracer(grid_idx=1), frontend="nki")(copy_kernel)

    n_elements = 12
    x = NDArray(value=np.arange(n_elements, dtype=np.float32))
    out = NDArray(value=np.empty_like(x.data))

    grid = (div_ceil(n_elements, 4),)
    traced[grid](x, out)

    records = launches[-1].records

    record_types = [type(r) for r in records]

    # first and third blocks skipped upon seeing Grid record hence [Grid]
    assert record_types == [Grid] + [Grid, Load, Store] + [Grid]

    grid_records = [r for r in records if isinstance(r, Grid)]
    assert all([r.idx == (grid_idx, 0, 0) for grid_idx, r in enumerate(grid_records)])


def test_tracer_records_reduce_sum():
    tilelens.clear()

    @tilelens.trace(client=Tracer(), frontend="nki")
    def reduce_sum_kernel(x_ptr, out_ptr):
        block_m = 4
        block_n = 8
        pid = nl.program_id(axis=0)
        offs_m = pid * block_m + nl.arange(block_m)
        offs_n = nl.arange(block_n)
        mask = (offs_m[:, None] < x_ptr.shape[0]) & (offs_n[None, :] < x_ptr.shape[1])
        x = nl.load(x_ptr[offs_m[:, None], offs_n[None, :]], mask=mask)
        s = nl.sum(x, axis=1)
        out_mask = offs_m < out_ptr.shape[0]
        nl.store(out_ptr[offs_m], s, mask=out_mask)

    block_m = 4
    block_n = 8
    x = NDArray(
        value=np.arange(block_m * block_n, dtype=np.float32).reshape(block_m, block_n)
    )
    out = NDArray(value=np.empty(block_m, dtype=np.float32))

    grid = (1,)
    reduce_sum_kernel[grid](x, out)

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

    TILE_M = 2
    TILE_K = 2
    TILE_N = 4

    @tilelens.trace(client=Tracer(), frontend="nki")
    def matmul_kernel(lhs, rhs, result):
        """NKI matmul_kernel to compute a matrix multiplication operation in a tiled manner

        Args:
            lhs: an input tensor of shape [K,M], where both K and M are multiples for
            128.  It is the left-hand-side argument of the matrix multiplication,
            delivered transposed for optimal performance.
            rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
            is a multiple of 512.  It is the right-hand-side argument of the
            matrix multiplication.
        Returns:
            result: the resulting output tensor of shape [M,N]
        """
        TILE_M = 2
        TILE_K = 2
        TILE_N = 4

        M, K = lhs.shape
        K_, N = rhs.shape
        assert K == K_, "lhs and rhs must have the same contraction dimension"

        # Use affine_range to loop over tiles
        for m in nl.affine_range(math.ceil(M / TILE_M)):
            for n in nl.affine_range(math.ceil(N / TILE_N)):
                # Allocate a tensor in PSUM
                res_psum = nl.zeros((TILE_M, TILE_N), nl.int32, buffer=nl.psum)

                for k in nl.affine_range(math.ceil(K / TILE_K)):
                    # Declare the tiles on SBUF
                    lhs_tile = nl.ndarray(
                        (TILE_K, TILE_M), dtype=lhs.dtype, buffer=nl.sbuf
                    )
                    rhs_tile = nl.ndarray(
                        (TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf
                    )

                    # Load tiles from lhs and rhs
                    lhs_p = nl.arange(TILE_M)[:, None] + m * TILE_M
                    lhs_f = nl.arange(TILE_K)[None, :] + k * TILE_K
                    lhs_mask = (lhs_p < M) & (lhs_f < K)
                    lhs_tile = nl.load(lhs[lhs_p, lhs_f], mask=lhs_mask)

                    rhs_p = nl.arange(TILE_K)[:, None] + k * TILE_K
                    rhs_f = nl.arange(TILE_N)[None, :] + n * TILE_N
                    rhs_mask = (rhs_p < K) & (rhs_f < N)
                    rhs_tile = nl.load(rhs[rhs_p, rhs_f], mask=rhs_mask)

                    # Accumulate partial-sums into PSUM
                    x = nl.matmul(lhs_tile[...], rhs_tile[...], transpose_x=False)
                    res_psum += x

                # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                res_sb = nl.copy(res_psum, dtype=result.dtype)

                out_p = nl.arange(TILE_M)[:, None] + m * TILE_M
                out_f = nl.arange(TILE_N)[None, :] + n * TILE_N
                out_mask = (out_p < M) & (out_f < N)
                nl.store(
                    result[
                        m * TILE_M : (m + 1) * TILE_M, n * TILE_N : (n + 1) * TILE_N
                    ],
                    value=res_sb,
                    mask=out_mask,
                )

    kernel_grid = (1, 1, 1)
    lhs_small = np.arange(16).astype(np.float32).reshape(4, 4)
    rhs_small = np.arange(32).astype(np.float32).reshape(4, 8)
    # lhs_small = np.arange(9).astype(np.float32).reshape(3, 3)
    # rhs_small = np.arange(18).astype(np.float32).reshape(3, 6)
    result = np.empty((lhs_small.shape[0], rhs_small.shape[1]), dtype=lhs_small.dtype)
    kernel_args = (lhs_small, rhs_small, result)

    print("Executing matmul_kernel with NKI interpreter...")
    kernel_instance = matmul_kernel[kernel_grid]
    kernel_instance(*kernel_args)

    records = launches[-1].records
    dot_records = [r for r in records if isinstance(r, Dot)]

    assert len(dot_records) == 8
    for record in dot_records:
        assert record.input_shape == (TILE_M, TILE_K)
        assert record.other_shape == (TILE_K, TILE_N)
        assert record.output_shape == (TILE_M, TILE_N)


def test_tracer_records_dot_transpose_x_kwarg():
    tilelens.clear()

    @tilelens.trace(client=Tracer(), frontend="nki")
    def dot_kernel(lhs, rhs, out):
        out[...] = nl.matmul(lhs, rhs, transpose_x=True)

    lhs = NDArray(value=np.arange(6, dtype=np.float32).reshape(2, 3))
    rhs = NDArray(value=np.arange(8, dtype=np.float32).reshape(2, 4))
    out = NDArray(value=np.empty((3, 4), dtype=np.float32))

    dot_kernel[(1,)](lhs, rhs, out)

    assert np.allclose(out.data, lhs.data.T @ rhs.data)

    dot_records = [r for r in launches[-1].records if isinstance(r, Dot)]
    assert len(dot_records) == 1
    record = dot_records[0]
    assert record.input_shape == (3, 2)
    assert record.other_shape == (2, 4)
    assert record.output_shape == (3, 4)


def test_nki_trace_save_load_roundtrip(tmp_path: Path):
    """NKI traces should serialize and reload through the shared .tvz path."""
    tilelens.clear()

    traced = tilelens.trace(client=Tracer(), frontend="nki")(copy_kernel)

    n_elements = 6
    x = NDArray(value=np.arange(n_elements, dtype=np.float32))
    out = NDArray(value=np.empty_like(x.data))

    traced[(div_ceil(n_elements, 4),)](x, out)

    path = tmp_path / "nki_trace.tvz"
    tilelens.save(path)
    tilelens.clear()
    tilelens.load(path)

    records = launches[-1].records
    record_types = [type(r) for r in records]

    assert record_types == [Grid, Load, Store] * div_ceil(n_elements, 4)
