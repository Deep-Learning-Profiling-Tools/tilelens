import numpy as np
import pytest
from pathlib import Path

import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.data import (
    Dot,
    Grid,
    Load,
    NkiCompute,
    ReduceSum,
    Store,
    TensorTranspose,
)
from triton_viz.core.trace import launches
import math

try:
    import neuronxcc.nki.language as nl
    from triton_viz.core.simulation.nki import NDArray
    from triton_viz.tools.nki_trace_dump import records_to_events, summarize_events
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install triton-viz[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki  # only run at "pytest -m nki"


def div_ceil(n, d):
    return (n + d - 1) // d


def test_tracer_records_masked_load_store():
    triton_viz.clear()

    @triton_viz.trace(client=Tracer(), frontend="nki")
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
    assert [r.bytes for r in load_records] == [16, 16, 8, 8]
    assert [r.bytes for r in store_records] == [16, 8]


def test_nl_storage_identity_builds_versioned_load_compute_store_chain():
    triton_viz.clear()

    @triton_viz.trace(client=Tracer(), frontend="nki")
    def kernel(x_ptr, out_ptr):
        offs = nl.arange(8)
        value = nl.load(x_ptr[offs])
        view = nl.broadcast_to(value[0:1], shape=(8,))
        result = nl.add(value, view, dtype=x_ptr.dtype)
        nl.store(out_ptr[offs], result)

    x = NDArray(value=np.arange(8, dtype=np.float32))
    out = NDArray(value=np.empty_like(x.data))
    kernel[(1,)](x, out)

    records = launches[-1].records
    load = next(record for record in records if isinstance(record, Load))
    computes = [record for record in records if isinstance(record, NkiCompute)]
    broadcast = next(record for record in computes if record.api_op == "broadcast_to")
    compute = next(record for record in computes if record.api_op == "add")
    store = next(record for record in records if isinstance(record, Store))

    assert load.dst_storage == broadcast.input_storages[0]
    assert load.dst_version == broadcast.input_versions[0] == 0
    assert broadcast.output_storages[0] == compute.input_storages[1]
    assert compute.input_storages[0] == load.dst_storage
    assert compute.input_ranges[1][1] <= load.dst_range[1]
    assert compute.output_storages[0] == store.src_storage
    assert compute.output_versions[0] == store.src_version == 0

    from triton_viz.tools.nki_cost_model import CostModel, simulate

    result = simulate(records_to_events(records), CostModel(cross_engine_sync_ns=0))
    load_timing = result.timeline["dma"][0]
    compute_timing = result.timeline["vector"][0]
    store_timing = result.timeline["dma"][1]
    assert compute_timing.start >= load_timing.end
    assert store_timing.start >= compute_timing.end


def test_compute_mask_is_preserved_in_source_event():
    triton_viz.clear()

    @triton_viz.trace(client=Tracer(), frontend="nki")
    def kernel(x_ptr, out_ptr):
        offs = nl.arange(8)
        mask = offs < 6
        value = nl.load(x_ptr[offs], mask=mask)
        result = nl.maximum(value, 0.0, mask=mask)
        nl.store(out_ptr[offs], result, mask=mask)

    x = NDArray(value=np.arange(8, dtype=np.float32))
    out = NDArray(value=np.empty_like(x.data))
    kernel[(1,)](x, out)

    compute = next(
        record for record in launches[-1].records if isinstance(record, NkiCompute)
    )
    event = next(
        event for event in records_to_events(launches[-1].records)
        if event.get("op") == "compute"
    )
    assert compute.attrs["compute_mask_provided"] is True
    assert event["compute_mask_provided"] is True


def test_inplace_assignment_retargets_compute_to_new_tensor_version():
    triton_viz.clear()

    @triton_viz.trace(client=Tracer(), frontend="nki")
    def kernel(x_ptr, out_ptr):
        offs = nl.arange(8)
        acc = nl.load(x_ptr[offs])
        acc[...] = nl.add(acc, 1.0, dtype=x_ptr.dtype)
        result = nl.multiply(acc, 2.0, dtype=x_ptr.dtype)
        nl.store(out_ptr[offs], result)

    x = NDArray(value=np.arange(8, dtype=np.float32))
    out = NDArray(value=np.empty_like(x.data))
    kernel[(1,)](x, out)
    computes = [r for r in launches[-1].records if isinstance(r, NkiCompute)]

    assert len(computes) == 2
    writer, consumer = computes
    assert writer.output_storages[0] == consumer.input_storages[0]
    assert writer.output_versions[0] == consumer.input_versions[0] == 1


def copy_kernel(x_ptr, out_ptr):
    block_size = 4
    pid = nl.program_id(axis=0)
    offs = pid * block_size + nl.arange(block_size)
    mask = offs < x_ptr.shape[0]
    x = nl.load(x_ptr[offs], mask=mask)
    nl.store(out_ptr[offs], x, mask=mask)


def test_tracer_records_masked_bytes_for_float16():
    triton_viz.clear()

    traced = triton_viz.trace(client=Tracer(), frontend="nki")(copy_kernel)
    x = NDArray(value=np.arange(6, dtype=np.float16))
    out = NDArray(value=np.empty_like(x.data))

    traced[(2,)](x, out)

    loads = [r for r in launches[-1].records if isinstance(r, Load)]
    stores = [r for r in launches[-1].records if isinstance(r, Store)]
    assert [r.bytes for r in loads] == [8, 4]
    assert [r.bytes for r in stores] == [8, 4]

    events = records_to_events(launches[-1].records)
    load_events = [event for event in events if event["op"] == "load"]
    summary = summarize_events(events)
    assert [event["active_lanes"] for event in load_events] == [4, 2]
    assert [event["bytes"] for event in load_events] == [8, 4]
    assert summary["bytes_by_edge"] == {
        "HBM->SBUF": 12,
        "SBUF->HBM": 12,
    }


def test_tracer_grid_idx_sampling():
    triton_viz.clear()

    traced = triton_viz.trace(client=Tracer(grid_idx=1), frontend="nki")(copy_kernel)

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
    triton_viz.clear()

    @triton_viz.trace(client=Tracer(), frontend="nki")
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
    assert record.input_dtypes == ("float32",)
    assert record.output_dtype == "float32"

    event = next(
        event
        for event in records_to_events(records)
        if event.get("op") == "reduce_sum"
    )
    assert event["input_dtypes"] == ["float32"]
    assert event["output_dtype"] == "float32"


def _silu_kernel(x_ptr):
    nl.silu(nl.load(x_ptr[nl.arange(8)]))


def _gelu_kernel(x_ptr):
    nl.gelu(nl.load(x_ptr[nl.arange(8)]))


@pytest.mark.parametrize(
    ("api", "kernel"), [("silu", _silu_kernel), ("gelu", _gelu_kernel)]
)
def test_tracer_records_all_registered_activation_compute_apis(api, kernel):
    triton_viz.clear()
    x = NDArray(value=np.arange(8, dtype=np.float32))
    traced = triton_viz.trace(client=Tracer(), frontend="nki")(kernel)
    traced[(1,)](x)

    computes = [
        record for record in launches[-1].records if isinstance(record, NkiCompute)
    ]
    assert [record.api_op for record in computes] == [api]
    assert computes[0].input_storages
    assert computes[0].output_storages


def test_legacy_nc_transpose_records_tensor_event_and_storage_identity():
    triton_viz.clear()

    def transpose_kernel(x_ptr):
        value = nl.load(x_ptr[nl.arange(4)[:, None], nl.arange(8)[None, :]])
        import neuronxcc.nki.isa as nisa_local

        nisa_local.nc_transpose(value)

    x = NDArray(value=np.arange(32, dtype=np.float32).reshape(4, 8))
    traced = triton_viz.trace(client=Tracer(), frontend="nki")(transpose_kernel)
    traced[(1,)](x)
    records = launches[-1].records
    load = next(record for record in records if isinstance(record, Load))
    transpose = next(
        record for record in records if isinstance(record, TensorTranspose)
    )
    assert transpose.input_storages == (load.dst_storage,)
    assert transpose.input_versions == (load.dst_version,)
    assert transpose.output_shape == (8, 4)


def test_tracer_records_dot():
    triton_viz.clear()

    TILE_M = 2
    TILE_K = 2
    TILE_N = 4

    @triton_viz.trace(client=Tracer(), frontend="nki")
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
    traced_kernel = triton_viz.trace(client=Tracer(), frontend="nki")(matmul_kernel)
    kernel_instance = traced_kernel[kernel_grid]
    kernel_instance(*kernel_args)

    records = launches[-1].records
    dot_records = [r for r in records if isinstance(r, Dot)]

    assert len(dot_records) == 8
    for record in dot_records:
        assert record.input_shape == (TILE_M, TILE_K)
        assert record.other_shape == (TILE_K, TILE_N)
        assert record.output_shape == (TILE_M, TILE_N)


def test_tracer_records_dot_transpose_x_kwarg():
    triton_viz.clear()

    @triton_viz.trace(client=Tracer(), frontend="nki")
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
    triton_viz.clear()

    traced = triton_viz.trace(client=Tracer(), frontend="nki")(copy_kernel)

    n_elements = 6
    x = NDArray(value=np.arange(n_elements, dtype=np.float32))
    out = NDArray(value=np.empty_like(x.data))

    traced[(div_ceil(n_elements, 4),)](x, out)

    path = tmp_path / "nki_trace.tvz"
    triton_viz.save(path)
    triton_viz.clear()
    triton_viz.load(path)

    records = launches[-1].records
    record_types = [type(r) for r in records]

    assert record_types == [Grid, Load, Store] * div_ceil(n_elements, 4)


def test_opt_in_compute_mask_geometry_preserves_declared_tile_shape():
    def run(capture):
        triton_viz.clear()
        @triton_viz.trace(client=Tracer(compute_masks=capture), frontend="nki")
        def kernel(x_ptr, out_ptr):
            offs = nl.arange(8)
            mask = offs < 6
            value = nl.load(x_ptr[offs], mask=mask)
            result = nl.maximum(value, 0.0, mask=mask)
            nl.store(out_ptr[offs], result, mask=mask)
        kernel[(1,)](NDArray(value=np.arange(8, dtype=np.float32)), NDArray(value=np.zeros(8, dtype=np.float32)))
        return next(e for e in records_to_events(launches[-1].records) if e.get('op')=='compute')
    default, captured = run(False), run(True)
    geometry = captured.pop('compute_mask_geometry')
    # Runtime storage addresses differ between runs; shape/mask facts do not.
    assert captured['input_shape'] == default['input_shape'] == [8]
    assert captured['output_shape'] == default['output_shape'] == [8]
    assert 'compute_mask_geometry' not in default
    assert geometry == dict(status='known',shape=[8],active_elements=6,bounds=[[0,6]],rectangular=True)
