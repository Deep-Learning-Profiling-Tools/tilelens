import json

import numpy as np
import pytest
import triton_viz

try:
    import nki.isa as nisa
    import nki.language as nl
    from triton_viz.clients import Tracer
    from triton_viz.core.trace import launches
    from triton_viz.tools.nki_trace_dump import records_to_events, summarize_events, write_jsonl
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install triton-viz[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki


def _kernel(lhs_t, rhs, out):
    lhs_tile = nl.ndarray((128, 128), dtype=lhs_t.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    res_psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    out_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
    nisa.dma_copy(lhs_tile, lhs_t)
    nisa.dma_copy(rhs_tile, rhs)
    nisa.nc_matmul(dst=res_psum, stationary=lhs_tile, moving=rhs_tile)
    nisa.tensor_copy(out_tile, res_psum)
    nisa.dma_copy(out, out_tile)


def _records():
    triton_viz.clear()
    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(_kernel)
    lhs_t = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    rhs = np.arange(128 * 512, dtype=np.float32).reshape(128, 512)
    out = np.empty((128, 512), dtype=np.float32)
    traced[(1,)](lhs_t, rhs, out)
    return launches[-1].records


def test_records_to_events_has_perf_model_fields():
    events = records_to_events(_records())
    summary = summarize_events(events)

    assert summary["op_counts"]["grid"] == 1
    assert summary["op_counts"]["transfer"] == 4
    assert summary["op_counts"]["dot"] == 1
    assert summary["bytes_by_edge"] == {
        "hbm->sbuf": (128 * 128 + 128 * 512) * 4,
        "psum->sbuf": 128 * 512 * 4,
        "sbuf->hbm": 128 * 512 * 4,
    }
    assert summary["flops"] == 2 * 128 * 512 * 128
    dot = next(event for event in events if event["op"] == "dot")
    assert dot["engine"] == "tensor"
    assert dot["input_shape"] == [128, 128]
    assert dot["other_shape"] == [128, 512]
    assert dot["fusion_signature"] == "dot"


def test_fusion_signature_groups_only_contiguous_compute_in_one_grid():
    from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature

    events = [
        {"op": "compute", "api_op": "subtract", "grid_idx": [0]},
        {"op": "compute", "api_op": "exp", "grid_idx": [0]},
        {"op": "reduce_sum", "grid_idx": [0]},
        {"op": "store", "grid_idx": [0]},
        {"op": "compute", "api_op": "divide", "grid_idx": [0]},
        {"op": "compute", "api_op": "add", "grid_idx": [1]},
    ]
    _annotate_fusion_signature(events)

    assert [event.get("fusion_signature") for event in events] == [
        "subtract_exp_reduce_sum",
        "subtract_exp_reduce_sum",
        "subtract_exp_reduce_sum",
        None,
        "divide",
        "add",
    ]
    assert [events[i]["fusion_group_index"] for i in range(3)] == [0, 1, 2]
    assert events[4]["fusion_group"] != events[5]["fusion_group"]


def test_write_jsonl_round_trip(tmp_path):
    out = tmp_path / "trace.jsonl"
    events = write_jsonl(_records(), out)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines == events
    assert lines[0]["op"] == "grid"
    assert any(line["op"] == "transfer" and line["engine"] == "dma" for line in lines)
    dma = next(line for line in lines if line["op"] == "transfer" and line["engine"] == "dma")
    assert dma["partition_count"] > 0
    assert dma["free_bytes_per_partition"] == dma["bytes"] // dma["partition_count"]


def test_tensor_add_trace_records_binary_and_par_dim_geometry():
    triton_viz.clear()

    def kernel(a, b, out):
        a_tile = nl.ndarray((2, nl.par_dim(8), 16), dtype=a.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((2, nl.par_dim(8), 16), dtype=b.dtype, buffer=nl.sbuf)
        c_tile = nl.ndarray((2, nl.par_dim(8), 16), dtype=out.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a)
        nisa.dma_copy(dst=b_tile, src=b)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)
        nisa.dma_copy(dst=out, src=c_tile)

    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    a = np.ones((2, 8, 16), dtype=np.float32)
    b = np.ones_like(a)
    out = np.empty_like(a)
    traced[(1,)](a, b, out, pre_trace=False)
    events = records_to_events(launches[-1].records)
    dma = next(event for event in events if event["op"] == "transfer")
    binary = next(event for event in events if event["op"] == "binary")
    assert dma["partition_axis"] == 1
    assert dma["partition_count"] == 8
    assert dma["free_bytes_per_partition"] == 2 * 16 * 4
    assert binary["binary_op"] == "add"
    assert binary["elements"] == 2 * 8 * 16
    assert len(binary["input_ptrs"]) == 2
    assert binary["output_ptr"] is not None


def test_square_dma_transpose_is_explicitly_marked():
    triton_viz.clear()

    def kernel(src, out):
        tile = nl.ndarray((nl.par_dim(8), 8), dtype=src.dtype, buffer=nl.sbuf)
        nisa.dma_transpose(dst=tile, src=src)
        nisa.dma_copy(dst=out, src=tile)

    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    src = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = np.empty_like(src)
    traced[(1,)](src, out, pre_trace=False)
    events = records_to_events(launches[-1].records)
    transfer = next(event for event in events if event.get("mem_src") == "hbm")
    assert transfer["src_offsets_shape"] == transfer["dst_offsets_shape"] == [8, 8]
    assert transfer["dma_pattern"] == "transpose"


def test_hbm_load_records_onchip_destination_stride_separately():
    def kernel(src, out):
        packed=nl.ndarray((128,32),dtype=src.dtype,buffer=nl.sbuf)
        dense=nl.ndarray((128,16),dtype=src.dtype,buffer=nl.sbuf)
        nisa.dma_copy(dst=packed[:,0:32:2],src=src)
        nisa.dma_copy(dst=dense,src=src)
        nisa.dma_copy(dst=out,src=dense)
    triton_viz.clear()
    src=np.arange(128*16,dtype=np.float32).reshape(128,16);out=np.zeros_like(src)
    triton_viz.trace(client=Tracer(),frontend='nki_beta2')(kernel)[(1,)](src,out)
    loads=[e for e in records_to_events(launches[-1].records)
           if e.get('mem_src')=='hbm' and e.get('mem_dst')=='sbuf']
    assert len(loads)==2 and np.array_equal(src,out)
    assert loads[0]['sbuf_destination_geometry']['dma_pattern']=='strided'
    assert loads[0]['sbuf_destination_geometry']['free_stride_items']==2
    assert loads[1]['sbuf_destination_geometry']['dma_pattern']=='contiguous'
    assert loads[1]['sbuf_destination_geometry']['free_stride_items']==1
    # Existing HBM-side geometry remains unchanged.
    assert loads[0]['free_stride_items']==loads[1]['free_stride_items']==1


def test_masked_byte_ranges_ignore_sentinel_and_preserve_disjoint_segments():
    from triton_viz.tools.nki_trace_dump import _byte_ranges, _byte_span

    sentinel = np.iinfo(np.int64).max
    offsets = np.array([0, 4, sentinel, 12], dtype=np.int64)
    mask = np.array([True, True, False, True])
    assert _byte_span(offsets, 12, mask) == [0, 16]
    assert _byte_ranges(offsets, 12, mask) == [[0, 8], [12, 16]]


def test_dma_geometry_uses_active_mask_partitions():
    from triton_viz.tools.nki_trace_dump import _dma_geometry

    mask = np.zeros((128, 16), dtype=bool)
    mask[:16, :8] = True
    geometry = _dma_geometry([128, 16], 16 * 8 * 4, masks=mask)
    assert geometry["partition_count"] == 16
    assert geometry["free_bytes_per_partition"] == 8 * 4


def test_large_interleaved_byte_ranges_fall_back_to_compact_bounding_span():
    from triton_viz.tools.nki_trace_dump import (
        MAX_EXACT_BYTE_RANGES,
        _byte_ranges,
        _byte_span,
    )

    offsets = np.arange(MAX_EXACT_BYTE_RANGES + 1, dtype=np.int64) * 8
    assert _byte_ranges(offsets, offsets.size * 4) == []
    assert _byte_span(offsets, offsets.size * 4) == [0, int(offsets[-1]) + 4]


def test_sbuf_scalar_scatter_is_grouped_as_static_dma():
    triton_viz.clear()

    def kernel(src, out):
        src_tile = nl.ndarray((nl.par_dim(2), 12), dtype=src.dtype, buffer=nl.sbuf)
        dst_tile = nl.ndarray((nl.par_dim(2), 12), dtype=src.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=src_tile, src=src)
        for i in nl.affine_range(3):
            for j in nl.affine_range(4):
                nisa.tensor_copy(
                    dst=dst_tile[:, nl.ds(j * 3 + i, 1)],
                    src=src_tile[:, nl.ds(i * 4 + j, 1)],
                )
        nisa.dma_copy(dst=out, src=dst_tile)

    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    src = np.arange(24, dtype=np.float32).reshape(2, 12)
    traced[(1,)](src, np.empty_like(src), pre_trace=False)
    events = records_to_events(launches[-1].records)
    static = [event for event in events if event.get("engine") == "static_dma"]
    assert len(static) == 12
    assert {event["static_dma_group"] for event in static} == {0}
    assert {event["static_dma_group_copies"] for event in static} == {12}
    assert {event["static_dma_group_x"] for event in static} == {3}
    assert {event["static_dma_group_y"] for event in static} == {4}
    assert [event["dst_offset_first"] for event in static[:4]] == [0, 12, 24, 36]


def test_dot_input_ptrs_match_producing_transfers_and_serialize():
    """Regression for the pseudo-pointer bug in _nki_beta2_dot_adapter.

    The stationary operand is consumed transposed by nc_matmul. The adapter must
    keep the *original* SBUF pointer (not a materialized transpose copy) so both
    Dot.input_ptrs match the DMAs that produced the tiles, and the matmul is
    scheduled after BOTH input DMAs even with parallel DMA queues.
    """
    from triton_viz.tools.nki_cost_model import simulate, CostModel

    triton_viz.clear()

    def kernel(a, b):
        a_tile = nl.ndarray((64, 32), dtype=a.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((64, 16), dtype=b.dtype, buffer=nl.sbuf)
        out = nl.ndarray((32, 16), dtype=nl.float32, buffer=nl.psum)
        nisa.dma_copy(dst=a_tile, src=a)
        nisa.dma_copy(dst=b_tile, src=b)
        nisa.nc_matmul(dst=out, stationary=a_tile, moving=b_tile)

    a = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    b = np.arange(64 * 16, dtype=np.float32).reshape(64, 16)
    traced = triton_viz.trace(client=Tracer(), frontend="nki_beta2")(kernel)
    traced[(1,)](a, b, pre_trace=False)
    events = records_to_events(launches[-1].records)

    transfer_dsts = {e["dst_ptr"] for e in events if e["op"] == "transfer"}
    dot = next(e for e in events if e["op"] == "dot")
    assert dot["input_shape"] == [32, 64]
    assert dot["other_shape"] == [64, 16]
    assert len(dot["input_ptrs"]) == 2
    assert all(ptr in transfer_dsts for ptr in dot["input_ptrs"]), (
        dot["input_ptrs"], transfer_dsts
    )

    result = simulate(events, CostModel(dma_queue_count=2))
    dot_start = result.timeline["tensor"][0].start
    dma_end = max(e.end for e in result.timeline["dma"])
    assert dot_start >= dma_end - 1e-9
