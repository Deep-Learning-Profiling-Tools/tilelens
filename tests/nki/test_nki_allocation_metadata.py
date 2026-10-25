"""Opt-in root allocation metadata for producer/consumer lifetime analysis."""
import numpy as np
import pytest
import triton_viz

try:
    import nki.isa as nisa
    import nki.language as nl
    from triton_viz.clients import Tracer
    from triton_viz.core.trace import launches
    from triton_viz.tools.nki_trace_dump import records_to_events
except ModuleNotFoundError:
    pytest.skip("NKI dependencies unavailable", allow_module_level=True)

pytestmark = pytest.mark.nki


def _kernel(a, b, out):
    lhs = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    trans_psum = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    trans = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    rhs = nl.ndarray((128, 32), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray((128, 32), dtype=nl.float32, buffer=nl.psum)
    sbuf_result = nl.ndarray((128, 32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lhs, src=a)
    nisa.nc_transpose(dst=trans_psum, data=lhs)
    nisa.tensor_copy(dst=trans, src=trans_psum)
    nisa.dma_copy(dst=rhs, src=b)
    nisa.nc_matmul(dst=result, stationary=trans, moving=rhs)
    nisa.tensor_copy(dst=sbuf_result, src=result)
    nisa.dma_copy(dst=out, src=sbuf_result)


def _trace(enabled):
    triton_viz.clear()
    out = np.zeros((128, 32), np.float32)
    traced = triton_viz.trace(client=Tracer(allocation_metadata=enabled), frontend="nki_beta2")(_kernel)
    traced[(1,)](np.ones((128, 128), np.float32), np.ones((128, 32), np.float32), out)
    assert np.all(out == 128)
    return records_to_events(launches[-1].records)


def test_opt_in_metadata_covers_dot_transpose_and_onchip_copy():
    events = _trace(True)
    producers = [e for e in events if e["op"] in ("dot", "tensor_transpose")]
    assert len(producers) == 2
    for event in producers:
        descriptor = event["output_allocations"][0]
        assert descriptor["memory"] == "psum"
        assert descriptor["storage"] == event["output_storage"]
        assert descriptor["shape"] == event["output_shape"]
        assert descriptor["partition_axis"] == 0
        assert descriptor["dtype"] == "float32"
        reader = next(e for e in events if e.get("src_storage") == descriptor["storage"])
        assert reader["input_allocations"][0] == descriptor


def test_default_records_do_not_gain_allocation_descriptors():
    assert all("input_allocations" not in e and "output_allocations" not in e for e in _trace(False))


def _view_kernel(out):
    parent = nl.ndarray((4, 8), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=parent, value=5., engine=nisa.engine.vector)
    tile = nl.ndarray((4, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tile, src=parent[:, 2:6])
    nisa.dma_copy(dst=out, src=tile)


def test_transfer_view_keeps_root_identity_and_declared_extent():
    triton_viz.clear()
    out = np.zeros((4, 4), np.float32)
    triton_viz.trace(client=Tracer(allocation_metadata=True), frontend="nki_beta2")(_view_kernel)[(1,)](out)
    events = records_to_events(launches[-1].records)
    fill = next(e for e in events if e["op"] == "memory_fill")
    copy = next(e for e in events if e.get("mem_src") == "psum")
    descriptor = copy["input_allocations"][0]
    assert descriptor["storage"] == fill["allocation_storage"]
    assert descriptor["shape"] == [4, 8]
    assert descriptor["view_storage"] != descriptor["storage"]
    assert np.all(out == 5)
