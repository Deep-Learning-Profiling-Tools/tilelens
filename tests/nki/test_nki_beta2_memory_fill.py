"""Initializer writes must be visible before physical lifetime scheduling."""
import numpy as np
import pytest

import triton_viz
try:
    from triton_viz.clients import Tracer
    from triton_viz.core.trace import launches
    from triton_viz.core.frontend.base import _LangPatchScope
    from triton_viz.core.simulation import nki_beta2 as b2
    from triton_viz.tools.nki_trace_dump import records_to_events
    from triton_viz.tools.nki_cost_model import CostModel, simulate
    import nki.isa as nisa
    import nki.language as nl
except ModuleNotFoundError:
    pytest.skip("NKI dependencies unavailable", allow_module_level=True)

pytestmark = pytest.mark.nki


def test_memset_initializes_a_view_and_records_scalar_semantics():
    scope = _LangPatchScope()
    b2.nki_patch_lang(scope)
    try:
        parent = b2.NDArray(value=np.zeros((4, 8), np.float32), buffer='psum')
        dst = parent[:, 2:6]
        result = b2.memset(dst, np.float32(3.5), engine=nisa.engine.vector)
        assert result is dst and result._nki_inputs == ()
        assert result._nki_fill_value == 3.5 and result._nki_engine == 'vector'
        assert np.all(parent.data[:, 2:6] == 3.5)
        assert np.all(parent.data[:, :2] == 0) and np.all(parent.data[:, 6:] == 0)
    finally:
        b2.nki_unpatch_lang(scope)


def _fills_kernel(out):
    outs = nl.ndarray((16, 64, 160), dtype=nl.float32, buffer=nl.sbuf)
    for i in range(16):
        acc = nl.ndarray((64, 160), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(dst=acc, value=float(i + 1), engine=nisa.engine.vector)
        nisa.tensor_copy(dst=outs[i], src=acc)
    for i in range(16):
        nisa.dma_copy(dst=out[i], src=outs[i])


def test_trace_retains_sixteen_distinct_psum_writers_and_copy_links():
    triton_viz.clear()
    out = np.zeros((16, 64, 160), np.float32)
    traced = triton_viz.trace(client=Tracer(), frontend='nki_beta2')(_fills_kernel)
    traced[1,](out)
    assert np.all(out == np.arange(1, 17, dtype=np.float32)[:, None, None])
    events = records_to_events(launches[-1].records)
    fills = [e for e in events if e['op'] == 'memory_fill']
    copies = [e for e in events if e['op'] == 'transfer' and e.get('mem_src') == 'psum']
    assert len(fills) == len(copies) == 16
    assert len({e['output_storage'] for e in fills}) == 16
    for fill, copy in zip(fills, copies):
        assert fill['output_memory'] == 'psum' and fill['engine'] == 'vector'
        assert fill['output_storage'] == copy['src_storage']
        assert fill['output_range'] == [0, 64 * 160 * 4]
        assert fill['input_storages'] == [] and fill['timing_status'] == 'unpriced_initializer'
    with pytest.raises(ValueError, match='Unpriced memory initializer'):
        simulate(events, CostModel())


def _inplace_kernel(out):
    tile = nl.ndarray((2, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=tile, value=2.0, engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=tile, data=tile, op0=nl.multiply, operand0=3.0, engine=nisa.engine.vector)
    nisa.dma_copy(dst=out, src=tile)


def test_initializer_metadata_does_not_leak_to_following_inplace_arithmetic():
    triton_viz.clear()
    out = np.zeros((2, 8), np.float32)
    triton_viz.trace(client=Tracer(), frontend='nki_beta2')(_inplace_kernel)[1,](out)
    events = records_to_events(launches[-1].records)
    assert np.all(out == 6)
    assert sum((e['op'] == 'memory_fill' for e in events)) == 1
    arithmetic = [e for e in events if e['op'] == 'compute']
    assert len(arithmetic) == 1 and arithmetic[0]['api_op'] == 'multiply'


def test_memset_rejects_nonconstant_values_offchip_and_unsupported_engines():
    dst = b2.NDArray(shape=(2, 8), dtype='float32', buffer='sbuf')
    with pytest.raises(ValueError, match='scalar constant'):
        b2.memset(dst, np.ones(8))
    with pytest.raises(ValueError, match='on-chip'):
        b2.memset(b2.NDArray(shape=(2, 8), buffer='hbm'), 1.0)
    with pytest.raises(ValueError, match='engine'):
        b2.memset(dst, 1.0, engine='tensor')


def test_gen2_psum_geometry_constants_are_available_to_source_kernels():
    assert b2.tile_size.psum_num_banks == 8
    assert b2.tile_size.psum_bank_fmax == 512
    assert b2.tile_size.psum_bank_fmax_bytes == 2048


def _cast_kernel(out):
    acc = nl.ndarray((4, 8), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=acc, value=3.0, engine=nisa.engine.vector)
    tile = nl.ndarray((4, 8), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tile, src=acc)
    nisa.dma_copy(dst=out, src=tile)


def test_cast_transfer_keeps_distinct_source_and_destination_byte_spans():
    import ml_dtypes
    triton_viz.clear()
    out = np.zeros((4, 8), dtype=ml_dtypes.bfloat16)
    triton_viz.trace(client=Tracer(), frontend='nki_beta2')(_cast_kernel)[1,](out)
    transfer = next((e for e in records_to_events(launches[-1].records) if e.get('mem_src') == 'psum'))
    assert transfer['src_bytes'] == 128 and transfer['dst_bytes'] == 64
    assert transfer['src_range'] == [0, 128] and transfer['dst_range'] == [0, 64]
    assert transfer['src_ranges'] == [[0, 128]]
    assert np.all(out == 3)


def _view_kernel(out):
    parent = nl.ndarray((4, 8), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=parent, value=0.0, engine=nisa.engine.vector)
    nisa.memset(dst=parent[:, 2:6], value=4.0, engine=nisa.engine.vector)
    tile = nl.ndarray((4, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tile, src=parent)
    nisa.dma_copy(dst=out, src=tile)


def test_view_initializer_retains_declared_allocation_extent():
    triton_viz.clear()
    out = np.zeros((4, 8), np.float32)
    triton_viz.trace(client=Tracer(), frontend='nki_beta2')(_view_kernel)[1,](out)
    fills = [e for e in records_to_events(launches[-1].records) if e['op'] == 'memory_fill']
    assert len(fills) == 2 and fills[1]['output_shape'] == [4, 4]
    assert fills[1]['allocation_shape'] == [4, 8]
    assert fills[1]['allocation_partition_axis'] == 0
    assert fills[0]['allocation_storage'] == fills[1]['allocation_storage']
    assert np.all(out[:, 2:6] == 4) and np.all(out[:, :2] == 0)
