import numpy as np
import pytest
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_cost_model import simulate
from triton_viz.tools.nki_trace_dump import records_to_events

pytestmark = pytest.mark.nki


def test_addressed_sbuf_executes_virtually_but_is_never_priced():
    import nki.isa as nisa
    import nki.language as nl
    def kernel(out):
        tile = nl.ndarray((2, 8), dtype=nl.float32, buffer=nl.sbuf, address=(0, 32768))
        nisa.memset(dst=tile, value=3.0)
        nisa.dma_copy(dst=out, src=tile)
    out = np.zeros((2, 8), dtype=np.float32)
    triton_viz.clear()
    triton_viz.trace(client=Tracer(), frontend='nki_beta2')(kernel)[1,](out)
    events = records_to_events(launches[-1].records)
    assert np.array_equal(out, np.full((2, 8), 3, dtype=np.float32))
    assert any(event.get('unmodeled_explicit_address') for event in events)
    with pytest.raises(ValueError, match='explicit SBUF address'):
        simulate(events)


def test_addressed_view_is_marked_even_without_allocation_metadata():
    import nki.isa as nisa
    import nki.language as nl
    def kernel(out):
        tile = nl.ndarray((2, 8), dtype=nl.float32, buffer=nl.sbuf, address=(0, 4096))
        nisa.memset(dst=tile, value=1.0)
        nisa.dma_copy(dst=out, src=tile[:, :4])
    out = np.zeros((2, 4), dtype=np.float32)
    triton_viz.clear()
    triton_viz.trace(client=Tracer(), frontend='nki_beta2')(kernel)[1,](out)
    events = records_to_events(launches[-1].records)
    assert any(event['op'] == 'transfer' and event.get('unmodeled_explicit_address') for event in events)
    assert np.array_equal(out, np.ones_like(out))


def test_unsupported_explicit_addresses_are_rejected():
    from triton_viz.core.simulation.nki_beta2 import ndarray, sbuf, hbm, psum
    for address in ((-1, 0), (0, -1), (True, 0), (0,), [0, 32]):
        with pytest.raises(ValueError, match='explicit SBUF address'):
            ndarray((2, 8), np.float32, buffer=sbuf, address=address)
    for buffer in (hbm, psum):
        with pytest.raises(ValueError, match='explicit SBUF address'):
            ndarray((2, 8), np.float32, buffer=buffer, address=(0, 32))


def test_unused_address_allocation_still_marks_the_trace():
    import nki.isa as nisa
    import nki.language as nl
    def kernel(x, out):
        nl.ndarray((2, 8), dtype=nl.float32, buffer=nl.sbuf, address=(0, 4096))
        tile = nl.ndarray((2, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=x)
        nisa.dma_copy(dst=out, src=tile)
    x = np.ones((2, 8), dtype=np.float32)
    out = np.zeros_like(x)
    triton_viz.clear()
    triton_viz.trace(client=Tracer(), frontend='nki_beta2')(kernel)[1,](x, out)
    events = records_to_events(launches[-1].records)
    assert np.array_equal(x, out)
    assert any(event.get('unmodeled_explicit_address') for event in events)
    with pytest.raises(ValueError, match='explicit SBUF address'):
        simulate(events)
