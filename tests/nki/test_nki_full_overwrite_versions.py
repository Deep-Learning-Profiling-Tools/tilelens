import copy
from dataclasses import replace

import numpy as np
import pytest
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_cost_model import CostModel, InitializerCalibration, simulate
from triton_viz.tools.nki_full_overwrite_versions import apply
from triton_viz.tools.nki_trace_dump import records_to_events

pytestmark = pytest.mark.nki


def source_events(reuse=False):
    import nki.isa as nisa
    import nki.language as nl
    def kernel(x, out):
        data = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf)
        zero = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf)
        result = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf)
        if reuse:
            nisa.memset(dst=zero, value=1.0, engine=nisa.engine.gpsimd)
        for block in range(3):
            if not reuse:
                nisa.memset(dst=zero, value=1.0, engine=nisa.engine.gpsimd)
            nisa.dma_copy(dst=data, src=x[nl.ds(block * 128, 128), :])
            nisa.tensor_tensor(dst=result, data1=data, data2=zero, op=nl.add)
            nisa.dma_copy(dst=out[nl.ds(block * 128, 128), :], src=result)
    x = np.ones((384, 256), dtype=np.float32)
    out = np.zeros_like(x)
    triton_viz.clear()
    triton_viz.trace(client=Tracer(allocation_metadata=True), frontend='nki_beta2')(kernel)[1,](x, out)
    assert np.array_equal(out, x + 1)
    return list(records_to_events(launches[-1].records))


def calibrated_model():
    shapes = ((128, 64), (128, 256), (128, 1024), (128, 4096), (32, 512))
    calibration = InitializerCalibration({'vector': (50., .7, 0.), 'gpsimd': (95., .7, .01)},
                                         {'vector': shapes, 'gpsimd': shapes}, frozenset({1.0}))
    return CostModel(initializer_calibration=calibration, cross_engine_sync_ns=0)


def test_real_trace_alternates_full_overwrites_without_losing_readers():
    events = source_events()
    original = copy.deepcopy(events)
    changed, info = apply(events)
    assert events == original
    fills = [e for e in changed if e['op'] == 'memory_fill']
    binaries = [e for e in changed if e['op'] == 'binary']
    ids = [e['output_storage'] for e in fills]
    assert ids[0] != ids[1] and ids[0] == ids[2]
    assert all(fill['output_storage'] in binary['input_storages'] for fill, binary in zip(fills, binaries))
    assert info['depth'] == 2 and sorted(info['writes_by_root']) == [3, 3, 3]
    strict = simulate(events, calibrated_model())
    scenario = simulate(events, replace(calibrated_model(), full_overwrite_version_scenario=True))
    assert scenario.predicted_latency_ns <= strict.predicted_latency_ns
    assert scenario.components_ns['hypothetical_full_overwrite_version_depth'] == 2


def test_once_initialized_tile_keeps_its_single_version():
    events, _ = apply(source_events(reuse=True))
    fill = next(e for e in events if e['op'] == 'memory_fill')
    assert all(fill['output_storage'] in e['input_storages'] for e in events if e['op'] == 'binary')


@pytest.mark.parametrize('corruption', ['partial', 'sparse', 'alias', 'inplace', 'missing', 'oversize', 'two_grids'])
def test_unsupported_version_scenario_is_rejected(corruption):
    events = source_events()
    fill = next(e for e in events if e['op'] == 'memory_fill')
    binary = next(e for e in events if e['op'] == 'binary')
    if corruption == 'partial':
        fill['output_range'][1] -= 4
    elif corruption == 'sparse':
        fill['output_range_lists'][0]['runs'] = [[0, 4], [8, 128*256*4]]
    elif corruption == 'alias':
        fill['output_allocations'][0]['view_storage'] += 1
    elif corruption == 'inplace':
        binary['output_storage'] = binary['input_storages'][0]
        binary['output_ptr'] = binary['input_ptrs'][0]
    elif corruption == 'missing':
        del fill['output_allocations']
    elif corruption == 'oversize':
        for event in events:
            for alloc in event.get('input_allocations', []) + event.get('output_allocations', []):
                if alloc['memory'] == 'sbuf':
                    alloc['shape'][1] = 20000
    else:
        events.append(dict(events[0]))
    with pytest.raises(ValueError):
        apply(events)


def test_scenario_requires_calibration_and_is_explicitly_opt_in():
    events = source_events()
    with pytest.raises(ValueError):
        simulate(events, CostModel(full_overwrite_version_scenario=True))
    assert 'hypothetical_full_overwrite_version_depth' not in simulate(events, calibrated_model()).components_ns
