"""Opt-in consumer-visible instruction latency (InstructionLatencyCalibration)."""

from dataclasses import replace

import pytest

from triton_viz.tools.nki_cost_model import CostModel, InstructionLatencyCalibration, simulate


def vec(seq, src, dst, f=512):
    return dict(op="compute", engine="vector", seq=seq, input_shape=[128, f], output_shape=[128, f],
                input_dtypes=["float32"], input_storages=[src], input_ranges=[[0, 4 * f]],
                output_storage=dst, output_range=[0, 4 * f])


def chain():
    return [vec(0, 1, 2), vec(1, 2, 3), vec(2, 3, 4)]       # each op reads the previous result


def independent():
    return [vec(0, 1, 2), vec(1, 1, 3), vec(2, 1, 4)]       # all read the same input


LAT = InstructionLatencyCalibration({("vector", "float32", 1): (5000.0, 0.0)})


def test_default_is_unchanged():
    base = CostModel()
    assert base.instruction_latency is None
    for events in (chain(), independent()):
        a = simulate(events, base)
        b = simulate(events, replace(base, instruction_latency=None))
        assert a.predicted_latency_ns == b.predicted_latency_ns


def test_dependent_chain_waits_for_latency_independent_stream_does_not():
    base = CostModel()
    model = replace(base, instruction_latency=LAT)
    dep0, dep1 = simulate(chain(), base), simulate(chain(), model)
    ind0, ind1 = simulate(independent(), base), simulate(independent(), model)
    starts = [t.start for t in dep1.timeline["vector"]]
    assert starts[1] - starts[0] >= 5000.0 - 1e-6 and starts[2] - starts[1] >= 5000.0 - 1e-6
    ind_starts = [t.start for t in ind1.timeline["vector"]]
    assert ind_starts == [t.start for t in ind0.timeline["vector"]]   # throughput unchanged
    assert dep1.timeline["vector"][0].end == dep0.timeline["vector"][0].end  # engine release unchanged


def test_latency_lookup_falls_back_to_one_stream():
    cal = InstructionLatencyCalibration({("vector", "float32", 1): (100.0, 1.0)})
    assert cal.latency_ns("vector", "float32", 2, 10) == pytest.approx(110.0)
    assert cal.latency_ns("scalar", "float32", 1, 10) is None


def test_rewriting_one_buffer_waits_for_release_not_latency():
    # Three ops overwrite the same output buffer from independent inputs (WAW only): ordering the writes
    # needs engine release, not the result latency. A final reader of that buffer does wait for it.
    model = replace(CostModel(), instruction_latency=LAT)
    waw = [vec(0, 1, 9), vec(1, 2, 9), vec(2, 3, 9)]
    base = simulate(waw, CostModel())
    lat = simulate(waw, model)
    assert [t.start for t in lat.timeline["vector"]] == [t.start for t in base.timeline["vector"]]
    read = simulate(waw + [vec(3, 9, 10)], model)
    starts = [t.start for t in read.timeline["vector"]]
    assert starts[3] - starts[2] >= 5000.0 - 1e-6
