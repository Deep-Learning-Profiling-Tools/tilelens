from copy import deepcopy
from dataclasses import replace

from triton_viz.tools.nki_cost_model import CostModel, ComputeCalibration, InstructionLatencyCalibration
from triton_viz.core.compute_mask import masked_free_dim


def event():
    return dict(op='compute',api_op='multiply',engine='vector',input_shape=[128,16384],output_shape=[128,16384],
                input_dtypes=['float32'],output_dtype='float32',input_stream_count=1,compute_mask_provided=True,
                compute_mask_geometry=dict(status='known',shape=[128,16384],bounds=[[0,1],[0,192]],
                                           rectangular=True,active_elements=192))


def test_same_mask_geometry_prices_issue_and_consumer_latency_without_changing_allocation():
    e=event(); before=deepcopy(e)
    base=CostModel(compute_calibration=ComputeCalibration({('vector','float32',1):(10.,1.)}),
                   instruction_latency=InstructionLatencyCalibration({('vector','float32',1):(200.,1.,16384)}))
    active=replace(base,active_compute_masks=True)
    assert base.cost_ns(e)==16394
    assert active.cost_ns(e)==202
    assert active.result_ready_ns(e,'vector',202)==392
    assert base.result_ready_ns(e,'vector',16394)==16584
    assert e==before


def test_sparse_reduction_missing_mask_and_inconsistent_geometry_preserve_declared_extent():
    for change in [dict(api_op='sum'),dict(op='reduce_sum'),dict(compute_mask_provided=False)]:
        assert masked_free_dim({**event(),**change}) is None
    for change in [dict(rectangular=False),dict(active_elements=191),dict(bounds=[[0,1],[0,20000]])]:
        e=event();e['compute_mask_geometry'].update(change)
        assert masked_free_dim(e) is None


def test_unknown_geometry_descriptor_preserves_declared_pricing():
    for geometry in (None, [], 'unknown'):
        assert masked_free_dim({**event(),'compute_mask_geometry':geometry}) is None
