import copy

import pytest

from triton_viz.tools.nki_cost_model import CostModel, InitializerCalibration, simulate

pytestmark=pytest.mark.nki


def model():
    shapes=((128,64),(128,256),(128,1024),(128,4096),(32,512))
    calibration=InitializerCalibration({'vector':(50.0,.7,0.0),'gpsimd':(95.0,.7,.01)},
        {'vector':shapes,'gpsimd':shapes},frozenset({1.0,2.0}))
    return CostModel(initializer_calibration=calibration,cross_engine_sync_ns=0)


def fill():
    return {'seq':0,'op':'memory_fill','api_op':'memset','engine':'vector',
        'output_memory':'sbuf','output_dtype':'float32','output_partition_axis':0,
        'allocation_partition_axis':0,'output_shape':[128,256],'allocation_shape':[128,256],
        'output_ptr':101,'output_storage':101,'allocation_storage':101,
        'output_strides':[256,1],'output_range':[0,128*256*4],
        'input_ptrs':[],'fill_value':1.0}


def test_only_explicit_calibrated_geometry_receives_a_cost():
    event=fill();assert model().cost_ns(event)==pytest.approx(50+.7*256)
    for key,value in [('engine','unknown'),('engine','scalar'),('output_memory','psum'),
        ('output_dtype','uint32'),('fill_value',0.0),('fill_value',True),('output_strides',[1,128]),
        ('allocation_shape',[128,512]),('allocation_storage',102),('input_ptrs',[9]),
        ('fusion_group','hidden'),('scheduler_duration_override_ns',0.0)]:
        bad=copy.deepcopy(event);bad[key]=value
        with pytest.raises(ValueError,match='Unpriced memory initializer'):model().cost_ns(bad)
    for shape in ([128,32],[16,512],[128,8192]):
        bad=copy.deepcopy(event);bad.update(output_shape=shape,allocation_shape=shape,
            output_strides=[shape[1],1],output_range=[0,shape[0]*shape[1]*4])
        with pytest.raises(ValueError,match='Unpriced memory initializer'):model().cost_ns(bad)


def test_initializer_write_orders_its_dma_consumer():
    event=fill()
    store={'seq':1,'op':'transfer','engine':'dma','mem_src':'sbuf','mem_dst':'hbm',
           'bytes':128*256*4,'src_ptr':101,'src_storage':101,'src_range':[0,128*256*4],
           'dst_ptr':102,'dst_storage':102,'dst_range':[0,128*256*4]}
    result=simulate([event,store],model())
    assert result.timeline['dma'][0].start>=result.timeline['vector'][0].end


def test_unsupported_fill_cannot_hide_behind_override_or_fusion():
    event=fill();event['scheduler_duration_override_ns']=1
    with pytest.raises(ValueError,match='Unpriced memory initializer'):simulate([event],model())
    event=fill();event['fusion_group']='x';event['fusion_signature']='x'
    with pytest.raises(ValueError,match='Unpriced memory initializer'):simulate([event],model())
    with pytest.raises(ValueError,match='Unpriced memory initializer'):simulate([fill()],CostModel())


def test_real_trace_carries_contiguous_geometry_and_dependency():
    import numpy as np
    import triton_viz
    from triton_viz.clients import Tracer
    from triton_viz.core.trace import launches
    from triton_viz.tools.nki_trace_dump import records_to_events
    import nki.isa as nisa
    import nki.language as nl
    def kernel(out):
        tile=nl.ndarray((128,256),dtype=nl.float32,buffer=nl.sbuf)
        nisa.memset(dst=tile,value=1.0,engine=nisa.engine.vector)
        nisa.dma_copy(dst=out,src=tile)
    triton_viz.clear();out=np.zeros((128,256),np.float32)
    triton_viz.trace(client=Tracer(),frontend='nki_beta2')(kernel)[1,](out)
    events=records_to_events(launches[-1].records)
    event=next(e for e in events if e['op']=='memory_fill')
    assert event['output_strides']==[256,1] and np.all(out==1)
    result=simulate(events,model())
    assert result.timeline['dma'][0].start>=result.timeline['vector'][0].end


def test_loader_checks_validation_and_engine_scope(tmp_path):
    import json
    shapes=[[128,64],[128,256],[128,1024],[128,4096],[32,512]]
    entry={'coefficients':[50.0,.7,0.0],'fit_geometries':shapes,'opcode':'MEMSET',
           'fit_validation':'passed','zero_validation':'passed'}
    value={'schema':'nki-explicit-sbuf-fp32-initializer-v1','device':'inf2-trn1',
           'admitted_values':list(range(9)),'engines':{'vector':entry,'gpsimd':dict(entry)},
           'evidence_sha256':{k:'a'*64 for k in ('registration','raw','fit','zero_registration','zero_result')}}
    path=tmp_path/'calibration.json';path.write_text(json.dumps(value))
    calibrated=InitializerCalibration.from_json(path)
    event=fill();event['fill_value']=0.0
    assert calibrated.instruction_ns(event)==pytest.approx(50+.7*256)
    value['engines']['vector']['zero_validation']='failed';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='invalid initializer calibration'):InitializerCalibration.from_json(path)
