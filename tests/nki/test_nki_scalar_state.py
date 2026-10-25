"""NC-v2 register transitions measured by independent reset/idle/deferred probes."""
import numpy as np
import pytest
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_trace_dump import records_to_events
import nki.isa as nisa
import nki.language as nl
pytestmark=pytest.mark.nki


def state_kernel(x,out):
    tile=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    act=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    rows=nl.ndarray((2,1),dtype=nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=tile,src=x)
    nisa.activation(dst=act,op=nl.exp,data=tile,reduce_op=nl.add,reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.activation(dst=act,op=nl.exp,data=tile,reduce_op=nl.add,reduce_cmd=nisa.reduce_cmd.reduce,reduce_res=rows)
    nisa.dma_copy(dst=out[:,0:1],src=rows)
    nisa.activation(dst=act,op=nl.exp,data=tile)
    nisa.activation(dst=act,op=nl.exp,data=tile,reduce_op=nl.add,reduce_cmd=nisa.reduce_cmd.reduce,reduce_res=rows)
    nisa.dma_copy(dst=out[:,1:2],src=rows)
    nisa.activation(dst=act,op=nl.exp,data=tile,reduce_op=nl.add,reduce_cmd=nisa.reduce_cmd.idle,reduce_res=rows)
    nisa.dma_copy(dst=out[:,2:3],src=rows)
    nisa.activation(dst=act,op=nl.exp,data=tile,reduce_op=nl.add,reduce_cmd=nisa.reduce_cmd.reset,reduce_res=rows)
    nisa.dma_copy(dst=out[:,3:4],src=rows)
    nisa.activation(dst=act,op=nl.exp,data=tile,reduce_op=nl.add,reduce_cmd=nisa.reduce_cmd.reduce,reduce_res=rows)
    nisa.dma_copy(dst=out[:,4:5],src=rows)
    nisa.tensor_scalar(dst=act,data=act,op0=nl.add,operand0=.5)


def test_deferred_reductions_plain_call_idle_replace_and_reset():
    x=np.arange(8,dtype=np.float32).reshape(2,4)/10;out=np.zeros((2,5),np.float32)
    triton_viz.clear();triton_viz.trace(client=Tracer(scalar_state_metadata=True),frontend='nki_beta2')(state_kernel)[1,](x,out)
    sums=np.exp(x).sum(axis=1,keepdims=True)
    assert np.allclose(out,np.concatenate([2*sums,3*sums,sums,0*sums,sums],axis=1),rtol=1e-6)
    events=records_to_events(launches[-1].records);states=[e['scalar_state'] for e in events if 'scalar_state' in e]
    assert len(states)==6
    assert states[0]['command']=='reset_reduce' and not states[0]['reads'] and states[0]['writes'] and not states[0]['readout']
    assert states[1]['command']=='reduce' and states[1]['reads'] and states[1]['writes']
    assert states[3]['command']=='idle' and not states[3]['reads'] and states[3]['writes']
    assert all(not s['physical_result_phase_modelled'] for s in states)
    assert 'scalar_state' not in next(e for e in events if e.get('api_op')=='add')


def uninitialized_kernel(x,out,initialize):
    tile=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    act=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    rows=nl.ndarray((2,1),dtype=nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=tile,src=x)
    if initialize and nl.program_id(0)==0:
        nisa.activation(dst=act,op=nl.exp,data=tile,reduce_op=nl.add,reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.activation(dst=act,op=nl.exp,data=tile,reduce_op=nl.add,reduce_cmd=nisa.reduce_cmd.reduce,reduce_res=rows)
    nisa.dma_copy(dst=out,src=rows)


def test_state_does_not_leak_between_launches():
    x=np.zeros((2,4),np.float32);out=np.zeros((2,1),np.float32)
    traced=triton_viz.trace(client=Tracer(),frontend='nki_beta2')(uninitialized_kernel)
    traced[1,](x,out,True)
    with pytest.raises(ValueError,match='registers are uninitialized'):traced[1,](x,out,False)


def test_state_does_not_leak_between_grid_programs():
    x=np.zeros((2,4),np.float32);out=np.zeros((2,1),np.float32)
    with pytest.raises(ValueError,match='registers are uninitialized'):
        triton_viz.trace(client=Tracer(),frontend='nki_beta2')(uninitialized_kernel)[2,](x,out,True)


def test_default_trace_has_no_scalar_state_fields():
    x=np.zeros((2,4),np.float32);out=np.zeros((2,5),np.float32)
    triton_viz.clear();triton_viz.trace(client=Tracer(),frontend='nki_beta2')(state_kernel)[1,](x,out)
    assert all('scalar_state' not in e for e in records_to_events(launches[-1].records))
