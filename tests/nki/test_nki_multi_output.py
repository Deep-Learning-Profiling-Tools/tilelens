"""Companion activation reductions are outputs of the same instruction."""
import numpy as np
import pytest
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_trace_dump import records_to_events
from triton_viz.tools.nki_cost_model import _read_accesses,_write_accesses
import nki.isa as nisa
import nki.language as nl
pytestmark=pytest.mark.nki


def kernel(x,out):
    tile=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    primary=nl.ndarray((2,4),dtype=nl.float16,buffer=nl.sbuf)
    rows=nl.ndarray((2,1),dtype=nl.float32,buffer=nl.sbuf)
    result=nl.ndarray((2,1),dtype=nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=tile,src=x)
    nisa.activation(dst=primary,op=nl.exp,data=tile,reduce_op=nl.add,reduce_res=rows)
    nisa.tensor_scalar(dst=result,data=rows,op0=nl.multiply,operand0=.5)
    nisa.dma_copy(dst=out,src=result)


def run(multi):
    x=np.arange(8,dtype=np.float32).reshape(2,4);out=np.zeros((2,1),np.float32)
    triton_viz.clear();triton_viz.trace(client=Tracer(allocation_metadata=True,multi_output_metadata=multi),frontend='nki_beta2')(kernel)[1,](x,out)
    assert np.allclose(out,.5*np.sum(np.exp(x),axis=1,keepdims=True),rtol=1e-6)
    return records_to_events(launches[-1].records)


def test_companion_output_has_dtype_geometry_and_same_producer_as_primary():
    events=run(True);activation=next(e for e in events if e.get('api_op')=='exp')
    consumer=next(e for e in events if e.get('api_op')=='multiply')
    assert activation['output_shapes']==[[2,4],[2,1]]
    assert activation['output_dtypes']==['float16','float32']
    assert activation['output_roles']==['primary','reduction'] and activation['reduction_op']=='add'
    assert len(activation['output_allocations'])==2
    assert len(_write_accesses(activation))==2
    read=_read_accesses(consumer)[0]
    assert any(w[0]==read[0] and w[4]==read[4] for w in _write_accesses(activation))
    assert activation['output_storage']==activation['output_storages'][0]
    assert activation['output_ptr']==activation['output_ptrs'][0]


def test_default_retains_single_output_trace_schema():
    events=run(False);activation=next(e for e in events if e.get('api_op')=='exp')
    assert 'output_storages' not in activation and 'output_dtypes' not in activation
    assert len(activation['output_allocations'])==1


def reuse_kernel(x,out):
    tile=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    primary=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    rows=nl.ndarray((2,1),dtype=nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=tile,src=x)
    nisa.activation(dst=primary,op=nl.exp,data=tile,reduce_op=nl.add,reduce_res=rows)
    nisa.tensor_scalar(dst=primary,data=primary,op0=nl.multiply,operand0=2.)
    nisa.dma_copy(dst=out,src=primary)


def test_later_inplace_write_does_not_inherit_companion_output():
    x=np.ones((2,4),np.float32);out=np.zeros_like(x)
    triton_viz.clear();triton_viz.trace(client=Tracer(multi_output_metadata=True),frontend='nki_beta2')(reuse_kernel)[1,](x,out)
    events=records_to_events(launches[-1].records)
    activation=next(e for e in events if e.get('api_op')=='exp');multiply=next(e for e in events if e.get('api_op')=='multiply')
    assert len(activation['output_storages'])==2
    assert 'output_storages' not in multiply
