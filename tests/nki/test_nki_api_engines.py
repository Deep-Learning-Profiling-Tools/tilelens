"""Convenience reciprocal and activation(reciprocal) use different engines."""
import numpy as np
import pytest
import triton_viz
from triton_viz.clients import Tracer
from triton_viz.core.trace import launches
from triton_viz.tools.nki_trace_dump import records_to_events
import nki.isa as nisa
import nki.language as nl
pytestmark=pytest.mark.nki


def kernel(x,out):
    tile=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    result=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=tile,src=x)
    nisa.reciprocal(dst=result,data=tile)
    nisa.dma_copy(dst=out,src=result)
    nisa.activation(dst=result,op=nl.reciprocal,data=tile)
    nisa.tensor_scalar(dst=result,data=result,op0=nl.add,operand0=.125,engine=nisa.engine.scalar)


@pytest.mark.parametrize('enabled,expected',[(False,['scalar','scalar','vector']),(True,['vector','scalar','scalar'])])
def test_engine_override_is_opt_in_and_does_not_leak_to_later_writes(enabled,expected):
    x=np.arange(1,9,dtype=np.float32).reshape(2,4);out=np.zeros_like(x)
    triton_viz.clear();triton_viz.trace(client=Tracer(hardware_engine_metadata=enabled),frontend='nki_beta2')(kernel)[1,](x,out)
    assert np.allclose(out,1/x)
    events=records_to_events(launches[-1].records)
    assert [e['engine'] for e in events if e.get('api_op') in ('reciprocal','add')]==expected


def combined_kernel(x,out):
    tile=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    primary=nl.ndarray((2,4),dtype=nl.float32,buffer=nl.sbuf)
    rows=nl.ndarray((2,1),dtype=nl.float32,buffer=nl.sbuf)
    inv=nl.ndarray((2,1),dtype=nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=tile,src=x)
    nisa.activation(dst=primary,op=nl.exp,data=tile,reduce_op=nl.add,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,reduce_res=rows)
    nisa.reciprocal(dst=inv,data=rows)
    nisa.dma_copy(dst=out,src=inv)


def test_engine_metadata_with_partition_multi_output_and_scalar_state():
    from triton_viz.core.simulation.nki_isa_contracts import partition_contract_checks
    x=np.arange(8,dtype=np.float32).reshape(2,4)/10;out=np.zeros((2,1),np.float32)
    tracer=Tracer(allocation_metadata=True,multi_output_metadata=True,
                  scalar_state_metadata=True,hardware_engine_metadata=True)
    triton_viz.clear()
    with partition_contract_checks():
        triton_viz.trace(client=tracer,frontend='nki_beta2')(combined_kernel)[1,](x,out)
    assert np.allclose(out,1/np.exp(x).sum(axis=1,keepdims=True),rtol=1e-6)
    events=records_to_events(launches[-1].records)
    activation=next(e for e in events if e.get('api_op')=='exp')
    reciprocal=next(e for e in events if e.get('api_op')=='reciprocal')
    assert reciprocal['engine']=='vector'
    assert activation['output_roles']==['primary','reduction']
    assert activation['scalar_state']['command']=='reset_reduce'
    assert not activation['scalar_state']['physical_result_phase_modelled']
