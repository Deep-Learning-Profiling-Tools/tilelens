"""Implicit native SDK preparation is metadata, not extra source execution."""
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
    rows=nl.ndarray((2,1),dtype=nl.float32,buffer=nl.sbuf)
    nisa.dma_copy(dst=tile,src=x)
    nisa.memset(dst=rows,value=.125)
    nisa.activation(dst=result,op=nl.exp,data=tile)
    nisa.activation(dst=result,op=nl.exp,data=tile,bias=rows,scale=.25)
    nisa.activation(dst=result,op=nl.exp,data=tile,bias=rows,scale=rows)
    nisa.activation(dst=result,op=nl.reciprocal,data=tile)
    nisa.tensor_scalar(dst=result,data=result,op0=nl.add,operand0=.125)
    nisa.dma_copy(dst=out,src=result)


def run(enabled):
    x=np.arange(1,9,dtype=np.float32).reshape(2,4);out=np.zeros_like(x)
    triton_viz.clear();triton_viz.trace(client=Tracer(sdk_setup_metadata=enabled),frontend='nki_beta2')(kernel)[1,](x,out)
    assert np.allclose(out,1/x+.125)
    return records_to_events(launches[-1].records)


def test_setup_roles_bits_and_tensor_operands_without_fake_source_ops():
    events=run(True);plans=[e['sdk_setup'] for e in events if 'sdk_setup' in e]
    assert len(plans)==4 and [len(p['materializations']) for p in plans]==[2,1,0,0]
    assert plans[0]['materializations'][0]['input_role']=='bias'
    assert plans[0]['materializations'][1]['value_bits_fp32']==1065353216
    assert plans[1]['materializations'][0]['input_role']=='scale'
    assert plans[1]['materializations'][0]['value_bits_fp32']==1048576000
    assert all(p['supported'] and not p['compiler_lowering_complete'] for p in plans)
    assert 'sdk_setup' not in next(e for e in events if e.get('api_op')=='add')
    old=run(False)
    assert len(old)==len(events) and all('sdk_setup' not in e for e in old)


def test_other_generations_are_not_silently_covered(monkeypatch):
    from triton_viz.core.simulation.nki_beta2 import nki_builder
    monkeypatch.setattr(nki_builder,'nc_version',nisa.nc_version.gen3)
    events=run(True)
    assert all(not e['sdk_setup']['supported'] for e in events if 'sdk_setup' in e)
