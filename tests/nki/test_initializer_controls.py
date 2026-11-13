import ast

import numpy as np
import pytest

from microbench.inf2_nki.tests.initializer_controls.generate import FillControl,controls

pytestmark=pytest.mark.nki


def test_fit_and_validation_geometries_are_fixed_and_disjoint():
    cases=controls()
    assert len(cases)==14 and len({c.name for c in cases})==len(cases)
    for engine in ('vector','gpsimd'):
        fit={(c.partitions,c.width) for c in cases if c.engine==engine and c.split=='fit'}
        validation={(c.partitions,c.width) for c in cases if c.engine==engine and c.split=='validation'}
        assert len(fit)==5 and len(validation)==2 and not fit & validation


@pytest.mark.parametrize('case',controls())
def test_each_fill_has_distinct_observable_output(case):
    tree=ast.parse(case.source());fn=tree.body[-1]
    fills=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr=='memset']
    writes=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr=='dma_copy']
    assert len(fills)==len(writes)==case.repeats
    assert [next(k.value.value for k in n.keywords if k.arg=='value') for n in fills]==list(range(1,case.repeats+1))
    module={};exec(case.task(),module)
    values=module['make_inputs'](0);ref=module['reference'](values)
    assert ref.shape==values['out'].shape
    assert np.unique(ref).tolist()==list(range(1,case.repeats+1))


@pytest.mark.parametrize('kwargs',[{'engine':'unknown'},{'width':0},{'partitions':129},{'split':'other'},
                                   {'repeats':True}])
def test_reject_uncontrolled_parameters(kwargs):
    fields={'partitions':128,'width':64,'engine':'vector','split':'fit'};fields.update(kwargs)
    with pytest.raises(ValueError):FillControl(**fields)


def test_instruction_audit_rejects_missing_extra_and_wrong_engine(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from microbench.inf2_nki.tests.initializer_controls.collect import audit_capture
    source=tmp_path/'kernel.py';table=tmp_path/'instructions.parquet'
    rows=[{'nki_source_location':str(source)+':'+str(6+2*i),'engine':'Vector',
           'opcode':'MEMSET','duration_ns':20} for i in range(8)]
    def run(items):
        pq.write_table(pa.Table.from_pylist(items),table)
        return audit_capture(table,source,'vector',8)
    assert run(rows)['mean_instruction_ns']==20
    assert run(rows[:-1])['status']=='unusable'
    extra=dict(rows[0]);extra['engine']='GpSimd'
    assert run(rows+[extra])['status']=='unusable'
    wrong=[dict(r) for r in rows];wrong[0]['engine']='GpSimd'
    assert run(wrong)['status']=='unusable'
    wrong=[dict(r) for r in rows];wrong[0]['opcode']='COPY'
    assert run(wrong)['status']=='unusable'
