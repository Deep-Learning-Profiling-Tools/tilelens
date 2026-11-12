"""Differential test of region DAG liveness on independent synthetic traces."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
OLD=Path(os.environ.get('AKO_REGION_REFERENCE_SIM','/home/ubuntu/nki-simulator-initializer-development-20261112'))
PROGRAM=r'''
import hashlib,json,random
from triton_viz.tools.nki_region_ir import build_region_ir
for seed in range(120):
    rng=random.Random(seed);members=[]
    for i in range(rng.randrange(0,220)):
        inputs=[rng.randrange(0,max(1,i+4)) for _ in range(rng.randrange(0,4))]
        output=rng.randrange(0,i+5) if rng.randrange(8) else None
        width=rng.choice([1,16,64,128]);op=rng.choice(['add','multiply','reduce_sum','rsqrt'])
        members.append({'op':'compute','api_op':op,'input_ptrs':inputs,'output_ptr':output,
                        'input_shape':[128,width],'output_shape':[128,width],
                        'input_dtypes':['float32']*len(inputs),'output_dtype':'float32'})
    context=[{'op':'load','active_lanes':128*64,'partition_count':128,
              'offsets_shape':[128,64],'mem_src':'HBM','mem_dst':'SBUF'}]
    result=build_region_ir(members,context)
    print(hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest())
'''


def test_full_region_outputs_match_previous_algorithm():
    if not (OLD/'triton_viz/tools/nki_region_ir.py').exists():pytest.skip('reference simulator unavailable')
    outputs=[]
    for repo in (OLD,ROOT):
        env={**os.environ,'PYTHONPATH':str(repo),'OPENBLAS_NUM_THREADS':'1'}
        p=subprocess.run([sys.executable,'-c',PROGRAM],cwd='/tmp',env=env,text=True,
                         capture_output=True,check=True,timeout=90)
        outputs.append(p.stdout.splitlines())
    assert len(outputs[0])==120 and outputs[0]==outputs[1]
