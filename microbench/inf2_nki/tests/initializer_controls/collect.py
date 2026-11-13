"""Register and collect fixed, source-audited initializer controls on Inf2."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

SIM=Path(__file__).resolve().parents[4]
PROJECT=Path('/home/ubuntu/nki-shared-static-baseline-20261113')
sys.path[:0]=[str(SIM),str(PROJECT)]
from microbench.inf2_nki.tests.initializer_controls.generate import controls
from emit_ako.harness import loop
from emit_ako.profile_environment import current

PY=Path('/opt/aws_neuronx_venv_pytorch_2_9/bin/python')
NAME='initializer_controls_20261113'
OUT=Path('/home/ubuntu/nki-initializer-controls-20261113')
REG=PROJECT/'data'/f'{NAME}_registration.json';RESULT=PROJECT/'data'/f'{NAME}_result.json'
EVIDENCE=PROJECT/'data'/f'{NAME}_evidence.zip'
PROTOCOL=Path(__file__).with_name('PROTOCOL.md');GENERATOR=Path(__file__).with_name('generate.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state():
    if current()!={'NEURON_RT_ENABLE_DGE_NOTIFICATIONS':'1'}:raise ValueError('enable DMA notifications')
    if subprocess.check_output(['git','-C',str(SIM),'status','--porcelain','--untracked-files=all']).strip():
        raise ValueError('commit the simulator worktree first')
    paths=[Path(__file__),GENERATOR,PROTOCOL,PY.resolve(),(PY.parent/'neuronx-cc').resolve()]
    return {'files':{str(p):sha(p) for p in paths},'harness_sha256':loop.harness_sha256(),
            'simulator_commit':subprocess.check_output(['git','-C',str(SIM),'rev-parse','HEAD'],text=True).strip(),
            'environment':current()}


def committed(path,root):
    rel=path.relative_to(root).as_posix()
    if subprocess.check_output(['git','-C',str(root),'show','HEAD:'+rel])!=path.read_bytes():
        raise ValueError('commit first: '+rel)


def prepare():
    if any(p.exists() for p in (OUT,REG,RESULT,EVIDENCE)):raise ValueError('controls already started')
    for path in (Path(__file__),GENERATOR,PROTOCOL):committed(path,SIM)
    initial=state();OUT.mkdir();rows=[]
    for case in controls():
        folder=OUT/case.name;folder.mkdir();task=folder/'task.py';source=folder/'kernel.py'
        task.write_text(case.task());source.write_text(case.source())
        rows.append({'name':case.name,'partitions':case.partitions,'width':case.width,'engine':case.engine,
                     'split':case.split,'repeats':case.repeats,'task_sha256':sha(task),'source_sha256':sha(source)})
    if state()!=initial:raise ValueError('inputs changed during preparation')
    REG.write_text(json.dumps({'protocol':str(PROTOCOL),'state':initial,'cases':rows,
        'captures_per_case':2,'target':'trn1','policy':'exact source/engine/opcode audit; no outcome exclusion or timing extrapolation'},indent=2)+'\n')
    print('Registered initializer controls before hardware capture',flush=True)


def command(cmd,folder,label,*,device=False):
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','NEURON_RT_VISIBLE_CORES':'0',
         'PYTHONPATH':str(PROJECT)+os.pathsep+str(SIM)}
    try:
        if device:
            with open('/tmp/emit_neuron_device.lock','a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                p=subprocess.run(cmd,cwd=PROJECT,env=env,text=True,capture_output=True,timeout=900)
        else:p=subprocess.run(cmd,cwd=PROJECT,env=env,text=True,capture_output=True,timeout=900)
        stdout,stderr,code=p.stdout,p.stderr,p.returncode
    except subprocess.TimeoutExpired as exc:
        def decode(x):return x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
        stdout,stderr,code=decode(exc.stdout),decode(exc.stderr)+'\ntimeout',None
    (folder/(label+'.stdout')).write_text(stdout);(folder/(label+'.stderr')).write_text(stderr)
    try:result=json.loads(stdout.rsplit('@@RESULT@@',1)[1].strip())
    except (IndexError,json.JSONDecodeError):result={'status':'invalid_output'}
    return {'returncode':code,'result':result}


def audit_capture(table,source,engine,repeats):
    """Only exact, one-instruction source matches can become calibration data."""
    import pyarrow.parquet as pq
    expected={6+2*i for i in range(repeats)};matched={line:[] for line in expected}
    for row in pq.read_table(table).to_pylist():
        loc=str(row.get('nki_source_location') or '')
        try:filename,line=loc.rsplit(':',1);line=int(line)
        except (ValueError,TypeError):continue
        if Path(filename).name!=source.name or line not in matched:continue
        actual=str(row.get('engine') or '').lower()
        opcode=str(row.get('opcode') or '').upper()
        if opcode in ('EVENT_SEMAPHORE','DRAIN','NOP'):continue
        matched[line].append({'engine':actual,'opcode':opcode,'duration_ns':row.get('duration_ns')})
    if any(len(v)!=1 for v in matched.values()):
        return {'status':'unusable','reason':'missing or multiple mapped instructions','matched':matched}
    values=[v[0] for v in matched.values()];opcodes={v['opcode'] for v in values}
    if (len(opcodes)!=1 or not next(iter(opcodes)) or any(v['engine']!=engine for v in values) or
        any(type(v['duration_ns']) not in (int,float) or v['duration_ns']<=0 for v in values)):
        return {'status':'unusable','reason':'opcode or duration mismatch','matched':matched}
    return {'status':'ok','opcode':next(iter(opcodes)),
            'mean_instruction_ns':sum(v['duration_ns'] for v in values)/len(values),'matched':matched}


def run():
    committed(REG,PROJECT);reg=json.loads(REG.read_text())
    if state()!=reg['state'] or any(p.exists() for p in (RESULT,EVIDENCE,OUT/'started.json')):
        raise ValueError('registered inputs changed or collection already started')
    for case in reg['cases']:
        folder=OUT/case['name']
        if sha(folder/'kernel.py')!=case['source_sha256'] or sha(folder/'task.py')!=case['task_sha256']:
            raise ValueError('generated control changed')
    (OUT/'started.json').write_text(json.dumps({'registration_sha256':sha(REG)})+'\n');results=[]
    for case in reg['cases']:
        folder=OUT/case['name'];task=folder/'task.py';source=folder/'kernel.py';build=folder/'build'
        base=[str(PY),'-m','emit_ako.hw_eval',str(task),str(source),str(build)]
        compiled=command(base+['--phase','compile','--seed','0','--target','trn1'],folder,'compile')
        captures=[]
        if compiled['returncode']==0 and compiled['result'].get('status')=='compiled':
            for i in range(reg['captures_per_case']):
                capture=command(base+['--phase','run','--seed','0','--target','trn1','--profile',
                                      '--simulator-repo',str(SIM)],folder,f'capture{i}',device=True)
                parquet=(build/'explorer_parquet/Instruction.parquet' if i==0 else
                         next(iter(sorted(build.glob('profile_capture_*/explorer_parquet/Instruction.parquet'))),None))
                if capture['result'].get('status')=='ok' and parquet is not None and parquet.is_file():
                    copy=folder/f'instructions{i}.parquet';shutil.copyfile(parquet,copy)
                    capture['instruction_audit']=audit_capture(copy,source,case['engine'],case['repeats'])
                else:capture['instruction_audit']={'status':'unusable','reason':'capture or instruction table missing'}
                captures.append(capture)
        row={'case':case,'compile':compiled,'captures':captures};results.append(row)
        (OUT/'partial.json').write_text(json.dumps(results,indent=2)+'\n')
        print(case['name'],compiled['result'].get('status'),
              [c['instruction_audit']['status'] for c in captures],flush=True)
    if state()!=reg['state']:raise ValueError('registered inputs changed during collection')
    RESULT.write_text(json.dumps({'registration_sha256':sha(REG),'results':results},indent=2)+'\n')
    with zipfile.ZipFile(EVIDENCE,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.stdout','.stderr','.parquet','.log'):
                archive.write(p,p.relative_to(OUT).as_posix())
    print('Finished initializer controls',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'))
    {'prepare':prepare,'run':run}[parser.parse_args().stage]()
