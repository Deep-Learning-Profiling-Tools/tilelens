"""Validate the frozen nonzero fill model on new zero-valued controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

SIM=Path(__file__).resolve().parents[4]
PROJECT=Path('/home/ubuntu/nki-shared-static-baseline-20261113')
sys.path[:0]=[str(SIM),str(PROJECT)]
from microbench.inf2_nki.tests.initializer_controls import collect as base
from microbench.inf2_nki.tests.initializer_controls.generate import FillControl

NAME='initializer_zero_validation_20261113'
REG=PROJECT/'data'/f'{NAME}_registration.json';RESULT=PROJECT/'data'/f'{NAME}_result.json'
EVIDENCE=PROJECT/'data'/f'{NAME}_evidence.zip';FIT=PROJECT/'data/initializer_controls_fit_20261113_result.json'
PROTOCOL=Path(__file__).with_name('ZERO_PROTOCOL.md')
OUT=Path('/home/ubuntu/nki-initializer-zero-validation-20261113')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    return tuple(FillControl(p,f,engine,'validation') for engine in ('vector','gpsimd')
                 for p,f in ((128,128),(128,2048),(64,512)))


def source(case):
    old=case.source();result=old.replace('value=1.0,','value=0.0,',1)
    if result==old:raise ValueError('source template changed')
    return result


def task(case):
    old=case.task();needle='np.arange(1,9,dtype=np.float32)'
    if old.count(needle)!=1:raise ValueError('task template changed')
    return old.replace(needle,'np.array([0,2,3,4,5,6,7,8],dtype=np.float32)')


def state():
    value=base.state()
    value['files'].update({str(p):sha(p) for p in (Path(__file__),PROTOCOL,FIT)})
    return value


def prepare():
    if any(p.exists() for p in (REG,RESULT,EVIDENCE,OUT)):raise ValueError('zero validation already exists')
    for path in (Path(__file__),PROTOCOL):base.committed(path,SIM)
    base.committed(FIT,PROJECT)
    initial=state();model=json.loads(FIT.read_text())['engine_models']
    if any(model[e].get('status')!='validation_passed' for e in ('vector','gpsimd')):
        raise ValueError('frozen prerequisite model did not pass')
    OUT.mkdir();rows=[]
    for c in cases():
        folder=OUT/c.name;folder.mkdir();(folder/'task.py').write_text(task(c));(folder/'kernel.py').write_text(source(c))
        rows.append({'name':c.name,'engine':c.engine,'partitions':c.partitions,'width':c.width,
                     'repeats':c.repeats,'task_sha256':sha(folder/'task.py'),'source_sha256':sha(folder/'kernel.py')})
    if state()!=initial:raise ValueError('inputs changed during preparation')
    REG.write_text(json.dumps({'protocol':str(PROTOCOL),'state':initial,'cases':rows,'captures_per_case':2,
                               'relative_error_limit':0.30,'target':'trn1'},indent=2)+'\n')
    print('Registered independent zero-fill validation',flush=True)


def run():
    base.committed(REG,PROJECT);reg=json.loads(REG.read_text())
    if state()!=reg['state'] or any(p.exists() for p in (RESULT,EVIDENCE,OUT/'started.json')):
        raise ValueError('registered inputs changed or collection already started')
    for case in reg['cases']:
        folder=OUT/case['name']
        if sha(folder/'task.py')!=case['task_sha256'] or sha(folder/'kernel.py')!=case['source_sha256']:
            raise ValueError('generated source changed')
    (OUT/'started.json').write_text(json.dumps({'registration_sha256':sha(REG)})+'\n')
    models=json.loads(FIT.read_text())['engine_models'];rows=[]
    for case in reg['cases']:
        folder=OUT/case['name'];source_path=folder/'kernel.py';task_path=folder/'task.py';build=folder/'build'
        command=[str(base.PY),'-m','emit_ako.hw_eval',str(task_path),str(source_path),str(build)]
        compiled=base.command(command+['--phase','compile','--seed','0','--target','trn1'],folder,'compile')
        captures=[]
        if compiled['returncode']==0 and compiled['result'].get('status')=='compiled':
            for i in range(reg['captures_per_case']):
                capture=base.command(command+['--phase','run','--seed','0','--target','trn1','--profile',
                                              '--simulator-repo',str(SIM)],folder,f'capture{i}',device=True)
                parquet=(build/'explorer_parquet/Instruction.parquet' if i==0 else
                         next(iter(sorted(build.glob('profile_capture_*/explorer_parquet/Instruction.parquet'))),None))
                if capture['result'].get('status')=='ok' and parquet is not None and parquet.is_file():
                    copy=folder/f'instructions{i}.parquet';shutil.copyfile(parquet,copy)
                    capture['audit']=base.audit_capture(copy,source_path,case['engine'],case['repeats'])
                else:capture['audit']={'status':'unusable','reason':'capture or instruction table missing'}
                captures.append(capture)
        values=[c['audit']['matched'][6][0]['duration_ns'] for c in captures if c['audit'].get('status')=='ok' and
                c['audit'].get('opcode')==models[case['engine']]['opcode']]
        prediction=None;measured=None;error=None
        if len(values)==reg['captures_per_case']:
            import statistics
            measured=statistics.median(values);a,b,c=models[case['engine']]['coefficients']
            prediction=a+b*case['width']+c*case['width']*case['partitions']/128
            error=abs(prediction/measured-1)
        row={'case':case,'compile':compiled,'captures':captures,'zero_measured_ns':measured,
             'frozen_prediction_ns':prediction,'relative_error':error,
             'passed':error is not None and error<=reg['relative_error_limit']}
        rows.append(row);(OUT/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(case['name'],'zero_validation',row['passed'],flush=True)
    if state()!=reg['state']:raise ValueError('inputs changed during validation')
    RESULT.write_text(json.dumps({'registration_sha256':sha(REG),'fit_sha256':sha(FIT),'results':rows,
        'passed_by_engine':{e:len([r for r in rows if r['case']['engine']==e])==3 and
            all(r['passed'] for r in rows if r['case']['engine']==e) for e in ('vector','gpsimd')}},indent=2)+'\n')
    with zipfile.ZipFile(EVIDENCE,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.stdout','.stderr','.parquet','.log'):
                archive.write(p,p.relative_to(OUT).as_posix())
    print('Finished zero-fill validation',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'))
    {'prepare':prepare,'run':run}[parser.parse_args().stage]()
