"""Minimal actual learner/native inputs; metadata only before numerical imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth498_binding.json';FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic();PROC=psutil.Process();PROC.cpu_affinity([0]);PEAK=HASHED=0
def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset);assert PEAK<=256<<20 and time.monotonic()-START<=90
def item(path,expected=None):
    global HASHED
    if expected:assert len(expected)==64
    p=Path(path).resolve();before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(4<<20):h.update(data);HASHED+=len(data);guard()
    after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':before.st_size,'sha256':h.hexdigest()}
def head(p):
    p=Path(p);saved=subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT,timeout=5)
    assert p.read_bytes().replace(b'\r\n',b'\n')==saved.replace(b'\r\n',b'\n');guard()
try:
    records=[];catalog={}
    hashes={'meth497_binding.json':'89a7a5829cd15f933ed63f2378f7b5a7cc9f8498877da85625f47836291e2864',
        'ADMISSION_497_20261006.json':'6121e58ce83bb53fa12eaf3cda554c3d33288027c5ab7957ee35f97d780a35b5',
        'meth495_binding.json':'c2fac85bb7ace619f2de0031a53dfc30acdc8800824d5bc9ceb811fe77c4f3d4',
        'ADMISSION_495_20261006.json':'de9e28638e3abc6b9a512c7fdf7c6e6751f4505d9501d51d1e6763fd8e98280f'}
    assert all(len(v)==64 for v in hashes.values())
    for name,sha in hashes.items():p=DOC/name;head(p);v=item(p,sha);records.append(v);catalog[v['path']]=v
    old=json.loads((DOC/'meth497_binding.json').read_bytes());source=json.loads((DOC/'meth495_binding.json').read_bytes())
    a497=json.loads((DOC/'ADMISSION_497_20261006.json').read_bytes());a495=json.loads((DOC/'ADMISSION_495_20261006.json').read_bytes())
    for admission in (a497,a495):
        assert admission['main_completed'] and all(admission['gates'].values())
        for key in ('raw','retention'):
            desc=admission[key];head(desc['path']);v=item(desc['path'],desc['sha256']);records.append(v);catalog[v['path']]=v
    rank=json.loads(Path(a497['raw']['path']).read_bytes());assert rank['summary']['development_full_nonempty']==rank['summary']['ALL_full_nonempty']==127
    raw=json.loads(Path(a495['raw']['path']).read_bytes());inventory={Path(v['path']).name:v for v in raw['output_inventory']};data={}
    def add(key,desc):
        data[key]=item(desc['path'],desc['sha256']);catalog[data[key]['path']]=data[key]
    for key in ('uid','occurrences','features'):add(key,old['data'][key])
    for key,sourcekey in (('targets','targets'),('geometry','geometry'),('C0','C0'),('calibration','calibration')):add(key,source['fit_inputs'][sourcekey])
    for key,name in (('bank','bank.bin'),('inputs','inputs.bin'),('baseline_predictions','predictions.bin'),('native','meth495_hybrid_physical.exe')):add(key,inventory[name])
    runtime=old['runtime']
    for files in [runtime['files']]+[v['files'] for v in runtime['packages'].values()]:
        for path,v in files.items():
            p=Path(path).resolve()
            if str(p) not in catalog:catalog[str(p)]=item(p,v['sha256'])
    historical={Path(v['path']).resolve():v for v in source['catalog']};native_modules=[]
    command=next(v for v in raw['commands'] if v['label']=='predict')
    for path in command['observed_modules']:
        p=Path(path).resolve();v=catalog.get(str(p))
        if v is None:v=item(p,historical[p]['sha256'] if p in historical else None);catalog[v['path']]=v
        native_modules.append(v)
    scientific=[];paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth498*'))+[DOC/'METH_498_MINIMUM_PRIOR_PROTOCOL_20261006.md']
    paths.extend(ROOT/'benchmarks/native_expert_scaling'/name for name in ('meth494_operations.py','meth490_r1_operations.py','meth495_hybrid_physical.c'))
    for p in paths:head(p);v=item(p);scientific.append(v);catalog[v['path']]=v
    for rel,sha in old['preserved'].items():v=item(ROOT/rel,sha);catalog[v['path']]=v
    v=item(ROOT/'benchmarks/phase60/engine.c',old['engine_sha256']);catalog[v['path']]=v
    binding={'experiment':'METH498 one minimum-source-prior learner/physical artifact','runtime':runtime,'native_modules':native_modules,'scientific':scientific,'records':records,
        'catalog':list(catalog.values()),'data':data,'preserved':old['preserved'],'engine_sha256':old['engine_sha256'],
        'source_admission':catalog[str(DOC/'ADMISSION_497_20261006.json')],'source495_admission':catalog[str(DOC/'ADMISSION_495_20261006.json')],
        'inherited_output_bytes':0,'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip(),
        'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},'preparation_seconds':time.monotonic()-START,
        'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'builder_wall_limit_seconds':90,'builder_host_limit_bytes':256<<20,'compiler_or_source_FFN_payload_current_inputs':False}
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(binding,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    guard();sha=item(DEST)['sha256'];guard()
    print(json.dumps({'binding':str(DEST),'sha256':sha,'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED,'process_instance':binding['process_instance']}))
except BaseException as exc:
    with FAIL.open('xb') as f:f.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
