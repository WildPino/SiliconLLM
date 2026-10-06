"""Current decomposition dependencies only; no numerical values/control observation."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil
ROOT=Path(__file__).resolve().parents[2]; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth496_binding.json'; FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic(); PROC=psutil.Process(); PROC.cpu_affinity([0]); PEAK=HASHED=0
def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset);assert PEAK<=256<<20 and time.monotonic()-START<=90
def item(path,expected=None):
    global HASHED
    p=Path(path).resolve(); before=p.stat(); h=hashlib.sha256()
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
    hashes={'meth495_binding.json':'c2fac85bb7ace619f2de0031a53dfc30acdc8800824d5bc9ceb811fe77c4f3d4',
        'meth495_weighted_hybrid_fit_result.json':'6bb28ccf065afe0a132a0f9893a6d6e5f42395a7df3206ceea0cf9a473cd5a1b',
        'RETENTION_495_20261006.json':'eb8523f9c86a7864fade4e49f1f56bc938d80795e0e0322925e65ab68874e342',
        'ADMISSION_495_20261006.json':'de9e28638e3abc6b9a512c7fdf7c6e6751f4505d9501d51d1e6763fd8e98280f'}
    for name,sha in hashes.items():
        p=DOC/name;head(p);v=item(p,sha);records.append(v);catalog[v['path']]=v
    old=json.loads((DOC/'meth495_binding.json').read_bytes()); raw=json.loads((DOC/'meth495_weighted_hybrid_fit_result.json').read_bytes())
    admission=json.loads((DOC/'ADMISSION_495_20261006.json').read_bytes());assert admission['main_completed'] and all(admission['gates'].values())
    inventory={Path(v['path']).name:v for v in raw['output_inventory']}; data={}
    for key,name in (('features','features.bin'),('predictions','predictions.bin'),('coefficients','fitted_coefficients.bin'),('bank','bank.bin')):
        v=inventory[name];data[key]=item(v['path'],v['sha256']);catalog[data[key]['path']]=data[key]
    for key,oldkey in (('uid','uid'),('occurrences','occurrences'),('targets','targets')):
        v=old['fit_inputs'][oldkey];data[key]=item(v['path'],v['sha256']);catalog[data[key]['path']]=data[key]
    runtime=old['runtime'];required={Path(v).resolve() for v in runtime['files']}
    for package in runtime['packages'].values():required.update(Path(v).resolve() for v in package['files'])
    for v in old['catalog']:
        p=Path(v['path']).resolve()
        if p in required and str(p) not in catalog:catalog[str(p)]=item(p,v['sha256'])
    for files in [runtime['files']]+[v['files'] for v in runtime['packages'].values()]:
        for path,v in files.items():
            p=Path(path).resolve()
            if str(p) not in catalog:catalog[str(p)]=item(p,v['sha256'])
    assert required<=set(Path(p) for p in catalog)
    scientific=[]
    paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth496*'))+[DOC/'METH_496_READOUT_ERROR_PROTOCOL_20261006.md']
    paths.extend(ROOT/'benchmarks/native_expert_scaling'/name for name in ('meth494_operations.py','meth490_r1_operations.py'))
    for p in paths:head(p);v=item(p);scientific.append(v);catalog[v['path']]=v
    for rel,sha in old['preserved'].items():v=item(ROOT/rel,sha);catalog[v['path']]=v
    v=item(ROOT/'benchmarks/phase60/engine.c',old['engine_sha256']);catalog[v['path']]=v
    binding={'experiment':'METH496 unchanged-artifact error decomposition','runtime':runtime,'scientific':scientific,'records':records,
        'catalog':list(catalog.values()),'data':data,'preserved':old['preserved'],'engine_sha256':old['engine_sha256'],
        'source_admission':catalog[str(DOC/'ADMISSION_495_20261006.json')],'inherited_output_bytes':0,
        'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip(),
        'preparation_seconds':time.monotonic()-START,'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,
        'builder_wall_limit_seconds':90,'builder_host_limit_bytes':256<<20,'native_compiler_or_source_payload_current_inputs':False}
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(binding,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED}))
except BaseException as exc:
    with FAIL.open('xb') as f:f.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
