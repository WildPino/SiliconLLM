"""Fresh metadata-only actual feature/identity/runtime binding, no field observation."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth497_binding.json';FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic();PROC=psutil.Process();PROC.cpu_affinity([0]);PEAK=HASHED=0
def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset);assert PEAK<=256<<20 and time.monotonic()-START<=90
def item(path,expected=None):
    global HASHED
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
    hashes={'meth496_binding.json':'830c69075edcbd8a1055351ef10f68e737a0be29e56f5fa2770e5de76e26c9ab',
        'meth496_readout_error_result.json':'a633e57526de03d6bf17848504e0b2318be6e80b4109f10c0dbbe3143e346456',
        'RETENTION_496_20261006.json':'984679b08f896483ec29381c972c8afef8c21f87bbf85ed49ad9aefe080c095b',
        'ADMISSION_496_20261006.json':'fa7a535528f0e913466ec83e163147ba2d31a5f95327b8b68dfb3fb7870a5e12',
        'ADMISSION_495_20261006.json':'de9e28638e3abc6b9a512c7fdf7c6e6751f4505d9501d1e6763fd8e98280f'}
    for name,sha in hashes.items():
        p=DOC/name;head(p);v=item(p,sha);records.append(v);catalog[v['path']]=v
    old=json.loads((DOC/'meth496_binding.json').read_bytes());admission=json.loads((DOC/'ADMISSION_496_20261006.json').read_bytes())
    assert admission['main_completed'] and all(admission['gates'].values()) and admission['raw']['sha256']==hashes['meth496_readout_error_result.json'] and admission['retention']['sha256']==hashes['RETENTION_496_20261006.json']
    source495=json.loads((DOC/'ADMISSION_495_20261006.json').read_bytes());assert source495['main_completed'] and all(source495['gates'].values())
    data={}
    for key in ('features','uid','occurrences'):
        v=old['data'][key];data[key]=item(v['path'],v['sha256']);catalog[data[key]['path']]=data[key]
    runtime=old['runtime']
    for files in [runtime['files']]+[v['files'] for v in runtime['packages'].values()]:
        for path,v in files.items():
            p=Path(path).resolve()
            if str(p) not in catalog:catalog[str(p)]=item(p,v['sha256'])
    scientific=[];paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth497*'))+[DOC/'METH_497_EXACT_FEATURE_RANK_PROTOCOL_20261006.md']
    paths.extend(ROOT/'benchmarks/native_expert_scaling'/name for name in ('meth494_operations.py','meth490_r1_operations.py'))
    for p in paths:head(p);v=item(p);scientific.append(v);catalog[v['path']]=v
    for rel,sha in old['preserved'].items():v=item(ROOT/rel,sha);catalog[v['path']]=v
    v=item(ROOT/'benchmarks/phase60/engine.c',old['engine_sha256']);catalog[v['path']]=v
    binding={'experiment':'METH497 one fixed exact dyadic field rank certificate','runtime':runtime,'scientific':scientific,'records':records,
        'catalog':list(catalog.values()),'data':data,'preserved':old['preserved'],'engine_sha256':old['engine_sha256'],
        'source_admission':catalog[str(DOC/'ADMISSION_496_20261006.json')],'inherited_output_bytes':0,
        'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip(),
        'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},'preparation_seconds':time.monotonic()-START,
        'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'builder_wall_limit_seconds':90,'builder_host_limit_bytes':256<<20,
        'bank_coefficients_targets_compiler_or_source_payload_current_inputs':False}
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(binding,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    guard();sha=item(DEST)['sha256'];guard()
    print(json.dumps({'binding':str(DEST),'sha256':sha,'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED,'process_instance':binding['process_instance']}))
except BaseException as exc:
    with FAIL.open('xb') as f:f.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
