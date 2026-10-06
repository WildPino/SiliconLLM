"""Metadata-only binding of actual METH500 inputs, no numerical observation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth500_binding.json';FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic();PROC=psutil.Process();PROC.cpu_affinity([0]);PEAK=HASHED=0
def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset);assert PEAK<=256<<20 and time.monotonic()-START<=180
def item(path,expected=None):
    global HASHED
    p=Path(path).resolve();s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while d:=f.read(4<<20):h.update(d);HASHED+=len(d);guard()
    assert (p.stat().st_size,p.stat().st_mtime_ns)==(s.st_size,s.st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()}
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
try:
    catalog={};records=[];scientific=[]
    def add(v):
        d=item(v['path'],v['sha256']);catalog[d['path']]=d;return d
    old=json.loads((DOC/'meth499_binding.json').read_bytes())
    for name in ['meth499_binding.json','ADMISSION_499_20261006.json','meth499_factor_fit_result.json','meth380_switch_base128_export_result.json','meth472_prospective_bindings.json']:
        p=DOC/name;head(p);v=item(p);records.append(v);catalog[v['path']]=v
    admission=json.loads((DOC/'ADMISSION_499_20261006.json').read_bytes())
    assert admission['main_completed'] and all(admission['gates'].values())
    for k in ['raw','retention']:add(admission[k])
    raw=json.loads((DOC/'meth499_factor_fit_result.json').read_bytes());inv={Path(v['path']).name:v for v in raw['output_inventory']}
    data={k:add(old['data'][k]) for k in ['uid','occurrences']}
    data.update(inputs=add(inv['inputs.bin']),targets=add(inv['unweighted_targets.bin']))
    src=json.loads((DOC/'meth472_prospective_bindings.json').read_bytes())['original_artifact']
    data['payload']=add({'path':src['payload'],'sha256':src['sha256']})
    data['manifest']=add({'path':src['manifest'],'sha256':src['manifest_sha256']})
    runtime=old['runtime']
    for files in [runtime['files']]+[v['files'] for v in runtime['packages'].values()]:
        for path,v in files.items():add({'path':path,'sha256':v['sha256']})
    paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth500*'))+[ROOT/'benchmarks/native_expert_scaling/meth490_r1_operations.py',DOC/'METH_500_OVERLAPPING_SOURCE_PROTOCOL_20261006.md']
    for p in paths:head(p);v=item(p);scientific.append(v);catalog[v['path']]=v
    for rel,sha in old['preserved'].items():add({'path':str(ROOT/rel),'sha256':sha})
    add({'path':str(ROOT/'benchmarks/phase60/engine.c'),'sha256':old['engine_sha256']})
    b={'experiment':'METH500 overlapping source branches; supersedes unimplemented query-plan500',
       'runtime':runtime,'catalog':list(catalog.values()),'records':records,'scientific':scientific,'data':data,
       'preserved':old['preserved'],'engine_sha256':old['engine_sha256'],'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},'preparation_seconds':time.monotonic()-START,
       'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'argv':sys.argv,
       'new_information':'Complete original-coordinate ReLU/A16 supports and partial-source functions on fixed original UID inputs',
       'main_and_audit_limits':{'seconds_each':900,'OS_peak_bytes_each':2<<30,'main_output_bytes':1<<30,'audit_output_bytes':128<<20},
       'new_model_native_GPU_calls':0,'optimizer_updates':0}
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    v=item(DEST);guard();print(json.dumps({'binding':v,'resource':{'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},'process_instance':b['process_instance']}))
except BaseException as e:
    with FAIL.open('xb') as f:f.write((json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
