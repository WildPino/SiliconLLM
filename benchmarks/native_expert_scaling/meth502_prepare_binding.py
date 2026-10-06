"""Metadata-only actual support/ownership binding; no source payload or old main."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth502_binding.json';FAIL=DEST.with_suffix('.failure.json');assert not DEST.exists() and not FAIL.exists()
START=time.monotonic();P=psutil.Process();P.cpu_affinity([0]);PEAK=HASHED=0
def item(path,expected=None):
    global PEAK,HASHED
    p=Path(path).resolve();s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while d:=f.read(4<<20):
            h.update(d);HASHED+=len(d);PEAK=max(PEAK,P.memory_info().peak_wset);assert PEAK<=128<<20 and time.monotonic()-START<=60
    assert (p.stat().st_size,p.stat().st_mtime_ns)==(s.st_size,s.st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()}
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
try:
    old=json.loads((DOC/'meth501_binding.json').read_bytes());ad=json.loads((DOC/'ADMISSION_501_20261006.json').read_bytes())
    assert ad['main_completed'] and all(ad['gates'].values())
    raw=json.loads((DOC/'meth501_support_cover_result.json').read_bytes());inv={Path(v['path']).name:v for v in raw['output_inventory']}
    catalog={};records=[];scientific=[]
    def add(v):
        d=item(v['path'],v['sha256']);catalog[d['path']]=d;return d
    for name in ['meth501_binding.json','ADMISSION_501_20261006.json','meth501_support_cover_result.json','RETENTION_501_20261006.json']:
        p=DOC/name;head(p);v=item(p);records.append(v);catalog[v['path']]=v
    for k in ['raw','retention']:add(ad[k])
    data={k:add(v) for k,v in old['data'].items()}
    data.update({k:add(inv[v]) for k,v in {'base':'base_unions.npy','final':'final_masks.npy','assignment':'development_assignment.npy','order':'development_order.bin'}.items()})
    data['raw501']=add({'path':str(DOC/'meth501_support_cover_result.json'),'sha256':ad['raw']['sha256']})
    runtime=old['runtime'];runtime['packages'].pop('threadpoolctl',None)
    for files in [runtime['files']]+[v['files'] for v in runtime['packages'].values()]:
        for path,v in files.items():add({'path':path,'sha256':v['sha256']})
    paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth502*'))+[ROOT/'benchmarks/native_expert_scaling/meth490_r1_operations.py',DOC/'METH_502_VARIABLE_COVER_PROTOCOL_20261006.md']
    for p in paths:head(p);v=item(p);scientific.append(v);catalog[v['path']]=v
    for rel,sha in old['preserved'].items():add({'path':str(ROOT/rel),'sha256':sha})
    add({'path':str(ROOT/'benchmarks/phase60/engine.c'),'sha256':old['engine_sha256']})
    b={'experiment':'METH502 exact support cover ONE greedy witness','runtime':runtime,'catalog':list(catalog.values()),'records':records,'scientific':scientific,
       'data':data,'preserved':old['preserved'],'engine_sha256':old['engine_sha256'],
       'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'process_instance':{'pid':P.pid,'create_time_unix':P.create_time()},'preparation_seconds':time.monotonic()-START,
       'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'argv':sys.argv,
       'no_source_payload_or_new_source_model_native_GPU_gradient':True}
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    desc=item(DEST);print(json.dumps({'binding':desc,'resource':{'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},'process_instance':b['process_instance']}))
except BaseException:
    with FAIL.open('xb') as f:f.write((json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
