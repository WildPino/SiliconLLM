"""Freeze only actual 495 input dependencies; no numerical feature/fit observation."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil
ROOT=Path(__file__).resolve().parents[2]; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth495_binding.json'; FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic(); PROC=psutil.Process(); PROC.cpu_affinity([0]); PEAK=HASHED=0
def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset); assert PEAK<=256<<20 and time.monotonic()-START<=90
def item(path,expected=None):
    global HASHED
    p=Path(path).resolve(); before=p.stat(); h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(4<<20): h.update(data); HASHED+=len(data); guard()
    after=p.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    if expected: assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':before.st_size,'sha256':h.hexdigest()}
def head(path):
    p=Path(path); saved=subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT,timeout=5)
    assert p.read_bytes().replace(b'\r\n',b'\n')==saved.replace(b'\r\n',b'\n'); guard()
try:
    old=item(DOC/'meth494_binding.json','91fb15d7db330d6a36b2f25809bfa769b99fb62fe9df3b9c074d884c25b55b58')
    b=json.loads(Path(old['path']).read_bytes()); catalog={old['path']:old}; records=[old]
    required={Path(v['path']).resolve() for v in b['scientific']+b['records']+b['rational_runtime']}
    required.update(Path(v).resolve() for v in b['runtime']['files'])
    required.update(Path(b[k]['path']).resolve() for k in ('qualified_source_raw','qualified_source_retention','compiler'))
    required.update(Path(b['prior_inputs'][k]['path']).resolve() for k in ('query','uid','weighted_targets_future_only'))
    for v in b['catalog']:
        p=Path(v['path']).resolve()
        if p in required and str(p) not in catalog: catalog[str(p)]=item(p,v['sha256'])
    # 494 did not compile; its compiler snapshot was historical, not fresh.
    compiler_root=Path(b['compiler']['path']).resolve().parent.parent
    compiler_snapshot=[]
    for v in b['historical_catalog']:
        p=Path(v['path']).resolve()
        if p.is_relative_to(compiler_root):
            if str(p) not in catalog:catalog[str(p)]=item(p,v['sha256'])
            compiler_snapshot.append(catalog[str(p)])
    assert len(compiler_snapshot)==5353
    assert required<=set(Path(p) for p in catalog)
    fresh={}
    for name,expected in (('meth494_hybrid_prior_result.json','ab3501d8add8164ef0adb2ccc2c2512ea56166a491afff0b24f887752124e527'),
        ('RETENTION_494_20261006.json','4436ef88bd926c6c5b42e7ebbec6db92ccfe31bbf099c80134f319c3fee72029'),
        ('ADMISSION_494_20261006.json','432054f90a0b426b386001e391609851981e3fd111d506005dd13ea588c2eb59')):
        p=DOC/name; head(p); v=item(p,expected); records.append(v); fresh[name]=json.loads(p.read_bytes())
    admission=fresh['ADMISSION_494_20261006.json']; assert all(admission['gates'].values()) and admission['main_completed']
    assert admission['decision']=='HYBRID_GAUSSIAN_PRIOR_AND_LOGICAL_COST_INDEPENDENTLY_ADMITTED'
    prior=fresh['meth494_hybrid_prior_result.json']; assert all(prior['gates'].values())
    inventory={Path(v['path']).name:v for v in prior['output_inventory']}
    fit={}
    for key,name in (('bank','bank_prior.bin'),('C0','C0_prior.bin'),('geometry','geometry.bin'),('calibration','calibration.json')):
        v=inventory[name]; fit[key]=item(v['path'],v['sha256']); catalog[fit[key]['path']]=fit[key]
    source=json.loads((DOC/'meth493_r2_weighted_targets_result.json').read_bytes()); assert all(source['gates'].values())
    occ={Path(v['path']).name:v for v in source['output_inventory']}['occurrences.bin']; fit['occurrences']=item(occ['path'],occ['sha256'])
    catalog[fit['occurrences']['path']]=fit['occurrences']
    for key,oldkey in (('query','query'),('uid','uid'),('targets','weighted_targets_future_only')): fit[key]=b['prior_inputs'][oldkey]
    new=[]
    for p in sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth495*'))+[DOC/'METH_495_WEIGHTED_HYBRID_FIT_PROTOCOL_20261006.md']:
        head(p); new.append(item(p))
    for v in records+new: catalog[v['path']]=v
    for rel,sha in b['preserved'].items(): catalog[str(ROOT/rel)]=item(ROOT/rel,sha)
    catalog[str(ROOT/'benchmarks/phase60/engine.c')]=item(ROOT/'benchmarks/phase60/engine.c',b['engine_sha256'])
    b['scientific']+=new; b['records']+=records
    b.update(experiment='METH495 ONE fixed weighted hybrid learner and physical evaluation',catalog=list(catalog.values()),
        source_extents=[],data={},fit_inputs=fit,compiler_snapshot=compiler_snapshot,inherited_output_bytes=0,
        freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip(),
        preparation_seconds=time.monotonic()-START,preparation_peak_bytes=PEAK,preparation_hashed_bytes=HASHED,
        builder_wall_limit_seconds=90,builder_host_limit_bytes=256<<20)
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED}))
except BaseException as exc:
    with FAIL.open('xb') as f:f.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
