"""Fresh actual unweighted reference/prior/native compilation inputs; metadata only."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth499_binding.json';FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic();PROC=psutil.Process();PROC.cpu_affinity([0]);PEAK=HASHED=0
def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset);assert PEAK<=256<<20 and time.monotonic()-START<=180
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
    records={};catalog={}
    hashes={'meth498_binding.json':'b85fd8a136e546f7d940f518811081b2b9e993554f11c84adb6039990524bf37',
        'ADMISSION_498_20261006.json':'bff9a2ed8533bc28a0fc1892151cbedd7fb1274da62103bddea09cc4f032bac6',
        'meth495_binding.json':'c2fac85bb7ace619f2de0031a53dfc30acdc8800824d5bc9ceb811fe77c4f3d4',
        'meth493_r2_binding.json':'884db0ede443ce5f1e058949c951677de022eb1265a51f6b6b8c72eb631954b0',
        'ADMISSION_493_R2_20261006.json':'aae3cd3bc182d16f805101bc5e0cd7c1e9be79891b5a48c14bf50341881bedda',
        'ADMISSION_494_20261006.json':'432054f90a0b426b386001e391609851981e3fd111d506005dd13ea588c2eb59',
        'ADMISSION_497_20261006.json':'6121e58ce83bb53fa12eaf3cda554c3d33288027c5ab7957ee35f97d780a35b5'}
    assert all(len(v)==64 for v in hashes.values())
    def record(path,expected):
        head(path);v=item(path,expected);records[v['path']]=catalog[v['path']]=v;return v
    for name,sha in hashes.items():record(DOC/name,sha)
    old=json.loads((DOC/'meth498_binding.json').read_bytes());s495=json.loads((DOC/'meth495_binding.json').read_bytes());s493=json.loads((DOC/'meth493_r2_binding.json').read_bytes())
    for v in old['records']:record(v['path'],v['sha256'])
    for name in ('ADMISSION_498_20261006.json','ADMISSION_493_R2_20261006.json','ADMISSION_494_20261006.json','ADMISSION_497_20261006.json'):
        a=json.loads((DOC/name).read_bytes());assert a['main_completed'] and all(a['gates'].values())
        for key in ('raw','retention'):record(a[key]['path'],a[key]['sha256'])
    prior=json.loads((DOC/'meth494_hybrid_prior_result.json').read_bytes());inv={Path(v['path']).name:v for v in prior['output_inventory']};data={}
    def add(key,desc):
        data[key]=item(desc['path'],desc['sha256']);catalog[data[key]['path']]=data[key]
    for key in ('uid','occurrences','features','targets','geometry','C0','bank','inputs','baseline_predictions'):add(key,old['data'][key])
    add('L0',inv['L0_prior.bin'])
    for key,name in (('queries','query_inputs.npy'),('references','reference_functions.npy')):add(key,s493['weighted_source'][name])
    runtime=old['runtime']
    for files in [runtime['files']]+[v['files'] for v in runtime['packages'].values()]:
        for path,v in files.items():
            p=Path(path).resolve()
            if str(p) not in catalog:catalog[str(p)]=item(p,v['sha256'])
    compiler_snapshot=[]
    for desc in s495['compiler_snapshot']:
        p=Path(desc['path']).resolve()
        if str(p) not in catalog:catalog[str(p)]=item(p,desc['sha256'])
        compiler_snapshot.append(catalog[str(p)])
    assert len(compiler_snapshot)==5353
    native_modules=[]
    for desc in old['native_modules']:
        if not desc['path'].lower().endswith('.dll'):continue
        p=Path(desc['path']).resolve()
        if str(p) not in catalog:catalog[str(p)]=item(p,desc['sha256'])
        native_modules.append(catalog[str(p)])
    scientific=[];paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth499*'))+[DOC/'METH_499_FACTOR_PROTOCOL_20261006.md']
    paths.extend(ROOT/'benchmarks/native_expert_scaling'/name for name in ('meth498_math.py','meth494_operations.py','meth490_r1_operations.py'))
    for p in paths:head(p);v=item(p);scientific.append(v);catalog[v['path']]=v
    for rel,sha in old['preserved'].items():v=item(ROOT/rel,sha);catalog[v['path']]=v
    v=item(ROOT/'benchmarks/phase60/engine.c',old['engine_sha256']);catalog[v['path']]=v
    b={'experiment':'METH499 unweighted-prior variable-mass pipeline','runtime':runtime,'compiler':catalog[str(Path(s495['compiler']['path']).resolve())],
        'compiler_snapshot':compiler_snapshot,'native_modules':native_modules,'scientific':scientific,'records':list(records.values()),
        'catalog':list(catalog.values()),'data':data,'preserved':old['preserved'],'engine_sha256':old['engine_sha256'],
        'source_role_rule':s493['source_role_rule'],'inherited_output_bytes':0,
        'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip(),
        'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},'preparation_seconds':time.monotonic()-START,
        'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'builder_wall_limit_seconds':180,'builder_host_limit_bytes':256<<20,
        'source_FFN_payload_current_input':False,'new_source_FFN_or_model_calls':0}
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    guard();sha=item(DEST)['sha256'];guard()
    print(json.dumps({'binding':str(DEST),'sha256':sha,'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED,'process_instance':b['process_instance']}))
except BaseException as exc:
    with FAIL.open('xb') as f:f.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
