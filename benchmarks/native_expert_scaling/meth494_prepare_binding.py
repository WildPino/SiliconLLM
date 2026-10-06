"""Fresh direct prior dependencies and original per-tensor extents; no NumPy or values."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[2]; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth494_binding.json'; FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic(); PROC=psutil.Process(); PROC.cpu_affinity([0]); PEAK=HASHED=0

def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset); assert PEAK<=256<<20 and time.monotonic()-START<=90

def item(path,expected=None):
    global HASHED
    path=Path(path).resolve(); before=path.stat(); h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(4<<20):h.update(data); HASHED+=len(data); guard()
    after=path.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    v={'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}
    if expected: assert v['sha256']==expected,str(path)
    return v

def head(path):
    path=Path(path); saved=subprocess.check_output(['git','show','HEAD:'+path.relative_to(ROOT).as_posix()],cwd=ROOT,timeout=5)
    assert path.read_bytes().replace(b'\r\n',b'\n')==saved.replace(b'\r\n',b'\n'); guard()

try:
    old=item(DOC/'meth493_r2_binding.json','884db0ede443ce5f1e058949c951677de022eb1265a51f6b6b8c72eb631954b0')
    b=json.loads(Path(old['path']).read_bytes()); records=[old]; catalog={old['path']:old}
    required={Path(v['path']).resolve() for v in b['records']+b['scientific']+b['rational_runtime']}
    required.update(Path(v).resolve() for v in b['runtime']['files'])
    required.update(Path(b[k]['path']).resolve() for k in ('qualified_source_raw','qualified_source_retention'))
    query=b['weighted_source']['query_inputs.npy']; required.add(Path(query['path']).resolve())
    for v in b['catalog']:
        p=Path(v['path']).resolve()
        if p in required and str(p) not in catalog: catalog[str(p)]=item(p,v['sha256'])
    assert required<=set(Path(v) for v in catalog)
    for name in ('meth493_r2_weighted_targets_result.json','RETENTION_493_R2_20261006.json','ADMISSION_493_R2_20261006.json','meth380_switch_base128_export_result.json'):
        p=DOC/name; head(p); records.append(item(p))
    admission=json.loads((DOC/'ADMISSION_493_R2_20261006.json').read_bytes())
    assert all(admission['gates'].values()) and admission['decision']=='FULL_BANK11_WEIGHTED_SOURCE_TARGETS_INDEPENDENTLY_ADMITTED'
    raw=json.loads((DOC/'meth493_r2_weighted_targets_result.json').read_bytes()); assert all(raw['gates'].values())
    inventory={Path(v['path']).name:v for v in raw['output_inventory']}
    uid=item(inventory['uid_info.bin']['path'],inventory['uid_info.bin']['sha256']); target=item(inventory['weighted_targets.bin']['path'],inventory['weighted_targets.bin']['sha256'])
    catalog[uid['path']]=uid; catalog[target['path']]=target
    source=json.loads((DOC/'meth380_switch_base128_export_result.json').read_bytes()); assert all(source['gates'].values())
    artifact=source['artifact']; manifest=item(artifact['manifest'],artifact['manifest_sha256']); catalog[manifest['path']]=manifest
    data=Path(manifest['path']).read_bytes(); assert data[:8]==b'SWI8A001'; entries=source['tensors']; pos=72
    def text():
        global pos
        n=struct.unpack_from('<I',data,pos)[0]; pos+=4; v=data[pos:pos+n].decode('utf8'); pos+=n; return v
    assert Path(text()).resolve()==Path(artifact['payload']).resolve()
    for name,e in sorted(entries.items()):
        assert text()==name; v=struct.unpack_from('<5I3Q',data,pos); pos+=44
        assert v==(0,len(e['shape']),e['shape'][0],e['shape'][1] if len(e['shape'])==2 else 1,e['encoding'],e['offset'],e['scale_offset'],e['elements'])
    assert pos==len(data) and len(entries)==3320
    extents=[]
    for expert in range(128):
        for kind,shape in (('wi',(3072,768)),('wo',(768,3072))):
            name=f'decoder.block.11.layer.2.mlp.experts.expert_{expert}.{kind}.weight'; e=entries[name]
            assert tuple(e['shape'])==shape and e['encoding']==1 and e['bytes']==2359296
            for suffix,offset,size,sha in (('code',e['offset'],e['bytes'],e['sha256']),('scale',e['scale_offset'],e['scale_bytes'],e['scale_sha256'])):
                assert size==(2359296 if suffix=='code' else 4*shape[0])
                extents.append({'path':str(Path(artifact['payload']).resolve()),'offset':offset,'bytes':size,'sha256':sha,
                    'name':name+'.'+suffix,'expert':expert,'kind':kind,'component':suffix,'shape':list(shape)})
    e=entries['decoder.block.11.layer.2.mlp.router.classifier.weight']; assert e['encoding']==0 and e['shape']==[128,768]
    extents.append({'path':str(Path(artifact['payload']).resolve()),'offset':e['offset'],'bytes':e['bytes'],'sha256':e['sha256'],
        'name':'bank11.router','expert':None,'kind':'router','component':'code','shape':[128,768]})
    assert len(extents)==513
    for v in extents:
        p=Path(v['path']); before=p.stat(); h=hashlib.sha256(); remaining=v['bytes']
        with p.open('rb') as f:
            f.seek(v['offset'])
            while remaining:
                chunk=f.read(min(1<<20,remaining)); assert chunk; h.update(chunk); remaining-=len(chunk); HASHED+=len(chunk); guard()
        after=p.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        assert before.st_size==artifact['bytes'] and h.hexdigest()==v['sha256'],v['name']
    new=[]
    for p in sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth494*'))+[DOC/'METH_494_HYBRID_DONOR_PRIOR_PROTOCOL_20261006.md']:
        head(p); new.append(item(p))
    for v in records+new: catalog[v['path']]=v
    for rel,sha in b['preserved'].items(): catalog[str(ROOT/rel)]=item(ROOT/rel,sha)
    catalog[str(ROOT/'benchmarks/phase60/engine.c')]=item(ROOT/'benchmarks/phase60/engine.c',b['engine_sha256'])
    b['records']+=records; b['scientific']+=new
    b.update(experiment='METH494 Gaussian hybrid donor prior',catalog=list(catalog.values()),data={},
        prior_inputs={'query':query,'uid':uid,'weighted_targets_future_only':target},source_extents=extents,
        historical_payload=artifact,inherited_output_bytes=0,
        freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip(),
        preparation_seconds=time.monotonic()-START,preparation_peak_bytes=PEAK,preparation_hashed_bytes=HASHED,
        builder_wall_limit_seconds=90,builder_host_limit_bytes=256<<20)
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],'seconds':time.monotonic()-START,
        'peak_bytes':PEAK,'hashed_bytes':HASHED,'source_extents':len(extents)}))
except BaseException as exc:
    with FAIL.open('xb') as f:f.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc(),
        'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
