"""Metadata/hash preparation only. Never import NumPy, run controls or inspect values."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth490_binding.json';start=time.monotonic();assert not DEST.exists()
def item(path,expected=None):
    p=Path(path).resolve();before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(4<<20):h.update(data);assert time.monotonic()-start<=120
    assert p.stat().st_size==before.st_size and p.stat().st_mtime_ns==before.st_mtime_ns
    v={'path':str(p),'bytes':before.st_size,'sha256':h.hexdigest()}
    if expected:assert v['sha256']==expected,str(p)
    return v
def head(p):
    p=Path(p);rel=p.relative_to(ROOT).as_posix()
    assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT).replace(b'\r\n',b'\n'),rel
old=DOC/'meth480_prospective_bindings.json';head(old);source=json.loads(old.read_bytes());prior=source['prior']
scientific=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth490*'))+[DOC/'METH_490_SEPARATE_ROOT_MASS_PROTOCOL_20261006.md']
for p in scientific:head(p)
scientific=list(map(item,scientific));data={k:item(v['path'],v['sha256']) for k,v in source['data_files'].items()}
records=[item(old),item(DOC/'meth479_r2_routing_supervision_result.json','c03efee574a62743ac214f0c94213f41e2cbb34db0b6db86ddd18ebab53d9200'),
         item(DOC/'RETENTION_479_R3_20261005.json','86c57ea27d948a366344466acb4eef86944c65899c1876ec9a630964aac28d3d'),
         item(DOC/'meth479_prospective_bindings.json','0523b1822b84f6e11b40025b3c677116042a06853f4094375d271df73a4016ac'),
         item(ROOT/'benchmarks/native_expert_scaling/meth479_r3_retention_audit.py','09430f9e819cdf77c3de90f49745b923c79fa9d1dce78155cde95c0947e8fd12')]
for v in records:head(v['path'])
runtime=prior['runtime'];catalog={}
def add(v):
    record=item(v['path'],v['sha256']);catalog[record['path']]=record
for v in scientific+records+list(data.values())+prior['compile_assets']+prior['system_files']+[prior['compiler']]:add(v)
for p,v in runtime['files'].items():add({'path':p,**v})
for name,v in runtime['packages'].items():
    for p,record in v['files'].items():add({'path':p,**record})
artifact=prior['artifact'];add({'path':artifact['payload'],'bytes':artifact['bytes'],'sha256':artifact['sha256']});add({'path':artifact['manifest'],'sha256':artifact['manifest_sha256']})
preserved={
 'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
 'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
 'docs/research/RESEARCH_INDEX.md':'639afab9f5330bb7170a2a8e6fad61759bb848839f0a86a6479ce688edd61f61'}
for p,sha in preserved.items():add(item(ROOT/p,sha))
engine=item(ROOT/'benchmarks/phase60/engine.c','ea3b9f17e2664b8191397a6afa4fabe48af3f583d5dfaf1583a734ba115203db');add(engine)
binding={'experiment':'METH490 one separate normalized conditional-mass root recipe','freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'scientific':scientific,'records':records,'data':data,'runtime':runtime,'compiler':prior['compiler'],'catalog':list(catalog.values()),
         'artifact':artifact,'engine_sha256':engine['sha256'],'preserved':preserved,'qualified_source_raw':records[1],'qualified_source_retention':records[2],
         'configuration':{'roots':12,'steps':32,'rate':.03,'optimizer':'Adam beta=.9/.999 epsilon=1e-8','source_only_n128':True,'main_seconds':600,'audit_seconds':300,'main_host_bytes':1<<30,'audit_host_bytes':512<<20,'output_bytes':64<<20},
         'preparation_seconds':time.monotonic()-start,'preparation_numeric_imports_controls_fits':0}
with DEST.open('xb') as f:f.write((json.dumps(binding,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],'catalog_files':len(catalog),'seconds':binding['preparation_seconds']}))
