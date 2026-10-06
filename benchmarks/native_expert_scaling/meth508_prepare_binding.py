"""Metadata-only current-merge and exact98.7MB source readout extent binding."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
import meth508_operations as O

DEST=O.BIND;assert not DEST.exists() and not DEST.with_suffix('.failure.json').exists()
start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);catalog={};hashed=peak=0
def guard():
    global peak
    peak=max(peak,proc.memory_info().peak_wset);assert peak<=512<<20 and time.monotonic()-start<=180
def add(path,expected=None):
    global hashed
    p=Path(path).resolve();key=str(p)
    if key not in catalog:
        before=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            while data:=f.read(8<<20):h.update(data);hashed+=len(data);guard()
        assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
        catalog[key]={'path':key,'bytes':before.st_size,'sha256':h.hexdigest()}
    if expected:assert catalog[key]['sha256']==expected,key
    return catalog[key]
try:
    previous=add(O.DOC/'meth507_binding.json','59f6ebee73460b58745a6637499ef1eb3f35ecf150db5c31da2f46bc6c4afc09');b=json.loads(Path(previous['path']).read_bytes())
    preserved={**b['preserved'],'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    for row in b['catalog']:
        p=Path(row['path']);rel=p.relative_to(O.ROOT).as_posix() if p.is_relative_to(O.ROOT) else None
        add(p,preserved[rel] if rel in preserved else row['sha256'])
    for name,sha in [('meth507_main_result.json','6b21f71416ffda30e9c5b8631aad3408515221b519590e09486a6f92463035ab'),
                     ('meth507_audit_result.json','439facabc4a201978065069f776f5355215330dcc8bde3d586f1f46616cd2476'),
                     ('COMPLETION_507_20261006.json','1f257b52d698e51c42df5564c8ffe330fe3f057f852c4919bc5b3c1a52cab441')]:add(O.DOC/name,sha)
    prep=add(O.DOC/'meth506_r3_preparation_result.json','695d5438942220e82431eb2cc64e39d95f39cf7bc63af834e2a16ca1eb718dab');a=json.loads(Path(prep['path']).read_bytes())['artifact']
    manifest=add(a['manifest'],a['manifest_sha256']);cfg,files,tensors=O.manifest(manifest['path'])
    head=tensors['shared.weight'];assert head[0:5]==(0,2,32128,768,0) and head[6]==0 and head[7]==32128*768
    assert Path(files[head[0]]).resolve()==Path(a['payload']).resolve()
    extent={'path':str(Path(a['payload']).resolve()),'file_bytes':a['bytes'],'offset':head[5],'bytes':98697216,
            'shape':[32128,768],'dtype':'<f4','sha256':'0c5ee029008c814daeae89420ff7345c474425847c018d7cef46c814fa7d4a6c',
            'retained_whole_SHA256_NOT_rehashed':a['sha256'],'manifest':manifest}
    original=add(O.DOC/'meth379_switch_tensor_binding_result.json');v=json.loads(Path(original['path']).read_bytes())['tensors']
    assert v['shared.weight']['sha256']==v['lm_head.weight']['sha256']==extent['sha256'] and v['shared.weight']['shape']==extent['shape']
    p=Path(extent['path']);before=p.stat();assert before.st_size==extent['file_bytes'];left=extent['bytes'];h=hashlib.sha256()
    with p.open('rb') as f:
        f.seek(extent['offset'])
        while left:
            data=f.read(min(left,8<<20));assert data;h.update(data);left-=len(data);hashed+=len(data);guard()
    assert h.hexdigest()==extent['sha256'] and (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
    scientific=[]
    for path in sorted((O.ROOT/'benchmarks/native_expert_scaling').glob('meth508*.py'))+[O.DOC/'METH_508_SOURCE_HEAD_PROTOCOL_20261006.md']:
        assert path.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+path.relative_to(O.ROOT).as_posix()],cwd=O.ROOT).replace(b'\r\n',b'\n');scientific.append(add(path))
    assert len(scientific)==5
    raw507=json.loads((O.DOC/'meth507_main_result.json').read_bytes());assert len(raw507['cases'])==96 and sum(c['comparable_steps'] for c in raw507['cases'])==996
    changed=subprocess.check_output(['git','diff','--name-only','91e56a3..e3441d5'],cwd=O.ROOT,text=True).splitlines();assert all(p.startswith('docs/research/') for p in changed)
    merge={'revision':'e3441d5','changed_tracked_paths':len(changed),'only_research_documentation':True,
           'engine_and_506_507_scientific_files_changed':False,'preserved_RESEARCH_INDEX_working_SHA256':preserved['docs/research/RESEARCH_INDEX.md']}
    value={**b,'experiment':'METH508 source-head fixed-state decomposition','previous507':previous,
           'scientific':scientific,'catalog':list(catalog.values()),'preserved':preserved,'source_head':extent,'post_merge':merge,
           'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),
           'raw507':catalog[str((O.DOC/'meth507_main_result.json').resolve())],
           'price':{'selected_array_bytes':b['price']['selected_array_bytes'],'source_head_extent_bytes':98697216,
                    'catalog_bytes':sum(v['bytes'] for v in catalog.values()),'catalog_files':len(catalog),
                    'source_head_F64_bytes':197394432,'maximum_comparable_steps_per_case':max(c['comparable_steps'] for c in raw507['cases']),
                    'two_source_head_MACs':2*996*32128*768,'maximum_retained_decisions':996,'first_pair_decompositions':31,
                    'main_audit_each_limits':{'seconds':180,'OS_peak_bytes':2<<30,'output_bytes':32<<20},
                    'whole_model_native_compiler_calls':0,'new_cohorts':0},
           'preparation_resource_before_serialization':{'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed},
           'process_instance':{'pid':proc.pid,'create_time_unix':proc.create_time()}}
    O.write(DEST,value);guard();print(json.dumps({'binding':add(DEST),'price':value['price'],'post_merge':merge,
                                               'resource_after_serialization':{'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed}}),flush=True)
except BaseException:
    O.write(DEST.with_suffix('.failure.json'),{'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed});raise
