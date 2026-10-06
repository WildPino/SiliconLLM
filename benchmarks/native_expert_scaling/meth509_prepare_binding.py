"""Metadata-only selected actual509 inputs; no shortlist/rank/source dot inspection."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil
import meth509_operations as O

DEST=O.BIND;assert not DEST.exists() and not DEST.with_suffix('.failure.json').exists()
start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);peak=hashed=0;catalog={}
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
        assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns);catalog[key]={'path':key,'bytes':before.st_size,'sha256':h.hexdigest()}
    if expected:assert catalog[key]['sha256']==expected,key
    return catalog[key]
try:
    old=add(O.DOC/'meth508_binding.json','906499fd54f132c045d2f27bb84f188b9548fa93af51781e8097499734a92e39');b=json.loads(Path(old['path']).read_bytes())
    # Actual used runtime and common operations only; no unused donor NPZ or full checkpoint.
    used={'benchmarks/native_expert_scaling/meth507_operations.py','benchmarks/native_expert_scaling/meth508_operations.py','benchmarks/phase60/engine.c',*b['preserved']}
    for row in b['catalog']:
        p=Path(row['path']);rel=p.relative_to(O.ROOT).as_posix() if p.is_relative_to(O.ROOT) else None
        if rel is None or rel in used or rel.startswith('results/native_expert_scaling/meth324_switch_reference/venv/'):
            add(p,row['sha256'])
    oracle=add(O.DOC/'meth508_main_result.json','cf7badf270be0fce0334e4e20f3355a122d05938deb2ccab42fa079e1a1bf241')
    add(O.DOC/'meth508_audit_result.json','e67f3cc1a5bb65ee39cf9364d68e33c9e2a4ce6c2a391c7d1ff02cfd8968e831')
    add(O.DOC/'COMPLETION_508_20261006.json','2a30be882e5c4365b99188af1615bd80b506b881fbe6861fb2dd96ef121073e5')
    r=json.loads(Path(oracle['path']).read_bytes());assert len(r['cases'])==96 and r['summary']['source_state_bridge_qualified']
    selected=[]
    for i,row in enumerate(b['selected']):
        assert (row['book'],row['index'])==divmod(i,4);add(row['wire']['path'],row['wire']['sha256'])
        selected.append({'book':row['book'],'index':row['index'],'wire':row['wire']})
    add(b['source_head']['manifest']['path'],b['source_head']['manifest']['sha256']);extent=b['source_head'];p=Path(extent['path']);before=p.stat();assert before.st_size==extent['file_bytes']
    h=hashlib.sha256();left=extent['bytes']
    with p.open('rb') as f:
        f.seek(extent['offset'])
        while left:
            data=f.read(min(left,8<<20));assert data;left-=len(data);h.update(data);hashed+=len(data);guard()
    assert h.hexdigest()==extent['sha256'] and (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
    scientific=[]
    for path in sorted((O.ROOT/'benchmarks/native_expert_scaling').glob('meth509*.py'))+[O.DOC/'METH_509_SHORTLIST_HEAD_PROTOCOL_20261006.md']:
        assert path.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+path.relative_to(O.ROOT).as_posix()],cwd=O.ROOT).replace(b'\r\n',b'\n');scientific.append(add(path))
    assert len(scientific)==5
    value={'experiment':'METH509 ONE top8 conditional head','freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),
           'previous508':old,'oracle508':oracle,'selected':selected,'source_head':extent,'scientific':scientific,'catalog':list(catalog.values()),
           'python':b['python'],'executable':b['executable'],'packages':b['packages'],'preserved':b['preserved'],'engine_sha256':b['engine_sha256'],
           'price':{'wire_bytes':sum(v['wire']['bytes'] for v in selected),'catalog_files':len(catalog),'catalog_bytes':sum(v['bytes'] for v in catalog.values()),
                    'source_head_extent_bytes':98697216,'maximum_wire_bytes':max(v['wire']['bytes'] for v in selected),'maximum_comparable_steps':15,
                    'K':8,'positions':996,'selected_source_MACs':6119424,'extra_logical_source_weight_bytes_per_position':24576,
                    'main_audit_each_limits':{'seconds':180,'OS_peak_bytes':2<<30,'output_bytes':32<<20},'new_model_C_compiler_corpus_NPZ_calls':0},
           'preparation_resource_before_serialization':{'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed}}
    O.write(DEST,value);guard();print(json.dumps({'binding':add(DEST),'price':value['price'],'resource_after_serialization':{'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed}}),flush=True)
except BaseException:O.write(DEST.with_suffix('.failure.json'),{'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed});raise
