"""Fresh actual compiler/auditor inputs and runtime; unused historical assets stay in ancestry."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[2]; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth493_r2_binding.json'; FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic(); PROC=psutil.Process(); PROC.cpu_affinity([0]); PEAK=HASHED=0

def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset)
    assert PEAK<=256<<20 and time.monotonic()-START<=90

def item(path,expected=None):
    global HASHED
    path=Path(path).resolve(); before=path.stat(); h=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(4<<20): h.update(data); HASHED+=len(data); guard()
    after=path.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    result={'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}
    if expected: assert result['sha256']==expected,str(path)
    return result

def head(path):
    path=Path(path)
    saved=subprocess.check_output(['git','show','HEAD:'+path.relative_to(ROOT).as_posix()],cwd=ROOT,timeout=5)
    assert path.read_bytes().replace(b'\r\n',b'\n')==saved.replace(b'\r\n',b'\n'); guard()

try:
    inherited=item(DOC/'meth493_r1_binding.json','47cd764d4c4abe5f0dc6e0fcbef434e18cda676786b4583b127705592117a5f2')
    binding=json.loads(Path(inherited['path']).read_bytes()); history=binding['catalog']
    required={Path(v['path']).resolve() for v in binding['scientific']+binding['records']+binding['rational_runtime']}
    required.update(Path(v).resolve() for v in binding['runtime']['files'])
    binding['data']={k:v for k,v in binding['data'].items() if k in ('unique_inputs.bin','ownership.bin','query_links.bin','source_fields.bin')}
    required.update(Path(v['path']).resolve() for v in binding['data'].values())
    required.update(Path(v['path']).resolve() for v in binding['weighted_source'].values())
    required.update(Path(binding[k]['path']).resolve() for k in ('qualified_source_raw','qualified_source_retention','retained_original_controls'))
    catalog={}
    for v in history:
        path=Path(v['path']).resolve()
        if path in required and str(path) not in catalog: catalog[str(path)]=item(path,v['sha256'])
    assert required<=set(Path(k) for k in catalog), sorted(map(str,required-set(Path(k) for k in catalog)))
    # Historical payloads/binaries/captures are not read by this program. Their expected descriptors remain verbatim.
    records=[inherited]
    for name in ('meth493_r1_weighted_targets_result.failure.json','meth493_r1_main_sole_first_registration.json',
                 'RETENTION_493_R1_20261006.json','meth493_r1_audit_sole_first_registration.json','ADMISSION_493_R1_20261006.json'):
        path=DOC/name; head(path); records.append(item(path))
    reg=json.loads((DOC/'meth493_r1_main_sole_first_registration.json').read_bytes())
    assert reg['record_sha256']=='c8c77615d9a5ecafa8db7512b7b42fe1e0c3741b0bd83cc219e68523fd58a2fc' and reg['no_numerical_import_or_product']
    ret=json.loads((DOC/'RETENTION_493_R1_20261006.json').read_bytes()); admission=json.loads((DOC/'ADMISSION_493_R1_20261006.json').read_bytes())
    assert all(ret['gates'].values()) and all(admission['gates'].values()) and not ret['main_completed'] and not ret['numerical_imports']
    for v in reg['partial_inventory']: catalog[str(Path(v['path']).resolve())]=item(v['path'],v['sha256'])
    for name in ('meth493_r1_windows_terminal.json','meth493_r1_audit_windows_terminal.json'):
        path=ROOT/'results/native_expert_scaling'/name; catalog[str(path)]=item(path)
        event=json.loads(path.read_bytes()); assert event['query_available'] and not event['matching_scientific_events']
    for rel,sha in binding['preserved'].items(): catalog[str(ROOT/rel)]=item(ROOT/rel,sha)
    catalog[str(ROOT/'benchmarks/phase60/engine.c')]=item(ROOT/'benchmarks/phase60/engine.c',binding['engine_sha256'])
    new=[]
    for path in sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth493_r2*'))+[DOC/'METH_493_R2_DIRECT_DEPENDENCY_PROTOCOL_20261006.md']:
        head(path); new.append(item(path))
    for v in records+new: catalog[v['path']]=v
    binding['records']+=records; binding['scientific']+=new
    binding.update(experiment='METH493-R2 fresh direct inputs with retained historical provenance',catalog=list(catalog.values()),
        historical_catalog=history,historical_catalog_status='EXPECTED_DESCRIPTORS_FROM_FULLY_HASHED_R1_BINDING_NOT_FRESHLY_REHASHED_HERE',
        direct_catalog_bytes=sum(v['bytes'] for v in catalog.values()),
        unused_historical_bytes=sum(v['bytes'] for v in history if Path(v['path']).resolve() not in required),
        inherited_output_bytes=0,freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip(),
        preparation_seconds=time.monotonic()-START,preparation_peak_bytes=PEAK,preparation_hashed_bytes=HASHED,
        builder_wall_limit_seconds=90,builder_host_limit_bytes=256<<20,new_native_calls=0,new_optimizer_updates=0)
    guard()
    with DEST.open('xb') as stream: stream.write((json.dumps(binding,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],'seconds':time.monotonic()-START,
        'peak_bytes':PEAK,'hashed_bytes':HASHED,'direct_catalog_bytes':binding['direct_catalog_bytes'],
        'unused_historical_bytes':binding['unused_historical_bytes']}))
except BaseException as exc:
    with FAIL.open('xb') as stream:
        stream.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc(),
            'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
