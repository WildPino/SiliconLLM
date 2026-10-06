"""Sole R1 fresh metadata freeze; original first fault/control/partials are immutable."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[2]; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth493_r1_binding.json'; FAIL=DEST.with_suffix('.failure.json')
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
    inherited=item(DOC/'meth493_binding.json','d3b7f101b86a8bfb25a12aaa4389da63e9ebb216a26f93b9b5e18d22c2f2e461')
    binding=json.loads(Path(inherited['path']).read_bytes())
    catalog={v['path']:item(v['path'],v['sha256']) for v in binding['catalog']}
    records=[inherited]
    for name in ('meth493_weighted_targets_result.failure.json','meth493_main_sole_first_registration.json',
                 'RETENTION_493_20261006.json','meth493_audit_sole_first_registration.json','ADMISSION_493_20261006.json',
                 'meth493_role_join_first_fault_diagnosis.json'):
        path=DOC/name; head(path); records.append(item(path))
    reg=json.loads((DOC/'meth493_main_sole_first_registration.json').read_bytes())
    assert reg['actual_exit_code']==1 and reg['record_sha256']=='37a81a2d502c574ffbef9c5f3c1dfecfdfc6e142fb076806513e9c27813ca1d1'
    admission=json.loads((DOC/'ADMISSION_493_20261006.json').read_bytes())
    assert all(admission['gates'].values()) and admission['main_completed'] is False
    ret=json.loads((DOC/'RETENTION_493_20261006.json').read_bytes())
    assert all(ret['gates'].values()) and ret['numerical_imports'] is False and ret['main_completed'] is False
    controls=None
    for v in reg['partial_inventory']:
        value=item(v['path'],v['sha256']); assert value['bytes']==v['bytes']; catalog[value['path']]=value
        if Path(value['path']).name=='rounding_controls.json': controls=value
    assert controls is not None
    for name in ('meth493_windows_terminal.json','meth493_audit_windows_terminal.json'):
        path=ROOT/'results/native_expert_scaling'/name; value=item(path); catalog[value['path']]=value
        event=json.loads(path.read_bytes()); assert event['query_available'] and not event['matching_scientific_events']
    new=[]
    for path in sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth493_r1*'))+[DOC/'METH_493_R1_ROLE_MAPPING_PROTOCOL_20261006.md']:
        head(path); new.append(item(path))
    for v in records+new: catalog[v['path']]=v
    binding['records']+=records; binding['scientific']+=new
    binding.update(experiment='METH493-R1 complete weighted targets with source-role translation',catalog=list(catalog.values()),
        retained_original_controls=controls,inherited_output_bytes=0,
        freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip(),
        preparation_seconds=time.monotonic()-START,preparation_peak_bytes=PEAK,preparation_hashed_bytes=HASHED,
        builder_wall_limit_seconds=90,builder_host_limit_bytes=256<<20,
        original_builder_actual_seconds=67.67200000000594,combined_builder_wall_limit_seconds=157.67200000000594,
        source_role_rule={'472': 'validation iff64<=book<128, otherwise development0',
                          '479': 'development0 below64, consumedvalidation1 below128, extension2 otherwise'},
        new_native_calls=0,new_optimizer_updates=0)
    guard()
    with DEST.open('xb') as stream: stream.write((json.dumps(binding,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],
        'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED}))
except BaseException as exc:
    with FAIL.open('xb') as stream:
        stream.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc(),
            'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
