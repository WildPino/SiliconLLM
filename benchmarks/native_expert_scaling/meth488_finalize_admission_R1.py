"""Metadata-only final linkage of original main, first audit and numbered recovery."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_488_R1_20261006.json'
assert not DEST.exists() and sys.flags.optimize==0

def item(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4<<20):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

retpath=DOC/'RETENTION_488_R1_20261006.json'
ret=json.loads(retpath.read_bytes())
assert all(ret['gates'].values()) and not ret['main_completed']
assert [ret['sources'][str(n)]['complete_cases'] for n in (128,256)]==[96,78]
records=[]
for label,source,win,exit_code in [
    ('main','meth488_shared_integer_backend_result.failure.json','meth488_windows_terminal.json',1),
    ('audit','RETENTION_488_20261006.failure.json','meth488_audit_windows_terminal.json',1),
    ('audit_R1','RETENTION_488_R1_20261006.json','meth488_audit_r1_windows_terminal.json',0)]:
    path=DOC/source;scientific=json.loads(path.read_bytes())
    regpath=DOC/f'meth488_{label}_sole_first_registration.json';reg=json.loads(regpath.read_bytes())
    assert reg['attempt']==1 and reg['actual_exit_code']==exit_code
    assert reg['process_instance']==scientific['process_instance'] and reg['record_sha256']==item(path)['sha256']
    winpath=ROOT/'results/native_expert_scaling'/win;events=json.loads(winpath.read_bytes())
    assert events['query_available'] and not events['matching_scientific_events']
    own=events['instances'][0]
    assert own['pid']==scientific['process_instance']['pid'] and own['create_time_unix']==scientific['process_instance']['create_time_unix']
    for key in ('start_utc','end_utc'):
        assert datetime.datetime.fromisoformat(own[key])==datetime.datetime.fromisoformat(scientific[key])
    records.extend([item(path),item(regpath),item(winpath)])
assert ret['raw']['sha256']==records[0]['sha256']
binding=DOC/'meth488_audit_r1_binding.json'
assert ret['extension_binding_sha256']==item(binding)['sha256']
value={'experiment':'METH488-R1 independent retained integer controls and complete case prefix',
       'main_completed':False,'sources':ret['sources'],'raw':ret['raw'],'retention':item(retpath),
       'extension_binding':item(binding),'records':records,
       'gates':{'ALL9_retention_gates':True,'sole_first_actual_main_audit_audit_R1':True,
                'ALL3_actual_PID_creation_aware_ISO_Windows_without_fault':True,
                'ALL11_integer_controls_and_174_complete_case_prefix_only':True,
                'no_complete192_or_speed_or_bootstrap_promotion':True},
       'decision':'Admit only exact arithmetic controls and completed174-case prefix. Original time stop, audit isolation failure and bootstrap discrepancy retained. Full goal active/incomplete.'}
with DEST.open('xb') as f:f.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':str(DEST),'sha256':item(DEST)['sha256'],'gates':value['gates']}))
