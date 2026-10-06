"""Sole metadata finalizer; no model, control, main or audit replay."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';DEST=DOC/'ADMISSION_488_20261006.json'
assert not DEST.exists() and sys.flags.optimize==0
def item(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4<<20):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
rawpath=DOC/'meth488_shared_integer_backend_result.json'
if not rawpath.exists():rawpath=rawpath.with_suffix('.failure.json')
raw=json.loads(rawpath.read_bytes());retpath=DOC/'RETENTION_488_20261006.json';ret=json.loads(retpath.read_bytes());assert all(ret['gates'].values()) and ret['raw']['sha256']==item(rawpath)['sha256']
records=[]
for label,scientific,source in [('main',raw,rawpath),('audit',ret,retpath)]:
    regpath=DOC/f'meth488_{label}_sole_first_registration.json';reg=json.loads(regpath.read_bytes());assert reg['attempt']==1 and reg['actual_exit_code']==(1 if label=='main' and 'failure' in raw else 0)
    assert reg['process_instance']==scientific['process_instance'] and reg['record_sha256']==item(source)['sha256']
    winpath=ROOT/f'results/native_expert_scaling/meth488_{"audit_" if label=="audit" else ""}windows_terminal.json';win=json.loads(winpath.read_bytes())
    assert win['query_available'] and not win['matching_scientific_events'] and win['instances'][0]['pid']==scientific['process_instance']['pid'] and win['instances'][0]['create_time_unix']==scientific['process_instance']['create_time_unix']
    records.extend([item(regpath),item(winpath)])
value={'experiment':'METH488 complete sole execution and independent admission','main_completed':ret['main_completed'],
       'raw':item(rawpath),'retention':item(retpath),'records':records,'gates':{'sole_first_actual_main_and_audit':True,'raw_retention_digest_link':True,'ALL_retention_gates':True,'BOTH_actual_Windows_instance_without_fault':True},
       'decision':'Retain exactly the completed gates. If main failed, no backend promotion; if speed failed, close this fixed layout. Full goal remains active/incomplete.'}
with DEST.open('xb') as f:f.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':str(DEST),'sha256':item(DEST)['sha256'],'main_completed':value['main_completed'],'gates':value['gates']}))
