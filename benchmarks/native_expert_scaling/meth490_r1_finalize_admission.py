"""Metadata-only final link of sole actual executions, independent admission and events."""
import hashlib
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_490_R1_20261006.json';assert not DEST.exists()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def receipt(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}
mainpath=DOC/'meth490_r1_root_mass_result.json'
completed=mainpath.exists()
if not completed:mainpath=mainpath.with_suffix('.failure.json')
main=json.loads(mainpath.read_bytes());retpath=DOC/'RETENTION_490_R1_20261006.json';ret=json.loads(retpath.read_bytes())
registrations=[];events=[]
for label,raw in [('main',main),('audit',ret)]:
    p=DOC/('meth490_r1_'+label+'_sole_first_registration.json');reg=json.loads(p.read_bytes())
    assert reg['attempt']==1 and reg['process_instance']==raw['process_instance']
    assert reg['record_sha256']==digest(mainpath if label=='main' else retpath)
    assert reg['actual_exit_code']==0 if label=='audit' or completed else reg['actual_exit_code']!=0
    event=ROOT/('results/native_expert_scaling/meth490_r1_'+('audit_' if label=='audit' else '')+'windows_terminal.json')
    v=json.loads(event.read_bytes());assert v['query_available'] and v['query_error'] is None and not v['matching_scientific_events']
    assert v['instances'][0]['pid']==raw['process_instance']['pid'] and v['instances'][0]['create_time_unix']==raw['process_instance']['create_time_unix']
    assert all(datetime.datetime.fromisoformat(v['instances'][0][key])==datetime.datetime.fromisoformat(raw[key]) for key in ('start_utc','end_utc'))
    registrations.append(receipt(p));events.append(receipt(event))
assert ret['raw']['sha256']==digest(mainpath) and all(ret['gates'].values()) and ret['main_windows_sha256']==events[0]['sha256']
if completed:
    assert all(main['gates'].values()) and ret['optimizer_updates_audited']==384 and ret['unique_inputs_audited']==238872
    assert ret['feasibility']==main['feasibility'] and ret['decision']==main['decision']
gates={'sole_first_actual_main_and_audit':True,'raw_retention_digest_link':True,'ALL_independent_retention_gates':True,
       'BOTH_actual_Windows_instances_without_application_fault':True,'no_incomplete_fit_or_source_fidelity_promotion':True}
assert not DEST.exists()
value={'inherited_first_admission':receipt(DOC/'ADMISSION_490_20261006.json'),'experiment':'METH490 complete sole-execution independent admission','main_completed':completed,'raw':receipt(mainpath),'retention':receipt(retpath),
       'registrations':registrations,'Windows_events':events,'gates':gates,'decision':ret['decision'],
       'scope':'Only the observed fixed root recipe/retained first fault; full goal remains active/incomplete.'}
with DEST.open('xb') as f:f.write((json.dumps(value,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':str(DEST),'sha256':digest(DEST),'gates':gates,'decision':value['decision']}))
