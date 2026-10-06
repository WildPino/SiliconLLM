"""Sole metadata admission of actual independent target compilation executions."""
import datetime
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[2]; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_493_20261006.json'; FAIL=DOC/'meth493_first_finalizer_fault.json'
assert not DEST.exists() and not FAIL.exists()

def receipt(path): return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

try:
    rawpath=DOC/'meth493_weighted_targets_result.json'; completed=rawpath.exists()
    if not completed: rawpath=rawpath.with_suffix('.failure.json')
    raw=json.loads(rawpath.read_bytes()); retpath=DOC/'RETENTION_493_20261006.json'; ret=json.loads(retpath.read_bytes())
    registrations=[]; events=[]; resources=[]
    for label,record,path,cap in (('main',raw,rawpath,180),('audit',ret,retpath,300)):
        regpath=DOC/f'meth493_{label}_sole_first_registration.json'; reg=json.loads(regpath.read_bytes())
        assert reg['attempt']==1 and reg['process_instance']==record['process_instance']
        assert reg['record_sha256']==receipt(path)['sha256']
        assert reg['actual_exit_code']==0 if completed or label=='audit' else reg['actual_exit_code']!=0
        eventpath=ROOT/f'results/native_expert_scaling/meth493_{"audit_" if label=="audit" else ""}windows_terminal.json'
        event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        instance=event['instances'][0]
        assert instance['pid']==record['process_instance']['pid'] and instance['create_time_unix']==record['process_instance']['create_time_unix']
        for key in ('start_utc','end_utc'): assert datetime.datetime.fromisoformat(instance[key])==datetime.datetime.fromisoformat(record[key])
        registrations.append(receipt(regpath)); events.append(receipt(eventpath))
        if completed or label=='audit':
            terminalpath=Path(record['terminal_resource_path']); terminal=json.loads(terminalpath.read_bytes())
            assert terminal['raw_sha256']==receipt(path)['sha256'] and terminal['raw_path']==str(path)
            assert terminal['resource']['wall_seconds']<=cap and terminal['resource']['parent_peak_bytes']+terminal['resource']['native_peak_bytes']<=256<<20
            resources.append(receipt(terminalpath))
    assert ret['raw']['sha256']==receipt(rawpath)['sha256'] and ret['main_completed']==completed and all(ret['gates'].values())
    assert ret['main_windows_sha256']==events[0]['sha256']
    if completed:
        assert all(raw['gates'].values()) and (ret['unique_inputs_audited'],ret['occurrences_audited'],ret['cells_audited'],ret['views_audited'])==(17540,19962,128,768)
        assert ret['target_coordinates_audited']==17540*768 and ret['occurrence_product_coordinates_audited']==19962*768
        assert raw['native_calls']==raw['optimizer_updates']==raw['SVD_calls']==ret['native_calls']==ret['optimizer_updates']==ret['SVD_calls']==0
    gates={'sole_first_actual_main_and_audit':True,'source_raw_retention_and_product_scope_digest_link':True,
        'ALL_independent_retention_gates':True,'BOTH_actual_Windows_without_application_fault':True,
        'supervision_only_no_student_quality_speed_or_goal_promotion':True}
    value={'experiment':'METH493 full bank11 weighted-function source supervision admission','main_completed':completed,
        'raw':receipt(rawpath),'retention':receipt(retpath),'registrations':registrations,'Windows_events':events,
        'terminal_resources':resources,'gates':gates,'decision':ret['decision'],'scope':'Source target compiler only; goal ACTIVE/INCOMPLETE.'}
    with DEST.open('xb') as stream: stream.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'admission':str(DEST),'sha256':receipt(DEST)['sha256'],'gates':gates,'decision':ret['decision']}))
except BaseException as exc:
    with FAIL.open('xb') as stream: stream.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n').encode())
    raise
