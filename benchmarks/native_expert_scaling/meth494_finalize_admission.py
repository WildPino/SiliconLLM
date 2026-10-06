"""Metadata admission of sole actual executions; no numerical replay."""
import datetime
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[2]; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_494_20261006.json'; FAIL=DOC/'meth494_first_finalizer_fault.json'
assert not DEST.exists() and not FAIL.exists()
def receipt(path): return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
try:
    rawpath=DOC/'meth494_hybrid_prior_result.json'; completed=rawpath.exists()
    if not completed: rawpath=rawpath.with_suffix('.failure.json')
    raw=json.loads(rawpath.read_bytes()); retpath=DOC/'RETENTION_494_20261006.json'; ret=json.loads(retpath.read_bytes())
    registrations=[]; events=[]; resources=[]
    for label,record,path,cap in (('main',raw,rawpath,600),('audit',ret,retpath,900)):
        regpath=DOC/f'meth494_{label}_sole_first_registration.json'; reg=json.loads(regpath.read_bytes())
        assert reg['attempt']==1 and reg['process_instance']==record['process_instance']
        assert reg['record_sha256']==receipt(path)['sha256']
        if completed or label=='audit': assert reg['actual_exit_code']==0
        else: assert reg['actual_exit_code']!=0
        eventpath=ROOT/f'results/native_expert_scaling/meth494_{"audit_" if label=="audit" else ""}windows_terminal.json'
        event=json.loads(eventpath.read_bytes()); assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        instance=event['instances'][0]; assert instance['pid']==record['process_instance']['pid'] and instance['create_time_unix']==record['process_instance']['create_time_unix']
        for key in ('start_utc','end_utc'): assert datetime.datetime.fromisoformat(instance[key])==datetime.datetime.fromisoformat(record[key])
        registrations.append(receipt(regpath)); events.append(receipt(eventpath))
        if completed or label=='audit':
            terminalpath=Path(record['terminal_resource_path']); terminal=json.loads(terminalpath.read_bytes())
            assert terminal['raw_sha256']==receipt(path)['sha256'] and terminal['raw_path']==str(path)
            assert terminal['resource']['wall_seconds']<=cap and terminal['resource']['parent_peak_bytes']<=512<<20 and terminal['resource']['native_peak_bytes']==0
            resources.append(receipt(terminalpath))
    assert ret['main_completed']==completed and ret['raw']['sha256']==receipt(rawpath)['sha256'] and all(ret['gates'].values())
    assert ret['main_windows_sha256']==events[0]['sha256']
    if completed:
        assert all(raw['gates'].values()) and (ret['experts_audited'],ret['dictionary_rows_audited'],ret['development_inputs_audited'])==(128,512,11721)
        assert raw['Gaussian_projection_solves']==128 and raw['gradient_updates']==raw['source_FFN_calls']==raw['model_calls']==ret['source_FFN_calls']==ret['model_calls']==0
    gates={'sole_first_actual_main_and_audit':True,'raw_retention_ALL_output_and_scope_digest_link':True,
        'ALL_independent_retention_gates':True,'BOTH_actual_Windows_without_application_fault':True,
        'prior_and_logical_cost_only_no_model_quality_rate_or_goal_promotion':True}
    value={'experiment':'METH494 hybrid donor Gaussian prior admission','main_completed':completed,'raw':receipt(rawpath),'retention':receipt(retpath),
        'registrations':registrations,'Windows_events':events,'terminal_resources':resources,'gates':gates,'decision':ret['decision'],
        'scope':'One untrained bank prior and logical cost. Goal ACTIVE/INCOMPLETE.'}
    with DEST.open('xb') as f:f.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'admission':str(DEST),'sha256':receipt(DEST)['sha256'],'gates':gates,'decision':ret['decision']}))
except BaseException as exc:
    with FAIL.open('xb') as f:f.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n').encode())
    raise
