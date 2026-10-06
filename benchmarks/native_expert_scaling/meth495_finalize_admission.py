"""Admission of actual whole local fit/audit executions; metadata only."""
import datetime
import hashlib
import json
from pathlib import Path
import traceback
ROOT=Path(__file__).resolve().parents[2]; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_495_20261006.json'; FAIL=DOC/'meth495_first_finalizer_fault.json'
assert not DEST.exists() and not FAIL.exists()
def receipt(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
try:
    rp=DOC/'meth495_weighted_hybrid_fit_result.json'; completed=rp.exists()
    if not completed:rp=rp.with_suffix('.failure.json')
    raw=json.loads(rp.read_bytes()); ap=DOC/'RETENTION_495_20261006.json'; audit=json.loads(ap.read_bytes())
    regs=[]; events=[]; resources=[]
    for label,record,path in (('main',raw,rp),('audit',audit,ap)):
        regpath=DOC/f'meth495_{label}_sole_first_registration.json'; reg=json.loads(regpath.read_bytes())
        assert reg['attempt']==1 and reg['process_instance']==record['process_instance'] and reg['record_sha256']==receipt(path)['sha256']
        if completed or label=='audit':assert reg['actual_exit_code']==0
        else:assert reg['actual_exit_code']!=0
        evpath=ROOT/f'results/native_expert_scaling/meth495_{"audit_" if label=="audit" else ""}windows_terminal.json'
        event=json.loads(evpath.read_bytes()); assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        expected=[{'process_instance':record['process_instance'],'start_utc':record['start_utc'],'end_utc':record['end_utc']}]+record.get('commands',[])
        assert len(event['instances'])==len(expected)
        for instance,source in zip(event['instances'],expected):
            assert instance['pid']==source['process_instance']['pid'] and instance['create_time_unix']==source['process_instance']['create_time_unix']
            for key in ('start_utc','end_utc'):assert datetime.datetime.fromisoformat(instance[key])==datetime.datetime.fromisoformat(source[key])
        regs.append(receipt(regpath)); events.append(receipt(evpath))
        if completed or label=='audit':
            terminalpath=Path(record['terminal_resource_path']); terminal=json.loads(terminalpath.read_bytes()); resource=terminal['resource']
            assert terminal['raw_sha256']==receipt(path)['sha256'] and resource['wall_seconds']<=1200
            assert resource['parent_peak_bytes']+resource['native_peak_bytes']<=1536<<20; resources.append(receipt(terminalpath))
    assert audit['main_completed']==completed and audit['raw']['sha256']==receipt(rp)['sha256'] and all(audit['gates'].values())
    assert audit['main_windows_sha256']==events[0]['sha256']
    if completed:
        assert all(raw['gates'].values()) and (audit['UIDs_audited'],audit['occurrences_audited'],audit['experts_audited'],audit['metric_groups_audited'],audit['exposure_groups_audited'])==(17540,19962,128,1040,384)
        assert raw['function_solves']==127 and raw['Adam_updates']==128 and audit['recipe_gates']==raw['reports']['recipe_gates']
        assert raw['source_FFN_calls']==raw['model_calls']==audit['source_FFN_calls']==audit['model_calls']==0
    gates={'sole_first_actual_main_audit_and_terminal_resources':True,'ALL_independent_full_domain_gates':True,
        'BOTH_actual_Windows_parent_and_child_process_instances_without_fault':True,'source_raw_retention_scope_digest_link':True,
        'local_candidate_only_no_model_quality_rate_useful_n_or_goal_promotion':True}
    value={'experiment':'METH495 full local fixed hybrid recipe admission','main_completed':completed,'raw':receipt(rp),'retention':receipt(ap),
        'registrations':regs,'Windows_events':events,'terminal_resources':resources,'gates':gates,'decision':audit['decision'],
        'scope':'One-bank consumed-domain candidate. Goal ACTIVE/INCOMPLETE.'}
    with DEST.open('xb') as f:f.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'admission':str(DEST),'sha256':receipt(DEST)['sha256'],'gates':gates,'decision':audit['decision']}))
except BaseException as exc:
    with FAIL.open('xb') as f:f.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n').encode())
    raise
