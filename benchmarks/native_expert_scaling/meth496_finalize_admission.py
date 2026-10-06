"""Actual fixed-artifact main/audit execution admission; metadata only."""
import datetime
import hashlib
import json
from pathlib import Path
import traceback
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_496_20261006.json';FAIL=DOC/'meth496_first_finalizer_fault.json'
assert not DEST.exists() and not FAIL.exists()
def receipt(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
try:
    rp=DOC/'meth496_readout_error_result.json';completed=rp.exists()
    if not completed:rp=rp.with_suffix('.failure.json')
    raw=json.loads(rp.read_bytes());ap=DOC/'RETENTION_496_20261006.json';audit=json.loads(ap.read_bytes());regs=[];events=[];resources=[]
    for label,record,path in (('main',raw,rp),('audit',audit,ap)):
        regpath=DOC/f'meth496_{label}_sole_first_registration.json';reg=json.loads(regpath.read_bytes())
        assert reg['attempt']==1 and reg['process_instance']==record['process_instance'] and reg['record_sha256']==receipt(path)['sha256']
        assert reg['actual_exit_code']==(0 if completed or label=='audit' else reg['actual_exit_code'])
        if not completed and label=='main':assert reg['actual_exit_code']!=0
        evpath=ROOT/f'results/native_expert_scaling/meth496_{"audit_" if label=="audit" else ""}windows_terminal.json';event=json.loads(evpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        assert len(event['instances'])==1 and not record.get('commands')
        instance=event['instances'][0];assert instance['pid']==record['process_instance']['pid'] and instance['create_time_unix']==record['process_instance']['create_time_unix']
        for key in ('start_utc','end_utc'):assert datetime.datetime.fromisoformat(instance[key])==datetime.datetime.fromisoformat(record[key])
        regs.append(receipt(regpath));events.append(receipt(evpath))
        if completed or label=='audit':
            terminalpath=Path(record['terminal_resource_path']);terminal=json.loads(terminalpath.read_bytes());resource=terminal['resource']
            assert terminal['raw_sha256']==receipt(path)['sha256'] and resource['wall_seconds']<=300
            assert resource['parent_peak_bytes']<=768<<20 and resource['native_peak_bytes']==0;resources.append(receipt(terminalpath))
    assert audit['main_completed']==completed and audit['raw']['sha256']==receipt(rp)['sha256'] and all(audit['gates'].values())
    assert audit['main_windows_sha256']==events[0]['sha256']
    if completed:
        assert all(raw['gates'].values()) and (audit['UIDs_audited'],audit['occurrences_audited'],audit['experts_audited'],audit['metric_groups_audited'],audit['exposure_groups_audited'])==(17540,19962,128,1040,384)
        assert audit['diagnostic_outcomes']==raw['reports']['diagnostic_outcomes'] and audit['decision']==raw['decision']
        for key in ('source_FFN_calls','model_calls','native_calls'):assert raw[key]==audit[key]==0
        assert raw['optimizer_updates']==raw['projection_solves']==0
    gates={'sole_first_actual_main_audit_and_terminal_resources':True,'ALL_independent_full_domain_decomposition_gates':True,
        'BOTH_actual_Windows_process_instances_without_fault':True,'raw_retention_scope_digest_link':True,'fixed_artifact_only_no_candidate_quality_rate_useful_n_or_goal_promotion':True}
    value={'experiment':'METH496 unchanged-artifact readout error admission','main_completed':completed,'raw':receipt(rp),'retention':receipt(ap),
        'registrations':regs,'Windows_events':events,'terminal_resources':resources,'gates':gates,'decision':audit['decision'],'scope':'One-bank fixed artifact consumed-domain diagnostic. Goal ACTIVE/INCOMPLETE.'}
    with DEST.open('xb') as f:f.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'admission':str(DEST),'sha256':receipt(DEST)['sha256'],'gates':gates,'decision':audit['decision']}))
except BaseException as exc:
    with FAIL.open('xb') as f:f.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n').encode())
    raise
