"""Sole metadata admission of actual executions and independent exact outcomes."""
import datetime
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_492_20261006.json';FAIL=DOC/'meth492_first_finalizer_fault.json'
assert not DEST.exists() and not FAIL.exists()

def receipt(path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

try:
    rawpath=DOC/'meth492_selected_amplitude_result.json'
    failure=rawpath.with_suffix('.failure.json')
    if failure.exists():rawpath=failure
    completed=rawpath.name=='meth492_selected_amplitude_result.json'
    raw=json.loads(rawpath.read_bytes());retpath=DOC/'RETENTION_492_20261006.json';ret=json.loads(retpath.read_bytes())
    registrations=[];events=[]
    for label,record,path in (('main',raw,rawpath),('audit',ret,retpath)):
        regpath=DOC/('meth492_'+label+'_sole_first_registration.json');reg=json.loads(regpath.read_bytes())
        assert reg['attempt']==1 and reg['process_instance']==record['process_instance']
        assert reg['record_sha256']==receipt(path)['sha256']
        assert reg['actual_exit_code']==0 if completed or label=='audit' else reg['actual_exit_code']!=0
        eventpath=ROOT/('results/native_expert_scaling/meth492_'+('audit_' if label=='audit' else '')+'windows_terminal.json')
        event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        instance=event['instances'][0]
        assert instance['pid']==record['process_instance']['pid'] and instance['create_time_unix']==record['process_instance']['create_time_unix']
        assert all(datetime.datetime.fromisoformat(instance[k])==datetime.datetime.fromisoformat(record[k]) for k in ('start_utc','end_utc'))
        registrations.append(receipt(regpath));events.append(receipt(eventpath))
    assert ret['raw']['sha256']==receipt(rawpath)['sha256'] and all(ret['gates'].values())
    assert ret['main_windows_sha256']==events[0]['sha256'] and ret['main_completed']==completed
    if completed:
        assert all(raw['gates'].values()) and ret['cells_audited']==1536 and ret['unique_inputs_audited']==238872 and ret['occurrences_audited']==387036
        assert raw['outcome_counts']==ret['outcome_counts'] and raw['decision']==ret['source_amplitude_decision']
        assert raw['native_calls']==raw['optimizer_updates']==ret['native_calls']==ret['optimizer_updates']==ret['SVD_calls']==0
        assert ret['report_counts_audited']=={'unique':39,'occurrence':72,'book':4608,'source_ID':9216,'accepted':144}
        terminal_receipts=[]
        for label,record,path,cap in (('main',raw,rawpath,180),('audit',ret,retpath,300)):
            terminalpath=Path(record['terminal_resource_path']);terminal=json.loads(terminalpath.read_bytes())
            assert terminal['raw_sha256']==receipt(path)['sha256'] and terminal['raw_path']==str(path)
            assert terminal['resource']['wall_seconds']<=cap
            assert terminal['resource']['parent_peak_bytes']+terminal['resource']['native_peak_bytes']<=256<<20
            terminal_receipts.append(receipt(terminalpath))
    gates={'sole_first_actual_main_and_audit':True,'raw_retention_digest_and_scientific_decision_link':True,
        'ALL_independent_retention_gates':True,'BOTH_actual_Windows_without_application_fault':True,
        'no_function_quality_rate_DRAM_or_useful_n_promotion_from_amplitude_alone':True}
    value={'experiment':'METH492 exact selected-amplitude constant eligibility admission','main_completed':completed,
        'raw':receipt(rawpath),'retention':receipt(retpath),'registrations':registrations,'Windows_events':events,
        'terminal_resources':terminal_receipts if completed else [],
        'gates':gates,'decision':ret['decision'],'scope':'Exact selected-amplitude class only; full goal active/incomplete.'}
    with DEST.open('xb') as stream:stream.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'admission':str(DEST),'sha256':receipt(DEST)['sha256'],'gates':gates,'decision':value['decision']}))
except BaseException as exc:
    with FAIL.open('xb') as stream:stream.write((json.dumps({'error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n').encode())
    raise
