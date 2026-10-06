"""Metadata admission with corrected actual UTC Windows queries; no scientific replay."""
import hashlib,json
from pathlib import Path
from datetime import datetime
import meth506_r4_operations as O
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while data:=f.read(4<<20):h.update(data)
    return h.hexdigest()
def item(p):
    p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}
dest=O.DOC/'ADMISSION_506_R4_4_20261006.json';assert not dest.exists()
b=json.loads(O.BIND.read_bytes());r=json.loads(O.RAW.read_bytes());retpath=O.DOC/'RETENTION_506_R4_4_20261006.json';a=json.loads(retpath.read_bytes())
assert all(r['gates'].values()) and all(a['gates'].values()) and a['raw_sha256']==digest(O.RAW)
assert r['binding_sha256']==a['binding_sha256']==digest(O.BIND)
assert r['new_compile_native_calls']==0 and not r['timing_isolation']['strictly_verified'] and r['summary']['whole_recipe_eligible'] is False
wp=O.ROOT/'results/native_expert_scaling/meth506_r4_3_windows_terminal.json';w=json.loads(wp.read_bytes())
assert w['positive_control']['query_available']
assert set(w['known_control_record_ids'])<=set(v['RecordId'] for v in w['positive_control']['events'])
assert [s['name'] for s in w['stages']]==['prepareR3','originalMainAndAll578Commands','qualityR4p2','auditR4p2']
for s in w['stages']:
    assert digest(s['raw_path'])==s['raw_sha256'] and s['query']['query_available'] and not s['matching_scientific_events']
    lo=datetime.fromisoformat(s['query']['start_utc']);hi=datetime.fromisoformat(s['query']['end_utc'])
    assert all(lo<=datetime.fromisoformat(e['time_utc'])<=hi for e in s['query']['events'])
assert a['quality_gates']==r['summary']['quality_gates'] and a['economic_gates']==r['summary']['economic_gates']
for rel,sha in b['preserved'].items():assert digest(O.ROOT/rel)==sha
assert digest(O.ROOT/'benchmarks/phase60/engine.c')==b['engine_sha256']
folders=[O.ROOT/'results/native_expert_scaling'/n for n in ['meth506_artifact','meth506_r2_artifact','meth506_r3_artifact','meth506_whole','meth506_r4_quality','meth506_r4_2_quality','meth506_r4_2_retention']]
size=sum(p.stat().st_size for d in folders if d.exists() for p in d.iterdir() if p.is_file())+sum(p.stat().st_size for p in O.DOC.glob('meth506*.json'));assert size<=24<<30
sources=[O.ROOT/'benchmarks/native_expert_scaling'/n for n in ['meth506_r4_3_windows_terminal.ps1','meth506_r4_3_finalize_admission.py']]+[O.DOC/'METH_506_R4_3_UTC_EVENT_REPAIR_20261006.md']
value={'experiment':'METH506-R4.4 corrected metadata admission of sole complete whole-function/fresh-quality observation',
       'binding':item(O.BIND),'raw':item(O.RAW),'retention':item(retpath),'corrected_actual_UTC_windows':item(wp),'metadata_sources':[item(p) for p in sources],
       'main_completed':True,'gates':{'ALL_bank_artifact_whole_states_and_native_outputs_independently_audited':True,
       'fresh_original_F32_teacher_own_generate_tasks_and_fixed_decisions_independently_audited':True,
       'actual_UTC_event_windows_positive_control_process_scope_and_retained_first_query_defect':True,
       'zero_native_replay_preserved_foreign3_engine_and_bounded_outputs':True},
       'quality_gates':a['quality_gates'],'observed_economic_gates':a['economic_gates'],
       'strict_timing_isolation_verified':False,'whole_recipe_eligible':False,'total_new_outputs_before_receipt':size,
       'known_import_diagnostic':r['import_diagnostic'],'audit_serialization_exception':{'original_exit':a['original_audit_exit_code'],'scope':a['scope'],'resource':a['resource'],'process_instance':a['process_instance']},
       'scope':'Apparatus/function and negative decisions admitted via once-only independent source/log assertion-path recovery; original audit serializer failed, precise peak/hash/creation unavailable.14/15 quality,4/5 economics; original timing isolation unverified. Legacy Windows queries superseded by actual XML UTC queries. No final goal completion.'}
O.write(dest,value);print(json.dumps({'admission':item(dest),'gates':value['gates'],'quality':value['quality_gates'],'observed_economics':value['observed_economic_gates'],'outputs_bytes':size}),flush=True)
