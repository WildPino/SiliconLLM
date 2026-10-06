"""Final evidence evaluation: retain negative quality/economy/clean-event outcomes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
def item(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(4<<20):h.update(data)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
dest=DOC/'EVALUATION_506_20261006.json';assert not dest.exists()
rp=DOC/'meth506_r4_2_result.json';ap=DOC/'RETENTION_506_R4_4_20261006.json';bp=DOC/'meth506_r4_2_binding.json'
wp=ROOT/'results/native_expert_scaling/meth506_r4_3_windows_terminal.json'
r=json.loads(rp.read_bytes());a=json.loads(ap.read_bytes());b=json.loads(bp.read_bytes());w=json.loads(wp.read_bytes())
assert all(r['gates'].values()) and all(a['gates'].values()) and a['raw_sha256']==item(rp)['sha256']
assert a['binding_sha256']==r['binding_sha256']==item(bp)['sha256']
assert a['quality_gates']==r['summary']['quality_gates'] and a['economic_gates']==r['summary']['economic_gates']
assert w['positive_control']['query_available'] and set(w['known_control_record_ids'])<=set(e['RecordId'] for e in w['positive_control']['events'])
assert all(s['query']['query_available'] and item(s['raw_path'])['sha256']==s['raw_sha256'] for s in w['stages'])
for rel,sha in b['preserved'].items():assert item(ROOT/rel)['sha256']==sha
assert item(ROOT/'benchmarks/phase60/engine.c')['sha256']==b['engine_sha256']
matches=[{'stage':s['name'],'event':m['event'],'instance':m['instance']} for s in w['stages'] for m in s['matching_scientific_events']]
assert len(matches)==3 and {m['stage'] for m in matches}=={'prepareR3','originalMainAndAll578Commands','qualityR4p2'}
assert all(m['event']['data']['ModuleName']=='arrow.dll' and m['event']['data']['ExceptionCode']=='c0000005' and m['event']['data']['FaultingOffset']=='0000000000bc5431' for m in matches)
folders=[ROOT/'results/native_expert_scaling'/n for n in ['meth506_artifact','meth506_r2_artifact','meth506_r3_artifact','meth506_whole','meth506_r4_quality','meth506_r4_2_quality','meth506_r4_2_retention']]
size=sum(p.stat().st_size for d in folders if d.exists() for p in d.iterdir() if p.is_file())+sum(p.stat().st_size for p in DOC.glob('meth506*.json'));assert size<=24<<30
g={'complete_physical_function_and_retained_numerical_assertion_path':True,'all_frozen_quality_criteria':all(a['quality_gates'].values()),
   'all_observed_economic_criteria':all(a['economic_gates'].values()),'actual_UTC_zero_matching_event_criterion':not matches,
   'strict_original_native_timing_isolation':r['timing_isolation']['strictly_verified']}
v={'experiment':'METH506 final evidence evaluation with complete negative outcomes and provenance exceptions','raw':item(rp),'binding':item(bp),'source_log_independent_retention':item(ap),'corrected_actual_UTC_windows':item(wp),
   'evaluation_gates':g,'full_admission_pass':all(g.values()),'whole_recipe_eligible':False,'goal_complete':False,
   'quality_gates':a['quality_gates'],'observed_economic_gates':a['economic_gates'],'actual_matching_parent_import_events':matches,
   'audit_serializer_exception':{'original_exit':a['original_audit_exit_code'],'exact_peak_hash_creation_unavailable':True,'resource':a['resource']},
   'known_import_diagnostic':r['import_diagnostic'],'total_new_outputs_before_evaluation':size,
   'scope':'Full numerical function/quality/cost evidence retained.14/15 quality and4/5 observed economics; clean-event and strict-isolation criteria FAIL/unverified, no final admission or SAME>=50 goal promotion.Three actual parent arrow.dll startup events correspond in time/module to retained import diagnostics; cause unidentified. Sole independent audit completed computations but serializer exit1 recovered by source/log metadata, exact private resources unavailable. No scientific replay.'}
data=(json.dumps(v,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode()
with dest.open('xb') as f:f.write(data)
print(json.dumps({'evaluation':item(dest),'gates':g,'full_admission_pass':False,'outputs_bytes':size}),flush=True)
