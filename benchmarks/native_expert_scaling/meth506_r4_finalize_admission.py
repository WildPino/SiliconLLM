"""Metadata-only R4 semantic/quality admission; strict original rate provenance unresolved."""
import hashlib
import json
from pathlib import Path
import meth506_r4_operations as O
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while data:=f.read(4<<20):h.update(data)
    return h.hexdigest()
def item(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':digest(p)}
dest=O.DOC/'ADMISSION_506_R4_2_20261006.json';assert not dest.exists()
b=json.loads(O.BIND.read_bytes());r=json.loads(O.RAW.read_bytes());retpath=O.DOC/'RETENTION_506_R4_2_20261006.json';a=json.loads(retpath.read_bytes())
assert all(r['gates'].values()) and all(a['gates'].values()) and a['raw_sha256']==digest(O.RAW)
assert r['binding_sha256']==a['binding_sha256']==digest(O.BIND)
assert r['new_compile_native_calls']==0 and not r['timing_isolation']['strictly_verified'] and r['summary']['whole_recipe_eligible'] is False
windows=[]
for stage in ['main','audit']:
    p=O.ROOT/f'results/native_expert_scaling/meth506_r4_2_{stage}_windows_terminal.json';v=json.loads(p.read_bytes());assert v['query_available'] and not v['matching_scientific_events'];windows.append(item(p))
v=json.loads(Path(b['original_native_windows']['path']).read_bytes());assert v['query_available'] and not v['matching_scientific_events'];windows.append(b['original_native_windows'])
assert a['quality_gates']==r['summary']['quality_gates'] and a['economic_gates']==r['summary']['economic_gates']
for rel,sha in b['preserved'].items():assert digest(O.ROOT/rel)==sha
assert digest(O.ROOT/'benchmarks/phase60/engine.c')==b['engine_sha256']
folders=[O.ROOT/'results/native_expert_scaling'/n for n in ['meth506_artifact','meth506_r2_artifact','meth506_r3_artifact','meth506_whole','meth506_r4_quality','meth506_r4_2_quality','meth506_r4_2_retention']]
size=sum(p.stat().st_size for d in folders if d.exists() for p in d.iterdir() if p.is_file())+sum(p.stat().st_size for p in O.DOC.glob('meth506*.json'));assert size<=24<<30
value={'experiment':'METH506-R4 independent whole function/fresh donor quality admission','binding':item(O.BIND),'raw':item(O.RAW),'retention':item(retpath),'windows':windows,'main_completed':True,
       'gates':{'immutable_ALL_bank_artifact_whole_state_wires_and_native_outputs_independently_audited':True,'fresh_original4p57p6_F32_teacher_own_generate_tasks_and_fixed_decisions_audited':True,'original_and_recovery_actual_Windows_exits_and_known_import_diagnostic_retained':True,'zero_native_replay_foreign3_engine_and_bounded_outputs':True},
       'quality_gates':a['quality_gates'],'observed_economic_gates':a['economic_gates'],'strict_timing_isolation_verified':False,'whole_recipe_eligible':False,'total_new_outputs_before_receipt':size,
       'scope':'Complete original-source128 conditional function and fresh donor-relative quality evidence with explicit economic outcomes. Original two transient Python groups have no retained birth/end/command identity; original strict rate provenance unverified, no final-goal>=50 claim. No CPU LUT/useful n/DRAM/other-family/scale completion.'}
O.write(dest,value);print(json.dumps({'admission':item(dest),'gates':value['gates'],'quality':value['quality_gates'],'observed_economics':value['observed_economic_gates'],'strict_timing_isolation_verified':False,'outputs_bytes':size}),flush=True)
