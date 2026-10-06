"""Metadata admission only; scientific results retained without changed thresholds."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_502_20261006.json';assert not DEST.exists()
def item(p):
    p=Path(p);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rp=DOC/'meth502_variable_cover_result.json';ap=DOC/'RETENTION_502_20261006.json';bp=DOC/'meth502_binding.json'
r=json.loads(rp.read_bytes());a=json.loads(ap.read_bytes());b=json.loads(bp.read_bytes())
for stage,p in [('main',rp),('audit',ap)]:
    reg=json.loads((DOC/f'meth502_{stage}_tool_registration.json').read_bytes());assert reg['tool_result']['exit_code']==0 and not reg['tool_result'].get('session_id')
assert item(rp)['sha256']==a['raw']['sha256'] and all(r['gates'].values()) and all(a['gates'].values()) and a['main_completed']
assert r['summary']==a['summary'] and r['eligibility']==a['eligibility']=={'ALL127_finite_development_exact_support_cover':True}
assert a['unique_inputs_audited']==17540 and len(r['experts'])==128
windows=[]
for stage,name in [('main','meth502_windows_terminal.json'),('audit','meth502_audit_windows_terminal.json')]:
    p=ROOT/'results/native_expert_scaling'/name;v=json.loads(p.read_bytes());record=r if stage=='main' else a
    assert v['query_available'] and v['query_error'] is None and not v['matching_scientific_events'] and v['instances'][0]['pid']==record['process_instance']['pid'];windows.append(item(p))
for rel,sha in b['preserved'].items():assert item(ROOT/rel)['sha256']==sha
assert item(ROOT/'benchmarks/phase60/engine.c')['sha256']==b['engine_sha256']
assert r['summary']['baseline_experts']==61 and r['summary']['resumed_experts']==52
result={'experiment':'METH502 admitted variable storage construction','main_completed':True,
    'gates':{'all_five_main_apparatus':True,'all_six_independent_Boolean_apparatus':True,'complete_original_cohort_and_summary_unchanged':True,
        'actual_main_audit_exit0_Windows_available_zero_faults':True,'engine_foreign_files_preserved':True},
    'raw':item(rp),'retention':item(ap),'binding':item(bp),'windows':windows,'summary':r['summary'],'eligibility':r['eligibility'],
    'decision':'NEXT_ACTUAL_FUNCTION_ORACLE_AND_INPUT_SELECTOR','scope':'Exact finite development cover/copy witness only; no quality/fresh/physical/rate promotion.'}
with DEST.open('xb') as f:f.write((json.dumps(result,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':item(DEST),'gates':result['gates'],'decision':result['decision']}))
