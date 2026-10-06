"""Metadata-only admission of complete new functions and unchanged oracle decision."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_503_20261006.json';assert not DEST.exists()
def item(p):
    p=Path(p);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rp=DOC/'meth503_function_result.json';ap=DOC/'RETENTION_503_20261006.json';bp=DOC/'meth503_binding.json'
r=json.loads(rp.read_bytes());a=json.loads(ap.read_bytes());b=json.loads(bp.read_bytes())
for stage in ['main','audit']:
    reg=json.loads((DOC/f'meth503_{stage}_tool_registration.json').read_bytes());assert reg['actual_exit_code']==reg['wait_tool_results'][-1]['exit_code']==0
assert item(rp)['sha256']==a['raw']['sha256'] and all(r['gates'].values()) and all(a['gates'].values()) and a['main_completed']
assert r['views']==a['views'] and r['rare_consumed']==a['rare_consumed'] and r['eligibility']==a['eligibility'] and r['decision']==a['decision']
assert a['unique_inputs_audited']==17540 and a['partial_functions_audited']==r['partial_source_outputs']==127570 and len(r['experts'])==128
assert sum(v.get('children',0) for v in r['experts'])==783 and sum(r['eligibility'].values())==4 and len(r['eligibility'])==6
assert r['decision']=='RECONSIDER_REPRESENTATION_OR_SUPPORT_DOMAIN'
windows=[]
for stage,name in [('main','meth503_windows_terminal.json'),('audit','meth503_audit_windows_terminal.json')]:
    p=ROOT/'results/native_expert_scaling'/name;v=json.loads(p.read_bytes());record=r if stage=='main' else a
    assert v['query_available'] and v['query_error'] is None and not v['matching_scientific_events'] and v['instances'][0]['pid']==record['process_instance']['pid'];windows.append(item(p))
for rel,sha in b['preserved'].items():assert item(ROOT/rel)['sha256']==sha
assert item(ROOT/'benchmarks/phase60/engine.c')['sha256']==b['engine_sha256']
result={'experiment':'METH503 admitted variable-child actual function and input-only selector inquiry','main_completed':True,
    'gates':{'all_six_main_apparatus':True,'all_seven_independent_function_apparatus':True,'ALL_original_domain_views_rare_and_frozen_eligibility_unchanged':True,
        'actual_main_audit_exit0_Windows_available_zero_faults':True,'engine_foreign_files_preserved':True},
    'raw':item(rp),'retention':item(ap),'binding':item(bp),'windows':windows,'views':r['views'],'rare_consumed':r['rare_consumed'],
    'eligibility':r['eligibility'],'local_recipe_eligible':False,'decision':r['decision'],
    'raw_experiment_label_note':'Inherited operations label says four overlapping children. The frozen binding/protocol, masks and checked per-expert counts define783 variable children. RAW remains unchanged.',
    'scope':'Exact development interpolation but failed consumed oracle tail. Not a refutation of all redundancy or of other source-derived regions; no fresh/model/native/rate promotion.'}
with DEST.open('xb') as f:f.write((json.dumps(result,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':item(DEST),'gates':result['gates'],'decision':result['decision'],'local_recipe_eligible':False}))
