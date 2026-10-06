"""Metadata-only admission: certificate soundness and failed economy kept separate."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_504_20261006.json';assert not DEST.exists()
def item(p):
    p=Path(p);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rp=DOC/'meth504_region_result.json';ap=DOC/'RETENTION_504_20261006.json';bp=DOC/'meth504_binding.json'
r=json.loads(rp.read_bytes());a=json.loads(ap.read_bytes());b=json.loads(bp.read_bytes())
for stage in ['main','audit']:
    reg=json.loads((DOC/f'meth504_{stage}_tool_registration.json').read_bytes());assert reg['actual_exit_code']==reg['wait_tool_results'][-1]['exit_code']==0
assert item(rp)['sha256']==a['raw']['sha256'] and all(r['gates'].values()) and all(a['gates'].values()) and a['main_completed']
assert r['views']==a['views'] and r['rare_consumed']==a['rare_consumed'] and r['logical_cost']==a['logical_cost'] and r['eligibility']==a['eligibility']
assert a['unique_inputs_audited']==17540 and len(r['experts'])==128 and r['centres']==637 and r['valid_regions']==637 and not r['unsupported_regions']
assert not any(r['eligibility'].values()) and r['decision']==a['decision']=='CLOSE_THIS_FINITE_ISOTROPIC_INPUT_CONE_RECIPE'
windows=[]
for stage,name in [('main','meth504_windows_terminal.json'),('audit','meth504_audit_windows_terminal.json')]:
    p=ROOT/'results/native_expert_scaling'/name;v=json.loads(p.read_bytes());record=r if stage=='main' else a
    assert v['query_available'] and v['query_error'] is None and not v['matching_scientific_events'] and v['instances'][0]['pid']==record['process_instance']['pid'];windows.append(item(p))
for rel,sha in b['preserved'].items():assert item(ROOT/rel)['sha256']==sha
assert item(ROOT/'benchmarks/phase60/engine.c')['sha256']==b['engine_sha256']
result={'experiment':'METH504 admitted source-region certificate and finite economic failure','main_completed':True,
    'gates':{'all_six_main_apparatus':True,'all_seven_independent_sign_geometry_apparatus':True,'ALL_original_domain_costs_and_frozen_economy_unchanged':True,
        'actual_main_audit_exit0_Windows_available_zero_faults':True,'engine_foreign_files_preserved':True},
    'raw':item(rp),'retention':item(ap),'binding':item(bp),'windows':windows,'views':r['views'],'rare_consumed':r['rare_consumed'],'logical_cost':r['logical_cost'],
    'eligibility':r['eligibility'],'economic_recipe_eligible':False,'decision':r['decision'],
    'scope':'Certificate is sound but coverage insufficient; no native/whole model/fresh/rate/DRAM/LUT/family promotion. Not a general nonexistence proof.'}
with DEST.open('xb') as f:f.write((json.dumps(result,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':item(DEST),'gates':result['gates'],'decision':result['decision'],'economic_recipe_eligible':False}))
