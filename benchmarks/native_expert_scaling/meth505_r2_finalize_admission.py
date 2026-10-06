"""Metadata-only admission of the completed new physical operator/cost inquiry."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_505_R2_20261006.json';assert not DEST.exists()
def item(p):
    p=Path(p);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rp=DOC/'meth505_r2_result.json';ap=DOC/'RETENTION_505_R2_20261006.json';bp=DOC/'meth505_r2_binding.json'
r=json.loads(rp.read_bytes());a=json.loads(ap.read_bytes());b=json.loads(bp.read_bytes())
for stage in ['main','audit']:
    reg=json.loads((DOC/f'meth505_r2_{stage}_tool_registration.json').read_bytes());assert reg['actual_exit_code']==reg['wait_tool_results'][-1]['exit_code']==0
assert item(rp)['sha256']==a['raw']['sha256'] and all(r['gates'].values()) and all(a['gates'].values()) and a['main_completed']
for key in ['views','books','rare_consumed','experts','storage','eligibility','decision']:assert r[key]==a[key]
assert a['unique_inputs_audited']==17540 and len(r['experts'])==128 and len(r['books'])==192
windows=[]
for stage,name in [('main','meth505_r2_windows_terminal.json'),('audit','meth505_r2_audit_windows_terminal.json')]:
    p=ROOT/'results/native_expert_scaling'/name;v=json.loads(p.read_bytes());record=r if stage=='main' else a
    assert v['query_available'] and v['query_error'] is None and not v['matching_scientific_events'] and v['instances'][0]['pid']==record['process_instance']['pid'];windows.append(item(p))
oldreg=json.loads((DOC/'meth505_first_fault_tool_registration.json').read_bytes());assert oldreg['actual_exit_code']==1
oldwin=json.loads(Path(b['original_windows']['path']).read_bytes());assert oldwin['query_available'] and not oldwin['matching_scientific_events'] and oldwin['query_error'] is None
assert r['new_native_calls']==a['new_native_calls']==0 and r['unbound_primal_OS_exception']==a['unbound_primal_OS_exception']==b['unbound_primal_OS_exception']
for rel,sha in b['preserved'].items():assert item(ROOT/rel)['sha256']==sha
assert item(ROOT/'benchmarks/phase60/engine.c')['sha256']==b['engine_sha256']
result={'experiment':'METH505 independently admitted exact original WI/sparse WO physical inquiry','main_completed':True,'metadata_recovery':True,'first_caller_exit':1,'retained_native_exits':[0,0,0],'unbound_primal_OS_exception':b['unbound_primal_OS_exception'],'new_native_calls':0,
        'gates':{'main_apparatus':True,'independent_bank_function_cost_audit':True,'all_frozen_domains_gates_preserved':True,'actual_main_audit_exit0_Windows_available_zero_faults':True,'engine_foreign_files_preserved':True},
        'raw':item(rp),'retention':item(ap),'binding':item(bp),'windows':windows,'views':r['views'],'storage':r['storage'],'eligibility':r['eligibility'],
        'economic_recipe_eligible':all(r['eligibility'].values()),'decision':r['decision'],
        'scope':'Exact physical local FFN and cost only; full integration/fresh own-state quality/same artifact rate/useful n/LUT/DRAM/families remain open.'}
with DEST.open('xb') as f:f.write((json.dumps(result,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':item(DEST),'gates':result['gates'],'decision':result['decision'],'economic_recipe_eligible':result['economic_recipe_eligible']}))
