"""Post-observation metadata admission; unchanged scientific thresholds and outputs."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_500_20261006.json';assert not DEST.exists()
def item(path):
    p=Path(path);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rawp=DOC/'meth500_overlap_result.json';retp=DOC/'RETENTION_500_20261006.json'
raw=json.loads(rawp.read_bytes());ret=json.loads(retp.read_bytes());binding=json.loads((DOC/'meth500_binding.json').read_bytes())
mainreg=json.loads((DOC/'meth500_main_sole_first_registration.json').read_bytes());auditreg=json.loads((DOC/'meth500_audit_sole_first_registration.json').read_bytes())
assert item(rawp)['sha256']==mainreg['record_sha256']==ret['raw']['sha256']
assert item(retp)['sha256']==auditreg['record_sha256']
assert mainreg['actual_exit_code']==auditreg['actual_exit_code']==0
assert all(raw['gates'].values()) and ret['main_completed'] and all(ret['gates'].values())
assert ret['unique_inputs_audited']==raw['source_FFN_rows_reconstructed']==17540 and ret['partial_functions_audited']==157860
assert raw['eligibility']==ret['eligibility'] and len(raw['eligibility'])==5 and not all(raw['eligibility'].values())
windows=[]
for stage,name in [('main','meth500_windows_terminal.json'),('audit','meth500_audit_windows_terminal.json')]:
    p=ROOT/'results/native_expert_scaling'/name;v=json.loads(p.read_bytes());record=raw if stage=='main' else ret
    assert v['query_available'] and v['query_error'] is None and not v['matching_scientific_events']
    assert v['instances'][0]['pid']==record['process_instance']['pid'];windows.append(item(p))
for rel,sha in binding['preserved'].items():assert item(ROOT/rel)['sha256']==sha
assert item(ROOT/'benchmarks/phase60/engine.c')['sha256']==binding['engine_sha256']
tail=raw['views']['natural_consumed']['oracle']['p95_UID_RMS'];assert tail>.05
result={'experiment':'METH500 complete independently admitted ONE overlap recipe','main_completed':True,
        'gates':{'main_all_six_apparatus':True,'audit_all_seven_apparatus':True,'ALL17540_source_support_and157860_partial_functions':True,
                 'unchanged_ALL_five_eligibility_outcomes':True,'actual_main_audit_exit0_Windows_available_no_matching_fault_and_preserved_files':True},
        'raw':item(rawp),'retention':item(retp),'binding':item(DOC/'meth500_binding.json'),'windows':windows,
        'eligibility':raw['eligibility'],'recipe_eligible':False,'frozen_automatic_decision':raw['decision'],
        'decision':'NEXT_SOURCE_SUPPORT_COVER_GEOMETRY_BEFORE_SELECTOR',
        'deduction':{'best_actual_child_natural_consumed_p95_UID_RMS':tail,'threshold':.05,
            'reason':'At every input any single choice among these fixed four children has relative error >= the best-child error; quantiles are monotone. Oracle tail already fails, so changing only selector cannot satisfy the frozen tail gate.',
            'scope':'This four-mask construction, not all overlap/expanded representations. The frozen automatic label branches on aggregate RMS only and is retained verbatim.'},
        'counts':{'source_FFN_rows_main':17540,'source_FFN_rows_independent_audit':17540,'partial_functions_main':157860,'partial_functions_independent_audit':157860,
                  'model_native_GPU_calls':0,'gradient_updates':0},'argv':sys.argv,
        'scope':'Local original-domain mask/selector screening FAIL; no head/composed/fresh/DRAM/LUT/rate/useful-n/family-scale or goal-completion promotion.'}
with DEST.open('xb') as f:f.write((json.dumps(result,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':item(DEST),'gates':result['gates'],'decision':result['decision'],'recipe_eligible':False}))
