"""Post-observation metadata admission, preserving the complete-cohort geometry outcome."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'ADMISSION_501_20261006.json';assert not DEST.exists()
def item(p):
    p=Path(p);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rp=DOC/'meth501_support_cover_result.json';ap=DOC/'RETENTION_501_20261006.json';bp=DOC/'meth501_binding.json'
r=json.loads(rp.read_bytes());a=json.loads(ap.read_bytes());b=json.loads(bp.read_bytes())
mr=json.loads((DOC/'meth501_main_sole_first_registration.json').read_bytes());ar=json.loads((DOC/'meth501_audit_sole_first_registration.json').read_bytes())
assert item(rp)['sha256']==mr['record_sha256']==a['raw']['sha256'] and item(ap)['sha256']==ar['record_sha256']
assert mr['actual_exit_code']==ar['actual_exit_code']==0 and all(r['gates'].values()) and a['main_completed'] and all(a['gates'].values())
assert r['summary']==a['summary'] and r['eligibility']==a['eligibility']=={'ALL127_complete_development_cover':False}
assert a['unique_inputs_audited']==17540 and len(r['experts'])==128
windows=[]
for stage,name in [('main','meth501_windows_terminal.json'),('audit','meth501_audit_windows_terminal.json')]:
    p=ROOT/'results/native_expert_scaling'/name;v=json.loads(p.read_bytes());record=r if stage=='main' else a
    assert v['query_available'] and v['query_error'] is None and not v['matching_scientific_events'] and v['instances'][0]['pid']==record['process_instance']['pid'];windows.append(item(p))
for rel,sha in b['preserved'].items():assert item(ROOT/rel)['sha256']==sha
assert item(ROOT/'benchmarks/phase60/engine.c')['sha256']==b['engine_sha256']
counts={status:sum(row['status']==status for row in r['experts']) for status in {row['status'] for row in r['experts']}}
assert sorted(counts.values())==[1,14,52,61]
result={'experiment':'METH501 independently admitted ONE complete-cohort support-cover attempt','main_completed':True,
    'gates':{'all_five_main_apparatus':True,'all_six_independent_bitset_apparatus':True,'ALL128_UID_domain_and_complete_summary_eligibility_unchanged':True,
             'actual_main_audit_exit0_Windows_available_zero_matching_faults':True,'engine_foreign_files_preserved':True},
    'raw':item(rp),'retention':item(ap),'binding':item(bp),'windows':windows,'summary':r['summary'],'status_counts':counts,
    'eligibility':r['eligibility'],'complete_cohort_eligible':False,'frozen_automatic_decision':r['decision'],
    'decision':'NEXT_VARIABLE_CHILD_RAM_CERTIFICATE_AT_FIXED_ACTIVE_WIDTH',
    'next_reason':'Use retained first conflicts and global-copy deficits to construct explicit data-driven additional branches at fixed1536 active width; price this construction\'s RAM requirement without a C grid or replay of completed prefixes. This is not a minimum-memory or generalization claim.',
    'scope':'Finite support geometry.61 positive development witnesses do not qualify all127 experts or consumed/fresh model quality. No new source/model/native/GPU/gradient/rate/DRAM/LUT/large-n or goal promotion.'}
with DEST.open('xb') as f:f.write((json.dumps(result,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':item(DEST),'gates':result['gates'],'decision':result['decision'],'complete_cohort_eligible':False}))
