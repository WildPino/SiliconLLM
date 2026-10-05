"""Descriptive algebra on retained admission counts; no model/fit/main import."""
import hashlib
import json
from pathlib import Path
ROOT=Path.cwd();DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
RAW=DOC/'meth469_switch_native_domain_capture_result.json';RET=DOC/'RETENTION_469_20261005.json';OUT=DOC/'meth469_information_gap_analysis.json'
assert not OUT.exists()
assert hashlib.sha256(RAW.read_bytes()).hexdigest()=='ccb34e789d6e3ed955171b619639dc4d34bc032f960e44996b5dc1f7fa60f870'
assert hashlib.sha256(RET.read_bytes()).hexdigest()=='22e1cb909ca77097cb5a572462b191355ade0aad92ebe837facaa625f9d355c5'
j=json.loads(RAW.read_text());bank=j['banks'][11];groups={};dev_repairable=[];val_blocked=[]
for e in bank['experts']:
 if e['data_ready']:continue
 failed=tuple(k for k,v in e['readiness_gates'].items() if not v)
 group=groups.setdefault(failed,{'experts':[],'natural_validation_executed_queries':0})
 group['experts'].append(e['expert']);group['natural_validation_executed_queries']+=e['by_split_mode']['val_natural']['executed']
 row={'expert':e['expert'],'development_unique_codes':e['dev_union']['unique_code_SHA'],'development_books':len(e['dev_union']['books']),
      'minimum_additional_development_unique_codes_to32':max(0,32-e['dev_union']['unique_code_SHA']),
      'minimum_additional_development_books_to4':max(0,4-len(e['dev_union']['books'])),
      'validation_novel_codes':e['validation_novel_codes'],'validation_novel_books':len(e['validation_novel_code_books']),
      'natural_validation_executed_queries':e['by_split_mode']['val_natural']['executed'],'failed_gates':list(failed)}
 (val_blocked if any(k.startswith('validation_') for k in failed) else dev_repairable).append(row)
total=bank['val_natural_executed_queries'];covered=bank['covered_val_natural_executed_queries']
repairable_queries=sum(v['natural_validation_executed_queries'] for v in dev_repairable)
blocked_queries=sum(v['natural_validation_executed_queries'] for v in val_blocked)
assert len(dev_repairable)==25 and len(val_blocked)==19 and repairable_queries==389 and blocked_queries==71
assert covered+repairable_queries+blocked_queries==total==3065
needed=(9*total+9)//10-covered
assert needed==154
value={'purpose':'posthoc descriptive information bottleneck algebra; fixed data decision unchanged; not a new experiment/fit or adaptive validation selection',
 'source_raw_sha256':'ccb34e789d6e3ed955171b619639dc4d34bc032f960e44996b5dc1f7fa60f870','retention_sha256':'22e1cb909ca77097cb5a572462b191355ade0aad92ebe837facaa625f9d355c5',
 'analysis_helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fixed_bank':11,'ready_IDs':84,'unsupported_IDs':44,
 'natural_validation_queries':total,'covered_queries':covered,'current_coverage':covered/total,'threshold_coverage':.90,'minimum_additional_covered_queries_to_threshold':needed,
 'deficiency_groups':[{'failed_gates':list(k),**v} for k,v in sorted(groups.items())],
 'development_only_deficit_IDs':dev_repairable,'already_validation_floor_blocked_IDs':val_blocked,
 'development_only_deficit_queries':repairable_queries,'already_validation_floor_blocked_queries':blocked_queries,
 'pure_development_augmentation_coverage_UPPER_bound':{'queries':covered+repairable_queries,'coverage':(covered+repairable_queries)/total,'not_a_guarantee':True},
 'minimum_distinct_additional_dev_codes_if_ALL25_currently_val_eligible_deficits_repaired':sum(v['minimum_additional_development_unique_codes_to32'] for v in dev_repairable),
 'minimum_distinct_additional_dev_codes_if_ALL44_dev_deficits_repaired':sum(v['minimum_additional_development_unique_codes_to32'] for v in dev_repairable+val_blocked),
 'monotonic_novelty_proof':'For fixed validation code set V_e, development D_e subset Dprime_e implies V_e\\Dprime_e subset V_e\\D_e. Novel code counts and novel-code book unions cannot grow by adding development alone;19already validation-floor-blocked IDs cannot be repaired that way.25current val-eligible IDs have only development deficits; extra development may still remove novel validation codes.',
 'prospective_action':'ONE bounded model-free64-new-DEVELOPMENT-book/256-context manifest, then separate512native-stream augmentation/combined readiness with old64validation books unchanged and no factor until SAME90%fixed-bank gate. Source selection uses no validation scores/route IDs or the25-ID list; cover development information deficits uniformly. Stop after this one augmentation if inadequate; no endless new-data ladder.',
 'limitations':'Counts are information floors, not rank or function error.389potentially repairable queries only define an upper bound; added development can lower validation novelty. No selection of expert IDs or contexts by validation performance. Pure development supports prospective fit; final donor-relative quality needs separate untouched cases.'}
with OUT.open('xb') as s:s.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'groups':[(list(k),len(v['experts']),v['natural_validation_executed_queries']) for k,v in sorted(groups.items())],
 'dev_repairable_IDs':len(dev_repairable),'dev_repairable_queries':repairable_queries,'val_blocked_IDs':len(val_blocked),'val_blocked_queries':blocked_queries,
 'minimum_extra_covered_queries':needed,'coverage_upper':(covered+repairable_queries)/total,'minimum_extra_codes_ALL25':value['minimum_distinct_additional_dev_codes_if_ALL25_currently_val_eligible_deficits_repaired'],'minimum_extra_codes_ALL44':value['minimum_distinct_additional_dev_codes_if_ALL44_dev_deficits_repaired']}))
