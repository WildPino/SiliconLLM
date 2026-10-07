"""Metadata-only terminal admission of fixed511 decisions; zero numerical/model/C calls."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);peak=0
def guard():
    global peak
    peak=max(peak,proc.memory_info().peak_wset);assert peak<=512<<20 and time.monotonic()-start<=180
def receipt(path,expected=None):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(8<<20):h.update(data);guard()
    r={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    if expected:assert r['sha256']==expected
    return r
def read(name,sha=None):
    r=receipt(DOC/name,sha);return r,json.loads(Path(r['path']).read_bytes())
expected={
 'meth511_r3_cohort_result.json':'90b790c2723e96bc17408f8da4b2285022965276968d61be226b94981059de4c',
 'meth511_r7_native_result.json':'278df0c17a7a07000c1bda446f6b090d1f2f27dad8d853464a9ccbf9f6ca681f',
 'meth511_r8_donor_result.json':'4b73168632b025f3d5977e2f2c6e5fa75c495a4d00777bf1488332785d7ae195',
 'meth511_r9_audit_result.json':'48e95e9fa6d0466c89337fadc77877def862a42165d10a9b5f45c525b30fab27',
 'meth511_r9_windows_terminal.json':'34e4de5a96486e0aac59810881410545b4ad3cfe914fb6531393c0c2397b7063'}
receipts={};raw={}
for name,sha in expected.items():receipts[name],raw[name]=read(name,sha)
cohort,native,donor,audit,events=[raw[n] for n in expected]
binding_receipt,b=read('meth511_r9_binding.json','8ac4e645149554677cbd6100334e1ca4d5aad5b3a481dc5e87fa2f9cdbb3987b')
assert all(all(r['gates'].values()) for r in [cohort,native,donor,audit])
assert native['native_calls']==23 and native['retained_native_calls']==553 and len(native['commands'])==576
assert donor['model_calls']==384 and donor['canonical_F32_bytes_hashed_before_first_inference']==29956961280
assert donor['summary']==audit['summary'] and audit['independent_head_states']==4983
assert all(r['cohort_sha256']==cohort['cohort_sha256'] for r in [native,donor,audit])
assert events['known_record_ids']==[179791,179810] and events['positive_control']['query_available']
assert set(events['known_record_ids'])<={e['RecordId'] for e in events['positive_control']['events']}
assert [s['name'] for s in events['stages']]==['cohort','native','donor','audit']
for s in events['stages']:
    assert s['query']['query_available'] and not s['matching_events']
    assert receipt(s['raw_path'])['sha256']==s['raw_sha256']
instances=[]
for s in events['stages']:
    for v in s['instances']:
        try:assert abs(psutil.Process(v['pid']).create_time()-v['create_time_unix'])>=.002
        except psutil.NoSuchProcess:pass
        instances.append(v)
for rel,sha in b['preserved'].items():assert receipt(ROOT/rel)['sha256']==sha
assert set(subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines())==set(b['preserved'])
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
resources={}
for kind,prefix in [('cohort','meth511_r3'),('native','meth511_r7'),('donor','meth511_r8'),('audit','meth511_r9')]:
    p=ROOT/'results/native_expert_scaling'/f'{prefix}_{kind}'/'terminal_resource.json';rr=receipt(p);v=json.loads(p.read_bytes())
    assert v['seconds']<=v['limits'][0] and v['OS_peak_bytes']<=v['limits'][1]
    assert v['native_OS_peak_bytes']<=4<<30 and v['result_sha256']==receipts[{'cohort':'meth511_r3_cohort_result.json','native':'meth511_r7_native_result.json','donor':'meth511_r8_donor_result.json','audit':'meth511_r9_audit_result.json'}[kind]]['sha256']
    resources[kind]={'receipt':rr,**v}
_,partial=read('meth511_r6_native_result.failure.json','1384c005600e796dfd1cf47936c084e0ca35f1144f18109dedca2d12bb166e1b')
_,donor_fault=read('meth511_r7_donor_result.failure.json','ccf675ff6cea63e8988c8ed60cfde13a2ed522dcc69b271f5f623c6a4209cfeb')
_,audit_fault=read('meth511_r8_audit_result.failure.json','e91cfcdf9883e66af001b4a9ad1d4b693bacf62b4dff28d739b761998d1242fe')
combined={'native':partial['seconds']+resources['native']['seconds'],'donor':donor_fault['seconds']+resources['donor']['seconds'],'audit':audit_fault['seconds']+resources['audit']['seconds']}
assert combined['native']<2400 and combined['donor']<3600 and combined['audit']<1200
owned=[]
for p in (ROOT/'results/native_expert_scaling').glob('meth511*'):
    if p.is_dir():
        for parent,dirs,files in os.walk(p,followlinks=False):
            dirs[:]=[n for n in dirs if not (Path(parent)/n).is_symlink() and not (Path(parent)/n).is_junction()]
            owned.extend(Path(parent)/n for n in files if not (Path(parent)/n).is_symlink() and not (Path(parent)/n).is_junction())
owned.extend(DOC.glob('meth511*.json'));output_bytes=sum(p.stat().st_size for p in owned);assert output_bytes<=18<<30
s=donor['summary'];quality=all(s['quality_gates'].values());economics=all(s['economic_gates'].values())
qualified=quality and s['economic_gates']['candidate_ordinary_warm_and_lower_ge50'] and s['economic_gates']['candidate_prose_warm_and_lower_ge50']
assert qualified and not economics and not s['whole_recipe_eligible']
result={'experiment':'METH511 fixed8 original-row readout plus exact conditional WO, fresh whole evaluation',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'receipts':receipts,'final_binding':binding_receipt,'artifacts':{k:b[k] for k in ['source_binary','candidate_binary','source_manifest','candidate_manifest','source_payload','candidate_payload']},
 'admission':{'all_native_donor_and_independent_audit_gates':True,'typed_UTC_positive_controls_and_zero_relevant_Event1000':True,
     'all_stage_and_576_C_instances_closed':True,'preserved_foreign_bytes':True,'resource_and_total_output_bounds':True},
 'instances_checked':instances,'resource_receipts':resources,'combined_guard_seconds_with_retained_first_faults':combined,'new_physical_output_bytes_excluding_foreign_junction_targets':output_bytes,
 'timing_scope':'Two separately bound quiet intervals; all553 retained and23 new C calls occurred once. Original untimed preflight idle CPU tick0.015625s and interrupted first controller remain explicit. Sampled scientific topology/idle counters; other OS work remains.',
 'fresh_project_excluded_books':24,'cases':96,'source_unique_parameters':7415217408,'experts_per_original_bank':128,'original_banks':12,
 'audited_head_states':4983,'audited_full_head_cells':4983*32128,'candidate_tail_overtakes':audit['candidate_tail_overtakes'],
 'canonical_original_F32_bytes_hashed':donor['canonical_F32_bytes_hashed_before_first_inference'],'summary':s,
 'same_artifact_fresh_quality_AND_warm_accepted_rate_ge50_qualified_in_declared_scope':qualified,
 'whole_fixed_recipe_eligible':False,'goal_complete':False,
 'limits':['24-book project-heldout bounded infilling; no donor-pretraining exclusion claim.','Warm generation includes encoder/cross-KV/decode/router/head, failed cases zero accepted IDs with all time retained. First-request prose27.1382/s differs from warm52.5586/lower50.4685.',
    'Six book regressions; worst1.2586986. Frozen all-book criterion fails, not relaxed.','Full source-sized WI/fixed128 original parents; compact reusable geometry/useful much larger n/LUT winner and normalized mass remain open.','Physical DRAM unverified; logical counters and process working sets are not memory-controller measurements.','Additional actual families/scales~10B/~100B remain unqualified.'],
 'metadata_finalization':{'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'limits':[180,512<<20],'new_numerical_model_C_calls':0}}
destination=DOC/'EVALUATION_511_20261007.json';guard()
with destination.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
r=receipt(destination);guard();print(json.dumps({'evaluation':r,'qualified_warm_quality_rate':qualified,'whole_recipe_eligible':False,'goal_complete':False,'output_bytes':output_bytes,'combined_seconds':combined,'metadata_seconds':time.monotonic()-start,'OS_peak_bytes':peak}),flush=True)
