"""Independent retained-byte/ledger audit; no main/helper import or native run."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
import psutil
ROOT=Path.cwd(); DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
RAW=DOC/'meth469_switch_native_domain_capture_result.json'; RET=DOC/'RETENTION_469_20261005.json'
OUT=ROOT/'results/native_expert_scaling/meth469_switch_native_domain_capture'
EVENTS=ROOT/'results/native_expert_scaling/meth469_windows_terminal_v2.json'
ap=argparse.ArgumentParser();ap.add_argument('--expected-raw-sha',required=True);args=ap.parse_args()
assert not RET.exists();start=time.monotonic();hashed=peak=0
def guard():
 global peak
 info=psutil.Process().memory_info();peak=max(peak,info.rss,getattr(info,'peak_wset',0))
 assert peak<=1<<30 and time.monotonic()-start<=300
def sha(data):return hashlib.sha256(data).hexdigest()
def digest(p):
 global hashed
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  while data:=s.read(4<<20):h.update(data);hashed+=len(data);guard()
 return h.hexdigest()
def head(p):
 rel=p.relative_to(ROOT).as_posix();assert p.read_bytes()==subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters','--path='+rel,'HEAD:'+rel])
assert digest(RAW)==args.expected_raw_sha
j=json.loads(RAW.read_text(encoding='utf-8'))
b=json.loads((DOC/'meth468_prospective_bindings.json').read_text(encoding='utf-8'))
assert all(j['gates'].values()) and len(j['gates'])==10
for p,h in [(ROOT/'benchmarks/native_expert_scaling/meth469_switch_native_domain_capture.py',j['controller_sha256']),
 (DOC/'METH_469_SWITCH_NATIVE_DOMAIN_CAPTURE_PROTOCOL_20261005.md',j['protocol_sha256']),
 (DOC/'meth468_prospective_bindings.json',j['source_binding_sha256']),
 (ROOT/'benchmarks/phase60/engine.c',j['preserved_engine_sha256'])]:head(p);assert digest(p)==h
assert (ROOT/'.gitattributes').read_bytes()==subprocess.check_output(['git','show','HEAD:.gitattributes'])
for rel,v in b['helpers'].items():head(ROOT/rel);assert digest(ROOT/rel)==v['sha256']
for name,v in b['records'].items():head(DOC/name);assert digest(DOC/name)==v['sha256']
for name,h in j['first468_fault_bindings'].items():head(DOC/name);assert digest(DOC/name)==h
for path,v in b['runtime']['files'].items():assert Path(path).stat().st_size==v['bytes'] and digest(path)==v['sha256']
for pkg,v in b['runtime']['packages'].items():
 for path,row in v['files'].items():assert Path(path).stat().st_size==row['bytes'] and digest(path)==row['sha256']
for path,v in b['native_files'].items():assert Path(path).stat().st_size==v['bytes'] and digest(path)==v['sha256']
assert j['artifact']==b['original_artifact'] and digest(j['artifact']['payload'])==j['artifact']['sha256']
p=Path(j['artifact']['payload']);assert [p.stat().st_size,p.stat().st_mtime_ns]==j['payload_stat_before']==[b['payload_stat_before']['bytes'],b['payload_stat_before']['mtime_ns']]
cohort=json.loads((DOC/'meth467_switch_rust_query_manifest.json').read_text(encoding='utf-8'))
capture=json.loads((DOC/'meth393_switch_router_audit_result.json').read_text(encoding='utf-8'))
target=next(v for v in capture['targets'] if v['n']==128)
old_fault=json.loads((DOC/'meth468_switch_native_domain_capture_result.failure.json').read_text(encoding='utf-8'))
old_ret=json.loads((DOC/'RETENTION_468_20261005.json').read_text(encoding='utf-8'))
for v in old_ret['files']:assert Path(v['path']).stat().st_size==v['bytes'] and digest(v['path'])==v['sha256']
assert j['first_observed468_dev_teacher_bytes_exact_reconstruction_counted_once'] is True
assert len(j['reused_files'])==192
for r in j['reused_files']:
 assert digest(r['whole'])==r['whole_sha256'] and digest(r['trace'])==r['trace_sha256']
assert len(j['commands'])==1033 and len(j['cases'])==512 and len(cohort['items'])==128
for i,c in enumerate(j['commands']):
 assert c['returncode']==(2 if i==0 else 0) and c['negative']==(i==0)
 for k in ('stdout','stderr'):assert digest(c[k]['path'])==c[k]['sha256']
 if i:
  assert digest(c['whole_output_path'])==c['whole_output_sha256'] and digest(c['trace_path'])==c['trace_sha256']
  native=c['native_row'];assert native['threads']==3 and native['profile']==0 and native['repetition']==0
  assert [v['actual_mask'] for v in native['worker_affinity']]==[1,4,16]
  assert all(v['group']==0 for v in native['worker_affinity'])
for i,bi in enumerate((0,11,12,23)):
 e=next(v for v in target['quality_bridges'] if v['book']==bi and v['case']==0)
 for mi,mode in enumerate(('teacher','natural')):
  c=j['commands'][1+2*i+mi];old=e['teacher_rows' if mi==0 else 'generation_rows'][0]
  assert c['whole_output_sha256']==old['output_sha256'] and c['trace_sha256']==old['router_trace_sha256']
first=j['commands'][9];assert first['whole_output_sha256']==old_fault['commands'][-1]['whole_output_sha256'] and first['trace_sha256']==old_fault['commands'][-1]['trace_sha256']

def bucket():return {'selected':0,'executed':0,'rejected':0,'probability_sum':0.,'books':set(),'cases':set(),'input':set(),'code':{},'pair':set(),'scale':set()}
states=[[bucket() for _ in range(4)] for _ in range(1536)]
wire=struct.Struct('<H6BHIff32s32s32s');assert wire.size==118
dtype=np.dtype([('index','<u4'),('phase','<u4'),('input','<f4',(768,)),('scores','<f4',(128,))])
ledger_path=Path(j['ledger']['path']);assert digest(ledger_path)==j['ledger']['sha256']
assert ledger_path.stat().st_size==20+j['ledger']['records']*118==j['ledger']['bytes']
records_seen=0;totals={'teacher':0,'natural':0};healthy_seen=0
def healthy(ids):
 fields=[[] for _ in range(4)];expected=0;current=-1;bad=closed=False
 for token in ids:
  if token==1:break
  if token==0:bad=True;continue
  if token>=32000:
   if expected<=4 and token==32099-expected:
    if expected==4:closed=True;current=-1
    else:current=expected
    expected+=1
   else:bad=True
  elif 0<=current<4:fields[current].append(token)
  else:bad=True
 if bad:fields=[[] for _ in range(4)]
 triples=[[tuple(f[i:i+3]) for i in range(max(len(f)-2,0))] for f in fields]
 total=sum(map(len,triples));repeat=(total-sum(len(set(v)) for v in triples))/max(total,1)
 return bool(closed and expected==5 and not bad and all(fields) and ids[-1]==32095 and repeat<=.50)
with ledger_path.open('rb') as ledger:
 assert ledger.read(20)==struct.pack('<8sIQ',b'MQ469L01',118,j['ledger']['records'])
 for case_index,entry in enumerate(j['cases']):
  bi,ci=divmod(case_index,4);book=cohort['items'][bi];case=book['cases'][ci];split=int(bi>=64)
  assert entry['book']==bi and entry['case']==ci and entry['role']==book['role']==('development' if bi<64 else 'diagnostic_validation')
  assert entry['source_ids_sha256']==case['source_ids_sha256'];teacher=None
  for mi,mode in enumerate(('teacher','natural')):
   command_index=9+2*case_index+mi;assert entry['modes'][mode]==command_index
   c=j['commands'][command_index];assert c['label']==f'book{bi}.case{ci}.{mode}' and c['mode']==mode
   argv=c['argv'];assert argv[0]==j['native_binary']['path'] and argv[4]==str(OUT/f'book{bi}.case{ci}.{mode}')
   if mi==0:assert argv[2]==','.join(map(str,case['source_ids'])) and argv[3]==','.join(map(str,case['decoder_ids']))
   else:assert argv[1]=='--generate' and argv[3]==','.join(map(str,case['source_ids']))
   t=14 if mi==0 else c['native_row']['actual_generated_tokens'];nrows=6*(29+t)
   trace_data=Path(c['trace_path']).read_bytes();whole_data=Path(c['whole_output_path']).read_bytes()
   assert trace_data[:16]==struct.pack('<8sII',b'SWRTA001',128,768) and len(trace_data)==16+nrows*dtype.itemsize
   traced=np.frombuffer(trace_data,dtype,offset=16)
   assert np.array_equal(traced['index'],np.arange(nrows)) and np.all(traced['phase'][:174]==0) and np.all(traced['phase'][174:]==1)
   assert np.all(np.isfinite(traced['input'])) and np.all(np.isfinite(traced['scores']))
   assert whole_data[:36]==struct.pack('<8s7I',b'SWR32O01',29,t,768,12,12,32128,nrows)
   route_offset=36+4*(14*29*768+14*t*768+t*32128)
   assert len(whole_data)==route_offset+12*nrows
   routes=np.frombuffer(whole_data,dtype=[('expert','<i4'),('accepted','<i4'),('p','<f4')],offset=route_offset)
   assert np.all(routes['accepted']==1) and np.array_equal(np.argmax(traced['scores'],axis=1),routes['expert'])
   logits_offset=36+4*(14*29*768+14*t*768)
   bridge=(trace_data[16:16+180*dtype.itemsize],whole_data[36:36+4*14*29*768],
           whole_data[36+4*14*29*768:36+4*14*29*768+4*14*768],whole_data[logits_offset:logits_offset+4*32128])
   if mi==0:teacher=bridge
   else:
    assert bridge==teacher
    logits=np.frombuffer(whole_data,'<f4',t*32128,logits_offset).reshape(t,32128)
    assert np.argmax(logits,axis=1).tolist()==c['native_row']['generated_ids']
    good=healthy(c['native_row']['generated_ids']);assert good==c['natural_health']['healthy_complete_nonempty_fields'];healthy_seen+=good
   # Independent vector reconstruction from retained raw F32 normalized inputs.
   maximum=np.max(np.abs(traced['input']),axis=1)
   with np.errstate(under='ignore'):
    scale=(maximum/np.float32(32767)).astype(np.float32)
   scale[maximum==0]=1.
   quant=np.clip(np.rint(np.divide(traced['input'],scale[:,None],dtype=np.float32)),-32767,32767).astype('<i2')
   raw_ledger=ledger.read(nrows*118);assert len(raw_ledger)==nrows*118
   for index,r in enumerate(wire.iter_unpack(raw_ledger)):
    ni,m,book_idx,case_idx,bank,accepted,role,expert,trace_idx,alpha,probability,hi,hq,hp=r
    expected_bank=index//29 if index<174 else 6+(index-174)%6
    assert (ni,m,book_idx,case_idx,bank,accepted,role,expert,trace_idx)==(128,mi,bi,ci,expected_bank,1,split,int(routes[index]['expert']),index)
    sb=struct.pack('<f',float(scale[index]));qbytes=quant[index].tobytes()
    assert struct.pack('<f',alpha)==sb and struct.pack('<f',probability)==routes[index]['p'].tobytes()
    assert hi==hashlib.sha256(traced[index]['input'].tobytes()).digest()
    assert hq==hashlib.sha256(qbytes).digest() and hp==hashlib.sha256(qbytes+sb).digest()
    st=states[bank*128+expert][2*split+mi]
    st['selected']+=1;st['executed']+=1;st['probability_sum']+=probability
    st['books'].add(bi);st['cases'].add((bi,ci));st['input'].add(hi);st['pair'].add(hp);st['scale'].add(sb)
    st['code'][hq]=st['code'].get(hq,0)|(1<<bi);records_seen+=1
   totals[mode]+=nrows;guard()
 assert not ledger.read(1)
assert records_seen==j['ledger']['records'] and totals==j['route_query_totals']
assert healthy_seen==j['natural_health']['healthy'] and j['natural_health']['all_cases']==512 and j['natural_health']['unhealthy']==512-healthy_seen
def union(bs):
 answer={}
 for st in bs:
  for k,v in st['code'].items():answer[k]=answer.get(k,0)|v
 return answer
def summary(bs):
 return {'selected':sum(s['selected'] for s in bs),'executed':sum(s['executed'] for s in bs),'rejected':sum(s['rejected'] for s in bs),
 'executed_router_probability_sum':sum(s['probability_sum'] for s in bs),'books':sorted(set.union(*(s['books'] for s in bs))),
 'cases':len(set.union(*(s['cases'] for s in bs))),'unique_input_SHA':len(set.union(*(s['input'] for s in bs))),
 'unique_code_SHA':len(union(bs)),'unique_code_scale_SHA':len(set.union(*(s['pair'] for s in bs))),'unique_scale_bytes':len(set.union(*(s['scale'] for s in bs)))}
assert len(j['banks'])==12
for bank,stored in enumerate(j['banks']):
 ready_count=covered=denominator=0
 assert stored['bank']==bank and len(stored['experts'])==128
 for expert,row in enumerate(stored['experts']):
  bs=states[bank*128+expert];dev=union(bs[:2]);val=union(bs[2:]);novel=set(val)-set(dev);mask=0
  for key in novel:mask|=val[key]
  novel_books=[i for i in range(128) if mask&(1<<i)];ds=summary(bs[:2]);vs=summary(bs[2:])
  assert row['expert']==expert and row['dev_union']==ds and row['val_union']==vs
  assert row['validation_code_seen_in_dev']==len(set(dev)&set(val)) and row['validation_novel_codes']==len(novel) and row['validation_novel_code_books']==novel_books
  assert row['by_split_mode']=={f'{split}_{mode}':summary([bs[2*si+mi]]) for si,split in enumerate(('dev','val')) for mi,mode in enumerate(('teacher','natural'))}
  assert row['cross_mode_identical_input_dev_val']==[len(bs[i]['input']&bs[i+1]['input']) for i in (0,2)]
  assert row['cross_mode_identical_code_dev_val']==[len(set(bs[i]['code'])&set(bs[i+1]['code'])) for i in (0,2)]
  gates={'development_unique_codes_ge32':len(dev)>=32,'development_books_ge4':len(ds['books'])>=4,'validation_novel_codes_ge16':len(novel)>=16,'validation_novel_code_books_ge4':len(novel_books)>=4}
  ready=all(gates.values());assert gates==row['readiness_gates'] and row['data_ready']==ready
  assert row['sampled_dev_span_rank_UPPER_bound_NOT_measured_rank']==min(768,len(dev))
  ready_count+=ready;denominator+=bs[3]['executed'];covered+=bs[3]['executed'] if ready else 0
 coverage=covered/denominator if denominator else 0.
 assert (stored['data_ready_IDs'],stored['val_natural_executed_queries'],stored['covered_val_natural_executed_queries'],stored['coverage'])==(ready_count,denominator,covered,coverage)
 assert stored['passed']==(ready_count>=32 and coverage>=.90);guard()
assert j['fixed_future_probe']['data_ready']==j['banks'][11]['passed']
expected_decision='data_admitted_for_separately_frozen_ONE_rank32_private_INPUT_probe' if j['banks'][11]['passed'] else 'new_domain_insufficient_fixed_bank11_stop_before_factors_reassess_information_or_geometry'
assert j['decision']==expected_decision

progress=[json.loads(line) for line in (OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
assert progress[-1]['admitted'] is True and progress[-1]['native_commands']==1033 and progress[-1]['raw_sha256']==args.expected_raw_sha
assert len([r for r in progress if 'completed_book' in r])==128
assert [r['child_started'] for r in progress if 'child_started' in r]==[c['pid'] for c in j['commands']]
assert [r['child_terminal'] for r in progress if 'child_terminal' in r]==[c['pid'] for c in j['commands']]
assert (OUT/'fatal_native.log').stat().st_size==0
assert j['forbidden_modules_after_all_cases']==[]
events=json.loads(EVENTS.read_text(encoding='utf-8'))
metadata_fault_path=DOC/'meth469_terminal_query_metadata_fault.json'
metadata_fault=json.loads(metadata_fault_path.read_text(encoding='utf-8'))
assert digest(metadata_fault_path)=='a66d15a54a909561337245c455246643462c28216ccc4a690627608a55630f4c'
assert datetime.fromisoformat(events['query_start_utc']).timestamp() <= datetime.fromisoformat(j['start_utc']).timestamp()
assert datetime.fromisoformat(events['query_end_utc']).timestamp() >= datetime.fromisoformat(j['end_utc']).timestamp()
assert digest(EVENTS)==metadata_fault['corrected_query']['sha256']
assert events['query_available'] and events['query_error'] is None and events['matching_main_or_native_events']==[]
assert events['main_pid']==progress[0]['pid'] and set(events['native_pids'])=={c['pid'] for c in j['commands']}
reused_pids=[]
for pid in {events['main_pid'],*events['native_pids']}:
 if not psutil.pid_exists(pid):continue
 try:proc=psutil.Process(pid);created=proc.create_time();name=proc.name().lower()
 except psutil.NoSuchProcess:continue
 if pid==events['main_pid']:
  assert not (name.startswith('python') and datetime.fromisoformat(j['start_utc']).timestamp()-2<=created<=datetime.fromisoformat(j['end_utc']).timestamp())
 else:
  for c in j['commands']:
   if c['pid']==pid:assert not (name=='meth393_switch_router_audit.exe' and datetime.fromisoformat(c['start_utc']).timestamp()-2<=created<=datetime.fromisoformat(c['end_utc']).timestamp())
 reused_pids.append({'pid':pid,'current_name':name,'current_creation_unix':created})
own={psutil.Process().pid,*(p.pid for p in psutil.Process().parents())};daemons=[]
for p in psutil.process_iter(['name','cmdline']):
 if p.pid in own:continue
 name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
 if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():daemons.append(p.pid);continue
 assert not (name.startswith('python') or name=='clang.exe' or (name.startswith('meth') and name.endswith('.exe')))
for rel,row in b['preserved_unrelated_files'].items():assert digest(ROOT/rel)==row['sha256']
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).splitlines()==b['tracked_status']
files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.iterdir())]
all_bytes=sum(v['bytes'] for v in files)+RAW.stat().st_size
terminal_progress_row_bytes=len((OUT/'progress.jsonl').read_bytes().splitlines(keepends=True)[-1])
assert all_bytes==progress[-1]['all_output_bytes_including_raw']+terminal_progress_row_bytes+RAW.stat().st_size<=12<<30
assert digest(DOC/'meth469_output_counter_metadata_fault.json')=='1bbefb2145654fa2b50804e789cba3042f9cbd34804efddd7282b7a2f62c918c'
assert j['resource']['main_seconds_including_heavy_imports_through_admission']<=2100
assert progress[-1]['conservative_parent_plus_largest_child_peak_bytes']<=16<<30 and progress[-1]['maximum_sampled_parent_all_descendant_RSS_sum']<=16<<30
value={'experiment':'METH469-independent-complete-capture-ledger-retention','raw_sha256':args.expected_raw_sha,'helper_sha256':digest(__file__),
 'gates':{'frozen_sources_original_payload_runtime_records_physicalHEAD_SHA_exact':True,'ALL1033_command_logs_1024cohort_streams_8golden_fullbytes_retained':True,
 'ALL_ledger_queries_reconstructed_from_actual_trace_inputs_A16_codes_scales_route_ids_roles_exact':True,'all1536_perID4role_tables_novelty_books_and12bank_coverage_rederived':True,
 'all512_correlated_encoder_firstdecoder_views_exact_not_independent':True,'all128immutable_book_roles_all512cases_no_health_filter_exact':True,
 'first468fault_and_same_observed_teacher_bytes_retained_counted_once':True,'actual_terminal_original_instances_exited_empty_fault_forbiddenmodules_no_WindowsAPPCRASH':True,
 'all_outputs_SHA_resources_and3unrelated_status_M_byte_hashes_preserved':True},
 'ledger_records':records_seen,'books':128,'cases':512,'native_commands':1033,'fixed_bank11':{k:j['banks'][11][k] for k in ('data_ready_IDs','coverage','passed','val_natural_executed_queries','covered_val_natural_executed_queries')},
 'decision':j['decision'],'main_terminal':{'tool_session':5698,'tool_exit_code':0,'actual_pid':events['main_pid']},'windows_events':events,'windows_events_sha256':digest(EVENTS),'first_metadata_query_fault_sha256':digest(metadata_fault_path),
 'files':files,'all_outputs_including_raw_bytes':all_bytes,'main_terminal_progress':progress[-1],
 'preserved_daemon_pids':daemons,'reused_PID_identity_observations':reused_pids,'terminal_progress_row_bytes_after_output_snapshot':terminal_progress_row_bytes,'output_counter_metadata_fault_sha256':digest(DOC/'meth469_output_counter_metadata_fault.json'),
 'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed,
 'scope':'Independent exact retained-byte/query-ledger reconstruction with standard NumPy; no scientific main/helper import, no native/model/tokenizer run or fit. Data readiness is not rank/generalization/compressed knowledge/quality/speed/useful-n/physicalDRAM.'}
guard()
with RET.open('xb') as stream:stream.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'sha256':digest(RET),'gates':value['gates'],'ledger_records':records_seen,'bank11':value['fixed_bank11'],'seconds':value['seconds'],'OS_peak_bytes':peak,'all_output_bytes':all_bytes}))
