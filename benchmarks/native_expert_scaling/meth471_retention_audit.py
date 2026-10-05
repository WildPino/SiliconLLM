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
RAW=DOC/'meth471_switch_development_capture_result.json'; RET=DOC/'RETENTION_471_20261005.json'
OUT=ROOT/'results/native_expert_scaling/meth471_switch_development_capture'
EVENTS=ROOT/'results/native_expert_scaling/meth471_windows_terminal.json'
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
b=json.loads((DOC/'meth471_prospective_bindings.json').read_text(encoding='utf-8'))
assert all(j['gates'].values()) and len(j['gates'])==13
for p,h in [(ROOT/'benchmarks/native_expert_scaling/meth471_switch_development_capture.py',j['controller_sha256']),
 (DOC/'METH_471_SWITCH_DEVELOPMENT_CAPTURE_PROTOCOL_20261005.md',j['protocol_sha256']),
 (DOC/'meth471_prospective_bindings.json',j['source_binding_sha256']),
 (ROOT/'benchmarks/phase60/engine.c',j['preserved_engine_sha256'])]:head(p);assert digest(p)==h
assert (ROOT/'.gitattributes').read_bytes()==subprocess.check_output(['git','show','HEAD:.gitattributes'])
for rel,v in b['helpers'].items():head(ROOT/rel);assert digest(ROOT/rel)==v['sha256']
for name,v in b['reused_protocols'].items():head(DOC/name);assert digest(DOC/name)==v['sha256']
assert digest(b['preparation_helper']['path'])==b['preparation_helper']['sha256']
for row in b['previous_output_inventory']:
 p=Path(row['path']);assert p.stat().st_size==row['bytes'] and p.stat().st_mtime_ns==row['mtime_ns'] and digest(p)==row['sha256']
assert len(b['previous_output_inventory'])==4136 and sum(v['bytes'] for v in b['previous_output_inventory'])==4566198260
for name,v in b['records'].items():head(DOC/name);assert digest(DOC/name)==v['sha256']
for name,h in j['first468_fault_bindings'].items():head(DOC/name);assert digest(DOC/name)==h
for path,v in b['runtime']['files'].items():assert Path(path).stat().st_size==v['bytes'] and digest(path)==v['sha256']
for pkg,v in b['runtime']['packages'].items():
 for path,row in v['files'].items():assert Path(path).stat().st_size==row['bytes'] and digest(path)==row['sha256']
for path,v in b['native_files'].items():assert Path(path).stat().st_size==v['bytes'] and digest(path)==v['sha256']
assert j['artifact']==b['original_artifact'] and digest(j['artifact']['payload'])==j['artifact']['sha256']
p=Path(j['artifact']['payload']);assert [p.stat().st_size,p.stat().st_mtime_ns]==j['payload_stat_before']==[b['payload_stat_before']['bytes'],b['payload_stat_before']['mtime_ns']]
original_cohort=json.loads((DOC/'meth467_switch_rust_query_manifest.json').read_text(encoding='utf-8'))
cohort=json.loads((DOC/'meth470_switch_development_manifest.json').read_text(encoding='utf-8'))
previous=json.loads((DOC/'meth469_switch_native_domain_capture_result.json').read_text(encoding='utf-8'))
books=original_cohort['items']+cohort['items']
assert len(books)==192 and [v['book'] for v in books]==list(range(192))
roles={v['book']:{'development':0,'diagnostic_validation':1,'development_augmentation':0}[v['role']] for v in books}
assert roles[63]==roles[128]==roles[191]==0 and roles[64]==roles[127]==1
assert j['immutable_book_roles']==[{'book':v['book'],'role':v['role'],'split':roles[v['book']],
 'source_id':v['source_id'],'whole_source_utf8_sha256':v['whole_source_utf8_sha256']} for v in books]
capture=json.loads((DOC/'meth393_switch_router_audit_result.json').read_text(encoding='utf-8'))
target=next(v for v in capture['targets'] if v['n']==128)
old_fault=json.loads((DOC/'meth468_switch_native_domain_capture_result.failure.json').read_text(encoding='utf-8'))
old_ret=json.loads((DOC/'RETENTION_468_20261005.json').read_text(encoding='utf-8'))
for v in old_ret['files']:assert Path(v['path']).stat().st_size==v['bytes'] and digest(v['path'])==v['sha256']
assert len(j['reused_files'])==192
for r in j['reused_files']:
 assert digest(r['whole'])==r['whole_sha256'] and digest(r['trace'])==r['trace_sha256']
assert len(j['commands'])==521 and len(j['cases'])==256 and len(cohort['items'])==64
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
for name,path in j['actual_loaded_package_paths'].items():
 assert path in b['runtime']['packages'][name.partition('.')[0]]['files'],(name,path)
for c in previous['commands']:
 assert c['returncode']==(2 if c['negative'] else 0)
 for k in ('stdout','stderr'):assert digest(c[k]['path'])==c[k]['sha256']
 if not c['negative']:
  assert digest(c['whole_output_path'])==c['whole_output_sha256'] and digest(c['trace_path'])==c['trace_sha256']

def bucket():return {'selected':0,'executed':0,'rejected':0,'probability_sum':0.,'books':set(),'cases':set(),'input':set(),'code':{},'pair':set(),'scale':set()}
states=[[bucket() for _ in range(4)] for _ in range(1536)]
wire=struct.Struct('<H6BHIff32s32s32s');assert wire.size==118
dtype=np.dtype([('index','<u4'),('phase','<u4'),('input','<f4',(768,)),('scores','<f4',(128,))])
ledger_path=Path(j['ledger']['path']);assert digest(ledger_path)==j['ledger']['sha256']
assert ledger_path.stat().st_size==20+j['ledger']['records']*118==j['ledger']['bytes']
records_seen=0;totals={'teacher':0,'natural':0};new_totals={'teacher':0,'natural':0};healthy_seen={'old':0,'new':0}
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
 assert ledger.read(20)==struct.pack('<8sIQ',b'MQ471L01',118,j['ledger']['records'])
 for case_index in range(768):
  bi,ci=divmod(case_index,4);book=books[bi];case=book['cases'][ci];split=roles[bi]
  source_j=previous if bi<128 else j
  local_case=case_index if bi<128 else case_index-512
  entry=source_j['cases'][local_case]
  assert entry['book']==bi and entry['case']==ci and entry['role']==book['role']
  assert entry['source_ids_sha256']==case['source_ids_sha256'];teacher=None
  for mi,mode in enumerate(('teacher','natural')):
   command_index=9+2*local_case+mi;assert entry['modes'][mode]==command_index
   c=source_j['commands'][command_index];assert c['label']==f'book{bi}.case{ci}.{mode}' and c['mode']==mode
   argv=c['argv'];assert argv[0]==j['native_binary']['path'] and argv[4]==str((Path(previous['ledger']['path']).parent if bi<128 else OUT)/f'book{bi}.case{ci}.{mode}')
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
   all_floats=np.frombuffer(whole_data,'<f4',count=(route_offset-36)//4,offset=36)
   assert np.all(np.isfinite(all_floats))
   selected=np.argmax(traced['scores'],axis=1)
   differences=traced['scores']-traced['scores'][np.arange(nrows),selected,None]
   expected_p=(1/np.exp(differences.astype(np.float64)).astype(np.float32).astype(np.float64).sum(axis=1)).astype(np.float32)
   assert np.max(np.abs(expected_p.astype(np.float64)/routes['p']-1))<=1e-6
   native_rows=[json.loads(v) for v in Path(c['stdout']['path']).read_text(encoding='utf-8').splitlines()]
   assert native_rows==[c['native_row']]
   counts=c['native_row']['counters'][1]
   assert sum(v['code_bytes'] for v in counts)==123764736*t
   assert sum(v['scale_bytes'] for v in counts)==534016*t
   assert sum(v['f32_bytes'] for v in counts)==18432*128*t
   logits_offset=36+4*(14*29*768+14*t*768)
   bridge=(trace_data[16:16+180*dtype.itemsize],whole_data[36:36+4*14*29*768],
           whole_data[36+4*14*29*768:36+4*14*29*768+4*14*768],whole_data[logits_offset:logits_offset+4*32128])
   if mi==0:teacher=bridge
   else:
    assert bridge==teacher
    logits=np.frombuffer(whole_data,'<f4',t*32128,logits_offset).reshape(t,32128)
    assert np.argmax(logits,axis=1).tolist()==c['native_row']['generated_ids']
    good=healthy(c['native_row']['generated_ids']);assert good==c['natural_health']['healthy_complete_nonempty_fields'];healthy_seen['old' if bi<128 else 'new']+=good
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
   totals[mode]+=nrows
   if bi>=128:new_totals[mode]+=nrows
   guard()
 assert not ledger.read(1)
assert records_seen==j['ledger']['records'] and totals==j['route_query_totals']
assert healthy_seen['old']==j['natural_health']['reused_healthy']==previous['natural_health']['healthy']
assert healthy_seen['new']==j['natural_health']['new_healthy'] and j['natural_health']['new_all_cases']==256 and j['natural_health']['new_unhealthy']==256-healthy_seen['new']
assert sum(healthy_seen.values())==j['natural_health']['combined_healthy'] and j['natural_health']['combined_all_cases']==768
assert new_totals==j['new_route_query_totals'] and {k:totals[k]-new_totals[k] for k in totals}==previous['route_query_totals']
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
  novel_books=[i for i in range(192) if mask&(1<<i)];ds=summary(bs[:2]);vs=summary(bs[2:])
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
expected_decision='data_admitted_for_separately_frozen_ONE_rank32_private_INPUT_probe' if j['banks'][11]['passed'] else 'ONE_development_augmentation_DATAFAIL_stop_this_data_ladder_reassess_geometry_or_data_contract'
assert j['decision']==expected_decision

assert j['banks'][11]['val_natural_executed_queries']==3065
for bank,oldbank in zip(j['banks'],previous['banks']):
 assert bank['val_natural_executed_queries']==oldbank['val_natural_executed_queries']
 for row,oldrow in zip(bank['experts'],oldbank['experts']):
  assert row['val_union']==oldrow['val_union']
  assert all(row['by_split_mode'][key]==oldrow['by_split_mode'][key] for key in ('val_teacher','val_natural'))
  assert row['dev_union']['unique_code_SHA']>=oldrow['dev_union']['unique_code_SHA']
  assert row['validation_novel_codes']<=oldrow['validation_novel_codes']
  assert set(row['validation_novel_code_books']).issubset(oldrow['validation_novel_code_books'])
with Path(previous['ledger']['path']).open('rb') as oldledger,ledger_path.open('rb') as combined:
 assert oldledger.read(20)==struct.pack('<8sIQ',b'MQ469L01',118,258120)
 combined.seek(20)
 remaining=258120*118
 while remaining:
  count=min(remaining,4<<20);assert oldledger.read(count)==combined.read(count);remaining-=count;guard()
 assert not oldledger.read(1)
assert j['reused_469_ledger_and_tables']['records']==258120 and j['reused_469_ledger_and_tables']['all_old_banks_and_ID_tables_exact'] is True
progress_bytes=(OUT/'progress.jsonl').read_bytes()
progress=[json.loads(line) for line in progress_bytes.splitlines()]
terminal=progress[-1]
assert terminal['admitted'] is True and terminal['native_commands']==521 and terminal['raw_sha256']==args.expected_raw_sha
assert len([r for r in progress if 'completed_book' in r])==64
assert len([r for r in progress if 'reused_book' in r])==128
assert [r['child_started'] for r in progress if 'child_started' in r]==[c['pid'] for c in j['commands']]
assert [r['child_terminal'] for r in progress if 'child_terminal' in r]==[c['pid'] for c in j['commands']]
assert (OUT/'fatal_native.log').stat().st_size==0 and j['forbidden_modules_after_all_cases']==[]
events=json.loads(EVENTS.read_text(encoding='utf-8'))
assert datetime.fromisoformat(events['query_start_utc']).timestamp()<=j['main_process_instance']['create_time_unix']
assert datetime.fromisoformat(events['query_end_utc']).timestamp()>=datetime.fromisoformat(terminal['utc']).timestamp()
assert events['query_available'] and events['query_error'] is None and events['matching_main_or_native_events']==[]
assert events['main_pid']==progress[0]['pid']==j['main_process_instance']['pid']
assert set(events['native_pids'])=={c['pid'] for c in j['commands']}
instances=[j['main_process_instance']]+[c['process_instance'] for c in j['commands']]
reused_pids=[]
for instance in instances:
 try:
  proc=psutil.Process(instance['pid']);created=proc.create_time();name=proc.name();exe=proc.exe()
 except psutil.NoSuchProcess:continue
 assert not (name.lower()==instance['name'].lower() and abs(created-instance['create_time_unix'])<.01 and Path(exe).resolve()==Path(instance['executable']).resolve()),instance
 observation={'pid':instance['pid'],'current_name':name,'current_creation_unix':created,'current_executable':exe}
 if observation not in reused_pids:reused_pids.append(observation)
for c in j['commands']:
 inst=c['process_instance'];assert inst['pid']==c['pid'] and inst['name']==Path(j['native_binary']['path']).name
 assert inst['create_time_unix']==(inst['creation_filetime_100ns']-116444736000000000)/10000000.
 assert datetime.fromisoformat(c['start_utc']).timestamp()-2<=inst['create_time_unix']<=datetime.fromisoformat(c['end_utc']).timestamp()
 assert c['OS_peak_bytes']<=16<<30 and c['wall_seconds']<=120
own={psutil.Process().pid,*(p.pid for p in psutil.Process().parents())};daemons=[]
for p in psutil.process_iter(['name','cmdline']):
 if p.pid in own:continue
 try:
  name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
  if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():daemons.append(p.pid);continue
  assert not (name.startswith('python') or name=='clang.exe' or (name.startswith('meth') and name.endswith('.exe'))),(p.pid,name)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
for rel,row in b['preserved_unrelated_files'].items():assert digest(ROOT/rel)==row['sha256']
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).splitlines()==b['tracked_status']
files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.iterdir())]
OUT_bytes=sum(v['bytes'] for v in files);all_bytes=OUT_bytes+RAW.stat().st_size
terminal_row_bytes=len(progress_bytes.splitlines(keepends=True)[-1])
assert OUT_bytes==terminal['OUT_bytes_before_terminal_progress_row']+terminal_row_bytes
assert terminal['raw_bytes']==RAW.stat().st_size and all_bytes<=12<<30
assert terminal['final_file_bytes_hashed']==j['resource']['file_bytes_hashed_before_raw']+RAW.stat().st_size
assert j['resource']['main_seconds_including_heavy_imports_through_admission']<=terminal['seconds']<=2100
assert terminal['conservative_parent_plus_largest_child_peak_bytes']<=16<<30 and terminal['maximum_sampled_parent_all_descendant_RSS_sum']<=16<<30
assert j['prospective_byte_bounds']['new_native_and_combined_ledger_and_metadata_upper_bound']==5215181284
assert records_seen<=467016 and len(j['commands'])==521
assert not RAW.with_suffix('.failure.json').exists()
value={'experiment':'METH471-independent-development-extension-combined-ledger-retention',
 'raw_sha256':args.expected_raw_sha,'helper_sha256':digest(__file__),
 'gates':{'frozen_sources_original_payload_runtime_records_physicalHEAD_SHA_exact':True,
 'ALL4136_old469_files_SHA_mtime_and_first_fault_inputs_immutable':True,
 'ALL521_new_command_logs_512new_streams_eight_original_golden_bytes_exact':True,
 'ALL_combined_ledger_queries_reconstructed_from_actual_trace_A16_route_scale_role_inputs':True,
 'all1536_perID4mode_tables_novelty_books_and12bank_coverage_rederived':True,
 'all768_encoder_firstdecoder_correlated_teacher_natural_prefixes_exact':True,
 '192explicit_book_roles_and_all64_original_validation_books_counts_sets_unchanged':True,
 'old258120_records_reused_byte_exact_once_no_old_native_or_main_rerun':True,
 'actual_main_native_instances_exited_creation_FILETIME_empty_fault_Windows_query':True,
 'exact_RAW_OUT_terminal_row_resource_totals_and_unrelated3_preserved':True},
 'ledger_records':records_seen,'books':192,'cases':768,'new_cases':256,'new_native_commands':521,
 'new_route_query_totals':new_totals,'fixed_bank11':{k:j['banks'][11][k] for k in ('data_ready_IDs','coverage','passed','val_natural_executed_queries','covered_val_natural_executed_queries')},
 'decision':j['decision'],'main_terminal':{'tool_session':24999,'tool_exit_code':0,'actual_instance':j['main_process_instance']},
 'windows_events':events,'windows_events_sha256':digest(EVENTS),'files':files,
 'all_new_outputs_including_raw_bytes':all_bytes,'OUT_only_bytes':OUT_bytes,
 'reused_469_input_files_count':4136,'reused_469_input_output_bytes':4566198260,
 'main_terminal_progress':terminal,'preserved_daemon_pids':daemons,'reused_PID_identity_observations':reused_pids,
 'terminal_progress_row_bytes_after_output_snapshot':terminal_row_bytes,
 'seconds_before_RET_write':time.monotonic()-start,'OS_peak_bytes_before_RET_write':peak,'file_bytes_hashed_before_RET_write':hashed,
 'scope':'Independent standardNumPy full retained-byte/query reconstruction, no scientific main/helper import, native/model/codec/tokenization/fit replay. Original and new contexts counted once; fixed validation roles/denominators and exact novelty recomputed. Data admission is not measured rank/function preservation/compressed knowledge/quality/rate/causal useful-n/mass/DRAM.'}
guard()
with RET.open('xb') as stream:stream.write((json.dumps(value,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
guard()
print(json.dumps({'sha256':digest(RET),'gates':value['gates'],'ledger_records':records_seen,'bank11':value['fixed_bank11'],
 'seconds_before_RET_write':value['seconds_before_RET_write'],'final_OS_peak_bytes':peak,'all_new_output_bytes':all_bytes}))
