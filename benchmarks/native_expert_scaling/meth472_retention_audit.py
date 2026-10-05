"""Independent paired-vector/bank/query/spectral witness audit, no SVD or forward."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import time
START=time.monotonic()
import argparse
from datetime import datetime
import hashlib,json,math
from pathlib import Path
import struct,subprocess
import numpy as np
import psutil
import threadpoolctl

ROOT=Path.cwd();DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
RAW=DOC/'meth472_switch_private_input_probe_result.json';RET=DOC/'RETENTION_472_20261005.json'
OUT=ROOT/'results/native_expert_scaling/meth472_switch_private_input_probe'
EVENTS=ROOT/'results/native_expert_scaling/meth472_windows_terminal.json'
ap=argparse.ArgumentParser();ap.add_argument('--expected-raw-sha',required=True);args=ap.parse_args()
assert not RET.exists();peak=hashed=0
psutil.Process().cpu_affinity([0]);limits=threadpoolctl.threadpool_limits(limits=1)
def guard():
 global peak
 info=psutil.Process().memory_info();peak=max(peak,info.rss,getattr(info,'peak_wset',0))
 assert peak<=2<<30 and time.monotonic()-START<=300,'audit300s2GiB'
def digest(p):
 global hashed
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  while d:=s.read(4<<20):h.update(d);hashed+=len(d);guard()
 return h.hexdigest()
def head(p):
 rel=Path(p).resolve().relative_to(ROOT).as_posix()
 assert Path(p).read_bytes()==subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters','--path='+rel,'HEAD:'+rel])
def same(a,b):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def quant(x):
 maximum=np.max(np.abs(x),axis=1)
 with np.errstate(under='ignore'):alpha=(maximum/np.float32(32767)).astype('<f4')
 alpha[maximum==0]=1.;assert np.all(alpha>0)
 return np.clip(np.rint(np.divide(x,alpha[:,None],dtype=np.float32)),-32767,32767).astype('<i2'),alpha
assert digest(RAW)==args.expected_raw_sha
j=json.loads(RAW.read_text(encoding='utf-8'));assert len(j['apparatus_gates'])==8 and all(j['apparatus_gates'].values())
assert j['native_or_model_commands']==j['updates']==0
bpath=DOC/'meth472_prospective_bindings.json';head(bpath);assert digest(bpath)==j['source_binding_sha256']
b=json.loads(bpath.read_text(encoding='utf-8'))
for path,h in j['scientific_sources'].items():head(path);assert digest(path)==h
assert (ROOT/'.gitattributes').read_bytes()==subprocess.check_output(['git','show','HEAD:.gitattributes'])
for rel,v in b['helpers'].items():head(ROOT/rel);assert digest(ROOT/rel)==v['sha256']
for name,v in b['records'].items():head(DOC/name);assert (DOC/name).stat().st_size==v['bytes'] and digest(DOC/name)==v['sha256']
assert digest(b['preparation_helper']['path'])==b['preparation_helper']['sha256']
for path,v in b['runtime']['files'].items():assert Path(path).stat().st_size==v['bytes'] and digest(path)==v['sha256']
for pkg,v in b['runtime']['packages'].items():
 for path,r in v['files'].items():assert Path(path).stat().st_size==r['bytes'] and digest(path)==r['sha256']
for module in (np,psutil,threadpoolctl):assert module.__version__==b['runtime']['packages'][module.__name__]['version'] and str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():assert pool['num_threads']==1 and str(Path(pool['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
for path,v in b['native_files'].items():assert Path(path).stat().st_size==v['bytes'] and digest(path)==v['sha256']
assert j['artifact']==b['original_artifact'] and digest(j['artifact']['payload'])==j['artifact']['sha256']
p=Path(j['artifact']['payload']);assert [p.stat().st_size,p.stat().st_mtime_ns]==j['payload_stat_before']==[b['payload_stat_before']['bytes'],b['payload_stat_before']['mtime_ns']]
for v in b['retained_inventory']:
 p=Path(v['path']);assert p.stat().st_size==v['bytes'] and p.stat().st_mtime_ns==v['mtime_ns'] and digest(p)==v['sha256']
assert len(b['retained_inventory'])==6224
for origin in ('469','471'):
 rows=[v for v in b['retained_inventory'] if v['origin']==origin]
 assert {str(p.resolve()) for p in Path(rows[0]['path']).parent.iterdir() if p.is_file()}=={str(Path(v['path']).resolve()) for v in rows}
for v in b['native_FFN_controls']:assert Path(v['path']).stat().st_size==v['bytes'] and digest(v['path'])==v['sha256']
for path,v in b['integer_fixtures'].items():assert digest(path)==v['sha256']
files={Path(v['path']).resolve():v for v in j['output_inventory']}
assert len(files)==113 and set(files)=={p.resolve() for p in OUT.iterdir() if p.is_file() and p.name!='progress.jsonl'}
for path,v in files.items():assert path.stat().st_size==v['bytes'] and digest(path)==v['sha256']
assert (OUT/'fatal_native.log').stat().st_size==0 and not RAW.with_suffix('.failure.json').exists()

q=np.load(OUT/'query_inputs.npy',mmap_mode='r',allow_pickle=False)
f0=np.load(OUT/'reference_functions.npy',mmap_mode='r',allow_pickle=False)
fc=np.load(OUT/'candidate_functions.npy',mmap_mode='r',allow_pickle=False)
metrics=np.load(OUT/'per_query_metrics.npy',mmap_mode='r',allow_pickle=False)
assert q.shape==(19962,) and q.dtype.itemsize==4732 and json.loads(json.dumps(q.dtype.descr))==j['domain']['query_dtype_descr']
assert f0.shape==fc.shape==(19962,768) and f0.dtype==fc.dtype==np.dtype('<f4') and metrics.shape==(19962,20) and metrics.dtype==np.dtype('<f8')
assert np.isfinite(f0).all() and np.isfinite(fc).all() and np.isfinite(metrics).all()
assert np.sum(q['role']==0)==13313 and np.sum((q['role']==1)&(q['mode']==1))==3065
capture=json.loads((DOC/'meth471_switch_development_capture_result.json').read_text(encoding='utf-8'))
previous=json.loads((DOC/'meth469_switch_native_domain_capture_result.json').read_text(encoding='utf-8'))
roles={v['book']:v['split'] for v in capture['immutable_book_roles']}
wire=struct.Struct('<H6BHIff32s32s32s');trdtype=np.dtype([('index','<u4'),('phase','<u4'),('input','<f4',(768,)),('scores','<f4',(128,))])
rdtype=np.dtype([('expert','<i4'),('accepted','<i4'),('p','<f4')])
cursor=seen=0
with Path(capture['ledger']['path']).open('rb') as ledger:
 assert ledger.read(20)==struct.pack('<8sIQ',b'MQ471L01',118,387036)
 for case_index in range(768):
  bi,ci=divmod(case_index,4);source=previous if bi<128 else capture;local=case_index if bi<128 else case_index-512
  for mi,mode in enumerate(('teacher','natural')):
   c=source['commands'][source['cases'][local]['modes'][mode]];t=14 if mi==0 else c['native_row']['actual_generated_tokens'];nr=6*(29+t)
   assert c['label']==f'book{bi}.case{ci}.{mode}' and c['returncode']==0
   d=Path(c['trace_path']).read_bytes();assert d[:16]==struct.pack('<8sII',b'SWRTA001',128,768) and len(d)==16+nr*trdtype.itemsize
   tr=np.frombuffer(d,trdtype,offset=16)
   wd=Path(c['whole_output_path']).read_bytes();routeoff=36+4*(14*29*768+14*t*768+t*32128)
   assert wd[:36]==struct.pack('<8s7I',b'SWR32O01',29,t,768,12,12,32128,nr) and len(wd)==routeoff+nr*12
   rt=np.frombuffer(wd,rdtype,offset=routeoff);ix=179+6*np.arange(t);v=q[cursor:cursor+t]
   same(v['input'],tr['input'][ix]);codes,alpha=quant(tr['input'][ix]);same(v['codes'],codes);same(v['alpha'],alpha);same(v['probability'],rt['p'][ix])
   assert np.array_equal(v['expert'],np.argmax(tr['scores'][ix],axis=1)) and np.all(v['accepted']==1)
   assert np.array_equal(v['index'],ix) and np.array_equal(v['ledger_record'],seen+ix)
   assert np.all(v['book']==bi) and np.all(v['case']==ci) and np.all(v['mode']==mi) and np.all(v['role']==roles[bi])
   selected=np.argmax(tr['scores'][ix],axis=1);diff=tr['scores'][ix]-tr['scores'][ix,selected,None]
   prob=(1/np.exp(diff.astype(np.float64)).astype(np.float32).astype(np.float64).sum(axis=1)).astype('<f4')
   assert np.max(np.abs(prob.astype(np.float64)/v['probability']-1))<=1e-6
   ld=ledger.read(nr*118);assert len(ld)==nr*118
   for k,index in enumerate(ix):
    ni,lm,lb,lc,bank,accepted,role,e,li,al,pr,hi,hc,hp=wire.unpack_from(ld,int(index)*118)
    assert (ni,lm,lb,lc,bank,accepted,role,e,li)==(128,mi,bi,ci,11,1,roles[bi],int(v['expert'][k]),int(index))
    sb=v['alpha'][k].tobytes();qb=v['codes'][k].tobytes()
    assert struct.pack('<f',al)==sb and struct.pack('<f',pr)==v['probability'][k].tobytes()
    assert hi==hashlib.sha256(v['input'][k].tobytes()).digest()==v['input_sha'][k:k+1].tobytes()
    assert hc==hashlib.sha256(qb).digest()==v['code_sha'][k:k+1].tobytes() and hp==hashlib.sha256(qb+sb).digest()==v['pair_sha'][k:k+1].tobytes()
   cursor+=t;seen+=nr;guard()
 assert cursor==19962 and seen==387036 and not ledger.read(1)

def code(i):return q['code_sha'][i:i+1].tobytes()
def reps(indices):
 first={}
 for i in indices:first.setdefault((int(q['book'][i]),q['codes'][i].tobytes(),q['alpha'][i].tobytes()),int(i))
 ids=np.asarray(sorted(first.values()),dtype='<u4');books=q['book'][ids];bs,counts=np.unique(books,return_counts=True)
 lookup=dict(zip(map(int,bs),map(int,counts)));w=np.asarray([1/(len(bs)*lookup[int(b)]) for b in books],dtype='<f8')
 assert abs(float(w.sum())-1)<=1e-12
 for b in bs:assert abs(float(w[books==b].sum())-1/len(bs))<=1e-12
 return ids,w
def rms(err,ref,w=None):
 if w is None:n=float(np.sum(err));d=float(np.sum(ref))
 else:n=float(np.sum(w*err));d=float(np.sum(w*ref))
 ratio=math.sqrt(n/d) if d else (0. if n==0 else None)
 return {'RMS_relative':ratio,'weighted_error_energy':n,'weighted_reference_energy':d,'zero_reference_nonzero_error':d==0 and n>0,'count':len(err)}
def within(v,t):return v['RMS_relative'] is not None and v['RMS_relative']<=t
ready=set(b['fixed_ready_IDs']);novel={}
for e,row in enumerate(capture['banks'][11]['experts']):
 for split,role in enumerate(('dev','val')):
  for mode,name in enumerate(('teacher','natural')):
   ids=np.flatnonzero((q['expert']==e)&(q['role']==split)&(q['mode']==mode));v=q[ids];s=row['by_split_mode'][role+'_'+name]
   assert len(ids)==s['executed']==s['selected'] and s['rejected']==0
   assert len({code(i) for i in ids})==s['unique_code_SHA'] and sorted(set(map(int,v['book'])))==s['books']
   assert len({(int(q['book'][i]),int(q['case'][i])) for i in ids})==s['cases']
 di=np.flatnonzero((q['expert']==e)&(q['role']==0));vi=np.flatnonzero((q['expert']==e)&(q['role']==1))
 dc={code(i) for i in di};vc={code(i) for i in vi};novel[e]=vc-dc;nb=sorted({int(q['book'][i]) for i in vi if code(i) in novel[e]})
 assert len(dc)==row['dev_union']['unique_code_SHA'] and len(vc)==row['val_union']['unique_code_SHA'] and len(novel[e])==row['validation_novel_codes'] and nb==row['validation_novel_code_books']
 actual=all((len(dc)>=32,len(set(map(int,q['book'][di])))>=4,len(novel[e])>=16,len(nb)>=4))
 assert actual==row['data_ready']==(e in ready)

bm=np.memmap(OUT/'private_input_bank.bin',dtype='u1',mode='r');original=np.memmap(j['artifact']['payload'],dtype='u1',mode='r')
assert bm[:64].tobytes()==struct.pack('<8s14I',b'MQ472B01',1,768,3072,128,11,32,128,64,16448,107,21,406093824,0,0) and len(bm)==406110272
entry=struct.Struct('<4I14Q');export=json.loads((DOC/'meth380_switch_base128_export_result.json').read_text(encoding='utf-8'))
offset=16448;failed=[];witnesses=0
for e in range(128):
 desc=list(entry.unpack_from(bm,64+e*128));assert desc==j['bank_descriptors'][e] and desc[:4]==[e,int(e in ready),32 if e in ready else 0,0] and desc[-2:]==[0,0]
 sections=dict(zip(('P','A','WI','WI_scale','WO','WO_scale'),[desc[i:i+2] for i in range(4,16,2)]))
 lengths=[98304,393216,0,12288,2359296,3072] if e in ready else [0,0,2359296,12288,2359296,3072]
 for name,length in zip(sections,lengths):
  start,count=sections[name]
  if length:assert (start,count)==(offset,length) and start%64==0;offset+=length
  else:assert [start,count]==[0,0]
 def arr(name,dtype,shape):
  start,count=sections[name];a=np.frombuffer(bm,dtype,count//np.dtype(dtype).itemsize,start).reshape(shape);assert a.nbytes==count;return a
 parts=[]
 for kind,shape in [('wi',(3072,768)),('wo',(768,3072))]:
  t=export['tensors'][f'decoder.block.11.layer.2.mlp.experts.expert_{e}.{kind}.weight']
  w=np.frombuffer(original,'<i1',t['elements'],t['offset']).reshape(shape);s=np.frombuffer(original,'<f4',shape[0],t['scale_offset']);parts.extend((w,s))
 wi,si,wo,so=parts;same(arr('WI_scale','<f4',(3072,)),si);same(arr('WO','<i1',(768,3072)),wo);same(arr('WO_scale','<f4',(768,)),so)
 ids=np.flatnonzero(q['expert']==e);stats=j['experts'][e];assert stats['expert']==e and stats['all_query_count']==len(ids)
 if e in ready:
  assert stats['policy']=='factor32'
  with np.load(OUT/f'svd_witness_e{e:03d}.npz',allow_pickle=False) as w:
   s=w['singular_values'];vt=w['Vt'];di,dw=reps(np.flatnonzero((q['expert']==e)&(q['role']==0)))
   vi,vw=reps([i for i in ids if q['role'][i]==1 and code(i) in novel[e]])
   same(di,w['development_representatives']);same(dw,w['development_weights']);same(vi,w['validation_novel_representatives']);same(vw,w['validation_novel_weights'])
   assert len(s)>=32 and vt.shape==(len(s),768) and len(s)==min(len(di),768) and np.isfinite(vt).all() and np.all(s>=0) and np.all(s[:-1]>=s[1:])
   assert np.array_equal(s,np.asarray(stats['input_spectrum']['singular_values']))
   for r in vt:assert r[int(np.argmax(np.abs(r)))]>=0
   x=(q['codes'][di].astype(np.float64)*q['alpha'][di].astype(np.float64)[:,None])*np.sqrt(dw)[:,None];energy=float(np.sum(x*x))
   assert energy==stats['input_spectrum']['energy'] and np.linalg.norm(vt @ vt.T-np.eye(len(s)))<=1e-10*math.sqrt(len(s))
   assert np.linalg.norm(x.T @ (x @ vt.T)-vt.T*(s*s))/energy<=1e-10
   assert np.linalg.norm(x-(x @ vt.T) @ vt)/math.sqrt(energy)<=1e-10
   assert abs(float(np.sum(s*s))-energy)/energy<=1e-10
   tol=float(s[0]*max(x.shape)*np.finfo(np.float64).eps)
   assert tol==stats['input_spectrum']['numerical_rank_tolerance'] and int(np.sum(s>tol))==stats['input_spectrum']['numerical_rank']
   p32=vt[:32].T.astype('<f4');same(arr('P','<f4',(768,32)),p32)
   a32=(wi.astype(np.float64) @ p32.astype(np.float64)).astype('<f4');same(arr('A','<f4',(3072,32)),a32)
   assert np.linalg.norm(p32.astype(np.float64).T @ p32.astype(np.float64)-np.eye(32))<=2e-6*math.sqrt(32)
   xres=x-(x @ vt[:32].T) @ vt[:32];retained=1-float(np.sum(xres*xres))/energy
   assert abs(retained-stats['input_spectrum']['ideal_rank32_development_energy_retained'])<=1e-12
   actual=rms(metrics[vi,2],metrics[vi,0],vw);assert actual==stats['novel_validation_FFN'] and within(actual,.05)==stats['per_expert_gate_novel_FFN_RMS_le0p05']
   inputstats=rms(metrics[vi,6],metrics[vi,5],vw);assert inputstats==stats['novel_validation_input']
   if not within(actual,.05):failed.append(e)
   for begin in range(0,len(ids),256):
    ii=ids[begin:begin+256];xx=q['codes'][ii].astype(np.float64)*q['alpha'][ii].astype(np.float64)[:,None];res=xx-(xx @ vt[:32].T) @ vt[:32]
    assert np.allclose(np.sum(res*res,axis=1),metrics[ii,6],rtol=1e-12,atol=1e-20)
   witnesses+=1
 else:
  assert stats['policy']=='original_I8_fallback';same(arr('WI','<i1',(3072,768)),wi)
  assert fc[ids].tobytes()==f0[ids].tobytes() and np.all(metrics[ids,2]==0)
 guard()
assert offset==len(bm) and witnesses==107

for begin in range(0,len(q),256):
 ii=slice(begin,begin+256);ref=f0[ii].astype(np.float64);cand=fc[ii].astype(np.float64);delta=cand-ref
 expected=[np.sum(ref*ref,axis=1),np.sum(cand*cand,axis=1),np.sum(delta*delta,axis=1)]
 for column,v in enumerate(expected):assert np.array_equal(v,metrics[ii,column])
 xin=q['codes'][ii].astype(np.float64)*q['alpha'][ii].astype(np.float64)[:,None]
 assert np.array_equal(np.sum(xin*xin,axis=1),metrics[ii,5])
 assert np.array_equal(expected[2]*q['probability'][ii].astype(np.float64)**2,metrics[ii,18])
 assert np.array_equal(expected[0]*q['probability'][ii].astype(np.float64)**2,metrics[ii,19])
 parts=metrics[ii,12:18];scale=np.maximum(np.sum(np.abs(parts),axis=1),np.maximum(metrics[ii,2],1e-30))
 assert np.all(np.abs(np.sum(parts,axis=1)-metrics[ii,2])<=2e-12*scale);guard()
vn=np.flatnonzero((q['role']==1)&(q['mode']==1));valteacher=np.flatnonzero((q['role']==1)&(q['mode']==0))
natural=rms(metrics[vn,2],metrics[vn,0]);assert natural==j['full_natural_validation_FFN']
assert rms(metrics[vn,18],metrics[vn,19])==j['full_natural_validation_routed_FFN']
assert rms(metrics[valteacher,2],metrics[valteacher,0])==j['full_teacher_validation_FFN']
books=[]
for bi in range(64,128):
 ii=vn[q['book'][vn]==bi];books.append({'book':bi,**rms(metrics[ii,2],metrics[ii,0])})
assert books==j['natural_validation_books']
gates={'actual_complete_bank_bytes_le0p70_original':len(bm)<=.7*605945856,
 'ALL107_novel_validation_equal_book_FFN_RMS_le0p05':len(failed)==0,
 'ALL3065_natural_validation_FFN_RMS_le0p05':within(natural,.05),
 'ALL64_natural_validation_book_FFN_RMS_le0p10':all(within(v,.10) for v in books)}
assert gates==j['recipe_gates']
assert j['storage']['actual_bank_bytes']==len(bm) and j['storage']['stored_ratio']==len(bm)/605945856
expected='rank32_private_INPUT_recipe_admitted_for_separately_frozen_native_operator_cost_inquiry' if all(gates.values()) else 'fixed_rank32_private_INPUT_PCA_function_recipe_FAIL_close_this_representation_no_rank_grid_or_validation_selected_fallback'
assert j['decision']==expected

progress_bytes=(OUT/'progress.jsonl').read_bytes();progress=[json.loads(v) for v in progress_bytes.splitlines()];terminal=progress[-1]
assert terminal['raw_sha256']==args.expected_raw_sha and terminal['raw_bytes']==RAW.stat().st_size and terminal['admitted']==all(gates.values())
assert terminal['final_file_bytes_hashed']==j['resource']['file_bytes_hashed_before_raw']+RAW.stat().st_size
rowbytes=len(progress_bytes.splitlines(keepends=True)[-1]);outbytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
assert outbytes==terminal['OUT_bytes_before_terminal_progress_row']+rowbytes
assert outbytes+RAW.stat().st_size<=j['prospective_byte_bounds']['new_output_upper_bytes']<768<<20
assert j['resource']['seconds_before_raw_write']<=terminal['seconds']<=900 and terminal['OS_peak_bytes_after_raw']<=8<<30
events=json.loads(EVENTS.read_text(encoding='utf-8'))
assert events['query_available'] and events['query_error'] is None and events['matching_main_events']==[]
assert events['main_pid']==j['main_process_instance']['pid']==progress[0]['pid']
assert datetime.fromisoformat(events['query_start_utc']).timestamp()<=datetime.fromisoformat(j['start_utc']).timestamp()-1
assert datetime.fromisoformat(events['query_end_utc']).timestamp()>=datetime.fromisoformat(terminal['utc']).timestamp()
instance=j['main_process_instance'];reuse=[]
try:
 p=psutil.Process(instance['pid']);assert not(abs(p.create_time()-instance['create_time_unix'])<.01 and p.name().lower()==instance['name'].lower() and Path(p.exe()).resolve()==Path(instance['executable']).resolve())
 reuse.append({'pid':p.pid,'current_name':p.name(),'current_creation_unix':p.create_time(),'current_executable':p.exe()})
except psutil.NoSuchProcess:pass
own={os.getpid(),*(p.pid for p in psutil.Process().parents())};daemons=[]
for p in psutil.process_iter(['name','cmdline']):
 if p.pid in own:continue
 try:
  name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
  if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():daemons.append(p.pid);continue
  assert not(name.startswith('python') or name=='clang.exe' or(name.startswith('meth') and name.endswith('.exe'))),(p.pid,name)
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
for rel,v in b['preserved_unrelated_files'].items():assert digest(ROOT/rel)==v['sha256']
assert digest(ROOT/'benchmarks/phase60/engine.c')==j['preserved_engine_sha256']
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).splitlines()==b['tracked_status']
retfiles=[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
byid=[]
totalenergy=natural['weighted_reference_energy']
for e in range(128):
 ids=vn[q['expert'][vn]==e]
 if not len(ids):continue
 refenergy=float(np.sum(metrics[ids,0]));errenergy=float(np.sum(metrics[ids,2]))
 byid.append({'expert':e,'policy':'factor32' if e in ready else 'original_I8_fallback','queries':len(ids),
  'reference_energy':refenergy,'reference_energy_fraction':refenergy/totalenergy,'error_energy':errenergy})
value={'experiment':'METH472 independent retained query/bank/paired vector/spectral witness audit',
 'raw_sha256':args.expected_raw_sha,'helper_sha256':digest(__file__),
 'gates':{'frozen_actual_runtime_source_records_payload_ALL6224_input_SHA_immutable':True,
  'ALL19962_query_input_q_scale_native_route_mass_role_actual_ledger_and128_readiness_exact':True,
  'ALL128_actual_bank_header_policy_sections_source_parts_P_A_bytes_no_supported_original_WI':True,
  'ALL107_development_dedup_weights_SVD_witness_axes_energy_eigen_and_factors_no_refit':True,
  'ALL19962_paired_F32_function_vectors_primary_and_routed_error_energies_rederived':True,
  'saved_metric_fields_finite_input_projection_and_signed_energy_accounting_verified':True,
  'ALL107_novel_validation_ALL3065natural_ALL64book_storage_gates_rederived_no_policy_selection':True,
  'actual_main_instance_exited_empty_fault_literalUTC_Windows_query_and_exact_output_resources':True,
  'original_engine_unrelated3_status_SHA_and_complete_input_mtimes_preserved':True},
 'recipe_gates':gates,'decision':expected,'failed_factor_IDs':failed,'natural_validation_FFN':natural,
 'natural_validation_energy_by_ID_descriptive_from_saved_vectors':byid,
 'main_terminal':{'tool_session':89758,'tool_exit_code':0,'actual_instance':instance,'PID_reuse_observations':reuse},
 'windows_events':events,'windows_events_sha256':digest(EVENTS),'files':retfiles,
 'OUT_bytes':outbytes,'raw_bytes':RAW.stat().st_size,'all_main_output_bytes':outbytes+RAW.stat().st_size,
 'terminal_progress_row_bytes':rowbytes,'terminal_progress':terminal,'preserved_daemons':daemons,
 'audit_seconds_before_RET_write':time.monotonic()-START,'audit_OS_peak_bytes_before_RET_write':peak,'audit_file_bytes_hashed_before_RET_write':hashed,
 'scope':'Independent standardNumPy witness/paired-vector metadata algebra, no SVD/refit/scientific main/native/model/codec/tokenizer replay. Source controls remain fresh byte-bound. Intermediate WI/ReLU/hidden-quantizer direction fields are source-recorded; audit checks their finite signed-energy identity, not new intermediate forward re-evaluation. All primary frozen function-error gates rederived from full saved vectors and independently reconstructed role/novelty/weights.'}
guard()
with RET.open('xb') as s:s.write((json.dumps(value,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
guard();print(json.dumps({'ret_sha256':digest(RET),'gates':value['gates'],'recipe_gates':gates,'failed_factors':len(failed),'audit_seconds_before_RET_write':value['audit_seconds_before_RET_write'],'audit_final_OS_peak_bytes':peak,'all_main_outputs':value['all_main_output_bytes']}))
limits.restore_original_limits()
