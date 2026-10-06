"""One frozen complete integer backend inquiry, including failure retention."""
import argparse
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import psutil
import numpy as np
import meth486_checks as checks

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT=ROOT/'results/native_expert_scaling/meth487_shared_integer_backend';RAW=DOC/'meth487_shared_integer_backend_result.json';BIND=DOC/'meth487_binding.json'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,data):
    with path.open('xb') as f:f.write((json.dumps(data,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
def jobs():
    own={os.getpid(),*(p.pid for p in psutil.Process().parents())};kept=[]
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in own:continue
        name=(p.info['name'] or '').lower();args=p.info['cmdline'] or []
        if name=='pythonw.exe' and len(args)==2 and Path(args[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():kept.append(p.pid);continue
        assert not(name.startswith(('python','clang')) or (name.startswith('meth') and name.endswith('.exe'))),('foreign_science',p.pid,name)
    return kept
class Memory(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);arg=ap.parse_args()
    assert arg.out.resolve()==RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and not OUT.exists()
    assert os.name=='nt' and sys.flags.optimize==0 and ctypes.sizeof(Memory)==72
    began=time.monotonic();stage='bindings';peak=child_peak=hashed=0;child=None;binding=None
    result={'experiment':'METH487 same C486 fixed shared integer GPU backend','start_utc':stamp(),'process_instance':{'pid':os.getpid(),'create_time_unix':psutil.Process().create_time()},
            'commands':[],'output_inventory':[],'sources':{},'gates':{},'feasibility_gates':{},'attempts_per_namespace':1}
    api=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;api.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];api.restype=wintypes.BOOL
    def memory(p):
        v=Memory();v.cb=ctypes.sizeof(v);assert api(wintypes.HANDLE(int(p._handle)),ctypes.byref(v),v.cb)
        return {'working_set_bytes':int(v.WorkingSetSize),'peak_working_set_bytes':int(v.PeakWorkingSetSize),'peak_pagefile_bytes':int(v.PeakPagefileUsage)}
    def guard(p=None,command_start=None):
        nonlocal peak,child_peak
        m=psutil.Process().memory_info();peak=max(peak,m.rss,getattr(m,'peak_wset',0));observation=None
        if p is not None:observation=memory(p);child_peak=max(child_peak,observation['peak_working_set_bytes'])
        assert time.monotonic()-began<=2700 and peak+child_peak<=24<<30 and peak<=1<<30,'main45min_host24GiB_parent1GiB'
        if command_start is not None:assert time.monotonic()-command_start<=180,'one_command180s'
        return observation
    def digest(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while b:=f.read(4<<20):h.update(b);hashed+=len(b);guard()
        return h.hexdigest()
    def exact(item,stat=False):
        p=Path(item['path']);s=p.stat();assert s.st_size==item['bytes'] and (not stat or s.st_mtime_ns==item['mtime_ns'])
        assert digest(p)==item['sha256'],('digest',str(p))
    def inventory(path):
        p=Path(path);r={'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)};result['output_inventory'].append(r);return r['sha256']
    def quiet(label):
        waited=time.monotonic();observed=[]
        while True:
            try:
                preserved=jobs();break
            except AssertionError as error:
                observed.append({'utc':stamp(),'reason':repr(error)})
                guard();assert time.monotonic()-waited<=600,'quiet_wait600_seconds'
                print(json.dumps({'awaiting_exclusive_machine':label,'wait_seconds':time.monotonic()-waited,'reason':repr(error)}),flush=True)
                time.sleep(10)
        result.setdefault('isolation_waits',[]).append({'before':label,'seconds':time.monotonic()-waited,'observations':observed,'preserved_daemons':preserved})
        return preserved

    def run(argv,label,affinity,env):
        nonlocal child
        quiet(label);prefix=OUT/label;stdout=prefix.with_suffix('.stdout');stderr=prefix.with_suffix('.stderr')
        # label has dots: append, never replace its last component.
        stdout=Path(str(prefix)+'.stdout');stderr=Path(str(prefix)+'.stderr');started=time.monotonic()
        cmd={'label':label,'argv':list(map(str,argv)),'start_utc':stamp(),'requested_affinity':affinity,'descendant_process_peaks':[]};result['commands'].append(cmd)
        with stdout.open('xb') as so,stderr.open('xb') as se:
            child=psutil.Popen(cmd['argv'],cwd=ROOT,env=env,stdout=so,stderr=se);child.cpu_affinity(affinity)
            cmd['process_instance']={'pid':child.pid,'create_time_unix':child.create_time()};cmd['actual_affinity_after_dispatch']=child.cpu_affinity()
            observed=0;modules=set();descendants={};next_module_observation=started
            while child.poll() is None:
                m=guard(child,started);observed=max(observed,m['peak_working_set_bytes'])
                if time.monotonic()>=next_module_observation:
                    try:modules.update(v.path for v in child.memory_maps() if v.path.lower().endswith(('.dll','.exe')))
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                    next_module_observation=time.monotonic()+5
                for d in child.children(recursive=True):
                    try:
                        key=(d.pid,d.create_time());dm=d.memory_info();descendants[key]=max(descendants.get(key,0),dm.rss,getattr(dm,'peak_wset',0))
                    except psutil.NoSuchProcess:pass
                live=sum(p.stat().st_size for p in OUT.glob(label+'*') if p.is_file())
                assert live<=256<<20 and sum(x['bytes'] for x in result['output_inventory'])+live<=12<<30,'outputs12GiB_child256MiB'
                time.sleep(.1)
            cmd.update({'returncode':child.returncode,'end_utc':stamp(),'wall_seconds':time.monotonic()-started,'terminal_memory':memory(child),'sampled_peak_bytes':observed,
                        'loaded_module_paths_observed':sorted(modules),'descendant_process_peaks':[{'pid':k[0],'create_time_unix':k[1],'peak_working_set_bytes':v} for k,v in descendants.items()]})
            child=None
        cmd['stdout_sha256']=inventory(stdout);cmd['stderr_sha256']=inventory(stderr)
        cmd['raw_outputs_before_gate']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':inventory(p)} for p in sorted(OUT.glob(label+'*.bin'))]
        assert cmd['returncode']==0,('native_exit',label,cmd['returncode'])
        assert stderr.read_bytes()==b'',('stderr',label)
        rows=[json.loads(line) for line in stdout.read_text().splitlines() if line.strip()]
        print(json.dumps({'finished':label,'rows':len(rows),'wall_seconds':cmd['wall_seconds'],'peak_bytes':cmd['terminal_memory']['peak_working_set_bytes']}),flush=True)
        return rows,cmd
    try:
        binding=json.loads(BIND.read_bytes());result['binding_sha256']=digest(BIND);result['freeze_head']=binding['freeze_head']
        quiet('before_metadata_admission')
        assert sys.version==binding['runtime']['python'] and Path(sys.executable).resolve()==Path(binding['runtime']['executable']).resolve()
        assert np.__version__==binding['runtime']['versions']['numpy'] and psutil.__version__==binding['runtime']['versions']['psutil']
        for i in binding['helpers']+binding['records']+binding['toolchain']+binding['cuda']+binding['runtime']['files']:exact(i)
        for i in binding['helpers']:
            rel=Path(i['path']).relative_to(ROOT).as_posix();committed=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
            assert Path(i['path']).read_bytes().replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n'),rel
        result['preserved_daemons_before']=jobs();assert binding['complete_output_upper_bound_bytes']<=12<<30
        prior=json.loads((DOC/'meth458_switch_matched_whole_cost_result.json').read_bytes());eligibility=json.loads((DOC/'meth484_shared_integer_eligibility_result.json').read_bytes())
        assert all(prior['apparatus_gates'].values()) and all(eligibility['gates'].values())
        sources={};result['refreshed_inputs']=[]
        for source in binding['sources']:
            n=source['n'];exact(source['manifest']);exact(source['payload'],True);result['refreshed_inputs'].append(source['payload'])
            quality=json.loads(Path(source['quality_path']).read_bytes());cohort=json.loads(Path(source['cohort_path']).read_bytes())
            assert len(quality['gates'])==18 and all(quality['gates'].values()) and len(cohort['items'])==24
            for row in source['references']:
                for kind in ('teacher','natural','donor_teacher','donor_generation'):exact(row[kind]);result['refreshed_inputs'].append(row[kind])
            sources[n]=(source,quality,cohort)
        result['gates']['all_frozen_runtime_sources_payloads_and_references']=True;OUT.mkdir(parents=True)
        env=os.environ.copy();env.update(binding['runtime_environment']);env.update({'SILICON486_LIBDIR':binding['cuda_dir'],'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
        for prohibited in ('SILICON_WORKER_BINDING_FAULT','SILICON_ROUTER_FAULT'):assert prohibited not in env
        compiler=binding['toolchain'][0];binary=OUT/'meth487_shared_integer.exe';dll=OUT/'libomp.dll';shutil.copyfile(binding['toolchain'][1]['path'],dll);exact(dict(binding['toolchain'][1],path=str(dll)))
        stage='one_frozen_compile';argv=[compiler['path'],'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp','-DSILICON_SWITCH_SHARED_INTEGER_GPU_R1',str(ROOT/'benchmarks/phase60/engine.c'),'-o',str(binary)]
        compile_rows,compile_cmd=run(argv,'compile',list(range(psutil.cpu_count())),env)
        assert compile_rows==[]
        result['compile']={'argv':argv,'returncode':compile_cmd['returncode'],'process_instance':compile_cmd['process_instance'],
                           'stdout':(OUT/'compile.stdout').read_text(),'stderr':(OUT/'compile.stderr').read_text()}
        result['compile']['binary_sha256']=inventory(binary);inventory(dll)
        stage='first_CUDA_runtime_and_complete_controls';controlfile=OUT/'controls.bin'
        rows,command=run([binary,'--controls',controlfile],'controls',[0],env)
        assert len(rows)==11 and [r['control'] for r in rows]==list(range(11)) and all(r['integer_mismatches']==0 and r['output_byte_exact'] for r in rows)
        for r in rows:
            g=r['gpu'];assert g['runtime_version']==12040 and g['compute_major']==8 and g['compute_minor']==6 and g['calls']==1 and g['queries']==r['queries'] and g['device_explicit_bytes']<=2<<30
            assert [Path(p).resolve() for p in g['module_paths']]==[Path(binding['cuda_dir'])/v for v in ('cudart64_12.dll','cublasLt64_12.dll','cublas64_12.dll')]+[Path('C:/Windows/System32/nvcuda.dll')]
        result['controls']={'native_rows':rows,'independent_inquiry_checks':checks.control_check(controlfile)}
        result['gates']['complete_runtime_controls_and_integer_float_bytes']=True
        stage='whole_ownstate_teacher_and_generation'
        for n in (128,256):
            source,quality,cohort=sources[n];cases=[];result['sources'][str(n)]={'n':n,'cases':cases,'source_quality_gates_inherited':quality['gates'],'quality_scope':'Exact new ownstate outputs on the SAME 96 archived heldout cases, transitive to admitted donor comparison; no new books or donor replay.'}
            for bi,book in enumerate(cohort['items']):
                for ci,case in enumerate(book['cases']):
                    index=4*bi+ci;old=prior['sources'][str(n)]['cases'][index];qc=quality['books'][bi]['cases'][ci];refs=source['references'][index]
                    assert qc['index']==ci and refs['book']==bi and refs['case']==ci and old['source_id']==book['source_id']
                    c={'book':bi,'case':ci,'source_id':book['source_id'],'healthy_accepted':old['healthy_accepted'],'generated_tokens':old['generated_tokens'],'prose_tokens':old['prose_tokens'],
                       'backend_order':[index%2,1-index%2],'teacher':{},'generation':{'0':{},'1':{}}};cases.append(c)
                    src=','.join(map(str,case['source_ids']));dec=','.join(map(str,case['decoder_ids']));base=[binary,'--backend']
                    for backend in c['backend_order']:
                        label=f'n{n}.book{bi}.case{ci}.backend{backend}.teacher';prefix=OUT/label
                        argv=base+[str(backend),source['manifest']['path'],src,dec,prefix,str(source['threads']),'0','0','1','0']
                        tr,cmd=run(argv,label,source['affinity'],env);assert len(tr)==1 and tr[0]['repetition']==0
                        path=Path(str(prefix)+'.0.bin');assert digest(path)==refs['teacher']['sha256']
                        choices,logits=checks.wire(path,n);metrics=checks.prediction(logits,case)
                        for key in metrics:
                            if key in ('correct_span_tokens','correct_fields','greedy_teacher_forced_ids'):assert metrics[key]==qc['native'][key]
                            else:assert np.allclose(metrics[key],qc['native'][key],rtol=0,atol=1e-12),key
                        tr[0]['output_sha256']=digest(path);c['teacher'][str(backend)]={'rows':tr,'prediction':metrics,'exact_qualified_source_bytes':True}
                    for profile in (index%2,1-index%2):
                        for backend in c['backend_order']:
                            label=f'n{n}.book{bi}.case{ci}.backend{backend}.profile{profile}';prefix=OUT/label
                            argv=base+[str(backend),'--generate',source['manifest']['path'],src,prefix,str(source['threads']),'64','32095',str(profile),'1','3','0']
                            gr,cmd=run(argv,label,source['affinity'],env);c['generation'][str(backend)][str(profile)]=gr
                            assert [r['repetition'] for r in gr]==[-1,0,1,2]
                            for r in gr:
                                path=Path(str(prefix)+f".{r['repetition']}.bin");assert digest(path)==refs['natural']['sha256'];r['output_sha256']=digest(path)
                                ids,logits=checks.wire(path,n);assert ids==r['generated_ids']==qc['generation']['native']['generated_ids'];assert checks.accepted(ids)==(c['healthy_accepted'],c['generated_tokens'],c['prose_tokens'])
                                for key in ('source_tokens','actual_generated_tokens','closing_id','stopped_on_EOS','stopped_on_closing','stopped_at_cap'):assert r[key]==old['modes'][str(profile)][r['repetition']+1][key]
                                assert r['threads']==source['threads'] and r['profile']==profile and r['worker_physical_cores']==6
                                assert [a['actual_mask'] for a in r['worker_affinity']]==[1<<a for a in source['affinity']]
                                assert all(a['actual_mask']==1<<source['affinity'][a['slot']] and a['group']==0 for a in r['worker_binding_events'])
                                for phase in (0,1):
                                    for kind in range(4):
                                        for key in ('calls','code_bytes','f32_bytes','scale_bytes'):assert r['counters'][phase][kind][key]==old['modes'][str(profile)][r['repetition']+1]['counters'][phase][kind][key]
                                g=r['gpu'];assert g['enabled']==bool(backend)
                                if backend:
                                    for key,v in checks.expected_gpu(r['source_tokens'],r['actual_generated_tokens']).items():assert g[key]==v,(label,key)
                                    assert g['matrix_count']==169 and g['weight_bytes']==166232064 and g['device_explicit_bytes']==178552832 and g['cpu_query_bytes']==394240
                                else:assert all(g[key]==0 for key in ('calls','queries','h2d_bytes','d2h_bytes'))
                                assert r['full_generation_seconds']>0 and abs(r['full_generation_seconds']-sum(r[k] for k in ('encoder_seconds','cross_kv_seconds','decode_greedy_seconds')))<=3e-8
            assert [sum(c['healthy_accepted'] for c in cases),sum(c['generated_tokens'] for c in cases if c['healthy_accepted']),sum(c['prose_tokens'] for c in cases if c['healthy_accepted'])]==source['totals']
            result['sources'][str(n)]['summaries']={str(b):{str(p):checks.rate_summary(cases,b,p) for p in (0,1)} for b in (0,1)}
            admission={}
            for backend in (0,1):
                s0=result['sources'][str(n)]['summaries'][str(backend)]['0'];s1=result['sources'][str(n)]['summaries'][str(backend)]['1']
                ratio=s1['sum_case_mean_seconds']/s0['sum_case_mean_seconds'];bookratios=[]
                for bi in range(24):
                    sums=[sum(statistics.mean(r['full_generation_seconds'] for r in c['generation'][str(backend)][str(p)] if r['repetition']>=0) for c in cases[bi*4:bi*4+4]) for p in (0,1)];bookratios.append(sums[1]/sums[0])
                admission[str(backend)]={'aggregate_ratio':ratio,'book_ratios':bookratios,'pass':.9<=ratio<=1.1 and all(.85<=v<=1.15 for v in bookratios)}
            result['sources'][str(n)]['profile_admission']=admission
        result['gates']['ALL192_teacher_ownstate_and_natural_bytes_quality_counters']=True
        result['gates']['profile_cost_admission']=all(a['pass'] for s in result['sources'].values() for a in s['profile_admission'].values())
        for n,s in result['sources'].items():
            gpu=s['summaries']['1']['0'];cpu=s['summaries']['0']['0'];result['feasibility_gates'][n]={'warm_ordinary_lower95_ge50':gpu['ordinary']['warm_lower95']>=50,'warm_prose_lower95_ge50':gpu['prose']['warm_lower95']>=50,
                'fresh_matched_GPU_faster':gpu['sum_case_mean_seconds']<cpu['sum_case_mean_seconds'],'stretch_prose_lower95_ge100':gpu['prose']['warm_lower95']>=100}
        result['decision']='Admit exact arithmetic backend only if all apparatus gates pass. Fixed layout speed qualifies only with BOTH source warm ordinary/prose lower95 >=50 and matched speed benefit; full goal remains incomplete.'
        result['preserved_daemons_after']=jobs();guard();result['end_utc']=stamp();result['resource']={'wall_seconds':time.monotonic()-began,'parent_peak_bytes':peak,'native_peak_bytes':child_peak,'bytes_hashed':hashed,'output_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),'device_limit_scope':'Explicit operator buffers + weights + workspace <=2GiB. CUDA context/library allocation is not a measured WDDM process peak; global MemGetInfo snapshots retained.'}
        assert result['resource']['output_bytes']<=12<<30;write(RAW,result)
        print(json.dumps({'result':str(RAW),'gates':result['gates'],'feasibility':result['feasibility_gates'],'resource':result['resource']}),flush=True)
    except BaseException as exc:
        if child is not None:
            try:child.terminate();child.wait(timeout=10)
            except (psutil.NoSuchProcess,psutil.TimeoutExpired):pass
            command=result['commands'][-1];command['end_utc']=stamp();command['returncode']=child.poll();command['guard_termination']=True
        result['failure']={'stage':stage,'exception':repr(exc),'traceback':traceback.format_exc()};result['end_utc']=stamp();result['resource']={'wall_seconds':time.monotonic()-began,'parent_peak_bytes':peak,'native_peak_bytes':child_peak,'bytes_hashed':hashed}
        if OUT.exists():result['all_partial_files']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.iterdir()) if p.is_file()]
        write(RAW.with_suffix('.failure.json'),result);print(json.dumps({'failure':result['failure'],'resource':result['resource']}),flush=True);raise

if __name__=='__main__':main()
