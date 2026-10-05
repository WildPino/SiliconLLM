"""One prospective stage-cost diagnosis, not a kernel promotion or quality run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT=ROOT/'results/native_expert_scaling/meth455_switch_component_cost'
CPU=Path(__file__).with_name('meth455_switch_component_cpu.c')
PROTOCOL=DOC/'METH_455_SWITCH_COMPONENT_COST_PROTOCOL_20261005.md'
PARENT_SHA='efc087e32c2f88631654efaa4d0f30bf9b23288f66243deb51ed5f07a947d277'
RETENTION_SHA='4747e27123ea23c2aaf94b415e57b58683f9d35696c7f079fb15684f9e6233a2'
COMPONENTS=('signed_basis','WI_A16','LUT_build','WI_dot_decode_scale','ReLU','WO_A16','WO_active_scan','WO_column_accum','WO_output_scale','original_WO_dot_scale')
ARMS=('original_I8','direct_I4_sparse_I8','pair_LUT_I4_sparse_I8')
MODES=('unchanged454','split_without_internal_timers','split_with_internal_timers')
WIRE=np.dtype([('query','<u4'),('arm','<u4'),('wrong','<u4'),('id','<u4'),('nonzero','<u4'),('wi_scale','<f4'),('wo_scale','<f4'),
    ('basis','<f4',(768,)),('raw','<f4',(3072,)),('down','<f4',(768,)),('wi_codes','<i2',(768,)),('wo_codes','<i2',(3072,))])
WIRE455=np.dtype([('mode','<u4'),('record',WIRE)])
assert WIRE.itemsize==26140 and WIRE455.itemsize==26144

def committed(path):
    p=Path(path);rel=p.resolve().relative_to(ROOT).as_posix()
    expected=subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters',f'--path={rel}',f'HEAD:{rel}'],cwd=ROOT)
    assert p.read_bytes()==expected,('physical_HEAD',rel)

def write(path,value):
    Path(path).write_bytes((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').replace('\n','\r\n').encode('utf-8'))

def jobs(result):
    own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in own:continue
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            result.setdefault('preserved_daemons',[]).append(p.pid);continue
        assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',p.pid,name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();numeric=None;peak=hashed=0;stage='fresh_bindings'
    result={'experiment':'METH455-equivalent-stage-profile-and-perturbation-controls','native_commands':{},'helper_sha256':{}}
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info();peak=max(peak,mi.rss,getattr(mi,'peak_wset',0));elapsed=time.monotonic()-start
        assert peak<=512<<20 and elapsed<=900 and ((numeric is None and elapsed<=300) or (numeric is not None and time.monotonic()-numeric<=600)),'parent512MiB_admission300_numeric600_total900'
        assert not OUT.exists() or sum(p.stat().st_size for p in OUT.glob('*') if p.is_file())<=128<<20,'output128MiB'
    def sha(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while b:=f.read(4<<20):h.update(b);hashed+=len(b);guard()
        return h.hexdigest()
    try:
        result['git_head_before_execution']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert (ROOT/'.gitattributes').read_bytes()==subprocess.check_output(['git','show','HEAD:.gitattributes'],cwd=ROOT)
        for p in (Path(__file__),CPU,PROTOCOL):committed(p);result['helper_sha256'][str(p)]=sha(p)
        p=DOC/'meth454_switch_sparse_wo_cost_result.json';committed(p);assert sha(p)==PARENT_SHA;parent=json.loads(p.read_bytes())
        assert all(parent['apparatus_gates'].values()) and not any(parent['candidate_PASS'].values())
        p=DOC/'RETENTION_454_20261005.json';committed(p);assert sha(p)==RETENTION_SHA
        for path,expected in parent['helper_sha256'].items():committed(path);assert sha(path)==expected;result['helper_sha256'][path]=expected
        for name,expected in parent['retained_record_sha256'].items():committed(DOC/name);assert sha(DOC/name)==expected
        outputs={}
        for item in parent['output_inventory']:
            p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];outputs[p.name]=p
        a=parent['artifacts']['128'];assert Path(a['payload']).stat().st_size==a['bytes'] and sha(a['payload'])==a['sha256'] and sha(a['manifest'])==a['manifest_sha256']
        engine=ROOT/'benchmarks/phase60/engine.c';committed(engine);assert sha(engine)==parent['preserved_engine_sha256']
        for path,expected in parent['preserved_original_binary_sha256'].items():assert sha(path)==expected
        for path,expected in parent['runtime']['binary_sha256'].items():assert sha(path)==expected
        compiler=Path(parent['compile']['argv'][0]);assert sha(compiler)==parent['compile']['compiler_sha256'];dll=outputs['libomp.dll'];assert sha(dll)==parent['compile']['runtime_sha256']
        result.update(retained_record_sha256=parent['retained_record_sha256']|{'meth454_switch_sparse_wo_cost_result.json':PARENT_SHA,'RETENTION_454_20261005.json':RETENTION_SHA},
            artifacts=parent['artifacts'],preserved_engine_sha256=parent['preserved_engine_sha256'],preserved_original_binary_sha256=parent['preserved_original_binary_sha256'],
            preserved_numerical_runtime_sha256=parent['runtime']['binary_sha256'],reused_output_inventory=parent['output_inventory'])
        jobs(result);psutil.Process().cpu_affinity([0]);assert np.__version__=='2.4.6' and psutil.disk_usage(str(ROOT)).free>=1<<30
        initial={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in (Path(a['payload']),Path(a['manifest']),outputs['bank.bin'],outputs['trace.bin'],outputs['primal.bin'])}
        numeric=time.monotonic();OUT.mkdir(parents=True);result['admission']={'seconds':numeric-start,'bytes_hashed':hashed};print(json.dumps({'admission':result['admission']}),flush=True)
        env=os.environ.copy();env.update(parent['runtime']['native_environment']);env['PATH']=str(compiler.parent)+os.pathsep+env.get('PATH','')
        result['runtime']={'numpy':np.__version__,'controller_affinity':[0],'native_environment':parent['runtime']['native_environment'],'GPU':False,'BLAS_arithmetic_used':False}
        def run(label,argv,timeout):
            result['native_commands'][label]={'argv':argv,'timeout_seconds':timeout};before=time.monotonic()
            try:q=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=min(timeout,max(1,600-(time.monotonic()-numeric))))
            except subprocess.TimeoutExpired as error:
                (OUT/(label+'.stdout.log')).write_bytes(error.stdout or b'');(OUT/(label+'.stderr.log')).write_bytes(error.stderr or b'');result['native_commands'][label]['terminal']='subprocess_timeout_killed_and_waited';raise
            (OUT/(label+'.stdout.log')).write_bytes(q.stdout);(OUT/(label+'.stderr.log')).write_bytes(q.stderr);result['native_commands'][label].update(exit_code=q.returncode,seconds=time.monotonic()-before);guard();return q
        stage='compile_frozen455';binary=OUT/'meth455_switch_component_cpu.exe';argv=[str(compiler),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',str(CPU),'-lpsapi','-o',str(binary)]
        q=run('compile',argv,120);assert q.returncode==0,q.stderr.decode(errors='replace');shutil.copyfile(dll,OUT/'libomp.dll')
        result['compile']={'argv':argv,'compiler_sha256':sha(compiler),'runtime_sha256':sha(OUT/'libomp.dll'),'binary_sha256':sha(binary)}
        stage='split_and_profile_primal';primal=OUT/'primal.bin';common=[str(binary),'--qualify',a['manifest'],str(outputs['bank.bin']),str(outputs['trace.bin']),str(outputs['primal.bin']),str(primal)]
        q=run('primal',common,240);assert q.returncode==0,q.stderr.decode(errors='replace');native=json.loads(q.stdout);assert native['tiny_qualified'];result['native_primal']=native
        with primal.open('rb') as f:header=f.read(28)
        assert header[:8]==b'MC455Q01' and struct.unpack_from('<5I',header,8)==(336,768,3072,3360,26144) and primal.stat().st_size==87843868
        gold=np.memmap(outputs['primal.bin'],dtype=WIRE,mode='r',offset=24,shape=(1680,));new=np.memmap(primal,dtype=WIRE455,mode='r',offset=28,shape=(3360,))
        for j,record in enumerate(new):
            assert int(record['mode'])==1+j//1680 and record['record'].tobytes()==gold[j%1680].tobytes(),('all_fields_BYTE_exact454',j)
        qualified_counts=np.array([gold[offset:offset+336]['nonzero'] for offset in (0,336,1008)])
        del new,gold;guard();result['C_primal']={'split_without_internal_timers_records':1680,'split_with_internal_timers_records':1680,'all_basis_raw_down_codes_scales_IDs_nonzero_BYTE_exact454':True}
        print(json.dumps({'C_primal':result['C_primal'],'native_seconds':native['native_seconds']}),flush=True)
        stage='nine_combination_whole_trace_profile';timings=OUT/'timings.jsonl';common[1]='--profile';common[-1]=str(timings)
        q=run('profile',common,300);assert q.returncode==0,q.stderr.decode(errors='replace');result['native_profile']=json.loads(q.stdout)
        rows=[json.loads(line) for line in timings.read_text(encoding='utf-8').splitlines()]
        expected=[(rep,(((rep+2)%9+slot)%9)//3,(((rep+2)%9+slot)%9)%3,i) for rep in range(-2,5) for slot in range(9) for i in range(336)]
        assert len(rows)==21168 and [(r['repetition'],r['arm'],r['mode'],r['query']) for r in rows]==expected
        books=np.array(parent['data']['books']);ids=parent['data']['selected_ids'];values=np.empty((5,3,3,336));components=np.empty((5,3,336,10));active=np.empty((3,336),np.int64)
        active_sets={0:{1,3,4,5,9},1:{0,1,3,4,5,6,7,8},2:{0,1,2,3,4,5,6,7,8}}
        for r in rows:
            rep,arm,mode,i=(r[k] for k in ('repetition','arm','mode','query'));assert r['id']==ids[i] and r['book']==int(books[i])
            assert np.isfinite(r['seconds']) and r['seconds']>0 and len(r['components'])==10 and all(np.isfinite(v) and v>=0 for v in r['components'])
            assert sum(r['components'])<=r['seconds']+1e-9
            if mode!=2:assert all(v==0 for v in r['components'])
            else:assert all(r['components'][k]==0 for k in set(range(10))-active_sets[arm])
            assert (r['nonzero']==-1 if arm==0 else r['nonzero']==int(qualified_counts[arm,i]))
            if rep>=0:
                values[rep,arm,mode,i]=r['seconds']
                if mode==2:components[rep,arm,i]=r['components'];active[arm,i]=r['nonzero']
        metrics={};admissibility={};profile={}
        for arm,label in enumerate(ARMS):
            metrics[label]={}
            for mode,name in enumerate(MODES):
                v=values[:,arm,mode,:];metrics[label][name]={'mean_seconds':float(np.mean(v)),'p95_seconds':float(np.quantile(v,.95)),'max_seconds':float(np.max(v)),
                    'per_sweep_seconds':np.sum(v,axis=1).tolist(),'per_book_mean_seconds':{str(b):float(np.mean(v[:,books==b])) for b in range(18,24)}}
            baseline,split,measured=(metrics[label][m] for m in MODES);sr=split['mean_seconds']/baseline['mean_seconds'];pr=measured['mean_seconds']/split['mean_seconds']
            book_sr={b:split['per_book_mean_seconds'][b]/v for b,v in baseline['per_book_mean_seconds'].items()};book_pr={b:measured['per_book_mean_seconds'][b]/v for b,v in split['per_book_mean_seconds'].items()}
            means=np.mean(components[:,arm,:,:],axis=(0,1));elapsed=measured['mean_seconds'];empty=result['native_profile']['empty_boundary_sequence_mean_seconds'][arm]
            profile[label]={'component_mean_seconds':dict(zip(COMPONENTS,means.tolist())),'component_mean_fraction_of_profiled_whole':dict(zip(COMPONENTS,(means/elapsed).tolist())),
                'boundary_and_return_residual_mean_seconds':elapsed-float(np.sum(means)),'split_to_unchanged_mean_ratio':sr,'profiled_to_split_mean_ratio':pr,
                'split_to_unchanged_per_book_ratios':book_sr,'profiled_to_split_per_book_ratios':book_pr,'empty_boundary_sequence_mean_seconds':empty,'empty_boundary_fraction_of_profiled_mean':empty/elapsed,
                'raw_times_not_overhead_subtracted':True}
            admissibility[label]={'split_mean_within10percent':.90<=sr<=1.10,'split_each_book_within15percent':all(.85<=v<=1.15 for v in book_sr.values()),
                'profile_mean_within10percent_of_split':.90<=pr<=1.10,'profile_each_book_within15percent_of_split':all(.85<=v<=1.15 for v in book_pr.values()),
                'empty_timer_fraction_le0_02':empty/elapsed<=.02}
        result['cost']=metrics;result['component_profile']=profile;result['diagnostic_admissibility_gates']=admissibility
        fractions=profile[ARMS[1]]['component_mean_fraction_of_profiled_whole'];usable=all(all(v.values()) for v in admissibility.values())
        if not usable:decision='profile_perturbation_FAIL_no_physical_recipe_selected'
        elif fractions['WI_dot_decode_scale']>=.50:decision='prepare_ONE_NEW_exact_output_tiled_WI_format'
        elif fractions['WO_column_accum']>=.30:decision='prepare_ONE_NEW_paired_active_WO_columns'
        else:decision='valid_diagnosis_without_predeclared_dominant_component_reassess_no_sweep'
        result['decision']=decision
        result['apparatus_gates']={'all_fresh_parent_outputs_helpers_sources_compiler_runtime_payload_bound':True,'all3360_split_profile_records_BYTE_exact454':True,
            'tiny_original454_and_split_WO_extrema_zero_single_fullI64_qualified':True,'all21168_timed_outputs_BYTE_exact454':True,'fixed_nine_combination_whole_trace_order':True,
            'component_partition_finite_nonnegative_and_unused_slots_zero':True,'one_physical_worker_mask1_in_both_native_children':all(len(n['worker_affinity'])==1 and n['worker_affinity'][0]['actual_mask']==1 for n in (result['native_primal'],result['native_profile']))}
        for path,state in initial.items():assert state==(Path(path).stat().st_size,Path(path).stat().st_mtime_ns)
        nativepeak=max(n['native_peak_bytes'] for n in (result['native_primal'],result['native_profile']));assert peak+nativepeak<=(5<<29),'conservative_total2_5GiB'
        result['data']=parent['data']|{'combinations':9,'timed_records':21168,'components':COMPONENTS,'arms':ARMS,'modes':MODES,'empty_timer_sequences_per_arm':10000}
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource']={'seconds':time.monotonic()-start,'admission_seconds':numeric-start,'numeric_seconds':time.monotonic()-numeric,'parent_peak_bytes':peak,'native_peak_bytes':nativepeak,
            'conservative_aggregate_peak_bytes':peak+nativepeak,'file_bytes_hashed':hashed,'output_bytes':sum(p['bytes'] for p in result['output_inventory']),'GPU':False,'optimizer_updates':0,'new_coefficient_roundings':0}
        result['scope']='Consumed fixed original bank11 inputs/IDs, one CPU worker. Same wire/floats/functions. Profile decomposition includes separation and timer bias controls; no adjusted promotion of454 failed kernels. Interleaved WI dot/decode/scales and original WO dot/scales stay coupled. No whole-model f, accepted rate, fresh quality, routing/n, DRAM counters or cross-family100B proof.'
        jobs(result);guard();write(args.out,result);print(json.dumps({'apparatus':result['apparatus_gates'],'diagnostic_gates':admissibility,'component_profile':profile,'decision':decision,'resource':result['resource'],'sha256':sha(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,parent_peak_bytes=peak,file_bytes_hashed=hashed)
        if OUT.exists():
            partial=[]
            for p in sorted(OUT.glob('*')):
                if p.is_file():
                    with p.open('rb') as f:partial.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(f,'sha256').hexdigest()})
            result['partial_output_inventory']=partial
        write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
