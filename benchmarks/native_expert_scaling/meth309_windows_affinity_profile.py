#!/usr/bin/env python3
"""New direct Windows binding profile; exact immutable306 mathematical controls."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import psutil
import meth308_physical_core_profile as A
W=A.W
B=A.B
M=A.M
CPU=M.ROOT/'benchmarks/native_expert_scaling/meth309_windows_affinity_cpu.c'

def native(result,ledger,selected):
    available=psutil.virtual_memory().available;assert available>=16*(1<<30),'available RAM stop'
    log=M.OUT/'meth309_native_run1.jsonl';err=M.OUT/'meth309_native_run1.stderr.log'
    assert not log.exists() and not err.exists()
    command=[str(M.EXE),str(M.SPEC),str(M.MODEL),','.join(map(str,selected))]
    env=os.environ.copy();env['PATH']=str(M.TOOLCHAIN/'bin')+os.pathsep+env.get('PATH','')
    env.update({'OMP_NUM_THREADS':'6','OMP_DYNAMIC':'FALSE','OMP_WAIT_POLICY':'ACTIVE','KMP_BLOCKTIME':'200','KMP_SETTINGS':'TRUE','KMP_AFFINITY':'none'})
    for key in ('KMP_LIBRARY','OMP_PLACES','OMP_PROC_BIND'):env.pop(key,None)
    run={'command':command,'log_path':str(log),'stderr_path':str(err),'available_ram_before_bytes':available,
         'environment':{k:env[k] for k in ('OMP_NUM_THREADS','OMP_DYNAMIC','OMP_WAIT_POLICY','KMP_BLOCKTIME','KMP_SETTINGS','KMP_AFFINITY')},'peak_rss_bytes':0}
    result['native']=run;start=time.monotonic()
    with log.open('x',encoding='utf-8') as out,err.open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=M.ROOT,env=env,stdout=out,stderr=stderr);run['pid']=process.pid
        child=psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:run['peak_rss_bytes']=max(run['peak_rss_bytes'],child.memory_info().rss)
                except psutil.NoSuchProcess:pass
                if time.monotonic()-start>600 or run['peak_rss_bytes']>24*(1<<30):raise RuntimeError('native resource stop')
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        finally:
            if process.poll() is None:process.kill();process.wait()
            run.update({'exit_code':process.returncode,'seconds':time.monotonic()-start})
    run.update({'log_sha256':M.digest(log),'stderr_sha256':M.digest(err)})
    assert process.returncode==0,err.read_text(encoding='utf-8')[-2000:]
    events=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()]
    def one(event):
        values=[v for v in events if v['event']==event];assert len(values)==1;return values[0]
    ready,checks,finished=one('ready'),one('selftest'),one('finished');operators=[v for v in events if v['event']=='operator']
    run.update({'ready':ready,'selftest':checks,'finished':finished,'operators':operators})
    assert len(operators)==30 and len(events)==40
    affinity=[v for v in events if v['event']=='native_affinity'];assert [v['phase'] for v in affinity]==list(range(7))
    for phase in affinity:
        assert len(phase['threads'])==6
        for t,item in enumerate(phase['threads']):
            assert item['thread']==t and item['group']==0 and item['mask']==1<<selected[t]
            if phase['phase']:assert item['current_cpu']==selected[t]
    run['affinity_checks']=affinity
    assert ready['active_coefficients']==ledger['active_coefficients'] and ready['active_row_scale_bytes']==ledger['active_scale_bytes']
    assert ready['stored_coefficients']==ledger['stored_coefficients'] and ready['functions_per_layer']==640
    assert ready['code_bits']==4 and ready['row_tile']==4 and ready['serial_if_coefficients_less_than']==32768
    assert ready['allocated_bytes']==5_106_289_792
    assert checks['exact_integer_rows']==ledger['exact_sampled_integer_rows'] and checks['stored_bank_offset_checks']==ledger['bank_edge_checks']
    assert checks['integer_relative_l2']<=1e-6 and checks['real_head64_relative_l2']<=1e-5 and checks['reference_route_layers']==25
    assert all(checks[k] for k in ('bad_bank_rejected','packed_head_change_detected','integer_extrema_and_quantizer_checks'))
    hashes=[];medians=[]
    for rep in range(3):
        obs=[v for v in operators if v['rep']==rep];assert [v['input'] for v in obs]==list(range(10))
        assert all(v['warmup']==(v['input']<2) and 0<v['seconds']<600 for v in obs)
        hashes.append([v['output_route_hash'] for v in obs]);medians.append(statistics.median(v['seconds'] for v in obs if not v['warmup']))
    assert hashes[0]==hashes[1]==hashes[2]
    assert 1<=finished['minimum_selected_child_union']<=finished['maximum_selected_child_union']<=40
    run.update({'rep_medians_seconds':medians,'max_over_min_rep_median':max(medians)/min(medians),
                'peak_rss_bytes':max(run['peak_rss_bytes'],finished['peak_rss_bytes'])})
    assert run['peak_rss_bytes']<=24*(1<<30)
    result['gates'].update({'numeric_and_route_controls':True,'repeatability_1_10':run['max_over_min_rep_median']<=1.10,
                           'all_operator_medians_14ms':max(medians)<=.014})

def run(path):
    start=time.monotonic();stage='bindings';result={'experiment':'METH-309-direct-Windows-physical-core-profile','runs':[]}
    try:
        assert M.digest(W.PRIOR)==W.PRIOR_SHA and M.digest(Path(B.__file__))==W.DRIVER_SHA and M.digest(B.CPU)==W.CPU_SHA
        assert M.digest(Path(W.__file__))=='3c08734e08a890028d6bc16923358ebde9ac727a32a1271237c0b9464d276352'
        assert M.digest(Path(A.__file__))=='7f0a8951dcc011c898edae3b2128fde568fce6654d366dc3da6239aeb5cb001f'
        assert M.digest(Path(M.__file__))=='3b0300ada9b645fbf4576dff301cfa93bc0635ab8254e66d080de83570ef4765'
        for rel,sha in M.PINS.items():assert M.digest(M.ROOT/rel)==sha,rel
        prior=json.loads(W.PRIOR.read_text(encoding='utf-8'))
        topo=A.topology();result['topology']=topo
        result.update({'prior_sha256':W.PRIOR_SHA,'immutable306_controller_sha256':W.DRIVER_SHA,'immutable306_cpu_sha256':W.CPU_SHA,
                       'controller_sha256':M.digest(Path(__file__)),'cpu_source_sha256':M.digest(CPU),'engine_sha256':M.digest(M.ENGINE),
                       'ledger':B.ledger(),'inherited_omp_kmp_environment':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','KMP_'))}})
        root=M.OUT/'meth309_windows_affinity';root.mkdir(exist_ok=True)
        M.SPEC=root/'meth309_spec.bin';M.EXE=root/'meth309_windows_affinity_cpu.exe'
        stage='source_and_compile';result['spec'],result['source_binding']=M.make_spec()
        assert result['spec']['sha256']==prior['spec']['sha256'] and result['source_binding']==prior['source_binding']
        assert not M.EXE.exists()
        command=[str(M.COMPILER),'-O3','-mavx2','-mssse3','-mfma','-fopenmp','-ffp-contract=off','-std=c11',
                 '-DSILICON_WINDOWS_AFFINITY_PROFILE',str(M.ENGINE),'-o',str(M.EXE),'-lm','-lpsapi','-lbcrypt']
        c=subprocess.run(command,cwd=M.ROOT,capture_output=True,text=True,timeout=120)
        result['compile']={'command':command,'exit_code':c.returncode,'stdout':c.stdout,'stderr':c.stderr,
                           'compiler_sha256':M.digest(M.COMPILER),'libomp_sha256':M.digest(M.TOOLCHAIN/'bin/libomp.dll')}
        assert c.returncode==0,c.stderr
        result['executable_sha256']=M.digest(M.EXE)
        for number in range(1,4):
            stage='native_'+str(number);M.OUT=root/str(number);M.OUT.mkdir(exist_ok=True)
            item={'number':number,'gates':{}};result['runs'].append(item);native(item,result['ledger'],topo['selected_logical_ids'])
            n=item['native'];assert n['selftest']==prior['native']['selftest']
            for key in ('parent_count','children_per_parent','functions_per_layer','threads','code_bits','row_tile','serial_if_coefficients_less_than','allocated_bytes','active_coefficients','active_row_scale_bytes','stored_coefficients'):
                assert n['ready'][key]==prior['native']['ready'][key],key
            assert [x['output_route_hash'] for x in n['operators']]==[x['output_route_hash'] for x in prior['native']['operators']]
            stderr=Path(n['stderr_path']).read_text(encoding='utf-8')
            assert W.effective(stderr,'OMP_WAIT_POLICY')=='ACTIVE' and re.fullmatch(r'200(?:MS)?',W.effective(stderr,'KMP_BLOCKTIME'))
            assert W.effective(stderr,'KMP_AFFINITY').endswith('NONE')
        process_medians=[statistics.median(x['native']['rep_medians_seconds']) for x in result['runs']]
        drift=max(process_medians)/min(process_medians)
        result['process_medians_seconds']=process_medians;result['max_over_min_process_median']=drift
        result['gates']={'all90_hashes_selftests_counts_exact306':True,'seven_binding_checks_per_process':True,
                         'effective_active200ms_library_affinity_none':True,
                         'each_process_repeatability_1_10':all(x['gates']['repeatability_1_10'] for x in result['runs']),
                         'cross_process_repeatability_1_10':drift<=1.10,
                         'all_nine_medians_14ms':all(x['gates']['all_operator_medians_14ms'] for x in result['runs'])}
        if not result['gates']['each_process_repeatability_1_10'] or not result['gates']['cross_process_repeatability_1_10']:
            result['decision']='direct_windows_profile_inconclusive_variation_no_training'
        elif not result['gates']['all_nine_medians_14ms']:
            result['decision']='direct_windows_profile_cost_stop_before_teacher_collection_or_training'
        else:result['decision']='eligible_only_for_separately_frozen_real_teacher_function_fit_and_I4_quality'
        result['scope']='New binary with exact306 mathematical controls and verified Windows binding. Synthetic640-function bank; no isolated speedup, trained quality, useful n, causal inference, physical DRAM or accepted-rate claim.'
        result['total_seconds']=time.monotonic()-start;M.write(path,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'rep_medians_ms':[[v*1000 for v in x['native']['rep_medians_seconds']] for x in result['runs']],
                          'cross_process_ratio':drift,'result_sha256':M.digest(path),'total_seconds':result['total_seconds']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'total_seconds':time.monotonic()-start})
        M.write(path.with_suffix('.failure.json'),result);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists();run(args.out)
