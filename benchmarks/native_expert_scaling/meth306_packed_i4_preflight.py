#!/usr/bin/env python3
"""Packed-I4/row-tile4 cost with exact format controls; no quality claim."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import psutil

import meth303_compact_i8_preflight as M

PRIOR=M.DOC/'meth303_compact_i8_preflight_result.json'
PRIOR_SHA='396edaee9c201324ffcb35eb57cadfb915a4ea6899694ca9e1055d8c29dd968b'
SOURCE_SHA='eee66947fec515b1f7805a20bd7de2be7b92e665c8a0c4602d1a7a80e2fcdcf5'
DRIVER_SHA='3b0300ada9b645fbf4576dff301cfa93bc0635ab8254e66d080de83570ef4765'
CPU=M.ROOT/'benchmarks/native_expert_scaling/meth306_packed_i4_cpu.c'

def ledger():
    v=M.catalogue();v['active_coefficients']=v.pop('active_i8_coefficients');v['stored_coefficients']=v.pop('stored_i8_coefficients')
    v['exact_sampled_integer_rows']=v.pop('exact_sampled_i8_rows');v['stored_code_bytes']=v['stored_coefficients']//2
    v['active_code_bytes']=v['active_coefficients']//2
    v['complete_addressed_weight_bytes']-=v['active_code_bytes'];assert v['complete_addressed_weight_bytes']==310571680
    v['scope']='Packed signedI4 synthetic geometry, all coefficients retained but precision changed. Descriptor addresses, not physical DRAM.'
    return v

def unchanged_composition():
    old=M.CPU.read_text(encoding='utf-8');new=CPU.read_text(encoding='utf-8')
    for begin,end in [('static void execute(void){','static Matrix *matrix('),('static void prepare(unsigned token){','static Matrix *matrix('),('static double decoded_head(unsigned row){','static void selftest(void){')]:
        assert old[old.index(begin):old.index(end)]==new[new.index(begin):new.index(end)],begin
    return True

def native(result,ledger):
    available=psutil.virtual_memory().available;assert available>=16*(1<<30),'available RAM stop'
    log=M.OUT/'meth306_native_run1.jsonl';err=M.OUT/'meth306_native_run1.stderr.log'
    assert not log.exists() and not err.exists()
    command=[str(M.EXE),str(M.SPEC),str(M.MODEL)]
    env=os.environ.copy();env['PATH']=str(M.TOOLCHAIN/'bin')+os.pathsep+env.get('PATH','')
    env.update({'OMP_NUM_THREADS':'6','OMP_DYNAMIC':'FALSE','OMP_WAIT_POLICY':'PASSIVE'})
    run={'command':command,'log_path':str(log),'stderr_path':str(err),'available_ram_before_bytes':available,
         'environment':{k:env[k] for k in ('OMP_NUM_THREADS','OMP_DYNAMIC','OMP_WAIT_POLICY')},'peak_rss_bytes':0}
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
    assert len(operators)==30 and len(events)==33
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
    start=time.monotonic();stage='bindings';result={'experiment':'METH-306-packed-I4-row-tile4-cost'}
    try:
        assert M.digest(PRIOR)==PRIOR_SHA and M.digest(M.CPU)==SOURCE_SHA and M.digest(Path(M.__file__))==DRIVER_SHA
        for rel,sha in M.PINS.items():assert M.digest(M.ROOT/rel)==sha,rel
        p=json.loads(PRIOR.read_text(encoding='utf-8'))
        result.update({'prior_sha256':PRIOR_SHA,'immutable303_source_sha256':SOURCE_SHA,'immutable303_controller_sha256':DRIVER_SHA,
                       'controller_sha256':M.digest(Path(__file__)),'cpu_source_sha256':M.digest(CPU),'engine_sha256':M.digest(M.ENGINE),
                       'full_composition_source_routing_and_head_exact303':unchanged_composition(),'ledger':ledger(),'gates':{}})
        result['gates']['complete_weight_560mb']=result['ledger']['complete_addressed_weight_bytes']<=560000000
        M.OUT=M.OUT/'meth306_packed_i4';M.OUT.mkdir(exist_ok=True)
        M.EXE=M.OUT/'meth306_packed_i4_cpu.exe';M.SPEC=M.OUT/'meth306_spec.bin'
        stage='source_and_compile';result['spec'],result['source_binding']=M.make_spec()
        assert result['spec']['sha256']==p['spec']['sha256'] and result['source_binding']==p['source_binding']
        assert not M.EXE.exists()
        command=[str(M.COMPILER),'-O3','-mavx2','-mssse3','-mfma','-fopenmp','-ffp-contract=off','-std=c11',
                 '-DSILICON_PACKED_I4_PREFLIGHT',str(M.ENGINE),'-o',str(M.EXE),'-lm','-lpsapi','-lbcrypt']
        c=subprocess.run(command,cwd=M.ROOT,capture_output=True,text=True,timeout=120)
        result['compile']={'command':command,'exit_code':c.returncode,'stdout':c.stdout,'stderr':c.stderr,
                           'compiler_sha256':M.digest(M.COMPILER),'libomp_sha256':M.digest(M.TOOLCHAIN/'bin/libomp.dll')}
        assert c.returncode==0,c.stderr
        result['executable_sha256']=M.digest(M.EXE);stage='native';native(result,result['ledger'])
        checks=result['native']['selftest']
        assert checks['exhaustive_nibble_input_cases']==4080 and checks['tile_lane_cases']==16320 and checks['mixed_adjacent_pairs']==30976
        assert checks['row_bias_error_detected'] and checks['packed_nibble_fault_detected']
        result['gates'].update({'packed_format_tile_lanes_and_fault_controls':True,'complete_coefficient_capacity_retained':True})
        if not result['gates']['repeatability_1_10']:result['decision']='inconclusive_packed_I4_variation_stop_before_training'
        elif not result['gates']['all_operator_medians_14ms']:result['decision']='reject_unchanged_packed_I4_kernel_before_teacher_collection_or_training'
        else:result['decision']='eligible_only_for_separately_frozen_teacher_function_fit_and_I4_precision_quality'
        result['scope']='Precision and scheduling candidate, synthetic640-function bank with real Q6 head/router/norms. Own format fidelity, no source-I8 equivalence or donor quality/useful n/causal inference/physical DRAM/accepted-rate claim.'
        result['total_seconds']=time.monotonic()-start;M.write(path,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'rep_medians_ms':[v*1000 for v in result['native']['rep_medians_seconds']],
                          'peak_rss_bytes':result['native']['peak_rss_bytes'],'result_sha256':M.digest(path),'total_seconds':result['total_seconds']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'total_seconds':time.monotonic()-start})
        M.write(path.with_suffix('.failure.json'),result);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists();run(args.out)

