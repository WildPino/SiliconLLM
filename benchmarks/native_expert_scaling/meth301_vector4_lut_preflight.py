#!/usr/bin/env python3
"""Source-bound descriptor accounting plus actual phase60 vector4 LUT costs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import time

import psutil

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT=ROOT/'results/native_expert_scaling'
PRIOR=DOC/'meth300_gigachat_vector_lut_cost_result.json'
PRIOR_SHA='9458aec3f6baafbe00eaa266a3ac7da84c43040153cbfee3fc24ea9fd64dcd75'
CPU=ROOT/'benchmarks/native_expert_scaling/meth301_vector4_lut_cpu.c'
ENGINE=ROOT/'benchmarks/phase60/engine.c'
TOOLCHAIN=Path('C:/Users/giosa/AppData/Local/Microsoft/WinGet/Packages/MartinStorsjo.LLVM-MinGW.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/llvm-mingw-20251216-msvcrt-x86_64')
COMPILER=TOOLCHAIN/'bin/clang.exe'
EXE=OUT/'meth301_vector4_lut_cpu.exe'
SPEC=OUT/'meth301_vector4_lut_spec.bin'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    with Path(path).open('x',encoding='utf-8',newline='\r\n') as f:f.write(json.dumps(obj,indent=2,ensure_ascii=True)+'\n')

def descriptors(prior):
    encoded={'mla','routed','shared','dense0','head'}
    selected=[r for r in prior['tensor_rows'] if r['organ'] in encoded]
    assert len(selected)==284 and all(r['input_width']%4==0 and r['output_width']%32==0 for r in selected)
    codes=sum(r['active_elements']//4 for r in selected)
    scales=sum(r['scale_bytes'] for r in selected)
    palettes=len(selected)*256*4*4
    unchanged=sum(r['q4_active_bytes'] for r in prior['tensor_rows'] if r['organ'] not in encoded)
    entries=sum(r['input_width']//4*r['distinct_query_inputs']*256 for r in selected)
    stored=sum(r['stored_code_bytes']//2+r['stored_scale_bytes']+4096 for r in selected)
    stored+=sum(r['q4_stored_bytes'] for r in prior['tensor_rows'] if r['organ'] not in encoded)
    per_expert=(prior['organs']['routed']['stored_code_bytes']//2+prior['organs']['routed']['stored_scale_bytes'])//64
    router_per_expert=prior['organs']['router']['q4_active_bytes']//64
    result={'coefficients_per_U8':4,'palette_entries':256,'codes_bytes_per_token':codes,
            'scales_bytes_per_token':scales,'palette_bytes':palettes,'unchanged_q4_bytes_per_token':unchanged,
            'complete_weight_bytes_per_token':codes+scales+palettes+unchanged,
            'table_entries_per_token':entries,'logical_table_write_bytes_per_token':entries*4,
            'logical_table_gather_bytes_per_token':codes*4,'table_construct_multiply_add_operations':entries*7,
            'encoded_model_payload_bytes':stored,'parametric_n640_complete_weight_bytes':codes+scales+palettes+unchanged+576*(router_per_expert+100),
            'parametric_n640_encoded_model_payload_bytes':stored+576*(per_expert+router_per_expert+100)}
    assert codes==406_405_120 and entries==70_352_896
    return result

def make_spec(prior):
    rows=[];names=[]
    for r in prior['tensor_rows']:
        if r['organ'] not in {'mla','routed','shared','dense0','head','router'}:continue
        name=r['name'];layer=int(name.split('.')[1]) if name.startswith('blk.') else 26
        if r['organ']=='router':d,o=r['gguf_shape'];b=a=q=1;kind=3
        else:
            d,o,b,a,q=(r[k] for k in ('input_width','output_width','stored_banks','active_banks','distinct_query_inputs'))
            kind=2 if name.endswith('ffn_down_exps.weight') else 1 if r['organ']=='routed' else 0
        rows.append(struct.pack('<7I',layer,kind,d,o,b,a,q));names.append(name)
    assert len(rows)==309
    with SPEC.open('xb') as f:f.write(b'M301SPC1'+struct.pack('<2I',len(rows),64)+b''.join(rows))
    return {'path':str(SPEC),'bytes':SPEC.stat().st_size,'sha256':digest(SPEC),'ordered_source_tensor_names':names}

def native(n,number):
    available=psutil.virtual_memory().available
    assert available>(8 if n==64 else 36)*(1<<30),'RAM availability stop'
    log=OUT/f'meth301_native_run{number}_n{n}.jsonl';err=log.with_suffix('.stderr.log')
    assert not log.exists() and not err.exists()
    command=[str(EXE),str(SPEC),str(n)]
    env=os.environ.copy();env['PATH']=str(TOOLCHAIN/'bin')+os.pathsep+env.get('PATH','')
    env['OMP_NUM_THREADS']='6';env['OMP_DYNAMIC']='FALSE'
    env['OMP_WAIT_POLICY']='PASSIVE'
    start=time.monotonic();peak=0
    with log.open('x',encoding='utf-8') as out,err.open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=stderr)
        child=psutil.Process(process.pid)
        while process.poll() is None:
            try:peak=max(peak,child.memory_info().rss)
            except psutil.NoSuchProcess:pass
            if time.monotonic()-start>600 or peak>40*(1<<30):
                process.kill();process.wait();raise RuntimeError('native resource stop')
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
    assert process.returncode==0,(n,process.returncode,err.read_text(encoding='utf-8')[-2000:])
    data=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()]
    ready=[v for v in data if v['event']=='ready'];checks=[v for v in data if v['event']=='selftest']
    finished=[v for v in data if v['event']=='finished'];tokens=[v for v in data if v['event']=='token']
    assert len(ready)==len(checks)==len(finished)==1 and len(tokens)==30
    assert ready[0]['lookups_per_token']==406405120 and ready[0]['table_entries_per_token']==70352896
    assert checks[0]['bad_table_detected'] and checks[0]['pooled_relative_l2']<=1e-5
    hashes=[[v['output_hash'] for v in tokens if v['rep']==i] for i in range(3)]
    assert hashes[0]==hashes[1]==hashes[2]
    medians=[statistics.median(v['seconds'] for v in tokens if v['rep']==i and not v['warmup']) for i in range(3)]
    peak=max(peak,finished[0]['peak_rss_bytes'])
    result={'n':n,'number':number,'command':command,'environment':{'OMP_NUM_THREADS':'6','OMP_DYNAMIC':'FALSE','OMP_WAIT_POLICY':'PASSIVE'},
            'log_path':str(log),'log_sha256':digest(log),'stderr_path':str(err),'stderr_sha256':digest(err),
            'ready':ready[0],'selftest':checks[0],'tokens':tokens,'finished':finished[0],
            'rep_medians_seconds':medians,'max_over_min_rep_median':max(medians)/min(medians),
            'peak_rss_bytes':peak,'available_ram_before_bytes':available,'seconds':time.monotonic()-start}
    print(json.dumps({'completed_native_run':number,'n':n,'rep_medians_ms':[v*1000 for v in medians],
                      'peak_rss_bytes':peak,'seconds':result['seconds']}),flush=True)
    return result

def run(path):
    start=time.monotonic();result={'experiment':'METH-301-source-shaped-vector4-U8-LUT-cost-preflight','runs':[]}
    stage='bindings'
    try:
        assert digest(PRIOR)==PRIOR_SHA
        prior=json.loads(PRIOR.read_text(encoding='utf-8'))
        result.update({'prior_sha256':PRIOR_SHA,'source_bindings':prior['source_bindings'],
                       'controller_sha256':digest(Path(__file__)),'cpu_source_sha256':digest(CPU),'engine_sha256':digest(ENGINE),
                       'scientific_scope':'Source-shaped synthetic fixtures and real allocated/consulted RAM; no donor weights encoded, quality, useful extra capacity or accepted end-to-end rate.'})
        ledger=descriptors(prior);result['descriptor_ledger']=ledger
        result['gates']={'n64_weight_560mb':ledger['complete_weight_bytes_per_token']<=560_000_000,
                         'n640_weight_560mb':ledger['parametric_n640_complete_weight_bytes']<=560_000_000}
        if not all(result['gates'].values()):result['decision']='descriptor_cost_stop';return
        stage='spec_and_compile';result['spec']=make_spec(prior)
        assert not EXE.exists()
        command=[str(COMPILER),'-O3','-mavx2','-mssse3','-mfma','-fopenmp','-ffp-contract=off','-std=c11',
                 '-DSILICON_VECTOR4_LUT_PREFLIGHT',str(ENGINE),'-o',str(EXE),'-lm','-lpsapi']
        compiled=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
        result['compile']={'command':command,'exit_code':compiled.returncode,'stdout':compiled.stdout,'stderr':compiled.stderr,
                           'compiler_sha256':digest(COMPILER)}
        assert compiled.returncode==0,compiled.stderr
        result['executable_sha256']=digest(EXE)
        stage='native_n64';small=native(64,1);result['runs'].append(small)
        result['gates'].update({'n64_numeric_controls':True,'n64_repeatability_1_10':small['max_over_min_rep_median']<=1.10,
                               'n64_operator_all_medians_14ms':max(small['rep_medians_seconds'])<=.014})
        if not result['gates']['n64_repeatability_1_10']:
            result['decision']='n64_native_inconclusive_variation';return
        if not result['gates']['n64_operator_all_medians_14ms']:
            result['decision']='reject_unchanged_vector4_native_layout_before_palette_training';return
        stage='native_n640';large=native(640,2);result['runs'].append(large)
        result['gates'].update({'n640_numeric_controls':True,'n640_repeatability_1_10':large['max_over_min_rep_median']<=1.10,
                               'n640_operator_all_medians_14ms':max(large['rep_medians_seconds'])<=.014,
                               'native_scaling_ratio_1_25':statistics.median(large['rep_medians_seconds'])/statistics.median(small['rep_medians_seconds'])<=1.25})
        result['decision']='eligible_only_for_separately_frozen_source_quality' if all(result['gates'].values()) else 'n640_native_cost_or_variation_stop'
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'decision':'apparatus_or_resource_failure'})
        write(path.with_suffix('.failure.json'),result);raise
    finally:
        result['total_seconds']=time.monotonic()-start
        if 'error' not in result:
            write(path,result)
            print(json.dumps({'decision':result.get('decision'),'gates':result.get('gates'),
                              'result_sha256':digest(path),'total_seconds':result['total_seconds']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    run(args.out)
