#!/usr/bin/env python3
"""Frozen source-bound compact I8 operator cost, not student quality/rate."""
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
import meth300_gigachat_vector_lut_cost as source

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT=ROOT/'results/native_expert_scaling'
CPU=ROOT/'benchmarks/native_expert_scaling/meth303_compact_i8_cpu.c'
ENGINE=ROOT/'benchmarks/phase60/engine.c'
MODEL=source.Q4
SPEC=OUT/'meth303_compact_i8_spec.bin'
EXE=OUT/'meth303_compact_i8_cpu.exe'
CONFIG=source.MODELS/'strat01_gigachat_source_189fff27/config.json'
TOOLCHAIN=Path('C:/Users/giosa/AppData/Local/Microsoft/WinGet/Packages/MartinStorsjo.LLVM-MinGW.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/llvm-mingw-20251216-msvcrt-x86_64')
COMPILER=TOOLCHAIN/'bin/clang.exe'
PINS={
    'benchmarks/native_expert_scaling/meth300_gigachat_vector_lut_cost.py':'986abfee67cc7137bed3aecaf6a32c469d38c61d481b16859275476ce6e92ea4',
    'benchmarks/phase60/strat01_q6k_q8k_avx2.h':'4de37b009a9d3173178c60a7db309d73179bb1909fba049e36c6240f23d7544d',
    'benchmarks/phase60/strat01_q4k_q8k.h':'49b5f39fd1a1313e15038208420cf7484765b01bc3f83952382577c6be68bc5f',
    'benchmarks/phase60/strat01_f32_dot_reference_generic.h':'362abe0f3fddb6e49b55ce289cbb4212f3b5fa001b991b4a84db894e573d0ccd',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/meth300_gigachat_vector_lut_cost_result.json':'9458aec3f6baafbe00eaa266a3ac7da84c43040153cbfee3fc24ea9fd64dcd75',
}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    with Path(path).open('x',encoding='utf-8',newline='\r\n') as f:f.write(json.dumps(obj,indent=2,ensure_ascii=True)+'\n')

def catalogue():
    rows=[]
    def add(layer,name,d,o,b,a):rows.append({'layer':layer,'name':name,'d':d,'o':o,'stored_banks':b,'active_banks':a})
    for l in range(26):
        for name,d,o,b,a in [('q',1536,1536,1,1),('kv',1536,576,1,1),('kb',128,512,8,8),('vb',512,192,8,8),('wo',1536,1536,1,1)]:add(l,name,d,o,b,a)
        if not l:
            for name,d,o in [('g',1536,1024),('u',1536,1024),('down',1024,1536)]:add(l,name,d,o,1,1)
        else:
            for name,d,o in [('g',1536,128),('u',1536,128),('down',128,1536)]:add(l,name,d,o,640,4)
            for name,d,o in [('sg',1536,256),('su',1536,256),('sd',256,1536),('cq',1536,16)]:add(l,name,d,o,1,1)
    active=sum(r['d']*r['o']*r['active_banks'] for r in rows)
    scales=sum(4*r['o']*r['active_banks'] for r in rows)
    stored=sum(r['d']*r['o']*r['stored_banks'] for r in rows)
    expected_checks=sum(8*r['active_banks'] for r in rows)
    bank_checks=sum(2*r['stored_banks'] for r in rows)
    assert len(rows)==308 and active==273571840 and scales==1902656
    complete=active+scales+25*64*1536*4+161602560+386144+25*4*10*16*4
    assert complete==447357600
    return {'matrices':rows,'active_i8_coefficients':active,'active_scale_bytes':scales,'stored_i8_coefficients':stored,
            'exact_sampled_i8_rows':expected_checks,'bank_edge_checks':bank_checks,'complete_addressed_weight_bytes':complete,
            'scope':'Descriptor address count, not physical DRAM measurement. Includes priced embedding row absent from timing.'}

def make_spec():
    assert digest(CONFIG)=='6a6b8260f08791c4968f70934903e9aa892a53ee3f2e2e05b61a68ea1ff33503'
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    for key,value in {'hidden_size':1536,'num_hidden_layers':26,'num_attention_heads':32,'n_routed_experts':64,
                      'num_experts_per_tok':4,'moe_intermediate_size':1280,'topk_method':'noaux_tc','n_group':1,
                      'topk_group':1,'norm_topk_prob':True,'scoring_func':'sigmoid','routed_scaling_factor':1.0}.items():assert config[key]==value,(key,config.get(key))
    fields,tensors,binding=source.header(MODEL,6474702976)
    assert binding['header_sha256']=='3d96ca766891be60b055441fcaa6d127014423d90431f693e2e14ee4db0db0c6'
    pieces=[b'M303SPC1',struct.pack('<6I',26,64,10,128,256,8)]
    components=[]
    with MODEL.open('rb') as f:
        def get(name,shape,kind):
            t=tensors[name];assert tuple(t['shape'])==shape and t['type']==kind
            offset=binding['header_bytes_read']+t['offset'];f.seek(offset);data=f.read(t['bytes']);assert len(data)==t['bytes']
            sha=hashlib.sha256(data).digest();components.append({'name':name,'offset':offset,'bytes':len(data),'sha256':sha.hex()})
            return offset,data,sha
        off,head,sha=get('output.weight',(1536,128256),'Q6_K')
        pieces.extend([struct.pack('<3Q',6474702976,off,len(head)),sha]);del head
        off,data,sha=get('output_norm.weight',(1536,),'F32');pieces.extend([struct.pack('<Q',off),sha])
        for l in range(26):
            prefix=f'blk.{l}.';offsets=[];payloads=[]
            for name,shape in [('attn_norm.weight',(1536,)),('ffn_norm.weight',(1536,)),('attn_kv_a_norm.weight',(512,))]:
                off,data,sha=get(prefix+name,shape,'F32');offsets.append(off);payloads.append(data)
            if l:
                for name,shape in [('ffn_gate_inp.weight',(1536,64)),('exp_probs_b.bias',(64,))]:
                    off,data,sha=get(prefix+name,shape,'F32');offsets.append(off);payloads.append(data)
            else:offsets.extend([0,0])
            pieces.extend([struct.pack('<5Q',*offsets),hashlib.sha256(b''.join(payloads)).digest()])
    raw=b''.join(pieces);assert len(raw)==2000
    with SPEC.open('xb') as f:f.write(raw)
    binding.update({'components':components,'component_bytes_freshly_hashed':sum(c['bytes'] for c in components),
                    'previous_full_sha256_not_rechecked':'68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb',
                    'config_sha256':digest(CONFIG)})
    return {'path':str(SPEC),'bytes':len(raw),'sha256':digest(SPEC)},binding

def native(result,ledger):
    available=psutil.virtual_memory().available;assert available>=16*(1<<30),'available RAM stop'
    log=OUT/'meth303_native_run1.jsonl';err=OUT/'meth303_native_run1.stderr.log'
    assert not log.exists() and not err.exists()
    command=[str(EXE),str(SPEC),str(MODEL)]
    env=os.environ.copy();env['PATH']=str(TOOLCHAIN/'bin')+os.pathsep+env.get('PATH','')
    env.update({'OMP_NUM_THREADS':'6','OMP_DYNAMIC':'FALSE','OMP_WAIT_POLICY':'PASSIVE'})
    run={'command':command,'log_path':str(log),'stderr_path':str(err),'available_ram_before_bytes':available,
         'environment':{k:env[k] for k in ('OMP_NUM_THREADS','OMP_DYNAMIC','OMP_WAIT_POLICY')},'peak_rss_bytes':0}
    result['native']=run;start=time.monotonic()
    with log.open('x',encoding='utf-8') as out,err.open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=stderr);run['pid']=process.pid
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
    run.update({'log_sha256':digest(log),'stderr_sha256':digest(err)})
    assert process.returncode==0,err.read_text(encoding='utf-8')[-2000:]
    events=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()]
    def one(event):
        values=[v for v in events if v['event']==event];assert len(values)==1;return values[0]
    ready,checks,finished=one('ready'),one('selftest'),one('finished');operators=[v for v in events if v['event']=='operator']
    run.update({'ready':ready,'selftest':checks,'finished':finished,'operators':operators})
    assert len(operators)==30 and len(events)==33
    assert ready['active_i8_coefficients']==ledger['active_i8_coefficients'] and ready['active_row_scale_bytes']==ledger['active_scale_bytes']
    assert ready['stored_i8_coefficients']==ledger['stored_i8_coefficients'] and ready['functions_per_layer']==640
    assert checks['exact_i8_rows']==ledger['exact_sampled_i8_rows'] and checks['stored_bank_offset_checks']==ledger['bank_edge_checks']
    assert checks['i8_relative_l2']<=1e-6 and checks['real_head64_relative_l2']<=1e-5 and checks['reference_route_layers']==25
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
    start=time.monotonic();result={'experiment':'METH-303-compact-I8-source-Q6-head-cost','stage':'bindings'}
    try:
        for rel,sha in PINS.items():assert digest(ROOT/rel)==sha,rel
        result.update({'immutable_pins':PINS,'controller_sha256':digest(Path(__file__)),'cpu_source_sha256':digest(CPU),
                       'engine_sha256':digest(ENGINE),'ram_total_bytes':psutil.virtual_memory().total,'cpu_logical_threads':psutil.cpu_count(),
                       'scope':'Synthetic compact functions with real source Q6 head/F32 macro router/norms. Input-ready attention contexts; no causal LM, quality, useful n or accepted rate.'})
        ledger=catalogue();result['ledger']=ledger;result['gates']={'complete_weight_560mb':ledger['complete_addressed_weight_bytes']<=560000000}
        assert result['gates']['complete_weight_560mb']
        result['stage']='spec_and_compile';result['spec'],result['source_binding']=make_spec()
        assert not EXE.exists()
        command=[str(COMPILER),'-O3','-mavx2','-mssse3','-mfma','-fopenmp','-ffp-contract=off','-std=c11',
                 '-DSILICON_COMPACT_I8_PREFLIGHT',str(ENGINE),'-o',str(EXE),'-lm','-lpsapi','-lbcrypt']
        compile_run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
        result['compile']={'command':command,'exit_code':compile_run.returncode,'stdout':compile_run.stdout,'stderr':compile_run.stderr,
                           'compiler_sha256':digest(COMPILER),'libomp_sha256':digest(TOOLCHAIN/'bin/libomp.dll')}
        assert compile_run.returncode==0,compile_run.stderr
        result['executable_sha256']=digest(EXE);result['stage']='native';native(result,ledger)
        if not result['gates']['repeatability_1_10']:result['decision']='inconclusive_native_variation_stop_before_training'
        elif not result['gates']['all_operator_medians_14ms']:result['decision']='reject_unchanged_compact_I8_cost_before_teacher_collection_or_training'
        else:result['decision']='eligible_only_for_separately_frozen_teacher_function_fit'
        result['stage']='complete'
    except BaseException as error:
        result.update({'error':repr(error),'decision':'apparatus_or_resource_failure','total_seconds':time.monotonic()-start})
        write(path.with_suffix('.failure.json'),result);raise
    else:
        result['total_seconds']=time.monotonic()-start;write(path,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'rep_medians_ms':[x*1000 for x in result['native']['rep_medians_seconds']],
                          'peak_rss_bytes':result['native']['peak_rss_bytes'],'result_sha256':digest(path),'total_seconds':result['total_seconds']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists();run(args.out)
