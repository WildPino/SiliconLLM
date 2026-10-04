"""Exact additive I8 representation via I32 activation tables: numeric/cost only."""
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
import meth303_compact_i8_preflight as M
import meth299_gigachat_block_selection as P
import meth324_switch_reference as B

ROOT=M.ROOT;DOC=M.DOC
CPU=ROOT/'benchmarks/native_expert_scaling/meth399_integer_lut_cost_profile_cpu.c'
PROTOCOL=DOC/'METH_399_INTEGER_LUT_COST_PROFILE_PROTOCOL_20261004.md'
PRIOR=DOC/'meth317_gigachat_additive_descriptor_result.json'
OUT=M.OUT/'meth399_additive_i8';SPEC=OUT/'meth399_spec.bin';EXE=OUT/'meth399_additive_integer_lut_cpu.exe'
FAILURE=DOC/'meth318_additive_i8_preflight_result.failure.json'
FAILURE_SHA='d51677076dbc21aa94a3409c6f576fa5a8a255c61809056acdacca03bacd8830'
PREVIOUS=DOC/'meth398_additive_integer_lut_result.json'
PREVIOUS_SHA='44647e766b0918504ff3d3209f88bd6254d452c979bce9079296d81aeb7beb48'
LUT_CAPACITY=32*8960*256
PRIOR_SHA='93e778d724490138ce1b187f9f887c8e86d78778d6d0c67a15d1ace71edc2e83'
CONTROLLER303_SHA='3b0300ada9b645fbf4576dff301cfa93bc0635ab8254e66d080de83570ef4765'

def ledger(prior):
    rows=[v for v in prior['tensor_rows'] if v['organ'] in ('mla','dense0','routed','shared')]
    active=sum(v['active_coefficients'] for v in rows)
    scales=sum(v['active_scale_bytes'] for v in rows)
    palettes=sum(v['active_palette_bytes'] for v in rows)
    stored=sum(v['stored_code_bytes']*4 for v in rows)
    expected_allocated=sum(v['stored_code_bytes']+v['stored_scale_bytes']+v['stored_palette_bytes']+v['active_scale_bytes'] for v in rows)+161602560+128256*1536*2
    assert len(rows)==283 and active==1428619264 and scales==5327360 and palettes==8683520
    return {'matrices':rows,'active_coefficients':active,'active_code_bytes':active//4,
        'active_scale_bytes':scales,'active_palette_bytes':palettes,'stored_coefficients':stored,
        'allocated_dynamic_bytes':expected_allocated,'exact_sampled_integer_rows':8*sum(v['active_banks'] for v in rows),
        'bank_edge_checks':2*sum(v['stored_banks'] for v in rows),
        'complete_addressed_weight_bytes':prior['scenarios'][0]['full_flat_router_active_weight_bytes']}

def make_spec():
    # Reuse fresh source-Q6/router/norm extraction without changing immutable303.
    M.SPEC=OUT/'source303_spec_binding.bin'
    original,binding=M.make_spec()
    raw=M.SPEC.read_bytes()
    assert len(raw)==2000 and raw[:8]==b'M303SPC1'
    pieces=[b'M318SPC1',struct.pack('<6I',26,64,1,1280,1280,32),raw[32:]]
    _,tensors,base=M.source.header(M.source.SOURCE,21356264448)
    assert base['header_sha256']=='2e18041f5c90d897f4ab3f882887c63b797f736fec63b1147c8cd9e0d1bcbe08'
    item=tensors['token_embd.weight'];assert item['shape']==(1536,128256) and item['type']=='BF16'
    offset=base['header_bytes_read']+item['offset']
    with M.source.SOURCE.open('rb') as stream:
        stream.seek(offset);data=stream.read(item['bytes']);assert len(data)==item['bytes']
    digest=hashlib.sha256(data).digest();del data
    pieces.extend([struct.pack('<2Q',offset,item['bytes']),digest])
    with SPEC.open('xb') as stream:stream.write(b''.join(pieces))
    base.update({'embedding':{'offset':offset,'bytes':item['bytes'],'sha256':digest.hex()},
                 'previous_full_sha256_not_rechecked':'fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47'})
    return {'path':str(SPEC),'bytes':SPEC.stat().st_size,'sha256':M.digest(SPEC),'source303_spec':original},binding,base

def native(result,book,number,start):
    assert psutil.virtual_memory().available>=12*(1<<30)
    log=OUT/f'meth399_native_run{number}.jsonl';err=OUT/f'meth399_native_run{number}.stderr.log'
    assert not log.exists() and not err.exists()
    command=[str(EXE),str(SPEC),str(M.MODEL),str(M.source.SOURCE)]
    env=os.environ.copy();env['PATH']=str(M.TOOLCHAIN/'bin')+os.pathsep+env.get('PATH','')
    for key in list(env):
        if key.startswith(('OMP_','KMP_','GOMP_','SILICON_WORKER_BINDING_')):env.pop(key)
    env.update({'OMP_NUM_THREADS':'6','OMP_DYNAMIC':'FALSE','OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
    run={'OMP_PROC_BIND_absent':'OMP_PROC_BIND' not in env,'command':command,'log_path':str(log),'stderr_path':str(err),'peak_sampled_rss_bytes':0,'all_inherited_OMP_KMP_GOMP_binding_options_removed':True,
         'environment':{key:env[key] for key in ('OMP_NUM_THREADS','OMP_DYNAMIC','OMP_WAIT_POLICY','KMP_AFFINITY')}}
    result['native_runs'].append(run);run_start=time.monotonic()
    with log.open('x',encoding='utf-8') as out,err.open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=stderr);run['pid']=process.pid;child=psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:run['peak_sampled_rss_bytes']=max(run['peak_sampled_rss_bytes'],child.memory_info().rss)
                except psutil.NoSuchProcess:pass
                if time.monotonic()-start>1200 or run['peak_sampled_rss_bytes']>12*(1<<30):raise RuntimeError('native total20min/12GiB stop')
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        finally:
            if process.poll() is None:process.kill();process.wait()
            run.update({'exit_code':process.returncode,'seconds':time.monotonic()-run_start})
    run.update({'log_sha256':M.digest(log),'stderr_sha256':M.digest(err)})
    assert process.returncode==0,err.read_text(encoding='utf-8')[-2000:]
    events=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    def one(name):
        items=[v for v in events if v['event']==name];assert len(items)==1;return items[0]
    ready,checks,finish=one('ready'),one('selftest'),one('finished');obs=[v for v in events if v['event']=='operator']
    run.update({'ready':ready,'selftest':checks,'finished':finish,'operators':obs})
    assert len(events)==64 and len(obs)==30
    profiles=[v for v in events if v['event']=='cost_profile'];assert len(profiles)==30
    run['profiles']=profiles
    for v,w in zip(obs,profiles):
        assert (v['rep'],v['input'])==(w['rep'],w['input'])
        assert w['calls']==[130,75,3,75]
        assert all(y>0 for k in ('builder_seconds','row_seconds') for y in w[k])
        assert w['router_seconds']>0 and w['head_seconds']>0
        accounted=sum(w['builder_seconds'])+sum(w['row_seconds'])+w['router_seconds']+w['head_seconds']
        assert 0<accounted<=v['seconds']+1e-8
        w['other_seconds']=v['seconds']-accounted
        assert w['other_seconds']>=-1e-8
    lut=one('integer_lut_controls');run['integer_lut_controls']=lut
    assert lut['exact_pair_group_cases']==262144 and lut['MAX_INPUT_extrema'] and lut['table_fault_detected']
    assert ready['parent_count']==ready['functions_per_layer']==64 and ready['children_per_parent']==1
    assert ready['threads']==6 and ready['code_bits']==2 and ready['row_tile']==4
    assert ready['active_coefficients']==book['active_coefficients'] and ready['active_palette_bytes']==book['active_palette_bytes']
    assert ready['active_row_scale_bytes']==book['active_scale_bytes'] and ready['stored_coefficients']==book['stored_coefficients']
    assert ready['allocated_bytes']==book['allocated_dynamic_bytes']+LUT_CAPACITY
    assert checks['exact_integer_rows']==book['exact_sampled_integer_rows']
    assert checks['stored_bank_offset_checks']==checks['palette_bank_offset_checks']==book['bank_edge_checks']
    assert checks['integer_relative_l2']<=1e-6 and checks['real_head64_relative_l2']<=1e-5 and checks['reference_route_layers']==25
    for key in ('bad_bank_rejected','packed_head_change_detected','integer_extrema_and_quantizer_checks',
        'row_bias_error_detected','palette_fault_detected','code_fault_detected','S8_saturation_fault_detected','embedding1536_exact'):assert checks[key]
    assert checks['exhaustive_decoded_input_cases']==32131 and checks['tile_lane_cases']==128524 and checks['mixed_adjacent_pairs']==14641
    hashes=[];medians=[]
    for rep in range(3):
        selected=[v for v in obs if v['rep']==rep];assert [v['input'] for v in selected]==list(range(10))
        assert all(v['warmup']==(v['input']<2) and 0<v['seconds']<1200 for v in selected)
        hashes.append([v['output_route_hash'] for v in selected]);medians.append(statistics.median(v['seconds'] for v in selected if not v['warmup']))
    assert hashes[0]==hashes[1]==hashes[2]
    assert 1<=finish['minimum_selected_child_union']<=finish['maximum_selected_child_union']<=40
    run.update({'rep_medians_seconds':medians,'within_max_over_min':max(medians)/min(medians),
        'fixed_input_medians_seconds':[statistics.median(v['seconds'] for v in obs if v['input']==i) for i in range(2,10)],
        'peak_rss_bytes':max(run['peak_sampled_rss_bytes'],finish['peak_rss_bytes'])})
    assert run['peak_rss_bytes']<=12*(1<<30)
    print(json.dumps({'completed_process':number,'rep_medians_ms':[v*1000 for v in medians],'peak_rss_bytes':run['peak_rss_bytes']}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=time.monotonic();stage='bindings';result={'experiment':'METH-399-same398-integer-LUT-cost-attribution','native_runs':[]}
    try:
        for path in (Path(__file__),CPU,PROTOCOL,M.ENGINE,PRIOR,FAILURE,PREVIOUS,Path(__file__).with_name('meth319_additive_i8_execution_repair.py'),Path(__file__).with_name('meth398_additive_integer_lut_preflight.py'),DOC/'METH_398_ADDITIVE_INTEGER_LUT_PROTOCOL_20261004.md'):B.committed(path)
        assert M.digest(FAILURE)==FAILURE_SHA and M.digest(PREVIOUS)==PREVIOUS_SHA
        previous=json.loads(PREVIOUS.read_text(encoding='utf-8'))
        for path in (Path(M.__file__),Path(M.source.__file__),Path(B.__file__),ROOT/'benchmarks/native_expert_scaling/meth318_additive_i8_cpu.c'):B.committed(path)
        base=ROOT/'benchmarks/native_expert_scaling/meth398_additive_integer_lut_cpu.c';B.committed(base)
        assert M.digest(base)==previous['cpu_source_sha256']
        import re
        assert re.sub(r'/\* M399_BEGIN \*/.*?/\* M399_END \*/','',CPU.read_text(encoding='utf-8'),flags=re.S)==base.read_text(encoding='utf-8')
        failed=json.loads(FAILURE.read_text(encoding='utf-8'))
        result['immutable318_failure_sha256']=FAILURE_SHA
        assert M.digest(PRIOR)==PRIOR_SHA and M.digest(Path(M.__file__))==CONTROLLER303_SHA
        for rel,sha in M.PINS.items():assert M.digest(ROOT/rel)==sha
        # Concrete legacy branch must remain exact; new prefixes alone are permitted.
        original=subprocess.check_output(['git','show','0ff9705:benchmarks/phase60/engine.c'])
        current=M.ENGINE.read_bytes();needle=b'#elif defined(SILICON_COMPLETE_I16_NATIVE_GENERATE)'
        assert current[current.index(needle):].replace(needle,b'#ifdef SILICON_COMPLETE_I16_NATIVE_GENERATE',1)==original
        prior=json.loads(PRIOR.read_text(encoding='utf-8'));book=ledger(prior)
        result.update({'controller_sha256':M.digest(Path(__file__)),'cpu_source_sha256':M.digest(CPU),
            'protocol_sha256':M.digest(PROTOCOL),'engine_sha256':M.digest(M.ENGINE),'prior_sha256':PRIOR_SHA,
            'legacy_default_byte_exact':True,'ledger':book,'gates':{'complete_weight_560mb':book['complete_addressed_weight_bytes']<=560000000},
            'ram_total_bytes':psutil.virtual_memory().total,'cpu_logical_threads':psutil.cpu_count()})
        assert result['gates']['complete_weight_560mb'];OUT.mkdir(exist_ok=True)
        stage='source_and_compile';result['spec'],result['q4_source_binding'],result['bf16_embedding_binding']=make_spec()
        assert result['spec']['sha256']==failed['spec']['sha256']
        assert result['q4_source_binding']==failed['q4_source_binding']
        assert result['bf16_embedding_binding']==failed['bf16_embedding_binding']
        command=[str(M.COMPILER),'-O3','-mavx2','-mssse3','-mfma','-fopenmp','-ffp-contract=off','-std=c11','-DSILICON_ADDITIVE_INTEGER_LUT_COST_PROFILE',str(M.ENGINE),'-o',str(EXE),'-lm','-lpsapi','-lbcrypt']
        assert M.digest(M.COMPILER)==previous['compile']['compiler_sha256']
        libomp=M.TOOLCHAIN/'bin/libomp.dll';assert M.digest(libomp)==previous['compile']['libomp_sha256']
        compiled=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
        result['compile']={'command':command,'exit_code':compiled.returncode,'stdout':compiled.stdout,'stderr':compiled.stderr,'compiler_sha256':M.digest(M.COMPILER),'libomp_sha256':M.digest(libomp)}
        assert compiled.returncode==0,compiled.stderr
        result['executable_sha256']=M.digest(EXE)
        result['exact318_representation_spec_source_match']=True
        result['previous398_raw_sha256']=PREVIOUS_SHA
        result['instrumentation_removed_CPU_exact398']=True
        result['integer_lut_workspace_bytes']=LUT_CAPACITY
        result['integer_lut_logical_writes_per_token_bytes']=sum(r['shape'][0]*r['active_banks']*256 for r in book['matrices'])
        result['integer_lut_logical_gathered_values_per_token']=book['active_coefficients']//4
        for number in (1,2,3):
            stage=f'native_process{number}';native(result,book,number,start)
        runs=result['native_runs'];hashes=[[v['output_route_hash'] for v in run['operators']] for run in runs]
        assert hashes[0]==hashes[1]==hashes[2]
        original_hashes=[[v['output_route_hash'] for v in run['operators']] for run in previous['native_runs']]
        assert hashes==original_hashes, 'all90 exact318 full-operator outputs including source head/router'
        result['gates']['all90_fingerprints_exact398']=True
        medians=[v for run in runs for v in run['rep_medians_seconds']]
        ratio=max(medians)/min(medians);result['all_rep_medians_seconds']=medians;result['cross_max_over_min']=ratio
        result['gates'].update({'all_numeric_route_capacity_controls':True,'all90_output_hashes_repeat_exactly':True,
            'within_and_cross_repeatability_1_10':ratio<=1.10 and all(run['within_max_over_min']<=1.10 for run in runs),
            'every_rep_and_fixed_input_median14ms':max(medians)<=.014 and all(max(run['fixed_input_medians_seconds'])<=.014 for run in runs)})
        component_medians={k:[] for k in ('builder','rows','router','head','other')}
        category_medians={k:[[] for _ in range(4)] for k in ('builder','rows')}
        for run in runs:
            for rep in range(3):
                chosen=[v for v in run['profiles'] if v['rep']==rep and v['input']>=2]
                for name in component_medians:
                    vals=[sum(v['builder_seconds']) if name=='builder' else sum(v['row_seconds']) if name=='rows' else v[name+'_seconds'] for v in chosen]
                    component_medians[name].append(statistics.median(vals))
                for name,field in (('builder','builder_seconds'),('rows','row_seconds')):
                    for cat in range(4):category_medians[name][cat].append(statistics.median(v[field][cat] for v in chosen))
        result['component_rep_medians_seconds']=component_medians
        result['categories']=['mla','routed','dense0','shared'];result['category_rep_medians_seconds']=category_medians
        result['all9_component_medians_over14ms']={k:min(v)>.014 for k,v in component_medians.items()}
        result['gates'].update({'all_profile_calls_and_positive_nested_boundaries':True,'instrumentation_only_source_exact398':True})
        builder_over=result['all9_component_medians_over14ms']['builder'];rows_over=result['all9_component_medians_over14ms']['rows']
        result['decision']='both_builder_and_rows_individually_exceed14ms_require_joint_change' if builder_over and rows_over else 'rows_alone_exceed14ms_no_builder_only_rescue' if rows_over else 'builder_alone_exceeds14ms_no_rows_only_rescue' if builder_over else 'no_individual_component_closure_joint_cost_requires_new_gate'
        result['scope']='SAME398 instrumentation-only diagnostic, 64-bit output fingerprints not exhaustive archived byte comparison. Exact318 source geometry/representation, NEW I32 activation LUT builder/gather kernel; synthetic additive codes/palettes plus actual original Q6 head/F32 router/norm/BF16 embedding. Input-ready per-layer residual/context fixtures, no causal attention/RoPE/KV/composition, learned book quality, useful extra n, physical DRAM or accepted model rate.'
        result['total_seconds']=time.monotonic()-start;assert result['total_seconds']<=1200;M.write(args.out,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'rep_medians_ms':[v*1000 for v in medians],'component_rep_medians_seconds':component_medians,
            'cross_max_over_min':ratio,'result_sha256':M.digest(args.out),'total_seconds':result['total_seconds']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'total_seconds':time.monotonic()-start})
        M.write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
