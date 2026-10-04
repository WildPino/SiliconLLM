"""Bounded Ling source-sized synthetic active-cost rejection screen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import psutil
import meth324_switch_reference as M
import meth359_switch_cost_topology as A

CPU = M.ROOT/'benchmarks/native_expert_scaling/meth408_ling_additive_cost_cpu.c'
ENGINE = M.ROOT/'benchmarks/phase60/engine.c'
PROTOCOL = M.DOC/'METH_408_LING_ADDITIVE_COST_PROTOCOL_20261004.md'
PRIOR = M.DOC/'meth407_ling_mini_source_headers_result.json'
PRIOR_SHA = '431d9e2978c46b1d938a1efb0fc67ea345b18d01e671baa47604f4b8ddf94a56'
PROFILE = M.DOC/'meth374_switch_physical_workers_contract_result.json'
PROFILE_SHA = '4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1'
OUT = M.ROOT/'results/native_expert_scaling/meth408_ling_additive_cost'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4<<20):h.update(block)
    return h.hexdigest()


def placement(row, threads):
    assert row['worker_physical_cores']==6
    assert [v['slot'] for v in row['worker_affinity']]==list(range(threads))
    assert [v['actual_mask'] for v in row['worker_affinity']]==[1<<(2*i) for i in range(threads)]
    assert len({v['windows_thread_id'] for v in row['worker_affinity']})==threads
    assert all(v['group']==0 for v in row['worker_affinity'])
    assert {v['slot'] for v in row['worker_binding_events']}==set(range(threads))
    assert all(v['group']==0 and v['actual_mask']==1<<(2*v['slot']) for v in row['worker_binding_events'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();peak=0;stage='bindings'
    result={'experiment':'METH-408-Ling-full-width-synthetic-additive-active-cost','runs':[]}

    def guard(child=None):
        nonlocal peak
        rss=psutil.Process().memory_info().rss
        if child is not None and child.poll() is None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        peak=max(peak,rss);assert peak<=6<<30 and time.monotonic()-start<=600,'main_10min_6GiB'

    try:
        helpers=[Path(__file__),CPU,ENGINE,PROTOCOL,PRIOR,PROFILE,Path(M.__file__),Path(A.__file__),
                 CPU.parent/'meth388_switch_thread_binding.h',CPU.parent/'meth318_additive_i8_cpu.c',
                 M.ROOT/'benchmarks/phase60/strat01_q6k_q8k_avx2.h',M.ROOT/'benchmarks/phase60/strat01_q4k_q8k.h',
                 M.ROOT/'benchmarks/phase60/strat01_f32_dot_reference_generic.h']
        result['bindings']={}
        for p in helpers:M.committed(p);result['bindings'][str(p)]=sha(p)
        assert sha(PRIOR)==PRIOR_SHA and sha(PROFILE)==PROFILE_SHA
        old=(CPU.parent/'meth318_additive_i8_cpu.c').read_text(encoding='utf-8');new=CPU.read_text(encoding='utf-8')
        def region(text,begin,end):return text[text.index(begin):text.index(end)]
        make=region(old,'static int8_t palette_value','static void initialize')
        assert region(new,'static int8_t palette_value','static int quantize')==make
        quant=region(old,'static int quantize','static void prepare')
        current=region(new,'static int quantize','static void initialize').replace('\n    worker_bind();','')
        current=current.replace('executed_coeff+=(uint64_t)m->d*m->o*m->nactive;executed_scales+=(uint64_t)m->o*m->nactive*4;executed_palettes+=(uint64_t)m->nactive*4096;','')
        assert current==quant,'318_quant_decode_integer_row_math_exact'
        marker=b'#elif defined(SILICON_BLOCKED_INTEGER_LUT_PREFLIGHT)';engine=ENGINE.read_bytes()
        old_engine=subprocess.check_output(['git','show','02afce0:benchmarks/phase60/engine.c'])
        assert engine[engine.index(marker):].replace(marker,b'#ifdef SILICON_BLOCKED_INTEGER_LUT_PREFLIGHT',1)==old_engine
        marker=b'#elif defined(SILICON_COMPLETE_I16_NATIVE_GENERATE)'
        assert engine[engine.index(marker):].replace(marker,b'#ifdef SILICON_COMPLETE_I16_NATIVE_GENERATE',1)==subprocess.check_output(['git','show','0ff9705:benchmarks/phase60/engine.c'])
        prior=json.loads(PRIOR.read_text(encoding='utf-8'));profile=json.loads(PROFILE.read_text(encoding='utf-8'))
        assert all(prior['gates'].values()) and all(profile['gates'].values())
        ledger=prior['analysis']['full_width_additive_decode_ledger'];assert ledger['addressed_weight_descriptor_bytes']==503905280
        stored=prior['analysis']['hypothetical_main_storage'];assert stored['total_bytes']==4969196544
        result['ledger']={'source407':ledger,'stored407':stored,'expected_allocated_dynamic_bytes':stored['total_bytes']-stored['router_and_control_F32_bytes']+ledger['row_scale_bytes'],
                          'expected_integer_rows':8*(40+3+19*(3*8+3)), 'expected_bank_edge_checks':2*(100+19*3*256)}
        topology=A.physical_topology();assert topology==profile['fresh_topology'];result['fresh_topology']=topology
        ancestors={p.pid for p in psutil.Process().parents()}|{os.getpid()}
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid in ancestors:continue
            name=(process.info['name'] or '').lower();argv=process.info['cmdline'] or []
            unrelated=(name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve())
            if unrelated:result.setdefault('unrelated_background_processes',[]).append({'pid':process.pid,'argv':argv});continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_model_job',process.pid,name)
        OUT.mkdir(parents=True);compiler=Path(profile['compile']['argv'][0]);binary=OUT/'meth408_ling_additive_cost.exe'
        assert sha(compiler)==profile['compile']['compiler_sha256']
        dll=compiler.parent/'libomp.dll';assert sha(dll)==profile['compile']['runtime_sha256'];shutil.copyfile(dll,OUT/'libomp.dll')
        argv=[str(compiler),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',
              '-DSILICON_LING_ADDITIVE_COST_PREFLIGHT',str(ENGINE),'-o',str(binary),'-lm','-lpsapi','-lbcrypt']
        stage='compile';compiled=subprocess.run(argv,capture_output=True,timeout=120)
        result['compile']={'argv':argv,'returncode':compiled.returncode,'stdout':compiled.stdout.decode(errors='replace'),'stderr':compiled.stderr.decode(errors='replace'),
                           'compiler_sha256':sha(compiler),'runtime_sha256':sha(OUT/'libomp.dll')}
        assert compiled.returncode==0,result['compile']['stderr'];result['compile']['binary_sha256']=sha(binary)
        env={k:v for k,v in os.environ.items() if not k.startswith(('OMP_','KMP_','GOMP_','SILICON_WORKER_BINDING_'))}
        env.update(profile['runtime_environment']);result['runtime_environment']=profile['runtime_environment']

        def invoke(threads,label,negative=False):
            prefix=OUT/label;child_env=dict(env);child_env['OMP_NUM_THREADS']=str(threads)
            if negative:child_env['SILICON_WORKER_BINDING_FAULT']='1'
            command=[str(binary),str(threads),str(prefix)];row={'argv':command,'negative':negative,'threads':threads};result['runs'].append(row)
            before=time.monotonic()
            with (OUT/(label+'.stdout.log')).open('xb') as stdout,(OUT/(label+'.stderr.log')).open('xb') as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=child_env)
                try:
                    while child.poll() is None:
                        guard(child);assert time.monotonic()-before<=120,'native_2min';time.sleep(.1)
                    child.wait()
                except BaseException:
                    if child.poll() is None:child.kill();child.wait()
                    raise
            row.update({'returncode':child.returncode,'seconds':time.monotonic()-before,'stdout_sha256':sha(OUT/(label+'.stdout.log')),'stderr_sha256':sha(OUT/(label+'.stderr.log'))})
            if negative:
                assert child.returncode==2 and b'worker_affinity_readback' in (OUT/(label+'.stderr.log')).read_bytes();return row
            assert child.returncode==0,(label,(OUT/(label+'.stderr.log')).read_text(encoding='utf-8'))
            events=[json.loads(line) for line in (OUT/(label+'.stdout.log')).read_text(encoding='utf-8').splitlines()]
            def one(name):
                values=[v for v in events if v['event']==name];assert len(values)==1;return values[0]
            ready,checks,finish=one('ready'),one('selftest'),one('finished');obs=[v for v in events if v['event']=='operator'];assert len(events)==33 and len(obs)==30
            placement(ready,threads);placement(finish,threads)
            assert ready['threads']==threads and ready['all_values_synthetic'] and ready['layers']==20 and ready['dimension']==2048 and ready['stored_banks']==256 and ready['selected']==8
            assert ready['allocated_dynamic_bytes']==result['ledger']['expected_allocated_dynamic_bytes']
            assert ready['active_coefficients']==779091968 and ready['active_scales']==2560000 and ready['active_palettes']==2277376
            assert ready['stored_coded_coefficients']==4*stored['encoded_matrix_code_bytes']
            assert checks['exact_integer_rows']==result['ledger']['expected_integer_rows']
            assert checks['stored_bank_offset_checks']==checks['palette_bank_offset_checks']==result['ledger']['expected_bank_edge_checks']
            assert checks['integer_relative_l2']<=1e-6 and checks['synthetic_head64_relative_l2']<=1e-5 and checks['reference_route_layers']==19
            for key in ('bad_bank_rejected','packed_head_change_detected','integer_extrema_and_quantizer_checks','row_bias_error_detected',
                        'palette_fault_detected','code_fault_detected','S8_saturation_fault_detected','embedding2048_exact'):assert checks[key]
            assert checks['exhaustive_decoded_input_cases']==32131 and checks['tile_lane_cases']==128524 and checks['mixed_adjacent_pairs']==14641
            hashes=[];medians=[]
            for rep in range(3):
                values=[v for v in obs if v['rep']==rep];assert [v['input'] for v in values]==list(range(10))
                assert all(v['warmup']==(v['input']<2) and v['executed_coefficients']==779091968 and v['addressed_descriptor_bytes']==503905280 and 0<v['seconds']<120 for v in values)
                hashes.append([v['complete_output_sha256'] for v in values]);medians.append(statistics.median(v['seconds'] for v in values if not v['warmup']))
            assert hashes[0]==hashes[1]==hashes[2]
            row['outputs']=[]
            for token in range(10):
                path=Path(str(prefix)+f'.input{token}.bin');assert sha(path)==hashes[0][token]
                assert path.read_bytes()[:8]==b'S408OUT1';row['outputs'].append({'input':token,'path':str(path),'sha256':hashes[0][token]})
            assert 8<=finish['minimum_selected_union']<=finish['maximum_selected_union']<=80
            row.update({'ready':ready,'checks':checks,'finished':finish,'operators':obs,'rep_medians_seconds':medians,
                        'fixed_input_medians_seconds':[statistics.median(v['seconds'] for v in obs if v['input']==i) for i in range(2,10)]})
            print(json.dumps({'run':label,'threads':threads,'rep_medians_ms':[v*1000 for v in medians],'peak_rss_bytes':finish['peak_rss_bytes']}),flush=True)
            return row

        stage='negative_worker';invoke(3,'negative_worker',True)
        for number,threads in enumerate((3,6,6,3,3,6)):
            stage=f'run{number}_t{threads}';run=invoke(threads,stage)
            if number:
                first=result['runs'][1]
                for original,current in zip(first['outputs'],run['outputs']):
                    assert original['sha256']==current['sha256'] and Path(original['path']).read_bytes()==Path(current['path']).read_bytes()
        result['profiles']=[]
        for threads in (3,6):
            runs=[r for r in result['runs'] if not r['negative'] and r['threads']==threads];assert len(runs)==3
            medians=[v for r in runs for v in r['rep_medians_seconds']];fixed=[v for r in runs for v in r['fixed_input_medians_seconds']]
            result['profiles'].append({'threads':threads,'all9_rep_medians_seconds':medians,'max_over_min':max(medians)/min(medians),
                                       'repeatability_1_10':max(medians)/min(medians)<=1.10,'every_rep_and_fixed_input_median14ms':max(medians+fixed)<=.014})
        result['gates']={'exact_reused318_integer_primitives_and_old_engine_paths':True,'full407_stored_active_descriptor_reconciled':True,
                         'all_scalar_head_route_bank_quantizer_and_negative_controls':True,'negative_actual_worker_fault_detected':True,
                         'all_actual_3_and6_worker_process_event_readbacks':True,'ALL180_complete_output_SHA_match60_archives':True,
                         'all60_archived_complete_outputs_byte_exact_across_process_profiles':True}
        eligible=any(p['repeatability_1_10'] and p['every_rep_and_fixed_input_median14ms'] for p in result['profiles'])
        result['decision']='eligible_only_for_source_reference_and_representation_quality_protocol' if eligible else 'reject_this_full_width_additive_format_before_Ling_acquisition_or_book_fitting'
        result['resource']={'main_seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':peak,'new_output_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())}
        assert result['resource']['new_output_bytes']<=2<<30;guard()
        result['scope']='ALL values synthetic, exact407 source-sized shapes/counts. Complete coded attention projections/shared/dense/routed matrices plus Q6 head/F32 group router/norms/BF16 row. Input-ready independent layer/context fixtures; QK norm executed, no causal attention/RoPE/KV/cache/composition/prefill/tokenization/generation/task/original accuracy. Timed operator stream lower-bound rejection screen, not50 accepted tokens/s. SHA/archive full operator outputs are not pretrained-model predictions. No learned codebooks/useful extra n/physical DRAM/source values/network/GPU/T4.'
        M.write(args.out,result);print(json.dumps({'sha256':sha(args.out),'decision':result['decision'],'profiles':result['profiles'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':peak})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
