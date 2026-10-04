"""M412: bounded actual-Granite-geometry synthetic I8/A16 cost screen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import time
import psutil
import meth324_switch_reference as M
import meth359_switch_cost_topology as A

CPU = M.ROOT/'benchmarks/native_expert_scaling/meth412_granite_i8_cost_cpu.c'
ENGINE = M.ROOT/'benchmarks/phase60/engine.c'
PROTOCOL = M.DOC/'METH_412_GRANITE_I8_COST_PROTOCOL_20261004.md'
PRIOR = M.DOC/'meth411_granite_moe_source_headers_result.json'
PRIOR_SHA = 'a3260f95f5d93b4fd457487dda9a586d1c7dbc0d50596176f84042d4e0cb76bc'
PROFILE = M.DOC/'meth374_switch_physical_workers_contract_result.json'
PROFILE_SHA = '4601be80b787341f6d60e01f7219c1cd4cd71c0451f835d25e31ded3c468f4b1'
OUT = M.ROOT/'results/native_expert_scaling/meth412_granite_i8_cost'


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


def archive(path, token):
    data=path.read_bytes();assert len(data)==3515716 and data[:8]==b'S412OUT1'
    assert struct.unpack_from('<7I',data,8)==(token,1024,24,32,8,49152,512)
    pos=36+1024*4
    shapes=((1024,1024,1,1,1),(1024,512,1,1,1),(1024,512,1,1,1),
            (1024,1024,1,1,1),(1024,1024,32,8,1),(512,1024,32,8,8))
    def matrix(shape):
        nonlocal pos
        assert struct.unpack_from('<5I',data,pos)==shape;pos+=20
        ids=struct.unpack_from('<'+'I'*shape[3],data,pos);pos+=shape[3]*4
        assert list(ids)==sorted(set(ids)) and all(v<shape[2] for v in ids)
        pos+=shape[1]*shape[3]*4
        return ids
    def activation(n):
        nonlocal pos
        assert struct.unpack_from('<I',data,pos)[0]==n;pos+=8+n*2
    for layer in range(24):
        ids=[matrix(shape) for shape in shapes]
        assert ids[:4]==[(0,)]*4 and ids[4]==ids[5]
        pos+=(7*1024+8*512)*4
        assert struct.unpack_from('<8I',data,pos)==ids[4];pos+=(8+32+8)*4
        for _ in range(3):activation(1024)
        for _ in range(8):activation(512)
    pos+=2*1024*4;activation(1024);assert matrix((1024,49152,1,1,1))==(0,)
    assert pos==len(data),'all_archive_bytes_parsed_no_trailing_or_missing_outputs'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();peak=0;stage='bindings'
    result={'experiment':'METH-412-Granite-source0-full-row-I8-A16-synthetic-cost','runs':[]}

    def guard(child=None):
        nonlocal peak
        rss=psutil.Process().memory_info().rss
        if child is not None and child.poll() is None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        peak=max(peak,rss);assert peak<=6<<30 and time.monotonic()-start<=600,'main_10min_6GiB'
        if OUT.exists():assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<=2<<30,'output_2GiB'

    def idle():
        ancestors={p.pid for p in psutil.Process().parents()}|{os.getpid()}
        allowed=[]
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid in ancestors:continue
            name=(process.info['name'] or '').lower();argv=process.info['cmdline'] or []
            unrelated=(name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve())
            if unrelated:allowed.append({'pid':process.pid,'argv':argv});continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_model_job',process.pid,name)
        return allowed

    try:
        helpers=[Path(__file__),CPU,ENGINE,PROTOCOL,PRIOR,PROFILE,Path(M.__file__),Path(A.__file__),
                 CPU.parent/'meth388_switch_thread_binding.h',CPU.parent/'meth388_switch_three_workers.c',
                 M.ROOT/'benchmarks/phase60/strat01_f32_dot_reference_generic.h']
        result['bindings']={}
        for p in helpers:M.committed(p);result['bindings'][str(p)]=sha(p)
        result['frozen_HEAD']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        assert sha(PRIOR)==PRIOR_SHA and sha(PROFILE)==PROFILE_SHA
        old=(CPU.parent/'meth388_switch_three_workers.c').read_text(encoding='utf-8');new=CPU.read_text(encoding='utf-8')
        begin='/* For cols<=4096 each I32 lane';end='static void head_mv('
        region=old[old.index(begin):old.index(end)]
        current=new[new.index(begin):new.index('static int8_t expected_code(')]
        assert current==region,'388_exact_A16_quantizer_and_I64_global_integer_dot'
        marker=b'#elif defined(SILICON_LING_STATIC_PAIR_LUT_PREFLIGHT)';engine=ENGINE.read_bytes()
        assert engine[engine.index(marker):].replace(marker,b'#ifdef SILICON_LING_STATIC_PAIR_LUT_PREFLIGHT',1)==subprocess.check_output(['git','show','3f87d37:benchmarks/phase60/engine.c'])
        marker=b'#elif defined(SILICON_COMPLETE_I16_NATIVE_GENERATE)'
        assert engine[engine.index(marker):].replace(marker,b'#ifdef SILICON_COMPLETE_I16_NATIVE_GENERATE',1)==subprocess.check_output(['git','show','0ff9705:benchmarks/phase60/engine.c'])
        prior=json.loads(PRIOR.read_text(encoding='utf-8'));profile=json.loads(PROFILE.read_text(encoding='utf-8'))
        assert all(prior['gates'].values()) and all(profile['gates'].values())
        source=prior['models'][0];analysis=source['analysis'];assert analysis['config_geometry']==[1024,512,32,8,24,16,8,49152]
        result['source_assets']=[]
        for item in source['assets']:
            assert sha(item['path'])==item['sha256'];result['source_assets'].append(item)
        for item in source['shards']:
            assert sha(item['header_path'])==item['header_sha256'];result['source_assets'].append({'path':item['header_path'],'sha256':item['header_sha256']})
        for item in prior['reference_modules']:
            assert sha(item['path'])==sha(item['stored_copy'])==item['sha256'];result['source_assets'].append(item)
        ledger=analysis['hypothetical_row_I8_active'];stored=analysis['hypothetical_row_I8_storage']
        assert ledger['descriptor_bytes']==433231872 and stored['total_bytes']==1444581376
        result['ledger']={'source411':ledger,'stored411':stored,
            'expected_allocated_dynamic_bytes':stored['total_bytes']-stored['router_and_norm_F32_bytes']+ledger['coded_row_F32_scale_bytes']+ledger['head_row_F32_scale_bytes'],
            'expected_integer_rows':8*(24*(4+2*8)+1),'expected_bank_edge_checks':2*(24*(4+2*32)+1),
            'expected_archive_bytes':3515716}
        assert result['ledger']['expected_allocated_dynamic_bytes']==1443299328
        topology=A.physical_topology();assert topology==profile['fresh_topology'];result['fresh_topology']=topology
        result['unrelated_background_processes']=idle()
        OUT.mkdir(parents=True);compiler=Path(profile['compile']['argv'][0]);binary=OUT/'meth412_granite_i8_cost.exe'
        assert sha(compiler)==profile['compile']['compiler_sha256']
        dll=compiler.parent/'libomp.dll';assert sha(dll)==profile['compile']['runtime_sha256'];shutil.copyfile(dll,OUT/'libomp.dll')
        argv=[str(compiler),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',
              '-DSILICON_GRANITE_I8_COST_PREFLIGHT',str(ENGINE),'-o',str(binary),'-lm','-lpsapi','-lbcrypt']
        stage='compile';compiled=subprocess.run(argv,capture_output=True,timeout=120)
        result['compile']={'argv':argv,'returncode':compiled.returncode,'stdout':compiled.stdout.decode(errors='replace'),'stderr':compiled.stderr.decode(errors='replace'),
                           'compiler_sha256':sha(compiler),'runtime_sha256':sha(OUT/'libomp.dll')}
        assert compiled.returncode==0,result['compile']['stderr'];result['compile']['binary_sha256']=sha(binary)
        env={k:v for k,v in os.environ.items() if not k.startswith(('OMP_','KMP_','GOMP_','SILICON_WORKER_BINDING_'))}
        env.update(profile['runtime_environment']);result['runtime_environment']=profile['runtime_environment']

        def invoke(threads,label,negative=False):
            idle();prefix=OUT/label;child_env=dict(env);child_env['OMP_NUM_THREADS']=str(threads)
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
            assert ready['threads']==threads and ready['all_values_synthetic'] and ready['layers']==24 and ready['dimension']==1024 and ready['stored_banks']==32 and ready['selected']==8
            assert ready['allocated_dynamic_bytes']==result['ledger']['expected_allocated_dynamic_bytes']
            assert ready['active_coefficients']==427819008 and ready['active_scales']==2064384
            assert ready['stored_coded_coefficients']==stored['coded_core_and_all_expert_I8_bytes']+stored['head_I8_bytes']
            assert checks['exact_integer_rows']==result['ledger']['expected_integer_rows']
            assert checks['stored_bank_offset_checks']==result['ledger']['expected_bank_edge_checks']
            assert checks['integer_relative_l2']<=1e-6 and checks['synthetic_head64_relative_l2']<=1e-6 and checks['reference_route_layers']==24
            assert checks['all_weight_representative_A16_cases']==2805
            for key in ('bad_bank_rejected','head_bit_change_detected','code_fault_detected','row_scale_fault_detected',
                        'I32_global_truncation_detected','zero_ties_nonfinite_quantizer','scalar_all_fixture_activation_codes',
                        'router_ties_and_sorted_softmax','independent_norm_SiLU_residual_controls','BF16_lookup1024_times12_exact'):assert checks[key]
            hashes=[];medians=[]
            for rep in range(3):
                values=[v for v in obs if v['rep']==rep];assert [v['input'] for v in values]==list(range(10))
                assert all(v['warmup']==(v['input']<2) and v['executed_coefficients']==427819008 and v['weight_descriptor_bytes']==433231872 and 0<v['seconds']<120 for v in values)
                hashes.append([v['complete_output_sha256'] for v in values]);medians.append(statistics.median(v['seconds'] for v in values if not v['warmup']))
            assert hashes[0]==hashes[1]==hashes[2]
            row['outputs']=[]
            for token in range(10):
                path=Path(str(prefix)+f'.input{token}.bin');assert sha(path)==hashes[0][token];archive(path,token)
                row['outputs'].append({'input':token,'path':str(path),'bytes':path.stat().st_size,'sha256':hashes[0][token]})
            assert 8<=finish['minimum_selected_union']<=finish['maximum_selected_union']<=32
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
        result['gates']={'exact_reused388_A16_I64_primitives_and_old_engine_paths':True,'full411_stored_active_descriptor_reconciled':True,
                         'all_scalar_head_route_bank_quantizer_norm_nonlinear_and_negative_controls':True,'negative_actual_worker_fault_detected':True,
                         'all_actual_3_and6_worker_process_event_readbacks':True,'ALL180_complete_output_SHA_match60_archives':True,
                         'all60_archived_complete_outputs_byte_exact_across_process_profiles':True,'all_archive_lengths_shapes_ids_and_inputs_independently_parsed':True}
        eligible=any(p['repeatability_1_10'] and p['every_rep_and_fixed_input_median14ms'] for p in result['profiles'])
        result['decision']='eligible_only_for_source_acquisition_reference_and_NEW_quality_protocol' if eligible else 'reject_this_complete_row_I8_cost_profile_before_Granite_source_values'
        result['resource']={'main_seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':peak,'new_output_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())}
        guard();result['scope']='ALL values synthetic, actual411 source0 geometry. ALL24 complete I8 attention projections and selected8 packed gate/up/down matrices, full49152x1024 I8 head, full32-score F32 router, norms, A16 quantization, SiLU/mixtures/residuals, BF16 lookup*12, residual*.22 and head/6. Independent layer/context fixtures, F64 norm reduction and declared lower-ID ties; no original Torch/backend parity. Causal attention/attention scale .015625/RoPE/GQA execution/KV/cache/layer composition/prefill/tokenization/generation/task/source values/source quality/physical DRAM are OMITTED. Conservative full weight descriptor is not physical traffic. Lower-bound cost screen only, not accepted tokens/s or useful32/>256/10x/RAM-scale functions. No network/training/GPU/T4.'
        M.write(args.out,result);print(json.dumps({'sha256':sha(args.out),'decision':result['decision'],'profiles':result['profiles'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':peak})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
