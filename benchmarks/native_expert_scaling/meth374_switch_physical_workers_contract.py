"""NEW explicit worker placement/active waiting; unchanged356 complete model math."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import psutil
import meth324_switch_reference as M
import meth359_switch_cost_topology as A
import meth372_switch_nested_bank_usefulness as U

OUT=M.ROOT/'results/native_expert_scaling/meth374_switch_physical_workers_contract'
PROTOCOL=M.DOC/'METH_374_SWITCH_PHYSICAL_WORKERS_CONTRACT_PROTOCOL_20261004.md'
ENGINE=M.ROOT/'benchmarks/phase60/engine.c'
BASE=M.ROOT/'benchmarks/native_expert_scaling'
OLDOUT=M.ROOT/'results/native_expert_scaling/meth356_switch_all_a16_contract'
INPUTS={338:('meth338_switch_tensor_recovery_result.json','19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'),
        356:('meth356_switch_all_a16_contract_result.json','ec51e76c08277e4f874cacac9a49d5b60477c0abb6a651cbe648f9bdad3f66d3'),
        362:('meth362_switch_multi_span_manifest.json','c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'),
        363:('meth363_switch_all_a16_multi_span_quality_result.json','ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'),
        371:('meth371_switch_nested_bank_contract_result.json','2d534664bd715c771e62c33fa50f4d85211dc994efefbddaecef0d7e0c80614d')}


def source_identity():
    source=(BASE/'meth374_switch_physical_workers.c').read_text(encoding='utf-8')
    source=source.removeprefix('#define _WIN32_WINNT 0x0601\n').replace('#include "meth374_switch_thread_binding.h"\n','').replace('worker_bind();','')
    # A standalone inserted line has indentation; remove precisely that line.
    source=source.replace('    \n    require(n>0&&n<=4096,"head_integer_bound")','    require(n>0&&n<=4096,"head_integer_bound")')
    assert source.replace('meth374','meth356')==(BASE/'meth356_switch_all_a16.c').read_text(encoding='utf-8'),'projection_math_identity'
    for kind in ('cost_entry','entry'):
        source=(BASE/f'meth374_switch_physical_workers_{kind}.c').read_text(encoding='utf-8')
        source=source.replace('worker_setup(threads);','').replace('worker_json();','')
        if kind=='cost_entry':source=source.replace('printf("]");printf("}\\n");','printf("]}\\n");')
        source=source.replace('meth374_switch_physical_workers','meth374_switch_all_a16').replace('meth374','meth356')
        assert source==(BASE/f'meth356_switch_all_a16_{kind}.c').read_text(encoding='utf-8'),'driver_identity'
    old=subprocess.check_output(['git','show','d4b22a9:benchmarks/phase60/engine.c'],cwd=M.ROOT)
    new=ENGINE.read_bytes();prefix=b'#ifdef SILICON_SWITCH_PHYSICAL_WORKERS\n#include "../native_expert_scaling/meth374_switch_physical_workers_entry.c"\n#elif defined(SILICON_SWITCH_ENCODER_BATCHES)'
    assert new.startswith(prefix) and new.replace(prefix,b'#ifdef SILICON_SWITCH_ENCODER_BATCHES',1)==old,'earlier_engine_paths_identity'
    marker=b'#elif defined(SILICON_COMPLETE_I16_NATIVE_GENERATE)';start=new.index(marker)
    assert new[start:].replace(marker,b'#ifdef SILICON_COMPLETE_I16_NATIVE_GENERATE',1)==subprocess.check_output(['git','show','0ff9705:benchmarks/phase60/engine.c'],cwd=M.ROOT)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    started=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-374-physical-worker-active-wait-complete-contract','commands':[],'tiny':[],'controls':[],'quality_bridges':[]}
    def guard(child=None):
        nonlocal maximum
        rss=psutil.Process().memory_info().rss
        if child is not None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        maximum=max(maximum,rss)
        if rss>16<<30 or time.monotonic()-started>1800:
            if child is not None and child.poll() is None:child.kill();child.wait()
            raise RuntimeError('physical_workers_30min_16GiB')
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        sources=[BASE/f'meth374_switch_physical_workers{suffix}' for suffix in ('.c','_entry.c','_cost_entry.c')]+[BASE/'meth374_switch_thread_binding.h']
        for path in [Path(__file__),PROTOCOL,ENGINE,*sources,Path(M.__file__),Path(A.__file__),Path(U.__file__)]:M.committed(path)
        for kind in ('.c','_entry.c','_cost_entry.c'):M.committed(BASE/('meth356_switch_all_a16'+kind))
        source_identity();records={}
        for n,(name,sha) in INPUTS.items():
            path=M.DOC/name;M.committed(path);assert digest(path)==sha;records[n]=json.loads(path.read_text(encoding='utf-8'));assert all(records[n]['gates'].values())
        topology=A.physical_topology();assert topology==records[371]['fresh_topology']
        masks=topology['selected_one_logical_per_physical_core'];assert masks==[0,2,4,6,8,10]
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid','name','cmdline']):
            cmd=' '.join(process.info['cmdline'] or []).replace('\\','/')
            if process.pid not in own and process.info['name'].lower() in ('python.exe','meth356_switch_all_a16.exe','meth365_switch_encoder_batches.exe','meth374_switch_physical_workers.exe'):
                assert not ('benchmarks/native_expert_scaling/' in cmd or 'switch_' in process.info['name']),'concurrent_worker'
        qualified=records[356];artifact=records[338]['artifact'];payload=Path(artifact['payload']);before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert before[0]==artifact['bytes'] and digest(payload)==artifact['sha256'] and digest(artifact['manifest'])==artifact['manifest_sha256']
        compiler=Path(qualified['compile']['argv'][0]);dll=OLDOUT/'libomp.dll'
        assert digest(compiler)==qualified['compile']['compiler_sha256'] and digest(dll)==qualified['compile']['runtime_sha256']
        result.update({'controller_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'source_sha256':{p.name:digest(p) for p in sources},
                       'engine_sha256':digest(ENGINE),'input_sha256':{str(n):v[1] for n,v in INPUTS.items()},'artifact':artifact,'fresh_topology':topology,
                       'model_and_driver_math_exact356_by_source_reversal':True,'earlier_engine_paths_byte_exact':True,'primary_threads':6,'primary_affinity':masks})
        OUT.mkdir(parents=True);shutil.copyfile(dll,OUT/'libomp.dll');binary=OUT/'meth374_switch_physical_workers.exe'
        argv=[str(compiler),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp','-DSILICON_SWITCH_PHYSICAL_WORKERS',str(ENGINE),'-o',str(binary)]
        stage='compile';compiled=subprocess.run(argv,capture_output=True,timeout=120);(OUT/'compile.stdout.log').write_bytes(compiled.stdout);(OUT/'compile.stderr.log').write_bytes(compiled.stderr)
        assert compiled.returncode==0,compiled.stderr.decode(errors='replace')
        result['compile']={'argv':argv,'binary_sha256':digest(binary),'compiler_sha256':digest(compiler),'runtime_sha256':digest(OUT/'libomp.dll')}
        env={k:v for k,v in os.environ.items() if not k.startswith(('OMP_','KMP_','GOMP_','SILICON_WORKER_BINDING_'))}
        runtime={'OMP_WAIT_POLICY':'ACTIVE','KMP_AFFINITY':'none','KMP_BLOCKTIME':'infinite','OMP_DYNAMIC':'FALSE','OMP_MAX_ACTIVE_LEVELS':'1'};env.update(runtime);result['runtime_environment']=runtime
        def invoke(argv,label,threads=1,negative=False,rows_expected=False):
            child_env=dict(env,OMP_NUM_THREADS=str(threads));affinity=masks[:threads]
            if negative:child_env['SILICON_WORKER_BINDING_FAULT']='1'
            with (OUT/(label+'.stdout.log')).open('wb') as stdout,(OUT/(label+'.stderr.log')).open('wb') as stderr:
                child=subprocess.Popen([str(binary),*map(str,argv)],stdout=stdout,stderr=stderr,env=child_env)
                try:
                    process=psutil.Process(child.pid);process.cpu_affinity(affinity);actual=process.cpu_affinity();assert actual==affinity
                    while child.poll() is None:guard(child);time.sleep(.1)
                except BaseException:
                    if child.poll() is None:child.kill();child.wait()
                    raise
            result['commands'].append({'argv':[str(binary),*map(str,argv)],'returncode':child.returncode,'actual_process_affinity':actual,
                                       'negative_control':negative,'stdout_sha256':digest(OUT/(label+'.stdout.log')),'stderr_sha256':digest(OUT/(label+'.stderr.log'))})
            if negative:
                assert child.returncode==2 and b'worker_affinity_readback' in (OUT/(label+'.stderr.log')).read_bytes();return []
            assert child.returncode==0,(label,child.returncode)
            rows=[json.loads(line) for line in (OUT/(label+'.stdout.log')).read_text(encoding='utf-8').splitlines()] if rows_expected else []
            for row in rows:
                assert row['worker_physical_cores']==6
                assert [v['slot'] for v in row['worker_affinity']]==list(range(threads))
                assert [v['actual_mask'] for v in row['worker_affinity']]==[1<<cpu for cpu in affinity]
                assert all(v['group']==0 for v in row['worker_affinity'])
                assert len({v['windows_thread_id'] for v in row['worker_affinity']})==threads
                assert all(v['actual_mask']==1<<affinity[v['slot']] and v['group']==0 for v in row['worker_binding_events'])
                assert set(v['slot'] for v in row['worker_binding_events'])==set(range(threads))
            return rows
        def execute(spec,source,decoder,label,threads,profile=0,generate=False,cap=64,closing=32095):
            prefix=OUT/label
            if generate:argv=['--generate',spec,','.join(map(str,source)),prefix,threads,cap,closing,profile,1,1,0]
            else:argv=[spec,','.join(map(str,source)),','.join(map(str,decoder)),prefix,threads,profile,1,1,0]
            rows=invoke(argv,label,threads,rows_expected=True);assert [r['repetition'] for r in rows]==[-1,0]
            for row in rows:
                path=Path(str(prefix)+'.'+str(row['repetition'])+'.bin');row['output_sha256']=digest(path)
                if generate:assert row['generated_ids']==U.read_output(path)[2].argmax(-1).tolist()
            assert rows[0]['output_sha256']==rows[1]['output_sha256'];return rows
        stage='negative_worker_placement_control'
        invoke([artifact['manifest'],'2,1','0',OUT/'negative',6,0,0,1,0],'negative',6,negative=True)
        result['negative_worker_placement_detected']=True
        stage='unchanged_integer_primitives';inp=OLDOUT/'integer_cases.bin'
        assert digest(inp)=='e404e9a2501810532630e483450ce2bb3bcf7db7e5100d1a947b49a19ecdeb94'
        result['integer_primitives']=[]
        for old in qualified['integer_primitives']:
            output=OUT/(old['path'][2:]+'.bin');invoke([old['path'],inp,output],old['path'][2:])
            assert digest(output)==old['output_sha256'];result['integer_primitives'].append({'path':old['path'],'all13_full_output_exact356':True,'sha256':digest(output)})
        for tiny in qualified['tiny_cases']:
            cap=tiny['capacity'];stage=f'tiny{cap}';target=tiny['artifact']
            assert digest(target['payload'])==target['payload_sha256'] and digest(target['manifest'])==target['manifest_sha256']
            reference=OLDOUT/f'tiny{cap}.reference.npz';assert digest(reference)==tiny['reference_sha256']
            with np.load(reference) as data:
                expected=tuple(data[k].copy() for k in ('encoder','decoder','logits','routes'))
                natural=tuple(data[k].copy() for k in ('gen_encoder','gen_decoder','gen_logits','gen_routes'))
            faults=[]
            for fault in tiny['faults']:
                number=fault['fault'];output=OUT/f'tiny{cap}.fault{number}.bin'
                invoke([target['manifest'],'2,3,4,5,6,7','0,8,9,10',output,number],f'tiny{cap}.fault{number}')
                logits=U.read_output(output)[2];error=float(np.linalg.norm(logits.astype(np.float64)-expected[2])/max(float(np.linalg.norm(expected[2].astype(np.float64))),1e-30))
                assert error>1e-4;faults.append({'fault':number,'logit_relative_l2':error})
            checks=[]
            for threads in (1,6):
                for profile in (0,1):
                    for generate in (False,True):
                        label=f'tiny{cap}.t{threads}.p{profile}.g{int(generate)}'
                        rows=execute(target['manifest'],[2,3,4,5,6,7],[0,8,9,10],label,threads,profile,generate,16,31)
                        oracle=natural if generate else expected
                        for row in rows:assert all(np.array_equal(a,b) for a,b in zip(U.read_output(OUT/f'{label}.{row["repetition"]}.bin'),oracle))
                        checks.append({'threads':threads,'profile':profile,'generate':generate,'rows':rows,'complete_independent356_arrays_exact':True})
            result['tiny'].append({'capacity':cap,'faults':faults,'checks':checks})
        for kind in ('cases','long_cases'):
            for index,control in enumerate(qualified[kind]):
                for threads in (1,6):
                    for profile in (0,1):
                        for generate in ((False,True) if kind=='cases' else (False,)):
                            stage=f'{kind}{index}.t{threads}.p{profile}.g{int(generate)}';label=stage
                            rows=execute(artifact['manifest'],control['source_ids'],control['decoder_ids'],label,threads,profile,generate,16 if kind=='cases' else 32,32098)
                            expected=control['generation_rows'][0]['output_sha256'] if generate else control['native_sha256']
                            assert all(r['output_sha256']==expected for r in rows)
                            result['controls'].append({'kind':kind,'index':index,'threads':threads,'profile':profile,'generate':generate,'rows':rows,'complete356_output_exact':True})
        for bi,book in enumerate(records[362]['items']):
            for ci,fixture in enumerate(book['cases']):
                stage=f'quality.book{bi}.case{ci}';old=records[363]['books'][bi]['cases'][ci]
                forced=execute(artifact['manifest'],fixture['source_ids'],fixture['decoder_ids'],stage+'.teacher',6)
                generated=execute(artifact['manifest'],fixture['source_ids'],[],stage+'.natural',6,generate=True)
                assert all(r['output_sha256']==old['native_output_sha256'] for r in forced)
                assert all(r['output_sha256']==old['generation']['native_generation_sha256'] for r in generated)
                result['quality_bridges'].append({'book':bi,'case':ci,'forced_rows':forced,'generation_rows':generated,'complete363_teacher_and_own_natural_exact':True})
            print(json.dumps({'book':bi,'complete_quality_bridges':len(result['quality_bridges'])}),flush=True)
        assert before==(payload.stat().st_size,payload.stat().st_mtime_ns)
        result['gates']={'source_math_and_previous_engine_paths_exact':True,'negative_actual_worker_affinity_fault_detected':True,
                         'both13_integer_primitive_paths_exact356':len(result['integer_primitives'])==2,
                         'both_Tiny_caps_all9_faults_and_full_arrays_exact':len(result['tiny'])==2 and sum(len(t['faults']) for t in result['tiny'])==9,
                         'engineering_forced_natural_and_long_forced_1_6_profiles_exact356':len(result['controls'])==24,
                         'ALL96_teacher_and_own_natural_complete_bytes_exact363':len(result['quality_bridges'])==96,
                         'all_actual_worker_and_process_affinity_readbacks_exact':True}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-started,'maximum_checked_combined_RSS_bytes':maximum,'available_RAM_end_bytes':psutil.virtual_memory().available}
        result['decision']='exact_new_worker_execution_inherits_only_scoped363_quality_and_licenses375_cost' if all(result['gates'].values()) else 'new_worker_execution_not_qualified'
        result['scope']='NEW Windows one actual worker per physical core and active wait/infinite blocktime, unchanged356 all-A16 math and full256 target. ALL96 consumed363 full teacher/natural bytes exact, only scoped363 quality inheritance. Apparatus timings are not accepted-rate; no original reload/new untouched quality/physicalDRAM/more-n quality/cross-family/100B proof.'
        M.write(args.out,result);print(json.dumps({'sha256':digest(args.out),'gates':result['gates'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-started,'maximum_checked_combined_RSS_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
