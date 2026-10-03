"""Qualify unchanged345 forward through natural cached greedy wrapper351."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
import argparse
import gc
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import psutil
import torch
import transformers
from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
import meth324_switch_reference as M
import meth328_switch_native_contract as B
import meth336_switch_w8a8_contract as C
import meth345_switch_head_a16_reference as R
import meth351_switch_generation_reference as H

PROTOCOL=M.DOC/'METH_353_SWITCH_GENERATION_CONTRACT_PROTOCOL_20261003.md'
OUT=M.ROOT/'results/native_expert_scaling/meth353_switch_generation_contract'
ENTRY=M.ROOT/'benchmarks/native_expert_scaling/meth351_switch_multi_span_generate_entry.c'
SOURCE=M.ROOT/'benchmarks/native_expert_scaling/meth345_switch_head_a16.c'
NUMERIC=M.DOC/'meth347_switch_head_a16_contract_resume_result.json'
NUMERIC_SHA='0242039aef21bbb087ae7c973a4bfe718507d8da2f80b8ba796d5d5ee8792c45'
QUALITY=M.DOC/'meth349_switch_head_a16_fresh_prediction_result.json'
QUALITY_SHA='c75f3e43aec5ed73e8365bbb739fcbb8ceb492f5ffcdbba1b94b07009942e75d'


def exact(actual,expected):
    checks={name:bool(np.array_equal(a,b)) for name,a,b in zip(('encoder','decoder','logits','routes'),actual,expected)}
    assert all(checks.values()),checks
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-353-natural-greedy-head-A16-wrapper-contract','commands':[],'tiny_cases':[],'forced_bridges':[],'cases':[]}
    def guard(child=None):
        nonlocal maximum
        rss=psutil.Process().memory_info().rss
        if child is not None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        maximum=max(maximum,rss)
        if rss>16<<30 or time.monotonic()-start>1200:
            if child is not None and child.poll() is None:child.kill();child.wait()
            raise RuntimeError('generation_contract_resource_guard')
    try:
        recovered=C.EXPORTED
        for path in (Path(__file__),PROTOCOL,ENTRY,SOURCE,NUMERIC,QUALITY,recovered,B.ENGINE,C.BRIDGE,
                     Path(M.__file__),Path(B.__file__),Path(C.__file__),Path(R.__file__),Path(R.R.__file__),Path(R.R.R.__file__),Path(H.__file__),
                     M.ROOT/'benchmarks/native_expert_scaling/meth345_switch_head_a16_entry.c'):M.committed(path)
        assert M.digest(NUMERIC)==NUMERIC_SHA and M.digest(QUALITY)==QUALITY_SHA and M.digest(recovered)==C.EXPORTED_SHA
        numeric=json.loads(NUMERIC.read_text(encoding='utf-8'));quality=json.loads(QUALITY.read_text(encoding='utf-8'));exported=json.loads(recovered.read_text(encoding='utf-8'))
        assert all(numeric['gates'].values()) and all(quality['gates'].values())
        assert M.digest(C.BRIDGE)==C.BRIDGE_SHA and M.digest(B.COMPILER)==B.COMPILER_SHA and M.digest(B.DLL)==B.DLL_SHA
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        assert M.digest(SOURCE)==numeric['source_sha256']
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'entry_sha256':M.digest(ENTRY),'source_sha256':M.digest(SOURCE),
                       'generation_reference_sha256':M.digest(H.__file__),'numeric347_sha256':NUMERIC_SHA,'quality349_sha256':QUALITY_SHA,'engine_sha256':M.digest(B.ENGINE)})
        OUT.mkdir(parents=True);shutil.copyfile(B.DLL,OUT/'libomp.dll');binary=OUT/'meth353_switch_generate.exe'
        command=[str(B.COMPILER),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp','-DSILICON_SWITCH_MULTI_SPAN_GENERATE',str(B.ENGINE),'-o',str(binary)]
        compiled=subprocess.run(command,capture_output=True,timeout=120);(OUT/'compile.stdout.log').write_bytes(compiled.stdout);(OUT/'compile.stderr.log').write_bytes(compiled.stderr)
        assert compiled.returncode==0,compiled.stderr.decode(errors='replace')
        result['compile']={'argv':command,'binary_sha256':M.digest(binary),'compiler_sha256':B.COMPILER_SHA,'runtime_sha256':B.DLL_SHA}
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def run(argv,label,threads):
            env['OMP_NUM_THREADS']=str(threads)
            with (OUT/(label+'.stdout.log')).open('wb') as out,(OUT/(label+'.stderr.log')).open('wb') as err:
                child=subprocess.Popen([str(binary),*argv],stdout=out,stderr=err,env=env)
                while child.poll() is None:guard(child);time.sleep(.25)
            result['commands'].append({'argv':[str(binary),*argv],'returncode':child.returncode,'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode);guard()
            return [json.loads(line) for line in (OUT/(label+'.stdout.log')).read_text(encoding='utf-8').splitlines()]
        def generation(spec,ids,cap,closing,label,threads,profile,warmups,reps):
            rows=run(['--generate',str(spec),','.join(map(str,ids)),str(OUT/label),str(threads),str(cap),str(closing),str(profile),str(warmups),str(reps),'0'],label,threads)
            assert len(rows)==warmups+reps
            for row in rows:
                output=OUT/f'{label}.{row["repetition"]}.bin';row['output_sha256']=M.digest(output)
                assert len(row['generated_ids'])==row['actual_generated_tokens']<=cap
                assert row['generated_ids']==B.read_output(output)[2].argmax(-1).tolist()
            assert len({row['output_sha256'] for row in rows})==1
            return rows
        torch.set_num_threads(1);bridge=json.loads(C.BRIDGE.read_text(encoding='utf-8'))
        for capacity in (1,64):
            stage=f'tiny_capacity{capacity}';torch.manual_seed(328);old=next(v for v in bridge['cases'] if v['capacity']==capacity)
            model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**old['tiny_config'])).eval()
            original=B.export(model,OUT/f'tiny_original{capacity}');assert original['weights_sha256']==old['artifact']['weights_sha256']
            source=torch.tensor([[2,3,4,5,6,7]]);cap=16;closing=31
            original_ids,original_arrays=H.cached_greedy(model,source,cap,closing);official_ids,official_scores=H.official_greedy(model,source,cap,closing)
            original_error=B.relative(official_scores,original_arrays[2]);assert original_ids==official_ids and original_error<=1e-6
            entries,artifact=C.tiny_export(model,OUT/f'tiny_target{capacity}')
            with R.compact_reference(model,entries,artifact['payload']) as mode:
                choices,expected=H.cached_greedy(model,source,cap,closing)
                generate_choices,generate_scores=H.official_greedy(model,source,cap,closing)
            assert choices==generate_choices and np.array_equal(expected[2],generate_scores)
            assert mode.calls['head_a16_integer_projection']==2*len(choices)
            checks=[]
            for threads in (1,6):
                for profile in (0,1):
                    label=f'tiny{capacity}.t{threads}.p{profile}';rows=generation(artifact['manifest'],source[0].tolist(),cap,closing,label,threads,profile,1,1)
                    for row in rows:
                        assert row['generated_ids']==choices;checks.append(exact(B.read_output(OUT/f'{label}.{row["repetition"]}.bin'),expected))
            arrays=OUT/f'tiny{capacity}.reference.npz';np.savez(arrays,**dict(zip(('encoder','decoder','logits','routes'),expected)))
            result['tiny_cases'].append({'capacity':capacity,'same328_weights':True,'original_generate_cache_choices_exact':True,'original_generate_cache_logit_relative':original_error,
                                        'compact_generate_cache_scores_exact':True,'artifact':artifact,'generated_ids':choices,'checks':checks,'reference_sha256':M.digest(arrays),'passed':True})
            del model;gc.collect();guard()
        stage='fresh_unchanged_target_identity';artifact=exported['artifact'];payload=Path(artifact['payload']);spec=Path(artifact['manifest']);before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        import hashlib
        h=hashlib.sha256()
        with payload.open('rb') as f:
            while block:=f.read(4<<20):h.update(block);guard()
        assert before[0]==artifact['bytes'] and h.hexdigest()==artifact['sha256'] and M.digest(spec)==artifact['manifest_sha256'];result['artifact']=artifact
        for i,control in enumerate(numeric['cases']):
            stage=f'forced347_bridge{i}'
            for threads in (1,6):
                for profile in (0,1):
                    label=f'forced{i}.t{threads}.p{profile}'
                    rows=run([str(spec),','.join(map(str,control['source_ids'])),','.join(map(str,control['decoder_ids'])),str(OUT/label),str(threads),str(profile),'0','1','0'],label,threads)
                    sha=M.digest(OUT/(label+'.0.bin'));assert sha==control['native_sha256'];result['forced_bridges'].append({'case':i,'threads':threads,'profile':profile,'exact347':True,'output_sha256':sha})
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**exported['original_config']))
        result['reference_model']=R.R.target_control_model(model,exported['tensors'],payload)
        for i,control in enumerate(numeric['cases']):
            stage=f'actual_generation_independent{i}'
            with R.compact_reference(model,exported['tensors'],payload) as mode:
                choices,expected=H.cached_greedy(model,torch.tensor([control['source_ids']]),16,32098)
            assert mode.calls['head_a16_integer_projection']==len(choices)
            label=f'actual_generation{i}';rows=generation(spec,control['source_ids'],16,32098,label,6,0,1,1)
            for row in rows:assert row['generated_ids']==choices;exact(B.read_output(OUT/f'{label}.{row["repetition"]}.bin'),expected)
            result['cases'].append({'source_ids':control['source_ids'],'cap':16,'closing':32098,'generated_ids':choices,'rows':rows,'independent_complete_exact':True})
        assert before==(payload.stat().st_size,payload.stat().st_mtime_ns)
        result['gates']={'both_same328_original_official_and_manual_cached_greedy':True,'both_compact_tiny_generated_full_arrays_exact_all_threads_profiles':True,
                         'fresh_same338_target_identity':True,'all_eight_forced_bridges_exact347':len(result['forced_bridges'])==8,'actual_engineering_generated_full_integer_reference_exact':len(result['cases'])==2}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='wrapper_numeric_eligible_for_separate_NEW_original_prediction_generation_task_then_accepted_rate'
        result['scope']='Same345 arithmetic/new natural greedy driver; Tiny and consumed engineering numerical controls only. No original-source generation quality or accepted speed/n-scaling proof. New binary not substituted silently for349.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'gates':result['gates'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
