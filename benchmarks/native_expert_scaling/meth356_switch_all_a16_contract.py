"""ALL I8 inputs A16: independent full numerical/cache contract, consumed controls."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
import argparse
import gc
import hashlib
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
import meth347_switch_head_a16_contract_resume as P
import meth351_switch_generation_reference as H
import meth356_switch_all_a16_reference as R

PROTOCOL=M.DOC/'METH_356_SWITCH_ALL_A16_CONTRACT_PROTOCOL_20261004.md'
OUT=M.ROOT/'results/native_expert_scaling/meth356_switch_all_a16_contract'
SOURCE=M.ROOT/'benchmarks/native_expert_scaling/meth356_switch_all_a16.c'
ENTRY=M.ROOT/'benchmarks/native_expert_scaling/meth356_switch_all_a16_entry.c'
COST_ENTRY=M.ROOT/'benchmarks/native_expert_scaling/meth356_switch_all_a16_cost_entry.c'
INPUTS={
    338:('meth338_switch_tensor_recovery_result.json','19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'),
    347:('meth347_switch_head_a16_contract_resume_result.json','0242039aef21bbb087ae7c973a4bfe718507d8da2f80b8ba796d5d5ee8792c45'),
    350:('meth350_switch_multi_span_manifest.json','14e7782dc5e60313b2f3341862dff5f825811b5f41e0e78ab2ba0260c6683f43'),
    351:('meth351_switch_multi_span_quality_result.json','6859c57d40bfc32bd80be948d1c227381cef8e165d2ef3820b0c314053e2c306'),
    355:('meth355_switch_multispan_attribution_result.json','639308005e6f7740e96b747d74a6b29df401e9b401fbef7212d5362d76e58d10')}


def exact(actual,expected):
    checks={name:bool(np.array_equal(a,b)) for name,a,b in zip(('encoder','decoder','logits','routes'),actual,expected)}
    assert all(checks.values()),checks
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();maximum=0;stage='bindings'
    result={'experiment':'METH-356-all-I8-inputs-A16-full-contract','commands':[],'tiny_cases':[],'cases':[],'long_cases':[],'consumed_cases':[]}
    def guard(child=None):
        nonlocal maximum
        rss=psutil.Process().memory_info().rss
        if child is not None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        maximum=max(maximum,rss)
        if rss>16<<30 or time.monotonic()-start>1800:
            if child is not None and child.poll() is None:child.kill();child.wait()
            raise RuntimeError('all_a16_contract_resource_guard_30min_16GiB')
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),PROTOCOL,SOURCE,ENTRY,COST_ENTRY,B.ENGINE,C.BRIDGE,Path(M.__file__),Path(B.__file__),Path(C.__file__),Path(P.__file__),
                     Path(H.__file__),Path(R.__file__),Path(R.R.__file__),Path(R.H.__file__),Path(R.R.R.__file__)):M.committed(path)
        records={}
        for n,(name,sha) in INPUTS.items():
            path=M.DOC/name;M.committed(path);assert digest(path)==sha;records[n]=json.loads(path.read_text(encoding='utf-8'))
        assert all(records[338]['gates'].values()) and all(records[347]['gates'].values()) and all(records[350]['gates'].values()) and all(records[355]['gates'].values())
        assert not records[351]['gates']['masked_original_top1_agreement_ge0p95']
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        assert digest(C.BRIDGE)==C.BRIDGE_SHA and digest(B.COMPILER)==B.COMPILER_SHA and digest(B.DLL)==B.DLL_SHA
        result.update({'controller_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'source_sha256':digest(SOURCE),'entry_sha256':digest(ENTRY),
                       'cost_entry_sha256':digest(COST_ENTRY),'reference_sha256':digest(R.__file__),'engine_sha256':digest(B.ENGINE),'input_sha256':{str(n):pair[1] for n,pair in INPUTS.items()}})
        OUT.mkdir(parents=True);shutil.copyfile(B.DLL,OUT/'libomp.dll');binary=OUT/'meth356_switch_all_a16.exe'
        command=[str(B.COMPILER),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp','-DSILICON_SWITCH_W8A16_ALL',str(B.ENGINE),'-o',str(binary)]
        compiled=subprocess.run(command,capture_output=True,timeout=120);(OUT/'compile.stdout.log').write_bytes(compiled.stdout);(OUT/'compile.stderr.log').write_bytes(compiled.stderr)
        assert compiled.returncode==0,compiled.stderr.decode(errors='replace')
        result['compile']={'argv':command,'binary_sha256':digest(binary),'compiler_sha256':B.COMPILER_SHA,'runtime_sha256':B.DLL_SHA}
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def run(argv,label,threads=1):
            env['OMP_NUM_THREADS']=str(threads)
            with (OUT/(label+'.stdout.log')).open('wb') as out,(OUT/(label+'.stderr.log')).open('wb') as err:
                child=subprocess.Popen([str(binary),*argv],stdout=out,stderr=err,env=env)
                while child.poll() is None:guard(child);time.sleep(.25)
            result['commands'].append({'argv':[str(binary),*argv],'returncode':child.returncode,'stdout_sha256':digest(OUT/(label+'.stdout.log')),'stderr_sha256':digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode);guard()
            return [json.loads(line) for line in (OUT/(label+'.stdout.log')).read_text(encoding='utf-8').splitlines()] if len(argv) in (9,11) else []
        def forced(spec,source,decoder,label,threads,profile=0):
            run([str(spec),','.join(map(str,source)),','.join(map(str,decoder)),str(OUT/label),str(threads),str(profile),'0','1','0'],label,threads)
            return OUT/(label+'.0.bin')
        def generated(spec,source,cap,closing,label,threads,profile=0):
            rows=run(['--generate',str(spec),','.join(map(str,source)),str(OUT/label),str(threads),str(cap),str(closing),str(profile),'1','1','0'],label,threads)
            assert [row['repetition'] for row in rows]==[-1,0]
            for row in rows:
                path=OUT/f'{label}.{row["repetition"]}.bin';row['output_sha256']=digest(path)
                assert row['generated_ids']==B.read_output(path)[2].argmax(-1).tolist()
                assert row['actual_generated_tokens']==len(row['generated_ids'])<=cap
            assert rows[0]['output_sha256']==rows[1]['output_sha256']
            return rows
        stage='both_integer_primitive_paths';inp=OUT/'integer_cases.bin';primitives=P.primitive_inputs(inp);result['integer_primitives']=[]
        for flag in ('--int-dot','--head-int-dot'):
            answer=OUT/(flag[2:]+'.bin');run([flag,str(inp),str(answer)],flag[2:]);entry=P.primitive_result(answer,primitives);entry['path']=flag;result['integer_primitives'].append(entry)
        torch.set_num_threads(1);bridge=json.loads(C.BRIDGE.read_text(encoding='utf-8'))
        for capacity in (1,64):
            stage=f'tiny_capacity{capacity}';torch.manual_seed(328);old=next(v for v in bridge['cases'] if v['capacity']==capacity)
            model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**old['tiny_config'])).eval()
            original=B.export(model,OUT/f'tiny_original{capacity}');assert original['weights_sha256']==old['artifact']['weights_sha256']
            entries,artifact=C.tiny_export(model,OUT/f'tiny_target{capacity}');source=[2,3,4,5,6,7];decoder=[0,8,9,10]
            with R.compact_reference(model,entries,artifact['payload']) as mode:
                expected=B.torch_reference(model,torch.tensor([source]),torch.tensor([decoder]))
                choices,gen_expected=H.cached_greedy(model,torch.tensor([source]),16,31)
                official_choices,official_scores=H.official_greedy(model,torch.tensor([source]),16,31)
            assert choices==official_choices and np.array_equal(official_scores,gen_expected[2]) and mode.calls['integer_projection']>0
            faults=[];checks=[]
            for number in (1,2,3,4,5):
                if number==4 and capacity==64:continue
                output=OUT/f'tiny{capacity}.fault{number}.bin';run([artifact['manifest'],','.join(map(str,source)),','.join(map(str,decoder)),str(output),str(number)],f'tiny{capacity}.fault{number}')
                error=B.relative(B.read_output(output)[2],expected[2]);assert error>1e-4;faults.append({'fault':number,'logit_relative_l2':error,'detected':True})
            for threads in (1,6):
                for profile in (0,1):
                    label=f'tiny{capacity}.t{threads}.p{profile}';path=forced(artifact['manifest'],source,decoder,label,threads,profile);checks.append(exact(B.read_output(path),expected))
                    rows=generated(artifact['manifest'],source,16,31,label+'.gen',threads,profile)
                    for row in rows:assert row['generated_ids']==choices;checks.append(exact(B.read_output(OUT/f'{label}.gen.{row["repetition"]}.bin'),gen_expected))
            arrays=OUT/f'tiny{capacity}.reference.npz';np.savez(arrays,encoder=expected[0],decoder=expected[1],logits=expected[2],routes=expected[3],gen_encoder=gen_expected[0],gen_decoder=gen_expected[1],gen_logits=gen_expected[2],gen_routes=gen_expected[3])
            result['tiny_cases'].append({'capacity':capacity,'same328_weights':True,'artifact':artifact,'checks':checks,'faults':faults,'compact_cached_official_greedy_exact':True,'reference_sha256':digest(arrays),'passed':True})
            del model;gc.collect();guard()
        stage='fresh_full_target_identity';recovered=records[338];artifact=recovered['artifact'];payload=Path(artifact['payload']);spec=Path(artifact['manifest']);before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert before[0]==artifact['bytes'] and digest(payload)==artifact['sha256'] and digest(spec)==artifact['manifest_sha256'];result['artifact']=artifact
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**recovered['original_config']))
        result['reference_model']=R.target_control_model(model,recovered['tensors'],payload)
        def complete_control(source,decoder,label,cap,closing,profiles=False):
            with R.compact_reference(model,recovered['tensors'],payload) as mode:
                expected=B.torch_reference(model,torch.tensor([source]),torch.tensor([decoder]))
                choices,gen_expected=H.cached_greedy(model,torch.tensor([source]),cap,closing) if cap else ([],None)
            arrays=OUT/(label+'.reference.npz');values=dict(zip(('encoder','decoder','logits','routes'),expected))
            if cap:values.update(dict(zip(('gen_encoder','gen_decoder','gen_logits','gen_routes'),gen_expected)))
            np.savez(arrays,**values);checks=[];gen_checks=[];hashes=[];gen_rows=[]
            for threads in (1,6):
                for profile in ((0,1) if profiles else (0,)):
                    prefix=f'{label}.t{threads}.p{profile}';path=forced(spec,source,decoder,prefix,threads,profile);checks.append(exact(B.read_output(path),expected));hashes.append(digest(path))
                    if cap:
                        rows=generated(spec,source,cap,closing,prefix+'.gen',threads,profile)
                        for row in rows:assert row['generated_ids']==choices;gen_checks.append(exact(B.read_output(OUT/f'{prefix}.gen.{row["repetition"]}.bin'),gen_expected))
                        gen_rows.extend(rows)
            assert len(set(hashes))==1
            return {'source_ids':source,'decoder_ids':decoder,'native_sha256':hashes[0],'checks':checks,'reference_sha256':digest(arrays),
                    'generated_ids':choices,'generation_checks':gen_checks,'generation_rows':gen_rows,'reference_calls':dict(mode.calls),'passed':True}
        for i,control in enumerate(records[347]['cases']):
            stage=f'engineering{i}';result['cases'].append(complete_control(control['source_ids'],control['decoder_ids'],f'case{i}',16,32098,True));print(json.dumps({'stage':stage,'passed':True}),flush=True)
        for i,fixture in enumerate(records[347]['long_cases']):
            stage=f'long{i}';entry=complete_control(fixture['source_ids'],fixture['decoder_ids'],f'long{i}',0,0);entry['source_length']=fixture['source_length'];result['long_cases'].append(entry);print(json.dumps({'stage':stage,'passed':True}),flush=True)
        for bi,ci in ((0,0),(23,3)):
            stage=f'consumed{bi}_{ci}';control=records[350]['items'][bi]['cases'][ci];entry=complete_control(control['source_ids'],control['decoder_ids'],f'consumed{bi}_{ci}',64,32095);entry.update({'book':bi,'case':ci,'consumed351_source':True});result['consumed_cases'].append(entry);print(json.dumps({'stage':stage,'passed':True}),flush=True)
        assert before==(payload.stat().st_size,payload.stat().st_mtime_ns) and R.R.integer_projection is not R.projection,'reference_restore_or_payload_changed'
        result['gates']={'both_all_A16_primitive_paths_scalar_extremes_exact':True,'same328_tiny_nine_faults_and_greedy_profiles_exact':True,
                         'fresh_same338_complete_artifact_identity':True,'full_engineering_forced_and_natural_arrays_exact':len(result['cases'])==2,
                         'both_long_fixtures_complete_independent_exact':len(result['long_cases'])==2,'both_consumed_multispan_forced_and_own_greedy_exact':len(result['consumed_cases'])==2,'scoped_reference_restored':True}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='all_A16_numeric_eligible_for_actual_cost_then_NEW_original_primary_quality'
        result['scope']='All quantized projections use A16 inputs with SAME338 I8 weights. Independent full consumed/Tiny numerical/cache/greedy contracts; no untouched quality, accepted rate, useful n or DRAM proof.351 failures preserved; F32 router/lookup/norm/attention unchanged.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':digest(args.out),'gates':result['gates'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
