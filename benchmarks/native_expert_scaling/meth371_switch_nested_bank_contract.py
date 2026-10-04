"""Full independent actualnested64/128 dynamicn native target contract."""
import os
os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import psutil
import torch
import transformers
from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
import meth324_switch_reference as M
import meth328_switch_native_contract as B
import meth351_switch_generation_reference as H
import meth356_switch_all_a16_reference as K
import meth359_switch_cost_topology as A
import meth368_switch_bank_manifest as I
import meth371_switch_subset_reference as R

OUT=M.ROOT/'results/native_expert_scaling/meth371_switch_nested_bank_contract'
PROTOCOL=M.DOC/'METH_371_SWITCH_NESTED_BANK_CONTRACT_PROTOCOL_20261004.md'
INPUTS={356:('meth356_switch_all_a16_contract_result.json','ec51e76c08277e4f874cacac9a49d5b60477c0abb6a651cbe648f9bdad3f66d3'),
        361:('meth361_switch_single_core_cost_result.json','9446c5cc259311afead2dbaa537debd3fcee34e93959b34d9beaa677fc49a85b'),
        362:('meth362_switch_multi_span_manifest.json','c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'),
        363:('meth363_switch_all_a16_multi_span_quality_result.json','ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'),
        370:('meth370_switch_nested_bank_export_result.json','7668901af8a81fd797f0295063a6839c74c5ff7f32cce56e81d82b04cdb48243')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-371-actualnested64-128-full-dynamicn-native-contract','commands':[],'targets':[]}
    def guard(child=None):
        nonlocal maximum
        rss=psutil.Process().memory_info().rss
        if child is not None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        maximum=max(maximum,rss)
        if rss>16<<30 or time.monotonic()-start>1800:
            if child is not None and child.poll() is None:child.kill();child.wait()
            raise RuntimeError('actualnested_bank_contract_30min_16GiB')
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),PROTOCOL,Path(M.__file__),B.ENGINE,Path(B.__file__),Path(H.__file__),Path(K.__file__),Path(K.R.__file__),Path(K.H.__file__),Path(K.R.R.__file__),Path(A.__file__),Path(I.__file__),Path(R.__file__)):M.committed(path)
        records={}
        for n,(name,sha) in INPUTS.items():
            path=M.DOC/name;M.committed(path);assert digest(path)==sha;records[n]=json.loads(path.read_text(encoding='utf-8'))
        assert all(all(row['gates'].values()) for row in records.values())
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        fresh=A.physical_topology();assert fresh==records[361]['fresh_topology'];affinity=[0]
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid','name','cmdline']):
            cmd=' '.join(process.info['cmdline'] or []).replace('\\','/')
            if process.pid not in own and process.info['name'].lower() in ('python.exe','meth356_switch_all_a16.exe','meth365_switch_encoder_batches.exe'):
                assert not ('benchmarks/native_expert_scaling/' in cmd or 'meth356_switch_all_a16.exe' in cmd or 'meth365_switch_encoder_batches.exe' in cmd),'concurrent_model_or_native_worker'
        numeric=records[356];binary=Path(numeric['compile']['argv'][-1]);assert digest(binary)==numeric['compile']['binary_sha256']==records[363]['native_binary_sha256']
        assert digest(binary.parent/'libomp.dll')==numeric['compile']['runtime_sha256'] and digest(numeric['compile']['argv'][0])==numeric['compile']['compiler_sha256']
        result.update({'controller_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'reference371_sha256':digest(R.__file__),'input_sha256':{str(n):v[1] for n,v in INPUTS.items()},
                       'compile':numeric['compile'],'native_threads':1,'native_process_affinity':affinity,'fresh_topology':fresh,'unchanged356_primitive_Tiny_nine_fault_runtime_reused':True})
        OUT.mkdir(parents=True);env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def run(argv,label,threads=1):
            env['OMP_NUM_THREADS']=str(threads)
            with (OUT/(label+'.stdout.log')).open('wb') as out,(OUT/(label+'.stderr.log')).open('wb') as err:
                child=subprocess.Popen([str(binary),*argv],stdout=out,stderr=err,env=env)
                try:
                    mask=affinity if threads==1 else fresh['selected_one_logical_per_physical_core'];process=psutil.Process(child.pid)
                    process.cpu_affinity(mask);actual_affinity=process.cpu_affinity();assert actual_affinity==mask
                except BaseException:
                    if child.poll() is None:child.kill();child.wait()
                    raise
                while child.poll() is None:guard(child);time.sleep(.25)
            assert child.returncode==0,(label,child.returncode)
            result['commands'].append({'argv':[str(binary),*argv],'returncode':child.returncode,'actual_affinity':actual_affinity,'stdout_sha256':digest(OUT/(label+'.stdout.log')),'stderr_sha256':digest(OUT/(label+'.stderr.log'))})
            return [json.loads(line) for line in (OUT/(label+'.stdout.log')).read_text(encoding='utf-8').splitlines()]
        torch.set_num_threads(1)
        for target in records[370]['targets']:
            n=target['n'];assert n in (64,128);stage=f'fresh_target{n}'
            assert digest(target['payload'])==target['sha256'] and digest(target['manifest'])==target['manifest_sha256'] and digest(target['metadata'])==target['metadata_sha256']
            payload=Path(target['payload']);before=(payload.stat().st_size,payload.stat().st_mtime_ns);assert before[0]==target['bytes']
            metadata=json.loads(Path(target['metadata']).read_text(encoding='utf-8'));entries=metadata['tensors'];config=target['original_config']
            assert config==metadata['original_config'] and config['num_experts']==n and len(entries)==24*n+248==target['namespace']
            I.read_manifest(target['manifest'],config,entries,payload)
            with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**config))
            assert sum(v.numel() for v in model.parameters())==target['retained_original_unique_coefficients']
            record={'n':n,'artifact':target,'reference_model':R.target_control_model(model,entries,payload),'engineering':[],'long':[],'consumed':[]};result['targets'].append(record)
            def complete(source,decoder,label,cap=0,closing=0,profiles=False):
                with R.compact_reference(model,entries,payload) as mode:
                    expected=B.torch_reference(model,torch.tensor([source]),torch.tensor([decoder]))
                    if cap:
                        choices,natural=H.cached_greedy(model,torch.tensor([source]),cap,closing)
                        official_choices,official_logits=H.official_greedy(model,torch.tensor([source]),cap,closing)
                        assert choices==official_choices and np.array_equal(official_logits,natural[2])
                    else:choices=[];natural=None
                arrays=OUT/(label+'.reference.npz');values=dict(zip(('encoder','decoder','logits','routes'),expected))
                if cap:values.update(dict(zip(('gen_encoder','gen_decoder','gen_logits','gen_routes'),natural)))
                np.savez(arrays,**values);checks=[];hashes=[];genhashes=[];rows=[];consultedhashes=[]
                for threads in (1,6):
                    for profile in ((0,1) if profiles else (0,)):
                        prefix=OUT/f'{label}.t{threads}.p{profile}'
                        observed=run([target['manifest'],','.join(map(str,source)),','.join(map(str,decoder)),str(prefix),str(threads),str(profile),'0','1','0'],prefix.name,threads)
                        path=Path(str(prefix)+'.0.bin');actual=B.read_output(path);check={k:bool(np.array_equal(a,b)) for k,a,b in zip(('encoder','decoder','logits','routes'),actual,expected)};assert all(check.values()),(label,check)
                        counts=observed[0]['counters'][1];positions=len(decoder)
                        assert sum(v['code_bytes'] for v in counts)==123764736*positions and sum(v['scale_bytes'] for v in counts)==534016*positions and sum(v['f32_bytes'] for v in counts)==18432*n*positions
                        checks.append(check);hashes.append(digest(path))
                        if cap:
                            generated=run(['--generate',target['manifest'],','.join(map(str,source)),str(prefix)+'.gen',str(threads),str(cap),str(closing),str(profile),'1','1','0'],prefix.name+'.gen',threads)
                            assert [v['repetition'] for v in generated]==[-1,0]
                            for row in generated:
                                path=Path(str(prefix)+f'.gen.{row["repetition"]}.bin');actual=B.read_output(path)
                                check={k:bool(np.array_equal(a,b)) for k,a,b in zip(('encoder','decoder','logits','routes'),actual,natural)};assert all(check.values()),(label,check)
                                assert row['generated_ids']==choices;row['output_sha256']=digest(path);checks.append(check);genhashes.append(row['output_sha256']);rows.append(row)
                                consulted=I.consultations(config,len(source),len(choices),actual[3],0);assert all(v['source_expert'] is None or 0<=v['source_expert']<n for v in consulted)
                                sidecar=path.with_suffix('.consultations.json');M.write(sidecar,consulted);consultedhashes.append(digest(sidecar))
                assert len(set(hashes))==1 and (not cap or len(set(genhashes))==len(set(consultedhashes))==1)
                return {'source_ids':source,'decoder_ids':decoder,'native_sha256':hashes[0],'reference_sha256':digest(arrays),'checks':checks,'generated_ids':choices,
                        'natural_sha256':genhashes[0] if cap else None,'consultations_sha256':consultedhashes[0] if cap else None,'generation_rows_not_rate':rows,'reference_calls':dict(mode.calls),'passed':True}
            for ci,control in enumerate(numeric['cases']):
                stage=f'n{n}.engineering{ci}';record['engineering'].append(complete(control['source_ids'],control['decoder_ids'],stage,16,32098,True))
            for ci,control in enumerate(numeric['long_cases']):
                stage=f'n{n}.long{ci}';entry=complete(control['source_ids'],control['decoder_ids'],stage);entry['source_length']=control['source_length'];record['long'].append(entry)
            for bi,ci in ((0,0),(23,3)):
                stage=f'n{n}.consumed{bi}_{ci}';control=records[362]['items'][bi]['cases'][ci];entry=complete(control['source_ids'],control['decoder_ids'],stage,64,32095);entry['book']=bi;entry['case']=ci;record['consumed'].append(entry)
            assert before==(payload.stat().st_size,payload.stat().st_mtime_ns) and K.R.integer_projection is not K.projection
            record['passed']=True;print(json.dumps({'n':n,'namespace':target['namespace'],'full_independent_pass':True}),flush=True)
            del model;gc.collect();guard()
        assert [v['n'] for v in result['targets']]==[64,128]
        result['gates']={'SAME356_runtime_compile_profile_identity_exact':True,'both_actual_smaller_payload_spec_metadata_fresh_identity_exact':True,
                         'both_NEW_actual_n_META_namespace_shapes_source_counts_exact':True,'both_targets_engineering_forced_own_greedy_profile_thread_arrays_exact':True,
                         'both_targets_source9_64_forced32_long_arrays_exact':True,'both_targets_fixed_source29_full_forced_own_greedy_arrays_exact':True,
                         'actual_retained_function_ids_and_dynamic_router_bytes_exact':True,'scoped_reference_restored_and_payloads_unchanged':True}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='actual_nested64_128_numeric_eligible_for_frozen372_paired_additional_real_bank_intervention'
        result['scope']='Actualsubset64/128 from SAME14.664B donor, unchanged356 executable/dynamicn/precision/core/retainedexpert coefficients, new physicallysmaller370 payloads/F32routerprefix. Full independent targetarithmetic/cache/route contract, NOT originalquality/usefuladditionaln/acceptedrate/physicalDRAM. Changedmodels cannotinherit363 n256quality.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':digest(args.out),'gates':result['gates'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
