"""SAME356 runtime: independent complete fixed expert-identity control contract."""
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
import meth336_switch_w8a8_contract as C
import meth351_switch_generation_reference as H
import meth356_switch_all_a16_reference as R
import meth359_switch_cost_topology as A
import meth368_switch_bank_manifest as I

OUT=M.ROOT/'results/native_expert_scaling/meth368_switch_bank_identity_contract'
PROTOCOL=M.DOC/'METH_368_SWITCH_BANK_IDENTITY_CONTRACT_PROTOCOL_20261004.md'
INPUTS={338:('meth338_switch_tensor_recovery_result.json','19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'),
        356:('meth356_switch_all_a16_contract_result.json','ec51e76c08277e4f874cacac9a49d5b60477c0abb6a651cbe648f9bdad3f66d3'),
        361:('meth361_switch_single_core_cost_result.json','9446c5cc259311afead2dbaa537debd3fcee34e93959b34d9beaa677fc49a85b'),
        362:('meth362_switch_multi_span_manifest.json','c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'),
        363:('meth363_switch_all_a16_multi_span_quality_result.json','ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'),
        364:('meth364_switch_single_core_accepted_rate_result.json','96fdecb5557161fed1849ba8322c83e55f685754d9cbba00886e07b8b58fb4ba'),
        366:('meth366_switch_encoder_batches_cost_result.json','7a487bb58f0a9bbcc3f1d2dff8917d46457942f6a73b3b822ba2de30daee03dd')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-368-fixed-learned-bank-identity-independent-contract','commands':[],'tiny_cases':[],'controls':[],'manifests':[]}
    def guard(child=None):
        nonlocal maximum
        rss=psutil.Process().memory_info().rss
        if child is not None:
            try:rss+=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:pass
        maximum=max(maximum,rss)
        if rss>16<<30 or time.monotonic()-start>1800:
            if child is not None and child.poll() is None:child.kill();child.wait()
            raise RuntimeError('bank_identity_contract_30min_16GiB')
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),PROTOCOL,Path(I.__file__),B.ENGINE,C.BRIDGE,Path(M.__file__),Path(B.__file__),Path(C.__file__),Path(C.E.__file__),Path(H.__file__),Path(R.__file__),Path(R.R.__file__),Path(R.H.__file__),Path(R.R.R.__file__),Path(A.__file__)):M.committed(path)
        records={}
        for n,(name,sha) in INPUTS.items():
            path=M.DOC/name;M.committed(path);assert digest(path)==sha;records[n]=json.loads(path.read_text(encoding='utf-8'))
        for n in (338,356,361,362,363):assert all(records[n]['gates'].values())
        assert not records[364]['gates']['accepted_full_lower95_ge50'] and not records[366]['gates']['all_threads1_repeat_ratio_le1p10']
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        assert digest(C.BRIDGE)==C.BRIDGE_SHA
        fresh=A.physical_topology();assert fresh==records[361]['fresh_topology'];affinity=[0]
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid','name','cmdline']):
            cmd=' '.join(process.info['cmdline'] or []).replace('\\','/')
            if process.pid not in own and process.info['name'].lower() in ('python.exe','meth356_switch_all_a16.exe','meth365_switch_encoder_batches.exe'):
                assert not ('benchmarks/native_expert_scaling/' in cmd or 'meth356_switch_all_a16.exe' in cmd or 'meth365_switch_encoder_batches.exe' in cmd),'concurrent_model_or_native_worker'
        numeric=records[356];binary=Path(numeric['compile']['argv'][-1]);dll=binary.parent/'libomp.dll';compiler=Path(numeric['compile']['argv'][0])
        assert digest(binary)==numeric['compile']['binary_sha256']==records[363]['native_binary_sha256']
        assert digest(dll)==numeric['compile']['runtime_sha256'] and digest(compiler)==numeric['compile']['compiler_sha256']
        recovered=records[338];artifact=recovered['artifact'];payload=Path(artifact['payload']);before=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert before[0]==artifact['bytes'] and digest(payload)==artifact['sha256'] and digest(artifact['manifest'])==artifact['manifest_sha256']
        result.update({'controller_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'manifest_helper_sha256':digest(I.__file__),
                       'input_sha256':{str(n):v[1] for n,v in INPUTS.items()},'artifact':artifact,'compile':numeric['compile'],'native_threads':1,'native_process_affinity':affinity,'fresh_topology':fresh})
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
        def prepare(config,entries,path,offset,label):
            shifted,mapping=I.remap(entries,config['num_experts'],offset);spec=OUT/(label+'.manifest.bin')
            C.E.manifest(config,shifted,path,spec);checks=I.read_manifest(spec,config,shifted,path)
            mappingpath=OUT/(label+'.mapping.json');M.write(mappingpath,{'offset_requested':offset,'offset_effective':offset%config['num_experts'],'n':config['num_experts'],'source_names':mapping})
            entry={'offset':offset,'n':config['num_experts'],'manifest':str(spec),'manifest_sha256':digest(spec),'mapping_path':str(mappingpath),'mapping_sha256':digest(mappingpath),'checks':checks,'expert_pairs':len(mapping)//2}
            return shifted,spec,entry
        def complete(model,config,entries,spec,source,decoder,cap,closing,offset,label,profiles=False):
            with R.compact_reference(model,entries,payload if config['num_experts']==256 else tiny_payload) as mode:
                expected=B.torch_reference(model,torch.tensor([source]),torch.tensor([decoder]))
                choices,natural=H.cached_greedy(model,torch.tensor([source]),cap,closing)
                official_choices,official_logits=H.official_greedy(model,torch.tensor([source]),cap,closing)
            assert choices==official_choices and np.array_equal(official_logits,natural[2])
            arrays=OUT/(label+'.reference.npz');np.savez(arrays,**dict(zip(('encoder','decoder','logits','routes'),expected)),**dict(zip(('gen_encoder','gen_decoder','gen_logits','gen_routes'),natural)))
            hashes=[];genhashes=[];checks=[];generation_rows=[];consultation_hashes=[]
            for threads in ((1,6) if profiles else (1,)):
                for profile in ((0,1) if profiles else (0,)):
                    prefix=OUT/f'{label}.t{threads}.p{profile}'
                    run([str(spec),','.join(map(str,source)),','.join(map(str,decoder)),str(prefix),str(threads),str(profile),'0','1','0'],prefix.name,threads)
                    path=Path(str(prefix)+'.0.bin');actual=B.read_output(path);assert all(np.array_equal(a,e) for a,e in zip(actual,expected));hashes.append(digest(path));checks.append(True)
                    rows=run(['--generate',str(spec),','.join(map(str,source)),str(prefix)+'.gen',str(threads),str(cap),str(closing),str(profile),'1','1','0'],prefix.name+'.gen',threads)
                    assert [v['repetition'] for v in rows]==[-1,0]
                    for row in rows:
                        path=Path(str(prefix)+f'.gen.{row["repetition"]}.bin');actual=B.read_output(path)
                        assert row['generated_ids']==choices and all(np.array_equal(a,e) for a,e in zip(actual,natural))
                        row['output_sha256']=digest(path);genhashes.append(row['output_sha256']);generation_rows.append(row)
                        consulted=I.consultations(config,len(source),len(choices),actual[3],offset)
                        sidecar=path.with_suffix('.consultations.json');M.write(sidecar,consulted);consultation_hashes.append(digest(sidecar))
                        assert all(v['source_expert']==(v['router_slot']+offset)%config['num_experts'] for v in consulted if v['accepted'])
            assert len(set(hashes))==len(set(genhashes))==len(set(consultation_hashes))==1
            return {'label':label,'offset':offset,'n':config['num_experts'],'source_ids':source,'decoder_ids':decoder,'checks':checks,'forced_sha256':hashes[0],
                    'natural_sha256':genhashes[0],'consultations_sha256':consultation_hashes[0],'generated_ids':choices,'generation_rows_not_accepted_timing':generation_rows,
                    'reference_sha256':digest(arrays),'compact_official_vs_manual_greedy_exact':True,'reference_calls':dict(mode.calls),'passed':True}
        torch.set_num_threads(1);bridge=json.loads(C.BRIDGE.read_text(encoding='utf-8'))
        for capacity in (1,64):
            stage=f'tiny{capacity}';torch.manual_seed(328);old=next(v for v in bridge['cases'] if v['capacity']==capacity)
            model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**old['tiny_config'])).eval()
            original=B.export(model,OUT/f'tiny_original{capacity}');assert original['weights_sha256']==old['artifact']['weights_sha256']
            entries,target=C.tiny_export(model,OUT/f'tiny_target{capacity}');tiny_payload=Path(target['payload']);config=model.config.to_dict()
            for offset in (0,1):
                shifted,spec,manifest=prepare(config,entries,tiny_payload,offset,f'tiny{capacity}.offset{offset}')
                result['tiny_cases'].append(complete(model,config,shifted,spec,[2,3,4,5,6,7],[0,8,9,10],16,31,offset,f'tiny{capacity}.offset{offset}',True))
            # AtTiny n2, requested127 has effective1 and exact same mapping;
            # actual256 controls below use different nonidentity bijections.
            a,ma=I.remap(entries,2,1);b,mb=I.remap(entries,2,127);assert a==b and ma==mb
            del model;gc.collect();guard()
        config=recovered['original_config']
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**config))
        result['reference_model']=R.target_control_model(model,recovered['tensors'],payload)
        for offset in (0,1,127):
            stage=f'actual_manifest{offset}';shifted,spec,manifest=prepare(config,recovered['tensors'],payload,offset,f'actual.offset{offset}')
            assert manifest['expert_pairs']==3072
            result['manifests'].append(manifest)
            if offset==0:assert manifest['manifest_sha256']==artifact['manifest_sha256']
            for ci,control in enumerate(numeric['cases']):
                stage=f'engineering{ci}.offset{offset}';record=complete(model,config,shifted,spec,control['source_ids'],control['decoder_ids'],16,32098,offset,stage)
                if offset==0:assert record['forced_sha256']==control['native_sha256'] and record['generated_ids']==control['generated_ids']
                result['controls'].append(record)
            for bi,ci in ((0,0),(23,3)):
                stage=f'consumed363.{bi}_{ci}.offset{offset}';control=records[362]['items'][bi]['cases'][ci]
                record=complete(model,config,shifted,spec,control['source_ids'],control['decoder_ids'],64,32095,offset,stage)
                if offset==0:
                    old=records[363]['books'][bi]['cases'][ci];assert record['forced_sha256']==old['native_output_sha256'] and record['natural_sha256']==old['generation']['native_generation_sha256']
                result['controls'].append(record)
            print(json.dumps({'actual_offset':offset,'controls':len(result['controls']),'independent_pass':True}),flush=True)
        assert before==(payload.stat().st_size,payload.stat().st_mtime_ns) and R.R.integer_projection is not R.projection
        result['gates']={'SAME356_binary_payload_profile_identity_exact':True,'all_serialized_WI_WO_bijections_nonexperts_exact':len(result['manifests'])==3,
                         'same328_tiny_caps_profile_threads_independent_exact':len(result['tiny_cases'])==4,'all_three_actual_full_forced_and_own_greedy_independent_exact':len(result['controls'])==12,
                         'matched_controls_complete_bytes_exact356_and363':True,'actual_consulted_function_identities_recorded_and_exact':True,'scoped_reference_restored_payload_unchanged':True}
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='fixed_identity_control_manifests_numeric_qualified_for_paired_consumed369_usefulness_assay'
        result['scope']='SAME356 target/executable/CPU1/affinity[0], only WI/WO lookup identity fixedbijections0/1/127. Router coefficients/algorithm/probability/capacity preserved on EACH own trajectory, no original forcedroute replay. All functions remain distinct/real. Numerical and manifest semantic controls, NOT untouched quality/causal bank harm/extra n gain/accepted rate/physical DRAM.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':digest(args.out),'gates':result['gates'],'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
