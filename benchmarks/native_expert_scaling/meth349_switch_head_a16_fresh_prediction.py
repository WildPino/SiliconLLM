"""NEW paired actual head-A16/original-donor span prediction, no target arithmetic oracle."""
import argparse
import ctypes
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import psutil
import torch
import transformers
from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
import meth324_switch_reference as M
import meth327_switch_tensor_binding as T
import meth328_switch_native_contract as B

MANIFEST=M.DOC/'meth348_switch_head_a16_fresh_manifest.json'
MANIFEST_SHA='ecc09f1257bdc3917ae65942ad6e0f3f815eb0d265f44e6208cd11e7b232cc72'
COST=M.DOC/'meth346_switch_head_a16_cost_result.json'
COST_SHA='1d32e982f8392632f4e1745f3fec91b3cb59884aa3f679a085a5307eb1432fcf'
BOUND=M.DOC/'meth327_switch_tensor_binding_result.json'
BOUND_SHA='e72fc17b5527dc34df5c00ed7c44498b2ee4fc5d338ed8b130fff101a30c20a7'
OLD=M.DOC/'meth331_switch_router_diagnostic_result.json'
OLD_SHA='668a01a42276ee51955a9cde930f224cbb57180016bc7e88f1696226738948a1'
CONTROL=M.DOC/'meth347_switch_head_a16_contract_resume_result.json'
CONTROL_SHA='0242039aef21bbb087ae7c973a4bfe718507d8da2f80b8ba796d5d5ee8792c45'
RECOVERED=M.DOC/'meth338_switch_tensor_recovery_result.json'
RECOVERED_SHA='19ef2f987444d17396beea8e489d2fee341d64b61cff00714521ed806b407a7c'
PROTOCOL=M.DOC/'METH_349_SWITCH_HEAD_A16_FRESH_PREDICTION_PROTOCOL_20261003.md'
OUT=M.ROOT/'results/native_expert_scaling/meth349_switch_head_a16_fresh_prediction'


def trim():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetProcessWorkingSetSize.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
    assert kernel.SetProcessWorkingSetSize(kernel.GetCurrentProcess(),ctypes.c_size_t(-1).value,ctypes.c_size_t(-1).value)


def metrics(logits,target,span):
    assert logits.shape==(11,32128) and len(target)==11 and len(span)==8
    values=logits.astype(np.float64);shift=values.max(-1,keepdims=True)
    nll=np.log(np.exp(values-shift).sum(-1))+shift[:,0]-values[np.arange(11),target]
    choice=logits.argmax(-1)
    return {'mean_nll':float(nll.mean()),'mean_span_nll':float(nll[1:9].mean()),
            'correct_tokens':int(np.sum(choice==target)),'correct_span_tokens':int(np.sum(choice[1:9]==span)),
            'all_span_teacher_forced_correct':bool(np.array_equal(choice[1:9],span)),
            'greedy_teacher_forced_ids':choice.tolist(),'per_target_nll':nll.tolist()}


def paired_interval(values):
    rng=np.random.default_rng(343343);values=np.asarray(values,dtype=np.float64);assert values.shape==(24,)
    draws=values[rng.integers(0,24,size=(10000,24))].mean(-1)
    return {'mean':float(values.mean()),'one_sided_lower95':float(np.quantile(draws,.05)),
            'one_sided_upper95':float(np.quantile(draws,.95)),'bootstrap_unit':'book','draws':10000,'seed':343343}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';maximum=0
    result={'experiment':'METH-349-NEW-head-A16-original-donor-span-prediction','source_shards':[],'original_bridges':[],'native_bridges':[],'books':[],'commands':[]}
    def guard():
        nonlocal maximum
        rss=psutil.Process().memory_info().rss;maximum=max(maximum,rss)
        assert rss<=48<<30 and time.monotonic()-start<=3600,'quality_resource_guard'
    def stream_digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);guard()
        return h.hexdigest()
    try:
        for path in (Path(__file__),PROTOCOL,MANIFEST,COST,BOUND,OLD,CONTROL,RECOVERED,T.ACQUISITION,T.HEADERS,B.ENGINE,
                     Path(M.__file__),Path(T.__file__),Path(B.__file__)):M.committed(path)
        for path,sha in ((MANIFEST,MANIFEST_SHA),(COST,COST_SHA),(BOUND,BOUND_SHA),(OLD,OLD_SHA),(CONTROL,CONTROL_SHA),(RECOVERED,RECOVERED_SHA)):
            assert M.digest(path)==sha,str(path)
        cohort=json.loads(MANIFEST.read_text(encoding='utf-8'));cost=json.loads(COST.read_text(encoding='utf-8'));bound=json.loads(BOUND.read_text(encoding='utf-8'))
        old=json.loads(OLD.read_text(encoding='utf-8'));control=json.loads(CONTROL.read_text(encoding='utf-8'));recovered=json.loads(RECOVERED.read_text(encoding='utf-8'))
        assert all(cohort['gates'].values()) and all(cost['gates'].values()) and bound['passed']
        assert len({book['whole_source_utf8_sha256'] for book in cohort['items']})==24
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        binary=Path(cost['compile']['argv'][-1]);assert M.digest(binary)==cost['compile']['binary_sha256']
        assert M.digest(binary.parent/'libomp.dll')==cost['compile']['runtime_sha256']
        artifact=cost['artifact'];payload=Path(artifact['payload']);spec=Path(artifact['manifest']);payload_stat=(payload.stat().st_size,payload.stat().st_mtime_ns)
        assert payload_stat[0]==artifact['bytes'] and stream_digest(payload)==artifact['sha256'] and M.digest(spec)==artifact['manifest_sha256']
        for row in cohort['source_tokenizer_files']:assert stream_digest(row['path'])==row['sha256']
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'manifest348_sha256':MANIFEST_SHA,
                       'cost346_sha256':COST_SHA,'original327_binding_sha256':BOUND_SHA,'reference331_sha256':OLD_SHA,
                       'artifact':artifact,'native_binary_sha256':cost['compile']['binary_sha256'],'native_threads':6,'original_threads':1,
                       'installed_model_source_sha256':M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))})
        OUT.mkdir(parents=True);torch.set_num_threads(1)
        stage='fresh_all_original_coefficients_and_reference_model'
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**recovered['original_config']))
        namespace=model.state_dict();assert set(namespace)==set(bound['tensors']);seen=set()
        for shard in sorted(T.SOURCE.glob('pytorch_model-*-of-*.bin')):
            before=time.monotonic();state=torch.load(shard,weights_only=True,mmap=True,map_location='cpu')
            assert not seen.intersection(state)
            for ordinal,(name,value) in enumerate(sorted(state.items(),key=lambda item:item[1].data_ptr())):
                expected=bound['tensors'][name]
                assert value.dtype==torch.float32 and value.is_contiguous() and list(value.shape)==expected['shape']==list(namespace[name].shape)
                assert hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()==expected['sha256'],name
                seen.add(name)
                if ordinal%32==31:trim()
                guard()
            missing=model.load_state_dict(state,strict=False,assign=True);assert not missing.unexpected_keys
            del state;gc.collect();trim();guard();result['source_shards'].append({'name':shard.name,'tensor_names_so_far':len(seen),'seconds':time.monotonic()-before})
        assert len(seen)==6392 and seen==set(bound['tensors'])
        model.tie_weights();model.requires_grad_(False);model.eval()
        assert sum(p.numel() for p in model.parameters())==14664154368
        assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in model.parameters())
        result['original_source_identity']='Fresh ALL6392 canonical original F32 coefficient bytes327 plus config/tokenizer; no new ZIP envelope claim. Unmodified official CPU1 forward.'
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_NUM_THREADS':'6','OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def native(source_ids,decoder_ids,label):
            nonlocal maximum
            prefix=OUT/label;argv=[str(binary),str(spec),','.join(map(str,source_ids)),','.join(map(str,decoder_ids)),str(prefix),'6','0','0','1','0']
            with (OUT/(label+'.stdout.log')).open('wb') as stdout,(OUT/(label+'.stderr.log')).open('wb') as stderr:
                child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env)
                while child.poll() is None:
                    try:
                        rss=psutil.Process().memory_info().rss+psutil.Process(child.pid).memory_info().rss;maximum=max(maximum,rss)
                        if rss>48<<30 or time.monotonic()-start>3600:child.kill();child.wait();raise RuntimeError('native_quality_resource_guard')
                    except psutil.NoSuchProcess:pass
                    time.sleep(.1)
            result['commands'].append({'argv':argv,'returncode':child.returncode,'stdout_sha256':M.digest(OUT/(label+'.stdout.log')),'stderr_sha256':M.digest(OUT/(label+'.stderr.log'))})
            assert child.returncode==0,(label,child.returncode)
            output=Path(str(prefix)+'.0.bin');return B.read_output(output),M.digest(output)
        stage='fresh_original_and_native_consumed_bridges_before_NEW_scores'
        for index,case in enumerate(control['cases']):
            source=torch.tensor([case['source_ids']]);decoder=torch.tensor([case['decoder_ids']]);reference=B.torch_reference(model,source,decoder)
            cached=M.ROOT/f'results/native_expert_scaling/meth331_switch_router_diagnostic/case{index}.reference_arrays.npz'
            assert M.digest(cached)==old['cases'][index]['reference_array_sha256']
            with np.load(cached) as saved:checks={name:bool(np.array_equal(values,saved[name])) for name,values in zip(('encoder','decoder','logits','routes'),reference)}
            assert all(checks.values()),('original_bridge_changed',index,checks)
            result['original_bridges'].append({'case':index,'exact_cached331':checks})
            actual,output_sha=native(case['source_ids'],case['decoder_ids'],f'bridge{index}')
            assert output_sha==case['native_sha256'],'head_A16_native347_bridge_changed'
            result['native_bridges'].append({'case':index,'exact347_native':True,'sha256':output_sha})
            trim();guard()
        stage='NEW_all96_paired_span_predictions'
        for ordinal,book in enumerate(cohort['items']):
            record={'source_id':book['source_id'],'cases':[]}
            for case in book['cases']:
                label=f'book{ordinal}.case{case["index"]}';source=torch.tensor([case['source_ids']]);decoder=torch.tensor([case['decoder_ids']])
                reference=B.torch_reference(model,source,decoder);guard()
                saved=OUT/(label+'.original_reference.npz');np.savez(saved,encoder=reference[0],decoder=reference[1],logits=reference[2],routes=reference[3])
                actual,native_sha=native(case['source_ids'],case['decoder_ids'],label)
                original_metrics=metrics(reference[2],case['target_ids'],case['masked_span_ids']);target_metrics=metrics(actual[2],case['target_ids'],case['masked_span_ids'])
                agreement=float(np.mean(actual[2].argmax(-1)==reference[2].argmax(-1)))
                record['cases'].append({'index':case['index'],'original':original_metrics,'native':target_metrics,'original_top1_agreement':agreement,
                                        'changed_route_choices':int(np.sum(actual[3][:,0]!=reference[3][:,0])),
                                        'original_reference_sha256':M.digest(saved),'native_output_sha256':native_sha})
                del reference,actual;gc.collect();trim();guard()
            result['books'].append(record);print(json.dumps({'book':ordinal,'source_id':book['source_id'],'case_mean_nll_deltas':[c['native']['mean_nll']-c['original']['mean_nll'] for c in record['cases']],
                'agreements':[c['original_top1_agreement'] for c in record['cases']],'elapsed_seconds':time.monotonic()-start}),flush=True)
        nll=[];span_nll=[];span_accuracy=[];token_accuracy=[];agreement=[]
        for book in result['books']:
            cases=book['cases'];nll.append(np.mean([c['native']['mean_nll']-c['original']['mean_nll'] for c in cases]))
            span_nll.append(np.mean([c['native']['mean_span_nll']-c['original']['mean_span_nll'] for c in cases]))
            span_accuracy.append(np.mean([int(c['native']['all_span_teacher_forced_correct'])-int(c['original']['all_span_teacher_forced_correct']) for c in cases]))
            token_accuracy.append(np.mean([(c['native']['correct_span_tokens']-c['original']['correct_span_tokens'])/8 for c in cases]))
            agreement.append(np.mean([c['original_top1_agreement'] for c in cases]))
        result['paired']={'all_token_nll':paired_interval(nll),'masked_span_nll':paired_interval(span_nll),
                          'teacher_forced_span_exact_accuracy':paired_interval(span_accuracy),'masked_token_accuracy':paired_interval(token_accuracy),
                          'original_top1_agreement':float(np.mean(agreement))}
        result['gates']={'all_fresh_original_source_identity_and_bridges':True,'all_native_same_artifact_bridges':True,'all96_known_span_scores':len(result['books'])==24,
                         'all_token_nll_upper95_le0p05':result['paired']['all_token_nll']['one_sided_upper95']<=.05,
                         'masked_span_nll_upper95_le0p05':result['paired']['masked_span_nll']['one_sided_upper95']<=.05,
                         'masked_token_accuracy_lower95_ge_minus0p02':result['paired']['masked_token_accuracy']['one_sided_lower95']>=-.02,
                         'teacher_forced_span_exact_accuracy_lower95_ge_minus0p05':result['paired']['teacher_forced_span_exact_accuracy']['one_sided_lower95']>=-.05,
                         'original_top1_agreement_ge0p95':result['paired']['original_top1_agreement']>=.95}
        assert payload_stat==(payload.stat().st_size,payload.stat().st_mtime_ns)
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum,'end_rss_bytes':psutil.Process().memory_info().rss}
        result['decision']='eligible_for_separate_NEW_free_span_generation_task_and_native_accepted_rate' if all(result['gates'].values()) else 'fixed_head_A16_candidate_prediction_not_qualified_preserve_failure_before_diagnostics'
        result['scope']='96 known8-token masked spans from24 NEW project-heldout PG19book rows, book bootstrap. Primary ORIGINAL unmodified full14.664B source, actual head-A16 native same346 timed executable/target. Teacher-forced prediction/reconstruction only, no free generation quality, useful larger-n/LUT/DRAM, accepted rate or cross-family proof.'
        guard();M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'gates':result['gates'],'paired':result['paired'],'resource':result['resource'],'decision':result['decision']}),flush=True)
    except BaseException as error:
        result.update({'stage':stage,'error':repr(error),'seconds':time.monotonic()-start,'maximum_checked_combined_rss_bytes':maximum});M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
