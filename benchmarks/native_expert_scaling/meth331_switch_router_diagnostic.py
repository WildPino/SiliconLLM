"""Actual state decomposition of330 router probability mismatch, no repair."""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import numpy as np
import psutil
import torch
import transformers
from transformers import AutoTokenizer,SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router
import meth324_switch_reference as M
import meth327_switch_tensor_binding as T
import meth328_switch_native_contract as B

BASELINE=M.DOC/'meth330_switch_full_source_result.json'
BASELINE_SHA='b28dc0f5a8a289be7aa0c23270d3abba2f7b67d07904dc576153618e0bc46745'
PROTOCOL=M.DOC/'METH_331_SWITCH_ROUTER_DIAGNOSTIC_PROTOCOL_20261003.md'
ARITHMETIC=M.ROOT/'benchmarks/native_expert_scaling/meth331_switch_router_trace.c'
ENTRY=M.ROOT/'benchmarks/native_expert_scaling/meth331_switch_router_trace_entry.c'
OLD=M.ROOT/'benchmarks/native_expert_scaling/meth330_switch_stable_rms.c'
OUT=M.ROOT/'results/native_expert_scaling/meth331_switch_router_diagnostic'


def read_trace(path):
    data=path.read_bytes();offset=0;inputs=[];logits=[]
    while offset<len(data):
        d,n=struct.unpack_from('<2I',data,offset);offset+=8;assert (d,n)==(768,256)
        inputs.append(np.frombuffer(data,dtype='<f4',count=d,offset=offset).copy());offset+=d*4
        logits.append(np.frombuffer(data,dtype='<f4',count=n,offset=offset).copy());offset+=n*4
    assert offset==len(data)
    return np.stack(inputs),np.stack(logits)


def softmax64(logits):
    logits=logits.astype(np.float64);value=np.exp(logits-logits.max(axis=-1,keepdims=True));return value/value.sum(axis=-1,keepdims=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';peak=0;result={'experiment':'METH-331-actual-router-error-decomposition','cases':[]}
    try:
        for path in (Path(__file__),PROTOCOL,BASELINE,ARITHMETIC,ENTRY,OLD,B.ENGINE,Path(M.__file__),Path(T.__file__),Path(B.__file__)):M.committed(path)
        assert M.digest(BASELINE)==BASELINE_SHA
        base=json.loads(BASELINE.read_text());assert base['gates']['all_native_actual_source_tensors'] and not base['gates']['complete_actual_source_native_controls']
        line='static FILE *router_trace;\nstatic void trace_router(int d,int n,const float *input,const float *logits){if(router_trace){uint32_t dimensions[2]={(uint32_t)d,(uint32_t)n};fwrite(dimensions,4,2,router_trace);fwrite(input,4,d,router_trace);fwrite(logits,4,n,router_trace);}}\n'
        expected=OLD.read_text().replace('static void feed(Config c,Block *b,float *hidden,int tokens){',line+'static void feed(Config c,Block *b,float *hidden,int tokens){').replace('mv(b->router,x,logits,c.n,c.d);int chosen=0;','mv(b->router,x,logits,c.n,c.d);trace_router(c.d,c.n,x,logits);int chosen=0;')
        assert ARITHMETIC.read_text()==expected and M.digest(OLD)==base['arithmetic_source_sha256']
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        assert M.digest(B.COMPILER)==B.COMPILER_SHA and M.digest(B.DLL)==B.DLL_SHA
        spec=Path(base['native_manifest']['path']);assert M.digest(spec)==base['native_manifest']['sha256']
        acquisition=json.loads(T.ACQUISITION.read_text())
        for item in acquisition['files']:assert T.file_digest(Path(item['path']),start)==item['sha256']
        cfg=SwitchTransformersConfig(**json.loads((T.SOURCE/'config.json').read_text()))
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(cfg)
        for item in base['original_config'],:assert item['num_experts']==256
        for shard in sorted(T.SOURCE.glob('pytorch_model-*-of-*.bin')):
            state=torch.load(shard,map_location='cpu',weights_only=True,mmap=True);extra=model.load_state_dict(state,strict=False,assign=True)
            assert not extra.unexpected_keys;del state;gc.collect()
        model.tie_weights();model.eval();model.requires_grad_(False);torch.set_num_threads(1)
        assert sum(p.numel() for p in model.parameters())==14664154368 and all(p.device.type=='cpu' for p in model.parameters())
        tokenizer=AutoTokenizer.from_pretrained(T.SOURCE,use_fast=True,local_files_only=True,trust_remote_code=False)
        OUT.mkdir(parents=True);shutil.copyfile(B.DLL,OUT/'libomp.dll');binary=OUT/'meth331_switch_router_trace.exe'
        command=[str(B.COMPILER),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',
            '-DSILICON_SWITCH_ROUTER_TRACE',str(B.ENGINE),'-o',str(binary)]
        compiled=subprocess.run(command,capture_output=True,timeout=120)
        (OUT/'compile.stdout.log').write_bytes(compiled.stdout);(OUT/'compile.stderr.log').write_bytes(compiled.stderr)
        assert compiled.returncode==0,compiled.stderr.decode(errors='replace')
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),'baseline_sha256':BASELINE_SHA,
            'arithmetic_sha256':M.digest(ARITHMETIC),'entry_sha256':M.digest(ENTRY),'engine_sha256':M.digest(B.ENGINE),
            'native_binary_sha256':M.digest(binary),'native_manifest_sha256':M.digest(spec),'compile_argv':command})
        names={module:name for name,module in model.named_modules() if type(module) is SwitchTransformersTop1Router}
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None);env.update({'OMP_NUM_THREADS':'1','KMP_AFFINITY':'none','OMP_WAIT_POLICY':'PASSIVE'})
        for index,previous in enumerate(base['cases']):
            stage=f'case{index}';calls=[];hooks=[]
            def hook(module,inputs,output):
                mask,prob,logits=output
                calls.append({'name':names[module],'input':inputs[0].detach().numpy()[0].copy(),
                    'weight':module.classifier.weight.detach().numpy(),'logits':logits.detach().numpy()[0].copy(),
                    'prob':prob.detach().numpy()[0,:,0].copy(),'choice':logits.argmax(-1).numpy()[0].copy()})
            for module in names:hooks.append(module.register_forward_hook(hook))
            source=torch.tensor([previous['source_ids']]);decoder=torch.tensor([previous['decoder_ids']])
            assert tokenizer(previous['prompt'],return_tensors='pt')['input_ids'].tolist()==source.tolist()
            reference=B.torch_reference(model,source,decoder)
            for handle in hooks:handle.remove()
            native_output=OUT/f'case{index}.bin';trace=OUT/f'case{index}.trace.bin'
            argv=[str(binary),str(spec),','.join(map(str,source[0].tolist())),','.join(map(str,decoder[0].tolist())),str(native_output),'0',str(trace)]
            with (OUT/f'case{index}.stdout.log').open('wb') as stdout,(OUT/f'case{index}.stderr.log').open('wb') as stderr:
                child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env)
                while child.poll() is None:
                    rss=psutil.Process().memory_info().rss
                    try:rss+=psutil.Process(child.pid).memory_info().rss
                    except psutil.NoSuchProcess:pass
                    peak=max(peak,rss)
                    if rss>32<<30 or time.monotonic()-start>1200:child.kill();child.wait();raise RuntimeError('resource_guard')
                    time.sleep(.25)
            assert child.returncode==0 and M.digest(native_output)==previous['native_output_sha256'],'unchanged330_native_output_required'
            x_native,l_native=read_trace(trace);native=B.read_output(native_output)
            assert len(x_native)==len(native[3])==sum(len(c['input']) for c in calls)
            entries=[];position=0;x_torch=[];l_torch=[];p_original=[];p_replay=[];p_native64=[];p_torch64=[];banks=[]
            for call in calls:
                length=len(call['input']);a=x_native[position:position+length];z=l_native[position:position+length]
                weight=call['weight'];chosen=call['choice'];assert np.array_equal(native[3][position:position+length,0],chosen)
                with torch.no_grad():replay=torch.nn.functional.linear(torch.from_numpy(a.copy())[None],torch.from_numpy(weight)).numpy()[0]
                replay_prob=torch.softmax(torch.from_numpy(replay),dim=-1).numpy()[np.arange(length),chosen]
                exact_a=a.astype(np.float64)@weight.astype(np.float64).T
                exact_b=call['input'].astype(np.float64)@weight.astype(np.float64).T
                pa=softmax64(exact_a)[np.arange(length),chosen];pb=softmax64(exact_b)[np.arange(length),chosen]
                pc=softmax64(z)[np.arange(length),chosen]
                for row in range(length):
                    observed=float(native[3][position+row,2]);original=float(call['prob'][row]);reproduced=float(replay_prob[row])
                    entries.append({'ordinal':position+row,'bank':call['name'],'token_in_call':row,'choice':int(chosen[row]),
                        'observed_probability_signed_error':observed-original,
                        'classifier_softmax_signed_error_at_same_native_input':observed-reproduced,
                        'upstream_input_signed_error_replayed_official':reproduced-original,
                        'identity_decomposition_error':abs((observed-reproduced)+(reproduced-original)-(observed-original)),
                        'F64_dot_softmax_at_native_input_error_to_official':float(pa[row])-original,
                        'F64_dot_softmax_at_original_input_error_to_official':float(pb[row])-original,
                        'native_selected_softmax_rounding_error':observed-float(pc[row]),
                        'native_raw_logit_maxabs_to_F64_same_input':float(np.max(np.abs(z[row]-exact_a[row]))),
                        'official_raw_logit_maxabs_to_F64_same_input':float(np.max(np.abs(call['logits'][row]-exact_b[row]))),
                        'native_input_relative_l2':B.relative(a[row],call['input'][row])})
                x_torch.append(call['input']);l_torch.append(call['logits']);p_original.extend(call['prob'].tolist())
                p_replay.extend(replay_prob.tolist());p_native64.extend(pa.tolist());p_torch64.extend(pb.tolist());banks.extend([call['name']]*length)
                position+=length
            assert position==len(x_native) and all(v['identity_decomposition_error']==0 for v in entries)
            observed=max(abs(v['observed_probability_signed_error']) for v in entries)
            assert observed==previous['maximum_selected_probability_abs_error']
            arrays=OUT/f'case{index}.reference_arrays.npz'
            np.savez(arrays,encoder=reference[0],decoder=reference[1],logits=reference[2],routes=reference[3],
                original_router_inputs=np.concatenate(x_torch),original_router_logits=np.concatenate(l_torch),
                original_probabilities=np.array(p_original),replay_probabilities=np.array(p_replay),
                F64_native_input_probabilities=np.array(p_native64),F64_original_input_probabilities=np.array(p_torch64),
                native_router_inputs=x_native,native_router_logits=l_native)
            worst=max(entries,key=lambda v:abs(v['observed_probability_signed_error']))
            entry={'index':index,'native_output_exact330':True,'trace_sha256':M.digest(trace),'reference_array_sha256':M.digest(arrays),
                'router_rows':len(entries),'observed_maxabs':observed,'worst_row':worst,'rows':entries,
                'F64_router_counterfactual_maxabs':max(abs(v['F64_dot_softmax_at_native_input_error_to_official']) for v in entries)}
            result['cases'].append(entry);print(json.dumps({k:entry[k] for k in ('index','router_rows','observed_maxabs','worst_row','F64_router_counterfactual_maxabs')}),flush=True)
        result['resource']={'main_seconds_excluding_imports':time.monotonic()-start,'maximum_sampled_combined_rss_bytes':peak,'end_controller_rss_bytes':psutil.Process().memory_info().rss}
        assert time.monotonic()-start<=1200 and result['resource']['end_controller_rss_bytes']<=32<<30
        result['decision']='diagnostic_only_preserve_original_probability_failure_choose_next_specific_arithmetic_variable'
        result['scope']='Unchanged330 outputs byte exact and actual native router inputs/raw logits captured. Replay and F64 counterfactual separate same-input classifier/softmax error from upstream-state error; no whole model repair, threshold waiver, quality/native rate or general causality claim.'
        M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'resource':result['resource'],'decision':result['decision']}),flush=True)
    except BaseException as error:
        result.update({'error':repr(error),'stage':stage,'main_seconds_excluding_imports':time.monotonic()-start})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
