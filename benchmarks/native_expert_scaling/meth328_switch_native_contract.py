"""Freezeable complete native Switch FP32 bridge vs official tiny reference."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import psutil
import numpy as np
import torch
import transformers
from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router,SwitchTransformersAttention
import meth324_switch_reference as M

SOURCE=M.ROOT/'benchmarks/native_expert_scaling/meth328_switch_f32_reference.c'
ENGINE=M.ROOT/'benchmarks/phase60/engine.c'
PROTOCOL=M.DOC/'METH_328_SWITCH_NATIVE_CONTRACT_PROTOCOL_20261003.md'
REFERENCE=M.DOC/'meth324_switch_reference_result.json'
REFERENCE_SHA='9420d46235cdcf2e48a1153ae250aac59deecee12d6d243e0319ed33fb91f082'
OUT=M.ROOT/'results/native_expert_scaling/meth328_switch_native_contract'
COMPILER=Path(r'C:\Users\giosa\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20251216-msvcrt-x86_64\bin\clang.exe')
COMPILER_SHA='8ba7ddd7fce5574275dec0302caa1eb9f3d9ae812c6e1f44276a27b6d14fb9b7'
DLL=COMPILER.parent/'libomp.dll'
DLL_SHA='6fc163dd513538a92a187d987438bebc5509afd1f824b20a88dd51463b3d5698'


def pack_string(stream,text):
    data=text.encode('utf-8');stream.write(struct.pack('<I',len(data)));stream.write(data)


def export(model,directory):
    directory.mkdir()
    weights=directory/'weights.bin';spec=directory/'manifest.bin';entries=[]
    state=model.state_dict()
    with weights.open('xb') as stream:
        for name,tensor in state.items():
            assert tensor.dtype==torch.float32 and tensor.ndim in (1,2) and tensor.is_contiguous()
            padding=(-stream.tell())%64;stream.write(b'\0'*padding);offset=stream.tell()
            stream.write(memoryview(tensor.numpy()).cast('B'))
            entries.append((name,list(tensor.shape),offset,tensor.numel()))
    cfg=model.config
    values=(cfg.d_model,cfg.d_ff,cfg.num_heads,cfg.d_kv,cfg.num_layers,cfg.num_decoder_layers,cfg.num_experts,
            cfg.expert_capacity,cfg.vocab_size,cfg.relative_attention_num_buckets,cfg.relative_attention_max_distance,
            cfg.encoder_sparse_step,cfg.decoder_sparse_step)
    with spec.open('xb') as stream:
        stream.write(b'SWF32A01');stream.write(struct.pack('<13IfII',*values,cfg.layer_norm_epsilon,1,len(entries)))
        pack_string(stream,str(weights.resolve()))
        for name,shape,offset,elements in entries:
            pack_string(stream,name);stream.write(struct.pack('<4I2Q',0,len(shape),shape[0],shape[1] if len(shape)==2 else 1,offset,elements))
    return {'spec':str(spec),'manifest_sha256':M.digest(spec),'weights':str(weights),'weight_bytes':weights.stat().st_size,
            'weights_sha256':M.digest(weights),'tensor_names':len(entries)}


def torch_reference(model,source,decoder):
    encoder_states=[model.shared(source).detach().numpy()[0].copy()]
    decoder_states=[];route=[];current=[];hooks=[]
    def encoder_hook(_module,_args,output):encoder_states.append(output[0].detach().numpy()[0].copy())
    def decoder_hook(_module,_args,output):current.append(output[0].detach().numpy()[0,0].copy())
    def router_hook(_module,_args,output):
        mask,prob,logits=output
        chosen=logits.argmax(dim=-1).reshape(-1).tolist();accepted=mask.any(dim=-1).reshape(-1).tolist();p=prob.reshape(-1).tolist()
        route.extend(zip(chosen,accepted,p))
    for block in model.encoder.block:hooks.append(block.register_forward_hook(encoder_hook))
    for block in model.decoder.block:hooks.append(block.register_forward_hook(decoder_hook))
    for module in model.modules():
        if type(module) is SwitchTransformersTop1Router:hooks.append(module.register_forward_hook(router_hook))
    logits=[];cache=None
    with torch.no_grad():
        encoded=model.encoder(input_ids=source,return_dict=True)
        encoder_states.append(encoded.last_hidden_state.numpy()[0].copy())
        for step in range(decoder.shape[1]):
            current.clear();current.append(model.shared(decoder[:,step:step+1]).numpy()[0,0].copy())
            output=model(encoder_outputs=encoded,decoder_input_ids=decoder[:,step:step+1],past_key_values=cache,use_cache=True,output_hidden_states=True)
            cache=output.past_key_values;current.append(output.decoder_hidden_states[-1].numpy()[0,0].copy())
            decoder_states.append(np.stack(current));logits.append(output.logits.numpy()[0,0].copy())
    for hook in hooks:hook.remove()
    return np.stack(encoder_states),np.stack(decoder_states),np.stack(logits),np.array(route,dtype=np.float64)


def read_output(path):
    data=path.read_bytes();assert data[:8]==b'SWR32O01'
    s,t,d,enc,dec,vocab,nroutes=struct.unpack_from('<7I',data,8);offset=36
    def floats(shape):
        nonlocal offset
        count=int(np.prod(shape));result=np.frombuffer(data,dtype='<f4',count=count,offset=offset).reshape(shape).copy();offset+=count*4;return result
    a=floats((enc+2,s,d));b=floats((t,dec+2,d));c=floats((t,vocab))
    routes=np.array([struct.unpack_from('<iif',data,offset+12*i) for i in range(nroutes)],dtype=np.float64)
    assert len(data)==offset+12*nroutes
    return a,b,c,routes


def relative(a,b):
    denominator=float(np.linalg.norm(b.astype(np.float64)));assert denominator>0
    return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/denominator)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();stage='bindings';result={'experiment':'METH-328-complete-native-Switch-FP32-contract','cases':[]}
    maximum_rss=0
    try:
        for path in (Path(__file__),SOURCE,ENGINE,PROTOCOL,REFERENCE,Path(M.__file__)):M.committed(path)
        assert M.digest(REFERENCE)==REFERENCE_SHA and json.loads(REFERENCE.read_text())['passed']
        assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
        assert M.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        assert M.digest(COMPILER)==COMPILER_SHA and M.digest(DLL)==DLL_SHA
        result.update({'controller_sha256':M.digest(__file__),'source_sha256':M.digest(SOURCE),
            'engine_sha256':M.digest(ENGINE),'protocol_sha256':M.digest(PROTOCOL),'reference_sha256':REFERENCE_SHA,
            'compiler_sha256':COMPILER_SHA,'runtime_sha256':DLL_SHA})
        OUT.mkdir(parents=True);shutil.copyfile(DLL,OUT/'libomp.dll');binary=OUT/'meth328_switch_f32_reference.exe'
        command=[str(COMPILER),'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',
                 '-DSILICON_SWITCH_F32_REFERENCE',str(ENGINE),'-o',str(binary)]
        completed=subprocess.run(command,capture_output=True,timeout=120)
        (OUT/'compile.stdout.log').write_bytes(completed.stdout);(OUT/'compile.stderr.log').write_bytes(completed.stderr)
        assert completed.returncode==0,completed.stderr.decode(errors='replace')
        result['compile']={'argv':command,'returncode':completed.returncode,'binary_sha256':M.digest(binary)}
        env=os.environ.copy();env.pop('OMP_PROC_BIND',None)
        env.update({'OMP_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE','KMP_AFFINITY':'none'})
        def run(argv,label):
            nonlocal maximum_rss
            with (OUT/(label+'.stdout.log')).open('wb') as stdout,(OUT/(label+'.stderr.log')).open('wb') as stderr:
                process=subprocess.Popen([str(binary),*argv],stdout=stdout,stderr=stderr,env=env)
                while process.poll() is None:
                    try:
                        rss=psutil.Process(process.pid).memory_info().rss+psutil.Process().memory_info().rss;maximum_rss=max(maximum_rss,rss)
                        if rss>3<<30 or time.monotonic()-start>600:process.kill();process.wait();raise RuntimeError('resource_guard')
                    except psutil.NoSuchProcess:pass
                    time.sleep(.1)
            assert process.returncode==0,(label,process.returncode)
            return (OUT/(label+'.stdout.log')).read_bytes()
        stage='buckets'
        observed=json.loads(run(['--buckets'],'buckets'))
        positions=torch.tensor([-4096,-2048,-128,-64,-16,-8,-1,0,1,8,16,64,128,2048,4096])
        expected=[]
        for bidi in (False,True):expected.extend(SwitchTransformersAttention._relative_position_bucket(positions,bidirectional=bidi,num_buckets=32,max_distance=128).tolist())
        result['relative_buckets']={'observed':observed,'expected':expected,'passed':observed==expected}
        torch.set_num_threads(1)
        source=torch.tensor([[2,3,4,5,6,7]]);decoder=torch.tensor([[0,8,9,10]])
        for capacity in (1,64):
            stage=f'capacity{capacity}';torch.manual_seed(328)
            cfg=SwitchTransformersConfig(d_model=8,d_ff=16,d_kv=4,num_heads=2,num_experts=2,
                num_layers=2,num_decoder_layers=2,num_sparse_encoder_layers=1,num_sparse_decoder_layers=1,
                expert_capacity=capacity,dropout_rate=0.,router_jitter_noise=0.,vocab_size=32,
                pad_token_id=0,eos_token_id=1,decoder_start_token_id=0,router_dtype='float32')
            model=SwitchTransformersForConditionalGeneration(cfg).eval()
            artifact=export(model,OUT/f'capacity{capacity}')
            reference=torch_reference(model,source,decoder)
            baseline_file=OUT/f'capacity{capacity}.output.bin'
            argv=[artifact['spec'],'2,3,4,5,6,7','0,8,9,10',str(baseline_file),'0'];run(argv,f'capacity{capacity}')
            native=read_output(baseline_file)
            errors=[relative(a,b) for a,b in zip(native[:3],reference[:3])]
            per_encoder=[relative(a,b) for a,b in zip(native[0],reference[0])]
            per_decoder=[relative(a,b) for a,b in zip(native[1].reshape(-1,8),reference[1].reshape(-1,8))]
            routes=native[3];actual=reference[3]
            exact_route=bool(np.array_equal(routes[:,:2],actual[:,:2]));gate_error=float(np.max(np.abs(routes[:,2]-actual[:,2])))
            top1=bool(np.array_equal(native[2].argmax(-1),reference[2].argmax(-1)))
            faults=[]
            for number in (1,2,3,4,5):
                if number==4 and capacity==64:continue
                fault_file=OUT/f'capacity{capacity}.fault{number}.bin'
                run([artifact['spec'],argv[1],argv[2],str(fault_file),str(number)],f'capacity{capacity}_fault{number}')
                output=read_output(fault_file);error=relative(output[2],reference[2])
                faults.append({'fault':number,'relative_logit_l2':error,'detected':error>1e-4})
            entry={'capacity':capacity,'tiny_config':cfg.to_dict(),'artifact':artifact,'baseline_output_sha256':M.digest(baseline_file),
                'encoder_decoder_logit_relative_l2':errors,'encoder_state_errors':per_encoder,'decoder_state_errors':per_decoder,
                'exact_route_choice_and_acceptance':exact_route,'maximum_selected_probability_abs_error':gate_error,
                'greedy_top1_equal':top1,'reference_dropped_routes':int(np.sum(actual[:,1]==0)),
                'faults':faults,'passed':max(errors+per_encoder+per_decoder)<=1e-4 and exact_route and gate_error<=1e-6 and top1 and all(v['detected'] for v in faults)}
            assert capacity!=1 or entry['reference_dropped_routes']>0
            result['cases'].append(entry)
            print(json.dumps({'capacity':capacity,'errors':errors,'route':exact_route,'gate_error':gate_error,'faults':faults,'passed':entry['passed']}),flush=True)
            del model
        result['gates']={'relative_bucket_boundaries':result['relative_buckets']['passed'],
                         'all_complete_native_tiny_contracts':all(v['passed'] for v in result['cases'])}
        result['resource']={'seconds':time.monotonic()-start,'maximum_sampled_parent_child_rss_bytes':maximum_rss,
                            'end_rss_bytes':psutil.Process().memory_info().rss}
        assert result['resource']['seconds']<=600 and result['resource']['end_rss_bytes']<=3<<30
        result['decision']='eligible_for_separate_actual_source_native_binding' if all(result['gates'].values()) else 'unchanged_native_bridge_not_qualified'
        result['scope']='Full C encoder/decoder/cache/head and capacity semantics on fixed tiny synthetic weights, through opt-in engine prefix. No real source quality, useful parameter transfer, full-size native cost, precision/LUT or accepted token rate. F32 bridge alone not final compact representation.'
        M.write(args.out,result);print(json.dumps({'sha256':M.digest(args.out),'gates':result['gates'],'resource':result['resource'],'decision':result['decision']}),flush=True)
    except BaseException as error:
        result.update({'error':repr(error),'stage':stage,'seconds':time.monotonic()-start})
        M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
