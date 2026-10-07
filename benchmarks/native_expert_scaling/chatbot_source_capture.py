"""One-pass original Qwen own-history pre-MLP/complete-FFN calibration capture.

Exclusive conversation files and a per-forward journal retain completed prefixes
on any first fault. No fitting or native calls. Direct Python with pinned local
packages; the monitoring launcher controls deadline and through-exit OS peak.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def save(path,value):
    raw=(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
    assert len(raw)<=2<<20
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


def main(args):
    start=time.monotonic()
    import psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    r=dict(schema='QWEN_ORIGINAL_CHAT_CAPTURE_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        scope='CALIBRATION_AND_RESERVED_DEVELOPMENT_NOT_MODEL_QUALITY',conversations=[],gates={})
    torch=None; stage='binding'; current=None
    def resource():
        return dict(elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_peak_reserved=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved())
    def guard():
        z=resource()
        assert z['elapsed_seconds']<=300 and z['OS_peak_snapshot']<=12<<30,'worker deadline/RSS budget'
        assert z['GPU_peak_allocated']<=9<<30 and z['GPU_peak_reserved']<=10<<30,'GPU budget'
        assert not proc.children(recursive=True),'unexpected worker children'
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes())
        assert b['schema']=='QWEN_ORIGINAL_CAPTURE_BINDING_V1'
        for item in b['inputs']:
            assert str(Path(item['path']).resolve())==item['resolved_path'],'logical source/input mapping changed'
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path']
            guard()
        args.directory.mkdir() # caller must select a fresh namespace
        r['gates']['frozen_input_files']=True
        # Identify and prevent Python subprocesses BEFORE creation. The capture
        # contract has no subprocess work; retain argv/stack, never environment.
        def reject_subprocess(event,arguments):
            if event=='subprocess.Popen':
                attempted=dict(executable=str(arguments[0]),argv=str(arguments[1]),stage=stage,
                    stack=traceback.format_stack(limit=12))
                r.setdefault('blocked_subprocess_attempts',[]).append(attempted)
                print(json.dumps(dict(blocked_subprocess_attempt=attempted)),flush=True)
                raise RuntimeError('Subprocess forbidden by original single-process capture contract')
        sys.addaudithook(reject_subprocess)
        stage='imports'
        print(json.dumps(dict(stage=stage,phase='before_numpy_torch_HF_imports')),flush=True)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from chatbot_interaction import serialize,terminate_generated
        from chatbot_capture_cases import manifest
        import transformers,tokenizers
        import importlib.util
        assert importlib.util.find_spec('pyarrow') is None and importlib.util.find_spec('datasets') is None
        assert not any(name=='pyarrow' or name.startswith('pyarrow.') for name in sys.modules)
        assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='5.13.1' and tokenizers.__version__=='0.22.2' and np.__version__=='2.4.6'
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3060'
        device=torch.device('cuda:0');torch.cuda.set_device(device)
        torch.set_num_threads(6);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats()
        source=Path(b['source_directory'])
        print(json.dumps(dict(stage=stage,phase='runtime_and_GPU_ready')),flush=True)
        spec=manifest();r['manifest']=spec
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
        stage='original_model_load'
        print(json.dumps(dict(stage=stage,phase='before_source_decode')),flush=True)
        model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
        for value in model.parameters():value.requires_grad_(False)
        assert (model.config.hidden_size,model.config.intermediate_size,model.config.num_hidden_layers,model.config.vocab_size)==(896,4864,24,151936)
        assert model.config.tie_word_embeddings and model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
        assert model.config.hidden_act=='silu' and model.config.rms_norm_eps==1e-6
        assert model.config.num_attention_heads==14 and model.config.num_key_value_heads==2
        assert model.config.use_sliding_window is False
        # Preserve pure-original module identity: no added experts/core wrappers.
        captured={};hooks=[]
        for layer_id,layer in enumerate(model.model.layers):
            assert layer.mlp.__class__.__name__=='Qwen2MLP'
            assert layer.mlp.gate_proj.bias is layer.mlp.up_proj.bias is layer.mlp.down_proj.bias is None
            def hook(module,inputs,output,layer_id=layer_id):
                x=inputs[0]
                assert x.ndim==3 and x.shape[0]==1 and x.shape[-1]==896 and output.shape==x.shape
                assert x.dtype==output.dtype==torch.bfloat16 and x.device==output.device==device
                assert layer_id not in captured,'duplicate MLP call in a source forward'
                captured[layer_id]=(x.detach(),output.detach())
            hooks.append(layer.mlp.register_forward_hook(hook))
        r['gates']['pure_original_source_and_tied_head']=True
        r['gates']['unrelated_Arrow_dataset_packages_absent']=True
        guard();stage='capture'
        print(json.dumps(dict(stage=stage,phase='before_first_source_forward')),flush=True)
        total_bytes=0
        with torch.inference_mode():
            for case in spec['cases']:
                current=dict(id=case['id'],split=case['split'],category=case['category'],mode=case['mode'],
                    messages=case['messages'],frames=[],generated_ids=[])
                r['pending_conversation']=current
                rendered=serialize(tokenizer,case['messages'],case['mode'])
                prompt=tokenizer.encode(rendered,add_special_tokens=False)
                assert 0<len(prompt)<=spec['max_prompt_tokens'] and all(0<=n<151936 for n in prompt)
                current.update(rendered_text=rendered,prompt_ids=prompt,
                    rendered_UTF8_SHA256=hashlib.sha256(rendered.encode('utf8')).hexdigest(),
                    prompt_U32LE_SHA256=hashlib.sha256(struct.pack('<'+'I'*len(prompt),*prompt)).hexdigest())
                binary=args.directory/(case['id']+'.bf16.bin')
                journal=args.directory/(case['id']+'.frames.jsonl')
                cache=None;input_ids=prompt
                with binary.open('xb') as stream,journal.open('xb') as log:
                    stream.write(struct.pack('<8s4I',b'QWCAP001',24,896,2,0))
                    for step in range(spec['max_new_tokens']):
                        guard();captured.clear()
                        ids=torch.tensor(input_ids,dtype=torch.int64,device=device)[None]
                        result=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
                        assert sorted(captured)==list(range(24)) and result.logits.shape==(1,1,151936)
                        assert torch.isfinite(result.logits).all()
                        next_id=int(result.logits[0,0].argmax().item())
                        # Shape [input rows,24 layers,2 operands(x,y),896]. Original
                        # BF16 values are exported as little-endian U16 bits in C order.
                        frame=torch.stack([torch.stack((captured[i][0][0],captured[i][1][0]),dim=1) for i in range(24)],dim=1)
                        bits=frame.contiguous().cpu().view(torch.uint16).numpy()
                        assert bits.dtype==np.dtype('<u2') and bits.flags.c_contiguous and bits.shape==(len(input_ids),24,2,896)
                        assert not ((bits & 0x7fff)>=0x7f80).any(),'nonfinite captured original operand'
                        raw=bits.tobytes(order='C')
                        assert total_bytes+len(raw)<=2<<30,'capture output cap'
                        row=dict(step=step,input_ids=input_ids,shape=list(bits.shape),dtype='<u2 BF16 bits',order='C',
                            offset=stream.tell(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),next_id=next_id)
                        stream.write(raw);stream.flush();os.fsync(stream.fileno())
                        log.write((json.dumps(row,separators=(',',':'))+'\n').encode('utf8'));log.flush();os.fsync(log.fileno())
                        total_bytes+=len(raw);current['frames'].append(row);current['generated_ids'].append(next_id)
                        stopped=terminate_generated(current['generated_ids'],spec['EOS'],spec['max_new_tokens'])
                        cache=result.past_key_values
                        del frame,bits,raw,result
                        captured.clear()
                        guard()
                        if stopped['termination'] in ('eos','length'):break
                        input_ids=[next_id]
                current.update(binary_path=str(binary),binary_SHA256=sha(binary),binary_bytes=binary.stat().st_size,
                    journal_path=str(journal),journal_SHA256=sha(journal),
                    accepted_generated_ids=stopped['accepted_ids'],termination=stopped['termination'],
                    displayed_text=tokenizer.decode(stopped['accepted_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False),
                    captured_rows=sum(a['shape'][0] for a in current['frames']))
                assert current['captured_rows']==len(prompt)+len(current['generated_ids'])-1
                save(args.directory/(case['id']+'.json'),current)
                r['conversations'].append(current);r.pop('pending_conversation')
                print(json.dumps(dict(id=case['id'],split=case['split'],captured_rows=current['captured_rows'],resource=resource())),flush=True)
                del cache;current=None
        for hook in hooks:hook.remove()
        assert len(r['conversations'])==48 and sum(c['split']=='fit' for c in r['conversations'])==32
        assert sum(c['split']=='development' for c in r['conversations'])==16
        r['gates']['all_frozen_original_conversations_captured']=True
        r.update(capture_payload_bytes=total_bytes,decision='ORIGINAL_SOURCE_CALIBRATION_AVAILABLE_NOT_CONVERTER_QUALIFIED',
            resource_before_final_serialization=resource(),ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            runtime=dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,numpy=np.__version__,
                dtype='BF16 source eager attention; capture exact original operand bits',TF32=False,gpu=torch.cuda.get_device_name()))
        guard();save(args.out,r)
        print(json.dumps(dict(decision=r['decision'],conversations=len(r['conversations']),resource=resource())),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,resource_snapshot=resource(),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save(args.out.with_suffix('.failure.json'),r)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--binding-sha',required=True)
    parser.add_argument('--freeze',required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
