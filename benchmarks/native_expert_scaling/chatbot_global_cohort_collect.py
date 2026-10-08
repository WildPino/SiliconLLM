"""One-pass NEW original CHATBOT x/full-vocabulary teacher labels for global fit.

Generated decision positions only; all input x retained for ALL24 calibration.
No old source query, source FFN y/J/H, target field, fitting or endpoint query.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'))
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import sha,write_once


def main(args):
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(11)))
    torch=None;case_start=None;stage='binding'
    r=dict(schema='QWEN_GLOBAL_COHORT_COLLECT_RESULT_V1',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
        procedure_gates={},conversations=[],new_original_BF16_full_forwards=0,new_old_source_queries=0,
        new_source_FFN_y_targets=0,new_student_responses=0,new_optimizer_updates=0,new_endpoint_queries=0)
    def resource():
        return dict(elapsed_seconds=time.monotonic()-start,OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_peak_allocated=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_peak_reserved=0 if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved())
    def guard():
        v=resource();assert v['elapsed_seconds']<=900 and v['OS_peak_snapshot']<=16<<30,'collector time/OS'
        assert v['GPU_peak_allocated']<=8<<30 and v['GPU_peak_reserved']<=9<<30,'GPU envelope'
        assert case_start is None or time.monotonic()-case_start<=60,'per conversation60s'
        assert not proc.children(recursive=True),'unexpected descendants'
        assert not args.directory.exists() or sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file())<=3<<30,'total output cap'
    try:
        assert sys.version_info[:3]==(3,12,10) and sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['job']['name']=='global_cohort_collect'
        for item in b['inputs']:
            path=Path(item['path']);assert str(path.resolve())==item['resolved_path']
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];guard()
        def no_subprocess(event,arguments):
            if event=='subprocess.Popen':raise RuntimeError('Collector forbids subprocess')
        sys.addaudithook(no_subprocess)
        stage='imports';print(json.dumps(dict(stage=stage)),flush=True)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from transformers import AutoModelForCausalLM,AutoTokenizer
        import transformers,tokenizers
        from chatbot_interaction import serialize,terminate_generated
        assert (torch.__version__,transformers.__version__,tokenizers.__version__,np.__version__)==('2.6.0+cu124','5.13.1','0.22.2','2.4.6')
        assert importlib.util.find_spec('pyarrow') is None and importlib.util.find_spec('datasets') is None
        assert not any(n=='pyarrow' or n.startswith('pyarrow.') for n in sys.modules)
        assert torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3060'
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        args.directory.mkdir();manifest=json.loads(Path(b['manifest_path']).read_bytes())
        assert (manifest['max_prompt_tokens'],manifest['max_context_tokens'],manifest['max_new_tokens'],manifest['EOS'])==(128,256,16,[151645,151643])
        assert len(manifest['cases'])==200 and sum(c['split']=='fit' for c in manifest['cases'])==160
        stage='tokenize_NEW_transfer_only';tokenizer=AutoTokenizer.from_pretrained(b['source_directory'],local_files_only=True,trust_remote_code=False)
        tokenized=[]
        for case in manifest['cases']:
            rendered=serialize(tokenizer,case['messages'],case['mode']);ids=tokenizer.encode(rendered,add_special_tokens=False)
            assert 0<len(ids)<=128 and len(ids)+16<=256 and all(type(i) is int and 0<=i<151936 for i in ids),(case['id'],len(ids))
            tokenized.append(dict(id=case['id'],rendered_text=rendered,prompt_ids=ids,
                rendered_UTF8_SHA256=hashlib.sha256(rendered.encode('utf8')).hexdigest(),
                prompt_U32LE_SHA256=hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()))
        tokenized_path=args.directory/'tokenized_inputs.json';write_once(tokenized_path,dict(schema='QWEN_GLOBAL_TOKENIZED_V1',cases=tokenized))
        r['tokenized_inputs']=dict(path=str(tokenized_path.resolve()),sha256=sha(tokenized_path),bytes=tokenized_path.stat().st_size)
        r['procedure_gates']['all_NEW_transfer_prompts_tokenized_WITHOUT_endpoint_queries']=True;guard()
        stage='load_pure_original_source';print(json.dumps(dict(stage=stage)),flush=True)
        model=AutoModelForCausalLM.from_pretrained(b['source_directory'],local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        cfg=model.config
        assert (cfg.model_type,cfg.hidden_size,cfg.intermediate_size,cfg.num_hidden_layers,cfg.vocab_size)==('qwen2',896,4864,24,151936)
        assert cfg.hidden_act=='silu' and cfg.rms_norm_eps==1e-6 and cfg.num_attention_heads==14 and cfg.num_key_value_heads==2 and not cfg.use_sliding_window
        assert cfg.tie_word_embeddings and model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
        captured={};hooks=[]
        for layer_id,layer in enumerate(model.model.layers):
            assert layer.mlp.__class__.__name__=='Qwen2MLP'
            assert layer.mlp.gate_proj.bias is layer.mlp.up_proj.bias is layer.mlp.down_proj.bias is None
            def hook(module,inputs,layer_id=layer_id):
                x=inputs[0];assert x.ndim==3 and x.shape[0]==1 and x.shape[-1]==896 and x.dtype==torch.bfloat16 and x.device==device
                assert layer_id not in captured;captured[layer_id]=x.detach()
            hooks.append(layer.mlp.register_forward_pre_hook(hook))
        r['procedure_gates'].update(pure_original_source_BF16_tied_head=True,safe_no_Arrow_dataset_runtime=True)
        payload_bytes=0;stage='NEW_original_own_history_capture';print(json.dumps(dict(stage=stage)),flush=True)
        with torch.inference_mode():
            for case,encoded in zip(manifest['cases'],tokenized):
                case_start=time.monotonic();current={**case,**encoded,'frames':[],'generated_ids':[]}
                r['pending_conversation']=current;xfile=args.directory/(case['id']+'.x.bf16.bin')
                logitsfile=args.directory/(case['id']+'.logits.bf16.bin');journal=args.directory/(case['id']+'.frames.jsonl')
                cache=None;input_ids=encoded['prompt_ids'];xrows=0
                with xfile.open('xb') as xs,logitsfile.open('xb') as ls,journal.open('xb') as js:
                    xs.write(struct.pack('<8s4I',b'QWGX0001',24,896,2,0));ls.write(struct.pack('<8s4I',b'QWGL0001',151936,2,0,0))
                    for step in range(16):
                        guard();captured.clear();ids=torch.tensor(input_ids,dtype=torch.int64,device=device)[None]
                        result=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
                        r['new_original_BF16_full_forwards']+=1
                        assert sorted(captured)==list(range(24)) and result.logits.shape==(1,1,151936) and result.logits.dtype==torch.bfloat16
                        x=torch.stack([captured[i][0] for i in range(24)],dim=1).contiguous()
                        xbits=x.cpu().view(torch.uint16).numpy();lbits=result.logits[0,0].contiguous().cpu().view(torch.uint16).numpy()
                        assert xbits.shape==(len(input_ids),24,896) and lbits.shape==(151936,)
                        assert xbits.dtype==lbits.dtype==np.dtype('<u2') and xbits.flags.c_contiguous and lbits.flags.c_contiguous
                        assert not ((xbits&0x7fff)>=0x7f80).any() and not ((lbits&0x7fff)>=0x7f80).any()
                        next_id=int(result.logits[0,0].argmax().item());xr=xbits.tobytes(order='C');lr=lbits.tobytes(order='C')
                        assert len(xr)==len(input_ids)*43008 and len(lr)==303872
                        assert payload_bytes+len(xr)+len(lr)<=2202419200,'fixed manifest tensor upper bound'
                        frame=dict(step=step,input_ids=input_ids,next_id=next_id,decision_history_position=len(encoded['prompt_ids'])+step-1,
                            x_offset=xs.tell(),x_rows=len(input_ids),x_bytes=len(xr),x_sha256=hashlib.sha256(xr).hexdigest(),
                            logits_offset=ls.tell(),logits_bytes=len(lr),logits_sha256=hashlib.sha256(lr).hexdigest())
                        xs.write(xr);xs.flush();ls.write(lr);ls.flush()
                        js.write((json.dumps(frame,separators=(',',':'))+'\n').encode('utf8'));js.flush()
                        current['frames'].append(frame);current['generated_ids'].append(next_id);xrows+=len(input_ids);payload_bytes+=len(xr)+len(lr)
                        stopped=terminate_generated(current['generated_ids'],manifest['EOS'],16);cache=result.past_key_values
                        del x,xbits,lbits,xr,lr,result,ids;captured.clear();guard()
                        if stopped['termination'] in ('eos','length'):break
                        input_ids=[next_id]
                    for stream in (xs,ls,js):os.fsync(stream.fileno())
                assert xrows==len(encoded['prompt_ids'])+len(current['generated_ids'])-1
                current.update(x_path=str(xfile.resolve()),x_bytes=xfile.stat().st_size,x_sha256=sha(xfile),
                    logits_path=str(logitsfile.resolve()),logits_bytes=logitsfile.stat().st_size,logits_sha256=sha(logitsfile),
                    journal_path=str(journal.resolve()),journal_sha256=sha(journal),x_rows=xrows,
                    termination=stopped['termination'],accepted_generated_ids=stopped['accepted_ids'],
                    displayed_text=tokenizer.decode(stopped['accepted_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False))
                metadata=args.directory/(case['id']+'.meta.json');write_once(metadata,current);assert metadata.stat().st_size<=64<<10
                summary=dict(id=case['id'],split=case['split'],category=case['category'],mode=case['mode'],
                    metadata_path=str(metadata.resolve()),metadata_sha256=sha(metadata),metadata_bytes=metadata.stat().st_size,
                    x_path=current['x_path'],x_bytes=current['x_bytes'],x_sha256=current['x_sha256'],x_rows=xrows,
                    logits_path=current['logits_path'],logits_bytes=current['logits_bytes'],logits_sha256=current['logits_sha256'],
                    journal_path=current['journal_path'],journal_sha256=current['journal_sha256'],prompt_tokens=len(encoded['prompt_ids']),
                    source_forwards=len(current['generated_ids']),termination=stopped['termination'],elapsed_seconds=time.monotonic()-case_start)
                r['conversations'].append(summary);r.pop('pending_conversation');guard();case_start=None;del cache;current=None
                if len(r['conversations'])%8==0:print(json.dumps(dict(completed=len(r['conversations']),payload_bytes=payload_bytes,resource=resource())),flush=True)
        for hook in hooks:hook.remove()
        assert len(r['conversations'])==200 and r['new_original_BF16_full_forwards']==sum(c['source_forwards'] for c in r['conversations'])
        r['procedure_gates'].update(ALL200_NEW_named_conversations_once=True,ALL24_x_and_full_vocabulary_generated_logits=True,
            BOTH_EOS_generated_only_and_own_history=True,incremental_typed_wires_journals_complete=True,no_old_source_or_student_or_endpoint_queries=True)
        r.update(decision='NEW_WHOLE_TRANSFER_COHORT_REQUIRE_FIRST_BYTE_ID_AUDIT',tensor_payload_bytes=payload_bytes,
            resource_before_final_serialization=resource(),ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            runtime=dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,numpy=np.__version__,
                GPU=torch.cuda.get_device_name(),source='Original BF16/eager',TF32=False,deterministic=True),
            scope='NEW finite transfer FIT/development only; generated positions supervision; no compact artifact or endpoint quality/rate admission')
        guard();write_once(args.out,r);print(json.dumps(dict(decision=r['decision'],cases=200,source_forwards=r['new_original_BF16_full_forwards'],resource=resource())),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),fault_stage=stage,resource_snapshot=resource(),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_once(args.out.with_suffix('.failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
