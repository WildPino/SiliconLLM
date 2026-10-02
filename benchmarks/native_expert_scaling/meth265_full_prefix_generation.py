#!/usr/bin/env python3
"""Same weights/prompts/health gates, explicit full-prefix reference decoding."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import meth264_complete_core_generation as C

P,L,R,Q,G=C.P,C.L,C.R,C.Q,C.G
STOP=P.DOC/'meth264_complete_core_generation_result.json'
STOP_SHA='c016596e419476e7033fc95e774b06bc184e54f8f6ceaf08989dcc54e3865029'
PREDICTION_SHA='7bcb8774bf335762e5e9b89dee29223bf1fa2bfea8d59480ed0621fa1c4d7821'
ARMS=C.ARMS


def generate(model,items,tokenizer,device,start,approx,save):
    eos=model.config.eos_token_id;assert isinstance(eos,int);rows=[]
    with torch.inference_mode():
        for item in items:
            ids=torch.as_tensor(item['prompt_ids'],device=device)[None];continuation=[];shortlist=[]
            for step in range(128):
                token,cache,hidden,logits=G.choose(model,ids,cached=False);assert cache is None
                assert torch.isfinite(logits).all()
                if approx is not None:
                    probe=L.D.H.score_item(hidden,model.lm_head.weight,approx);assert probe['positions']==1
                    shortlist.append({'step':step,'full_choice':token,**probe})
                continuation.append(token);P.budget(start,device)
                if token==eos:break
                ids=torch.cat((ids,torch.as_tensor([[token]],device=device)),dim=1)
            row={'source_id':item['source_id'],'category':item['category'],'prompt_ids_sha256':item['prompt_ids_sha256'],
                'continuation_ids':continuation,'continuation_text':tokenizer.decode(continuation,skip_special_tokens=False),
                'eos_terminated':bool(continuation) and continuation[-1]==eos,
                'early_non_eos_under16':len(continuation)<16 and (not continuation or continuation[-1]!=eos),
                'repeated_8gram_3x':G.G.repeated_8gram(continuation),'distinct2':G.G.distinct2(continuation),'shortlist':shortlist}
            rows.append(row);save(rows)
            print(json.dumps({'generated':len(rows),'total':24,'runtime':P.budget(start,device)}),flush=True)
            if approx is not None and any(r['misses']['64'] or r['rerank_mismatches']['64'] for r in shortlist):return rows,False
    return rows,True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();assert not args.out.exists();start=time.monotonic();stage='bindings';arms={};gates={}
    def save():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'arms':arms,'gates':gates,
            'seconds':time.monotonic()-start},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    def finish(decision,summary=None):
        result={'experiment':'METH-265-same-artifact-independent-full-prefix-generation',
            'cached_reference_stop_sha256':STOP_SHA,'prediction_sha256':PREDICTION_SHA,
            'manifest_sha256':Q.MANIFEST_SHA,'artifact_sha256':Q.D.CORE_SHA,'source_sha256':P.M57.MODEL_SHA,
            'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,
            'arms':arms,'summary':summary,'gates':gates,
            'decoding':'Full-prefix recomputation,no KV cache,unpenalized greedy BF16 full tied head,lowest ID ties,config EOS,128 token cap',
            'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'generation_helper_sha256':P.digest(Path(G.__file__)),'decision':decision,
            'scope':'Same artifact/source/reference full-prefix semantic evaluation path,not efficient production decoding. Finite health/K64 controls,no task/blind or native cached quality/accepted-rate,n/RAM/other-donor proof.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','gates','summary','runtime')}),flush=True)
    try:
        for path,sha in ((STOP,STOP_SHA),(C.PREDICTION,PREDICTION_SHA),(Q.MANIFEST,Q.MANIFEST_SHA),
                         (Q.D.EXPORT,Q.D.EXPORT_SHA),(Q.D.CORE,Q.D.CORE_SHA),
                         (P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),(P.M122.TRAINING,P.M57.TRAINING_SHA)):
            assert P.digest(path)==sha
        stop=json.loads(STOP.read_text(encoding='utf-8'));assert stop['decision']=='cached_reference_choices_fail_hold_generation_and_tasks'
        assert not stop['arms'] and stop['gates']=={'bf16_donor_cache_choices':False}
        prediction=json.loads(C.PREDICTION.read_text(encoding='utf-8'));assert all(prediction['gates'].values())
        manifest=json.loads(Q.MANIFEST.read_text(encoding='utf-8'));items=manifest['items'];assert len(items)==24
        for item in items:
            for key in ('document_ids','prompt_ids'):assert P.M17.sha(np.asarray(item[key],dtype=np.int32).tobytes())==item[key+'_sha256']
        export=json.loads(Q.D.EXPORT.read_text(encoding='utf-8'));assert P.digest(Path(R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        parent=json.loads(P.M122.TRAINING.read_text(encoding='utf-8'))['checkpoints']['512'];assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True));assert P.digest(source)==P.M57.MODEL_SHA
        tokenizer=AutoTokenizer.from_pretrained(P.M42.MODEL,revision=P.M42.REV,local_files_only=True)
        assert P.M15.M13.C.tok_fingerprint(tokenizer)==P.M15.M13.TOK_FP
        device=R.G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=70*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);G.P=P;G.L=L
        model=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to(device).eval();model.config.use_cache=False
        L.D.H.load_centered(model,device,parent['path']);wrappers=[layer.mlp for layer in model.model.layers]
        for arm,enabled in ((ARMS[0],False),(ARMS[1],True),(ARMS[2],True)):
            approx=None
            if arm==ARMS[2]:
                del model,wrappers;gc.collect();torch.cuda.empty_cache()
                model,wrappers,proposal,load_record=R.load_stored(Q.D.CORE,device,Q.D.CORE_SHA)
                approx=(proposal['codes'].float()*proposal['scales'].float()[:,None]).bfloat16();L.D.H.KS=(64,)
            P.M44.set_experts(wrappers,enabled);stage=arm+'_full_prefix_generation'
            def save_rows(rows):arms[arm]=rows;save()
            rows,passed=generate(model,items,tokenizer,device,start,approx,save_rows);arms[arm]=rows
            if approx is not None:
                gates['generated_k64_inclusion_and_exact_rerank']=passed
                if not passed:finish('full_prefix_generated_k64_fail_hold_tasks_native_promotion');return
        summary=G.summarize(arms)
        for control in ARMS[:2]:
            gates[control+'_pooled_eos']=summary['pooled'][ARMS[2]]['eos_terminated']>=summary['pooled'][control]['eos_terminated']-2
            for metric in ('early_non_eos_under16','repeated_8gram_3x'):
                gates[control+'_'+metric]=all(summary[c][ARMS[2]][metric]<=summary[c][control][metric]+1 for c in ('pooled',*P.CATEGORIES))
        finish('full_prefix_generation_health_pass_freeze_task_and_blind' if all(gates.values()) else 'full_prefix_generation_health_fail_close_fixed_candidate',summary)
    except BaseException as error:
        save();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
