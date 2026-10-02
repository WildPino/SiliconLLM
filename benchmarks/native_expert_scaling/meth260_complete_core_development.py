#!/usr/bin/env python3
"""Frozen full-model development screen of the actual complete unique core."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
import meth259_unique_bank_core_export as R
import meth214_stored_core_fresh_prediction as F

P=R.P;L=R.L
EXPORT=P.DOC/'meth259_unique_bank_core_export_result.json'
EXPORT_SHA='39f796651522a2af907f4123756b9c7ab1e838057debbb142db5c1e02765b663'
CORE=P.ROOT/'results/native_expert_scaling/meth259_qwen05b_unique_source_core.safetensors'
CORE_SHA='3c0949fb2b7f99dc887c6aec1041aa162b35801939349966d18e52b8807a6f71'
PRIOR=P.DOC/'meth194_q8_core_e1280_development_result.json'
PRIOR_SHA='9063af7040e063070a3f590e5be19001ffb48f8b007a2375db8dd17aa656ec1b'
ARMS=('bf16_donor','bf16_e1280','stored_unique_e1280')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();assert not args.out.exists();start=time.monotonic();stage='bindings';arms={}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'arms':arms,
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
    try:
        for path,sha in ((EXPORT,EXPORT_SHA),(CORE,CORE_SHA),(PRIOR,PRIOR_SHA),
                         (P.M122.MANIFEST,P.M122.MANIFEST_SHA),(P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),
                         (P.M122.TRAINING,P.M57.TRAINING_SHA)):
            assert P.digest(path)==sha
        export=json.loads(EXPORT.read_text(encoding='utf-8'));assert all(export['gates'].values()) and export['artifact']['sha256']==CORE_SHA
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        assert P.digest(Path(R.__file__))==export['script_sha256']
        prior=json.loads(PRIOR.read_text(encoding='utf-8'));items=json.loads(P.M122.MANIFEST.read_text(encoding='utf-8'))['items']
        assert len(items)==24 and prior['manifest_sha256']==P.M122.MANIFEST_SHA
        for item in items:
            assert P.M17.sha(item['text'].encode())==item['text_sha256']
            for key in ('document_ids','prompt_ids'):
                assert P.M17.sha(np.asarray(item[key],dtype=np.int32).tobytes())==item[key+'_sha256']
        parent=json.loads(P.M122.TRAINING.read_text(encoding='utf-8'))['checkpoints']['512'];assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True));assert P.digest(source)==P.M57.MODEL_SHA
        device=R.G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=20*60
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        stage='original_BF16_control_load'
        model=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,
            dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device).eval()
        model.config.use_cache=False;L.D.H.load_centered(model,device,parent['path'])
        wrappers=[layer.mlp for layer in model.model.layers];donor_top={}
        with torch.inference_mode():
            for arm,enabled in ((ARMS[0],False),(ARMS[1],True)):
                stage=arm;documents=[];prompts=[]
                for index,item in enumerate(items):
                    nats=P.M17.score_doc(model,item['document_ids'],wrappers,enabled,device,start)
                    assert item['source_id']==prior['document_rows'][index]['source_id']
                    assert nats==prior['document_rows'][index]['nats'][arm]
                    documents.append({'source_id':item['source_id'],'category':item['category'],'bytes':item['bytes'],'nats':nats})
                P.M44.set_experts(wrappers,enabled)
                for index,item in enumerate(items):
                    ids=torch.as_tensor(item['prompt_ids'],device=device)[None];top=model(ids,use_cache=False).logits.argmax(-1)[0].cpu()
                    if not enabled:donor_top[item['source_id']]=top
                    matching=int((top==donor_top[item['source_id']]).sum())
                    old=prior['prompt_rows'][index];assert item['source_id']==old['source_id'] and matching==old['matching'][arm]
                    prompts.append({'source_id':item['source_id'],'category':item['category'],'positions':len(item['prompt_ids']),'matching':matching})
                    P.budget(start,device)
                arms[arm]={'document_rows':documents,'prompt_rows':prompts};partial()
                print(json.dumps({'arm':arm,'runtime':P.budget(start,device)}),flush=True)
        del model,wrappers;gc.collect();torch.cuda.empty_cache();stage='complete_stored_candidate_load'
        model,wrappers,proposal,load_record=R.load_stored(CORE,device,CORE_SHA)
        L.D.E.P=P;stage=ARMS[2]
        arms[ARMS[2]]=L.D.E.evaluate(model,wrappers,items,donor_top,ARMS[2],device,start);partial()
        stage='finite_prompt_shortlist_controls'
        approx=(proposal['codes'].float()*proposal['scales'].float()[:,None]).bfloat16();L.D.H.KS=(64,);shortlist=[]
        with torch.inference_mode():
            for item in items:
                ids=torch.as_tensor(item['prompt_ids'],device=device)[None]
                hidden=model.model(ids,use_cache=False).last_hidden_state[0]
                shortlist.append({'source_id':item['source_id'],'category':item['category'],
                    **L.D.H.score_item(hidden,model.lm_head.weight,approx)});P.budget(start,device)
        F.ARMS=ARMS;summary=F.summarize(arms)
        gates={'all_old_original_control_document_and_prompt_scores_exact':True,
            'complete_candidate_load_from_same_fixed_archive':True,
            'pooled_bpb_vs_both':all(summary['pooled'][k]<=.01 for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'category_bpb_vs_both':all(summary[c][k]<=.02 for c in P.CATEGORIES for k in ('candidate_minus_donor_bpb','candidate_minus_bf16_e1280_bpb')),
            'pooled_top1':summary['pooled']['candidate_minus_bf16_e1280_top1']>=-.01,
            'category_top1':all(summary[c]['candidate_minus_bf16_e1280_top1']>=-.02 for c in P.CATEGORIES),
            'prompt_k64_inclusion':all(r['misses']['64']==0 for r in shortlist),
            'prompt_exact_rerank':all(r['rerank_mismatches']['64']==0 for r in shortlist)}
        result={'experiment':'METH-260-complete-unique-core-full-model-consumed-development',
            'export_result_sha256':EXPORT_SHA,'artifact_sha256':CORE_SHA,'manifest_sha256':P.M122.MANIFEST_SHA,
            'original_control_result_sha256':PRIOR_SHA,'source_sha256':P.M57.MODEL_SHA,
            'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,
            'load_record':load_record,'arms':arms,'summary':summary,'bootstrap':F.bootstraps(arms),
            'shortlist':shortlist,'gates':gates,'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'decision':'complete_core_development_pass_freeze_new_independent_source_manifest' if all(gates.values()) else 'complete_core_development_fail_close_this_fixed_candidate',
            'scope':'Original consumed24-source METH121/122 cohort. Same complete saved artifact,full exact head probabilities,finite prompt K64 check. No fresh source/generation/task/RAM n/native LUT/DRAM or accepted rate/second donor/10B/100B promotion.'}
        args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        partial();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
