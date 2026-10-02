#!/usr/bin/env python3
"""Frozen cached/full and three-arm generation after independent prediction."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import meth263_complete_core_fresh_prediction as Q
import meth215_stored_core_generation as G

P,L,R=Q.P,Q.L,Q.R
PREDICTION=P.DOC/'meth263_complete_core_fresh_prediction_result.json'
ARMS=Q.ARMS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prediction-sha',required=True);ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();assert not args.out.exists();start=time.monotonic();stage='bindings';arms={};parity={};gates={}
    def save():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'arms':arms,'parity':parity,
            'gates':gates,'seconds':time.monotonic()-start},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    def finish(decision,summary=None):
        result={'experiment':'METH-264-same-complete-core-cached-greedy-generation',
            'prediction_sha256':args.prediction_sha,'manifest_sha256':Q.MANIFEST_SHA,'artifact_sha256':Q.D.CORE_SHA,
            'source_sha256':P.M57.MODEL_SHA,'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,
            'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,'arms':arms,'parity':parity,'summary':summary,'gates':gates,
            'decoding':'Unpenalized greedy full BF16 tied head,lowest ID ties,config EOS,128 token cap',
            'runtime':P.budget(start,device),'script_sha256':P.digest(Path(__file__)),
            'generation_helper_sha256':P.digest(Path(G.__file__)),'decision':decision,
            'scope':'Finite cached/full and same261-source full-head generation/K64 screen. Health controls only;no task/blind semantic/native quality/accepted rate/n RAM/other-family proof.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('decision','gates','summary','runtime')}),flush=True)
    try:
        assert P.digest(PREDICTION)==args.prediction_sha
        prediction=json.loads(PREDICTION.read_text(encoding='utf-8'))
        assert prediction['decision']=='independent_complete_core_prediction_pass_freeze_generation_tasks' and all(prediction['gates'].values())
        assert prediction['manifest_sha256']==Q.MANIFEST_SHA and prediction['artifact_sha256']==Q.D.CORE_SHA
        for path,sha in ((Q.MANIFEST,Q.MANIFEST_SHA),(Q.D.EXPORT,Q.D.EXPORT_SHA),(Q.D.CORE,Q.D.CORE_SHA),
                         (P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),(P.M122.TRAINING,P.M57.TRAINING_SHA)):
            assert P.digest(path)==sha
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
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        G.P=P;G.L=L
        model=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to(device).eval()
        L.D.H.load_centered(model,device,parent['path']);wrappers=[layer.mlp for layer in model.model.layers]
        for arm,enabled in ((ARMS[0],False),(ARMS[1],True),(ARMS[2],True)):
            approx=None
            if arm==ARMS[2]:
                del model,wrappers;gc.collect();torch.cuda.empty_cache()
                model,wrappers,proposal,load_record=R.load_stored(Q.D.CORE,device,Q.D.CORE_SHA)
                approx=(proposal['codes'].float()*proposal['scales'].float()[:,None]).bfloat16();L.D.H.KS=(64,)
            P.M44.set_experts(wrappers,enabled);stage=arm+'_cache_parity'
            parity[arm]=G.cache_parity(model,items,device,start)
            gates[arm+'_cache_choices']=all(row['choice_mismatches']==0 for row in parity[arm]);save()
            if not gates[arm+'_cache_choices']:
                finish('cached_reference_choices_fail_hold_generation_and_tasks');return
            stage=arm+'_generation'
            def save_rows(rows):arms[arm]=rows;save()
            rows,passed=G.generations(model,items,tokenizer,device,start,approx,save_rows);arms[arm]=rows
            if approx is not None:
                gates['generated_k64_inclusion_and_exact_rerank']=passed
                if not passed:finish('generated_k64_fail_hold_tasks_native_promotion');return
        summary=G.summarize(arms)
        for control in ARMS[:2]:
            gates[control+'_pooled_eos']=summary['pooled'][ARMS[2]]['eos_terminated']>=summary['pooled'][control]['eos_terminated']-2
            for metric in ('early_non_eos_under16','repeated_8gram_3x'):
                gates[control+'_'+metric]=all(summary[c][ARMS[2]][metric]<=summary[c][control][metric]+1 for c in ('pooled',*P.CATEGORIES))
        finish('generation_health_pass_freeze_task_and_blind_assessment' if all(gates.values()) else 'generation_health_fail_close_this_fixed_cached_candidate',summary)
    except BaseException as error:
        save();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),
            'seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
