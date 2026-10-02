#!/usr/bin/env python3
"""Prospective full PIQA regression of the semantically qualified M276 archive."""
import argparse
import gc
import json
from pathlib import Path
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import meth281_complete_i16_generation as E

P,L,R,Q=E.P,E.L,E.R,E.Q
T=P.M122.M21
S=P.M122.M90
ARMS=E.ARMS
GENERATION=P.DOC/'meth281_complete_i16_generation_result.json'
GENERATION_SHA='9699987e06ef6816a4fe07d373bbe44a888bc5be519af14ac079812c2b2c1304'
SEMANTIC=P.DOC/'meth282_complete_i16_semantic_result.json'
SEMANTIC_SHA='071e797942e9b225151a60b282f96c8d4ade0b561913e6098c5adc3b525906db'
MAX_SECONDS=45*60
MAX_RSS_BYTES=20*(1<<30)
MAX_GPU_BYTES=int(10.5*(1<<30))
MAX_OUTPUT_BYTES=96*(1<<20)
TASK_HELPER_SHA=('cde74aa44c7db89692f877a9a24b0cae37085048068cff3151af9f762df63609',
                 'c9c59e0272958283f1c7de90dee2f7ce00949c90badfd8a436df28ce2deaf932')


def budget(start,device):
    elapsed=time.monotonic()-start
    rss=psutil.Process().memory_info().rss
    gpu=torch.cuda.max_memory_allocated(device)
    if elapsed>MAX_SECONDS or rss>MAX_RSS_BYTES or gpu>MAX_GPU_BYTES:
        raise RuntimeError(f'METH-283 resource stop: {elapsed:.1f}s,RSS {rss},GPU {gpu}')
    return {'elapsed_seconds':elapsed,'rss_end_bytes':rss,'gpu_peak_allocated_bytes':gpu}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json')))
    start=time.monotonic();stage='bindings';arms={};gates={};load_record=None
    def save():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage':stage,'arms':arms,
            'gates':gates,'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
        assert args.out.with_suffix('.partial.json').stat().st_size<MAX_OUTPUT_BYTES
    try:
        for module,sha in zip((T,S),TASK_HELPER_SHA):assert P.digest(Path(module.__file__))==sha
        for path,sha in ((GENERATION,GENERATION_SHA),(SEMANTIC,SEMANTIC_SHA),
                         (Q.EXPORT,Q.EXPORT_SHA),(Q.CORE,Q.CORE_SHA),
                         (P.M122.SPECIALIZED,P.M122.SPECIALIZED_SHA),(P.M122.TRAINING,P.M57.TRAINING_SHA)):
            assert P.digest(path)==sha,str(path)
        gen=json.loads(GENERATION.read_text(encoding='utf-8'))
        semantic=json.loads(SEMANTIC.read_text(encoding='utf-8'))
        assert gen['decision']=='full_prefix_generation_health_pass_freeze_task_and_blind' and all(gen['gates'].values())
        assert semantic['decision']=='anonymous_semantic_pass_requires_PIQA_then_joint_native_gate' and all(semantic['gates'].values())
        assert semantic['generation_sha256']==GENERATION_SHA
        assert gen['artifact_sha256']==semantic['artifact_sha256']==Q.CORE_SHA
        assert gen['manifest_sha256']==semantic['manifest_sha256']==Q.MANIFEST_SHA
        assert semantic['findings_commit']=='70e4061e520399cfe8a0fb06158c08a58cc6eda7'
        for field,path in (('panel_sha256',P.DOC/'meth282_anonymous_response_panel.json'),
                           ('verdict_sha256',P.DOC/'meth282_anonymous_verdict.json'),
                           ('mapping_sha256',P.DOC/'meth282_anonymous_arm_map.json')):
            assert P.digest(path)==semantic[field]
        export=json.loads(Q.EXPORT.read_text(encoding='utf-8'))
        assert all(export['gates'].values()) and export['diagnostic_only'] and not export['native_promotion_qualified']
        assert P.digest(Path(R.__file__))==export['script_sha256']
        for path,sha in export['helper_sha256'].items():assert P.digest(Path(path))==sha
        parent=json.loads(P.M122.TRAINING.read_text(encoding='utf-8'))['checkpoints']['512']
        assert P.digest(parent['path'])==P.M57.CHECKPOINT_SHA
        source=Path(hf_hub_download(P.M42.MODEL,'model.safetensors',revision=P.M42.REV,local_files_only=True))
        assert P.digest(source)==P.M57.MODEL_SHA
        tokenizer=AutoTokenizer.from_pretrained(P.M42.MODEL,revision=P.M42.REV,local_files_only=True)
        assert P.M15.M13.C.tok_fingerprint(tokenizer)==P.M15.M13.TOK_FP
        task_manifest,items=T.bind_data(tokenizer);assert len(items)==1838
        task_manifest['binding_helper_legacy_adapter_sha256']=task_manifest.pop('adapter_sha256')
        task_manifest.update(experiment='METH-283',artifact_sha256=Q.CORE_SHA)
        assert T.BOOTSTRAP_SEED==212121 and T.BOOTSTRAP_DRAWS==20000
        device=R.G.Q.M.D.Q.setup();assert torch.get_num_threads()==6
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        stage='original_BF16_control_load'
        model=AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to(device).eval();model.config.use_cache=False
        L.D.H.load_centered(model,device,parent['path']);wrappers=[layer.mlp for layer in model.model.layers]
        for arm,enabled in ((ARMS[0],False),(ARMS[1],True),(ARMS[2],True)):
            if arm==ARMS[2]:
                del model,wrappers;gc.collect();torch.cuda.empty_cache();stage='same_archive_loader'
                model,wrappers,proposal,load_record=R.load_stored(Q.CORE,device,Q.CORE_SHA)
                assert not model.config.use_cache
                assert load_record['source_weights_or_conditional_checkpoints_loaded'] is False
                assert load_record['tensors_consumed']==725
            P.M44.set_experts(wrappers,enabled);stage=arm+'_piqa';rows=[];arms[arm]=rows
            for item in items:
                assert not model.config.use_cache
                scores=[T.score_option(model,item['prefix'],suffix,device) for suffix in item['suffixes']]
                rows.append({'index':item['index'],'label':item['label'],
                    'choice_mean':0 if scores[0][1]<=scores[1][1] else 1,
                    'choice_total':0 if scores[0][0]<=scores[1][0] else 1,
                    'options':[{'total_nll':total,'mean_nll':mean,'tokens':len(suffix)}
                        for (total,mean),suffix in zip(scores,item['suffixes'])]})
                runtime=budget(start,device)
                if len(rows)%128==0:
                    save();print(json.dumps({'arm':arm,'items':len(rows),'total':1838,'runtime':runtime}),flush=True)
            assert len(rows)==1838;save()
        summary={}
        for control in ARMS[:2]:
            pairs=list(zip(arms[control],arms[ARMS[2]]))
            assert all(a['index']==b['index'] and a['label']==b['label'] for a,b in pairs)
            summary[control]=S.task_summary([{'label':a['label'],'donor':a,'student':b} for a,b in pairs])
            gates[control+'_accuracy']=summary[control]['accuracy_delta']>=-.02
            gates[control+'_paired_lower95']=summary[control]['paired_bootstrap_lower95']>=-.05
        result={'experiment':'METH-283-complete-I16-private128-PIQA-regression',
            'generation_sha256':GENERATION_SHA,'semantic_sha256':SEMANTIC_SHA,
            'artifact_sha256':Q.CORE_SHA,'composition_sha256':Q.EXPORT_SHA,
            'source_sha256':P.M57.MODEL_SHA,'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,
            'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,'task_manifest':task_manifest,
            'arms':arms,'summary':summary,'gates':gates,'load_record':load_record,'runtime':budget(start,device),
            'script_sha256':P.digest(Path(__file__)),
            'task_helper_sha256':{str(Path(module.__file__)):P.digest(Path(module.__file__)) for module in (T,S)},
            'helper_sha256':export['helper_sha256'],'diagnostic_only':True,'native_promotion_qualified':False,
            'decision':'complete_I16_task_pass_requires_joint_native_quality_rate' if all(gates.values()) else 'complete_I16_task_fail_close_fixed_candidate',
            'scope':'Repeated full1838-item PIQA regression,not independent task/general capability proof. Same276 archive/full BF16 tied head/no cache;no native accepted-rate/useful RAM-scale n/DRAM/family qualification.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
        assert args.out.stat().st_size+args.out.with_suffix('.partial.json').stat().st_size<MAX_OUTPUT_BYTES
        print(json.dumps({key:result[key] for key in ('decision','summary','gates','runtime')}),flush=True)
    except BaseException as error:
        save();args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,
            'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8');raise


if __name__=='__main__':main()
