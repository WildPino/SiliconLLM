#!/usr/bin/env python3
"""Native generation/health and real state capture; GPU controls in a child process."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np
import meth295_native_primary_prediction as M
import meth281_complete_i16_generation as O
from transformers import AutoTokenizer

P=M.P;G=O.G
PREDICTION=M.DOC/'meth295_native_primary_prediction_repair1_result.json'

def bind(prediction_sha):
    assert P.digest(PREDICTION)==prediction_sha
    prediction=json.loads(PREDICTION.read_text(encoding='utf-8'))
    assert all(prediction['gates'].values()) and prediction['decision']=='native_primary_prediction_pass_freeze_generation'
    assert prediction['artifact_sha256']==M.CORE_SHA and prediction['manifest_sha256']==M.MANIFEST_SHA
    for path,sha in prediction['sha256'].items():
        if path.startswith('benchmarks/native_expert_scaling/'):assert P.digest(P.ROOT/path)==sha,path
    items,parent=M.bind()
    tokenizer=AutoTokenizer.from_pretrained(P.M42.MODEL,revision=P.M42.REV,local_files_only=True)
    assert P.M15.M13.C.tok_fingerprint(tokenizer)==P.M15.M13.TOK_FP
    return prediction,items,parent,tokenizer

def controls_worker(args):
    start=time.monotonic();stage='bindings';arms={}
    assert not args.controls_out.exists()
    try:
        prediction,items,parent,tokenizer=bind(args.prediction_sha)
        device=M.R.G.Q.M.D.Q.setup();P.MAX_SECONDS=P.M17.MAX_SECONDS=35*60
        M.torch.backends.cuda.matmul.allow_tf32=False;M.torch.backends.cudnn.allow_tf32=False
        M.torch.set_float32_matmul_precision('highest');M.torch.use_deterministic_algorithms(True)
        model=M.AutoModelForCausalLM.from_pretrained(P.M42.MODEL,revision=P.M42.REV,dtype=M.torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).to(device).eval();model.config.use_cache=False
        assert model.config.eos_token_id==M.EOS
        M.L.D.H.load_centered(model,device,parent['path']);wrappers=[l.mlp for l in model.model.layers]
        G.P=P;G.L=M.L
        for arm,enabled in ((M.ARMS[0],False),(M.ARMS[1],True)):
            stage=arm;P.M44.set_experts(wrappers,enabled)
            def save(rows):
                arms[arm]=rows;M.dump(args.controls_out.with_suffix('.partial.json'),{'stage':stage,'arms':arms})
            rows,passed=O.generate(model,items,tokenizer,device,start,None,save);assert passed and len(rows)==24
            arms[arm]=rows
        M.dump(args.controls_out,{'arms':arms,'prediction_sha256':args.prediction_sha,'manifest_sha256':M.MANIFEST_SHA,
            'source_sha256':P.M57.MODEL_SHA,'parent_checkpoint_sha256':P.M57.CHECKPOINT_SHA,
            'child_checkpoint_sha256':P.M122.SPECIALIZED_SHA,'script_sha256':P.digest(Path(__file__)),
            'runtime':P.budget(start,device),'generation_helper_sha256':P.digest(Path(O.__file__)),
            'decoding':'Original full-prefix BF16 reference/lowest ID/full head/no cache/128cap/EOS'})
    except BaseException as error:
        M.dump(args.controls_out.with_suffix('.failure.json'),{'stage':stage,'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start});raise

def prior_prompt_heads(prediction):
    path=P.ROOT/'results/native_expert_scaling/meth295_native_primary_repair1.bin'
    assert P.digest(path)==prediction['sha256'][str(path.relative_to(P.ROOT))]
    heads={}
    with path.open('rb') as file:
        assert struct.unpack('<8s2I',file.read(16))==(b'M294OUT1',1,len(prediction['cases']))
        for c in prediction['cases']:
            assert struct.unpack('<5I',file.read(20))==(1,c['case'],c['n'],c['offset'],c['first'])
            raw=file.read(M.V*4);assert len(raw)==M.V*4
            if c['kind']=='prompt':heads[c['source_id']]=raw
            file.seek((c['n']-c['first'])*20,1)
        assert not file.read(1)
    assert len(heads)==24;return heads

def health_row(item,ids,tokenizer):
    return {'source_id':item['source_id'],'category':item['category'],'prompt_ids_sha256':item['prompt_ids_sha256'],
        'continuation_ids':ids,'continuation_text':tokenizer.decode(ids,skip_special_tokens=False),
        'eos_terminated':bool(ids) and ids[-1]==M.EOS,'early_non_eos_under16':len(ids)<16 and (not ids or ids[-1]!=M.EOS),
        'repeated_8gram_3x':G.G.repeated_8gram(ids),'distinct2':G.G.distinct2(ids)}

def parse_native(path,items,tokenizer,prediction):
    heads=prior_prompt_heads(prediction);rows=[];state_counts={'prompt':0,'generated':0}
    def check_state(raw):
        assert len(raw)==1792
        values=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
        assert np.isfinite(values).all()
    with path.open('rb') as file:
        assert struct.unpack('<8s3I',file.read(20))==(b'M296GEN1',24,896,M.V)
        for index,item in enumerate(items):
            n=len(item['prompt_ids']);assert struct.unpack('<2I',file.read(8))==(index,n)
            first=file.read(M.V*4);assert first==heads[item['source_id']]
            prior=prediction['arms'][M.ARMS[2]]['prompt_rows'][index]
            last_bits=None
            for j in range(n):
                pos,choice=struct.unpack('<2i',file.read(8));bits=file.read(1792);check_state(bits)
                assert pos==j and choice==prior['token_top1'][j]
                last_bits=bits;state_counts['prompt']+=1
            count,eos=struct.unpack('<2I',file.read(8));assert 1<=count<=128
            ids=[]
            for step in range(count):
                stored_step,choice,logit=struct.unpack('<iif',file.read(12));bits=file.read(1792);check_state(bits)
                assert stored_step==step and 0<=choice<M.V and np.isfinite(logit)
                if step==0:assert bits==last_bits and choice==prior['token_top1'][-1]
                if step<count-1:assert choice!=M.EOS
                ids.append(choice);state_counts['generated']+=1
            assert bool(eos)==(ids[-1]==M.EOS) and (eos or count==128)
            rows.append(health_row(item,ids,tokenizer))
        assert not file.read(1)
    return rows,state_counts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prediction-sha',required=True)
    ap.add_argument('--out',type=Path);ap.add_argument('--controls-out',type=Path);args=ap.parse_args()
    if args.controls_out is not None:controls_worker(args);return
    assert args.out is not None
    bundle=M.ART/'meth296_native_prompt_ids.bin';native=M.ART/'meth296_native_generation.bin'
    log=M.ART/'meth296_native_generation_stdout.log';err=M.ART/'meth296_native_generation_stderr.log'
    exe=M.ART/'meth296_native_generation_cpu.exe';controls=M.DOC/'meth296_original_generation_controls.json'
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.partial.json'),args.out.with_suffix('.failure.json'),bundle,native,log,err,controls))
    start=time.monotonic();stage='bindings';arms={}
    try:
        prediction,items,parent,tokenizer=bind(args.prediction_sha)
        stage='original_full_prefix_controls'
        subprocess.run([sys.executable,str(Path(__file__)), '--prediction-sha',args.prediction_sha,'--controls-out',str(controls)],check=True,timeout=40*60)
        saved=json.loads(controls.read_text(encoding='utf-8'));assert saved['prediction_sha256']==args.prediction_sha
        assert saved['manifest_sha256']==M.MANIFEST_SHA and saved['script_sha256']==P.digest(Path(__file__))
        arms=saved['arms'];assert set(arms)==set(M.ARMS[:2])
        for name,values in arms.items():
            assert len(values)==24
            for item,row in zip(items,values):
                assert item['source_id']==row['source_id'] and row['prompt_ids_sha256']==item['prompt_ids_sha256']
        assert M.torch.cuda.is_initialized() is False
        with bundle.open('wb') as file:
            file.write(struct.pack('<8sI',b'M296IDS1',24))
            for item in items:
                file.write(struct.pack('<I',len(item['prompt_ids'])));file.write(np.asarray(item['prompt_ids'],dtype='<i4').tobytes())
        command=['clang','-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11','-DSILICON_COMPLETE_I16_NATIVE_GENERATE',
            str(P.ROOT/'benchmarks/phase60/engine.c'),'-o',str(exe),'-lm','-lpsapi','-lbcrypt']
        subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        stage='native_cached_generation';M.CPU_SECONDS=20*60
        print(json.dumps({'stage':stage,'sources':24,'log':str(log),'timing_qualification':False}),flush=True)
        runtime=M.run_cpu(exe,bundle,native,log,err)
        assert len(runtime['reports'])==25 and [r['source_complete'] for r in runtime['reports'][:-1]]==list(range(1,25))
        stage='native_parse';arms[M.ARMS[2]],state_counts=parse_native(native,items,tokenizer,prediction)
        assert [r['generated'] for r in runtime['reports'][:-1]]==[len(r['continuation_ids']) for r in arms[M.ARMS[2]]]
        summary=G.summarize(arms);report=runtime['reports'][-1]
        expected_fresh=sum(min(3,len(arms[M.ARMS[2]][i]['continuation_ids'])) for i in (0,8,16))
        gates={'same_artifact_all725_no_fallback':True,'all_prompt_top1_and_first_heads_exact_native295':True,
            'all_real_prompt_and_generated_hidden_states_finite_retained':True,'native_cached_fresh_bytes':report['fresh_byte_checks']==expected_fresh,
            'erased_history_negative_controls':report['erased_history_controls']==3,'CUDA_control_process_terminated_before_native':True}
        for control in M.ARMS[:2]:
            gates[control+'_pooled_eos']=summary['pooled'][M.ARMS[2]]['eos_terminated']>=summary['pooled'][control]['eos_terminated']-2
            for metric in ('early_non_eos_under16','repeated_8gram_3x'):
                gates[control+'_'+metric]=all(summary[c][M.ARMS[2]][metric]<=summary[c][control][metric]+1 for c in ('pooled',*P.CATEGORIES))
        paths=(Path(__file__),Path(O.__file__),Path(G.__file__),P.ROOT/'benchmarks/native_expert_scaling/meth296_native_generation_cpu.c',
            P.ROOT/'benchmarks/phase60/engine.c',P.ROOT/'benchmarks/native_expert_scaling/meth285_model_operator.h',bundle,native,log,err,exe,controls)
        result={'experiment':'METH-296-actual-native-complete-bank-generation-health-and-state-capture',
            'prediction_sha256':args.prediction_sha,'manifest_sha256':M.MANIFEST_SHA,'artifact_sha256':M.CORE_SHA,
            'answerability_sha256':M.ANSWER_SHA,'annotations_sha256':M.ANNOTATIONS_SHA,'policy_sha256':M.POLICY_SHA,
            'arms':arms,'summary':summary,'gates':gates,'native_runtime':runtime,'GPU_runtime':saved['runtime'],
            'state_counts':state_counts,'sha256':{str(p.relative_to(P.ROOT)):P.digest(p) for p in paths},
            'compile_command':command,'seconds':time.monotonic()-start,'diagnostic_only':True,'native_promotion_qualified':False,
            'decoding':'CPU original285 cached full head; BF16 donor/E1280 full-prefix references; unpenalized greedy lowest ID/EOS/128cap',
            'decision':'native_generation_health_pass_freeze_anonymous_semantics' if all(gates.values()) else 'native_generation_health_fail_close_fixed_native_profile',
            'scope':'Same24 source prompts/actual72 continuations, finite health and scoped native cache closure. Real native normalized hidden states retained for later CPU K64. No anonymous semantics/PIQA/CPU K64/accepted rate/useful new n/DRAM/family qualification.'}
        assert native.stat().st_size<64*1024**2 and M.torch.cuda.is_initialized() is False
        M.dump(args.out,result);print(json.dumps({k:result[k] for k in ('decision','summary','gates','seconds')}),flush=True)
    except BaseException as error:
        M.dump(args.out.with_suffix('.failure.json'),{'stage':stage,'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start});raise

if __name__=='__main__':main()
