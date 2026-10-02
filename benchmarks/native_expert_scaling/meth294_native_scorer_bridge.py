#!/usr/bin/env python3
"""Frozen scoring/plumbing bridge on old285 data, not new held-out quality."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';ART=ROOT/'results/native_expert_scaling'
CORE=ART/'meth276_qwen05b_i16_private128_diagnostic_repair1.safetensors'
CORE_SHA='4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9'
PRIOR=DOC/'meth285_whole_prefix_qualification_result.json'
PRIOR_SHA='c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def proof():
    original=(ROOT/'benchmarks/native_expert_scaling/meth285_model_operator.h').read_text(encoding='utf-8')
    original=original[original.index('static void cc_forward('):original.index('static int cc_argmax')]
    new=(ROOT/'benchmarks/native_expert_scaling/meth294_window_forward.h').read_text(encoding='utf-8').split('\n',1)[1]
    new=new.replace('static void cc_forward_window(CcState *s,int token,int pos,int experts,int head,int position_offset)',
                    'static void cc_forward(CcState *s,int token,int pos,int experts,int head)',1)
    new=new.replace('float phase=(float)(pos+position_offset)*frequency;','float phase=(float)pos*frequency;',1)
    assert new==original

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    bundle=ART/'meth294_scorer_bridge_ids.bin';native=ART/'meth294_scorer_bridge.bin';exe=ART/'meth294_native_prediction_cpu.exe'
    assert all(not p.exists() for p in (args.out,args.out.with_suffix('.failure.json'),bundle,native))
    start=time.monotonic();stage='bindings';proof()
    try:
        assert digest(PRIOR)==PRIOR_SHA and digest(CORE)==CORE_SHA
        prior=json.loads(PRIOR.read_text(encoding='utf-8'));old=ART/'meth285_native_prefixes.bin';assert digest(old)==prior['native_sha256']
        manifest=DOC/'meth278_complete_i16_fresh_manifest.json';assert digest(manifest)==prior['manifest_sha256']
        item=json.loads(manifest.read_text(encoding='utf-8'))['items'][0];ids=np.asarray(item['prompt_ids'],dtype='<i4');assert len(ids)==147
        targets=np.concatenate((ids[-7:],np.asarray([-1],dtype='<i4')))
        with bundle.open('wb') as file:
            file.write(struct.pack('<8sI3I',b'M294IN01',1,147,0,139));file.write(ids.tobytes());file.write(targets.tobytes())
        command=['clang','-O3','-mavx2','-mssse3','-mfma','-fopenmp','-std=c11','-DSILICON_COMPLETE_I16_NATIVE_SCORE',
                 str(ROOT/'benchmarks/phase60/engine.c'),'-o',str(exe),'-lm','-lpsapi','-lbcrypt']
        subprocess.run(command,capture_output=True,text=True,check=True,timeout=60);stage='native_scorer_bridge'
        run=subprocess.run([str(exe),str(CORE),str(bundle),str(native)],capture_output=True,text=True,check=True,timeout=120)
        loader=json.loads(run.stderr);assert loader['fields_bound']==725 and not loader['fallback'] and loader['archive_sha256']==CORE_SHA
        with old.open('rb') as file:
            assert struct.unpack('<8s5I',file.read(28))==(b'M285LG01',2,6,8,896,151936)
            originals=[]
            for arm in range(2):
                at=28+(arm*6)*(20+9*(896+151936)*4);file.seek(at);meta=struct.unpack('<5I',file.read(20));assert meta[:3]==(arm,0,147)
                originals.append(np.frombuffer(file.read(8*(896+151936)*4),dtype='<f4').copy().reshape(8,896+151936)[:,896:])
        checks=[]
        with native.open('rb') as file:
            assert struct.unpack('<8s2I',file.read(16))==(b'M294OUT1',2,1)
            for arm in range(2):
                assert struct.unpack('<5I',file.read(20))==(arm,0,147,0,139)
                first=np.frombuffer(file.read(151936*4),dtype='<f4').copy();assert np.array_equal(first,originals[arm][0])
                rows=[]
                for i in range(8):
                    pos,top,target,loss=struct.unpack('<iiid',file.read(20));assert pos==139+i and target==targets[i]
                    logits=originals[arm][i].astype(np.float64);maximum=logits.max()
                    expected=0.0 if target<0 else float(np.log(np.exp(logits-maximum).sum())+maximum-logits[target])
                    rows.append({'position':pos,'top1_exact':top==int(logits.argmax()),'target':target,
                                 'native_nll':loss,'numpy_float64_nll':expected,'absolute_nll_error':abs(loss-expected)})
                checks.append({'arm':arm,'first_full_head_bytes_exact_original285':True,'rows':rows})
            assert not file.read(1)
        gates={'same_archive_all725_no_fallback':True,'window_source_exact285_except_absolute_offset':True,
               'both_first_full_head_rows_exact_original285':True,'all16_top1_exact_original285':all(r['top1_exact'] for c in checks for r in c['rows']),
               'all16_native_nll_within1e_minus9_of_float64_oracle':all(r['absolute_nll_error']<=1e-9 for c in checks for r in c['rows'])}
        result={'experiment':'METH-294-native-full-head-scoring-entry-bridge','artifact_sha256':CORE_SHA,'prior285_sha256':PRIOR_SHA,
                'checks':checks,'gates':gates,'loader':loader,'native_report':json.loads(run.stdout),'compile_command':command,
                'sha256':{str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'benchmarks/native_expert_scaling/meth294_native_prediction_cpu.c',
                    ROOT/'benchmarks/native_expert_scaling/meth294_window_forward.h',ROOT/'benchmarks/phase60/engine.c',bundle,native,exe)},
                'seconds':time.monotonic()-start,'diagnostic_only':True,'native_promotion_qualified':False,
                'decision':'native_scorer_bridge_pass_freeze_new_source_quality' if all(gates.values()) else 'native_scorer_bridge_fail_stop_new_source_quality',
                'scope':'Old consumed285 source0/tail16 bankoff/on rows;source/scorer/plumbing qualification. No new292 source/model quality,nonzero-offset numerical control,generation/task/K64/accepted rate/useful capacity/family evidence.'}
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('decision','gates','seconds')}),flush=True)
    except BaseException as error:
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage':stage,'error':type(error).__name__+': '+str(error),'seconds':time.monotonic()-start},indent=2,ensure_ascii=True)+'\n',encoding='utf-8');raise

if __name__=='__main__':main()
