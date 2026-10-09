"""Post-hoc fixed-direction scale control of immutable actual native outputs."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))
import numpy as np


def main(a):
    start=time.monotonic()
    probe=DOC/'original_readout_information_result_20261009.json'
    terminal=probe.with_suffix('.terminal.json');t=json.loads(terminal.read_bytes())
    assert t['exit_code']==0 and sha(probe)==t['result_sha256']
    binding=DOC/'original_readout_information_binding_20261009.json';b=json.loads(binding.read_bytes())
    assert sha(binding)==t['binding_sha256']
    rec=next(r for r in b['selected'] if r['id']=='broad_fit_smol_magpie_ultra_022')
    corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    full=next(r for r in json.loads(corpus.read_bytes())['records'] if r['id']==rec['id'])
    inputs=[Path(__file__),probe,terminal,binding,corpus,Path(b['native']['path']),Path(rec['teacher']['path'])]
    receipts=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in inputs]
    assert receipts[-2]==b['native'] and receipts[-1]['sha256']==rec['teacher']['sha256']
    native=np.memmap(b['native']['path'],dtype='<f4',mode='r',shape=(1507,65537))
    bits=np.memmap(rec['teacher']['path'],dtype='<u2',mode='r',shape=(256,65537))
    KL=[[],[]];uniform=[];entropy=[];disagreement=0
    for j,pos in enumerate(full['positions']):
        qlog=(bits[j].astype('<u4')<<16).view('<f4').astype('f8');qlog-=qlog.max()
        qlog-=np.log(np.exp(qlog).sum());q=np.exp(qlog)
        H=-float(np.sum(q*qlog));entropy.append(H);uniform.append(math.log(65537)-H)
        disagreement+=int(np.argmax(native[pos])!=np.argmax(qlog))
        for k,scale in enumerate([1.,math.sqrt(8.)]):
            z=native[pos].astype('f8')*scale;z-=z.max();z-=np.log(np.exp(z).sum())
            KL[k].append(float(np.sum(q*(qlog-z))))
    assert abs(float(np.mean(KL[0]))-10.580769334518816)<1e-10
    for item in receipts:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    result=dict(schema='ORIGINAL_READOUT_STORED_SCALE_RESULT_V1',inputs=receipts,labels=256,
        source_case=rec['id'],entropy_mean=float(np.mean(entropy)),uniform_KL_mean=float(np.mean(uniform)),
        controls=[dict(scale=s,KL_mean=float(np.mean(KL[k])),per_label_KL=KL[k]) for k,s in enumerate([1.,math.sqrt(8.)])],
        disagreement=disagreement,scale_helped_at_actual_fixed_directions=float(np.mean(KL[1]))<float(np.mean(KL[0])),
        numerical_scope='Post-hoc F64 scaling of saved native F32 logits;not a re-exported native gamma variant.',
        new_model_forwards=0,new_teacher_queries=0,new_optimizer_updates=0,GPU_calls=0,quality_admission=False,
        elapsed_seconds=time.monotonic()-start)
    write(a.out,result)
    print(json.dumps(dict(result=str(a.out),sha256=sha(a.out),seconds=result['elapsed_seconds'],
        uniform=result['uniform_KL_mean'],actual=result['controls'][0]['KL_mean'],scaled=result['controls'][1]['KL_mean'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
