"""Post-hoc interpretation of immutable complete outputs; no model/teacher calls.

Does not replace/relax the failed prospective native numerical gate.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))
import numpy as np


def main(a):
    result=json.loads(a.result.read_bytes());terminal=json.loads(a.result.with_suffix('.terminal.json').read_bytes())
    assert sha(a.result)==terminal['result_sha256'] and terminal['exit_code']==0
    assert result['decision']=='ORIGINAL_FALCON_NATIVE_NUMERICAL_GATE_FAIL' and result['durable_updates']==1
    paths={Path(p['path']).name:p for p in terminal['output_files']}
    binding=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_transfer_binding_20261009.json'
    b=json.loads(binding.read_bytes());assert sha(binding)==result['binding_sha256']
    corpus=Path(b['corpus']);rec=next(r for r in json.loads(corpus.read_bytes())['records'] if r['id']==b['selected_id'])
    sources=[a.result,a.result.with_suffix('.terminal.json'),binding,corpus,Path(rec['logits']['path']),Path(__file__)]
    for name in ('native.f32','learner.f32','native.routes','learner.routes'):
        item=paths[name];p=Path(item['path'])
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];sources.append(p)
    receipts=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in sources]
    native=np.memmap(paths['native.f32']['path'],dtype='<f4',mode='r',shape=(1507,65537))
    learner=np.memmap(paths['learner.f32']['path'],dtype='<f4',mode='r',shape=(1507,65537))
    bits=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4')
    teacher=(bits<<16).view('<f4').reshape(256,65537)
    metrics={};positions=rec['positions'];bad=[r['position'] for r in result['native_comparison']['rows'] if r['relative_RMS']>1e-4]
    for name,array in [('native',native),('learner',learner)]:
        KLs=[];dis=0;self_prediction=0
        for j,t in enumerate(positions):
            a0=teacher[j].astype('f8');a0-=a0.max();a0-=np.log(np.exp(a0).sum())
            pred=array[t].astype('f8');pred-=pred.max();pred-=np.log(np.exp(pred).sum())
            KLs.append(float(np.sum(np.exp(a0)*(a0-pred))))
            idx=int(np.argmax(array[t]));dis+=int(idx!=np.argmax(teacher[j]))
            self_prediction+=int(idx==rec['student_input_ids'][t])
        metrics[name]=dict(KL_F64_mean=float(np.mean(KLs)),per_label_KL=KLs,
            disagreement=dis,labels=256,current_input_ID_predictions=self_prediction)
    assert abs(metrics['learner']['KL_F64_mean']-result['KL_after'])<5e-5
    nr=Path(paths['native.routes']['path']).read_bytes();lr=Path(paths['learner.routes']['path']).read_bytes()
    assert len(nr)==len(lr)==1507*6*64
    mass_failures=[];ids_mismatches=0
    for t in range(1507):
        for l in range(6):
            off=(t*6+l)*64
            ni=np.frombuffer(nr,dtype='<i4',count=8,offset=off);li=np.frombuffer(lr,dtype='<i4',count=8,offset=off)
            ids_mismatches+=int(not np.array_equal(ni,li))
            nm=np.frombuffer(nr,dtype='<f4',count=8,offset=off+32).astype('f8')
            lm=np.frombuffer(lr,dtype='<f4',count=8,offset=off+32).astype('f8');d=float(np.max(np.abs(nm-lm)))
            if d>1e-6:mass_failures.append(dict(position=t,site=l,max_abs=d))
    assert ids_mismatches==result['native_comparison']['route_mismatch_calls']==0
    for item in receipts:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    value=dict(schema='ORIGINAL_FALCON_STORED_NATIVE_AUDIT_V1',inputs=receipts,
        interpretation='Post-hoc saved-output analysis; original prospective numerical FAIL remains unchanged.',
        metrics=metrics,bad_output_positions=bad,bad_output_positions_in_labels=sorted(set(bad)&set(positions)),
        mass_failure_calls=mass_failures,mass_failure_positions=sorted({r['position'] for r in mass_failures}),
        selected_ID_mismatch_calls=ids_mismatches,native_teacher_KL_minus_learner_teacher_KL=
            metrics['native']['KL_F64_mean']-metrics['learner']['KL_F64_mean'],
        new_model_forwards=0,new_teacher_queries=0,new_optimizer_updates=0,GPU_calls=0,quality_admission=False)
    write(a.out,value)
    print(json.dumps(dict(native_KL=metrics['native']['KL_F64_mean'],learner_KL=metrics['learner']['KL_F64_mean'],
        difference=value['native_teacher_KL_minus_learner_teacher_KL'],mass_failure_calls=len(mass_failures),
        bad_output_positions_in_labels=value['bad_output_positions_in_labels'],native_self_predictions=metrics['native']['current_input_ID_predictions'],
        result=str(a.out),sha256=sha(a.out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
