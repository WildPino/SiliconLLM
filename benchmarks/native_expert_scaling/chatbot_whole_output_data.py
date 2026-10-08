"""Immutable cached teacher trajectories for a complete-output learner.

No original model, tokenization, endpoint query or work at import.
"""
import hashlib
import json
from pathlib import Path
import struct


def load_case(summary):
    import numpy as np
    import torch
    metadata=Path(summary['metadata_path']).read_bytes()
    assert hashlib.sha256(metadata).hexdigest()==summary['metadata_sha256']
    meta=json.loads(metadata);assert meta['id']==summary['id'] and meta['split']==summary['split']
    prompt=meta['prompt_ids'];generated=meta['generated_ids'];k=len(generated);p=len(prompt)
    assert p==summary['prompt_tokens'] and k==summary['source_forwards'] and 1<=k<=16 and 1<=p<=128
    assert len(meta['frames'])==k and all(type(i) is int and 0<=i<151936 for i in prompt+generated)
    for step,frame in enumerate(meta['frames']):
        assert frame['step']==step and frame['next_id']==generated[step]
        assert frame['decision_history_position']==p-1+step
        assert frame['input_ids']==(prompt if step==0 else [generated[step-1]])
        assert frame['logits_offset']==24+step*303872 and frame['logits_bytes']==303872
    payload=Path(summary['logits_path']).read_bytes()
    assert len(payload)==summary['logits_bytes']==24+k*303872
    assert hashlib.sha256(payload).hexdigest()==summary['logits_sha256']
    assert struct.unpack('<8s4I',payload[:24])==(b'QWGL0001',151936,2,0,0)
    bits=np.frombuffer(payload,dtype='<u2',offset=24).reshape(k,151936).copy(order='C')
    assert not ((bits&0x7fff)>=0x7f80).any()
    for step,frame in enumerate(meta['frames']):
        assert hashlib.sha256(memoryview(bits[step])).hexdigest()==frame['logits_sha256']
    input_ids=prompt+generated[:-1];positions=list(range(p-1,p+k-1))
    assert len(input_ids)==p+k-1<=143 and positions[-1]==len(input_ids)-1
    return dict(id=meta['id'],split=meta['split'],category=summary['category'],prompt_length=p,label_count=k,
        input_ids=input_ids,positions=positions,source_generated_ids=generated,
        teacher_logits=torch.from_numpy(bits).view(torch.bfloat16)[None].contiguous())


def logit_metrics(student,teacher,source_ids):
    import torch
    from torch.nn import functional as F
    assert student.shape==teacher.shape and student.ndim==3 and student.shape[0]==1 and student.shape[2]==151936
    assert student.dtype==teacher.dtype==torch.bfloat16 and torch.isfinite(student).all() and torch.isfinite(teacher).all()
    p_lp=F.log_softmax(teacher.float(),dim=-1);q_lp=F.log_softmax(student.float(),dim=-1)
    per=(p_lp.exp()*(p_lp-q_lp)).sum(-1)[0];ids=student[0].argmax(-1)
    target=torch.tensor(source_ids,dtype=torch.int64,device=student.device)
    assert len(target)==len(per) and bool(torch.isfinite(per).all())
    return dict(KL_F32_per_label=per.detach().cpu().tolist(),mean_KL=float(per.mean()),
        student_ids=ids.detach().cpu().tolist(),disagreements=int((ids!=target).sum()),labels=len(per),
        source_ID_log_probability=q_lp[0].gather(1,target[:,None]).squeeze(1).detach().cpu().tolist())


def summarize(cases):
    result={}
    for split in ('fit','development'):
        subset=[v for v in cases if v['split']==split];n=sum(v['labels'] for v in subset)
        assert len(subset)==(160 if split=='fit' else 40) and n==(2437 if split=='fit' else 613)
        categories={}
        for category in sorted({v['category'] for v in subset}):
            group=[v for v in subset if v['category']==category];count=sum(v['labels'] for v in group)
            categories[category]=dict(labels=count,mean_KL=sum(sum(v['KL_F32_per_label']) for v in group)/count,
                disagreement_fraction=sum(v['disagreements'] for v in group)/count)
        result[split]=dict(cases=len(subset),labels=n,mean_KL=sum(sum(v['KL_F32_per_label']) for v in subset)/n,
            disagreement_fraction=sum(v['disagreements'] for v in subset)/n,categories=categories)
    return result


def transfer_gates(initial,final):
    gates={}
    for split in ('fit','development'):
        current=final[split];baseline=initial[split]
        gates[split+'_absolute_KL_le_0p10']=current['mean_KL']<=.10
        gates[split+'_disagreement_le_0p15']=current['disagreement_fraction']<=.15
        gates[split+'_relative_KL_le_0p20']=current['mean_KL']<=.2*baseline['mean_KL']
    gates['DEV_each_category_KL_le_0p20']=all(v['mean_KL']<=.2 for v in final['development']['categories'].values())
    gates['DEV_each_category_disagreement_le_0p25']=all(v['disagreement_fraction']<=.25 for v in final['development']['categories'].values())
    return gates
