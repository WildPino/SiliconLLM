"""Independent metrics, normalized-feature and scalar final-head witnesses."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
sys.path.insert(0,str(SITE))


def reference(q,p):
    import numpy as np
    q=q.astype('f8');p=p.astype('f8');assert np.isfinite(q).all() and np.isfinite(p).all()
    bad=int(np.argmax(q)!=np.argmax(p));q-=q.max();p-=p.max()
    q-=np.logaddexp.reduce(q);p-=np.logaddexp.reduce(p)
    kl=float(np.dot(np.exp(q),q-p));entropy=-float(np.dot(np.exp(q),q))
    return kl,bad,entropy,math.log(len(q))-entropy


def main(a):
    import numpy as np
    import psutil
    from source_final_readout import alignment
    from original_latent_readout_attribution import aggregate
    from original_joint_history_recovery_audit import aggregate_equal
    start=time.monotonic();psutil.Process().cpu_affinity(list(range(6)))
    r=json.loads(a.result.read_bytes());b=json.loads(a.binding.read_bytes());t=json.loads(a.result.with_suffix('.terminal.json').read_bytes())
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.result)
    assert t['binding_sha256']==r['binding_sha256']==sha(a.binding) and t['inputs_before_after_exact']
    assert t['elapsed_seconds']<=b['limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['limits']['OS_bytes']
    assert t['GPU_allocated_peak']<=b['limits']['GPU_allocated_bytes'] and t['GPU_reserved_peak']<=b['limits']['GPU_reserved_bytes']
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item,item['path']
    V,D=65537,2048;hf=b['fields']['lm_head.weight'];nf=b['fields']['model.final_layernorm.weight'];weight=b['weights']['path']
    head=np.memmap(weight,dtype='<u2',mode='r',offset=hf['offset'],shape=(V,D))
    gamma=(np.fromfile(weight,dtype='<u2',count=D,offset=nf['offset']).astype('<u4')<<16).view('<f4').astype('f8')
    recs={x['id']:x for x in b['records']};assert len(recs)==48 and [arm['name'] for arm in r['arms']]==['BF16','F64']
    by_format={arm['name']:{row['id']:row for row in arm['records']} for arm in r['arms']}
    for name in by_format:assert set(by_format[name])==set(recs)
    recomputed={'BF16':[],'F64':[]};max_metric=max_norm=max_scalar=0.;witnesses=0;rounding={p['id']:p for p in r['rounding']}
    for ident,rec in recs.items():
        n=len(rec['student_input_ids']);m=len(rec['positions']);saved={name:rows[ident] for name,rows in by_format.items()}
        assert saved['BF16']['features']==saved['F64']['features']
        for name,row in saved.items():
            assert row['positions']==rec['positions'] and row['split']==rec['split'] and row['domain']==rec['domain']
            assert row['scores']['bytes']==m*V*(2 if name=='BF16' else 8) and row['features']['bytes']==m*D*2
            assert row['teacher']==rec['logits'] and row['h24']==b['targets'][ident]
        bf=np.memmap(saved['BF16']['scores']['path'],mode='r',dtype='<u2',shape=(m,V))
        double=np.memmap(saved['F64']['scores']['path'],mode='r',dtype='<f8',shape=(m,V))
        teacher=np.memmap(rec['logits']['path'],mode='r',dtype='<u2',shape=(m,V))
        features=np.memmap(saved['BF16']['features']['path'],mode='r',dtype='<u2',shape=(m,D))
        h=np.memmap(b['targets'][ident]['path'],mode='r',dtype='<u2',shape=(n,D))
        raw=(h[rec['positions']].astype('<u4')<<16).view('<f4').astype('f8')
        exact=raw/np.sqrt(np.mean(raw*raw,axis=-1,keepdims=True)+b['eps'])*gamma
        actual=(features.astype('<u4')<<16).view('<f4').astype('f8')
        relative=math.sqrt(float(np.sum((exact-actual)**2))/max(float(np.sum(exact**2)),1e-300));max_norm=max(max_norm,relative)
        assert relative<=b['numeric_gates']['feature_relative_RMS'];del h,raw,exact
        values={name:[] for name in saved};round_losses=[];round_bad=0;score_delta=0.
        for j in range(m):
            q=(teacher[j].astype('<u4')<<16).view('<f4');bb=(bf[j].astype('<u4')<<16).view('<f4');dd=double[j]
            for name,pred in (('BF16',bb),('F64',dd)):
                kl,bad,entropy,uniform=reference(q,pred);values[name].append((kl,bad,entropy,uniform))
                row=saved[name];max_metric=max(max_metric,abs(kl-row['KL_per_label'][j]),abs(entropy-row['entropy_per_label'][j]),abs(uniform-row['uniform_KL_per_label'][j]))
            kl,bad,_,_=reference(bb,dd);round_losses.append(kl);round_bad+=bad
            max_metric=max(max_metric,abs(kl-rounding[ident]['KL_per_label'][j]));score_delta=max(score_delta,float(np.max(np.abs(bb.astype('f8')-dd))))
            if j in {0,m//2,m-1}:
                for v in {0,1,V-1,int(np.argmax(q))}:
                    w=(head[v].astype('<u4')<<16).view('<f4').astype('f8')
                    reference_score=math.fsum(float(x)*float(y) for x,y in zip(actual[j],w))*b['multiplier']
                    max_scalar=max(max_scalar,abs(reference_score-float(dd[v])));witnesses+=1
        assert round_bad==rounding[ident]['disagreement'] and score_delta==rounding[ident]['max_score_absolute_delta']
        assert abs(float(np.mean(round_losses))-rounding[ident]['case_KL_BF16_to_F64'])<=1e-10
        for name,vv in values.items():
            bad=sum(p[1] for p in vv);assert bad==saved[name]['disagreement'] and bad/m==saved[name]['disagreement_rate']
            recomputed[name].append(dict(saved[name],KL=float(np.mean([p[0] for p in vv])),KL_per_label=[p[0] for p in vv],
                teacher_entropy=float(np.mean([p[2] for p in vv])),entropy_per_label=[p[2] for p in vv],
                uniform_KL=float(np.mean([p[3] for p in vv])),uniform_KL_per_label=[p[3] for p in vv]))
        del bf,double,teacher,features,actual,q,bb,dd,w
    assert max_metric<=b['numeric_gates']['metric_absolute'] and max_scalar<=b['numeric_gates']['scalar_head_absolute']
    arms=[];max_aggregate=0.
    for arm in r['arms']:
        agg={s:aggregate([row for row in recomputed[arm['name']] if row['split']==s]) for s in ('FIT','DEV')}
        max_aggregate=max(max_aggregate,aggregate_equal(agg,arm['aggregates']));arms.append(dict(name=arm['name'],records=recomputed[arm['name']],aggregates=agg))
    decision,flags=alignment(arms,b['source_label_gate']);assert decision==r['decision'] and flags==r['alignment_flags']
    assert r['BF16_source_head_calls']==r['F64_head_contractions']==8808
    assert r['source_history_forwards']==r['source_generations']==r['model_instances']==r['optimizer_updates']==r['native_calls']==r['reserved_queries']==0
    assert not r['quality_admission'] and not r['speed_admission'] and not r['useful_large_n_admission'] and r['physical_DRAM_bytes'] is None
    write(a.out,dict(schema='SOURCE_FINAL_READOUT_AUDIT_V1',result=extent(a.result),binding=extent(a.binding),decision=decision,flags=flags,
        aggregates=[dict(name=arm['name'],aggregates=arm['aggregates']) for arm in arms],max_absolute_metric_delta=max_metric,
        max_aggregate_delta=max_aggregate,max_feature_relative_RMS=max_norm,max_scalar_head_absolute_delta=max_scalar,
        scalar_head_witnesses=witnesses,complete_input_output_hashes=True,source_history_forwards=0,new_head_contractions=0,
        optimizer_updates=0,seconds=time.monotonic()-start,
        scope='Stored score/norm/scalar audits only; no codec fit, causal source reconstruction, compact chatbot, useful speed/n/DRAM admission.'))
    print(json.dumps(dict(audit=str(a.out),sha256=sha(a.out),decision=decision,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('result','binding','out'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
