"""Independent F64 metrics from stored intervention scores, no head contraction."""
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


def main(a):
    import numpy as np
    import psutil
    from original_latent_readout_attribution import aggregate,decision
    from original_joint_history_recovery_audit import aggregate_equal
    started=time.monotonic();psutil.Process().cpu_affinity(list(range(6)))
    r=json.loads(a.result.read_bytes());b=json.loads(a.binding.read_bytes())
    t=json.loads(a.result.with_suffix('.terminal.json').read_bytes())
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.result)
    assert t['binding_sha256']==r['binding_sha256']==sha(a.binding) and t['inputs_before_after_exact']
    assert t['elapsed_seconds']<=b['limits']['seconds']
    assert t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['limits']['OS_bytes']
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item,item['path']
    records={rec['id']:rec for rec in b['records']};audits=[];max_delta=0.;score_bytes=0
    assert [arm['name'] for arm in r['arms']]==['A','B']
    for arm in r['arms']:
        rows=[];assert len(arm['records'])==24 and {x['id'] for x in arm['records']}==set(records)
        assert arm['native']==next(h['native'] for h in b['heads'] if h['name']==arm['name'])
        for saved in arm['records']:
            rec=records[saved['id']];m=len(rec['positions']);V=65537
            assert saved['positions']==rec['positions'] and saved['domain']==rec['domain']
            assert saved['target']==b['targets'][rec['id']] and saved['teacher']==rec['logits']
            assert saved['scores']['bytes']==m*V*8;score_bytes+=saved['scores']['bytes']
            scores=np.memmap(saved['scores']['path'],mode='r',dtype='<f8',shape=(m,V))
            teacher=np.memmap(rec['logits']['path'],mode='r',dtype='<u2',shape=(m,V))
            losses=[];entropies=[];uniform=[];dis=0
            for j in range(m):
                p=scores[j].copy();q=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8')
                assert np.isfinite(p).all() and np.isfinite(q).all()
                dis+=int(np.argmax(p)!=np.argmax(q));p-=p.max();q-=q.max()
                p-=np.logaddexp.reduce(p);q-=np.logaddexp.reduce(q)
                loss=float(np.dot(np.exp(q),q-p));entropy=-float(np.dot(np.exp(q),q))
                losses.append(loss);entropies.append(entropy);uniform.append(math.log(V)-entropy)
                max_delta=max(max_delta,abs(loss-saved['KL_per_label'][j]),abs(entropy-saved['entropy_per_label'][j]),abs(uniform[-1]-saved['uniform_KL_per_label'][j]))
            assert dis==saved['disagreement'] and dis/m==saved['disagreement_rate']
            rows.append(dict(saved,KL=float(np.mean(losses)),KL_per_label=losses,
                teacher_entropy=float(np.mean(entropies)),entropy_per_label=entropies,
                uniform_KL=float(np.mean(uniform)),uniform_KL_per_label=uniform))
            del scores,teacher,p,q
        agg=aggregate(rows);delta=aggregate_equal(agg,arm['aggregate'])
        audits.append(dict(name=arm['name'],labels=agg['labels'],case_KL=agg['case_KL'],case_disagreement=agg['case_disagreement'],max_aggregate_delta=delta))
    assert max_delta<=b['gates']['reference_absolute']==1e-10
    branch,flags=decision(r['arms'],b['gates']);assert branch==r['decision'] and flags==r['decision_flags']
    assert score_bytes==r['new_label_score_bytes']==b['score_bytes']==4599124512
    assert r['label_head_contractions']==8772 and sum(x['labels'] for x in audits)==8772
    assert r['source_calls']==r['model_calls']==r['native_calls']==r['optimizer_updates']==r['GPU_calls']==r['reserved_queries']==0
    assert not r['quality_admission'] and not r['speed_admission'] and not r['useful_large_n_admission'] and r['physical_DRAM_bytes'] is None
    write(a.out,dict(schema='ORIGINAL_LATENT_READOUT_ATTRIBUTION_AUDIT_V1',result=extent(a.result),binding=extent(a.binding),
        audit_code=extent(Path(__file__)),arms=audits,max_absolute_metric_delta=max_delta,decision=branch,flags=flags,
        label_scores_bytes=score_bytes,complete_input_output_hashes=True,model_calls=0,head_contractions=0,
        optimizer_updates=0,seconds=time.monotonic()-started,
        limits='Recomputes full-V metrics from lossless stored scores; head contractions belong to held worker. '
            'Injected states are diagnostic and do not qualify chatbot/rate/general applicability.'))
    print(json.dumps(dict(audit=str(a.out),sha256=sha(a.out),decision=branch,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('result','binding','out'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
