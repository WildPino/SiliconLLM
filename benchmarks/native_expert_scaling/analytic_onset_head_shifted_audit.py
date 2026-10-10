"""Shifted independent probability audit; preserve the original failed audit."""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
from categorical_onset_control import save,load_array,norm,relative
from causal_categorical_readout import export_check
sys.path.insert(0,str(SITE))
V,D,K=65537,256,255


def gates(rows,b):
    result={}
    for split in ['FIT','DEV']:
        chosen=[r for r in rows if r['split']==split];mkl=sum(r['KL'] for r in chosen)/24;dis=sum(r['disagreement_rate'] for r in chosen)/24;g=b['strict_gates']
        domain={d:dict(KL=sum(r['KL'] for r in chosen if r['domain']==d)/2,disagreement=sum(r['disagreement_rate'] for r in chosen if r['domain']==d)/2) for d in sorted({r['domain'] for r in chosen})}
        result[split]=dict(case_KL=mkl,case_disagreement=dis,domains=domain,first_case_KL=sum(r['KL_per_label'][0] for r in chosen)/24,first_disagreement=sum(r['disagreement_per_label'][0] for r in chosen),continuation_case_KL=sum(sum(r['KL_per_label'][1:])/(r['labels']-1) for r in chosen)/24,
            strict_flags=dict(mean_KL=mkl<=g['mean_KL'],mean_disagreement=dis<=g['mean_disagreement'],every_case_KL=all(r['KL']<=g['case_KL'] for r in chosen),every_case_disagreement=all(r['disagreement_rate']<=g['case_disagreement'] for r in chosen),domains=all(r['KL']<=g['case_KL'] and r['disagreement']<=g['case_disagreement'] for r in domain.values())),
            original_flags=dict(mean_KL=mkl<=1.,mean_disagreement=dis<=.20,domain_KL=all(r['KL']<=2 for r in domain.values()),domain_disagreement=all(r['disagreement']<=.35 for r in domain.values())))
    return result

def rows_for(b,H,norms64,h32,quant,independent,guard):
    import numpy as np
    rows=[];count=0
    for rec in b['records']:
        f=load_array(rec['features']);qbits=np.memmap(rec['teacher']['path'],mode='r',dtype='<u2',shape=(rec['labels'],V));kl=[];wrong=[];entropy=[];uncertainty=[];predicted=[]
        for first in range(0,rec['labels'],b['chunk']):
            last=min(first+b['chunk'],rec['labels']);z=f[first:last]@H.T;q=(qbits[first:last].astype('<u4')<<16).view('<f4').astype('f8');assert np.isfinite(z).all() and np.isfinite(q).all()
            if independent:
                lq=q-q.max(1,keepdims=True);lp=z-z.max(1,keepdims=True);lq-=np.logaddexp.reduce(lq,axis=1)[:,None];lp-=np.logaddexp.reduce(lp,axis=1)[:,None]
            else:
                lq=q-q.max(1,keepdims=True);lp=z-z.max(1,keepdims=True);lq-=np.log(np.exp(lq).sum(1,keepdims=True));lp-=np.log(np.exp(lp).sum(1,keepdims=True))
            pq=np.exp(lq);loss=np.sum(pq*(lq-lp),axis=1);ent=-np.sum(pq*lq,axis=1);ids=z.argmax(1);dis=(ids!=q.argmax(1)).astype('i4');e=np.asarray(rec['feature_error'][first:last]);epsilon=e*norms64+(np.linalg.norm(f[first:last],axis=1)+e)*(quant+b['native_rounding']['F32_gamma_dot']*h32)
            kl.extend(loss.tolist());wrong.extend(dis.tolist());entropy.extend(ent.tolist());predicted.extend(ids.tolist());uncertainty.extend(epsilon.tolist());count+=last-first;guard()
        row=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],labels=rec['labels'],KL=sum(kl)/rec['labels'],disagreement=sum(wrong),disagreement_rate=sum(wrong)/rec['labels'],KL_per_label=kl,disagreement_per_label=wrong,predicted_ids=predicted,entropy_per_label=entropy,score_uncertainty_per_label=uncertainty)
        rows.append(row);print(json.dumps(dict(stage='stored_probability_case',id=rec['id'],KL=row['KL'],first_KL=kl[0],count=count)),flush=True)
    assert count==8808;return rows,count


def audit(a):
    import numpy as np
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([1]);b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes());assert sha(a.binding)==a.binding_sha and r['binding_sha256']==t['binding_sha256']==b['parent_binding_sha256'] and r['freeze']==t['freeze']==b['parent_producer_freeze'];a.directory.mkdir(exist_ok=False);assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    def guard():assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes'] and not proc.children(recursive=True)
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item;guard()
    art=r['artifacts'];by={item['path']:item for item in t['outputs']}
    for key,item in art.items():assert by[item['path']]=={k:item[k] for k in ('path','bytes','sha256')}
    theta=load_array(art['theta']);J=load_array(art['adapter']);H=load_array(art['head_real']);H32=np.fromfile(art['head_native']['path'],dtype='<f4').reshape(V,D);assert np.array_equal(theta,load_array(b['theta']))
    deltaJ=float(np.max(abs(J-theta@load_array(b['W']))));deltaH=float(np.max(abs(H-load_array(b['decoder'])*b['gain']@J)));assert max(deltaJ,deltaH)<=b['numeric']['fusion_absolute'] and np.array_equal(H32,H.astype('<f4'))
    exported=export_check(b,art['packed']['path']);assert json.loads(json.dumps(exported))==r['export']
    with Path(art['packed']['path']).open('rb') as f:f.seek(b['fields']['head']['offset']);packed_H=np.frombuffer(f.read(b['fields']['head']['size']),dtype='<f4').reshape(V,D)
    assert np.array_equal(packed_H,H32) and r['radius_constraint_removed'] and r['parent_fit_radius']==16
    assert abs(norm(theta)-r['theta_norm'])<=b['numeric']['fusion_absolute']
    h64=float(np.linalg.norm(H,axis=1).max());h32=float(np.linalg.norm(H32.astype('f8'),axis=1).max());quant=float(np.linalg.norm(H32.astype('f8')-H,axis=1).max())
    for value,key in [(h64,'head_row_norm_F64_max'),(h32,'head_row_norm_F32_max'),(quant,'head_cast_row_error_max')]:assert abs(value-r[key])<=b['numeric']['fusion_absolute']
    rows,count=rows_for(b,H,h64,h32,quant,True,guard);maxKL=maxE=0.
    for actual,saved in zip(rows,r['records']):
        assert abs(actual['KL']-saved['KL'])<=b['numeric']['metric_absolute'] and actual['disagreement_rate']==saved['disagreement_rate']
        assert actual['id']==saved['id'] and actual['predicted_ids']==saved['predicted_ids'] and actual['disagreement_per_label']==saved['disagreement_per_label'] and actual['disagreement']==saved['disagreement']
        maxKL=max(maxKL,float(np.max(abs(np.asarray(actual['KL_per_label'])-saved['KL_per_label']))));maxE=max(maxE,float(np.max(abs(np.asarray(actual['score_uncertainty_per_label'])-saved['score_uncertainty_per_label']))));assert float(np.max(abs(np.asarray(actual['entropy_per_label'])-saved['entropy_per_label'])))<=b['numeric']['metric_absolute']
    write(a.directory/'normalization_checkpoint.json',dict(max_KL_absolute_delta=maxKL,max_uncertainty_absolute_delta=maxE,stored_gates_exact=gates(r['records'],b)==r['aggregate'],all8808_shifted_independent_probabilities_complete=True,all48_case_ID_entropy_checks_passed=True))
    assert maxKL<=b['numeric']['metric_absolute'] and maxE<=b['numeric']['bound_absolute'] and gates(r['records'],b)==r['aggregate']
    agg=gates(rows,b)
    for split in ['FIT','DEV']:
        assert agg[split]['strict_flags']==r['aggregate'][split]['strict_flags'] and agg[split]['original_flags']==r['aggregate'][split]['original_flags'] and abs(agg[split]['case_KL']-r['aggregate'][split]['case_KL'])<=b['numeric']['metric_absolute']
    selected={rec['id']:rec for rec in b['selected']};firstdelta=max(abs(row['KL_per_label'][0]-selected[row['id']]['reference_KL']) for row in rows if row['split']=='FIT');assert firstdelta<=b['numeric']['metric_absolute'] and agg['FIT']['first_disagreement']==0
    decision='ANALYTIC_FIRST_CORRECTION_PROXY_PASS_PENDING_NATIVE' if all(flag for split in agg.values() for flag in split['strict_flags'].values()) else 'ANALYTIC_FIRST_CORRECTION_PROXY_QUALITY_FAIL';assert decision==r['decision']
    for key in ['source_calls','native_calls','history_forwards','optimizer_updates','head_FG_calls','GPU_calls','reserved_queries','SVD_calls']:assert r[key]==0
    assert r['analytic_head_pairs']==1 and r['stored_head_label_products']==count==8808 and not r['quality_admission'] and not r['speed_admission']
    assert t['elapsed_seconds']<=b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['capture_limits']['OS_bytes'] and sum(item['bytes'] for item in t['outputs'])+t['result']['bytes']<=b['capture_limits']['output_bytes'];guard()
    write(a.out,dict(schema='ANALYTIC_ONSET_HEAD_SHIFTED_AUDIT_V1',freeze=a.freeze,parent_producer_freeze=b['parent_producer_freeze'],first_audit_fault=b['first_audit_fault'],normalization='Subtract row maxima before sequential logaddexp reduction; independent of producer exp/sum normalization',checkpoint=extent(a.directory/'normalization_checkpoint.json'),result=extent(a.source_result),binding=extent(a.binding),decision=decision,complete_input_output_hashes=True,all8808_independent_probabilities_KL_argmax_verified=True,all48_case_domain_flags_verified=True,all24_reference_first_labels_verified=True,all_forced_uncertainty_verified=True,original_packed_bytes_except_head_exact=True,full_fusion_verified=True,max_adapter_absolute_delta=deltaJ,max_fusion_absolute_delta=deltaH,max_KL_absolute_delta=maxKL,max_uncertainty_absolute_delta=maxE,first_reference_max_KL_delta=firstdelta,stored_head_labels_verified=count,source_calls=0,native_calls=0,optimizer_updates=0,GPU_calls=0,SVD_calls=0,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,scope='Full stored-only independent audit, no core/source/history/optimizer/SVD replay. Proxy probabilities not actual new-head C/chatbot admission.'))
    print(json.dumps(dict(decision=decision,seconds=time.monotonic()-start,maxKL=maxKL)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bind','capture-worker','audit-worker'):p.add_argument('--'+key,action='store_true')
    for key in ('binding','out','directory','source-result','protocol'):p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.audit_worker:audit(a)
    else:raise RuntimeError('Use separate qualified held launcher')
