"""Exact finite-logit KL identity and declared quadratic/cost diagnostics."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write


def main(a):
    import numpy as np,psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    assert extent(a.report)['sha256']==a.report_sha and extent(a.audit)['sha256']==a.audit_sha
    r=json.loads(a.report.read_bytes());ad=json.loads(a.audit.read_bytes())
    assert ad['report']==extent(a.report) and ad['decision']=='INDEPENDENT_STORED_PASS'
    assert not a.out.exists();b=json.loads(Path(r['binding']['path']).read_bytes());rd=r['readout']
    inputs=[extent(__file__),extent(a.report),extent(a.audit),r['binding'],rd['file'],rd['reference'],
        {k:b['qbar'][k] for k in ('path','bytes','sha256')},
        extent(DOC/'FUNCTION_COORDINATE_READOUT_DIAGNOSTIC_PROTOCOL_20261012.md')]+[x['file'] for x in r['bases']]
    def check():
        for item in inputs:
            assert time.monotonic()-start<300 and proc.memory_info().peak_wset<=2<<30
            assert not proc.children(recursive=True) and extent(item['path'])==item
    check();qbar=np.fromfile(b['qbar']['path'],'<f8')
    ref=np.fromfile(rd['reference']['path'],'<f8').reshape(72,65537)
    with np.load(rd['file']['path'],allow_pickle=False) as pack:scores=pack['scores']
    rows=[];max_identity=0.
    for i,label in enumerate(rd['labels']):
        logs=ref[i]-np.logaddexp.reduce(ref[i]);q=np.exp(logs);winner=int(np.argmax(ref[i]))
        top2=np.partition(ref[i],-2)[-2:];margin=float(top2.max()-top2.min())
        for mi,mode in [(0,'oracle_norm'),(1,'own_norm')]:
            delta=scores[mi,i]-ref[i];mean=float(q@delta);center=delta-mean
            # Exact KL(q||softmax(l+delta)) = log E_q exp(delta-E_q delta).
            exact=float(np.logaddexp.reduce(logs+center))
            local=.5*float(q@(center*center));mixmean=float(qbar@delta)
            mix=.5*float(qbar@((delta-mixmean)**2))
            target=next(x for x in rd['rows'] if x['mode']==mode and all(x[k]==v for k,v in label.items()))
            error=abs(exact-target['KL']);max_identity=max(max_identity,error);assert error<=1e-10
            predwinner=int(np.argmax(scores[mi,i]));rank=int(np.count_nonzero(scores[mi,i]>scores[mi,i,winner]))+1
            entropy=-float(q@logs)
            # Hoeffding bound on log moment generating function; exact but can be loose.
            range_bound=float((delta.max()-delta.min())**2/8)
            assert exact<=range_bound+1e-10
            rows.append(dict(**label,mode=mode,source_winner=winner,predicted_winner=predwinner,
                source_top2_margin=margin,source_entropy=entropy,predicted_rank_of_source_winner=rank,
                source_winner_FIT_qbar=float(qbar[winner]),exact_KL=exact,local_half_Fisher_variance=local,
                mixture_half_Fisher_variance=mix,centered_logit_error_max=float(np.max(np.abs(center))),
                Hoeffding_KL_upper_bound=range_bound,KL_identity_delta=error))
    summaries=[]
    for mode in ['oracle_norm','own_norm']:
        for wrong in [False,True]:
            rs=[x for x in rows if x['mode']==mode and (x['source_winner']!=x['predicted_winner'])==wrong]
            summaries.append(dict(mode=mode,wrong=wrong,count=len(rs),
                mean_KL=math.fsum(x['exact_KL'] for x in rs)/len(rs),
                mean_local_half_Fisher=math.fsum(x['local_half_Fisher_variance'] for x in rs)/len(rs),
                mean_mixture_half_Fisher=math.fsum(x['mixture_half_Fisher_variance'] for x in rs)/len(rs),
                median_source_margin=float(np.median([x['source_top2_margin'] for x in rs])),
                median_centered_error_max=float(np.median([x['centered_logit_error_max'] for x in rs])),
                median_FIT_qbar_winner=float(np.median([x['source_winner_FIT_qbar'] for x in rs])),
                worst_source_winner_rank=max(x['predicted_rank_of_source_winner'] for x in rs)))
    spectra=[]
    for item in r['bases']:
        with np.load(item['file']['path'],allow_pickle=False) as p:eig=p['eigenvalues']
        remaining=1.-np.cumsum(eig)/eig.sum()
        spectra.append(dict(boundary=item['boundary'],
            rank_for_FIT_quadratic_99pct=int(np.searchsorted(-remaining,-.01))+1,
            rank_for_FIT_quadratic_999pct=int(np.searchsorted(-remaining,-.001))+1,
            fixed_ranks=[dict(rank=k,FIT_metric_tail_fraction=float(remaining[k-1])) for k in [128,256,384,512,768,1024]],
            scope='Retained-support FIT oracle-norm reading objective only; not DEV error, KL, capacity or deployment.'))
    check()
    write(a.out,dict(schema='FUNCTION_COORDINATE_READOUT_DIAGNOSTIC_V1',freeze=a.freeze,command=sys.orig_argv,
        inputs=inputs,rows=rows,summary=summaries,spectra=spectra,max_KL_identity_delta=max_identity,
        algebra='KL(q||softmax(l+delta))=log E_q exp(delta-E_q delta). Quadratic Fisher is its local second-order expansion.',
        costs=dict(D=256,full_norm_metric_F32_bytes=256*256*4,norm_metric_dense_MAC=256*256,
            skip_F32_bytes_per_transition=256*256*4,skip_dense_MAC=256*256,
            projected_F32_head_bytes=65537*256*4,head_dense_MAC_per_token=65537*256,
            source_2048_to_256_encoding_MAC=2048*256,
            scope='Shape-derived dense upper operation counts/storage; not measured LUT throughput or DRAM bandwidth. '
                'The full-source encoder is unavailable as a cheap causal deployment input.'),
        before_after_hashes_exact=True,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
        scope='Stored logits algebra only; no coefficients changed or bases/source/runtime/refit/generator/native/GPU/T4 calls. '
            'Error and qbar association descriptive, not causation or an optimal metric claim. No new scientific gates.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','audit','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','audit-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
