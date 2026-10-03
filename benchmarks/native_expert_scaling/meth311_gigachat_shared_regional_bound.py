#!/usr/bin/env python3
"""Shared256 output space plus regional128 residual: optimistic representation bound."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import psutil
import scipy.linalg
from safetensors import safe_open
import torch
from threadpoolctl import threadpool_limits
import meth310_gigachat_regional_subspace as R
C=R.C
P=R.P
DOC=C.DOC
PROTOCOL=DOC/'METH_311_GIGACHAT_SHARED_REGIONAL_BOUND_PROTOCOL_20261003.md'
PRIOR=DOC/'meth310_gigachat_regional_subspace_result.json'
PRIOR_SHA='1ce6972c99d2db3057b989ec327c7d3e015f2be2d125107cadc3d1ce4058cac1'
R_SHA='8fed130998c32a3cd1ba342958bce2ab179e9834b6bc64b666c30ab8fb352719'
COMMON=256
RANK=128
peak_checked=0

def budget(start):
    global peak_checked
    rss=psutil.Process().memory_info().rss;peak_checked=max(peak_checked,rss)
    elapsed=time.monotonic()-start
    assert elapsed<=20*60 and rss<=12*(1<<30),'M311 wall/RSS stop'
    return {'seconds':elapsed,'rss_bytes':rss,'maximum_checked_rss_bytes':peak_checked}

def evaluate(original,residual,q):
    norm=np.linalg.norm(original,axis=1);assert (norm>0).all()
    error=residual-(residual@q)@q.T
    return np.linalg.norm(error,axis=1)/norm,float(np.sum(error**2)),float(np.sum(original**2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=time.monotonic();stage='bindings';result={'experiment':'METH-311-GigaChat-common256-regional128-output-bound','rows':[],'shared_records':[]}
    torch.set_num_threads(6)
    try:
        for path in (Path(__file__),PROTOCOL,PRIOR,Path(R.__file__),Path(P.__file__),*[DOC/f'meth298_{d}_capture_result.json' for d in P.CAP_SHA]):P.committed(path)
        assert C.digest(PRIOR)==PRIOR_SHA and C.digest(Path(R.__file__))==R_SHA
        old=C.read(PRIOR);assert not any(old['gates'].values())
        assert C.digest(C.PRIOR)==C.PRIOR_SHA and C.digest(P.SCREEN)==P.SCREEN_SHA
        C.validate_build();source_prior=C.read(C.PRIOR)
        assert C.digest(C.SOURCE/'model.safetensors.index.json')==source_prior['source_index_sha256']
        assert C.digest(C.SOURCE/'config.json')==source_prior['source_config_sha256']
        arrays={};captures={}
        for d,sha in P.CAP_SHA.items():
            path=DOC/f'meth298_{d}_capture_result.json';assert C.digest(path)==sha
            info=C.read(path);assert all(info['gates'].values())
            cap=Path(info['capture']['path']);assert C.binding(cap)==info['capture']==old['capture_bindings'][d]
            data,co,n=C.parse_capture(cap,info['chunks']);assert n==info['records']
            for l in (1,13,25):
                for e in (0,32,63):
                    assert co[l,e,0]==co[l,e,1]==co[l,e,2]
                    assert np.array_equal(data[l,e,0],data[l,e,1])
            arrays[d]={k:v for k,v in data.items() if k[2] in (0,2)};captures[d]=info['capture'];del data,co
        result.update({'capture_bindings':captures,'prior310_sha256':PRIOR_SHA,'immutable310_controller_sha256':R_SHA,
                       'controller_sha256':C.digest(Path(__file__)),'protocol_sha256':C.digest(PROTOCOL),
                       'source_revision':source_prior['source_revision'],'common_width':COMMON,'regional_width':RANK,
                       'known_prior_exposure_failure_preserved':True,'query_keys_route_counts_exact310':True})
        refs=[v for v in source_prior['rows'] if v['projection']=='down_proj'];assert len(refs)==9
        stage='fit_only_shared_and_regional_output_spaces'
        with threadpool_limits(limits=6):
            for l in (1,13,25):
                pooled_x=np.concatenate([arrays['fit'][l,e,0] for e in (0,32,63)]).astype(np.float64)
                eig,u=scipy.linalg.eigh(pooled_x.T@pooled_x,check_finite=False,driver='evd')
                query=R.canonical(u[:,-16:]);del pooled_x
                oldquery=next(v for v in old['query_records'] if v['layer']==l)
                assert R.array_sha(query)==oldquery['fit_only_shared_query_sha256']
                targets={};source_hashes={};closure_max=0.
                for ref in [v for v in refs if v['layer']==l]:
                    e=ref['expert']
                    with safe_open(C.SOURCE/ref['source_shard'],framework='pt',device='cpu') as source:
                        tensor=source.get_tensor(f'model.layers.{l}.mlp.experts.{e}.down_proj.weight')
                    assert tensor.dtype==torch.bfloat16
                    sha=hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
                    assert sha==ref['tensor_bf16_sha256'];source_hashes[e]=sha
                    w=tensor.float().numpy().astype(np.float64);assert w.shape==(1536,1280)
                    targets[e]={}
                    for d in P.CAP_SHA:
                        z=arrays[d][l,e,2].astype(np.float64);y=z@w.T
                        closure=z[:,:640]@w[:,:640].T+z[:,640:]@w[:,640:].T
                        delta=float(np.max(np.linalg.norm(closure-y,axis=1)/np.linalg.norm(y,axis=1)))
                        assert delta<=1e-12;closure_max=max(closure_max,delta);targets[e][d]=y
                pooled_y=np.concatenate([targets[e]['fit'] for e in (0,32,63)])
                eig,u=scipy.linalg.eigh(pooled_y.T@pooled_y,check_finite=False,driver='evd')
                assert eig[-1]>0 and eig[0]>=-1e-8*eig[-1]
                common=R.canonical(u[:,-COMMON:]);assert np.max(np.abs(common.T@common-np.eye(COMMON)))<=1e-10
                retained=float(np.sum(eig[-COMMON:])/np.sum(eig))
                fit_res=pooled_y-(pooled_y@common)@common.T
                assert abs(1-float(np.sum(fit_res**2)/np.sum(pooled_y**2))-retained)<=1e-8
                result['shared_records'].append({'layer':l,'basis_sha256':R.array_sha(common),'fit_count':len(pooled_y),
                    'pooled_fit_retained_energy':retained,'maximum_source_split_closure':closure_max,
                    'query_sha256':R.array_sha(query)})
                del pooled_y,fit_res
                for e in (0,32,63):
                    y=targets[e];previous=next(v for v in old['rows'] if v['layer']==l and v['expert']==e)
                    keys=np.asarray(previous['keys'],dtype=np.float64);assert R.array_sha(keys)==previous['fit_only_keys_sha256']
                    ids={d:np.argmax(R.normalized(arrays[d][l,e,0].astype(np.float64)@query)@keys.T,axis=1) for d in P.CAP_SHA}
                    for d in P.CAP_SHA:assert np.bincount(ids[d],minlength=10).tolist()==previous['route_counts'][d]
                    residual={d:y[d]-(y[d]@common)@common.T for d in P.CAP_SHA}
                    global_q,_,_=R.basis(residual['fit']);_,global_ss,total=evaluate(y['test'],residual['test'],global_q)
                    error={d:np.zeros(len(y[d])) for d in P.CAP_SHA};sse={d:0. for d in P.CAP_SHA};oracle_sse=0.;children=[]
                    for j in range(10):
                        select={d:ids[d]==j for d in P.CAP_SHA}
                        rf=residual['fit'][select['fit']];rt=residual['test'][select['test']]
                        q,_,numerical=R.basis(rf)
                        if q.shape[1]:assert np.max(np.abs(common.T@q))<=1e-8,'shared/regional nonorthogonality'
                        row={'child':j,'fit_count':len(rf),'test_count':len(rt),'fit_numerical_residual_rank':numerical,
                             'basis_width':q.shape[1],'fit_residual_basis_sha256':R.array_sha(q)}
                        for d in P.CAP_SHA:
                            if np.any(select[d]):
                                err,ss,tt=evaluate(y[d][select[d]],residual[d][select[d]],q)
                                error[d][select[d]]=err;sse[d]+=ss;row[d+'_full_output_energy']=1-ss/tt
                        if len(rt):
                            oracle_q,_,_=R.basis(rt);_,ss,tt=evaluate(y['test'][select['test']],rt,oracle_q)
                            oracle_sse+=ss;oracle_energy=1-ss/tt
                            assert row['test_full_output_energy']<=oracle_energy+1e-8
                            row['test_fixed_shared_regional_oracle_energy']=oracle_energy
                        children.append(row);budget(start)
                    energies={d:1-sse[d]/float(np.sum(y[d]**2)) for d in P.CAP_SHA}
                    row={'layer':l,'expert':e,'source_tensor_sha256':source_hashes[e],'fit_count':len(y['fit']),'test_count':len(y['test']),
                         'summary':{d:P.stats(error[d]) for d in P.CAP_SHA},'energy':energies,'children':children,
                         'shared_only_test_energy':1-float(np.sum(residual['test']**2))/total,
                         'shared_global_rank128_test_energy':1-global_ss/total,
                         'fixed_shared_test_regional_oracle_energy':1-oracle_sse/total,
                         'prior310_test_energy':previous['energy']['test'],'gain_over310_energy':energies['test']-previous['energy']['test'],
                         'relative_l2_by_state':{d:error[d].tolist() for d in P.CAP_SHA},
                         'route_counts':previous['route_counts'],'keys_sha256':previous['fit_only_keys_sha256']}
                    result['rows'].append(row)
                    print(json.dumps({'completed':[l,e],'test_summary':row['summary']['test'],'test_energy':energies['test'],
                        'oracle_energy':row['fixed_shared_test_regional_oracle_energy'],'gain_over310':row['gain_over310_energy'],
                        'seconds':budget(start)['seconds']}),flush=True)
        rows=result['rows'];assert len(rows)==9
        result['gates']={'every_test_parent_median1_p95_5':all(r['summary']['test']['median']<=.01 and r['summary']['test']['p95']<=.05 for r in rows),
                         'every_test_parent_energy99':all(r['energy']['test']>=.99 for r in rows),
                         'no_matched_global_loss_over_half_point':all(r['energy']['test']>=r['shared_global_rank128_test_energy']-.005 for r in rows),
                         'each_child_fit32_test16':all(c['fit_count']>=32 and c['test_count']>=16 for r in rows for c in r['children'])}
        result['decision']='local_shared_regional_bound_eligible_only_for_new_cost_and_precision_checks' if all(result['gates'].values()) else 'reject_fixed_common256_PCA16_spherical10_fit_residual128_conversion'
        result['scope']='Fit-only common output space and regional residuals, actual nonlinear source outputs and exact310 input routing. Optimistic full-output projection coefficients, not realized shared input function/children, complete MoE mixture, quantized/native model, useful new n or accepted rate. Consumed diagnostic domains. Test residual oracle holds fitted common basis fixed and is not a population/general architecture bound.'
        result['runtime']=budget(start);C.write(args.out,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'runtime':result['runtime'],'result_sha256':C.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start})
        C.write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
