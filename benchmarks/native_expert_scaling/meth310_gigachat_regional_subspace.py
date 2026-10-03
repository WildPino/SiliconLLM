#!/usr/bin/env python3
"""Fit-only input regions and full nonlinear-output rank128 bounds, not a student."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import psutil
import scipy.linalg
from safetensors import safe_open
import torch
from threadpoolctl import threadpool_limits
import meth299_gigachat_block_selection as P
C=P.C
DOC=C.DOC
PROTOCOL=DOC/'METH_310_GIGACHAT_REGIONAL_SUBSPACE_PROTOCOL_20261003.md'
RANK=128
CHILDREN=10
QUERY=16

def array_sha(x):
    return hashlib.sha256(np.asarray(x,dtype='<f8',order='C').tobytes()).hexdigest()

def budget(start):
    r={'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss}
    assert r['seconds']<=20*60 and r['rss_bytes']<=12*(1<<30),'M310 resource stop'
    return r

def normalized(x):
    norm=np.linalg.norm(x,axis=1)
    assert np.isfinite(x).all() and (norm>0).all(),'zero/nonfinite input query'
    return x/norm[:,None]

def canonical(q):
    q=q.copy()
    for i in range(q.shape[1]):
        j=int(np.argmax(np.abs(q[:,i])))
        if q[j,i]<0:q[:,i]*=-1
    return q

def fit_keys(x,seed):
    rng=np.random.default_rng(seed)
    keys=[x[int(rng.integers(len(x)))].copy()]
    while len(keys)<CHILDREN:
        proximity=np.max(x@np.stack(keys).T,axis=1)
        keys.append(x[int(np.argmin(proximity))].copy())
    keys=np.stack(keys);previous=None
    for iteration in range(100):
        ids=np.argmax(x@keys.T,axis=1)
        if previous is not None and np.array_equal(ids,previous):break
        previous=ids.copy()
        for j in range(CHILDREN):
            selected=x[ids==j]
            if len(selected):
                mean=selected.mean(axis=0);norm=np.linalg.norm(mean)
                if norm>0:keys[j]=mean/norm
    ids=np.argmax(x@keys.T,axis=1)
    return keys,ids,iteration+1

def basis(y):
    if not len(y):return np.zeros((1536,0)),0.,0
    _,s,vh=scipy.linalg.svd(y,full_matrices=False,check_finite=False,lapack_driver='gesdd')
    numerical=int(np.sum(s>np.finfo(np.float64).eps*max(y.shape)*s[0]))
    n=min(RANK,numerical);q=canonical(vh[:n].T)
    assert np.max(np.abs(q.T@q-np.eye(n)))<=1e-10
    energy=float(np.sum(s[:n]**2)/np.sum(s**2))
    return q,energy,numerical

def errors(y,q):
    norm=np.linalg.norm(y,axis=1);assert (norm>0).all()
    residual=y-(y@q)@q.T
    return np.linalg.norm(residual,axis=1)/norm,float(np.sum(residual**2)),float(np.sum(y**2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=time.monotonic();stage='bindings';result={'experiment':'METH-310-GigaChat-fit-only-input-regions-full-output-subspace','rows':[]}
    torch.set_num_threads(6)
    try:
        for path in (Path(__file__),PROTOCOL,P.SCREEN,*[DOC/f'meth298_{d}_capture_result.json' for d in P.CAP_SHA]):P.committed(path)
        assert C.digest(P.SCREEN)==P.SCREEN_SHA and C.digest(C.PRIOR)==C.PRIOR_SHA
        # Existing immutable capture apparatus remains pinned; do not rerun collection.
        C.validate_build();prior=C.read(C.PRIOR)
        assert C.digest(C.SOURCE/'model.safetensors.index.json')==prior['source_index_sha256']
        assert C.digest(C.SOURCE/'config.json')==prior['source_config_sha256']
        arrays={};coords={};captures={}
        for d,sha in P.CAP_SHA.items():
            path=DOC/f'meth298_{d}_capture_result.json';assert C.digest(path)==sha
            info=C.read(path);assert all(info['gates'].values())
            cap=Path(info['capture']['path']);assert C.binding(cap)==info['capture']
            data,co,n=C.parse_capture(cap,info['chunks']);assert n==info['records']
            for l in (1,13,25):
                for e in (0,32,63):
                    assert co[l,e,0]==co[l,e,1]==co[l,e,2],'unaligned nonlinear states'
                    assert np.array_equal(data[l,e,0],data[l,e,1])
            arrays[d]={k:v for k,v in data.items() if k[2] in (0,2)}
            coords[d]={k:v for k,v in co.items() if k[2] in (0,2)}
            captures[d]=info['capture'];del data,co
        result.update({'capture_bindings':captures,'controller_sha256':C.digest(Path(__file__)),
                       'protocol_sha256':C.digest(PROTOCOL),'prior180_sha256':C.PRIOR_SHA,'prior298_sha256':P.SCREEN_SHA,
                       'source_revision':prior['source_revision'],'rank':RANK,'query_width':QUERY,'children_per_parent':CHILDREN,
                       'query_records':[]})
        refs=[r for r in prior['rows'] if r['projection']=='down_proj'];assert len(refs)==9
        stage='fit_only_query_regions_and_output_subspace'
        with threadpool_limits(limits=6):
            for l in (1,13,25):
                pooled=np.concatenate([arrays['fit'][l,e,0] for e in (0,32,63)]).astype(np.float64)
                eig,u=scipy.linalg.eigh(pooled.T@pooled,check_finite=False,driver='evd')
                assert eig[-1]>0 and eig[0]>=-1e-8*eig[-1]
                query=canonical(u[:,-QUERY:]);del pooled
                result['query_records'].append({'layer':l,'fit_only_shared_query_sha256':array_sha(query),
                    'retained_input_energy':float(np.sum(eig[-QUERY:])/np.sum(eig))})
                for ref in [r for r in refs if r['layer']==l]:
                    e=ref['expert']
                    with safe_open(C.SOURCE/ref['source_shard'],framework='pt',device='cpu') as source:
                        tensor=source.get_tensor(f'model.layers.{l}.mlp.experts.{e}.down_proj.weight')
                    assert tensor.dtype==torch.bfloat16
                    assert hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()==ref['tensor_bf16_sha256']
                    w=tensor.float().numpy().astype(np.float64);assert w.shape==(1536,1280)
                    y={d:arrays[d][l,e,2].astype(np.float64)@w.T for d in P.CAP_SHA}
                    for d in P.CAP_SHA:
                        z=arrays[d][l,e,2].astype(np.float64)
                        closure=z[:,:640]@w[:,:640].T+z[:,640:]@w[:,640:].T
                        assert np.max(np.linalg.norm(closure-y[d],axis=1)/np.linalg.norm(y[d],axis=1))<=1e-12
                    qx={d:normalized(arrays[d][l,e,0].astype(np.float64)@query) for d in P.CAP_SHA}
                    seed=310000+l*100+e;keys,fit_ids,iterations=fit_keys(qx['fit'],seed)
                    ids={'fit':fit_ids,'test':np.argmax(qx['test']@keys.T,axis=1)}
                    global_q,_,_=basis(y['fit']);global_error,global_sse,total=errors(y['test'],global_q)
                    err={d:np.zeros(len(y[d])) for d in P.CAP_SHA};sse={d:0. for d in P.CAP_SHA};oracle_sse=0.;children=[]
                    for j in range(CHILDREN):
                        yf=y['fit'][ids['fit']==j];yt=y['test'][ids['test']==j]
                        q,fit_energy,numerical=basis(yf)
                        row={'child':j,'fit_count':len(yf),'test_count':len(yt),'fit_numerical_rank':numerical,
                             'basis_width':q.shape[1],'fit_output_basis_sha256':array_sha(q),'fit_svd_energy':fit_energy}
                        for d in P.CAP_SHA:
                            select=ids[d]==j;subset=y[d][select]
                            if len(subset):
                                error,ss,tt=errors(subset,q);err[d][select]=error;sse[d]+=ss
                                row[d+'_energy']=1-ss/tt
                                if d=='fit':assert abs(row[d+'_energy']-fit_energy)<=1e-8
                        if len(yt):
                            oracle_q,oracle_energy,_=basis(yt)
                            _,os,ot=errors(yt,oracle_q);oracle_sse+=os
                            assert row['test_energy']<=oracle_energy+1e-8
                            row['test_oracle_energy']=oracle_energy
                        children.append(row);budget(start)
                    row={'layer':l,'expert':e,'source_tensor_sha256':ref['tensor_bf16_sha256'],
                         'fit_only_keys_sha256':array_sha(keys),'keys':keys.tolist(),'seed':seed,'kmeans_iterations':iterations,
                         'children':children,'fit_count':len(y['fit']),'test_count':len(y['test']),
                         'global_rank128_test_energy':1-global_sse/total,'regional_test_oracle_energy':1-oracle_sse/total,
                         'summary':{d:P.stats(err[d]) for d in P.CAP_SHA},
                         'energy':{d:1-sse[d]/float(np.sum(y[d]**2)) for d in P.CAP_SHA},
                         'relative_l2_by_state':{d:err[d].tolist() for d in P.CAP_SHA},
                         'route_counts':{d:np.bincount(ids[d],minlength=10).tolist() for d in P.CAP_SHA}}
                    result['rows'].append(row)
                    print(json.dumps({'completed':[l,e],'test_summary':row['summary']['test'],
                        'test_energy':row['energy']['test'],'test_oracle_energy':row['regional_test_oracle_energy'],
                        'seconds':budget(start)['seconds']}),flush=True)
        rows=result['rows'];assert len(rows)==9
        result['gates']={'every_test_parent_median1_p95_5':all(r['summary']['test']['median']<=.01 and r['summary']['test']['p95']<=.05 for r in rows),
                         'every_test_parent_energy99':all(r['energy']['test']>=.99 for r in rows),
                         'no_global_rank128_loss_over_half_point':all(r['energy']['test']>=r['global_rank128_test_energy']-.005 for r in rows),
                         'each_child_fit32_test16':all(c['fit_count']>=32 and c['test_count']>=16 for r in rows for c in r['children'])}
        result['decision']='local_representation_eligible_only_for_new_cost_and_precision_checks' if all(result['gates'].values()) else 'reject_fixed_PCA16_spherical10_fit_rank128_regional_conversion'
        result['scope']='Actual source post-SwiGLU down states/full BF16 mathematical outputs, input-only fit routing. FP64 output oracle coefficients require full source outputs; no realized child/student, quantized router, native cost, full-model quality or useful added capacity. Previously consumed298 domains; future whole quality needs untouched data. Held-out regional oracle is diagnostic and finite-sample optimistic.'
        result['runtime']=budget(start);C.write(args.out,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'result_sha256':C.digest(args.out),'runtime':result['runtime']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start})
        C.write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
