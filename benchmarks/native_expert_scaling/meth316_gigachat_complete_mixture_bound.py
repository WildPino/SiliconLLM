#!/usr/bin/env python3
"""Complete source MoE: shared256 + four regional128 projections and joint span oracle."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time
import numpy as np
import psutil
import scipy
import scipy.linalg
import torch
from threadpoolctl import threadpool_limits
import meth316_joint_projection as J
import meth310_gigachat_regional_subspace as R
C=R.C
P=R.P
DOC=C.DOC
PRIOR=DOC/'meth314_gigachat_mixture_targets_result.json'
PRIOR_SHA='a46b9395df437442e113651c06d9ee8fc253c6731f9026deaf431a5fb01b30b5'
PROTOCOL=DOC/'METH_316_GIGACHAT_COMPLETE_MIXTURE_BOUND_PROTOCOL_20261003.md'
peak_checked=0

def budget(start):
    global peak_checked
    rss=psutil.Process().memory_info().rss;peak_checked=max(peak_checked,rss);seconds=time.monotonic()-start
    assert seconds<=20*60 and rss<=12*(1<<30),'M316 wall/RSS stop'
    return {'seconds':seconds,'rss_bytes':rss,'maximum_checked_rss_bytes':peak_checked}

def load_case(binding,count):
    path=Path(binding['path']);assert C.binding(path)==binding
    with np.load(path,allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
    assert set(data)=={'input','coordinates','selected_parents','gates','parent_outputs','shared_output','full_mixture_output'}
    shapes={'input':(count,1536),'coordinates':(count,3),'selected_parents':(count,4),'gates':(count,4),
            'parent_outputs':(count,4,1536),'shared_output':(count,1536),'full_mixture_output':(count,1536)}
    for key,shape in shapes.items():
        assert data[key].shape==shape and np.isfinite(data[key]).all(),key
        assert data[key].dtype==(np.int32 if key=='coordinates' else np.int16 if key=='selected_parents' else np.float32),key
    ids=data['selected_parents'];assert ((ids>=0)&(ids<64)).all() and all(len(set(v.tolist()))==4 for v in ids)
    assert (data['gates']>0).all() and np.max(np.abs(data['gates'].sum(axis=1)-1))<=1e-6
    coordinates=[tuple(v) for v in data['coordinates'].tolist()];assert coordinates==sorted(set(coordinates))
    full=(np.sum(data['parent_outputs'].astype(np.float64)*data['gates'].astype(np.float64)[:,:,None],axis=1)+data['shared_output'].astype(np.float64)).astype(np.float32)
    assert np.array_equal(full,data['full_mixture_output']),'full source mixture reconstruction'
    assert (np.linalg.norm(full.astype(np.float64),axis=1)>0).all()
    return data

def output_basis(y):
    if not len(y) or not np.any(y):return np.zeros((1536,0)),0.,0
    return R.basis(y)

def summary(error,reference):
    norm=np.linalg.norm(reference,axis=1);assert (norm>0).all()
    values=np.linalg.norm(error,axis=1)/norm
    return {'relative_l2_by_state':values.tolist(),'summary':P.stats(values),
            'retained_energy':1-float(np.sum(error**2)/np.sum(reference**2))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start=time.monotonic();stage='bindings';result={'experiment':'METH-316-equivalent-conditioned-joint-span','layers':[]}
    torch.set_num_threads(6)
    try:
        for path in (Path(__file__),PROTOCOL,PRIOR,Path(R.__file__),Path(P.__file__),Path(J.__file__)):P.committed(path)
        assert C.digest(PRIOR)==PRIOR_SHA and C.digest(Path(R.__file__))=='8fed130998c32a3cd1ba342958bce2ab179e9834b6bc64b666c30ab8fb352719'
        partial_path=DOC/'meth315_gigachat_complete_mixture_bound_result.failure.json'
        P.committed(partial_path);assert C.digest(partial_path)=='41cf6365c6268d2e47d76fadd3f309c2adeb99db2f85e3bc711c5456d8a265ae'
        previous=C.read(partial_path)
        result['numerical_controls']=J.controls()
        result['projection_helper_sha256']=C.digest(Path(J.__file__))
        prior=C.read(PRIOR);assert all(prior['gates'].values()) and len(prior['cases'])==6
        result.update({'qualified314_sha256':PRIOR_SHA,'controller_sha256':C.digest(Path(__file__)),
                       'protocol_sha256':C.digest(PROTOCOL),'libraries':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
                       'gates':{},'source_archive_bindings':[v['archive'] for v in prior['cases']]})
        with threadpool_limits(limits=6):
            for layer in (1,13,25):
                stage=f'layer{layer}_archive_and_fit'
                info={d:next(v for v in prior['cases'] if v['layer']==layer and v['domain']==d) for d in ('fit','test')}
                data={d:load_case(info[d]['archive'],info[d]['unique_source_inputs']) for d in info}
                for d in data:assert np.bincount(data[d]['selected_parents'].ravel(),minlength=64).tolist()==info[d]['selected_expert_counts']
                x=data['fit']['input'].astype(np.float64);y={d:data[d]['full_mixture_output'].astype(np.float64) for d in data}
                eig,u=scipy.linalg.eigh(x.T@x,check_finite=False,driver='evd');assert eig[-1]>0 and eig[0]>=-1e-8*eig[-1]
                query=R.canonical(u[:,-16:]);del x
                eig,u=scipy.linalg.eigh(y['fit'].T@y['fit'],check_finite=False,driver='evd');assert eig[-1]>0 and eig[0]>=-1e-8*eig[-1]
                shared=R.canonical(u[:,-256:]);assert np.max(np.abs(shared.T@shared-np.eye(256)))<=1e-10
                shared_prediction={d:(y[d]@shared)@shared.T for d in data}
                shared_energy=float(np.sum(eig[-256:])/np.sum(eig));assert abs(1-float(np.sum((y['fit']-shared_prediction['fit'])**2)/np.sum(y['fit']**2))-shared_energy)<=1e-8
                qx={d:R.normalized(data[d]['input'].astype(np.float64)@query) for d in data}
                candidate={d:shared_prediction[d].copy() for d in data};global_control={d:shared_prediction[d].copy() for d in data}
                children_ids={d:np.full((len(y[d]),4),-1,dtype=np.int16) for d in data};bases={};records=[]
                entry={'layer':layer,'fit_count':len(y['fit']),'test_count':len(y['test']),
                       'shared_basis_sha256':R.array_sha(shared),'query_sha256':R.array_sha(query),'shared_fit_energy':shared_energy,
                       'parents':records,'cases':{}};result['layers'].append(entry)
                for expert in range(64):
                    positions={d:np.nonzero(data[d]['selected_parents']==expert) for d in data}
                    fi,fs=positions['fit'];assert len(fi)>=10,'known source fit coverage changed'
                    keys,fit_ids,iterations=R.fit_keys(qx['fit'][fi],315000+layer*100+expert)
                    ids={'fit':fit_ids,'test':np.argmax(qx['test'][positions['test'][0]]@keys.T,axis=1)}
                    source={d:data[d]['parent_outputs'][positions[d]].astype(np.float64) for d in data}
                    residual={d:source[d]-(source[d]@shared)@shared.T for d in data}
                    global_q,_,_=output_basis(residual['fit'])
                    for d in data:
                        ix,slots=positions[d];weights=data[d]['gates'][ix,slots].astype(np.float64)
                        global_control[d][ix]+=weights[:,None]*((residual[d]@global_q)@global_q.T)
                        children_ids[d][ix,slots]=expert*10+ids[d]
                    child_records=[]
                    for child in range(10):
                        pick={d:ids[d]==child for d in data};q,energy,numerical=output_basis(residual['fit'][pick['fit']])
                        if q.shape[1]:assert np.max(np.abs(shared.T@q))<=1e-8
                        bases[expert*10+child]=q
                        child_records.append({'child':child,'fit_count':int(pick['fit'].sum()),'test_count':int(pick['test'].sum()),
                            'width':q.shape[1],'numerical_rank':numerical,'residual_fit_energy':energy,'basis_sha256':R.array_sha(q)})
                        for d in data:
                            ix,slots=positions[d];selected=pick[d];weights=data[d]['gates'][ix[selected],slots[selected]].astype(np.float64)
                            candidate[d][ix[selected]]+=weights[:,None]*((residual[d][selected]@q)@q.T)
                    records.append({'expert':expert,'fit_count':len(fi),'test_count':len(positions['test'][0]),'seed':315000+layer*100+expert,
                        'key_sha256':R.array_sha(keys),'keys':keys.tolist(),'kmeans_iterations':iterations,'children':child_records})
                    budget(start)
                assert all((v>=0).all() for v in children_ids.values())
                previous_layer=next((v for v in previous['layers'] if v['layer']==layer),None)
                if previous_layer:
                    assert entry['shared_basis_sha256']==previous_layer['shared_basis_sha256']
                    assert entry['query_sha256']==previous_layer['query_sha256']
                    assert entry['parents']==previous_layer['parents'],'all fitted fields must match315'
                entry['exact_previous_fit_field_match']=previous_layer is not None
                cache=OrderedDict();hits=0;misses=0
                for domain in ('fit','test'):
                    stage=f'layer{layer}_{domain}_joint_oracle'
                    residual=y[domain]-shared_prediction[domain];oracle_error=np.empty_like(residual);ranks=[]
                    metrics={'cholesky_states':0,'svd_fallback_states':0,'maximum_normal_equation_relative':0.,
                             'maximum_fixed_index_svd_difference':0.,'svd_reference_count':0,'minimum_cholesky_rcond':1.}
                    with threadpool_limits(limits=1):
                        for i in range(len(residual)):
                            key=tuple(sorted(children_ids[domain][i].tolist()))
                            if key in cache:record,matrix=cache.pop(key);hits+=1
                            else:
                                matrix=np.concatenate([bases[k] for k in key],axis=1)
                                record=J.factor(matrix);misses+=1
                                if record[0]=='svd' and record[1].shape[1]:
                                    span=record[1]
                                    assert np.max(np.abs(span.T@span-np.eye(span.shape[1])))<=1e-10
                                    assert np.max(np.abs(shared.T@span))<=1e-8
                            cache[key]=(record,matrix)
                            if len(cache)>64:cache.popitem(last=False)
                            prediction,rank=J.project(record,residual[i]);oracle_error[i]=residual[i]-prediction;ranks.append(rank)
                            fullnorm=float(np.linalg.norm(y[domain][i]))
                            normal=float(np.max(np.abs(matrix.T@oracle_error[i]))/fullnorm) if matrix.shape[1] else 0.
                            assert normal<=1e-8,'every-state projection optimality'
                            metrics['maximum_normal_equation_relative']=max(metrics['maximum_normal_equation_relative'],normal)
                            metrics['cholesky_states' if record[0]=='cholesky' else 'svd_fallback_states']+=1
                            if record[0]=='cholesky':metrics['minimum_cholesky_rcond']=min(metrics['minimum_cholesky_rcond'],record[3])
                            if i<32:
                                span=J.svd_span(matrix);reference=span@(span.T@residual[i])
                                difference=float(np.linalg.norm(prediction-reference)/fullnorm)
                                assert difference<=1e-8 and rank==span.shape[1],'fixed-index SVD equivalence'
                                metrics['maximum_fixed_index_svd_difference']=max(metrics['maximum_fixed_index_svd_difference'],difference)
                                metrics['svd_reference_count']+=1
                            if i%512==0:
                                runtime=budget(start)
                                if i%2048==0:print(json.dumps({'joint_oracle_progress':[layer,domain,i,len(residual)],'seconds':runtime['seconds']}),flush=True)
                    errors=y[domain]-candidate[domain];norm=np.linalg.norm(y[domain],axis=1)
                    assert np.all(np.linalg.norm(oracle_error,axis=1)<=np.linalg.norm(errors,axis=1)+1e-10*norm),'joint oracle cannot be worse than contained candidate'
                    values={'regional_sum':summary(errors,y[domain]),'global_parent_control':summary(y[domain]-global_control[domain],y[domain]),
                            'shared_only':summary(residual,y[domain]),'joint_span_oracle':summary(oracle_error,y[domain])}
                    assert np.array_equal(np.linalg.norm(-y[domain],axis=1)/norm,np.ones(len(norm))),'zero-output control'
                    values['selected_children']=children_ids[domain].tolist();values['joint_residual_rank_by_state']=ranks
                    values['numerical_projection_controls']=metrics
                    if previous_layer and domain in previous_layer['cases']:
                        old=previous_layer['cases'][domain]
                        assert values['selected_children']==old['selected_children']
                        assert ranks==old['joint_residual_rank_by_state']
                        for arm in ('regional_sum','global_parent_control','shared_only','joint_span_oracle'):
                            assert np.max(np.abs(np.asarray(values[arm]['relative_l2_by_state'])-np.asarray(old[arm]['relative_l2_by_state'])))<=1e-8,'all completed315 row errors equivalent'
                        values['every_previous_case_row_equivalent']=True
                    entry['cases'][domain]=values
                    print(json.dumps({'completed':[layer,domain],'regional':values['regional_sum']['summary'],
                        'oracle':values['joint_span_oracle']['summary'],'regional_energy':values['regional_sum']['retained_energy'],
                        'oracle_energy':values['joint_span_oracle']['retained_energy'],'seconds':budget(start)['seconds']}),flush=True)
                entry['oracle_cache']={'capacity':64,'hits':hits,'misses':misses}
                cache.clear();del data,candidate,global_control,bases,shared_prediction,y,residual,oracle_error
        def error_gates(arm):
            cases=[l['cases'][d][arm] for l in result['layers'] for d in ('fit','test')]
            return {'every_case_median1_p95_5':all(v['summary']['median']<=.01 and v['summary']['p95']<=.05 for v in cases),
                    'every_case_energy99':all(v['retained_energy']>=.99 for v in cases)}
        result['gates'].update(error_gates('regional_sum'))
        result['gates']['no_matched_global_loss_over_half_point']=all(l['cases'][d]['regional_sum']['retained_energy']>=l['cases'][d]['global_parent_control']['retained_energy']-.005 for l in result['layers'] for d in ('fit','test'))
        result['gates']['every_child_fit32_test16']=all(c['fit_count']>=32 and c['test_count']>=16 for l in result['layers'] for p in l['parents'] for c in p['children'])
        result['diagnostic_joint_oracle_gates']=error_gates('joint_span_oracle')
        result['decision']='close_fixed_complete_mixture_projection_rule_before_student_training' if not all(result['gates'].values()) else 'local_mixture_eligible_only_for_separate_realized_function_cost_precision_checks'
        result['scope']='Complete actual source FFN targets, fit-only shared/query/keys/bases. True-output projection coefficients and per-state joint span oracle are undeployable; no actual compact nonlinear child/shared function, quantized/native quality, useful extra n or accepted rate. Sparse anchor-conditioned domains consumed; no whole causal model or general conditional-architecture bound.'
        result['runtime']=budget(start);C.write(args.out,result)
        print(json.dumps({'decision':result['decision'],'gates':result['gates'],'joint_oracle_gates':result['diagnostic_joint_oracle_gates'],
            'runtime':result['runtime'],'result_sha256':C.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage,'error':repr(error),'seconds':time.monotonic()-start})
        C.write(args.out.with_suffix('.failure.json'),result);raise

if __name__=='__main__':main()
