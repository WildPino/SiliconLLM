"""Source-width, stored-chart full audit; independent fit/update arithmetic."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes())
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.report)
    assert r['schema']=='CONDITIONAL_RAW_RESIDUAL_CHARTS_RESULT_V1' and not a.out.exists()
    b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding']
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;delta_max=relative_max=0.;ambiguous=0;checks=[]
    audit_inputs=[extent(a.report),extent(a.terminal),extent(__file__),extent(DOC/'CONDITIONAL_RAW_RESIDUAL_CHARTS_STORED_AUDIT_PROTOCOL_20261012.md')]
    receipt=dict(schema='CONDITIONAL_RAW_RESIDUAL_CHARTS_AUDIT_TERMINAL_V1',command=sys.orig_argv,
        source=extent(__file__),freeze=a.freeze,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=1800 and proc.memory_info().peak_wset<=8<<30
        assert not proc.children(recursive=True)
    def check():
        for item in b['inputs']+r['outputs']+[r['binding']]+audit_inputs:
            guard();assert extent(item['path'])==item,item['path']
    def scalar(x,y,tol=1e-8):
        nonlocal delta_max
        delta=abs(float(x)-float(y));delta_max=max(delta_max,delta)
        assert delta<=tol*max(1.,abs(float(y))),(x,y,delta)
    def compare(x,y,tol=1e-10):
        nonlocal relative_max
        rel=float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-300));relative_max=max(relative_max,rel)
        assert rel<=tol,rel
    def raw(rec,bd):
        item=next(x['h'] for x in rec['boundaries'] if x['boundary']==bd)
        return (np.fromfile(item['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(rec['history'],2048).astype('f8')
    def factored_distance(Xc,centers,L):
        mc=centers@L
        return np.maximum(np.sum(Xc*Xc,axis=1)[:,None]+np.sum(mc*mc,axis=1)[None,:]-2*(Xc@mc.T),0.)
    def routing(d,labels):
        nonlocal ambiguous
        least=d.min(axis=1);chosen=d[np.arange(len(d)),labels];tol=1e-10*np.maximum(1.,d.max(axis=1))
        assert np.all(chosen-least<=tol)
        discrep=np.argmin(d,axis=1)!=labels;ambiguous+=int(discrep.sum())
    def partial(x,A):
        y=np.zeros((*x.shape[:-1],A.shape[0]),'f8')
        for lo in range(0,2048,256):y+=x[...,lo:lo+256]@A[:,lo:lo+256].T
        return y
    try:
        check();get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
        cfg=json.loads((Path(b['source'])/'config.json').read_bytes());obs_count=0;case_count=0
        for bd in b['boundaries']:
            guard();metric=next(x for x in b['metrics'] if x['boundary']==bd)
            with np.load(metric['file']['path'],allow_pickle=False) as pack:M=pack['M']
            # Audit numerical factorization, not a new covariance/metric fit.
            L=np.linalg.cholesky(M);compare(L@L.T,M);fit=[x for x in b['records'] if x['split']=='FIT']
            H=np.concatenate([raw(x,bd) for x in fit]);w=np.concatenate([1/(24*x['history']*(np.mean(raw(x,bd)**2,axis=1)+b['epsilon'])) for x in fit])
            fp=next(x for x in r['fit_packets'] if x['boundary']==bd)
            with np.load(fp['file']['path'],allow_pickle=False) as pack:
                assert np.array_equal(pack['H'],H) and np.array_equal(pack['weights'],w)
            assert fp['labels']==[dict(id=x['id'],position=i) for x in fit for i in range(x['history'])]
            HC=H@L;HM=H@M
            for Kslots in b['K_values']:
                family=next(x for x in r['families'] if x['boundary']==bd and x['K']==Kslots)
                with np.load(family['clustering']['path'],allow_pickle=False) as pack:
                    mu=pack['centers'];assignment=pack['assignment'];history=pack['center_history'];ah=pack['assignment_history'];costs=pack['costs'];seeds=pack['seeds']
                assert history.shape==(33,Kslots,2048) and ah.shape==(33,11148)
                mean=np.sum(H*w[:,None],axis=0)/w.sum();distance=factored_distance(HC,mean[None,:],L)[:,0]
                for j,ix in enumerate(seeds):
                    candidate=w*distance;candidate[seeds[:j]]=-1.;best=float(candidate.max())
                    assert best-candidate[ix]<=1e-10*max(1.,abs(best));assert np.array_equal(history[0,j],H[ix])
                    new=factored_distance(HC,H[ix:ix+1],L)[:,0];distance=new if j==0 else np.minimum(distance,new)
                for iteration in range(33):
                    guard();distance=factored_distance(HC,history[iteration],L);labels=ah[iteration];routing(distance,labels)
                    scalar(costs[iteration],float(w@distance[np.arange(len(H)),labels]))
                    if iteration<32:
                        expected=history[iteration].copy()
                        for j in range(Kslots):
                            mask=labels==j
                            if mask.any():expected[j]=np.average(H[mask],axis=0,weights=w[mask])
                        compare(history[iteration+1],expected)
                assert np.array_equal(mu,history[-1]) and np.array_equal(assignment,ah[-1])
                assert np.all(np.diff(costs)<=1e-8*np.maximum(1.,costs[:-1]))
                chartdata=[];active=[]
                for j,item in enumerate(family['charts']):
                    guard();mask=assignment==j;mass=float(w[mask].sum());X=H[mask]-mu[j]
                    C=(X.T*w[mask])@X/mass if mass else np.zeros((2048,2048),'f8')
                    with np.load(item['file']['path'],allow_pickle=False) as pack:
                        ce=pack['cov_eigenvalues'];Q=pack['cov_eigenvectors'];kernel=pack['kernel'];eig=pack['eigenvalues'];U=pack['eigenvectors'];V=pack['V'];W=pack['W'];G=pack['norm_metric'];cross=pack['norm_cross'];a0=pack['norm_center_squared']
                        compare(pack['C'],C);assert np.array_equal(pack['center'],mu[j])
                    compare((Q*ce)@Q.T,C);compare(Q.T@Q,np.eye(2048))
                    keep=ce>max(float(ce[0]),0)*b['covariance_cut_relative'];rank=int(keep.sum());nd=min(rank,256)
                    assert rank==item['support_dimension'] and nd==item['active_dimension']
                    assert item['FIT_positions']==int(mask.sum());scalar(mass,item['FIT_mass'])
                    S=Q[:,keep]*np.sqrt(ce[keep]);J=Q[:,keep]/np.sqrt(ce[keep]);compare(kernel,S.T@M@S)
                    if rank:
                        compare(U.T@U,np.eye(rank));compare((U*eig)@U.T,kernel)
                        assert eig[-1]>=-1e-10*max(float(eig[0]),1.)
                    assert np.all(ce[:-1]>=ce[1:]) and np.all(eig[:-1]>=eig[1:])
                    compare(V[:,:nd],S@U[:,:nd]);compare(W[:,:nd],J@U[:,:nd])
                    assert np.count_nonzero(V[:,nd:])==0 and np.count_nonzero(W[:,nd:])==0
                    compare(W.T@V,np.diag(np.arange(256)<nd),1e-8);compare(G,V.T@V);compare(cross,V.T@mu[j]);scalar(a0,mu[j]@mu[j])
                    z=X@W;xm=HM[mask]-mu[j]@M
                    source=np.sum(xm*X,axis=1);ct=np.sum((X@(M@V))*z,axis=1);decoded=np.sum((z@(V.T@M@V))*z,axis=1)
                    energy=float(w[mask]@(source-2*ct+decoded));scalar(energy,item['FIT_error_energy']);scalar(mass*eig[nd:].sum(),item['FIT_spectral_error'])
                    chartdata.append(dict(mu=mu[j].copy(),V=V.copy(),W=W.copy()));active.append(nd)
                    if j%4==0:print(json.dumps(dict(audited_boundary=bd,K=Kslots,chart=j,seconds=time.monotonic()-START)),flush=True)
                    del C,Q,S,J,kernel,U,X
                selected=[];samplelabels=[]
                for rec in b['records']:
                    guard();case=next(x for x in r['rows'] if x['boundary']==bd and x['K']==Kslots and x['id']==rec['id'])
                    source=raw(rec,bd)
                    with np.load(case['packet']['path'],allow_pickle=False) as pack:
                        labels=pack['chart_ids'];Z=pack['z'];radius=pack['true_radius'];own=pack['own_radius']
                    routing(factored_distance(source@L,mu,L),labels);hat=np.empty_like(source)
                    assert case['occupancy']==[int(np.count_nonzero(labels==j)) for j in range(Kslots)]
                    for j,c in enumerate(chartdata):
                        mask=labels==j;z=(source[mask]-c['mu'])@c['W'];compare(Z[mask],z);hat[mask]=c['mu']+z@c['V'].T
                    compare(radius,np.sqrt(np.mean(source**2,axis=1)+b['epsilon']))
                    compare(own,np.sqrt(np.mean(hat**2,axis=1)+b['epsilon']),1e-9)
                    u=source/radius[:,None];preds=[hat/radius[:,None],hat/own[:,None]]
                    source_energy=float(np.sum((u@M)*u))
                    for mode,pred in zip(['oracle_norm','own_norm'],preds):
                        # Different quadratic expression from producer's direct error vectors.
                        energy=float(np.sum((u@M)*u)-2*np.sum((u@M)*pred)+np.sum((pred@M)*pred));energy=max(energy,0.)
                        val=case['values'][mode];scalar(energy/len(source),val['error_energy_per_token']);scalar(source_energy/len(source),val['source_energy_per_token']);scalar(math.sqrt(energy/max(source_energy,1e-300)),val['relative_RMS'])
                    ix=[0,len(source)//2,len(source)-1];selected.append(np.stack([u[ix],preds[0][ix],preds[1][ix]]));samplelabels.extend(dict(id=rec['id'],split=rec['split'],position=i) for i in ix);case_count+=1
                for summary in [x for x in r['summary'] if x['boundary']==bd and x['K']==Kslots]:
                    vals=[x['values'][summary['mode']]['relative_RMS'] for x in r['rows'] if x['boundary']==bd and x['K']==Kslots and x['split']=='DEV']
                    scalar(math.fsum(vals)/24,summary['DEV_metric_relative_RMS_mean']);scalar(max(vals),summary['DEV_metric_relative_RMS_worst']);assert summary['proxy_gates']=={k:summary[k]<=v for k,v in b['proxy'].items()}
                if bd<24:
                    sample=np.concatenate(selected,axis=1);assert samplelabels==family['sample_labels']
                    base=f'model.layers.{bd}.';gamma=get(base+'input_layernorm.weight');proj=get(base+'mamba.in_proj.weight');bounds=[0,3072,6144,6400,6656,6704];gs=[]
                    for j,name in enumerate(['gate','x','B','C','dt']):gs.append((name,proj[bounds[j]:bounds[j+1]]*gamma*cfg['ssm_in_multiplier']*cfg['ssm_multipliers'][j]))
                    for name in ['q','k','v']:
                        A=get(base+'self_attn.'+name+'_proj.weight')*gamma*cfg['attention_in_multiplier'];gs.append((name,A*cfg['key_multiplier'] if name=='k' else A))
                    with np.load(family['observables_file']['path'],allow_pickle=False) as obs:
                        compare(obs['normalized'],sample)
                        for name,A in gs:
                            calc=partial(sample,A);compare(obs[name],calc)
                            for split in ['FIT','DEV']:
                                mask=np.array([x['split']==split for x in samplelabels]);truth=calc[0,mask]
                                for i,mode in [(1,'oracle_norm'),(2,'own_norm')]:
                                    err=np.linalg.norm(calc[i,mask]-truth)/np.linalg.norm(truth);target=next(x for x in r['observables'] if x['boundary']==bd and x['K']==Kslots and x['group']==name and x['split']==split and x['mode']==mode)
                                    scalar(err,target['relative_RMS']);obs_count+=1
                checks.append(dict(boundary=bd,K=Kslots,charts=len(chartdata),active_dimensions=active,cases=48))
            del H,w,HM,HC
        probcount=0
        for rd in r['readouts']:
            guard();Kslots=rd['K'];family=next(x for x in r['families'] if x['boundary']==24 and x['K']==Kslots)
            with np.load(rd['file']['path'],allow_pickle=False) as pack:H=pack['raw'];hat=pack['decoded'];radius=pack['true_radius'];own=pack['own_radius'];scores=pack['scores']
            actual=[];decoded=[];tr=[];ow=[]
            for rec in [x for x in b['records'] if x['split']=='DEV']:
                labels=[x for x in rd['labels'] if x['id']==rec['id']];positions=[x['position'] for x in labels];actual.append(raw(rec,24)[positions])
                case=next(x for x in r['rows'] if x['boundary']==24 and x['K']==Kslots and x['id']==rec['id'])
                with np.load(case['packet']['path'],allow_pickle=False) as pack:ids=pack['chart_ids'][positions];zs=pack['z'][positions];tr.append(pack['true_radius'][positions]);ow.append(pack['own_radius'][positions])
                hh=np.empty((3,2048),'f8')
                for j in set(ids.tolist()):
                    with np.load(family['charts'][j]['file']['path'],allow_pickle=False) as pack:hh[ids==j]=pack['center']+zs[ids==j]@pack['V'].T
                decoded.append(hh)
            assert np.array_equal(H,np.concatenate(actual));compare(hat,np.concatenate(decoded));compare(radius,np.concatenate(tr));compare(own,np.concatenate(ow))
            ref=np.fromfile(rd['reference']['path'],'<f8').reshape(72,65537);inp=np.stack([H/radius[:,None],hat/radius[:,None],hat/own[:,None]]);gamma=get('model.final_layernorm.weight')
            for lo in range(0,65537,1024):
                guard();hi=min(65537,lo+1024);A=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier'];calc=partial(inp,A);compare(calc[0],ref[:,lo:hi]);compare(calc[1:],scores[:,:,lo:hi])
            for mi,mode in [(0,'oracle_norm'),(1,'own_norm')]:
                losses=[];wrong=[]
                for i,label in enumerate(rd['labels']):
                    q=ref[i]-np.logaddexp.reduce(ref[i]);p=scores[mi,i]-np.logaddexp.reduce(scores[mi,i]);loss=float(np.exp(q)@(q-p));wr=int(np.argmax(ref[i])!=np.argmax(scores[mi,i]));losses.append(loss);wrong.append(wr)
                    target=next(x for x in rd['rows'] if x['mode']==mode and all(x[k]==v for k,v in label.items()));scalar(loss,target['KL']);assert wr==target['argmax_disagreement'];probcount+=1
                ss=next(x for x in rd['summary'] if x['mode']==mode);scalar(math.fsum(losses)/72,ss['mean_KL']);scalar(max(losses),ss['worst_KL']);scalar(sum(wrong)/72,ss['disagreement_rate']);assert ss['proxy_gates']=={k:ss[k]<=v for k,v in b['readout_proxy'].items()}
            assert np.array_equal(np.argmax(scores[0],axis=1),np.argmax(scores[1],axis=1))
        pairs=0
        for skip in r['skips']:
            first=next(x for x in r['families'] if x['boundary']==skip['source'] and x['K']==skip['K']);second=next(x for x in r['families'] if x['boundary']==skip['destination'] and x['K']==skip['K'])
            with np.load(skip['file']['path'],allow_pickle=False) as pack:T=pack['matrix'];offset=pack['offset']
            for j,c in enumerate(second['charts']):
                with np.load(c['file']['path'],allow_pickle=False) as pack:W=pack['W'];mu=pack['center']
                for i,c0 in enumerate(first['charts']):
                    guard()
                    with np.load(c0['file']['path'],allow_pickle=False) as pack:V=pack['V'];old=pack['center']
                    compare(T[j,i],W.T@V);compare(offset[j,i],W.T@(old-mu));pairs+=1
        expected='ORACLE_LOCAL_PROXY_PASS' if any(all(all(x['proxy_gates'].values()) for x in r['summary'] if x['K']==Kslots and x['mode']=='own_norm') and all(all(x['proxy_gates'].values()) for x in next(y for y in r['readouts'] if y['K']==Kslots)['summary']) for Kslots in b['K_values']) else 'COMPLETED_ORACLE_PROXY_FAILURE'
        assert expected==r['decision'] and (case_count,obs_count,probcount,pairs)==(288,128,288,544)
        stream.close();stream=None;check();guard()
        result=dict(schema='CONDITIONAL_RAW_RESIDUAL_CHARTS_STORED_AUDIT_V1',source=extent(__file__),freeze=a.freeze,
            report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],checks=checks,
            all60_chart_covariances_full_spectra_centers_maps_and198_clustering_states_checked=True,
            all288_histories_affine_norms_coordinates_and128_observable_metrics_checked=True,
            all2_full_readout_packets_288_probability_rows_and544_affine_skip_pairs_checked=True,
            ambiguous_nearest_center_rounding_rows=ambiguous,max_numeric_relative=relative_max,max_scalar_delta=delta_max,
            before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',scientific_decision=expected,
            seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            scope='Independent source-space factor distances, all Lloyd updates, covariance reaccumulation and fixed-spectrum reconstruction; '
                'source-width projection/head sums, expanded quadratic errors and direct decoded norms. No chart/covariance eigensolver refit '
                '(metric Cholesky only), source/shadow/optimizer/native/GPU/T4 calls. Oracle geometry, no causal routing qualification.')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
