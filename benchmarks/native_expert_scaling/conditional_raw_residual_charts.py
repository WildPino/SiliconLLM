"""FIT raw affine charts: oracle geometric screen, not cheap causal routing."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from source_function_coordinates_v2 import raw,groups,probabilities
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    report=DOC/'source_function_coordinates_result_v2_20261012.json'
    audit=DOC/'source_function_coordinates_stored_adjudication_20261012.json'
    r=json.loads(report.read_bytes());ad=json.loads(audit.read_bytes())
    assert ad['report']==extent(report) and ad['decision']=='INDEPENDENT_STORED_PASS'
    parent=json.loads(Path(r['binding']['path']).read_bytes())
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'source_function_coordinates_v2.py',report,audit,
        Path(r['binding']['path']),DOC/'CONDITIONAL_RAW_RESIDUAL_CHARTS_PROTOCOL_20261012.md',
        Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),
        Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__),
        Path(parent['source'])/'model.safetensors',Path(parent['source'])/'config.json',
        Path(r['readout']['reference']['path'])]
    paths+=list((SITE/'numpy.libs').glob('*.dll'))+list((SITE/'psutil').glob('*.pyd'))
    selected=[x for x in r['bases'] if x['boundary'] in [0,12,24]]
    paths+=[Path(x['file']['path']) for x in selected]
    for rec in parent['records']:
        paths+=[Path(x['h']['path']) for x in rec['boundaries'] if x['boundary'] in [0,12,24]]
    b=dict(schema='CONDITIONAL_RAW_RESIDUAL_CHARTS_BINDING_V1',python=str(Path(sys.executable).resolve()),
        source=parent['source'],records=parent['records'],global_control=r,metrics=selected,
        boundaries=[0,12,24],K_values=[4,16],D=256,epsilon=1e-5,covariance_cut_relative=1e-12,
        lloyd_iterations=32,numeric=dict(dual_max=1e-8,eigen_relative=1e-10,FIT_tail_relative=1e-8),
        proxy=parent['proxy'],readout_proxy=parent['readout_proxy'],
        limits=dict(seconds=1800,OS_bytes=8<<30,output_bytes=12<<30),
        inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
        scope='FIT-only raw source-RMS weighted affine charts, fixed4/16 slots and<=256 active coordinates. '
            'Oracle source-space nearest-center selection and true source encoder, not compact causal routing or useful experts. '
            'Fixed source reading metrics; nonlinear/composed/task quality and same-artifact50 remain unqualified. '
            'No source/shadow/optimizer/native/GPU/T4 calls or DEV calibration.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),inputs=len(b['inputs']),seconds=time.monotonic()-START)),flush=True)


def distances(H,HM,hh,centers,M,np):
    d=hh[:,None]-2*(HM@centers.T)+np.sum((centers@M)*centers,axis=1)[None,:]
    assert d.min()>=-1e-8*max(float(np.max(hh)),1.)
    return np.maximum(d,0.)


def fit_centers(H,w,M,K,iterations,np,guard):
    HM=H@M;hh=np.sum(HM*H,axis=1);mu=np.sum(H*w[:,None],axis=0)/w.sum()
    d=distances(H,HM,hh,mu[None,:],M,np)[:,0];seeds=[];centers=[]
    for j in range(K):
        guard();candidate=w*d;candidate[seeds]=-1.;ix=int(np.argmax(candidate));assert candidate[ix]>0
        seeds.append(ix);centers.append(H[ix].copy())
        newdistance=distances(H,HM,hh,H[ix:ix+1],M,np)[:,0]
        d=newdistance if j==0 else np.minimum(d,newdistance)
    centers=np.stack(centers);history=[centers.copy()];assignments=[];costs=[]
    # Fixed32 rounds; empty slot retains its previous center and later has rank0.
    for iteration in range(iterations):
        guard();d=distances(H,HM,hh,centers,M,np);labels=np.argmin(d,axis=1).astype('i2')
        assignments.append(labels);costs.append(float(w@d[np.arange(len(H)),labels]))
        updated=centers.copy()
        for j in range(K):
            mask=labels==j
            if mask.any():updated[j]=np.sum(H[mask]*w[mask,None],axis=0)/w[mask].sum()
        centers=updated;history.append(centers.copy())
    d=distances(H,HM,hh,centers,M,np);labels=np.argmin(d,axis=1).astype('i2')
    assignments.append(labels);costs.append(float(w@d[np.arange(len(H)),labels]))
    assert all(y<=x+1e-8*max(1.,x) for x,y in zip(costs,costs[1:]))
    return centers,labels,np.stack(history),np.stack(assignments),np.array(costs),seeds


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='CONDITIONAL_RAW_RESIDUAL_CHARTS_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert not a.directory.exists() and not a.out.exists() and not a.out.with_suffix('.terminal.json').exists()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='before_hash';families=[]
    receipt=dict(schema='CONDITIONAL_RAW_RESIDUAL_CHARTS_TERMINAL_V1',binding=extent(a.binding),freeze=a.freeze,
        command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'],'deadline'
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'OS cap'
        assert not proc.children(recursive=True),'unexpected child'
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes'],'output cap'
    def check():
        for item in b['inputs']:guard();assert extent(item['path'])==item,item['path']
    try:
        check();a.directory.mkdir();cfg=json.loads((Path(b['source'])/'config.json').read_bytes())
        get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np);fit=[r for r in b['records'] if r['split']=='FIT']
        assert len(fit)==24;fit_packets=[]
        for bd in b['boundaries']:
            guard();stage=f'FIT{bd}'
            metric=next(x for x in b['metrics'] if x['boundary']==bd)
            with np.load(metric['file']['path'],allow_pickle=False) as pack:M=pack['M']
            hs=[];weights=[];fitlabels=[]
            for rec in fit:
                H=raw(rec,bd,np);radius2=np.mean(H*H,axis=1)+b['epsilon'];hs.append(H)
                weights.append(1/(24*len(H)*radius2));fitlabels.extend(dict(id=rec['id'],position=i) for i in range(len(H)))
            H=np.concatenate(hs);w=np.concatenate(weights);assert H.shape==(11148,2048)
            fp=a.directory/f'boundary{bd:02d}.FIT.npz';np.savez(fp,H=H,weights=w);fit_packets.append(dict(boundary=bd,file=extent(fp),labels=fitlabels))
            for Kslots in b['K_values']:
                guard();stage=f'FIT{bd}_K{Kslots}'
                centers,assignment,chistory,ahistory,costs,seeds=fit_centers(H,w,M,Kslots,b['lloyd_iterations'],np,guard)
                cf=a.directory/f'boundary{bd:02d}.K{Kslots:02d}.clustering.npz'
                np.savez(cf,centers=centers,assignment=assignment,center_history=chistory,assignment_history=ahistory,costs=costs,seeds=np.array(seeds,dtype='i4'))
                charts=[]
                for j in range(Kslots):
                    guard();mask=assignment==j;mass=float(w[mask].sum());mu=centers[j].copy()
                    # Last assignment may differ from last update; covariance remains around the frozen center.
                    X=H[mask]-mu;C=(X.T@(w[mask,None]*X))/mass if mass else np.zeros((2048,2048),'f8')
                    C=(C+C.T)*.5;ce,Q=np.linalg.eigh(C);ce=ce[::-1].copy();Q=Q[:,::-1].copy()
                    Q*=np.where(Q[np.argmax(np.abs(Q),axis=0),np.arange(2048)]<0,-1.,1.)
                    support=ce>max(float(ce[0]),0)*b['covariance_cut_relative'];rank=int(support.sum())
                    assert ce[-1]>=-1e-10*max(float(ce[0]),1.)
                    S=Q[:,support]*np.sqrt(ce[support]);J=Q[:,support]/np.sqrt(ce[support])
                    kernel=S.T@M@S;kernel=(kernel+kernel.T)*.5
                    eig,U=np.linalg.eigh(kernel);eig=eig[::-1].copy();U=U[:,::-1].copy()
                    if rank:assert eig[-1]>=-1e-10*max(float(eig[0]),1.)
                    if rank:U*=np.where(U[np.argmax(np.abs(U),axis=0),np.arange(rank)]<0,-1.,1.)
                    active=min(rank,256);V=np.zeros((2048,256),'f8');W=V.copy()
                    V[:,:active]=S@U[:,:active];W[:,:active]=J@U[:,:active];G=V.T@V;cross=V.T@mu;center2=float(mu@mu)
                    dual=float(np.max(np.abs(W.T@V-np.diag(np.arange(256)<active))));assert dual<=b['numeric']['dual_max']
                    cr=float(np.linalg.norm(C@Q-Q*ce)/max(float(np.linalg.norm(C)),1e-300))
                    kr=float(np.linalg.norm(kernel@U-U*eig)/max(float(np.linalg.norm(kernel)),1e-300))
                    assert max(cr,kr)<=b['numeric']['eigen_relative']
                    # Fit-only weighted explicit reading loss around frozen center vs spectral tail.
                    e=X-X@W@V.T;observed=float(np.sum(np.sum((e@M)*e,axis=1)*w[mask]))
                    tail=mass*float(eig[active:].sum());delta=abs(observed-tail)/max(mass*float(np.trace(C@M)),1e-300)
                    assert delta<=b['numeric']['FIT_tail_relative']
                    chartpath=a.directory/f'boundary{bd:02d}.K{Kslots:02d}.chart{j:02d}.npz'
                    np.savez(chartpath,center=mu,C=C,cov_eigenvalues=ce,cov_eigenvectors=Q,kernel=kernel,
                        eigenvalues=eig,eigenvectors=U,V=V,W=W,norm_metric=G,norm_cross=cross,norm_center_squared=np.array(center2))
                    charts.append(dict(chart=j,file=extent(chartpath),FIT_positions=int(mask.sum()),FIT_mass=mass,
                        support_dimension=rank,active_dimension=active,dual_max=dual,cov_eigen_relative=cr,kernel_eigen_relative=kr,
                        FIT_error_energy=observed,FIT_spectral_error=tail,FIT_tail_relative=delta))
                    print(json.dumps(dict(boundary=bd,K=Kslots,chart=j,FIT_positions=int(mask.sum()),active=active,seconds=time.monotonic()-START)),flush=True)
                    del C,Q,S,J,kernel,U,e,V,W,G,X
                families.append(dict(boundary=bd,K=Kslots,clustering=extent(cf),charts=charts,metric=metric['file'],FIT_packet=extent(fp)))
            del H,w,hs,weights
        stage='evaluate';rows=[];readouts=[];sample_groups=[]
        for family in families:
            bd,Kslots=family['boundary'],family['K'];guard()
            with np.load(family['clustering']['path'],allow_pickle=False) as cc:centers=cc['centers']
            with np.load(family['metric']['path'],allow_pickle=False) as pack:M=pack['M']
            chartdata=[]
            for chart in family['charts']:
                with np.load(chart['file']['path'],allow_pickle=False) as c:
                    chartdata.append({k:c[k].copy() for k in ['center','V','W','norm_metric','norm_cross','norm_center_squared']})
            sm=[];samplelabels=[];devh=[];devhat=[];devnorm=[];devown=[];labelmeta=[]
            for rec in b['records']:
                guard();H=raw(rec,bd,np);HM=H@M;hh=np.sum(HM*H,axis=1);d=distances(H,HM,hh,centers,M,np)
                labels=np.argmin(d,axis=1).astype('i2');Z=np.empty((len(H),256),'f8');hat=np.empty_like(H);own=np.empty(len(H),'f8')
                for j,c in enumerate(chartdata):
                    mask=labels==j
                    z=(H[mask]-c['center'])@c['W'];Z[mask]=z;hat[mask]=c['center']+z@c['V'].T
                    norm2=(c['norm_center_squared']+2*(z@c['norm_cross'])+np.sum((z@c['norm_metric'])*z,axis=1))/2048+b['epsilon']
                    assert np.all(norm2>0);own[mask]=np.sqrt(norm2)
                norm=np.sqrt(np.mean(H*H,axis=1)+b['epsilon']);u=H/norm[:,None];oracle=hat/norm[:,None];decoded=hat/own[:,None]
                truth=float(np.sum((u@M)*u));values={}
                for mode,pred in [('oracle_norm',oracle),('own_norm',decoded)]:
                    e=u-pred;num=float(np.sum((e@M)*e));assert num>=-1e-8*max(1.,truth);num=max(num,0.)
                    values[mode]=dict(error_energy_per_token=num/len(H),source_energy_per_token=truth/len(H),relative_RMS=math.sqrt(num/max(truth,1e-300)))
                packet=a.directory/f'boundary{bd:02d}.K{Kslots:02d}.{rec["id"]}.npz';np.savez(packet,chart_ids=labels,z=Z,true_radius=norm,own_radius=own)
                rows.append(dict(boundary=bd,K=Kslots,id=rec['id'],split=rec['split'],domain=rec['domain'],history=len(H),values=values,
                    packet=extent(packet),occupancy=[int(np.count_nonzero(labels==j)) for j in range(Kslots)]))
                ix=[0,len(H)//2,len(H)-1];sm.append(np.stack([u[ix],oracle[ix],decoded[ix]]))
                samplelabels.extend(dict(id=rec['id'],split=rec['split'],position=i) for i in ix)
                if bd==24 and rec['split']=='DEV':
                    item=next(x for x in b['global_control']['readout']['labels'] if x['id']==rec['id'])
                    labelsrec=[x for x in b['global_control']['readout']['labels'] if x['id']==rec['id']]
                    positions=[x['position'] for x in labelsrec]
                    devh.append(H[positions]);devhat.append(hat[positions]);devnorm.append(norm[positions]);devown.append(own[positions]);labelmeta.extend(labelsrec)
            sample=np.concatenate(sm,axis=1);assert sample.shape==(3,144,2048)
            if bd<24:
                obs={'normalized':sample}
                for name,A in groups(get,cfg,bd):
                    value=sample@A.T;obs[name]=value
                    for split in ['FIT','DEV']:
                        mask=np.array([x['split']==split for x in samplelabels]);truth=value[0,mask]
                        for i,mode in [(1,'oracle_norm'),(2,'own_norm')]:
                            sample_groups.append(dict(boundary=bd,K=Kslots,group=name,split=split,mode=mode,relative_RMS=float(np.linalg.norm(value[i,mask]-truth)/np.linalg.norm(truth))))
                sf=a.directory/f'boundary{bd:02d}.K{Kslots:02d}.observables.npz';np.savez(sf,**obs);family['observables_file']=extent(sf);family['sample_labels']=samplelabels
                del obs
            else:
                H=np.concatenate(devh);hat=np.concatenate(devhat);norm=np.concatenate(devnorm);own=np.concatenate(devown)
                scores=np.empty((2,72,65537),'f8');gamma=get('model.final_layernorm.weight')
                for lo in range(0,65537,2048):
                    guard();hi=min(65537,lo+2048);A=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
                    scores[0,:,lo:hi]=(hat/norm[:,None])@A.T;scores[1,:,lo:hi]=(hat/own[:,None])@A.T
                refitem=b['global_control']['readout']['reference'];reference=np.fromfile(refitem['path'],'<f8').reshape(72,65537)
                assert labelmeta==b['global_control']['readout']['labels'];rs=[];ss=[]
                for i,mode in [(0,'oracle_norm'),(1,'own_norm')]:
                    rr,summary=probabilities(reference,scores[i],labelmeta,np);summary['mode']=mode
                    summary['proxy_gates']={k:summary[k]<=v for k,v in b['readout_proxy'].items()};ss.append(summary);rs+= [dict(**x,mode=mode) for x in rr]
                rf=a.directory/f'boundary24.K{Kslots:02d}.readout.npz';np.savez(rf,raw=H,decoded=hat,true_radius=norm,own_radius=own,scores=scores)
                readouts.append(dict(K=Kslots,file=extent(rf),reference=refitem,labels=labelmeta,rows=rs,summary=ss))
                del scores,reference
            print(json.dumps(dict(evaluated_boundary=bd,K=Kslots,seconds=time.monotonic()-START)),flush=True)
        stage='affine_skips';skips=[]
        for Kslots in b['K_values']:
            for src,dst in [(0,12),(12,24)]:
                first=next(x for x in families if x['boundary']==src and x['K']==Kslots);second=next(x for x in families if x['boundary']==dst and x['K']==Kslots)
                T=np.empty((Kslots,Kslots,256,256),'f8');offset=np.empty((Kslots,Kslots,256),'f8')
                for j,c in enumerate(second['charts']):
                    with np.load(c['file']['path'],allow_pickle=False) as z:W=z['W'];mu=z['center']
                    for i,c0 in enumerate(first['charts']):
                        guard()
                        with np.load(c0['file']['path'],allow_pickle=False) as z:V=z['V'];old=z['center']
                        T[j,i]=W.T@V;offset[j,i]=W.T@(old-mu)
                sp=a.directory/f'skip{src:02d}_{dst:02d}.K{Kslots:02d}.npz';np.savez(sp,matrix=T,offset=offset)
                skips.append(dict(K=Kslots,source=src,destination=dst,file=extent(sp)));del T,offset
        summaries=[]
        for family in families:
            bd,Kslots=family['boundary'],family['K']
            for mode in ['oracle_norm','own_norm']:
                dev=[x['values'][mode]['relative_RMS'] for x in rows if x['boundary']==bd and x['K']==Kslots and x['split']=='DEV']
                ss=dict(boundary=bd,K=Kslots,mode=mode,DEV_metric_relative_RMS_mean=math.fsum(dev)/24,DEV_metric_relative_RMS_worst=max(dev))
                ss['proxy_gates']={k:ss[k]<=v for k,v in b['proxy'].items()};summaries.append(ss)
        stream.close();stream=None;stage='after_hash';check();guard();outputs=[extent(p) for p in sorted(a.directory.iterdir())]
        result=dict(schema='CONDITIONAL_RAW_RESIDUAL_CHARTS_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,fit_packets=fit_packets,
            families=families,rows=rows,summary=summaries,observables=sample_groups,readouts=readouts,skips=skips,outputs=outputs,
            inputs_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope'],
            decision='ORACLE_LOCAL_PROXY_PASS' if any(all(all(x['proxy_gates'].values()) for x in summaries if x['K']==Kslots and x['mode']=='own_norm') and all(all(x['proxy_gates'].values()) for x in next(y for y in readouts if y['K']==Kslots)['summary']) for Kslots in b['K_values']) else 'COMPLETED_ORACLE_PROXY_FAILURE')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage,completed_families=families);raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
