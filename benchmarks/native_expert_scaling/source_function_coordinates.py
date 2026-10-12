"""FIT-only source read/write residual bases and fixed-source observables.

Quadratic surrogate optimum, not a compact causal chatbot or a KL optimum.
"""
import argparse, json, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def raw(rec,boundary,np):
    item=next(x['h'] for x in rec['boundaries'] if x['boundary']==boundary)
    return (np.fromfile(item['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(rec['history'],2048).astype('f8')


def groups(get,cfg,boundary):
    base=f'model.layers.{boundary}.'
    gamma=get(base+'input_layernorm.weight')
    inp=get(base+'mamba.in_proj.weight')*gamma
    sizes=[3072,3072,256,256,48]; names=['gate','x','B','C','dt'];lo=0;out=[]
    for name,size,mul in zip(names,sizes,cfg['ssm_multipliers']):
        A=inp[lo:lo+size]*cfg['ssm_in_multiplier']*mul
        out.append((name,A));lo+=size
    assert lo==inp.shape[0]
    for name in ['q','k','v']:
        A=get(base+'self_attn.'+name+'_proj.weight')*gamma*cfg['attention_in_multiplier']
        if name=='k':A*=cfg['key_multiplier']
        out.append((name,A))
    return out


def prepare(a):
    import numpy as np,psutil
    report=DOC/'source_residual_coordinate_feasibility_result_20261012.json'
    audit=DOC/'source_residual_coordinate_stored_adjudication_20261012.json'
    r=json.loads(report.read_bytes());ad=json.loads(audit.read_bytes())
    assert ad['report']==extent(report) and ad['all7_covariances_full_eigensystems_and336_case_metrics_checked']
    parent=json.loads(Path(r['binding']['path']).read_bytes())
    paired=DOC/'paired_output_codec_result_20261010.json'
    paudit=DOC/'paired_output_codec_stored_adjudication_20261010.json'
    pr=json.loads(paired.read_bytes());pad=json.loads(paudit.read_bytes())
    assert pad['result']==extent(paired) and all(pad['numeric_flags'].values())
    qbar=pr['artifacts']['qbar'];assert extent(qbar['path'])==qbar
    probe=DOC/'residual_basis_readout_probe_result_20261012.json'
    probeaudit=DOC/'residual_basis_readout_stored_adjudication_20261012.json'
    prob=json.loads(probe.read_bytes());pbad=json.loads(probeaudit.read_bytes())
    assert pbad['report']==extent(probe) and pbad['all72_states5_full_vocab_arrays2_projected_heads_checked']
    pb=json.loads(Path(prob['binding']['path']).read_bytes())
    source=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',report,audit,Path(r['binding']['path']),
        paired,paudit,Path(qbar['path']),probe,probeaudit,Path(prob['binding']['path']),
        DOC/'SOURCE_FUNCTION_COORDINATES_PROTOCOL_20261012.md',source/'model.safetensors',source/'config.json',
        SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',Path(sys.executable),
        Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),
        Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]
    paths+=list((SITE/'numpy.libs').glob('*.dll'))+list((SITE/'psutil').glob('*.pyd'))
    paths += [Path(x['file']['path']) for x in r['bases']]
    paths += [Path(x['path']) for x in prob['outputs']]
    for rec in parent['records']:
        paths += [Path(x['h']['path']) for x in rec['boundaries']]
    b=dict(schema='SOURCE_FUNCTION_COORDINATES_BINDING_V1',python=str(Path(sys.executable).resolve()),
        source=str(source.resolve()),records=parent['records'],covariances=r['bases'],qbar=qbar,
        readout_reference=prob,readout_samples=pb['samples'],boundaries=parent['boundaries'],dimension=256,
        covariance_support_cut=1e-12,discarded_covariance_trace_gate=1e-8,epsilon=1e-5,
        numeric=dict(dual_identity_max=1e-8,eigen_relative=1e-10,FIT_surrogate_relative=1e-8),
        proxy=dict(DEV_metric_relative_RMS_mean=.1,DEV_metric_relative_RMS_worst=.2),
        readout_proxy=dict(mean_KL=.01,worst_KL=.1,disagreement_rate=.01),
        limits=dict(seconds=1200,OS_bytes=6<<30,output_bytes=3<<30),
        inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
        scope='FIT-only quadratic source reading objective, full raw source histories; actual linear input projections at '
            'boundaries0/4/8/12/16/20, final mixture-Fisher head metric at24. Cached FIT qbar and full-prefill '
            'raw states are distinct paths; qbar is a surrogate metric only. Eight groups get equal FIT energy weight. '
            'No nonlinear SSM/attention/FFN composition or causal encoder trained; no source/shadow/optimizer/native/GPU/T4 calls.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),inputs=len(b['inputs']),seconds=time.monotonic()-START)),flush=True)


def quadratic(x,M,np):
    value=float(np.sum((x@M)*x))
    assert value>=-1e-8*max(float(np.sum(x*x)),1.)
    return max(value,0.)


def probabilities(reference,prediction,labels,np):
    rows=[]
    for i,label in enumerate(labels):
        q=reference[i]-reference[i].max();q-=math.log(float(np.exp(q).sum()))
        p=prediction[i]-prediction[i].max();p-=math.log(float(np.exp(p).sum()))
        rows.append(dict(**label,KL=float(np.sum(np.exp(q)*(q-p))),
            argmax_disagreement=int(np.argmax(q)!=np.argmax(p))))
    return rows,dict(mean_KL=math.fsum(x['KL'] for x in rows)/len(rows),
        worst_KL=max(x['KL'] for x in rows),disagreement_rate=sum(x['argmax_disagreement'] for x in rows)/len(rows))


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='SOURCE_FUNCTION_COORDINATES_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert not a.directory.exists() and not a.out.exists() and not a.out.with_suffix('.terminal.json').exists()
    assert (np.__version__,psutil.__version__)==('2.4.6','7.2.2')
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='before_hash';stream=None;done=[]
    receipt=dict(schema='SOURCE_FUNCTION_COORDINATES_TERMINAL_V1',binding=extent(a.binding),freeze=a.freeze,
        command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'],'deadline'
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'OS cap'
        assert not proc.children(recursive=True),'unexpected child'
    def check():
        for item in b['inputs']:guard();assert extent(item['path'])==item,item['path']
    try:
        check();a.directory.mkdir();cfg=json.loads((Path(b['source'])/'config.json').read_bytes())
        get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
        rows=[];sample_rows=[];readout=None
        for boundary in b['boundaries']:
            guard();stage=f'boundary{boundary}'
            prior=next(x for x in b['covariances'] if x['boundary']==boundary)
            with np.load(prior['file']['path'],allow_pickle=False) as z:
                C=z['gram'].copy();ce=z['eigenvalues'].copy();cq=z['eigenvectors'].copy();angular=z['basis'].copy()
            support=ce>ce[0]*b['covariance_support_cut'];assert support.sum()>=256
            lost=float(ce[~support].sum()/ce.sum());assert abs(lost)<=b['discarded_covariance_trace_gate']
            root=(cq[:,support]*np.sqrt(ce[support]))@cq[:,support].T
            inverse=(cq[:,support]/np.sqrt(ce[support]))@cq[:,support].T
            M=np.zeros((2048,2048),'f8');descriptors=[]
            if boundary<24:
                gs=groups(get,cfg,boundary)
                for name,A in gs:
                    G=A.T@A;energy=float(np.sum(C*G));assert energy>0
                    M+=G/(8*energy);descriptors.append(dict(name=name,shape=list(A.shape),FIT_energy=energy))
                del G
            else:
                gs=[];qbar=np.fromfile(b['qbar']['path'],'<f8');assert qbar.shape==(65537,) and abs(qbar.sum()-1)<1e-12
                gamma=get('model.final_layernorm.weight');headmean=np.zeros(2048,'f8')
                for lo in range(0,65537,2048):
                    guard();hi=min(lo+2048,65537)
                    A=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
                    M+=A.T@(qbar[lo:hi,None]*A);headmean+=qbar[lo:hi]@A
                M-=np.outer(headmean,headmean);energy=float(np.sum(C*M));assert energy>0;M/=energy
                descriptors=[dict(name='mixture_Fisher_head',shape=[65537,2048],FIT_energy=energy)]
            M=(M+M.T)*.5;K=root@M@root;K=(K+K.T)*.5
            eig,U=np.linalg.eigh(K);eig=eig[::-1].copy();U=U[:,::-1].copy()
            U*=np.where(U[np.argmax(np.abs(U),axis=0),np.arange(2048)]<0,-1.,1.)
            assert eig[-1]>=-1e-10*max(float(eig[0]),1.)
            er=float(np.linalg.norm(K@U-U*eig)/max(float(np.linalg.norm(K)),1e-300))
            assert er<=b['numeric']['eigen_relative']
            V=root@U[:,:256];W=inverse@U[:,:256];G=V.T@V
            dual=float(np.max(np.abs(W.T@V-np.eye(256))));assert dual<=b['numeric']['dual_identity_max']
            f32dual=float(np.max(np.abs(W.astype('f4').astype('f8').T@V.astype('f4').astype('f8')-np.eye(256))))
            f=a.directory/f'boundary{boundary:02d}.readwrite.npz'
            np.savez(f,M=M,root=root,inverse=inverse,K=K,eigenvalues=eig,eigenvectors=U,V=V,W=W,norm_metric=G,
                V_f32=V.astype('f4'),W_f32=W.astype('f4'))
            sampleU=[];sampleMeta=[]
            for rec in b['records']:
                guard();H=raw(rec,boundary,np);radius=np.sqrt(np.mean(H*H,axis=1,keepdims=True)+b['epsilon']);u=H/radius
                z=H@W;hat=z@V.T;own=np.sqrt(np.sum((z@G)*z,axis=1,keepdims=True)/2048+b['epsilon'])
                decode=u@W@V.T;own_decode=hat/own;ap=u@angular@angular.T
                denom=quadratic(u,M,np)
                values={}
                for mode,pred in [('oracle_norm',decode),('own_norm',own_decode),('angular_oracle_norm',ap)]:
                    numerator=quadratic(u-pred,M,np)
                    values[mode]=dict(error_energy_per_token=numerator/len(u),source_energy_per_token=denom/len(u),
                        metric_relative_RMS=math.sqrt(numerator/max(denom,1e-300)))
                rows.append(dict(id=rec['id'],split=rec['split'],domain=rec['domain'],boundary=boundary,history=len(H),
                    values=values,true_to_own_radius_mean=float(np.mean(radius/own))))
                indices=[0,len(H)//2,len(H)-1];assert len(set(indices))==3
                sampleU.append(np.stack([u[indices],decode[indices],own_decode[indices]]))
                sampleMeta.extend(dict(id=rec['id'],split=rec['split'],position=i) for i in indices)
            sampleU=np.concatenate(sampleU,axis=1);assert sampleU.shape==(3,144,2048)
            if boundary<24:
                arrays={'normalized':sampleU}
                for name,A in gs:
                    obs=sampleU@A.T;arrays[name]=obs
                    for split in ['FIT','DEV']:
                        mask=np.array([r['split']==split for r in sampleMeta]);truth=obs[0,mask]
                        for i,mode in [(1,'oracle_norm'),(2,'own_norm')]:
                            err=float(np.linalg.norm(obs[i,mask]-truth)/max(float(np.linalg.norm(truth)),1e-300))
                            sample_rows.append(dict(boundary=boundary,group=name,split=split,mode=mode,samples=int(mask.sum()),relative_RMS=err))
                sf=a.directory/f'boundary{boundary:02d}.observables.npz';np.savez(sf,**arrays)
                sample_file=extent(sf)
            else:
                sample_file=None;selected=[];labels=[]
                for item in b['readout_samples']:
                    rec=next(r for r in b['records'] if r['id']==item['id']);H=raw(rec,24,np)
                    selected.append(H[item['positions']]);labels.extend(dict(id=item['id'],label_index=i,position=p) for i,p in zip(item['label_indices'],item['positions']))
                H=np.concatenate(selected);z=H@W;norm=np.sqrt(np.mean(H*H,axis=1,keepdims=True)+b['epsilon'])
                own=np.sqrt(np.sum((z@G)*z,axis=1,keepdims=True)/2048+b['epsilon'])
                head=np.empty((65537,256),'f8');pred=np.empty((2,72,65537),'f8')
                for lo in range(0,65537,2048):
                    guard();hi=min(lo+2048,65537);A=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
                    head[lo:hi]=A@V
                    pred[0,:,lo:hi]=(z/norm)@head[lo:hi].T;pred[1,:,lo:hi]=(z/own)@head[lo:hi].T
                reference_item=next(x for x in b['readout_reference']['outputs'] if Path(x['path']).name=='full_source.f64')
                reference=np.fromfile(reference_item['path'],'<f8').reshape(72,65537)
                summaries=[];probrows=[]
                for i,mode in [(0,'oracle_norm'),(1,'own_norm')]:
                    rr,ss=probabilities(reference,pred[i],labels,np);ss['mode']=mode
                    ss['proxy_gates']={k:ss[k]<=v for k,v in b['readout_proxy'].items()}
                    summaries.append(ss);probrows.extend(dict(**r,mode=mode) for r in rr)
                pf=a.directory/'readout72.npz';np.savez(pf,raw_states=H,encoded=z,head=head,scores=pred)
                readout=dict(file=extent(pf),labels=labels,rows=probrows,summary=summaries,reference=reference_item)
                del pred,head,reference
            fit=[x for x in rows if x['boundary']==boundary and x['split']=='FIT']
            observed=math.fsum(x['values']['oracle_norm']['error_energy_per_token'] for x in fit)/24
            tail=float(eig[256:].sum());delta=abs(observed-tail)/max(float(eig.sum()),1e-300)
            assert delta<=b['numeric']['FIT_surrogate_relative']
            done.append(dict(boundary=boundary,file=extent(f),samples_file=sample_file,sample_labels=sampleMeta,
                groups=descriptors,support_dimension=int(support.sum()),discarded_covariance_trace=lost,
                eigen_relative=er,dual_identity_max=dual,F32_dual_identity_max=f32dual,
                FIT_observed_error_energy=observed,FIT_spectral_tail=tail,FIT_surrogate_delta=delta,
                FIT_metric_retained_fraction=float(eig[:256].sum()/eig.sum())))
            print(json.dumps(dict(boundary=boundary,FIT_retention=done[-1]['FIT_metric_retained_fraction'],seconds=time.monotonic()-START)),flush=True)
            if boundary<24:del arrays,obs,truth
            del C,ce,cq,root,inverse,M,K,U,V,W,G,gs,sampleU
        stage='skip_transport';transports=[]
        for first,second in zip(done,done[1:]):
            with np.load(first['file']['path'],allow_pickle=False) as x,np.load(second['file']['path'],allow_pickle=False) as y:
                T=y['W'].T@x['V'];f=a.directory/f"skip{first['boundary']:02d}_{second['boundary']:02d}.f64";T.astype('<f8').tofile(f)
                singular=np.linalg.svd(T,compute_uv=False)
                transports.append(dict(source=first['boundary'],destination=second['boundary'],file=extent(f),max_singular=float(singular[0]),min_singular=float(singular[-1])))
        summaries=[]
        for boundary in b['boundaries']:
            dev=[r for r in rows if r['split']=='DEV' and r['boundary']==boundary]
            for mode in ['oracle_norm','own_norm','angular_oracle_norm']:
                vals=[r['values'][mode]['metric_relative_RMS'] for r in dev]
                ss=dict(boundary=boundary,mode=mode,DEV_metric_relative_RMS_mean=math.fsum(vals)/24,DEV_metric_relative_RMS_worst=max(vals))
                ss['proxy_gates']={k:ss[k]<=v for k,v in b['proxy'].items()};summaries.append(ss)
        stream.close();stream=None;stage='after_hash';check();guard()
        outputs=[extent(p) for p in sorted(a.directory.iterdir())];assert sum(x['bytes'] for x in outputs)<=b['limits']['output_bytes']
        result=dict(schema='SOURCE_FUNCTION_COORDINATES_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,bases=done,
            rows=rows,summary=summaries,observables=sample_rows,readout=readout,transport=transports,outputs=outputs,
            inputs_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope'],
            source_shadow_optimizer_native_GPU_T4_calls=0,
            decision='PROXY_PASS' if all(all(s['proxy_gates'].values()) for s in summaries if s['mode']=='own_norm') and all(all(s['proxy_gates'].values()) for s in readout['summary']) else 'COMPLETED_PROXY_FAILURE')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage,completed_bases=done);raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
