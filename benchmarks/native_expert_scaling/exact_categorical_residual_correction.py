"""Actual source-head KL: full RMS/factor Jacobian and one FIT-selected step."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    report=DOC/'source_function_coordinates_result_v2_20261012.json';audit=DOC/'source_function_coordinates_stored_adjudication_20261012.json'
    r=json.loads(report.read_bytes());ad=json.loads(audit.read_bytes());assert ad['report']==extent(report) and ad['decision']=='INDEPENDENT_STORED_PASS'
    parent=json.loads(Path(r['binding']['path']).read_bytes());basis=next(x['file'] for x in r['bases'] if x['boundary']==24)
    corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    cr={x['id']:x for x in json.loads(corpus.read_bytes())['records']};fit=[];dev=[]
    for rec in parent['records']:
        h=next(x['h'] for x in rec['boundaries'] if x['boundary']==24)
        if rec['split']=='FIT':
            cc=cr[rec['id']];ix=[0,len(cc['positions'])//2,len(cc['positions'])-1];assert len(set(ix))==3
            fit.append(dict(id=rec['id'],domain=rec['domain'],history=rec['history'],h=h,label_indices=ix,positions=[cc['positions'][i] for i in ix]))
        else:
            labels=[x for x in r['readout']['labels'] if x['id']==rec['id']];assert len(labels)==3
            dev.append(dict(id=rec['id'],domain=rec['domain'],history=rec['history'],h=h,label_indices=[x['label_index'] for x in labels],positions=[x['position'] for x in labels]))
    assert len(fit)==len(dev)==24
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',report,audit,Path(r['binding']['path']),corpus,Path(basis['path']),
        DOC/'EXACT_CATEGORICAL_RESIDUAL_CORRECTION_PROTOCOL_20261012.md',Path(parent['source'])/'model.safetensors',Path(parent['source'])/'config.json',
        Path(r['readout']['reference']['path']),Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),
        Path(np._core._multiarray_umath.__file__),Path(psutil.__file__)]+[Path(x['h']['path']) for x in fit+dev]
    paths+=list((SITE/'numpy.libs').glob('*.dll'))+list((SITE/'psutil').glob('*.pyd'))
    b=dict(schema='EXACT_CATEGORICAL_RESIDUAL_CORRECTION_BINDING_V1',python=str(Path(sys.executable).resolve()),source=parent['source'],basis=basis,
        FIT=fit,DEV=dev,DEV_reference=r['readout']['reference'],DEV_labels=r['readout']['labels'],prior_DEV=r['readout']['summary'],
        epsilon=1e-5,finite_epsilon=[1e-5,2e-6],finite_derivative_relative_absolute=2e-6,
        alpha=[.0001,.001,.01,.1],gradient_norm_floor=1e-12,FIT_relative_improvement_required=1e-4,
        probability_proxy=parent['readout_proxy'],limits=dict(seconds=900,OS_bytes=8<<30,output_bytes=2<<30),
        inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
        scope='Exact full-V F64 categorical loss on actual raw source FIT72 states; source normalized trained head fixed. '
            'One normalized-group gradient, fixed4-step ladder, FIT-only selection sealed before semantic DEV evaluation. '
            'Generic rank<=256 factor correction is not an idempotent projector; not causal deployed state acquisition, '
            'internal source operator preservation, chatbot, ternary/native/speed/useful-n or family admission. No model/GPU/T4 call.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def selected(samples,np):
    H=[];labels=[]
    for rec in samples:
        raw=(np.fromfile(rec['h']['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(rec['history'],2048).astype('f8')
        H.append(raw[rec['positions']]);labels.extend(dict(id=rec['id'],label_index=i,position=p) for i,p in zip(rec['label_indices'],rec['positions']))
    return np.concatenate(H),labels


def logprob(score,np):
    x=score-score.max(axis=1,keepdims=True);return x-np.log(np.exp(x).sum(axis=1,keepdims=True))


def quality(score,reference,np):
    q=logprob(reference,np);p=logprob(score,np);loss=np.sum(np.exp(q)*(q-p),axis=1)
    return loss,dict(mean_KL=math.fsum(loss.tolist())/len(loss),worst_KL=float(loss.max()),disagreement_rate=float(np.mean(np.argmax(score,axis=1)!=np.argmax(reference,axis=1))))


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='EXACT_CATEGORICAL_RESIDUAL_CORRECTION_BINDING_V1' and Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert not a.directory.exists() and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='before_hash'
    receipt=dict(schema='EXACT_CATEGORICAL_RESIDUAL_CORRECTION_TERMINAL_V1',binding=extent(a.binding),freeze=a.freeze,command=sys.orig_argv,
        pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for item in b['inputs']:guard();assert extent(item['path'])==item,item['path']
    def save(name,**arrays):
        guard();path=a.directory/(name+'.npz');np.savez(path,**arrays);return extent(path)
    try:
        check();a.directory.mkdir();cfg=json.loads((Path(b['source'])/'config.json').read_bytes());get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np)
        gamma=get('model.final_layernorm.weight');A=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close();stream=None
        H,fitlabels=selected(b['FIT'],np);assert H.shape==(72,2048)
        t0=time.monotonic();reference=(H/np.sqrt(np.mean(H*H,axis=1,keepdims=True)+b['epsilon']))@A.T;first_matrix_seconds=time.monotonic()-t0
        reference_file=save('FIT_reference',H=H,scores=reference)
        with np.load(b['basis']['path'],allow_pickle=False) as pack:theta={'V':pack['V'].copy(),'W':pack['W'].copy(),'mu':np.zeros(2048,'f8')}
        initial_file=save('initial_parameters',**theta)
        def evaluate(params,name):
            guard();X=H-params['mu'];Z=X@params['W'];Y=params['mu']+Z@params['V'].T
            R=np.sqrt(np.mean(Y*Y,axis=1)+b['epsilon']);score=(Y/R[:,None])@A.T;loss,summary=quality(score,reference,np)
            file=save(name,z=Z,decoded=Y,radius=R,scores=score,KL=loss)
            return dict(file=file,summary=summary),Z,Y,R,score
        stage='baseline_gradient';baseline,Z,Y,R,score=evaluate(theta,'FIT_baseline')
        p=np.exp(logprob(score,np));q=np.exp(logprob(reference,np));g=((p-q)/72)@A
        gy=g/R[:,None]-Y*np.sum(Y*g,axis=1,keepdims=True)/(2048*R[:,None]**3)
        gradients=dict(V=gy.T@Z,W=(H-theta['mu']).T@(gy@theta['V']),mu=np.sum(gy-(gy@theta['V'])@theta['W'].T,axis=0))
        gradient_file=save('gradient',**gradients,logit_gradient=(p-q)/72,normalized_gradient=g,raw_gradient=gy)
        scales={key:float(np.linalg.norm(theta[key])) for key in ['V','W']};scales['mu']=math.sqrt(float(np.mean(np.sum(H*H,axis=1))))
        norms={key:float(np.linalg.norm(value)) for key,value in gradients.items()}
        direction={key:-gradients[key]*scales[key]/norms[key] if norms[key]>b['gradient_norm_floor'] else np.zeros_like(gradients[key]) for key in theta}
        direction_file=save('trust_direction',**direction)
        frames=[]
        for key in ['V','W','mu']:
            for phase in [0.,.7]:
                d=np.sin(np.arange(theta[key].size,dtype='f8')*.017+phase).reshape(theta[key].shape);d*=scales[key]/np.linalg.norm(d)
                frames.append((key+'_'+str(phase),{k:d if k==key else np.zeros_like(theta[k]) for k in theta}))
        frames.append(('joint_trust',direction));finite=[]
        for name,frame in frames:
            guard();stage='derivative_'+name;analytic=math.fsum(float(np.sum(gradients[k]*frame[k])) for k in theta)
            df=save('direction_'+name,**frame)
            for eps in b['finite_epsilon']:
                points=[]
                for sign in [-1,1]:
                    pp={k:theta[k]+sign*eps*frame[k] for k in theta};point,*_=evaluate(pp,'finite_'+name+'_'+str(eps)+'_'+str(sign));points.append(point)
                derivative=(points[1]['summary']['mean_KL']-points[0]['summary']['mean_KL'])/(2*eps);delta=abs(derivative-analytic)/max(1.,abs(analytic))
                finite.append(dict(name=name,direction=df,epsilon=eps,minus=points[0],plus=points[1],analytic=analytic,finite=derivative,relative_absolute_delta=delta,gate=delta<=b['finite_derivative_relative_absolute']))
            print(json.dumps(dict(derivative=name,all_scales_pass=all(x['gate'] for x in finite if x['name']==name),seconds=time.monotonic()-START)),flush=True)
        stage='FIT_ladder';candidates=[]
        for alpha in b['alpha']:
            pp={k:theta[k]+alpha*direction[k] for k in theta};parameters=save('candidate_'+str(alpha)+'_parameters',**pp)
            point,*_=evaluate(pp,'candidate_'+str(alpha)+'_FIT');candidates.append(dict(alpha=alpha,parameters=parameters,**point))
        best=min(candidates,key=lambda x:x['summary']['mean_KL']);improvement=(baseline['summary']['mean_KL']-best['summary']['mean_KL'])/max(baseline['summary']['mean_KL'],1e-300)
        accepted=all(x['gate'] for x in finite) and improvement>=b['FIT_relative_improvement_required']
        chosen=best['parameters'] if accepted else initial_file
        selection_path=a.directory/'selection.json';write(selection_path,dict(accepted=accepted,selected_parameters=chosen,
            selected_alpha=best['alpha'] if accepted else 0.,FIT_relative_improvement=improvement,derivative_gates=all(x['gate'] for x in finite),
            rule='Only FIT determines selection; all parameter/point extents fixed before DEV semantic evaluation.'))
        selection=extent(selection_path);print(json.dumps(dict(selection=selection,accepted=accepted,FIT_relative_improvement=improvement,seconds=time.monotonic()-START)),flush=True)
        stage='DEV_after_selection'
        with np.load(chosen['path'],allow_pickle=False) as pack:pp={k:pack[k] for k in theta}
        devH,devlabels=selected(b['DEV'],np);assert devlabels==b['DEV_labels']
        devref=np.fromfile(b['DEV_reference']['path'],'<f8').reshape(72,65537)
        devZ=(devH-pp['mu'])@pp['W'];devY=pp['mu']+devZ@pp['V'].T;devR=np.sqrt(np.mean(devY*devY,axis=1)+b['epsilon']);devScore=(devY/devR[:,None])@A.T
        devLoss,devSummary=quality(devScore,devref,np);devSummary['proxy_gates']={k:devSummary[k]<=v for k,v in b['probability_proxy'].items()}
        devFile=save('DEV_selected',H=devH,z=devZ,decoded=devY,radius=devR,scores=devScore,KL=devLoss)
        assert extent(selection_path)==selection;stage='after_hash';check();guard();outputs=[extent(p) for p in sorted(a.directory.iterdir())]
        result=dict(schema='EXACT_CATEGORICAL_RESIDUAL_CORRECTION_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
            FIT_labels=fitlabels,FIT_reference=reference_file,initial_parameters=initial_file,baseline=baseline,gradient=gradient_file,
            gradient_norms=norms,trust_scales=scales,trust_direction=direction_file,finite=finite,candidates=candidates,selection=selection,
            selected_parameters=chosen,accepted=accepted,FIT_relative_improvement=improvement,DEV=dict(file=devFile,labels=devlabels,summary=devSummary),
            first72_full_source_head_matrix_seconds=first_matrix_seconds,outputs=outputs,inputs_before_after_exact=True,
            seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope'],
            decision='VERIFIED_GRADIENT_FIT_DESCENT' if accepted else 'COMPLETED_NO_VERIFIED_FIT_DESCENT')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage);raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true');p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
