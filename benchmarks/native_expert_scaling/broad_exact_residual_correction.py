"""All existing source labels, exact categorical gradient and score-factor custody."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def prepare(a):
    import numpy as np,psutil
    report=DOC/'exact_categorical_residual_correction_result_20261012.json';audit=DOC/'exact_categorical_residual_correction_stored_adjudication_20261012.json'
    r=json.loads(report.read_bytes());ad=json.loads(audit.read_bytes());assert ad['report']==extent(report) and ad['decision']=='INDEPENDENT_STORED_PASS'
    parent=json.loads(Path(r['binding']['path']).read_bytes());corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    cr={x['id']:x for x in json.loads(corpus.read_bytes())['records']};cases={}
    for split in ['FIT','DEV']:
        cases[split]=[]
        for x in parent[split]:
            cc=cr[x['id']];assert cc['split']==split and len(cc['positions'])>0
            cases[split].append(dict(id=x['id'],split=split,domain=x['domain'],history=x['history'],h=x['h'],positions=cc['positions']))
    assert sum(len(x['positions']) for x in cases['FIT'])==4422 and sum(len(x['positions']) for x in cases['DEV'])==4386
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',report,audit,Path(r['binding']['path']),corpus,Path(r['initial_parameters']['path']),
        DOC/'BROAD_EXACT_RESIDUAL_CORRECTION_PROTOCOL_20261012.md',Path(parent['source'])/'model.safetensors',Path(parent['source'])/'config.json',
        Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(psutil.__file__)]+[Path(x['h']['path']) for x in cases['FIT']+cases['DEV']]
    paths+=list((SITE/'numpy.libs').glob('*.dll'))+list((SITE/'psutil').glob('*.pyd'))
    b=dict(schema='BROAD_EXACT_RESIDUAL_CORRECTION_BINDING_V1',python=str(Path(sys.executable).resolve()),source=parent['source'],cases=cases,
        initial_parameters=r['initial_parameters'],initializer='Original global source-function256 factors/raw zero center;not72-label corrected factors.',
        epsilon=1e-5,batch=64,finite_epsilon=[2e-6,2e-7],derivative_tolerance=2e-6,
        alpha=[.0001,.001,.01,.1],gradient_floor=1e-12,FIT_relative_improvement_gate=1e-4,
        factor_numeric=dict(score_sup_relative=1e-10,KL_absolute=1e-9,argmax_differences=0),
        quality=dict(case_mean_KL=.01,case_mean_disagreement=.01,worst_label_KL=.1,every_case_KL=.05,every_case_disagreement=.05),
        limits=dict(seconds=900,OS_bytes=6<<30,output_bytes=3<<30),inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
        scope='All4422 FIT/all4386 DEV stored raw source labels, equal case weighting. Actual source head F64 full-V probabilities. '
            'One broad gradient/joint directional checks/fixed ladder/FIT-only sealed selection. Exact-real score factors with '
            'floating operation-order control and full direct independent audit;not dense score storage. '
            'Teacher-injected source encoders;no source model/native/GPU/T4/RESERVED/longalign or causal/useful-n/speed admission.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def dataset(cases,np):
    rows=[];weights=[];meta=[];lo=0
    for c in cases:
        raw=(np.fromfile(c['h']['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(c['history'],2048).astype('f8');h=raw[c['positions']];rows.append(h)
        n=len(h);weights.extend([1/(24*n)]*n);meta.append(dict(**c,lo=lo,hi=lo+n));lo+=n
    return np.concatenate(rows),np.array(weights,'f8'),meta


def logs(s,np):
    s=s-s.max(axis=1,keepdims=True);return s-np.log(np.exp(s).sum(axis=1,keepdims=True))


def metrics(score,reference,np):
    q=logs(reference,np);p=logs(score,np)
    return np.column_stack([np.sum(np.exp(q)*(q-p),axis=1),np.argmax(score,axis=1)!=np.argmax(reference,axis=1),
        -np.sum(np.exp(q)*q,axis=1),-np.sum(np.exp(p)*p,axis=1)])


def summarize(values,meta,limits):
    import numpy as np
    cases=[]
    for c in meta:
        x=values[c['lo']:c['hi']];cases.append(dict(id=c['id'],domain=c['domain'],labels=len(x),KL=math.fsum(x[:,0].tolist())/len(x),disagreement_rate=float(x[:,1].mean()),worst_KL=float(x[:,0].max())))
    summary=dict(case_mean_KL=math.fsum(c['KL'] for c in cases)/24,case_mean_disagreement=math.fsum(c['disagreement_rate'] for c in cases)/24,
        label_mean_KL=math.fsum(values[:,0].tolist())/len(values),label_disagreement=float(values[:,1].mean()),
        worst_label_KL=float(values[:,0].max()),worst_case_KL=max(c['KL'] for c in cases))
    flags=dict(case_mean_KL=summary['case_mean_KL']<=limits['case_mean_KL'],case_mean_disagreement=summary['case_mean_disagreement']<=limits['case_mean_disagreement'],
        worst_label_KL=summary['worst_label_KL']<=limits['worst_label_KL'],every_case_KL=all(c['KL']<=limits['every_case_KL'] for c in cases),
        every_case_disagreement=all(c['disagreement_rate']<=limits['every_case_disagreement'] for c in cases))
    domains=[]
    for d in sorted(set(c['domain'] for c in cases)):
        cc=[c for c in cases if c['domain']==d];assert len(cc)==2
        domains.append(dict(domain=d,KL=math.fsum(c['KL'] for c in cc)/2,disagreement_rate=math.fsum(c['disagreement_rate'] for c in cc)/2))
    flags['domains']=all(c['KL']<=limits['every_case_KL'] and c['disagreement_rate']<=limits['every_case_disagreement'] for c in domains)
    return dict(summary=summary,cases=cases,domains=domains,quality_gates=flags)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert not a.directory.exists() and not a.out.exists();proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;stage='before_hash'
    receipt=dict(schema='BROAD_EXACT_RESIDUAL_CORRECTION_TERMINAL_V1',binding=extent(a.binding),freeze=a.freeze,command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(x.stat().st_size for x in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for x in b['inputs']:guard();assert extent(x['path'])==x,x['path']
    def load(item):
        with np.load(item['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def save(name,**fields):
        guard();p=a.directory/(name+'.npz');np.savez(p,**fields);return extent(p)
    try:
        check();a.directory.mkdir();get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np);cfg=json.loads((Path(b['source'])/'config.json').read_bytes());gamma=get('model.final_layernorm.weight');A=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close();stream=None;H,weights,meta=dataset(b['cases']['FIT'],np);fitfile=save('FIT_dataset',H=H,weights=weights);theta=load(b['initial_parameters'])
        def evaluate(pp,name,H=H,weights=weights,meta=meta,gradient=False,preflight=False):
            guard();params=save(name+'_parameters',**pp);z=(H-pp['mu'])@pp['W'];y=pp['mu']+z@pp['V'].T;radius=np.sqrt(np.mean(y*y,axis=1)+b['epsilon']);head=A@pp['V'];bias=A@pp['mu'];values=np.empty((len(H),4),'f8')
            g=np.empty_like(H) if gradient else None;gy=np.empty_like(H) if gradient else None;controls=[]
            for lo in range(0,len(H),b['batch']):
                guard();hi=min(lo+b['batch'],len(H));u=H[lo:hi]/np.sqrt(np.mean(H[lo:hi]**2,axis=1,keepdims=True)+b['epsilon']);ref=u@A.T
                score=(z[lo:hi]@head.T+bias)/radius[lo:hi,None];values[lo:hi]=metrics(score,ref,np)
                if preflight:
                    direct=(y[lo:hi]/radius[lo:hi,None])@A.T;dm=metrics(direct,ref,np)
                    controls.append(dict(lo=lo,hi=hi,score_sup_relative=float(np.max(np.abs(score-direct))/max(1.,float(np.max(np.abs(direct))))),
                        KL_absolute=float(np.max(np.abs(values[lo:hi,0]-dm[:,0]))),argmax_differences=int(np.count_nonzero(np.argmax(score,axis=1)!=np.argmax(direct,axis=1)))))
                if gradient:
                    residual=(np.exp(logs(score,np))-np.exp(logs(ref,np)))*weights[lo:hi,None];g[lo:hi]=residual@A
                    gy[lo:hi]=g[lo:hi]/radius[lo:hi,None]-y[lo:hi]*np.sum(y[lo:hi]*g[lo:hi],axis=1,keepdims=True)/(2048*radius[lo:hi,None]**3)
            packet=save(name+'_score_factors',head=head,bias=bias,z=z,radius=radius,metrics=values)
            result=dict(name=name,parameters=params,file=packet,**summarize(values,meta,b['quality']))
            if gradient:
                grads=dict(V=gy.T@z,W=(H-pp['mu']).T@(gy@pp['V']),mu=np.sum(gy-(gy@pp['V'])@pp['W'].T,axis=0))
                result['gradient']=save('gradient',**grads,normalized_gradient=g,raw_gradient=gy)
                return result,grads,controls
            return result
        stage='FIT_baseline_gradient';t0=time.monotonic();baseline,grads,control=evaluate(theta,'FIT_baseline',gradient=True,preflight=True);baseline_seconds=time.monotonic()-t0
        factor_pass=all(all(c[k]<=v for k,v in b['factor_numeric'].items()) for c in control)
        scales={k:float(np.linalg.norm(theta[k])) for k in ['V','W']};scales['mu']=math.sqrt(float(np.sum(weights*np.sum(H*H,axis=1))))
        norms={k:float(np.linalg.norm(grads[k])) for k in theta};direction={k:-grads[k]*scales[k]/norms[k] if norms[k]>b['gradient_floor'] else np.zeros_like(grads[k]) for k in theta}
        dfile=save('trust_direction',**direction);analytic=math.fsum(float(np.sum(grads[k]*direction[k])) for k in theta);finite=[]
        print(json.dumps(dict(baseline=baseline['summary'],factor_pass=factor_pass,seconds=time.monotonic()-START)),flush=True)
        for eps in b['finite_epsilon']:
            stage='broad_joint_derivative';points=[]
            for sign in [-1,1]:points.append(evaluate({k:theta[k]+sign*eps*direction[k] for k in theta},'finite_'+str(eps)+'_'+str(sign)))
            fd=(points[1]['summary']['case_mean_KL']-points[0]['summary']['case_mean_KL'])/(2*eps);delta=abs(fd-analytic)/max(1.,abs(analytic))
            finite.append(dict(epsilon=eps,minus=points[0],plus=points[1],analytic=analytic,finite=fd,delta=delta,gate=delta<=b['derivative_tolerance']))
            print(json.dumps(dict(epsilon=eps,delta=delta,seconds=time.monotonic()-START)),flush=True)
        candidates=[]
        for alpha in b['alpha']:
            stage='FIT_ladder';point=evaluate({k:theta[k]+alpha*direction[k] for k in theta},'candidate_'+str(alpha));point['alpha']=alpha;candidates.append(point)
            print(json.dumps(dict(alpha=alpha,FIT=point['summary'],seconds=time.monotonic()-START)),flush=True)
        best=min(candidates,key=lambda x:x['summary']['case_mean_KL']);improvement=(baseline['summary']['case_mean_KL']-best['summary']['case_mean_KL'])/max(baseline['summary']['case_mean_KL'],1e-300)
        accepted=factor_pass and all(f['gate'] for f in finite) and improvement>=b['FIT_relative_improvement_gate'];chosen=best if accepted else baseline
        seal=a.directory/'selection.json';write(seal,dict(accepted=accepted,alpha=best['alpha'] if accepted else 0.,parameters=chosen['parameters'],score_factors=chosen['file'],FIT_relative_improvement=improvement,
            rule='Original initializer;all FIT labels,equal-case exact loss;fixed ladder. Seal before new DEV semantics.'))
        selection=extent(seal);stage='DEV_after_selection';devH,devweights,devmeta=dataset(b['cases']['DEV'],np);devfile=save('DEV_dataset',H=devH,weights=devweights)
        # Reuse evaluate's equations with the same fixed trained source head; no DEV affects choice.
        devbase=evaluate(theta,'DEV_baseline',H=devH,weights=devweights,meta=devmeta)
        dev=evaluate(load(chosen['parameters']),'DEV_selected',H=devH,weights=devweights,meta=devmeta)
        assert extent(seal)==selection;check();guard();outputs=[extent(p) for p in sorted(a.directory.iterdir())]
        result=dict(schema='BROAD_EXACT_RESIDUAL_CORRECTION_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,FIT_dataset=fitfile,FIT_cases=meta,DEV_dataset=devfile,DEV_cases=devmeta,
            baseline=baseline,factor_control=control,factor_pass=factor_pass,baseline_gradient_seconds=baseline_seconds,gradient_norms=norms,trust_scales=scales,direction=dfile,
            finite=finite,candidates=candidates,selection=selection,accepted=accepted,FIT_relative_improvement=improvement,selected=chosen,
            DEV_baseline=devbase,DEV_selected=dev,outputs=outputs,inputs_before_after_exact=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            decision='BROAD_VERIFIED_FIT_DESCENT' if accepted else 'COMPLETED_BROAD_NONACCEPTANCE',scope=b['scope'])
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage);raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.prepare:prepare(a)
    else:assert all((a.binding,a.binding_sha,a.freeze,a.directory));run(a)
