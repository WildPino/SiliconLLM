"""Full-label entropy-mixture dual witnesses; no primal optimization or model run."""
import argparse,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic(); CLOCK=time.perf_counter()


def prepare(a):
    import numpy as np,psutil
    report=DOC/'all_label_native_probability_gap_result_20261012.json'; audit=DOC/'all_label_native_probability_gap_adjudication_20261012.json'
    r=json.loads(report.read_bytes()); ad=json.loads(audit.read_bytes()); assert ad['report']==extent(report) and ad['decision']=='INDEPENDENT_STORED_PASS'
    parent=json.loads(Path(r['binding']['path']).read_bytes()); native=parent['native']
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',report,audit,Path(r['binding']['path']),Path(r['final_checkpoint']['path']),
           DOC/'DUAL_ENTROPY_MIXTURE_PROTOCOL_20261012.md',Path(parent['prior']['source'])/'model.safetensors',Path(parent['prior']['source'])/'config.json',
           Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(psutil.__file__)]
    paths += [Path(native[k]['path']) for k in ['head_F32','bias_F32','epsilon_F32']]
    paths += [Path(parent['prior']['parent'][s+'_dataset']['path']) for s in ['FIT','DEV']]
    paths += [Path(s['file']['path']) for s in parent['preflight']['splits']]+[Path(s['scores']['path']) for s in r['splits']]
    paths += list((SITE/'numpy.libs').glob('*.dll'))+list((SITE/'psutil').glob('*.pyd'))
    b=dict(schema='DUAL_ENTROPY_MIXTURE_BINDING_V1',python=str(Path(sys.executable).resolve()),oracle=r,parent=parent,
           families=['final_relaxed_real','final_actual_native'],lambdas=[0.,.25,.5,.75,.9,.99,.999,1.],batch=64,threads=6,
           numeric=dict(moment_relative=1e-9,entropy_absolute=1e-8,mass_absolute=1e-11,endpoint_absolute=1e-8,concavity_slack=1e-8,allowance=1e-8),
           limits=dict(seconds=900,OS_bytes=6<<30,output_bytes=512<<20),inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
           scope='All8808 existing labels,2 recorded candidate distributions,fixed8-lambda entropy-mixture dual family. '
                 'Full source/operator probabilities/moments and real half-ball supports;not interval/globalF32 floor. '
                 'No primal optimization/source model/GPU/T4/RESERVED/longalign/causal model/n/50 admission.')
    write(a.out,b); print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha; b=json.loads(a.binding.read_bytes()); assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']) and not a.directory.exists() and not a.out.exists()
    proc=psutil.Process(); proc.cpu_affinity(list(range(6))); stream=mapping=None
    receipt=dict(schema='DUAL_ENTROPY_MIXTURE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=b['limits']['output_bytes']
    def check():
        for e in b['inputs']:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def prob(score):
        shifted=score-score.max(axis=1,keepdims=True); lp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True)); pp=np.exp(lp); mass=pp.sum(axis=1)
        return pp/mass[:,None],lp-np.log(mass)[:,None],mass
    def entropy(pp):
        log=np.zeros_like(pp); np.log(pp,out=log,where=pp>0); return -np.sum(pp*log,axis=1)
    def support(g):return 16*np.sqrt(np.sum(g[:,:256]**2,axis=1)+np.minimum(g[:,-1],0)**2)
    def summaries(lower,cases,quality,weights):
        cc=[dict(id=c['id'],domain=c['domain'],labels=c['hi']-c['lo'],lower=float(lower[c['lo']:c['hi']].mean())) for c in cases]
        dom=[dict(domain=d,lower=sum(x['lower'] for x in cc if x['domain']==d)/2) for d in sorted({c['domain'] for c in cases})]
        value=float(weights@lower); worst=float(lower.max()); violation=bool(value>quality['case_mean_KL'] or worst>quality['worst_label_KL'] or any(c['lower']>quality['every_case_KL'] for c in cc))
        return dict(case_mean_lower=value,label_mean_lower=float(lower.mean()),worst_label_lower=worst,positive_labels=int(np.count_nonzero(lower>0)),cases=cc,domains=dom,real_domain_budget_violation=violation)
    try:
        check(); a.directory.mkdir(); parent=b['parent']; native=parent['native']; checkpoint=load(b['oracle']['final_checkpoint'])
        C=np.fromfile(native['head_F32']['path'],'<f4').reshape(65537,256).astype('f8'); beta=np.fromfile(native['bias_F32']['path'],'<f4').astype('f8'); epsilon=float(np.fromfile(native['epsilon_F32']['path'],'<f4')[0]); aug=np.column_stack([C,beta/math.sqrt(256*epsilon)])
        get,stream=tensor_reader(Path(parent['prior']['source'])/'model.safetensors',np); cfg=json.loads((Path(parent['prior']['source'])/'config.json').read_bytes()); gamma=get('model.final_layernorm.weight'); source=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):
            guard(); hi=min(lo+1024,65537); source[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close(); stream=None; outputs=[]; splits=[]; offset=0
        for oracle_split in b['oracle']['splits']:
            name=oracle_split['split']; data=load(parent['prior']['parent'][name+'_dataset']); old=load(next(s for s in parent['preflight']['splits'] if s['split']==name)['file']); n=len(data['H'])
            egrid=np.empty((n,2,8),'f8'); mgrid=np.empty((n,2,8,257),'f8'); sgrid=np.empty((n,2,8),'f8'); rawgrid=np.empty((n,2,8),'f8'); massgrid=np.empty((n,2,8),'f8')
            pmom=np.empty((n,2,257),'f8'); ph=np.empty((n,2),'f8'); reverse=np.empty((n,2),'f8'); derivative=np.empty((n,2),'f8'); pzero=np.empty((n,2),'i8'); qzero=np.empty(n,'i8'); derivative_valid=np.empty((n,2),bool)
            qm=np.empty((n,257),'f8'); qh=np.empty(n,'f8'); qmass=np.empty(n,'f8'); tail_log_mass=np.full(n,-np.inf,'f8'); tail_log_entropy=np.full(n,-np.inf,'f8')
            chosen_family=np.empty(n,'i8'); chosen_lambda=np.empty(n,'i8'); chosen_moment=np.empty((n,257),'f8'); selected_moment_delta=np.empty(n,'f8'); endpoint=np.empty((n,2),'f8')
            mapping=np.memmap(oracle_split['scores']['path'],mode='r',dtype='<f4',shape=(n,65537))
            for lo in range(0,n,64):
                guard(); hi=min(lo+64,n); H=data['H'][lo:hi]; source_score=(H/np.sqrt(np.mean(H*H,axis=1,keepdims=True)+parent['epsilon_source']))@source.T
                q,qlog,mass=prob(source_score); qm[lo:hi]=q@aug; qh[lo:hi]=entropy(q); qmass[lo:hi]=mass; qzero[lo:hi]=(q==0).sum(axis=1)
                for row in range(hi-lo):
                    zz=q[row]==0
                    if zz.any():
                        tail_log_mass[lo+row]=math.log(int(zz.sum()))+float(qlog[row,zz].max())
                        tail_log_entropy[lo+row]=math.log(int(zz.sum()))+float((qlog[row,zz]+np.log(-qlog[row,zz])).max())
                vv=checkpoint['v'][offset+lo:offset+hi]; candidates=[prob(vv@aug.T),prob(mapping[lo:hi].astype('f8'))]; probabilities=[]
                for family,(p,plog,pmass) in enumerate(candidates):
                    assert np.max(np.abs(pmass-1))<=b['numeric']['mass_absolute']; probabilities.append(p)
                    pp_m=p@aug; pp_h=entropy(p); pmom[lo:hi,family]=pp_m; ph[lo:hi,family]=pp_h; pzero[lo:hi,family]=(p==0).sum(axis=1)
                    reverse[lo:hi,family]=np.sum(p*(plog-qlog),axis=1); base_support=support(pp_m-qm[lo:hi]); d0=pp_h-qh[lo:hi]-base_support
                    derivative[lo:hi,family]=-d0-reverse[lo:hi,family]; derivative_valid[lo:hi,family]=(pzero[lo:hi,family]==0)&(qzero[lo:hi]==0)
                    for li,lam in enumerate(b['lambdas']):
                        mixture=(1-lam)*p+lam*q; mmass=mixture.sum(axis=1); mixture/=mmass[:,None]
                        mm=((1-lam)*pp_m+lam*qm[lo:hi])/mmass[:,None]; hh=entropy(mixture); ss=support(mm-qm[lo:hi])
                        egrid[lo:hi,family,li]=hh; mgrid[lo:hi,family,li]=mm; sgrid[lo:hi,family,li]=ss; massgrid[lo:hi,family,li]=mmass; rawgrid[lo:hi,family,li]=hh-qh[lo:hi]-ss
                    endpoint[lo:hi,family]=rawgrid[lo:hi,family,-1]
                    # Unequal lambda spacing: concavity is checked in secant slopes.
                    slopes=np.diff(rawgrid[lo:hi,family],axis=1)/np.diff(np.array(b['lambdas']))[None,:]
                    assert np.max(np.diff(slopes,axis=1))<=b['numeric']['concavity_slack']
                safe=np.maximum(0.,rawgrid[lo:hi]-b['numeric']['allowance']); flat=np.argmax(safe.reshape(hi-lo,16),axis=1); families=flat//8; lambdas=flat%8
                chosen_family[lo:hi]=families; chosen_lambda[lo:hi]=lambdas
                pstack=np.stack(probabilities,axis=1); pp=pstack[np.arange(hi-lo),families]; ll=np.array(b['lambdas'])[lambdas]; mixture=(1-ll[:,None])*pp+ll[:,None]*q; mixture/=mixture.sum(axis=1,keepdims=True)
                actual_moment=mixture@aug; chosen_moment[lo:hi]=actual_moment; expected=mgrid[lo:hi][np.arange(hi-lo),families,lambdas]
                selected_moment_delta[lo:hi]=np.linalg.norm(actual_moment-expected,axis=1); assert np.max(selected_moment_delta[lo:hi])<=1e-9
                assert np.max(np.abs(endpoint[lo:hi]))<=b['numeric']['endpoint_absolute']
            mapping._mmap.close(); mapping=None
            moment_rel=float(np.linalg.norm(qm-old['moments'])/max(1.,float(np.linalg.norm(old['moments'])))); entropy_delta=float(np.max(np.abs(qh-old['entropy'])))
            assert moment_rel<=b['numeric']['moment_relative'] and entropy_delta<=b['numeric']['entropy_absolute'] and np.max(np.abs(qmass-1))<=b['numeric']['mass_absolute']
            safe=np.maximum(0.,rawgrid-b['numeric']['allowance']); selected=safe[np.arange(n),chosen_family,chosen_lambda]
            path=a.directory/(name+'_dual.npz'); np.savez(path,entropy=egrid,moments=mgrid,support=sgrid,raw_lower=rawgrid,safe_lower=safe,mixture_mass=massgrid,
                p_moments=pmom,p_entropy=ph,reverse_KL=reverse,derivative_at1=derivative,derivative_valid=derivative_valid,p_zeros=pzero,q_zeros=qzero,q_moment=qm,q_entropy=qh,q_mass=qmass,
                q_tail_log_mass_upper=tail_log_mass,q_tail_log_entropy_upper=tail_log_entropy,chosen_family=chosen_family,chosen_lambda=chosen_lambda,chosen_moment_direct=chosen_moment,
                chosen_moment_linear_delta=selected_moment_delta,endpoint=endpoint,selected_lower=selected,weights=data['weights'])
            e=extent(path); outputs.append(e); summary=summaries(selected,parent['prior']['parent'][name+'_cases'],parent['quality'],data['weights'])
            splits.append(dict(split=name,rows=n,file=e,summary=summary,source_moment_relative=moment_rel,source_entropy_delta=entropy_delta,source_zero_words=int(qzero.sum()),
                               candidate_zero_words=int(pzero.sum()),maximum_selected_moment_delta=float(selected_moment_delta.max()),selected_family_counts=np.bincount(chosen_family,minlength=2).tolist(),selected_lambda_counts=np.bincount(chosen_lambda,minlength=8).tolist()))
            offset+=n; print(json.dumps(dict(split=name,summary={k:summary[k] for k in ['case_mean_lower','worst_label_lower','positive_labels','real_domain_budget_violation']},seconds=time.perf_counter()-CLOCK)),flush=True)
        assert offset==8808; check(); guard()
        result=dict(schema='DUAL_ENTROPY_MIXTURE_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,outputs=outputs,splits=splits,
                    decision='COMPLETED_ENTROPY_MIXTURE_WITNESSES_PENDING_INDEPENDENT_AUDIT',numerical_real_domain_budget_violation=any(s['summary']['real_domain_budget_violation'] for s in splits),
                    before_after_hashes_exact=True,interval_certified=False,global_F32_floor_proved=False,primal_optimization_run=False,seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope=b['scope'])
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        if stream is not None:stream.close()
        if mapping is not None:mapping._mmap.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset); write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args(); assert a.prepare!=a.run
    prepare(a) if a.prepare else run(a)
