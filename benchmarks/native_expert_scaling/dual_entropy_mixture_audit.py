"""Every source probability, family/lambda entropy and direct moment audit."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic(); CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes()); term=json.loads(a.terminal.read_bytes()); assert term['exit_code']==0 and term['result']==extent(a.report)
    assert extent(r['binding']['path'])==r['binding']; b=json.loads(Path(r['binding']['path']).read_bytes()); parent=b['parent']
    proc=psutil.Process(); proc.cpu_affinity(list(range(6))); stream=mapping=None; maxrel=maxscalar=0.; rows=grid_rows=0
    extra=[extent(__file__),extent(DOC/'DUAL_ENTROPY_MIXTURE_AUDIT_PROTOCOL_20261012.md'),extent(a.report),extent(a.terminal),r['binding']]
    receipt=dict(schema='DUAL_ENTROPY_MIXTURE_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+r['outputs']+extra:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def compare(x,y,tolerance=1e-9):
        nonlocal maxrel
        assert x.shape==y.shape and np.array_equal(np.isnan(x),np.isnan(y)) and np.array_equal(np.isinf(x),np.isinf(y))
        mask=np.isfinite(x); assert np.array_equal(x[~mask],y[~mask],equal_nan=True)
        error=float(np.linalg.norm(x[mask]-y[mask])/max(1.,float(np.linalg.norm(y[mask])))); maxrel=max(maxrel,error); assert error<=tolerance,error
    def scalar(x,y,tolerance=1e-8):
        nonlocal maxscalar
        error=abs(float(x)-float(y)); maxscalar=max(maxscalar,error); assert error<=tolerance*max(1.,abs(float(y))),(x,y)
    def partial(x,A,width):
        out=np.zeros((len(x),len(A)),'f8')
        for col in range(0,x.shape[1],width):out+=x[:,col:col+width]@A[:,col:col+width].T
        return out
    def probability(score):
        lp=score-np.logaddexp.reduce(score,axis=1)[:,None]; p=np.exp(lp); mass=p.sum(axis=1); return p/mass[:,None],lp-np.log(mass)[:,None],mass
    def entropy(p):
        out=np.zeros(len(p),'f8')
        for lo in range(0,65537,4096):
            x=p[:,lo:lo+4096]; log=np.zeros_like(x); np.log(x,out=log,where=x>0); out-=np.sum(x*log,axis=1)
        return out
    def moment(p,A):
        out=np.zeros((len(p),257),'f8')
        for lo in range(0,65537,4096):out+=p[:,lo:lo+4096]@A[lo:lo+4096]
        return out
    def support(g):
        direction=-g.copy(); direction[:,-1]=np.maximum(direction[:,-1],0.); norm=np.sqrt(np.einsum('ij,ij->i',direction,direction)); endpoint=direction*(16/np.maximum(norm,1e-300))[:,None]
        return -np.einsum('ij,ij->i',g,endpoint)
    try:
        check(); native=parent['native']; C=np.fromfile(native['head_F32']['path'],'<f4').reshape(65537,256).astype('f8'); beta=np.fromfile(native['bias_F32']['path'],'<f4').astype('f8')
        epsilon=float(np.fromfile(native['epsilon_F32']['path'],'<f4')[0]); scale=math.sqrt(256*epsilon); aug=np.column_stack([C,beta/scale]); cp=load(b['oracle']['final_checkpoint'])
        get,stream=tensor_reader(Path(parent['prior']['source'])/'model.safetensors',np); cfg=json.loads((Path(parent['prior']['source'])/'config.json').read_bytes()); gamma=get('model.final_layernorm.weight'); source=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):
            guard(); hi=min(lo+1024,65537); source[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close(); stream=None; offset=0; splitchecks=[]
        for s in r['splits']:
            data=load(parent['prior']['parent'][s['split']+'_dataset']); old=load(next(x for x in parent['preflight']['splits'] if x['split']==s['split'])['file']); p=load(s['file']); n=len(data['H'])
            assert p['moments'].shape==(n,2,8,257); score_extent=next(x for x in b['oracle']['splits'] if x['split']==s['split'])['scores']; mapping=np.memmap(score_extent['path'],mode='r',dtype='<f4',shape=(n,65537))
            for lo in range(0,n,64):
                guard(); hi=min(lo+64,n); nn=hi-lo; x=data['H'][lo:hi]; score=partial(x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+parent['epsilon_source']),source,256)
                q,ql,mass=probability(score); mq=moment(q,aug); hq=entropy(q); compare(mq,p['q_moment'][lo:hi]); compare(hq,p['q_entropy'][lo:hi]); compare(mass,p['q_mass'][lo:hi]); compare(mq,old['moments'][lo:hi]); compare(hq,old['entropy'][lo:hi])
                zz=(q==0).sum(axis=1); assert np.array_equal(zz,p['q_zeros'][lo:hi]); tlm=np.full(nn,-np.inf,'f8'); tle=tlm.copy()
                for row in range(nn):
                    zero=q[row]==0
                    if zero.any():tlm[row]=math.log(int(zero.sum()))+float(ql[row,zero].max()); tle[row]=math.log(int(zero.sum()))+float((ql[row,zero]+np.log(-ql[row,zero])).max())
                compare(tlm,p['q_tail_log_mass_upper'][lo:hi]); compare(tle,p['q_tail_log_entropy_upper'][lo:hi])
                v=cp['v'][offset+lo:offset+hi]; real=partial(v[:,:256],C,32)+v[:,-1,None]*(beta/scale); candidates=[probability(real),probability(mapping[lo:hi].astype('f8'))]; independent=np.empty((nn,2,8),'f8')
                for family,(pp,pl,pmass) in enumerate(candidates):
                    mp=moment(pp,aug); hp=entropy(pp); compare(mp,p['p_moments'][lo:hi,family]); compare(hp,p['p_entropy'][lo:hi,family])
                    zero=(pp==0).sum(axis=1); assert np.array_equal(zero,p['p_zeros'][lo:hi,family]); valid=(zero==0)&(zz==0); assert np.array_equal(valid,p['derivative_valid'][lo:hi,family])
                    reverse=np.sum(pp*(pl-ql),axis=1); compare(reverse,p['reverse_KL'][lo:hi,family]); base=hp-hq-support(mp-mq); compare(-base-reverse,p['derivative_at1'][lo:hi,family])
                    for li,lam in enumerate(b['lambdas']):
                        guard(); mix=(1-lam)*pp+lam*q; mixmass=mix.sum(axis=1); mix/=mixmass[:,None]
                        # Direct moment at EVERY lambda,not producer linear reconstruction.
                        mr=moment(mix,aug); hr=entropy(mix); ss=support(mr-mq); raw=hr-hq-ss; independent[:,family,li]=np.maximum(0.,raw-b['numeric']['allowance'])
                        compare(mr,p['moments'][lo:hi,family,li]); compare(hr,p['entropy'][lo:hi,family,li]); compare(ss,p['support'][lo:hi,family,li]); compare(mixmass,p['mixture_mass'][lo:hi,family,li]); compare(raw,p['raw_lower'][lo:hi,family,li]); compare(independent[:,family,li],p['safe_lower'][lo:hi,family,li])
                        assert np.max(np.abs(mixmass-1))<=b['numeric']['mass_absolute']; grid_rows+=nn
                    slopes=np.diff(p['raw_lower'][lo:hi,family],axis=1)/np.diff(np.array(b['lambdas']))[None,:]; assert np.max(np.diff(slopes,axis=1))<=b['numeric']['concavity_slack']
                    compare(p['endpoint'][lo:hi,family],p['raw_lower'][lo:hi,family,-1]); assert np.max(np.abs(p['endpoint'][lo:hi,family]))<=b['numeric']['endpoint_absolute']
                # Exact selection rule is audited on immutable producer grid;independent
                # values agree numerically,without refitting ties through roundoff.
                stored=p['safe_lower'][lo:hi]; chosen=np.argmax(stored.reshape(nn,16),axis=1); assert np.array_equal(chosen//8,p['chosen_family'][lo:hi]) and np.array_equal(chosen%8,p['chosen_lambda'][lo:hi])
                pp=np.stack([candidate[0] for candidate in candidates],axis=1)[np.arange(nn),chosen//8]; lam=np.array(b['lambdas'])[chosen%8]
                mix=(1-lam[:,None])*pp+lam[:,None]*q; mix/=mix.sum(axis=1,keepdims=True); direct=moment(mix,aug)
                compare(direct,p['chosen_moment_direct'][lo:hi]); expected=p['moments'][lo:hi][np.arange(nn),chosen//8,chosen%8]
                compare(np.linalg.norm(p['chosen_moment_direct'][lo:hi]-expected,axis=1),p['chosen_moment_linear_delta'][lo:hi])
                compare(p['selected_lower'][lo:hi],stored[np.arange(nn),chosen//8,chosen%8]); rows+=nn
            mapping._mmap.close(); mapping=None; lower=p['selected_lower']; compare(p['weights'],data['weights'],0.); summary=s['summary']; cases=parent['prior']['parent'][s['split']+'_cases']
            means=[]
            for c in cases:
                value=float(lower[c['lo']:c['hi']].mean()); means.append(value); oldcase=next(x for x in summary['cases'] if x['id']==c['id']); scalar(value,oldcase['lower']); assert oldcase['labels']==c['hi']-c['lo'] and oldcase['domain']==c['domain']
            scalar(data['weights']@lower,summary['case_mean_lower']); scalar(lower.mean(),summary['label_mean_lower']); scalar(lower.max(),summary['worst_label_lower']); assert np.count_nonzero(lower>0)==summary['positive_labels']
            for d in summary['domains']:scalar(sum(value for c,value in zip(cases,means) if c['domain']==d['domain'])/2,d['lower'])
            violation=bool(data['weights']@lower>parent['quality']['case_mean_KL'] or lower.max()>parent['quality']['worst_label_KL'] or max(means)>parent['quality']['every_case_KL']); assert violation==summary['real_domain_budget_violation']
            scalar(np.linalg.norm(p['q_moment']-old['moments'])/max(1.,np.linalg.norm(old['moments'])),s['source_moment_relative']); scalar(np.max(np.abs(p['q_entropy']-old['entropy'])),s['source_entropy_delta'])
            assert int(p['q_zeros'].sum())==s['source_zero_words'] and int(p['p_zeros'].sum())==s['candidate_zero_words']; scalar(p['chosen_moment_linear_delta'].max(),s['maximum_selected_moment_delta'])
            assert np.bincount(p['chosen_family'],minlength=2).tolist()==s['selected_family_counts'] and np.bincount(p['chosen_lambda'],minlength=8).tolist()==s['selected_lambda_counts']; offset+=n
            splitchecks.append(dict(split=s['split'],rows=n,all_grid_verified=True,summary=summary)); print(json.dumps(dict(audited=s['split'],seconds=time.perf_counter()-CLOCK)),flush=True)
        assert rows==offset==8808 and grid_rows==140928 and not r['interval_certified'] and not r['global_F32_floor_proved'] and not r['primal_optimization_run']
        assert r['numerical_real_domain_budget_violation']==any(s['summary']['real_domain_budget_violation'] for s in splitchecks); check(); guard()
        result=dict(schema='DUAL_ENTROPY_MIXTURE_AUDIT_V1',source=extent(__file__),freeze=a.freeze,report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],source_rows=rows,direct_family_lambda_moment_rows=grid_rows,
                    all_source_distributions_entropies_direct_grid_moments_supports_underflows_selections_summaries_verified=True,split_checks=splitchecks,
                    max_relative=maxrel,max_scalar_delta=maxscalar,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,
                    scope='All entropy-mixture dual witnesses;numeric real-half-ball lower only,not interval/globalF32 or causal/model/n/50 admission.')
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        if stream is not None:stream.close()
        if mapping is not None:mapping._mmap.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset); write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
