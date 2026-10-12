"""Direct full-source-head replay of all score factors and broad gradient."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.report)
    b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'] and not a.out.exists()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;maxrel=maxscalar=0.;points=0;labelrows=0;replay_metrics={};factor_replay=[]
    extra=[extent(a.report),extent(a.terminal),extent(__file__),extent(DOC/'BROAD_EXACT_RESIDUAL_STORED_AUDIT_PROTOCOL_V2_20261012.md')]
    receipt=dict(schema='BROAD_EXACT_RESIDUAL_AUDIT_TERMINAL_V1',command=sys.orig_argv,freeze=a.freeze,source=extent(__file__),pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():assert time.monotonic()-START<=1200 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    def check():
        for x in b['inputs']+r['outputs']+[r['binding']]+extra:guard();assert extent(x['path'])==x,x['path']
    def compare(x,y,tol=1e-10):
        nonlocal maxrel
        rel=float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-300));maxrel=max(maxrel,rel);assert rel<=tol,rel
    def scalar(x,y,tol=1e-8):
        nonlocal maxscalar
        d=abs(float(x)-float(y));maxscalar=max(maxscalar,d);assert d<=tol*max(1.,abs(float(y))),(x,y,d)
    def load(x):
        with np.load(x['path'],allow_pickle=False) as p:return {k:p[k].copy() for k in p.files}
    def data(split):
        H=[];weights=[];meta=[];lo=0
        for c in b['cases'][split]:
            raw=(np.fromfile(c['h']['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(c['history'],2048).astype('f8');hh=raw[c['positions']];H.append(hh);n=len(hh)
            weights.extend([1/(24*n)]*n);meta.append(dict(**c,lo=lo,hi=lo+n));lo+=n
        return np.concatenate(H),np.array(weights,'f8'),meta
    def full_head(x,A):
        # Source-width partials, never the producer's compact projected score shortcut.
        score=np.zeros((len(x),65537),'f8')
        for lo in range(0,2048,256):score+=x[:,lo:lo+256]@A[:,lo:lo+256].T
        return score
    def probs(s):return s-np.logaddexp.reduce(s,axis=1)[:,None]
    def summary(values,meta,point):
        cc=[]
        for c in meta:
            vv=values[c['lo']:c['hi']];cc.append(dict(id=c['id'],domain=c['domain'],labels=len(vv),KL=math.fsum(vv[:,0].tolist())/len(vv),disagreement_rate=float(vv[:,1].mean()),worst_KL=float(vv[:,0].max())))
        ss=dict(case_mean_KL=math.fsum(c['KL'] for c in cc)/24,case_mean_disagreement=math.fsum(c['disagreement_rate'] for c in cc)/24,label_mean_KL=math.fsum(values[:,0].tolist())/len(values),
            label_disagreement=float(values[:,1].mean()),worst_label_KL=float(values[:,0].max()),worst_case_KL=max(c['KL'] for c in cc))
        for key in ss:scalar(ss[key],point['summary'][key])
        for calc,stored in zip(cc,point['cases']):
            assert all(calc[k]==stored[k] for k in ['id','domain','labels'])
            for k in ['KL','disagreement_rate','worst_KL']:scalar(calc[k],stored[k])
        domains=[]
        for d in sorted(set(c['domain'] for c in cc)):
            members=[c for c in cc if c['domain']==d];assert len(members)==2
            row=dict(domain=d,KL=math.fsum(c['KL'] for c in members)/2,disagreement_rate=math.fsum(c['disagreement_rate'] for c in members)/2);domains.append(row)
        for calc,stored in zip(domains,point['domains']):
            assert calc['domain']==stored['domain'];scalar(calc['KL'],stored['KL']);scalar(calc['disagreement_rate'],stored['disagreement_rate'])
        q=b['quality'];flags=dict(case_mean_KL=ss['case_mean_KL']<=q['case_mean_KL'],case_mean_disagreement=ss['case_mean_disagreement']<=q['case_mean_disagreement'],
            worst_label_KL=ss['worst_label_KL']<=q['worst_label_KL'],every_case_KL=all(c['KL']<=q['every_case_KL'] for c in cc),every_case_disagreement=all(c['disagreement_rate']<=q['every_case_disagreement'] for c in cc),
            domains=all(c['KL']<=q['every_case_KL'] and c['disagreement_rate']<=q['every_case_disagreement'] for c in domains))
        assert flags==point['quality_gates']
    try:
        check();get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np);cfg=json.loads((Path(b['source'])/'config.json').read_bytes());gamma=get('model.final_layernorm.weight');A=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close();stream=None;initial=load(b['initial_parameters']);computed_gradient=None
        def auditpoint(point,H,weights,meta,gradient=False):
            nonlocal points,labelrows,computed_gradient
            guard();params=load(point['parameters']);packet=load(point['file']);Z=np.zeros((len(H),256),'f8')
            for lo in range(0,2048,256):Z+=(H[:,lo:lo+256]-params['mu'][lo:lo+256])@params['W'][lo:lo+256]
            Y=params['mu']+Z@params['V'].T;R=np.sqrt(np.sum(Y*Y,axis=1)/2048+b['epsilon']);compare(packet['z'],Z);compare(packet['radius'],R)
            for lo in range(0,65537,1024):
                guard();hi=min(lo+1024,65537);bh=np.zeros((hi-lo,256),'f8');bias=np.zeros(hi-lo,'f8')
                for j in range(0,2048,256):bh+=A[lo:hi,j:j+256]@params['V'][j:j+256];bias+=A[lo:hi,j:j+256]@params['mu'][j:j+256]
                compare(packet['head'][lo:hi],bh);compare(packet['bias'][lo:hi],bias)
            values=np.empty((len(H),4),'f8');GV=np.zeros_like(params['V']);GW=np.zeros_like(params['W']);GM=np.zeros_like(params['mu'])
            gnorm=np.empty_like(H) if gradient else None;graw=np.empty_like(H) if gradient else None;identity=np.eye(2048) if gradient else None
            for lo in range(0,len(H),64):
                guard();hi=min(lo+64,len(H));rr=np.sqrt(np.mean(H[lo:hi]**2,axis=1)+b['epsilon']);ref=full_head(H[lo:hi]/rr[:,None],A);direct=full_head(Y[lo:hi]/R[lo:hi,None],A)
                lq=probs(ref);lp=probs(direct);q=np.exp(lq);p=np.exp(lp)
                values[lo:hi]=np.column_stack([np.sum(q*(lq-lp),axis=1),np.argmax(ref,axis=1)!=np.argmax(direct,axis=1),-np.sum(q*lq,axis=1),-np.sum(p*lp,axis=1)])
                if gradient:
                    g=np.zeros((hi-lo,2048),'f8')
                    for j in range(0,65537,1024):end=min(j+1024,65537);g+=(p[:,j:end]-q[:,j:end])*weights[lo:hi,None]@A[j:end]
                    gnorm[lo:hi]=g
                    for ii,i in enumerate(range(lo,hi)):
                        jac=identity/R[i]-np.outer(Y[i],Y[i])/(2048*R[i]**3);dy=jac@g[ii];graw[i]=dy;dz=params['V'].T@dy
                        GV+=np.outer(dy,Z[i]);GW+=np.outer(H[i]-params['mu'],dz);GM+=dy-params['W']@dz
                    compact=(packet['z'][lo:hi]@packet['head'].T+packet['bias'])/packet['radius'][lo:hi,None]
                    c=next(x for x in r['factor_control'] if x['lo']==lo);scalar(float(np.max(np.abs(compact-direct))/max(1.,float(np.max(np.abs(direct))))),c['score_sup_relative'])
                    replay=dict(score_sup_relative=float(np.max(np.abs(compact-direct))/max(1.,float(np.max(np.abs(direct))))),
                        KL_absolute=float(np.max(np.abs(np.sum(q*(lq-probs(compact)),axis=1)-values[lo:hi,0]))),
                        argmax_differences=int(np.count_nonzero(np.argmax(compact,axis=1)!=np.argmax(direct,axis=1))))
                    assert all(replay[k]<=limit for k,limit in b['factor_numeric'].items());factor_replay.append(replay)
                    # Original preflight direct used one full BLAS operation; independent source partial order is separately measured.
                    assert np.array_equal(np.argmax(compact,axis=1),np.argmax(direct,axis=1))
                compare(packet['metrics'][lo:hi],values[lo:hi],1e-9)
            summary(values,meta,point);replay_metrics[point['name']]=values;points+=1;labelrows+=len(H)
            if gradient:
                stored=load(point['gradient']);compare(stored['V'],GV);compare(stored['W'],GW);compare(stored['mu'],GM);compare(stored['normalized_gradient'],gnorm);compare(stored['raw_gradient'],graw)
                computed_gradient=stored
            print(json.dumps(dict(audited_point=point['name'],positions=len(H),seconds=time.monotonic()-START)),flush=True)
        H,weights,meta=data('FIT');stored=load(r['FIT_dataset']);assert np.array_equal(stored['H'],H) and np.array_equal(stored['weights'],weights) and meta==r['FIT_cases'];del stored
        auditpoint(r['baseline'],H,weights,meta,gradient=True);params=load(r['baseline']['parameters'])
        for k in initial:assert np.array_equal(initial[k],params[k])
        grad=computed_gradient;direction=load(r['direction']);scales={k:float(np.linalg.norm(initial[k])) for k in ['V','W']};scales['mu']=math.sqrt(float(np.sum(weights*np.sum(H*H,axis=1))))
        for k in initial:
            norm=float(np.linalg.norm(grad[k]));scalar(norm,r['gradient_norms'][k]);scalar(scales[k],r['trust_scales'][k]);expected=-grad[k]*scales[k]/norm if norm>b['gradient_floor'] else np.zeros_like(grad[k]);assert np.array_equal(expected,direction[k])
        analytic=math.fsum(float(np.sum(grad[k]*direction[k])) for k in initial)
        for f in r['finite']:
            assert f['epsilon'] in b['finite_epsilon'];eps=f['epsilon']
            for sign,p in [(-1,f['minus']),(1,f['plus'])]:
                params=load(p['parameters'])
                for k in initial:assert np.array_equal(params[k],initial[k]+sign*eps*direction[k])
                auditpoint(p,H,weights,meta)
            scalar(analytic,f['analytic']);fd=(f['plus']['summary']['case_mean_KL']-f['minus']['summary']['case_mean_KL'])/(2*eps);scalar(fd,f['finite']);delta=abs(fd-analytic)/max(1.,abs(analytic));scalar(delta,f['delta']);assert f['gate']==(delta<=b['derivative_tolerance'])
        assert len(r['finite'])==2
        for p in r['candidates']:
            assert p['alpha'] in b['alpha'];params=load(p['parameters'])
            for k in initial:assert np.array_equal(params[k],initial[k]+p['alpha']*direction[k])
            auditpoint(p,H,weights,meta)
        factorpass=all(all(c[k]<=v for k,v in b['factor_numeric'].items()) for c in r['factor_control']);assert factorpass==r['factor_pass']
        best=min(r['candidates'],key=lambda x:x['summary']['case_mean_KL']);improve=(r['baseline']['summary']['case_mean_KL']-best['summary']['case_mean_KL'])/max(r['baseline']['summary']['case_mean_KL'],1e-300);scalar(improve,r['FIT_relative_improvement'])
        accepted=factorpass and all(f['gate'] for f in r['finite']) and improve>=b['FIT_relative_improvement_gate'];assert accepted==r['accepted'];chosen=best if accepted else r['baseline'];assert chosen==r['selected']
        selection=json.loads(Path(r['selection']['path']).read_bytes());assert extent(r['selection']['path'])==r['selection'];assert selection['parameters']==chosen['parameters'] and selection['score_factors']==chosen['file'] and selection['accepted']==accepted and selection['alpha']==(best['alpha'] if accepted else 0.);scalar(selection['FIT_relative_improvement'],improve)
        H,weights,meta=data('DEV');stored=load(r['DEV_dataset']);assert np.array_equal(stored['H'],H) and np.array_equal(stored['weights'],weights) and meta==r['DEV_cases'];del stored
        for p,expected in [(r['DEV_baseline'],initial),(r['DEV_selected'],load(chosen['parameters']))]:
            params=load(p['parameters'])
            for k in initial:assert np.array_equal(params[k],expected[k])
            auditpoint(p,H,weights,meta)
        assert points==11 and labelrows==48570
        expected='BROAD_VERIFIED_FIT_DESCENT' if accepted else 'COMPLETED_BROAD_NONACCEPTANCE';assert r['decision']==expected;check();guard()
        result=dict(schema='BROAD_EXACT_RESIDUAL_STORED_AUDIT_V1',source=extent(__file__),freeze=a.freeze,report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],
            all8808_raw_source_labels_weights_and11_score_factor_points48570_probability_rows_checked=True,
            all4422_explicit_RMS_Jacobians_outer_gradients4_difference_points4_bit_updates_FIT_seal_and_DEV_checked=True,
            independent_all_FIT_factor_numeric_controls=factor_replay,
            max_numeric_relative=maxrel,max_scalar_delta=maxscalar,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',scientific_decision=expected,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            scope='All scores/probabilities from direct source-width2048 features,not producer projected-head shortcut;actual source references reread,full per-state RMS matrices and outer gradients. '
                'Frozen derivative subtraction uses independently verified producer means. No model/refit/GPU/native/T4 replay;selection order supported by frozen control flow,not semantic-read timestamps.')
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
