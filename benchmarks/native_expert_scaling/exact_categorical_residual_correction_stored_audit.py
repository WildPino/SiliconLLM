"""Full source-head audit, explicit RMS Jacobians and per-state outer products."""
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
    assert r['schema']=='EXACT_CATEGORICAL_RESIDUAL_CORRECTION_RESULT_V1' and t['exit_code']==0 and t['error'] is None and t['result']==extent(a.report)
    b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'] and not a.out.exists()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;maxrel=maxscalar=0.;points=0
    extra=[extent(a.report),extent(a.terminal),extent(__file__),extent(DOC/'EXACT_CATEGORICAL_RESIDUAL_CORRECTION_STORED_AUDIT_PROTOCOL_20261012.md')]
    receipt=dict(schema='EXACT_CATEGORICAL_RESIDUAL_CORRECTION_AUDIT_TERMINAL_V1',command=sys.orig_argv,freeze=a.freeze,
        source=extent(__file__),pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():
        assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=8<<30 and not proc.children(recursive=True)
    def check():
        for item in b['inputs']+r['outputs']+[r['binding']]+extra:guard();assert extent(item['path'])==item,item['path']
    def compare(x,y,tol=1e-10):
        nonlocal maxrel
        rel=float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-300));maxrel=max(maxrel,rel);assert rel<=tol,rel
    def scalar(x,y,tol=1e-8):
        nonlocal maxscalar
        delta=abs(float(x)-float(y));maxscalar=max(maxscalar,delta);assert delta<=tol*max(1.,abs(float(y))),(x,y)
    def load(item):
        with np.load(item['path'],allow_pickle=False) as p:return {k:p[k].copy() for k in p.files}
    def selected(samples):
        H=[];labels=[]
        for rec in samples:
            raw=(np.fromfile(rec['h']['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(rec['history'],2048).astype('f8');H.append(raw[rec['positions']])
            labels.extend(dict(id=rec['id'],label_index=i,position=p) for i,p in zip(rec['label_indices'],rec['positions']))
        return np.concatenate(H),labels
    def partial_head(x,A):
        out=np.zeros((len(x),len(A)),'f8')
        for lo in range(0,2048,256):out+=x[:,lo:lo+256]@A[:,lo:lo+256].T
        return out
    def stats(scores,ref):
        q=ref-np.logaddexp.reduce(ref,axis=1)[:,None];p=scores-np.logaddexp.reduce(scores,axis=1)[:,None]
        loss=np.sum(np.exp(q)*(q-p),axis=1)
        return loss,dict(mean_KL=math.fsum(loss.tolist())/len(loss),worst_KL=float(loss.max()),disagreement_rate=float(np.mean(np.argmax(scores,axis=1)!=np.argmax(ref,axis=1))))
    try:
        check();get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np);cfg=json.loads((Path(b['source'])/'config.json').read_bytes())
        A=np.empty((65537,2048),'f8');gamma=get('model.final_layernorm.weight')
        for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close();stream=None;H,labels=selected(b['FIT']);assert labels==r['FIT_labels']
        refpack=load(r['FIT_reference']);assert np.array_equal(H,refpack['H']);ref=refpack['scores'];rn=np.sqrt(np.mean(H**2,axis=1)+b['epsilon'])
        for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);compare(partial_head(H/rn[:,None],A[lo:hi]),ref[:,lo:hi])
        theta=load(r['initial_parameters']);basis=load(b['basis']);assert np.array_equal(theta['V'],basis['V']) and np.array_equal(theta['W'],basis['W']) and np.count_nonzero(theta['mu'])==0;del basis
        def point(params,item,summary,H=H,ref=ref):
            nonlocal points
            guard();packet=load(item);z=np.zeros((len(H),256),'f8')
            for lo in range(0,2048,256):z+=(H[:,lo:lo+256]-params['mu'][lo:lo+256])@params['W'][lo:lo+256]
            decoded=params['mu']+z@params['V'].T;rad=np.sqrt(np.sum(decoded*decoded,axis=1)/2048+b['epsilon'])
            compare(packet['z'],z);compare(packet['decoded'],decoded);compare(packet['radius'],rad)
            for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);compare(partial_head(decoded/rad[:,None],A[lo:hi]),packet['scores'][:,lo:hi])
            loss,ss=stats(packet['scores'],ref);compare(packet['KL'],loss)
            for key in ss:scalar(ss[key],summary[key])
            points+=1;return packet
        baseline=point(theta,r['baseline']['file'],r['baseline']['summary']);Z,Y,R=baseline['z'],baseline['decoded'],baseline['radius']
        lp=baseline['scores']-np.logaddexp.reduce(baseline['scores'],axis=1)[:,None];lq=ref-np.logaddexp.reduce(ref,axis=1)[:,None]
        logit=(np.exp(lp)-np.exp(lq))/72;g=np.zeros((72,2048),'f8')
        for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);g+=logit[:,lo:hi]@A[lo:hi]
        gy=np.empty_like(g);GV=np.zeros_like(theta['V']);GW=np.zeros_like(theta['W']);GM=np.zeros_like(theta['mu']);identity=np.eye(2048)
        for i in range(72):
            guard();jac=identity/R[i]-np.outer(Y[i],Y[i])/(2048*R[i]**3);dy=jac@g[i];gy[i]=dy
            dz=theta['V'].T@dy;GV+=np.outer(dy,Z[i]);GW+=np.outer(H[i]-theta['mu'],dz);GM+=dy-theta['W']@dz
        grad=load(r['gradient']);compare(grad['V'],GV);compare(grad['W'],GW);compare(grad['mu'],GM)
        compare(grad['logit_gradient'],logit);compare(grad['normalized_gradient'],g);compare(grad['raw_gradient'],gy)
        scales={k:float(np.linalg.norm(theta[k])) for k in ['V','W']};scales['mu']=math.sqrt(float(np.mean(np.sum(H*H,axis=1))))
        direction=load(r['trust_direction'])
        for key in theta:
            norm=float(np.linalg.norm(grad[key]));scalar(norm,r['gradient_norms'][key]);scalar(scales[key],r['trust_scales'][key])
            expected=-grad[key]*scales[key]/norm if norm>b['gradient_norm_floor'] else np.zeros_like(grad[key]);assert np.array_equal(expected,direction[key])
        finiteflags=[]
        for f in r['finite']:
            frame=load(f['direction'])
            if f['name']=='joint_trust':
                for key in theta:assert np.array_equal(frame[key],direction[key])
            else:
                key,phase=f['name'].rsplit('_',1);d=np.sin(np.arange(theta[key].size,dtype='f8')*.017+float(phase)).reshape(theta[key].shape);d*=scales[key]/np.linalg.norm(d)
                for k in theta:assert np.array_equal(frame[k],d if k==key else np.zeros_like(theta[k]))
            eps=f['epsilon'];assert eps in b['finite_epsilon']
            analytic=math.fsum(float(np.sum(grad[key]*frame[key])) for key in theta);scalar(analytic,f['analytic'])
            minus=point({k:theta[k]-eps*frame[k] for k in theta},f['minus']['file'],f['minus']['summary']);plus=point({k:theta[k]+eps*frame[k] for k in theta},f['plus']['file'],f['plus']['summary'])
            # Replay the frozen stored producer summaries for derivative subtraction; their probability values independently checked above.
            finite=(f['plus']['summary']['mean_KL']-f['minus']['summary']['mean_KL'])/(2*eps);scalar(finite,f['finite']);delta=abs(finite-analytic)/max(1.,abs(analytic));scalar(delta,f['relative_absolute_delta']);flag=delta<=b['finite_derivative_relative_absolute'];assert flag==f['gate'];finiteflags.append(flag)
            print(json.dumps(dict(audited_direction=f['name'],epsilon=eps,seconds=time.monotonic()-START)),flush=True)
            del minus,plus
        assert len(finiteflags)==14
        for candidate in r['candidates']:
            params=load(candidate['parameters']);alpha=candidate['alpha'];assert alpha in b['alpha']
            for key in theta:assert np.array_equal(params[key],theta[key]+alpha*direction[key])
            point(params,candidate['file'],candidate['summary'])
        best=min(r['candidates'],key=lambda x:x['summary']['mean_KL']);improve=(r['baseline']['summary']['mean_KL']-best['summary']['mean_KL'])/max(r['baseline']['summary']['mean_KL'],1e-300)
        accepted=all(finiteflags) and improve>=b['FIT_relative_improvement_required'];assert accepted==r['accepted'];scalar(improve,r['FIT_relative_improvement'])
        selection=json.loads(Path(r['selection']['path']).read_bytes());assert extent(r['selection']['path'])==r['selection']
        chosen=best['parameters'] if accepted else r['initial_parameters'];assert chosen==r['selected_parameters']==selection['selected_parameters'] and selection['accepted']==accepted
        assert selection['selected_alpha']==(best['alpha'] if accepted else 0.) and selection['derivative_gates']==all(finiteflags);scalar(selection['FIT_relative_improvement'],improve)
        devH,devlabels=selected(b['DEV']);assert devlabels==r['DEV']['labels']==b['DEV_labels'];devref=np.fromfile(b['DEV_reference']['path'],'<f8').reshape(72,65537)
        dr=np.sqrt(np.mean(devH**2,axis=1)+b['epsilon'])
        for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);compare(partial_head(devH/dr[:,None],A[lo:hi]),devref[:,lo:hi])
        dev=point(load(chosen),r['DEV']['file'],r['DEV']['summary'],H=devH,ref=devref);assert np.array_equal(dev['H'],devH)
        assert r['DEV']['summary']['proxy_gates']=={k:r['DEV']['summary'][k]<=v for k,v in b['probability_proxy'].items()}
        assert points==34 and r['decision']==('VERIFIED_GRADIENT_FIT_DESCENT' if accepted else 'COMPLETED_NO_VERIFIED_FIT_DESCENT')
        check();guard();result=dict(schema='EXACT_CATEGORICAL_RESIDUAL_CORRECTION_STORED_AUDIT_V1',source=extent(__file__),freeze=a.freeze,
            report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],all_actual_source_FIT_DEV_references_and34_full_vocab_points_checked=True,
            all_full_RMS_Jacobians72_outer_product_gradients14_difference_controls_and4_parameter_updates_checked=True,
            FIT_only_selection_parameters_and_DEV_quality_checked=True,
            selection_order_scope='Actual before-DEV order follows bound source control flow and immutable selection receipt; stored artifacts alone cannot timestamp semantic reads.',
            max_numeric_relative=maxrel,max_scalar_delta=maxscalar,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',scientific_decision=r['decision'],
            seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            scope='Independent full RMS Jacobian/per-state outer-product gradients, source-width encoding/head sums, logaddexp probabilities and exact bit update audit; '
                'stored derivative subtraction reuses independently verified point means. No source model/refit/optimizer/GPU/native/T4 replay.')
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
