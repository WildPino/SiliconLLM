"""Full smaller-scale point/selection audit, reusing audited parent gradients."""
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
    b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'];pr=b['parent'];pb=b['parent_binding'];assert not pr['accepted'] and not a.out.exists()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;maxrel=maxscalar=0.
    extra=[extent(a.report),extent(a.terminal),extent(__file__),extent(DOC/'EXACT_CATEGORICAL_RESIDUAL_REFINEMENT_STORED_AUDIT_PROTOCOL_20261012.md')]
    receipt=dict(schema='EXACT_CATEGORICAL_RESIDUAL_REFINEMENT_AUDIT_TERMINAL_V1',command=sys.orig_argv,source=extent(__file__),freeze=a.freeze,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():assert time.monotonic()-START<=300 and proc.memory_info().peak_wset<=4<<30 and not proc.children(recursive=True)
    def check():
        for x in b['inputs']+r['outputs']+[r['binding']]+extra:guard();assert extent(x['path'])==x
    def load(x):
        with np.load(x['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def compare(x,y):
        nonlocal maxrel
        rel=float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-300));maxrel=max(maxrel,rel);assert rel<=1e-10
    def scalar(x,y):
        nonlocal maxscalar
        d=abs(float(x)-float(y));maxscalar=max(maxscalar,d);assert d<=1e-8*max(1.,abs(float(y)))
    try:
        check();ad=json.loads(Path(b['parent_audit']['path']).read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['all_full_RMS_Jacobians72_outer_product_gradients14_difference_controls_and4_parameter_updates_checked']
        get,stream=tensor_reader(Path(pb['source'])/'model.safetensors',np);cfg=json.loads((Path(pb['source'])/'config.json').read_bytes());gamma=get('model.final_layernorm.weight');A=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):guard();hi=min(65537,lo+1024);A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close();stream=None;fit=load(pr['FIT_reference']);H,ref=fit['H'],fit['scores'];theta=load(pr['initial_parameters']);grad=load(pr['gradient'])
        def point(params,item,ss,H=H,ref=ref):
            pack=load(item);z=np.zeros((len(H),256),'f8')
            for lo in range(0,2048,256):z+=(H[:,lo:lo+256]-params['mu'][lo:lo+256])@params['W'][lo:lo+256]
            y=params['mu']+z@params['V'].T;rad=np.sqrt(np.sum(y*y,axis=1)/2048+pb['epsilon']);compare(pack['z'],z);compare(pack['decoded'],y);compare(pack['radius'],rad)
            for lo in range(0,65537,1024):
                guard();hi=min(65537,lo+1024);pred=np.zeros((len(H),hi-lo),'f8')
                for j in range(0,2048,256):pred+=(y[:,j:j+256]/rad[:,None])@A[lo:hi,j:j+256].T
                compare(pred,pack['scores'][:,lo:hi])
            q=ref-np.logaddexp.reduce(ref,axis=1)[:,None];p=pack['scores']-np.logaddexp.reduce(pack['scores'],axis=1)[:,None];loss=np.sum(np.exp(q)*(q-p),axis=1);compare(loss,pack['KL'])
            vals=dict(mean_KL=math.fsum(loss.tolist())/len(loss),worst_KL=float(loss.max()),disagreement_rate=float(np.mean(np.argmax(pack['scores'],axis=1)!=np.argmax(ref,axis=1))))
            for key,value in vals.items():scalar(value,ss[key])
            return pack
        flags=[]
        for f in r['rows']:
            frame=load(f['direction']);assert f['epsilon']==b['epsilon'];analytic=math.fsum(float(np.sum(grad[k]*frame[k])) for k in theta);scalar(analytic,f['analytic'])
            point({k:theta[k]-b['epsilon']*frame[k] for k in theta},f['minus']['file'],f['minus']['summary']);point({k:theta[k]+b['epsilon']*frame[k] for k in theta},f['plus']['file'],f['plus']['summary'])
            finite=(f['plus']['summary']['mean_KL']-f['minus']['summary']['mean_KL'])/(2*b['epsilon']);scalar(finite,f['finite']);delta=abs(finite-analytic)/max(1.,abs(analytic));scalar(delta,f['delta']);assert f['gate']==(delta<=b['tolerance'])
            fine=next(x for x in pr['finite'] if x['name']==f['name'] and x['epsilon']==2e-6);coarse=next(x for x in pr['finite'] if x['name']==f['name'] and x['epsilon']==1e-5);assert f['prior_fine_gate']==fine['gate']
            rich=(25*fine['finite']-coarse['finite'])/24;scalar(rich,f['Richardson_two_prior_scales']);scalar(abs(rich-analytic)/max(1.,abs(analytic)),f['Richardson_relative_absolute_delta']);flags.append(f['gate'] and fine['gate'])
            print(json.dumps(dict(audited=f['name'],seconds=time.monotonic()-START)),flush=True)
        assert len(flags)==7;best=min(pr['candidates'],key=lambda x:x['summary']['mean_KL']);accepted=all(flags) and pr['FIT_relative_improvement']>=pb['FIT_relative_improvement_required'];assert accepted==r['refined_acceptance'] and r['parent_acceptance_stays_false']
        chosen=best['parameters'] if accepted else pr['initial_parameters'];assert chosen==r['selected_parameters'] and r['selected_alpha']==(best['alpha'] if accepted else 0.)
        assert r['FIT_before']==pr['baseline']['summary'] and r['FIT_selected']==(best['summary'] if accepted else pr['baseline']['summary'])
        selection=json.loads(Path(r['selection']['path']).read_bytes());assert extent(r['selection']['path'])==r['selection'];assert selection['parameters']==chosen and selection['refined_acceptance']==accepted and selection['alpha']==r['selected_alpha']
        assert selection['parent_acceptance_unchanged']==pr['accepted'] # This field stores the unchanged parent VALUE(false), not a flag saying it changed.
        dev=load(r['DEV']['file']);devH=[];labels=[]
        for rec in pb['DEV']:
            h=(np.fromfile(rec['h']['path'],'<u2').astype('<u4')<<16).view('<f4').reshape(rec['history'],2048).astype('f8');devH.append(h[rec['positions']]);labels.extend(dict(id=rec['id'],label_index=i,position=p) for i,p in zip(rec['label_indices'],rec['positions']))
        devH=np.concatenate(devH);assert np.array_equal(devH,dev['H']) and labels==r['DEV']['labels']==pb['DEV_labels'];devref=np.fromfile(pb['DEV_reference']['path'],'<f8').reshape(72,65537);params=load(chosen)
        point(params,r['DEV']['file'],r['DEV']['summary'],H=devH,ref=devref);assert r['DEV']['summary']['proxy_gates']=={k:r['DEV']['summary'][k]<=v for k,v in pb['probability_proxy'].items()}
        scalar(np.max(np.abs(params['W'].T@params['V']-np.eye(256))),r['selected_WtV_identity_defect'])
        assert r['decision']==('REFINED_DERIVATIVE_VERIFIED_FIT_DESCENT' if accepted else 'COMPLETED_REFINED_NONACCEPTANCE');check();guard()
        result=dict(schema='EXACT_CATEGORICAL_RESIDUAL_REFINEMENT_STORED_AUDIT_V1',source=extent(__file__),freeze=a.freeze,report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],
            all14_refined_FIT_and_selected_DEV_full_vocab_packets7_derivatives_and_selection_checked=True,
            parent_nonacceptance_and_all_original_inputs_outputs_unchanged=True,max_numeric_relative=maxrel,max_scalar_delta=maxscalar,before_after_hashes_exact=True,
            decision='INDEPENDENT_STORED_PASS',scientific_decision=r['decision'],seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            scope='Reuse fully audited parent Jacobian/gradient/candidates; source-width point encoding/head, independent logaddexp probabilities, frozen derivative subtraction and FIT seal/DEV audit. No refit/source model/GPU/native/T4.')
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
