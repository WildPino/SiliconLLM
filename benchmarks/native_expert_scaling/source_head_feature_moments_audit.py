"""All source feature moments and different decoder derivatives, CPU only."""
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
    assert extent(r['binding']['path'])==r['binding']; b=json.loads(Path(r['binding']['path']).read_bytes()); proc=psutil.Process(); proc.cpu_affinity(list(range(6))); stream=None; maxrel=maxscalar=0.
    extra=[extent(__file__),extent(DOC/'SOURCE_HEAD_FEATURE_MOMENTS_AUDIT_PROTOCOL_20261012.md'),extent(a.report),extent(a.terminal),r['binding']]
    receipt=dict(schema='SOURCE_HEAD_FEATURE_MOMENTS_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():assert time.monotonic()-START<=1200 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+r['outputs']+extra:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def one(e,key):
        with np.load(e['path'],allow_pickle=False) as z:return z[key].copy()
    def compare(x,y,tolerance=1e-9):
        nonlocal maxrel
        error=float(np.linalg.norm(x-y)/max(1.,float(np.linalg.norm(y)))); maxrel=max(maxrel,error); assert error<=tolerance,error
    def scalar(x,y,tolerance=1e-8):
        nonlocal maxscalar
        error=abs(float(x)-float(y)); maxscalar=max(maxscalar,error); assert error<=tolerance*max(1.,abs(float(y))),(x,y)
    def partial(x,A):
        score=np.zeros((len(x),65537),'f8')
        for col in range(0,2048,256):score+=x[:,col:col+256]@A[:,col:col+256].T
        return score
    def logz(score):
        value=np.full(len(score),-np.inf,'f8')
        for lo in range(0,65537,4096):
            x=score[:,lo:lo+4096]; center=x.max(axis=1); value=np.logaddexp(value,center+np.log(np.exp(x-center[:,None]).sum(axis=1)))
        return value
    def expectation(prob,A):
        result=np.zeros((len(prob),2048),'f8')
        for lo in range(0,65537,4096):result+=prob[:,lo:lo+4096]@A[lo:lo+4096]
        return result
    try:
        check(); parent=b['parent']; broad=b['broad']; get,stream=tensor_reader(Path(parent['prior']['source'])/'model.safetensors',np); cfg=json.loads((Path(parent['prior']['source'])/'config.json').read_bytes()); gamma=get('model.final_layernorm.weight'); A=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):
            guard(); hi=min(lo+1024,65537); A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close(); stream=None; params=[load(broad['baseline']['parameters']),load(broad['selected']['parameters'])]; gauge=load(parent['native']['gauge']); best=one(b['oracle']['final_checkpoint'],'best_t_f32'); offset=0; control_rows=0; checks=[]
        for s in r['splits']:
            name=s['split']; data=load(broad[name+'_dataset']); old=load(next(x for x in parent['preflight']['splits'] if x['split']==name)['file']); compiled=load(s['moments']); n=len(data['H'])
            zpoints=[one(broad['baseline']['file'] if name=='FIT' else broad['DEV_baseline']['file'],'z'),one(broad['selected']['file'] if name=='FIT' else broad['DEV_selected']['file'],'z')]; controls=[load(c['file']) for c in s['controls']]
            assert compiled['moment'].shape==(n,2048) and compiled['moment'].dtype==np.dtype('f8'); compare(compiled['weights'],data['weights'],0.)
            for lo in range(0,n,64):
                guard(); hi=min(lo+64,n); x=data['H'][lo:hi]; radius=np.sqrt(np.sum(x*x,axis=1)/2048+b['epsilon_source']); source=partial(x/radius[:,None],A); qlog=source-logz(source)[:,None]; q=np.exp(qlog)
                m=expectation(q,A); entropy=-np.sum(q*qlog,axis=1); compare(compiled['moment'][lo:hi],m); compare(compiled['entropy'][lo:hi],entropy); compare(entropy,old['entropy'][lo:hi]); compare(compiled['qmass'][lo:hi],q.sum(axis=1))
                assert np.max(np.abs(q.sum(axis=1)-1))<=1e-11 and np.array_equal(compiled['source_ids'][lo:hi],np.argmax(source,axis=1))
                assert np.max(compiled['self_error'][lo:hi])<=b['numeric']['self_absolute']; self_loss=logz(source)-np.sum(m*x/radius[:,None],axis=1)-entropy; assert np.max(np.abs(self_loss))<=b['numeric']['self_absolute']
                ys=[p['mu']+zpoints[k][lo:hi]@p['V'].T for k,p in enumerate(params)]; ys.append(best[offset+lo:offset+hi].astype('f8')@gauge['Q'].T+gauge['center_perpendicular'])
                for ci,y in enumerate(ys):
                    rsource=np.sqrt(np.sum(y*y,axis=1)/2048+b['epsilon_source']); score=partial(y/rsource[:,None],A); plog=score-logz(score)[:,None]; prob=np.exp(plog)
                    direct=np.sum(q*(qlog-plog),axis=1); linear=np.zeros(hi-lo,'f8')
                    for col in range(0,2048,128):linear+=np.sum(m[:,col:col+128]*(y[:,col:col+128]/rsource[:,None]),axis=1)
                    loss=logz(score)-linear-entropy; grad=expectation(prob-q,A); compare(controls[ci]['gradient_u'][lo:hi],grad)
                    # Full RMS Jacobian action through independent radial/tangent eigenspaces.
                    ynorm=np.sqrt(np.einsum('ij,ij->i',y,y)); unit=y/np.maximum(ynorm,1e-300)[:,None]; radial_dot=np.einsum('ij,ij->i',unit,grad); radial=unit*radial_dot[:,None]; tangent=grad-radial
                    grad_y=tangent/rsource[:,None]+radial*(b['epsilon_source']/rsource**3)[:,None]; compare(controls[ci]['gradient_y'][lo:hi],grad_y)
                    values=np.column_stack([loss,direct,np.argmax(score,axis=1)!=np.argmax(source,axis=1),-np.sum(prob*plog,axis=1),rsource])
                    compare(controls[ci]['values'][lo:hi],values); compare(controls[ci]['radius'][lo:hi],rsource); control_rows+=hi-lo
            scalar(np.max(np.abs(compiled['entropy']-old['entropy'])),s['source_entropy_delta']); scalar(compiled['self_error'].max(),s['maximum_self_error'])
            for ci,c in enumerate(s['controls']):
                assert c['name']==b['controls'][ci]; scalar(data['weights']@controls[ci]['values'][:,1],c['case_mean_KL']); scalar(data['weights']@controls[ci]['values'][:,2],c['case_disagreement'])
                recorded=c['comparison']; flags=dict(loss=recorded['loss_absolute']<=b['numeric']['loss_absolute'],gradient_absolute=recorded['gradient_absolute']<=b['numeric']['feature_gradient_absolute'],gradient_relative=recorded['gradient_relative']<=b['numeric']['feature_gradient_relative']); assert flags==c['flags']
            offset+=n; checks.append(dict(split=name,rows=n,all_source_moments_and3_decoder_gradients_verified=True)); print(json.dumps(dict(audited=name,seconds=time.perf_counter()-CLOCK)),flush=True)
        assert offset==8808 and control_rows==26424 and r['source_moment_payload_bytes']==144310272 and not r['optimization_run'] and not r['native_quantized_export_admitted'] and not r['timing_admitted']
        expected='SOURCE_FEATURE_MOMENTS_AND_CONTROLS_PASS_PENDING_AUDIT' if all(all(c['flags'].values()) for s in r['splits'] for c in s['controls']) else 'COMPLETED_SOURCE_FEATURE_CONTROL_FAILURE'; assert r['decision']==expected; check(); guard()
        result=dict(schema='SOURCE_HEAD_FEATURE_MOMENTS_AUDIT_V1',source=extent(__file__),freeze=a.freeze,report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],source_rows=offset,control_rows=control_rows,
                    all18038784_source_feature_moment_words_verified=True,all54116352_feature_and54116352_normalized_decoder_gradient_words_verified=True,
                    full_radial_tangent_RMS_Jacobian_actions_verified=True,split_checks=checks,max_relative=maxrel,max_scalar_delta=maxscalar,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',
                    seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='Full changing-decoder source-head loss/moment/feature/RMS derivative qualification. GPU internal comparison maxima are bound observations,not exact GPU replay;no causal/native/model/n/50 admission.')
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset); write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
