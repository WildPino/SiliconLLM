"""Independent source-map/probability proof and every native F32 bit."""
import argparse,ctypes,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes());t=json.loads(a.terminal.read_bytes());assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.report)
    b=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'];parent=b['parent'];assert not a.out.exists()
    prep=json.loads(a.reference.read_bytes());assert prep['exit_code']==0
    extra=[extent(a.report),extent(a.terminal),extent(a.reference),extent(__file__),extent(DOC/'GAUGED_NATIVE_STORED_AUDIT_PROTOCOL_20261012.md'),extent(DOC/'GAUGED_NATIVE_AUDIT_SCOPE_ERRATUM_20261012.md')]
    extra+=prep['inputs']+[prep[k] for k in ['dll','log','assembly']]
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));stream=None;mapping=None;maxrel=maxscalar=0.;scorewords=0;rows=0
    receipt=dict(schema='GAUGED_NATIVE_STORED_AUDIT_TERMINAL_V1',command=sys.orig_argv,source=extent(__file__),freeze=a.freeze,pid=proc.pid,creation_time=proc.create_time(),timing_control_admitted=False)
    def guard():assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    def check():
        for x in b['inputs']+r['outputs']+[r['binding']]+extra:guard();assert extent(x['path'])==x,x['path']
    def load(x):
        with np.load(x['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def compare(x,y,tol=1e-10):
        nonlocal maxrel
        rel=float(np.linalg.norm(x-y)/max(float(np.linalg.norm(y)),1e-300));maxrel=max(maxrel,rel);assert rel<=tol,rel
    def scalar(x,y,tol=1e-8):
        nonlocal maxscalar
        d=abs(float(x)-float(y));maxscalar=max(maxscalar,d);assert d<=tol*max(1.,abs(float(y))),(x,y)
    def partial(x,A):
        out=np.zeros((len(x),len(A)),'f8')
        for j in range(0,2048,256):out+=x[:,j:j+256]@A[:,j:j+256].T
        return out
    def logprob(x):return x-np.logaddexp.reduce(x,axis=1)[:,None]
    def measures(score,ref):
        q=logprob(ref);p=logprob(score);return np.column_stack([np.sum(np.exp(q)*(q-p),axis=1),np.argmax(score,axis=1)!=np.argmax(ref,axis=1),-np.sum(np.exp(q)*q,axis=1),-np.sum(np.exp(p)*p,axis=1)])
    def quality(values,cases,expected):
        cc=[]
        for c in cases:
            v=values[c['lo']:c['hi']];cc.append(dict(id=c['id'],domain=c['domain'],labels=len(v),KL=math.fsum(v[:,0].tolist())/len(v),disagreement_rate=float(v[:,1].mean()),worst_KL=float(v[:,0].max())))
        ss=dict(case_mean_KL=math.fsum(c['KL'] for c in cc)/24,case_mean_disagreement=math.fsum(c['disagreement_rate'] for c in cc)/24,label_mean_KL=math.fsum(values[:,0].tolist())/len(values),label_disagreement=float(values[:,1].mean()),worst_label_KL=float(values[:,0].max()),worst_case_KL=max(c['KL'] for c in cc))
        for key in ss:scalar(ss[key],expected['summary'][key])
        for calc,stored in zip(cc,expected['cases']):
            assert all(calc[k]==stored[k] for k in ['id','domain','labels'])
            for k in ['KL','disagreement_rate','worst_KL']:scalar(calc[k],stored[k])
        domains=[]
        for d in sorted(set(c['domain'] for c in cc)):
            entries=[c for c in cc if c['domain']==d];domains.append(dict(domain=d,KL=math.fsum(c['KL'] for c in entries)/2,disagreement_rate=math.fsum(c['disagreement_rate'] for c in entries)/2))
        for calc,stored in zip(domains,expected['domains']):assert calc['domain']==stored['domain'];scalar(calc['KL'],stored['KL']);scalar(calc['disagreement_rate'],stored['disagreement_rate'])
        lim=b['quality'];flags=dict(case_mean_KL=ss['case_mean_KL']<=lim['case_mean_KL'],case_mean_disagreement=ss['case_mean_disagreement']<=lim['case_mean_disagreement'],worst_label_KL=ss['worst_label_KL']<=lim['worst_label_KL'],every_case_KL=all(c['KL']<=lim['every_case_KL'] for c in cc),every_case_disagreement=all(c['disagreement_rate']<=lim['every_case_disagreement'] for c in cc),domains=all(c['KL']<=lim['every_case_KL'] and c['disagreement_rate']<=lim['every_case_disagreement'] for c in domains))
        assert flags==expected['quality_gates']
    try:
        check();params=load(parent['selected']['parameters']);V,W,mu=params['V'],params['W'],params['mu'];gauge=load(r['gauge']);Q,R,E=gauge['Q'],gauge['R'],gauge['E'];perp=gauge['center_perpendicular'];c=gauge['center_in_code'];codebias=gauge['code_bias'];epsilon=float(gauge['epsilon'])
        compare(Q.T@Q,np.eye(256));compare(Q@R,V);compare(E,W@R.T);compare(c,Q.T@mu);compare(perp,mu-Q@c);compare(codebias,c-E.T@mu)
        scalar(epsilon,(perp@perp+2048*b['epsilon_source'])/256);scalar(epsilon,r['epsilon']);scalar(np.max(np.abs(E.T@Q-np.eye(256))),r['source_reencoding_identity_defect'])
        scalar(np.max(np.abs(Q.T@Q-np.eye(256))),r['orthogonal_max']);scalar(np.linalg.norm(Q@R-V)/np.linalg.norm(V),r['QR_reconstruction_relative']);scalar(np.linalg.norm(Q.T@perp)/max(1.,float(np.linalg.norm(perp))),r['perpendicular_defect_relative'])
        h64=load(r['head_F64']);C,beta=h64['head'],h64['bias'];C32=np.fromfile(r['head_F32']['path'],'<f4').reshape(65537,256);beta32=np.fromfile(r['bias_F32']['path'],'<f4');eps32=np.fromfile(r['epsilon_F32']['path'],'<f4')
        assert np.array_equal(C32.view('u4'),C.astype('f4').view('u4')) and np.array_equal(beta32.view('u4'),beta.astype('f4').view('u4'))
        assert eps32.shape==(1,) and eps32.view('u4')[0]==np.array([epsilon],dtype='f4').view('u4')[0];scalar(float(eps32[0]),r['epsilon_f32'])
        get,stream=tensor_reader(Path(b['source'])/'model.safetensors',np);cfg=json.loads((Path(b['source'])/'config.json').read_bytes());gamma=get('model.final_layernorm.weight');A=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):
            guard();hi=min(65537,lo+1024);A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier'];compare(C[lo:hi],math.sqrt(8)*partial(Q.T,A[lo:hi]).T);compare(beta[lo:hi],math.sqrt(8)*partial(perp[None,:],A[lo:hi])[0])
        stream.close();stream=None;dll=ctypes.CDLL(prep['dll']['path']);fn=dll.reference_readout;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_float,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];fn.restype=ctypes.c_int
        totalcalls=0;splitchecks=[]
        for split in r['splits']:
            name=split['split'];dataset=parent[name+'_dataset'];point=parent['selected'] if name=='FIT' else parent['DEV_selected'];cases=parent[name+'_cases'];data=load(dataset);H=data['H'];old=load(point['file']);pack=load(split['codes'])
            Y=mu+old['z']@V.T;t=old['z']@R.T+c;compare(pack['t'],t);compare(t,H@E+codebias);assert np.array_equal(pack['t_f32'].view('u4'),t.astype('f4').view('u4'));t32=pack['t_f32']
            decoded=t@Q.T+perp;compare(decoded,Y);radius=np.sqrt(np.mean(t*t,axis=1)+epsilon);sr=np.sqrt(np.mean(Y*Y,axis=1)+b['epsilon_source']);compare(radius/math.sqrt(8),sr)
            scalar(np.linalg.norm(decoded-Y)/np.linalg.norm(Y),split['decoded_vector_relative']);scalar(np.linalg.norm(t-(H@E+codebias))/np.linalg.norm(t),split['code_relation_relative']);scalar(np.max(np.abs(radius/math.sqrt(8)-sr)/sr),split['radius_relative'])
            mapping=np.memmap(split['scores']['path'],mode='r',dtype='<f4',shape=(len(H),65537));nv_values=np.empty((len(H),4),'f8');gv_values=np.empty_like(nv_values);pres_values=np.empty_like(nv_values);calc_batches=[]
            for lo in range(0,len(H),64):
                guard();hi=min(lo+64,len(H));n=hi-lo;refnorm=np.empty((n,256),'<f4');refinv=np.empty(n,'<f4');refout=np.empty((n,65537),'<f4')
                rc=fn(C32.ctypes.data,beta32.ctypes.data,t32[lo:hi].ctypes.data,n,float(eps32[0]),refnorm.ctypes.data,refinv.ctypes.data,refout.ctypes.data);assert rc==0
                assert np.array_equal(refnorm.view('u4'),pack['normalized_f32'][lo:hi].view('u4')) and np.array_equal(refinv.view('u4'),pack['inverse_f32'][lo:hi].view('u4'))
                assert np.array_equal(refout.view('u4'),mapping[lo:hi].view('u4')),'native score bits differ from independent scalar reference'
                scorewords+=n*65537;rows+=n;totalcalls+=1
                original=partial(Y[lo:hi]/old['radius'][lo:hi,None],A);teacher=partial(H[lo:hi]/np.sqrt(np.mean(H[lo:hi]**2,axis=1,keepdims=True)+b['epsilon_source']),A)
                gauged=(t[lo:hi]@C.T+beta)/radius[lo:hi,None];native=refout.astype('f8');nv_values[lo:hi]=measures(native,teacher);gv_values[lo:hi]=measures(gauged,teacher);pres_values[lo:hi]=measures(native,original)
                compare(pack['native_metrics'][lo:hi],nv_values[lo:hi],1e-9);compare(pack['gauge_metrics'][lo:hi],gv_values[lo:hi],1e-9);compare(pack['preservation_metrics'][lo:hi],pres_values[lo:hi],1e-9)
                # Preservation KL is tiny;compare absolute KL separately to avoid dividing by roundoff.
                assert np.max(np.abs(pack['preservation_metrics'][lo:hi,0]-pres_values[lo:hi,0]))<=1e-10
                normref=t[lo:hi]/radius[lo:hi,None]
                batch=dict(lo=lo,hi=hi,gauge_score_sup_relative=float(np.max(np.abs(gauged-original))/max(1.,float(np.max(np.abs(original))))),gauge_source_KL_delta=float(np.max(np.abs(gv_values[lo:hi,0]-old['metrics'][lo:hi,0]))),gauge_argmax_changes=int(np.count_nonzero(np.argmax(gauged,axis=1)!=np.argmax(original,axis=1))),
                    native_norm_relative=float(np.linalg.norm(refnorm.astype('f8')-normref)/np.linalg.norm(normref)),native_inverse_relative=float(np.max(np.abs(refinv.astype('f8')*radius[lo:hi]-1))),native_score_absolute=float(np.max(np.abs(native-original))),native_argmax_changes=int(np.count_nonzero(np.argmax(native,axis=1)!=np.argmax(original,axis=1))))
                stored=next(x for x in split['batches'] if x['lo']==lo)
                for k in batch:scalar(batch[k],stored[k])
                calc_batches.append(batch)
            mapping._mmap.close();mapping=None
            quality(nv_values,cases,split['quality']);mean=float(np.mean(pres_values[:,0]));scalar(mean,split['preservation_mean_KL']);lim=b['numeric']
            flags=dict(orthogonal_max=r['orthogonal_max']<=lim['orthogonal_max'],gauge_vector_relative=max(split['decoded_vector_relative'],split['code_relation_relative'])<=lim['gauge_vector_relative'],gauge_radius_relative=split['radius_relative']<=lim['gauge_radius_relative'],
                gauge_score_sup_relative=max(x['gauge_score_sup_relative'] for x in calc_batches)<=lim['gauge_score_sup_relative'],gauge_source_KL_delta=max(x['gauge_source_KL_delta'] for x in calc_batches)<=lim['gauge_source_KL_delta'],gauge_argmax_changes=sum(x['gauge_argmax_changes'] for x in calc_batches)<=lim['gauge_argmax_changes'],
                native_norm_relative=max(x['native_norm_relative'] for x in calc_batches)<=lim['native_norm_relative'],native_inverse_relative=max(x['native_inverse_relative'] for x in calc_batches)<=lim['native_inverse_relative'],native_score_absolute=max(x['native_score_absolute'] for x in calc_batches)<=lim['native_score_absolute'],native_to_factor_mean_KL=mean<=lim['native_to_factor_mean_KL'],native_argmax_changes=sum(x['native_argmax_changes'] for x in calc_batches)<=lim['native_argmax_changes'])
            assert flags==split['numeric_gates'];splitchecks.append(dict(split=name,rows=len(H),all_native_bits_exact=True,numeric_gates=flags))
            print(json.dumps(dict(audited_split=name,all_bits_exact=True,seconds=time.monotonic()-START)),flush=True)
        assert rows==8808 and scorewords==577249896 and totalcalls==r['native_calls']==139
        expected='GAUGE_AND_NATIVE_PRECISION_PASS' if all(all(x['numeric_gates'].values()) for x in r['splits']) else 'COMPLETED_GAUGE_OR_NATIVE_PRECISION_FAILURE';assert expected==r['decision'];check();guard()
        result=dict(schema='GAUGED_NATIVE_STORED_AUDIT_V1',source=extent(__file__),freeze=a.freeze,report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],reference_preparation=extent(a.reference),
            all_gauge_source_coefficients_F32_casts_and8808_source_code_probability_rows_checked=True,
            all577249896_native_scores_and2254848_normalized_state_words8808_inverse_words_bit_exact=True,
            independent_scalar_reference_calls=totalcalls,split_checks=splitchecks,max_numeric_relative=maxrel,max_scalar_delta=maxscalar,before_after_hashes_exact=True,
            decision='INDEPENDENT_STORED_PASS',scientific_decision=expected,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,
            scope='Independent scalar explicit FMA native output replay,actual source coefficients and direct2048-feature reference/old predictor probabilities. No producer DLL replay/QR refit/source model/GPU/T4. '
                'Preserves poor predictor;not whole-engine/source causality/quality/n/50 admission.')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if stream is not None:stream.close()
        if mapping is not None:mapping._mmap.close()
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['report','terminal','reference','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['report-sha','terminal-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
