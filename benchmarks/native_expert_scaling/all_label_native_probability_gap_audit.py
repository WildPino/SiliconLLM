"""All stored mathematical points plus full native-bit/source-probability audit."""
import argparse,ctypes,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
START=time.monotonic(); CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    assert extent(a.report)['sha256']==a.report_sha and extent(a.terminal)['sha256']==a.terminal_sha
    r=json.loads(a.report.read_bytes()); terminal=json.loads(a.terminal.read_bytes()); assert terminal['exit_code']==0 and terminal['result']==extent(a.report)
    assert extent(r['binding']['path'])==r['binding']; b=json.loads(Path(r['binding']['path']).read_bytes())
    proc=psutil.Process(); proc.cpu_affinity(list(range(6))); mapping=stream=None; maxrel=maxscalar=0.
    extra=[extent(__file__),extent(DOC/'ALL_LABEL_NATIVE_PROBABILITY_GAP_AUDIT_PROTOCOL_20261012.md'),extent(a.report),extent(a.terminal),r['binding']]
    receipt=dict(schema='ALL_LABEL_NATIVE_PROBABILITY_GAP_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():assert time.monotonic()-START<=1200 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+r['outputs']+extra:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def compare(x,y,tolerance=1e-9):
        nonlocal maxrel
        error=float(np.linalg.norm(x-y)/max(1.,float(np.linalg.norm(y)))); maxrel=max(maxrel,error); assert error<=tolerance,error
    def scalar(x,y,tolerance=1e-8):
        nonlocal maxscalar
        error=abs(float(x)-float(y)); maxscalar=max(maxscalar,error); assert error<=tolerance*max(1.,abs(float(y))),(x,y)
    def logz(score):
        # Hierarchical full-vocabulary log partition,not CUDA reduction.
        result=np.full(len(score),-np.inf,'f8')
        for lo in range(0,score.shape[1],4096):
            block=score[:,lo:lo+4096]; center=block.max(axis=1); value=center+np.log(np.exp(block-center[:,None]).sum(axis=1)); result=np.logaddexp(result,value)
        return result
    def partial(x,A,width):
        out=np.zeros((len(x),len(A)),'f8')
        for col in range(0,x.shape[1],width):out+=x[:,col:col+width]@A[:,col:col+width].T
        return out
    def point(v,mm,hh,aug,C,beta,scale,gradient=True):
        score=partial(v[:,:256],C,32)+v[:,-1,None]*(beta/scale); z=logz(score); linear=np.zeros(len(v),'f8')
        for col in range(0,257,8):linear+=np.sum(mm[:,col:col+8]*v[:,col:col+8],axis=1)
        loss=z-linear-hh
        if not gradient:return loss
        probability=np.exp(score-z[:,None]); compare(probability.sum(axis=1),np.ones(len(v)),1e-11); g=-mm.copy()
        for row in range(0,65537,4096):g+=probability[:,row:row+4096]@aug[row:row+4096]
        return loss,g
    def quality(values,cases,expected):
        cc=[]
        for c in cases:
            x=values[c['lo']:c['hi']]; kl=math.fsum(x[:,0])/len(x); dis=float(x[:,1].mean()); worst=float(x[:,0].max())
            old=next(t for t in expected['cases'] if t['id']==c['id']); scalar(kl,old['KL']); scalar(dis,old['disagreement_rate']); scalar(worst,old['worst_KL']); cc.append((c,kl,dis,worst))
        calc=dict(case_mean_KL=math.fsum(x[1] for x in cc)/24,case_mean_disagreement=math.fsum(x[2] for x in cc)/24,label_mean_KL=math.fsum(values[:,0])/len(values),
                  label_disagreement=float(values[:,1].mean()),worst_label_KL=float(values[:,0].max()),worst_case_KL=max(x[1] for x in cc))
        for k in calc:scalar(calc[k],expected['summary'][k])
        lim=b['quality']; flags=dict(case_mean_KL=calc['case_mean_KL']<=lim['case_mean_KL'],case_mean_disagreement=calc['case_mean_disagreement']<=lim['case_mean_disagreement'],
          worst_label_KL=calc['worst_label_KL']<=lim['worst_label_KL'],every_case_KL=all(x[1]<=lim['every_case_KL'] for x in cc),every_case_disagreement=all(x[2]<=lim['every_case_disagreement'] for x in cc))
        domains=[]
        for domain in expected['domains']:
            selected=[x for x in cc if x[0]['domain']==domain['domain']]; assert len(selected)==2
            kl=sum(x[1] for x in selected)/2; dis=sum(x[2] for x in selected)/2; scalar(kl,domain['KL']); scalar(dis,domain['disagreement_rate']); domains.append((kl,dis))
        flags['domains']=all(kl<=lim['every_case_KL'] and dis<=lim['every_case_disagreement'] for kl,dis in domains); assert flags==expected['quality_gates']
    try:
        check(); nr=b['native']; C32=np.fromfile(nr['head_F32']['path'],'<f4').reshape(65537,256); beta32=np.fromfile(nr['bias_F32']['path'],'<f4'); C=C32.astype('f8'); beta=beta32.astype('f8')
        epsilon=float(np.fromfile(nr['epsilon_F32']['path'],'<f4')[0]); scale=math.sqrt(256*epsilon); aug=np.column_stack([C,beta/scale])
        originals=[load(s['file']) for s in b['preflight']['splits']]; m=np.concatenate([p['moments'] for p in originals]); H=np.concatenate([p['entropy'] for p in originals]); w=np.concatenate([p['weights']/2 for p in originals])
        warm=np.concatenate([load(s['codes'])['t_f32'] for s in nr['splits']]); warm_loss=np.concatenate([p['loss'] for p in originals]); previous=None; checked_points=0; same_lower_rows=0
        for cp,trace in zip(r['checkpoints'],r['trace']):
            p=load(cp['file']); assert p['v'].shape==p['lower_v'].shape==p['gradient'].shape==p['lower_gradient'].shape==(8808,257)
            assert p['best_t_f32'].dtype==np.dtype('f4') and np.all(np.isfinite(p['best_t_f32'])) and np.all(p['steps']>=b['solver']['L_floor'])
            assert np.all(p['v'][:,-1]>=0) and np.max(np.sum(p['v']**2,axis=1))<=256+1e-9 and np.all(p['lower_v'][:,-1]>=0) and np.max(np.sum(p['lower_v']**2,axis=1))<=256+1e-9
            assert p['updates'].min()==trace['min_updates'] and p['updates'].max()==trace['max_updates']; assert np.all(p['updates']<=128)
            if previous is not None:
                assert np.max(p['loss']-previous['loss'])<=1e-8 and np.max(p['best_real_loss']-previous['best_real_loss'])<=1e-8
                assert np.min(p['raw_lower']-previous['raw_lower'])>=-1e-8 and np.all(p['updates']>=previous['updates'])
            else:
                assert np.array_equal(p['best_t_f32'].view('u4'),warm.view('u4')); compare(p['loss'],warm_loss); assert not np.any(p['updates'])
            for lo in range(0,8808,64):
                guard(); hi=min(lo+64,8808); loss,g=point(p['v'][lo:hi],m[lo:hi],H[lo:hi],aug,C,beta,scale)
                compare(p['loss'][lo:hi],loss); compare(p['gradient'][lo:hi],g)
                equal=np.all(p['v'][lo:hi]==p['lower_v'][lo:hi],axis=1); lower_loss=loss.copy(); lower_g=g.copy(); same_lower_rows+=int(equal.sum())
                if not equal.all():
                    ll,gg=point(p['lower_v'][lo:hi][~equal],m[lo:hi][~equal],H[lo:hi][~equal],aug,C,beta,scale); lower_loss[~equal]=ll; lower_g[~equal]=gg
                compare(p['lower_loss'][lo:hi],lower_loss); compare(p['lower_gradient'][lo:hi],lower_g)
                # Construct support minimizer explicitly from independent gradient.
                direction=-lower_g.copy(); direction[:,-1]=np.maximum(direction[:,-1],0.); norm=np.sqrt(np.sum(direction*direction,axis=1)); endpoint=direction*(16/np.maximum(norm,1e-300))[:,None]
                raw=lower_loss+np.sum(lower_g*(endpoint-p['lower_v'][lo:hi]),axis=1)
                compare(raw,p['raw_lower'][lo:hi]); compare(np.maximum(0.,raw-b['solver']['lower_allowance']),p['lower'][lo:hi])
                t=p['best_t_f32'][lo:hi].astype('f8'); radius=np.sqrt(np.mean(t*t,axis=1)+epsilon); vv=np.column_stack([t/radius[:,None],scale/radius]); upper=point(vv,m[lo:hi],H[lo:hi],aug,C,beta,scale,gradient=False)
                compare(upper,p['best_real_loss'][lo:hi]); assert np.max(p['lower'][lo:hi]-upper)<=1e-8; checked_points+=hi-lo
            scalar(w@p['best_real_loss'],trace['weighted_real_upper']); scalar(w@p['lower'],trace['weighted_real_lower']); assert cp['iteration']==trace['iteration'] and cp['final']==trace['final']
            previous=p; print(json.dumps(dict(audited_checkpoint=cp['iteration'],final=cp['final'],seconds=time.perf_counter()-CLOCK)),flush=True)
        assert len(r['checkpoints'])==len(r['trace']) and r['final_checkpoint']==r['checkpoints'][-1]['file'] and r['checkpoints'][-1]['final']; final=previous
        assert np.all(final['updates']>=r['completed_outer']) and np.all(final['updates']<=r['completed_outer']+int(r['partial_outer']))
        get,stream=tensor_reader(Path(b['prior']['source'])/'model.safetensors',np); cfg=json.loads((Path(b['prior']['source'])/'config.json').read_bytes()); gamma=get('model.final_layernorm.weight'); source=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):
            guard(); hi=min(lo+1024,65537); source[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close(); stream=None; dll=ctypes.CDLL(b['reference']['dll']['path']); fn=dll.reference_readout
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_float,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]; fn.restype=ctypes.c_int
        offset=scorewords=0; nativecalls=0; splitchecks=[]
        for s in r['splits']:
            old=load(b['prior']['parent'][s['split']+'_dataset']); pack=load(s['pack']); n=len(old['H']); compare(pack['lower'],final['lower'][offset:offset+n]); assert np.array_equal(pack['t_f32'].view('u4'),final['best_t_f32'][offset:offset+n].view('u4'))
            mapping=np.memmap(s['scores']['path'],mode='r',dtype='<f4',shape=(n,65537)); calculated=np.empty((n,4),'f8'); precision={k:True for k in ['score_absolute','norm_relative','inverse_relative']}
            for lo in range(0,n,64):
                guard(); hi=min(lo+64,n); count=hi-lo; t32=pack['t_f32'][lo:hi]; norm=np.empty((count,256),'<f4'); inv=np.empty(count,'<f4'); out=np.empty((count,65537),'<f4')
                assert fn(C32.ctypes.data,beta32.ctypes.data,t32.ctypes.data,count,epsilon,norm.ctypes.data,inv.ctypes.data,out.ctypes.data)==0
                assert np.array_equal(norm.view('u4'),pack['normalized_f32'][lo:hi].view('u4')) and np.array_equal(inv.view('u4'),pack['inverse_f32'][lo:hi].view('u4')) and np.array_equal(out.view('u4'),mapping[lo:hi].view('u4'))
                scorewords+=out.size; nativecalls+=1; t=t32.astype('f8'); radius=np.sqrt(np.mean(t*t,axis=1)+epsilon); vv=np.column_stack([t/radius[:,None],scale/radius]); real=partial(vv[:,:256],C,32)+vv[:,-1,None]*(beta/scale)
                x=old['H'][lo:hi]; teacher=partial(x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+b['epsilon_source']),source,256); qlog=teacher-logz(teacher)[:,None]; plog=out.astype('f8')-logz(out.astype('f8'))[:,None]
                calculated[lo:hi]=np.column_stack([np.sum(np.exp(qlog)*(qlog-plog),axis=1),np.argmax(out,axis=1)!=np.argmax(teacher,axis=1),-np.sum(np.exp(qlog)*qlog,axis=1),-np.sum(np.exp(plog)*plog,axis=1)])
                compare(pack['metrics'][lo:hi],calculated[lo:hi]); real_loss=point(vv,m[offset+lo:offset+hi],H[offset+lo:offset+hi],aug,C,beta,scale,gradient=False); compare(real_loss,pack['real_loss'][lo:hi]); compare(real_loss,final['best_real_loss'][offset+lo:offset+hi])
                delta=out.astype('f8')-real; span=delta.max(axis=1)-delta.min(axis=1); compare(span,pack['score_span'][lo:hi]); assert np.max(np.abs(calculated[lo:hi,0]-real_loss)-span)<=1e-9
                calc=dict(score_absolute=float(np.max(np.abs(delta))),norm_relative=float(np.linalg.norm(norm.astype('f8')-vv[:,:256])/np.linalg.norm(vv[:,:256])),inverse_relative=float(np.max(np.abs(inv.astype('f8')*radius-1))))
                batch=s['batches'][lo//64]; assert batch['lo']==lo and batch['hi']==hi
                for key in calc:scalar(calc[key],batch[key]); precision[key]&=calc[key]<=b['numeric'][key]
            mapping._mmap.close(); mapping=None; quality(calculated,b['prior']['parent'][s['split']+'_cases'],s['quality']); assert precision==s['precision_flags']
            lower=pack['lower']; cases=b['prior']['parent'][s['split']+'_cases']; cc=[float(lower[c['lo']:c['hi']].mean()) for c in cases]
            scalar(old['weights']@lower,s['lower_case_mean']); scalar(lower.max(),s['lower_worst_label']); violation=bool(old['weights']@lower>b['quality']['case_mean_KL'] or lower.max()>b['quality']['worst_label_KL'] or max(cc)>b['quality']['every_case_KL'])
            assert violation==s['real_domain_quality_lower_violation']; splitchecks.append(dict(split=s['split'],rows=n,native_bits_exact=True,quality_gates=s['quality']['quality_gates'],real_lower_violation=violation)); offset+=n
            print(json.dumps(dict(audited_native=s['split'],seconds=time.perf_counter()-CLOCK)),flush=True)
        assert offset==8808 and scorewords==577249896 and nativecalls==139 and nativecalls==r['native_calls']
        assert r['native_oracle_quality_pass']==all(all(s['quality_gates'].values()) for s in splitchecks) and r['numerical_real_domain_lower_violation']==any(s['real_lower_violation'] for s in splitchecks)
        assert not r['interval_certified'] and not r['global_F32_floor_proved'] and not r['causal_encoder_implemented'] and not r['timing_admitted']; check(); guard()
        result=dict(schema='ALL_LABEL_NATIVE_PROBABILITY_GAP_AUDIT_V1',source=extent(__file__),freeze=a.freeze,report=extent(a.report),terminal=extent(a.terminal),binding=r['binding'],checkpoints=len(r['checkpoints']),
                    all_stored_checkpoint_rows=checked_points,identical_current_lower_rows_reused_exactly=same_lower_rows,all_checkpoint_functions_bounds_upper_losses_checked=True,
                    all577249896_native_scores2254848_normalized_words8808_inverse_words_bit_exact=True,all_source_native_probability_quality_rows_checked=True,split_checks=splitchecks,
                    max_relative=maxrel,max_scalar_delta=maxscalar,before_after_hashes_exact=True,decision='INDEPENDENT_STORED_PASS',seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,
                    scope='Every stored point/full native/source probability audit. Numerical real-domain bounds,not interval/globalF32 floor;no unstored trajectory replay/convergence/causal model/useful-n/50 admission.')
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
