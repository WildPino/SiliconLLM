"""Bounded full-label fixed-head oracle search, then actual native/source capture."""
import argparse,ctypes,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from native_head_gpu_preflight import gpu_metadata
from broad_exact_residual_correction import summarize
START=time.monotonic(); CLOCK=time.perf_counter()


def prepare(a):
    gp=DOC/'native_head_gpu_preflight_result_20261012.json'; ga=DOC/'native_head_gpu_preflight_adjudication_20261012.json'
    r=json.loads(gp.read_bytes()); ad=json.loads(ga.read_bytes()); assert ad['report']==extent(gp) and ad['decision']=='INDEPENDENT_STORED_PASS' and ad['GPU_numeric_pass']
    gb=json.loads(Path(r['binding']['path']).read_bytes()); prior=gb['prior']; native=prior['native_report']; nb=json.loads(Path(native['binding']['path']).read_bytes())
    ep=DOC/'native_scalar_readout_preparation_20261012.json'; rp=DOC/'gauged_native_reference_preparation_20261012.json'
    engine=json.loads(ep.read_bytes()); reference=json.loads(rp.read_bytes())
    extra=[Path(__file__),B/'native_head_gpu_preflight.py',B/'broad_exact_residual_correction.py',gp,ga,Path(r['binding']['path']),Path(native['binding']['path']),ep,rp,
           DOC/'ALL_LABEL_NATIVE_PROBABILITY_GAP_PROTOCOL_20261012.md',Path(prior['source'])/'model.safetensors',Path(prior['source'])/'config.json']
    extra += [Path(s['codes']['path']) for s in native['splits']]+[Path(prior['parent'][s+'_dataset']['path']) for s in ['FIT','DEV']]
    extra += [Path(engine[k]['path']) for k in ['source','dll','log','compiler','engine']]+[Path(reference[k]['path']) for k in ['dll','log','assembly']]+[Path(e['path']) for e in reference['inputs']]
    inputs=gb['inputs']+[extent(p) for p in extra]; inputs=list({e['path']:e for e in inputs}.values())
    b=dict(schema='ALL_LABEL_NATIVE_PROBABILITY_GAP_BINDING_V1',python=gb['python'],gpu=gb['gpu'],versions=gb['versions'],GPU_qualification=r,
           preflight=gb['preflight'],prior=prior,native=native,engine=engine,reference=reference,quality=nb['quality'],epsilon_source=prior['epsilon_source'],
           solver=dict(max_outer=128,max_backtracks=30,initial_L=1.,L_floor=1e-12,majorant_slack=1e-12,descent_slack=1e-9,
                       lower_allowance=1e-8,upper_improvement=1e-12,tau=1e-4,checkpoint_every=16,seconds=1200),
           batch=64,threads=6,numeric=dict(loss_absolute=1e-8,gradient_absolute=1e-8,sphere_absolute=1e-10,score_absolute=1e-4,norm_relative=2e-5,inverse_relative=2e-5,score_span_slack=1e-9),
           limits=dict(seconds=1800,OS_bytes=6<<30,GPU_allocated_bytes=2<<30,GPU_reserved_bytes=2<<30,output_bytes=4<<30),inputs=inputs,
           scope='All8808 existing source labels,independent per-label codes including DEV,unchanged actual F32 head. Oracle probability-image diagnostic, '
                 'not held-out learned model/causal core/n/50. Convex real half-ball supporting lowers and executable sphere-code uppers are distinct. '
                 'All stored mathematical checkpoints/final native bits/source probabilities must be independently audited;no trajectory optimality claim.')
    assert gpu_metadata()==b['gpu']; write(a.out,b); print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    assert extent(a.binding)['sha256']==a.binding_sha; b=json.loads(a.binding.read_bytes()); assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert gpu_metadata()==b['gpu'] and not a.directory.exists() and not a.out.exists()
    assert all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']) and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions']
    proc=psutil.Process(); proc.cpu_affinity(list(range(6))); torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.manual_seed(0)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.init(); torch.cuda.reset_peak_memory_stats(); mapping=stream=None
    receipt=dict(schema='ALL_LABEL_NATIVE_PROBABILITY_GAP_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        lim=b['limits']; assert time.monotonic()-START<=lim['seconds'] and proc.memory_info().peak_wset<=lim['OS_bytes'] and not proc.children(recursive=True)
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=lim['output_bytes']
    def check():
        for e in b['inputs']:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def save(name,**fields):
        p=a.directory/(name+'.npz'); np.savez(p,**fields); return extent(p)
    try:
        check(); a.directory.mkdir(); native=b['native']; C32=np.fromfile(native['head_F32']['path'],'<f4').reshape(65537,256); beta32=np.fromfile(native['bias_F32']['path'],'<f4')
        epsilon=float(np.fromfile(native['epsilon_F32']['path'],'<f4')[0]); scale=math.sqrt(256*epsilon)
        cpu=np.column_stack([C32.astype('f8'),beta32.astype('f8')/scale]); matrix=torch.from_numpy(cpu).to('cuda'); del cpu
        packets=[load(s['file']) for s in b['preflight']['splits']]; moment_np=np.concatenate([p['moments'] for p in packets]); entropy_np=np.concatenate([p['entropy'] for p in packets])
        initial_np=np.concatenate([p['v'] for p in packets]); loss_gold=np.concatenate([p['loss'] for p in packets]); grad_gold=np.concatenate([p['gradient'] for p in packets]); weights_np=np.concatenate([p['weights']/2 for p in packets])
        warm_np=np.concatenate([load(s['codes'])['t_f32'] for s in native['splits']]); N=len(initial_np); assert N==8808
        m=torch.from_numpy(moment_np).to('cuda'); H=torch.from_numpy(entropy_np).to('cuda'); v=torch.from_numpy(initial_np).to('cuda'); best_code=torch.from_numpy(warm_np).to('cuda')
        L=torch.empty(N,dtype=torch.float64,device='cuda'); g=torch.empty_like(v); steps=torch.ones_like(L); updates=torch.zeros(N,dtype=torch.int32,device='cuda'); calls=0
        def objective(x,mm,hh,gradient=True):
            nonlocal calls
            calls+=1; guard(); assert x.dtype==mm.dtype==hh.dtype==matrix.dtype==torch.float64
            score=x@matrix.T; value=torch.logsumexp(score,dim=1)-(mm*x).sum(dim=1)-hh
            return (value,torch.softmax(score,dim=1)@matrix-mm) if gradient else value
        def projection(x):
            x=x.clone(); x[:,-1].clamp_(min=0); norm=x.norm(dim=1); return x*torch.clamp(16/norm.clamp_min(1e-300),max=1.)[:,None]
        def support(value,gradient,point):return value-(gradient*point).sum(dim=1)-16*torch.sqrt((gradient[:,:256]**2).sum(dim=1)+torch.minimum(gradient[:,-1],torch.zeros_like(value))**2)
        def sphere_code(point):
            u=point[:,:256].clone(); square=(u*u).sum(dim=1); limit=256-b['solver']['tau']**2
            u*=torch.sqrt(torch.clamp(limit/square.clamp_min(1e-300),max=1.))[:,None]
            last=torch.sqrt(torch.clamp(256-(u*u).sum(dim=1),min=b['solver']['tau']**2)); code=(scale*u/last[:,None]).float(); assert torch.isfinite(code).all()
            t=code.double(); radius=torch.sqrt((t*t).mean(dim=1)+epsilon); vv=torch.cat([t/radius[:,None],(scale/radius)[:,None]],dim=1); return code,vv
        checkpoints=[]; trace=[]; solver=b['solver']; solver_started=time.monotonic(); best_lower_point=v.clone(); best_lower_gradient=torch.empty_like(v); best_lower_loss=torch.empty_like(L)
        with torch.inference_mode():
            for lo in range(0,N,64):
                hi=min(lo+64,N); L[lo:hi],g[lo:hi]=objective(v[lo:hi],m[lo:hi],H[lo:hi])
            assert np.max(np.abs(L.cpu().numpy()-loss_gold))<=b['numeric']['loss_absolute'] and np.max(np.abs(g.cpu().numpy()-grad_gold))<=b['numeric']['gradient_absolute']
            best_upper=L.clone(); best_raw_lower=support(L,g,v); best_lower_gradient.copy_(g); best_lower_loss.copy_(L)
            def checkpoint(iteration,final=False):
                guard(); fields=dict(v=v.cpu().numpy(),loss=L.cpu().numpy(),gradient=g.cpu().numpy(),steps=steps.cpu().numpy(),updates=updates.cpu().numpy(),
                    lower_v=best_lower_point.cpu().numpy(),lower_gradient=best_lower_gradient.cpu().numpy(),lower_loss=best_lower_loss.cpu().numpy(),raw_lower=best_raw_lower.cpu().numpy(),
                    lower=torch.clamp(best_raw_lower-solver['lower_allowance'],min=0).cpu().numpy(),best_t_f32=best_code.cpu().numpy(),best_real_loss=best_upper.cpu().numpy())
                e=save('checkpoint_'+str(iteration).zfill(3)+('_final' if final else ''),**fields); checkpoints.append(dict(iteration=iteration,final=final,file=e))
                row=dict(iteration=iteration,final=final,calls=calls,solver_seconds=time.monotonic()-solver_started,weighted_real_upper=float(weights_np@fields['best_real_loss']),weighted_real_lower=float(weights_np@fields['lower']),
                         min_updates=int(fields['updates'].min()),max_updates=int(fields['updates'].max())); trace.append(row); print(json.dumps(row),flush=True)
            checkpoint(0); completed=0; stop='MAX_OUTER'; partial=False
            for iteration in range(1,solver['max_outer']+1):
                if time.monotonic()-solver_started>=solver['seconds']:stop='SOLVER_BUDGET'; break
                for lo in range(0,N,64):
                    if time.monotonic()-solver_started>=solver['seconds']:stop='SOLVER_BUDGET_PARTIAL_OUTER'; partial=True; break
                    hi=min(lo+64,N); old_v=v[lo:hi]; old_L=L[lo:hi]; old_g=g[lo:hi]; local_L=torch.clamp(steps[lo:hi]/2,min=solver['L_floor']); accepted=torch.zeros(hi-lo,dtype=torch.bool,device='cuda')
                    next_v=old_v.clone(); next_L=old_L.clone(); next_g=old_g.clone()
                    for backtrack in range(solver['max_backtracks']):
                        candidate=projection(old_v-old_g/local_L[:,None]); candidate=torch.where(accepted[:,None],next_v,candidate)
                        values,gradient=objective(candidate,m[lo:hi],H[lo:hi]); delta=candidate-old_v
                        majorant=old_L+(old_g*delta).sum(dim=1)+.5*local_L*(delta*delta).sum(dim=1)
                        newly=(values<=majorant+solver['majorant_slack'])&(~accepted); next_v[newly]=candidate[newly]; next_L[newly]=values[newly]; next_g[newly]=gradient[newly]; accepted|=newly
                        if bool(accepted.all()):break
                        local_L[~accepted]*=2
                    assert bool(accepted.all()),'backtracking limit'; assert float((next_L-old_L).max())<=solver['descent_slack'],'descent violation'
                    v[lo:hi]=next_v; L[lo:hi]=next_L; g[lo:hi]=next_g; steps[lo:hi]=local_L; updates[lo:hi]+=1
                    raw=support(next_L,next_g,next_v); improve=raw>best_raw_lower[lo:hi]; best_raw_lower[lo:hi][improve]=raw[improve]
                    best_lower_point[lo:hi][improve]=next_v[improve]; best_lower_gradient[lo:hi][improve]=next_g[improve]; best_lower_loss[lo:hi][improve]=next_L[improve]
                    candidate_code,candidate_v=sphere_code(next_v); candidate_loss=objective(candidate_v,m[lo:hi],H[lo:hi],gradient=False)
                    improve=candidate_loss<best_upper[lo:hi]-solver['upper_improvement']; best_code[lo:hi][improve]=candidate_code[improve]; best_upper[lo:hi][improve]=candidate_loss[improve]
                if partial:break
                completed=iteration
                if iteration%solver['checkpoint_every']==0:checkpoint(iteration)
            checkpoint(completed,final=True); final_checkpoint=checkpoints[-1]['file']; final=load(final_checkpoint)
        solver_seconds=time.monotonic()-solver_started
        del matrix,m,H,v,L,g,steps,updates,best_code,best_upper,best_raw_lower,best_lower_point,best_lower_gradient,best_lower_loss
        del packets,initial_np,loss_gold,grad_gold; torch.cuda.empty_cache()
        get,stream=tensor_reader(Path(b['prior']['source'])/'model.safetensors',np); cfg=json.loads((Path(b['prior']['source'])/'config.json').read_bytes()); gamma=get('model.final_layernorm.weight')
        source_head=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):
            guard(); hi=min(lo+1024,65537); source_head[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close(); stream=None; C=C32.astype('f8'); beta=beta32.astype('f8')
        dll=ctypes.CDLL(b['engine']['dll']['path']); fn=dll.scalar_readout; fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_float,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]; fn.restype=ctypes.c_int
        outputs=[p['file'] for p in checkpoints]; splits=[]; native_calls=0; offset=0
        for split in ['FIT','DEV']:
            data=load(b['prior']['parent'][split+'_dataset']); old=load(next(s for s in b['preflight']['splits'] if s['split']==split)['file']); Hs=data['H']; n=len(Hs)
            codes=np.ascontiguousarray(final['best_t_f32'][offset:offset+n]); norm=np.empty_like(codes); inverse=np.empty(n,'<f4'); metrics=np.empty((n,4),'f8'); real_losses=np.empty(n,'f8'); spans=np.empty(n,'f8'); batches=[]
            path=a.directory/(split+'_native_scores.f32'); mapping=np.memmap(path,mode='w+',dtype='<f4',shape=(n,65537))
            for lo in range(0,n,64):
                guard(); hi=min(lo+64,n); count=hi-lo; out=np.empty((count,65537),'<f4'); began=time.perf_counter()
                assert fn(C32.ctypes.data,beta32.ctypes.data,codes[lo:hi].ctypes.data,count,epsilon,norm[lo:hi].ctypes.data,inverse[lo:hi].ctypes.data,out.ctypes.data)==0
                native_seconds=time.perf_counter()-began; mapping[lo:hi]=out; native_calls+=1
                t=codes[lo:hi].astype('f8'); radius=np.sqrt(np.mean(t*t,axis=1)+epsilon); vv=np.column_stack([t/radius[:,None],scale/radius]); real=(t@C.T+beta)/radius[:,None]
                raw=Hs[lo:hi]; teacher=(raw/np.sqrt(np.mean(raw*raw,axis=1,keepdims=True)+b['epsilon_source']))@source_head.T
                qlog=teacher-np.logaddexp.reduce(teacher,axis=1)[:,None]; q=np.exp(qlog); native_score=out.astype('f8'); plog=native_score-np.logaddexp.reduce(native_score,axis=1)[:,None]
                metrics[lo:hi]=np.column_stack([np.sum(q*(qlog-plog),axis=1),np.argmax(native_score,axis=1)!=np.argmax(teacher,axis=1),-np.sum(q*qlog,axis=1),-np.sum(np.exp(plog)*plog,axis=1)])
                real_losses[lo:hi]=np.logaddexp.reduce(real,axis=1)-np.sum(old['moments'][lo:hi]*vv,axis=1)-old['entropy'][lo:hi]
                delta=native_score-real; spans[lo:hi]=delta.max(axis=1)-delta.min(axis=1); assert np.max(np.abs(metrics[lo:hi,0]-real_losses[lo:hi])-spans[lo:hi])<=b['numeric']['score_span_slack']
                assert np.max(np.abs(real_losses[lo:hi]-final['best_real_loss'][offset+lo:offset+hi]))<=b['numeric']['loss_absolute']
                batches.append(dict(lo=lo,hi=hi,native_seconds=native_seconds,score_absolute=float(np.max(np.abs(delta))),norm_relative=float(np.linalg.norm(norm[lo:hi].astype('f8')-vv[:,:256])/np.linalg.norm(vv[:,:256])),inverse_relative=float(np.max(np.abs(inverse[lo:hi].astype('f8')*radius-1)))))
            mapping.flush(); mapping._mmap.close(); mapping=None; score_extent=extent(path)
            pack=save(split+'_native',t_f32=codes,normalized_f32=norm,inverse_f32=inverse,metrics=metrics,real_loss=real_losses,score_span=spans,lower=final['lower'][offset:offset+n]); outputs.extend([score_extent,pack])
            cases=b['prior']['parent'][split+'_cases']; quality=summarize(metrics,cases,b['quality']); lower=final['lower'][offset:offset+n]; lower_cases=[float(lower[c['lo']:c['hi']].mean()) for c in cases]
            lower_violation=bool(float(data['weights']@lower)>b['quality']['case_mean_KL'] or float(lower.max())>b['quality']['worst_label_KL'] or max(lower_cases)>b['quality']['every_case_KL'])
            flags={k:all(bb[k]<=b['numeric'][k] for bb in batches) for k in ['score_absolute','norm_relative','inverse_relative']}
            splits.append(dict(split=split,rows=n,pack=pack,scores=score_extent,quality=quality,precision_flags=flags,batches=batches,
                               lower_case_mean=float(data['weights']@lower),lower_worst_label=float(lower.max()),real_domain_quality_lower_violation=lower_violation)); offset+=n
        assert offset==8808 and native_calls==139; check(); guard(); assert gpu_metadata()==b['gpu']
        result=dict(schema='ALL_LABEL_NATIVE_PROBABILITY_GAP_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,solver_stop=stop,completed_outer=completed,partial_outer=partial,solver_seconds=solver_seconds,
                    checkpoints=checkpoints,final_checkpoint=final_checkpoint,trace=trace,objective_calls=calls,native_calls=native_calls,splits=splits,outputs=outputs,
                    native_oracle_quality_pass=all(all(s['quality']['quality_gates'].values()) for s in splits),numerical_real_domain_lower_violation=any(s['real_domain_quality_lower_violation'] for s in splits),
                    decision='COMPLETED_ORACLE_PROBABILITY_IMAGE_PENDING_INDEPENDENT_AUDIT',before_after_hashes_exact=True,seconds=time.perf_counter()-CLOCK,
                    OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),timing_admitted=False,
                    interval_certified=False,global_F32_floor_proved=False,causal_encoder_implemented=False,scope=b['scope'])
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out)); print(json.dumps(dict(stop=stop,completed_outer=completed,quality=result['native_oracle_quality_pass'],lower_violation=result['numerical_real_domain_lower_violation'],seconds=result['seconds'])),flush=True)
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        if stream is not None:stream.close()
        if mapping is not None:mapping._mmap.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved()); write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args(); assert a.prepare!=a.run
    prepare(a) if a.prepare else run(a)
