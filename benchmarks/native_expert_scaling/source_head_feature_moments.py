"""Source-head2048 moments, full different-decoder loss/gradient controls."""
import argparse,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from native_head_gpu_preflight import gpu_metadata
START=time.monotonic(); CLOCK=time.perf_counter()


def prepare(a):
    import numpy as np,psutil,torch
    op=DOC/'all_label_native_probability_gap_result_20261012.json'; oa=DOC/'all_label_native_probability_gap_adjudication_20261012.json'; r=json.loads(op.read_bytes()); ad=json.loads(oa.read_bytes())
    assert ad['report']==extent(op) and ad['decision']=='INDEPENDENT_STORED_PASS'; parent=json.loads(Path(r['binding']['path']).read_bytes()); broad=parent['prior']['parent']
    gp=DOC/'native_head_gpu_preflight_binding_20261012.json'; gb=json.loads(gp.read_bytes())
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',B/'native_head_gpu_preflight.py',op,oa,Path(r['binding']['path']),gp,
           DOC/'SOURCE_HEAD_FEATURE_MOMENTS_PROTOCOL_20261012.md',Path(parent['prior']['source'])/'model.safetensors',Path(parent['prior']['source'])/'config.json',Path(r['final_checkpoint']['path']),
           Path(parent['native']['gauge']['path']),Path(broad['baseline']['parameters']['path']),Path(broad['selected']['parameters']['path'])]
    paths += [Path(broad[s+'_dataset']['path']) for s in ['FIT','DEV']]
    paths += [Path(x['path']) for x in [broad['baseline']['file'],broad['selected']['file'],broad['DEV_baseline']['file'],broad['DEV_selected']['file']]]
    runtime=[e for e in gb['inputs'] if e['path'].startswith(str(Path(sys.executable).parent)) or e['path'].startswith(str(SITE.resolve())) or e['path'].startswith('C:\\Windows\\System32')]
    inputs=runtime+[extent(p) for p in paths]
    b=dict(schema='SOURCE_HEAD_FEATURE_MOMENTS_BINDING_V1',python=gb['python'],gpu=gb['gpu'],versions=gb['versions'],oracle=r,parent=parent,broad=broad,
           batch=64,threads=6,controls=['original_decoder','corrected_decoder','oracle_source_reconstruction'],epsilon_source=parent['epsilon_source'],
           numeric=dict(source_entropy_absolute=1e-8,loss_absolute=1e-8,feature_gradient_absolute=1e-8,feature_gradient_relative=1e-9,self_absolute=1e-8),
           limits=dict(seconds=900,OS_bytes=6<<30,GPU_allocated_bytes=2<<30,GPU_reserved_bytes=2<<30,output_bytes=2<<30),inputs=list({e['path']:e for e in inputs}.values()),
           scope='All8808 actual source/operator q,2048-feature moments and3 existing source-derived decoder points/full vocabulary. '
                 'GPUF64 compilation/real loss+feature+RMS gradients,not optimization/native quantized export/causal model/n/50. '
                 'Fixed actual source head column space only;native F32 head rounding remains separate.')
    assert gpu_metadata()==b['gpu']; write(a.out,b); print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    assert extent(a.binding)['sha256']==a.binding_sha; b=json.loads(a.binding.read_bytes()); assert Path(sys.executable).resolve()==Path(b['python']).resolve() and gpu_metadata()==b['gpu']
    assert not a.out.exists() and not a.directory.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions']
    proc=psutil.Process(); proc.cpu_affinity(list(range(6))); torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.manual_seed(0)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.cuda.init(); torch.cuda.reset_peak_memory_stats(); stream=None
    receipt=dict(schema='SOURCE_HEAD_FEATURE_MOMENTS_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        lim=b['limits']; assert time.monotonic()-START<=lim['seconds'] and proc.memory_info().peak_wset<=lim['OS_bytes'] and not proc.children(recursive=True)
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=lim['output_bytes']
    def check():
        for e in b['inputs']:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def one(e,key):
        with np.load(e['path'],allow_pickle=False) as z:return z[key].copy()
    def save(name,**fields):
        p=a.directory/(name+'.npz'); np.savez(p,**fields); return extent(p)
    try:
        check(); a.directory.mkdir(); parent=b['parent']; broad=b['broad']; get,stream=tensor_reader(Path(parent['prior']['source'])/'model.safetensors',np)
        cfg=json.loads((Path(parent['prior']['source'])/'config.json').read_bytes()); gamma=get('model.final_layernorm.weight'); A=np.empty((65537,2048),'f8')
        for lo in range(0,65537,1024):
            guard(); hi=min(lo+1024,65537); A[lo:hi]=get('lm_head.weight',list(range(lo,hi)))*gamma*cfg['lm_head_multiplier']
        stream.close(); stream=None; matrix=torch.from_numpy(A).to('cuda'); assert matrix.dtype==torch.float64; del A
        params=[load(broad['baseline']['parameters']),load(broad['selected']['parameters'])]; gauge=load(parent['native']['gauge']); best_t=one(b['oracle']['final_checkpoint'],'best_t_f32')
        outputs=[]; splits=[]; offset=0
        with torch.inference_mode():
            for name in ['FIT','DEV']:
                data=load(broad[name+'_dataset']); old=load(next(s for s in parent['preflight']['splits'] if s['split']==name)['file']); raw=data['H']; n=len(raw)
                zpoints=[one(broad['baseline']['file'] if name=='FIT' else broad['DEV_baseline']['file'],'z'),one(broad['selected']['file'] if name=='FIT' else broad['DEV_selected']['file'],'z')]
                moments=np.empty((n,2048),'f8'); entropy=np.empty(n,'f8'); source_ids=np.empty(n,'i8'); self_error=np.empty((n,2),'f8'); qmass=np.empty(n,'f8')
                control_values=[np.empty((n,5),'f8') for _ in range(3)]; gu=[np.empty((n,2048),'f8') for _ in range(3)]; gy=[np.empty((n,2048),'f8') for _ in range(3)]; radius_out=[np.empty(n,'f8') for _ in range(3)]; comparisons=[[] for _ in range(3)]; timings=[]
                for lo in range(0,n,64):
                    guard(); hi=min(lo+64,n); h=torch.from_numpy(raw[lo:hi]).to('cuda'); u=h/torch.sqrt((h*h).mean(dim=1,keepdim=True)+b['epsilon_source']); torch.cuda.synchronize(); began=time.perf_counter()
                    source_score=u@matrix.T; qlog=torch.log_softmax(source_score,dim=1); q=qlog.exp(); ms=q@matrix; Hq=-(q*qlog).sum(dim=1)
                    torch.cuda.synchronize(); compile_seconds=time.perf_counter()-began; moments[lo:hi]=ms.cpu().numpy(); entropy[lo:hi]=Hq.cpu().numpy(); source_ids[lo:hi]=source_score.argmax(dim=1).cpu().numpy(); qmass[lo:hi]=q.sum(dim=1).cpu().numpy()
                    self_compiled=torch.logsumexp(source_score,dim=1)-(ms*u).sum(dim=1)-Hq; self_error[lo:hi,0]=self_compiled.abs().cpu().numpy(); self_error[lo:hi,1]=(q@matrix-ms).abs().max(dim=1).values.cpu().numpy()
                    ys=[p['mu']+zpoints[k][lo:hi]@p['V'].T for k,p in enumerate(params)]
                    ys.append(best_t[offset+lo:offset+hi].astype('f8')@gauge['Q'].T+gauge['center_perpendicular'])
                    times=[]
                    for ci,y_np in enumerate(ys):
                        y=torch.from_numpy(np.ascontiguousarray(y_np)).to('cuda'); rsource=torch.sqrt((y*y).mean(dim=1)+b['epsilon_source']); value=y/rsource[:,None]
                        torch.cuda.synchronize(); began=time.perf_counter(); score=value@matrix.T; plog=torch.log_softmax(score,dim=1); probability=plog.exp()
                        compiled=torch.logsumexp(score,dim=1)-(ms*value).sum(dim=1)-Hq; direct=(q*(qlog-plog)).sum(dim=1)
                        grad=probability@matrix-ms; direct_grad=(probability-q)@matrix
                        grad_y=grad/rsource[:,None]-y*(y*grad).sum(dim=1)[:,None]/(2048*rsource[:,None]**3)
                        torch.cuda.synchronize(); times.append(time.perf_counter()-began)
                        control_values[ci][lo:hi]=np.column_stack([compiled.cpu().numpy(),direct.cpu().numpy(),(score.argmax(dim=1)!=source_score.argmax(dim=1)).cpu().numpy(),-(probability*plog).sum(dim=1).cpu().numpy(),rsource.cpu().numpy()])
                        gu[ci][lo:hi]=grad.cpu().numpy(); gy[ci][lo:hi]=grad_y.cpu().numpy(); radius_out[ci][lo:hi]=rsource.cpu().numpy()
                        comparisons[ci].append(dict(lo=lo,hi=hi,loss_absolute=float((compiled-direct).abs().max()),gradient_absolute=float((grad-direct_grad).abs().max()),gradient_relative=float((grad-direct_grad).norm()/direct_grad.norm().clamp_min(1.))))
                    timings.append(dict(lo=lo,hi=hi,compile_seconds=compile_seconds,control_seconds=times))
                source_extent=save(name+'_source_moments',moment=moments,entropy=entropy,source_ids=source_ids,self_error=self_error,qmass=qmass,weights=data['weights']); outputs.append(source_extent)
                entropy_delta=float(np.max(np.abs(entropy-old['entropy']))); assert entropy_delta<=b['numeric']['source_entropy_absolute'] and np.max(self_error)<=b['numeric']['self_absolute']
                controls=[]
                for ci,control in enumerate(b['controls']):
                    e=save(name+'_'+control,values=control_values[ci],gradient_u=gu[ci],gradient_y=gy[ci],radius=radius_out[ci]); outputs.append(e)
                    worst={k:max(row[k] for row in comparisons[ci]) for k in ['loss_absolute','gradient_absolute','gradient_relative']}
                    flags=dict(loss=worst['loss_absolute']<=b['numeric']['loss_absolute'],gradient_absolute=worst['gradient_absolute']<=b['numeric']['feature_gradient_absolute'],gradient_relative=worst['gradient_relative']<=b['numeric']['feature_gradient_relative'])
                    controls.append(dict(name=control,file=e,comparison=worst,flags=flags,case_mean_KL=float(data['weights']@control_values[ci][:,1]),case_disagreement=float(data['weights']@control_values[ci][:,2])))
                splits.append(dict(split=name,rows=n,moments=source_extent,controls=controls,source_entropy_delta=entropy_delta,maximum_self_error=float(self_error.max()),timings=timings)); offset+=n
                print(json.dumps(dict(split=name,rows=n,control_flags=[c['flags'] for c in controls],seconds=time.perf_counter()-CLOCK)),flush=True)
        assert offset==8808; check(); guard(); assert gpu_metadata()==b['gpu']
        result=dict(schema='SOURCE_HEAD_FEATURE_MOMENTS_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,splits=splits,outputs=outputs,
                    decision='SOURCE_FEATURE_MOMENTS_AND_CONTROLS_PASS_PENDING_AUDIT' if all(all(c['flags'].values()) for s in splits for c in s['controls']) else 'COMPLETED_SOURCE_FEATURE_CONTROL_FAILURE',
                    source_moment_payload_bytes=8808*2048*8,feature_width=2048,vocabulary=65537,optimization_run=False,native_quantized_export_admitted=False,
                    before_after_hashes_exact=True,seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),timing_admitted=False,scope=b['scope'])
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved()); write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args(); assert a.prepare!=a.run
    prepare(a) if a.prepare else run(a)
