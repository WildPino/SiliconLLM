"""Full CPU/GPU F64 initial head loss/gradient qualification and cost only."""
import argparse,ctypes,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'; SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def gpu_metadata():
    dll=ctypes.WinDLL('C:/Windows/System32/nvml.dll')
    assert dll.nvmlInit_v2()==0
    try:
        handle=ctypes.c_void_p(); get=dll.nvmlDeviceGetHandleByIndex_v2
        get.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)]; assert get(0,ctypes.byref(handle))==0
        out={}
        for key,name in [('driver','nvmlSystemGetDriverVersion'),('name','nvmlDeviceGetName'),('uuid','nvmlDeviceGetUUID')]:
            buffer=ctypes.create_string_buffer(128); fn=getattr(dll,name)
            if key=='driver':fn.argtypes=[ctypes.c_void_p,ctypes.c_uint]; rc=fn(buffer,128)
            else:fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint]; rc=fn(handle,buffer,128)
            assert rc==0; out[key]=buffer.value.decode('ascii')
        return out
    finally:assert dll.nvmlShutdown()==0


def prepare(a):
    import numpy as np,psutil,torch
    report=DOC/'native_head_domain_preflight_result_v2_20261012.json'; audit=DOC/'native_head_domain_preflight_adjudication_v2_20261012.json'
    p=json.loads(report.read_bytes()); ad=json.loads(audit.read_bytes()); assert ad['report']==extent(report) and ad['decision']=='INDEPENDENT_STORED_PASS'
    prior=json.loads(Path(p['binding']['path']).read_bytes()); nr=prior['native_report']
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',report,audit,Path(p['binding']['path']),
           DOC/'NATIVE_HEAD_GPU_PREFLIGHT_PROTOCOL_20261012.md',Path(sys.executable),Path(sys.executable).parent/'python312.dll',
           Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(psutil.__file__),Path(torch.__file__),Path(torch._C.__file__),
           Path(torch.__file__).parent/'functional.py',Path(torch.__file__).parent/'_tensor.py',Path(torch.__file__).parent/'nn/functional.py',
           Path(torch.__file__).parent/'cuda/__init__.py',Path('C:/Windows/System32/nvcuda.dll'),Path('C:/Windows/System32/nvml.dll')]
    paths += [Path(nr[k]['path']) for k in ['head_F32','bias_F32','epsilon_F32']]+[Path(s['file']['path']) for s in p['splits']]
    paths += list((SITE/'torch/lib').glob('*.dll'))+list((SITE/'numpy.libs').glob('*.dll'))+list((SITE/'psutil').glob('*.pyd'))
    b=dict(schema='NATIVE_HEAD_GPU_PREFLIGHT_BINDING_V1',python=str(Path(sys.executable).resolve()),preflight=p,prior=prior,gpu=gpu_metadata(),
           versions=dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda),
           threads=6,batch=64,warmups=2,repeats=7,numeric=dict(loss_absolute=1e-8,gradient_absolute=1e-8,gradient_relative=1e-9,argmax_changes=0),
           limits=dict(seconds=600,OS_bytes=4<<30,GPU_allocated_bytes=2<<30,GPU_reserved_bytes=2<<30,output_bytes=64<<20),
           inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
           scope='All8808 qualified CPU initial codes/moments/entropies/losses/gradients and current actual F32 head promoted F64. '
                 'Local GPU float64 forward loss/gradient qualification/cost only,no optimization/native C/source model/T4/RESERVED/longalign. '
                 'Existing CPU-v-native1e-5 flags remain false. Desktop GPU sharing/timing unqualified,no useful-n/50 claim.')
    write(a.out,b); print(json.dumps(dict(binding=extent(a.out),gpu=b['gpu'],seconds=time.monotonic()-START)),flush=True)


def run(a):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np,psutil,torch
    assert extent(a.binding)['sha256']==a.binding_sha; b=json.loads(a.binding.read_bytes())
    assert Path(sys.executable).resolve()==Path(b['python']).resolve() and gpu_metadata()==b['gpu']
    assert all(os.environ.get(k)=='6' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']) and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert dict(torch=torch.__version__,numpy=np.__version__,psutil=psutil.__version__,cuda=torch.version.cuda)==b['versions']
    assert not a.directory.exists() and not a.out.exists()
    proc=psutil.Process(); proc.cpu_affinity(list(range(6))); torch.set_num_threads(6); torch.set_num_interop_threads(1)
    torch.manual_seed(0); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    receipt=dict(schema='NATIVE_HEAD_GPU_PREFLIGHT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def guard():
        lim=b['limits']; assert time.monotonic()-START<=lim['seconds'] and proc.memory_info().peak_wset<=lim['OS_bytes'] and not proc.children(recursive=True)
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        if a.directory.exists():assert sum(p.stat().st_size for p in a.directory.iterdir())<=lim['output_bytes']
    def check():
        for e in b['inputs']:guard(); assert extent(e['path'])==e,e['path']
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    try:
        torch.cuda.init(); torch.cuda.reset_peak_memory_stats(); assert torch.cuda.get_device_name(0)==b['gpu']['name']; check(); a.directory.mkdir()
        nr=b['prior']['native_report']; C=np.fromfile(nr['head_F32']['path'],'<f4').reshape(65537,256).astype('f8'); beta=np.fromfile(nr['bias_F32']['path'],'<f4').astype('f8')
        epsilon=float(np.fromfile(nr['epsilon_F32']['path'],'<f4')[0]); A=np.column_stack([C,beta/math.sqrt(256*epsilon)])
        matrix=torch.from_numpy(A).to('cuda'); assert matrix.dtype==torch.float64; assert not matrix.requires_grad
        def fg(v,m,H):
            assert v.dtype==m.dtype==H.dtype==matrix.dtype==torch.float64
            scores=v@matrix.T; logz=torch.logsumexp(scores,dim=1); probability=torch.softmax(scores,dim=1)
            return logz-(m*v).sum(dim=1)-H,probability@matrix-m,scores.argmax(dim=1)
        outputs=[]; splits=[]; benchmark=None; calls=0
        with torch.inference_mode():
            for s in b['preflight']['splits']:
                old=load(s['file']); n=len(old['v']); values=np.empty(n,'f8'); gradients=np.empty((n,257),'f8'); ids=np.empty(n,'i8'); batches=[]
                for lo in range(0,n,64):
                    guard(); hi=min(lo+64,n); v=torch.from_numpy(old['v'][lo:hi]).to('cuda'); m=torch.from_numpy(old['moments'][lo:hi]).to('cuda'); H=torch.from_numpy(old['entropy'][lo:hi]).to('cuda')
                    torch.cuda.synchronize(); began=time.monotonic(); result=fg(v,m,H); torch.cuda.synchronize(); seconds=time.monotonic()-began; calls+=1
                    values[lo:hi]=result[0].cpu().numpy(); gradients[lo:hi]=result[1].cpu().numpy(); ids[lo:hi]=result[2].cpu().numpy()
                    batches.append(dict(lo=lo,hi=hi,FG_seconds=seconds))
                    if s['split']=='FIT' and lo==0:
                        times=[]
                        for repeat in range(b['warmups']+b['repeats']):
                            guard(); torch.cuda.synchronize(); began=time.monotonic(); again=fg(v,m,H); torch.cuda.synchronize(); duration=time.monotonic()-began; calls+=1
                            assert all(torch.equal(x,y) for x,y in zip(result,again))
                            if repeat>=b['warmups']:times.append(duration)
                        benchmark=dict(batch=hi-lo,GPU_seconds=times,median_seconds=float(np.median(times)),nominal139_batch_evaluations_seconds=float(np.median(times))*139,
                                       CPU_prior_median_seconds=b['preflight']['benchmark']['median_seconds'],CPU_prior_nominal_seconds=b['preflight']['benchmark']['nominal139_batch_evaluations_seconds'],
                                       admission=False,scope='Synchronized resident-batch component calls,desktop sharing;not optimizer/end-to-end speed.')
                loss_delta=float(np.max(np.abs(values-old['loss']))); grad_delta=float(np.max(np.abs(gradients-old['gradient']))); grad_relative=float(np.linalg.norm(gradients-old['gradient'])/max(1.,float(np.linalg.norm(old['gradient'])))); id_changes=int(np.count_nonzero(ids!=old['ids']))
                numeric=dict(loss_absolute=loss_delta,gradient_absolute=grad_delta,gradient_relative=grad_relative,argmax_changes=id_changes)
                flags={k:numeric[k]<=b['numeric'][k] for k in numeric}
                path=a.directory/(s['split']+'_GPU_initial.npz'); np.savez(path,loss=values,gradient=gradients,ids=ids); outputs.append(extent(path))
                splits.append(dict(split=s['split'],rows=n,file=outputs[-1],numeric=numeric,flags=flags,batches=batches))
                print(json.dumps(dict(split=s['split'],rows=n,flags=flags,seconds=time.monotonic()-START)),flush=True)
        check(); guard(); assert gpu_metadata()==b['gpu']
        result=dict(schema='NATIVE_HEAD_GPU_PREFLIGHT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,gpu=b['gpu'],versions=b['versions'],outputs=outputs,splits=splits,benchmark=benchmark,
                    calls=calls,all148_calls_float64=True,repeated_outputs_bit_exact=True,before_after_hashes_exact=True,
                    decision='GPU_INITIAL_NUMERIC_PASS_PENDING_INDEPENDENT_AUDIT' if all(all(s['flags'].values()) for s in splits) else 'COMPLETED_GPU_INITIAL_NUMERIC_FAILURE',
                    seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
                    optimization_implemented=False,timing_admitted=False,scope=b['scope'])
        assert calls==148
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved()); write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',action='store_true')
    for name in ['binding','directory','out']:p.add_argument('--'+name,type=Path)
    for name in ['binding-sha','freeze']:p.add_argument('--'+name)
    a=p.parse_args(); assert a.prepare!=a.run
    prepare(a) if a.prepare else run(a)
