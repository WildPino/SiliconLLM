"""FIT-only convex image of an existing original-engine RMS/readout.

No donor/model forward or parameter update. A feasible feature is not a model.
The first-order ball bound is mathematical, evaluated in F64, not interval arithmetic.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))
FIELD = struct.Struct('<64s6I2Q')


def extent(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def check_inputs(binding):
    for item in binding['inputs']:
        p = Path(item['path'])
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], p


def bind(a):
    corpus = ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    records = sorted((r for r in json.loads(corpus.read_bytes())['records'] if r['split'] == 'FIT'), key=lambda r:r['id'])
    assert len(records) == 24 and len({r['domain'] for r in records}) == 12
    selection = []
    for r in records:
        n = len(r['positions']); indices = [0, n//2, n-1]
        assert len(set(indices)) == 3 and r['logits']['shape'] == [n,65537]
        assert sha(r['logits']['path']) == r['logits']['sha256']
        selection.append(dict(id=r['id'],domain=r['domain'],label_indices=indices,
            positions=[r['positions'][j] for j in indices],teacher=r['logits']))
    terminal = DOC/'original_falcon_transfer_result_20261009.terminal.json'
    old = json.loads(terminal.read_bytes())
    old_result = DOC/'original_falcon_transfer_result_20261009.json'
    assert old['exit_code'] == 0 and sha(old_result) == old['result_sha256']
    outputs = {Path(i['path']).name:i for i in old['output_files']}
    for name in ('candidate.packed','native.f32'):
        i=outputs[name]; assert extent(i['path']) == i
    files = [Path(__file__),B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
        DOC/'ORIGINAL_READOUT_INFORMATION_PROTOCOL_20261009.md',corpus,terminal,old_result,
        Path(outputs['candidate.packed']['path']),Path(outputs['native.f32']['path']),Path(sys.executable)]
    files += [Path(r['teacher']['path']) for r in selection]
    files += [SITE/'torch'/p for p in ('__init__.py','_C.cp312-win_amd64.pyd','lib/torch_cpu.dll','lib/torch_cuda.dll','cuda/__init__.py')]
    files += [SITE/'numpy/__init__.py',SITE/'psutil/__init__.py']
    files += [ROOT/p for p in ('benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    write(a.out,dict(schema='ORIGINAL_READOUT_INFORMATION_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),packed=outputs['candidate.packed'],native=outputs['native.f32'],
        selected=selection,inputs=[extent(p) for p in dict.fromkeys(files)],
        solver=dict(radius=16.,scales=[1.,math.sqrt(8.)],max_iterations=512,max_backtracks=30,
            initial_L=1.,L_floor=1e-12,gap_tolerance=1e-5,majorant_slack=1e-12,certificate_slack=1e-8),
        decisions=dict(expressible_mean_KL=1.,fixed_image_lower_mean_KL=1.,scale_upper_to_current_lower=.75),
        limits=dict(seconds=600,reserve_seconds=45,OS_bytes=4<<30,GPU_allocated_bytes=3<<30,
            GPU_reserved_bytes=4<<30,output_bytes=8<<20),
        runtime_binding_scope='Existing packed head/final_norm,24 FIT full source packets,existing longest-FIT native output;principal Torch/NumPy/psutil and Python binaries,not full DLL tree.'))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),labels=72,inputs=len(files))),flush=True)


def launch(a):
    import psutil
    from chatbot_falcon_usability_launch import Memory
    start=time.monotonic(); assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes()); assert b['schema']=='ORIGINAL_READOUT_INFORMATION_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(v.pid for v in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','clang','engine','packed_original')),('overlap',p.pid,name)
    log=a.out.with_suffix('.worker.log');terminal=a.out.with_suffix('.terminal.json')
    assert not a.directory.exists() and not any(p.exists() for p in (log,terminal,a.out,a.out.with_suffix('.launcher_failure.json')))
    record=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_pid=proc.pid)
    worker=None;peak=0
    getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];getmem.restype=wintypes.BOOL
    def memory():
        nonlocal peak
        m=Memory();m.cb=ctypes.sizeof(m)
        assert getmem(wintypes.HANDLE(int(worker._handle)),ctypes.byref(m),m.cb)
        peak=max(peak,m.PeakWorkingSetSize)
    def guard():
        assert time.monotonic()-start<=b['limits']['seconds'],'family deadline'
        assert peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family OS cap'
        assert log.stat().st_size<=4<<20,'log cap'
    try:
        check_inputs(b)
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        argv=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),
            '--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
        record['command']=argv
        with log.open('xb') as f:
            worker=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=8)
            record['worker_pid']=worker.pid;record['worker_creation_time']=psutil.Process(worker.pid).create_time();position=0
            while worker.poll() is None:
                memory();guard()
                try:assert not psutil.Process(worker.pid).children(recursive=True),'unexpected child'
                except psutil.NoSuchProcess:pass
                with log.open('rb') as src:
                    src.seek(position);chunk=src.read();boundary=chunk.rfind(b'\n')+1
                    if boundary:print(chunk[:boundary].decode('utf8',errors='replace'),end='',flush=True);position+=boundary
                time.sleep(.1)
        memory();guard();assert worker.returncode==0,log.read_text(errors='replace')
        check_inputs(b);r=json.loads(a.out.read_bytes())
        assert r['schema']=='ORIGINAL_READOUT_INFORMATION_RESULT_V1' and r['labels']==72 and len(r['arms'])==2
        assert r['new_model_forwards']==r['new_teacher_queries']==r['new_optimizer_updates']==0
        assert not r['quality_admission'] and r['GPU_allocated_peak']<=b['limits']['GPU_allocated_bytes']
        assert r['GPU_reserved_peak']<=b['limits']['GPU_reserved_bytes']
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        assert sum(i['bytes'] for i in outputs)<=b['limits']['output_bytes']
        record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak,
            launcher_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            result_sha256=sha(a.out),output_files=outputs,resource_gates=True)
        write(terminal,record);print(json.dumps(record),flush=True)
    except BaseException as error:
        if worker is not None:
            if worker.poll() is None:worker.kill();worker.wait()
            memory();record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak)
        record.update(fault=repr(error),elapsed_seconds=time.monotonic()-start)
        write(a.out.with_suffix('.launcher_failure.json'),record);raise


def worker(a):
    start=time.monotonic(); assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='ORIGINAL_READOUT_INFORMATION_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    import numpy as np
    import psutil
    import torch
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.cuda.reset_peak_memory_stats()
    lim=b['limits'];solver=b['solver'];R=solver['radius'];N=72;V=65537;D=256
    def guard():
        assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds'],'worker reserve'
        assert proc.memory_info().peak_wset<=lim['OS_bytes'],'worker OS cap'
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'],'GPU allocated cap'
        assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU reserved cap'
    def event(**values):
        guard();print(json.dumps(dict(seconds=time.monotonic()-start,**values)),flush=True)
    p=Path(b['packed']['path'])
    with p.open('rb') as f:
        header=f.read(80);assert header[:8]==b'E4BPv001'
        h=struct.unpack('<16I',header[8:72]);assert h[:3]==(1,V,D) and h[15]==110
        assert struct.unpack('<Q',header[72:])[0]==p.stat().st_size
        entries={}
        for _ in range(110):
            name,dtype,ndim,d0,d1,d2,d3,offset,size=FIELD.unpack(f.read(104))
            entries[name.rstrip(b'\0').decode()]=(dtype,(d0,d1,d2,d3)[:ndim],offset,size)
    def field(name,shape):
        dtype,actual,offset,size=entries[name];assert dtype==1 and actual==shape and size==4*math.prod(shape)
        assert offset+size<=p.stat().st_size
        return np.memmap(p,dtype='<f4',mode='r',offset=offset,shape=shape).astype('f8')
    head=field('head',(V,D));gamma=field('final_norm',(D,))
    assert np.isfinite(head).all() and np.isfinite(gamma).all()
    A=torch.as_tensor(head*gamma,device='cuda',dtype=torch.float64)
    # Common additive vocabulary logit is probability-invariant in real arithmetic.
    A-=A.mean(0,keepdim=True)
    labels=[];teacher=[];actual=[]
    native=np.memmap(b['native']['path'],dtype='<f4',mode='r',shape=(1507,V))
    for rec in b['selected']:
        raw=np.memmap(rec['teacher']['path'],dtype='<u2',mode='r',shape=tuple(rec['teacher']['shape']))
        for j,pos in zip(rec['label_indices'],rec['positions']):
            row=(raw[j].astype('<u4')<<16).view('<f4').astype('f8');assert np.isfinite(row).all()
            teacher.append(row);labels.append(dict(id=rec['id'],domain=rec['domain'],label_index=j,position=pos))
            if rec['id']=='broad_fit_smol_magpie_ultra_022':
                logq=row-np.max(row);logq-=np.log(np.exp(logq).sum())
                z=native[pos].astype('f8');z-=z.max();z-=np.log(np.exp(z).sum())
                actual.append(dict(label_index=len(labels)-1,KL=float(np.sum(np.exp(logq)*(logq-z))),
                    native_argmax=int(np.argmax(native[pos])),teacher_argmax=int(np.argmax(row))))
        del raw
    assert len(labels)==N and len(actual)==3
    qlog=torch.log_softmax(torch.as_tensor(np.stack(teacher),dtype=torch.float64,device='cuda'),dim=-1)
    q=qlog.exp();negH=(q*qlog).sum(-1);uniform=math.log(V)+negH
    assert float((q.sum(-1)-1).abs().max())<1e-12
    del teacher,head
    event(stage='loaded',labels=N,uniform_mean=float(uniform.mean()))
    arms=[];solutions={}
    with torch.inference_mode():
        for arm,scale in enumerate(solver['scales']):
            matrix=A*scale;target=q@matrix
            calls=0
            def fg(u):
                nonlocal calls
                calls+=1;guard();z=u@matrix.T;logp=torch.log_softmax(z,dim=-1)
                loss=negH-(q*logp).sum(-1);grad=logp.exp()@matrix-target
                assert torch.isfinite(loss).all() and torch.isfinite(grad).all()
                return loss,grad
            def project(u):return u*(R/u.norm(dim=-1,keepdim=True).clamp_min(R))
            u=torch.zeros((N,D),dtype=torch.float64,device='cuda');loss,grad=fg(u)
            assert float((loss-uniform).abs().max())<1e-10
            upper=loss.clone();lower=torch.zeros_like(loss);best=u.clone()
            steps=torch.full_like(loss,solver['initial_L']);trace=[];converged=False;iteration=0
            for iteration in range(solver['max_iterations']+1):
                bound=loss-(grad*u).sum(-1)-R*grad.norm(dim=-1)
                lower=torch.maximum(lower,bound);improved=loss<upper
                best[improved]=u[improved];upper=torch.minimum(upper,loss)
                assert float((lower-upper).max())<=solver['certificate_slack'],'inconsistent F64 bounds'
                gap=(upper-lower).clamp_min(0)
                converged=bool((gap<=solver['gap_tolerance']).all())
                if iteration%16==0 or converged or iteration==solver['max_iterations']:
                    rec=dict(iteration=iteration,upper_mean=float(upper.mean()),lower_mean=float(lower.mean()),
                        max_gap=float(gap.max()),mean_gap=float(gap.mean()),converged_labels=int((gap<=solver['gap_tolerance']).sum()))
                    trace.append(rec);event(stage='solve',arm=arm,scale=scale,**rec)
                if converged or iteration==solver['max_iterations']:break
                steps=(steps*.5).clamp_min(solver['L_floor'])
                for backtrack in range(solver['max_backtracks']):
                    candidate=project(u-grad/steps[:,None]);delta=candidate-u
                    next_loss,next_grad=fg(candidate)
                    majorant=loss+(grad*delta).sum(-1)+.5*steps*(delta*delta).sum(-1)
                    accepted=next_loss<=majorant+solver['majorant_slack']
                    if bool(accepted.all()):break
                    steps[~accepted]*=2.
                else:raise RuntimeError('backtracking cap')
                assert float((next_loss-loss).max())<=1e-9,'projected descent violation'
                u,loss,grad=candidate,next_loss,next_grad
            final_loss,final_grad=fg(best)
            assert float((final_loss-upper).abs().max())<1e-9
            assert float(best.norm(dim=-1).max())<=R+1e-10
            final_bound=final_loss-(final_grad*best).sum(-1)-R*final_grad.norm(dim=-1)
            lower=torch.maximum(lower,final_bound);gap=(upper-lower).clamp_min(0)
            assert float((lower-upper).max())<=solver['certificate_slack']
            logits=best@matrix.T;ids=logits.argmax(-1);teacher_ids=q.argmax(-1)
            arm_result=dict(scale=scale,iterations=iteration,objective_gradient_calls=calls,
                converged=bool((gap<=solver['gap_tolerance']).all()),converged_labels=int((gap<=solver['gap_tolerance']).sum()),
                upper_mean=float(upper.mean()),lower_mean=float(lower.mean()),mean_gap=float(gap.mean()),max_gap=float(gap.max()),
                per_label_upper=upper.cpu().tolist(),per_label_lower=lower.cpu().tolist(),
                per_label_gap=gap.cpu().tolist(),feature_norms=best.norm(dim=-1).cpu().tolist(),
                optimistic_argmax_disagreement=int((ids!=teacher_ids).sum()),trace=trace)
            arms.append(arm_result);solutions['features_'+str(arm)]=best.cpu().numpy()
            event(stage='arm_done',arm=arm,upper=arm_result['upper_mean'],lower=arm_result['lower_mean'],calls=calls)
        solutions['head_gamma']=gamma;solutions['uniform_KL']=uniform.cpu().numpy()
        with (a.directory/'solutions.npz').open('xb') as f:np.savez(f,**solutions)
    dc=b['decisions'];flags=dict(current_feasible_mean_KL_le_1=arms[0]['upper_mean']<=dc['expressible_mean_KL'],
        current_fixed_image_mean_KL_gt_1=arms[0]['lower_mean']>dc['fixed_image_lower_mean_KL'],
        scaled_feasible_loss_le_75pct_current_lower=arms[1]['upper_mean']<=dc['scale_upper_to_current_lower']*arms[0]['lower_mean'])
    guard();result=dict(schema='ORIGINAL_READOUT_INFORMATION_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
        labels=N,label_metadata=labels,arms=arms,decision_flags=flags,teacher_entropy_mean=float((-negH).mean()),
        uniform_KL_mean=float(uniform.mean()),uniform_KL_per_label=uniform.cpu().tolist(),
        actual_native_matched_labels=actual,actual_native_scope='Only3 selected labels of existing1507-history case;no outputs for other23 cases.',
        numerical_scope='F64 real-arithmetic convex tangent bound;not interval-certified or native F32/whole-model quality.',
        GPU_name=torch.cuda.get_device_name(),GPU_allocated_peak=torch.cuda.max_memory_allocated(),
        GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,
        elapsed_seconds=time.monotonic()-start,new_model_forwards=0,new_teacher_queries=0,new_optimizer_updates=0,
        quality_admission=False,historical_native_numerical_gate='FAIL retained')
    write(a.out,result);event(stage='complete',decision_flags=flags,result_sha256=sha(a.out))


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bind',action='store_true');mode.add_argument('--launch',action='store_true');mode.add_argument('--worker',action='store_true')
    parser.add_argument('--binding',type=Path);parser.add_argument('--binding-sha');parser.add_argument('--freeze')
    parser.add_argument('--directory',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.bind:bind(args)
    elif args.launch:launch(args)
    else:worker(args)
