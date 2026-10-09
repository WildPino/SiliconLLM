"""Finite direct-master export / original-operator learner bridge, no donor calls."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))
from original_packed_capacity import extent,extract,raw


def bind(a):
    import numpy as np
    import psutil
    old=ROOT/'results/native_expert_scaling/original_packed_capacity_20261009'
    compiler=Path(json.loads((DOC/'original_packed_capacity_binding_20261009.json').read_bytes())['compiler'])
    checkpoint=ROOT/'results/phase57/moe_gran.pt'
    files=[Path(__file__),B/'original_tensor_learner.py',B/'original_tensor_bank_probe.c',
        B/'original_packed_capacity.py',B/'original_packed_capacity.c',B/'chatbot_falcon_usability.py',
        B/'chatbot_falcon_usability_launch.py',DOC/'ORIGINAL_TENSOR_LEARNER_PROTOCOL_20261009.md',
        checkpoint,ROOT/'results/phase55/ids.u16',ROOT/'benchmarks/phase60/engine.c',
        old/'original_e32.packed',old/'small_queries.u32',old/'small.f32',old/'small.routes',
        old/'original_bodies.json',old/'original_packed_bodies.h',old/'packed_original.exe',
        DOC/'original_packed_capacity_result_20261009.json',Path(sys.executable),Path(np.__file__),
        Path(psutil.__file__),SITE/'torch/__init__.py',SITE/'torch/_C.cp312-win_amd64.pyd',
        SITE/'torch/lib/torch_cpu.dll',compiler]
    files += [p for p in compiler.parent.iterdir() if p.name in ('clang-21.exe','ld.lld.exe','lld.exe','libclang-cpp.dll','libLLVM.dll')]
    files += [ROOT/p for p in ('benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    assert sha(checkpoint)=='356478b2f63ace9d6ec429056fee5c0e14e1c0f36253edbca36300eba02d4525'
    write(a.out,dict(schema='ORIGINAL_TENSOR_LEARNER_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),old_directory=str(old),checkpoint=str(checkpoint),
        compiler=str(compiler),inputs=[extent(p) for p in files],
        limits=dict(seconds=240,reserve_seconds=20,OS_bytes=3<<30,output_bytes=1<<30),
        allowed_worker_children=['clang.exe','clang-21.exe','ld.lld.exe','lld.exe','conhost.exe','tensor_bank.exe','packed_original.exe'],
        allowed_system_child_path='C:/Windows/System32/conhost.exe',
        runtime_binding_scope='CPU Torch2.6.0+cu124, NumPy2.4.6, psutil7.2.2; principal CPU binary/code and reused native artifacts bound; no GPU execution.',
        gates=dict(relative_row_RMS=1e-4,mass_abs=1e-6,ids_exact=True,zero_exact=True),
        updates_max=1,inputs_count=16,seed=1709))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='ORIGINAL_TENSOR_LEARNER_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    a.directory.mkdir(exist_ok=False)
    import numpy as np
    import psutil
    import torch
    from original_tensor_learner import Learner,export,quant_weight,D,EH,K,L
    from chatbot_falcon_usability_launch import Memory
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(b['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    children=[];max_child=0;stage='start';last_checkpoint=None;updates=0
    old=Path(b['old_directory'])
    def guard(reserve=False):
        assert time.monotonic()-start<=b['limits']['seconds']-(b['limits']['reserve_seconds'] if reserve else 0),'worker deadline'
        assert proc.memory_info().peak_wset+max_child<=b['limits']['OS_bytes'],'worker/direct child OS cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=b['limits']['output_bytes'],'output cap'
    def event(**values):
        guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**values)),flush=True)
    def save(name,value):
        path=a.directory/name
        with path.open('xb') as f:torch.save(value,f)
        guard();return extent(path)
    def run(argv,name):
        nonlocal max_child
        t=time.monotonic();log=a.directory/(name+'.log')
        getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];getmem.restype=wintypes.BOOL
        peak=0
        with log.open('xb') as f:
            p=subprocess.Popen([str(v) for v in argv],stdout=f,stderr=subprocess.STDOUT,creationflags=8)
            def sample():
                nonlocal peak,max_child
                m=Memory();m.cb=ctypes.sizeof(m)
                assert getmem(wintypes.HANDLE(int(p._handle)),ctypes.byref(m),m.cb)
                peak=max(peak,m.PeakWorkingSetSize);max_child=max(max_child,peak)
            try:
                while p.poll() is None:sample();guard();time.sleep(.05)
                sample()
            except BaseException:
                if p.poll() is None:p.kill();p.wait()
                sample();raise
            finally:
                rec=dict(command=[str(v) for v in argv],pid=p.pid,exit_code=p.returncode,
                    held_OS_peak=peak,seconds=time.monotonic()-t,log=extent(log))
                children.append(rec);write(a.directory/(name+'.receipt.json'),rec)
        assert p.returncode==0,(name,p.returncode,log.read_text(errors='replace'));guard()
    def arrays(prefix,logits,routes):
        raw(a.directory/(prefix+'.learner.f32'),logits.detach().numpy().astype('<f4').tobytes())
        data=bytearray()
        for t in range(logits.shape[-2]):
            for ids,mass in routes:
                data.extend(ids[t].numpy().astype('<i4').tobytes());data.extend(mass[t].numpy().astype('<f4').tobytes())
        raw(a.directory/(prefix+'.learner.routes'),data)
    def compare(prefix,pred,reference,routes,route_bytes):
        pred=np.asarray(pred,dtype=np.float64);reference=np.asarray(reference,dtype=np.float64)
        assert pred.shape==reference.shape and np.all(np.isfinite(pred)) and np.all(np.isfinite(reference))
        rows=[];route_records=[];ok=True
        for t in range(pred.shape[0]):
            delta=pred[t]-reference[t];rms=float(np.sqrt(np.mean(delta**2)))
            norm=max(float(np.sqrt(np.mean(reference[t]**2))),1e-8)
            rel=rms/norm;greedy=int(np.argmax(pred[t]))==int(np.argmax(reference[t]))
            rows.append(dict(row=t,absolute_RMS=rms,reference_RMS=norm,relative_RMS=rel,
                max_abs=float(np.max(np.abs(delta))),greedy_agreement=greedy))
            ok &= rel<=b['gates']['relative_row_RMS'] and (greedy if prefix!='bank' else True)
        steps=routes[0][0].shape[0];assert len(route_bytes)==steps*L*64
        for t in range(steps):
            for l,(ids,mass) in enumerate(routes):
                off=(t*L+l)*64
                ni=np.frombuffer(route_bytes,dtype='<i4',count=K,offset=off)
                nm=np.frombuffer(route_bytes,dtype='<f4',count=K,offset=off+32)
                pi=ids[t].numpy();pm=mass[t].numpy()
                exact=bool(np.array_equal(ni,pi));md=float(np.max(np.abs(nm.astype('f8')-pm)))
                defect=abs(float(np.sum(pm,dtype='f8'))-1)
                route_records.append(dict(position=t,layer=l,IDs_exact=exact,mass_max_abs=md,mass_sum_defect=defect))
                ok &= exact and md<=b['gates']['mass_abs'] and defect<=1e-6
        value=dict(pass_gate=bool(ok),rows=rows,routes=route_records)
        write(a.directory/(prefix+'.comparison.json'),value);return value
    try:
        stage='adopt_real_masters'
        packet=torch.load(b['checkpoint'],map_location='cpu',weights_only=True)
        model=Learner(32,1024);model.adopt_phase57(packet);del packet
        parameters=sum(p.numel() for p in model.parameters());tensors=sum(1 for p in model.parameters())
        stage='direct_master_export';export_info=export(model,a.directory/'masters.packed')
        assert sha(a.directory/'masters.packed')==sha(old/'original_e32.packed'),'direct master export differs'
        event(bytes=export_info['bytes'],parameters=parameters,tensors=tensors)
        stage='extract_compile_bank_probe';body=extract(a.directory);exe=a.directory/'tensor_bank.exe'
        run([b['compiler'],'-O3','-mavx2','-mfma','-march=znver2',B/'original_tensor_bank_probe.c','-I',a.directory,'-o',exe,'-lm'],'compile_bank')
        stage='bank_forward'
        i=torch.arange(D,dtype=torch.float32)
        rng=torch.Generator().manual_seed(b['seed'])
        inputs=torch.stack([torch.zeros(D),(i.remainder(127)-63)/63,
                            ((7*i+11).remainder(127)-63)/31,torch.randn(D,generator=rng)])
        raw(a.directory/'bank_inputs.f32',struct.pack('<I',4)+inputs.numpy().astype('<f4').tobytes())
        run([exe,a.directory/'masters.packed',a.directory/'bank_inputs.f32',a.directory/'bank.native.f32',a.directory/'bank.native.routes'],'bank_native')
        with torch.no_grad():
            bank_outputs=[layer.bank(inputs) for layer in model.layers]
            routes=[layer.bank.last_routes for layer in model.layers]
        bank=torch.stack(bank_outputs,1).reshape(4*L,D)
        raw(a.directory/'bank.learner.f32',bank.numpy().astype('<f4').tobytes())
        bank_routes=bytearray()
        for t in range(4):
            for ids,mass in routes:
                bank_routes.extend(ids[t].numpy().astype('<i4').tobytes())
                bank_routes.extend(mass[t].numpy().astype('<f4').tobytes())
        raw(a.directory/'bank.learner.routes',bank_routes)
        bank_ref=np.fromfile(a.directory/'bank.native.f32',dtype='<f4').reshape(4*L,D)
        bank_result=compare('bank',bank.numpy(),bank_ref,routes,(a.directory/'bank.native.routes').read_bytes())
        zero_exact=bool(np.array_equal(bank.numpy()[:L],bank_ref[:L]) and np.count_nonzero(bank_ref[:L])==0)
        tied=model.layers[0].bank
        w0,b0=tied.router.weight.detach().clone(),tied.router.bias.detach().clone()
        with torch.no_grad():
            tied.router.weight.zero_();tied.router.bias.zero_();tied(inputs[:1])
            ti,tm=tied.last_routes
            tie_exact=bool(torch.equal(ti[0],torch.arange(K)) and torch.equal(tm[0],torch.full((K,),1/K)))
            tied.router.weight.copy_(w0);tied.router.bias.copy_(b0)
        assert tie_exact and zero_exact,'tie/zero contract'
        event(pass_gate=bank_result['pass_gate'],zero_exact=zero_exact,tie_exact=tie_exact)
        stage='whole_forward'
        q=(old/'small_queries.u32').read_bytes();assert struct.unpack_from('<I',q)[0]==64
        tokens=np.frombuffer(q,dtype='<u4',offset=4,count=16).copy();ids=torch.from_numpy(tokens.astype('i8'))[None,:]
        raw(a.directory/'queries.u32',struct.pack('<I',16)+tokens.astype('<u4').tobytes())
        with torch.no_grad():logits=model(ids)[0]
        routes=[layer.bank.last_routes for layer in model.layers];arrays('before',logits,routes)
        ref=np.fromfile(old/'small.f32',dtype='<f4',count=16*1024).reshape(16,1024)
        before=compare('before',logits.numpy(),ref,routes,(old/'small.routes').read_bytes()[:16*L*64])
        event(pass_gate=before['pass_gate'],max_relative_RMS=max(r['relative_RMS'] for r in before['rows']))
        after=None;integer_coordinates=0;gradients=None;checkpoint_info=None
        if bank_result['pass_gate'] and before['pass_gate']:
            stage='one_update';guard(True)
            source=ROOT/'results/phase55/ids.u16';n=source.stat().st_size//2
            with source.open('rb') as f:f.seek(int(n*.9)*2);labels=np.frombuffer(f.read(17*2),dtype='<u2').astype('i8')
            assert np.array_equal(labels[:16],tokens)
            target=torch.from_numpy(labels[1:])[None,:]
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
            model.use_checkpoint=True;optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(ids).reshape(-1,1024),target.flatten())
            loss.backward();groups={}
            for name,p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
                group='bank' if any(name.endswith('.'+key) for key in ('gate','up','down')) else ('router' if '.router.' in name else 'organ')
                groups[group]=groups.get(group,0)+float(p.grad.double().square().sum())
            assert set(groups)=={'organ','bank','router'} and min(groups.values())>0
            gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step();updates=1
            for p in model.parameters():assert torch.isfinite(p).all()
            for state in optimizer.state.values():
                assert int(state['step'])==1 and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
            gradients=dict(group_squared_norm_before_clip=groups,global_norm_before_clip=float(gradnorm),loss=float(loss.detach()))
            checkpoint_info=save('candidate.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(),updates=updates,input_IDs=tokens.tolist(),labels=labels[1:].tolist(),gradients=gradients,
                provenance='Original small-vocabulary master bridge only; no donor training history.'))
            last_checkpoint=checkpoint_info;event(checkpoint=checkpoint_info)
            stage='changed_export';changed=export(model,a.directory/'updated.packed');assert sha(a.directory/'updated.packed')!=sha(a.directory/'masters.packed')
            stage='changed_native';run([old/'packed_original.exe',a.directory/'updated.packed',a.directory/'queries.u32',
                a.directory/'after.native.f32',a.directory/'after.native.routes',a.directory/'after.witness',a.directory/'after.native.json'],'after_native')
            with torch.no_grad():logits=model(ids)[0]
            routes=[layer.bank.last_routes for layer in model.layers];arrays('after',logits,routes)
            after=compare('after',logits.numpy(),np.fromfile(a.directory/'after.native.f32',dtype='<f4').reshape(16,1024),
                routes,(a.directory/'after.native.routes').read_bytes())
            candidates=[0,31];expected=[]
            for e in candidates:
                for layer in model.layers:
                    for name,length in [('gate',D),('up',D),('down',EH)]:
                        w=getattr(layer.bank,name)[e];qw,_=quant_weight(w)
                        qi=np.arange(length,dtype='i8');qi=(qi%127-63) if name!='down' else ((7*qi)%127-63)
                        expected.append(qw.numpy().astype('i8') @ qi)
            actual=np.fromfile(a.directory/'after.witness',dtype='<i4');expected=np.concatenate(expected)
            assert np.array_equal(actual,expected),'new integer witnesses';integer_coordinates=actual.size
            event(pass_gate=after['pass_gate'],integer_coordinates=integer_coordinates)
        stage='complete';guard()
        passed=bool(bank_result['pass_gate'] and before['pass_gate'] and after and after['pass_gate'])
        result=dict(schema='ORIGINAL_TENSOR_LEARNER_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
            decision='ORIGINAL_TENSOR_LEARNER_PASS' if passed else 'ORIGINAL_TENSOR_LEARNER_NUMERICAL_GATE_FAIL',
            export=export_info,direct_master_export_bit_exact=True,parameters=parameters,parameter_tensors=tensors,
            zero_exact=zero_exact,tie_exact=tie_exact,bank=bank_result,before=before,after=after,
            optimizer_updates=updates,checkpoint=checkpoint_info,gradients=gradients,integer_coordinates=integer_coordinates,
            source_calls=0,GPU_calls=0,source_informed_learner_available=False,quality_admission=False,
            useful_large_n_admission=False,speed_admission=False,original_bodies=body,children=children,
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=max_child,
            resources_scope='CPU worker held by launcher; direct child handles held through exit; nested compiler children not separately held.',
            elapsed_seconds=time.monotonic()-start)
        write(a.out,result);event(decision=result['decision'],updates=updates)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(stage=stage,fault=repr(error),completed_updates=updates,
            checkpoint=last_checkpoint,children=children,elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['bind'],default=None)
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
