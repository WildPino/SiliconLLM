"""Frozen source initialization, streamed-adjoint qualification and one whole step."""
import argparse
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))
from original_packed_capacity import extent,raw


def bind(a):
    source=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    package=json.loads((source/'source_package.json').read_bytes())
    assert package['revision']=='80ebc50d7799a440b96c93bb6686a3924a09b0cb'
    corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    fit=sorted([r for r in json.loads(corpus.read_bytes())['records'] if r['split']=='FIT'],
               key=lambda r:(len(r['student_input_ids']),r['id']))
    rec=fit[-1];assert rec['id']=='broad_fit_smol_magpie_ultra_022' and len(rec['student_input_ids'])==1507
    assert len(rec['positions'])==len(rec['output_ids'])==256 and rec['logits']['shape']==[256,65537]
    basis=ROOT/'results/native_expert_scaling/chatbot_hybrid_pilot_repair1_20261008/basis.pt'
    native=ROOT/'results/native_expert_scaling/original_packed_capacity_20261009/packed_original.exe'
    files=[Path(v['path']) for v in package['files']]
    files += [source/'source_package.json',basis,corpus,Path(rec['logits']['path']),native,
        ROOT/'results/phase57/moe_gran.pt',ROOT/'benchmarks/phase60/engine.c',Path(sys.executable),
        Path(__file__),B/'original_falcon_learner.py',B/'original_tensor_learner.py',
        B/'original_packed_capacity.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
        DOC/'ORIGINAL_FALCON_TRANSFER_PROTOCOL_20261009.md',DOC/'original_tensor_learner_result_20261009.json',
        DOC/'chatbot_broad_adoption_result_20261009.json',DOC/'chatbot_broad_adoption_result_20261009.terminal.json']
    files += [SITE/'torch'/p for p in ('__init__.py','_C.cp312-win_amd64.pyd','lib/torch_cpu.dll',
        'lib/torch_cuda.dll','cuda/__init__.py','utils/checkpoint.py','optim/adam.py','optim/optimizer.py')]
    files += [SITE/'safetensors'/p for p in ('__init__.py','_safetensors_rust.pyd')]
    files += [SITE/'numpy/__init__.py',SITE/'psutil/__init__.py']
    files += [ROOT/p for p in ('benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    files=list(dict.fromkeys(p.resolve() for p in files))
    adoption=json.loads((DOC/'chatbot_broad_adoption_result_20261009.json').read_bytes())
    assert adoption['corpus']['sha256']==sha(corpus)
    assert sha(rec['logits']['path'])==rec['logits']['sha256']
    write(a.out,dict(schema='ORIGINAL_FALCON_TRANSFER_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),source=str(source),basis=str(basis),corpus=str(corpus),
        selected_id=rec['id'],native=str(native),checkpoint_reference=str(ROOT/'results/phase57/moe_gran.pt'),
        inputs=[extent(p) for p in files],limits=dict(seconds=900,reserve_seconds=120,OS_bytes=24<<30,
            GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=16<<30),
        gates=dict(VJP_relative=1e-4,output_relative_row_RMS=1e-4,mass_abs=1e-6),
        projected=dict(master_coefficients=717877248,bank_coefficients=679477248,packed_bytes=507505920,
                       full_logits_bytes=395057036),allowed_worker_children=['packed_original.exe','conhost.exe'],
        allowed_system_child_path='C:/Windows/System32/conhost.exe',
        runtime_binding_scope='Pinned source/cached basis/whole teacher packet/new construction and first-order adjoint/CPU+CUDA principal Torch binaries/optimizer/checkpoint/runtime/foreign hashes;not full DLL-tree.'))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='ORIGINAL_FALCON_TRANSFER_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    a.directory.mkdir(exist_ok=False);stage='startup';updates=0;durable=0;last_snapshot=None;children=[];child_peak=0
    import numpy as np
    import psutil
    import torch
    from original_falcon_learner import SourceLearner,UnionBank,initialize,E,V
    from original_tensor_learner import Bank,export,quant_weight,D,EH,K,L
    from chatbot_falcon_usability_launch import Memory
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(1713);torch.cuda.manual_seed_all(1713);torch.cuda.reset_peak_memory_stats()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));last_event=0
    def guard(reserve=True):
        lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-(lim['reserve_seconds'] if reserve else 0),'worker reserve/deadline'
        assert proc.memory_info().peak_wset+child_peak<=lim['OS_bytes'],'worker/child OS cap'
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'],'GPU allocated cap'
        assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU reserved cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes'],'namespace cap'
    def event(**values):
        guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**values)),flush=True)
    def progress(mode,count,total,site=None):
        nonlocal last_event
        guard()
        now=time.monotonic()
        if now-last_event>=15 or count==total:
            torch.cuda.synchronize();event(operation=mode,completed=count,total=total,site=site);last_event=now
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    def save(name,value):
        path=a.directory/name
        with path.open('xb') as f:torch.save(cpu(value),f)
        guard(False);return extent(path)
    def run(argv,name):
        nonlocal child_peak
        t=time.monotonic();log=a.directory/(name+'.log')
        getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];getmem.restype=wintypes.BOOL
        peak=0
        with log.open('xb') as f:
            p=subprocess.Popen([str(v) for v in argv],stdout=f,stderr=subprocess.STDOUT,creationflags=8)
            def sample():
                nonlocal peak,child_peak
                m=Memory();m.cb=ctypes.sizeof(m);assert getmem(wintypes.HANDLE(int(p._handle)),ctypes.byref(m),m.cb)
                peak=max(peak,m.PeakWorkingSetSize);child_peak=max(child_peak,peak)
            try:
                while p.poll() is None:sample();guard();time.sleep(.05)
                sample()
            except BaseException:
                if p.poll() is None:p.kill();p.wait()
                sample();raise
            finally:
                rec=dict(command=[str(v) for v in argv],pid=p.pid,exit_code=p.returncode,held_OS_peak=peak,seconds=time.monotonic()-t)
                children.append(rec);write(a.directory/(name+'.receipt.json'),rec)
        assert p.returncode==0,(name,p.returncode,log.read_text(errors='replace'));guard()
    def dump(name,value):raw(a.directory/name,value.detach().cpu().numpy().astype('<f4').tobytes())
    def route_file(name,routes):
        data=bytearray()
        for t in range(routes[0][0].shape[0]):
            for ids,mass in routes:
                data.extend(ids[t].numpy().astype('<i4').tobytes());data.extend(mass[t].numpy().astype('<f4').tobytes())
        raw(a.directory/name,data)
    def kl(logits,teacher):
        logp=torch.log_softmax(teacher,-1);lp=torch.log_softmax(logits,-1)
        return (logp.exp()*(logp-lp)).sum(-1).mean()
    model=None;optimizer=None
    try:
        stage='adjoint_control'
        packet=torch.load(b['checkpoint_reference'],map_location='cpu',weights_only=True)['model']
        ref=Bank(32,'cpu')
        with torch.no_grad():
            for label in ('gate','up','down'):
                key='blocks.0.mlp.Wd' if label=='down' else 'blocks.0.mlp.'+label+'.weight'
                getattr(ref,label).copy_(packet[key].reshape_as(getattr(ref,label)))
            ref.router.weight.copy_(packet['blocks.0.mlp.router.weight']);ref.router.bias.copy_(packet['blocks.0.mlp.router.bias'])
        del packet
        generator=torch.Generator().manual_seed(1711);x=torch.randn(17,D,generator=generator)
        cotangent=torch.randn(17,D,generator=torch.Generator().manual_seed(1712))
        xr=x.clone().requires_grad_(True);yr=ref(xr);yr.backward(cotangent)
        reference={'output':yr.detach().cpu(),'input':xr.grad.detach().clone(),
                   **{name:p.grad.detach().clone() for name,p in ref.named_parameters()}}
        with torch.no_grad():ref(x);reference['routes']=ref.last_routes
        comparisons=[];control={'reference':reference};all_control=True
        for device in ('cpu','cuda'):
            test=UnionBank(32,device);test.load_state_dict(ref.state_dict())
            xt=x.to(device).detach().requires_grad_(True);yt=test(xt);yt.backward(cotangent.to(device))
            actual={'output':yt.detach().cpu(),'input':xt.grad.detach().cpu(),
                    **{name:p.grad.detach().cpu() for name,p in test.named_parameters()}}
            rows=[]
            for name,r in reference.items():
                if name=='routes':continue
                value=actual[name];delta=(value.double()-r.double()).norm();den=max(float(r.double().norm()),1e-8)
                rel=float(delta)/den;rows.append(dict(tensor=name,relative_L2=rel));all_control &= rel<=b['gates']['VJP_relative']
            ri,rm=reference['routes'];ti,tm=test.last_routes
            exact=bool(torch.equal(ri,ti));md=float((rm-tm).abs().max());all_control &= exact and md<=b['gates']['mass_abs']
            comparisons.append(dict(device=device,tensors=rows,IDs_exact=exact,mass_max_abs=md));control[device]=actual
            del test,xt,yt,actual
        assert all_control,'streamed adjoint qualification failed'
        save('adjoint_control.pt',control);write(a.directory/'adjoint_control.json',comparisons)
        del control,reference,ref,x,xr,yr,cotangent;gc.collect();torch.cuda.empty_cache();event(pass_gate=True)
        stage='source_initialization'
        model=SourceLearner(device='cuda')
        ledger,p=initialize(model,b['source'],b['basis'],progress=progress)
        assert ledger['total_master_coefficients']==b['projected']['master_coefficients']
        assert ledger['bank_coefficients']==b['projected']['bank_coefficients'] and len(ledger['expert_maps'])==L*E
        for name,param in model.named_parameters():
            bank=name.endswith(('.gate','.up','.down'))
            assert param.dtype==torch.float32 and param.device.type==('cpu' if bank else 'cuda'),name
            assert torch.isfinite(param).all(),name
        write(a.directory/'initialization_ledger.json',ledger);save('basis256.pt',dict(P=p));del p
        records=json.loads(Path(b['corpus']).read_bytes())['records'];rec=next(r for r in records if r['id']==b['selected_id'])
        assert rec['student_input_ids']==rec['input_ids']+rec['output_ids'][:-1]
        assert len(rec['student_input_ids'])==1507 and rec['positions']==list(range(1251,1507))
        ids=torch.tensor([rec['student_input_ids']],device='cuda');positions=torch.tensor(rec['positions'],device='cuda')
        torch.manual_seed(1713);torch.cuda.manual_seed_all(1713)
        last_snapshot=save('source_initial.pt',dict(schema='ORIGINAL_FALCON_STATE_V1',model=model.state_dict(),updates=0,
            CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),initialization_ledger_sha256=sha(a.directory/'initialization_ledger.json'),
            input_ids=rec['student_input_ids'],positions=rec['positions'],provenance='Fresh original core; source-derived FFN/readout; no old optimizer history.'))
        event(snapshot=last_snapshot)
        for site,layer in enumerate(model.layers):
            layer.bank.progress=lambda mode,count,total,site=site:progress(mode,count,total,site)
        stage='whole_forward_backward'
        bits=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4')
        teacher=torch.from_numpy((bits<<16).view('<f4').reshape(256,V)).to('cuda');del bits
        optimizer=torch.optim.Adam(model.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        model.use_checkpoint=True;optimizer.zero_grad(set_to_none=True)
        t=time.monotonic();logits=model(ids,positions)[0];torch.cuda.synchronize()
        assert torch.isfinite(logits).all();before_loss=kl(logits,teacher)
        before_routes=[layer.bank.last_routes for layer in model.layers]
        dump('before.labels.f32',logits);route_file('before.routes',before_routes)
        before_dis=int((logits.argmax(-1)!=teacher.argmax(-1)).sum())
        exposure=[dict(site=l,union=len(torch.unique(r[0])),selected_pairs=r[0].numel()) for l,r in enumerate(before_routes)]
        forward_seconds=time.monotonic()-t;event(KL=float(before_loss.detach()),exposure=exposure,forward_seconds=forward_seconds)
        before_loss.backward();torch.cuda.synchronize();backward_seconds=time.monotonic()-t-forward_seconds
        stage='gradient_inspection';groups={};norm_squared=0.
        for name,param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
            squared=float(param.grad.double().square().sum());norm_squared+=squared
            if name.startswith('layers.'):
                site=int(name.split('.')[1]);label='bank' if name.endswith(('.gate','.up','.down')) else ('router' if '.router.' in name else ('norm' if name.endswith(('.norm','.ff_norm')) else 'core'))
                key=f'{site}.{label}';groups[key]=groups.get(key,0)+squared
        assert all(groups.get(f'{l}.{kind}',0)>0 for l in range(L) for kind in ('bank','router','norm','core')),groups
        gradnorm=math.sqrt(norm_squared);clip=min(1.,1/(gradnorm+1e-6))
        with torch.no_grad():
            for param in model.parameters():param.grad.mul_(clip)
        guard();optimizer.step();updates=1;torch.cuda.synchronize()
        for param in model.parameters():assert torch.isfinite(param).all()
        for state in optimizer.state.values():
            assert int(state['step'])==1 and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
        gradients=dict(KL_before=float(before_loss.detach()),disagreement_before=before_dis,global_L2_before_clip=gradnorm,
            clip_coefficient=clip,groups_squared_norm=groups,forward_seconds=forward_seconds,backward_seconds=backward_seconds)
        write(a.directory/'update1.json',dict(gradients=gradients,exposure=exposure,input_ids=1507,labels=256,updates=updates))
        stage='durable_update';last_snapshot=save('candidate.pt',dict(schema='ORIGINAL_FALCON_STATE_V1',model=model.state_dict(),
            optimizer=optimizer.state_dict(),updates=updates,CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),
            initialization_ledger_sha256=sha(a.directory/'initialization_ledger.json'),input_ids=rec['student_input_ids'],positions=rec['positions'],gradients=gradients))
        durable=1;event(snapshot=last_snapshot)
        optimizer.zero_grad(set_to_none=True);del logits,before_loss,before_routes;gc.collect();torch.cuda.empty_cache()
        stage='export_updated';packed=export(model,a.directory/'candidate.packed');assert packed['bytes']==b['projected']['packed_bytes'];event(packed_bytes=packed['bytes'])
        distinct=[]
        for l,layer in enumerate(model.layers):
            hashes=[]
            for e in range(E):
                h=hashlib.sha256()
                for name in ('gate','up','down'):
                    q,s=quant_weight(getattr(layer.bank,name)[e])
                    h.update(q.numpy().astype('i1').tobytes());h.update(s.numpy().astype('<f4').tobytes())
                hashes.append(h.hexdigest())
            distinct.append(dict(site=l,distinct_effective_bundles=len(set(hashes)),bundle_hashes=hashes))
        write(a.directory/'effective_bundle_hashes.json',distinct)
        tokens=np.asarray(rec['student_input_ids'],dtype='<u4');raw(a.directory/'queries.u32',struct.pack('<I',len(tokens))+tokens.tobytes())
        stage='native_whole_history';run([b['native'],a.directory/'candidate.packed',a.directory/'queries.u32',a.directory/'native.f32',
            a.directory/'native.routes',a.directory/'native.witness',a.directory/'native.json'],'native')
        stage='GPU_whole_history'
        with torch.no_grad():after=model(ids)[0]
        assert after.shape==(1507,V) and torch.isfinite(after).all();after_loss=float(kl(after[positions],teacher))
        after_dis=int((after[positions].argmax(-1)!=teacher.argmax(-1)).sum());dump('learner.f32',after)
        after_routes=[layer.bank.last_routes for layer in model.layers];route_file('learner.routes',after_routes)
        del after,teacher;gc.collect();torch.cuda.empty_cache()
        stage='complete_output_comparison'
        learner=np.memmap(a.directory/'learner.f32',dtype='<f4',mode='r',shape=(1507,V))
        native=np.memmap(a.directory/'native.f32',dtype='<f4',mode='r',shape=(1507,V))
        rows=[];output_pass=True;greedy_mismatch=0
        for t in range(1507):
            r=native[t].astype('f8');q=learner[t].astype('f8');delta=q-r
            rel=float(np.sqrt(np.mean(delta**2)))/max(float(np.sqrt(np.mean(r*r))),1e-8)
            mismatch=int(np.argmax(q)!=np.argmax(r));greedy_mismatch+=mismatch;output_pass &= rel<=b['gates']['output_relative_row_RMS']
            rows.append(dict(position=t,relative_RMS=rel,max_abs=float(np.max(np.abs(delta))),greedy_mismatch=mismatch))
        del learner,native
        nr=(a.directory/'native.routes').read_bytes();assert len(nr)==1507*L*64
        route_mismatches=0;max_mass=0.;max_defect=0.
        for t in range(1507):
            for l,(ri,rm) in enumerate(after_routes):
                off=(t*L+l)*64;ni=np.frombuffer(nr,dtype='<i4',count=K,offset=off);nm=np.frombuffer(nr,dtype='<f4',count=K,offset=off+32)
                assert len(set(ni.tolist()))==K and ni.min()>=0 and ni.max()<E and np.isfinite(nm).all() and (nm>=0).all()
                route_mismatches+=int(not np.array_equal(ni,ri[t].numpy()))
                max_mass=max(max_mass,float(np.max(np.abs(nm.astype('f8')-rm[t].numpy()))))
                max_defect=max(max_defect,abs(float(np.sum(nm,dtype='f8'))-1),abs(float(rm[t].double().sum())-1))
        expected=[]
        for e in (0,31,32,1023,1024,E-1):
            for layer in model.layers:
                for name,length in [('gate',D),('up',D),('down',EH)]:
                    qw,_=quant_weight(getattr(layer.bank,name)[e]);qi=np.arange(length,dtype='i8')
                    qi=qi%127-63 if name!='down' else (7*qi)%127-63
                    expected.append(qw.numpy().astype('i8')@qi)
        actual=np.fromfile(a.directory/'native.witness',dtype='<i4');assert np.array_equal(actual,np.concatenate(expected));assert actual.size==18432
        parity=bool(output_pass and route_mismatches==0 and max_mass<=b['gates']['mass_abs'] and max_defect<=1e-6)
        comparison=dict(pass_gate=parity,rows=rows,route_mismatch_calls=route_mismatches,mass_max_abs=max_mass,
            max_mass_defect=max_defect,greedy_mismatch=greedy_mismatch,integer_coordinates=actual.size)
        write(a.directory/'native_comparison.json',comparison);guard(False)
        result=dict(schema='ORIGINAL_FALCON_TRANSFER_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
            decision='ORIGINAL_FALCON_TRANSFER_STEP_PASS' if parity else 'ORIGINAL_FALCON_NATIVE_NUMERICAL_GATE_FAIL',
            source_informed_learner_available=True,adjoint_qualification=comparisons,initialization_ledger=extent(a.directory/'initialization_ledger.json'),
            parameters=717877248,bank_coefficients=679477248,E=E,V=V,input_ids=1507,labels=256,
            optimizer_updates=updates,durable_updates=durable,checkpoint=last_snapshot,gradients=gradients,
            KL_after=after_loss,disagreement_after=after_dis,exposure=exposure,packed=extent(a.directory/'candidate.packed'),
            native_comparison=comparison,source_calls=0,GPU_execution=True,quality_admission=False,
            useful_large_n_admission=False,speed_admission=False,children=children,max_direct_child_OS_peak=child_peak,
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-start)
        write(a.out,result);stage='complete';guard(False)
        print(json.dumps(dict(stage=stage,decision=result['decision'],seconds=time.monotonic()-start,
            KL_before=gradients['KL_before'],KL_after=after_loss)),flush=True)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(stage=stage,fault=repr(error),completed_updates=updates,durable_updates=durable,
            last_snapshot=last_snapshot,children=children,elapsed_seconds=time.monotonic()-start,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['bind'],default=None)
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
