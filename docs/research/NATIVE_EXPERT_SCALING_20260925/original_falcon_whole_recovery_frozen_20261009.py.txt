"""One finite original-operator whole-model continuation, measured in actual C.

Restores actual Adam1; historical numerical FAIL is retained. No donor queries.
"""
import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def extent(path):
    p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def check_inputs(b):
    for i in b['inputs']:
        p=Path(i['path']);assert p.stat().st_size==i['bytes'] and sha(p)==i['sha256'],p


def memory_reader():
    from chatbot_falcon_usability_launch import Memory
    fn=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];fn.restype=wintypes.BOOL
    def read(p):
        m=Memory();m.cb=ctypes.sizeof(m)
        assert fn(wintypes.HANDLE(int(p._handle)),ctypes.byref(m),m.cb)
        return m.PeakWorkingSetSize
    return read


def bind(a):
    corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    records=json.loads(corpus.read_bytes())['records'];assert len(records)==48
    old_result=DOC/'original_falcon_transfer_result_20261009.json'
    old_terminal=old_result.with_suffix('.terminal.json');old=json.loads(old_terminal.read_bytes())
    assert old['exit_code']==0 and sha(old_result)==old['result_sha256']
    r=json.loads(old_result.read_bytes());assert r['durable_updates']==r['optimizer_updates']==1
    outputs={Path(i['path']).name:i for i in old['output_files']}
    adopted={name:outputs[name] for name in ('candidate.pt','candidate.packed','native.f32','native.routes','native.witness','native.json','initialization_ledger.json')}
    for i in adopted.values():assert extent(i['path'])==i
    inventory=DOC/'original_falcon_recovery_inventory_20261009.json';inv=json.loads(inventory.read_bytes())
    assert sha(inventory)=='b9e7de41671a639a4ef963eeccc3ed5d66dd38f1f3eb6e5ea0aa81028695ded8'
    for split in ('FIT','DEV'):
        rows=[r for r in records if r['split']==split]
        assert len(rows)==24 and len({r['domain'] for r in rows})==12
        assert sum(len(r['student_input_ids']) for r in rows)==inv['splits'][split]['histories']
    for rec in records:
        assert rec['student_input_ids']==rec['input_ids']+rec['output_ids'][:-1]
        assert rec['logits']['shape']==[len(rec['positions']),65537]
        assert Path(rec['logits']['path']).stat().st_size==len(rec['positions'])*65537*2
        assert sha(rec['logits']['path'])==rec['logits']['sha256']
    files=[Path(__file__),B/'original_falcon_learner.py',B/'original_tensor_learner.py',B/'original_packed_capacity.py',
        B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',DOC/'ORIGINAL_FALCON_WHOLE_RECOVERY_PROTOCOL_20261009.md',
        corpus,old_result,old_terminal,inventory,Path(sys.executable)]
    files += [Path(i['path']) for i in adopted.values()]+[Path(rec['logits']['path']) for rec in records]
    native=ROOT/'results/native_expert_scaling/original_packed_capacity_20261009/packed_original.exe';files.append(native)
    files += [SITE/'torch'/p for p in ('__init__.py','_C.cp312-win_amd64.pyd','lib/torch_cpu.dll','lib/torch_cuda.dll',
        'cuda/__init__.py','utils/checkpoint.py','optim/adam.py','optim/optimizer.py')]
    files += [SITE/'numpy/__init__.py',SITE/'psutil/__init__.py']
    files += [ROOT/p for p in ('benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    assert shutil.disk_usage(ROOT).free>=80<<30
    write(a.out,dict(schema='ORIGINAL_FALCON_WHOLE_RECOVERY_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),corpus=str(corpus.resolve()),native=str(native.resolve()),adopted=adopted,
        adopted_id='broad_fit_smol_magpie_ultra_022',FIT_order=inv['splits']['FIT']['ordered_ids'],
        limits=dict(seconds=3600,reserve_seconds=300,OS_bytes=32<<30,GPU_allocated_bytes=10<<30,
            GPU_reserved_bytes=11<<30,output_bytes=40<<30),
        gates=dict(DEV_case_KL=1.,DEV_case_disagreement=.20,domain_case_KL=2.,domain_case_disagreement=.35,
            relative_DEV_KL=.90,relative_DEV_disagreement=.95,mass_defect=1e-6),
        optimizer=dict(lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=0.,foreach=False,clip_global_L2=1.),
        start_counter=1,new_updates=24,durable_counters=[13,25],
        inputs=[extent(p) for p in dict.fromkeys(files)],
        runtime_binding_scope='Actual source-informed original Adam1/RNG/ledger/packed/cached48 source vectors,unchanged original endpoint and principal Torch/NumPy/psutil binaries;not full DLL tree.'))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def launch(a):
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='ORIGINAL_FALCON_WHOLE_RECOVERY_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for other in psutil.process_iter(['name','cmdline']):
        name=(other.info['name'] or '').lower();argv=other.info['cmdline'] or []
        if other.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','clang','engine','packed_original')),('overlap',other.pid,name)
    log=a.out.with_suffix('.worker.log');terminal=a.out.with_suffix('.terminal.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out,log,terminal,a.out.with_suffix('.launcher_failure.json')))
    worker=None;peak=0;read_memory=memory_reader();seen={}
    record=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_pid=proc.pid)
    def memory():
        nonlocal peak
        peak=max(peak,read_memory(worker))
    def guard():
        assert time.monotonic()-start<=b['limits']['seconds'],'family deadline'
        assert peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family OS cap'
        assert log.stat().st_size<=4<<20,'log cap'
    try:
        check_inputs(b)
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        argv=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),
            '--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
        record['command']=argv
        with log.open('xb') as stream:
            worker=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            record.update(worker_pid=worker.pid,worker_creation_time=psutil.Process(worker.pid).create_time());offset=0
            while worker.poll() is None:
                memory();guard()
                try:
                    for child in psutil.Process(worker.pid).children(recursive=True):
                        try:
                            name=child.name().lower();created=child.create_time();exe=Path(child.exe().removeprefix('\\\\?\\')).resolve()
                        except psutil.NoSuchProcess:continue
                        seen[(child.pid,created)]=child
                        assert (name=='packed_original.exe' and exe==Path(b['native']).resolve()) or (name=='conhost.exe' and exe==Path('C:/Windows/System32/conhost.exe').resolve()),('unexpected child',name,exe)
                except psutil.NoSuchProcess:pass
                with log.open('rb') as f:
                    f.seek(offset);chunk=f.read();boundary=chunk.rfind(b'\n')+1
                    if boundary:print(chunk[:boundary].decode('utf8',errors='replace'),end='',flush=True);offset+=boundary
                time.sleep(.1)
        memory();guard();record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak)
        assert worker.returncode==0,log.read_text(errors='replace')[-8000:]
        check_inputs(b);r=json.loads(a.out.read_bytes())
        assert r['schema']=='ORIGINAL_FALCON_WHOLE_RECOVERY_RESULT_V1' and r['new_updates']==24 and r['final_counter']==25
        assert len(r['before'])==len(r['after'])==48 and len(r['updates'])==24 and len(r['children'])==95
        assert r['source_calls']==0 and not r['quality_admission'] and not r['speed_admission']
        assert r['historical_native_numerical_gate']=='FAIL retained'
        assert r['GPU_allocated_peak']<=b['limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['limits']['GPU_reserved_bytes']
        assert peak+r['max_direct_child_OS_peak']+proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        files=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        assert sum(i['bytes'] for i in files)<=b['limits']['output_bytes']
        record.update(launcher_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            result_sha256=sha(a.out),decision=r['decision'],output_files=files,resource_gates=True)
        write(terminal,record);print(json.dumps(dict(terminal=str(terminal),worker_pid=worker.pid,exit_code=worker.returncode,
            seconds=record['elapsed_seconds'],OS_peak=peak,decision=r['decision'])),flush=True)
    except BaseException as error:
        for (pid,created),child in reversed(list(seen.items())):
            try:
                if child.is_running() and child.create_time()==created:child.kill()
            except psutil.NoSuchProcess:pass
        if worker is not None:
            if worker.poll() is None:worker.kill();worker.wait()
            memory();record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak)
        record.update(fault=repr(error),elapsed_seconds=time.monotonic()-start)
        write(a.out.with_suffix('.launcher_failure.json'),record);raise


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='ORIGINAL_FALCON_WHOLE_RECOVERY_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    import numpy as np
    import psutil
    import torch
    from original_falcon_learner import SourceLearner,E,V
    from original_tensor_learner import export,quant_weight,D,EH,L,K
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.cuda.reset_peak_memory_stats()
    records=sorted(json.loads(Path(b['corpus']).read_bytes())['records'],key=lambda r:r['id'])
    by_id={r['id']:r for r in records};order=[by_id[i] for i in b['FIT_order']]
    before=[];after=[];updates=[];snapshots=[];children=[];child_peak=0;stage='startup';counter=1;durable=1
    model=None;optimizer=None;last_event=0.;last_snapshot=b['adopted']['candidate.pt'];read_memory=memory_reader()
    def guard(reserve=True):
        limit=b['limits'];assert time.monotonic()-start<=limit['seconds']-(limit['reserve_seconds'] if reserve else 0),'worker reserve/deadline'
        assert proc.memory_info().peak_wset+child_peak<=limit['OS_bytes'],'worker/child OS cap'
        assert torch.cuda.max_memory_allocated()<=limit['GPU_allocated_bytes'],'GPU allocated cap'
        assert torch.cuda.max_memory_reserved()<=limit['GPU_reserved_bytes'],'GPU reserved cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=limit['output_bytes'],'namespace cap'
    def event(**values):
        guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,counter=counter,durable=durable,**values)),flush=True)
    def progress(mode,count,total,site):
        nonlocal last_event
        guard();now=time.monotonic()
        if now-last_event>=15:
            torch.cuda.synchronize();event(operation=mode,completed=count,total=total,site=site);last_event=now
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    def snapshot(name):
        nonlocal durable,last_snapshot
        saved=time.monotonic();path=a.directory/name
        state=dict(schema='ORIGINAL_FALCON_WHOLE_CONTINUATION_STATE_V1',model=model.state_dict(),optimizer=optimizer.state_dict(),
            updates=counter,new_updates=counter-1,CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),
            initialization_ledger_sha256=b['adopted']['initialization_ledger.json']['sha256'],
            parent=b['adopted']['candidate.pt'],binding_sha256=a.binding_sha,completed_FIT_ids=[r['id'] for r in updates])
        with path.open('xb') as f:torch.save(cpu(state),f);f.flush();os.fsync(f.fileno())
        del state;gc.collect();last_snapshot=extent(path);durable=counter
        snapshots.append(dict(**last_snapshot,counter=counter,save_and_hash_seconds=time.monotonic()-saved))
        guard(False);event(snapshot=snapshots[-1])
    def native(packed,rec,prefix,expected):
        nonlocal child_peak
        n=len(rec['student_input_ids']);query=a.directory/(prefix+'.u32')
        with query.open('xb') as f:f.write(struct.pack('<I',n)+np.asarray(rec['student_input_ids'],dtype='<u4').tobytes())
        files=[a.directory/(prefix+suffix) for suffix in ('.f32','.routes','.witness','.native.json')]
        command=[str(v) for v in [b['native'],packed,query,*files]];t=time.monotonic();peak=0
        with (a.directory/(prefix+'.native.log')).open('xb') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,creationflags=8)
            created=psutil.Process(child.pid).create_time()
            try:
                while child.poll() is None:
                    peak=max(peak,read_memory(child));child_peak=max(child_peak,peak);guard();time.sleep(.05)
                peak=max(peak,read_memory(child));child_peak=max(child_peak,peak)
            except BaseException:
                if child.poll() is None:child.kill();child.wait()
                peak=max(peak,read_memory(child));child_peak=max(child_peak,peak);raise
            finally:
                receipt=dict(command=command,pid=child.pid,creation_time=created,exit_code=child.returncode,held_OS_peak=peak,
                    seconds=time.monotonic()-t,prefix=prefix)
                children.append(receipt);write(a.directory/(prefix+'.receipt.json'),receipt)
        assert child.returncode==0,(prefix,child.returncode)
        assert files[2].read_bytes()==expected,'native integer witness differs'
        return metrics(rec,files[0],files[1],False,packed)
    def metrics(rec,logits_path,routes_path,adopted,packed):
        n=len(rec['student_input_ids']);labels=len(rec['positions'])
        assert Path(logits_path).stat().st_size==n*V*4 and Path(routes_path).stat().st_size==n*L*64
        logits=np.memmap(logits_path,dtype='<f4',mode='r',shape=(n,V))
        for first in range(0,n,128):assert np.isfinite(logits[first:first+128]).all()
        teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(labels,V))
        losses=[];uniform=[];dis=0
        for j,pos in enumerate(rec['positions']):
            qlog=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8');qlog-=qlog.max();qlog-=np.log(np.exp(qlog).sum())
            q=np.exp(qlog);z=logits[pos].astype('f8');z-=z.max();z-=np.log(np.exp(z).sum())
            losses.append(float(np.sum(q*(qlog-z))));uniform.append(math.log(V)+float(np.sum(q*qlog)))
            dis+=int(np.argmax(logits[pos])!=np.argmax(qlog))
        routes=np.fromfile(routes_path,dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))])).reshape(n,L)
        ids=routes['ids'];mass=routes['mass'];assert ids.min()>=0 and ids.max()<E
        assert (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all()
        assert np.isfinite(mass).all() and (mass>=0).all()
        defect=float(np.max(np.abs(mass.astype('f8').sum(-1)-1)));assert defect<=b['gates']['mass_defect']
        unions=[int(np.unique(ids[:,site]).size) for site in range(L)]
        result=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,labels=labels,KL=float(np.mean(losses)),
            KL_per_label=losses,uniform_KL=float(np.mean(uniform)),disagreement=dis,disagreement_rate=dis/labels,
            unions=unions,mass_defect=defect,adopted=adopted,packed_path=str(packed),logits=extent(logits_path),routes=extent(routes_path))
        del logits,teacher,routes;guard();return result
    def aggregate(rows):
        def group(values):
            return dict(cases=len(values),labels=sum(r['labels'] for r in values),
                case_KL=sum(r['KL'] for r in values)/len(values),
                label_KL=sum(sum(r['KL_per_label']) for r in values)/sum(r['labels'] for r in values),
                case_disagreement=sum(r['disagreement_rate'] for r in values)/len(values),
                label_disagreement=sum(r['disagreement'] for r in values)/sum(r['labels'] for r in values))
        return {split:dict(**group([r for r in rows if r['split']==split]),
            domains={domain:group([r for r in rows if r['split']==split and r['domain']==domain])
                for domain in sorted({r['domain'] for r in rows})}) for split in ('FIT','DEV')}
    try:
        stage='native_before';initial=b['adopted']['candidate.packed']['path'];expected=Path(b['adopted']['native.witness']['path']).read_bytes()
        assert len(expected)==18432*4
        for i,rec in enumerate(records):
            if rec['id']==b['adopted_id']:
                measured=metrics(rec,b['adopted']['native.f32']['path'],b['adopted']['native.routes']['path'],True,initial)
            else:measured=native(initial,rec,'before_'+rec['id'],expected)
            before.append(measured);write(a.directory/('before_'+rec['id']+'.metrics.json'),measured)
            event(completed_cases=i+1,case=rec['id'],KL=measured['KL'],disagreement=measured['disagreement_rate'])
        stage='restore_actual_Adam1'
        state=torch.load(b['adopted']['candidate.pt']['path'],map_location='cpu',weights_only=True)
        assert state['schema']=='ORIGINAL_FALCON_STATE_V1' and state['updates']==1
        assert state['initialization_ledger_sha256']==b['adopted']['initialization_ledger.json']['sha256']
        model=SourceLearner(device='cuda');model.load_state_dict(state['model'],strict=True)
        params=list(model.parameters());assert len(params)==92 and sum(p.numel() for p in params)==717877248
        cfg=b['optimizer'];optimizer=torch.optim.Adam(params,lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'],foreach=False)
        optimizer.load_state_dict(state['optimizer'])
        group=optimizer.param_groups[0]
        assert (group['lr'],group['betas'],group['eps'],group['weight_decay'],group['foreach'])==(5e-5,(.9,.999),1e-8,0,False)
        for name,p in model.named_parameters():
            assert p.dtype==torch.float32 and torch.isfinite(p).all() and torch.equal(p.detach().cpu(),state['model'][name]),name
            assert p.device.type==('cpu' if name.endswith(('.gate','.up','.down')) else 'cuda'),name
        old_ids=state['optimizer']['param_groups'][0]['params'];assert len(old_ids)==92
        for p,idx in zip(params,old_ids):
            new=optimizer.state[p];old=state['optimizer']['state'][idx]
            assert int(new['step'])==1 and torch.equal(new['step'].cpu(),old['step'])
            for field in ('exp_avg','exp_avg_sq'):
                assert new[field].shape==p.shape and new[field].dtype==torch.float32 and new[field].device==p.device
                assert torch.isfinite(new[field]).all() and torch.equal(new[field].cpu(),old[field])
        torch.set_rng_state(state['CPU_rng']);torch.cuda.set_rng_state_all(state['CUDA_rng'])
        assert torch.equal(torch.get_rng_state(),state['CPU_rng'])
        assert all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),state['CUDA_rng']))
        del state;gc.collect();torch.cuda.empty_cache();model.use_checkpoint=True
        for site,layer in enumerate(model.layers):layer.bank.progress=lambda mode,count,total,site=site:progress(mode,count,total,site)
        event(adoption_exact=True,parameter_tensors=92,moments_counter=1)
        for ordinal,rec in enumerate(order):
            stage='whole_forward_backward';case_start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            ids=torch.tensor([rec['student_input_ids']],device='cuda');positions=torch.tensor(rec['positions'],device='cuda')
            raw=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4')
            teacher=torch.from_numpy((raw<<16).view('<f4').reshape(len(rec['positions']),V)).to('cuda');del raw
            t=time.monotonic();logits=model(ids,positions)[0];torch.cuda.synchronize();assert torch.isfinite(logits).all()
            logq=torch.log_softmax(teacher,-1);logp=torch.log_softmax(logits,-1)
            loss=(logq.exp()*(logq-logp)).sum(-1).mean();loss_value=float(loss.detach())
            dis=int((logits.argmax(-1)!=teacher.argmax(-1)).sum())
            exposure=[dict(site=l,union=int(torch.unique(layer.bank.last_routes[0]).numel()),selected_pairs=layer.bank.last_routes[0].numel())
                for l,layer in enumerate(model.layers)]
            forward=time.monotonic()-t;event(case=rec['id'],ordinal=ordinal,history=len(rec['student_input_ids']),KL=loss_value,exposure=exposure)
            loss.backward();torch.cuda.synchronize();backward=time.monotonic()-t-forward
            stage='gradient_inspection';groups={};squared=0.
            for name,p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
                value=float(p.grad.double().square().sum());squared+=value
                if name.startswith('layers.'):
                    site=name.split('.')[1];kind='bank' if name.endswith(('.gate','.up','.down')) else ('router' if '.router.' in name else ('norm' if name.endswith(('.norm','.ff_norm')) else 'core'))
                    key=site+'.'+kind;groups[key]=groups.get(key,0)+value
            norm=math.sqrt(squared);clip=min(1.,1./(norm+1e-6))
            with torch.no_grad():
                for p in params:p.grad.mul_(clip)
            stage='optimizer_step';guard();t=time.monotonic();optimizer.step();counter+=1;torch.cuda.synchronize()
            for p in params:assert torch.isfinite(p).all()
            for st in optimizer.state.values():assert int(st['step'])==counter and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
            measured=dict(id=rec['id'],ordinal=ordinal,domain=rec['domain'],counter=counter,history=len(rec['student_input_ids']),
                labels=len(rec['positions']),KL_before_step=loss_value,disagreement_before_step=dis,exposure=exposure,
                global_gradient_L2=norm,clip_coefficient=clip,groups_squared_norm=groups,forward_seconds=forward,
                backward_seconds=backward,optimizer_and_finite_inspection_seconds=time.monotonic()-t,
                case_seconds=time.monotonic()-case_start)
            updates.append(measured);write(a.directory/('update_'+str(counter)+'.json'),measured)
            optimizer.zero_grad(set_to_none=True);del ids,positions,teacher,logits,loss,logp,logq;gc.collect();torch.cuda.empty_cache()
            stage='updated';event(update=measured)
            if counter in b['durable_counters']:
                stage='durable_boundary';snapshot('candidate_'+str(counter)+'.pt')
        assert counter==durable==25 and len(updates)==24
        stage='final_export';packed=export(model,a.directory/'candidate_25.packed');assert packed['bytes']==507505920
        event(packed=packed)
        expected=[]
        for e in (0,31,32,1023,1024,E-1):
            for layer in model.layers:
                for name,length in [('gate',D),('up',D),('down',EH)]:
                    q,_=quant_weight(getattr(layer.bank,name)[e]);qi=np.arange(length,dtype='i8')
                    qi=qi%127-63 if name!='down' else (7*qi)%127-63
                    expected.append(q.numpy().astype('i8')@qi)
        expected=np.concatenate(expected).astype('<i4').tobytes();assert len(expected)==18432*4
        with (a.directory/'expected_final.witness').open('xb') as f:f.write(expected)
        stage='native_after'
        for i,rec in enumerate(records):
            measured=native(packed['path'],rec,'after_'+rec['id'],expected);after.append(measured)
            write(a.directory/('after_'+rec['id']+'.metrics.json'),measured)
            event(completed_cases=i+1,case=rec['id'],KL=measured['KL'],disagreement=measured['disagreement_rate'])
        stage='decision';ab=aggregate(before);aa=aggregate(after);g=b['gates'];dev=aa['DEV']
        gates=dict(absolute_DEV_KL=dev['case_KL']<=g['DEV_case_KL'],absolute_DEV_disagreement=dev['case_disagreement']<=g['DEV_case_disagreement'],
            all_domain_KL=all(v['case_KL']<=g['domain_case_KL'] for v in dev['domains'].values()),
            all_domain_disagreement=all(v['case_disagreement']<=g['domain_case_disagreement'] for v in dev['domains'].values()),
            relative_DEV_KL=dev['case_KL']<=g['relative_DEV_KL']*ab['DEV']['case_KL'],
            relative_DEV_disagreement=dev['case_disagreement']<=g['relative_DEV_disagreement']*ab['DEV']['case_disagreement'])
        decision='ORIGINAL_WHOLE_RECOVERY_DEV_DISTRIBUTION_GATES_PASS' if all(gates.values()) else 'ORIGINAL_WHOLE_RECOVERY_DEV_DISTRIBUTION_GATES_FAIL'
        result=dict(schema='ORIGINAL_FALCON_WHOLE_RECOVERY_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision=decision,initial_counter=1,final_counter=counter,durable_counter=durable,new_updates=24,updates=updates,
            before=before,after=after,before_aggregates=ab,after_aggregates=aa,gates=gates,snapshots=snapshots,packed=packed,
            historical_native_numerical_gate='FAIL retained',training_forward_scope='GPU numerical surrogate;native full probabilities govern quality.',
            excluded_quality_scopes=['old32 retention','fresh RESERVED','own-history generation','chatbot tasks','family/scale variants'],
            source_calls=0,quality_admission=False,speed_admission=False,useful_large_n_admission=False,
            children=children,max_direct_child_OS_peak=child_peak,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start)
        write(a.out,result);stage='complete';event(decision=decision,result_sha256=sha(a.out))
    except BaseException as error:
        fault=dict(schema='ORIGINAL_FALCON_WHOLE_RECOVERY_FAULT_V1',stage=stage,error=repr(error),actual_counter=counter,
            durable_counter=durable,last_snapshot=last_snapshot,updates=updates,before=before,after=after,children=children,
            elapsed_seconds=time.monotonic()-start,optimizer_partial_possible=stage=='optimizer_step')
        write(a.directory/'first_fault.json',fault)
        if model is not None and optimizer is not None and (counter!=durable or stage=='optimizer_step'):
            try:
                optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();snapshot('fault_actual_state.pt')
                write(a.directory/'fault_snapshot.json',dict(snapshot=last_snapshot,counter=counter,
                    optimizer_counters=[int(s['step']) for s in optimizer.state.values()],partial_step_possible=fault['optimizer_partial_possible']))
            except BaseException as save_error:write(a.directory/'fault_save_error.json',dict(error=repr(save_error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bind',action='store_true');mode.add_argument('--launch',action='store_true');mode.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
