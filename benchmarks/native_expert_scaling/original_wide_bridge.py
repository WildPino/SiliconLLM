"""One actual longest-FIT width-injected Adam25->26/native export bridge.

No source queries;retained native before packets adopted only after exact export
identity. GPU is a numerical training surrogate; native probabilities are measured.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from original_falcon_whole_recovery import extent,check_inputs,memory_reader
from original_packed_capacity import SITE,sha,write,raw
sys.path.insert(0,str(SITE))


def bind(a):
    import shutil
    import numpy as np
    import numpy._core._multiarray_umath as numpy_ext
    import psutil
    assert (np.__version__,psutil.__version__)==('2.4.6','7.2.2')
    state=ROOT/'results/native_expert_scaling/original_falcon_whole_recovery_20261009/candidate_25.pt'
    assert sha(state)=='1a6f366e5df0d913f4fcffd205f2f32c3da844e4c0c4c08b66b8c29accdc8790'
    previous=DOC/'original_native_envelope_finish_result_20261009.json'
    terminal=previous.with_suffix('.terminal.json');r=json.loads(previous.read_bytes());t=json.loads(terminal.read_bytes())
    assert t['exit_code']==0 and t['result_sha256']==sha(previous) and r['parity']['passed']
    old=ROOT/'results/native_expert_scaling/original_native_envelope_finish_20261009'
    packed=Path(r['wide_export']['artifact']['path']);assert extent(packed)==r['wide_export']['artifact']
    corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    records=json.loads(corpus.read_bytes())['records'];chosen=['broad_dev_self_oss_instruct_036','broad_fit_smol_magpie_ultra_022','broad_dev_smol_magpie_ultra_036']
    sequences=[next(x for x in records if x['id']==name) for name in chosen]
    assert [len(x['student_input_ids']) for x in sequences]==[357,1507,1488]
    files=[Path(__file__),B/'original_wide_learner.py',B/'original_tensor_learner.py',B/'original_falcon_learner.py',
        B/'original_falcon_whole_recovery.py',B/'original_packed_capacity.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
        B/'original_native_envelope.py',B/'original_native_envelope_finish.py',B/'original_native_envelope_main.c',
        DOC/'ORIGINAL_WIDE_BRIDGE_PROTOCOL_20261009.md',previous,terminal,corpus,state,packed,Path(sys.executable)]
    files += [old/p for p in ('native_wide.exe','requests.u32','parity_wide.f32','parity_wide.routes','parity_wide.witness')]
    files += [Path(x['logits']['path']) for x in sequences]
    files += [SITE/'torch'/p for p in ('__init__.py','_C.cp312-win_amd64.pyd','lib/torch_cpu.dll','lib/torch_cuda.dll',
        'optim/adam.py','optim/optimizer.py','utils/checkpoint.py','cuda/__init__.py')]
    files += [SITE/'numpy/__init__.py',Path(numpy_ext.__file__),SITE/'psutil/__init__.py']
    files += [ROOT/p for p in ('benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    assert shutil.disk_usage(ROOT).free>=20<<30
    write(a.out,dict(schema='ORIGINAL_WIDE_BRIDGE_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
        source_state=extent(state),initial_fixture=extent(packed),native=extent(old/'native_wide.exe'),queries=extent(old/'requests.u32'),
        before_logits=extent(old/'parity_wide.f32'),before_routes=extent(old/'parity_wide.routes'),before_witness=extent(old/'parity_wide.witness'),
        sequences=sequences,train_id=chosen[1],initial_counter=25,new_updates=1,
        optimizer=dict(lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=0.,foreach=False,clip_global_L2=1.),
        criteria=dict(initial_export_bit_exact=True,transport_bit_exact=True,finite_all_gradients=True,
            actual_new_read_coefficients=True,integer_exact=True,mass_defect=1e-6,GPU_native_row_relative_RMS=1e-4,GPU_native_mass_delta=1e-6),
        limits=dict(seconds=3600,reserve_seconds=300,OS_bytes=32<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=12<<30),
        allowed_child_names=['native_wide.exe','conhost.exe'],inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in files)],
        scope='Actual masters/moments/RNG25 width injection,one whole longest-FIT update;three native fixed histories after. No source/RESERVED/T4. No accepted-speed/useful quality admission. Principal binaries,not DLL tree.'))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def launch(a):
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='ORIGINAL_WIDE_BRIDGE_BINDING_V1' and Path(sys.executable).resolve()==Path(b['python']).resolve()
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','gcc','clang','engine','packed_original','native_base','native_wide')),('overlap',p.pid,name)
    log=a.out.with_suffix('.worker.log');terminal=a.out.with_suffix('.terminal.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out,log,terminal,a.out.with_suffix('.launcher_failure.json')))
    worker=None;peak=0;reader=memory_reader();seen={};record=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_pid=proc.pid)
    def guard():
        nonlocal peak
        peak=max(peak,reader(worker));assert time.monotonic()-start<=b['limits']['seconds'],'family deadline'
        assert peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family OS'
        assert log.stat().st_size<=4<<20,'log cap'
    try:
        check_inputs(b);env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        argv=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),
            '--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
        record['command']=argv
        with log.open('xb') as stream:
            worker=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            record.update(worker_pid=worker.pid,worker_creation_time=psutil.Process(worker.pid).create_time());offset=0
            while worker.poll() is None:
                guard()
                try:
                    for child in psutil.Process(worker.pid).children(recursive=True):
                        try:name=child.name().lower();created=child.create_time();exe=Path(child.exe().removeprefix('\\\\?\\')).resolve()
                        except psutil.NoSuchProcess:continue
                        seen[(child.pid,created)]=child
                        assert name in b['allowed_child_names'],('child name',name)
                        if name.startswith('native_'):assert exe==a.directory.resolve()/name
                        elif name=='conhost.exe':assert exe==Path('C:/Windows/System32/conhost.exe').resolve()
                        else:raise AssertionError(('unexpected child',name,exe))
                except psutil.NoSuchProcess:pass
                with log.open('rb') as f:
                    f.seek(offset);chunk=f.read();end=chunk.rfind(b'\n')+1
                    if end:print(chunk[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.1)
        guard();record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        assert worker.returncode==0,log.read_text(errors='replace')[-6000:]
        check_inputs(b);r=json.loads(a.out.read_bytes());assert r['schema']=='ORIGINAL_WIDE_BRIDGE_RESULT_V1'
        assert r['source_calls']==0 and r['optimizer_updates']==r['new_updates']==1 and r['final_counter']==r['durable_counter']==26 and len(r['children'])==1
        assert r['initial_export']['sha256']==b['initial_fixture']['sha256'] and r['GPU_allocated_peak']<=b['limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['limits']['GPU_reserved_bytes']
        assert peak+r['max_direct_child_OS_peak']+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family worker+direct child+launcher'
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        assert sum(i['bytes'] for i in outputs)<=b['limits']['output_bytes']
        record.update(elapsed_seconds=time.monotonic()-start,result_sha256=sha(a.out),output_files=outputs,
            decision=r['decision'],resource_gates=True,scope='Worker plus direct native child held through exit;GPU allocator maxima measured in worker,not full device residency.')
        write(terminal,record);print(json.dumps(dict(terminal=str(terminal),worker_pid=worker.pid,exit_code=worker.returncode,seconds=record['elapsed_seconds'],decision=r['decision'])),flush=True)
    except BaseException as e:
        for (pid,created),child in reversed(list(seen.items())):
            try:
                if child.is_running() and child.create_time()==created:child.kill()
            except psutil.NoSuchProcess:pass
        if worker is not None:
            if worker.poll() is None:worker.kill();worker.wait()
            peak=max(peak,reader(worker));record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak)
        record.update(fault=repr(e),elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.launcher_failure.json'),record);raise


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());a.directory.mkdir(exist_ok=False)
    import numpy as np
    import psutil
    import torch
    from original_wide_learner import SourceLearner,export,quant_weight,widen,verify,tensor,source
    assert (torch.__version__,np.__version__,psutil.__version__)==('2.6.0+cu124','2.4.6','7.2.2')
    assert (tensor.DN,tensor.DT,source.DN,source.DT)==(1024,48,1024,48)
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.cuda.reset_peak_memory_stats()
    reader=memory_reader();child_peak=0;children=[];stage='startup';counter=25;durable=25;last_event=0.
    model=None;optimizer=None;last_snapshot=None;transport=[];update=None;parent_ledger=None;inherited_ids=None
    def guard(reserve=True):
        lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-(lim['reserve_seconds'] if reserve else 0),'reserve/deadline'
        assert proc.memory_info().peak_wset+child_peak<=lim['OS_bytes'],'worker/child OS'
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'],'GPU allocated'
        assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU reserved'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes'],'output cap'
    def event(**values):
        guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,counter=counter,durable=durable,**values)),flush=True)
    def progress(mode,count,total,site):
        nonlocal last_event
        guard();now=time.monotonic()
        if now-last_event>=15:torch.cuda.synchronize();event(operation=mode,completed=count,total=total,site=site);last_event=now
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    def snapshot(filename,partial=False):
        nonlocal last_snapshot,durable
        path=a.directory/filename
        packet=dict(schema='ORIGINAL_WIDE_STATE_V1',model=model.state_dict(),optimizer=optimizer.state_dict(),updates=counter,new_updates=counter-25,
            source_state=b['source_state'],geometry=dict(D=256,N=96,DN=1024,DT=48,L=6,E=1152,V=65537,HID_E=128,K=8),
            optimizer_partial_possible=partial,CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),
            initialization_ledger_sha256=parent_ledger,transport_ledger=extent(a.directory/'transport_ledger.json'),
            binding_sha256=a.binding_sha,inherited_completed_FIT_ids=inherited_ids,completed_new_FIT_ids=[b['train_id']] if update else [])
        with path.open('xb') as f:torch.save(cpu(packet),f);f.flush();os.fsync(f.fileno())
        del packet;gc.collect();last_snapshot=extent(path)
        if not partial:durable=counter
        guard(False);event(snapshot=last_snapshot)
    def native_metrics(rec,logits_path,routes_path,row_offset):
        n=len(rec['student_input_ids']);labels=len(rec['positions']);total=3352
        logits=np.memmap(logits_path,dtype='<f4',mode='r',shape=(total,65537));teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(labels,65537))
        losses=[];uniform=[];dis=0
        for j,pos in enumerate(rec['positions']):
            qlog=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8');qlog-=qlog.max();qlog-=np.log(np.exp(qlog).sum());q=np.exp(qlog)
            z=logits[row_offset+pos].astype('f8');z-=z.max();z-=np.log(np.exp(z).sum())
            losses.append(float(np.sum(q*(qlog-z))));uniform.append(math.log(65537)+float(np.sum(q*qlog)));dis+=int(np.argmax(z)!=np.argmax(qlog))
        for first in range(0,n,128):assert np.isfinite(logits[row_offset+first:row_offset+min(first+128,n)]).all()
        routes=np.fromfile(routes_path,dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))])).reshape(total,6)[row_offset:row_offset+n]
        ids=routes['ids'];mass=routes['mass'];assert ids.min()>=0 and ids.max()<1152 and (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all()
        assert np.isfinite(mass).all() and (mass>=0).all();defect=float(np.max(np.abs(mass.astype('f8').sum(-1)-1)));assert defect<=1e-6
        result=dict(id=rec['id'],split=rec['split'],history=n,labels=labels,row_offset=row_offset,KL=float(np.mean(losses)),KL_per_label=losses,
            uniform_KL=float(np.mean(uniform)),disagreement=dis,disagreement_rate=dis/labels,unions=[int(np.unique(ids[:,l]).size) for l in range(6)],mass_defect=defect)
        del logits,teacher,routes;return result
    try:
        stage='load_actual25';state=torch.load(b['source_state']['path'],map_location='cpu',weights_only=True)
        assert state['schema']=='ORIGINAL_FALCON_WHOLE_CONTINUATION_STATE_V1' and state['updates']==25
        parent_ledger=state['initialization_ledger_sha256'];inherited_ids=state['completed_FIT_ids'];assert len(inherited_ids)==24
        model=SourceLearner(device='cuda');names=list(model.named_parameters());assert len(names)==92 and sum(p.numel() for _,p in names)==721008128
        assert list(state['model'])==[name for name,_ in names],'parameter ordering'
        stage='master_injection'
        with torch.no_grad():
            for name,p in names:
                old=state['model'][name];mapped=widen(old,name,p.shape);assert torch.isfinite(mapped).all();p.copy_(mapped.to(p.device))
                rec=dict(name=name,old_shape=list(old.shape),new_shape=list(p.shape),master=verify(p,old,name))
                assert p.device.type==('cpu' if name.endswith(('.gate','.up','.down')) else 'cuda');transport.append(rec);del mapped
        cfg=b['optimizer'];optimizer=torch.optim.Adam([p for _,p in names],lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=0.,foreach=False)
        old_opt=state['optimizer'];old_ids=old_opt['param_groups'][0]['params'];assert len(old_ids)==92
        mapped_states={}
        for row,(name,p),idx in zip(transport,names,old_ids):
            old=old_opt['state'][idx];assert int(old['step'])==25 and set(old)=={'step','exp_avg','exp_avg_sq'}
            mapped_states[idx]=dict(step=old['step'],exp_avg=widen(old['exp_avg'],name,p.shape,True),exp_avg_sq=widen(old['exp_avg_sq'],name,p.shape,True))
        optimizer.load_state_dict(dict(state=mapped_states,param_groups=old_opt['param_groups']))
        assert optimizer.param_groups[0]['lr']==5e-5 and optimizer.param_groups[0]['betas']==(.9,.999) and optimizer.param_groups[0]['eps']==1e-8 and optimizer.param_groups[0]['weight_decay']==0 and optimizer.param_groups[0]['foreach']==False
        for row,(name,p),idx in zip(transport,names,old_ids):
            st=optimizer.state[p];old=old_opt['state'][idx];assert torch.equal(st['step'].cpu(),old['step']) and int(st['step'])==25
            row['moments']={}
            for field in ('exp_avg','exp_avg_sq'):
                assert st[field].dtype==torch.float32 and st[field].device==p.device and torch.isfinite(st[field]).all()
                row['moments'][field]=verify(st[field],old[field],name,True)
        torch.set_rng_state(state['CPU_rng']);torch.cuda.set_rng_state_all(state['CUDA_rng'])
        assert torch.equal(torch.get_rng_state(),state['CPU_rng']) and all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),state['CUDA_rng']))
        ledger=dict(schema='ORIGINAL_WIDE_TRANSPORT_V1',source=b['source_state'],parameters=721008128,bank_coefficients=679477248,
            rows=transport,unchanged_master_tensors=sum(not x['master']['changed_shape'] for x in transport),changed_master_tensors=sum(x['master']['changed_shape'] for x in transport),
            old_counter=25,RNG_restored_exact=True,new_moments_zero=True)
        assert ledger['changed_master_tensors']==45;write(a.directory/'transport_ledger.json',ledger)
        del state,old_opt,mapped_states,old;gc.collect();torch.cuda.empty_cache();event(transport_exact=True,parameters=721008128)
        stage='initial_export';export(model,a.directory/'initial_wide.packed');initial=extent(a.directory/'initial_wide.packed')
        assert initial['bytes']==b['initial_fixture']['bytes'] and initial['sha256']==b['initial_fixture']['sha256'],'initial packed identity'
        event(initial_export=initial)
        before=[];offset=0
        for rec in b['sequences']:before.append(native_metrics(rec,b['before_logits']['path'],b['before_routes']['path'],offset));offset+=len(rec['student_input_ids'])
        write(a.directory/'adopted_native_before.json',dict(initial_export=initial,before=before,parent_logits=b['before_logits'],parent_routes=b['before_routes']))
        model.use_checkpoint=True
        for site,layer in enumerate(model.layers):layer.bank.progress=lambda mode,count,total,site=site:progress(mode,count,total,site)
        rec=next(x for x in b['sequences'] if x['id']==b['train_id']);n=1507
        stage='whole_forward_backward';optimizer.zero_grad(set_to_none=True);ids=torch.tensor([rec['student_input_ids']],device='cuda');pos=torch.tensor(rec['positions'],device='cuda')
        teacher_raw=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4');teacher=torch.from_numpy((teacher_raw<<16).view('<f4').reshape(len(rec['positions']),65537)).cuda();del teacher_raw
        ts=time.monotonic();logits=model(ids,pos)[0];torch.cuda.synchronize();assert torch.isfinite(logits).all()
        logq=torch.log_softmax(teacher,-1);logp=torch.log_softmax(logits,-1);loss=(logq.exp()*(logq-logp)).sum(-1).mean();assert torch.isfinite(loss)
        forward=time.monotonic()-ts;loss_before=float(loss.detach());exposure=[dict(site=l,union=int(torch.unique(layer.bank.last_routes[0]).numel()),selected_pairs=layer.bank.last_routes[0].numel()) for l,layer in enumerate(model.layers)]
        event(KL_before_surrogate=loss_before,exposure=exposure);loss.backward();torch.cuda.synchronize();backward=time.monotonic()-ts-forward
        stage='gradient_inspection';squared=0.;new_read_grads=[]
        for name,p in names:
            assert p.grad is not None and torch.isfinite(p.grad).all(),name;squared+=float(p.grad.double().square().sum())
            if name.endswith('.organs.out_proj'):new_read_grads.append(dict(name=name,squared_norm=float(p.grad[:,512:].double().square().sum())))
        gradient_norm=math.sqrt(squared);clip=min(1.,1./(gradient_norm+1e-6))
        assert any(x['squared_norm']>0 for x in new_read_grads),'new read gradients'
        with torch.no_grad():
            for _,p in names:p.grad.mul_(clip)
        stage='optimizer_step';guard();ts=time.monotonic();optimizer.step();counter=26;torch.cuda.synchronize()
        for _,p in names:
            assert torch.isfinite(p).all()
            st=optimizer.state[p];assert int(st['step'])==26 and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
        new_read_changes=[]
        for name,p in names:
            if name.endswith('.organs.out_proj'):new_read_changes.append(dict(name=name,nonzero_new_coefficients=int(torch.count_nonzero(p[:,512:])),new_read_L2=float(p[:,512:].double().norm())))
        assert any(x['nonzero_new_coefficients']>0 for x in new_read_changes)
        update=dict(counter=26,id=rec['id'],history=1507,labels=len(rec['positions']),KL_before_surrogate=loss_before,exposure=exposure,
            forward_seconds=forward,backward_seconds=backward,optimizer_finite_inspection_seconds=time.monotonic()-ts,
            gradient_norm=gradient_norm,clip_coefficient=clip,new_read_gradients=new_read_grads,new_read_changes=new_read_changes)
        write(a.directory/'actual_update26.json',update);optimizer.zero_grad(set_to_none=True);del logits,loss,logq,logp;gc.collect();torch.cuda.empty_cache()
        stage='durable_actual26';snapshot('candidate_wide26.pt')
        stage='updated_export';export(model,a.directory/'candidate_wide26.packed');packed=extent(a.directory/'candidate_wide26.packed')
        assert packed['bytes']==520029440 and packed['sha256']!=initial['sha256'];event(packed=packed)
        stage='GPU_after_surrogate'
        with torch.no_grad():logits=model(ids)[0];torch.cuda.synchronize()
        assert logits.shape==(1507,65537) and torch.isfinite(logits).all();after_routes=[(layer.bank.last_routes[0].clone(),layer.bank.last_routes[1].clone()) for layer in model.layers]
        glp=torch.log_softmax(logits[pos],-1);glq=torch.log_softmax(teacher,-1);gpu_after=float((glq.exp()*(glq-glp)).sum(-1).mean())
        raw(a.directory/'GPU_after.f32',logits.cpu().numpy().astype('<f4').tobytes());del logits,glp,glq,teacher,ids,pos;gc.collect();torch.cuda.empty_cache()
        expected=[]
        for e in (0,31,32,1023,1024,1151):
            for layer in model.layers:
                for organ,length in [('gate',256),('up',256),('down',128)]:
                    q,_=quant_weight(getattr(layer.bank,organ)[e]);qi=np.arange(length,dtype='i8');qi=qi%127-63 if organ!='down' else (qi*7)%127-63
                    expected.append(q.numpy().astype('i8')@qi)
        expected=np.concatenate(expected).astype('<i4').tobytes();assert len(expected)==18432*4;raw(a.directory/'expected_updated.witness',expected)
        stage='native_after';native=a.directory/'native_wide.exe';raw(native,Path(b['native']['path']).read_bytes());query=a.directory/'requests.u32';raw(query,Path(b['queries']['path']).read_bytes())
        prefix=a.directory/'after';command=[str(x) for x in (native,packed['path'],query,'parity',prefix)];ts=time.monotonic();peak=0
        with (a.directory/'after.native.log').open('xb') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,creationflags=8);created=psutil.Process(child.pid).create_time()
            try:
                while child.poll() is None:peak=max(peak,reader(child));child_peak=max(child_peak,peak);guard();time.sleep(.05)
                peak=max(peak,reader(child));child_peak=max(child_peak,peak)
            except BaseException:
                if child.poll() is None:child.kill();child.wait()
                peak=max(peak,reader(child));child_peak=max(child_peak,peak);raise
            finally:
                receipt=dict(command=command,pid=child.pid,creation_time=created,exit_code=child.returncode,held_OS_peak=peak,seconds=time.monotonic()-ts);children.append(receipt);write(a.directory/'native_after.receipt.json',receipt)
        assert child.returncode==0 and (a.directory/'after.witness').read_bytes()==expected,'native/witness'
        after=[];offset=0
        for r in b['sequences']:after.append(native_metrics(r,a.directory/'after.f32',a.directory/'after.routes',offset));offset+=len(r['student_input_ids'])
        stage='numeric_adjudication';gpu=np.memmap(a.directory/'GPU_after.f32',dtype='<f4',mode='r',shape=(1507,65537));native_logits=np.memmap(a.directory/'after.f32',dtype='<f4',mode='r',shape=(3352,65537))
        row_metrics=[];worst=0.;bad_rows=0;greedy_bad=0
        for t in range(1507):
            x=gpu[t].astype('f8');y=native_logits[357+t].astype('f8');err=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12));worst=max(worst,err);bad_rows+=int(err>1e-4);greedy_bad+=int(np.argmax(x)!=np.argmax(y));row_metrics.append(dict(position=t,relative_RMS=err))
        routebytes=(a.directory/'after.routes').read_bytes();id_bad=0;mass_delta=0.
        for t in range(1507):
            for l,(ri,rm) in enumerate(after_routes):
                off=((357+t)*6+l)*64;ni=np.frombuffer(routebytes,dtype='<i4',count=8,offset=off);nm=np.frombuffer(routebytes,dtype='<f4',count=8,offset=off+32)
                id_bad+=int(not np.array_equal(ni,ri[t].numpy()));mass_delta=max(mass_delta,float(np.max(np.abs(nm.astype('f8')-rm[t].numpy()))))
        del gpu,native_logits
        numeric=dict(passed=bad_rows==0 and id_bad==0 and mass_delta<=1e-6,worst_row_relative_RMS=worst,failed_rows=bad_rows,
            greedy_mismatch_rows=greedy_bad,ID_mismatch_calls=id_bad,mass_max_delta=mass_delta,rows=row_metrics)
        write(a.directory/'GPU_native_numeric.json',numeric)
        result=dict(schema='ORIGINAL_WIDE_BRIDGE_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='WIDE_ACTUAL_STEP_EXPORT_QUALIFIED' if numeric['passed'] else 'WIDE_ACTUAL_STEP_EXPORT_NATIVE_NUMERIC_FAIL',
            transport=ledger,initial_export=initial,checkpoint=last_snapshot,packed=packed,update=update,initial_counter=25,final_counter=26,durable_counter=durable,new_updates=1,
            before_native_adopted=before,after_native=after,GPU_KL_before=loss_before,GPU_KL_after=gpu_after,native_numerical_comparison=numeric,
            historical_native_numerical_gate='FAIL retained',source_calls=0,optimizer_updates=1,quality_admission=False,speed_admission=False,useful_large_n_admission=False,
            children=children,max_direct_child_OS_peak=child_peak,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-start,
            scope='Actual new width25->26 whole-FIT step;GPU surrogate,native3case endpoints. No general quality/causal width advantage/accepted speed/T4/DRAM/family admission.')
        write(a.out,result);stage='complete';event(decision=result['decision'],GPU_KL_before=loss_before,GPU_KL_after=gpu_after,before=before,after=after)
    except BaseException as e:
        fault=dict(stage=stage,error=repr(e),counter=counter,durable=durable,last_snapshot=last_snapshot,update=update,transport=transport,children=children,
            optimizer_partial_possible=stage=='optimizer_step',GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=child_peak,elapsed_seconds=time.monotonic()-start)
        write(a.directory/'first_fault.json',fault)
        if model is not None and optimizer is not None and (counter!=durable or stage=='optimizer_step'):
            try:optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();snapshot('fault_actual_state.pt',stage=='optimizer_step')
            except BaseException as se:write(a.directory/'fault_snapshot_failure.json',dict(error=repr(se),counter=counter,last_snapshot=last_snapshot))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
