"""Actual26 FIT-response delta fork, preserved native function, one step27.

All old workers/checkpoints/operators remain immutable. Calibration uses only
the retained FIT core inputs. GPU training remains a numerical surrogate.
"""
import argparse,gc,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from original_packed_capacity import SITE,sha,write,raw,extent,FIELD
from original_falcon_whole_recovery import check_inputs,memory_reader
sys.path.insert(0,str(SITE))

def bind(a):
    import shutil,numpy as np,numpy._core._multiarray_umath as ext,psutil
    parent=DOC/'original_wide_bridge_result_20261009.json';pr=json.loads(parent.read_bytes())
    terminal=parent.with_suffix('.terminal.json');pt=json.loads(terminal.read_bytes());assert pt['exit_code']==0 and pt['result_sha256']==sha(parent)
    oldpath=DOC/'original_wide_bridge_binding_20261009.json';old=json.loads(oldpath.read_bytes())
    tracepath=DOC/'original_wide_trace_result_20261009.json';tr=json.loads(tracepath.read_bytes())
    trace_terminal=tracepath.with_suffix('.terminal.json');tt=json.loads(trace_terminal.read_bytes());assert tt['exit_code']==0 and tt['result_sha256']==sha(tracepath) and tr['observer_identity']
    ns=ROOT/'results/native_expert_scaling/original_wide_bridge_20261009';trace=ROOT/'results/native_expert_scaling/original_wide_trace_20261009/GPU_trace.bin'
    files=[Path(__file__),B/'original_delta_fit_seed.py',B/'original_wide_learner.py',B/'original_tensor_learner.py',B/'original_falcon_learner.py',B/'original_wide_trace.py',
        B/'original_packed_capacity.py',B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
        DOC/'ORIGINAL_DELTA_SEED_FINISH_PROTOCOL_20261009.md',DOC/'original_delta_fit_seed_algebra_20261009.json',parent,terminal,oldpath,tracepath,trace_terminal,
        Path(pr['checkpoint']['path']),Path(pr['packed']['path']),trace,ns/'after.f32',ns/'after.routes',ns/'after.witness',ns/'GPU_after.f32',ns/'native_wide.exe',ns/'requests.u32',Path(sys.executable)]
    files += [Path(rec['logits']['path']) for rec in old['sequences']]
    files += [SITE/'torch'/p for p in ('__init__.py','_C.cp312-win_amd64.pyd','lib/torch_cpu.dll','lib/torch_cuda.dll','optim/adam.py','optim/optimizer.py','utils/checkpoint.py','cuda/__init__.py')]
    files += [SITE/'numpy/__init__.py',Path(ext.__file__),SITE/'psutil/__init__.py']
    files += [ROOT/p for p in ('benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    assert shutil.disk_usage(ROOT).free>=20<<30
    packet=dict(schema='ORIGINAL_DELTA_SEED_FINISH_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),source_state=pr['checkpoint'],parent_result=extent(parent),
        parent_packed=pr['packed'],trace=extent(trace),before_logits=extent(ns/'after.f32'),before_routes=extent(ns/'after.routes'),before_witness=extent(ns/'after.witness'),
        before_GPU_logits=extent(ns/'GPU_after.f32'),native=extent(ns/'native_wide.exe'),queries=extent(ns/'requests.u32'),before_native=pr['after_native'],
        sequences=old['sequences'],train_id=old['train_id'],optimizer=old['optimizer'],seed=dict(count=32,max_norm_multiple=10.,rcond=1e-12),
        criteria=dict(native_initial_heads_routes_witnesses_bit_exact=True,unchanged_coordinates_bit_exact=True,FIT_cross_relative=1e-6,all_new_V_columns_active=True,integer_exact=True,mass_defect=1e-6,GPU_native_row_relative_RMS=1e-4,GPU_native_mass_delta=1e-6),
        limits=dict(seconds=3600,reserve_seconds=300,OS_bytes=32<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=12<<30),
        allowed_child_names=['native_wide.exe','conhost.exe'],inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in files)],
        scope='FIT-calibrated explicit actual26 fork;one new whole-history step27 andthree native cases. No source/RESERVED/T4/quality/accepted-speed/useful-n admission. Principal binaries,not DLL tree.')

    oldbinding=DOC/'original_delta_seed_binding_20261009.json';legacy=json.loads(oldbinding.read_bytes());check_inputs(legacy)
    oldns=ROOT/'results/native_expert_scaling/original_delta_seed_bridge_20261009';fault=oldns/'first_fault.json'
    old_failure=DOC/'original_delta_seed_result_20261009.launcher_failure.json';failure=json.loads(old_failure.read_bytes());frozen_fault=json.loads(fault.read_bytes())
    assert failure['exit_code']==1 and frozen_fault['counter']==frozen_fault['durable']==26 and frozen_fault['update'] is None and not frozen_fault['children'] and frozen_fault['stage']=='FIT_calibration'
    assert frozen_fault['error']=="AssertionError('layers.0.organs.x_proj')"
    features=[extent(oldns/f'site{l}.FIT_X.f32') for l in range(5)];assert all(x['bytes']==1507*1024*4 for x in features)
    assert {p.name for p in oldns.iterdir()}=={'first_fault.json',*(f'site{l}.FIT_X.f32' for l in range(5))}
    extra=[oldbinding,old_failure,old_failure.with_name('original_delta_seed_result_20261009.worker.log'),fault,B/'original_delta_seed_bridge.py',DOC/'ORIGINAL_DELTA_SEED_PROTOCOL_20261009.md']+[Path(x['path']) for x in features]
    present={x['path'] for x in packet['inputs']};packet['inputs'] += [extent(p) for p in extra if str(p.resolve()) not in present]
    packet.update(adopted_features=features,parent_failure=extent(old_failure),parent_fault=extent(fault),parent_family_held_seconds=failure['elapsed_seconds'],old_output_files=[extent(p) for p in sorted(oldns.iterdir())],completed_feature_captures=5,missing_seed_solves=5)
    packet['limits']['seconds']=3560;packet['limits']['output_bytes']-=sum(x['bytes'] for x in packet['old_output_files'])
    packet['scope']='Missing-only U serialization/initial preservation/actualstep27 after field-name fault;five saved feature captures adopted,five seed solves repeated becauseoldU arrays were notdurable. No source/update replay/T4/admission.'
    write(a.out,packet)
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(packet['inputs']))),flush=True)

def native_metrics(np,rec,path,routes_path,offset):
    logits=np.memmap(path,dtype='<f4',mode='r',shape=(3352,65537));labels=len(rec['positions']);n=len(rec['student_input_ids'])
    teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(labels,65537));losses=[];dis=0
    for j,pos in enumerate(rec['positions']):
        q=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8');q-=q.max();q-=math.log(float(np.exp(q).sum()))
        p=logits[offset+pos].astype('f8');p-=p.max();p-=math.log(float(np.exp(p).sum()))
        losses.append(float(np.sum(np.exp(q)*(q-p))));dis+=int(np.argmax(p)!=np.argmax(q))
    for first in range(offset,offset+n,64):assert np.isfinite(logits[first:min(first+64,offset+n)]).all()
    rd=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))]);rt=np.memmap(routes_path,dtype=rd,mode='r',shape=(3352,6))[offset:offset+n]
    ids=rt['ids'];mass=rt['mass'];assert ids.min()>=0 and ids.max()<1152 and (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all()
    assert np.isfinite(mass).all() and (mass>=0).all();defect=float(np.max(np.abs(mass.astype('f8').sum(-1)-1)));assert defect<=1e-6
    return dict(id=rec['id'],split=rec['split'],history=n,labels=labels,row_offset=offset,KL=float(np.mean(losses)),KL_per_label=losses,disagreement=dis,disagreement_rate=dis/labels,
        unions=[int(np.unique(ids[:,l]).size) for l in range(6)],mass_defect=defect)

def packed_identity(np,parent,seeded,seeds):
    with Path(parent).open('rb') as f:table=f.read(80+110*104)
    with Path(seeded).open('rb') as f:assert f.read(len(table))==table
    checked=0;changed=[]
    for i in range(110):
        row=FIELD.unpack(table[80+i*104:80+(i+1)*104]);name=row[0].split(b'\0')[0].decode();offset,size=row[-2:]
        x=np.memmap(parent,dtype='u1',mode='r',offset=offset,shape=(size,));y=np.memmap(seeded,dtype='u1',mode='r',offset=offset,shape=(size,))
        if name in seeds:
            xb=x.view('<u4').reshape(240,1024);yb=y.view('<u4').reshape(240,1024)
            assert np.array_equal(xb[:16],yb[:16]) and np.array_equal(xb[48:],yb[48:]) and not np.any(xb[16:48])
            assert np.array_equal(yb[16:48],seeds[name.replace('.organs.','.')].view('<u4'));changed.append(name)
        else:assert np.array_equal(x,y),name;checked+=1
    assert checked==105 and len(changed)==5
    return dict(header_table_bit_exact=True,unchanged_fields=checked,seeded_fields=changed,seeded_F32_coordinates=5*32*1024)

def worker(a):
    import numpy as np,psutil,torch
    from torch.nn import functional as F
    from original_wide_learner import SourceLearner,export,quant_weight,tensor,source
    from original_wide_trace import layout
    from original_delta_fit_seed import seed_from_fit
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());a.directory.mkdir(exist_ok=False)
    assert (np.__version__,psutil.__version__,torch.__version__)==('2.4.6','7.2.2','2.6.0+cu124') and (tensor.DN,tensor.DT,source.DN,source.DT)==(1024,48,1024,48)
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    reader=memory_reader();child_peak=0;children=[];stage='startup';counter=26;durable=26;last_event=0.;model=None;optimizer=None;last_snapshot=None;update=None;seed_ledger=None;ancestry=None
    def guard(reserve=True):
        lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-(lim['reserve_seconds'] if reserve else 0),'reserve/deadline'
        assert proc.memory_info().peak_wset+child_peak<=lim['OS_bytes'],'worker/direct child OS'
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes'],'output cap'
    def event(**kw):guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,counter=counter,durable=durable,**kw)),flush=True)
    def progress(mode,count,total,site):
        nonlocal last_event
        guard()
        if time.monotonic()-last_event>=15:torch.cuda.synchronize();event(operation=mode,completed=count,total=total,site=site);last_event=time.monotonic()
    def cpu(x):
        if isinstance(x,torch.Tensor):return x.detach().cpu()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        if isinstance(x,list):return [cpu(v) for v in x]
        if isinstance(x,tuple):return tuple(cpu(v) for v in x)
        return x
    def snapshot(filename,partial=False):
        nonlocal durable,last_snapshot
        packet=dict(schema='ORIGINAL_DELTA_SEEDED_STATE_V1',model=model.state_dict(),optimizer=optimizer.state_dict(),updates=counter,new_updates=counter-26,
            optimizer_partial_possible=partial,source_state=b['source_state'],ancestry=ancestry,seed_ledger=extent(a.directory/'seed_ledger.json'),binding_sha256=a.binding_sha,
            geometry=dict(D=256,N=96,DN=1024,DT=48,L=6,E=1152,V=65537,HID_E=128,K=8),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),
            completed_new_FIT_ids=[b['train_id']] if update else [])
        path=a.directory/filename
        with path.open('xb') as f:torch.save(cpu(packet),f);f.flush();os.fsync(f.fileno())
        del packet;gc.collect();last_snapshot=extent(path)
        if not partial:durable=counter
        guard(False);event(snapshot=last_snapshot)
    def run(prefix,packed):
        nonlocal child_peak
        command=[a.directory/'native_wide.exe',packed,a.directory/'requests.u32','parity',a.directory/prefix];t=time.monotonic();peak=0
        with (a.directory/(prefix+'.native.log')).open('xb') as f:
            child=subprocess.Popen([str(x) for x in command],stdout=f,stderr=subprocess.STDOUT,creationflags=8);created=psutil.Process(child.pid).create_time()
            try:
                while child.poll() is None:peak=max(peak,reader(child));child_peak=max(child_peak,peak);guard();time.sleep(.05)
                peak=max(peak,reader(child));child_peak=max(child_peak,peak)
            except BaseException:
                if child.poll() is None:child.kill();child.wait()
                peak=max(peak,reader(child));child_peak=max(child_peak,peak);raise
            finally:
                rec=dict(command=[str(x) for x in command],pid=child.pid,creation_time=created,exit_code=child.returncode,held_OS_peak=peak,seconds=time.monotonic()-t)
                children.append(rec);write(a.directory/(prefix+'.receipt.json'),rec)
        assert child.returncode==0;event(native=prefix,receipt=rec)
    def exact(x,y):return torch.equal(x.detach().cpu().contiguous().view(torch.int32),y.detach().cpu().contiguous().view(torch.int32))
    try:
        stage='actual26_load';state=torch.load(b['source_state']['path'],map_location='cpu',weights_only=True,mmap=True)
        assert state['schema']=='ORIGINAL_WIDE_STATE_V1' and state['updates']==26 and state['new_updates']==1
        ancestry={k:state[k] for k in ('initialization_ledger_sha256','transport_ledger','inherited_completed_FIT_ids','completed_new_FIT_ids','binding_sha256')}
        model=SourceLearner(device='cuda');names=list(model.named_parameters());assert len(names)==92 and sum(p.numel() for _,p in names)==721008128 and list(state['model'])==[n for n,_ in names]
        model.load_state_dict(state['model']);cfg=b['optimizer'];optimizer=torch.optim.Adam([p for _,p in names],lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=0.,foreach=False)
        optimizer.load_state_dict(state['optimizer']);old_ids=state['optimizer']['param_groups'][0]['params'];checks=[]
        for (name,p),idx in zip(names,old_ids):
            assert exact(p,state['model'][name]),name;st=optimizer.state[p];old=state['optimizer']['state'][idx]
            assert int(st['step'])==26 and exact(st['step'],old['step']) and exact(st['exp_avg'],old['exp_avg']) and exact(st['exp_avg_sq'],old['exp_avg_sq'])
            assert torch.isfinite(p).all() and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all();checks.append(name)
        assert optimizer.param_groups[0]['lr']==5e-5 and optimizer.param_groups[0]['betas']==(.9,.999) and optimizer.param_groups[0]['eps']==1e-8 and optimizer.param_groups[0]['weight_decay']==0 and optimizer.param_groups[0]['foreach']==False
        torch.set_rng_state(state['CPU_rng']);torch.cuda.set_rng_state_all(state['CUDA_rng']);assert torch.equal(torch.get_rng_state(),state['CPU_rng']) and all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),state['CUDA_rng']))
        del old;gc.collect();torch.cuda.empty_cache();event(exact_92_master_and_Adam_states=True)
        stage='adopted_FIT_seed_completion';seeds={};ledgers=[]
        for l in range(5):
            w=model.layers[l].organs;st_u=optimizer.state[w['x_proj']];st_v=optimizer.state[w['dt_proj']]
            assert not torch.count_nonzero(w['x_proj'][16:48]) and not torch.count_nonzero(w['dt_proj'][:,16:48])
            for st,key in ((st_u,'U'),(st_v,'V')):
                for field in ('exp_avg','exp_avg_sq'):
                    value=st[field][16:48] if key=='U' else st[field][:,16:48];assert not torch.count_nonzero(value)
            old_u=w['x_proj'][:16].detach().cpu().numpy().copy()
            X=np.fromfile(b['adopted_features'][l]['path'],dtype='<f4').reshape(1507,1024)
            assert np.isfinite(X).all();U,ledger=seed_from_fit(X,old_u,**b['seed'])
            with torch.no_grad():w['x_proj'][16:48].copy_(torch.from_numpy(U).to(w['x_proj'].device))
            assert np.array_equal(w['x_proj'][16:48].detach().cpu().numpy().view('<u4'),U.view('<u4')) and not torch.count_nonzero(w['dt_proj'][:,16:48])
            seeds[f'layers.{l}.x_proj']=U;ledger.update(site=l,FIT_features=b['adopted_features'][l]);ledgers.append(ledger)
            del X;gc.collect();event(calibrated_site=l,rank=ledger['new_rank'],cross=ledger['FIT_response_cross_relative'])
        for name,p in names:
            old=state['model'][name]
            if name.replace('.organs.','.') in seeds:assert exact(p[:16],old[:16]) and exact(p[48:],old[48:]),name
            else:assert exact(p,old),name
        assert torch.equal(torch.get_rng_state(),state['CPU_rng']) and all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),state['CUDA_rng']))
        del old,state;gc.collect();torch.cuda.empty_cache()
        np.savez(a.directory/'seed_rows.npz',**{f'site{l}':seeds[f'layers.{l}.x_proj'] for l in range(5)})
        seed_ledger=dict(schema='ORIGINAL_DELTA_FIT_SEED_LEDGER_V1',source_state=b['source_state'],FIT_id=b['train_id'],rows=extent(a.directory/'seed_rows.npz'),sites=ledgers,unchanged_92_loaded_states_bit_exact=True,
            replacement='Only five x_proj[16:48,:] master blocks replaced;their moments zero andV partner/moments zero. Seeded26 reconstructable fromimmutable parent plus these exact rows;no full duplicate pre-step checkpoint.')
        write(a.directory/'seed_ledger.json',seed_ledger);gc.collect();torch.cuda.empty_cache()
        stage='initial_export';initial_path=a.directory/'seeded26.packed';export(model,initial_path);wire=packed_identity(np,b['parent_packed']['path'],initial_path,seeds)
        raw(a.directory/'native_wide.exe',Path(b['native']['path']).read_bytes());raw(a.directory/'requests.u32',Path(b['queries']['path']).read_bytes())
        stage='native_initial_preservation';run('initial',initial_path)
        for name,key in (('f32','before_logits'),('routes','before_routes'),('witness','before_witness')):assert sha(a.directory/('initial.'+name))==b[key]['sha256'],('initial native identity',name)
        preservation=dict(passed=True,wire=wire,heads_routes_witnesses_bit_exact=True,history_positions=3352,initial_export=extent(initial_path));write(a.directory/'initial_preservation.json',preservation);event(preservation=preservation)
        model.use_checkpoint=True
        for l,layer in enumerate(model.layers):layer.bank.progress=lambda mode,count,total,l=l:progress(mode,count,total,l)
        rec=next(x for x in b['sequences'] if x['id']==b['train_id']);ids=torch.tensor([rec['student_input_ids']],device='cuda');pos=torch.tensor(rec['positions'],device='cuda')
        raw_teacher=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4');teacher=torch.from_numpy((raw_teacher<<16).view('<f4').reshape(256,65537)).cuda();del raw_teacher
        stage='whole_forward_backward';optimizer.zero_grad(set_to_none=True);t=time.monotonic();logits=model(ids,pos)[0];torch.cuda.synchronize();assert torch.isfinite(logits).all()
        logq=teacher.log_softmax(-1);logp=logits.log_softmax(-1);loss=(logq.exp()*(logq-logp)).sum(-1).mean();assert torch.isfinite(loss)
        forward=time.monotonic()-t;before=float(loss.detach());assert abs(before-6.313983917236328)<=1e-5,'seeded GPU before loss preservation'
        exposure=[dict(site=l,union=int(torch.unique(layer.bank.last_routes[0]).numel()),selected_pairs=layer.bank.last_routes[0].numel()) for l,layer in enumerate(model.layers)]
        event(KL_before_surrogate=before,exposure=exposure);loss.backward();torch.cuda.synchronize();backward=time.monotonic()-t-forward
        stage='gradient_inspection';squared=0.;new_V_grad=[]
        for name,p in names:
            assert p.grad is not None and torch.isfinite(p.grad).all(),name;squared+=float(p.grad.double().square().sum())
            if name.endswith('.organs.dt_proj'):
                new=p.grad[:,16:48];assert torch.any(new!=0,dim=0).all(),name;new_V_grad.append(dict(name=name,per_column_L2=new.double().norm(dim=0).cpu().tolist(),squared_norm=float(new.double().square().sum())))
            if name.endswith('.organs.x_proj'):assert not torch.count_nonzero(p.grad[16:48]),('U first-step gradient',name)
        norm=math.sqrt(squared);clip=min(1.,1./(norm+1e-6))
        with torch.no_grad():
            for _,p in names:p.grad.mul_(clip)
        stage='optimizer_step';guard();t=time.monotonic();optimizer.step();counter=27;torch.cuda.synchronize();changes=[]
        for name,p in names:
            st=optimizer.state[p];assert int(st['step'])==27 and torch.isfinite(p).all() and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
            if name.endswith('.organs.dt_proj'):
                new=p[:,16:48];assert torch.any(new!=0,dim=0).all();changes.append(dict(name=name,nonzero=int(torch.count_nonzero(new)),per_column_L2=new.double().norm(dim=0).cpu().tolist(),new_V_L2=float(new.double().norm()),rank_F64=int(torch.linalg.matrix_rank(new.double()).item())))
            if name.endswith('.organs.x_proj'):assert np.array_equal(p[16:48].detach().cpu().numpy().view('<u4'),seeds[name.replace('.organs.','.')].view('<u4'))
        update=dict(counter=27,id=rec['id'],history=1507,labels=256,KL_before_surrogate=before,exposure=exposure,forward_seconds=forward,backward_seconds=backward,optimizer_finite_inspection_seconds=time.monotonic()-t,
            gradient_norm=norm,clip_coefficient=clip,new_V_gradients=new_V_grad,new_V_changes=changes,new_U_first_step_gradient_zero_and_rows_unchanged=True)
        write(a.directory/'actual_update27.json',update);optimizer.zero_grad(set_to_none=True);del logits,loss,logq,logp;gc.collect();torch.cuda.empty_cache()
        stage='durable_actual27';snapshot('candidate_seeded27.pt');stage='updated_export';after_path=a.directory/'candidate_seeded27.packed';export(model,after_path);packed=extent(after_path);event(packed=packed)
        stage='GPU_after'
        with torch.no_grad():logits=model(ids)[0];torch.cuda.synchronize()
        assert logits.shape==(1507,65537) and torch.isfinite(logits).all();glp=logits[pos].log_softmax(-1);glq=teacher.log_softmax(-1);after_gpu=float((glq.exp()*(glq-glp)).sum(-1).mean())
        raw(a.directory/'GPU_after.f32',logits.cpu().numpy().astype('<f4').tobytes());rd=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))]);gpu_routes=np.zeros((1507,6),dtype=rd)
        for l,layer in enumerate(model.layers):gpu_routes['ids'][:,l]=layer.bank.last_routes[0].numpy();gpu_routes['mass'][:,l]=layer.bank.last_routes[1].numpy()
        raw(a.directory/'GPU_after.routes',gpu_routes.tobytes());del logits,glp,glq,teacher,ids,pos;gc.collect();torch.cuda.empty_cache()
        expected=[]
        for e in (0,31,32,1023,1024,1151):
            for layer in model.layers:
                for organ,length in [('gate',256),('up',256),('down',128)]:
                    q,_=quant_weight(getattr(layer.bank,organ)[e]);qi=np.arange(length,dtype='i8');qi=qi%127-63 if organ!='down' else (qi*7)%127-63;expected.append(q.numpy().astype('i8')@qi)
        expected=np.concatenate(expected).astype('<i4').tobytes();assert len(expected)==18432*4;raw(a.directory/'expected_updated.witness',expected)
        stage='native_after';run('after',after_path);assert (a.directory/'after.witness').read_bytes()==expected
        after=[];offset=0
        for case in b['sequences']:after.append(native_metrics(np,case,a.directory/'after.f32',a.directory/'after.routes',offset));offset+=len(case['student_input_ids'])
        stage='numeric_adjudication';gpu=np.memmap(a.directory/'GPU_after.f32',dtype='<f4',mode='r',shape=(1507,65537));native=np.memmap(a.directory/'after.f32',dtype='<f4',mode='r',shape=(3352,65537))[357:1864]
        rows=[];greedy=0
        for t in range(1507):
            x=gpu[t].astype('f8');y=native[t].astype('f8');value=math.sqrt(float(np.sum((x-y)**2))/max(float(np.sum(y*y)),1e-24));rows.append(dict(position=t,relative_RMS=value));greedy+=int(np.argmax(x)!=np.argmax(y))
        nr=np.memmap(a.directory/'after.routes',dtype=rd,mode='r',shape=(3352,6))[357:1864];id_bad=int(np.any(nr['ids']!=gpu_routes['ids'],axis=-1).sum());set_bad=int(np.any(np.sort(nr['ids'],axis=-1)!=np.sort(gpu_routes['ids'],axis=-1),axis=-1).sum());mass_delta=float(np.max(np.abs(nr['mass'].astype('f8')-gpu_routes['mass'].astype('f8'))));bad=sum(row['relative_RMS']>1e-4 for row in rows)
        numeric=dict(passed=bad==0 and id_bad==0 and mass_delta<=1e-6,failed_rows=bad,worst_row_relative_RMS=max(x['relative_RMS'] for x in rows),greedy_mismatch_rows=greedy,ID_mismatch_calls=id_bad,set_ID_mismatch_calls=set_bad,mass_max_delta=mass_delta,rows=rows)
        write(a.directory/'GPU_native_numeric.json',numeric)
        result=dict(schema='ORIGINAL_DELTA_SEED_FINISH_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,decision='DELTA_SEED_ACTUAL27_NATIVE_NUMERIC_PASS' if numeric['passed'] else 'DELTA_SEED_ACTUAL27_NATIVE_NUMERIC_FAIL',
            seed=seed_ledger,initial_preservation=preservation,checkpoint=last_snapshot,packed=packed,update=update,initial_counter=26,final_counter=27,durable_counter=durable,new_updates=1,
            source_calls=0,optimizer_updates=1,before_native_adopted=b['before_native'],after_native=after,GPU_KL_before=before,GPU_KL_after=after_gpu,native_numerical_comparison=numeric,
            quality_admission=False,speed_admission=False,useful_large_n_admission=False,children=children,max_direct_child_OS_peak=child_peak,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-start,
            scope='Explicit actual26 FIT delta initialization/function preservation andone new actual27 step;no causal width advantage/useful chatbot/accepted-speed/DRAM/family admission.')
        write(a.out,result);stage='complete';event(decision=result['decision'],GPU_KL_before=before,GPU_KL_after=after_gpu,native_KL=[x['KL'] for x in after],new_V_changes=changes)
    except BaseException as e:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),counter=counter,durable=durable,last_snapshot=last_snapshot,update=update,seed_ledger=seed_ledger,children=children,
            optimizer_partial_possible=stage=='optimizer_step',elapsed_seconds=time.monotonic()-start,worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=child_peak,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved()))
        if model is not None and optimizer is not None and (counter!=durable or stage=='optimizer_step'):
            try:optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();snapshot('fault_actual_state.pt',stage=='optimizer_step')
            except BaseException as se:write(a.directory/'fault_snapshot_failure.json',dict(error=repr(se),counter=counter,last_snapshot=last_snapshot))
        raise


def launch(a):
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='ORIGINAL_DELTA_SEED_FINISH_BINDING_V1' and Path(sys.executable).resolve()==Path(b['python']).resolve()
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
        check_inputs(b);r=json.loads(a.out.read_bytes());assert r['schema']=='ORIGINAL_DELTA_SEED_FINISH_RESULT_V1'
        assert r['source_calls']==0 and r['optimizer_updates']==r['new_updates']==1 and r['final_counter']==r['durable_counter']==27 and len(r['children'])==2
        assert r['initial_preservation']['passed'] and r['GPU_allocated_peak']<=b['limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['limits']['GPU_reserved_bytes']
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


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
