"""One new original-operator causal forward/backward and native packed bridge."""
import argparse, hashlib, json, math, os, shutil, struct, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'; DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write,raw
from original_falcon_whole_recovery import memory_reader
from original_joint_history_recovery_audit import packed_fields
from causal_categorical_readout import range_sha
from original_categorical_supervision import load_fit_supervision
from original_categorical_loss_compile import probabilities,decode
sys.path.insert(0,str(SITE))
V,D=65537,256

def native_values(x):
    import numpy as np
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:native_values(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native_values(v) for v in x]
    return x

def emit(path,x):write(path,native_values(x))

def bind(a):
    import numpy as np
    import psutil
    import torch
    import torch._dynamo
    from original_categorical_history_learner import CategoricalHistoryLearner
    assert not a.binding.exists()
    names=['original_joint_history_recovery_binding_20261009.json','categorical_native_assessment_binding_20261010.json','causal_categorical_readout_binding_20261010.json','causal_coordinate_dot_bound_result_20261010.json','original_categorical_loss_compile_result_20261010.json','original_categorical_loss_compile_scalar_adjudication_20261010.json']
    paths=[DOC/n for n in names];jb,nb,cb,cr,r,ar=[json.loads(p.read_bytes()) for p in paths]
    assert ar['decision']=='LOSS_SUPERVISION_QUALIFIED' and all(ar['numeric_flags'].values()) and ar['result']==extent(paths[4])
    terminal=paths[5].with_suffix('.terminal.json');t=json.loads(terminal.read_bytes());assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(paths[5])
    rec=min((r for r in nb['records'] if r['split']=='FIT'),key=lambda r:(len(r['student_input_ids']),r['id']))
    assert rec['id']=='broad_fit_explore_instruct_rewriting_008' and len(rec['student_input_ids'])==58 and len(rec['positions'])==18
    target=next(x for x in r['records'] if x['id']==rec['id']);coord=next(x for x in cr['records'] if x['id']==rec['id']);old=next(x for x in nb['old_native'] if x['id']==rec['id'])
    assert target['positions']==coord['positions']==rec['positions'] and coord['split']==target['split']=='FIT'
    witness=next(i for i in nb['inputs'] if i['path'].endswith('expected_updated.witness'))
    with Path(jb['source_packed']['path']).open('rb') as f:f.seek(cb['fields']['final_norm']['offset']);gamma=np.frombuffer(f.read(D*4),'<f4').copy()
    assert np.isfinite(gamma).all() and np.min(abs(gamma))>0
    files=[Path(__file__),a.protocol,*paths,terminal,Path(jb['source_state']['path']),Path(jb['source_packed']['path']),Path(nb['executable']['path']),Path(nb['original_bodies']['source']['path']),Path(witness['path']),Path(old['routes']['path']),Path(coord['features']['path']),Path(rec['logits']['path']),Path(r['head']['path']),Path(r['final_norm']['path'])]
    files += [Path(x[k]['path']) for x in r['records'] if x['split']=='FIT' for k in ('moments','negative_entropy','source_argmax')]
    for m in list(sys.modules.values()):
        for key in ('__file__','__cached__'):
            p=getattr(m,key,None)
            if isinstance(p,(str,os.PathLike)) and Path(p).is_file():files.append(Path(p))
    files += [Path(sys.executable),Path(sys.executable).parent/'python312.dll']
    files += sorted((SITE/'torch/lib').glob('*.dll'))+sorted((SITE/'numpy.libs').glob('*.dll'))+sorted((SITE/'psutil').glob('*.pyd'))
    inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in files)];by={i['path']:i for i in inputs}
    for item in (jb['source_state'],jb['source_packed'],nb['executable'],nb['original_bodies']['source'],witness,old['routes'],coord['features'],rec['logits'],r['head'],r['final_norm']):assert by[str(Path(item['path']).resolve())]=={k:item[k] for k in ('path','bytes','sha256')}
    assert (torch.__version__,np.__version__,psutil.__version__)==('2.6.0+cu124','2.4.6','7.2.2')
    emit(a.binding,dict(schema='ORIGINAL_CATEGORICAL_CAUSAL_PREFLIGHT_BINDING_V1',inputs=inputs,record=rec,target=target,compiled_result=extent(paths[4]),compiled_audit=extent(paths[5]),source_state=jb['source_state'],packed=jb['source_packed'],executable=nb['executable'],original_bodies=nb['original_bodies'],fields=cb['fields'],head=r['head'],final_norm=r['final_norm'],coordinate=coord,native_rounding=cr['calibration'],old_routes=old['routes'],witness=witness,
        numeric=dict(loss_absolute=1e-8,gradient_absolute=1e-8,F32_upstream_absolute=1e-6,GPU_native_score_absolute=1e-3,GPU_native_KL_absolute=1e-3,metric_absolute=1e-8,bound_slack=1e-8),
        capture_limits=dict(seconds=900,reserve_seconds=120,OS_bytes=24<<30,GPU_allocated_bytes=6<<30,GPU_reserved_bytes=8<<30,output_bytes=4<<30,log_bytes=8<<20,child_seconds=100),
        audit_limits=dict(seconds=900,OS_bytes=12<<30,output_bytes=2<<20,log_bytes=8<<20),
        runtime=dict(Torch=torch.__version__,NumPy=np.__version__,psutil=psutil.__version__,actual_imported_modules_and_bytecode=True,junction_resolved=True),scope='One NEW changed coherent head/norm original causal case forward/backward/native prefix. Model and original ternary masters unchanged; no Adam restore/update, source generation, DEV/RESERVED/T4. Training-only F64 fixed-coefficient readout, C F32. Not a useful chatbot/speed admission.'))
    print(json.dumps(dict(binding_sha256=sha(a.binding),inputs=len(inputs),bytes=sum(i['bytes'] for i in inputs))),flush=True)

def array(directory,name,x):
    import numpy as np
    x=np.ascontiguousarray(x);p=directory/(name+'.bin')
    with p.open('xb') as f:f.write(x.tobytes());f.flush();os.fsync(f.fileno())
    return dict(**extent(p),shape=list(x.shape),dtype=x.dtype.str)

def read(item):
    import numpy as np
    x=np.fromfile(item['path'],dtype=item['dtype']).reshape(item['shape']);assert np.isfinite(x).all();return x

def tensor_info(value):
    import numpy as np
    x=value.detach().cpu().contiguous().numpy();assert x.dtype==np.float32 and np.isfinite(x).all()
    buf=memoryview(x).cast('B');h=hashlib.sha256()
    for first in range(0,len(buf),1<<20):h.update(buf[first:first+(1<<20)])
    return dict(shape=list(x.shape),entries=x.size,sha256=h.hexdigest(),squared_norm=float(np.sum(x.astype('f8')**2)))

def export_check(b,path):
    fields=packed_fields(path);assert fields=={k:{**v,'shape':tuple(v['shape'])} for k,v in b['fields'].items()}
    n,h=fields['final_norm'],fields['head'];assert n['offset']+n['size']==h['offset'];size=Path(path).stat().st_size;assert size==b['packed']['bytes'];checks={}
    for key,off,length in [('prefix',0,n['offset']),('suffix',h['offset']+h['size'],size-h['offset']-h['size'])]:
        digest=range_sha(path,off,length);assert digest==range_sha(b['packed']['path'],off,length);checks[key]=dict(offset=off,bytes=length,sha256=digest)
    with Path(path).open('rb') as f:f.seek(n['offset']);assert f.read(n['size'])==Path(b['final_norm']['path']).read_bytes();assert f.read(h['size'])==Path(b['head']['path']).read_bytes()
    return dict(fields=fields,all108_other_fields_exact=True,checks=checks)

def diagnostics(b,art,independent):
    import numpy as np
    f=read(art['features']);z=read(art['GPU_logits']);dense=read(art['direct_loss']);comp=read(art['compiled_loss']);gd=read(art['direct_gradient']);gc=read(art['compiled_gradient']);gf=read(art['backward_feature_gradient'])
    n=18;rec=b['record'];bits=np.fromfile(rec['logits']['path'],dtype='<u2').reshape(n,V);q,lq=probabilities(decode(bits),independent)
    p,lp=probabilities(z[0],independent)
    native=np.fromfile(art['native_scores']['path'],dtype='<f4').reshape(58,V);assert np.isfinite(native).all();sn=native[rec['positions']].astype('f8');pn,lpn=probabilities(sn,independent)
    kl=np.sum(q*(lq-lp),axis=1);nkl=np.sum(q*(lq-lpn),axis=1)
    H=read(b['head']).astype('f8');real=f[0].astype('f8')@H.T;delta=float(np.max(abs(real-z[0])))
    oldf=read(b['coordinate']['features'])
    with Path(b['packed']['path']).open('rb') as stream:stream.seek(b['fields']['final_norm']['offset']);gamma=np.frombuffer(stream.read(D*4),'<f4').astype('f8')
    g=oldf/gamma;u=2.**-24;g2=2*u/(1-2*u);e=np.asarray(b['coordinate']['feature_error_L2_upper_per_label'])/float(np.min(abs(gamma)));en=e+(2*g2/(1-g2))*(np.linalg.norm(g,axis=1)+e)
    hn=float(np.linalg.norm(H,axis=1).max());bound=hn*(en+b['native_rounding']['F32_gamma_dot']*(np.linalg.norm(g,axis=1)+en));pred=g@H.T;error=np.max(abs(pred-sn),axis=1)
    m=read(b['target']['moments']);c=read(b['target']['negative_entropy']);weights=np.full(n,.5/(n-1));weights[0]=.5
    shift=z[0]-z[0].max(1,keepdims=True);L=np.logaddexp.reduce(shift,axis=1) if independent else np.log(np.exp(shift).sum(1));cl=c+z[0].max(1)+L-np.sum(m*f[0].astype('f8'),axis=1)
    dg=((p-q)@H)*weights[:,None];cg=(p@H-m)*weights[:,None]
    nums=dict(loss_absolute=float(np.max(abs(dense-comp))),gradient_absolute=float(np.max(abs(gd-gc))),upstream_absolute=float(np.max(abs(gf-gc.astype('<f4')))),GPU_CPU_real_score_absolute=delta,GPU_CPU_dense_loss_absolute=float(np.max(abs(dense[0]-kl))),GPU_CPU_compiled_loss_absolute=float(np.max(abs(comp[0]-cl))),GPU_CPU_direct_gradient_absolute=float(np.max(abs(gd[0]-dg))),GPU_CPU_compiled_gradient_absolute=float(np.max(abs(gc[0]-cg))),GPU_native_score_absolute=float(np.max(abs(z[0]-sn))),GPU_native_KL_absolute=float(np.max(abs(kl-nkl))),GPU_native_argmax_disagreements=int(np.count_nonzero(z[0].argmax(1)!=sn.argmax(1))),native_coordinate_score_error_max=float(error.max()),native_coordinate_bound_max=float(bound.max()),native_coordinate_bound_utilization=float(np.max(error/bound)))
    gnum=b['numeric'];flags=dict(loss=nums['loss_absolute']<=gnum['loss_absolute'],gradient=nums['gradient_absolute']<=gnum['gradient_absolute'],upstream=nums['upstream_absolute']<=gnum['F32_upstream_absolute'],GPU_CPU_real_readout=delta<=gnum['loss_absolute'],GPU_CPU_losses=max(nums['GPU_CPU_dense_loss_absolute'],nums['GPU_CPU_compiled_loss_absolute'])<=gnum['loss_absolute'],GPU_CPU_gradients=max(nums['GPU_CPU_direct_gradient_absolute'],nums['GPU_CPU_compiled_gradient_absolute'])<=gnum['gradient_absolute'],GPU_native_scores=nums['GPU_native_score_absolute']<=gnum['GPU_native_score_absolute'],GPU_native_KL=nums['GPU_native_KL_absolute']<=gnum['GPU_native_KL_absolute'],GPU_native_argmax=nums['GPU_native_argmax_disagreements']==0,coordinate_bound=bool(np.all(error<=bound+gnum['bound_slack'])))
    wrong=(sn.argmax(1)!=decode(bits).argmax(1)).astype('i4')
    return dict(numeric=nums,numeric_flags=flags,native_KL=nkl.tolist(),native_case_KL=float(nkl.mean()),native_disagreement=int(wrong.sum()),native_quality_flags=dict(case_KL=float(nkl.mean())<=.05,case_disagreement=float(wrong.mean())<=.05),native_score_bounds=bound.tolist(),native_score_errors=error.tolist(),GPU_feature_native_distance_upper=(np.linalg.norm(g-f[0],axis=1)+en).tolist())

def capture(a):
    import numpy as np,psutil,torch
    from original_categorical_history_learner import CategoricalHistoryLearner
    start=time.monotonic();b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha;a.directory.mkdir(exist_ok=False);proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='restore';art={};reader=memory_reader();child_peak=0
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8';assert torch.cuda.device_count()==1;torch.cuda.reset_peak_memory_stats()
    device=dict(name=torch.cuda.get_device_name(),total_memory=torch.cuda.get_device_properties(0).total_memory,Torch=torch.__version__,CUDA=torch.version.cuda)
    def guard():
        lim=b['capture_limits'];assert time.monotonic()-start<lim['seconds']-lim['reserve_seconds'] and proc.memory_info().peak_wset+child_peak<=lim['OS_bytes'];assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
    last=0.
    def progress(mode,n,total,site):
        nonlocal last
        guard()
        if time.monotonic()-last>15:print(json.dumps(dict(stage=stage,operation=mode,site=site,n=n,total=total,seconds=time.monotonic()-start)),flush=True);last=time.monotonic()
    try:
        state=torch.load(b['source_state']['path'],map_location='cpu',weights_only=True,mmap=True);assert state['schema']=='ORIGINAL_DELTA_SEEDED_STATE_V1' and state['updates']==27 and not state['optimizer_partial_possible']
        model=CategoricalHistoryLearner(device='cuda');model.load_state_dict(state['model']);assert len(list(model.named_parameters()))==92 and sum(p.numel() for p in model.parameters())==721008128
        model.install_fixed_head(torch.from_numpy(read(b['head']).copy()));model.use_checkpoint=True;calls=[0]*6
        for site,layer in enumerate(model.layers):
            layer.bank.progress=lambda mode,n,total,site=site:progress(mode,n,total,site)
            layer.register_forward_pre_hook(lambda module,args,site=site:calls.__setitem__(site,calls[site]+1))
        payload=load_fit_supervision(b['compiled_result']['path'],b['compiled_audit']['path']);row=next(r for r in payload['records'] if r['id']==b['record']['id']);ids=torch.tensor([b['record']['student_input_ids']],device='cuda');pos=row['positions'].tolist();guard();stage='causal_forward'
        f=model.forward_features(ids,pos);assert f.dtype==torch.float32;f.retain_grad();fd=f.double();m=torch.from_numpy(row['moments'].copy()).to('cuda')[None];c=torch.from_numpy(row['negative_entropy'].copy()).to('cuda')[None];w=torch.from_numpy(row['loss_weights'].copy()).to('cuda')[None]
        z,comp,loss=model.categorical_objective(fd,m,c,w);bits=np.fromfile(b['record']['logits']['path'],dtype='<u2').reshape(18,V);source=torch.from_numpy(decode(bits)).to('cuda')[None];lq=torch.log_softmax(source,-1);q=lq.exp();dense=(q*(lq-torch.log_softmax(z,-1))).sum(-1)
        gd=torch.autograd.grad((dense*w).sum(),fd,retain_graph=True)[0];gc=torch.autograd.grad(loss,fd,retain_graph=True)[0];stage='causal_backward';loss.backward();torch.cuda.synchronize();guard()
        for key,value in [('features',f),('GPU_logits',z),('direct_loss',dense),('compiled_loss',comp),('direct_gradient',gd),('compiled_gradient',gc),('backward_feature_gradient',f.grad)]:art[key]=array(a.directory,key,value.detach().cpu().numpy())
        stage='gradient_state';stats={};parameters={};grads={};roles={}
        for name,p in model.named_parameters():
            info=tensor_info(p);expected=b['head'] if name=='head' else (b['final_norm'] if name=='final_norm' else None)
            if expected:assert torch.equal(p.detach().cpu(),torch.from_numpy(read(expected)))
            else:assert torch.equal(p.detach().cpu(),state['model'][name])
            parameters[name]=info
            if p.requires_grad:
                assert p.grad is not None;stats[name]=tensor_info(p.grad);grads[name]=p.grad.detach().cpu();role=gradient_role(name);roles[role]=roles.get(role,0.)+stats[name]['squared_norm']
            else:assert p.grad is None and name in ('head','final_norm')
            guard()
        assert len(stats)==90;gradient_path=a.directory/'gradients.pt';torch.save(grads,gradient_path);art['gradients']=extent(gradient_path);model.validate_fixed_geometry();guard()
        stage='packed';packed=a.directory/'candidate_categorical_causal.packed'
        with Path(b['packed']['path']).open('rb') as src,packed.open('xb') as dest:shutil.copyfileobj(src,dest,1<<20);dest.flush();os.fsync(dest.fileno())
        with packed.open('r+b') as stream:
            for key in ('final_norm','head'):stream.seek(b['fields'][key]['offset']);stream.write(Path(b[key]['path']).read_bytes())
            stream.flush();os.fsync(stream.fileno())
        exported=export_check(b,packed);art['packed']=extent(packed);stage='native_prefix';query=a.directory/'query.u32';raw(query,struct.pack('<I',58)+np.asarray(b['record']['student_input_ids'],dtype='<u4').tobytes());art['query']=extent(query)
        paths=[a.directory/x for x in ('native.scores.f32','native.routes','native.witness','native.summary.json')];command=[b['executable']['path'],'--prefix',str(packed),str(query),*[str(p) for p in paths]];began=time.monotonic()
        with (a.directory/'native.log').open('xb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=8);created=psutil.Process(child.pid).create_time()
            try:
                while child.poll() is None:child_peak=max(child_peak,reader(child));guard();assert time.monotonic()-began<b['capture_limits']['child_seconds'];time.sleep(.1)
                child_peak=max(child_peak,reader(child));assert child.returncode==0
            finally:
                if child.poll() is None:child.kill();child.wait()
                emit(a.directory/'native.receipt.json',dict(command=command,pid=child.pid,creation_time=created,exit_code=child.returncode,OS_peak=child_peak,seconds=time.monotonic()-began))
        for key,path in zip(('native_scores','native_routes','native_witness','native_summary'),paths):art[key]=extent(path)
        assert paths[1].read_bytes()==Path(b['old_routes']['path']).read_bytes() and paths[2].read_bytes()==Path(b['witness']['path']).read_bytes();stage='diagnostics';diag=diagnostics(b,art,False)
        diag['numeric_flags']['gradient_roles_nonzero']=all(roles.get(k,0.)>0 for k in ('bank','router','core','embed'));diag['numeric_flags']['checkpoint_blocks']=calls==[2]*6
        emit(a.out,dict(schema='ORIGINAL_CATEGORICAL_CAUSAL_PREFLIGHT_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,decision=decision(diag['numeric_flags']),device=device,artifacts=art,parameters=parameters,gradient_stats=stats,gradient_role_squared_norms=roles,diagnostics=diag,export=exported,block_forward_calls=calls,outer_history_forwards=1,checkpoint_recompute_blocks=sum(calls)-6,whole_backward_calls=1,state_gradient_checks=2,state_gradient_labels=36,native_histories=1,native_positions=58,paired_labels=18,original_routes_masses_witnesses_exact=True,all_core_bank_parameters_unchanged=True,optimizer_restore=False,optimizer_updates=0,source_calls=0,DEV_queries=0,reserved_queries=0,T4_calls=0,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak=proc.memory_info().peak_wset,native_OS_peak=child_peak,seconds=time.monotonic()-start,quality_admission=False,speed_admission=False,scope=b['scope']));guard();print(json.dumps(native_values(dict(decision=json.loads(a.out.read_bytes())['decision'],diagnostics=diag,seconds=time.monotonic()-start))),flush=True)
    except BaseException as error:emit(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),artifacts=art,seconds=time.monotonic()-start));raise

def gradient_role(name):
    return 'bank' if '.bank.' in name and name.endswith(('gate','up','down')) else ('router' if '.router.' in name else ('embed' if name=='embed' else 'core'))

def decision(flags):
    return 'CAUSAL_BRIDGE_QUALIFIED_PENDING_TRAINING' if all(flags.values()) else 'CAUSAL_BRIDGE_NUMERIC_NOT_QUALIFIED'

def audit(a):
    import numpy as np,psutil,torch
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([1]);torch.set_num_threads(1);b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes());assert sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256'] and r['freeze']==t['freeze']==a.freeze and t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    def guard():assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes'] and not proc.children(recursive=True)
    for i in b['inputs']+t['outputs']:assert extent(i['path'])==i;guard()
    by={i['path']:i for i in t['outputs']}
    for i in r['artifacts'].values():assert by[i['path']]=={k:i[k] for k in ('path','bytes','sha256')}
    assert json.loads(json.dumps(export_check(b,r['artifacts']['packed']['path'])))==r['export'];state=torch.load(b['source_state']['path'],map_location='cpu',weights_only=True,mmap=True)
    assert state['schema']=='ORIGINAL_DELTA_SEEDED_STATE_V1' and state['updates']==27 and not state['optimizer_partial_possible']
    assert set(r['parameters'])==set(state['model']) and len(r['parameters'])==92 and sum(x['entries'] for x in r['parameters'].values())==721008128
    for name,info in r['parameters'].items():
        value=torch.from_numpy(read(b[name])) if name in ('head','final_norm') else state['model'][name];assert tensor_info(value)==info;guard()
    gradients=torch.load(r['artifacts']['gradients']['path'],map_location='cpu',weights_only=True,mmap=True);assert set(gradients)==set(r['gradient_stats']) and len(gradients)==90
    roles={}
    for name,value in gradients.items():
        assert name not in ('head','final_norm') and tensor_info(value)==r['gradient_stats'][name];role=gradient_role(name);roles[role]=roles.get(role,0.)+r['gradient_stats'][name]['squared_norm'];guard()
    assert roles==r['gradient_role_squared_norms'] and set(roles)=={'bank','router','embed','core'}
    assert Path(r['artifacts']['query']['path']).read_bytes()==struct.pack('<I',58)+np.asarray(b['record']['student_input_ids'],dtype='<u4').tobytes()
    receipt_path=Path(r['artifacts']['packed']['path']).parent/'native.receipt.json';assert extent(receipt_path)==by[str(receipt_path.resolve())];native=json.loads(receipt_path.read_bytes())
    expected_command=[b['executable']['path'],'--prefix',r['artifacts']['packed']['path'],r['artifacts']['query']['path'],*[r['artifacts'][key]['path'] for key in ('native_scores','native_routes','native_witness','native_summary')]]
    assert native['command']==expected_command and native['exit_code']==0 and native['OS_peak']==r['native_OS_peak'] and 0<native['seconds']<=b['capture_limits']['child_seconds']
    assert Path(r['artifacts']['native_routes']['path']).read_bytes()==Path(b['old_routes']['path']).read_bytes() and Path(r['artifacts']['native_witness']['path']).read_bytes()==Path(b['witness']['path']).read_bytes()
    diag=diagnostics(b,r['artifacts'],True);saved=r['diagnostics'];delta=max(abs(diag['numeric'][k]-saved['numeric'][k]) for k in diag['numeric']);assert delta<=b['numeric']['metric_absolute'];assert max(abs(np.asarray(diag['native_KL'])-saved['native_KL']))<=b['numeric']['metric_absolute']
    assert diag['native_disagreement']==saved['native_disagreement'] and diag['native_quality_flags']==saved['native_quality_flags']
    for k,v in diag['numeric_flags'].items():assert v==saved['numeric_flags'][k]
    diag['numeric_flags']['gradient_roles_nonzero']=all(roles.get(k,0.)>0 for k in ('bank','router','core','embed'));diag['numeric_flags']['checkpoint_blocks']=r['block_forward_calls']==[2]*6
    assert diag['numeric_flags']==saved['numeric_flags'] and decision(diag['numeric_flags'])==r['decision']
    assert len(r['block_forward_calls'])==6 and all(n>=1 for n in r['block_forward_calls']) and r['outer_history_forwards']==r['whole_backward_calls']==r['native_histories']==1 and r['checkpoint_recompute_blocks']==sum(r['block_forward_calls'])-6 and r['state_gradient_checks']==2 and r['state_gradient_labels']==36 and r['native_positions']==58 and r['paired_labels']==18
    assert r['all_core_bank_parameters_unchanged'] and r['original_routes_masses_witnesses_exact'] and not r['optimizer_restore'] and r['optimizer_updates']==r['source_calls']==r['DEV_queries']==r['reserved_queries']==r['T4_calls']==0
    assert not r['quality_admission'] and not r['speed_admission'];assert r['GPU_allocated_peak']<=b['capture_limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['capture_limits']['GPU_reserved_bytes']
    assert r['worker_OS_peak']+r['native_OS_peak']<=b['capture_limits']['OS_bytes'] and r['seconds']<b['capture_limits']['seconds']-b['capture_limits']['reserve_seconds'] and sum(i['bytes'] for i in t['outputs'])+a.source_result.stat().st_size<=b['capture_limits']['output_bytes'];guard()
    emit(a.out,dict(schema='ORIGINAL_CATEGORICAL_CAUSAL_PREFLIGHT_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=r['decision'],complete_input_output_hashes=True,all92_parameters_and90_gradients_verified=True,all108_other_packed_fields_exact=True,all18_independent_native_metrics_and_coordinate_bounds_verified=True,numeric_flags=saved['numeric_flags'],max_diagnostic_delta=delta,gradient_tensors=90,optimizer_updates=0,source_calls=0,history_forwards=0,backward_calls=0,native_calls=0,GPU_calls=0,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,scope='Stored-only audit; no model/history/backward replay. Finite gradient/hash custody is not an independent proof of every nonlinear surrogate derivative.'))
    print(json.dumps(dict(decision=r['decision'],seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('bind','capture-worker','audit-worker'):p.add_argument('--'+k,action='store_true')
    for k in ('binding','directory','out','source-result','protocol'):p.add_argument('--'+k,type=Path)
    for k in ('binding-sha','freeze'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.bind:bind(a)
    elif a.capture_worker:capture(a)
    elif a.audit_worker:audit(a)
    else:raise RuntimeError('Use held GPU/CPU launcher')
