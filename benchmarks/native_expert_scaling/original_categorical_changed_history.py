"""One changed-master causal backward, compared with already qualified C traces."""
import argparse, json, os, struct, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from original_packed_capacity import SITE, extent, sha
sys.path.insert(0, str(SITE))
from original_categorical_causal_preflight import emit, array, read, tensor_info, gradient_role
from original_categorical_supervision import load_fit_supervision
from original_categorical_loss_compile import probabilities, decode

V, D, T, L, K, E = 65537, 256, 58, 6, 8, 1152


def qualified_result(path):
    result = json.loads(path.read_bytes())
    terminal = json.loads(path.with_suffix('.terminal.json').read_bytes())
    assert terminal['exit_code'] == 0 and terminal['error'] is None
    assert terminal['inputs_before_after_exact'] and terminal['result'] == extent(path)
    return result


def bind(a):
    import numpy as np, psutil, torch
    import torch._dynamo
    from original_categorical_history_learner import CategoricalHistoryLearner
    assert not a.binding.exists()
    parent = DOC / 'original_categorical_trust_step_binding_20261010.json'
    result = DOC / 'original_categorical_trust_step_result_20261010.json'
    adjudication = DOC / 'original_categorical_trust_fsum_adjudication_20261010.json'
    causal = DOC / 'original_categorical_causal_preflight_reviewed_binding_20261010.json'
    b = json.loads(parent.read_bytes()); cb = json.loads(causal.read_bytes())
    r, ar = qualified_result(result), qualified_result(adjudication)
    assert r['decision'] == ar['decision'] == 'GROUPED_DISCRETE_DESCENT_QUALIFIED'
    assert all(ar['selection']['flags'].values()) and ar['selection'] == r['selection']
    assert ar['result'] == extent(result) and ar['binding'] == extent(parent)
    assert ar['complete_input_output_hashes'] and ar['all3_92_master_states_110_fields_verified']
    assert ar['all_group_updates_independent_fsum_verified'] and r['binding_sha256'] == sha(parent)
    row = next(x for x in r['records'] if x['alpha'] == r['selection']['alpha'])
    assert row['alpha'] == .001 and all(row['numeric_flags'].values())
    assert cb['record'] == b['record'] and cb['head'] == b['head'] and cb['final_norm'] == b['final_norm']
    files = [Path(__file__), a.protocol, parent, result, adjudication, causal,
             result.with_suffix('.terminal.json'), adjudication.with_suffix('.terminal.json'),
             Path(cb['compiled_result']['path']), Path(cb['compiled_audit']['path']),
             Path(cb['compiled_audit']['path']).with_suffix('.terminal.json'),
             Path(b['record']['logits']['path']), Path(b['head']['path']), Path(b['final_norm']['path'])]
    files += [Path(item['path']) for item in row['artifacts'].values()]
    files += [Path(row['artifacts']['packed']['path']).parent / 'native.receipt.json']
    compiled = json.loads(Path(cb['compiled_result']['path']).read_bytes())
    files += [Path(x[k]['path']) for x in compiled['records'] if x['split'] == 'FIT'
              for k in ('moments', 'negative_entropy', 'source_argmax')]
    for module in list(sys.modules.values()):
        for key in ('__file__', '__cached__'):
            p = getattr(module, key, None)
            if isinstance(p, (str, os.PathLike)) and Path(p).is_file(): files.append(Path(p))
    files += [Path(sys.executable), Path(sys.executable).parent / 'python312.dll']
    files += sorted((SITE/'torch/lib').glob('*.dll')) + sorted((SITE/'numpy.libs').glob('*.dll'))
    files += sorted((SITE/'psutil').glob('*.pyd'))
    inputs = [extent(p) for p in dict.fromkeys(p.resolve() for p in files)]
    for item in [*row['artifacts'].values(), b['head'], b['final_norm'], b['record']['logits'],
                 cb['compiled_result'], cb['compiled_audit']]:
        assert extent(item['path']) == {k: item[k] for k in ('path', 'bytes', 'sha256')}
    assert (torch.__version__, np.__version__, psutil.__version__) == ('2.6.0+cu124', '2.4.6', '7.2.2')
    emit(a.binding, dict(schema='ORIGINAL_CATEGORICAL_CHANGED_HISTORY_BINDING_V1', inputs=inputs,
        parent_binding=extent(parent), parent_result=extent(result), parent_audit=extent(adjudication),
        source_state=row['artifacts']['masters'], source_parameters=row['parameters'],
        selected_alpha=row['alpha'], selected=row, original_gradient_source=b['gradient_source'],
        baseline_state=b['source_state'], record=b['record'], target=cb['target'],
        head=b['head'], final_norm=b['final_norm'], compiled_result=cb['compiled_result'],
        compiled_audit=cb['compiled_audit'], native=row['artifacts'],
        numeric=dict(loss_absolute=1e-8, gradient_absolute=1e-8, F32_upstream_absolute=1e-6,
                     GPU_native_score_absolute=1e-3, GPU_native_KL_absolute=1e-3,
                     routing_mass_absolute=1e-5, routing_mass_defect=1e-6, metric_absolute=1e-8),
        capture_limits=dict(seconds=900, reserve_seconds=120, OS_bytes=24<<30,
                            GPU_allocated_bytes=6<<30, GPU_reserved_bytes=8<<30,
                            output_bytes=4<<30, log_bytes=8<<20),
        audit_limits=dict(seconds=900, OS_bytes=12<<30, output_bytes=2<<20, log_bytes=8<<20),
        runtime=dict(Torch=torch.__version__, NumPy=np.__version__, psutil=psutil.__version__,
                     actual_imported_modules_and_bytecode=True, junction_resolved=True),
        scope='ONE new GPU history/backward at the qualified changed alpha.001 master state. '
              'Reuse existing C scores/routes; no C prefix, export, old Adam, optimizer update, '
              'old-point coordinate bound, source/DEV/RESERVED/T4 or quality/speed admission.'))
    print(json.dumps(dict(binding_sha256=sha(a.binding), inputs=len(inputs),
                          bytes=sum(i['bytes'] for i in inputs))), flush=True)


def changed_state(b):
    import torch
    state = torch.load(b['source_state']['path'], map_location='cpu', weights_only=True, mmap=True)
    assert state['schema'] == 'ORIGINAL_CATEGORICAL_GROUPED_MASTERS_V1'
    assert state['alpha'] == b['selected_alpha'] and state['binding_sha256'] == b['parent_binding']['sha256']
    assert state['source_state'] == b['baseline_state'] and state['gradient_source'] == b['original_gradient_source']
    assert state['optimizer_updates'] == 0 and state['baseline_updates'] == 27
    assert state['saved_gradient_displacements'] == 1
    assert set(state['model']) == set(b['source_parameters']) and len(state['model']) == 92
    return state


def native_routes(item):
    import numpy as np
    dtype = np.dtype([('ids','<i4',(K,)), ('mass','<f4',(K,))])
    assert item['bytes'] == T*L*dtype.itemsize
    rows = np.fromfile(item['path'], dtype=dtype).reshape(T,L)
    return rows['ids'], rows['mass']


def route_diagnostics(b, art):
    import numpy as np
    ids, mass = read(art['forward_route_ids']), read(art['forward_route_mass'])
    post_ids, post_mass = read(art['backward_route_ids']), read(art['backward_route_mass'])
    ci, cm = native_routes(b['native']['native_routes'])
    assert ids.shape == mass.shape == post_ids.shape == post_mass.shape == (T,L,K)
    assert ids.dtype == post_ids.dtype == np.dtype('<i4') and mass.dtype == post_mass.dtype == np.dtype('<f4')
    valid = bool(np.all((ids >= 0) & (ids < E)) and np.all(np.diff(np.sort(ids,axis=-1),axis=-1) > 0)
                 and np.all(mass >= 0))
    cvalid = bool(np.all((ci >= 0) & (ci < E)) and np.all(np.diff(np.sort(ci,axis=-1),axis=-1) > 0)
                  and np.isfinite(cm).all() and np.all(cm >= 0))
    numbers = dict(route_ID_slot_disagreements=int(np.count_nonzero(ids != ci)),
        route_set_disagreements=int(np.any(np.sort(ids,axis=-1) != np.sort(ci,axis=-1),axis=-1).sum()),
        route_mass_slot_absolute=float(np.max(abs(mass.astype('f8')-cm.astype('f8')))),
        GPU_mass_defect=float(np.max(abs(mass.astype('f8').sum(-1)-1))),
        C_mass_defect=float(np.max(abs(cm.astype('f8').sum(-1)-1))),
        checkpoint_ID_bit_disagreements=int(np.count_nonzero(ids != post_ids)),
        checkpoint_mass_bit_disagreements=int(np.count_nonzero(mass.view('<u4') != post_mass.view('<u4'))))
    flags = dict(route_valid=valid and cvalid, route_ID_exact=numbers['route_ID_slot_disagreements']==0,
        route_mass=numbers['route_mass_slot_absolute'] <= b['numeric']['routing_mass_absolute'],
        route_normalization=max(numbers['GPU_mass_defect'],numbers['C_mass_defect']) <= b['numeric']['routing_mass_defect'],
        checkpoint_route_exact=numbers['checkpoint_ID_bit_disagreements']==numbers['checkpoint_mass_bit_disagreements']==0)
    return numbers, flags


def diagnostics(b, art, independent):
    import numpy as np
    f,z,dense,comp,gd,gc,gf = [read(art[k]) for k in ('features','GPU_logits','direct_loss',
        'compiled_loss','direct_gradient','compiled_gradient','backward_feature_gradient')]
    assert f.shape == gd.shape == gc.shape == gf.shape == (1,18,D)
    assert z.shape == (1,18,V) and dense.shape == comp.shape == (1,18)
    bits=np.fromfile(b['record']['logits']['path'],dtype='<u2').reshape(18,V)
    q,lq=probabilities(decode(bits),independent);p,lp=probabilities(z[0],independent)
    native=np.fromfile(b['native']['native_scores']['path'],dtype='<f4').reshape(T,V)
    assert np.isfinite(native).all();sn=native[b['record']['positions']].astype('f8')
    _,lpn=probabilities(sn,independent);kl=np.sum(q*(lq-lp),axis=1);nkl=np.sum(q*(lq-lpn),axis=1)
    H=read(b['head']).astype('f8');real=f[0].astype('f8')@H.T
    m=read(b['target']['moments']);c=read(b['target']['negative_entropy'])
    weights=np.full(18,.5/17);weights[0]=.5
    shift=z[0]-z[0].max(1,keepdims=True)
    normalizer=np.logaddexp.reduce(shift,axis=1) if independent else np.log(np.exp(shift).sum(1))
    cl=c+z[0].max(1)+normalizer-np.sum(m*f[0].astype('f8'),axis=1)
    direct=((p-q)@H)*weights[:,None];compiled=(p@H-m)*weights[:,None]
    nums=dict(loss_absolute=float(np.max(abs(dense-comp))),gradient_absolute=float(np.max(abs(gd-gc))),
        upstream_absolute=float(np.max(abs(gf-gc.astype('<f4')))),
        GPU_CPU_real_score_absolute=float(np.max(abs(real-z[0]))),
        GPU_CPU_dense_loss_absolute=float(np.max(abs(dense[0]-kl))),
        GPU_CPU_compiled_loss_absolute=float(np.max(abs(comp[0]-cl))),
        GPU_CPU_direct_gradient_absolute=float(np.max(abs(gd[0]-direct))),
        GPU_CPU_compiled_gradient_absolute=float(np.max(abs(gc[0]-compiled))),
        GPU_native_score_absolute=float(np.max(abs(z[0]-sn))),
        GPU_native_KL_absolute=float(np.max(abs(kl-nkl))),
        GPU_native_argmax_disagreements=int(np.count_nonzero(z[0].argmax(1)!=sn.argmax(1))))
    n=b['numeric'];flags=dict(loss=nums['loss_absolute']<=n['loss_absolute'],
        gradient=nums['gradient_absolute']<=n['gradient_absolute'], upstream=nums['upstream_absolute']<=n['F32_upstream_absolute'],
        GPU_CPU_real_readout=nums['GPU_CPU_real_score_absolute']<=n['loss_absolute'],
        GPU_CPU_losses=max(nums['GPU_CPU_dense_loss_absolute'],nums['GPU_CPU_compiled_loss_absolute'])<=n['loss_absolute'],
        GPU_CPU_gradients=max(nums['GPU_CPU_direct_gradient_absolute'],nums['GPU_CPU_compiled_gradient_absolute'])<=n['gradient_absolute'],
        GPU_native_scores=nums['GPU_native_score_absolute']<=n['GPU_native_score_absolute'],
        GPU_native_KL=nums['GPU_native_KL_absolute']<=n['GPU_native_KL_absolute'],
        GPU_native_argmax=nums['GPU_native_argmax_disagreements']==0)
    route_nums,route_flags=route_diagnostics(b,art);nums.update(route_nums);flags.update(route_flags)
    wrong=int(np.count_nonzero(sn.argmax(1)!=decode(bits).argmax(1)))
    return dict(numeric=nums,numeric_flags=flags,native_KL=nkl.tolist(),native_case_KL=float(nkl.mean()),
        native_weighted_KL=float(nkl@weights),native_disagreement=wrong,
        native_quality_flags=dict(case_KL=float(nkl.mean())<=.05,case_disagreement=wrong/18<=.05))


def decision(flags):
    return 'CHANGED_CAUSAL_BRIDGE_QUALIFIED' if all(flags.values()) else 'CHANGED_CAUSAL_BRIDGE_NUMERIC_NOT_QUALIFIED'


def capture(a):
    import numpy as np,psutil,torch
    from original_categorical_history_learner import CategoricalHistoryLearner
    start=time.monotonic();b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    a.directory.mkdir(exist_ok=False);proc=psutil.Process();proc.cpu_affinity(list(range(6)));stage='restore';art={}
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and torch.cuda.device_count()==1
    torch.cuda.reset_peak_memory_stats()
    device=dict(name=torch.cuda.get_device_name(),total_memory=torch.cuda.get_device_properties(0).total_memory,
                Torch=torch.__version__,CUDA=torch.version.cuda)
    def guard():
        lim=b['capture_limits']
        assert time.monotonic()-start<lim['seconds']-lim['reserve_seconds'] and proc.memory_info().peak_wset<=lim['OS_bytes']
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        assert not proc.children(recursive=True)
    last=0.
    def progress(mode,n,total,site):
        nonlocal last
        guard()
        if time.monotonic()-last>15:
            print(json.dumps(dict(stage=stage,operation=mode,site=site,n=n,total=total,
                                  seconds=time.monotonic()-start)),flush=True);last=time.monotonic()
    try:
        state=changed_state(b);model=CategoricalHistoryLearner(device='cuda');model.load_state_dict(state['model'])
        assert len(list(model.named_parameters()))==92 and sum(p.numel() for p in model.parameters())==721008128
        model.install_fixed_head(torch.from_numpy(read(b['head']).copy()));model.use_checkpoint=True;calls=[0]*L
        for site,layer in enumerate(model.layers):
            layer.bank.progress=lambda mode,n,total,site=site:progress(mode,n,total,site)
            layer.register_forward_pre_hook(lambda module,args,site=site:calls.__setitem__(site,calls[site]+1))
        payload=load_fit_supervision(b['compiled_result']['path'],b['compiled_audit']['path'])
        row=next(r for r in payload['records'] if r['id']==b['record']['id'])
        assert row['positions'].tolist()==b['record']['positions'] and payload['head']==b['head'] and payload['final_norm']==b['final_norm']
        ids=torch.tensor([b['record']['student_input_ids']],device='cuda');guard();stage='causal_forward'
        f=model.forward_features(ids,row['positions'].tolist());assert f.dtype==torch.float32;f.retain_grad()
        forward=[(layer.bank.last_routes[0].clone(),layer.bank.last_routes[1].clone()) for layer in model.layers]
        for k,idx,dtype in (('forward_route_ids',0,'<i4'),('forward_route_mass',1,'<f4')):
            art[k]=array(a.directory,k,np.stack([trace[idx].numpy() for trace in forward],axis=1).astype(dtype))
        fd=f.double();m=torch.from_numpy(row['moments'].copy()).to('cuda')[None]
        c=torch.from_numpy(row['negative_entropy'].copy()).to('cuda')[None];w=torch.from_numpy(row['loss_weights'].copy()).to('cuda')[None]
        z,comp,loss=model.categorical_objective(fd,m,c,w)
        bits=np.fromfile(b['record']['logits']['path'],dtype='<u2').reshape(18,V)
        source=torch.from_numpy(decode(bits)).to('cuda')[None];lq=torch.log_softmax(source,-1);q=lq.exp()
        dense=(q*(lq-torch.log_softmax(z,-1))).sum(-1)
        gd=torch.autograd.grad((dense*w).sum(),fd,retain_graph=True)[0]
        gc=torch.autograd.grad(loss,fd,retain_graph=True)[0]
        stage='causal_backward';loss.backward();torch.cuda.synchronize();guard()
        for k,idx,dtype in (('backward_route_ids',0,'<i4'),('backward_route_mass',1,'<f4')):
            art[k]=array(a.directory,k,np.stack([layer.bank.last_routes[idx].numpy() for layer in model.layers],axis=1).astype(dtype))
        for key,value in [('features',f),('GPU_logits',z),('direct_loss',dense),('compiled_loss',comp),
                          ('direct_gradient',gd),('compiled_gradient',gc),('backward_feature_gradient',f.grad)]:
            art[key]=array(a.directory,key,value.detach().cpu().numpy())
        stage='gradient_state';stats={};parameters={};grads={};roles={}
        for name,p in model.named_parameters():
            info=tensor_info(p);assert info==b['source_parameters'][name],name
            assert np.array_equal(p.detach().cpu().numpy().view('<u4'),state['model'][name].numpy().view('<u4')),name
            parameters[name]=info
            if p.requires_grad:
                assert p.grad is not None;stats[name]=tensor_info(p.grad);grads[name]=p.grad.detach().cpu()
                role=gradient_role(name);roles[role]=roles.get(role,0.)+stats[name]['squared_norm']
            else:assert p.grad is None and name in ('head','final_norm')
            guard()
        assert len(stats)==90;gradient_path=a.directory/'gradients.pt';torch.save(grads,gradient_path)
        art['gradients']=extent(gradient_path);model.validate_fixed_geometry();guard();stage='diagnostics'
        diag=diagnostics(b,art,False)
        assert max(abs(np.asarray(diag['native_KL'])-b['selected']['metrics']['KL_per_label']))<=b['numeric']['metric_absolute']
        diag['numeric_flags']['gradient_roles_nonzero']=all(roles.get(k,0.)>0 for k in ('bank','router','core','embed'))
        diag['numeric_flags']['checkpoint_blocks']=calls==[2]*L
        emit(a.out,dict(schema='ORIGINAL_CATEGORICAL_CHANGED_HISTORY_RESULT_V1',freeze=a.freeze,
            binding_sha256=a.binding_sha,decision=decision(diag['numeric_flags']),device=device,artifacts=art,
            parameters=parameters,gradient_stats=stats,gradient_role_squared_norms=roles,diagnostics=diag,
            block_forward_calls=calls,outer_history_forwards=1,checkpoint_recompute_blocks=sum(calls)-L,
            whole_backward_calls=1,state_gradient_checks=2,state_gradient_labels=36,paired_labels=18,
            native_histories=0,native_positions=0,reused_native_histories=1,reused_native_positions=T,
            route_rows=T*L,all_changed_parameters_unchanged=True,optimizer_restore=False,
            optimizer_updates=0,source_calls=0,DEV_queries=0,reserved_queries=0,T4_calls=0,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak=proc.memory_info().peak_wset,seconds=time.monotonic()-start,
            quality_admission=False,speed_admission=False,scope=b['scope']))
        guard();print(json.dumps(dict(decision=decision(diag['numeric_flags']),diagnostics=diag,
                                      seconds=time.monotonic()-start)),flush=True)
    except BaseException as error:
        emit(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),artifacts=art,
                                                seconds=time.monotonic()-start));raise


def audit(a):
    import numpy as np,psutil,torch
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([1]);torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes())
    t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256']
    assert r['freeze']==t['freeze']==a.freeze and t['exit_code']==0 and t['error'] is None
    assert t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    def guard():
        assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes']
        assert not proc.children(recursive=True)
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item;guard()
    by={item['path']:item for item in t['outputs']}
    for item in r['artifacts'].values():assert by[item['path']]=={k:item[k] for k in ('path','bytes','sha256')}
    state=changed_state(b)
    assert set(r['parameters'])==set(state['model']) and len(r['parameters'])==92
    assert sum(x['entries'] for x in r['parameters'].values())==721008128
    for name,value in state['model'].items():
        assert tensor_info(value)==r['parameters'][name]==b['source_parameters'][name]
        if name in ('head','final_norm'):assert np.array_equal(value.numpy().view('<u4'),read(b[name]).view('<u4'))
        guard()
    gradients=torch.load(r['artifacts']['gradients']['path'],map_location='cpu',weights_only=True,mmap=True)
    assert set(gradients)==set(r['gradient_stats'])==set(state['model'])-{'head','final_norm'} and len(gradients)==90
    roles={}
    for name,value in gradients.items():
        assert tensor_info(value)==r['gradient_stats'][name];role=gradient_role(name)
        roles[role]=roles.get(role,0.)+r['gradient_stats'][name]['squared_norm'];guard()
    assert roles==r['gradient_role_squared_norms'] and set(roles)=={'bank','router','embed','core'}
    diag=diagnostics(b,r['artifacts'],True);saved=r['diagnostics']
    delta=max(abs(diag['numeric'][k]-saved['numeric'][k]) for k in diag['numeric'])
    assert delta<=b['numeric']['metric_absolute']
    assert max(abs(np.asarray(diag['native_KL'])-saved['native_KL']))<=b['numeric']['metric_absolute']
    assert max(abs(np.asarray(diag['native_KL'])-b['selected']['metrics']['KL_per_label']))<=b['numeric']['metric_absolute']
    for key in ('native_case_KL','native_weighted_KL'):
        assert abs(diag[key]-saved[key])<=b['numeric']['metric_absolute']
    assert diag['native_disagreement']==saved['native_disagreement'] and diag['native_quality_flags']==saved['native_quality_flags']
    diag['numeric_flags']['gradient_roles_nonzero']=all(roles.get(k,0.)>0 for k in ('bank','router','core','embed'))
    diag['numeric_flags']['checkpoint_blocks']=r['block_forward_calls']==[2]*L
    assert diag['numeric_flags']==saved['numeric_flags'] and decision(diag['numeric_flags'])==r['decision']
    assert len(r['block_forward_calls'])==L and all(n>=1 for n in r['block_forward_calls'])
    assert r['outer_history_forwards']==r['whole_backward_calls']==r['reused_native_histories']==1
    assert r['checkpoint_recompute_blocks']==sum(r['block_forward_calls'])-L
    assert r['state_gradient_checks']==2 and r['state_gradient_labels']==36 and r['paired_labels']==18
    assert r['reused_native_positions']==T and r['route_rows']==T*L and r['all_changed_parameters_unchanged']
    assert not r['optimizer_restore'] and all(r[k]==0 for k in ('optimizer_updates','source_calls',
        'DEV_queries','reserved_queries','T4_calls','native_histories','native_positions'))
    assert not r['quality_admission'] and not r['speed_admission']
    lim=b['capture_limits']
    assert r['GPU_allocated_peak']<=lim['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=lim['GPU_reserved_bytes']
    assert r['worker_OS_peak']<=lim['OS_bytes'] and r['seconds']<lim['seconds']-lim['reserve_seconds']
    assert sum(i['bytes'] for i in t['outputs'])+a.source_result.stat().st_size<=lim['output_bytes'];guard()
    emit(a.out,dict(schema='ORIGINAL_CATEGORICAL_CHANGED_HISTORY_AUDIT_V1',result=extent(a.source_result),
        binding=extent(a.binding),decision=r['decision'],complete_input_output_hashes=True,
        all92_changed_parameters_and90_new_gradients_verified=True,all348_route_rows_and_checkpoint_traces_verified=True,
        all18_independent_native_metrics_verified=True,numeric_flags=saved['numeric_flags'],
        max_diagnostic_delta=delta,gradient_tensors=90,optimizer_updates=0,source_calls=0,
        history_forwards=0,backward_calls=0,native_calls=0,GPU_calls=0,seconds=time.monotonic()-start,
        OS_peak=proc.memory_info().peak_wset,
        scope='Stored-only full audit at one changed parameter point. Native artifacts reuse qualified '
              'full packing/quantization audit under exact hashes; no new export or C call. '
              'Finite gradient custody is not an independent proof of every nonlinear STE derivative.'))
    print(json.dumps(dict(decision=r['decision'],seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bind','capture-worker','audit-worker'):p.add_argument('--'+key,action='store_true')
    for key in ('binding','directory','out','source-result','protocol'):p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.bind:bind(a)
    elif a.capture_worker:capture(a)
    elif a.audit_worker:audit(a)
    else:raise RuntimeError('Use held GPU/CPU launcher')
