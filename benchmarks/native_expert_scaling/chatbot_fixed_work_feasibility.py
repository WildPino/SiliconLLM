"""Actual parent-state conversion and two priced whole-model updates, fixed2+6."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    report_path = DOC/'chatbot_broad_pilot_result_repair1_20261009.json'
    receipt_path = report_path.with_suffix('.terminal.json')
    parent_binding = DOC/'chatbot_broad_pilot_binding_repair1_20261009.json'
    report = json.loads(report_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    old = json.loads(parent_binding.read_bytes())
    assert receipt['exit_code'] == 0 and receipt['resource_gates']
    assert receipt['result_sha256'] == sha(report_path)
    assert sha(parent_binding) == report['binding_sha256'] == receipt['binding_sha256']
    assert report['new_updates'] == 24 and report['final_optimizer_step'] == 318
    state = report['checkpoint']
    assert state['sha256'] == '16a85448a706f96719c8d952bbfd5309b8bdf60237f763dfdf101bad1bb9983f'
    assert Path(state['path']).stat().st_size == state['bytes'] == 3059728406
    assert sha(state['path']) == state['sha256']
    inherited = {Path(v['path']).resolve(): v for v in old['inputs']}
    corpus = Path(old['corpus'])
    records = json.loads(corpus.read_bytes())['records']
    fit = [r for r in records if r['split'] == 'FIT']
    selected = [min(fit, key=lambda r:(len(r['student_input_ids']),r['id'])),
                min(fit, key=lambda r:(-len(r['student_input_ids']),r['id']))]
    assert [len(r['student_input_ids']) for r in selected] == [58,1507]
    baseline_by_id = {r['id']: r for r in report['after']['cases']}
    baseline = {r['id']: baseline_by_id[r['id']] for r in selected}
    paths = [Path(__file__),Path(sys.executable),B/'chatbot_hybrid_fixed_work_target.py',
             B/'chatbot_hybrid_shared_private_target.py',B/'chatbot_hybrid_target.py',
             B/'chatbot_target_ssd_storage.py',B/'chatbot_falcon_ssd_tiles.py',
             B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
             ROOT/'benchmarks/phase60/engine.c',report_path,receipt_path,parent_binding,
             DOC/'CHATBOT_FIXED_WORK_FEASIBILITY_PROTOCOL_20261009.md',
             Path(state['path']),corpus]
    checked = [corpus,B/'chatbot_hybrid_target.py',B/'chatbot_target_ssd_storage.py',B/'chatbot_falcon_ssd_tiles.py']
    for rec in selected:
        paths.extend([Path(rec['logits']['path']),Path(baseline[rec['id']]['logits']['path'])])
        for entry in (rec['logits'],baseline[rec['id']]['logits']):
            p = Path(entry['path'])
            assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
        assert rec['student_input_ids'] == rec['input_ids']+rec['output_ids'][:-1]
        checked.append(Path(rec['logits']['path']))
    for p in checked:
        v = inherited[p.resolve()]
        assert p.stat().st_size == v['bytes'] and sha(p) == v['sha256'], str(p)
    for p,v in inherited.items():
        if p.is_relative_to(SITE.resolve()):
            assert p.stat().st_size == v['bytes'] and sha(p) == v['sha256']
            paths.append(p)
    foreign = {
        'benchmarks/phase60/engine.c':'5f948fc0dcd28b1647a2a2c73067bdc0a76a7d41ae7f34395ede8120aeaa83ce',
        'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
        'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
        'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    for p,digest in foreign.items():
        assert sha(ROOT/p) == digest
        paths.append(ROOT/p)
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    assert a.freeze
    value = dict(schema='FIXED_WORK_FEASIBILITY_BINDING_V1',freeze=a.freeze,python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),parent_checkpoint=state,parent_result=str(report_path),
        corpus=str(corpus.resolve()),case_ids=[r['id'] for r in selected],baseline=baseline,
        update_case=selected[-1]['id'],new_updates=2,initial_optimizer_step=318,
        common_seed_private_ids=[0,36],initial_common_down_scale=5e-5,
        learning_rate=5e-5,normalized_AQ_dither=.025,clip_norm=1.,
        limits=dict(seconds=300,reserve_seconds=45,OS_bytes=8<<30,GPU_allocated_bytes=10<<30,
                    GPU_reserved_bytes=11<<30,output_bytes=4<<30),
        runtime_binding_scope='Actual parent state/receipt, used two source and baseline packets, corpus, frozen target/storage code and selected inherited runtime files; not complete DLL-tree or CUDA determinism.',
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths])
    write(a.out,value)
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(paths),cases=value['case_ids'])),flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'FIXED_WORK_FEASIBILITY_BINDING_V1'
    assert b['freeze'] == a.freeze
    a.directory.mkdir(exist_ok=False)
    phase = 'imports'; completed = 0; durable = None; updates = []; torch = None
    target = None; optimizer = None; snapshot = None
    try:
        assert sys.version_info[:3] == (3,12,10) and Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels','mamba_ssm','causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from torch.utils.checkpoint import checkpoint
        from transformers.models.falcon_h1 import modeling_falcon_h1 as source_code
        import chatbot_hybrid_target as native
        import chatbot_hybrid_fixed_work_target as variant
        import chatbot_target_ssd_storage as storage
        assert (torch.__version__,transformers.__version__,np.__version__) == ('2.6.0+cu124','5.13.1','2.4.6')
        proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        def guard():
            v=b['limits']
            assert time.monotonic()-start <= v['seconds']-v['reserve_seconds'], 'worker deadline reserve'
            assert proc.memory_info().peak_wset <= v['OS_bytes'], 'OS cap'
            assert torch.cuda.max_memory_allocated() <= v['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= v['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'unexpected worker child'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= v['output_bytes'], 'output cap'
        def event(stage,**fields):
            print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True)
            guard()
        generated=storage.install(source_code)
        (a.directory/'generated_method.py.txt').write_text(generated,encoding='utf8')
        dither=False; original_aq=native.aq63
        def robust_aq(x):
            if not dither or not torch.is_grad_enabled(): return original_aq(x)
            scale=x.detach().abs().amax(-1,keepdim=True).clamp_min(1e-12)/63
            noise=torch.empty_like(x).uniform_(-b['normalized_AQ_dither'],b['normalized_AQ_dither'])
            return torch.round(x.detach()*(1/scale)+noise).clamp(-63,63),scale
        native.aq63=robust_aq
        class TrainTarget(variant.Target):
            def forward(self,ids,positions):
                x=self.embed(ids)
                for block in self.layers:
                    x=checkpoint(block,x,use_reentrant=False,preserve_rng_state=True) if self.training and torch.is_grad_enabled() else block(x)
                return self.head(self.final_norm(x)[:,positions])
        def same_bits(left,right):
            assert left.dtype == right.dtype == torch.float32 and left.shape == right.shape
            return torch.equal(left.detach().cpu().contiguous().view(torch.int32),right.detach().cpu().contiguous().view(torch.int32))
        phase='parent_adoption'
        old=torch.load(b['parent_checkpoint']['path'],map_location='cpu',weights_only=True)
        assert old['target_schema']=='COMPACT_FALCON_TERNARY_TARGET_V1' and old['candidate_schema']=='BROAD_24_TRANSFER_CANDIDATE_V1'
        assert old['completed_broad_updates']==24 and old['completed_recovery_updates']==286
        prior={k:v for k,v in old.items() if k not in ('model','optimizer','torch_CPU_rng','torch_CUDA_rng')}
        cpu_rng=old['torch_CPU_rng'].clone();cuda_rng=[v.clone() for v in old['torch_CUDA_rng']]
        config=old['config'];target=TrainTarget(config,'ternary').to('cuda')
        mapped=variant.mapped_model(old['model'])
        for i in range(12): mapped[f'layers.{i}.banks.common.down_scale'].fill_(b['initial_common_down_scale'])
        target.load_state_dict(mapped,strict=True)
        named=list(target.named_parameters());names=[n for n,_ in named];pars=[p for _,p in named]
        assert len(named)==283 and sum(p.numel() for p in pars)==259669760
        remapped,mapping=variant.mapped_optimizer(old['model'],old['optimizer'],names)
        optimizer=variant.make_optimizer(target,dict(optimizer=remapped,optimizer_parameter_names=names))
        assert len(optimizer.state)==211
        moment_bytes=0
        for name,p in named:
            assert p.dtype==torch.float32 and torch.isfinite(p).all().item()
            if '.banks.common.' in name:
                assert p not in optimizer.state
                continue
            original_name=variant.canonical_private_name(name)
            assert same_bits(p,old['model'][original_name]), name
        by_name=dict(named)
        for m in mapping:
            slot=optimizer.state[by_name[m['new_name']]];parent=old['optimizer']['state'][m['original_id']]
            assert int(slot['step'].item())==int(parent['step'].item())==318
            for label in ('exp_avg','exp_avg_sq'):
                assert same_bits(slot[label],parent[label]),(m['new_name'],label)
                assert torch.isfinite(slot[label]).all().item()
                moment_bytes+=slot[label].numel()*4
            assert torch.all(slot['exp_avg_sq']>=0).item()
        assert moment_bytes==2039461888
        for group in optimizer.param_groups:
            assert group['lr']==b['learning_rate'] and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
            assert group['weight_decay']==0 and group['foreach'] is False and not group['amsgrad'] and not group['maximize']
        for layer in target.layers:
            assert torch.count_nonzero(layer.banks.common.down).item()==0
            assert torch.all(layer.banks.common.down_scale==b['initial_common_down_scale']).item()
        write(a.directory/'parameter_mapping.json',dict(private=mapping,new_common=[n for n in names if '.banks.common.' in n]))
        del old,mapped,remapped,parent;gc.collect()
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        event('adopted',parameters=259669760,tensors=283,old_slots=211,new_slots=0,old_moment_bytes=moment_bytes)
        records={r['id']:r for r in json.loads(Path(b['corpus']).read_bytes())['records']}
        selected=[records[i] for i in b['case_ids']]
        teachers={}
        for rec in selected:
            bits=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4')
            values=(bits<<16).view('<f4').reshape(rec['logits']['shape'])
            teacher=torch.from_numpy(values.copy()).to('cuda')
            logp=torch.log_softmax(teacher,-1).detach()
            teachers[rec['id']]=(values,logp,logp.exp())
        collection=None;exposure={}
        def row_for(stage,identifier):
            return exposure.setdefault(stage,{}).setdefault(identifier,
                dict(private=[[0]*72 for _ in range(12)],common_positions=[0]*12,
                     mass_max_defect=[0.]*12,common_norm2=[0.]*12,common_mean_energy=[0.]*12))
        def route_observer(site):
            def observe(ids,mass):
                if collection is None:return
                stage,identifier=collection;row=row_for(stage,identifier)
                assert ids.shape==mass.shape==(len(records[identifier]['student_input_ids']),6)
                assert torch.isfinite(mass).all().item() and torch.all(mass>0).item()
                assert torch.all((ids>=0)&(ids<72)).item()
                assert torch.all(ids.sort(-1).values.diff(dim=-1)>0).item()
                defect=(mass.sum(-1)-1).abs().max().item();assert defect<=1e-6
                counts=torch.bincount(ids.flatten(),minlength=72).cpu().tolist()
                row['private'][site]=[x+y for x,y in zip(row['private'][site],counts,strict=True)]
                row['mass_max_defect'][site]=max(row['mass_max_defect'][site],defect)
            return observe
        def common_hook(site):
            def observe(module,args,output):
                if collection is None:return
                stage,identifier=collection;row=row_for(stage,identifier)
                assert list(output.shape)==[1,len(records[identifier]['student_input_ids']),512]
                assert torch.isfinite(output).all().item()
                if stage=='initial':assert torch.count_nonzero(output).item()==0
                row['common_positions'][site]+=output.shape[1]*2
                v=output.detach().double()
                row['common_norm2'][site]+=v.square().sum().item()
                row['common_mean_energy'][site]+=v.shape[1]*v.mean(1).square().sum().item()
            return observe
        handles=[]
        for site,layer in enumerate(target.layers):
            layer.banks.private.routing_observer=route_observer(site)
            handles.append(layer.banks.common.register_forward_hook(common_hook(site)))
        def logp64(values):
            shifted=values.astype(np.float64)-values.max(-1,keepdims=True)
            return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
        def metrics(values,rec):
            lp=logp64(teachers[rec['id']][0]);lq=logp64(values)
            per=(np.exp(lp)*(lp-lq)).sum(-1)
            assert np.isfinite(per).all() and per.min()>=-1e-10
            predicted=values.argmax(-1).tolist()
            return dict(KL=float(per.mean()),label_KL=per.tolist(),predicted_ids=predicted,
                disagreements=sum(x!=y for x,y in zip(predicted,rec['output_ids'],strict=True)))
        def observe(rec,stage,retain):
            nonlocal collection
            collection=(stage,rec['id'])
            try:
                logits=target(torch.tensor([rec['student_input_ids']],device='cuda'),
                              torch.tensor(rec['positions'],device='cuda')).squeeze(0)
            finally:collection=None
            assert list(logits.shape)==rec['logits']['shape'] and torch.isfinite(logits).all().item()
            _,lp,prob=teachers[rec['id']]
            loss=(prob*(lp-torch.log_softmax(logits,-1))).sum(-1).mean()
            assert torch.isfinite(loss).item()
            row=dict(id=rec['id'],domain=rec['domain'],split=rec['split'],labels=len(rec['output_ids']),
                     teacher_forcing_ids=len(rec['student_input_ids']),training_KL_F32=loss.item(),dither=dither)
            if retain:
                values=logits.detach().cpu().numpy();row.update(metrics(values,rec))
                path=a.directory/(stage+'.'+rec['id']+'.logits.f32')
                with path.open('xb') as f:
                    f.write(values.astype('<f4',copy=False).tobytes());f.flush();os.fsync(f.fileno())
                row['logits']=dict(path=str(path.resolve()),shape=list(values.shape),bytes=path.stat().st_size,sha256=sha(path))
            return loss,row
        baseline=[]
        for rec in selected:
            entry=b['baseline'][rec['id']]['logits']
            values=np.fromfile(entry['path'],dtype='<f4').reshape(entry['shape'])
            row=dict(id=rec['id'],labels=len(rec['output_ids']),logits=entry,**metrics(values,rec))
            row['parent_report_KL_F32']=b['baseline'][rec['id']]['KL']
            assert abs(row['KL']-row['parent_report_KL_F32'])<=1e-4*max(1.,row['KL'])
            baseline.append(row)
        write(a.directory/'baseline_reused.json',dict(cases=baseline,new_baseline_forwards=0))
        def evaluate(stage):
            nonlocal phase,dither
            phase=stage;dither=False;target.eval();rows=[]
            with torch.no_grad():
                for rec in selected:
                    t0=time.monotonic();_,row=observe(rec,stage,True);torch.cuda.synchronize()
                    row['seconds']=time.monotonic()-t0;rows.append(row)
                    write(a.directory/(stage+'.'+rec['id']+'.json'),row)
                    event(stage,id=rec['id'],KL=row['KL'],disagreements=row['disagreements'])
            return rows
        initial=evaluate('initial')
        for rec,row,old_row in zip(selected,initial,baseline,strict=True):
            new=np.fromfile(row['logits']['path'],dtype='<f4').reshape(row['logits']['shape'])
            parent=np.fromfile(old_row['logits']['path'],dtype='<f4').reshape(old_row['logits']['shape'])
            row['initial_to_parent_KL_ratio']=row['KL']/old_row['KL']
            row['changed_parent_greedy_ids']=int(np.count_nonzero(new.argmax(-1)!=parent.argmax(-1)))
            row['changed_parent_F32_coordinates']=int(np.count_nonzero(new.view('<u4')!=parent.view('<u4')))
        write(a.directory/'initial.json',dict(cases=initial,baseline=baseline,zero_common=True))
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        def common_trits():
            result=[]
            with torch.no_grad():
                for i,layer in enumerate(target.layers):
                    c=layer.banks.common
                    code=torch.round(c.down/c.down_scale[...,None]).clamp(-1,1)
                    result.append(dict(layer=i,down_trits_nonzero=int(torch.count_nonzero(code).item()),
                        down_master_nonzero=int(torch.count_nonzero(c.down).item()),
                        down_scale_min=c.down_scale.min().item(),down_scale_max=c.down_scale.max().item()))
            return result
        def cpu_tree(value):
            if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
            if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu_tree(v) for v in value]
            if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
            return value
        def save_state(label):
            packet=dict(model=cpu_tree(target.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
                optimizer_parameter_names=names,config=config,target_schema=variant.SCHEMA,
                candidate_schema='FIXED_WORK_FEASIBILITY_CANDIDATE_V1',common_precision='ternary',
                common_count=2,private_k=6,private_n=72,completed_new_updates=completed,
                private_optimizer_step=318+completed,common_optimizer_step=completed,
                parent_checkpoint=b['parent_checkpoint'],parent_metadata=prior,new_updates=list(updates),
                torch_CPU_rng=torch.get_rng_state(),torch_CUDA_rng=torch.cuda.get_rng_state_all(),
                freeze=a.freeze,binding_sha256=a.binding_sha,training_contract=dict(
                    normalized_AQ_dither=.025,learning_rate=5e-5,clip_norm=1.,
                    initial_common_down_scale=b['initial_common_down_scale'],
                    storage_helper_sha256=sha(B/'chatbot_target_ssd_storage.py')))
            assert len(packet['model'])==283 and len(packet['optimizer']['state'])==(283 if completed else 211)
            for value in packet['model'].values():assert value.dtype==torch.float32 and np.isfinite(value.numpy()).all()
            for pid,slot in packet['optimizer']['state'].items():
                expected=completed if '.banks.common.' in names[pid] else 318+completed
                assert int(slot['step'].item())==expected
                for key in ('exp_avg','exp_avg_sq'):assert np.isfinite(slot[key].numpy()).all()
                assert (slot['exp_avg_sq'].numpy()>=0).all()
            temporary=a.directory/(label+'.next.pt');path=a.directory/(label+'.pt')
            with temporary.open('xb') as f:torch.save(packet,f);f.flush();os.fsync(f.fileno())
            os.replace(temporary,path);del packet;gc.collect()
            return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path))
        snapshot=save_state
        for step in range(1,3):
            phase='update_'+str(step);guard();t0=time.monotonic();target.train();dither=True
            optimizer.zero_grad(set_to_none=True)
            loss,row=observe(records[b['update_case']],phase,False);loss.backward()
            assert all(p.grad is not None and p.grad.shape==p.shape and torch.isfinite(p.grad).all().item() for p in pars)
            groups=[]
            for i,layer in enumerate(target.layers):
                fields={}
                for label,values in [('core',layer.core.parameters()),('private',layer.banks.private.parameters()),
                                     ('common',layer.banks.common.parameters()),('common_down',[layer.banks.common.down])]:
                    squared=sum(p.grad.double().square().sum().item() for p in values)
                    assert 0<squared<float('inf'),(i,label,'gradient path')
                    fields[label]=squared**.5
                groups.append(dict(layer=i,**fields))
            norm=torch.nn.utils.clip_grad_norm_(pars,b['clip_norm'],error_if_nonfinite=True).item()
            optimizer.step()
            with torch.no_grad():
                for layer in target.layers:
                    for bank in (layer.banks.private,layer.banks.common):
                        for scale in (bank.gate_scale,bank.up_scale,bank.down_scale):scale.clamp_(min=1e-8)
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();dither=False
            assert len(optimizer.state)==283
            for name,p in named:
                slot=optimizer.state[p];expected=step if '.banks.common.' in name else 318+step
                assert int(slot['step'].item())==expected
                assert torch.isfinite(p).all().item()
                assert torch.isfinite(slot['exp_avg']).all().item() and torch.isfinite(slot['exp_avg_sq']).all().item()
                assert torch.all(slot['exp_avg_sq']>=0).item()
            completed=step
            row.update(step=step,private_optimizer_step=318+step,common_optimizer_step=step,
                       gradient_norm_before_clip=norm,layer_gradient_norms=groups,
                       seconds_before_snapshot=time.monotonic()-t0,common=common_trits())
            updates.append(row);write(a.directory/(f'update_{step:02d}.json'),row)
            event('update',step=step,KL=row['training_KL_F32'],interval_seconds=row['seconds_before_snapshot'],
                  common_down_nonzero=sum(v['down_trits_nonzero'] for v in row['common']))
            del loss
        phase='snapshot';t0=time.monotonic();durable=save_state('candidate')
        snapshot_seconds=time.monotonic()-t0;event('durable',updates=completed,checkpoint=durable,interval_seconds=snapshot_seconds)
        after=evaluate('after')
        for stage,cases in exposure.items():
            for identifier,row in cases.items():
                positions=len(records[identifier]['student_input_ids'])
                assert all(sum(v)==positions*6 for v in row['private']), (stage,identifier,'private exposure')
                assert row['common_positions']==[positions*2]*12,(stage,identifier,'common exposure')
        for layer in target.layers:layer.banks.private.routing_observer=None
        for h in handles:h.remove()
        write(a.directory/'exposure.json',exposure)
        assert completed==2 and durable is not None
        native_common_progress=all(v['down_trits_nonzero']>0 for v in updates[-1]['common']) and all(
            v>0 for row in exposure['after'].values() for v in row['common_norm2'])
        guard();phase='result'
        result=dict(schema='FIXED_WORK_FEASIBILITY_RESULT_V1',decision='FIXED_WORK_FEASIBILITY_PASS',
            freeze=a.freeze,binding_sha256=a.binding_sha,process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
            cases=2,new_updates=2,parent_private_step=318,final_private_step=320,final_common_step=2,
            parameters=259669760,tensors=283,parent_adoption_bit_exact=True,
            parent_Adam_slots_bit_exact=True,initial_common_output_zero=True,actual_active_functions=8,
            actual_private_k=6,actual_common_count=2,actual_exposure_verified=True,
            common_ternary_progress=native_common_progress,checkpoint=durable,snapshot_seconds=snapshot_seconds,
            baseline=baseline,initial=initial,after=after,updates=updates,exposure=exposure,
            source_calls=0,baseline_new_forwards=0,reserved_queries=0,native_runs=0,
            quality_admission=False,native_admission=False,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            scope='Actual mapped parent, two new longest FIT updates and two FIT initial/final whole-output cases; feasible learning is not chatbot preservation, fresh generation, useful n, native parity or accepted50.')
        write(a.out,result);event('complete',decision=result['decision'],common_ternary_progress=native_common_progress)
    except BaseException as error:
        fault=dict(fault=repr(error),phase=phase,completed_new_updates=completed,durable_checkpoint=durable,
                   elapsed_seconds=time.monotonic()-start)
        if torch is not None:
            fault.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',fault)
        if durable is None and snapshot is not None:
            try:write(a.directory/'failure_state.json',dict(checkpoint=snapshot('failure_state'),completed_new_updates=completed))
            except BaseException as secondary:write(a.directory/'failure_state_fault.json',dict(fault=repr(secondary)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
