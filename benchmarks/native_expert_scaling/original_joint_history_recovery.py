"""Matched real-Adam actual27 forks: final KL versus KL+six residual losses.

Frozen original deployment operators, same one-pass24 FIT histories. Stored
states are durable before evaluation; partial faults never imply completed dose.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from original_packed_capacity import SITE,sha,write,raw,extent
from original_falcon_whole_recovery import memory_reader,check_inputs
sys.path.insert(0,str(SITE))


def bind(a):
    import numpy as np
    import numpy._core._multiarray_umath as ext
    parent=DOC/'original_delta_seed_finish_result_20261009.json';pr=json.loads(parent.read_bytes())
    assert pr['final_counter']==pr['durable_counter']==27
    baseline=DOC/'original_engine_baseline_result_20261009.json';br=json.loads(baseline.read_bytes())
    bt=baseline.with_suffix('.terminal.json');terminal=json.loads(bt.read_bytes())
    assert terminal['exit_code']==0 and terminal['result_sha256']==sha(baseline) and br['packed']==pr['packed']
    baseline_binding=DOC/'original_engine_baseline_binding_20261009.json';bb=json.loads(baseline_binding.read_bytes())
    audit=DOC/'original_engine_baseline_stored_adjudication_20261009.json';ar=json.loads(audit.read_bytes())
    assert ar['result']['sha256']==sha(baseline) and ar['all_prefix_heads_routes_bit_exact']
    boundaries=DOC/'original_history_boundaries_finish_result_20261009.json';hr=json.loads(boundaries.read_bytes())
    ht=boundaries.with_suffix('.terminal.json');hterm=json.loads(ht.read_bytes())
    assert hterm['exit_code']==0 and hterm['result_sha256']==sha(boundaries) and hr['cases']==48
    targets={r['id']:[x['z'] for x in r['boundaries'] if x['boundary']>0] for r in hr['records']}
    assert set(targets)=={r['id'] for r in bb['records']} and all(len(x)==6 for x in targets.values())
    first='broad_fit_smol_magpie_ultra_022'
    FIT=[first]+sorted(r['id'] for r in bb['records'] if r['split']=='FIT' and r['id']!=first)
    assert len(FIT)==24 and len(set(FIT))==24
    ns=ROOT/'results/native_expert_scaling/original_delta_seed_finish_20261009'
    files=[Path(__file__),B/'original_joint_history_learner.py',B/'original_joint_native_assessment.py',
        B/'original_engine_baseline.py',B/'original_engine_chat.py',B/'chatbot_hybrid_engine_chat.py',
        B/'original_wide_learner.py',B/'original_falcon_learner.py',B/'original_tensor_learner.py',
        B/'original_packed_capacity.py',B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability.py',
        B/'chatbot_falcon_usability_launch.py',DOC/'ORIGINAL_JOINT_HISTORY_RECOVERY_PROTOCOL_20261009.md',
        parent,parent.with_suffix('.terminal.json'),baseline,bt,baseline_binding,audit,boundaries,ht,
        Path(pr['checkpoint']['path']),Path(pr['packed']['path']),Path(br['executable']['path']),
        ns/'GPU_after.f32',ns/'GPU_after.routes',Path(sys.executable),Path(ext.__file__),SITE/'numpy/__init__.py',
        SITE/'psutil/__init__.py',SITE/'tokenizers/__init__.py',SITE/'tokenizers/tokenizers.pyd',SITE/'jinja2/__init__.py']
    files += [SITE/'torch'/p for p in ('__init__.py','_C.cp312-win_amd64.pyd','lib/torch_cpu.dll','lib/torch_cuda.dll',
        'optim/adam.py','optim/optimizer.py','utils/checkpoint.py','cuda/__init__.py')]
    files += [Path(r['logits']['path']) for r in bb['records']]+[Path(x['path']) for xs in targets.values() for x in xs]
    files += [Path(bb['source'])/n for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json')]
    files += [ROOT/n for n in ('benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    assert shutil.disk_usage(ROOT).free>=120<<30
    binding=dict(schema='ORIGINAL_JOINT_HISTORY_RECOVERY_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),source_state=pr['checkpoint'],source_packed=pr['packed'],native=br['executable'],
        before_native=br['native_aggregate']['DEV'],before_tasks=br['task_correct'],baseline=extent(baseline),
        records=bb['records'],cases=bb['cases'],source=bb['source'],special_ids=bb['special_ids'],followup=bb['followup'],
        boundary_targets=targets,source_boundary_result=extent(boundaries),parent_source_resource_gate=False,
        before_GPU_heads=extent(ns/'GPU_after.f32'),before_GPU_routes=extent(ns/'GPU_after.routes'),
        FIT_order=FIT,arms=[dict(name='A',auxiliary_weight=0.),dict(name='B',auxiliary_weight=1.)],
        start_counter=27,new_updates_per_arm=24,milestones=[6,12,24],
        optimizer=dict(lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=0.,foreach=False,clip_global_L2=1.),
        plateau=dict(regression_KL_multiple=1.05,regression_disagreement_add=.03,
            at12_KL_best_relative_to27=.98,at12_KL_relative_to6=.99),
        gates=dict(native_DEV_KL=1.,native_DEV_disagreement=.20,domain_KL=2.,domain_disagreement=.35,
            relative_DEV_KL=.90,relative_DEV_disagreement=.95,tasks_min=12,category_min=2,
            B_relative_to_A_KL=.95,B_relative_to_A_centered=.90,B_disagreement_add=.01,
            B_domain_KL_multiple=1.05,B_domain_disagreement_add=.03,B_behavioral_min=2,
            initial_GPU_bits_exact=True,initial_92_master_Adam_RNG_exact=True,initial_export_exact=True,
            mass_defect=1e-6,initial_B_aux_core_gradient_delta_min=1e-12,initial_B_aux_core_gradient_relative_min=1e-5),
        limits=dict(seconds=7200,reserve_seconds=600,OS_bytes=32<<30,GPU_allocated_bytes=10<<30,
            GPU_reserved_bytes=11<<30,output_bytes=96<<30,child_seconds=100,stream_seconds=300,log_bytes=8<<20),
        inputs=[extent(p) for p in dict.fromkeys(files)],
        scope='Matched actual27 real master/Adam/RNG forks,one24FIT pass each,max48 NEW updates. Native original20 bodies unchanged;'
            'same packed-only C binary reused. Source teacher boundaries have inherited resource/identity gaps,not retroqualified. '
            'Three full DEV auxiliary passes(initial27,Afinal,Bfinal);native24DEV at6/12/24. '
            'No source inference/RESERVED/T4/useful-n/DRAM/accepted-speed admission;GPU remains numerical surrogate.')
    write(a.out,binding);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(binding['inputs']))),flush=True)


def worker(a):
    import numpy as np
    import psutil
    import torch
    from original_joint_history_learner import JointHistoryLearner
    from original_wide_learner import export,quant_weight,tensor,source
    from original_joint_native_assessment import native_dev,tasks
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());a.directory.mkdir(exist_ok=False)
    assert (torch.__version__,np.__version__,psutil.__version__)==('2.6.0+cu124','2.4.6','7.2.2')
    assert (tensor.DN,tensor.DT,source.DN,source.DT)==(1024,48,1024,48)
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();reader=memory_reader();child_peak=0;children=[];arms=[];stage='startup';arm='none'
    model=None;optimizer=None;names=None;counter=27;durable=27;updates=[];last_snapshot=None;last_progress=0.;before_aux=None
    def observe_peak(peak):
        nonlocal child_peak
        child_peak=max(child_peak,peak)
    def guard(reserve=True):
        lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-(lim['reserve_seconds'] if reserve else 0),'reserve/deadline'
        assert proc.memory_info().peak_wset+child_peak<=lim['OS_bytes'],'worker/direct child OS cap'
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes'],'output cap'
    def event(**values):
        guard();print(json.dumps(dict(stage=stage,arm=arm,counter=counter,durable=durable,seconds=time.monotonic()-start,**values)),flush=True)
    def progress(mode,completed,total,site):
        nonlocal last_progress
        guard()
        if time.monotonic()-last_progress>=15:
            torch.cuda.synchronize();event(operation=mode,completed=completed,total=total,site=site);last_progress=time.monotonic()
    def run(command,name):
        began=time.monotonic();peak=0
        with (a.directory/(name+'.log')).open('xb') as f:
            child=subprocess.Popen([str(x) for x in command],stdout=f,stderr=subprocess.STDOUT,creationflags=8)
            created=psutil.Process(child.pid).create_time()
            try:
                while child.poll() is None:
                    peak=max(peak,reader(child));observe_peak(peak);guard();assert time.monotonic()-began<=b['limits']['child_seconds'],'child deadline';time.sleep(.05)
                peak=max(peak,reader(child));observe_peak(peak)
            except BaseException:
                if child.poll() is None:child.kill();child.wait()
                peak=max(peak,reader(child));observe_peak(peak);raise
            finally:
                receipt=dict(command=[str(x) for x in command],pid=child.pid,creation_time=created,exit_code=child.returncode,
                    held_OS_peak=peak,seconds=time.monotonic()-began)
                children.append(receipt);write(a.directory/(name+'.receipt.json'),receipt)
        assert child.returncode==0,(name,child.returncode);return receipt
    def exact(x,y):return torch.equal(x.detach().cpu().contiguous().view(torch.int32),y.detach().cpu().contiguous().view(torch.int32))
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    def snapshot(name,partial=False):
        nonlocal durable,last_snapshot
        began=time.monotonic();packet=dict(schema='ORIGINAL_JOINT_HISTORY_STATE_V1',arm=arm,model=model.state_dict(),optimizer=optimizer.state_dict(),
            updates=counter,new_updates=counter-27,optimizer_partial_possible=partial,source_state=b['source_state'],source_packed=b['source_packed'],
            inherited_lineage=ancestry,binding_sha256=a.binding_sha,completed_new_FIT_ids=[x['id'] for x in updates],
            auxiliary_weight=next(x['auxiliary_weight'] for x in b['arms'] if x['name']==arm),
            geometry=dict(D=256,N=96,DN=1024,DT=48,L=6,E=1152,V=65537,HID_E=128,K=8),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all())
        path=a.directory/name
        with path.open('xb') as f:torch.save(cpu(packet),f);f.flush();os.fsync(f.fileno())
        del packet;gc.collect();last_snapshot=extent(path)
        if not partial:durable=counter
        guard(False);print(json.dumps(dict(stage=stage,arm=arm,counter=counter,durable=durable,snapshot=last_snapshot,save_seconds=time.monotonic()-began)),flush=True)
        return last_snapshot
    records={r['id']:r for r in b['records']}
    def targets(rec):
        result=[]
        for item in b['boundary_targets'][rec['id']]:
            values=np.fromfile(item['path'],dtype='<f4').reshape(1,len(rec['student_input_ids']),256)
            assert np.isfinite(values).all();result.append(torch.from_numpy(values).cuda())
        return result
    def summaries(values):
        result=[]
        for row in values:
            item={k:(float(v.detach()) if isinstance(v,torch.Tensor) else v) for k,v in row.items()}
            assert all(math.isfinite(item[k]) for k in ('total','mean','centered','centered_relative'))
            assert abs(item['total']-item['mean']-item['centered'])<=1e-4*max(item['total'],1e-24)
            result.append(item)
        return result
    def auxiliary_dev(prefix):
        began=time.monotonic();values=[];identities={n:(id(p),p._version) for n,p in names}
        with torch.no_grad():
            for rec in b['records']:
                if rec['split']!='DEV':continue
                ids=torch.tensor([rec['student_input_ids']],device='cuda');target=targets(rec)
                logits,aux,terms=model(ids,torch.empty(0,dtype=torch.long,device='cuda'),target);torch.cuda.synchronize()
                assert logits.shape==(1,0,65537) and torch.isfinite(aux)
                row=dict(id=rec['id'],domain=rec['domain'],history=len(rec['student_input_ids']),auxiliary=float(aux),sites=summaries(terms))
                values.append(row);write(a.directory/(prefix+'.'+rec['id']+'.aux.json'),row)
                del ids,target,logits,aux,terms;gc.collect();torch.cuda.empty_cache();event(operation='auxiliary_DEV',prefix=prefix,id=rec['id'],cases=len(values))
        assert {n:(id(p),p._version) for n,p in names}==identities
        def mean(rows):return dict(cases=len(rows),total=float(np.mean([np.mean([s['total'] for s in r['sites']]) for r in rows])),
            mean=float(np.mean([np.mean([s['mean'] for s in r['sites']]) for r in rows])),
            centered=float(np.mean([np.mean([s['centered'] for s in r['sites']]) for r in rows])),
            centered_relative=float(np.mean([np.mean([s['centered_relative'] for s in r['sites']]) for r in rows])))
        result=dict(records=values,aggregate=dict(**mean(values),domains={d:mean([r for r in values if r['domain']==d]) for d in sorted({r['domain'] for r in values})}),
            parameter_ids_versions_unchanged=True,heads_computed=0,seconds=time.monotonic()-began)
        write(a.directory/(prefix+'.auxiliary_DEV.json'),result);return result
    def witness():
        values=[]
        with torch.no_grad():
            for e in (0,31,32,1023,1024,1151):
                for layer in model.layers:
                    for organ,length in (('gate',256),('up',256),('down',128)):
                        q,_=quant_weight(getattr(layer.bank,organ)[e]);operand=np.arange(length,dtype='i8')
                        operand=operand%127-63 if organ!='down' else (operand*7)%127-63;values.append(q.numpy().astype('i8')@operand)
        result=np.concatenate(values).astype('<i4').tobytes();assert len(result)==18432*4;return result
    def plateau(milestones):
        cfg=b['plateau'];now=milestones[-1]['native']['aggregate'];base=b['before_native']
        if now['case_KL']>cfg['regression_KL_multiple']*base['case_KL'] or now['case_disagreement']>base['case_disagreement']+cfg['regression_disagreement_add']:return 'native_DEV_regression'
        if milestones[-1]['new_updates']==12:
            best=min(x['native']['aggregate']['case_KL'] for x in milestones)
            if best>cfg['at12_KL_best_relative_to27']*base['case_KL'] and now['case_KL']>cfg['at12_KL_relative_to6']*milestones[0]['native']['aggregate']['case_KL']:return 'native_DEV_plateau_at12'
        return None
    try:
        exe=a.directory/'original_engine_chat.exe';raw(exe,Path(b['native']['path']).read_bytes());assert sha(exe)==b['native']['sha256']
        for recipe in b['arms']:
            arm=recipe['name'];counter=durable=27;updates=[];last_snapshot=None;milestones=[];stop=None
            stage='restore_actual27';state=torch.load(b['source_state']['path'],map_location='cpu',weights_only=True,mmap=True)
            assert state['schema']=='ORIGINAL_DELTA_SEEDED_STATE_V1' and state['updates']==27 and not state['optimizer_partial_possible']
            ancestry={k:state[k] for k in ('source_state','ancestry','seed_ledger','binding_sha256','completed_new_FIT_ids')}
            model=JointHistoryLearner(device='cuda');model.load_state_dict(state['model']);names=list(model.named_parameters())
            assert len(names)==92 and sum(p.numel() for _,p in names)==721008128 and list(state['model'])==[n for n,_ in names]
            cfg=b['optimizer'];optimizer=torch.optim.Adam([p for _,p in names],lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=0.,foreach=False)
            optimizer.load_state_dict(state['optimizer']);indices=state['optimizer']['param_groups'][0]['params']
            for (name,p),index in zip(names,indices):
                st=optimizer.state[p];old=state['optimizer']['state'][index]
                assert exact(p,state['model'][name]) and int(st['step'])==27
                assert exact(st['step'],old['step']) and exact(st['exp_avg'],old['exp_avg']) and exact(st['exp_avg_sq'],old['exp_avg_sq'])
                assert torch.isfinite(p).all() and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
            group=optimizer.param_groups[0];assert group['lr']==5e-5 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0 and group['foreach']==False
            torch.set_rng_state(state['CPU_rng']);torch.cuda.set_rng_state_all(state['CUDA_rng'])
            assert torch.equal(torch.get_rng_state(),state['CPU_rng']) and all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),state['CUDA_rng']))
            del state,old,st,p;gc.collect();torch.cuda.empty_cache();model.use_checkpoint=True
            for site,layer in enumerate(model.layers):layer.bank.progress=lambda mode,n,total,site=site:progress(mode,n,total,site)
            stage='initial_export';initial=a.directory/(arm+'.initial27.packed');export(model,initial)
            assert extent(initial)['sha256']==b['source_packed']['sha256'];event(initial_92_states_RNG_export_exact=True)
            if before_aux is None:
                stage='initial27_auxiliary_DEV';before_aux=auxiliary_dev('initial27')
            for index,identifier in enumerate(b['FIT_order']):
                rec=records[identifier];stage='whole_forward';began=time.monotonic();optimizer.zero_grad(set_to_none=True)
                ids=torch.tensor([rec['student_input_ids']],device='cuda');pos=torch.tensor(rec['positions'],device='cuda');target=targets(rec)
                teacher_bits=np.fromfile(rec['logits']['path'],dtype='<u2').astype('<u4')
                teacher=torch.from_numpy((teacher_bits<<16).view('<f4').reshape(len(rec['positions']),65537)).cuda();del teacher_bits
                logits,aux,terms=model(ids,None if index==0 else pos,target);torch.cuda.synchronize()
                assert torch.isfinite(logits).all() and torch.isfinite(aux);aux_summary=summaries(terms)
                if index==0:
                    observed=logits[0].detach().cpu().numpy().astype('<f4');raw(a.directory/(arm+'.first_GPU_heads.f32'),observed.tobytes())
                    assert sha(a.directory/(arm+'.first_GPU_heads.f32'))==b['before_GPU_heads']['sha256'],'adapter initial full-head bits'
                    routes=np.zeros((len(rec['student_input_ids']),6),dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))]))
                    for site,layer in enumerate(model.layers):routes['ids'][:,site]=layer.bank.last_routes[0].numpy();routes['mass'][:,site]=layer.bank.last_routes[1].numpy()
                    raw(a.directory/(arm+'.first_GPU_routes'),routes.tobytes());assert sha(a.directory/(arm+'.first_GPU_routes'))==b['before_GPU_routes']['sha256'],'initial GPU routes'
                    del observed,routes;selected=logits[0,pos]
                else:selected=logits[0]
                logq=teacher.log_softmax(-1);logp=selected.log_softmax(-1);kl=(logq.exp()*(logq-logp)).sum(-1).mean()
                loss=kl+recipe['auxiliary_weight']*aux;assert torch.isfinite(loss)
                exposure=[dict(site=site,union=int(torch.unique(layer.bank.last_routes[0]).numel()),selected_pairs=layer.bank.last_routes[0].numel()) for site,layer in enumerate(model.layers)]
                forward=time.monotonic()-began;event(operation='forward_complete',new_index=index+1,id=identifier,KL=float(kl.detach()),aux=float(aux.detach()),loss=float(loss.detach()))
                stage='whole_backward';loss.backward();torch.cuda.synchronize();backward=time.monotonic()-began-forward
                stage='gradient_inspection';squared=0.;core_delta=[]
                for name,p in names:
                    assert p.grad is not None and torch.isfinite(p.grad).all(),('gradient',name)
                    squared+=float(p.grad.detach().double().square().sum())
                if index==0:
                    for name in ('layers.0.organs.in_proj','layers.5.organs.qkv'):
                        p=dict(names)[name];values=p.grad.detach().cpu().numpy().astype('<f4');filename=name.replace('.','_')+'.gradient.f32'
                        raw(a.directory/(arm+'.'+filename),values.tobytes())
                        if arm=='B':
                            reference=np.fromfile(a.directory/('A.'+filename),dtype='<f4').reshape(values.shape).astype('f8');current=values.astype('f8');delta=current-reference
                            norm=float(np.linalg.norm(delta));assert norm>b['gates']['initial_B_aux_core_gradient_delta_min']
                            assert norm>=b['gates']['initial_B_aux_core_gradient_relative_min']*max(float(np.linalg.norm(reference)),float(np.linalg.norm(current)),1e-24),'aux gradient distinguishable from rounding'
                            core_delta.append(dict(name=name,auxiliary_gradient_difference_L2=norm,KL_gradient_L2=float(np.linalg.norm(reference)),total_gradient_L2=float(np.linalg.norm(current)),
                                dot_KL_auxiliary=float(np.sum(reference*delta)),scope='Matched identical first27 FIT trajectory/loss;Btotal minus savedA KL gradient,includes F32 accumulation roundoff.'))
                norm=math.sqrt(squared);assert math.isfinite(norm) and norm>0;clip=min(1.,1./(norm+1e-6))
                with torch.no_grad():
                    for _,p in names:p.grad.mul_(clip)
                stage='optimizer_step';guard();opt_begin=time.monotonic();optimizer.step();counter+=1;torch.cuda.synchronize();stage='optimizer_finite_inspection'
                for _,p in names:
                    st=optimizer.state[p];assert int(st['step'])==counter and torch.isfinite(p).all() and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
                row=dict(id=identifier,counter=counter,new_index=index+1,history=len(rec['student_input_ids']),labels=len(rec['positions']),
                    KL_before_surrogate=float(kl.detach()),auxiliary=float(aux.detach()),loss=float(loss.detach()),auxiliary_weight=recipe['auxiliary_weight'],sites=aux_summary,
                    exposure=exposure,gradient_norm=norm,clip_coefficient=clip,initial_auxiliary_core_gradient_deltas=core_delta,
                    forward_seconds=forward,backward_seconds=backward,optimizer_and_finite_seconds=time.monotonic()-opt_begin)
                updates.append(row);write(a.directory/(arm+f'.update{counter:03d}.json'),row)
                optimizer.zero_grad(set_to_none=True);del ids,pos,target,teacher,logits,selected,aux,terms,logq,logp,loss,kl;gc.collect();torch.cuda.empty_cache()
                stage='update_complete';event(id=identifier,new_index=index+1,gradient_norm=norm)
                if index+1 in b['milestones']:
                    prefix=arm+f'.actual{counter:03d}';stage='durable_snapshot';snapshot(prefix+'.pt')
                    stage='updated_export';packed=a.directory/(prefix+'.packed');export(model,packed);expected=witness();raw(a.directory/(prefix+'.expected.witness'),expected)
                    stage='native_DEV_milestone';native=native_dev(b,a.directory,exe,packed,expected,prefix,run,event)
                    milestone=dict(new_updates=index+1,counter=counter,checkpoint=last_snapshot,packed=extent(packed),native=native)
                    milestones.append(milestone);write(a.directory/(prefix+'.milestone.json'),milestone);stop=plateau(milestones)
                    if stop:break
            final=milestones[-1];stage='final_auxiliary_DEV';aux_result=auxiliary_dev(arm+'.final')
            stage='final_native_tasks';task_result=tasks(b,a.directory,exe,final['packed']['path'],arm+'.final',reader,observe_peak,guard,event);children.append(task_result['receipt'])
            dev=final['native']['aggregate'];base=b['before_native'];g=b['gates']
            quality=dict(DEV_KL=dev['case_KL']<=g['native_DEV_KL'],DEV_disagreement=dev['case_disagreement']<=g['native_DEV_disagreement'],
                domain_KL=all(x['case_KL']<=g['domain_KL'] for x in dev['domains'].values()),domain_disagreement=all(x['case_disagreement']<=g['domain_disagreement'] for x in dev['domains'].values()),
                relative_DEV_KL=dev['case_KL']<=g['relative_DEV_KL']*base['case_KL'],relative_DEV_disagreement=dev['case_disagreement']<=g['relative_DEV_disagreement']*base['case_disagreement'])
            record=dict(name=arm,auxiliary_weight=recipe['auxiliary_weight'],initial_counter=27,final_counter=counter,durable_counter=durable,
                new_updates=len(updates),complete_onepass=len(updates)==24,stop_reason=stop,updates=updates,milestones=milestones,
                final_checkpoint=last_snapshot,final_packed=final['packed'],auxiliary_DEV=aux_result,tasks=task_result,quality_gates=quality)
            arms.append(record);write(a.directory/(arm+'.arm_result.json'),record);stage='arm_complete';event(new_updates=len(updates),task_correct=task_result['correct'],DEV_KL=dev['case_KL'],stop=stop)
            optimizer.zero_grad(set_to_none=True);del p,st,model,optimizer,names;model=optimizer=names=None;gc.collect();torch.cuda.empty_cache()
        A,Barm=arms;g=b['gates'];aDEV=A['milestones'][-1]['native']['aggregate'];bDEV=Barm['milestones'][-1]['native']['aggregate']
        preference=dict(both_complete24=all(x['complete_onepass'] for x in arms),native_KL=bDEV['case_KL']<=g['B_relative_to_A_KL']*aDEV['case_KL'],
            centered_auxiliary=Barm['auxiliary_DEV']['aggregate']['centered_relative']<=g['B_relative_to_A_centered']*A['auxiliary_DEV']['aggregate']['centered_relative'],
            disagreement=bDEV['case_disagreement']<=aDEV['case_disagreement']+g['B_disagreement_add'],
            domains=all(bDEV['domains'][d]['case_KL']<=g['B_domain_KL_multiple']*x['case_KL'] and bDEV['domains'][d]['case_disagreement']<=x['case_disagreement']+g['B_domain_disagreement_add'] for d,x in aDEV['domains'].items()),
            behavioral_support=Barm['tasks']['correct']>=max(g['B_behavioral_min'],A['tasks']['correct']+2))
        stage='complete';guard()
        result=dict(schema='ORIGINAL_JOINT_HISTORY_RECOVERY_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,arms=arms,before_auxiliary_DEV=before_aux,
            B_preference_gates=preference,B_preferred=all(preference.values()),decision='JOINT_HISTORY_B_SUPPORTED' if all(preference.values()) else 'JOINT_HISTORY_B_NOT_SUPPORTED',
            new_optimizer_updates=sum(x['new_updates'] for x in arms),source_calls=0,reserved_queries=0,GPU_source_calls=0,
            initial_92_states_RNG_exports_GPU_heads_routes_exact=True,quality_admission=False,speed_admission=False,useful_large_n_admission=False,
            historical_GPU_native_numerical_gate='FAIL retained',physical_DRAM_bytes=None,inherited_source_capture_resource_gate=False,
            children=children,max_direct_child_OS_peak=child_peak,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-start,scope=b['scope'])
        write(a.out,result);event(decision=result['decision'],new_optimizer_updates=result['new_optimizer_updates'])
    except BaseException as error:
        fault=dict(stage=stage,arm=arm,error=repr(error),counter=counter,durable=durable,last_snapshot=last_snapshot,completed_updates=updates,completed_arms=arms,
            optimizer_partial_possible=stage=='optimizer_step',children=children,elapsed_seconds=time.monotonic()-start,worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            max_direct_child_OS_peak=child_peak,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_fault.json',fault)
        if model is not None and optimizer is not None and (counter!=durable or stage=='optimizer_step'):
            try:optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();snapshot(arm+'.fault_actual_state.pt',stage=='optimizer_step')
            except BaseException as save_error:write(a.directory/'fault_snapshot_failure.json',dict(error=repr(save_error),counter=counter,last_snapshot=last_snapshot))
        raise


def launch(a):
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());assert b['schema']=='ORIGINAL_JOINT_HISTORY_RECOVERY_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','native','original_engine','clang','lld','packed_original')),('overlap',p.pid,name)
    log=a.out.with_suffix('.worker.log');terminal=a.out.with_suffix('.terminal.json');failure=a.out.with_suffix('.launcher_failure.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out,log,terminal,failure))
    worker_process=None;peak=0;reader=memory_reader();seen={};record=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_pid=proc.pid)
    def guard():
        nonlocal peak
        peak=max(peak,reader(worker_process));assert time.monotonic()-start<=b['limits']['seconds'],'family deadline'
        assert peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family OS cap';assert log.stat().st_size<=b['limits']['log_bytes'],'log cap'
    try:
        check_inputs(b);env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,
            '--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())];record['command']=command
        with log.open('xb') as f:
            worker_process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=8)
            record.update(worker_pid=worker_process.pid,worker_creation_time=psutil.Process(worker_process.pid).create_time());offset=0
            while worker_process.poll() is None:
                guard()
                try:
                    for child in psutil.Process(worker_process.pid).children(recursive=True):
                        try:name=child.name().lower();created=child.create_time();exe=Path(child.exe().removeprefix('\\\\?\\')).resolve()
                        except psutil.NoSuchProcess:continue
                        seen[(child.pid,created)]=child
                        assert (name=='original_engine_chat.exe' and exe==a.directory.resolve()/name) or (name=='conhost.exe' and exe==Path('C:/Windows/System32/conhost.exe').resolve()),('unexpected child',name,exe)
                except psutil.NoSuchProcess:pass
                with log.open('rb') as stream:
                    stream.seek(offset);chunk=stream.read();end=chunk.rfind(b'\n')+1
                    if end:print(chunk[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.1)
        guard();record.update(exit_code=worker_process.returncode,worker_OS_peak_through_exit=peak)
        assert worker_process.returncode==0,log.read_text(errors='replace')[-6000:]
        check_inputs(b);result=json.loads(a.out.read_bytes());assert result['schema']=='ORIGINAL_JOINT_HISTORY_RECOVERY_RESULT_V1' and len(result['arms'])==2
        assert result['new_optimizer_updates']==sum(x['new_updates'] for x in result['arms'])<=48
        assert all(x['durable_counter']==x['final_counter']==27+x['new_updates'] for x in result['arms'])
        assert result['source_calls']==result['reserved_queries']==0 and all(c['exit_code']==0 for c in result['children'])
        assert peak+result['max_direct_child_OS_peak']+proc.memory_info().peak_wset<=b['limits']['OS_bytes']
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()];assert sum(x['bytes'] for x in outputs)<=b['limits']['output_bytes']
        record.update(elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,result_sha256=sha(a.out),output_files=outputs,
            decision=result['decision'],resource_gates=True)
        write(terminal,record);print(json.dumps(dict(terminal=str(terminal),exit_code=worker_process.returncode,seconds=record['elapsed_seconds'],decision=result['decision'])),flush=True)
    except BaseException as error:
        for (pid,created),child in reversed(list(seen.items())):
            try:
                if child.is_running() and child.create_time()==created:child.kill()
            except psutil.NoSuchProcess:pass
        if worker_process is not None:
            if worker_process.poll() is None:worker_process.kill();worker_process.wait()
            peak=max(peak,reader(worker_process));record.update(exit_code=worker_process.returncode,worker_OS_peak_through_exit=peak)
        record.update(error=repr(error),elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write(failure,record);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.bind:bind(args)
    elif args.launch:launch(args)
    else:worker(args)
