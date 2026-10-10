"""Finite full24 FIT categorical transfer in the original deployment geometry."""
import argparse,gc,json,math,os,shutil,struct,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,raw
sys.path.insert(0,str(SITE))
from original_categorical_campaign_geometry import array,emit,read,tensor_info,View,proposal,pack_check,backward_case,bridge,native_metrics,choose
from original_categorical_supervision import load_fit_supervision
from original_categorical_changed_history import qualified_result,changed_state
from original_master_transition import encode
from original_falcon_whole_recovery import memory_reader


def bind(a):
    start=time.monotonic()
    import numpy as np,psutil,torch
    import torch._dynamo
    from original_categorical_history_learner import CategoricalHistoryLearner
    from original_engine_chat import ChatTokenizer
    import original_joint_native_assessment
    assert not a.binding.exists()
    disk_free=shutil.disk_usage(ROOT).free;assert disk_free>=(140+24)<<30
    fixture=json.loads((DOC/'original_categorical_campaign_reviewed_preflight_20261010.json').read_bytes())
    assert all(fixture[k] for k in ('AST_PASS','constrained_feasible_selection_PASS','tie_smaller_alpha_PASS','baseline_numeric_rejection_PASS',
        'independent_all3_transition_modes_PASS','wrong_source_rejected','variable_history_synthetic_F64_loss_gradient_PASS','route_ID_fault_detected',
        'original_stream_counter_cache_EOS_PASS','core_counter_EOS_faults_rejected'))
    names=['original_categorical_changed_history_binding_20261010.json','original_categorical_changed_history_result_20261010.json','original_categorical_changed_history_stored_adjudication_20261010.json','original_master_transition_result_20261010.json','original_master_transition_stored_adjudication_20261010.json','categorical_native_assessment_binding_20261010.json','original_categorical_trust_step_binding_20261010.json']
    paths=[DOC/n for n in names];cb,r,ar,storage,sar,nb,tb=[json.loads(p.read_bytes()) for p in paths]
    for p in paths[1:5]:qualified_result(p)
    assert r['decision']==ar['decision']=='CHANGED_CAUSAL_BRIDGE_QUALIFIED' and all(ar['numeric_flags'].values())
    assert ar['result']==extent(paths[1]) and ar['binding']==extent(paths[0])
    assert sar['decision']=='LOSSLESS_MASTER_TRANSITION_QUALIFIED' and sar['all92_parameters_721008128_words_exact'] and sar['result']==extent(paths[3])
    records=nb['records'];fit=[x['id'] for x in records if x['split']=='FIT'];dev=[x['id'] for x in records if x['split']=='DEV']
    assert len(fit)==len(dev)==24 and set(fit).isdisjoint(dev)
    assert sum(len(x['positions']) for x in records if x['split']=='FIT')==4422
    assert sum(len(x['positions']) for x in records if x['split']=='DEV')==4386
    first=cb['record']['id'];assert first in fit
    order=[first]+[x for x in fit if x!=first]
    compiled=json.loads(Path(cb['compiled_result']['path']).read_bytes());target={x['id']:x for x in compiled['records']}
    for rec in records:assert target[rec['id']]['split']==rec['split'] and target[rec['id']]['positions']==rec['positions']
    tokenizer=ChatTokenizer(nb['source'])
    for case in nb['cases']:assert tokenizer.encode(case['messages'])==case['input_ids']
    files=[Path(__file__),B/'original_categorical_campaign_audit.py',B/'original_categorical_campaign_launch.py',
        B/'original_categorical_campaign_preflight.py',DOC/'original_categorical_campaign_structural_preflight_20261010.json',
        DOC/'original_categorical_campaign_reviewed_preflight_20261010.json',DOC/'original_categorical_campaign_preflight_registration_fault_20261010.json',a.protocol,*paths,
        *[p.with_suffix('.terminal.json') for p in paths[1:5]],Path(cb['compiled_result']['path']),Path(cb['compiled_audit']['path']),
        Path(cb['compiled_audit']['path']).with_suffix('.terminal.json'),Path(cb['source_state']['path']),
        Path(cb['head']['path']),Path(cb['final_norm']['path']),Path(nb['executable']['path']),Path(nb['original_bodies']['source']['path'])]
    files += [Path(x['path']) for x in r['artifacts'].values()]
    files += [Path(x['path']) for x in cb['native'].values()]
    files += [Path(cb['native']['packed']['path']).parent/'native.receipt.json']
    files += [Path(rec['logits']['path']) for rec in records]
    files += [Path(x[k]['path']) for x in target.values() for k in ('moments','negative_entropy','source_argmax')]
    files += [Path(case['source_record']['path']) for case in nb['cases']]
    files += [Path(nb['source'])/n for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json')]
    for module in list(sys.modules.values()):
        for key in ('__file__','__cached__'):
            p=getattr(module,key,None)
            if isinstance(p,(str,os.PathLike)) and Path(p).is_file():files.append(Path(p))
    files += [Path(sys.executable),Path(sys.executable).parent/'python312.dll']
    files += sorted((SITE/'torch/lib').glob('*.dll'))+sorted((SITE/'numpy.libs').glob('*.dll'))+sorted((SITE/'psutil').glob('*.pyd'))
    inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in files)]
    for item in [cb['source_state'],cb['head'],cb['final_norm'],nb['executable'],*r['artifacts'].values(),*cb['native'].values()]:assert extent(item['path'])=={k:item[k] for k in ('path','bytes','sha256')}
    assert (torch.__version__,np.__version__,psutil.__version__)==('2.6.0+cu124','2.4.6','7.2.2')
    emit(a.binding,dict(schema='ORIGINAL_CATEGORICAL_CAMPAIGN_BINDING_V1',inputs=inputs,
        initial_binding=cb,initial_binding_extent=extent(paths[0]),initial_result=extent(paths[1]),initial_audit=extent(paths[2]),
        initial_history=r,initial_masters=cb['source_state'],initial_native=cb['native'],source_parameters=cb['source_parameters'],
        records=records,targets=target,FIT_order=order,DEV_ids=dev,compiled_result=cb['compiled_result'],compiled_audit=cb['compiled_audit'],
        head=cb['head'],final_norm=cb['final_norm'],fields=tb['fields'],executable=nb['executable'],original_bodies=nb['original_bodies'],
        cases=nb['cases'],source=nb['source'],followup=nb['followup'],special_ids=nb['special_ids'],gates=nb['gates'],
        alphas=[1e-4,1e-3],RMS_radius_floor=1e-5,chunk_groups=32,milestones=[6,12,18,24],transition_chunk_bytes=4<<20,
        numeric=dict(**cb['numeric'],proposal_max_F32_ULP=1,group_relative=1e-9,aggregate_relative=1e-9,relative_radius_slack=1e-6),
        capture_limits=dict(seconds=18000,reserve_seconds=1800,OS_bytes=32<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,
            output_bytes=140<<30,free_disk_bytes=24<<30,log_bytes=8<<20,child_seconds=100,stream_seconds=300),
        audit_limits=dict(seconds=32400,OS_bytes=24<<30,output_bytes=4<<20,log_bytes=8<<20),
        scope='One finite24-direction FIT epoch, first saved changed-point gradient reused;23 NEW full histories/backwards. '
              'Two independent finite proposals per direction, both archives/packs retained, actual C acceptance. '
              'All48 final FIT/DEV and canonical16 tasks+own-answer followup; no DEV checkpoint selection, old Adam/source/RESERVED/T4. '
              'Conditional memo reuse only exact packed hash+input IDs. Not useful-n/family/DRAM or same-artifact50 admission.',
        disk_free_before_binding=disk_free,binding_preparation_seconds=time.monotonic()-start))
    print(json.dumps(dict(binding_sha256=sha(a.binding),inputs=len(inputs),bytes=sum(x['bytes'] for x in inputs),seconds=time.monotonic()-start)),flush=True)


def capture(a):
    import numpy as np,psutil,torch
    from original_categorical_history_learner import CategoricalHistoryLearner
    from original_wide_learner import export
    from original_joint_native_assessment import tasks
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    start=time.monotonic()
    assert shutil.disk_usage(a.directory.parent).free>=b['capture_limits']['output_bytes']+b['capture_limits']['free_disk_bytes']
    a.directory.mkdir(exist_ok=False);proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and torch.cuda.device_count()==1
    torch.cuda.reset_peak_memory_stats();reader=memory_reader();child_peak=0;children=[];steps=[];final_rows=[];snapshots=[]
    stage='restore';last_progress=last_scan=0.;cache={};native_calls=0;backwards=0;displacements=0;task_result=None;fault=None;pending=None;history_attempts=[];proposal_attempts=[];stream_attempts=0
    masters=changed_state(b['initial_binding'])['model'];current_pack=b['initial_native']['packed'];current_witness=b['initial_native']['expected_witness'];model=None
    def observe_peak(value):
        nonlocal child_peak
        child_peak=max(child_peak,value)
    def guard():
        nonlocal last_scan
        lim=b['capture_limits'];assert time.monotonic()-start<lim['seconds']-lim['reserve_seconds']
        assert proc.memory_info().peak_wset+child_peak<=lim['OS_bytes']
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        if time.monotonic()-last_scan>=1:
            assert sum(p.stat().st_size for p in a.directory.rglob('*') if p.is_file())<=lim['output_bytes']
            assert shutil.disk_usage(a.directory).free>=lim['free_disk_bytes'];last_scan=time.monotonic()
    def event(**values):
        guard();print(json.dumps(dict(stage=stage,directions=len(steps),seconds=time.monotonic()-start,**values)),flush=True)
    def progress(mode,n,total,site):
        nonlocal last_progress
        guard()
        if time.monotonic()-last_progress>=15:event(operation=mode,n=n,total=total,site=site);last_progress=time.monotonic()
    def run(command,directory):
        nonlocal native_calls
        began=time.monotonic();peak=0;log=directory/'native.log'
        with log.open('xb') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,creationflags=8);created=psutil.Process(child.pid).create_time()
            native_calls+=1
            try:
                while child.poll() is None:
                    peak=max(peak,reader(child));observe_peak(peak);guard()
                    assert time.monotonic()-began<b['capture_limits']['child_seconds'];time.sleep(.1)
                peak=max(peak,reader(child));observe_peak(peak);assert child.returncode==0
            finally:
                if child.poll() is None:child.kill();child.wait()
                receipt=dict(command=command,pid=child.pid,creation_time=created,exit_code=child.returncode,OS_peak=peak,seconds=time.monotonic()-began)
                children.append(receipt);emit(directory/'native.receipt.json',receipt)
        return receipt
    def native(rec,packed,directory):
        nonlocal native_calls
        key=(packed['sha256'],tuple(rec['student_input_ids']))
        if key in cache:return dict(**cache[key],reused=True)
        directory.mkdir(exist_ok=False);query=directory/'query.u32'
        raw(query,struct.pack('<I',len(rec['student_input_ids']))+np.asarray(rec['student_input_ids'],dtype='<u4').tobytes())
        paths=[directory/n for n in ('scores.f32','routes','witness','summary.json')]
        command=[b['executable']['path'],'--prefix',packed['path'],str(query),*[str(p) for p in paths]]
        receipt=run(command,directory)
        value=dict(packed=packed,query=extent(query),**{k:extent(p) for k,p in zip(('scores','routes','witness','summary'),paths)},receipt=extent(directory/'native.receipt.json'))
        cache[key]=value;return dict(**value,reused=False)
    initial=b['initial_native'];first=next(x for x in b['records'] if x['id']==b['FIT_order'][0])
    cache[(current_pack['sha256'],tuple(first['student_input_ids']))]=dict(packed=current_pack,query=initial['query'],scores=initial['native_scores'],routes=initial['native_routes'],witness=initial['native_witness'],summary=initial['native_summary'],receipt=extent(Path(current_pack['path']).parent/'native.receipt.json'))
    def save_result(complete):
        emit(a.out,dict(schema='ORIGINAL_CATEGORICAL_CAMPAIGN_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='COMPLETE_PENDING_FULL_AUDIT' if complete else 'INTERRUPTED_PENDING_PARTIAL_AUDIT',
            steps=steps,pending=pending,final_native=final_rows,tasks=task_result,snapshots=snapshots,final_packed=current_pack,
            source_parameters={n:tensor_info(v) for n,v in masters.items()},fault=fault,
            fresh_history_forwards=sum(x['history_completed'] for x in history_attempts),
            whole_backwards=sum(x['backward_completed'] for x in history_attempts),
            history_attempts=history_attempts,completed_gradient_cases=backwards,reused_history_gradient_cases=1,
            parameter_proposals=sum(x['F32_proposal_completed'] for x in proposal_attempts),completed_trial_candidates=displacements,
            proposal_attempts=proposal_attempts,accepted_displacements=sum(s['selection']['accepted'] for s in steps),
            native_prefix_calls=native_calls,native_stream_attempts=stream_attempts,
            native_stream_calls=int((a.directory/'final.stream.receipt.json').exists()),children=children,optimizer_restore=False,optimizer_updates=0,
            source_calls=0,DEV_optimizer_queries=0,reserved_queries=0,T4_calls=0,
            worker_OS_peak=proc.memory_info().peak_wset,native_OS_peak=child_peak,GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-start,quality_admission=False,
            speed_admission=False,scope=b['scope']))
    try:
        assert {n:tensor_info(v) for n,v in masters.items()}==b['source_parameters'];guard()
        model=CategoricalHistoryLearner(device='cuda');model.load_state_dict(masters)
        model.install_fixed_head(torch.from_numpy(read(b['head']).copy()));model.use_checkpoint=True
        payload=load_fit_supervision(b['compiled_result']['path'],b['compiled_audit']['path'])
        fit={r['id']:r for r in payload['records']};records={r['id']:r for r in b['records']}
        for ordinal,identifier in enumerate(b['FIT_order'],1):
            directory=a.directory/f'step_{ordinal:02d}';directory.mkdir();rec=records[identifier];stage='fresh_causal_direction'
            pending=dict(ordinal=ordinal,id=identifier,history=None,baseline_native=None,trials=[])
            event(id=identifier,ordinal=ordinal,history=len(rec['student_input_ids']))
            if ordinal==1:
                history={k:b['initial_history'][k] for k in ('artifacts','parameters','gradient_stats','gradient_role_squared_norms','block_forward_calls')}
                grads=torch.load(history['artifacts']['gradients']['path'],weights_only=True,map_location='cpu',mmap=True)
                assert {n:tensor_info(v) for n,v in grads.items()}==history['gradient_stats']
                history=dict(history,outer_history_forwards=0,whole_backward_calls=0,state_gradient_checks=0,reused=True)
            else:
                tracker=dict(id=identifier,history_started=0,history_completed=0,backward_started=0,backward_completed=0,state_checks_started=0,state_checks_completed=0,block_calls=[0]*6);history_attempts.append(tracker)
                history,grads=backward_case(model,masters,rec,fit[identifier],directory/'history',guard,progress,tracker);backwards+=1
                path=directory/'history'/'gradients.pt';torch.save(grads,path);history['artifacts']['gradients']=extent(path);history['reused']=False
            pending['history']=history;emit(directory/'history.json',history)
            stage='current_native';baseline=native(rec,current_pack,directory/'baseline_native')
            assert Path(baseline['witness']['path']).read_bytes()==Path(current_witness['path']).read_bytes()
            diag=bridge(rec,b['targets'][identifier],b['head'],baseline,history,b['numeric'])
            base_metrics=native_metrics(rec,baseline);trials=[];trial_models={}
            pending.update(baseline_native=baseline,baseline_metrics=base_metrics,bridge=diag)
            for alpha in b['alphas']:
                attempt=dict(ordinal=ordinal,id=identifier,alpha=alpha,F32_proposal_started=1,F32_proposal_completed=0,trial_completed=0);proposal_attempts.append(attempt)
                trialdir=directory/f'alpha_{alpha:g}';trialdir.mkdir();candidate={};group_stats={};dot=ideal=radius=0.;moved=0
                stage='finite_proposal';event(id=identifier,alpha=alpha)
                for name,value in masters.items():
                    if name in ('head','final_norm'):candidate[name]=value
                    else:
                        proposed,s=proposal(name,value,grads[name],alpha,b);candidate[name]=proposed
                        group_stats[name]=array(trialdir,name.replace('.','_')+'.group_stats',s)
                        dot+=float(s[:,4].sum());ideal-=alpha*float((s[:,2]*s[:,1]).sum());radius=max(radius,float((s[:,3]/s[:,2]).max()));moved+=int(s[:,5].sum())
                    guard()
                attempt['F32_proposal_completed']=1
                parameters={name:tensor_info(value) for name,value in candidate.items()}
                stage='lossless_trial_custody';index=encode(masters,candidate,trialdir/'transition',b['transition_chunk_bytes'],lambda *_:guard())
                for row in index['parameters']:assert row['target_sha256']==parameters[row['name']]['sha256']
                transition={k:extent(trialdir/'transition'/name) for k,name in [('index','transition.json'),('blob','transition.bin')]}
                stage='original_export';packed=trialdir/'candidate.packed';export(View(candidate),packed);packed_item=extent(packed)
                quant,witness=pack_check(masters,candidate,current_pack['path'],packed,b['fields'],guard)
                raw(trialdir/'expected.witness',witness);stage='trial_native';actual=native(rec,packed_item,trialdir/'native')
                assert Path(actual['witness']['path']).read_bytes()==witness
                met=native_metrics(rec,actual);flags=dict(group_radius=radius<=alpha+b['numeric']['relative_radius_slack'],
                    actual_direction_descent=dot<0,quantization_certificate=quant['changed_among_certified']==0,
                    route_mass=met['route_valid'] and met['route_mass_defect']<=b['numeric']['routing_mass_defect'])
                row=dict(alpha=alpha,parameters=parameters,group_stats=group_stats,transition=transition,packed=packed_item,native=actual,
                    expected_witness=extent(trialdir/'expected.witness'),quantization=quant,metrics=met,numeric_flags=flags,
                    actual_dot_gradient=dot,ideal_dot_gradient=ideal,max_relative_displacement=radius,moved_coefficients=moved)
                emit(trialdir/'trial.json',row);trials.append(row);displacements+=1;attempt['trial_completed']=1
                pending['trials']=list(trials)
                trial_models[alpha]=candidate
                del candidate;gc.collect();guard()
            selection=choose(trials,base_metrics,all(diag['numeric_flags'].values()))
            if selection['accepted']:
                winner=next(t for t in trials if t['alpha']==selection['alpha']);next_masters=trial_models[selection['alpha']];next_pack=winner['packed']
                next_witness=winner['expected_witness']
            else:next_masters=masters;next_pack=current_pack;next_witness=current_witness
            step=dict(ordinal=ordinal,id=identifier,history=history,baseline_native=baseline,baseline_metrics=base_metrics,bridge=diag,
                trials=trials,selection=selection,selected_packed=next_pack,selected_parameters={n:tensor_info(v) for n,v in next_masters.items()})
            emit(directory/'step.json',step);steps.append(step);masters=next_masters;current_pack=next_pack;current_witness=next_witness;pending=None
            if selection['accepted']:
                load=model.load_state_dict(masters,strict=False);assert load.missing_keys==['supervision_head'] and not load.unexpected_keys
                model.validate_fixed_geometry()
            if ordinal in b['milestones']:
                stage='milestone';path=directory/'masters.pt'
                torch.save(dict(schema='ORIGINAL_CATEGORICAL_CAMPAIGN_STATE_V1',model=masters,directions=ordinal,
                    binding_sha256=a.binding_sha,accepted_displacements=sum(s['selection']['accepted'] for s in steps),optimizer_updates=0),path)
                snapshots.append(dict(ordinal=ordinal,extent=extent(path)))
            model.zero_grad(set_to_none=True);del grads,history,trial_models,next_masters;gc.collect();torch.cuda.empty_cache()
            stage='direction_complete';event(id=identifier,accepted=selection['accepted'],weighted_before=base_metrics['weighted_KL'],weighted_best=min(r['metrics']['weighted_KL'] for r in trials))
        stage='full_final_FIT_DEV'
        for rec in b['records']:
            actual=native(rec,current_pack,a.directory/('final_'+rec['id']));met=native_metrics(rec,actual)
            assert Path(actual['witness']['path']).read_bytes()==Path(current_witness['path']).read_bytes()
            row=dict(id=rec['id'],split=rec['split'],domain=rec['domain'],native=actual,metrics=met);final_rows.append(row)
            emit(a.directory/('final_'+rec['id']+'.json'),row);event(id=rec['id'],split=rec['split'],mean_KL=met['mean_KL'])
        stage='canonical_own_history_tasks';task_binding={**b,'limits':b['capture_limits']}
        stream_attempts=1
        task_result=tasks(task_binding,a.directory,Path(b['executable']['path']),Path(current_pack['path']),'final',reader,observe_peak,guard,event)
        stage='complete';guard();event(operation='final_result_seal');save_result(True)
        print(json.dumps(dict(decision='COMPLETE_PENDING_FULL_AUDIT')),flush=True)
    except BaseException as error:
        fault=dict(stage=stage,error=repr(error),traceback=traceback.format_exc(),completed_directions=len(steps),completed_final_cases=len(final_rows),seconds=time.monotonic()-start)
        emit(a.directory/'first_fault.json',fault)
        # Exit normally with an explicitly interrupted record so completed data are sealed and audited.
        save_result(False);print(json.dumps(dict(decision='INTERRUPTED_PENDING_PARTIAL_AUDIT',fault=fault)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bind','capture-worker','audit-worker'):p.add_argument('--'+key,action='store_true')
    for key in ('binding','directory','out','source-result','protocol'):p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.bind:bind(a)
    elif a.capture_worker:capture(a)
    elif a.audit_worker:
        from original_categorical_campaign_audit import audit
        audit(a)
    else:raise RuntimeError('Use held GPU/CPU launcher')
