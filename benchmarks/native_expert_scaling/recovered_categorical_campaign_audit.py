"""Independent stored-math audit at an explicitly recovered failed-capture boundary.

The mathematical core is copied from the frozen original auditor; no original
receipt/result is forged and no model/history/native/optimizer replay occurs.
"""
import argparse,gc,json,math,os,struct,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha
sys.path.insert(0,str(SITE))
from original_categorical_campaign_geometry import decode_transition,tensor_info,read,pack_check,proposal_blocks,ulp_order,native_metrics,bridge,choose,emit
from original_categorical_causal_preflight import gradient_role
from original_categorical_changed_history import changed_state


def recovered_index(binding,seal,directory):
    """Index only durable complete units; never reconstruct transient trackers."""
    lookup={x['path']:x for x in seal['outputs']}
    def item(path):
        key=str(Path(path).resolve());expected=lookup[key]
        assert extent(path)==expected
        return expected
    def record(path):
        item(path);return json.loads(Path(path).read_bytes())
    assert len(binding['FIT_order'])==24 and len(set(binding['FIT_order']))==24
    expected_steps=[directory/f'step_{i:02d}'/'step.json' for i in range(1,25)]
    assert set(directory.glob('step_*/step.json'))==set(expected_steps)
    steps=[record(p) for p in expected_steps]
    for ordinal,(step,identifier) in enumerate(zip(steps,binding['FIT_order']),1):
        assert step['ordinal']==ordinal and step['id']==identifier
        assert [x['alpha'] for x in step['trials']]==binding['alphas']
        folder=directory/f'step_{ordinal:02d}'
        assert record(folder/'history.json')==step['history']
        for row in step['trials']:
            assert record(folder/f"alpha_{row['alpha']:g}"/'trial.json')==row
    finals=[record(directory/('final_'+rec['id']+'.json')) for rec in binding['records']]
    assert len(finals)==48 and len({x['id'] for x in finals})==48
    assert [x['id'] for x in finals]==[x['id'] for x in binding['records']]
    snapshots=[dict(ordinal=i,extent=item(directory/f'step_{i:02d}'/'masters.pt')) for i in binding['milestones']]
    assert not (directory/'final.stream.receipt.json').exists()
    assert not list(directory.glob('final.*.task.json'))
    assert not list(directory.glob('final.stream.[0-9][0-9][0-9][0-9].json'))
    return dict(steps=steps,pending=None,final_native=finals,snapshots=snapshots,
                final_packed=steps[-1]['selected_packed'],source_parameters=steps[-1]['selected_parameters'])


def audit(a):
    import numpy as np,psutil,torch
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([1]);torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    binding=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    b=json.loads(Path(binding['original_numeric_binding']['path']).read_bytes())
    seal=json.loads(Path(binding['recovery_seal']['path']).read_bytes())
    failed=json.loads(Path(binding['original_failed_terminal']['path']).read_bytes())
    assert seal['schema']=='CATEGORICAL_CAMPAIGN_RECOVERY_SEAL_V1'
    assert seal['original_numeric_binding']==binding['original_numeric_binding']
    assert seal['original_failed_terminal']==binding['original_failed_terminal']
    assert seal['original_holder_binding']==binding['original_holder_binding']
    assert not seal['original_successful_custody'] and seal['producer_result_missing']
    assert all(seal[k] for k in ('original_bound_inputs_currently_exact_before_after','output_tree_metadata_stable_during_seal','output_hashes_at_recovery_boundary','unrecorded_stream_execution_unknown'))
    assert failed['exit_code']!=0 and failed['error'] is not None
    assert failed['freeze']==seal['freeze']==binding['original_freeze']
    assert failed['binding_sha256']==binding['original_numeric_binding']['sha256']
    assert not Path(binding['missing_original_result']).exists()
    last_check=0.
    def guard():
        nonlocal last_check
        assert time.monotonic()-start<binding['limits']['seconds'] and proc.memory_info().peak_wset<=binding['limits']['OS_bytes']
        if time.monotonic()-last_check>=1:assert not proc.children(recursive=True);last_check=time.monotonic()
    for number,item in enumerate(binding['inputs']+seal['outputs'],1):
        assert extent(item['path'])==item;guard()
        if number%1000==0:print(json.dumps(dict(stage='worker_hash_check',files=number,seconds=time.monotonic()-start)),flush=True)
    lookup={i['path']:i for i in binding['inputs']+seal['outputs']}
    def sealed(item):
        assert lookup[str(Path(item['path']).resolve())]=={k:item[k] for k in ('path','bytes','sha256')}
    stored=recovered_index(b,seal,a.directory)
    original_sum=np.sum
    def independent_sum(value,axis=None,dtype=None,out=None,keepdims=False,initial=None,where=True):
        x=np.asarray(value)
        if axis==1 and x.ndim==2 and x.dtype==np.dtype('float64') and dtype is None and out is None and initial is None and where is True:
            v=np.asarray([math.fsum(row) for row in x],dtype='f8');return v[:,None] if keepdims else v
        kwargs=dict(axis=axis,dtype=dtype,out=out,keepdims=keepdims)
        if initial is not None:kwargs['initial']=initial
        if where is not True:kwargs['where']=where
        return original_sum(value,**kwargs)
    np.sum=independent_sum
    maximum_metric=maximum_group=0.;maximum_ulp=0;audited=[];native_receipts={};opaque_partial=[]
    try:
        masters=changed_state(b['initial_binding'])['model'];packed=b['initial_native']['packed'];current_witness=b['initial_native']['expected_witness']
        assert {n:tensor_info(v) for n,v in masters.items()}==b['source_parameters']
        records={x['id']:x for x in b['records']}
        def check_metrics(actual,saved):
            nonlocal maximum_metric
            error=float(np.max(abs(np.asarray(actual['KL_per_label'])-saved['KL_per_label'])))
            for key in ('mean_KL','weighted_KL','first_KL','route_mass_defect'):error=max(error,abs(actual[key]-saved[key]))
            assert error<=b['numeric']['metric_absolute'];maximum_metric=max(maximum_metric,error)
            for key in ('labels','disagreement','route_valid','route_unions'):assert actual[key]==saved[key]
        def check_native(rec,value,expected_pack):
            for key in ('packed','query','scores','routes','witness','summary','receipt'):sealed(value[key])
            assert value['packed']['sha256']==expected_pack['sha256'] and value['packed']['bytes']==expected_pack['bytes']
            assert Path(value['query']['path']).read_bytes()==struct.pack('<I',len(rec['student_input_ids']))+np.asarray(rec['student_input_ids'],dtype='<u4').tobytes()
            receipt=json.loads(Path(value['receipt']['path']).read_bytes())
            command=[b['executable']['path'],'--prefix',value['packed']['path'],value['query']['path'],*[value[k]['path'] for k in ('scores','routes','witness','summary')]]
            assert receipt['command']==command and receipt['exit_code']==0 and 0<receipt['seconds']<=b['capture_limits']['child_seconds']
            summary=json.loads(Path(value['summary']['path']).read_bytes())
            assert summary['n']==1152 and summary['V']==65537 and summary['inputs']==len(rec['student_input_ids']) and summary['max_input_id']==max(rec['student_input_ids'])
            assert summary['blob_bytes']==value['packed']['bytes'] and summary['integer_coordinates']==18432 and summary['expert_reference_bytes']==0
            assert math.isfinite(summary['elapsed_seconds']) and 0<summary['elapsed_seconds']<=receipt['seconds']+1e-5
            native_receipts[value['receipt']['path']]=receipt
            return native_metrics(rec,value,True)
        def check_history(history,rec):
            for item in history['artifacts'].values():sealed(item)
            assert history['parameters']=={n:tensor_info(v) for n,v in masters.items()}
            grads=torch.load(history['artifacts']['gradients']['path'],weights_only=True,map_location='cpu',mmap=True)
            assert set(grads)==set(history['gradient_stats'])==set(masters)-{'head','final_norm'} and len(grads)==90
            roles={}
            for name,v in grads.items():
                assert tensor_info(v)==history['gradient_stats'][name]
                role=gradient_role(name);roles[role]=roles.get(role,0.)+history['gradient_stats'][name]['squared_norm'];guard()
            assert roles==history['gradient_role_squared_norms']
            if history['reused']:assert rec['id']==b['FIT_order'][0] and history['artifacts']==b['initial_history']['artifacts']
            else:assert history['outer_history_forwards']==history['whole_backward_calls']==1 and history['state_gradient_checks']==2
            return grads
        units=[(step,True) for step in stored['steps']]
        if stored['pending'] and stored['pending']['history'] is not None:units.append((stored['pending'],False))
        assert [s['id'] for s in stored['steps']]==b['FIT_order'][:len(stored['steps'])]
        for unit,complete in units:
            rec=records[unit['id']];grads=check_history(unit['history'],rec)
            baseline=unit.get('baseline_native');diag=None;base=None
            if baseline is not None:
                base=check_native(rec,baseline,packed);check_metrics(base,unit['baseline_metrics'])
                assert Path(baseline['witness']['path']).read_bytes()==Path(current_witness['path']).read_bytes()
                diag=bridge(rec,b['targets'][rec['id']],b['head'],baseline,unit['history'],b['numeric'],True)
                saved=unit['bridge'];error=max(abs(diag['numeric'][k]-saved['numeric'][k]) for k in diag['numeric'])
                assert error<=b['numeric']['metric_absolute'] and diag['numeric_flags']==saved['numeric_flags'];maximum_metric=max(maximum_metric,error)
            assert [x['alpha'] for x in unit['trials']]==b['alphas'][:len(unit['trials'])]
            candidates=[];trial_models={}
            for row in unit['trials']:
                for item in row['transition'].values():sealed(item)
                model=decode_transition(masters,row['transition']['index']['path'],guard)
                assert {n:tensor_info(v) for n,v in model.items()}==row['parameters']
                dot=ideal=radius=0.;moved=0;case_ulp=0;case_group=0.
                for name,p in model.items():
                    if name in ('head','final_norm'):
                        assert np.array_equal(p.numpy().view('<u4'),read(b[name]).view('<u4'));continue
                    sealed(row['group_stats'][name]);stats=read(row['group_stats'][name]);groups=stats.shape[0]
                    assert stats.shape[1]==6;got=p.numpy().reshape(groups,-1)
                    for first,last,predicted,ind in proposal_blocks(name,masters[name],grads[name],row['alpha'],b,True):
                        case_ulp=max(case_ulp,int(np.max(abs(ulp_order(got[first:last])-ulp_order(predicted)))))
                        old=masters[name].numpy().reshape(groups,-1)[first:last].astype('f8')
                        g=grads[name].numpy().reshape(groups,-1)[first:last].astype('f8');delta=got[first:last].astype('f8')-old
                        ind[:,3]=np.sqrt(np.sum(delta*delta,axis=1));ind[:,4]=np.sum(delta*g,axis=1);ind[:,5]=np.count_nonzero(delta,axis=1)
                        reference=stats[first:last];error=float(np.max(abs(ind-reference)/np.maximum(1.,abs(reference))))
                        assert error<=b['numeric']['group_relative'];case_group=max(case_group,error)
                        dot+=float(ind[:,4].sum());ideal-=row['alpha']*float((ind[:,2]*ind[:,1]).sum())
                        radius=max(radius,float((ind[:,3]/ind[:,2]).max()));moved+=int(ind[:,5].sum());guard()
                    guard()
                assert case_ulp<=b['numeric']['proposal_max_F32_ULP'] and moved==row['moved_coefficients']
                for value,key in ((dot,'actual_dot_gradient'),(ideal,'ideal_dot_gradient'),(radius,'max_relative_displacement')):
                    assert abs(value-row[key])<=b['numeric']['aggregate_relative']*max(1.,abs(row[key]))
                sealed(row['packed']);sealed(row['expected_witness'])
                quant,witness=pack_check(masters,model,packed['path'],row['packed']['path'],b['fields'],guard)
                assert quant==row['quantization'] and witness==Path(row['expected_witness']['path']).read_bytes()==Path(row['native']['witness']['path']).read_bytes()
                met=check_native(rec,row['native'],row['packed']);check_metrics(met,row['metrics'])
                flags=dict(group_radius=radius<=row['alpha']+b['numeric']['relative_radius_slack'],actual_direction_descent=dot<0,
                    quantization_certificate=quant['changed_among_certified']==0,route_mass=met['route_valid'] and met['route_mass_defect']<=b['numeric']['routing_mass_defect'])
                assert flags==row['numeric_flags'];candidates.append(dict(alpha=row['alpha'],metrics=met,numeric_flags=flags))
                maximum_ulp=max(maximum_ulp,case_ulp);maximum_group=max(maximum_group,case_group)
                trial_models[row['alpha']]=model
                del model;gc.collect();guard()
            if complete:
                assert len(candidates)==2 and base is not None and diag is not None
                selection=choose(candidates,base,all(diag['numeric_flags'].values()));assert selection==unit['selection']
                if selection['accepted']:
                    masters=trial_models[selection['alpha']];packed=next(x['packed'] for x in unit['trials'] if x['alpha']==selection['alpha'])
                    current_witness=next(x['expected_witness'] for x in unit['trials'] if x['alpha']==selection['alpha'])
                assert packed==unit['selected_packed'] and {n:tensor_info(v) for n,v in masters.items()}==unit['selected_parameters']
            audited.append(dict(id=rec['id'],completed_direction=complete,trials=len(candidates),bridge_flags=diag['numeric_flags'] if diag else None))
            print(json.dumps(dict(audited_case=rec['id'],trials=len(candidates),seconds=time.monotonic()-start)),flush=True)
            del grads,trial_models;gc.collect();guard()
        assert {n:tensor_info(v) for n,v in masters.items()}==stored['source_parameters'] and packed==stored['final_packed']
        for snapshot in stored['snapshots']:
            sealed(snapshot['extent']);s=torch.load(snapshot['extent']['path'],weights_only=True,map_location='cpu',mmap=True)
            step=stored['steps'][snapshot['ordinal']-1]
            assert s['schema']=='ORIGINAL_CATEGORICAL_CAMPAIGN_STATE_V1' and s['binding_sha256']==a.binding_sha and s['directions']==snapshot['ordinal']
            assert s['optimizer_updates']==0 and s['accepted_displacements']==sum(x['selection']['accepted'] for x in stored['steps'][:snapshot['ordinal']])
            assert {n:tensor_info(v) for n,v in s['model'].items()}==step['selected_parameters'];del s;guard()
        final=[]
        for row in stored['final_native']:
            rec=records[row['id']];assert rec['split']==row['split'] and rec['domain']==row['domain']
            met=check_native(rec,row['native'],packed);check_metrics(met,row['metrics']);final.append(dict(**row,metrics=met));guard()
            assert Path(row['native']['witness']['path']).read_bytes()==Path(current_witness['path']).read_bytes()
        # Original stream stage never completed its durable seal; this audit
        # intentionally adjudicates all stored learning/forced observations only.
        owned_paths=sorted(a.directory.rglob('native.receipt.json'))
        for path in owned_paths:sealed(extent(path))
        owned=[json.loads(p.read_bytes()) for p in owned_paths]
        identity=lambda x:(x['pid'],x['creation_time'])
        assert len(owned)==115 and len({identity(x) for x in owned})==len(owned)
        own_map={identity(x):x for x in owned}
        referenced={identity(x):x for p,x in native_receipts.items() if Path(p).resolve().is_relative_to(a.directory.resolve())}
        assert referenced==own_map
        for row in owned:
            assert row['command'][:2]==[b['executable']['path'],'--prefix'] and len(row['command'])==8
            assert row['pid']>0 and math.isfinite(row['creation_time']) and row['OS_peak']>=0 and 0<row['seconds']<=b['capture_limits']['child_seconds']
            assert row['exit_code']==0
        observed=failed['observed_native_children']
        assert len({identity(x) for x in observed})==len(observed)
        assert {identity(x) for x in observed}==set(own_map)
        for row in observed:
            assert row['command']==own_map[identity(row)]['command']
            assert 0<=row['observed_OS_peak']<=failed['native_OS_peak']
        assert max(x['OS_peak'] for x in owned)<=failed['native_OS_peak']
        def aggregate(split):
            rows=[x for x in final if x['split']==split]
            assert len(rows)==24
            return dict(cases=len(rows),labels=sum(x['metrics']['labels'] for x in rows),
                case_KL=float(np.mean([x['metrics']['mean_KL'] for x in rows])),
                case_disagreement=float(np.mean([x['metrics']['disagreement']/x['metrics']['labels'] for x in rows])),
                first_weighted_KL=float(np.mean([x['metrics']['weighted_KL'] for x in rows])))
        summary={split:aggregate(split) for split in ('FIT','DEV')}
        domains={domain:[x for x in final if x['split']=='DEV' and x['domain']==domain] for domain in sorted({x['domain'] for x in b['records'] if x['split']=='DEV'})}
        domain_flags=dict(KL=all(rows and np.mean([x['metrics']['mean_KL'] for x in rows])<=b['gates']['domain_case_KL'] for rows in domains.values()),
            disagreement=all(rows and np.mean([x['metrics']['disagreement']/x['metrics']['labels'] for x in rows])<=b['gates']['domain_case_disagreement'] for rows in domains.values()))
        stored_numeric_flags=dict(all_baseline_bridges=all(all(x['bridge_flags'].values()) for x in audited),
            all_candidate_flags=all(all(row['numeric_flags'].values()) for step in stored['steps'] for row in step['trials']))
        lim=b['capture_limits']
        original_resource_flags=dict(GPU_allocated=None,GPU_reserved=None,
            holder_worker_native_OS=failed['worker_OS_peak_through_exit']+failed['native_OS_peak']+failed['launcher_OS_peak_snapshot']<=lim['OS_bytes'],
            stored_tree_output=sum(x['bytes'] for x in seal['outputs'])<=lim['output_bytes'],
            original_worker_reserve=None,original_holder_time=failed['elapsed_seconds']<=lim['seconds'],
            completed_prefix_child_time=all(x['seconds']<=lim['child_seconds'] for x in owned),
            original_stream_child_time=None)
        quality_flags=dict(DEV_case_KL=summary['DEV']['case_KL']<=b['gates']['DEV_case_KL'],
            DEV_case_disagreement=summary['DEV']['case_disagreement']<=b['gates']['DEV_case_disagreement'],
            DEV_domains=all(domain_flags.values()),canonical_tasks=None,own_answer_followup=None)
        unknowns=['original_successful_in_flight_custody','original_GPU_allocated_peak','original_GPU_reserved_peak',
                  'transient_history_and_proposal_attempt_trackers','original_worker_elapsed_seconds',
                  'initial_unrecorded_stream_execution','precise_original_holder_assertion']
        guard()
        emit(a.out,dict(schema='RECOVERED_CATEGORICAL_CAMPAIGN_AUDIT_V1',binding=extent(a.binding),recovery_seal=binding['recovery_seal'],
            original_failed_terminal=binding['original_failed_terminal'],original_successful_custody=False,
            decision='RECOVERED_STORED_MATH_AUDITED',complete_original_campaign=False,
            complete_input_output_hashes_at_recovery_boundary=True,independent_integer_byte_transitions=True,
            independent_fsum_all_proposals=True,full_original_packing_all48_trials=True,
            all24_completed_gradients_and_bridges=True,all48_final_forced_cases=True,
            audited=audited,final_summary=summary,final_packed=stored['final_packed'],
            final_parameters=stored['source_parameters'],quality_flags=quality_flags,DEV_domain_flags=domain_flags,
            stored_numeric_flags=stored_numeric_flags,original_resource_flags=original_resource_flags,unknown_original_measurements=unknowns,
            stored_counts=dict(completed_steps=len(stored['steps']),completed_trials=sum(len(x['trials']) for x in stored['steps']),
                accepted_displacements=sum(x['selection']['accepted'] for x in stored['steps']),
                completed_native_prefix_receipts=len(owned),milestone_ordinals=[x['ordinal'] for x in stored['snapshots']]),
            inferred_from_completed_history_metadata=dict(reused_history_gradient_cases=1,fresh_history_records=23,
                reported_outer_history_forwards=sum(x['history'].get('outer_history_forwards',0) for x in stored['steps']),
                reported_whole_backward_calls=sum(x['history'].get('whole_backward_calls',0) for x in stored['steps'])),
            max_metric_delta=maximum_metric,max_group_relative_error=maximum_group,max_proposal_ULP=maximum_ulp,
            quality_admission=False,speed_admission=False,history_forwards=0,backward_calls=0,native_calls=0,GPU_calls=0,optimizer_updates=0,
            seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            scope='Exhaustive independent stored learning and forced-case audit at a recovered boundary. '
                  'Original failed receipt preserved; unavailable original measurements remain unknown. '
                  'Stored numerical checks do not establish successful original capture, useful quality, native chat tasks, useful speed or general conversion.'))
    finally:np.sum=original_sum


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('binding','directory','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--binding-sha',required=True)
    a=p.parse_args()
    assert not a.out.exists() and not a.out.with_suffix('.fault.json').exists()
    try:audit(a)
    except BaseException as error:
        emit(a.out.with_suffix('.fault.json'),dict(schema='RECOVERED_CATEGORICAL_CAMPAIGN_AUDIT_FAULT_V1',error=repr(error),traceback=traceback.format_exc()))
        raise
