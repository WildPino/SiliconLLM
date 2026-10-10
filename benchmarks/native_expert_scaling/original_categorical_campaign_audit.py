"""Full stored campaign audit; accurate reductions and exact transition reconstruction."""
import gc,json,math,os,struct,time
from pathlib import Path
from original_packed_capacity import extent,sha
from original_categorical_campaign_geometry import decode_transition,tensor_info,read,pack_check,proposal_blocks,ulp_order,native_metrics,bridge,choose,emit,stream_counters
from original_categorical_causal_preflight import gradient_role
from original_categorical_changed_history import changed_state


def audit(a):
    import numpy as np,psutil,torch
    from original_engine_chat import ChatTokenizer
    from chatbot_falcon_usability import normalized
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([1]);torch.set_num_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes())
    t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert sha(a.binding)==a.binding_sha==r['binding_sha256']==t['binding_sha256'] and r['freeze']==t['freeze']==a.freeze
    assert t['exit_code']==0 and t['error'] is None and t['inputs_before_after_exact'] and t['result']==extent(a.source_result)
    last_check=0.
    def guard():
        nonlocal last_check
        assert time.monotonic()-start<b['audit_limits']['seconds'] and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes']
        if time.monotonic()-last_check>=1:assert not proc.children(recursive=True);last_check=time.monotonic()
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item;guard()
    lookup={i['path']:i for i in b['inputs']+t['outputs']}
    def sealed(item):
        assert lookup[str(Path(item['path']).resolve())]=={k:item[k] for k in ('path','bytes','sha256')}
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
        units=[(step,True) for step in r['steps']]
        if r['pending'] and r['pending']['history'] is not None:units.append((r['pending'],False))
        assert [s['id'] for s in r['steps']]==b['FIT_order'][:len(r['steps'])]
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
        assert {n:tensor_info(v) for n,v in masters.items()}==r['source_parameters'] and packed==r['final_packed']
        for snapshot in r['snapshots']:
            sealed(snapshot['extent']);s=torch.load(snapshot['extent']['path'],weights_only=True,map_location='cpu',mmap=True)
            step=r['steps'][snapshot['ordinal']-1]
            assert s['schema']=='ORIGINAL_CATEGORICAL_CAMPAIGN_STATE_V1' and s['binding_sha256']==a.binding_sha and s['directions']==snapshot['ordinal']
            assert s['optimizer_updates']==0 and s['accepted_displacements']==sum(x['selection']['accepted'] for x in r['steps'][:snapshot['ordinal']])
            assert {n:tensor_info(v) for n,v in s['model'].items()}==step['selected_parameters'];del s;guard()
        final=[]
        for row in r['final_native']:
            rec=records[row['id']];assert rec['split']==row['split'] and rec['domain']==row['domain']
            met=check_native(rec,row['native'],packed);check_metrics(met,row['metrics']);final.append(dict(**row,metrics=met));guard()
            assert Path(row['native']['witness']['path']).read_bytes()==Path(current_witness['path']).read_bytes()
        tokenizer=ChatTokenizer(b['source']);task_rows=[]
        paths=list(a.directory.glob('final.*.task.json'))
        for path in paths:
            row=json.loads(path.read_bytes());case=next(c for c in b['cases'] if c['id']==row['id']);request=row['request']
            assert row['category']==case['category'] and row['expected']==case['expected'] and row['messages']==case['messages']
            assert request['query_ids']==case['input_ids'];text=tokenizer.decode(request['generated_ids'])
            assert text==row['output_text'] and normalized(text)==row['normalized'] and row['correct']==(normalized(text)==case['expected'])
            assert row['blank']==(not text.strip()) and row['special_leak']==any(i in set(b['special_ids'])-{11,228} for i in request['generated_ids'])
            assert row['stop_policy_ok']==(bool(request['generated_ids']) and (request['generated_ids'][-1] in (11,228) or len(request['generated_ids'])==64))
            task_rows.append(row)
        stream_records=[json.loads(p.read_bytes()) for p in sorted(a.directory.glob('final.stream.[0-9][0-9][0-9][0-9].json'))]
        for index,request in enumerate(stream_records):
            assert request['index']==index
            if request['label']=='own_answer_followup':
                assert index>0 and stream_records[index-1]['label']==b['followup']['case_id']
                parent=stream_records[index-1];cached=parent['query_ids']+parent['generated_ids']
                compatible=request['query_ids'][:len(cached)]==cached
                stream_counters(request,len(cached) if compatible else 0,not compatible)
            else:
                case=next(c for c in b['cases'] if c['id']==request['label']);assert request['query_ids']==case['input_ids']
                stream_counters(request,0,True)
            for key in ('final_head','scores','routes'):sealed(request[key])
            head=np.fromfile(request['final_head']['path'],dtype='<f4');assert head.shape==(65537,) and np.isfinite(head).all()
            scores=np.fromfile(request['scores']['path'],dtype='<f4').reshape(-1,65537)
            assert len(scores)==len(request['generated_ids']) and np.isfinite(scores).all() and scores.argmax(-1).tolist()==request['generated_ids']
            from original_categorical_campaign_geometry import routes
            _,_,valid,defect=routes(request['routes'],request['core_calls']);assert valid and defect<=b['numeric']['routing_mass_defect']
            assert abs(defect-request['mass_defect'])<=b['numeric']['metric_absolute'];guard()
        assert all(row['request']==next(x for x in stream_records if x['label']==row['id']) for row in task_rows)
        assert len({x['id'] for x in task_rows})==len(task_rows)
        assert not stream_records or all(x['load_seconds']==stream_records[0]['load_seconds'] for x in stream_records)
        # All actual prefix attempts, including a child that failed before a full trace.
        owned_receipts=[json.loads(p.read_bytes()) for p in sorted(a.directory.rglob('native.receipt.json'))]
        identity=lambda x:(x['pid'],x['creation_time'])
        assert len({identity(x) for x in owned_receipts})==len(owned_receipts)==r['native_prefix_calls']==len(r['children'])
        assert {identity(x):x for x in owned_receipts}=={identity(x):x for x in r['children']}
        own_map={identity(x):x for x in owned_receipts}
        stream_path=a.directory/'final.stream.receipt.json';stream_receipt=None
        if stream_path.exists():
            stream_receipt=json.loads(stream_path.read_bytes());assert stream_receipt['command']==[b['executable']['path'],'--stream',packed['path']]
            assert identity(stream_receipt) not in own_map;own_map[identity(stream_receipt)]=stream_receipt
        assert r['native_stream_calls']==int(stream_receipt is not None) and r['native_stream_calls']<=r['native_stream_attempts']<=1
        for receipt in owned_receipts:
            assert receipt['command'][:2]==[b['executable']['path'],'--prefix'] and len(receipt['command'])==8
            assert math.isfinite(receipt['creation_time']) and receipt['pid']>0 and receipt['OS_peak']>=0 and receipt['seconds']>0
        for receipt_path,receipt in native_receipts.items():
            if Path(receipt_path).resolve().is_relative_to(a.directory.resolve()):assert own_map[identity(receipt)]==receipt
        for observed in t['observed_native_children']:
            actual=own_map[identity(observed)];assert observed['command']==actual['command']
            assert observed['observed_OS_peak']<=t['native_OS_peak']
        assert max([0]+[x['OS_peak'] for x in owned_receipts]+([stream_receipt['held_OS_peak']] if stream_receipt else []))<=r['native_OS_peak']<=t['native_OS_peak']
        assert r['worker_OS_peak']<=t['worker_OS_peak_through_exit']
        assert r['accepted_displacements']==sum(x['selection']['accepted'] for x in r['steps'])
        assert r['completed_trial_candidates']==sum(len(x['trials']) for x in units_as_steps(r))
        assert r['parameter_proposals']==sum(x['F32_proposal_completed'] for x in r['proposal_attempts'])
        assert r['completed_trial_candidates']==sum(x['trial_completed'] for x in r['proposal_attempts'])
        assert all(0<=x['trial_completed']<=x['F32_proposal_completed']<=x['F32_proposal_started']==1 for x in r['proposal_attempts'])
        expected_attempts=[(ordinal,identifier,alpha) for ordinal,identifier in enumerate(b['FIT_order'],1) for alpha in b['alphas']]
        assert [(x['ordinal'],x['id'],x['alpha']) for x in r['proposal_attempts']]==expected_attempts[:len(r['proposal_attempts'])]
        assert r['fresh_history_forwards']==sum(x['history_completed'] for x in r['history_attempts']) and r['whole_backwards']==sum(x['backward_completed'] for x in r['history_attempts'])
        assert [x['id'] for x in r['history_attempts']]==b['FIT_order'][1:1+len(r['history_attempts'])]
        assert all(0<=x['backward_completed']<=x['backward_started']<=x['history_completed']<=x['history_started']<=1 and
            0<=x['state_checks_completed']<=x['state_checks_started']<=2 for x in r['history_attempts'])
        complete=r['decision']=='COMPLETE_PENDING_FULL_AUDIT'
        if complete:
            assert r['fault'] is None and r['pending'] is None and len(r['steps'])==24 and len(final)==48 and len(task_rows)==16
            assert [x['id'] for x in final]==[x['id'] for x in b['records']] and r['fresh_history_forwards']==r['whole_backwards']==23
            assert r['parameter_proposals']==r['completed_trial_candidates']==48 and r['reused_history_gradient_cases']==1
            assert r['native_prefix_calls']<=119 and r['native_stream_calls']==r['native_stream_attempts']==1 and stream_receipt['exit_code']==0
            assert [x['ordinal'] for x in r['snapshots']]==b['milestones']
            assert r['tasks']['requests']==stream_records and len(stream_records)==17
            assert {x['id']:x for x in task_rows}=={x['id']:x for x in r['tasks']['records']}
            assert r['completed_gradient_cases']==23 and len(r['history_attempts'])==23
            assert [x['id'] for x in r['history_attempts']]==b['FIT_order'][1:]
            assert all(x['history_started']==x['history_completed']==x['backward_started']==x['backward_completed']==1 and x['state_checks_started']==x['state_checks_completed']==2 and x['block_calls']==[2]*6 for x in r['history_attempts'])
        else:
            assert r['decision']=='INTERRUPTED_PENDING_PARTIAL_AUDIT' and r['fault'] is not None
            # Seal every partial file while identifying the uncompleted unit explicitly.
            opaque_partial=[x['path'] for x in t['outputs'] if r['pending'] and f"step_{r['pending']['ordinal']:02d}" in Path(x['path']).parts]
        assert not r['optimizer_restore'] and all(r[k]==0 for k in ('optimizer_updates','source_calls','DEV_optimizer_queries','reserved_queries','T4_calls'))
        assert not r['quality_admission'] and not r['speed_admission']
        lim=b['capture_limits'];resource_flags=dict(GPU_allocated=r['GPU_allocated_peak']<=lim['GPU_allocated_bytes'],GPU_reserved=r['GPU_reserved_peak']<=lim['GPU_reserved_bytes'],
            worker_native_OS=r['worker_OS_peak']+r['native_OS_peak']<=lim['OS_bytes'],output=sum(x['bytes'] for x in t['outputs'])+a.source_result.stat().st_size<=lim['output_bytes'],
            worker_reserve=r['seconds']<lim['seconds']-lim['reserve_seconds'],prefix_child_time=all(x['seconds']<=lim['child_seconds'] for x in owned_receipts),
            stream_child_time=stream_receipt is None or stream_receipt['seconds']<=lim['stream_seconds'])
        def aggregate(split):
            rows=[x for x in final if x['split']==split]
            if not rows:return None
            return dict(cases=len(rows),labels=sum(x['metrics']['labels'] for x in rows),case_KL=float(np.mean([x['metrics']['mean_KL'] for x in rows])),
                case_disagreement=float(np.mean([x['metrics']['disagreement']/x['metrics']['labels'] for x in rows])),
                first_weighted_KL=float(np.mean([x['metrics']['weighted_KL'] for x in rows])))
        summary={s:aggregate(s) for s in ('FIT','DEV')}
        domains={domain:[x for x in final if x['split']=='DEV' and x['domain']==domain] for domain in sorted({x['domain'] for x in b['records'] if x['split']=='DEV'})}
        domain_flags=dict(KL=complete and all(rows and np.mean([x['metrics']['mean_KL'] for x in rows])<=b['gates']['domain_case_KL'] for rows in domains.values()),
            disagreement=complete and all(rows and np.mean([x['metrics']['disagreement']/x['metrics']['labels'] for x in rows])<=b['gates']['domain_case_disagreement'] for rows in domains.values()))
        categories={category:sum(x['correct'] for x in task_rows if x['category']==category) for category in sorted({x['category'] for x in b['cases']})}
        task_flags=dict(correct_min=sum(categories.values())>=b['gates']['task_correct_min'],
            every_category_min=all(v>=b['gates']['category_correct_min'] for v in categories.values()),
            blank_max=sum(x['blank'] for x in task_rows)<=b['gates']['blank_max'],
            no_special_leak=not any(x['special_leak'] for x in task_rows),stop_policy=all(x['stop_policy_ok'] for x in task_rows))
        if r['tasks'] is not None:
            assert r['tasks']['correct']==sum(categories.values()) and r['tasks']['categories']==categories and r['tasks']['gates']==task_flags
            follow=r['tasks']['own_answer_followup'];case=next(x for x in task_rows if x['id']==b['followup']['case_id'])
            messages=case['messages']+[dict(role='assistant',content=case['output_text']),dict(role='user',content=b['followup']['prompt'])]
            assert follow['messages']==messages and follow['request']['query_ids']==tokenizer.encode(messages)
            assert follow['output_text']==tokenizer.decode(follow['request']['generated_ids']) and follow['correct']==(normalized(follow['output_text'])==b['followup']['expected'])
            cached=case['request']['query_ids']+case['request']['generated_ids'];compatible=follow['request']['query_ids'][:len(cached)]==cached
            assert follow['exact_cached_prefix_compatible']==compatible and follow['request']['reused_prefix_ids']==(len(cached) if compatible else 0)
            receipt=r['tasks']['receipt'];assert receipt==stream_receipt and receipt['exit_code']==0
            assert follow==json.loads((a.directory/'final.own_answer_followup.json').read_bytes())
            assert follow['request']==next(x for x in stream_records if x['label']=='own_answer_followup')
            generated=sum(len(x['generated_ids']) for x in stream_records);assert generated==r['tasks']['generated_ids']
            for key,timing in (('raw_decode_ids_s','decode_seconds'),('raw_pipe_request_ids_s','pipe_request_seconds')):
                rate=generated/math.fsum(x[timing] for x in stream_records)
                assert abs(rate-r['tasks'][key])<=1e-9*max(1.,abs(rate))
        quality_flags=dict(complete=complete,DEV_case_KL=complete and summary['DEV']['case_KL']<=b['gates']['DEV_case_KL'],
            DEV_case_disagreement=complete and summary['DEV']['case_disagreement']<=b['gates']['DEV_case_disagreement'],
            DEV_domains=all(domain_flags.values()),tasks=complete and all(task_flags.values()))
        numeric_pass=complete and all(resource_flags.values()) and all(all(x['bridge_flags'].values()) for x in audited if x['bridge_flags']) and all(all(t['numeric_flags'].values()) for s in r['steps'] for t in s['trials'])
        decision='FULL_EPOCH_NUMERIC_QUALIFIED' if numeric_pass else ('FULL_EPOCH_NUMERIC_NOT_QUALIFIED' if complete else 'PARTIAL_EPOCH_STORED_PREFIX_AUDITED')
        guard();emit(a.out,dict(schema='ORIGINAL_CATEGORICAL_CAMPAIGN_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=decision,
            complete_campaign=complete,complete_input_output_hashes=True,independent_integer_byte_transitions=True,independent_fsum_all_proposals=True,
            full_original_packing_all_completed_trials=True,all_completed_gradients_and_bridges=True,audited=audited,final_summary=summary,
            task_correct=sum(x['correct'] for x in task_rows),task_cases=len(task_rows),task_flags=task_flags,quality_flags=quality_flags,DEV_domain_flags=domain_flags,resource_flags=resource_flags,max_metric_delta=maximum_metric,max_group_relative_error=maximum_group,
            max_proposal_ULP=maximum_ulp,opaque_partial_files=opaque_partial,quality_admission=False,speed_admission=False,
            history_forwards=0,backward_calls=0,native_calls=0,GPU_calls=0,optimizer_updates=0,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            scope='Full stored completed-unit audit, no history/backward/native/optimization replay. Partial raw files explicitly identified if interrupted. '
                  'Numeric qualification alone is not quality or useful speed/family/n/DRAM admission.'))
    finally:np.sum=original_sum


def units_as_steps(result):
    return result['steps']+([result['pending']] if result['pending'] is not None else [])
