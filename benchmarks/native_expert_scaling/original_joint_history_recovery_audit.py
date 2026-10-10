"""Stored-only adjudication of the frozen original-engine matched history pilot.

No learner construction, CUDA access, source inference or optimizer replay.
Run only after the experiment's launcher has terminated and sealed its outputs.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from original_packed_capacity import SITE, extent, sha, write
sys.path.insert(0, str(SITE))
V, L, E, K = 65537, 6, 1152, 8
FIELD = struct.Struct('<64s6I2Q')


def close_enough(x, y, atol=1e-10):
    assert math.isfinite(x) and math.isfinite(y)
    assert abs(x-y) <= atol * max(1., abs(x), abs(y)), (x, y)


def groups(rows, auxiliary=False):
    import numpy as np
    def group(values):
        if auxiliary:
            return dict(cases=len(values), **{key:float(np.mean([
                np.mean([s[key] for s in row['sites']]) for row in values]))
                for key in ('total', 'mean', 'centered', 'centered_relative')})
        labels = sum(row['labels'] for row in values)
        return dict(cases=len(values), labels=labels,
            case_KL=sum(row['KL'] for row in values)/len(values),
            label_KL=sum(sum(row['KL_per_label']) for row in values)/labels,
            case_disagreement=sum(row['disagreement_rate'] for row in values)/len(values),
            label_disagreement=sum(row['disagreement'] for row in values)/labels)
    return dict(**group(rows), domains={d:group([r for r in rows if r['domain']==d])
        for d in sorted({r['domain'] for r in rows})})


def aggregate_equal(actual, saved):
    maximum = 0.
    assert set(actual) == set(saved)
    for key, value in actual.items():
        if isinstance(value, dict):
            maximum = max(maximum, aggregate_equal(value, saved[key]))
        elif isinstance(value, int):
            assert value == saved[key]
        else:
            close_enough(value, saved[key]); maximum = max(maximum, abs(value-saved[key]))
    return maximum


def routes(path, count, saved_defect):
    import numpy as np
    dtype = np.dtype([('ids','<i4',(K,)),('mass','<f4',(K,))])
    assert Path(path).stat().st_size == count*L*64
    values = np.memmap(path, dtype=dtype, mode='r', shape=(count,L))
    ids, mass = values['ids'], values['mass']
    assert ids.min() >= 0 and ids.max() < E
    assert (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all()
    assert np.isfinite(mass).all() and (mass >= 0).all()
    defect = float(np.max(np.abs(mass.astype('f8').sum(-1)-1)))
    assert defect == saved_defect and defect <= 1e-6
    unions = [int(np.unique(ids[:,site]).size) for site in range(L)]
    del ids, mass, values
    return defect, unions


def native(rows, records):
    import numpy as np
    actual=[]; case_delta=label_delta=mass_peak=0.; labels=0; positions=0
    assert len(rows)==24 and {row['id'] for row in rows}=={r['id'] for r in records.values() if r['split']=='DEV'}
    for row in rows:
        rec=records[row['id']]; n=len(rec['student_input_ids']); m=len(rec['positions'])
        assert row['split']==rec['split']=='DEV' and row['domain']==rec['domain']
        assert row['history']==n and row['labels']==m and row['row_offset']==0 and not row['adopted']
        assert Path(row['logits']['path']).stat().st_size == n*V*4
        logits=np.memmap(row['logits']['path'],dtype='<f4',mode='r',shape=(n,V))
        for first in range(0,n,32): assert np.isfinite(logits[first:first+32]).all()
        teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(m,V))
        losses=[]; uniform=[]; disagreement=0
        for j,pos in enumerate(rec['positions']):
            q=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8')
            p=logits[pos].astype('f8'); assert np.isfinite(q).all()
            # Stable independent logaddexp reference; preserve the 1e-10 gate.
            q-=q.max(); p-=p.max(); q-=np.logaddexp.reduce(q); p-=np.logaddexp.reduce(p)
            losses.append(float(np.dot(np.exp(q),q-p)))
            uniform.append(math.log(V)+float(np.dot(np.exp(q),q)))
            disagreement+=int(np.argmax(p)!=np.argmax(q))
        value=float(np.mean(losses)); case_delta=max(case_delta,abs(value-row['KL']))
        label_delta=max(label_delta,float(np.max(np.abs(np.asarray(losses)-row['KL_per_label']))))
        close_enough(float(np.mean(uniform)),row['uniform_KL'])
        assert disagreement==row['disagreement'] and disagreement/m==row['disagreement_rate']
        defect,unions=routes(row['routes']['path'],n,row['mass_defect'])
        assert unions==row['unions']; mass_peak=max(mass_peak,defect)
        actual.append(dict(row,KL=value,KL_per_label=losses)); labels+=m; positions+=n
        del logits,teacher,q,p; gc.collect()
    assert case_delta<=1e-10 and label_delta<=1e-10
    return actual,dict(labels=labels,positions=positions,max_case_KL_delta=case_delta,
        max_label_KL_delta=label_delta,max_mass_defect=mass_peak)


def auxiliary(result, records):
    assert len(result['records'])==24 and result['heads_computed']==0 and result['parameter_ids_versions_unchanged']
    assert {r['id'] for r in result['records']}=={r['id'] for r in records.values() if r['split']=='DEV'}
    for row in result['records']:
        rec=records[row['id']]
        assert row['domain']==rec['domain'] and row['history']==len(rec['student_input_ids']) and len(row['sites'])==6
        close_enough(row['auxiliary'],sum(s['total'] for s in row['sites'])/6,1e-6)
        for s in row['sites']:
            assert all(math.isfinite(s[k]) and s[k]>=0 for k in ('total','mean','centered','centered_relative'))
            assert abs(s['total']-s['mean']-s['centered'])<=1e-4*max(s['total'],1e-24)
    return aggregate_equal(groups(result['records'],auxiliary=True),result['aggregate'])


def packed_fields(path):
    with Path(path).open('rb') as stream:
        header=stream.read(80)
        assert header[:8]==b'E4BPv001'
        assert struct.unpack_from('<16I',header,8)==(1,V,256,96,8,L,1024,48,4,128,5,E,128,K,1,110)
        assert struct.unpack_from('<Q',header,72)[0]==Path(path).stat().st_size
        result={}; cursor=80+110*FIELD.size
        for _ in range(110):
            name,dtype,rank,a,b,c,d,offset,size=FIELD.unpack(stream.read(FIELD.size))
            name=name.rstrip(b'\0').decode(); shape=(a,b,c,d)[:rank]
            assert name not in result and dtype in (1,2) and rank in (1,2,3) and all(shape)
            assert offset==cursor and size==math.prod(shape)*(4 if dtype==1 else 1)
            result[name]=dict(dtype=dtype,shape=shape,offset=offset,size=size); cursor+=size
        assert cursor==Path(path).stat().st_size
    return result


def master_name(name):
    if not name.startswith('layers.'): return name
    prefix,site,organ=name.split('.')
    base=f'{prefix}.{site}'
    if organ in ('gate','up','down'): return base+'.bank.'+organ
    if organ=='router': return base+'.bank.router.weight'
    if organ=='router_bias': return base+'.bank.router.bias'
    return base+'.organs.'+organ


def state_export(checkpoint, packed, witness_path, arm, milestone, binding, inherited_lineage):
    import numpy as np
    import torch
    state=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
    assert state['schema']=='ORIGINAL_JOINT_HISTORY_STATE_V1' and state['arm']==arm
    counter=milestone['counter'];dose=milestone['new_updates']
    assert state['updates']==counter==27+dose and state['new_updates']==dose
    assert not state['optimizer_partial_possible']
    assert state['source_state']==binding['source_state'] and state['source_packed']==binding['source_packed']
    assert state['binding_sha256']==sha(DOC/'original_joint_history_recovery_binding_20261009.json')
    assert state['inherited_lineage']==inherited_lineage
    assert state['completed_new_FIT_ids']==binding['FIT_order'][:dose]
    assert state['auxiliary_weight']==(0. if arm=='A' else 1.)
    assert state['geometry']==dict(D=256,N=96,DN=1024,DT=48,L=L,E=E,V=V,HID_E=128,K=K)
    assert state['CPU_rng'].dtype==torch.uint8 and state['CPU_rng'].ndim==1
    assert len(state['CUDA_rng'])==1 and state['CUDA_rng'][0].dtype==torch.uint8
    model=state['model']; optimizer=state['optimizer']; assert len(model)==92
    assert sum(p.numel() for p in model.values())==721008128
    assert sum(p.numel() for name,p in model.items() if name.endswith(('.bank.gate','.bank.up','.bank.down')))==679477248
    group,=optimizer['param_groups']; indices=group['params']
    assert len(indices)==92 and len(set(indices))==92 and set(indices)==set(optimizer['state'])
    assert group['lr']==5e-5 and group['betas']==(.9,.999) and group['eps']==1e-8
    assert group['weight_decay']==0. and group['foreach']==False and not group['amsgrad']
    for (name,p),index in zip(model.items(),indices):
        st=optimizer['state'][index]
        assert set(st)=={'step','exp_avg','exp_avg_sq'} and int(st['step'])==counter
        assert p.dtype==st['exp_avg'].dtype==st['exp_avg_sq'].dtype==torch.float32
        assert p.shape==st['exp_avg'].shape==st['exp_avg_sq'].shape
        for tensor in (p,st['exp_avg'],st['exp_avg_sq']):
            flat=tensor.reshape(-1)
            for first in range(0,flat.numel(),1<<20):assert torch.isfinite(flat[first:first+(1<<20)]).all(),name
        assert (st['exp_avg_sq']>=0).all(),name
    fields=packed_fields(packed); checked=0
    for name,f in fields.items():
        data=np.memmap(packed,dtype='<f4' if f['dtype']==1 else 'u1',mode='r',offset=f['offset'],shape=f['shape'])
        if name.endswith('_scale'): del data; continue # Paired with its code below.
        if name.endswith('_code'):
            base=name[:-5];p=model[master_name(base)]
            sf=fields[base+'_scale']; scales=np.memmap(packed,dtype='<f4',mode='r',offset=sf['offset'],shape=sf['shape'])
            assert np.isfinite(scales).all() and (scales>0).all()
            for first in range(0,E,32):
                last=min(first+32,E); w=p[first:last]
                scale=w.abs().mean(-1).clamp_min(1e-5)
                q=(w/scale.unsqueeze(-1)).round().clamp(-1,1).numpy().astype('i1')
                code=((q[...,0::2].astype('i2')+1)*3+q[...,1::2]+1).astype('u1')
                saved=data[first:last].transpose(0,2,1) if base.endswith('.down') else data[:,first*128:last*128].T.reshape(last-first,128,128)
                assert np.array_equal(code,saved),(name,first)
                assert np.array_equal(scale.numpy().view('<u4'),scales[first:last].view('<u4')),(base,first)
                del w,scale,q,code,saved
            assert data.min()>=0 and data.max()<=8; checked+=2;del scales
        else:
            p=model[master_name(name)];assert tuple(p.shape)==f['shape']
            assert np.array_equal(p.numpy().view('<u4'),data.view('<u4')),name;checked+=1
        del data,p
    assert checked==110
    values=[]
    for expert in (0,31,32,1023,1024,1151):
        for site in range(L):
            for organ,length in (('gate',256),('up',256),('down',128)):
                w=model[f'layers.{site}.bank.{organ}'][expert]
                scale=w.abs().mean(-1).clamp_min(1e-5)
                q=(w/scale.unsqueeze(-1)).round().clamp(-1,1).numpy().astype('i8')
                operand=np.arange(length,dtype='i8')
                operand=operand%127-63 if organ!='down' else (operand*7)%127-63
                values.append(q@operand)
    expected=np.concatenate(values).astype('<i4').tobytes()
    assert len(expected)==18432*4 and expected==Path(witness_path).read_bytes()
    result=dict(arm=arm,counter=counter,new_updates=dose,masters=92,coefficients=721008128,
        Adam_steps_all=counter,all_master_moments_finite=True,all110_export_fields_exact=True,
        independent_integer_coordinates=18432,source_lineage_exact=True,RNG_saved=True)
    del state,model,optimizer,st,tensor,flat,group,w,q,scale;gc.collect()
    return result,expected


def task_check(result,binding):
    import numpy as np
    from original_engine_chat import ChatTokenizer
    from chatbot_falcon_usability import normalized
    tokenizer=ChatTokenizer(binding['source']);cases={c['id']:c for c in binding['cases']}
    categories={c:0 for c in result['categories']};cached=[];greedy=0;mass_peak=0.
    assert len(result['records'])==16 and len(result['requests'])==17
    for row in result['requests']:
        if row['state_reset']:cached=[]
        assert row['query_ids'][:len(cached)]==cached and row['reused_prefix_ids']==len(cached)
        assert row['core_calls']==len(row['query_ids'])-len(cached)+len(row['generated_ids'])
        assert row['head_calls']==row['core_calls'] and row['cached_ids']==len(row['query_ids'])+len(row['generated_ids'])
        assert row['core_seconds'] is row['mlp_seconds'] is row['head_seconds'] is None
        cached=row['query_ids']+row['generated_ids']
        scores=np.fromfile(row['scores']['path'],dtype='<f4').reshape(-1,V)
        assert len(scores)==len(row['generated_ids']) and np.isfinite(scores).all()
        assert np.argmax(scores,-1).tolist()==row['generated_ids'];greedy+=len(scores)
        final=np.fromfile(row['final_head']['path'],dtype='<f4');assert final.shape==(V,) and np.isfinite(final).all()
        defect,_=routes(row['routes']['path'],row['core_calls'],row['mass_defect']);mass_peak=max(mass_peak,defect)
    for task in result['records']:
        case=cases[task['id']]; assert task['messages']==case['messages']
        assert task['request']['query_ids']==case['input_ids']==tokenizer.encode(case['messages'])
        ids=task['request']['generated_ids'];text=tokenizer.decode(ids)
        assert text==task['output_text'] and normalized(text)==task['normalized']
        assert task['correct']==(normalized(text)==case['expected']) and task['blank']==(not text.strip())
        assert task['special_leak']==any(i in set(binding['special_ids'])-{11,228} for i in ids)
        assert task['stop_policy_ok']==(bool(ids) and (ids[-1] in (11,228) or len(ids)==64))
        categories[case['category']]+=int(task['correct'])
    assert categories==result['categories'] and sum(categories.values())==result['correct']
    f=result['own_answer_followup']; parent=next(t for t in result['records'] if t['id']==binding['followup']['case_id'])
    messages=parent['messages']+[dict(role='assistant',content=parent['output_text']),dict(role='user',content=binding['followup']['prompt'])]
    assert f['messages']==messages and tokenizer.encode(messages)==f['request']['query_ids']
    assert f['output_text']==tokenizer.decode(f['request']['generated_ids']) and f['expected']==binding['followup']['expected']
    assert f['correct']==(normalized(f['output_text'])==f['expected'])
    cache=parent['request']['query_ids']+parent['request']['generated_ids'];compatible=f['request']['query_ids'][:len(cache)]==cache
    assert f['exact_cached_prefix_compatible']==compatible
    assert f['request']['reused_prefix_ids']==(len(cache) if compatible else 0)
    gates=dict(correct_min=sum(categories.values())>=12,every_category_min=all(v>=2 for v in categories.values()),
        blank_max=sum(t['blank'] for t in result['records'])<=1,no_special_leak=not any(t['special_leak'] for t in result['records']),
        stop_policy=all(t['stop_policy_ok'] for t in result['records']))
    assert gates==result['gates']; assert greedy==result['generated_ids']
    close_enough(greedy/sum(row['decode_seconds'] for row in result['requests']),result['raw_decode_ids_s'])
    close_enough(greedy/sum(row['pipe_request_seconds'] for row in result['requests']),result['raw_pipe_request_ids_s'])
    assert result['receipt']['exit_code']==0
    return dict(correct=result['correct'],categories=categories,generated_argmax_IDs=greedy,max_mass_defect=mass_peak,
        tasks_exact=True,own_answer_followup_exact=True,raw_rates_exact=True)


def main(args):
    import numpy as np
    import psutil
    import torch
    started=time.monotonic();psutil.Process().cpu_affinity(list(range(6)));torch.set_num_threads(6)
    result_path=args.result
    bpath=DOC/'original_joint_history_recovery_binding_20261009.json'
    r=json.loads(result_path.read_bytes());b=json.loads(bpath.read_bytes())
    finish_binding=json.loads(args.binding.read_bytes())
    terminal=json.loads(result_path.with_suffix('.terminal.json').read_bytes())
    assert terminal['exit_code']==0 and terminal['result_sha256']==sha(result_path)
    assert r['binding_sha256']==terminal['binding_sha256']==sha(args.binding)
    assert r['parent_binding_sha256']==sha(bpath)==finish_binding['parent_binding']['sha256']
    for pid in (terminal['worker_pid'],terminal['launcher_pid']):
        try:
            p=psutil.Process(pid)
            assert not (p.name().lower().startswith('python') and 'original_joint_history_recovery.py' in ' '.join(p.cmdline())),'family still live'
        except psutil.NoSuchProcess:pass
    assert r['schema']=='ORIGINAL_JOINT_HISTORY_RECOVERY_FINISH_RESULT_V1'
    for item in finish_binding['adjudication_inputs']+terminal['output_files']:assert extent(item['path'])==item,item['path']
    parent_failure=json.loads(Path(finish_binding['parent_failure']['path']).read_bytes())
    parent_fault=json.loads(Path(finish_binding['first_fault']['path']).read_bytes())
    assert parent_failure['exit_code']==1 and parent_fault['error']=="AssertionError('reserve/deadline')"
    assert parent_fault['counter']==parent_fault['durable']==51 and not parent_fault['optimizer_partial_possible']
    assert not r['parent_completion_reserve_gate'] and r['parent_first_fault']==finish_binding['first_fault']
    ns=Path(r['arms'][0]['final_checkpoint']['path']).parent;records={rec['id']:rec for rec in b['records']}
    assert len(records)==48 and len(b['FIT_order'])==24
    source=torch.load(b['source_state']['path'],map_location='cpu',weights_only=True,mmap=True)
    inherited={key:source[key] for key in ('source_state','ancestry','seed_ledger','binding_sha256','completed_new_FIT_ids')}
    assert source['updates']==27 and not source['optimizer_partial_possible'];del source;gc.collect()
    audit=[]; gradient_deltas=[];g=b['gates'];base=b['before_native']
    auxiliary_delta=auxiliary(r['before_auxiliary_DEV'],records)
    for arm in r['arms']:
        name=arm['name'];dose=arm['new_updates'];assert name in ('A','B') and dose in (6,12,24)
        assert arm['final_counter']==arm['durable_counter']==27+dose and arm['initial_counter']==27
        assert arm['complete_onepass']==(dose==24) and len(arm['updates'])==dose
        assert arm['auxiliary_weight']==(0. if name=='A' else 1.)
        assert sha(ns/(name+'.initial27.packed'))==b['source_packed']['sha256']
        assert sha(ns/(name+'.first_GPU_heads.f32'))==b['before_GPU_heads']['sha256']
        assert sha(ns/(name+'.first_GPU_routes'))==b['before_GPU_routes']['sha256']
        for index,row in enumerate(arm['updates']):
            rec=records[b['FIT_order'][index]]
            assert row['id']==rec['id'] and row['new_index']==index+1 and row['counter']==28+index
            assert row['history']==len(rec['student_input_ids']) and row['labels']==len(rec['positions'])
            assert row['auxiliary_weight']==arm['auxiliary_weight']
            close_enough(row['loss'],row['KL_before_surrogate']+row['auxiliary_weight']*row['auxiliary'],1e-6)
            assert row['gradient_norm']>0;close_enough(row['clip_coefficient'],min(1.,1./(row['gradient_norm']+1e-6)))
            assert len(row['sites'])==len(row['exposure'])==6
            for exposure in row['exposure']:assert 8<=exposure['union']<=E and exposure['selected_pairs']==row['history']*8
            assert row==json.loads((ns/(name+f'.update{row["counter"]:03d}.json')).read_bytes())
        milestones=arm['milestones'];assert [m['new_updates'] for m in milestones]==[x for x in (6,12,24) if x<=dose]
        milestone_audit=[];expected_stop=None
        for m in milestones:
            prefix=name+f'.actual{m["counter"]:03d}'
            st,expected=state_export(m['checkpoint']['path'],m['packed']['path'],ns/(prefix+'.expected.witness'),name,m,b,inherited)
            rows,metrics=native(m['native']['records'],records)
            delta=aggregate_equal(groups(rows),m['native']['aggregate'])
            for row in rows:
                assert (ns/(prefix+'.'+row['id']+'.witness')).read_bytes()==expected
                query=(ns/(prefix+'.'+row['id']+'.u32')).read_bytes();ids=records[row['id']]['student_input_ids']
                assert query==struct.pack('<I',len(ids))+np.asarray(ids,dtype='<u4').tobytes()
            now=m['native']['aggregate'];cfg=b['plateau']
            if now['case_KL']>cfg['regression_KL_multiple']*base['case_KL'] or now['case_disagreement']>base['case_disagreement']+cfg['regression_disagreement_add']:
                expected_stop='native_DEV_regression'
            elif m['new_updates']==12 and min(x['native']['aggregate']['case_KL'] for x in milestones[:2])>cfg['at12_KL_best_relative_to27']*base['case_KL'] and now['case_KL']>cfg['at12_KL_relative_to6']*milestones[0]['native']['aggregate']['case_KL']:
                expected_stop='native_DEV_plateau_at12'
            if expected_stop:assert m==milestones[-1]
            milestone_audit.append(dict(state=st,native=metrics,max_aggregate_delta=delta))
            print(json.dumps(dict(arm=name,audited_counter=m['counter'],seconds=time.monotonic()-started)),flush=True)
        assert arm['stop_reason']==expected_stop and arm['final_checkpoint']==milestones[-1]['checkpoint'] and arm['final_packed']==milestones[-1]['packed']
        auxiliary_delta=max(auxiliary_delta,auxiliary(arm['auxiliary_DEV'],records));task=task_check(arm['tasks'],b)
        dev=milestones[-1]['native']['aggregate']
        quality=dict(DEV_KL=dev['case_KL']<=g['native_DEV_KL'],DEV_disagreement=dev['case_disagreement']<=g['native_DEV_disagreement'],
            domain_KL=all(x['case_KL']<=g['domain_KL'] for x in dev['domains'].values()),domain_disagreement=all(x['case_disagreement']<=g['domain_disagreement'] for x in dev['domains'].values()),
            relative_DEV_KL=dev['case_KL']<=g['relative_DEV_KL']*base['case_KL'],relative_DEV_disagreement=dev['case_disagreement']<=g['relative_DEV_disagreement']*base['case_disagreement'])
        assert quality==arm['quality_gates'];audit.append(dict(arm=name,milestones=milestone_audit,tasks=task,quality_gates=quality))
    A,Barm=r['arms'];assert A['name']=='A' and Barm['name']=='B'
    for row in Barm['updates'][0]['initial_auxiliary_core_gradient_deltas']:
        filename=row['name'].replace('.','_')+'.gradient.f32'
        a=np.fromfile(ns/('A.'+filename),dtype='<f4').astype('f8');v=np.fromfile(ns/('B.'+filename),dtype='<f4').astype('f8')
        assert a.shape==v.shape and np.isfinite(a).all() and np.isfinite(v).all();delta=v-a
        actual=dict(auxiliary_gradient_difference_L2=float(np.linalg.norm(delta)),KL_gradient_L2=float(np.linalg.norm(a)),
            total_gradient_L2=float(np.linalg.norm(v)),dot_KL_auxiliary=float(np.dot(a,delta)))
        for key,value in actual.items():close_enough(value,row[key])
        assert actual['auxiliary_gradient_difference_L2']>g['initial_B_aux_core_gradient_delta_min']
        assert actual['auxiliary_gradient_difference_L2']>=g['initial_B_aux_core_gradient_relative_min']*max(actual['KL_gradient_L2'],actual['total_gradient_L2'],1e-24)
        gradient_deltas.append(dict(name=row['name'],**actual))
    assert len(gradient_deltas)==2
    aDEV=A['milestones'][-1]['native']['aggregate'];bDEV=Barm['milestones'][-1]['native']['aggregate']
    preference=dict(both_complete24=all(x['complete_onepass'] for x in r['arms']),native_KL=bDEV['case_KL']<=g['B_relative_to_A_KL']*aDEV['case_KL'],
        centered_auxiliary=Barm['auxiliary_DEV']['aggregate']['centered_relative']<=g['B_relative_to_A_centered']*A['auxiliary_DEV']['aggregate']['centered_relative'],
        disagreement=bDEV['case_disagreement']<=aDEV['case_disagreement']+g['B_disagreement_add'],
        domains=all(bDEV['domains'][d]['case_KL']<=g['B_domain_KL_multiple']*x['case_KL'] and bDEV['domains'][d]['case_disagreement']<=x['case_disagreement']+g['B_domain_disagreement_add'] for d,x in aDEV['domains'].items()),
        behavioral_support=Barm['tasks']['correct']>=max(g['B_behavioral_min'],A['tasks']['correct']+2))
    assert preference==r['B_preference_gates'] and r['B_preferred']==all(preference.values())
    assert r['decision']==('JOINT_HISTORY_B_SUPPORTED' if all(preference.values()) else 'JOINT_HISTORY_B_NOT_SUPPORTED')
    assert r['new_optimizer_updates']==0 and r['parent_optimizer_updates']==sum(x['new_updates'] for x in r['arms'])==48
    assert r['adopted_B_auxiliary_cases']==16 and r['candidate_auxiliary_calls']==8
    Baux=Barm['auxiliary_DEV'];assert Baux['adopted_cases']==16 and Baux['new_cases']==8
    for item in finish_binding['adopted_B_auxiliary']:
        row=json.loads(Path(item['path']).read_bytes());assert row==next(x for x in Baux['records'] if x['id']==row['id'])
    assert r['source_calls']==r['reserved_queries']==r['GPU_source_calls']==0
    assert r['quality_admission']==r['speed_admission']==r['useful_large_n_admission']==False
    assert r['physical_DRAM_bytes'] is None and not r['inherited_source_capture_resource_gate']
    assert all(child['exit_code']==0 for child in r['children'])
    assert len(r['children'])==1 and len(parent_fault['children'])==145
    assert all(child['exit_code']==0 for child in parent_fault['children'])
    peak=terminal['worker_OS_peak_through_exit']+r['max_direct_child_OS_peak']+terminal['launcher_OS_peak_snapshot']
    assert peak<=finish_binding['limits']['OS_bytes'] and terminal['elapsed_seconds']<=finish_binding['limits']['seconds']
    assert r['GPU_allocated_peak']<=finish_binding['limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=finish_binding['limits']['GPU_reserved_bytes']
    assert sum(item['bytes'] for item in terminal['output_files'])<=finish_binding['limits']['output_bytes']
    parent_peak=parent_failure['worker_OS_peak_through_exit']+parent_fault['max_direct_child_OS_peak']+parent_failure['launcher_OS_peak_snapshot']
    assert parent_peak<=b['limits']['OS_bytes'] and parent_failure['elapsed_seconds']<=b['limits']['seconds']
    assert parent_fault['GPU_allocated_peak']<=b['limits']['GPU_allocated_bytes'] and parent_fault['GPU_reserved_peak']<=b['limits']['GPU_reserved_bytes']
    assert sum(item['bytes'] for item in finish_binding['parent_outputs'])<=b['limits']['output_bytes']
    result=dict(schema='ORIGINAL_JOINT_HISTORY_RECOVERY_STORED_ADJUDICATION_V1',result=extent(result_path),binding=extent(bpath),
        audit_code=extent(Path(__file__)),completion_binding=extent(args.binding),input_hashes=len(finish_binding['adjudication_inputs']),
        parent_input_hashes=len(b['inputs']),parent_output_hashes=len(finish_binding['parent_outputs']),output_hashes=len(terminal['output_files']),arms=audit,
        auxiliary_summary_max_aggregate_delta=auxiliary_delta,initial_core_gradient_deltas=gradient_deltas,
        first_GPU_heads_routes_initial_packed_exact=True,worker_initial_RAM_92_master_moments_RNG_witness_retained=True,
        source_lineage_exact=True,B_preference_gates=preference,decision=r['decision'],parent_optimizer_updates=48,
        parent_completion_reserve_gate=False,adopted_B_auxiliary_cases=16,new_B_auxiliary_cases=8,
        conservative_held_OS_union=peak,parent_conservative_held_OS_union=parent_peak,
        model_calls=0,source_calls=0,GPU_calls=0,optimizer_updates=0,seconds=time.monotonic()-started,
        limits='Stored auxiliary summaries independently reduced; no saved intermediate candidate activations to recompute auxiliary loss. '
            'Initial92 RAM restoration witness belongs to worker; initial packs/heads/routes and all durable states checked here. '
            'Inherited source capture/numerical/full runtime DLL/DRAM/fresh quality gaps retained. Stored audit is outside held experiment family.')
    out=args.out;write(out,result)
    print(json.dumps(dict(audit=str(out),sha256=sha(out),decision=result['decision'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
