"""Stored-only independent adjudication; no compiler/model/GPU/update replay."""
import json
import math
from pathlib import Path
import re
import struct
import sys
import time
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from original_packed_capacity import SITE,sha,write,extent,FUNCTIONS,OBSERVER
sys.path.insert(0,str(SITE))


def main():
    import numpy as np
    import psutil
    from original_engine_chat import ChatTokenizer
    from chatbot_falcon_usability import normalized
    start=time.monotonic();psutil.Process().cpu_affinity([1])
    result_path=DOC/'original_engine_baseline_result_20261009.json';r=json.loads(result_path.read_bytes())
    terminal=json.loads(result_path.with_suffix('.terminal.json').read_bytes())
    bpath=DOC/'original_engine_baseline_binding_20261009.json';b=json.loads(bpath.read_bytes())
    assert terminal['exit_code']==0 and terminal['result_sha256']==sha(result_path)
    assert r['binding_sha256']==terminal['binding_sha256']==sha(bpath)
    checked=[]
    for item in b['inputs']+terminal['output_files']:
        assert extent(item['path'])==item,item['path'];checked.append(item['path'])
    ns=Path(r['executable']['path']).parent
    source=(ROOT/'benchmarks/phase60/engine.c').read_text()
    header=(ns/'original_packed_bodies.h').read_text()
    def body(text,name):
        m=re.search(r'^static[^\n]*\b'+name+r'\(',text,re.M);assert m,name
        first=text.index('{',m.start());last=first+1;depth=1
        while depth:depth+=(text[last]=='{')-(text[last]=='}');last+=1
        return text[m.start():last]
    for name in FUNCTIONS:
        original=body(source,name);compiled=body(header,name)
        assert (compiled.replace(OBSERVER,'') if name=='mlp_moe' else compiled)==original,name
    wrapper=(B/'original_packed_capacity.c').read_text()
    for name,value in (('DN',512),('DTR',16)):
        old=f'#define {name} {value}';wrapper=wrapper.replace(old,f'#ifndef {name}\n{old}\n#endif')
    assert (ns/'original_wide_prefix.c').read_bytes()==wrapper.encode('utf8')
    assert (ns/'original_engine_chat_main.c').read_bytes()==(B/'original_engine_chat_main.c').read_bytes()
    packet=Path(b['adopted']['requests.u32']['path']).read_bytes();cursor=4
    assert struct.unpack_from('<I',packet)[0]==3
    for rec in b['adopted_sequences']:
        n=struct.unpack_from('<I',packet,cursor)[0];cursor+=4
        ids=list(struct.unpack_from('<'+str(n)+'I',packet,cursor));cursor+=n*4
        assert ids==rec['student_input_ids']
    assert cursor==len(packet)
    records={c['id']:c for c in b['records']};max_KL=0.;max_label_KL=0.;max_defect=0.;label_count=0
    actual=[]
    for row in r['native_records']:
        rec=records[row['id']];n=len(rec['student_input_ids']);offset=row['row_offset']
        size=Path(row['logits']['path']).stat().st_size//(65537*4)
        logits=np.memmap(row['logits']['path'],dtype='<f4',mode='r',shape=(size,65537))[offset:offset+n]
        teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(len(rec['positions']),65537))
        losses=[];dis=0
        for j,pos in enumerate(rec['positions']):
            q=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8');p=logits[pos].astype('f8')
            assert np.isfinite(q).all() and np.isfinite(p).all()
            # Independent log-partition algorithm rather than max/exp/sum/log.
            q=q-np.logaddexp.reduce(q);p=p-np.logaddexp.reduce(p)
            losses.append(float(np.dot(np.exp(q),q-p)));dis+=int(np.argmax(p)!=np.argmax(q))
        loss=float(np.mean(losses));max_KL=max(max_KL,abs(loss-row['KL']))
        max_label_KL=max(max_label_KL,float(np.max(np.abs(np.asarray(losses)-row['KL_per_label']))))
        assert dis==row['disagreement'];label_count+=len(losses)
        route=np.memmap(row['routes']['path'],dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))]),mode='r',shape=(size,6))[offset:offset+n]
        ids=route['ids'];mass=route['mass'];assert ids.min()>=0 and ids.max()<1152
        assert (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all() and np.isfinite(mass).all() and (mass>=0).all()
        defect=float(np.max(np.abs(mass.astype('f8').sum(-1)-1)));max_defect=max(max_defect,defect)
        assert defect==row['mass_defect'] and defect<=1e-6
        assert [int(np.unique(ids[:,i]).size) for i in range(6)]==row['unions']
        actual.append(dict(row,KL=loss,KL_per_label=losses))
    assert max_KL<=1e-10 and max_label_KL<=1e-10
    aggregate_delta=0.
    for split in ('FIT','DEV'):
        for domain in [None,*sorted({x['domain'] for x in actual})]:
            rows=[x for x in actual if x['split']==split and (domain is None or x['domain']==domain)]
            total=sum(x['labels'] for x in rows)
            values=dict(case_KL=sum(x['KL'] for x in rows)/len(rows),
                label_KL=sum(sum(x['KL_per_label']) for x in rows)/total,
                case_disagreement=sum(x['disagreement_rate'] for x in rows)/len(rows),
                label_disagreement=sum(x['disagreement'] for x in rows)/total)
            saved=r['native_aggregate'][split] if domain is None else r['native_aggregate'][split]['domains'][domain]
            assert saved['cases']==len(rows) and saved['labels']==total
            aggregate_delta=max(aggregate_delta,max(abs(v-saved[k]) for k,v in values.items()))
    assert aggregate_delta<=1e-10
    tokenizer=ChatTokenizer(b['source']);cases={c['id']:c for c in b['cases']}
    category={c:0 for c in r['category_correct']};greedy_count=0;cached=[];max_stream_mass=0.
    for row in r['requests']:
        if row['state_reset']:cached=[]
        assert row['query_ids'][:len(cached)]==cached and row['reused_prefix_ids']==len(cached)
        assert row['core_calls']==len(row['query_ids'])-len(cached)+len(row['generated_ids'])
        assert row['head_calls']==row['core_calls'] and row['cached_ids']==len(row['query_ids'])+len(row['generated_ids'])
        assert row['core_seconds'] is row['mlp_seconds'] is row['head_seconds'] is None
        cached=row['query_ids']+row['generated_ids']
        head=np.fromfile(row['final_head']['path'],dtype='<f4');assert head.shape==(65537,) and np.isfinite(head).all()
        if row['generated_ids']:
            scores=np.fromfile(row['scores']['path'],dtype='<f4').reshape(-1,65537)
            assert np.isfinite(scores).all() and np.argmax(scores,-1).tolist()==row['generated_ids'];greedy_count+=len(scores)
        routes=np.fromfile(row['routes']['path'],dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))])).reshape(-1,6)
        assert len(routes)==row['core_calls']
        if len(routes):
            assert routes['ids'].min()>=0 and routes['ids'].max()<1152
            assert (np.diff(np.sort(routes['ids'],axis=-1),axis=-1)>0).all()
            assert np.isfinite(routes['mass']).all() and (routes['mass']>=0).all()
            defect=float(np.max(np.abs(routes['mass'].astype('f8').sum(-1)-1)));assert defect==row['mass_defect']
            max_stream_mass=max(max_stream_mass,defect)
    assert max_stream_mass<=1e-6
    for task in r['tasks']:
        case=cases[task['id']];source_case=json.loads(Path(case['source_record']['path']).read_bytes())
        assert task['messages']==case['messages']==source_case['messages']
        assert task['request']['query_ids']==tokenizer.encode(task['messages'])==source_case['input_ids']
        ids=task['request']['generated_ids'];text=tokenizer.decode(ids)
        assert text==task['output_text'] and normalized(text)==task['normalized']
        assert task['correct']==(normalized(text)==case['expected'])
        assert task['blank']==(not text.strip())
        assert task['special_leak']==any(i in set(b['special_ids'])-{11,228} for i in ids)
        assert task['stop_policy_ok']==(bool(ids) and (ids[-1] in (11,228) or len(ids)==64))
        category[case['category']]+=int(task['correct'])
    assert category==r['category_correct'] and sum(category.values())==r['task_correct']
    f=r['own_answer_followup'];assert tokenizer.encode(f['messages'])==f['request']['query_ids']
    parent_task=next(t for t in r['tasks'] if t['id']==b['followup']['case_id'])
    assert f['messages'][-2]==dict(role='assistant',content=parent_task['output_text'])
    assert f['correct']==(normalized(tokenizer.decode(f['request']['generated_ids']))==f['expected'])
    first=r['requests'][0];partial=r['requests'][1];tail=r['requests'][2];reuse=r['requests'][3]
    rec=records[b['qualification_id']];old=next(x for x in b['adopted_metrics'] if x['id']==rec['id'])
    retained=np.memmap(b['adopted']['after.f32']['path'],dtype='<f4',mode='r',shape=(3352,65537))
    assert Path(first['final_head']['path']).read_bytes()==retained[old['row_offset']+len(rec['student_input_ids'])-1].tobytes()
    assert Path(partial['final_head']['path']).read_bytes()==retained[old['row_offset']+b['qualification_split']-1].tobytes()
    assert Path(first['final_head']['path']).read_bytes()==Path(tail['final_head']['path']).read_bytes()==Path(reuse['final_head']['path']).read_bytes()
    route_bytes=Path(b['adopted']['after.routes']['path']).read_bytes();offset=old['row_offset']*6*64
    oldroute=route_bytes[offset:offset+len(rec['student_input_ids'])*6*64]
    assert Path(first['routes']['path']).read_bytes()==oldroute==Path(partial['routes']['path']).read_bytes()+Path(tail['routes']['path']).read_bytes()
    expected=Path(b['adopted']['expected_updated.witness']['path']).read_bytes()
    for path in ns.glob('*.witness'):assert path.read_bytes()==expected
    generated=[x for x in r['requests'] if x['generated_ids']];total=sum(len(x['generated_ids']) for x in generated)
    assert greedy_count==total==r['raw_rates']['generated_ids']
    for field,rate in (('decode_seconds','raw_decode_ids_s'),('request_seconds','raw_request_ids_s'),('pipe_request_seconds','raw_pipe_request_ids_s')):
        elapsed=sum(x[field] for x in generated);assert elapsed==r['raw_rates'][field]
        assert total/elapsed==r['raw_rates'][rate]
    gates=dict(correct_min=sum(category.values())>=12,every_category_min=all(v>=2 for v in category.values()),
        blank_max=sum(x['blank'] for x in r['tasks'])<=1,no_special_leak=not any(x['special_leak'] for x in r['tasks']),
        stop_policy=all(x['stop_policy_ok'] for x in r['tasks']))
    assert gates==r['task_gates']
    dev=r['native_aggregate']['DEV'];native=dict(DEV_case_KL=dev['case_KL']<=1.,DEV_case_disagreement=dev['case_disagreement']<=.20,
        all_domain_KL=all(x['case_KL']<=2. for x in dev['domains'].values()),
        all_domain_disagreement=all(x['case_disagreement']<=.35 for x in dev['domains'].values()))
    assert native==r['native_absolute_gates'] and r['quality_admission']==r['speed_admission']==r['useful_large_n_admission']==False
    assert r['source_calls']==r['optimizer_updates']==r['GPU_calls']==r['reserved_queries']==0
    assert all(x['exit_code']==0 for x in r['children']) and len(r['children'])==47
    peak=terminal['worker_OS_peak_through_exit']+r['max_direct_child_OS_peak']+terminal['launcher_OS_peak_snapshot']
    assert peak<=b['limits']['OS_bytes'] and terminal['elapsed_seconds']<=b['limits']['seconds']
    result=dict(schema='ORIGINAL_ENGINE_BASELINE_STORED_ADJUDICATION_V1',result=extent(result_path),
        binding=extent(bpath),input_hashes=len(b['inputs']),output_hashes=len(terminal['output_files']),
        original_computational_bodies=20,adopted_request_identity=True,independent_F64_labels=label_count,
        max_case_KL_delta=max_KL,max_label_KL_delta=max_label_KL,max_aggregate_delta=aggregate_delta,
        all_integer_witnesses_exact=True,max_native_mass_defect=max_defect,max_stream_mass_defect=max_stream_mass,
        all_prefix_heads_routes_bit_exact=True,all_generated_ID_argmax_exact=greedy_count,
        tasks_exact=True,followup_candidate_answer_exact=True,task_correct=r['task_correct'],categories=category,
        raw_rates_exact=True,conservative_held_OS_union=peak,nested_linker_peak_missing=True,
        source_calls=0,model_calls=0,GPU_calls=0,optimizer_updates=0,seconds=time.monotonic()-start,
        scope='Stored-only independent logaddexp F64 adjudication;not a held experiment family or new inference.')
    path=DOC/'original_engine_baseline_stored_adjudication_20261009.json';write(path,result)
    print(json.dumps(dict(audit=str(path),sha256=sha(path),seconds=result['seconds'],max_KL_delta=max_KL,task_correct=r['task_correct'])),flush=True)


if __name__=='__main__':main()
