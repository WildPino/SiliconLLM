"""Native assessments for new actual checkpoints through the qualified C ABI."""
import json
import os
from pathlib import Path
import struct
import time
from original_packed_capacity import raw,extent,write
from original_engine_baseline import metrics,aggregate
from original_engine_chat import OriginalEngineClient,ChatTokenizer
from chatbot_falcon_usability import normalized


def native_dev(binding,directory,exe,packed,expected,prefix,run,event):
    import numpy as np
    rows=[]
    for rec in binding['records']:
        if rec['split']!='DEV':continue
        name=prefix+'.'+rec['id'];query=directory/(name+'.u32')
        raw(query,struct.pack('<I',len(rec['student_input_ids']))+np.asarray(rec['student_input_ids'],dtype='<u4').tobytes())
        paths=[directory/(name+s) for s in ('.f32','.routes','.witness','.native.json')]
        run([exe,'--prefix',packed,query,*paths],name)
        assert paths[2].read_bytes()==expected,'updated ternary integer witness'
        row=metrics(rec,paths[0],paths[1]);rows.append(row);write(directory/(name+'.metrics.json'),row)
        event(operation='native_DEV',prefix=prefix,id=rec['id'],cases=len(rows),KL=row['KL'])
    assert len(rows)==24
    # aggregate expects both splits. Native-only DEV uses a separate direct reduction.
    def group(values):
        labels=sum(r['labels'] for r in values)
        return dict(cases=len(values),labels=labels,case_KL=sum(r['KL'] for r in values)/len(values),
            label_KL=sum(sum(r['KL_per_label']) for r in values)/labels,
            case_disagreement=sum(r['disagreement_rate'] for r in values)/len(values),
            label_disagreement=sum(r['disagreement'] for r in values)/labels)
    result=dict(records=rows,aggregate=dict(**group(rows),domains={d:group([r for r in rows if r['domain']==d]) for d in sorted({r['domain'] for r in rows})}))
    write(directory/(prefix+'.native_DEV.json'),result);return result


def tasks(binding,directory,exe,packed,prefix,reader,observe_peak,guard,event):
    import numpy as np
    import psutil
    tokenizer=ChatTokenizer(binding['source']);rows=[];requests=[];followup=None
    os.environ['SILICON_CHAT_SECONDS']=str(binding['limits']['stream_seconds'])
    os.environ['SILICON_CHAT_SCORE_PREFIX']=str(directory/(prefix+'.stream'))
    os.environ['SILICON_CHAT_ROUTE_PREFIX']=str(directory/(prefix+'.stream'))
    started=time.monotonic();client=None;created=None;peak=0
    with (directory/(prefix+'.stream.log')).open('xb') as log:
        try:
            client=OriginalEngineClient(exe,packed,log,timeout=binding['limits']['child_seconds'])
            created=psutil.Process(client.process.pid).create_time()
            def request(ids,reset,label):
                nonlocal peak
                record,head=client.request(ids,64,reset);peak=max(peak,reader(client.process));observe_peak(peak);guard()
                index=len(requests);name=f'{prefix}.stream.{index:04d}'
                raw(directory/(name+'.final.f32'),head);assert np.isfinite(np.frombuffer(head,dtype='<f4')).all()
                record.update(index=index,label=label,query_ids=ids,final_head=extent(directory/(name+'.final.f32')),
                    scores=extent(directory/(name+'.scores.f32')),routes=extent(directory/(name+'.routes')))
                scores=np.fromfile(record['scores']['path'],dtype='<f4').reshape(-1,65537)
                assert len(scores)==len(record['generated_ids']) and np.isfinite(scores).all()
                assert np.argmax(scores,-1).tolist()==record['generated_ids']
                routes=np.fromfile(record['routes']['path'],dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))])).reshape(-1,6)
                assert len(routes)==record['core_calls'] and routes['ids'].min()>=0 and routes['ids'].max()<1152
                assert (np.diff(np.sort(routes['ids'],axis=-1),axis=-1)>0).all()
                mass=routes['mass'];assert np.isfinite(mass).all() and (mass>=0).all()
                record['mass_defect']=float(np.max(np.abs(mass.astype('f8').sum(-1)-1)));assert record['mass_defect']<=1e-6
                requests.append(record);write(directory/(name+'.json'),record);return record
            for case in binding['cases']:
                record=request(case['input_ids'],True,case['id']);ids=record['generated_ids'];text=tokenizer.decode(ids)
                row=dict(id=case['id'],category=case['category'],expected=case['expected'],messages=case['messages'],request=record,
                    output_text=text,normalized=normalized(text),correct=normalized(text)==case['expected'],blank=not text.strip(),
                    special_leak=any(i in set(binding['special_ids'])-{11,228} for i in ids),
                    stop_policy_ok=bool(ids) and (ids[-1] in (11,228) or len(ids)==64))
                rows.append(row);write(directory/(prefix+'.'+case['id']+'.task.json'),row)
                event(operation='task',prefix=prefix,id=case['id'],answer=text,correct=row['correct'])
                if case['id']==binding['followup']['case_id']:
                    messages=case['messages']+[dict(role='assistant',content=text),dict(role='user',content=binding['followup']['prompt'])]
                    query=tokenizer.encode(messages);response=request(query,False,'own_answer_followup')
                    cached=case['input_ids']+ids;compatible=query[:len(cached)]==cached
                    assert response['reused_prefix_ids']==(len(cached) if compatible else 0)
                    answer=tokenizer.decode(response['generated_ids'])
                    followup=dict(messages=messages,request=response,output_text=answer,expected=binding['followup']['expected'],
                        correct=normalized(answer)==binding['followup']['expected'],exact_cached_prefix_compatible=compatible)
                    write(directory/(prefix+'.own_answer_followup.json'),followup)
            client.close()
        finally:
            if client:
                client.abort();peak=max(peak,reader(client.process));observe_peak(peak)
                receipt=dict(command=[str(exe),'--stream',str(packed)],pid=client.process.pid,creation_time=created,
                    exit_code=client.process.returncode,held_OS_peak=peak,seconds=time.monotonic()-started)
                write(directory/(prefix+'.stream.receipt.json'),receipt)
    categories={c:sum(r['correct'] for r in rows if r['category']==c) for c in sorted({r['category'] for r in rows})}
    gates=dict(correct_min=sum(categories.values())>=12,every_category_min=all(v>=2 for v in categories.values()),
        blank_max=sum(r['blank'] for r in rows)<=1,no_special_leak=not any(r['special_leak'] for r in rows),stop_policy=all(r['stop_policy_ok'] for r in rows))
    total=sum(len(r['generated_ids']) for r in requests)
    result=dict(records=rows,correct=sum(categories.values()),categories=categories,gates=gates,requests=requests,
        own_answer_followup=followup,receipt=receipt,generated_ids=total,
        raw_decode_ids_s=total/sum(r['decode_seconds'] for r in requests),
        raw_pipe_request_ids_s=total/sum(r['pipe_request_seconds'] for r in requests),
        scope='Same64/EOS/canonical source screen;own-answer followup separate. Raw incorrect-content rates,trace writes included in pipe time.')
    write(directory/(prefix+'.tasks.json'),result);return result
