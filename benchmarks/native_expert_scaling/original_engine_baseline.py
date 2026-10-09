"""Actual27 original-kernel native baseline and persistent own-history chat.

No optimizer, source-model inference, GPU or RESERVED observation. Three native
histories are adopted from retained actual27; the other45 are observed once.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from original_packed_capacity import SITE,sha,write,raw,extent,extract
from original_falcon_whole_recovery import check_inputs,memory_reader
sys.path.insert(0,str(SITE))
V=65537;L=6;E=1152


def metrics(rec,logits_path,routes_path,offset=0,adopted=False):
    import numpy as np
    n=len(rec['student_input_ids']);labels=len(rec['positions'])
    count=Path(logits_path).stat().st_size//(V*4)
    assert Path(logits_path).stat().st_size==count*V*4 and Path(routes_path).stat().st_size==count*L*64
    assert offset+n<=count
    logits=np.memmap(logits_path,dtype='<f4',mode='r',shape=(count,V))[offset:offset+n]
    for first in range(0,n,64):assert np.isfinite(logits[first:first+64]).all()
    teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(labels,V))
    losses=[];uniform=[];dis=0
    for j,pos in enumerate(rec['positions']):
        q=(teacher[j].astype('<u4')<<16).view('<f4').astype('f8');q-=q.max();q-=math.log(float(np.exp(q).sum()))
        p=logits[pos].astype('f8');p-=p.max();p-=math.log(float(np.exp(p).sum()))
        losses.append(float(np.sum(np.exp(q)*(q-p))))
        uniform.append(math.log(V)+float(np.sum(np.exp(q)*q)))
        dis+=int(np.argmax(p)!=np.argmax(q))
    dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))])
    routes=np.memmap(routes_path,dtype=dtype,mode='r',shape=(count,L))[offset:offset+n]
    ids=routes['ids'];mass=routes['mass']
    assert ids.min()>=0 and ids.max()<E and (np.diff(np.sort(ids,axis=-1),axis=-1)>0).all()
    assert np.isfinite(mass).all() and (mass>=0).all()
    defect=float(np.max(np.abs(mass.astype('f8').sum(-1)-1)));assert defect<=1e-6
    return dict(id=rec['id'],split=rec['split'],domain=rec['domain'],history=n,labels=labels,
        KL=float(np.mean(losses)),KL_per_label=losses,uniform_KL=float(np.mean(uniform)),
        disagreement=dis,disagreement_rate=dis/labels,mass_defect=defect,
        unions=[int(np.unique(ids[:,site]).size) for site in range(L)],row_offset=offset,adopted=adopted,
        logits=extent(logits_path),routes=extent(routes_path))


def aggregate(rows):
    def group(values):
        labels=sum(r['labels'] for r in values)
        return dict(cases=len(values),labels=labels,case_KL=sum(r['KL'] for r in values)/len(values),
            label_KL=sum(sum(r['KL_per_label']) for r in values)/labels,
            case_disagreement=sum(r['disagreement_rate'] for r in values)/len(values),
            label_disagreement=sum(r['disagreement'] for r in values)/labels)
    return {split:dict(**group([r for r in rows if r['split']==split]),
        domains={d:group([r for r in rows if r['split']==split and r['domain']==d])
            for d in sorted({r['domain'] for r in rows})}) for split in ('FIT','DEV')}


def bind(a):
    import shutil
    import numpy as np
    import numpy._core._multiarray_umath as ext
    from original_engine_chat import ChatTokenizer
    parent=DOC/'original_delta_seed_finish_result_20261009.json';pr=json.loads(parent.read_bytes())
    terminal=parent.with_suffix('.terminal.json');pt=json.loads(terminal.read_bytes())
    assert pt['exit_code']==0 and pt['result_sha256']==sha(parent) and pr['final_counter']==pr['durable_counter']==27
    old_binding=DOC/'original_delta_seed_finish_binding_20261009.json';old=json.loads(old_binding.read_bytes())
    corpus=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    records=json.loads(corpus.read_bytes())['records'];assert len(records)==48
    indexed={r['id']:r for r in records};assert len(indexed)==48
    for row in old['sequences']:assert row==indexed[row['id']],row['id']
    for split in ('FIT','DEV'):
        values=[r for r in records if r['split']==split]
        assert len(values)==24 and len({r['domain'] for r in values})==12
    assert sum(len(r['student_input_ids']) for r in records)==22547
    for r in records:
        assert r['student_input_ids']==r['input_ids']+r['output_ids'][:-1]
        assert r['logits']['shape']==[len(r['positions']),V]
    ns=ROOT/'results/native_expert_scaling/original_delta_seed_finish_20261009'
    outputs={Path(i['path']).name:i for i in pt['output_files']}
    adopted={name:outputs[name] for name in ('after.f32','after.routes','after.witness','expected_updated.witness','requests.u32')}
    for item in adopted.values():assert extent(item['path'])==item
    assert adopted['after.witness']['sha256']==adopted['expected_updated.witness']['sha256']
    source=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    cases_path=B/'chatbot_falcon_usability_cases_v1.json';cases=json.loads(cases_path.read_bytes())['cases']
    tokenizer=ChatTokenizer(source)
    source_result=DOC/'chatbot_hybrid_usability_result_repair1_20261008.json'
    sr=json.loads(source_result.read_bytes());st=source_result.with_suffix('.terminal.json');ss=json.loads(st.read_bytes())
    assert ss['exit_code']==0 and ss['result_sha256']==sha(source_result)
    assert sr['correct']==14 and all(sr['quality_gates'].values())
    source_ns=ROOT/'results/native_expert_scaling/falcon_1p5b_usability_repair1_20261008'
    source_cases=[]
    for case in cases:
        messages=case.get('history',[])+[dict(role='user',content=case['prompt'])]
        ids=tokenizer.encode(messages);saved=source_ns/(case['id']+'.json');s=json.loads(saved.read_bytes())
        assert s['input_ids']==ids and s['messages']==messages and len(ids)+64<=256
        source_cases.append(dict(**case,messages=messages,input_ids=ids,source_record=extent(saved)))
    compiler=Path(json.loads((DOC/'chatbot_hybrid_engine_probe_binding_20261009.json').read_bytes())['compiler'])
    files=[Path(__file__),B/'original_engine_chat_main.c',B/'original_engine_chat.py',B/'original_packed_capacity.c',
        B/'original_packed_capacity.py',B/'chatbot_hybrid_engine_chat.py',B/'original_falcon_whole_recovery.py',
        B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',DOC/'ORIGINAL_ENGINE_BASELINE_PROTOCOL_20261009.md',
        parent,terminal,old_binding,corpus,cases_path,source_result,st,Path(pr['packed']['path']),Path(sys.executable),compiler,
        SITE/'numpy/__init__.py',Path(ext.__file__),SITE/'psutil/__init__.py',SITE/'tokenizers/__init__.py',
        SITE/'tokenizers/tokenizers.pyd',SITE/'jinja2/__init__.py',SITE/'jinja2/sandbox.py',SITE/'jinja2/environment.py']
    files += [Path(i['path']) for i in adopted.values()]+[Path(r['logits']['path']) for r in records]
    files += [Path(c['source_record']['path']) for c in source_cases]
    files += [source/n for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json')]
    files += [compiler.parent/n for n in ('clang-21.exe','ld.lld.exe','lld.exe','libclang-cpp.dll','libLLVM.dll') if (compiler.parent/n).exists()]
    files += [ROOT/n for n in ('benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    assert shutil.disk_usage(ROOT).free>=12<<30
    binding=dict(schema='ORIGINAL_ENGINE_BASELINE_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
        compiler=str(compiler.resolve()),packed=pr['packed'],source=str(source),source_result=extent(source_result),
        records=records,cases=source_cases,adopted=adopted,adopted_metrics=pr['after_native'],adopted_sequences=old['sequences'],
        qualification_id=old['sequences'][0]['id'],qualification_split=178,max_new=64,
        followup=dict(case_id='history_pet',prompt="What is my pet's name? Answer with the name only.",expected='Milo'),
        special_ids=[t['id'] for t in json.loads((source/'tokenizer.json').read_bytes())['added_tokens'] if t['special']],
        gates=dict(DEV_case_KL=1.,DEV_case_disagreement=.20,domain_case_KL=2.,domain_case_disagreement=.35,
            task_correct_min=12,category_correct_min=2,blank_max=1,no_special_leak=True,stop_policy=True,
            prefix_logits_routes_bit_exact=True,integer_witness_bit_exact=True,mass_defect=1e-6),
        limits=dict(seconds=1200,reserve_seconds=60,OS_bytes=4<<30,output_bytes=8<<30,child_seconds=100,stream_seconds=300,log_bytes=4<<20),
        inputs=[extent(p) for p in dict.fromkeys(files)],
        scope='Actual27/fullV original20 bodies, expfast1, CPU0;19 exact/one pure route observer. Prefix heads on ALL inputs;'
            '45 new canonical histories +3 adopted, four stream qualifications +16 tasks +one own-answer followup.'
            'Score/route memcpy in timers, trace writes outside C timers but inside Python pipe time. '
            'Principal runtimes/compilers hashed, nested linker peak not separately held;no full DLL tree.' )
    write(a.out,binding);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(binding['inputs']))),flush=True)


def worker(a):
    import numpy as np
    import psutil
    from original_engine_chat import OriginalEngineClient,ChatTokenizer
    from chatbot_falcon_usability import normalized
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert (np.__version__,psutil.__version__)==('2.4.6','7.2.2')
    proc=psutil.Process();proc.cpu_affinity([1]);a.directory.mkdir(exist_ok=False)
    reader=memory_reader();child_peak=0;children=[];rows=[];tasks=[];requests=[];client=None;stage='startup'
    def guard():
        lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds'],'worker reserve'
        assert proc.memory_info().peak_wset+child_peak<=lim['OS_bytes'],'worker/direct child OS cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes'],'output cap'
    def event(**values):
        guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**values)),flush=True)
    def run(command,name):
        nonlocal child_peak
        began=time.monotonic();peak=0
        with (a.directory/(name+'.log')).open('xb') as f:
            child=subprocess.Popen([str(x) for x in command],stdout=f,stderr=subprocess.STDOUT,creationflags=8)
            created=psutil.Process(child.pid).create_time()
            try:
                while child.poll() is None:
                    peak=max(peak,reader(child));child_peak=max(child_peak,peak);guard()
                    assert time.monotonic()-began<=b['limits']['child_seconds'],'child deadline';time.sleep(.05)
                peak=max(peak,reader(child));child_peak=max(child_peak,peak)
            except BaseException:
                if child.poll() is None:child.kill();child.wait()
                peak=max(peak,reader(child));child_peak=max(child_peak,peak);raise
            finally:
                rec=dict(command=[str(x) for x in command],pid=child.pid,creation_time=created,exit_code=child.returncode,
                    held_OS_peak=peak,seconds=time.monotonic()-began)
                children.append(rec);write(a.directory/(name+'.receipt.json'),rec)
        assert child.returncode==0,(name,child.returncode);event(child=name,seconds_child=rec['seconds']);return rec
    try:
        stage='compile';body=extract(a.directory);wrapper=(B/'original_packed_capacity.c').read_text()
        for name,value in (('DN',512),('DTR',16)):
            old=f'#define {name} {value}';assert wrapper.count(old)==1;wrapper=wrapper.replace(old,f'#ifndef {name}\n{old}\n#endif')
        raw(a.directory/'original_wide_prefix.c',wrapper.encode('utf8'))
        raw(a.directory/'original_engine_chat_main.c',(B/'original_engine_chat_main.c').read_bytes())
        exe=a.directory/'original_engine_chat.exe'
        run([b['compiler'],'-O3','-mavx2','-mfma','-march=znver2','-DDN=1024','-DDTR=48',
            a.directory/'original_engine_chat_main.c','-I',a.directory,'-o',exe,'-lm'],'compile')
        stage='native_all48';adopted_index={r['id']:r for r in b['adopted_metrics']}
        expected=Path(b['adopted']['expected_updated.witness']['path']).read_bytes();assert len(expected)==18432*4
        for rec in b['records']:
            prefix=rec['id'];saved=adopted_index.get(prefix)
            if saved:
                row=metrics(rec,b['adopted']['after.f32']['path'],b['adopted']['after.routes']['path'],saved['row_offset'],True)
                assert abs(row['KL']-saved['KL'])<=1e-12 and row['disagreement']==saved['disagreement']
            else:
                query=a.directory/(prefix+'.u32');raw(query,struct.pack('<I',len(rec['student_input_ids']))+np.asarray(rec['student_input_ids'],dtype='<u4').tobytes())
                paths=[a.directory/(prefix+s) for s in ('.f32','.routes','.witness','.native.json')]
                run([exe,'--prefix',b['packed']['path'],query,*paths],prefix)
                assert paths[2].read_bytes()==expected,'integer witness'
                row=metrics(rec,paths[0],paths[1])
            rows.append(row);write(a.directory/(prefix+'.metrics.json'),row)
            event(cases=len(rows),id=prefix,KL=row['KL'],disagreement=row['disagreement_rate'],adopted=row['adopted'])
        agg=aggregate(rows);write(a.directory/'native_aggregate.json',agg)
        stage='stream_qualification';tokenizer=ChatTokenizer(b['source'])
        os.environ['SILICON_CHAT_SECONDS']=str(b['limits']['stream_seconds'])
        os.environ['SILICON_CHAT_SCORE_PREFIX']=str(a.directory/'stream')
        os.environ['SILICON_CHAT_ROUTE_PREFIX']=str(a.directory/'stream')
        stream_begin=time.monotonic();stream_peak=0;stream_receipt=None
        with (a.directory/'stream.log').open('xb') as log:
            client=OriginalEngineClient(exe,b['packed']['path'],log,timeout=b['limits']['child_seconds'])
            created=psutil.Process(client.process.pid).create_time()
            def request(ids,max_new,force=False,label=''):
                nonlocal stream_peak,child_peak
                row,head=client.request(ids,max_new,force)
                stream_peak=max(stream_peak,reader(client.process));child_peak=max(child_peak,stream_peak)
                index=len(requests);row.update(index=index,label=label,query_ids=ids,full_head=True)
                final=a.directory/f'stream.{index:04d}.final.f32';raw(final,head);row['final_head']=extent(final)
                route_path=a.directory/f'stream.{index:04d}.routes';row['routes']=extent(route_path)
                assert row['routes']['bytes']==row['core_calls']*L*64
                routes=np.fromfile(route_path,dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))])).reshape(-1,L)
                if len(routes):
                    assert routes['ids'].min()>=0 and routes['ids'].max()<E
                    assert (np.diff(np.sort(routes['ids'],axis=-1),axis=-1)>0).all()
                    assert np.isfinite(routes['mass']).all() and (routes['mass']>=0).all()
                    row['mass_defect']=float(np.max(np.abs(routes['mass'].astype('f8').sum(-1)-1)));assert row['mass_defect']<=1e-6
                else:row['mass_defect']=None
                if max_new:
                    score_path=a.directory/f'stream.{index:04d}.scores.f32';row['scores']=extent(score_path)
                    scores=np.fromfile(score_path,dtype='<f4').reshape(-1,V)
                    assert len(scores)==len(row['generated_ids']) and np.isfinite(scores).all()
                    assert np.argmax(scores,axis=-1).tolist()==row['generated_ids'],'score greedy IDs'
                    row['all_greedy_ids_exact']=True
                assert np.isfinite(np.frombuffer(head,dtype='<f4')).all()
                requests.append(row);write(a.directory/f'stream.{index:04d}.json',row);event(request=index,label=label,generated=len(row['generated_ids']))
                return row,head
            try:
                qualification=next(r for r in b['records'] if r['id']==b['qualification_id']);ids=qualification['student_input_ids']
                old=adopted_index[qualification['id']];offset=old['row_offset'];split=b['qualification_split']
                retained=np.memmap(b['adopted']['after.f32']['path'],dtype='<f4',mode='r',shape=(3352,V))
                full,head=request(ids,0,True,'full_prefix');assert head==retained[offset+len(ids)-1].tobytes()
                old_routes=Path(b['adopted']['after.routes']['path']).read_bytes()[offset*L*64:(offset+len(ids))*L*64]
                assert Path(full['routes']['path']).read_bytes()==old_routes
                part,parthead=request(ids[:split],0,True,'split_first');assert parthead==retained[offset+split-1].tobytes()
                tail,tailhead=request(ids,0,False,'split_tail');assert tail['reused_prefix_ids']==split and tailhead==head
                assert Path(part['routes']['path']).read_bytes()+Path(tail['routes']['path']).read_bytes()==old_routes
                reuse,reusehead=request(ids,0,False,'reuse_exact');assert reuse['reused_prefix_ids']==len(ids) and reuse['core_calls']==0 and reusehead==head
                qualification_result=dict(id=qualification['id'],full_prefix_bit_exact=True,split_prefix_bit_exact=True,
                    all_route_ids_masses_bit_exact=True,reuse_without_forward_bit_exact=True,history=len(ids),split=split)
                write(a.directory/'stream_qualification.json',qualification_result);del retained
                stage='native_tasks';followup=None
                for case in b['cases']:
                    row,_=request(case['input_ids'],b['max_new'],True,case['id'])
                    output=row['generated_ids'];text=tokenizer.decode(output)
                    result=dict(id=case['id'],category=case['category'],expected=case['expected'],request=row,
                        messages=case['messages'],output_text=text,output_with_special=tokenizer.tokenizer.decode(output,skip_special_tokens=False),
                        normalized=normalized(text),correct=normalized(text)==case['expected'],blank=not text.strip(),
                        special_leak=any(i in set(b['special_ids'])-{11,228} for i in output),
                        stop_policy_ok=bool(output) and (output[-1] in (11,228) or len(output)==64))
                    tasks.append(result);write(a.directory/(case['id']+'.task.json'),result)
                    event(task=case['id'],answer=text,correct=result['correct'])
                    if case['id']==b['followup']['case_id']:
                        # Canonical serialization includes the candidate's decoded answer.
                        # Prefix reuse only follows exact ID identity; retokenization can reset.
                        messages=case['messages']+[dict(role='assistant',content=text),dict(role='user',content=b['followup']['prompt'])]
                        qids=tokenizer.encode(messages);answer,_=request(qids,b['max_new'],False,'own_answer_followup')
                        decoded=tokenizer.decode(answer['generated_ids'])
                        cached=case['input_ids']+output;compatible=qids[:len(cached)]==cached
                        assert answer['reused_prefix_ids']==(len(cached) if compatible else 0)
                        followup=dict(messages=messages,request=answer,output_text=decoded,expected=b['followup']['expected'],
                            correct=normalized(decoded)==b['followup']['expected'],exact_cached_prefix_compatible=compatible,
                            scope='One canonical followup on candidate answer;not broad independent own-history usefulness.')
                        write(a.directory/'own_answer_followup.json',followup)
                client.close()
            finally:
                client.abort();stream_peak=max(stream_peak,reader(client.process));child_peak=max(child_peak,stream_peak)
                stream_receipt=dict(pid=client.process.pid,creation_time=created,exit_code=client.process.returncode,
                    held_OS_peak=stream_peak,seconds=time.monotonic()-stream_begin,command=[str(exe),'--stream',b['packed']['path']])
                children.append(stream_receipt);write(a.directory/'stream.receipt.json',stream_receipt)
        categories={c:sum(r['correct'] for r in tasks if r['category']==c) for c in sorted({r['category'] for r in tasks})}
        task_gates=dict(correct_min=sum(r['correct'] for r in tasks)>=12,every_category_min=all(v>=2 for v in categories.values()),
            blank_max=sum(r['blank'] for r in tasks)<=1,no_special_leak=not any(r['special_leak'] for r in tasks),stop_policy=all(r['stop_policy_ok'] for r in tasks))
        dev=agg['DEV'];native_gates=dict(DEV_case_KL=dev['case_KL']<=1.,DEV_case_disagreement=dev['case_disagreement']<=.20,
            all_domain_KL=all(r['case_KL']<=2. for r in dev['domains'].values()),
            all_domain_disagreement=all(r['case_disagreement']<=.35 for r in dev['domains'].values()))
        generated=[r for r in requests if r['generated_ids']];total=sum(len(r['generated_ids']) for r in generated)
        rates=dict(generated_ids=total,decode_seconds=sum(r['decode_seconds'] for r in generated),
            pipe_request_seconds=sum(r['pipe_request_seconds'] for r in generated),request_seconds=sum(r['request_seconds'] for r in generated))
        rates['raw_decode_ids_s']=total/rates['decode_seconds'];rates['raw_request_ids_s']=total/rates['request_seconds']
        rates['raw_pipe_request_ids_s']=total/rates['pipe_request_seconds']
        stage='complete';guard()
        result=dict(schema='ORIGINAL_ENGINE_BASELINE_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='ACTUAL27_BASELINE_COMPLETE',packed=b['packed'],native_records=rows,native_aggregate=agg,native_absolute_gates=native_gates,
            original_bodies=body,executable=extent(exe),stream_qualification=qualification_result,stream_hello=list(client.hello),
            tasks=tasks,task_correct=sum(r['correct'] for r in tasks),category_correct=categories,task_gates=task_gates,
            source_task_correct=14,own_answer_followup=followup,requests=requests,raw_rates=rates,children=children,
            adopted_histories=3,new_native_histories=45,new_native_history_positions=22547-3352,
            qualification_core_calls=sum(r['core_calls'] for r in requests[:4]),source_calls=0,GPU_calls=0,
            optimizer_updates=0,reserved_queries=0,quality_admission=False,speed_admission=False,useful_large_n_admission=False,
            physical_DRAM_bytes=None,historical_GPU_native_numerical_gate='FAIL retained',
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=child_peak,elapsed_seconds=time.monotonic()-start,
            scope=b['scope']+' Rates include all emitted IDs, including EOS and incorrect content. Quality-qualified accepted rate unavailable.'
                ' Followup is diagnostic and separately reported. Held child peaks survive exit; nested linker not held individually.')
        write(a.out,result);event(decision=result['decision'],task_correct=result['task_correct'],DEV=dev['case_KL'])
    except BaseException as error:
        if client:client.abort()
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),completed_histories=[r['id'] for r in rows],
            completed_tasks=[r['id'] for r in tasks],completed_requests=len(requests),children=children,
            elapsed_seconds=time.monotonic()-start,worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=child_peak))
        raise


def launch(a):
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='ORIGINAL_ENGINE_BASELINE_BINDING_V1' and Path(sys.executable).resolve()==Path(b['python']).resolve()
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for other in psutil.process_iter(['name','cmdline']):
        name=(other.info['name'] or '').lower();argv=other.info['cmdline'] or []
        if other.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','clang','native','original_engine','packed_original','lld')),('overlap',other.pid,name)
    log=a.out.with_suffix('.worker.log');terminal=a.out.with_suffix('.terminal.json');failure=a.out.with_suffix('.launcher_failure.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out,log,terminal,failure))
    record=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_pid=proc.pid)
    reader=memory_reader();worker_process=None;peak=0;seen={}
    def guard():
        nonlocal peak
        peak=max(peak,reader(worker_process));assert time.monotonic()-start<=b['limits']['seconds'],'family deadline'
        assert peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family OS cap'
        assert log.stat().st_size<=b['limits']['log_bytes'],'log cap'
    try:
        check_inputs(b)
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        command=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),
            '--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
        record['command']=command
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
                        if name=='original_engine_chat.exe':assert exe==a.directory.resolve()/name
                        elif name=='conhost.exe':assert exe==Path('C:/Windows/System32/conhost.exe').resolve()
                        else:
                            assert name in ('clang.exe','clang-21.exe','ld.lld.exe','lld.exe','ld.exe') and exe.parent==Path(b['compiler']).parent,('unexpected child',name,exe)
                except psutil.NoSuchProcess:pass
                with log.open('rb') as stream:
                    stream.seek(offset);chunk=stream.read();end=chunk.rfind(b'\n')+1
                    if end:print(chunk[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.1)
        guard();record.update(exit_code=worker_process.returncode,worker_OS_peak_through_exit=peak)
        assert worker_process.returncode==0,log.read_text(errors='replace')[-6000:]
        check_inputs(b);result=json.loads(a.out.read_bytes());assert result['schema']=='ORIGINAL_ENGINE_BASELINE_RESULT_V1'
        assert len(result['native_records'])==48 and len(result['tasks'])==16 and len(result['requests'])==21
        assert all(result['stream_qualification'][key] for key in ('full_prefix_bit_exact','split_prefix_bit_exact','all_route_ids_masses_bit_exact','reuse_without_forward_bit_exact'))
        assert all(c['exit_code']==0 for c in result['children']) and result['source_calls']==result['optimizer_updates']==result['GPU_calls']==0
        assert peak+result['max_direct_child_OS_peak']+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'conservative family OS union'
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        assert sum(i['bytes'] for i in outputs)<=b['limits']['output_bytes']
        record.update(elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,
            result_sha256=sha(a.out),output_files=outputs,resource_gates=True,decision=result['decision'])
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
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
