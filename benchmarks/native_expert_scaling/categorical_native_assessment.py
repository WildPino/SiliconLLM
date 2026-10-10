"""New categorical packed head: actual unchanged C forced histories and chat."""
import argparse,json,math,os,struct,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,sha,write,raw,extent
from original_falcon_whole_recovery import memory_reader,check_inputs
from original_engine_baseline import metrics,aggregate
sys.path.insert(0,str(SITE))
V,L,E=65537,6,1152

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
        stage='native_all48';exe=Path(b['executable']['path']);body=b['original_bodies'];old_index={r['id']:r for r in b['old_native']}
        expected=Path(b['adopted']['expected_updated.witness']['path']).read_bytes();assert len(expected)==18432*4
        for rec in b['records']:
            prefix=rec['id'];query=a.directory/(prefix+'.u32');raw(query,struct.pack('<I',len(rec['student_input_ids']))+np.asarray(rec['student_input_ids'],dtype='<u4').tobytes())
            paths=[a.directory/(prefix+s) for s in ('.f32','.routes','.witness','.native.json')]
            run([exe,'--prefix',b['packed']['path'],query,*paths],prefix)
            assert paths[2].read_bytes()==expected,'integer witness'
            row=metrics(rec,paths[0],paths[1]);old=old_index[prefix]
            with Path(old['routes']['path']).open('rb') as oldroute:
                oldroute.seek(old['row_offset']*L*64);expected_routes=oldroute.read(row['history']*L*64)
            assert paths[1].read_bytes()==expected_routes,'head-only forced route identity'
            row['old_route_ids_masses_bit_exact']=True
            rows.append(row);write(a.directory/(prefix+'.metrics.json'),row)
            event(cases=len(rows),id=prefix,KL=row['KL'],disagreement=row['disagreement_rate'],adopted=False)
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
                old=old_index[qualification['id']];offset=0;split=b['qualification_split'];qualified=next(row for row in rows if row['id']==qualification['id'])
                retained=np.memmap(qualified['logits']['path'],dtype='<f4',mode='r',shape=(len(ids),V))
                full,head=request(ids,0,True,'full_prefix');assert head==retained[offset+len(ids)-1].tobytes()
                old_routes=Path(old['routes']['path']).read_bytes()[old['row_offset']*L*64:(old['row_offset']+len(ids))*L*64]
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
        result=dict(schema='CATEGORICAL_NATIVE_ASSESSMENT_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='CATEGORICAL_NATIVE_ASSESSMENT_COMPLETE',packed=b['packed'],native_records=rows,native_aggregate=agg,native_absolute_gates=native_gates,
            original_bodies=body,executable=extent(exe),stream_qualification=qualification_result,stream_hello=list(client.hello),
            tasks=tasks,task_correct=sum(r['correct'] for r in tasks),category_correct=categories,task_gates=task_gates,
            source_task_correct=14,own_answer_followup=followup,requests=requests,raw_rates=rates,children=children,
            adopted_histories=0,new_native_histories=48,new_native_history_positions=22547,
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

def bind(a):
    import numpy as np
    import psutil
    from original_engine_chat import ChatTokenizer
    from original_joint_history_recovery_audit import packed_fields
    assert not a.out.exists()
    parent=DOC/'causal_categorical_readout_result_20261010.json';pr=json.loads(parent.read_bytes());ptpath=parent.with_suffix('.terminal.json');pt=json.loads(ptpath.read_bytes())
    audit=DOC/'causal_categorical_readout_stored_adjudication_20261010.json';ar=json.loads(audit.read_bytes());atpath=audit.with_suffix('.terminal.json');at=json.loads(atpath.read_bytes())
    assert pt['exit_code']==at['exit_code']==0 and pt['error'] is at['error'] is None
    assert ar['result']==pt['result']==extent(parent) and at['result']==extent(audit)
    for flag in ('complete_input_output_hashes','all_FIT_teacher_moments_entropy_argmax_verified','full_weighted_SVD_whitening_initializer_verified','all_checkpoint_Y_objectives_gradients_verified','final_feasible_upper_and_tangent_lower_verified','packed_all_bytes_except_head_exact','forced_native_uncertainty_bounds_verified'):assert ar[flag]
    cbpath=DOC/'causal_categorical_readout_binding_20261010.json';cb=json.loads(cbpath.read_bytes());assert pr['binding_sha256']==sha(cbpath)==pt['binding_sha256']
    oldpath=DOC/'original_engine_baseline_result_20261009.json';old=json.loads(oldpath.read_bytes());otpath=oldpath.with_suffix('.terminal.json');ot=json.loads(otpath.read_bytes());assert ot['exit_code']==0 and ot['result_sha256']==sha(oldpath)
    obpath=DOC/'original_engine_baseline_binding_20261009.json';ob=json.loads(obpath.read_bytes());assert sha(obpath)==old['binding_sha256']
    oldaudit=DOC/'original_engine_baseline_stored_adjudication_20261009.json';oa=json.loads(oldaudit.read_bytes());assert oa['result']==extent(oldpath)
    crpath=DOC/'causal_coordinate_dot_bound_result_20261010.json';cr=json.loads(crpath.read_bytes());assert cr['decision']=='CAUSAL_COORDINATES_QUALIFIED_APPROXIMATE'
    assert len(ob['records'])==len(cr['records'])==48 and sum(len(r['student_input_ids']) for r in ob['records'])==22547
    assert packed_fields(pr['artifacts']['packed']['path'])==packed_fields(cb['packed']['path'])
    old_index={r['id']:r for r in old['native_records']};coord_index={r['id']:r for r in cr['records']}
    for rec in ob['records']:
        c=coord_index[rec['id']];assert rec['positions']==c['positions'] and rec['split']==c['split'] and len(rec['positions'])==c['labels']
    files=[Path(__file__),B/'categorical_native_assessment_audit.py',DOC/'CATEGORICAL_NATIVE_ASSESSMENT_PROTOCOL_20261010.md',parent,ptpath,audit,atpath,cbpath,oldpath,otpath,obpath,oldaudit,crpath,Path(pr['artifacts']['packed']['path']),Path(pr['artifacts']['head_real']['path']),Path(old['executable']['path']),Path(old['original_bodies']['header']['path']),Path(old['executable']['path']).parent/'original_wide_prefix.c',Path(old['executable']['path']).parent/'original_engine_chat_main.c',Path(sys.executable),Path(sys.executable).parent/'python312.dll']
    files += [B/name for name in ('original_engine_baseline.py','original_engine_chat.py','original_packed_capacity.py','original_packed_capacity.c','original_engine_chat_main.c','original_falcon_whole_recovery.py','chatbot_hybrid_engine_chat.py','chatbot_falcon_usability.py','chatbot_falcon_usability_launch.py','original_joint_history_recovery_audit.py','paired_output_codec.py')]
    files += [Path(old['original_bodies']['source']['path'])]+[Path(ob['adopted'][key]['path']) for key in ('requests.u32','expected_updated.witness')]
    files += [Path(r['logits']['path']) for r in ob['records']]+[Path(r['routes']['path']) for r in old['native_records']]+[Path(c['features']['path']) for c in cr['records']]
    files += [Path(c['source_record']['path']) for c in ob['cases']]+[Path(ob['source'])/name for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json')]
    tokenizer=ChatTokenizer(ob['source'])
    for case in ob['cases']:assert tokenizer.encode(case['messages'])==case['input_ids']
    files += [Path(module.__file__) for module in list(sys.modules.values()) if getattr(module,'__file__',None) and str(SITE.resolve()).lower() in str(Path(module.__file__).resolve()).lower()]
    files += sorted((SITE/'numpy.libs').glob('*.dll'))+sorted((SITE/'psutil').glob('*.pyd'))
    inputs=[extent(p) for p in dict.fromkeys(files)];lookup={i['path']:i for i in inputs}
    inherited=cb['inputs']+ob['inputs']+ot['output_files']+pt['outputs']
    for item in inherited:
        if item['path'] in lookup:assert lookup[item['path']]=={key:item[key] for key in ('path','bytes','sha256')}
    binding=dict(schema='CATEGORICAL_NATIVE_ASSESSMENT_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),packed=pr['artifacts']['packed'],executable=old['executable'],original_bodies=old['original_bodies'],records=ob['records'],cases=ob['cases'],source=ob['source'],adopted=ob['adopted'],adopted_sequences=ob['adopted_sequences'],old_native=old['native_records'],coordinates=cr['records'],native_rounding=cr['calibration'],head=cb['fields']['head'],head_real=pr['artifacts']['head_real'],parent=extent(parent),parent_audit=extent(audit),source_result=ob['source_result'],qualification_id=ob['qualification_id'],qualification_split=ob['qualification_split'],max_new=ob['max_new'],followup=ob['followup'],special_ids=ob['special_ids'],gates=ob['gates'],strict_readout_gates=cb['quality_gates'],limits=dict(seconds=900,reserve_seconds=60,OS_bytes=4<<30,output_bytes=8<<30,child_seconds=100,stream_seconds=300,log_bytes=8<<20),audit_limits=dict(seconds=600,OS_bytes=4<<30,output_bytes=2<<20,log_bytes=8<<20),inputs=inputs,scope='New head artifact, all48 forced histories22547 positions/8808 labels, old route identity,16 original chat tasks+one own-answer followup. No source/GPU/optimizer/reserved/T4/compile call; unchanged20 original bodies. Timed full-V heads/routes/score copies, trace writes outside C timers but inside pipe. Raw rates include wrong tokens/EOS, not accepted useful50; no DRAM/n/family admission.')
    write(a.out,binding);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs),input_bytes=sum(i['bytes'] for i in inputs))),flush=True)

def launch(a):
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='CATEGORICAL_NATIVE_ASSESSMENT_BINDING_V1' and Path(sys.executable).resolve()==Path(b['python']).resolve()
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
                        if name=='original_engine_chat.exe':assert exe==Path(b['executable']['path']).resolve()
                        elif name=='conhost.exe':assert exe==Path('C:/Windows/System32/conhost.exe').resolve()
                        else:
                            raise AssertionError(('unexpected child',name,exe))
                except psutil.NoSuchProcess:pass
                with log.open('rb') as stream:
                    stream.seek(offset);chunk=stream.read();end=chunk.rfind(b'\n')+1
                    if end:print(chunk[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.1)
        guard();record.update(exit_code=worker_process.returncode,worker_OS_peak_through_exit=peak)
        assert worker_process.returncode==0,log.read_text(errors='replace')[-6000:]
        check_inputs(b);result=json.loads(a.out.read_bytes());assert result['schema']=='CATEGORICAL_NATIVE_ASSESSMENT_RESULT_V1'
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
