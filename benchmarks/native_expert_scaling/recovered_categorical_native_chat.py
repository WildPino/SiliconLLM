"""New canonical native chat capture and stored audit on the unchanged final pack."""
import argparse,json,math,os,sys,time,traceback
from pathlib import Path
BEGAN=time.monotonic()
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha
sys.path.insert(0,str(SITE))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
LIMITS=dict(seconds=900,reserve_seconds=120,OS_bytes=8<<30,output_bytes=512<<20,child_seconds=100,stream_seconds=300)


def emit(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def verify(items):
    for item in items:assert extent(item['path'])==item,('changed_input',item['path'])


def prepare(a):
    from original_joint_native_assessment import tasks
    from original_engine_chat import ChatTokenizer
    from original_categorical_campaign_geometry import stream_counters
    import numpy as np,psutil
    start=BEGAN;parent=DOC/'recovered_categorical_campaign_closure_adjudication_20261012.json'
    proof=json.loads(parent.read_bytes());assert proof['decision']=='MATH_AND_STORAGE_COMPLETE_ISOLATION_FAILED'
    assert proof['complete_numeric_worker'] and proof['input_output_hashes_before_after_proven_by_bound_control_flow']
    report=json.loads(Path(proof['report']['path']).read_bytes());assert extent(proof['report']['path'])==proof['report']
    old_path=DOC/'original_categorical_campaign_binding_20261010.json';old=json.loads(old_path.read_bytes())
    packed=report['final_packed'];assert extent(packed['path'])==packed
    assert extent(old['executable']['path'])==old['executable']
    tokenizer=ChatTokenizer(old['source'])
    assert len(old['cases'])==16 and all(tokenizer.encode(c['messages'])==c['input_ids'] for c in old['cases'])
    paths=[Path(__file__),a.protocol,parent,Path(proof['source']['path']),old_path,
        *[Path(proof[k]['path']) for k in ('report','terminal','log','binding','holder_source','worker_source')],
        Path(packed['path']),Path(old['executable']['path']),Path(old['original_bodies']['source']['path'])]
    paths += [Path(old['source'])/n for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json')]
    paths += [Path(c['source_record']['path']) for c in old['cases']]
    for module in list(sys.modules.values()):
        for key in ('__file__','__cached__'):
            p=getattr(module,key,None)
            if isinstance(p,(str,os.PathLike)) and Path(p).is_file():paths.append(Path(p))
    paths += [Path(sys.executable),Path(sys.executable).parent/'python312.dll']
    paths += sorted((SITE/'numpy.libs').glob('*.dll'))+sorted((SITE/'psutil').glob('*.pyd'))
    if 'torch' in sys.modules:paths += sorted((SITE/'torch/lib').glob('*.dll'))
    inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)]
    lookup={x['path']:x for x in inputs};prior={x['path']:x for x in old['inputs']}
    for key,item in lookup.items():
        if key in prior:assert item==prior[key]
    assert (np.__version__,psutil.__version__)==('2.4.6','7.2.2')
    emit(a.binding,dict(schema='RECOVERED_CATEGORICAL_NATIVE_CHAT_BINDING_V1',inputs=inputs,
        parent=extent(parent),packed=packed,executable=old['executable'],source=old['source'],
        cases=old['cases'],followup=old['followup'],special_ids=old['special_ids'],gates=old['gates'],
        limits=LIMITS,protocol=extent(a.protocol),source_file=extent(__file__),seconds=time.monotonic()-start,
        timing_control_admitted=False,original_capture_custody=False))
    print(json.dumps(dict(binding_sha256=sha(a.binding),inputs=len(inputs),seconds=time.monotonic()-start)),flush=True)


def capture(a):
    import psutil
    from original_joint_native_assessment import tasks
    from original_falcon_whole_recovery import memory_reader
    start=BEGAN;assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['limits']==LIMITS and b['source_file']==extent(__file__)
    verify(b['inputs']);a.directory.mkdir(exist_ok=False)
    proc=psutil.Process();proc.cpu_affinity([1]);reader=memory_reader();peak=0;error=None;result=None;stage='native_tasks'
    def guard():
        assert time.monotonic()-start<LIMITS['seconds']-LIMITS['reserve_seconds'],('time_cap',stage)
        assert proc.memory_info().peak_wset+peak<=LIMITS['OS_bytes'],('OS_cap',stage)
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=LIMITS['output_bytes'],('output_cap',stage)
    def observe(value):
        nonlocal peak
        peak=max(peak,value)
    def event(**values):guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**values)),flush=True)
    observations=[]
    try:
        result=tasks(b,a.directory,Path(b['executable']['path']),Path(b['packed']['path']),'chat',reader,observe,guard,event)
        assert result['receipt']['exit_code']==0 and len(result['requests'])==17 and len(result['records'])==16
        assert not proc.children(recursive=True)
        stage='input_hash_after';verify(b['inputs']);guard()
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        emit(a.out,dict(schema='RECOVERED_CATEGORICAL_NATIVE_CHAT_RESULT_V1',binding=extent(a.binding),
            binding_sha256=a.binding_sha,freeze=a.freeze,parent=b['parent'],packed=b['packed'],tasks=result,
            outputs=outputs,inputs_before_after_exact=True,original_successful_custody=False,
            timing_control_admitted=False,quality_admission=False,speed_admission=False,
            neural_model_constructions=0,history_backwards=0,exports=0,optimizer_updates=0,native_stream_processes=1,
            OS_peak=proc.memory_info().peak_wset,native_OS_peak_through_exit=peak,seconds=time.monotonic()-start))
    except BaseException as failure:
        error=dict(error=repr(failure),traceback=traceback.format_exc(),stage=stage)
        for child in proc.children(recursive=True):
            try:
                argv=child.cmdline();observations.append(dict(pid=child.pid,created=child.create_time(),ppid=child.ppid(),command=argv))
                if argv==[b['executable']['path'],'--stream',b['packed']['path']]:child.kill();child.wait()
            except psutil.NoSuchProcess:pass
        emit(a.out.with_suffix('.fault.json'),dict(schema='RECOVERED_CATEGORICAL_NATIVE_CHAT_FAULT_V1',
            **error,observed_children=observations,seconds=time.monotonic()-start))
        raise
    finally:
        emit(a.out.with_suffix('.terminal.json'),dict(schema='RECOVERED_CATEGORICAL_NATIVE_CHAT_TERMINAL_V1',
            binding=extent(a.binding),binding_sha256=a.binding_sha,freeze=a.freeze,error=error,
            exit_code=0 if error is None else 1,result=extent(a.out) if a.out.exists() else None,
            OS_peak=proc.memory_info().peak_wset,native_OS_peak_through_exit=peak,seconds=time.monotonic()-start,
            timing_control_admitted=False))


def audit(a):
    import numpy as np,psutil
    from original_engine_chat import ChatTokenizer
    from original_categorical_campaign_geometry import stream_counters
    from chatbot_falcon_usability import normalized
    start=BEGAN;psutil.Process().cpu_affinity([1])
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.source_result)
    assert r['binding']==t['binding']==extent(a.binding) and r['inputs_before_after_exact']
    verify(b['inputs']+r['outputs']);tokenizer=ChatTokenizer(b['source']);task=r['tasks']
    assert len(task['records'])==16 and len(task['requests'])==17 and task['receipt']['exit_code']==0
    assert task['receipt']['command']==[b['executable']['path'],'--stream',b['packed']['path']]
    assert task['receipt']['held_OS_peak']==r['native_OS_peak_through_exit']
    assert 0<task['receipt']['seconds']<=LIMITS['stream_seconds']
    generated=0;requests=task['requests'];cases={c['id']:c for c in b['cases']}
    assert [x['index'] for x in requests]==list(range(17)) and len({x['label'] for x in requests})==17
    for request in requests:
        if request['label']=='own_answer_followup':
            follow=task['own_answer_followup'];case=next(x for x in task['records'] if x['id']==b['followup']['case_id'])
            messages=case['messages']+[dict(role='assistant',content=case['output_text']),dict(role='user',content=b['followup']['prompt'])]
            assert messages==follow['messages'] and request['query_ids']==tokenizer.encode(messages)
            assert follow['request']==request and follow['output_text']==tokenizer.decode(request['generated_ids'])
            assert follow['expected']==b['followup']['expected']
            cached=case['request']['query_ids']+case['request']['generated_ids'];compatible=request['query_ids'][:len(cached)]==cached
            generated+=stream_counters(request,len(cached) if compatible else 0,not compatible)
            assert follow['exact_cached_prefix_compatible']==compatible
            assert follow['correct']==(normalized(tokenizer.decode(request['generated_ids']))==b['followup']['expected'])
        else:
            assert request['query_ids']==cases[request['label']]['input_ids']
            generated+=stream_counters(request,0,True)
        scores=np.fromfile(request['scores']['path'],dtype='<f4').reshape(-1,65537)
        assert np.isfinite(scores).all() and scores.argmax(1).tolist()==request['generated_ids']
        head=np.fromfile(request['final_head']['path'],dtype='<f4');assert head.shape==(65537,) and np.isfinite(head).all()
        route=np.fromfile(request['routes']['path'],dtype=np.dtype([('ids','<i4',(8,)),('mass','<f4',(8,))])).reshape(-1,6)
        assert len(route)==request['core_calls'] and np.all((route['ids']>=0)&(route['ids']<1152))
        assert np.all(np.diff(np.sort(route['ids'],axis=-1),axis=-1)>0) and np.isfinite(route['mass']).all() and (route['mass']>=0).all()
        assert float(np.max(abs(route['mass'].astype('f8').sum(-1)-1)))==request['mass_defect']<=1e-6
        assert time.monotonic()-start<900 and psutil.Process().memory_info().peak_wset<=LIMITS['OS_bytes']
    categories={c:0 for c in task['categories']};blank=leak=0
    assert [x['id'] for x in task['records']]==[x['id'] for x in b['cases']]
    for row in task['records']:
        case=cases[row['id']];text=tokenizer.decode(row['request']['generated_ids'])
        assert row['output_text']==text and row['expected']==case['expected'] and row['messages']==case['messages']
        assert row['category']==case['category'] and row['normalized']==normalized(text)
        assert row['request']==next(x for x in requests if x['label']==row['id'])
        correct=normalized(text)==case['expected'];assert row['correct']==correct;categories[row['category']]+=correct
        is_blank=not text.strip();is_leak=any(i in set(b['special_ids'])-{11,228} for i in row['request']['generated_ids'])
        ids=row['request']['generated_ids'];assert row['stop_policy_ok']==(bool(ids) and (ids[-1] in (11,228) or len(ids)==64))
        assert row['blank']==is_blank and row['special_leak']==is_leak
        blank+=is_blank;leak+=is_leak
    gates=dict(correct_min=sum(categories.values())>=12,every_category_min=all(v>=2 for v in categories.values()),
        blank_max=blank<=1,no_special_leak=leak==0,stop_policy=all(x['stop_policy_ok'] for x in task['records']))
    assert categories==task['categories'] and gates==task['gates'] and sum(categories.values())==task['correct']
    assert generated==task['generated_ids']
    rates=dict(raw_decode_ids_s=generated/math.fsum(x['decode_seconds'] for x in requests),
        raw_pipe_request_ids_s=generated/math.fsum(x['pipe_request_seconds'] for x in requests))
    assert all(abs(rates[k]-task[k])<=1e-10*max(1,abs(task[k])) for k in rates)
    verify(b['inputs']+r['outputs'])
    emit(a.out,dict(schema='RECOVERED_CATEGORICAL_NATIVE_CHAT_AUDIT_V1',binding=extent(a.binding),result=extent(a.source_result),
        all17_requests_16_tasks_complete=True,all_full_vocab_greedy_IDs_routes_counters_tokenization_exact=True,
        inputs_outputs_before_after_exact=True,correct=sum(categories.values()),categories=categories,gates=gates,
        own_answer_followup_correct=task['own_answer_followup']['correct'],generated_ids=generated,**rates,
        timing_control_admitted=False,quality_admission=False,speed_admission=False,seconds=time.monotonic()-start,
        OS_peak=psutil.Process().memory_info().peak_wset,scope='New same-final-pack chat behavior; raw rates are not useful50 admission.'))
    print(json.dumps(dict(correct=sum(categories.values()),cases=16,gates=gates,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','capture','audit'])
    for key in ('binding','protocol','directory','out','source-result'):parser.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'):parser.add_argument('--'+key)
    args=parser.parse_args();globals()[args.mode](args)
