"""Separately bounded own-history FIT/evaluation phases for source depth."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
from full_width_native_chat import metrics,quality,check_inputs

FAMILY=Path('E:/SiliconLLM/results/native_expert_scaling/fixed_depth_quality_20261014')
LIMITS=dict(seconds=2700,reserve_seconds=90,OS_bytes=8<<30,GPU_allocated_bytes=6<<30,
            GPU_reserved_bytes=10<<30,output_bytes=32<<30,audit_seconds=2700)


def load(p):return json.loads(Path(p).read_bytes())


def fit_quality(rows):
    assert len(rows)==24 and all(r['split']=='FIT' for r in rows)
    domains={d:dict(KL=sum(q['KL'] for q in rows if q['domain']==d)/sum(q['domain']==d for q in rows),
        disagreement=sum(q['disagreement_rate'] for q in rows if q['domain']==d)/sum(q['domain']==d for q in rows))
        for d in sorted({q['domain'] for q in rows})}
    kl=sum(r['KL'] for r in rows)/24;wrong=sum(r['disagreement_rate'] for r in rows)/24
    gates=dict(case_KL=kl<=1,case_disagreement=wrong<=.20,
        domain_KL=all(q['KL']<=2 for q in domains.values()),
        domain_disagreement=all(q['disagreement']<=.35 for q in domains.values()))
    return dict(KL=kl,disagreement=wrong,domains=domains,gates=gates,all_pass=all(gates.values()))


def same_sequence(x,y,np):
    assert x['consumed_ids']==y['consumed_ids'] and [s['site'] for s in x['sites']]==[s['site'] for s in y['sites']]
    same=x['attention']==y['attention']
    for s,t in zip(x['sites'],y['sites'],strict=True):
        same &= s['trace']['sha256']==t['trace']['sha256']
        with np.load(s['final']['path']) as zs,np.load(t['final']['path']) as zt:
            same &= set(zs.files)==set(zt.files) and all(np.array_equal(zs[k].view('u1'),zt[k].view('u1')) for k in zs.files)
    return bool(same)


def prepare(a):
    import numpy as np
    import psutil
    import torch
    import transformers
    import safetensors
    import tokenizers
    import jinja2
    assert not torch.cuda.is_initialized() and not FAMILY.exists()
    bp=DOC/'full_width_native_chat_binding_20261013.json';b=load(bp)
    rp=DOC/'full_width_native_chat_result_20261013.json';r=load(rp)
    cp=DOC/'full_width_native_chat_session_closure_20261013.json';c=load(cp)
    assert c['decision']=='KNOWN_CORPUS_FULL_WIDTH_NATIVE_BACKBONE_QUALIFIED'
    assert c['report']==extent(rp) and extent(r['binding']['path'])==r['binding']
    closure=DOC/'native_bf_activation_lut_session_closure_20261014.json';act=load(closure)
    assert act['decision']=='EXACT_STORED_ACCELERATION_ELIGIBLE'
    ab=load(act['binding']['path'])
    frozen=[Path(__file__),B/'fixed_depth_quality_audit.py',B/'fixed_depth_native_audit.py',
            B/'fixed_depth_bridge.py',DOC/'FIXED_DEPTH_PROTOCOL_20261014.md',
            DOC/'FIXED_DEPTH_EXECUTION_20261014.md']
    for p in frozen:
        blob=subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n'),str(p)
    source=Path(b['source']);nv=Path('C:/Windows/System32/nvidia-smi.exe')
    gpu=subprocess.check_output([str(nv),'--query-gpu=uuid,name,driver_version,memory.total,memory.free',
                                 '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    assert len(gpu.splitlines())==1 and 'GPU-1cce7fba-8103-2d4b-ad9b-df8e162f8221' in gpu
    assert int(gpu.split(',')[-1])>=7168
    paths=frozen+[bp,rp,cp,Path(c['adjudication']['path']),closure,Path(act['binding']['path']),
        Path(ab['candidate_compile']['dll']['path']),Path(b['compile']['dll']['path']),Path(b['table']['path']),
        Path(b['prior_source_method']['path']),Path(sys.executable),Path(sys.executable).parent/'python312.dll',nv,
        B/'full_width_native_chat.py',B/'full_width_native_chat_audit.py',B/'bf_activation_backend.py',
        B/'true_generator_input_capture_v3.py',B/'chatbot_falcon_ssd_tiles.py',B/'original_engine_chat.py',
        B/'chatbot_falcon_usability.py',B/'row_state.py',B/'token_state.py',B/'falcon_auxiliary_s6_transport.py',
        B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability_launch.py']
    paths+=list(source.glob('*'))+[Path(e['file']['path']) for e in b['exports']]
    paths += [Path(e['file']['path']) for e in ab['activation_tables']]
    # Every runtime Python/extension/library file in the numerical/tokenization packages.
    for mod in (np,psutil,torch,transformers,safetensors,tokenizers,jinja2):
        paths.extend(p for p in Path(mod.__file__).resolve().parent.rglob('*')
                     if p.is_file() and p.suffix in ('.py','.pyd','.dll'))
    paths.extend((Path(np.__file__).resolve().parent.parent/'numpy.libs').glob('*.dll'))
    paths.extend(Path(m.__file__) for m in tuple(sys.modules.values())
                 if getattr(m,'__file__',None) and Path(m.__file__).is_file())
    # Historical resolved Python helpers are reused; data/GPU tools get explicit new pins.
    old=load(DOC/'ffn_operand_capture_binding_v2_20261013.json')
    paths.extend(Path(e['path']) for e in old['inputs'] if Path(e['path']).suffix=='.py')
    for rec in b['records']:paths.append(Path(rec['logits']['path']))
    for rec in r['records']:
        if rec['split']=='FIT':
            paths.append(Path(rec['scores']['path']))
            paths.extend(Path(e[k]['path']) for e in rec['sequence']['sites'] for k in ('trace','final'))
    for control in r['prefix_controls']:
        paths.append(Path(control['scores']['path']))
        for name in ('wrapper','continuation'):
            paths.extend(Path(e[k]['path']) for e in control[name]['sites'] for k in ('trace','final'))
    paths.extend(Path('C:/Windows/System32')/p for p in ('nvcuda.dll','nvml.dll','nvapi64.dll'))
    paths.extend(ROOT/p for p in ('benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md'))
    for p in paths:assert p.is_file(),str(p)
    pins=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)]
    original_source=next(e for e in b['inputs'] if Path(e['path']).resolve()==(source/'model.safetensors').resolve())
    assert extent(source/'model.safetensors')==original_source
    prompt=sum(len(c['input_ids']) for c in b['cases'])
    fit_tokens=sum(len(q['student_input_ids']) for q in b['records'] if q['split']=='FIT')+2*prompt
    eval_tokens=sum(len(q['student_input_ids']) for q in b['records'] if q['split']=='DEV')+3*prompt+17*64+2048
    # All4 FIT arms plus worst evaluation at12 active. Native snapshots <=56/73 per phase.
    bounds=dict(FIT_processed_tokens=fit_tokens,evaluation_processed_tokens=eval_tokens,
        family_numeric_payload_bound=(fit_tokens*(sum((count-1)*21600+4096 for count in (3,6,12,24)))+
            eval_tokens*(11*21600+4096)+(4*4422+4386+17*64+16*2)*65537*4+
            (56*(3+6+12+24)+73*12)*(3072*96*4+3*3584*2+2048)))
    assert bounds['family_numeric_payload_bound']<(30<<30)
    assert shutil.disk_usage(FAMILY.parent).free>(LIMITS['output_bytes']+2*(1<<30))
    FAMILY.mkdir()
    b.update(schema='FIXED_DEPTH_QUALITY_BINDING_V1',freeze=a.freeze,inputs=pins,limits=LIMITS,
        family=str(FAMILY),worker_path=str(Path(__file__).resolve()),activation_closure=extent(closure),
        parent_report=extent(rp),parent_closure=extent(cp),versions=dict(numpy=np.__version__,psutil=psutil.__version__,
            torch=torch.__version__,transformers=transformers.__version__),GPU_query=gpu,processed_bounds=bounds,
        scope='Fixed whole source block depth on own histories, retained residual2048/operators/native state96. '
              'Source GPU FFN/attention/head/generators. Not CPU useful50/DRAM/useful-n/general-family admission.',
        custody='New runtime/NVIDIA/source binding; old consumed teacher distributions retained. No RESERVED queries.')
    check_inputs(b);assert not torch.cuda.is_initialized();write(a.out,b)
    print(json.dumps(dict(binding=extent(a.out),pins=len(pins),bounds=bounds,GPU=gpu)),flush=True)


def select(a):
    reports=[];eligible=[];pins=[]
    for count in (24,3,6,12):
        rp=DOC/f'fixed_depth_fit{count}_result_20261014.json';ap=DOC/f'fixed_depth_fit{count}_audit_20261014.json'
        r,ad=load(rp),load(ap)
        assert ad['decision']=='INDEPENDENT_STORED_NUMERIC_PASS' and ad['report']==extent(rp)
        assert r['depth']==count and r['phase']=='fit' and r['FIT_quality']==fit_quality(r['records'])
        assert all(q['exact'] for q in r['prefix_controls'])
        if count==24:assert r['FIT_quality']['all_pass'] and r['parent_24_exact']
        elif r['FIT_quality']['all_pass']:eligible.append(count)
        reports.append(dict(depth=count,report=extent(rp),audit=extent(ap),FIT_quality=r['FIT_quality']))
        pins.extend([extent(rp),extent(ap),r['binding']])
    assert len({q['binding']['sha256'] for q in [load(x['report']['path']) for x in reports]})==1
    chosen=min(eligible) if eligible else None
    write(a.out,dict(schema='FIXED_DEPTH_FIT_SELECTION_V1',reports=reports,selected_depth=chosen,
        evaluation_depth=chosen if chosen is not None else 6,nonselected_defect_diagnostic=chosen is None,
        decision='FIT_SELECTED' if chosen is not None else 'NO_FIT_CANDIDATE_DEFECT6',inputs=pins,
        DEV_queries=0,task_choice_queries=0))
    print(json.dumps(dict(selection=extent(a.out),chosen=chosen)),flush=True)


def worker(a):
    import numpy as np
    import psutil
    import torch
    import transformers
    from transformers import FalconH1ForCausalLM
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    from fixed_depth_bridge import FixedDepthBridge
    from true_generator_input_capture_v3 import verify_parameters,descendant_snapshot
    from chatbot_falcon_ssd_tiles import install
    from original_engine_chat import ChatTokenizer
    from chatbot_falcon_usability import normalized
    b=load(a.binding);assert extent(a.binding)['sha256']==a.binding_sha and b['limits']==LIMITS
    assert a.depth in (3,6,12,24) and not a.directory.exists() and not a.out.exists()
    assert a.phase in ('fit','evaluation') and Path(sys.executable).resolve()==Path(b['python']).resolve()
    assert dict(numpy=np.__version__,psutil=psutil.__version__,torch=torch.__version__,transformers=transformers.__version__)==b['versions']
    assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));start=time.monotonic()
    bridge=None;model=None;stage='input_hashes';records=[];controls=[];tasks=[];followup=None
    receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),phase=a.phase,depth=a.depth)
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def guard():
        assert time.monotonic()-start<=LIMITS['seconds']-LIMITS['reserve_seconds'],'worker deadline'
        assert proc.memory_info().peak_wset<=LIMITS['OS_bytes'],'OS cap'
        assert torch.cuda.max_memory_allocated()<=LIMITS['GPU_allocated_bytes'],'GPU allocation cap'
        assert torch.cuda.max_memory_reserved()<=LIMITS['GPU_reserved_bytes'],'GPU reservation cap'
        assert not descendant_snapshot(proc)['fresh'],'unexpected descendant'
        assert sum(p.stat().st_size for p in FAMILY.rglob('*') if p.is_file())<=LIMITS['output_bytes'],'family storage cap'
    def event(**x):guard();print(json.dumps(dict(stage=stage,depth=a.depth,seconds=time.monotonic()-start,**x)),flush=True)
    try:
        check_inputs(b,guard)
        selection=None
        if a.phase=='fit' and a.depth!=24:
            control=load(DOC/'fixed_depth_fit24_result_20261014.json')
            audit=load(DOC/'fixed_depth_fit24_audit_20261014.json')
            assert audit['decision']=='INDEPENDENT_STORED_NUMERIC_PASS'
            assert audit['report']==extent(DOC/'fixed_depth_fit24_result_20261014.json') and control['parent_24_exact']
            assert control['binding']==extent(a.binding)
        if a.phase=='evaluation':
            selection=load(a.selection);assert selection['evaluation_depth']==a.depth
            for e in selection['inputs']:assert extent(e['path'])==e
        gpu=subprocess.check_output(['C:/Windows/System32/nvidia-smi.exe',
            '--query-gpu=uuid,driver_version,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
        assert int(gpu.split(',')[-1])>=7168
        assert gpu.split(',')[0].strip()==b['GPU_query'].split(',')[0].strip()
        assert gpu.split(',')[1].strip()==b['GPU_query'].split(',')[2].strip()
        a.directory.mkdir();stage='source_load'
        method=install(code);assert method==load(b['prior_source_method']['path'])
        write(a.directory/'source_method.json',method)
        model=FalconH1ForCausalLM.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        identities={name:(id(p),p._version) for name,p in model.named_parameters()}
        before=verify_parameters(model,b,torch,np,guard);write(a.directory/'parameter_before.json',before)
        assert not code.is_fast_path_available and model.config.mamba_chunk_size==128
        assert all(layer.mamba.mup_vector[0,0].detach().cpu().view(torch.uint16).tolist()==b['MuP_bits'] for layer in model.model.layers)
        bridge=FixedDepthBridge(model,b,np,torch,a.depth,b['activation_closure']['path'])
        # Every omitted descendant invocation is mechanically prohibited throughout the run.
        def forbidden(module,args):raise AssertionError('Omitted source operator invoked')
        hooks=[m.register_forward_pre_hook(forbidden) for i,layer in enumerate(model.model.layers)
               if i not in bridge.active for m in layer.modules() if m is not layer]
        tokenizer=ChatTokenizer(b['source']);parent=load(b['parent_report']['path']);parent_exact=True
        with torch.inference_mode():
            stage='prefix_controls'
            for case in b['cases']:
                bridge.reset(a.directory,case['id']+'.prefix_wrapper');x=bridge.prefill(case['input_ids'])
                first=bridge.finish(a.directory,case['id']+'.prefix_wrapper')
                bridge.reset(a.directory,case['id']+'.prefix_steps')
                for t,token in enumerate(case['input_ids']):y=bridge.step(token,t==len(case['input_ids'])-1)
                second=bridge.finish(a.directory,case['id']+'.prefix_steps')
                exact=same_sequence(first,second,np) and np.array_equal(x.view('u4'),y.view('u4'))
                assert exact,'prefix/cache continuation mismatch'
                path=a.directory/(case['id']+'.prefix_scores.npz');np.savez(path,wrapper=x,continuation=y)
                controls.append(dict(id=case['id'],exact=bool(exact),scores=extent(path),wrapper=first,continuation=second))
                if a.depth==24:
                    old=next(c for c in parent['prefix_controls'] if c['id']==case['id'])
                    with np.load(old['scores']['path']) as z:assert np.array_equal(x.view('u4'),z['wrapper'].view('u4'))
                    assert same_sequence(first,old['wrapper'],np) and same_sequence(second,old['continuation'],np)
                event(operation='prefix',id=case['id'],exact=bool(exact))
            stage='FIT_histories' if a.phase=='fit' else 'DEV_histories'
            wanted='FIT' if a.phase=='fit' else 'DEV'
            for rec in b['records']:
                if rec['split']!=wanted:continue
                bridge.reset(a.directory,rec['id']);positions=set(rec['positions']);scores=[]
                for t,token in enumerate(rec['student_input_ids']):
                    if t%128==0:event(operation='history_progress',id=rec['id'],position=t)
                    value=bridge.step(token,t in positions)
                    if value is not None:scores.append(value)
                values=np.stack(scores);assert values.shape==(len(rec['positions']),65537)
                path=a.directory/(rec['id']+'.labels.f32');values.astype('f4').tofile(path)
                seq=bridge.finish(a.directory,rec['id']);row=metrics(rec,values,np)
                row.update(scores=extent(path),sequence=seq);records.append(row)
                if a.depth==24:
                    old=next(q for q in parent['records'] if q['id']==rec['id'])
                    assert extent(path)['sha256']==old['scores']['sha256'] and same_sequence(seq,old['sequence'],np)
                write(a.directory/(rec['id']+'.metrics.json'),row)
                event(operation='history_complete',id=rec['id'],KL=row['KL'],disagreement=row['disagreement_rate'])
            if a.phase=='evaluation':
                stage='canonical_generation'
                def generate(messages,label):
                    ids=tokenizer.encode(messages);assert len(ids)<=2048
                    bridge.reset(a.directory,label)
                    for t,token in enumerate(ids):
                        if t%32==0:guard()
                        value=bridge.step(token,t==len(ids)-1)
                    new=[];scores=[]
                    for t in range(64):
                        guard();scores.append(value.copy());token=int(value.argmax());new.append(token)
                        if token in b['eos_ids'] or t==63:break
                        value=bridge.step(token)
                    path=a.directory/(label+'.scores.f32');np.stack(scores).astype('f4').tofile(path)
                    seq=bridge.finish(a.directory,label);text=tokenizer.decode(new)
                    return dict(label=label,messages=messages,input_ids=ids,output_ids=new,output_text=text,
                        normalized=normalized(text),scores=extent(path),sequence=seq,state_reset=True,
                        special_leak=any(t in set(b['special_ids'])-set(b['eos_ids']) for t in new),
                        stop_policy=not any(t in b['eos_ids'] for t in new[:-1]) and (new[-1] in b['eos_ids'] or len(new)==64))
                for case in b['cases']:
                    row=generate(case['messages'],case['id']);assert row['input_ids']==case['input_ids']
                    row.update(id=case['id'],category=case['category'],expected=case['expected'],correct=row['normalized']==case['expected'])
                    tasks.append(row);write(a.directory/(case['id']+'.task.json'),row)
                    event(operation='task',id=case['id'],correct=row['correct'])
                    if case['id']==b['followup']['case_id']:
                        messages=case['messages']+[dict(role='assistant',content=row['output_text']),dict(role='user',content=b['followup']['prompt'])]
                        followup=generate(messages,'own_answer_followup')
                        followup.update(expected=b['followup']['expected'],correct=followup['normalized']==b['followup']['expected'])
                        write(a.directory/'own_answer_followup.json',followup)
        for hook in hooks:hook.remove()
        bridge.restore();stage='after_hash'
        assert identities=={name:(id(p),p._version) for name,p in model.named_parameters()}
        after=verify_parameters(model,b,torch,np,guard);assert after==before
        write(a.directory/'parameter_after.json',after);check_inputs(b,guard)
        assert len(records)==24 and len(controls)==16
        fit=fit_quality(records) if a.phase=='fit' else None
        q=quality(tasks,followup,records,controls,b['gates']) if a.phase=='evaluation' else None
        sequences=[q['sequence'] for q in records+tasks+([followup] if followup else [])]
        sequences += [q[k] for q in controls for k in ('wrapper','continuation')]
        bound=b['processed_bounds']['FIT_processed_tokens' if a.phase=='fit' else 'evaluation_processed_tokens']
        processed=sum(s['seen'] for s in sequences);assert processed<=bound
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        result=dict(schema='FIXED_DEPTH_QUALITY_RESULT_V1',binding=extent(a.binding),phase=a.phase,depth=a.depth,
            selection=extent(a.selection) if selection else None,nonselected_defect_diagnostic=selection['nonselected_defect_diagnostic'] if selection else False,
            records=records,prefix_controls=controls,tasks=tasks,followup=followup,FIT_quality=fit,quality=q,
            parent_24_exact=bool(parent_exact) if a.depth==24 else None,processed_tokens=processed,
            all411_source_parameters_exact=True,before_after_hashes_exact=True,outputs=outputs,
            optimizer_updates=0,RESERVED_queries=0,omitted_source_operator_calls=0,timing_admitted=False,
            seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            output_bytes=sum(e['bytes'] for e in outputs),scope=b['scope'],custody=b['custody'])
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out))
        event(operation='completed',FIT_quality=fit,quality=q)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage,
                       completed_histories=len(records),completed_controls=len(controls));raise
    finally:
        if bridge is not None:
            if bridge.streams is not None:
                for s in bridge.streams:
                    if s is not None:s.close()
                bridge.streams=None
            bridge.restore()
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.out.with_suffix('.worker_terminal.json'),receipt)


def launch(a):
    import psutil
    from original_falcon_whole_recovery import memory_reader
    b=load(a.binding);assert extent(a.binding)['sha256']==a.binding_sha and b['limits']==LIMITS
    assert not a.directory.exists() and not a.out.exists()
    check_inputs(b);env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0')
    argv=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'worker','--phase',a.phase,'--depth',str(a.depth),
          '--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
    if a.selection:argv+=['--selection',str(a.selection.resolve())]
    proc=psutil.Process();proc.cpu_affinity([11]);reader=memory_reader();start=time.monotonic()
    receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),worker_argv=argv,binding=extent(a.binding))
    log=a.out.with_suffix('.worker.log');assert not log.exists();child=None;peak=0;offset=0
    try:
        with log.open('xb') as f:
            child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=8)
            receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time())
            write(a.out.with_suffix('.owned_process.json'),receipt)
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-start<=LIMITS['seconds'],'holder deadline'
                assert peak+proc.memory_info().peak_wset<=LIMITS['OS_bytes'],'combined OS cap'
                with log.open('rb') as s:
                    s.seek(offset);data=s.read();end=data.rfind(b'\n')+1
                    if end:print(data[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.5)
        peak=max(peak,reader(child));receipt['worker_exit_code']=child.returncode
        assert child.returncode==0,log.read_text(encoding='utf8',errors='replace')[-4000:]
        check_inputs(b);receipt.update(exit_code=0,error=None,result=extent(a.out),
                                      worker_terminal=extent(a.out.with_suffix('.worker_terminal.json')))
    except BaseException as error:
        if child is not None and child.poll() is None:
            assert abs(psutil.Process(child.pid).create_time()-receipt['worker_creation_time'])<.001
            child.terminate();child.wait(timeout=15)
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),worker_exit_code=None if child is None else child.returncode);raise
    finally:
        receipt.update(seconds=time.monotonic()-start,worker_OS_peak=peak,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','launch','worker','select'])
    for k in ('binding','directory','out','selection'):parser.add_argument('--'+k,type=Path)
    parser.add_argument('--freeze');parser.add_argument('--binding-sha');parser.add_argument('--phase',choices=['fit','evaluation'])
    parser.add_argument('--depth',type=int);a=parser.parse_args()
    if a.mode in ('worker','launch'):{'worker':worker,'launch':launch}[a.mode](a)
    else:
        import psutil
        proc=psutil.Process();start=time.monotonic();t=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
        try:
            assert a.out and not a.out.exists();{'prepare':prepare,'select':select}[a.mode](a)
            assert time.monotonic()-start<=2700 and proc.memory_info().peak_wset<=8<<30
            t.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as error:t.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
        finally:
            t.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
            write(a.out.with_suffix('.preparation_terminal.json' if a.mode=='prepare' else '.terminal.json'),t)
