"""Capture every remaining short-stage adopted FIT/DEV donor packet once."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from fixed_depth_quality import load,check_inputs
PROTOCOL=DOC/'CHATBOT_SOURCE_COVERAGE_PROTOCOL_20261015.md'
RUNTIME=DOC/'fixed_depth_quality_io_binding_20261014.json'
ADOPTION=DOC/'chatbot_broad_data_result_repair1_20261009.json'
PRIOR_BINDING=DOC/'chatbot_broad_pilot_binding_repair1_20261009.json'
AUDIT=DOC/'chatbot_balanced_replay_audit_20261015.json'
BINDING=DOC/'chatbot_source_coverage_binding_20261015.json'
REPORT=DOC/'chatbot_source_coverage_result_20261015.json'
PLAN=ROOT/'results/native_expert_scaling/chatbot_source_coverage_plan_20261015'
FAMILY=Path('E:/SiliconLLM/results/native_expert_scaling/chatbot_source_coverage_20261015')
LIMITS=dict(seconds=21600,GPU_seconds=18000,reserve_seconds=180,OS_bytes=16<<30,
            GPU_allocated_bytes=8<<30,GPU_reserved_bytes=10<<30,output_bytes=32<<30,CPU_threads=6)


def choose(data,used):
    remaining=[r for r in data['records'] if r['split'] in ('FIT','DEV') and r['id'] not in used]
    cases=sorted((r for r in remaining if r['input_length']<=2048),key=lambda r:(r['source'],r['split'],r['id']))
    deferred=[dict(id=r['id'],split=r['split'],source=r['source'],input_length=r['input_length']) for r in remaining if r['input_length']>2048]
    assert len(cases)==432 and Counter(r['split'] for r in cases)==Counter(FIT=360,DEV=72)
    assert len({r['id'] for r in cases})==len({r['group_key'] for r in cases})==432
    assert set(Counter(r['source'] for r in cases).values())=={36}
    assert all(Counter(r['split'] for r in cases if r['source']==name)==Counter(FIT=30,DEV=6) for name in {r['source'] for r in cases})
    assert len(deferred)==40 and all(r['source']=='longalign' for r in deferred)
    assert not any(r['split']=='RESERVED' for r in cases)
    longest=min(cases,key=lambda r:(-r['input_length'],r['id']))
    cases=[longest]+[r for r in cases if r['id']!=longest['id']]
    return cases,deferred


def prepare(freeze):
    assert not PLAN.exists() and not FAMILY.exists() and not BINDING.exists()
    adopted=load(ADOPTION);assert adopted['decision']=='BROAD_DATA_ADOPTION_PASS'
    assert extent(adopted['corpus']['path'])==adopted['corpus']
    audit=load(AUDIT);terminal=load(AUDIT.with_suffix('.terminal.json'))
    assert terminal['exit_code']==0 and terminal['result']==extent(AUDIT)
    assert audit['all_stored_checks_pass'] and audit['quality_decision']=='KNOWN_PREFIX_GATES_FAIL'
    data=load(adopted['corpus']['path']);old=load(PRIOR_BINDING);previous=load(old['corpus'])['records']
    used={r['id'] for r in previous};cases,deferred=choose(data,used)
    assert not {r['group_key'] for r in cases}&{r['group_key'] for r in previous}
    for path in (Path(__file__),PROTOCOL):
        blob=subprocess.check_output(['git','show',freeze+':'+path.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==path.read_bytes().replace(b'\r\n',b'\n')
    runtime=load(RUNTIME);initial=load(DOC/'chatbot_balanced_replay_binding_20261015.json')
    controls=[next(r for r in previous if r['id']==name) for name in initial['control_ids']]
    assert all(r['split']=='FIT' for r in controls)
    PLAN.mkdir()
    plan=PLAN/'cases.json'
    write(plan,dict(schema='ALL_REMAINING_SHORT_SOURCE_CASES_V1',parent=adopted['corpus'],cases=cases,
        source_repo=data['source_repo'],source_revision=data['source_revision'],
        deferred=deferred,used_ids=sorted(used),FIT_cases=360,DEV_cases=72,RESERVED_queries=0,
        selection='All remaining adopted FIT/DEV prefixes<=2048; longest first, then source/split/ID. No scores or length cropping.',
        max_new_tokens=256,context_limit=2304,external_final_answers_are_labels=False))
    pins=runtime['inputs']+[extent(RUNTIME)]
    paths=[Path(__file__),PROTOCOL,ADOPTION,PRIOR_BINDING,AUDIT,AUDIT.with_suffix('.terminal.json'),
        Path(adopted['corpus']['path']),Path(old['corpus']),plan,
        B/'chatbot_falcon_ssd_tiles.py',B/'true_generator_input_capture_v3.py',
        DOC/'chatbot_balanced_replay_binding_20261015.json']
    paths.extend(Path(r['logits']['path']) for r in controls)
    pins.extend(extent(p) for p in dict.fromkeys(paths))
    check_inputs(dict(inputs=pins))
    write(BINDING,dict(schema='SOURCE_COVERAGE_BINDING_V1',freeze=freeze,inputs=pins,limits=LIMITS,
        source=runtime['source'],runtime=extent(RUNTIME),plan=extent(plan),controls=controls,
        family=str(FAMILY),worker=str(Path(__file__).resolve()),FIT_cases=360,DEV_cases=72,new_generations=432,
        maximum_labels=(432+2)*256,maximum_packet_bytes=(432+2)*256*65537*2,
        scope='New donor-output supervision only; no learner forward/training, RESERVED, longalign or transfer/rate admission.'))
    print(json.dumps(dict(binding=extent(BINDING),cases=432,FIT=360,DEV=72,deferred=40)),flush=True)


def worker(binding_sha):
    import numpy as np
    import psutil
    import torch
    import transformers
    from transformers import AutoTokenizer,FalconH1ForCausalLM,StoppingCriteria,StoppingCriteriaList
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    from chatbot_falcon_ssd_tiles import install
    from true_generator_input_capture_v3 import verify_parameters
    assert extent(BINDING)['sha256']==binding_sha and not FAMILY.exists() and not REPORT.exists()
    b=load(BINDING);assert b['limits']==LIMITS
    assert (np.__version__,torch.__version__,transformers.__version__)==('2.4.6','2.6.0+cu124','5.13.1')
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));torch.set_num_threads(6);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();gpu_start=None;gpu_end=None;last_children=last_inventory=0.;stage='input_hashes';records=[];active=None
    receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    def guard():
        nonlocal last_children,last_inventory
        now=time.monotonic();assert now-start<=LIMITS['seconds']-LIMITS['reserve_seconds'],'total deadline'
        assert proc.memory_info().peak_wset<=LIMITS['OS_bytes'],'OS cap'
        if now-last_children>=1:
            assert not proc.children(recursive=True),'unexpected child';last_children=now
        if FAMILY.exists() and now-last_inventory>=1:
            assert sum(p.stat().st_size for p in FAMILY.iterdir() if p.is_file())<=LIMITS['output_bytes'],'output cap';last_inventory=now
        if gpu_start is not None and gpu_end is None:
            assert now-gpu_start<=LIMITS['GPU_seconds']-180,'GPU deadline'
            assert torch.cuda.max_memory_allocated()<=LIMITS['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=LIMITS['GPU_reserved_bytes'],'GPU reserved cap'
    def event(**fields):
        guard();print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True)
    try:
        check_inputs(b,guard);runtime=load(RUNTIME);plan=load(b['plan']['path'])
        assert not torch.cuda.is_initialized() and len(plan['cases'])==432
        gpu=subprocess.check_output(['C:/Windows/System32/nvidia-smi.exe','--query-gpu=uuid,driver_version,memory.free',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',');expected=runtime['GPU_query'].split(',')
        assert gpu[0].strip()==expected[0].strip() and gpu[1].strip()==expected[2].strip() and int(gpu[-1])>=9216
        tokenizer=AutoTokenizer.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False)
        assert len(tokenizer)==65537 and tokenizer.bos_token_id==17
        for case in plan['cases']+b['controls']:
            assert case['messages'][-1]['role']=='user'
            ids=tokenizer.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
            assert ids==case['input_ids'] and len(ids)==case['input_length']<=2048
        FAMILY.mkdir();stage='source_load';gpu_start=time.monotonic()
        method=install(code);assert method==load(runtime['prior_source_method']['path'])
        write(FAMILY/'source_method.json',method)
        model=FalconH1ForCausalLM.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        assert not code.is_fast_path_available and model.config.mamba_chunk_size==128
        assert model.generation_config.eos_token_id==[11,228]
        identity={name:(id(p),p._version) for name,p in model.named_parameters()}
        before=verify_parameters(model,runtime,torch,np,guard);write(FAMILY/'parameter_before.json',before)
        def mup_check():
            assert all(layer.mamba.mup_vector[0,0].detach().cpu().view(torch.uint16).tolist()==runtime['MuP_bits'] for layer in model.model.layers)
        mup_check()
        class Budget(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                guard();return torch.zeros(input_ids.shape[0],dtype=torch.bool,device=input_ids.device)
        def packet(case,label,control=False):
            begin=time.monotonic();guard()
            with torch.inference_mode():
                inputs=torch.tensor([case['input_ids']],device='cuda')
                generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=256,
                    do_sample=False,use_cache=True,return_dict_in_generate=True,output_logits=True,
                    stopping_criteria=StoppingCriteriaList([Budget()]))
                outputs=generated.sequences[0,len(case['input_ids']):].tolist()
                logits=torch.stack(generated.logits).squeeze(1);encoded=logits.to(torch.bfloat16)
                assert list(logits.shape)==[len(outputs),65537] and 1<=len(outputs)<=256
                assert torch.isfinite(logits).all() and torch.equal(encoded.float(),logits.float())
                assert logits.argmax(-1).tolist()==outputs and not any(t in (11,228) for t in outputs[:-1])
                assert outputs[-1] in (11,228) or len(outputs)==256
                raw=FAMILY/(label+'.logits.bf16')
                with raw.open('xb') as stream:
                    stream.write(encoded.contiguous().view(torch.uint16).cpu().numpy().astype('<u2',copy=False).tobytes());stream.flush();os.fsync(stream.fileno())
                record=dict(id=case['id'],split=case['split'],domain=case.get('source',case.get('domain')),
                    messages=case['messages'],input_ids=case['input_ids'],input_length=len(case['input_ids']),
                    group_key=case['group_key'],prefix_sha256=case['prefix_sha256'],
                    output_ids=outputs,output_text=tokenizer.decode(outputs,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                    stop='EOS' if outputs[-1] in (11,228) else 'length',
                    student_input_ids=case['input_ids']+outputs[:-1],positions=list(range(len(case['input_ids'])-1,len(case['input_ids'])+len(outputs)-1)),
                    logits=dict(**extent(raw),shape=list(logits.shape),dtype='BF16'),raw_logits_dtype=str(logits.dtype),
                    BF16_lossless=True,raw_argmax_equals_generated=True,finite=True,control=control,
                    upstream=dict(repo=plan['source_repo'],revision=plan['source_revision'],source=case.get('source',case.get('domain')),
                                  parquet_path=case.get('parquet_path',case.get('upstream',{}).get('parquet_path')),
                                  row_index=case.get('row_index',case.get('upstream',{}).get('row_index'))),
                    context_origin='Public prior assistant turns retained; new reply generated by immutable donor.',
                    source_packet_seconds=time.monotonic()-begin)
                write(FAMILY/(label+'.json'),record)
            return record
        stage='source_controls';controls=[]
        for case in b['controls']:
            active=case['id'];row=packet(case,'control.'+case['id'],True)
            assert row['output_ids']==case['output_ids'] and row['logits']['sha256']==case['logits']['sha256'],'source reproducibility control'
            controls.append(row);event(operation='source_control_exact',id=case['id'],labels=len(row['output_ids']))
        stage='new_source_packets'
        for case in plan['cases']:
            active=case['id'];row=packet(case,case['id']);records.append(row)
            event(operation='packet_complete',id=case['id'],split=case['split'],cases=len(records),labels=sum(len(r['output_ids']) for r in records))
        active=None;stage='parameter_after'
        assert identity=={name:(id(p),p._version) for name,p in model.named_parameters()}
        after=verify_parameters(model,runtime,torch,np,guard);assert before==after;mup_check()
        write(FAMILY/'parameter_after.json',after)
        del model;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated()<=64<<20,'GPU ownership release';gpu_end=time.monotonic()
        write(FAMILY/'numerical_completion.json',dict(records=records,controls=controls,source411_exact=True,
            identities_versions_exact=True,MuP_exact=True,GPU_released=True,original_posthash_complete=False))
        stage='posthash';check_inputs(b,guard)
        assert len(records)==432 and Counter(r['split'] for r in records)==Counter(FIT=360,DEV=72)
        assert [r['id'] for r in records]==[r['id'] for r in plan['cases']]
        corpus=FAMILY/'corpus.json'
        write(corpus,dict(schema='SOURCE_COVERAGE_CORPUS_V1',records=records,source=runtime['source'],
            source_binding=extent(BINDING),max_new_tokens=256,source_dtype='BF16',RESERVED_queries=0,learner_forwards=0))
        outputs=[extent(p) for p in sorted(FAMILY.iterdir()) if p.is_file()]
        assert sum(e['bytes'] for e in outputs)<=LIMITS['output_bytes']
        write(REPORT,dict(schema='SOURCE_COVERAGE_RESULT_V1',binding=extent(BINDING),plan=b['plan'],corpus=extent(corpus),
            controls=controls,cases=432,FIT=360,DEV=72,labels=sum(len(r['output_ids']) for r in records),
            outputs=outputs,source411_exact=True,identities_versions_MuP_exact=True,inputs_before_after_exact=True,
            GPU_released_before_posthash=True,GPU_phase_seconds=gpu_end-gpu_start,seconds=time.monotonic()-start,
            OS_peak=proc.memory_info().peak_wset,GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),RESERVED_queries=0,learner_forwards=0,optimizer_updates=0,
            decision='SOURCE_COVERAGE_CAPTURE_COMPLETE',quality_admissible=False,scope=b['scope']))
        receipt.update(exit_code=0,error=None,result=extent(REPORT))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),stage=stage,active_case=active,completed_packets=len(records))
        raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(REPORT.with_suffix('.worker_terminal.json'),receipt)


def launch(binding_sha):
    import psutil
    from original_falcon_whole_recovery import memory_reader
    assert extent(BINDING)['sha256']==binding_sha and not FAMILY.exists() and not REPORT.exists()
    b=load(BINDING);check_inputs(b)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0')
    argv=[sys.executable,'-I','-S','-B','-X','utf8',str(Path(__file__).resolve()),'worker','--binding-sha',binding_sha]
    proc=psutil.Process();proc.cpu_affinity([11]);reader=memory_reader();start=time.monotonic();child=None;offset=0;peak=0
    receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),worker_argv=argv,binding=extent(BINDING))
    try:
        with REPORT.with_suffix('.worker.log').open('xb') as stream:
            child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time())
            write(REPORT.with_suffix('.owned_process.json'),receipt)
            while child.poll() is None:
                peak=max(peak,reader(child));assert time.monotonic()-start<=LIMITS['seconds'],'holder deadline'
                assert peak+proc.memory_info().peak_wset<=LIMITS['OS_bytes'],'combined OS cap'
                with REPORT.with_suffix('.worker.log').open('rb') as log:
                    log.seek(offset);data=log.read();end=data.rfind(b'\n')+1
                    if end:print(data[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.5)
        receipt['worker_exit_code']=child.returncode;assert child.returncode==0
        check_inputs(b);receipt.update(exit_code=0,error=None,result=extent(REPORT))
    except BaseException as error:
        if child is not None and child.poll() is None:
            assert abs(psutil.Process(child.pid).create_time()-receipt['worker_creation_time'])<.001
            child.terminate();child.wait(timeout=15)
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc(),worker_exit_code=None if child is None else child.returncode);raise
    finally:
        receipt.update(seconds=time.monotonic()-start,worker_OS_peak=peak,OS_peak=proc.memory_info().peak_wset)
        write(REPORT.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','launch','worker'])
    parser.add_argument('--freeze');parser.add_argument('--binding-sha');args=parser.parse_args()
    if args.mode=='prepare':
        import psutil
        proc=psutil.Process();start=time.monotonic();receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
        try:prepare(args.freeze);receipt.update(exit_code=0,error=None,result=extent(BINDING))
        except BaseException as error:receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
        finally:
            receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
            write(BINDING.with_suffix('.preparation_terminal.json'),receipt)
    elif args.mode=='launch':launch(args.binding_sha)
    else:worker(args.binding_sha)
