"""Bound native entry/chat/complete decode cost probe; no quality admission or fit."""
import argparse
import array
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))
FOREIGN={'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
 'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
 'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}


def bind(a):
    old_dir=ROOT/'results/native_expert_scaling/chatbot_hybrid_native_repair3_20261008'
    kernel=json.loads((old_dir/'original_kernel_binding.json').read_bytes())
    original=(ROOT/'benchmarks/phase60/engine.c').read_text()
    for row in kernel['functions']:
        match=re.search(r'^static[^\n]*\b'+row['name']+r'\(',original,re.M);assert match
        end=original.index('{',match.start())+1;depth=1
        while depth:
            depth+=(original[end]=='{')-(original[end]=='}');end+=1
        assert hashlib.sha256(original[match.start():end].encode()).hexdigest()==row['body_sha256']
    header=Path(kernel['header']['path']);assert sha(header)==kernel['header']['sha256']
    old_binding=json.loads((DOC/'chatbot_hybrid_native_binding_repair3_20261008.json').read_bytes())
    compiler=Path(old_binding['compiler'])
    export=DOC/'chatbot_hybrid_export_result_20261008.json';packed=json.loads(export.read_bytes())
    terminal=json.loads(export.with_suffix('.terminal.json').read_bytes())
    assert packed['decision']=='PACKED_EXPORT_COMPLETE' and terminal['exit_code']==0 and terminal['result_sha256']==sha(export)
    model=Path(packed['model']['path']);assert sha(model)==packed['model']['sha256']
    source=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    corpus=ROOT/'results/native_expert_scaling/chatbot_hybrid_transfer_capture_20261009/corpus.json'
    files=[Path(__file__),B/'chatbot_hybrid_engine_entry.c',B/'chatbot_hybrid_engine_chat.py',B/'chatbot_hybrid_native.c',
           B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',ROOT/'benchmarks/phase60/engine.c',
           DOC/'CHATBOT_HYBRID_ENGINE_PROBE_PROTOCOL_20261009.md',old_dir/'original_kernel_binding.json',header,
           export,export.with_suffix('.terminal.json'),model,corpus,Path(sys.executable),compiler,
           SITE/'tokenizers/__init__.py',SITE/'tokenizers/tokenizers.pyd',SITE/'jinja2/__init__.py']
    files += [source/n for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json')]
    files += [compiler.parent/n for n in ('clang-21.exe','ld.lld.exe','lld.exe','libclang-cpp.dll','libLLVM.dll') if (compiler.parent/n).exists()]
    files += [ROOT/p for p in FOREIGN]
    assert all(sha(ROOT/p)==s for p,s in FOREIGN.items())
    conhost=Path(os.environ['SystemRoot'])/'System32/conhost.exe';files.append(conhost)
    assert json.loads((source/'generation_config.json').read_bytes())['eos_token_id']==[11,228]
    binding=dict(schema='HYBRID_ENGINE_PROBE_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
      compiler=str(compiler),header_directory=str(old_dir),model=packed['model'],source_directory=str(source),corpus_path=str(corpus),
      selected_case_ids=['fit_history_00','fit_instruction_00'],requests=5,max_generation_lengths=[64,0,64,32,32],
      followup='Repeat your last answer in one short sentence.',threads=1,limits=dict(seconds=150,OS_bytes=4<<30,output_bytes=8<<20),
      allowed_worker_children=['clang.exe','clang-21.exe','ld.lld.exe','lld.exe','ld.exe','hybrid_engine.exe','conhost.exe'],
      allowed_system_child_path=str(conhost.resolve()),
      runtime_binding_scope='Packed artifact, original kernel body hashes, entry/native/client/code/protocol, tokenizer/template, selected Python/Rust/compiler extents; no full DLL tree.',
      inputs=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in dict.fromkeys(files)])
    write(a.out,binding);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    binding=json.loads(a.binding.read_bytes());assert binding['schema']=='HYBRID_ENGINE_PROBE_BINDING_V1'
    assert sys.version_info[:3]==(3,12,10) and Path(sys.executable).resolve()==Path(binding['python']).resolve()
    import psutil
    import tokenizers
    import jinja2
    from chatbot_falcon_usability_launch import Memory
    from chatbot_hybrid_engine_chat import ChatTokenizer,EngineClient
    assert psutil.__version__=='7.2.2' and tokenizers.__version__=='0.22.2'
    proc=psutil.Process();proc.cpu_affinity([0]);a.directory.mkdir(exist_ok=False)
    os.environ['SILICON_CHAT_SECONDS']='90'
    getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];getmem.restype=wintypes.BOOL
    def peak(p):
        mem=Memory();mem.cb=ctypes.sizeof(mem)
        assert getmem(wintypes.HANDLE(int(p._handle)),ctypes.byref(mem),mem.cb)
        return mem.PeakWorkingSetSize
    child_peaks=[];client=None;completed=[]
    def guard():
        assert time.monotonic()-start<binding['limits']['seconds']-15,'worker time reserve'
        extra=peak(client.process) if client else 0
        assert proc.memory_info().peak_wset+sum(child_peaks)+extra<=binding['limits']['OS_bytes'],'family OS'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=binding['limits']['output_bytes'],'output cap'
    def extent(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
    try:
        exe=a.directory/'hybrid_engine.exe';log=a.directory/'compile.log'
        command=[binding['compiler'],'-O3','-mavx2','-mfma','-ffp-contract=off','-DSILICON_FALCON_TERNARY_CHAT',
          str(ROOT/'benchmarks/phase60/engine.c'),'-I',binding['header_directory'],'-o',str(exe.resolve()),'-lm']
        begin=time.monotonic()
        with log.open('xb') as stream:
            p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000)
            try:p.wait(timeout=30)
            except BaseException:p.kill();p.wait();raise
            finally:
                compiler_peak=peak(p);child_peaks.append(compiler_peak)
                compile_record=dict(command=command,pid=p.pid,exit_code=p.returncode,seconds=time.monotonic()-begin,
                                    OS_peak_through_exit=compiler_peak,log=extent(log))
                write(a.directory/'compile.process.json',compile_record)
        assert p.returncode==0,log.read_text(errors='replace');guard()
        print(json.dumps(dict(stage='engine_compiled',exe=extent(exe))),flush=True)
        tokenizer=ChatTokenizer(binding['source_directory'])
        corpus=json.loads(Path(binding['corpus_path']).read_bytes())
        cases={r['id']:r for r in corpus['records'] if r['id'] in binding['selected_case_ids']}
        for row in cases.values():assert row['split']=='FIT' and tokenizer.encode(row['messages'])==row['input_ids']
        responses=[];raw=[];native_begin=time.monotonic()
        with (a.directory/'engine.stderr.log').open('xb') as stream:
            client=EngineClient(exe.resolve(),binding['model']['path'],stream,timeout=95)
            assert client.hello[6:9]==(425210736,84934656,9117696)
            def request(name,ids,max_new,force=False,messages=None):
                guard();row,logits=client.request(ids,max_new,force)
                data=array.array('f');data.frombytes(logits)
                assert sys.byteorder=='little' and len(data)==65537 and all(map(math.isfinite,data))
                row.update(name=name,input_token_ids=ids,messages=messages,output_text=tokenizer.decode(row['generated_ids']))
                path=a.directory/(name+'.next_logits.f32')
                with path.open('xb') as f:f.write(logits);f.flush();os.fsync(f.fileno())
                row['next_logits']=extent(path)
                row['raw_decode_ids_per_second']=len(row['generated_ids'])/row['decode_seconds'] if row['generated_ids'] else None
                row['raw_request_ids_per_second']=len(row['generated_ids'])/row['pipe_request_seconds'] if row['generated_ids'] else None
                write(a.directory/(name+'.json'),row);completed.append(name);responses.append(row);raw.append(logits)
                print(json.dumps(dict(stage='request_complete',name=name,generated=len(row['generated_ids']),
                       raw_decode_rate=row['raw_decode_ids_per_second'],reset=row['state_reset'],reused=row['reused_prefix_ids'])),flush=True)
                guard();return row
            first=cases['fit_history_00'];ids=first['input_ids'];half=len(ids)//2
            baseline=request('full_prefill',ids,64,True,first['messages'])
            request('split_prefill',ids[:half],0,True)
            split=request('appended_prefill',ids,64,False,first['messages'])
            segmentation=(baseline['generated_ids']==split['generated_ids'] and raw[0]==raw[2])
            assert split['reused_prefix_ids']==half and not split['state_reset'] and segmentation
            own=first['messages']+[dict(role='assistant',content=split['output_text']),dict(role='user',content=binding['followup'])]
            own_ids=tokenizer.encode(own)
            followup=request('own_history_followup',own_ids,32,False,own)
            cached=ids+split['generated_ids']
            safe_reuse=len(own_ids)>=len(cached) and own_ids[:len(cached)]==cached
            assert followup['reused_prefix_ids']==(len(cached) if safe_reuse else 0)
            assert followup['state_reset']==(not safe_reuse)
            last=cases['fit_instruction_00'];reset_case=request('changed_history',last['input_ids'],32,False,last['messages'])
            assert reset_case['state_reset'] and reset_case['reused_prefix_ids']==0
            client.close();native_peak=peak(client.process);child_peaks.append(native_peak)
            native_record=dict(command=client.process.args,pid=client.process.pid,exit_code=client.process.returncode,
              seconds=time.monotonic()-native_begin,OS_peak_through_exit=native_peak,hello=list(client.hello),stderr=extent(a.directory/'engine.stderr.log'))
            write(a.directory/'native.process.json',native_record);client=None;guard()
        assert not any(k in sys.modules for k in ('torch','transformers'))
        report=dict(schema='HYBRID_ENGINE_PROBE_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
          process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),decision='ENGINE_CHAT_APPARATUS_PASS',
          gates=dict(original_kernel_bodies_unchanged=True,source_template_consumed_FIT_IDs=True,split_prefill_bit_exact=segmentation,
                     canonical_own_history=True,safe_prefix_reuse_or_reset=True,full_head_and_final_ID_consumption=True,
                     EOS11_and228_contract=True,packed_only=True),requests=responses,request_count=len(responses),
          compile_process=compile_record,native_process=native_record,model=binding['model'],exe=extent(exe),
          worker_OS_peak_snapshot=proc.memory_info().peak_wset,children_OS_peaks=child_peaks,
          elapsed_seconds=time.monotonic()-start,source_calls=0,python_model_calls=0,optimizer_updates=0,GPU_calls=0,
          runtime=dict(tokenizers=tokenizers.__version__,jinja2=jinja2.__version__,psutil=psutil.__version__),
          quality_admission=False,native_numerical_admission=False,accepted50_admission=False,physical_DRAM_measured=False,
          scope='Old 8-update unqualified artifact; NEW own-history full-head raw batch1 decode. One split-prefill pair is an explicit apparatus control. No source-quality/native-parity/fresh/large-n claim.')
        write(a.out,report)
    except BaseException as error:
        if client:client.abort()
        write(a.directory/'first_failure.json',dict(fault=repr(error),completed_requests=completed,seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bind',action='store_true');p.add_argument('--binding',type=Path)
    p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();bind(a) if a.bind else worker(a)
