"""Packed-only target export and first original-kernel evolving-state C gate."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def bind(a):
    pilot=DOC/'chatbot_hybrid_pilot_result_repair1_20261008.json'
    result=json.loads(pilot.read_bytes())
    assert result['decision']=='PILOT_RECOVERY_PASS'
    files=[pilot,pilot.with_suffix('.terminal.json'),Path(result['checkpoint']['path']),B/'chatbot_hybrid_native.py',
        B/'chatbot_hybrid_native.c',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
        ROOT/'benchmarks/phase60/engine.c',DOC/'CHATBOT_HYBRID_NATIVE_PROTOCOL_20261008.md',Path(sys.executable)]
    if a.phase=='native':
        assert a.export_result
        exported=json.loads(a.export_result.read_bytes())
        assert exported['schema']=='HYBRID_PACKED_EXPORT_RESULT_V1'
        files += [a.export_result,a.export_result.with_suffix('.terminal.json'),Path(exported['model']['path'])]
        for row in result['after']['cases']:
            files.append(Path(row['logits']['path']))
        files.append(a.compiler)
        files += [a.compiler.parent/n for n in ('clang-21.exe','ld.lld.exe','lld.exe','libclang-cpp.dll','libLLVM.dll') if (a.compiler.parent/n).exists()]
        files.append(Path(os.environ['SystemRoot'])/'System32/conhost.exe')
    foreign={'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
        'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
        'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    assert all(sha(ROOT/p)==s for p,s in foreign.items())
    files += [ROOT/p for p in foreign]
    if a.resume:
        assert (a.resume/'first_failure.json').exists() or (a.resume/'external_failure.json').exists()
        files += [p for p in a.resume.iterdir() if p.is_file()]
    files=list(dict.fromkeys(p.resolve() for p in files))
    inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    assert next(v['sha256'] for v in inputs if v['path']==result['checkpoint']['path'])==result['checkpoint']['sha256']
    limits=dict(seconds=600,OS_bytes=(8 if a.phase=='export' else 2)<<30,
                output_bytes=(1<<30) if a.phase=='export' else (256<<20),GPU_allocated_bytes=512<<20,GPU_reserved_bytes=1<<30)
    b=dict(schema='HYBRID_NATIVE_BINDING_V1',phase=a.phase,python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),pilot_path=str(pilot.resolve()),
        compiler=str(a.compiler.resolve()) if a.compiler else None,export_result=str(a.export_result.resolve()) if a.export_result else None,
        resume_directory=str(a.resume.resolve()) if a.resume else None,limits=limits,inputs=inputs,
        allowed_worker_children=['clang.exe','clang-21.exe','ld.lld.exe','lld.exe','ld.exe','hybrid_native.exe','conhost.exe'] if a.phase=='native' else [],
        allowed_system_child_path=str((Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve()),
        runtime_binding_scope='Actual checkpoint/export/cases and selected code/compiler/Python extents; isolated versions/paths. Not a complete DLL or compiler-library tree hash.')
    write(a.out,b)
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs))),flush=True)


def worker(a):
    start=time.monotonic()
    assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes())
    assert b['schema']=='HYBRID_NATIVE_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    a.directory.mkdir(exist_ok=False)
    old=Path(b['resume_directory']) if b.get('resume_directory') else None
    import psutil
    proc=psutil.Process()
    proc.cpu_affinity(list(range(6)))
    child_peaks=[]
    torch=None
    def cached(name):
        return old/name if old and (old/name).is_file() else a.directory/name
    def guard():
        lim=b['limits']
        assert time.monotonic()-start<=lim['seconds'],'worker deadline'
        assert proc.memory_info().peak_wset+sum(child_peaks)<=lim['OS_bytes'],'family OS cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=lim['output_bytes'],'output cap'
        if torch is not None:
            assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes']
            assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
    def event(stage,**kw):
        print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**kw)),flush=True)
        guard()
    def extent(p):
        return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
    def save_raw(p,data):
        with p.open('xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
    result=json.loads(Path(b['pilot_path']).read_bytes())
    try:
        if b['phase']=='export':
            import struct
            import torch as t
            torch=t
            assert torch.__version__=='2.6.0+cu124' and torch.cuda.is_available()
            torch.set_num_threads(6);torch.set_num_interop_threads(1)
            packet=torch.load(result['checkpoint']['path'],map_location='cpu',weights_only=True)
            assert packet['target_schema']=='COMPACT_FALCON_TERNARY_TARGET_V1'
            state=packet['model'];del packet
            assert len(state)==211 and sum(v.numel() for v in state.values())==254932736
            event('checkpoint_loaded',parameters=254932736)
            state['rope_inv_freq']=1/(1e11**(torch.arange(0,128,2).float()/128))
            fields=[];new_fields=0;payload=0
            for index,(name,w) in enumerate(state.items()):
                path=cached(f'field_{index:03d}.bin');rec_path=cached(f'field_{index:03d}.json')
                if rec_path.exists():
                    rec=json.loads(rec_path.read_bytes());assert rec['name']==name and sha(rec['path'])==rec['sha256']
                else:
                    assert not path.exists(),'incomplete field requires a separate missing-extent repair'
                    assert w.dtype==torch.float32 and torch.isfinite(w).all()
                    if name.rsplit('.',1)[-1] in ('gate','up','down'):
                        scale=state[name+'_scale'];assert (scale>=1e-8).all()
                        q=torch.round(w.to('cuda')/scale.to('cuda')[:,:,None]).clamp(-1,1).to(torch.int8)
                        code=((q[:,:,0::2]+1)*3+(q[:,:,1::2]+1)).permute(0,2,1).contiguous()
                        decoded=code.permute(0,2,1)
                        assert torch.equal(decoded//3-1,q[:,:,0::2]) and torch.equal(decoded%3-1,q[:,:,1::2])
                        raw=code.cpu().numpy().tobytes();dtype=2;shape=list(code.shape)
                        del q,code,decoded
                    else:
                        raw=w.numpy().astype('<f4',copy=False).tobytes();dtype=1;shape=list(w.shape)
                    save_raw(path,raw)
                    rec=dict(name=name,dtype=dtype,shape=shape,**extent(path));write(rec_path,rec);new_fields+=1
                    del raw
                fields.append(rec);payload+=rec['bytes']
                if index%24==0:event('fields',complete=index+1,total=212)
            assert len(fields)==212 and payload==425188608
            model=a.directory/'model.bin'
            prefix=80+212*104;cursor=prefix;table=[]
            for rec in fields:
                assert len(rec['name'].encode())<64 and cursor%4==0
                table.append(struct.pack('<64sII4IQQ',rec['name'].encode(),rec['dtype'],len(rec['shape']),
                          *(rec['shape']+[0]*(4-len(rec['shape']))),cursor,rec['bytes']))
                cursor+=rec['bytes']
            header=struct.pack('<8s16IQ',b'SLH1PK01',1,512,12,65537,72,8,128,768,256,48,16,4,128,(1<<5)|(1<<11),212,0x01020304,cursor)
            with model.open('xb') as f:
                f.write(header);f.write(b''.join(table))
                for rec in fields:
                    with Path(rec['path']).open('rb') as source:
                        for chunk in iter(lambda:source.read(1<<20),b''):f.write(chunk)
                f.flush();os.fsync(f.fileno())
            assert model.stat().st_size==cursor==425210736
            manifest=dict(schema='HYBRID_PACKED_FORMAT_V1',header_bytes=80,field_record_bytes=104,fields=fields,
                payload_bytes=payload,model=extent(model),pair_layout='expert/pair/row tile-major; byte values0..8',
                quantization='CUDA F32 round(master/row_scale), clip[-1,1], same primitive/device as learner; reversible pair codes',
                master_or_unpacked_expert_in_model_bytes=0,extra_inference_rope_bytes=256)
            write(a.directory/'manifest.json',manifest)
            guard()
            report=dict(schema='HYBRID_PACKED_EXPORT_RESULT_V1',decision='PACKED_EXPORT_COMPLETE',freeze=a.freeze,
                binding_sha256=a.binding_sha,process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
                model=manifest['model'],manifest=extent(a.directory/'manifest.json'),payload_bytes=payload,
                expert_code_bytes=84934656,expert_master_or_unpacked_bytes=0,fields=212,new_fields=new_fields,
                checkpoint_sha256=result['checkpoint']['sha256'],GPU_allocated_peak=torch.cuda.max_memory_allocated(),
                GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,
                model_forwards=0,source_generations=0,training_updates=0,elapsed_seconds=time.monotonic()-start)
        else:
            import struct
            import numpy as np
            from chatbot_falcon_usability_launch import Memory
            assert np.__version__=='2.4.6'
            def run(argv,stem):
                logfile=a.directory/(stem+'.log');receipt=a.directory/(stem+'.process.json')
                getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
                getmem.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory),ctypes.c_uint32];getmem.restype=ctypes.c_int
                peak=0;p=None;begin=time.monotonic()
                def memory():
                    nonlocal peak
                    m=Memory();m.cb=ctypes.sizeof(m)
                    assert getmem(ctypes.c_void_p(int(p._handle)),ctypes.byref(m),m.cb)
                    peak=max(peak,m.PeakWorkingSetSize)
                with logfile.open('xb') as f:
                    p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
                    try:
                        while p.poll() is None:
                            memory();guard()
                            for child in proc.children(recursive=True):
                                try:
                                    name=child.name().lower()
                                    assert name in b['allowed_worker_children'],name
                                    if name=='conhost.exe':
                                        assert Path(child.exe().removeprefix('\\\\?\\')).resolve()==Path(b['allowed_system_child_path']).resolve()
                                    assert proc.memory_info().peak_wset+sum(child_peaks)+peak+child.memory_info().rss<=b['limits']['OS_bytes']
                                except psutil.NoSuchProcess:
                                    continue
                            assert proc.memory_info().peak_wset+sum(child_peaks)+peak<=b['limits']['OS_bytes']
                            time.sleep(.1)
                    except BaseException:
                        if p.poll() is None:p.kill();p.wait()
                        raise
                    finally:
                        memory();child_peaks.append(peak)
                        rec=dict(command=argv,pid=p.pid,exit_code=p.returncode,OS_peak_through_exit=peak,
                            seconds=time.monotonic()-begin,log=extent(logfile));write(receipt,rec)
                assert p.returncode==0,logfile.read_text(errors='replace')
                return rec
            original=(ROOT/'benchmarks/phase60/engine.c').read_text()
            functions=['hsum256','dotf','matvec','silu','softplus','acc_add_i8x32','matvec_lut_full','build_lut_t3','bc_tm','ref_t3','quant_i8']
            fragments=[]
            for name in functions:
                import re
                match=re.search(r'^static[^\n]*\b'+name+r'\(',original,re.M);assert match,name
                pos=original.index('{',match.start());depth=1;end=pos+1
                while depth:
                    if original[end]=='{':depth+=1
                    elif original[end]=='}':depth-=1
                    end+=1
                fragment=original[match.start():end];fragments.append(fragment)
            header=a.directory/'chatbot_hybrid_original_kernels.h'
            save_raw(header,('\n\n'.join(fragments)+'\n').encode())
            write(a.directory/'original_kernel_binding.json',dict(source_sha256=sha(ROOT/'benchmarks/phase60/engine.c'),
                  functions=[dict(name=n,body_sha256=hashlib.sha256(f.encode()).hexdigest()) for n,f in zip(functions,fragments)],header=extent(header)))
            exe=a.directory/'hybrid_native.exe'
            compile_record=run([b['compiler'],'-O3','-mavx2','-mfma','-ffp-contract=off',str(B/'chatbot_hybrid_native.c'),
                              '-I',str(a.directory.resolve()),'-o',str(exe.resolve()),'-lm'],'compile')
            event('compiled',exe_sha256=sha(exe))
            queries=a.directory/'queries.bin';metadata=[];parts=[struct.pack('<4I',0x31514853,6,32,65537)]
            for r in result['supervision_records']:
                ids=r['input_ids']+r['output_ids'][:-1];positions=list(range(len(r['input_ids'])-1,len(ids)))
                parts += [struct.pack('<2I',len(ids),len(positions)),struct.pack('<'+str(len(ids))+'I',*ids),struct.pack('<'+str(len(positions))+'I',*positions)]
                metadata.append(dict(id=r['id'],split=r['split'],input_ids=ids,positions=positions))
            save_raw(queries,b''.join(parts));write(a.directory/'queries.json',metadata)
            native_paths=[a.directory/p for p in ('logits.bin','trace.bin','states.bin','native.json')]
            exported=json.loads(Path(b['export_result']).read_bytes())
            native_record=run([str(exe.resolve()),exported['model']['path'],str(queries.resolve()),
                            *[str(p.resolve()) for p in native_paths]],'native')
            event('native_complete',child_seconds=native_record['seconds'])
            raw=native_paths[0].read_bytes();assert struct.unpack_from('<4I',raw)==(0x314c4853,6,32,65537)
            logits=np.frombuffer(raw,offset=16,dtype='<f4').reshape(32,65537);assert np.isfinite(logits).all()
            rows=[];offset=0
            for case in metadata:
                reference=next(v for v in result['after']['cases'] if v['id']==case['id'])
                assert sha(reference['logits']['path'])==reference['logits']['sha256']
                ref=np.fromfile(reference['logits']['path'],dtype='<f4').reshape(reference['logits']['shape'])
                actual=logits[offset:offset+len(ref)];delta=actual.astype(np.float64)-ref
                rms=np.sqrt((delta*delta).sum(-1)/np.square(ref.astype(np.float64)).sum(-1))
                winners=actual.argmax(-1);expected=ref.argmax(-1)
                for j in range(len(ref)):
                    rows.append(dict(id=case['id'],position=case['positions'][j],relative_logit_RMS=float(rms[j]),
                        C_ID=int(winners[j]),learner_ID=int(expected[j]),winner_equal=bool(winners[j]==expected[j]),
                        RMS_gate=bool(rms[j]<=1e-4)))
                offset+=len(ref)
            assert offset==32
            dtype=np.dtype([(n,'<f4',512) for n in ('input','core_input','core_output','ff_input')]+[
                ('scores','<f4',72),('ids','<u4',8),('mass','<f4',8),('ff_output','<f4',512),('output','<f4',512)])
            trace=np.fromfile(native_paths[1],dtype=dtype);assert dtype.itemsize==12640 and len(trace)==261*12
            for name in dtype.names:
                assert np.isfinite(trace[name]).all()
            selected=np.argsort(-trace['scores'],axis=-1,kind='stable')[:,:8].astype(np.uint32)
            assert np.array_equal(selected,trace['ids'])
            chosen=np.take_along_axis(trace['scores'].astype(np.float64),selected.astype(np.int64),-1)
            mass=np.exp(chosen-chosen.max(-1,keepdims=True));mass/=mass.sum(-1,keepdims=True)
            mass_error=float(np.abs(mass-trace['mass']).max());assert mass_error<=1e-6
            final=np.fromfile(native_paths[2],dtype='<f4');assert final.size==6*9117696//4 and np.isfinite(final).all()
            stats=json.loads(native_paths[3].read_bytes())
            assert stats['expert_master_or_unpacked_bytes']==0 and stats['coefficient_payload_bytes']==425188608
            gates=dict(all32_full_logit_RMS=all(r['RMS_gate'] for r in rows),all32_winners=all(r['winner_equal'] for r in rows),
                       packed_only=True,original_kernel_fixtures=stats['original_kernel_fixtures'],flat_router_ID_mass=True)
            guard()
            report=dict(schema='HYBRID_NATIVE_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
                process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
                decision='NATIVE_PREFIX_PARITY_PASS' if all(gates.values()) else 'NATIVE_PREFIX_PARITY_FAIL',gates=gates,
                rows=rows,max_relative_logit_RMS=max(r['relative_logit_RMS'] for r in rows),
                winners_equal=sum(r['winner_equal'] for r in rows),router_records=len(trace),max_router_mass_abs_error=mass_error,
                native_stats=stats,compile_process=compile_record,native_process=native_record,
                model=exported['model'],exe=extent(exe),queries=extent(queries),native_outputs=[extent(p) for p in native_paths],
                worker_OS_peak_snapshot=proc.memory_info().peak_wset,children_OS_peaks=child_peaks,
                model_python_forwards=0,source_generations=0,training_updates=0,
                quality_speed_scope='Fixed source-owned prefixes against saved learner. No fresh own-history quality, accepted50, window eviction or large-n routing qualification.',
                elapsed_seconds=time.monotonic()-start)
        write(a.out,report);event('complete',decision=report['decision'])
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),elapsed_seconds=time.monotonic()-start,children_OS_peaks=child_peaks))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--phase',choices=('export','native'))
    p.add_argument('--compiler',type=Path)
    p.add_argument('--export-result',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--binding',type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
