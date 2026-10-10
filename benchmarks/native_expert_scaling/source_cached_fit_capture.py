"""Capture missing FIT final states paired with their actual cached logits."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
from original_falcon_whole_recovery import memory_reader
sys.path.insert(0,str(SITE))
D,V=2048,65537


def bind(a):
    assert not a.binding.exists()
    parent_path=DOC/'source_cached_final_binding_repair1_sealed_20261010.json';parent=json.loads(parent_path.read_bytes())
    result_path=DOC/'source_cached_final_result_repair1_20261010.json';audit_path=DOC/'source_cached_final_stored_adjudication_repair1_20261010.json'
    audit=json.loads(audit_path.read_bytes());terminal_path=audit_path.with_suffix('.terminal.json');terminal=json.loads(terminal_path.read_bytes())
    assert audit['decision']=='PAIRED_CACHED_STATE_PASS' and audit['result']==extent(result_path)
    assert terminal['exit_code']==0 and terminal['error'] is None and terminal['result']==extent(audit_path)
    corpus_path=ROOT/'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json'
    corpus=json.loads(corpus_path.read_bytes());records=[r for r in corpus['records'] if r['split']=='FIT']
    assert len(records)==24 and sum(len(r['output_ids']) for r in records)==4422
    assert len({r['id'] for r in records})==24 and len({r['domain'] for r in records})==12
    assert all(n==2 for n in Counter(r['domain'] for r in records).values())
    files=[Path(x['path']) for x in parent['inputs']]+[parent_path,result_path,audit_path,terminal_path,corpus_path,Path(__file__),a.protocol]
    for rec in records:
        assert rec['student_input_ids']==rec['input_ids']+rec['output_ids'][:-1]
        assert rec['positions']==list(range(len(rec['input_ids'])-1,len(rec['input_ids'])+len(rec['output_ids'])-1))
        assert rec['BF16_lossless'] and rec['raw_argmax_equals_generated'] and rec['logits']['shape']==[len(rec['output_ids']),V]
        path=corpus_path.parent/(rec['id']+'.json');saved=json.loads(path.read_bytes())
        for key in ('id','split','input_ids','output_ids','positions','logits'):assert rec[key]==saved[key]
        files += [path,Path(rec['logits']['path'])]
    inputs=[extent(path) for path in dict.fromkeys(files)];by_path={x['path']:x for x in inputs}
    for item in parent['inputs']:assert by_path[item['path']]==item
    for rec in records:
        item=by_path[rec['logits']['path']];assert item['bytes']==rec['logits']['bytes'] and item['sha256']==rec['logits']['sha256']
    write(a.binding,dict(schema='SOURCE_CACHED_FIT_BINDING_V1',inputs=inputs,records=records,fields=parent['fields'],
        source=parent['source'],source_revision=parent['source_revision'],source_named_elements=parent['source_named_elements'],
        parent_qualification=extent(audit_path),source_format='Actual cached-generation final raw/postnorm BF16 states, same-call full-V logits.',
        capture_limits=dict(seconds=1500,reserve_seconds=25,OS_bytes=8<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=2<<30,log_bytes=8<<20),
        audit_limits=dict(seconds=300,OS_bytes=1<<30,output_bytes=512<<10,log_bytes=8<<20),
        expected=dict(cases=24,labels=4422,numeric_bytes=1213555992),
        gates=dict(tokens_exact=True,old_logits_exact=True,paired_norm_exact=True,paired_head_exact=True,feature_relative_RMS=.01,scalar_absolute=1.),
        scope='Missing FIT final-state observable only. No DEV generation, codec fit, optimizer/native/RESERVED calls.'))
    print(json.dumps(dict(binding=str(a.binding),sha256=sha(a.binding),inputs=len(inputs),bytes=sum(x['bytes'] for x in inputs))),flush=True)


def cpu_bf16(value):
    import torch
    assert value.dtype==torch.bfloat16
    return value.detach().cpu().contiguous().view(torch.uint16).numpy().astype('<u2',copy=False).copy()


def decode(bits):
    import numpy as np
    return (bits.astype('<u4')<<16).view('<f4').astype('f8')


def parameter_seal(model,b,guard):
    seals={}
    for name,p in model.named_parameters():
        guard();raw=cpu_bf16(p);field=b['fields'][name];digest=hashlib.sha256(raw.tobytes()).hexdigest()
        assert list(p.shape)==field['shape'] and raw.nbytes==field['bytes'] and digest==field['sha256'],name
        seals[name]=dict(identity=id(p),version=p._version,sha256=digest,bytes=raw.nbytes)
    assert set(seals)==set(b['fields']) and len(seals)==411
    assert sum(item['bytes'] for item in seals.values())==b['source_named_elements']*2
    return seals


def save(directory,name,array):
    path=directory/name
    with path.open('xb') as f:f.write(array.tobytes());f.flush();os.fsync(f.fileno())
    return dict(**extent(path),shape=list(array.shape),dtype=str(array.dtype))


def capture(a):
    import numpy as np
    import psutil
    import torch
    import transformers
    from transformers import FalconH1ForCausalLM
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    from chatbot_falcon_ssd_tiles import install
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and transformers.__version__=='5.13.1'
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.directory.mkdir(exist_ok=False);stage='source_loading';completed=[];model=None
    total_base=total_head=total_recon=generations=norm_calls=0;current=None;raw_states=[];features=[]
    def guard():
        lim=b['capture_limits'];assert time.monotonic()-start<lim['seconds']-lim['reserve_seconds'],'worker reserve'
        assert proc.memory_info().peak_wset<=lim['OS_bytes']
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        assert not proc.children(recursive=True)
        assert sum(path.stat().st_size for path in a.directory.iterdir() if path.is_file())<=lim['output_bytes']
    try:
        write(a.directory/'source_method.json',install(code))
        model=FalconH1ForCausalLM.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
        assert not code.is_fast_path_available
        assert sum(p.numel() for p in model.parameters())==b['source_named_elements']
        assert model.config.num_logits_to_keep==1 and model.generation_config.eos_token_id==[11,228]
        initial=parameter_seal(model,b,guard);write(a.directory/'parameters_before.json',initial)
        print(json.dumps(dict(stage='source_loaded',parameters=len(initial),seconds=time.monotonic()-start)),flush=True)
        for rec in b['records']:
            guard();current=rec['id'];case_start=time.monotonic();stage='cached_generation'
            raw_states=[];features=[];frames=[];calls=head_calls=0
            for name,p in model.named_parameters():assert (id(p),p._version)==(initial[name]['identity'],initial[name]['version'])
            def before_base(module,args,kwargs):
                nonlocal calls,total_base
                j=calls;calls+=1;total_base+=1;ids=kwargs['input_ids'];past=kwargs.get('past_key_values');positions=kwargs.get('position_ids')
                frames.append(dict(label_index=j,input_ids=ids[0].tolist(),shape=list(ids.shape),
                    cache_before_seq_length=0 if past is None else past.get_seq_length(),
                    position_ids=None if positions is None else positions[0].tolist()))
            def before_norm(module,args):raw_states.append(cpu_bf16(args[0][0,-1]))
            def after_norm(module,args,output):features.append(cpu_bf16(output[0,-1]));guard()
            def count_head(module,args,output):
                nonlocal head_calls,total_head
                head_calls+=1;total_head+=1
            hooks=[model.model.register_forward_pre_hook(before_base,with_kwargs=True),model.model.final_layernorm.register_forward_pre_hook(before_norm),
                model.model.final_layernorm.register_forward_hook(after_norm),model.lm_head.register_forward_hook(count_head)]
            try:
                generations+=1
                with torch.inference_mode():
                    ids=torch.tensor([rec['input_ids']],device='cuda')
                    generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,do_sample=False,use_cache=True,return_dict_in_generate=True,output_logits=True)
                    output_ids=generated.sequences[0,len(rec['input_ids']):].tolist()
                    logits=torch.stack(generated.logits).squeeze(1);encoded=logits.to(torch.bfloat16)
                    assert torch.equal(encoded.float(),logits.float()) and torch.isfinite(logits).all()
                    cached=cpu_bf16(encoded)
            finally:
                for hook in hooks:hook.remove()
            m=len(output_ids);assert 1<=m<=256 and calls==head_calls==len(raw_states)==len(features)==m
            assert logits.argmax(-1).tolist()==output_ids
            raw=np.stack(raw_states);normal=np.stack(features)
            files={key:save(a.directory,current+'.'+key+'.bf16',arr) for key,arr in (('h24',raw),('normalized',normal),('logits',cached))}
            frame_path=a.directory/(current+'.frames.json');write(frame_path,frames)
            stage='paired_reconstruction';pred=[]
            with torch.inference_mode():
                hh=torch.from_numpy(raw.copy()).view(torch.bfloat16).to('cuda').unsqueeze(1)
                reconstructed_normal=cpu_bf16(model.model.final_layernorm(hh).squeeze(1));norm_calls+=1
                files['reconstructed_norm']=save(a.directory,current+'.reconstructed_norm.bf16',reconstructed_normal)
                ff=torch.from_numpy(normal.copy()).view(torch.bfloat16).to('cuda')
                for j in range(m):
                    pred.append(cpu_bf16((model.lm_head(ff[j].view(1,1,D))*model.model.lm_head_multiplier).reshape(V)));total_recon+=1;guard()
            reconstructed=np.stack(pred);files['reconstructed_logits']=save(a.directory,current+'.reconstructed_logits.bf16',reconstructed)
            previous=np.fromfile(rec['logits']['path'],dtype='<u2').reshape(len(rec['output_ids']),V)
            gates=dict(tokens_exact=output_ids==rec['output_ids'],old_logits_exact=cached.shape==previous.shape and np.array_equal(cached,previous),
                paired_norm_exact=np.array_equal(normal,reconstructed_normal),paired_head_exact=np.array_equal(cached,reconstructed))
            row=dict(id=current,split=rec['split'],domain=rec['domain'],labels=m,output_ids=output_ids,gates=gates,frames=extent(frame_path),
                source_base_forwards=calls,source_LM_head_calls=head_calls,new_reconstruction_head_calls=m,new_reconstruction_norm_calls=1,
                seconds=time.monotonic()-case_start,**files)
            row_path=a.directory/(current+'.json');write(row_path,row);completed.append(dict(**row,case_record=extent(row_path)))
            print(json.dumps(dict(stage='case_complete',id=current,completed=len(completed),labels=m,gates=gates,seconds=time.monotonic()-start)),flush=True)
            del generated,logits,encoded,ids,hh,ff;guard()
        stage='parameter_seal';final=parameter_seal(model,b,guard);assert final==initial;write(a.directory/'parameters_after.json',final);guard()
        gates={key:all(row['gates'][key] for row in completed) for key in ('tokens_exact','old_logits_exact','paired_norm_exact','paired_head_exact')}
        result=dict(schema='SOURCE_CACHED_FIT_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,records=completed,labels=sum(row['labels'] for row in completed),
            decision='PAIRED_CACHED_FIT_PASS' if all(gates.values()) else 'PAIRED_CACHED_FIT_FAIL',gates=gates,
            source_generations=generations,source_model_instances=1,source_base_forwards=total_base,source_LM_head_calls=total_head,
            new_reconstruction_head_calls=total_recon,new_reconstruction_norm_calls=norm_calls,
            parameters=len(initial),all_source_parameter_values_before_after_exact=True,optimizer_updates=0,native_calls=0,reserved_queries=0,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            seconds=time.monotonic()-start,quality_admission=False,speed_admission=False,useful_large_n_admission=False,physical_DRAM_bytes=None,
            scope='FIT-only missing intermediate observable; no codec fit or new DEV/RESERVED observation.')
        write(a.out,result);print(json.dumps(dict(stage='complete',decision=result['decision'],seconds=result['seconds'])),flush=True)
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,id=current,error=repr(error),completed_ids=[row['id'] for row in completed],
            source_generations=generations,source_base_forwards=total_base,source_head_calls=total_head,new_reconstruction_head_calls=total_recon,
            current_observed_raw_states=len(raw_states),current_observed_features=len(features),seconds=time.monotonic()-start));raise


def audit(a):
    import numpy as np
    import psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));start=time.monotonic()
    b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.source_result)
    assert r['binding_sha256']==sha(a.binding)==t['binding_sha256'] and t['inputs_before_after_exact']
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item
    directory=Path(r['records'][0]['h24']['path']).parent
    before=json.loads((directory/'parameters_before.json').read_bytes());after=json.loads((directory/'parameters_after.json').read_bytes());assert before==after
    assert set(before)==set(b['fields']) and len(before)==r['parameters']==411
    for name,item in before.items():assert item['sha256']==b['fields'][name]['sha256'] and item['bytes']==b['fields'][name]['bytes']
    assert sum(item['bytes'] for item in before.values())==b['source_named_elements']*2
    weight=Path(b['source'])/'model.safetensors';gf=b['fields']['model.final_layernorm.weight'];hf=b['fields']['lm_head.weight']
    assert gf['shape']==[D] and hf['shape']==[V,D]
    gamma=decode(np.fromfile(weight,dtype='<u2',count=D,offset=gf['offset']))
    head=np.memmap(weight,mode='r',dtype='<u2',offset=hf['offset'],shape=(V,D))
    assert [row['id'] for row in r['records']]==[rec['id'] for rec in b['records']]
    records=[];numeric_bytes=labels=frames_count=total_witnesses=0
    for row,rec in zip(r['records'],b['records']):
        assert row['split']==rec['split']=='FIT' and row['domain']==rec['domain']
        saved=json.loads(Path(row['case_record']['path']).read_bytes());assert {key:value for key,value in row.items() if key!='case_record'}==saved
        assert extent(row['case_record']['path'])==row['case_record']
        m=row['labels'];assert 1<=m<=256 and m==len(row['output_ids']);labels+=m
        def array(key,width):
            nonlocal numeric_bytes
            item=row[key];assert item['shape']==[m,width] and item['dtype']=='uint16' and item['bytes']==m*width*2
            assert extent(item['path'])=={key:item[key] for key in ('path','bytes','sha256')};numeric_bytes+=item['bytes']
            return np.fromfile(item['path'],dtype='<u2').reshape(m,width)
        raw=array('h24',D);normal=array('normalized',D);nrecon=array('reconstructed_norm',D);scores=array('logits',V);precon=array('reconstructed_logits',V)
        old=np.fromfile(rec['logits']['path'],dtype='<u2').reshape(len(rec['output_ids']),V)
        gates=dict(tokens_exact=row['output_ids']==rec['output_ids'],old_logits_exact=scores.shape==old.shape and np.array_equal(scores,old),
            paired_norm_exact=np.array_equal(normal,nrecon),paired_head_exact=np.array_equal(scores,precon));assert gates==row['gates']
        assert [int(np.argmax(decode(x))) for x in scores]==row['output_ids']
        assert extent(row['frames']['path'])==row['frames'];frames=json.loads(Path(row['frames']['path']).read_bytes());assert len(frames)==m
        p=len(rec['input_ids'])
        for j,frame in enumerate(frames):
            expected=rec['input_ids'] if j==0 else [row['output_ids'][j-1]]
            assert frame==dict(label_index=j,input_ids=expected,shape=[1,len(expected)],cache_before_seq_length=0 if j==0 else p+j-1,
                position_ids=list(range(p)) if j==0 else [p+j-1]);frames_count+=1
        hh=decode(raw);ff=decode(normal);exact=hh/np.sqrt(np.mean(hh*hh,axis=-1,keepdims=True)+1e-5)*gamma
        assert np.isfinite(hh).all() and np.isfinite(ff).all()
        norm_rms=math.sqrt(float(np.sum((ff-exact)**2))/float(np.sum(exact**2)));assert norm_rms<=b['gates']['feature_relative_RMS']
        worst=0.;count=0
        for j in {0,m//2,m-1,*(k for k in (21,23,33) if k<m)}:
            q=decode(scores[j]);assert np.isfinite(q).all()
            for v in {0,1,V-1,int(np.argmax(q))}:
                expected=math.fsum(float(x)*float(y) for x,y in zip(ff[j],decode(head[v])))*.01953125
                worst=max(worst,abs(expected-float(q[v])));count+=1
        assert worst<=b['gates']['scalar_absolute'];total_witnesses+=count
        assert row['source_base_forwards']==row['source_LM_head_calls']==row['new_reconstruction_head_calls']==m
        assert row['new_reconstruction_norm_calls']==1
        records.append(dict(id=row['id'],labels=m,gates=gates,norm_feature_relative_RMS=norm_rms,scalar_head_witnesses=count,max_scalar_head_absolute_delta=worst,
            exact_old_logits_imply_KL_zero=gates['old_logits_exact']))
        assert time.monotonic()-start<b['audit_limits']['seconds']-10 and proc.memory_info().peak_wset<=b['audit_limits']['OS_bytes']
    gates={key:all(row['gates'][key] for row in records) for key in ('tokens_exact','old_logits_exact','paired_norm_exact','paired_head_exact')}
    decision='PAIRED_CACHED_FIT_PASS' if all(gates.values()) else 'PAIRED_CACHED_FIT_FAIL';assert gates==r['gates'] and decision==r['decision']
    assert labels==r['labels']==r['source_base_forwards']==r['source_LM_head_calls']==r['new_reconstruction_head_calls']==frames_count
    assert r['source_generations']==r['new_reconstruction_norm_calls']==len(records)==24 and r['source_model_instances']==1
    assert r['optimizer_updates']==r['native_calls']==r['reserved_queries']==0
    assert t['elapsed_seconds']<=b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['capture_limits']['OS_bytes']
    assert r['GPU_allocated_peak']<=b['capture_limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['capture_limits']['GPU_reserved_bytes']
    if gates['tokens_exact']:assert labels==4422 and numeric_bytes==b['expected']['numeric_bytes']
    write(a.out,dict(schema='SOURCE_CACHED_FIT_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=decision,gates=gates,records=records,
        complete_input_output_hashes=True,all_source_parameter_values_before_after_exact=True,frames_checked=frames_count,numeric_bytes=numeric_bytes,
        scalar_head_witnesses=total_witnesses,max_scalar_head_absolute_delta=max(row['max_scalar_head_absolute_delta'] for row in records),
        max_norm_feature_relative_RMS=max(row['norm_feature_relative_RMS'] for row in records),
        source_history_forwards=0,new_full_head_contractions=0,optimizer_updates=0,seconds=time.monotonic()-start,
        scope='Full stored FIT capture audit, all hashes/frames/exact gates plus analytic norm and scalar head witnesses; no codec or DEV fit.'))
    print(json.dumps(dict(audit=str(a.out),decision=decision,seconds=time.monotonic()-start)),flush=True)


def launch(a):
    import psutil
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','native','original_engine','clang','lld','packed_original')),('overlap',p.pid,name)
    terminal=a.out.with_suffix('.terminal.json');log=a.out.with_suffix('.worker.log')
    assert not any(path.exists() for path in (a.out,terminal,log)) and (a.audit or not a.directory.exists())
    assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());lim=b['audit_limits' if a.audit else 'capture_limits']
    began=time.monotonic();reader=memory_reader();peak=0
    for item in b['inputs']:assert extent(item['path'])==item
    assert time.monotonic()-began<lim['seconds'],'prehash deadline'
    extra=[]
    if a.audit:extra=[extent(a.source_result),extent(a.source_result.with_suffix('.terminal.json'))]
    command=[sys.executable,'-I','-S','-B','-X','utf8',str(Path(__file__).resolve()),'--audit-worker' if a.audit else '--capture-worker',
        '--binding',str(a.binding.resolve()),'--binding-sha',a.binding_sha,'--freeze',a.freeze,'--out',str(a.out.resolve())]
    command += ['--source-result',str(a.source_result.resolve())] if a.audit else ['--directory',str(a.directory.resolve())]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1' if a.audit else '6',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
             CUDA_VISIBLE_DEVICES='' if a.audit else '0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    receipt=dict(freeze=a.freeze,binding_sha256=a.binding_sha,command=command,launcher_pid=proc.pid,limits=lim)
    with log.open('xb') as output:
        child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,creationflags=8)
        receipt.update(worker_pid=child.pid,worker_creation_time=psutil.Process(child.pid).create_time());fault=None
        try:
            while child.poll() is None:
                peak=max(peak,reader(child))
                assert time.monotonic()-began<=lim['seconds'],'deadline'
                assert peak+proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
                assert log.stat().st_size<=lim['log_bytes'],'log cap'
                try:assert not psutil.Process(child.pid).children(recursive=True),'unexpected child'
                except psutil.NoSuchProcess:pass
                time.sleep(.1)
            assert child.returncode==0,log.read_text(errors='replace')[-5000:]
            for item in b['inputs']+extra:assert extent(item['path'])==item
            outputs=[] if a.audit else [extent(path) for path in sorted(a.directory.iterdir()) if path.is_file()]
            assert sum(item['bytes'] for item in outputs)+a.out.stat().st_size<=lim['output_bytes']
            if not a.audit:
                r=json.loads(a.out.read_bytes());assert r['GPU_allocated_peak']<=lim['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=lim['GPU_reserved_bytes']
            assert time.monotonic()-began<=lim['seconds'],'seal deadline'
            receipt.update(inputs_before_after_exact=True,result=extent(a.out),outputs=outputs,extra_inputs=extra)
        except BaseException as error:
            fault=repr(error)
            if child.poll() is None:child.kill();child.wait()
        finally:
            peak=max(peak,reader(child));receipt.update(exit_code=child.returncode,error=fault,elapsed_seconds=time.monotonic()-began,
                worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,log=extent(log))
            write(terminal,receipt)
    assert child.returncode==0 and fault is None,fault
    print(json.dumps(dict(terminal=str(terminal),seconds=receipt['elapsed_seconds'],result_sha256=receipt['result']['sha256'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bind','audit','capture-worker','audit-worker'):p.add_argument('--'+key,action='store_true')
    for key in ('binding','out','directory','source-result','protocol'):p.add_argument('--'+key,type=Path)
    for key in ('binding-sha','freeze'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.bind:bind(a)
    elif a.capture_worker:capture(a)
    elif a.audit_worker:audit(a)
    else:launch(a)
