"""Observe paired cached final states/logits; preserve the original readout FAIL."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
from original_falcon_whole_recovery import memory_reader
sys.path.insert(0,str(SITE))
IDENT='broad_dev_everyday_conversations_036';D,V=2048,65537


def sf_fields(path):
    with Path(path).open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));out={}
        for name,item in header.items():
            if name=='__metadata__':continue
            start,end=item['data_offsets'];assert item['dtype']=='BF16' and end-start==math.prod(item['shape'])*2
            f.seek(8+n+start);left=end-start;digest=hashlib.sha256()
            while left:
                raw=f.read(min(left,8<<20));assert raw;left-=len(raw);digest.update(raw)
            out[name]=dict(offset=8+n+start,bytes=end-start,shape=item['shape'],sha256=digest.hexdigest())
    return out


def bind(a):
    assert not a.binding.exists()
    old_path=DOC/'source_final_readout_binding_20261010.json';old=json.loads(old_path.read_bytes())
    previous=DOC/'source_final_readout_result_20261010.json';audit_path=DOC/'source_final_readout_stored_adjudication_20261010.json'
    audit=json.loads(audit_path.read_bytes());receipt_path=audit_path.with_suffix('.receipt.json');receipt=json.loads(receipt_path.read_bytes())
    assert audit['result']==extent(previous) and audit['decision']=='SOURCE_STATE_LABEL_ALIGNMENT_FAIL'
    assert receipt['exit_code']==0 and receipt['error'] is None and receipt['output']==extent(audit_path)
    source=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008';package=json.loads((source/'source_package.json').read_bytes())
    record_path=ROOT/f'results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/{IDENT}.json';record=json.loads(record_path.read_bytes())
    assert record['id']==IDENT and record['input_length']==148 and len(record['output_ids'])==53
    assert record['student_input_ids']==record['input_ids']+record['output_ids'][:-1]
    assert record['positions']==list(range(147,200)) and record['BF16_lossless'] and record['raw_argmax_equals_generated']
    prev=json.loads(previous.read_bytes());prefill=next(row for row in prev['arms'][0]['records'] if row['id']==IDENT)
    files=[Path(__file__),a.protocol,old_path,previous,audit_path,receipt_path,record_path,source/'source_package.json',DOC/'chatbot_broad_capture_binding_repair1_20261009.json']
    files += [Path(item['path']) for item in package['files']]
    files += [Path(item['path']) for item in old['inputs'] if str(SITE.resolve()).lower() in item['path'].lower() or item['path'].lower().endswith(('.py','.dll','.pyd','.exe'))]
    files += [Path(prefill[key]['path']) for key in ('h24','features','scores')]+[Path(record['logits']['path'])]
    files += [B/'chatbot_falcon_ssd_tiles.py',SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',SITE/'transformers/cache_utils.py',
              SITE/'transformers/generation/utils.py',SITE/'transformers/generation/configuration_utils.py']
    inputs=[extent(path) for path in dict.fromkeys(files)]
    bypath={item['path']:item for item in inputs}
    for item in package['files']:assert bypath[str(Path(item['path']).resolve())]['sha256']==item['sha256']
    assert bypath[str((source/'model.safetensors').resolve())]['sha256']==old['weights']['sha256']
    source_capture_binding=json.loads((DOC/'chatbot_broad_capture_binding_repair1_20261009.json').read_bytes())
    old_sources={item['path']:item for item in source_capture_binding['inputs']}
    for path in (SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',SITE/'transformers/cache_utils.py',SITE/'transformers/generation/utils.py'):
        assert bypath[str(path.resolve())]==old_sources[str(path.resolve())]
    fields=sf_fields(source/'model.safetensors')
    assert sum(item['bytes'] for item in fields.values())==1554863488*2
    write(a.binding,dict(schema='SOURCE_CACHED_FINAL_BINDING_V1',inputs=inputs,source=str(source.resolve()),source_revision=package['revision'],
        record=record,prefill=prefill,fields=fields,source_named_elements=1554863488,
        capture_limits=dict(seconds=600,OS_bytes=8<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=128<<20,log_bytes=4<<20),
        audit_limits=dict(seconds=120,OS_bytes=512<<20,output_bytes=256<<10,log_bytes=4<<20),
        gates=dict(tokens_exact=True,old_logits_exact=True,paired_norm_exact=True,paired_head_exact=True,feature_relative_RMS=.01,scalar_absolute=1.),
        scope='One new source generation for a missing intermediate observable; no whole cohort/history training replay.'))
    print(json.dumps(dict(binding=str(a.binding),sha256=sha(a.binding),inputs=len(inputs),bytes=sum(x['bytes'] for x in inputs))),flush=True)


def capture(a):
    import numpy as np
    import psutil
    import torch
    import transformers
    from transformers import FalconH1ForCausalLM
    from transformers.models.falcon_h1 import modeling_falcon_h1 as code
    from transformers.cache_utils import Cache
    from chatbot_falcon_ssd_tiles import install
    b=json.loads(a.binding.read_bytes());assert sha(a.binding)==a.binding_sha
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and transformers.__version__=='5.13.1'
    assert not code.is_fast_path_available
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.directory.mkdir(exist_ok=False);stage='source_loading';calls=0;head_calls=0;raw_states=[];features=[];frames=[];updates=[];update_count=0;model=None
    def guard():
        lim=b['capture_limits'];assert time.monotonic()-start<lim['seconds']-20,'worker reserve'
        assert proc.memory_info().peak_wset<=lim['OS_bytes']
        assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'] and torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes']
        assert not proc.children(recursive=True)
        assert sum(path.stat().st_size for path in a.directory.iterdir() if path.is_file())<=lim['output_bytes']
    def save(name,array):
        path=a.directory/name
        with path.open('xb') as f:f.write(array.tobytes());f.flush();os.fsync(f.fileno())
        return dict(**extent(path),shape=list(array.shape),dtype=str(array.dtype))
    def cpu_bf16(value):
        assert value.dtype==torch.bfloat16
        return value.detach().cpu().contiguous().view(torch.uint16).numpy().astype('<u2',copy=False).copy()
    def seal_parameters():
        seals={}
        for name,p in model.named_parameters():
            guard();raw=cpu_bf16(p);digest=hashlib.sha256(raw.tobytes()).hexdigest();field=b['fields'][name]
            assert list(p.shape)==field['shape'] and raw.nbytes==field['bytes'] and digest==field['sha256'],name
            seals[name]=dict(identity=id(p),version=p._version,sha256=digest,bytes=raw.nbytes)
        assert set(seals)==set(b['fields']) and sum(x['bytes'] for x in seals.values())==b['source_named_elements']*2
        return seals
    try:
        write(a.directory/'source_method.json',install(code))
        model=FalconH1ForCausalLM.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
        assert sum(p.numel() for p in model.parameters())==b['source_named_elements']
        assert model.config.num_logits_to_keep==1 and model.generation_config.eos_token_id==[11,228]
        initial=seal_parameters();write(a.directory/'parameters_before.json',initial);guard()
        print(json.dumps(dict(stage='source_loaded',parameters=len(initial),seconds=time.monotonic()-start)),flush=True)
        active_j=-1
        def before_base(module,args,kwargs):
            nonlocal calls,active_j
            active_j=calls;calls+=1;ids=kwargs['input_ids'];past=kwargs.get('past_key_values');positions=kwargs.get('position_ids')
            frames.append(dict(label_index=active_j,input_ids=ids[0].tolist(),shape=list(ids.shape),
                cache_before_seq_length=0 if past is None else past.get_seq_length(),
                position_ids=None if positions is None else positions[0].tolist()))
        def before_norm(module,args):raw_states.append(cpu_bf16(args[0][0,-1]))
        def after_norm(module,args,output):features.append(cpu_bf16(output[0,-1]));guard()
        def count_head(module,args,output):
            nonlocal head_calls
            head_calls+=1
        original_update=Cache.update_recurrent_state
        def observe_update(cache,value,layer_idx,**kwargs):
            nonlocal update_count
            update_count+=1;result=original_update(cache,value,layer_idx,**kwargs)
            if layer_idx in (0,11,23) and active_j in (0,21,23,33,52):
                indices=[0,1,17,value.numel()//2,value.numel()-2,value.numel()-1]
                ix=torch.tensor(indices,device=value.device)
                supplied=value.reshape(-1)[ix].float().detach().cpu().numpy().astype('<f4')
                stored=result.reshape(-1)[ix].detach().cpu().contiguous()
                assert stored.dtype==torch.bfloat16
                updates.append(dict(label_index=active_j,site=layer_idx,shape=list(value.shape),input_dtype=str(value.dtype),stored_dtype=str(result.dtype),
                    indices=indices,input_F32_bits=supplied.view('<u4').tolist(),stored_BF16_bits=stored.view(torch.uint16).numpy().astype('<u2').tolist()))
            return result
        hooks=[model.model.register_forward_pre_hook(before_base,with_kwargs=True),model.model.final_layernorm.register_forward_pre_hook(before_norm),
               model.model.final_layernorm.register_forward_hook(after_norm),model.lm_head.register_forward_hook(count_head)]
        Cache.update_recurrent_state=observe_update;stage='cached_generation'
        try:
            with torch.inference_mode():
                ids=torch.tensor([b['record']['input_ids']],device='cuda')
                generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,do_sample=False,use_cache=True,return_dict_in_generate=True,output_logits=True)
                output_ids=generated.sequences[0,len(b['record']['input_ids']):].tolist()
                logits=torch.stack(generated.logits).squeeze(1);encoded=logits.to(torch.bfloat16)
                assert torch.equal(encoded.float(),logits.float()) and torch.isfinite(logits).all()
                cached=cpu_bf16(encoded)
        finally:
            Cache.update_recurrent_state=original_update
            for hook in hooks:hook.remove()
        m=len(output_ids);assert 1<=m<=256 and calls==head_calls==len(raw_states)==len(features)==m and update_count==24*m
        assert logits.argmax(-1).tolist()==output_ids
        for j,frame in enumerate(frames):
            assert frame['input_ids']==(b['record']['input_ids'] if j==0 else [output_ids[j-1]])
            assert frame['cache_before_seq_length']==(0 if j==0 else 148+j-1)
            assert frame['position_ids']==(list(range(148)) if j==0 else [148+j-1])
        raw=np.stack(raw_states);normal=np.stack(features)
        observed=save('cached_logits.bf16',cached);hfile=save('cached_h24.bf16',raw);nfile=save('cached_normalized.bf16',normal)
        write(a.directory/'frames.json',frames);write(a.directory/'cache_quantization_witnesses.json',updates)
        print(json.dumps(dict(stage='cached_observed',labels=m,base_calls=calls,head_calls=head_calls,seconds=time.monotonic()-start)),flush=True)
        stage='paired_reconstruction';pred=[]
        with torch.inference_mode():
            hh=torch.from_numpy(raw.copy()).view(torch.bfloat16).to('cuda').unsqueeze(1)
            reconstructed_normal=cpu_bf16(model.model.final_layernorm(hh).squeeze(1))
            norm_file=save('cached_normalized_reconstructed.bf16',reconstructed_normal)
            ff=torch.from_numpy(normal.copy()).view(torch.bfloat16).to('cuda')
            for j in range(m):
                pred.append(cpu_bf16((model.lm_head(ff[j].view(1,1,D))*model.model.lm_head_multiplier).reshape(V)));guard()
        reconstructed=np.stack(pred);pfile=save('cached_logits_reconstructed.bf16',reconstructed)
        previous=np.fromfile(b['record']['logits']['path'],dtype='<u2').reshape(53,V)
        gates=dict(tokens_exact=output_ids==b['record']['output_ids'],old_logits_exact=cached.shape==previous.shape and np.array_equal(cached,previous),
            paired_norm_exact=np.array_equal(normal,reconstructed_normal),paired_head_exact=np.array_equal(cached,reconstructed))
        stage='parameter_seal';final=seal_parameters();assert final==initial;write(a.directory/'parameters_after.json',final);guard()
        result=dict(schema='SOURCE_CACHED_FINAL_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,id=IDENT,labels=m,output_ids=output_ids,
            decision='PAIRED_CACHED_STATE_PASS' if all(gates.values()) else 'PAIRED_CACHED_STATE_FAIL',gates=gates,
            h24=hfile,normalized=nfile,logits=observed,reconstructed_norm=norm_file,reconstructed_logits=pfile,
            source_generations=1,source_base_forwards=calls,source_LM_head_calls=head_calls,new_reconstruction_head_calls=m,
            new_reconstruction_norm_calls=1,cache_state_updates=update_count,parameters=len(initial),all_source_parameter_values_before_after_exact=True,
            optimizer_updates=0,native_calls=0,reserved_queries=0,GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,seconds=time.monotonic()-start,
            quality_admission=False,speed_admission=False,useful_large_n_admission=False,physical_DRAM_bytes=None,
            scope='One paired cached-state observation; original ALL48 full-prefill alignment FAIL retained. No compact codec or chatbot admission.')
        write(a.out,result);print(json.dumps(dict(stage='complete',decision=result['decision'],gates=gates,seconds=result['seconds'])),flush=True)
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),source_base_forwards=calls,observed_states=len(raw_states),source_head_calls=head_calls,
            cache_state_updates=update_count,seconds=time.monotonic()-start));raise


def decode(raw):
    import numpy as np
    return (raw.astype('<u4')<<16).view('<f4').astype('f8')


def audit(a):
    import numpy as np
    import psutil
    psutil.Process().cpu_affinity(list(range(6)));start=time.monotonic()
    b=json.loads(a.binding.read_bytes());r=json.loads(a.source_result.read_bytes());t=json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert t['exit_code']==0 and t['error'] is None and t['result']==extent(a.source_result)
    assert r['binding_sha256']==sha(a.binding)==t['binding_sha256'] and t['inputs_before_after_exact']
    for item in b['inputs']+t['outputs']:assert extent(item['path'])==item
    directory=Path(r['h24']['path']).parent;m=r['labels']
    def array(key,width):
        item=r[key];assert item['bytes']==m*width*2
        return np.fromfile(item['path'],dtype='<u2').reshape(m,width)
    raw=array('h24',D);normal=array('normalized',D);nrecon=array('reconstructed_norm',D);scores=array('logits',V);precon=array('reconstructed_logits',V)
    old=np.fromfile(b['record']['logits']['path'],dtype='<u2').reshape(53,V)
    gates=dict(tokens_exact=r['output_ids']==b['record']['output_ids'],old_logits_exact=scores.shape==old.shape and np.array_equal(scores,old),
        paired_norm_exact=np.array_equal(normal,nrecon),paired_head_exact=np.array_equal(scores,precon))
    assert gates==r['gates'];decision='PAIRED_CACHED_STATE_PASS' if all(gates.values()) else 'PAIRED_CACHED_STATE_FAIL';assert decision==r['decision']
    assert [int(np.argmax(decode(row))) for row in scores]==r['output_ids']
    before=json.loads((directory/'parameters_before.json').read_bytes());after=json.loads((directory/'parameters_after.json').read_bytes());assert before==after
    assert set(before)==set(b['fields']) and len(before)==r['parameters']
    for name,item in before.items():assert item['sha256']==b['fields'][name]['sha256'] and item['bytes']==b['fields'][name]['bytes']
    gamma_field=b['fields']['model.final_layernorm.weight'];weight=Path(b['source'])/'model.safetensors'
    gamma=decode(np.fromfile(weight,dtype='<u2',count=D,offset=gamma_field['offset']))
    hh=decode(raw);ff=decode(normal);exact=hh/np.sqrt(np.mean(hh*hh,axis=-1,keepdims=True)+1e-5)*gamma
    norm_rms=math.sqrt(float(np.sum((ff-exact)**2))/float(np.sum(exact**2)));assert norm_rms<=b['gates']['feature_relative_RMS']
    head_field=b['fields']['lm_head.weight'];head=np.memmap(weight,mode='r',dtype='<u2',offset=head_field['offset'],shape=(V,D))
    worst_scalar=0.;witnesses=0
    for j in {0,m//2,m-1,*(k for k in (21,23,33) if k<m)}:
        for v in {0,1,V-1,int(np.argmax(decode(scores[j])))}:
            expected=math.fsum(float(x)*float(y) for x,y in zip(ff[j],decode(head[v])))*.01953125
            worst_scalar=max(worst_scalar,abs(expected-float(decode(scores[j:j+1])[0,v])));witnesses+=1
    assert worst_scalar<=b['gates']['scalar_absolute']
    cache_rows=json.loads((directory/'cache_quantization_witnesses.json').read_bytes());quantized=changed=0
    for row in cache_rows:
        assert row['input_dtype']=='torch.float32' and row['stored_dtype']=='torch.bfloat16'
        for source,sink in zip(row['input_F32_bits'],row['stored_BF16_bits']):
            expected=((source+0x7fff+((source>>16)&1))>>16)&0xffff
            assert expected==sink;quantized+=1;changed+=int((source&0xffff)!=0)
    assert r['cache_state_updates']==r['source_base_forwards']*24 and r['source_generations']==1 and r['new_reconstruction_head_calls']==m
    assert r['source_base_forwards']==r['source_LM_head_calls']==m and r['optimizer_updates']==r['native_calls']==r['reserved_queries']==0
    assert t['elapsed_seconds']<=b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit']+t['launcher_OS_peak_snapshot']<=b['capture_limits']['OS_bytes']
    assert r['GPU_allocated_peak']<=b['capture_limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak']<=b['capture_limits']['GPU_reserved_bytes']
    prefix=0
    while prefix<min(m,53) and r['output_ids'][prefix]==b['record']['output_ids'][prefix]:prefix+=1
    comparable=min(prefix+1,m,53)
    source_h=np.memmap(b['prefill']['h24']['path'],mode='r',dtype='<u2',shape=(200,D))
    old_h=decode(source_h[b['record']['positions'][:comparable]]);delta=hh[:comparable]-old_h
    old_f=decode(np.fromfile(b['prefill']['features']['path'],dtype='<u2').reshape(53,D)[:comparable]);fd=ff[:comparable]-old_f
    old_scores=np.memmap(b['prefill']['scores']['path'],mode='r',dtype='<u2',shape=(53,V));margins=[]
    for j in (21,23,33):
        if j>=comparable:continue
        q=decode(scores[j]);p=decode(old_scores[j]);v=int(np.argmax(q));pv=int(np.argmax(p))
        margins.append(dict(label_index=j,cached_argmax=v,prefill_argmax=pv,cached_gap_to_prefill=float(q[v]-q[pv]),
            pairwise_prefill_minus_cached=float((p[pv]-q[pv])-(p[v]-q[v])),
            h24_relative_RMS=math.sqrt(float(np.sum(delta[j]**2))/float(np.sum(hh[j]**2))),
            normalized_relative_RMS=math.sqrt(float(np.sum(fd[j]**2))/float(np.sum(ff[j]**2)))))
    write(a.out,dict(schema='SOURCE_CACHED_FINAL_AUDIT_V1',result=extent(a.source_result),binding=extent(a.binding),decision=decision,gates=gates,
        complete_input_output_hashes=True,all_source_parameter_values_before_after_exact=True,norm_feature_relative_RMS=norm_rms,
        scalar_head_witnesses=witnesses,max_scalar_head_absolute_delta=worst_scalar,
        cache_quantization_coordinates=quantized,cache_nonzero_rounding_coordinates=changed,
        comparable_teacher_forced_positions=comparable,generated_prefix_exact=prefix,
        raw_h24_differing_coordinates=int(np.count_nonzero(raw[:comparable]!=source_h[b['record']['positions'][:comparable]])),
        raw_h24_relative_RMS=math.sqrt(float(np.sum(delta**2))/float(np.sum(hh[:comparable]**2))),
        normalized_relative_RMS=math.sqrt(float(np.sum(fd**2))/float(np.sum(ff[:comparable]**2))),
        first_label_raw_equal=bool(np.array_equal(raw[0],source_h[147])),margins=margins,
        source_history_forwards=0,new_full_head_contractions=0,optimizer_updates=0,seconds=time.monotonic()-start,
        scope='Stored one-case audit. Cached/full-prefill state drift observed if gates pass; cache rounding witnessed but not uniquely causal. Original ALL48 FAIL retained.'))
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
