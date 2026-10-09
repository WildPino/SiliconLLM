"""Price the first source-width ternary conversion on two saved trajectories."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def bind(a):
    source=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    package_path=source/'source_package.json';package=json.loads(package_path.read_bytes())
    assert package['revision']=='80ebc50d7799a440b96c93bb6686a3924a09b0cb'
    adoption_path=DOC/'chatbot_broad_adoption_result_20261009.json';adoption=json.loads(adoption_path.read_bytes())
    receipt_path=adoption_path.with_suffix('.terminal.json');receipt=json.loads(receipt_path.read_bytes())
    assert adoption['decision']=='SAVED_BROAD_TRANSPORT_PASS' and receipt['exit_code']==0 and receipt['result_sha256']==sha(adoption_path)
    corpus=Path(adoption['corpus']['path']);assert sha(corpus)==adoption['corpus']['sha256']
    fit=sorted([r for r in json.loads(corpus.read_bytes())['records'] if r['split']=='FIT'],key=lambda r:(len(r['student_input_ids']),r['id']))
    assert len(fit)==24;selected=[fit[0],fit[-1]]
    qualification_path=DOC/'chatbot_falcon_ssd_tiles_result_repair1_20261009.json'
    qualification=json.loads(qualification_path.read_bytes());qualification_receipt=qualification_path.with_suffix('.terminal.json')
    qr=json.loads(qualification_receipt.read_bytes())
    assert qualification['decision']=='TILED_SOURCE_TRAJECTORIES_PASS' and qr['exit_code']==0 and qr['result_sha256']==sha(qualification_path)
    qualified_binding_path=DOC/'chatbot_falcon_ssd_tiles_binding_repair1_20261009.json'
    assert sha(qualified_binding_path)==qualification['binding_sha256']
    qb=json.loads(qualified_binding_path.read_bytes())
    for name in ('chatbot_falcon_ssd_tiles.py','chatbot_hybrid_target.py'):
        path=B/name
        if name=='chatbot_falcon_ssd_tiles.py':
            item=next(v for v in qb['inputs'] if Path(v['path']).resolve()==path.resolve())
            assert sha(path)==item['sha256']
    files=[Path(v['path']) for v in package['files']]
    assert all(sha(v['path'])==v['sha256'] for v in package['files'])
    files += [package_path,adoption_path,receipt_path,corpus,qualification_path,qualification_receipt,qualified_binding_path,
              Path(__file__),B/'chatbot_source_ternary_ffn.py',B/'chatbot_falcon_ssd_tiles.py',B/'chatbot_hybrid_target.py',
              B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',Path(sys.executable),
              DOC/'CHATBOT_SOURCE_TERNARY_FFN_PREFLIGHT_PROTOCOL_20261009.md',ROOT/'benchmarks/phase60/engine.c',B/'chatbot_hybrid_native.py']
    files += [Path(r['logits']['path']) for r in selected]
    files += [SITE/'transformers'/v for v in ('models/falcon_h1/modeling_falcon_h1.py','cache_utils.py','activations.py')]
    files += [ROOT/v for v in ('benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    files=list(dict.fromkeys(p.resolve() for p in files))
    write(a.out,dict(schema='SOURCE_FFN_PREFLIGHT_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),source=str(source.resolve()),source_named_elements=package['total_named_elements'],
        corpus=str(corpus.resolve()),selected_ids=[r['id'] for r in selected],source_revision=package['revision'],
        limits=dict(seconds=600,reserve_seconds=45,OS_bytes=8<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=512<<20),
        runtime_binding_scope='Pinned original donor/adopted packets/qualified source storage/new FFN mapping/Python/selected runtime/original encoding/foreign hashes;not full DLL-tree.',
        inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='SOURCE_FFN_PREFLIGHT_BINDING_V1'
    a.directory.mkdir(exist_ok=False);phase='startup';completed=[];forwards=0;torch=None
    try:
        assert Path(sys.executable).resolve()==Path(b['python']).resolve() and sys.version_info[:3]==(3,12,10)
        assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels','mamba_ssm','causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from transformers import FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        from chatbot_falcon_ssd_tiles import install
        from chatbot_source_ternary_ffn import EngineFFN,packed_projection
        assert (torch.__version__,transformers.__version__,np.__version__)==('2.6.0+cu124','5.13.1','2.4.6')
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def guard():
            lim=b['limits'];assert time.monotonic()-start<=lim['seconds']-lim['reserve_seconds'],'time reserve'
            assert proc.memory_info().peak_wset<=lim['OS_bytes'],'OS cap'
            assert torch.cuda.max_memory_allocated()<=lim['GPU_allocated_bytes'],'GPU allocated cap'
            assert torch.cuda.max_memory_reserved()<=lim['GPU_reserved_bytes'],'GPU reserved cap'
            assert not proc.children(recursive=True),'worker child'
            assert sum(v.stat().st_size for v in a.directory.iterdir() if v.is_file())<=lim['output_bytes'],'output cap'
        def event(stage,**fields):
            print(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**fields)),flush=True);guard()
        modification=install(code);write(a.directory/'source_method.json',modification)
        phase='load'
        model=FalconH1ForCausalLM.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        assert sum(p.numel() for p in model.parameters())==b['source_named_elements']==1554863488
        assert not code.is_fast_path_available and model.config.mamba_chunk_size==128 and model.generation_config.eos_token_id==[11,228]
        unchanged={n:id(p) for n,p in model.named_parameters() if '.feed_forward.' not in n}
        assert len(unchanged)==339
        samples={};active=None
        def observer(site,name,q,matrix,dot):
            key=(active,name)
            if site!=0 or active is None or key in samples:return
            samples[key]=dict(q=q[0,-1].detach().cpu().contiguous(),matrix=matrix.detach().cpu().contiguous(),
                              dot=dot[0,-1].detach().cpu().contiguous())
        phase='conversion';t0=time.monotonic();fields=[];weight_metrics=[]
        payload=a.directory/'ffn.packed.bin'
        with payload.open('xb') as stream,torch.inference_mode():
            for site,layer in enumerate(model.model.layers):
                original=layer.feed_forward
                replacement=EngineFFN(original,site,observer)
                assert replacement.gate_multiplier==.4419417382415922 and replacement.down_multiplier==.13020833333333331
                for name in ('gate','up','down'):
                    packed,scale=packed_projection(replacement,name)
                    for label,value,dtype in (('pair_codes',packed,'u8'),('row_scale',scale,'F32')):
                        raw=value.cpu().numpy().tobytes();offset=stream.tell();stream.write(raw)
                        fields.append(dict(site=site,projection=name,field=label,shape=list(value.shape),offset=offset,bytes=len(raw),dtype=dtype))
                    del packed,scale,raw
                weight_metrics.extend(dict(site=site,**v) for v in replacement.weight_metrics)
                layer.feed_forward=replacement;del original,replacement
                event('converted_layer',layer=site)
            stream.flush();os.fsync(stream.fileno())
        conversion_seconds=time.monotonic()-t0
        assert payload.stat().st_size==340819968 and len(fields)==144 and len(weight_metrics)==72
        assert {n:id(p) for n,p in model.named_parameters()}==unchanged
        assert sum(p.numel() for p in model.parameters())==875386240
        write(a.directory/'ffn.manifest.json',dict(schema='SOURCE_TERNARY_FFN_SECTOR_V1',fields=fields,
            file=dict(path=str(payload.resolve()),bytes=payload.stat().st_size,sha256=sha(payload)),
            code_elements=679477248,row_scale_elements=270336,weight_metrics=weight_metrics,
            pair_layout='pair/row;first digit*3+second digit,byte0..8',full_native_model=False,
            arithmetic='F32 AQ63 integer dots/scales/SiLU/MUP;final FFN cast to original BF16',
            conversion_seconds=conversion_seconds))
        event('conversion_complete',conversion_seconds=conversion_seconds,packed_bytes=payload.stat().st_size)
        by_id={r['id']:r for r in json.loads(Path(b['corpus']).read_bytes())['records']};observations=[]
        def lp64(x):
            x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
        for identifier in b['selected_ids']:
            phase=identifier;rec=by_id[identifier];active=identifier;prompt=rec['input_ids'];outputs=rec['output_ids']
            assert rec['split']=='FIT' and prompt[0]==17
            past=None;actual=[];t0=time.monotonic()
            with torch.inference_mode():
                for row in range(len(outputs)):
                    ids=prompt if row==0 else [outputs[row-1]]
                    result=model(input_ids=torch.tensor([ids],device='cuda'),
                        attention_mask=torch.ones((1,len(prompt)+row),dtype=torch.long,device='cuda'),
                        past_key_values=past,use_cache=True,logits_to_keep=1)
                    past=result.past_key_values;logits=result.logits[0,-1];forwards+=1
                    assert logits.dtype==torch.bfloat16 and torch.isfinite(logits).all().item()
                    actual.append(logits.view(torch.uint16).cpu().numpy().copy());del result,logits;guard()
            active=None;torch.cuda.synchronize();trajectory_seconds=time.monotonic()-t0
            values=np.stack(actual);raw_path=a.directory/(identifier+'.logits.bf16')
            with raw_path.open('xb') as f:f.write(values.astype('<u2',copy=False).tobytes());f.flush();os.fsync(f.fileno())
            source_bits=np.fromfile(rec['logits']['path'],dtype='<u2').reshape(rec['logits']['shape'])
            source=(source_bits.astype('<u4')<<16).view('<f4')
            changed=(values.astype('<u4')<<16).view('<f4')
            label_KL=[]
            for offset in range(0,len(outputs),16):
                p=lp64(source[offset:offset+16].astype(np.float64));q=lp64(changed[offset:offset+16].astype(np.float64))
                label_KL.extend((np.exp(p)*(p-q)).sum(-1).tolist())
            assert all(np.isfinite(v) and v>=-1e-10 for v in label_KL)
            winners=changed.argmax(-1).tolist()
            row=dict(id=identifier,input_ids=len(prompt),teacher_forcing_ids=len(rec['student_input_ids']),labels=len(outputs),stop=rec['stop'],
                case_KL_F64=sum(label_KL)/len(label_KL),label_KL_F64=label_KL,predicted_ids=winners,
                disagreements=sum(x!=y for x,y in zip(winners,outputs,strict=True)),trajectory_seconds=trajectory_seconds,
                logits=dict(path=str(raw_path.resolve()),bytes=raw_path.stat().st_size,sha256=sha(raw_path)),
                GPU_allocated_peak_so_far=torch.cuda.max_memory_allocated(),GPU_reserved_peak_so_far=torch.cuda.max_memory_reserved())
            write(a.directory/(identifier+'.json'),row);observations.append(row);completed.append(identifier)
            event('case',**{k:v for k,v in row.items() if k not in ('label_KL_F64','predicted_ids','logits')})
            del past,actual,source_bits,source,changed,values
        phase='dot_qualification';assert len(samples)==6;dot_records=[]
        for (identifier,name),saved in samples.items():
            q=saved['q'];matrix=saved['matrix'];dot=saved['dot']
            assert q.dtype==torch.float32 and matrix.dtype==torch.int8 and dot.dtype==torch.float32
            assert torch.equal(q,q.round()) and q.abs().max().item()<=63 and matrix.abs().max().item()<=1
            reference=matrix.double()@q.double()
            differences=int((reference!=dot.double()).sum().item())
            record=dict(id=identifier,projection=name,input_columns=q.numel(),output_rows=dot.numel(),
                        differing_exact_integer_dots=differences,maximum_bound=63*q.numel())
            torch.save(saved,a.directory/(identifier+'.'+name+'.dot.pt'))
            write(a.directory/(identifier+'.'+name+'.dot.json'),record);dot_records.append(record)
            event('integer_dot',**record);assert differences==0,'integer dot mismatch'
        assert {n:id(p) for n,p in model.named_parameters()}==unchanged
        phase='result';guard()
        write(a.out,dict(schema='SOURCE_FFN_PREFLIGHT_RESULT_V1',decision='SOURCE_FFN_COST_ARITHMETIC_PASS',freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),cases=2,observations=observations,integer_dots=dot_records,
            conversion_seconds=conversion_seconds,converted_FFN_elements=679477248,packed_FFN_bytes=340819968,
            retained_source_parameter_tensors=339,retained_source_parameter_elements=875386240,
            unchanged_parameter_object_identity=True,source_forwards=forwards,source_generations=0,optimizer_updates=0,native_runs=0,
            quality_admission=False,full_native_model=False,GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            scope='Complete source-width FFN arithmetic conversion/packed sector plus two original forced cached FIT trajectories andsix actual integer-dot witnesses. No own-history/task/native/rate/large-n admission.'))
        event('complete',decision='SOURCE_FFN_COST_ARITHMETIC_PASS')
    except BaseException as error:
        failure=dict(fault=repr(error),phase=phase,completed=completed,source_forwards=forwards,elapsed_seconds=time.monotonic()-start)
        if torch is not None:failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
