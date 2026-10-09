"""New storage-schedule variable on saved source trajectories, no generation."""
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
    old=ROOT/'results/native_expert_scaling/chatbot_broad_capture_20261009'
    failure=json.loads((old/'first_failure.json').read_bytes())
    assert failure['fault']=="AssertionError('GPU allocated cap')" and len(failure['durable_completed'])==2
    cases=[old/(identifier+'.json') for identifier in failure['durable_completed']]
    records=[json.loads(p.read_bytes()) for p in cases]
    package=json.loads((source/'source_package.json').read_bytes())
    files=[Path(v['path']) for v in package['files']]+[source/'source_package.json',old/'first_failure.json']
    files+=cases+[Path(r['logits']['path']) for r in records]
    files+=[DOC/name for name in ('chatbot_broad_capture_binding_20261009.json',
            'chatbot_broad_capture_result_20261009.launcher_failure.json','chatbot_broad_capture_result_20261009.worker.log',
            'CHATBOT_BROAD_CAPTURE_FIRST_FAILURE_20261009.md','CHATBOT_FALCON_SSD_TILES_PROTOCOL_20261009.md')]
    files+=[B/name for name in ('chatbot_falcon_ssd_tiles.py','chatbot_falcon_ssd_tiles_qualify.py',
            'chatbot_falcon_usability.py','chatbot_falcon_usability_launch.py')]+[Path(sys.executable)]
    files+=[SITE/'transformers'/name for name in ('models/falcon_h1/modeling_falcon_h1.py',
            'cache_utils.py','generation/utils.py')]
    files+=[ROOT/name for name in ('benchmarks/donor_adaptation/configs/_manifest.json',
            'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    if a.prior_failure:
        prior=json.loads(a.prior_failure.read_bytes())
        assert prior['source_forwards']==0 and not prior['completed']
        files += [a.prior_failure, DOC/'CHATBOT_FALCON_SSD_TILES_REPAIR1_PROTOCOL_20261009.md']
        files += [DOC/name for name in ('chatbot_falcon_ssd_tiles_binding_20261009.json',
                  'chatbot_falcon_ssd_tiles_result_20261009.launcher_failure.json',
                  'chatbot_falcon_ssd_tiles_result_20261009.worker.log')]
    files=list(dict.fromkeys(p.resolve() for p in files))
    inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    write(a.out,dict(schema='FALCON_SSD_TILES_BINDING_V1',python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()),source=str(source),cases=[str(p) for p in cases],
        source_named_elements=package['total_named_elements'],
        criteria=dict(all_logit_bits_equal=True,all_argmax_equal=True),
        limits=dict(seconds=300,OS_bytes=8<<30,GPU_allocated_bytes=10<<30,GPU_reserved_bytes=11<<30,output_bytes=64<<20),
        runtime_binding_scope='Original package/two saved trajectories/selected source/runtime/code/Python/foreign hashes; not full DLL tree.',
        inputs=inputs))
    print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(inputs))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='FALCON_SSD_TILES_BINDING_V1'
    a.directory.mkdir(exist_ok=False);completed=[];forwards=0;torch=None
    try:
        assert sys.version_info[:3]==(3,12,10) and Path(sys.executable).resolve()==Path(b['python']).resolve()
        assert all(os.environ.get(k)=='1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels','mamba_ssm','causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from transformers import FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        from chatbot_falcon_ssd_tiles import install
        assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='5.13.1' and np.__version__=='2.4.6'
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        proc=psutil.Process();proc.cpu_affinity(list(range(6)))
        def guard():
            assert time.monotonic()-start <= b['limits']['seconds']-15
            assert proc.memory_info().peak_wset <= b['limits']['OS_bytes']
            assert torch.cuda.max_memory_allocated() <= b['limits']['GPU_allocated_bytes']
            assert torch.cuda.max_memory_reserved() <= b['limits']['GPU_reserved_bytes']
            assert not proc.children(recursive=True)
        modification=install(code);write(a.directory/'source_method.json',modification)
        model=FalconH1ForCausalLM.from_pretrained(b['source'],local_files_only=True,trust_remote_code=False,
                 dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
        assert sum(p.numel() for p in model.parameters())==b['source_named_elements']
        assert not code.is_fast_path_available and model.config.mamba_chunk_size==128
        observations=[]
        for path in b['cases']:
            case=json.loads(Path(path).read_bytes());packet=case['logits']
            assert sha(packet['path'])==packet['sha256']
            expected=np.fromfile(packet['path'],dtype='<u2').reshape(packet['shape'])
            actual=[];past=None;prompt=case['input_ids'];outputs=case['output_ids'];winners=[]
            t0=time.monotonic()
            with torch.inference_mode():
                for row in range(len(outputs)):
                    ids=prompt if row==0 else [outputs[row-1]]
                    result=model(input_ids=torch.tensor([ids],device='cuda'),
                        attention_mask=torch.ones((1,len(prompt)+row),dtype=torch.long,device='cuda'),
                        past_key_values=past,use_cache=True,logits_to_keep=1)
                    forwards+=1;past=result.past_key_values
                    logits=result.logits[0,-1]
                    assert torch.isfinite(logits).all() and torch.equal(logits.float(),logits.to(torch.bfloat16).float())
                    actual.append(logits.to(torch.bfloat16).view(torch.uint16).cpu().numpy().copy())
                    winners.append(int(logits.argmax().item()));del result,logits
                    guard()
            values=np.stack(actual);raw=a.directory/(case['id']+'.logits.bf16')
            with raw.open('xb') as stream:
                stream.write(values.astype('<u2',copy=False).tobytes());stream.flush();os.fsync(stream.fileno())
            differing=int(np.count_nonzero(values!=expected))
            record=dict(id=case['id'],input_length=len(prompt),labels=len(outputs),differing_BF16_coordinates=differing,
                        differing_argmax=sum(x!=y for x,y in zip(winners,outputs,strict=True)),seconds=time.monotonic()-t0,
                        logits=dict(path=str(raw.resolve()),bytes=raw.stat().st_size,sha256=sha(raw)))
            write(a.directory/(case['id']+'.json'),record);observations.append(record);completed.append(case['id'])
            del past,actual,values,expected
            print(json.dumps(record),flush=True)
        gates=dict(all_logit_bits_equal=all(r['differing_BF16_coordinates']==0 for r in observations),
                   all_argmax_equal=all(r['differing_argmax']==0 for r in observations))
        report=dict(schema='FALCON_SSD_TILES_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='TILED_SOURCE_TRAJECTORIES_PASS' if all(gates.values()) else 'TILED_SOURCE_TRAJECTORIES_FAIL',
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
            cases=2,labels=sum(r['labels'] for r in observations),source_forwards=forwards,source_generations=0,
            student_forwards=0,training_updates=0,native_runs=0,gates=gates,observations=observations,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
            scope='New source temporary-storage schedule on two saved fixed forced trajectories. Zero regenerated labels; exact observed BF16 logits and IDs required. Mathematical contraction identity is general; bit equality is observed only here.')
        write(a.out,report);guard();print(json.dumps(dict(decision=report['decision'],gates=gates)),flush=True)
    except BaseException as error:
        failure=dict(fault=repr(error),completed=completed,source_forwards=forwards,elapsed_seconds=time.monotonic()-start)
        if torch is not None:failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prior-failure',type=Path)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
