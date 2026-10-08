"""NEW stateless common-C-input bank probe; no whole-prefix/model replay."""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B));from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))


def bind(a):
    native=DOC/'chatbot_hybrid_native_result_repair3_20261008.json'
    n=json.loads(native.read_bytes())
    trace=next(Path(v['path']) for v in n['native_outputs'] if Path(v['path']).name=='trace.bin')
    queries=trace.parent/'queries.json'
    files=[native,native.with_suffix('.terminal.json'),Path(n['model']['path']),trace,queries,Path(__file__),
       B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',Path(sys.executable),
       DOC/'CHATBOT_HYBRID_COMMON_BANK_PROTOCOL_20261009.md']
    b=dict(schema='HYBRID_COMMON_BANK_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
       model=n['model']['path'],trace=str(trace.resolve()),queries=str(queries.resolve()),
       limits=dict(seconds=600,OS_bytes=4<<30,output_bytes=16<<20),
       runtime_binding_scope='Native packed bytes/common-C operands and code/Python hashed; isolated Torch/NumPy versions/paths checked, not full DLL-tree certificate.',
       inputs=[dict(path=str(v.resolve()),bytes=v.stat().st_size,sha256=sha(v)) for v in files])
    write(a.out,b);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(files))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='HYBRID_COMMON_BANK_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    import numpy as np
    import psutil
    import torch
    from torch.nn import functional as F
    assert torch.__version__=='2.6.0+cu124' and np.__version__=='2.4.6' and psutil.__version__=='7.2.2'
    assert torch.cuda.is_available()
    proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def guard():
        assert time.monotonic()-start<=600
        assert proc.memory_info().peak_wset<=4<<30
        assert torch.cuda.max_memory_allocated()<=512<<20 and torch.cuda.max_memory_reserved()<=1<<30
        assert not proc.children(recursive=True)
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=16<<20
    try:
        dtype=np.dtype([(n,'<f4',512) for n in ('input','core_input','core_output','ff_input')]+[
            ('scores','<f4',72),('ids','<u4',8),('mass','<f4',8),('ff_output','<f4',512),('output','<f4',512)])
        trace=np.fromfile(b['trace'],dtype=dtype);assert len(trace)==3132
        model=np.memmap(b['model'],mode='r',dtype=np.uint8)
        descriptors={}
        for i in range(212):
            r=struct.unpack_from('<64sII4IQQ',model,80+i*104)
            name=r[0].split(b'\0')[0].decode();descriptors[name]=(r[1],tuple(r[3:3+r[2]]),r[7])
        def array(name):
            typ,shape,off=descriptors[name]
            return np.ndarray(shape,dtype='<f4' if typ==1 else np.uint8,buffer=model,offset=off)
        def cuda(name):
            return torch.tensor(array(name).copy(),device='cuda')
        def decode(name):
            pair=cuda(name).to(torch.int16).transpose(1,2)
            w=torch.empty(pair.shape[0],pair.shape[1],pair.shape[2]*2,device='cuda')
            w[:,:,0::2]=(pair//3-1).float();w[:,:,1::2]=(pair%3-1).float()
            return w
        def quant(x):
            scale=x.abs().amax(-1,keepdim=True).clamp_min(1e-12)/63
            return torch.round(x*(1/scale)).clamp(-63,63),scale
        def linear(x,w,scale):
            q,a=quant(x);return F.linear(q,w)*scale*a
        records=[];frame_records=[]
        with torch.no_grad():
            for layer in range(12):
                prefix=f'layers.{layer}.banks.'
                x=torch.tensor(trace['ff_input'][layer::12].copy(),device='cuda')
                gate,up,down=[decode(prefix+n) for n in ('gate','up','down')]
                gs,us,ds=[cuda(prefix+n+'_scale') for n in ('gate','up','down')]
                scores=F.linear(x,cuda(prefix+'router.weight'),cuda(prefix+'router.bias'))
                ids=torch.argsort(scores,descending=True,stable=True)[:,:8]
                mass=torch.softmax(torch.gather(scores,1,ids),-1)
                y=torch.zeros_like(x)
                for e in range(72):
                    positions,slots=(ids==e).nonzero(as_tuple=True)
                    if not positions.numel():continue
                    xi=x[positions];g=linear(xi,gate[e],gs[e]);u=linear(xi,up[e],us[e])
                    out=linear(F.silu(g)*u,down[e],ds[e])
                    y=y.index_add(0,positions,out*mass[positions,slots,None])
                torch.cuda.synchronize()
                actual=y.cpu().numpy();reference=trace['ff_output'][layer::12]
                assert np.isfinite(actual).all()
                path=a.directory/(f'layer_{layer:02d}.common.ffn.f32')
                with path.open('xb') as f:
                    f.write(actual.astype('<f4',copy=False).tobytes());f.flush();os.fsync(f.fileno())
                error=np.square(actual.astype(np.float64)-reference).sum(-1)
                energy=np.square(reference.astype(np.float64)).sum(-1)
                assert np.all((energy>0)|(error==0)),'zero reference nonzero output needs explicit separate classification'
                rms=np.sqrt(error/np.where(energy==0,1,energy))
                py_ids=ids.cpu().numpy().astype(np.uint32);c_ids=trace['ids'][layer::12]
                id_changes=np.any(py_ids!=c_ids,axis=-1)
                score_error=np.abs(scores.cpu().numpy().astype(np.float64)-trace['scores'][layer::12]).max(-1)
                masses=mass.cpu().numpy().astype(np.float64)
                equal_ids=~id_changes
                mass_error=float(np.abs(masses[equal_ids]-trace['mass'][layer::12][equal_ids]).max()) if equal_ids.any() else None
                large=rms>1e-4
                record=dict(layer=layer,rows=261,router_ID_changed_rows=int(id_changes.sum()),
                     max_router_score_abs_error=float(score_error.max()),max_mass_error_where_IDs_equal=mass_error,
                     bank_RMS_max=float(rms.max()),bank_RMS_mean=float(rms.mean()),
                     rows_bank_RMS_over_1e_4=int(large.sum()),rows_large_with_same_IDs=int(np.count_nonzero(large&equal_ids)),
                     output=dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)))
                records.append(record)
                for i in range(261):
                    frame_records.append(dict(layer=layer,global_input_row=i,bank_relative_RMS=float(rms[i]),
                        router_ID_changed=bool(id_changes[i]),router_score_abs_error=float(score_error[i]),
                        C_ids=c_ids[i].tolist(),CUDA_ids=py_ids[i].tolist()))
                print(json.dumps(dict(layer=layer,large_rows=record['rows_bank_RMS_over_1e_4'],ID_changes=record['router_ID_changed_rows'],
                     max_RMS=record['bank_RMS_max'],seconds=time.monotonic()-start)),flush=True)
                del x,gate,up,down,gs,us,ds,scores,ids,mass,y
                guard()
        changed=sum(r['router_ID_changed_rows'] for r in records)
        large_same=sum(r['rows_large_with_same_IDs'] for r in records)
        decision='COMMON_INPUT_ROUTER_DIFFERENCE' if changed else 'COMMON_INPUT_BANK_DIFFERENCE' if large_same else 'COMMON_INPUT_BANKS_CLOSE'
        write(a.directory/'frame_metrics.json',frame_records)
        report=dict(schema='HYBRID_COMMON_BANK_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
             process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),decision=decision,
             layers=records,rows=3132,router_ID_changed_rows=changed,large_bank_error_same_ID_rows=large_same,
             frame_metrics=dict(path=str((a.directory/'frame_metrics.json').resolve()),sha256=sha(a.directory/'frame_metrics.json')),
             GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
             worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
             new_stateless_bank_input_rows=3132,whole_model_forwards=0,native_prefix_replays=0,source_generations=0,training_updates=0,
             limits='Common C-state operands isolate stateless backend differences. Does not reproduce original GPU trajectory or identify every full-prefix cause. Original C parity FAIL remains.')
        write(a.out,report);guard()
        print(json.dumps(dict(decision=decision,changed_router_rows=changed,large_same_ID_rows=large_same)),flush=True)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
