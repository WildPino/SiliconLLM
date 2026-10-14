"""Actual source classes on meta: omitted identity/hooks, native F0 ownership.

Mechanical evidence only. No source decoder quality, GPU or depth speed claim.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write


def main(a):
    import numpy as np
    import psutil
    import torch
    from transformers import FalconH1Config,FalconH1ForCausalLM
    from transformers.cache_utils import DynamicCache
    from fixed_depth_bridge import FixedDepthBridge,kept_sites
    from row_state import interface
    from token_state import arrays,arguments
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    proc=psutil.Process();proc.cpu_affinity([0]);start=time.monotonic()
    def guard():
        assert time.monotonic()-start<600 and proc.memory_info().peak_wset<=4<<30
        assert not proc.children(recursive=True) and not torch.cuda.is_initialized()
    b=json.loads((DOC/'full_width_native_chat_binding_20261013.json').read_bytes())
    closure=DOC/'native_bf_activation_lut_session_closure_20261014.json'
    paths=[Path(__file__),B/'fixed_depth_bridge.py',B/'full_width_native_chat.py',B/'bf_activation_backend.py',
        B/'token_state.py',B/'row_state.py',B/'falcon_auxiliary_s6_transport.py',
        DOC/'FIXED_DEPTH_PROTOCOL_20261014.md',DOC/'full_width_native_chat_binding_20261013.json',closure,
        Path(b['source'])/'config.json',Path(b['compile']['dll']['path']),Path(b['table']['path']),
        Path(torch.__file__),Path(torch._C.__file__),Path(np.__file__),Path(np._core._multiarray_umath.__file__),
        Path(sys.executable),Path(sys.executable).parent/'python312.dll']
    paths.extend(Path(e['file']['path']) for e in b['exports'])
    paths.extend(Path(m.__file__) for m in tuple(sys.modules.values())
                 if getattr(m,'__file__',None) and Path(m.__file__).is_file())
    pins=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)]
    config=FalconH1Config.from_pretrained(b['source'],local_files_only=True)
    assert config.hidden_size==2048 and config.num_hidden_layers==24
    config._attn_implementation='eager'
    with torch.device('meta'):model=FalconH1ForCausalLM(config)
    assert all(p.device.type=='meta' for p in model.parameters())
    original=[(layer.forward,layer.mamba.forward) for layer in model.model.layers]
    x=torch.arange(2048,dtype=torch.float32).reshape(1,1,2048).to(torch.bfloat16)/256
    api=interface(b);reports=[]
    def forbidden(module,args):raise AssertionError('An omitted source submodule executed')
    for count in (3,6,12,24):
        guard();bridge=FixedDepthBridge(model,b,np,torch,count,closure)
        assert bridge.active==kept_sites(count) and len(set(bridge.active))==count
        cache=DynamicCache(config=config);before=[cache.get_seq_length(i) for i in range(24)]
        omitted=0
        for site,layer in enumerate(model.model.layers):
            if site in bridge.active:continue
            hooks=[m.register_forward_pre_hook(forbidden) for m in layer.modules() if m is not layer]
            try:
                result=layer(x,past_key_values=cache,use_cache=True)
                assert isinstance(result,tuple) and len(result)==1 and result[0] is x
                omitted+=1
            finally:
                for hook in hooks:hook.remove()
        assert [cache.get_seq_length(i) for i in range(24)]==before
        # One actual source-only F0 token, own C state; no meta coefficient is evaluated.
        bridge.contexts=[bridge.create() if i in bridge.active else None for i in range(24)]
        assert all(bridge.contexts[i] is None for i in range(24) if i not in bridge.active)
        native_path=a.directory/f'depth{count}.first_native.bf16'
        bridge.streams=[native_path.open('xb') if i==0 else None for i in range(24)]
        bridge.token=42;bridge.steps=0
        out=model.model.layers[0].mamba(x,cache_params=cache)
        assert out.shape==(1,1,2048) and out.dtype==torch.bfloat16 and out.device.type=='cpu'
        state=np.empty((3072,96),'f4');ring=np.empty((3,3584),'u2');seen=np.zeros(1,'u8')
        assert bridge.snapshot(bridge.contexts[0],state.ctypes.data,ring.ctypes.data,seen.ctypes.data)==0
        assert seen[0]==1 and bridge.last(bridge.contexts[0])==42
        refctx=api[1]();z=arrays(1,np);tag=np.array([42],'u4');p=bridge.params[0]
        try:
            assert api[3](refctx,1,tag.ctypes.data,bridge.table.ctypes.data,bridge.weights[0],bridge.conv[0],
                          p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(z,0,1,np))==0
            assert np.array_equal(z['output'],bridge.outputs[0]['output'])
            st=np.empty_like(state);rg=np.empty_like(ring);sn=np.zeros(1,'u8')
            assert api[5](refctx,st.ctypes.data,rg.ctypes.data,sn.ctypes.data)==0
            assert np.array_equal(state,st) and np.array_equal(ring,rg) and np.array_equal(seen,sn)
        finally:api[2](refctx)
        bridge.streams[0].close();bridge.streams=None;bridge.restore()
        assert all(layer.forward==original[i][0] and layer.mamba.forward==original[i][1]
                   for i,layer in enumerate(model.model.layers))
        reports.append(dict(count=count,active=list(bridge.active),omitted_identity_checks=omitted,
            omitted_submodule_calls=0,cache_unchanged=True,native_F0_vs_original_exact=True,
            native_output=extent(native_path),source_methods_restored=True))
    for e in pins:guard();assert extent(e['path'])==e,e['path']
    guard();write(a.out,dict(schema='FIXED_DEPTH_MECHANICAL_PROBE_V1',records=reports,inputs=pins,
        actual_source_classes=True,source_parameters_on_meta=True,actual_native_source_parameters=True,
        GPU_initialized=False,quality_admitted=False,depth_timing_admitted=False,
        decision='SOURCE_CLASS_OMISSION_NATIVE_OWNERSHIP_AND_RESTORE_PASS',
        seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
        scope='No numerical source decoder forward or sparse KV continuation test. Meta source class bypass '
              'plus one actual F0 native token per schedule. Principal/loaded module pins, not entire DLL/OS tree. '
              'Whole own-history quality remains unmeasured.'))
    print(json.dumps(dict(probe=extent(a.out),schedules=[q['active'] for q in reports])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True);a=parser.parse_args()
    import psutil
    proc=psutil.Process();start=time.monotonic();receipt=dict(command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try:
        assert not a.out.exists() and not a.directory.exists();a.directory.mkdir()
        main(a);assert time.monotonic()-start<600 and proc.memory_info().peak_wset<=4<<30
        receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)
