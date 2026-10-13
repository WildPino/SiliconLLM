"""CPU-only consumer wiring test; no source model/chat/quality admission."""
import argparse
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'benchmarks/native_expert_scaling'),
               str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from full_width_native_chat import NativeBridge
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
from token_state import arrays,arguments


def main(a):
    import numpy as np
    import torch
    import psutil
    start = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized() and not a.directory.exists() and not a.out.exists()
    b = json.loads(a.binding.read_bytes()); assert extent(a.binding)['sha256'] == a.binding_sha
    a.directory.mkdir()
    get,stream = tensor_reader(Path(b['source'])/'model.safetensors',np)
    layers = [types.SimpleNamespace(mamba=types.SimpleNamespace()) for _ in range(24)]
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
    bridge = NativeBridge(model,b,np,torch)
    refs = [bridge.create() for _ in range(24)]
    bridge.reset(a.directory,'consumer')
    hidden = torch.from_numpy(np.linspace(-1,1,2048,dtype='f4')).to(torch.bfloat16).reshape(1,1,2048)
    all_exact = True; rows_checked = 0; retained = None
    try:
        for site in range(24):
            mixer = layers[site].mamba
            mixer.ssm_in_multiplier = .625
            if site:
                G = torch.from_numpy(get(f'model.layers.{site}.mamba.in_proj.weight').astype('f4')).to(torch.bfloat16)
                mixer.in_proj = lambda h,G=G: torch.nn.functional.linear(h,G)
                mixer.mup_vector = torch.from_numpy(np.array(b['MuP_bits'],'u2')).view(torch.bfloat16).reshape(1,1,6704)
            for pos,token in enumerate([17,65536]):
                bridge.token = token; bridge.steps = pos
                cache = types.SimpleNamespace(get_seq_length=lambda idx,site=site,pos=pos: pos if idx == site else -1)
                with torch.inference_mode(): value = mixer.forward(hidden,cache_params=cache)
                assert value.device.type == 'cpu' and value.dtype == torch.bfloat16 and value.shape == (1,1,2048)
                p,z = bridge.params[site],arrays(1,np); tags = np.array([token],'u4')
                if site:
                    with torch.inference_mode():
                        row = mixer.in_proj(hidden*.625)*mixer.mup_vector
                    rowbits = row[0].contiguous().view(torch.uint16).numpy()
                    rc = bridge.rows(refs[site],1,tags.ctypes.data,rowbits.ctypes.data,
                        bridge.weights[site],bridge.conv[site],p['q'].ctypes.data,p['scale'].ctypes.data,
                        *arguments(z,0,1,np))
                else:
                    rc = bridge.tokens(refs[site],1,tags.ctypes.data,bridge.table.ctypes.data,
                        bridge.weights[site],bridge.conv[site],p['q'].ctypes.data,p['scale'].ctypes.data,
                        *arguments(z,0,1,np))
                assert rc == 0
                all_exact &= np.array_equal(value.view(torch.uint16).numpy().reshape(1,2048),z['output'])
                if pos:
                    assert np.array_equal(retained[0].view(torch.uint16).numpy(),retained[1]), 'output aliases reused native buffer'
                retained = (value,value.view(torch.uint16).numpy().copy())
                rows_checked += 1
            for context in (bridge.contexts[site],refs[site]):
                st = np.empty((3072,96),'f4'); rg = np.empty((3,3584),'u2'); seen = np.zeros(1,'u8')
                assert bridge.snapshot(context,st.ctypes.data,rg.ctypes.data,seen.ctypes.data) == 0
                assert seen[0] == 2 and bridge.last(context) == 65536
                if context == bridge.contexts[site]: first = (st.copy(),rg.copy())
                else: all_exact &= np.array_equal(st.view('u4'),first[0].view('u4')) and np.array_equal(rg,first[1])
            assert time.monotonic()-start <= 300
        for f in bridge.streams: f.flush(); f.close()
        bridge.streams = None
        assert rows_checked == 48 and all_exact and not torch.cuda.is_initialized()
        result = dict(schema='FULL_WIDTH_NATIVE_CHAT_ADAPTER_SMOKE_V1',source=extent(Path(__file__)),
            adapter_source=extent(ROOT/'benchmarks/native_expert_scaling/full_width_native_chat.py'),
            binding=extent(a.binding),positions=48,sites=24,all_exact=bool(all_exact),
            persistent_seen_and_last_exact=True,output_owns_storage=True,
            GPU_initialized=False,source_model_forwards=0,
            outputs=[extent(p) for p in sorted(a.directory.iterdir())],
            seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            scope='CPU consumer adapter versus direct ABI,all24 actual source in_proj/packets '
                  'on two arbitrary fixed inputs. Mock attention cache tests layer-index check. '
                  'Not actual source-model/cache/generation quality or performance.')
        write(a.out,result); print(json.dumps(dict(result=extent(a.out),all_exact=bool(all_exact))),flush=True)
    finally:
        if bridge.streams is not None:
            for f in bridge.streams: f.close()
        bridge.close_contexts()
        for context in refs: bridge.destroy(context)
        stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('binding','directory','out'): p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--binding-sha',required=True)
    main(p.parse_args())
