"""Real original-source SSM packets for the frozen original-engine scan test."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/native_expert_scaling'))
from chatbot_falcon_usability import sha, write


def main(args):
    start = time.monotonic()
    assert sha(args.binding) == args.binding_sha
    binding = json.loads(args.binding.read_bytes())
    assert binding['schema'] == 'FALCON_SCAN_BRIDGE_BINDING_V1'
    assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
    assert not any(importlib.util.find_spec(v) for v in ('kernels', 'mamba_ssm', 'causal_conv1d'))
    args.directory.mkdir(exist_ok=False)
    completed = []
    try:
        import numpy as np
        import psutil
        import torch
        import transformers
        from transformers import AutoTokenizer, FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as source_code
        assert torch.__version__ == '2.6.0+cu124' and transformers.__version__ == '5.13.1'
        proc = psutil.Process()
        proc.cpu_affinity([0, 1, 2, 3, 4, 5])
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        def guard():
            assert time.monotonic()-start <= 600
            assert proc.memory_info().peak_wset <= 4 << 30
            assert torch.cuda.max_memory_allocated() <= 6 << 30
            assert torch.cuda.max_memory_reserved() <= 7 << 30
        source = Path(binding['source_directory'])
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        model = FalconH1ForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                    dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
        assert not source_code.is_fast_path_available
        prompt = binding['scan_probe']['prompt']
        messages = [dict(role='user', content=prompt)]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        assert ids == tokenizer.encode(text, add_special_tokens=False) and len(ids) <= 128
        current = {}
        call_index = 0
        packets = []
        def array(t):
            return t.detach().float().cpu().numpy().astype('<f4', copy=False)
        for index in binding['scan_probe']['layers']:
            m = model.model.layers[index].mamba
            assert m.intermediate_size == 768 and m.ssm_state_size == 64 and m.n_groups == 1 and not m.mamba_rms_norm
            def pre(module, positional, kw, i=index):
                cache = kw['cache_params']
                layer = cache.layers[i]
                old = layer.recurrent_states
                current[i] = dict(length=kw['hidden_states'].shape[1],
                    before=array(old) if old is not None else np.zeros((1,24,32,64), dtype='<f4'),
                    before_dtype=str(old.dtype) if old is not None else 'reset zero',
                    cache=cache)
            def proj(module, positional, output, i=index):
                current[i]['proj'] = output.detach().clone()
            def conv(module, positional, output, i=index):
                current[i]['conv'] = output.detach().clone()
            def gated(module, positional, i=index):
                current[i]['gated'] = array(positional[0]).reshape(-1,768)
            def post(module, positional, kw, output, i=index):
                r = current[i]
                T = r['length']
                projection = r['proj'] * module.mup_vector
                gate, _, dt = projection.split([768,896,24], dim=-1)
                dt = torch.nn.functional.softplus(dt + module.dt_bias)
                dt = torch.clamp(dt, module.time_step_limit[0], module.time_step_limit[1])
                convolved = r['conv']
                if T > 1:
                    convolved = convolved[:, :, :T].transpose(1,2)
                else:
                    convolved = convolved.reshape(1,1,896)
                x, B, C = convolved.split([768,64,64], dim=-1)
                A = (-torch.exp(module.A_log.float()))[:,None,None].expand(24,32,64)
                D = module.D[:,None].expand(24,32)
                fields = dict(A=array(A).reshape(768,64), D=array(D).reshape(768),
                    before=r['before'].reshape(768,64),
                    dt=np.repeat(array(dt).reshape(T,24),32,axis=1), x=array(x).reshape(T,768),
                    B=array(B).reshape(T,64), C=array(C).reshape(T,64), z=array(gate).reshape(T,768))
                state = r['cache'].layers[i].recurrent_states
                packet_name = f'call{call_index}_layer{i}'
                packet = args.directory / (packet_name + '.input.f32')
                with packet.open('xb') as f:
                    f.write(struct.pack('<4I',0x53434e31,768,64,T))
                    for name in ('A','D','before'):
                        f.write(fields[name].tobytes())
                    for t in range(T):
                        for name in ('dt','x','B','C','z'):
                            f.write(fields[name][t].tobytes())
                    f.flush()
                    os.fsync(f.fileno())
                oracle = args.directory / (packet_name + '.oracle.f32')
                with oracle.open('xb') as f:
                    f.write(array(state).reshape(768,64).tobytes())
                    f.write(r['gated'].tobytes())
                    f.flush()
                    os.fsync(f.fileno())
                rec = dict(name=packet_name, layer=i, call=call_index, tokens=T,
                     input=dict(path=str(packet.resolve()), bytes=packet.stat().st_size, sha256=sha(packet)),
                     oracle=dict(path=str(oracle.resolve()), bytes=oracle.stat().st_size, sha256=sha(oracle)),
                     source_state_dtype=str(state.dtype), source_previous_dtype=r['before_dtype'],
                     source_projection_dtype=str(r['proj'].dtype), source_convolution_dtype=str(r['conv'].dtype),
                     source_dt_dtype=str(dt.dtype), source_gate_out='BF16 out_proj input promoted to F32')
                write(args.directory / (packet_name+'.json'), rec)
                packets.append(rec)
                completed.append(packet_name)
                current.pop(i)
            m.register_forward_pre_hook(pre, with_kwargs=True)
            m.in_proj.register_forward_hook(proj)
            m.act.register_forward_hook(conv)
            m.out_proj.register_forward_pre_hook(gated)
            m.register_forward_hook(post, with_kwargs=True)
        generated = []
        cache = None
        seen = 0
        inputs = torch.tensor([ids], device='cuda')
        with torch.inference_mode():
            for call_index in range(4):
                seen += inputs.shape[1]
                output = model(input_ids=inputs, attention_mask=torch.ones((1,seen),device='cuda',dtype=torch.long),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                next_id = int(output.logits[0,-1].argmax())
                generated.append(next_id)
                inputs = torch.tensor([[next_id]],device='cuda')
                guard()
                print(json.dumps(dict(call=call_index, packets=len(packets), next_id=next_id,
                          seconds=time.monotonic()-start)), flush=True)
        torch.cuda.synchronize()
        result = dict(schema='FALCON_SCAN_CAPTURE_RESULT_V1', freeze=args.freeze,
                binding_sha256=args.binding_sha, process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
                decision='REAL_SCAN_OPERANDS_CAPTURED', packet_count=len(packets), packets=packets,
                prompt=prompt, serialized=text, input_ids=ids, next_ids=generated, new_source_forwards=4,
                source_training_updates=0, source_forward_path='BF16 eager attention, Torch naive SSD prefill/cached step',
                elapsed_seconds=time.monotonic()-start, GPU_allocated_peak=torch.cuda.max_memory_allocated(),
                GPU_reserved_peak=torch.cuda.max_memory_reserved(), worker_OS_peak_snapshot=proc.memory_info().peak_wset)
        assert len(packets) == 12
        guard()
        write(args.out,result)
    except BaseException as error:
        write(args.directory/'first_failure.json',dict(fault=repr(error),completed=completed,elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
