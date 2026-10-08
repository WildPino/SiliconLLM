"""One offline original-source teacher screen. No fitting or native speed claim."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path.insert(0, str(SITE))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def normalized(text):
    text = text.strip()
    if text.endswith('.'):
        text = text[:-1].strip()
    for mark in ('"', "'", '`'):
        if len(text) >= 2 and text.startswith(mark) and text.endswith(mark):
            text = text[1:-1].strip()
            break
    return text


def main(args):
    start = time.monotonic()
    assert sha(args.binding) == args.binding_sha
    binding = json.loads(args.binding.read_bytes())
    assert binding['schema'] == 'FALCON_USABILITY_BINDING_V1'
    assert sys.version_info[:3] == (3, 12, 10)
    assert Path(sys.executable).resolve() == Path(binding['python']).resolve()
    assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
    args.directory.mkdir(exist_ok=False)
    completed = []
    runtime = {}
    torch = None
    try:
        # The isolated view contains no optional hub/SSM kernels; reject a changed view.
        optional = {v: importlib.util.find_spec(v) is not None for v in ('kernels', 'mamba_ssm', 'causal_conv1d')}
        assert not any(optional.values()), optional
        import psutil
        import torch as torch_module
        import transformers
        import tokenizers
        import numpy as np
        from transformers import AutoTokenizer, FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as source_code
        torch = torch_module
        assert torch.__version__ == '2.6.0+cu124' and transformers.__version__ == '5.13.1'
        assert tokenizers.__version__ == '0.22.2' and np.__version__ == '2.4.6'
        assert psutil.__version__ == '7.2.2'
        assert torch.cuda.is_available()
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        proc = psutil.Process()
        proc.cpu_affinity([0, 1, 2, 3, 4, 5])
        runtime = {v.__name__: dict(version=v.__version__, path=v.__file__) for v in (torch, transformers, tokenizers, np, psutil)}
        runtime.update(optional_kernels=optional, fast_ssm_path=source_code.is_fast_path_available,
                       device=torch.cuda.get_device_name(), attention='eager', source_code=source_code.__file__)
        assert not source_code.is_fast_path_available
        def guard():
            assert time.monotonic() - start <= 600, 'worker deadline'
            assert proc.memory_info().peak_wset <= 4 << 30, 'OS cap'
            assert torch.cuda.max_memory_allocated() <= 6 << 30, 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= 7 << 30, 'GPU reserved cap'
            assert not proc.children(recursive=True), 'unexpected subprocess'
            assert sum(p.stat().st_size for p in args.directory.iterdir() if p.is_file()) <= 256 << 20, 'output cap'
        source = Path(binding['source_directory'])
        cases = json.loads(Path(binding['cases_path']).read_bytes())
        assert len(cases['cases']) == 16 and cases['max_new_tokens'] == 64
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        model = FalconH1ForCausalLM.from_pretrained(source, local_files_only=True,
                   trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        assert model.config.embedding_multiplier == .11083984375 and model.config.lm_head_multiplier == .078125
        assert len(model.model.layers) == 24
        eos = model.generation_config.eos_token_id
        assert eos == [228, 11] and model.generation_config.pad_token_id == 0
        assert tokenizer.chat_template == (source / 'chat_template.jinja').read_text(encoding='utf8')
        special = dict(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
        # Canonical serialization also includes the assistant-history newline.
        def serialize(messages):
            s = ''
            for m in messages:
                content = m['content'] + ('\n' if m['role'] == 'assistant' and m['content'] else '')
                s += '<|im_start|>' + m['role'] + '\n' + content + '<|im_end|>\n'
            return s + '<|im_start|>assistant\n'
        contract = dict(runtime=runtime, source_revision=binding['source_revision'], eos_ids=eos,
              special_tokens=special, tied_head=True, config=model.config.to_dict(),
              parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
              trainable_named_elements=sum(p.numel() for p in model.parameters()),
              buffer_dtypes=sorted({str(p.dtype) for p in model.buffers()}), source_freeze=args.freeze)
        write(args.directory / 'source_contract.json', contract)
        guard()
        cache_layout = None
        with torch.inference_mode():
            for case in cases['cases']:
                case_start = time.monotonic()
                messages = case.get('history', []) + [dict(role='user', content=case['prompt'])]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                assert text == serialize(messages), ('template', case['id'])
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                assert ids == tokenizer.encode(text, add_special_tokens=False), ('tokenizer', case['id'])
                assert len(ids) + 64 <= 256, ('context cap', case['id'])
                inputs = torch.tensor([ids], device='cuda', dtype=torch.long)
                torch.cuda.synchronize()
                generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                  max_new_tokens=64, do_sample=False, use_cache=True,
                  return_dict_in_generate=True, output_scores=True)
                torch.cuda.synchronize()
                new_ids = generated.sequences[0, len(ids):].tolist()
                scores = torch.stack(generated.scores).squeeze(1).float().cpu().numpy()
                assert scores.shape == (len(new_ids), model.config.vocab_size)
                assert np.isfinite(scores).all(), 'nonfinite generation scores'
                assert scores.argmax(axis=1).tolist() == new_ids, 'greedy ID mismatch'
                score_path = args.directory / (case['id'] + '.scores.f32')
                with score_path.open('xb') as f:
                    f.write(scores.astype('<f4', copy=False).tobytes())
                    f.flush()
                    os.fsync(f.fileno())
                decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
                stop_ok = not any(v in eos for v in new_ids[:-1]) and (new_ids[-1] in eos or len(new_ids) == 64)
                leak = any(v in set(tokenizer.all_special_ids) - set(eos) for v in new_ids)
                rec = dict(id=case['id'], category=case['category'], expected=case['expected'],
                    messages=messages, serialized=text, input_ids=ids, output_ids=new_ids,
                    output_text=decoded, output_with_special=tokenizer.decode(new_ids, skip_special_tokens=False),
                    normalized=normalized(decoded), correct=normalized(decoded) == case['expected'],
                    blank=not decoded.strip(), role_or_other_special_leak=leak, stop_policy_ok=stop_ok,
                    eos_stop=new_ids[-1] in eos, scores=dict(path=str(score_path.resolve()),
                    shape=list(scores.shape), dtype='little-endian F32 generation scores', sha256=sha(score_path)),
                    elapsed_seconds=time.monotonic()-case_start)
                write(args.directory / (case['id'] + '.json'), rec)
                completed.append(rec)
                if cache_layout is None:
                    cache_layout = [dict(index=i, conv_shape=list(layer.conv_states.shape),
                        conv_dtype=str(layer.conv_states.dtype), recurrent_shape=list(layer.recurrent_states.shape),
                        recurrent_dtype=str(layer.recurrent_states.dtype)) for i, layer in enumerate(generated.past_key_values.layers)]
                print(json.dumps(dict(case=rec['id'], answer=decoded, correct=rec['correct'],
                    completed=len(completed), seconds=time.monotonic()-start)), flush=True)
                del generated, inputs, scores
                guard()
        categories = {c: sum(v['correct'] for v in completed if v['category'] == c) for c in sorted({v['category'] for v in completed})}
        gates = dict(at_least_12=sum(v['correct'] for v in completed) >= 12,
                 every_category_at_least_2=all(v >= 2 for v in categories.values()),
                 blank_at_most_1=sum(v['blank'] for v in completed) <= 1,
                 no_special_leak=not any(v['role_or_other_special_leak'] for v in completed),
                 stop_policy=all(v['stop_policy_ok'] for v in completed))
        guard()
        result = dict(schema='FALCON_USABILITY_RESULT_V1', freeze=args.freeze, binding_sha256=args.binding_sha,
              process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()), runtime=runtime,
              decision='TEACHER_SCREEN_PASS' if all(gates.values()) else 'TEACHER_SCREEN_FAIL',
              quality_gates=gates, category_correct=categories, correct=sum(v['correct'] for v in completed),
              total=16, supplied_history_scope='Not model-own-history or broad/final chatbot quality',
              cache_layout=cache_layout, elapsed_seconds=time.monotonic()-start,
              GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
              worker_OS_peak_snapshot=proc.memory_info().peak_wset,
              new_source_generations=16, source_training_updates=0)
        write(args.out, result)
        print(json.dumps(dict(decision=result['decision'], correct=result['correct'], categories=categories)), flush=True)
    except BaseException as error:
        write(args.directory / 'first_failure.json', dict(fault=repr(error), completed=[v['id'] for v in completed],
                  runtime=runtime, elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--binding-sha', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
