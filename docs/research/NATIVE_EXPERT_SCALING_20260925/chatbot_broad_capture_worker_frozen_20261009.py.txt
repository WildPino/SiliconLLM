"""Capture the fixed broad Falcon cohort; original160 packets remain untouched."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    source = ROOT / 'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    package = json.loads((source / 'source_package.json').read_bytes())
    assert package['schema'] == 'HYBRID_ORIGINAL_SOURCE_PACKAGE_V1'
    data_result = DOC / 'chatbot_broad_data_result_repair1_20261009.json'
    adoption = json.loads(data_result.read_bytes())
    data_terminal = data_result.with_suffix('.terminal.json')
    terminal = json.loads(data_terminal.read_bytes())
    assert adoption['decision'] == 'BROAD_DATA_ADOPTION_PASS' and terminal['exit_code'] == 0
    assert terminal['result_sha256'] == sha(data_result)
    corpus = Path(adoption['corpus']['path'])
    assert sha(corpus) == adoption['corpus']['sha256']
    cases = corpus.parent / 'broad_capture_cases.json'
    spec = json.loads(cases.read_bytes())
    assert spec['schema'] == 'BROAD_CHAT_CAPTURE_CASES_V1' and len(spec['cases']) == 48
    assert spec['data_result_sha256'] == sha(data_result)
    assert spec['parent_corpus']['sha256'] == sha(corpus)
    files = [Path(v['path']) for v in package['files']] + [source / 'source_package.json']
    files += [data_result, data_terminal, corpus, cases,
              DOC / 'chatbot_broad_data_binding_repair1_20261009.json',
              DOC / 'CHATBOT_BROAD_CAPTURE_PROTOCOL_20261009.md',
              DOC / 'CHATBOT_BROAD_DATA_RESULT_20261009.md',
              DOC / 'CHATBOT_BROAD_CAPTURE_NEXT_20261009.md', Path(sys.executable)]
    files += [B / name for name in ('chatbot_broad_capture.py', 'chatbot_broad_capture_cases.py',
              'chatbot_hybrid_transfer_capture.py', 'chatbot_falcon_usability.py',
              'chatbot_falcon_usability_launch.py')]
    files += [ROOT / name for name in ('benchmarks/donor_adaptation/configs/_manifest.json',
              'benchmarks/donor_adaptation/density/build_document_holdout.py', 'docs/research/RESEARCH_INDEX.md')]
    files += [SITE / 'transformers' / name for name in ('models/falcon_h1/modeling_falcon_h1.py',
              'models/falcon_h1/configuration_falcon_h1.py', 'cache_utils.py', 'generation/utils.py',
              'generation/configuration_utils.py', 'tokenization_utils_base.py', 'tokenization_utils_tokenizers.py')]
    files += [SITE / 'tokenizers/tokenizers.pyd']
    files = list(dict.fromkeys(p.resolve() for p in files))
    inputs = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    by_path = {v['path']: v['sha256'] for v in inputs}
    assert all(by_path[str(Path(v['path']).resolve())] == v['sha256'] for v in package['files'])
    write(a.out, dict(schema='BROAD_CHAT_CAPTURE_BINDING_V1', python=str(Path(sys.executable).resolve()),
          worker_path=str(Path(__file__).resolve()), source=str(source.resolve()),
          source_revision=package['revision'], source_named_elements=package['total_named_elements'],
          cases=str(cases.resolve()), data_result=str(data_result.resolve()),
          limits=dict(seconds=2700, reserve_seconds=30, OS_bytes=8 << 30,
                      GPU_allocated_bytes=10 << 30, GPU_reserved_bytes=11 << 30, output_bytes=2 << 30),
          runtime_binding_scope='Pinned source package/adopted data/selected runtime/code/Python/foreign hashes; not a full native DLL-tree certificate.',
          inputs=inputs))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(inputs))), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'BROAD_CHAT_CAPTURE_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    completed = []
    new_calls = 0
    active_case = None
    torch = None
    try:
        assert Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert sys.version_info[:3] == (3, 12, 10)
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        import tokenizers
        from transformers import AutoTokenizer, FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as source_code
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        assert transformers.__version__ == '5.13.1' and tokenizers.__version__ == '0.22.2'
        assert psutil.__version__ == '7.2.2'
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))

        def guard(reserve=False):
            lim = b['limits']
            assert time.monotonic()-start <= lim['seconds']-(lim['reserve_seconds'] if reserve else 0), 'worker time reserve'
            assert proc.memory_info().peak_wset <= lim['OS_bytes'], 'worker OS cap'
            assert torch.cuda.max_memory_allocated() <= lim['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= lim['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'unexpected worker child'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= lim['output_bytes'], 'output cap'

        spec = json.loads(Path(b['cases']).read_bytes())
        cases = spec['cases']
        assert spec['schema'] == 'BROAD_CHAT_CAPTURE_CASES_V1' and len(cases) == 48
        assert Counter(c['split'] for c in cases) == Counter(FIT=24, DEV=24)
        assert len(spec['domains']) == 12 and spec['max_new_tokens'] == 256 and spec['context_limit'] == 2304
        assert len({c['group_key'] for c in cases}) == len({c['id'] for c in cases}) == 48
        assert all(c['split'] != 'RESERVED' for c in cases)
        source = Path(b['source'])
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        assert tokenizer.bos_token_id == 17 and len(tokenizer) == 65537
        assert json.loads((source / 'generation_config.json').read_bytes())['eos_token_id'] == [11, 228]
        preflight = []
        for case in cases:
            messages = case['messages']
            assert messages[-1]['role'] == 'user'
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            expected = tokenizer.bos_token + ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages) + '<|im_start|>assistant\n'
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
            assert text == expected and ids == tokenizer.encode(text, add_special_tokens=False) == case['input_ids']
            assert len(ids) == case['input_length'] and len(ids)+spec['max_new_tokens'] <= spec['context_limit']
            preflight.append(dict(**case, serialized=text))
        write(a.directory / 'interaction_preflight.json', preflight)
        guard(reserve=True)
        model = FalconH1ForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                    dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
        assert sum(p.numel() for p in model.parameters()) == b['source_named_elements']
        assert not source_code.is_fast_path_available and model.generation_config.eos_token_id == [11, 228]
        records = []
        for case in preflight:
            guard(reserve=True)
            active_case = case['id']
            case_start = time.monotonic()
            with torch.inference_mode():
                inputs = torch.tensor([case['input_ids']], device='cuda')
                result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        max_new_tokens=spec['max_new_tokens'], do_sample=False, use_cache=True,
                        return_dict_in_generate=True, output_logits=True)
                new_calls += 1
                outputs = result.sequences[0, len(case['input_ids']):].tolist()
                logits = torch.stack(result.logits).squeeze(1)
                encoded = logits.to(torch.bfloat16)
                raw = a.directory / (case['id'] + '.logits.bf16')
                with raw.open('xb') as stream:
                    stream.write(encoded.contiguous().view(torch.uint16).cpu().numpy().astype('<u2', copy=False).tobytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                record = dict(**case, output_ids=outputs,
                    output_text=tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    stop=('EOS' if outputs[-1] in (11, 228) else 'length') if outputs else 'empty',
                    raw_logits_dtype=str(logits.dtype), BF16_lossless=bool(torch.equal(encoded.float(), logits.float())),
                    raw_argmax_equals_generated=bool(logits.argmax(-1).tolist() == outputs),
                    finite=bool(torch.isfinite(logits).all()),
                    logits=dict(path=str(raw.resolve()), shape=list(logits.shape), dtype='BF16',
                                bytes=raw.stat().st_size, sha256=sha(raw)),
                    student_input_ids=case['input_ids']+outputs[:-1],
                    positions=list(range(len(case['input_ids'])-1, len(case['input_ids'])+len(outputs)-1)),
                    source_and_packet_seconds=time.monotonic()-case_start)
                # Save every completed generation before any transport or resource admission.
                write(a.directory / (case['id'] + '.json'), record)
                completed.append(case['id'])
                del result, logits, encoded, inputs
            assert record['finite'] and record['BF16_lossless'] and record['raw_argmax_equals_generated']
            assert record['logits']['shape'] == [len(outputs), 65537]
            assert 1 <= len(outputs) <= spec['max_new_tokens']
            assert not any(v in (11, 228) for v in outputs[:-1])
            assert record['stop'] == 'EOS' or len(outputs) == spec['max_new_tokens']
            records.append(record)
            active_case = None
            guard()
            print(json.dumps(dict(cases=len(records), labels=sum(len(r['output_ids']) for r in records),
                  case=case['id'], stop=record['stop'], case_seconds=record['source_and_packet_seconds'],
                  seconds=time.monotonic()-start)), flush=True)
        summary = []
        for domain in spec['domains']:
            for split in ('FIT', 'DEV'):
                rows = [r for r in records if r['domain'] == domain and r['split'] == split]
                assert len(rows) == 2
                summary.append(dict(domain=domain, split=split, cases=len(rows),
                    labels=sum(len(r['output_ids']) for r in rows), EOS=sum(r['stop'] == 'EOS' for r in rows),
                    length=sum(r['stop'] == 'length' for r in rows)))
        corpus_path = a.directory / 'corpus.json'
        write(corpus_path, dict(schema='BROAD_CHAT_CAPTURED_CORPUS_V1', records=records,
              source_revision=b['source_revision'], cases_path=b['cases'], cases_sha256=sha(b['cases']),
              data_result_path=b['data_result'], data_result_sha256=sha(b['data_result'])))
        report = dict(schema='BROAD_CHAT_CAPTURE_RESULT_V1', decision='BROAD_SOURCE_CAPTURE_COMPLETE',
            freeze=a.freeze, binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()),
            source_revision=b['source_revision'], cases=48, FIT_cases=24, DEV_cases=24,
            reserved_queries=0, new_source_generations=new_calls, labels=sum(len(r['output_ids']) for r in records),
            BF16_packet_bytes=sum(r['logits']['bytes'] for r in records), summary=summary,
            corpus=dict(path=str(corpus_path.resolve()), bytes=corpus_path.stat().st_size, sha256=sha(corpus_path)),
            GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset, elapsed_seconds=time.monotonic()-start,
            student_forwards=0, training_updates=0, native_runs=0,
            runtime=dict(torch=torch.__version__, transformers=transformers.__version__,
                         device=torch.cuda.get_device_name(), source_code=source_code.__file__, TF32=False),
            scope='New source supervision on fixed12 public-source strata; public prior assistant context is not donor own-history. All partial replies retained; no correctness or broad quality/speed admission. Longalign deferred intact.')
        write(a.out, report)
        guard()
        print(json.dumps(dict(decision=report['decision'], labels=report['labels'])), flush=True)
    except BaseException as error:
        detail = dict(fault=repr(error), active_case=active_case, durable_completed=completed,
                      new_source_generations=new_calls, elapsed_seconds=time.monotonic()-start)
        if torch is not None:
            detail.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory / 'first_failure.json', detail)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('bind', 'worker'), default='worker')
    p.add_argument('--binding', type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    bind(a) if a.mode == 'bind' else worker(a)
