"""Saved-only source-JSON-tokenizer and full BF16 supervision transport audit."""
import argparse
import json
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
    capture = json.loads(a.capture_result.read_bytes())
    terminal = a.capture_result.with_suffix('.terminal.json')
    assert json.loads(terminal.read_bytes())['exit_code'] == 0
    corpus = json.loads(Path(capture['corpus']['path']).read_bytes())
    source = ROOT / 'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    files = [a.capture_result, terminal, Path(capture['corpus']['path']), B / 'chatbot_hybrid_transfer_cases_v1.json',
             source / 'tokenizer.json', source / 'tokenizer_config.json', source / 'generation_config.json',
             Path(__file__), B / 'chatbot_falcon_usability.py', B / 'chatbot_falcon_usability_launch.py',
             Path(sys.executable), DOC / 'CHATBOT_HYBRID_TRANSFER_ADOPTION_PROTOCOL_20261009.md']
    files += [Path(r['logits']['path']) for r in corpus['records']]
    files += [B / name for name in ('chatbot_hybrid_pilot_cases_v1.json', 'chatbot_falcon_usability_cases_v1.json')]
    inputs = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    write(a.out, dict(schema='HYBRID_TRANSFER_ADOPTION_BINDING_V1', python=str(Path(sys.executable).resolve()),
                      worker_path=str(Path(__file__).resolve()), capture=str(a.capture_result.resolve()),
                      corpus=capture['corpus']['path'], cases=str((B / 'chatbot_hybrid_transfer_cases_v1.json').resolve()),
                      source=str(source.resolve()), limits=dict(seconds=120, OS_bytes=2 << 30, output_bytes=16 << 20),
                      runtime_binding_scope='Saved packets/canonical JSON tokenizer/code/Python hashes and isolated NumPy/tokenizers versions, not an independent BPE or full DLL-tree certificate.',
                      inputs=inputs))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(inputs))), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'HYBRID_TRANSFER_ADOPTION_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    try:
        import numpy as np
        import psutil
        import tokenizers
        assert np.__version__ == '2.4.6' and psutil.__version__ == '7.2.2' and tokenizers.__version__ == '0.22.2'
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))
        source = Path(b['source'])
        tokenizer = tokenizers.Tokenizer.from_file(str(source / 'tokenizer.json'))
        tokenizer_config = json.loads((source / 'tokenizer_config.json').read_bytes())
        bos = tokenizer_config['bos_token']
        if isinstance(bos, dict):
            bos = bos['content']
        assert json.loads((source / 'generation_config.json').read_bytes())['eos_token_id'] == [11, 228]
        spec = json.loads(Path(b['cases']).read_bytes())
        keys = [json.dumps([c.get('history', []), c['prompt']], sort_keys=True) for c in spec['cases'] + spec['reserved_cases']]
        old_cases = sum((json.loads((B / name).read_bytes())['cases'] for name in
                        ('chatbot_hybrid_pilot_cases_v1.json', 'chatbot_falcon_usability_cases_v1.json')), [])
        old_keys = {json.dumps([c.get('history', []), c['prompt']], sort_keys=True) for c in old_cases}
        assert len(set(keys)) == 224 and not (set(keys) & old_keys)
        corpus = json.loads(Path(b['corpus']).read_bytes())
        capture = json.loads(Path(b['capture']).read_bytes())
        assert corpus['schema'] == 'HYBRID_CAPTURED_CALIBRATION_V1' and len(corpus['records']) == 160
        assert sha(b['corpus']) == capture['corpus']['sha256'] and sha(b['cases']) == corpus['cases_sha256']
        records = []
        for expected, actual in zip(spec['cases'], corpus['records'], strict=True):
            assert all(actual[k] == v for k, v in expected.items())
            messages = expected.get('history', []) + [dict(role='user', content=expected['prompt'])]
            text = bos + ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages) + '<|im_start|>assistant\n'
            assert messages == actual['messages'] and text == actual['serialized']
            assert tokenizer.encode(text, add_special_tokens=False).ids == actual['input_ids']
            outputs = actual['output_ids']
            count = len(outputs)
            assert 1 <= count <= spec['max_new_tokens'] == 96
            assert not any(v in (11, 228) for v in outputs[:-1])
            stop = 'EOS' if outputs[-1] in (11, 228) else 'length'
            assert stop == actual['stop'] and (stop == 'EOS' or count == 96)
            assert actual['student_input_ids'] == actual['input_ids'] + outputs[:-1]
            assert actual['positions'] == list(range(len(actual['input_ids'])-1, len(actual['input_ids'])+count-1))
            assert len(actual['input_ids']) + count <= 512
            packet = actual['logits']
            assert packet['dtype'] == 'BF16' and packet['shape'] == [count, 65537]
            assert Path(packet['path']).stat().st_size == packet['bytes'] == count * 65537 * 2
            assert sha(packet['path']) == packet['sha256']
            # Exact bit embedding of each BF16 into F32, no re-quantization.
            values = (np.fromfile(packet['path'], dtype='<u2').astype('<u4') << 16).view('<f4').reshape(count, 65537)
            assert np.isfinite(values).all() and values.argmax(-1).tolist() == outputs
            assert actual['finite'] and actual['BF16_lossless'] and actual['raw_argmax_equals_generated']
            records.append(dict(id=actual['id'], split=actual['split'], domain=actual['domain'], labels=count,
                                prompt_ids=len(actual['input_ids']), stop=stop, bytes=packet['bytes']))
            del values
            assert time.monotonic()-start <= 120 and proc.memory_info().peak_wset <= 2 << 30
            assert not proc.children(recursive=True)
        summary = []
        for domain in spec['domains']:
            for split in ('FIT', 'DEV'):
                rows = [r for r in records if r['domain'] == domain and r['split'] == split]
                assert len(rows) == (16 if split == 'FIT' else 4)
                summary.append(dict(domain=domain, split=split, cases=len(rows), labels=sum(r['labels'] for r in rows),
                                    EOS=sum(r['stop'] == 'EOS' for r in rows), length=sum(r['stop'] == 'length' for r in rows)))
        assert summary == capture['summary'] and sum(r['labels'] for r in records) == capture['labels']
        assert len({r['id'] for r in records}) == 160 and not ({r['id'] for r in records} & {r['id'] for r in spec['reserved_cases']})
        result = dict(schema='HYBRID_TRANSFER_ADOPTION_RESULT_V1', decision='SAVED_CALIBRATION_TRANSPORT_PASS',
                      freeze=a.freeze, binding_sha256=a.binding_sha,
                      process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()),
                      cases=160, labels=capture['labels'], coordinates=capture['labels'] * 65537,
                      BF16_packet_bytes=sum(r['bytes'] for r in records), summary=summary, case_records=records,
                      source_generations=0, student_forwards=0, training_updates=0, native_runs=0,
                      worker_OS_peak_snapshot=proc.memory_info().peak_wset, elapsed_seconds=time.monotonic()-start,
                      scope='HF-free source-JSON Rust tokenizer and independent bit/shape/argmax/EOS/position transport verification. Shared Rust tokenizer implementation, not independent BPE. Source BF16-lossless flag remains producer observation; no source-answer truth or chatbot preservation admission.')
        write(a.out, result)
        print(json.dumps(dict(decision=result['decision'], cases=160, labels=capture['labels'])), flush=True)
    except BaseException as error:
        write(a.directory / 'first_failure.json', dict(fault=repr(error), elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('bind', 'worker'), default='worker')
    p.add_argument('--capture-result', type=Path)
    p.add_argument('--binding', type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    bind(a) if a.mode == 'bind' else worker(a)
