"""Saved-only verification of every new broad BF16 supervision coordinate."""
import argparse
from collections import Counter
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
    terminal_path = a.capture_result.with_suffix('.terminal.json')
    terminal = json.loads(terminal_path.read_bytes())
    assert capture['decision'] == 'BROAD_SOURCE_CAPTURE_COMPLETE' and terminal['exit_code'] == 0
    assert terminal['result_sha256'] == sha(a.capture_result)
    corpus_path = Path(capture['corpus']['path'])
    assert sha(corpus_path) == capture['corpus']['sha256']
    corpus = json.loads(corpus_path.read_bytes())
    cases = Path(corpus['cases_path'])
    assert sha(cases) == corpus['cases_sha256']
    data_result = Path(corpus['data_result_path'])
    assert sha(data_result) == corpus['data_result_sha256']
    source = ROOT / 'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    files = [a.capture_result, terminal_path, corpus_path, cases, data_result,
              data_result.with_suffix('.terminal.json'), Path(json.loads(data_result.read_bytes())['corpus']['path']),
              Path(capture['binding_path']),
              DOC / 'CHATBOT_BROAD_ADOPTION_PROTOCOL_20261009.md', Path(__file__),
              B / 'chatbot_falcon_usability.py', B / 'chatbot_falcon_usability_launch.py',
              source / 'tokenizer.json', source / 'tokenizer_config.json', source / 'generation_config.json',
              SITE / 'tokenizers/tokenizers.pyd', Path(sys.executable)]
    files += [Path(r['logits']['path']) for r in corpus['records']]
    files += [corpus_path.parent / (r['id']+'.json') for r in corpus['records']]
    files += [ROOT / name for name in ('benchmarks/donor_adaptation/configs/_manifest.json',
              'benchmarks/donor_adaptation/density/build_document_holdout.py', 'docs/research/RESEARCH_INDEX.md')]
    files = list(dict.fromkeys(p.resolve() for p in files))
    inputs = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    write(a.out, dict(schema='BROAD_CHAT_ADOPTION_BINDING_V1', python=str(Path(sys.executable).resolve()),
          worker_path=str(Path(__file__).resolve()), capture=str(a.capture_result.resolve()),
          corpus=str(corpus_path), cases=str(cases), data_result=str(data_result), source=str(source.resolve()),
          limits=dict(seconds=180, OS_bytes=2 << 30, output_bytes=16 << 20),
          runtime_binding_scope='Saved packets/case/source JSON/selected tokenizer binary/Python/code/foreign hashes; isolated NumPy and Rust-tokenizer versions. No independent BPE or full native DLL-tree certificate.',
          inputs=inputs))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(inputs))), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'BROAD_CHAT_ADOPTION_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    completed = []
    try:
        import numpy as np
        import psutil
        import tokenizers
        assert np.__version__ == '2.4.6' and psutil.__version__ == '7.2.2' and tokenizers.__version__ == '0.22.2'
        assert sys.version_info[:3] == (3, 12, 10) and Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert not any(name in sys.modules for name in ('torch', 'transformers', 'pyarrow'))
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))

        def guard():
            assert time.monotonic()-start <= b['limits']['seconds']
            assert proc.memory_info().peak_wset <= b['limits']['OS_bytes']
            assert not proc.children(recursive=True)
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= b['limits']['output_bytes']

        source = Path(b['source'])
        tokenizer = tokenizers.Tokenizer.from_file(str(source / 'tokenizer.json'))
        config = json.loads((source / 'tokenizer_config.json').read_bytes())
        bos = config['bos_token']
        if isinstance(bos, dict):
            bos = bos['content']
        assert tokenizer.token_to_id(bos) == 17 and tokenizer.get_vocab_size() == 65537
        assert json.loads((source / 'generation_config.json').read_bytes())['eos_token_id'] == [11, 228]
        spec = json.loads(Path(b['cases']).read_bytes())
        assert spec['schema'] == 'BROAD_CHAT_CAPTURE_CASES_V1' and len(spec['cases']) == 48
        assert Counter(c['split'] for c in spec['cases']) == Counter(FIT=24, DEV=24)
        assert spec['max_new_tokens'] == 256 and spec['context_limit'] == 2304
        data_result = json.loads(Path(b['data_result']).read_bytes())
        assert data_result['decision'] == 'BROAD_DATA_ADOPTION_PASS'
        parent_data = json.loads(Path(data_result['corpus']['path']).read_bytes())
        by_id = {r['id']: r for r in parent_data['records']}
        reserved = {r['id'] for r in parent_data['records'] if r['split'] == 'RESERVED'}
        assert len(reserved) == 104 and not ({c['id'] for c in spec['cases']} & reserved)
        corpus = json.loads(Path(b['corpus']).read_bytes())
        capture = json.loads(Path(b['capture']).read_bytes())
        assert corpus['schema'] == 'BROAD_CHAT_CAPTURED_CORPUS_V1' and len(corpus['records']) == 48
        assert sha(b['corpus']) == capture['corpus']['sha256'] and sha(b['cases']) == corpus['cases_sha256']
        assert corpus['source_revision'] == capture['source_revision'] == '80ebc50d7799a440b96c93bb6686a3924a09b0cb'
        records = []
        for expected, actual in zip(spec['cases'], corpus['records'], strict=True):
            assert all(actual[k] == v for k, v in expected.items())
            adopted = by_id[expected['id']]
            assert all(expected[k] == adopted[k] for k in ('id', 'split', 'messages', 'input_ids', 'input_length', 'group_key', 'prefix_sha256'))
            assert expected['domain'] == adopted['source']
            assert json.loads((Path(b['corpus']).parent / (actual['id']+'.json')).read_bytes()) == actual
            messages = expected['messages']
            text = bos + ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages) + '<|im_start|>assistant\n'
            assert text == actual['serialized'] and tokenizer.encode(text, add_special_tokens=False).ids == actual['input_ids']
            outputs = actual['output_ids']
            count = len(outputs)
            assert 1 <= count <= 256 and all(type(v) is int and 0 <= v < 65537 for v in outputs)
            assert not any(v in (11, 228) for v in outputs[:-1])
            stop = 'EOS' if outputs[-1] in (11, 228) else 'length'
            assert stop == actual['stop'] and (stop == 'EOS' or count == 256)
            assert tokenizer.decode(outputs, skip_special_tokens=True) == actual['output_text']
            assert actual['student_input_ids'] == actual['input_ids']+outputs[:-1]
            assert actual['positions'] == list(range(len(actual['input_ids'])-1, len(actual['input_ids'])+count-1))
            assert len(actual['input_ids']) == actual['input_length'] and len(actual['input_ids'])+count <= 2304
            packet = actual['logits']
            assert packet['dtype'] == 'BF16' and packet['shape'] == [count, 65537]
            assert Path(packet['path']).stat().st_size == packet['bytes'] == count*65537*2
            assert sha(packet['path']) == packet['sha256']
            bits = np.fromfile(packet['path'], dtype='<u2')
            values = (bits.astype('<u4') << 16).view('<f4').reshape(count, 65537)
            assert np.isfinite(values).all() and values.argmax(-1).tolist() == outputs
            assert actual['finite'] and actual['BF16_lossless'] and actual['raw_argmax_equals_generated']
            records.append(dict(id=actual['id'], split=actual['split'], domain=actual['domain'], labels=count,
                prompt_ids=len(actual['input_ids']), stop=stop, bytes=packet['bytes'],
                context_origin=actual['context_origin']))
            completed.append(actual['id'])
            del values, bits
            guard()
        summary = []
        for domain in spec['domains']:
            for split in ('FIT', 'DEV'):
                rows = [r for r in records if r['domain'] == domain and r['split'] == split]
                assert len(rows) == 2
                summary.append(dict(domain=domain, split=split, cases=len(rows), labels=sum(r['labels'] for r in rows),
                                    EOS=sum(r['stop'] == 'EOS' for r in rows), length=sum(r['stop'] == 'length' for r in rows)))
        assert summary == capture['summary'] and sum(r['labels'] for r in records) == capture['labels']
        assert sum(r['bytes'] for r in records) == capture['BF16_packet_bytes']
        assert len({r['id'] for r in records}) == len({r['group_key'] for r in spec['cases']}) == 48
        result = dict(schema='BROAD_CHAT_ADOPTION_RESULT_V1', decision='SAVED_BROAD_TRANSPORT_PASS',
            freeze=a.freeze, binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()),
            cases=48, labels=capture['labels'], coordinates=capture['labels']*65537,
            BF16_packet_bytes=capture['BF16_packet_bytes'], summary=summary, case_records=records,
            source_generations=0, student_forwards=0, training_updates=0, native_runs=0, reserved_queries=0,
            corpus=dict(path=b['corpus'], sha256=sha(b['corpus'])),
            gates=dict(all_packet_hashes_extents=True, all_finite_argmax=True, all_template_ids=True,
                       all_EOS_positions=True, all_public_context_provenance=True, reserved_unqueried=True),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset, elapsed_seconds=time.monotonic()-start,
            scope='All saved BF16 coordinates/IDs/text/EOS/positions transported without source/model/GPU calls. Shared Rust tokenizer,not independent BPE;losslessness remains producer observation. No answer correctness,own-history,quality or speed admission.')
        write(a.directory / 'packet_audit.json', dict(records=records, summary=summary))
        write(a.out, result)
        guard()
        print(json.dumps(dict(decision=result['decision'], cases=48, labels=result['labels'])), flush=True)
    except BaseException as error:
        write(a.directory / 'first_failure.json', dict(fault=repr(error), completed=completed, elapsed_seconds=time.monotonic()-start))
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
