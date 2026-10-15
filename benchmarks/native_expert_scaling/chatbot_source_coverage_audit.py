"""CPU-only, bounded adoption audit of the completed source-coverage capture."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

# Set these before importing any numerical or tokenizer package. No GPU/model API is used.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '6'

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0] = [str(Path(__file__).parent), str(SITE)]
from falcon_auxiliary_s6_transport import extent, write
from original_engine_chat import ChatTokenizer
from chatbot_broad_data_reader import dialogue_group

PROTOCOL = DOC / 'CHATBOT_SOURCE_COVERAGE_AUDIT_20261015.md'
REPORT = DOC / 'chatbot_source_coverage_result_20261015.json'
OUT = DOC / 'chatbot_source_coverage_audit_20261015.json'
EXPECTED_REPORT = '368b2c1d1f70248dfb66cc02e4b2fabbc022597b903599e902cae992c40766d1'
EXPECTED_BINDING = '23c93928549d2d16c12aa901b9f1f873daa3b0ff404fa777e7df2f0e58d6a64a'
EXPECTED_CORPUS = '1c42eb789f52258d7796bcaeb8b52387c648ffe803f8955ecea5c2f3bdbd9ad6'
LIMIT_SECONDS = 1800
LIMIT_OS_BYTES = 6 << 30


def load(path):
    return json.loads(Path(path).read_bytes())


def sealed(descriptor, guard):
    guard()
    actual = extent(descriptor['path'])
    assert actual == descriptor, (descriptor['path'], actual, descriptor)
    guard()


def frozen(path, revision):
    rel = path.relative_to(ROOT).as_posix()
    blob = subprocess.check_output(['git', 'show', revision + ':' + rel], cwd=ROOT)
    assert blob.replace(b'\r\n', b'\n') == path.read_bytes().replace(b'\r\n', b'\n'), rel


def source_tensor_seals(source, expected, guard):
    """Hash raw safetensors ranges; never materialize a parameter or use a model loader."""
    path = Path(source) / 'model.safetensors'
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        assert 0 < length < (1 << 24)
        header = json.loads(stream.read(length))
        specs = {k: v for k, v in header.items() if k != '__metadata__'}
        assert set(specs) == set(expected) and len(specs) == 411
        assert all(v['dtype'] == 'BF16' for v in specs.values())
        base = 8 + length
        ranges = sorted(tuple(v['data_offsets']) for v in specs.values())
        assert ranges[0][0] == 0
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        assert ranges[-1][1] + base == path.stat().st_size
        for name, spec in specs.items():
            guard()
            goal = expected[name]
            assert spec['shape'] == goal['shape']
            start, end = spec['data_offsets']
            assert end - start == goal['bytes'] == 2 * product(spec['shape'])
            stream.seek(base + start)
            h = hashlib.sha256()
            left = end - start
            while left:
                data = stream.read(min(left, 1 << 20))
                assert data, name
                h.update(data)
                left -= len(data)
                guard()
            assert h.hexdigest() == goal['sha256'], name
    return len(specs)


def product(values):
    out = 1
    for v in values:
        out *= v
    return out


def score_rows(row, original, np, guard):
    outputs = row['output_ids']
    shape = [len(outputs), 65537]
    spec = row['logits']
    assert spec['shape'] == shape and spec['dtype'] == 'BF16'
    assert spec['bytes'] == 2 * product(shape)
    scores = np.memmap(spec['path'], dtype='<u2', mode='r', shape=tuple(shape))
    old = None
    if original is not None:
        old_spec = original['logits']
        assert old_spec['shape'] == shape and old_spec['bytes'] == spec['bytes']
        old = np.memmap(old_spec['path'], dtype='<u2', mode='r', shape=tuple(shape))
    for start in range(0, len(outputs), 16):
        guard()
        words = np.asarray(scores[start:start + 16])
        decoded = (words.astype('<u4') << 16).view('<f4')
        assert np.isfinite(decoded).all(), row['id']
        assert decoded.argmax(axis=1).tolist() == outputs[start:start + 16], row['id']
        if old is not None:
            assert np.array_equal(words, old[start:start + 16]), row['id']
    del scores, old


def packet(row, case, tokenizer, control, source_repo, source_revision):
    """Check stored provenance and canonical plain-chat encoding, separately from GPU witnesses."""
    for key in ('id', 'split', 'messages', 'input_ids', 'input_length', 'group_key', 'prefix_sha256'):
        assert row[key] == case[key], (row['id'], key)
    assert row['control'] is control and row['domain'] == case.get('source', case.get('domain'))
    prefix = json.dumps(row['messages'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(prefix).hexdigest() == row['prefix_sha256']
    assert dialogue_group(row['messages']) == row['group_key']
    assert row['input_length'] == len(row['input_ids']) <= 2048
    assert case['messages'][-1]['role'] == 'user'
    assert tokenizer.encode(row['messages']) == row['input_ids'], row['id']
    assert row['upstream'] == dict(repo=source_repo, revision=source_revision,
        source=case.get('source', case.get('domain')),
        parquet_path=case.get('parquet_path', case.get('upstream', {}).get('parquet_path')),
        row_index=case.get('row_index', case.get('upstream', {}).get('row_index')))
    assert row['context_origin'] == 'Public prior assistant turns retained; new reply generated by immutable donor.'
    assert isinstance(row['source_packet_seconds'], (float, int)) and 0 <= row['source_packet_seconds'] < LIMIT_SECONDS * 20
    out = row['output_ids']
    assert 1 <= len(out) <= 256 and all(type(x) is int and 0 <= x < 65537 for x in out)
    assert not any(x in (11, 228) for x in out[:-1])
    assert out[-1] in (11, 228) or len(out) == 256
    assert row['stop'] == ('EOS' if out[-1] in (11, 228) else 'length')
    assert row['student_input_ids'] == row['input_ids'] + out[:-1]
    assert row['positions'] == list(range(row['input_length'] - 1, row['input_length'] + len(out) - 1))
    assert max(row['positions']) < 2304
    assert tokenizer.decode(out) == row['output_text'], row['id']
    assert row['raw_logits_dtype'] == 'torch.float32'
    assert row['BF16_lossless'] is row['raw_argmax_equals_generated'] is row['finite'] is True


def audit(args, proc, start):
    import numpy as np

    proc.cpu_affinity(proc.cpu_affinity()[:6])
    def guard():
        assert time.monotonic() - start <= LIMIT_SECONDS, 'audit deadline'
        assert proc.memory_info().peak_wset <= LIMIT_OS_BYTES, 'audit OS cap'
        assert not proc.children(recursive=True), 'audit child process'

    for path in (Path(__file__), PROTOCOL):
        frozen(path, args.freeze)
    assert extent(REPORT)['sha256'] == EXPECTED_REPORT
    report = load(REPORT)
    assert report['schema'] == 'SOURCE_COVERAGE_RESULT_V1'
    assert report['binding']['sha256'] == EXPECTED_BINDING
    assert report['corpus']['sha256'] == EXPECTED_CORPUS and report['corpus']['bytes'] == 7994412
    binding = load(report['binding']['path'])
    assert binding['schema'] == 'SOURCE_COVERAGE_BINDING_V1'
    assert binding['limits'] == dict(seconds=21600, GPU_seconds=18000, reserve_seconds=180,
        OS_bytes=16 << 30, GPU_allocated_bytes=8 << 30, GPU_reserved_bytes=10 << 30,
        output_bytes=32 << 30, CPU_threads=6)
    assert binding['worker'] == str((Path(__file__).parent / 'chatbot_source_coverage.py').resolve())
    assert binding['new_generations'] == 432 and binding['FIT_cases'] == 360 and binding['DEV_cases'] == 72
    assert binding['maximum_labels'] == 434 * 256
    assert binding['maximum_packet_bytes'] == 434 * 256 * 65537 * 2
    assert report['plan'] == binding['plan']
    assert report['cases'] == 432 and report['FIT'] == 360 and report['DEV'] == 72 and report['labels'] == 76512
    assert report['RESERVED_queries'] == report['learner_forwards'] == report['optimizer_updates'] == 0
    assert report['decision'] == 'SOURCE_COVERAGE_CAPTURE_COMPLETE' and report['quality_admissible'] is False
    assert report['source411_exact'] and report['identities_versions_MuP_exact']
    assert report['inputs_before_after_exact'] and report['GPU_released_before_posthash']
    assert report['OS_peak'] <= binding['limits']['OS_bytes']
    assert report['GPU_allocated_peak'] <= binding['limits']['GPU_allocated_bytes']
    assert report['GPU_reserved_peak'] <= binding['limits']['GPU_reserved_bytes']
    assert report['GPU_phase_seconds'] <= binding['limits']['GPU_seconds']
    assert report['seconds'] <= binding['limits']['seconds']
    holder = load(REPORT.with_suffix('.terminal.json'))
    worker = load(REPORT.with_suffix('.worker_terminal.json'))
    assert holder['exit_code'] == holder['worker_exit_code'] == worker['exit_code'] == 0
    assert holder['error'] is worker['error'] is None
    assert holder['result'] == worker['result'] == extent(REPORT)
    assert holder['binding'] == report['binding']
    assert holder['worker_pid'] == worker['pid'] and abs(holder['worker_creation_time'] - worker['creation_time']) < .001
    assert holder['pid'] != worker['pid'] and holder['worker_argv'] == worker['command']
    assert holder['worker_OS_peak'] <= binding['limits']['OS_bytes']
    assert holder['worker_OS_peak'] == worker['OS_peak'] == report['OS_peak']
    assert abs(worker['seconds'] - report['seconds']) < 1
    assert holder['seconds'] >= worker['seconds']
    for pid, creation in ((holder['pid'], holder['creation_time']), (worker['pid'], worker['creation_time'])):
        try:
            live = __import__('psutil').Process(pid)
            assert abs(live.create_time() - creation) >= .001, 'producer still live'
        except __import__('psutil').NoSuchProcess:
            pass
    owned = load(REPORT.with_suffix('.owned_process.json'))
    for key in ('pid', 'creation_time', 'worker_pid', 'worker_creation_time', 'worker_argv', 'binding'):
        assert owned[key] == holder[key], key

    fixed = [extent(Path(__file__)), extent(PROTOCOL), extent(REPORT), report['binding'], extent(REPORT.with_suffix('.terminal.json')),
             extent(REPORT.with_suffix('.worker_terminal.json')),
             extent(REPORT.with_suffix('.owned_process.json'))]
    def seals():
        for descriptor in fixed + binding['inputs'] + report['outputs']:
            sealed(descriptor, guard)
    seals()
    family = Path(binding['family'])
    outputs = report['outputs']
    assert len(outputs) == 873 and len({e['path'] for e in outputs}) == 873
    assert {str(p.resolve()) for p in family.iterdir()} == {e['path'] for e in outputs}
    assert all(Path(e['path']).parent.resolve() == family.resolve() for e in outputs)
    assert sum(e['bytes'] for e in outputs) <= binding['limits']['output_bytes']
    assert report['corpus'] in outputs
    plan = load(binding['plan']['path'])
    assert plan['schema'] == 'ALL_REMAINING_SHORT_SOURCE_CASES_V1'
    assert plan['parent'] in binding['inputs'] and plan['used_ids'] == sorted(plan['used_ids'])
    adopted = load(plan['parent']['path'])
    assert len(adopted['records']) == 624
    previous = load(load(DOC / 'chatbot_broad_pilot_binding_repair1_20261009.json')['corpus'])['records']
    assert len(previous) == 48 and plan['used_ids'] == sorted(q['id'] for q in previous)
    assert len({q['group_key'] for q in previous}) == 48
    eligible = [q for q in adopted['records'] if q['split'] in ('FIT', 'DEV') and q['id'] not in set(plan['used_ids'])]
    expected = sorted((q for q in eligible if q['input_length'] <= 2048), key=lambda q: (q['source'], q['split'], q['id']))
    longest = min(expected, key=lambda q: (-q['input_length'], q['id']))
    expected = [longest] + [q for q in expected if q['id'] != longest['id']]
    deferred = [dict(id=q['id'], split=q['split'], source=q['source'], input_length=q['input_length'])
                for q in eligible if q['input_length'] > 2048]
    assert plan['cases'] == expected and plan['deferred'] == deferred
    assert len(expected) == 432 and len(deferred) == 40 and all(q['source'] == 'longalign' for q in deferred)
    assert len([q for q in adopted['records'] if q['split'] == 'RESERVED']) == 104
    assert not {q['group_key'] for q in expected} & {q['group_key'] for q in previous}
    assert len({q['id'] for q in expected}) == len({q['group_key'] for q in expected}) == 432
    assert Counter(q['split'] for q in expected) == Counter(FIT=360, DEV=72)
    assert len({q['source'] for q in expected}) == 12
    for source in {q['source'] for q in expected}:
        assert Counter(q['split'] for q in expected if q['source'] == source) == Counter(FIT=30, DEV=6)
    assert plan['FIT_cases'] == 360 and plan['DEV_cases'] == 72
    assert plan['max_new_tokens'] == 256 and plan['context_limit'] == 2304
    assert plan['RESERVED_queries'] == 0 and plan['external_final_answers_are_labels'] is False
    corpus = load(report['corpus']['path'])
    assert corpus['schema'] == 'SOURCE_COVERAGE_CORPUS_V1'
    assert corpus['source_binding'] == report['binding'] and corpus['source'] == binding['source']
    assert corpus['max_new_tokens'] == 256 and corpus['source_dtype'] == 'BF16'
    assert corpus['RESERVED_queries'] == corpus['learner_forwards'] == 0
    records = corpus['records']
    controls = report['controls']
    assert len(records) == 432 and len(controls) == len(binding['controls']) == 2
    assert [r['id'] for r in records] == [q['id'] for q in expected]
    assert [r['id'] for r in controls] == [q['id'] for q in binding['controls']]
    assert [len(q['output_ids']) for q in controls] == [18, 256]
    assert sum(len(q['output_ids']) for q in records) == 76512
    completion = load(family / 'numerical_completion.json')
    assert completion['records'] == records and completion['controls'] == controls
    assert completion['source411_exact'] and completion['identities_versions_exact']
    assert completion['MuP_exact'] and completion['GPU_released']
    assert completion['original_posthash_complete'] is False
    runtime = load(binding['runtime']['path'])
    assert runtime['source'] == binding['source']
    tokenizer = ChatTokenizer(binding['source'])
    for index, (row, case) in enumerate(zip(records, expected, strict=True), 1):
        guard()
        packet(row, case, tokenizer, False, plan['source_repo'], plan['source_revision'])
        assert load(family / (row['id'] + '.json')) == row
        score_rows(row, None, np, guard)
        if index % 24 == 0:
            print(json.dumps(dict(operation='packets_audited', cases=index,
                                  seconds=time.monotonic() - start)), flush=True)
    for row, case in zip(controls, binding['controls'], strict=True):
        guard()
        packet(row, case, tokenizer, True, plan['source_repo'], plan['source_revision'])
        assert load(family / ('control.' + row['id'] + '.json')) == row
        assert row['output_ids'] == case['output_ids']
        score_rows(row, case, np, guard)
    before = load(family / 'parameter_before.json')
    after = load(family / 'parameter_after.json')
    assert before == after and before['exact_source_BF16_values']
    assert before['canonical_coefficient_words'] == 1554863488 and len(before['tensors']) == 411
    expected_tensors = runtime['checkpoint_tensors']
    assert len(expected_tensors) == 411 and {q['name'] for q in before['tensors']} == set(expected_tensors)
    for q in before['tensors']:
        goal = expected_tensors[q['name']]
        assert q['runtime_dtype'] == 'torch.bfloat16'
        assert q['shape'] == goal['shape'] and q['canonical_BF16_sha256'] == goal['sha256']
    source_count = source_tensor_seals(binding['source'], expected_tensors, guard)
    assert load(family / 'source_method.json') == load(runtime['prior_source_method']['path'])
    seals()
    guard()
    write(OUT, dict(schema='SOURCE_COVERAGE_STORED_AUDIT_V1', report=extent(REPORT), binding=report['binding'],
        corpus=report['corpus'], cases=432, FIT=360, DEV=72, labels=76512, controls=2,
        outputs=len(outputs), input_pins=len(binding['inputs']), source_BF16_tensors=source_count,
        freeze=args.freeze, fixed_inputs=fixed, BF16_words=(76512 + 274) * 65537,
        new_label_positions=76512, control_label_positions=274,
        stored_checks_pass=True, decision='INDEPENDENT_STORED_NUMERIC_PASS', quality_admissible=False,
        producer_witnesses=['GPU model identity/version', 'MuP', 'raw F32 logits before BF16', 'GPU source execution',
                            'GPU release before posthash'],
        scope='Complete stored custody, source BF16 tensor ranges, canonical chat tokens, every finite BF16 score row and first-index argmax, and byte-exact cached controls. No learner transfer, semantic quality, native rate, DRAM, or useful-capacity admission.',
        seconds=time.monotonic()-start, OS_peak=proc.memory_info().peak_wset, GPU_calls=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True, help='Git revision containing this script and its protocol')
    args = parser.parse_args()
    import psutil
    process = psutil.Process()
    began = time.monotonic()
    receipt = dict(command=sys.orig_argv, pid=process.pid, creation_time=process.create_time(),
                   freeze=args.freeze, limits=dict(seconds=LIMIT_SECONDS, OS_bytes=LIMIT_OS_BYTES, CPU_logical=6))
    try:
        assert not OUT.exists() and not OUT.with_suffix('.owned_process.json').exists()
        assert not OUT.with_suffix('.terminal.json').exists()
        write(OUT.with_suffix('.owned_process.json'), receipt)
        audit(args, process, began)
        receipt.update(exit_code=0, error=None, result=extent(OUT))
    except BaseException as error:
        receipt.update(exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(seconds=time.monotonic()-began, OS_peak=process.memory_info().peak_wset)
        write(OUT.with_suffix('.terminal.json'), receipt)
