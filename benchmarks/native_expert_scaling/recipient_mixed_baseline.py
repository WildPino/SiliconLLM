"""Frozen complete W4/W8 recipient cost comparison; no transfer admission."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path.insert(0, str(SITE))
import psutil

spec = importlib.util.spec_from_file_location('retained_baseline', HERE / 'recipient_int8_baseline.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sha, write = base.sha, base.write
DOC = ROOT / 'docs/research/PIPELINE_REASSESSMENT_20261015'
PREVIOUS = ROOT / 'results/native_expert_scaling/recipient_int8_baseline_repair2_20261015/binding.json'


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    old = json.loads(PREVIOUS.read_text())
    base.verify(old)
    files = {Path(x['path']) for x in old['inputs']}
    files.update((Path(__file__).resolve(), HERE / 'recipient_qwen_entry.c',
                  HERE / 'recipient_mixed_export.py', HERE / 'recipient_mixed_audit.py',
                  DOC / 'RECIPIENT_MIXED_BASELINE_PROTOCOL_20261015.md', PREVIOUS))
    cases = old['cases']
    for c in cases:
        ids = Path(c['ids']).read_bytes()
        assert len(ids) == c['tokens'] * 4 and sha(c['ids']) == c['sha256']
        request = directory / (c['name'] + '.request')
        request.write_bytes(b'QWCR0001' + struct.pack('<II', c['tokens'], c['new_tokens']) + ids)
        c['request'] = str(request)
        files.add(request)
    probe = directory / 'probe.request'
    ids = Path(old['parity_ids']).read_bytes()
    assert len(ids) == 32 * 4
    probe.write_bytes(b'QWCP0001' + struct.pack('<II4I', 32, 4, 0, 7, 15, 31) + ids)
    files.add(probe)
    records = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(files)]
    binding = {'schema': 'RECIPIENT_MIXED_BASELINE_V1',
               'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'previous_binding_sha256': sha(PREVIOUS), 'inputs': records,
               'compiler': old['compiler'], 'openmp_dll': old['openmp_dll'],
               'python': sys.executable, 'source_directory': old['source_directory'],
               'source_sha256': old['source_sha256'], 'cases': cases,
               'probe_request': str(probe), 'parity_ids': old['parity_ids'],
               'limits': {'job_seconds': 900, 'total_seconds': 2400,
                          'host_bytes': 8 << 30, 'output_bytes': 2 << 30},
               'gates': {'worst_forward_seconds': .018, 'maximum_case_rate_spread': 1.10,
                         'probe_top1_exact': True, 'probe_relative_l2': .001},
               'quality_admitted': False, 'donor_transfer_admitted': False}
    write(directory / 'binding.json', binding)
    print(json.dumps({'prepared': str(directory), 'inputs': len(records),
                      'binding_sha256': sha(directory / 'binding.json')}), flush=True)


def parse_generation(raw, log, context, maxnew):
    assert raw[:8] == b'QWCO0001' and len(raw) >= 24
    count, stop, elapsed = struct.unpack_from('<IId', raw, 8)
    assert 1 <= count <= maxnew and stop in (0, 1) and len(raw) == 24 + 4 * count
    ids = list(struct.unpack_from('<' + 'i' * count, raw, 24))
    assert all(0 <= x < 151936 for x in ids)
    records = [json.loads(x) for x in log.splitlines() if x.strip()]
    assert [x['phase'] for x in records] == ['binding_start', 'prefill'] + ['decode'] * count + ['binding_end']
    for x in (records[0], records[-1]):
        assert x['worker_masks'] == [1, 4, 16, 64, 256, 1024]
    prefill = records[1]
    assert prefill['context'] == context and prefill['threads'] == 6 and prefill['seconds'] > 0
    steps = records[2:-1]
    for i, x in enumerate(steps):
        assert (x['index'], x['generated_id'], x['context'], x['threads']) == (i, ids[i], context+i, 6)
        assert x['native_forward'] == (ids[i] not in (151645, 151643)) and x['seconds'] >= 0
    assert all(x not in (151645, 151643) for x in ids[:-1])
    assert stop == int(ids[-1] in (151645, 151643)) and (stop or count == maxnew)
    charged = sum(x['native_forward'] for x in steps)
    decode = sum(x['seconds'] for x in steps)
    assert charged > 0 and decode > 0 and elapsed >= prefill['seconds']
    return {'emitted_ids': ids, 'native_decode_forwards': charged, 'decode_s': decode,
            'prefill_s': prefill['seconds'], 'native_elapsed_s': elapsed,
            'stop': stop, 'decode_steps': steps, 'binding': [records[0], records[-1]]}


def run(directory):
    binding = json.loads((directory / 'binding.json').read_text())
    assert not (directory / 'started.json').exists()
    base.verify(binding)
    write(directory / 'started.json', {'time': time.time(), 'binding_sha256': sha(directory / 'binding.json')})
    started = time.monotonic()
    commands = []
    e = base.env()
    e['PATH'] = str(Path(binding['compiler']).parent) + os.pathsep + e['PATH']

    def job(name, argv, request=None):
        assert time.monotonic() - started < binding['limits']['total_seconds']
        t = time.monotonic()
        out, err = directory / f'{name}.stdout', directory / f'{name}.stderr'
        with out.open('xb') as stdout, err.open('xb') as stderr:
            stdin = Path(request).open('rb') if request else subprocess.DEVNULL
            try:
                child = subprocess.Popen(list(map(str, argv)), stdin=stdin, stdout=stdout, stderr=stderr,
                                         cwd=ROOT, env=e, creationflags=subprocess.CREATE_NO_WINDOW)
            finally:
                if request:
                    stdin.close()
            proc = psutil.Process(child.pid)
            created = proc.create_time()
            write(directory / f'{name}.owned.json', {'pid': child.pid, 'create_time': created,
                  'argv': list(map(str, argv)), 'request': str(request) if request else None})
            peak = 0
            try:
                while child.poll() is None:
                    try:
                        peak = max(peak, proc.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    assert peak <= binding['limits']['host_bytes'], f'{name}:memory'
                    assert time.monotonic() - t <= binding['limits']['job_seconds'], f'{name}:time'
                    assert time.monotonic() - started <= binding['limits']['total_seconds'], 'family:time'
                    time.sleep(.1)
            except BaseException:
                if child.poll() is None and proc.create_time() == created:
                    child.kill()
                    child.wait()
                raise
        receipt = {'name': name, 'argv': list(map(str, argv)), 'exit_code': child.returncode,
                   'seconds': time.monotonic()-t, 'peak_rss_observed': peak,
                   'stdout_sha256': sha(out), 'stderr_sha256': sha(err),
                   'request_sha256': sha(request) if request else None}
        write(directory / f'{name}.terminal.json', receipt)
        commands.append(receipt)
        print(json.dumps({'stage': name, 'exit_code': child.returncode, 'seconds': receipt['seconds']}), flush=True)
        assert child.returncode == 0, f'{name}:exit{child.returncode}'
        return out, err

    py = [binding['python'], '-I', '-S', '-B', '-X', 'utf8']
    artifact, header, manifest = (directory / x for x in ('recipient.mixed.bin', 'recipient_catalog.h', 'manifest.json'))
    exe = directory / 'recipient.mixed.exe'
    try:
        job('export', py + [HERE / 'recipient_mixed_export.py', '--source', Path(binding['source_directory']) / 'model.safetensors',
             '--out', artifact, '--catalog', header, '--manifest', manifest, '--threads', '6'])
        artifact_sha = sha(artifact)
        job('compile', [binding['compiler'], '-O3', '-std=c11', '-march=x86-64-v3', '-ffp-contract=off',
             '-fopenmp', '-Wl,--no-insert-timestamp', '-DSILICON_QWEN_RECIPIENT_ENTRY',
             '-DRECIPIENT_CATALOG="' + str(header).replace('\\', '/') + '"',
             ROOT / 'benchmarks/phase60/engine.c', '-o', exe, '-lbcrypt', '-lm'])
        logits, _ = job('probe', [exe, artifact, 'probe', '2176', '6'], binding['probe_request'])
        job('audit', py + [HERE / 'recipient_mixed_audit.py', '--source', Path(binding['source_directory']) / 'model.safetensors',
            '--archive', artifact, '--archive-sha', artifact_sha, '--catalog', header,
            '--manifest', manifest, '--ids', binding['parity_ids'],
            '--request', binding['probe_request'], '--probe', logits, '--out', directory / 'audit.json'])
        audit = json.loads((directory / 'audit.json').read_text())
        assert audit['status'] == 'PASS', audit
        results = []
        for repetition in range(3):
            for c in binding['cases']:
                out, err = job(f'generate_r{repetition}_{c["name"]}', [exe, artifact, 'stream', '2176', '6'], c['request'])
                r = parse_generation(out.read_bytes(), err.read_text(), c['tokens'], c['new_tokens'])
                results.append({'case': c['name'], 'repetition': repetition, **r})
        base.verify(binding)
        assert sha(artifact) == artifact_sha
        by_case = {}
        for c in binding['cases']:
            rows = [r for r in results if r['case'] == c['name']]
            assert all(x['emitted_ids'] == rows[0]['emitted_ids'] for x in rows)
            rates = [x['native_decode_forwards']/x['decode_s'] for x in rows]
            by_case[c['name']] = {'rates': rates, 'median': statistics.median(rates),
                 'worst': min(rates), 'worst_ms': 1000/min(rates), 'spread': max(rates)/min(rates)}
        passed = all(x['worst_ms'] <= 18 and x['spread'] <= 1.10 for x in by_case.values())
        output_bytes = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        assert output_bytes <= binding['limits']['output_bytes']
        result = {'status': 'MIXED_COST_ENVELOPE_PASS' if passed else 'MIXED_COST_ENVELOPE_FAIL',
                  'binding_sha256': sha(directory / 'binding.json'), 'artifact_sha256': artifact_sha,
                  'artifact_bytes': artifact.stat().st_size, 'executable_sha256': sha(exe),
                  'catalog_sha256': sha(header), 'manifest_sha256': sha(manifest),
                  'commands': commands, 'cases': results, 'cost_summary': by_case,
                  'seconds': time.monotonic()-started, 'output_bytes': output_bytes,
                  'active_matrix_payload_bytes_per_forward': 357484032,
                  'physical_dram_measured': False, 'quality_admitted': False,
                  'donor_transfer_admitted': False, 'goal_complete': False}
        write(directory / 'result.json', result)
        print(json.dumps({'status': result['status'], 'cost_summary': by_case}), flush=True)
    except BaseException as error:
        write(directory / 'failure.json', {'error': repr(error), 'commands': commands,
              'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run)(args.directory.resolve())
