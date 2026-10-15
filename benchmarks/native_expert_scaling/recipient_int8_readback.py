"""Independently reduce retained baseline timing/identity receipts; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'))
import psutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(directory, output):
    started = time.monotonic()
    assert not output.exists()
    result_path = directory / 'result.json'
    result_before = sha(result_path)
    result = json.loads(result_path.read_text())
    binding_path = directory / 'binding.json'
    binding = json.loads(binding_path.read_text())
    assert sha(binding_path) == result['binding_sha256']
    for x in binding['inputs']:
        p = Path(x['path'])
        assert p.stat().st_size == x['bytes'] and sha(p) == x['sha256'], str(p)
    assert sha(directory / 'recipient.i8.bin') == result['artifact_sha256']
    assert sha(directory / 'recipient.i8.exe') == result['executable_sha256']
    assert result['quality_admitted'] is False and result['donor_transfer_admitted'] is False
    audit = json.loads((directory / 'audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['top1_matches'] == audit['ids_count'] == 32
    assert audit['worst_relative_l2'] <= 1e-3
    assert (audit['source_tensors'], audit['vectors_checked'], audit['matrices_checked'],
            audit['artifact_extents_checked']) == (290, 122, 169, 460)
    seals = {}
    commands = result['commands']
    assert [x['name'] for x in commands[:4]] == ['export', 'compile', 'probe', 'audit']
    expected = [f'generate_r{rep}_{c["name"]}' for rep in range(3) for c in binding['cases']]
    assert [x['name'] for x in commands[4:]] == expected
    for r in commands:
        assert r == json.loads((directory / (r['name'] + '.terminal.json')).read_text())
        assert r['exit_code'] == 0 and r['seconds'] <= 900 and r['peak_rss_observed'] <= 8 << 30
        for channel in ('stdout', 'stderr'):
            p = directory / (r['name'] + '.' + channel)
            assert sha(p) == r[channel + '_sha256']
            seals[p.name] = sha(p)
        owned = json.loads((directory / (r['name'] + '.owned.json')).read_text())
        assert owned['argv'] == r['argv']
        try:
            p = psutil.Process(owned['pid'])
            assert p.create_time() != owned['create_time'], r['name'] + ':still live'
        except psutil.NoSuchProcess:
            pass
    summaries = {}
    for c in binding['cases']:
        ids_runs = []
        rates = []
        for rep in range(3):
            name = f'generate_r{rep}_{c["name"]}'
            r = json.loads((directory / (name + '.stdout')).read_text())
            assert next(x for x in result['cases'] if x['case'] == c['name'] and
                        x['repetition'] == rep) == {'repetition': rep, 'case': c['name'], **r}
            import struct
            p = directory / (name + '.ids')
            raw = p.read_bytes()
            ids = list(struct.unpack('<' + 'i' * (len(raw) // 4), raw))
            assert ids == r['emitted_ids'] and len(ids) == r['emitted_count'] <= 64
            assert r['input_tokens'] == c['tokens']
            steps = r['decode_steps']
            assert len(steps) == len(ids)
            assert abs(sum(s['seconds'] for s in steps) - r['decode_s']) <= 1e-6
            count = 0
            for i, s in enumerate(steps):
                assert s['index'] == i and s['generated_id'] == ids[i]
                assert s['context'] == c['tokens'] + i and s['seconds'] >= 0
                active = ids[i] not in (151643, 151645)
                assert s['native_forward'] == active
                count += active
                if not active:
                    assert i == len(ids) - 1
            assert count == r['native_decode_forwards'] > 0
            assert r['eos_count'] == int(ids[-1] in (151643, 151645))
            assert [x['actual_mask'] for x in r['worker_affinity']] == [1, 4, 16, 64, 256, 1024]
            assert len({x['windows_thread_id'] for x in r['worker_affinity']}) == 6
            assert all(x['group'] == 0 for x in r['worker_affinity'])
            rate = count / r['decode_s']
            rates.append(rate)
            ids_runs.append(ids)
            seals[p.name] = sha(p)
        assert all(ids == ids_runs[0] for ids in ids_runs), c['name'] + ':repeat IDs differ'
        old = result['cost_summary'][c['name']]
        assert rates == old['rates']
        assert old['worst'] == min(rates) and old['median'] == sorted(rates)[1]
        assert old['spread'] == max(rates) / min(rates)
        summaries[c['name']] = {'rates': rates, 'worst_ms': 1000 / min(rates),
                                 'spread': max(rates) / min(rates), 'repeat_ids_exact': True}
    passed = all(s['worst_ms'] <= 18 and s['spread'] <= 1.10 for s in summaries.values())
    decision = 'BASELINE_COST_ENVELOPE_PASS' if passed else 'BASELINE_COST_ENVELOPE_FAIL'
    assert decision == result['status']
    assert sha(result_path) == result_before, 'result changed during readback'
    final = {'status': 'PASS_RETAINED_COST_READBACK', 'independent_cost_decision': decision,
             'case_summaries': summaries, 'result_sha256': sha(result_path),
             'binding_sha256': sha(binding_path), 'audit_sha256': sha(directory / 'audit.json'),
             'retained_outputs_sha256': seals, 'script_sha256': sha(Path(__file__)),
             'seconds': time.monotonic() - started, 'quality_admitted': False,
             'scope': 'Stored timing arithmetic, repeat IDs, seals and owned-child exit checks. Does not replay model arithmetic or measure DRAM.'}
    with output.open('x', encoding='utf8', newline='\n') as f:
        json.dump(final, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': final['status'], 'decision': decision,
                      'case_summaries': summaries, 'seconds': final['seconds']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    main(a.directory.resolve(), a.output.resolve())
