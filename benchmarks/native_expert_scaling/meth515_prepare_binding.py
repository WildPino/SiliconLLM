"""515 full unchanged payload and two predetermined first/last probe cases."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil
sys.path.insert(0, str(Path(__file__).resolve().parent))
import meth515_operations as O

start = time.monotonic()
proc = psutil.Process()
proc.cpu_affinity([10])
hashed = peak = 0
catalog = {}
assert not O.BIND.exists() and not O.BIND.with_suffix('.failure.json').exists()


def item(path, expected=None):
    global hashed, peak
    p = Path(path).resolve()
    s = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        while block := f.read(8 << 20):
            h.update(block)
            hashed += len(block)
            peak = max(peak, proc.memory_info().peak_wset)
            assert peak <= 512 << 20 and time.monotonic() - start <= 180
    assert (p.stat().st_size, p.stat().st_mtime_ns) == (s.st_size, s.st_mtime_ns)
    if expected:
        assert h.hexdigest() == expected, str(p)
    v = {'path': str(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'sha256': h.hexdigest()}
    catalog[str(p)] = v
    return v


try:
    previous = json.loads((O.DOC / 'meth514_r1_binding.json').read_bytes())
    runtime = [item(v['path'], v['sha256']) for v in previous['runtime_files'] if '/numpy' not in v['path'].replace('\\', '/').lower() and not v['path'].endswith('threadpoolctl.py')]
    preserved = previous['preserved']
    for rel, sha in preserved.items():
        item(O.ROOT / rel, sha)
    records = {}
    for name in ['meth363_switch_all_a16_multi_span_quality_result.json', 'meth362_switch_multi_span_manifest.json', 'meth365_switch_encoder_batches_result.json',
                 'meth366_switch_encoder_batches_cost_result.json', 'meth374_switch_physical_workers_contract_result.json', 'meth458_switch_matched_whole_cost_result.json', 'meth489_primary_phase_algebra.json']:
        records[name] = item(O.DOC / name)
    raw = json.loads(Path(records['meth363_switch_all_a16_multi_span_quality_result.json']['path']).read_bytes())
    physical = json.loads(Path(records['meth374_switch_physical_workers_contract_result.json']['path']).read_bytes())
    cohort = json.loads(Path(records['meth362_switch_multi_span_manifest.json']['path']).read_bytes())
    assert all(raw['gates'].values()) and all(physical['gates'].values())
    artifact = physical['artifact']
    payload = item(artifact['payload'], artifact['sha256'])
    spec = item(artifact['manifest'], artifact['manifest_sha256'])
    compiler = item(physical['compile']['argv'][0], physical['compile']['compiler_sha256'])
    baseline = item(physical['compile']['argv'][-1], physical['compile']['binary_sha256'])
    libomp = item(Path(baseline['path']).parent / 'libomp.dll', physical['compile']['runtime_sha256'])
    cases = []
    for book, index in [(0, 0), (23, 0)]:
        c = raw['books'][book]['cases'][index]
        inputs = cohort['items'][book]['cases'][index]
        prefix = O.ROOT / f'results/native_expert_scaling/meth363_switch_all_a16_multi_span_quality/book{book}.case{index}'
        teacher = item(str(prefix) + '.0.bin', c['native_output_sha256'])
        generation = item(str(prefix) + '.generation.0.bin', c['generation']['native_generation_sha256'])
        cases.append({'book': book, 'index': index, 'source_id': raw['books'][book]['source_id'],
                      'source_ids': inputs['source_ids'], 'decoder_ids': inputs['decoder_ids'],
                      'expected_teacher': teacher, 'expected_generation': generation, 'expected_ids': c['generation']['native']['generated_ids']})
    expected_output = sum(c['expected_teacher']['bytes'] + 8 * c['expected_generation']['bytes'] for c in cases)
    assert expected_output == 56120440 and expected_output < (64 << 20) - (1 << 20)
    scientific = []
    used = sorted((O.ROOT / 'benchmarks/native_expert_scaling').glob('meth515*')) + [O.DOC / 'METH_515_ENCODER_REUSE_PROBE_PROTOCOL_20261007.md']
    used += [O.ROOT / 'benchmarks/native_expert_scaling' / n for n in ['meth365_switch_encoder_batches.c', 'meth374_switch_physical_workers.c', 'meth374_switch_physical_workers_entry.c', 'meth374_switch_physical_workers_cost_entry.c', 'meth374_switch_thread_binding.h']]
    for p in used:
        assert p.read_bytes().replace(b'\r\n', b'\n') == subprocess.check_output(['git', 'show', 'HEAD:' + p.relative_to(O.ROOT).as_posix()], cwd=O.ROOT).replace(b'\r\n', b'\n')
        scientific.append(item(p))
    v = {'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=O.ROOT, text=True).strip(), 'python': sys.version,
         'executable': str(Path(sys.executable).resolve()), 'preserved': preserved, 'catalog': list(catalog.values()), 'scientific': scientific,
         'payload': payload, 'manifest': spec, 'compiler': compiler, 'baseline': baseline, 'libomp': libomp,
         'compiler_flags': ['-O3', '-std=c11', '-march=x86-64-v3', '-fno-fast-math', '-ffp-contract=off', '-fopenmp'],
         'native_environment': physical['runtime_environment'], 'cases': cases, 'expected_native_output_bytes': expected_output,
         'main_limits': [300, 2 << 30, 64 << 20], 'audit_limits': [300, 2 << 30, 64 << 20],
         'encoder_ratio_necessary_on_retained489': 1 - (13.620031066666666 - 9.8) / 6.898050366666666,
         'resource': {'seconds': time.monotonic() - start, 'OS_peak_bytes': peak, 'bytes_hashed': hashed}}
    O.write(O.BIND, v)
    r = item(O.BIND)
    print(json.dumps({'binding': r, 'resource': v['resource'], 'expected_output_bytes': expected_output}), flush=True)
except BaseException:
    O.write(O.BIND.with_suffix('.failure.json'), {'traceback': traceback.format_exc(), 'seconds': time.monotonic() - start, 'bytes_hashed': hashed, 'OS_peak_bytes': peak})
    raise
