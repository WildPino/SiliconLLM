"""Post-hoc arithmetic on admitted outputs; no fit, source replay or native call."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import hashlib
import json
from pathlib import Path
import struct
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = DOC / 'meth490_admitted_output_algebra.json'
FAIL = OUT.with_suffix('.failure.json')
START = time.monotonic()
assert not OUT.exists() and not FAIL.exists()
READS = []

def read(path, expected):
    path = Path(path)
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    assert sha == expected, str(path)
    READS.append({'path': str(path), 'bytes': len(payload), 'sha256': sha})
    assert time.monotonic() - START < 30
    return payload

def save(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').replace('\n', '\r\n').encode()
    assert len(payload) < 128 << 10
    with path.open('xb') as stream:
        stream.write(payload)

try:
    admission = json.loads(read(DOC / 'ADMISSION_490_R1_20261006.json',
        '693541acf5e6b075c9ed2ec38853a595e49c91a4677c840d176cfa56e8f53745'))
    raw = json.loads(read(admission['raw']['path'], admission['raw']['sha256']))
    ret = json.loads(read(admission['retention']['path'], admission['retention']['sha256']))
    assert admission['main_completed'] and all(admission['gates'].values()) and all(ret['gates'].values())
    binding = json.loads(read(DOC / 'meth490_r1_binding.json', raw['binding_sha256']))
    known = {v['path']: v for v in binding['catalog'] + raw['output_inventory']}
    import numpy as np
    import psutil
    import threadpoolctl
    threadpoolctl.threadpool_limits(limits=1)
    assert np.__version__ == binding['runtime']['packages']['numpy']['version']
    proc = psutil.Process()
    tiny = np.array([1], '<u8').view('<f8')
    assert (tiny + tiny).view('<u8')[0] == 2 and float(np.float32(1 + 2**-24)) == 1
    def guard():
        assert time.monotonic() - START < 30
        assert proc.memory_info().peak_wset < 256 << 20
    def binary(name, magic, width, dtype):
        path = str(ROOT / 'results/native_expert_scaling/meth490_r1_root_mass' / name)
        payload = read(path, known[path]['sha256'])
        assert payload[:24] == struct.pack('<8sIIQ', magic, width, 0, 238872)
        data = np.frombuffer(payload, dtype=dtype, offset=24)
        assert len(data) == 238872
        return data
    diag = binary('diagnostics.bin', b'M490DIA1', 48,
        np.dtype([('uid', '<u4'), ('source', '<f8', (2,)), ('right', '<u4'),
                  ('error', '<f8'), ('relative', '<f8'), ('ideal', '<f8')]))
    pred = binary('predictions.bin', b'M490PRD1', 16,
        np.dtype([('uid', '<u4'), ('z', '<f4'), ('pair', '<f4', (2,))]))
    assert np.array_equal(diag['uid'], pred['uid'])
    right = diag['right'].astype(bool)
    sign = np.where(right, 1., -1.)
    source = np.where(right, diag['source'][:, 1], diag['source'][:, 0])
    actual = np.where(right, pred['pair'][:, 1], pred['pair'][:, 0]).astype('<f8')
    assert np.all(source > 0) and np.all(actual > 0)
    source_log = np.log(source)
    ideal_log = -np.logaddexp(0, -sign * diag['ideal'])
    exported_log = -np.logaddexp(0, -sign * pred['z'].astype('<f8'))
    actual_log = np.log(actual)
    ideal_error = np.abs(ideal_log - source_log)
    native_error = np.abs(actual_log - source_log)
    export_drift = np.abs(exported_log - ideal_log)
    probability_drift = np.abs(actual_log - exported_log)
    all_drift = np.abs(actual_log - ideal_log)
    logit_drift = np.abs(pred['z'].astype('<f8') - diag['ideal'])
    tolerance = 2e-12
    assert np.allclose(native_error, diag['error'], rtol=1e-12, atol=tolerance)
    assert np.all(export_drift <= logit_drift + tolerance)
    assert np.all(np.abs(native_error - ideal_error) <= all_drift + tolerance)
    eta = raw['log_error_budget']
    assert np.count_nonzero(native_error > eta) == raw['overall'][0]['over_log_budget']
    def stats(values):
        q = np.quantile(values, [.5, .9, .95, .99])
        return {'count': len(values), 'mean': float(values.mean()),
                **dict(zip(('p50', 'p90', 'p95', 'p99'), map(float, q))), 'max': float(values.max())}
    banks = []
    development = np.zeros(238872, bool)
    for model in raw['models']:
        path = model['history_path']
        import io
        with np.load(io.BytesIO(read(path, known[path]['sha256'])), allow_pickle=False) as archive:
            uid = archive['training_uid'].copy()
            omega = archive['omega'].copy()
            losses = np.r_[archive['loss'], archive['final_loss']]
            gradient = archive['gradient'][-1].copy()
            last_update = archive['after'][-1] - archive['before'][-1]
        development[uid] = True
        y = diag['source'][uid, 1]
        z = diag['ideal'][uid]
        entropy = np.zeros(len(uid))
        positive = y > 0
        entropy[positive] -= y[positive] * np.log(y[positive])
        complement = 1 - y
        positive = complement > 0
        entropy[positive] -= complement[positive] * np.log(complement[positive])
        bce = y * np.logaddexp(0, -z) + complement * np.logaddexp(0, z)
        final_bce = float(omega @ bce)
        target_entropy = float(omega @ entropy)
        excess_kl = float(omega @ (bce - entropy))
        assert abs(final_bce - model['final_loss']) < 1e-10
        assert abs(final_bce - target_entropy - excess_kl) < 1e-12 and excess_kl >= -1e-12
        banks.append({'bank': model['bank'], 'development_inputs': len(uid),
            'initial_BCE': float(losses[0]), 'final_BCE': final_bce,
            'target_Bernoulli_entropy': target_entropy, 'excess_Bernoulli_KL_nats': excess_kl,
            'observed_loss_increases': int(np.count_nonzero(np.diff(losses) > 0)),
            'pre_update32_gradient_Linf': float(np.max(np.abs(gradient))),
            'update32_L2': float(np.linalg.norm(last_update)),
            'development_native_over_eta': int(np.count_nonzero(native_error[uid] > eta)),
            'development_ideal_over_eta': int(np.count_nonzero(ideal_error[uid] > eta)),
            'development_native_log_error': stats(native_error[uid])})
        guard()
    assert development.sum() == 159414
    masks = {'ALL': np.ones(238872, bool), 'development': development,
             'consumed_validation': ~development}
    views = {}
    for role, mask in masks.items():
        views[role] = {'native_log_error': stats(native_error[mask]),
            'ideal_log_error': stats(ideal_error[mask]),
            'native_over_eta': int(np.count_nonzero(native_error[mask] > eta)),
            'ideal_over_eta': int(np.count_nonzero(ideal_error[mask] > eta)),
            'ideal_exceeds_eta_plus_numerical_drift': int(np.count_nonzero(
                ideal_error[mask] > eta + all_drift[mask] + tolerance)),
            'native_over_undivided_log1p01': int(np.count_nonzero(native_error[mask] > np.log1p(.01)))}
    guard()
    result = {'classification': 'POST_HOC_DESCRIPTIVE_ARITHMETIC_ON_INDEPENDENTLY_ADMITTED_OUTPUTS',
        'no_new_fit_source_native_or_quality_run': True,
        'source_helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': READS, 'eta': eta, 'comparison_tolerance': tolerance, 'views': views, 'banks': banks,
        'ALL_logit_export_difference': stats(logit_drift),
        'ALL_selected_log_probability_export_drift': stats(export_drift),
        'ALL_selected_log_probability_F32_rounding_drift': stats(probability_drift),
        'ALL_selected_log_probability_total_numerical_drift': stats(all_drift),
        'weighted_bank_mean_excess_Bernoulli_KL_nats': float(np.mean([v['excess_Bernoulli_KL_nats'] for v in banks])),
        'scope': 'Describes this fixed trajectory only; no affine-class impossibility, optimizer optimum, whole quality or physical timing claim.',
        'resource': {'wall_seconds': time.monotonic() - START, 'peak_working_set_bytes': proc.memory_info().peak_wset,
                     'wall_limit_seconds': 30, 'host_limit_bytes': 256 << 20}}
    save(OUT, result)
    print(json.dumps({k: result[k] for k in ('views', 'ALL_selected_log_probability_total_numerical_drift',
        'weighted_bank_mean_excess_Bernoulli_KL_nats', 'resource')}))
except BaseException as exc:
    save(FAIL, {'error': str(exc), 'traceback': traceback.format_exc(), 'reads': READS,
                'wall_seconds': time.monotonic() - START})
    raise
