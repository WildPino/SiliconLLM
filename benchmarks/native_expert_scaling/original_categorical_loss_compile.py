"""Compile fixed native-head categorical supervision; no model or optimizer."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from original_packed_capacity import SITE, extent, sha, write
from original_falcon_whole_recovery import memory_reader
sys.path.insert(0, str(SITE))
V, D, N = 65537, 256, 8808


def bind(a):
    import numpy as np
    import psutil
    assert not a.binding.exists()
    names = ['categorical_native_assessment_binding_20261010.json',
             'categorical_native_assessment_result_20261010.json',
             'categorical_native_assessment_stored_adjudication_20261010.json',
             'paired_head_image_binding_20261010.json',
             'paired_head_image_result_20261010.json',
             'paired_head_image_stored_adjudication_20261010.json']
    paths = [DOC / n for n in names]
    nb, nr, na, pb, pr, pa = [json.loads(p.read_bytes()) for p in paths]
    assert na['result'] == extent(paths[1]) and pa['result'] == extent(paths[4])
    assert na['input_hashes'] == 200 and na['output_hashes'] == 485 and na['independent_F64_labels'] == N
    assert na['proxy_labels_verified'] == N and na['all_integer_witnesses_exact'] and pa['complete_input_output_hashes']
    assert pa['all72_final_upper_and_lower_points_verified'] and all(pa['numeric_flags'].values())
    receipts = []
    for index in (1, 2, 4, 5):
        p = paths[index].with_suffix('.terminal.json')
        t = json.loads(p.read_bytes())
        assert t['exit_code'] == 0
        if index == 1:
            assert t['result_sha256'] == sha(paths[index]) and len(t['output_files']) == na['output_hashes']
            assert t['resource_gates'] is True and t['binding_sha256'] == sha(paths[0])
        else:
            assert t['error'] is None and t['inputs_before_after_exact'] and t['result'] == extent(paths[index])
        receipts.append(p)
    assert all(r['finite'] and r['BF16_lossless'] and r['raw_argmax_equals_generated'] for r in nb['records'])
    records = [dict(id=r['id'], split=r['split'], domain=r['domain'],
                    labels=len(r['positions']), positions=r['positions'], teacher=r['logits'], source_output_ids=r['output_ids'])
               for r in nb['records']]
    assert len(records) == 48 and sum(r['labels'] for r in records) == N
    assert [sum(r['labels'] for r in records if r['split'] == s) for s in ('FIT', 'DEV')] == [4422, 4386]
    lookup = {r['id']: r for r in records}
    for s in pb['selected']:
        r = lookup[s['id']]
        assert r['split'] == 'FIT' and r['teacher']['sha256'] == s['teacher']['sha256']
        assert r['teacher']['bytes'] == s['teacher']['bytes']
        assert [r['positions'][i] for i in s['label_indices']] == s['positions']
    assert len(pr['labels']) == 72 and len({r['id'] for r in pr['labels']}) == 24
    head, final_norm = pb['artifacts']['head'], pb['artifacts']['final_norm']
    assert head['shape'] == [V, D] and head['sha256'] == '768aaf9e17873707624e4489a8f14f345afb2722e256de871113b7d6e64baa66'
    assert np.array_equal(np.fromfile(final_norm['path'], dtype='<f4'), np.ones(D, '<f4'))
    files = [Path(__file__), a.protocol, *paths, *receipts, Path(head['path']),
             Path(final_norm['path']), Path(pr['points']['native_normalized']['path']),
             Path(pr['points']['teacher']['path'])]
    files += [Path(r['teacher']['path']) for r in records]
    modules = [m for m in list(sys.modules.values()) if getattr(m, '__file__', None)]
    for m in modules:
        for key in ('__file__', '__cached__'):
            p = getattr(m, key, None)
            if p and Path(p).is_file():
                files.append(Path(p))
    files += [Path(sys.executable), Path(sys.executable).parent / 'python312.dll']
    files += sorted((SITE / 'numpy.libs').glob('*.dll')) + sorted((SITE / 'psutil').glob('*.pyd'))
    assert (np.__version__, psutil.__version__) == ('2.4.6', '7.2.2')
    inputs = [extent(p) for p in dict.fromkeys(p.resolve() for p in files)]
    by = {r['path']: r for r in inputs}
    for r in records:
        assert by[str(Path(r['teacher']['path']).resolve())] == {k: r['teacher'][k] for k in ('path', 'bytes', 'sha256')}
        assert r['teacher']['shape'] == [r['labels'], V]
        assert len(r['source_output_ids']) == r['labels']
    write(a.binding, dict(schema='ORIGINAL_CATEGORICAL_LOSS_BINDING_V1', inputs=inputs,
          records=records, head=head, final_norm=final_norm, selected=pr['labels'],
          native_features=pr['points']['native_normalized'], selected_teacher=pr['points']['teacher'],
          FIT_ids=[r['id'] for r in records if r['split'] == 'FIT'],
          DEV_ids=[r['id'] for r in records if r['split'] == 'DEV'], chunk=16, vocab_block=4096,
          numeric=dict(moment_absolute=1e-8, entropy_absolute=1e-9, loss_absolute=1e-8,
                       gradient_absolute=1e-8, probability_mass_absolute=1e-12, scalar_absolute=1e-8),
          capture_limits=dict(seconds=300, reserve_seconds=30, OS_bytes=2 << 30,
                              output_bytes=128 << 20, log_bytes=4 << 20),
          audit_limits=dict(seconds=300, OS_bytes=2 << 30, output_bytes=1 << 20, log_bytes=4 << 20),
          runtime=dict(NumPy=np.__version__, psutil=psutil.__version__, Python=sys.version,
                       imported_module_file_scope=True, junction_paths_resolved=True),
          scope='Fixed actual F32 head interpreted in real arithmetic. Source moments/negative entropy, all8808 labels and72 stored loss/state-gradient witnesses. No source/model/native/history/optimizer/SVD/GPU/RESERVED/T4 call. Not a quality/speed admission.'))
    print(json.dumps(dict(binding_sha256=sha(a.binding), inputs=len(inputs),
                         bytes=sum(r['bytes'] for r in inputs))), flush=True)


def save(directory, name, x, dtype='<f8'):
    import numpy as np
    x = np.ascontiguousarray(x, dtype=dtype)
    p = directory / (name + ('.f64' if dtype == '<f8' else '.i32'))
    with p.open('xb') as f:
        f.write(x.tobytes()); f.flush(); os.fsync(f.fileno())
    return dict(**extent(p), shape=list(x.shape), dtype=dtype)


def read(item):
    import numpy as np
    x = np.fromfile(item['path'], dtype=item['dtype']).reshape(item['shape'])
    assert np.isfinite(x).all()
    return x


def teacher(rec):
    import numpy as np
    return np.memmap(rec['teacher']['path'], mode='r', dtype='<u2', shape=(rec['labels'], V))


def decode(bits):
    import numpy as np
    z = (bits.astype('<u4') << 16).view('<f4').astype('<f8')
    assert np.isfinite(z).all()
    return z


def probabilities(z, independent):
    import numpy as np
    s = z - z.max(1, keepdims=True)
    if independent:
        logZ = np.logaddexp.reduce(s, axis=1)
    else:
        logZ = np.log(np.exp(s).sum(1))
    lp = s - logZ[:, None]
    return np.exp(lp), lp


def contract(q, H, b, independent):
    import numpy as np
    if not independent:
        return q @ H
    out = np.zeros((len(q), D), '<f8')
    for first in range(0, V, b['vocab_block']):
        last = min(first + b['vocab_block'], V)
        out += q[:, first:last] @ H[first:last]
    return out


def witnesses(b, H, rows, independent, guard):
    import numpy as np
    f = read(b['native_features']).astype('<f8')
    old_q_logits = read(b['selected_teacher'])
    by = {r['id']: r for r in rows}
    src = {r['id']: r for r in b['records']}
    moments, constants, raw = [], [], []
    for i, label in enumerate(b['selected']):
        r = by[label['id']]; j = label['label_index']
        assert src[label['id']]['split'] == 'FIT' and label['position'] == src[label['id']]['positions'][j]
        bits = teacher(src[label['id']]); z = decode(bits[j:j+1])[0]
        assert np.array_equal(z, old_q_logits[i])
        raw.append(z); moments.append(read(r['moments'])[j]); constants.append(read(r['negative_entropy'])[j])
    raw = np.asarray(raw); moments = np.asarray(moments); constants = np.asarray(constants)
    q, lq = probabilities(raw, independent); z = f @ H.T; p, lp = probabilities(z, independent)
    direct_loss = np.sum(q * (lq - lp), axis=1)
    maxima = z.max(1); shifted = z - maxima[:, None]
    logZ = np.logaddexp.reduce(shifted, axis=1) if independent else np.log(np.exp(shifted).sum(1))
    compiled_loss = constants + maxima + logZ - np.sum(moments * f, axis=1)
    direct_gradient = contract(p - q, H, b, independent)
    compiled_gradient = contract(p, H, b, independent) - moments
    scalar_loss = max(abs(float(compiled_loss[i]) -
                          (constants[i] + maxima[i] + logZ[i] - math.fsum(float(moments[i, j] * f[i, j]) for j in range(D))))
                      for i in range(72))
    values = dict(loss_absolute_delta=float(np.max(abs(direct_loss - compiled_loss))),
                  gradient_absolute_delta=float(np.max(abs(direct_gradient - compiled_gradient))),
                  scalar_loss_absolute_delta=float(scalar_loss),
                  target_probability_mass_error=float(np.max(abs(p.sum(1) - 1))))
    scalar_moment = scalar_gradient = 0.
    if independent:
        for i in range(0, 72, 3):
            j = ((i // 3) * 17) % 255
            value = math.fsum(float(q[i, v] * H[v, j]) for v in range(V))
            scalar_moment = max(scalar_moment, abs(value - moments[i, j])); guard()
        for i in range(0, 72, 6):
            j = (i * 13 + 7) % 255
            value = math.fsum(float((p[i, v] - q[i, v]) * H[v, j]) for v in range(V))
            scalar_gradient = max(scalar_gradient, abs(value - direct_gradient[i, j])); guard()
    values.update(scalar_moment_absolute_delta=scalar_moment, scalar_gradient_absolute_delta=scalar_gradient)
    guard()
    return dict(direct_loss=direct_loss, compiled_loss=compiled_loss,
                direct_gradient=direct_gradient, compiled_gradient=compiled_gradient), values


def flags(values, b):
    g = b['numeric']
    return dict(loss_identity=values['loss_absolute_delta'] <= g['loss_absolute'],
                gradient_identity=values['gradient_absolute_delta'] <= g['gradient_absolute'],
                scalar_loss=values['scalar_loss_absolute_delta'] <= g['scalar_absolute'],
                target_mass=values['target_probability_mass_error'] <= g['probability_mass_absolute'])


def worker(a):
    import numpy as np
    import psutil
    start = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity([1])
    b = json.loads(a.binding.read_bytes()); assert sha(a.binding) == a.binding_sha
    a.directory.mkdir(exist_ok=False); stage = 'head'; rows = []
    def guard():
        assert time.monotonic() - start < b['capture_limits']['seconds'] - b['capture_limits']['reserve_seconds']
        assert proc.memory_info().peak_wset <= b['capture_limits']['OS_bytes'] and not proc.children(recursive=True)
    try:
        H = read(b['head']).astype('<f8'); assert np.isfinite(H).all() and np.count_nonzero(H[:, -1]) == 0
        mass_error = 0.; count = 0; stage = 'compile_all_labels'
        for rec in b['records']:
            bits = teacher(rec); m = np.empty((rec['labels'], D), '<f8'); c = np.empty(rec['labels'], '<f8'); ids = np.empty(rec['labels'], '<i4')
            for first in range(0, rec['labels'], b['chunk']):
                last = min(first + b['chunk'], rec['labels']); z = decode(bits[first:last]); assert np.isfinite(z).all()
                q, lq = probabilities(z, False); m[first:last] = contract(q, H, b, False)
                c[first:last] = np.sum(q * lq, axis=1); ids[first:last] = z.argmax(1)
                mass_error = max(mass_error, float(np.max(abs(q.sum(1) - 1)))); count += last - first; guard()
            assert np.isfinite(m).all() and np.isfinite(c).all() and np.count_nonzero(m[:, -1]) == 0
            assert ids.tolist() == rec['source_output_ids']
            rows.append(dict(id=rec['id'], split=rec['split'], domain=rec['domain'], labels=rec['labels'], positions=rec['positions'],
                             moments=save(a.directory, rec['id'] + '.moments', m),
                             negative_entropy=save(a.directory, rec['id'] + '.negative_entropy', c),
                             source_argmax=save(a.directory, rec['id'] + '.source_argmax', ids, '<i4')))
            print(json.dumps(dict(stage=stage, id=rec['id'], labels_compiled=count)), flush=True)
        assert count == N; stage = 'loss_gradient_witnesses'
        arrays, values = witnesses(b, H, rows, False, guard)
        artifacts = {k: save(a.directory, 'witness_' + k, v) for k, v in arrays.items()}
        numeric_flags = flags(values, b); numeric_flags['source_mass'] = mass_error <= b['numeric']['probability_mass_absolute']
        result = dict(schema='ORIGINAL_CATEGORICAL_LOSS_RESULT_V1', freeze=a.freeze, binding_sha256=a.binding_sha,
                      decision='LOSS_COMPILED_PENDING_AUDIT' if all(numeric_flags.values()) else 'LOSS_COMPILE_NUMERIC_FAIL',
                      records=rows, artifacts=artifacts, witness_metrics=values, numeric_flags=numeric_flags,
                      source_probability_mass_error=mass_error, labels_compiled=count, FIT_labels=4422, DEV_labels=4386,
                      FIT_ids=b['FIT_ids'], DEV_ids=b['DEV_ids'], head=b['head'], final_norm=b['final_norm'],
                      source_probability_rows=8880, target_probability_rows=72, stored_loss_gradient_witnesses=72,
                      source_calls=0, native_calls=0, history_forwards=0, optimizer_updates=0, SVD_calls=0, GPU_calls=0,
                      reserved_queries=0, quality_admission=False, speed_admission=False,
                      OS_peak=proc.memory_info().peak_wset, seconds=time.monotonic() - start, scope=b['scope'])
        write(a.out, result); guard()
        assert sum(p.stat().st_size for p in a.directory.iterdir()) + a.out.stat().st_size <= b['capture_limits']['output_bytes']
        print(json.dumps(dict(decision=result['decision'], witness_metrics=values, seconds=result['seconds'])), flush=True)
    except BaseException as error:
        write(a.directory / 'first_fault.json', dict(stage=stage, error=repr(error), records=rows, seconds=time.monotonic() - start)); raise


def audit(a):
    import numpy as np
    import psutil
    start = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity([1])
    b = json.loads(a.binding.read_bytes()); r = json.loads(a.source_result.read_bytes()); t = json.loads(a.source_result.with_suffix('.terminal.json').read_bytes())
    assert sha(a.binding) == a.binding_sha == r['binding_sha256'] == t['binding_sha256'] and a.freeze == r['freeze'] == t['freeze']
    assert t['exit_code'] == 0 and t['error'] is None and t['inputs_before_after_exact'] and t['result'] == extent(a.source_result)
    def guard():
        assert time.monotonic() - start < b['audit_limits']['seconds'] and proc.memory_info().peak_wset <= b['audit_limits']['OS_bytes'] and not proc.children(recursive=True)
    for item in b['inputs'] + t['outputs']:
        assert extent(item['path']) == item; guard()
    outputs = {i['path']: i for i in t['outputs']}
    for row in r['records']:
        for key in ('moments', 'negative_entropy', 'source_argmax'):
            item = row[key]; assert outputs[item['path']] == {k: item[k] for k in ('path', 'bytes', 'sha256')}
    for item in r['artifacts'].values():
        assert outputs[item['path']] == {k: item[k] for k in ('path', 'bytes', 'sha256')}
    H = read(b['head']).astype('<f8'); assert np.count_nonzero(H[:, -1]) == 0
    maxm = maxc = mass_error = 0.; count = 0
    assert len(r['records']) == len(b['records']) == 48
    for rec, row in zip(b['records'], r['records']):
        for key in ('id', 'split', 'domain', 'labels', 'positions'):
            assert rec[key] == row[key]
        m = read(row['moments']); c = read(row['negative_entropy']); ids = read(row['source_argmax']); bits = teacher(rec)
        assert m.shape == (rec['labels'], D) and c.shape == ids.shape == (rec['labels'],)
        assert ids.tolist() == rec['source_output_ids']
        for first in range(0, rec['labels'], b['chunk']):
            last = min(first + b['chunk'], rec['labels']); z = decode(bits[first:last]); q, lq = probabilities(z, True)
            mm = contract(q, H, b, True); cc = np.sum(q * lq, axis=1)
            maxm = max(maxm, float(np.max(abs(mm - m[first:last])))); maxc = max(maxc, float(np.max(abs(cc - c[first:last]))))
            assert np.array_equal(z.argmax(1), ids[first:last]) and np.count_nonzero(m[first:last, -1]) == 0
            mass_error = max(mass_error, float(np.max(abs(q.sum(1) - 1)))); count += last - first; guard()
        print(json.dumps(dict(stage='independent_compile_case', id=rec['id'], labels_verified=count)), flush=True)
    arrays, values = witnesses(b, H, r['records'], True, guard)
    comparison = {k: float(np.max(abs(v - read(r['artifacts'][k])))) for k, v in arrays.items()}
    af = dict(all_moments=maxm <= b['numeric']['moment_absolute'], all_entropy=maxc <= b['numeric']['entropy_absolute'],
              all_source_mass=mass_error <= b['numeric']['probability_mass_absolute'],
              scalar_moments=values['scalar_moment_absolute_delta'] <= b['numeric']['scalar_absolute'],
              scalar_gradients=values['scalar_gradient_absolute_delta'] <= b['numeric']['scalar_absolute'],
              witness_arrays=max(comparison.values()) <= b['numeric']['gradient_absolute'], **flags(values, b))
    assert count == r['labels_compiled'] == N and r['FIT_labels'] == 4422 and r['DEV_labels'] == 4386
    assert r['FIT_ids'] == b['FIT_ids'] and r['DEV_ids'] == b['DEV_ids'] and set(r['FIT_ids']).isdisjoint(r['DEV_ids'])
    assert r['head'] == b['head'] and r['final_norm'] == b['final_norm']
    assert r['source_probability_rows'] == 8880 and r['target_probability_rows'] == r['stored_loss_gradient_witnesses'] == 72
    for key in ('source_calls', 'native_calls', 'history_forwards', 'optimizer_updates', 'SVD_calls', 'GPU_calls', 'reserved_queries'):
        assert r[key] == 0
    assert not r['quality_admission'] and not r['speed_admission']
    assert flags(r['witness_metrics'], b) == {k: r['numeric_flags'][k] for k in flags(r['witness_metrics'], b)}
    assert r['numeric_flags']['source_mass'] == (r['source_probability_mass_error'] <= b['numeric']['probability_mass_absolute'])
    assert t['elapsed_seconds'] <= b['capture_limits']['seconds'] and t['worker_OS_peak_through_exit'] + t['launcher_OS_peak_snapshot'] <= b['capture_limits']['OS_bytes']
    assert sum(i['bytes'] for i in t['outputs']) + t['result']['bytes'] <= b['capture_limits']['output_bytes']; guard()
    decision = 'LOSS_SUPERVISION_QUALIFIED' if all(af.values()) and all(r['numeric_flags'].values()) else 'LOSS_SUPERVISION_NOT_QUALIFIED'
    write(a.out, dict(schema='ORIGINAL_CATEGORICAL_LOSS_AUDIT_V1', freeze=a.freeze, result=extent(a.source_result), binding=extent(a.binding),
          decision=decision, numeric_flags=af, complete_input_output_hashes=True, split_boundary_verified=True,
          all8808_moments_entropy_argmax_checked=True, all72_loss_gradient_witnesses_checked=True,
          scalar_moment_witnesses=24, scalar_gradient_witnesses=12, scalar_loss_witnesses=72,
          max_moment_absolute_delta=maxm, max_entropy_absolute_delta=maxc, source_probability_mass_error=mass_error,
          witness_metrics=values, witness_array_comparison=comparison, labels_verified=count,
          source_probability_rows=8880, target_probability_rows=72, stored_loss_gradient_witnesses=72,
          source_calls=0, native_calls=0, optimizer_updates=0, GPU_calls=0, SVD_calls=0,
          OS_peak=proc.memory_info().peak_wset, seconds=time.monotonic() - start,
          scope='Full independent stored supervision audit. Real fixed-head identities, not finite-precision GPU/native training equivalence or chatbot/speed admission.'))
    print(json.dumps(dict(decision=decision, numeric_flags=af, witness_metrics=values, seconds=time.monotonic() - start)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('bind', 'capture-worker', 'audit-worker'):
        p.add_argument('--' + key, action='store_true')
    for key in ('binding', 'out', 'directory', 'source-result', 'protocol'):
        p.add_argument('--' + key, type=Path)
    for key in ('binding-sha', 'freeze'):
        p.add_argument('--' + key)
    a = p.parse_args()
    if a.bind:
        bind(a)
    elif a.capture_worker:
        worker(a)
    elif a.audit_worker:
        audit(a)
    else:
        raise RuntimeError('Use separately frozen held launcher')
