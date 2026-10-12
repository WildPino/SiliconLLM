"""Exact teacher moments and bounded CPU cost for the qualified native head image.

No optimization, no source-model forward, and no native feasibility claim from
the relaxed half-ball. Uses actual F32 head/bias/epsilon promoted to F64.
"""
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
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0] = [str(B), str(SITE)]
from falcon_auxiliary_s6_transport import extent, write, tensor_reader
START = time.monotonic()


def prepare(a):
    import numpy as np
    import psutil
    report = DOC / 'gauged_native_readout_result_replay1_20261012.json'
    audit = DOC / 'gauged_native_stored_adjudication_20261012.json'
    r = json.loads(report.read_bytes())
    ad = json.loads(audit.read_bytes())
    assert ad['report'] == extent(report) and ad['decision'] == 'INDEPENDENT_STORED_PASS'
    parent = json.loads(Path(r['binding']['path']).read_bytes())
    broad = parent['parent']
    paths = [Path(__file__), B / 'falcon_auxiliary_s6_transport.py', report, audit,
             Path(r['binding']['path']), DOC / 'NATIVE_HEAD_DOMAIN_PREFLIGHT_PROTOCOL_V2_20261012.md',
             Path(r['head_F32']['path']), Path(r['bias_F32']['path']), Path(r['epsilon_F32']['path']),
             Path(parent['source']) / 'model.safetensors', Path(parent['source']) / 'config.json',
             Path(sys.executable), Path(sys.executable).parent / 'python312.dll',
             Path(np.__file__), Path(np._core._multiarray_umath.__file__), Path(psutil.__file__)]
    paths += [Path(broad[s + '_dataset']['path']) for s in ['FIT', 'DEV']]
    paths += [Path(s['codes']['path']) for s in r['splits']]
    paths += list((SITE / 'numpy.libs').glob('*.dll')) + list((SITE / 'psutil').glob('*.pyd'))
    binding = dict(schema='NATIVE_HEAD_DOMAIN_PREFLIGHT_BINDING_V1', python=str(Path(sys.executable).resolve()),
                   native_report=r, source=parent['source'], parent=broad, epsilon_source=parent['epsilon_source'],
                   batch=64, threads=6, warmups=2, repeats=5,
                   tolerance=dict(sphere_absolute=1e-12, native_KL_delta=1e-5, derivative_relative=2e-6),
                   finite_epsilon=[2e-6, 2e-7], limits=dict(seconds=900, OS_bytes=4 << 30, output_bytes=96 << 20),
                   inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)],
                   scope='All8808 existing raw source labels and qualified actual F32 head image. Exact categorical '
                         'moments/full initial loss+gradient+support lower and CPU cost;no optimization/GPU/native '
                         'replay/source model/causal code/useful-n/same-artifact50 claim. Existing DEV,not pristine.')
    write(a.out, binding)
    print(json.dumps(dict(binding=extent(a.out), seconds=time.monotonic() - START)), flush=True)


def run(a):
    import numpy as np
    import psutil
    assert extent(a.binding)['sha256'] == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    assert all(os.environ.get(k) == str(b['threads']) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'])
    assert not a.out.exists() and not a.directory.exists()
    proc = psutil.Process()
    proc.cpu_affinity(list(range(6)))
    stream = None
    receipt = dict(schema='NATIVE_HEAD_DOMAIN_PREFLIGHT_TERMINAL_V1', command=sys.orig_argv,
                   pid=proc.pid, creation_time=proc.create_time(), freeze=a.freeze, timing_admitted=False)

    def guard():
        assert time.monotonic() - START <= b['limits']['seconds']
        assert proc.memory_info().peak_wset <= b['limits']['OS_bytes'] and not proc.children(recursive=True)
        if a.directory.exists():
            assert sum(p.stat().st_size for p in a.directory.iterdir()) <= b['limits']['output_bytes']

    def check():
        for e in b['inputs']:
            guard()
            assert extent(e['path']) == e, e['path']

    def load(e):
        with np.load(e['path'], allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def logprob(score):
        shift = score - score.max(axis=1, keepdims=True)
        logz = np.log(np.exp(shift).sum(axis=1))
        return shift - logz[:, None]

    def fg(v, m, entropy, aug):
        score = v @ aug.T
        lp = logprob(score)
        probability = np.exp(lp)
        loss = score.max(axis=1) + np.log(np.exp(score - score.max(axis=1, keepdims=True)).sum(axis=1))
        loss -= np.sum(m * v, axis=1) + entropy
        gradient = probability @ aug - m
        return loss, gradient, np.argmax(score, axis=1)

    try:
        check()
        a.directory.mkdir()
        r = b['native_report']
        head = np.fromfile(r['head_F32']['path'], '<f4').reshape(65537, 256).astype('f8')
        bias = np.fromfile(r['bias_F32']['path'], '<f4').astype('f8')
        epsilon = float(np.fromfile(r['epsilon_F32']['path'], '<f4')[0])
        scale = math.sqrt(256 * epsilon)
        aug = np.column_stack([head, bias / scale])
        get, stream = tensor_reader(Path(b['source']) / 'model.safetensors', np)
        cfg = json.loads((Path(b['source']) / 'config.json').read_bytes())
        gamma = get('model.final_layernorm.weight')
        source_head = np.empty((65537, 2048), 'f8')
        for lo in range(0, 65537, 1024):
            guard()
            hi = min(lo + 1024, 65537)
            source_head[lo:hi] = get('lm_head.weight', list(range(lo, hi))) * gamma * cfg['lm_head_multiplier']
        stream.close()
        stream = None
        outputs = []
        splits = []
        bench = None
        derivatives = []
        for s in r['splits']:
            split = s['split']
            data = load(b['parent'][split + '_dataset'])
            old = load(s['codes'])
            h = data['H']
            t = old['t_f32'].astype('f8')
            radius = np.sqrt(np.sum(t * t, axis=1) / 256 + epsilon)
            v = np.column_stack([t / radius[:, None], scale / radius])
            sphere = float(np.max(np.abs(np.sum(v * v, axis=1) - 256)))
            assert sphere <= b['tolerance']['sphere_absolute'] and np.all(v[:, -1] > 0)
            n = len(h)
            moments = np.empty((n, 257), 'f8')
            entropy = np.empty(n, 'f8')
            teacher_ids = np.empty(n, 'i8')
            loss = np.empty(n, 'f8')
            gradients = np.empty_like(moments)
            ids = np.empty(n, 'i8')
            lower = np.empty(n, 'f8')
            compile_seconds = 0.
            evaluate_seconds = 0.
            for lo in range(0, n, b['batch']):
                guard()
                hi = min(lo + b['batch'], n)
                began = time.monotonic()
                normalized = h[lo:hi] / np.sqrt(np.mean(h[lo:hi] ** 2, axis=1, keepdims=True) + b['epsilon_source'])
                qlog = logprob(normalized @ source_head.T)
                q = np.exp(qlog)
                moments[lo:hi] = q @ aug
                entropy[lo:hi] = -np.sum(q * qlog, axis=1)
                teacher_ids[lo:hi] = np.argmax(qlog, axis=1)
                compile_seconds += time.monotonic() - began
                began = time.monotonic()
                loss[lo:hi], gradients[lo:hi], ids[lo:hi] = fg(v[lo:hi], moments[lo:hi], entropy[lo:hi], aug)
                evaluate_seconds += time.monotonic() - began
                g = gradients[lo:hi]
                support_norm = np.sqrt(np.sum(g[:, :256] ** 2, axis=1) + np.minimum(g[:, -1], 0) ** 2)
                lower[lo:hi] = loss[lo:hi] - np.sum(g * v[lo:hi], axis=1) - 16 * support_norm
                if split == 'FIT' and lo == 0:
                    times = []
                    for repeat in range(b['warmups'] + b['repeats']):
                        guard()
                        began = time.monotonic()
                        new = fg(v[lo:hi], moments[lo:hi], entropy[lo:hi], aug)
                        elapsed = time.monotonic() - began
                        assert np.array_equal(new[0], loss[lo:hi]) and np.array_equal(new[1], g)
                        if repeat >= b['warmups']:
                            times.append(elapsed)
                    bench = dict(batch=hi-lo, CPU_seconds=times, median_seconds=float(np.median(times)),
                                 nominal139_batch_evaluations_seconds=float(np.median(times)) * 139,
                                 admission=False, note='Component cost estimate,not optimizer or end-to-end throughput.')
                    direction = np.sin(np.arange(257) + 1.)
                    direction /= np.linalg.norm(direction)
                    analytic = float(g[0] @ direction)
                    for step in b['finite_epsilon']:
                        plus = fg(v[:1] + step * direction, moments[:1], entropy[:1], aug)[0][0]
                        minus = fg(v[:1] - step * direction, moments[:1], entropy[:1], aug)[0][0]
                        finite = float((plus - minus) / (2 * step))
                        error = abs(finite - analytic) / max(1., abs(finite), abs(analytic))
                        derivatives.append(dict(step=step, analytic=analytic, finite=finite, relative_error=error,
                                                passed=error <= b['tolerance']['derivative_relative']))
            native_delta = float(np.max(np.abs(loss - old['native_metrics'][:, 0])))
            native_delta_pass = native_delta <= b['tolerance']['native_KL_delta']
            path = a.directory / (split + '_moments.npz')
            np.savez(path, moments=moments, entropy=entropy, teacher_ids=teacher_ids, v=v, loss=loss,
                     gradient=gradients, ids=ids, raw_lower=lower, weights=data['weights'], native_KL_delta=loss-old['native_metrics'][:, 0])
            outputs.append(extent(path))
            splits.append(dict(split=split, rows=n, file=outputs[-1], sphere_absolute=sphere,
                               native_KL_max_delta=native_delta, native_KL_gate=native_delta_pass, compile_seconds=compile_seconds,
                               evaluate_seconds=evaluate_seconds, case_KL=float(data['weights'] @ loss),
                               case_disagreement=float(data['weights'] @ (ids != teacher_ids)),
                               maximum_raw_lower=float(lower.max()), positive_raw_lower_rows=int(np.count_nonzero(lower > 1e-8))))
            print(json.dumps(dict(split=split, rows=n, compiled_and_evaluated=True, seconds=time.monotonic()-START)), flush=True)
        check()
        guard()
        all_flags_pass = all(x['passed'] for x in derivatives) and all(s['native_KL_gate'] for s in splits)
        result = dict(schema='NATIVE_HEAD_DOMAIN_PREFLIGHT_RESULT_V1', binding=extent(a.binding), freeze=a.freeze,
                      augmented_head_shape=list(aug.shape), epsilon_F32_promoted=epsilon, last_scale=scale,
                      outputs=outputs, splits=splits, benchmark=bench, derivative_checks=derivatives,
                      decision='ALL_INITIAL_GATES_PASS_PENDING_AUDIT' if all_flags_pass else 'COMPLETED_INITIAL_GATE_FAILURE_PENDING_AUDIT',
                      optimization_implemented=False, numeric_lower_interval_certified=False,
                      seconds=time.monotonic()-START, OS_peak=proc.memory_info().peak_wset,
                      scope=b['scope'], before_after_hashes_exact=True)
        write(a.out, result)
        receipt.update(exit_code=0, error=None, result=extent(a.out))
    except BaseException as error:
        import traceback
        receipt.update(exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if stream is not None:
            stream.close()
        receipt.update(seconds=time.monotonic()-START, OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'), receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    for name in ['binding', 'directory', 'out']:
        parser.add_argument('--' + name, type=Path)
    for name in ['binding-sha', 'freeze']:
        parser.add_argument('--' + name)
    args = parser.parse_args()
    assert args.prepare != args.run
    if args.prepare:
        prepare(args)
    else:
        run(args)
