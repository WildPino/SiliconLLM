"""Full independent arithmetic audit of native-head teacher moments/preflight.

Different full-width/block accumulation, logaddexp and scalar support minimizer.
Does not import producer equations or replay its loss/gradient function.
"""
import argparse
import json
import math
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


def main(a):
    import numpy as np
    import psutil
    assert extent(a.report)['sha256'] == a.report_sha
    assert extent(a.terminal)['sha256'] == a.terminal_sha
    r = json.loads(a.report.read_bytes())
    terminal = json.loads(a.terminal.read_bytes())
    assert terminal['exit_code'] == 0 and terminal['result'] == extent(a.report)
    assert extent(r['binding']['path']) == r['binding']
    b = json.loads(Path(r['binding']['path']).read_bytes())
    proc = psutil.Process()
    proc.cpu_affinity(list(range(6)))
    stream = None
    max_relative = max_absolute = 0.
    extra = [extent(__file__), extent(DOC / 'NATIVE_HEAD_DOMAIN_PREFLIGHT_AUDIT_PROTOCOL_20261012.md'),
             extent(a.report), extent(a.terminal), r['binding']]
    receipt = dict(schema='NATIVE_HEAD_DOMAIN_PREFLIGHT_AUDIT_TERMINAL_V1', command=sys.orig_argv,
                   pid=proc.pid, creation_time=proc.create_time(), freeze=a.freeze, timing_admitted=False)

    def guard():
        assert time.monotonic() - START <= 900
        assert proc.memory_info().peak_wset <= 4 << 30 and not proc.children(recursive=True)

    def check():
        for e in b['inputs'] + r['outputs'] + extra:
            guard()
            assert extent(e['path']) == e, e['path']

    def load(e):
        with np.load(e['path'], allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def compare(x, y, tolerance=1e-9):
        nonlocal max_relative
        error = float(np.linalg.norm(x-y) / max(1., float(np.linalg.norm(y))))
        max_relative = max(max_relative, error)
        assert error <= tolerance, error

    def scalar(x, y, tolerance=1e-8):
        nonlocal max_absolute
        error = abs(float(x)-float(y))
        max_absolute = max(max_absolute, error)
        assert error <= tolerance * max(1., abs(float(y))), (x, y)

    def source_scores(x, head):
        out = np.zeros((len(x), 65537), 'f8')
        for j in range(0, 2048, 256):
            out += x[:, j:j+256] @ head[:, j:j+256].T
        return out

    def expectation(p, matrix):
        # Split vocabulary reduction into 4096 rows, unlike producer full BLAS.
        out = np.zeros((len(p), 257), 'f8')
        for j in range(0, 65537, 4096):
            out += p[:, j:j+4096] @ matrix[j:j+4096]
        return out

    try:
        check()
        nr = b['native_report']
        C = np.fromfile(nr['head_F32']['path'], '<f4').reshape(65537, 256).astype('f8')
        beta = np.fromfile(nr['bias_F32']['path'], '<f4').astype('f8')
        epsilon = float(np.fromfile(nr['epsilon_F32']['path'], '<f4')[0])
        scale = math.sqrt(256*epsilon)
        aug = np.concatenate((C, (beta / scale)[:, None]), axis=1)
        scalar(r['epsilon_F32_promoted'], epsilon)
        scalar(r['last_scale'], scale)
        assert r['augmented_head_shape'] == [65537, 257]
        get, stream = tensor_reader(Path(b['source']) / 'model.safetensors', np)
        cfg = json.loads((Path(b['source']) / 'config.json').read_bytes())
        gamma = get('model.final_layernorm.weight')
        A = np.empty((65537, 2048), 'f8')
        for lo in range(0, 65537, 1024):
            guard()
            hi = min(lo+1024, 65537)
            A[lo:hi] = get('lm_head.weight', list(range(lo, hi))) * gamma * cfg['lm_head_multiplier']
        stream.close()
        stream = None
        checked_rows = 0
        finite_checks = []
        for s in r['splits']:
            name = s['split']
            data = load(b['parent'][name + '_dataset'])
            old = load(next(x for x in nr['splits'] if x['split'] == name)['codes'])
            pack = load(s['file'])
            H = data['H']
            compare(pack['weights'], data['weights'], 0.)
            t = old['t_f32'].astype('f8')
            radius = np.sqrt(np.mean(t*t, axis=1) + epsilon)
            v = np.concatenate((t/radius[:, None], (scale/radius)[:, None]), axis=1)
            compare(pack['v'], v, 1e-12)
            defect = float(np.max(np.abs(np.sum(v*v, axis=1)-256)))
            assert defect <= b['tolerance']['sphere_absolute'] and np.all(v[:, -1] > 0)
            scalar(defect, s['sphere_absolute'])
            for lo in range(0, len(H), 64):
                guard()
                hi = min(lo+64, len(H))
                x = H[lo:hi]
                teacher = source_scores(x / np.sqrt(np.mean(x*x, axis=1, keepdims=True)+b['epsilon_source']), A)
                qlog = teacher - np.logaddexp.reduce(teacher, axis=1)[:, None]
                q = np.exp(qlog)
                compare(np.sum(q, axis=1), np.ones(hi-lo), 1e-11)
                m = expectation(q, aug)
                entropy = -np.sum(q*qlog, axis=1)
                # Explicit head and bias score decomposition rather than augmented BLAS.
                score = v[lo:hi, :256] @ C.T + v[lo:hi, -1, None] * (beta/scale)
                plog = score - np.logaddexp.reduce(score, axis=1)[:, None]
                probability = np.exp(plog)
                loss = np.sum(q*(qlog-plog), axis=1)
                g = expectation(probability, aug)-m
                compare(pack['moments'][lo:hi], m)
                compare(pack['entropy'][lo:hi], entropy)
                compare(pack['gradient'][lo:hi], g)
                compare(pack['loss'][lo:hi], loss)
                assert np.array_equal(pack['teacher_ids'][lo:hi], np.argmax(teacher, axis=1))
                assert np.array_equal(pack['ids'][lo:hi], np.argmax(score, axis=1))
                # Independently construct the half-ball linear-objective minimizer.
                for row in range(hi-lo):
                    direction = -g[row].copy()
                    if direction[-1] < 0:
                        direction[-1] = 0.
                    norm = math.sqrt(math.fsum(float(z)*float(z) for z in direction))
                    w = direction * (16/norm) if norm else np.zeros(257)
                    assert w[-1] >= 0 and float(w@w) <= 256+1e-10
                    lower = float(loss[row]) + math.fsum(float(a)*float(c) for a, c in zip(g[row], w-v[lo+row]))
                    scalar(lower, pack['raw_lower'][lo+row])
                    assert lower <= loss[row]+1e-8
                if name == 'FIT' and lo == 0:
                    direction = np.sin(np.arange(257)+1.)
                    direction /= np.linalg.norm(direction)
                    analytic = float(g[0]@direction)
                    for index, step in enumerate(b['finite_epsilon']):
                        shift = direction[:256] @ C.T + direction[-1]*(beta/scale)
                        score1 = score[:1] + step*shift
                        score2 = score[:1] - step*shift
                        l1 = np.logaddexp.reduce(score1, axis=1)[0] - float(m[0]@(v[0]+step*direction))-entropy[0]
                        l2 = np.logaddexp.reduce(score2, axis=1)[0] - float(m[0]@(v[0]-step*direction))-entropy[0]
                        finite = float((l1-l2)/(2*step))
                        error = abs(finite-analytic)/max(1., abs(finite), abs(analytic))
                        assert error <= b['tolerance']['derivative_relative']
                        scalar(analytic, r['derivative_checks'][index]['analytic'])
                        finite_checks.append(dict(step=step, independent_error=error))
                checked_rows += hi-lo
            delta = float(np.max(np.abs(pack['loss']-old['native_metrics'][:, 0])))
            compare(pack['native_KL_delta'], pack['loss']-old['native_metrics'][:, 0])
            assert s['native_KL_gate'] == (delta <= b['tolerance']['native_KL_delta'])
            scalar(delta, s['native_KL_max_delta'])
            scalar(data['weights']@pack['loss'], s['case_KL'])
            scalar(data['weights']@(pack['ids'] != pack['teacher_ids']), s['case_disagreement'])
            scalar(pack['raw_lower'].max(), s['maximum_raw_lower'])
            assert np.count_nonzero(pack['raw_lower'] > 1e-8) == s['positive_raw_lower_rows']
            assert len(H) == s['rows']
            print(json.dumps(dict(audited=name, rows=len(H), seconds=time.monotonic()-START)), flush=True)
        assert checked_rows == 8808
        assert len(r['benchmark']['CPU_seconds']) == 5 and r['benchmark']['batch'] == 64
        scalar(np.median(r['benchmark']['CPU_seconds']), r['benchmark']['median_seconds'])
        scalar(r['benchmark']['median_seconds']*139, r['benchmark']['nominal139_batch_evaluations_seconds'])
        assert not r['optimization_implemented'] and not r['numeric_lower_interval_certified']
        flags_pass = all(x['passed'] for x in r['derivative_checks']) and all(s['native_KL_gate'] for s in r['splits'])
        assert r['decision'] == ('ALL_INITIAL_GATES_PASS_PENDING_AUDIT' if flags_pass else 'COMPLETED_INITIAL_GATE_FAILURE_PENDING_AUDIT')
        check()
        guard()
        result = dict(schema='NATIVE_HEAD_DOMAIN_PREFLIGHT_AUDIT_V1', report=extent(a.report), terminal=extent(a.terminal),
                      source=extent(__file__), freeze=a.freeze, binding=r['binding'], independent_rows=checked_rows,
                      all_source_moments_entropies_ids_codes_losses_gradients_support_minima_verified=True,
                      derivative_checks=finite_checks, max_relative=max_relative, max_absolute=max_absolute,
                      before_after_hashes_exact=True, decision='INDEPENDENT_STORED_PASS', initial_gates_pass=flags_pass,
                      scope='Initial actual-head/domain preflight only;no optimized oracle code/native upper or '
                            'interval-certified floor/causal model/useful-n/50 admission.',
                      seconds=time.monotonic()-START, OS_peak=proc.memory_info().peak_wset)
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
    for name in ['report', 'terminal', 'out']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['report-sha', 'terminal-sha', 'freeze']:
        parser.add_argument('--'+name, required=True)
    main(parser.parse_args())
