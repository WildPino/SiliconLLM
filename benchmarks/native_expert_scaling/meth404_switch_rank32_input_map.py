"""One fixed rank32 PCR map, calibration book holdout; not model quality."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits, threadpool_info
import meth324_switch_reference as M
import meth403_switch_cross_source_inputs as P

PROTOCOL = M.DOC / 'METH_404_SWITCH_RANK32_INPUT_MAP_PROTOCOL_20261004.md'
RAW = M.DOC / 'meth403_switch_cross_source_inputs_result.json'
OUT = M.ROOT / 'results/native_expert_scaling/meth404_switch_rank32_input_map'
RANK = 32


def metrics(prediction, target, mean):
    difference = prediction - target
    norm = np.linalg.norm(target, axis=1); assert np.all(norm > 0)
    denominator = float(np.sum((target - mean)**2)); assert denominator > 0
    return {'relative_L2_quantiles': np.quantile(np.linalg.norm(difference, axis=1)/norm, [0, .05, .5, .95, 1]).tolist(),
            'squared_error_over_validation_mean_only_error': float(np.sum(difference**2)/denominator),
            'maximum_absolute_difference': float(np.max(np.abs(difference)))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; stage = 'bindings'
    result = {'experiment': 'METH-404-one-fixed-rank32-input-PCR-map', 'banks': [], 'rank': RANK}

    def guard():
        nonlocal peak
        peak = max(peak, psutil.Process().memory_info().rss)
        assert peak <= 1 << 30 and time.monotonic()-start <= 120, 'rank32_2min_1GiB'

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); guard()
        return h.hexdigest()

    try:
        result['bindings'] = {}
        for p in (Path(__file__), PROTOCOL, RAW, Path(P.__file__), Path(M.__file__)):
            M.committed(p); result['bindings'][str(p)] = sha(p)
        assert sha(RAW) == '364eab88aba2779329f13b91252602c558750b4767de801b58108d6f1d37faf0'
        prior = json.loads(RAW.read_text(encoding='utf-8')); assert all(prior['gates'].values()) and len(prior['pairs']) == 24
        arrays = {n: [[] for _ in range(12)] for n in (256, 128)}
        result['paired_input_keys'] = []
        for bi, pair in enumerate(prior['pairs']):
            assert pair['book'] == bi and pair['case'] == 0
            assert pair['prospective_split'] == ('development' if bi < 18 else 'validation')
            assert P.pairing_key(pair['source_ids'], pair['decoder_ids']) == pair['pairing_sha256']
            result['paired_input_keys'].append(pair['pairing_sha256'])
            for n, path, expected in ((256, pair['source256_trace_path'], pair['source256_trace_sha256']),
                                       (128, pair['source128_capture']['trace_path'], pair['source128_capture']['trace_sha256'])):
                assert sha(path) == expected
                for bank, values in enumerate(P.read_trace(path, n)): arrays[n][bank].append(values)
        OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime'] = {'numpy': np.__version__, 'BLAS': threadpool_info(), 'arithmetic': 'F64 fit; F32 two-factor inference diagnostic'}
            assert all(v['num_threads'] == 1 for v in threadpool_info())
            for bank in range(12):
                stage = f'bank_{bank}'; guard()
                x = np.concatenate(arrays[256][bank][:18]).astype(np.float64)
                y = np.concatenate(arrays[128][bank][:18]).astype(np.float64)
                vx = np.concatenate(arrays[256][bank][18:]).astype(np.float64)
                vy = np.concatenate(arrays[128][bank][18:]).astype(np.float64)
                xm = x.mean(axis=0); ym = y.mean(axis=0); xc = x-xm; yc = y-ym
                u, singular, vt = np.linalg.svd(xc, full_matrices=False); guard()
                assert singular[RANK-1]/singular[0] >= 1e-6, 'rank32_condition'
                basis = vt[:RANK].T.copy(); coefficient = (u[:, :RANK].T @ yc)/singular[:RANK, None]
                # Only development responses enter fit; six validation books enter metrics.
                prediction = ym + ((vx-xm) @ basis) @ coefficient
                basis32 = basis.astype(np.float32); coefficient32 = coefficient.astype(np.float32)
                xm32 = xm.astype(np.float32); ym32 = ym.astype(np.float32)
                prediction32 = ym32 + ((vx.astype(np.float32)-xm32) @ basis32) @ coefficient32
                mapped = metrics(prediction32.astype(np.float64), vy, ym)
                f64 = metrics(prediction, vy, ym)
                control = metrics(np.broadcast_to(ym, vy.shape), vy, ym)
                identity = metrics(vx, vy, ym)
                assert abs(control['squared_error_over_validation_mean_only_error']-1) <= 1e-12
                assert np.isfinite(prediction32).all()
                sensitivity = float(np.linalg.norm(prediction32-prediction)/max(np.linalg.norm(prediction), 1e-30))
                eligible = (mapped['squared_error_over_validation_mean_only_error'] <= .5
                            and mapped['relative_L2_quantiles'][2] <= .25 and mapped['relative_L2_quantiles'][3] <= .5
                            and sensitivity <= 1e-5)
                path = OUT / f'bank{bank}.npz'
                np.savez(path, basis=basis32, coefficient=coefficient32, input_mean=xm32, output_mean=ym32)
                result['banks'].append({'stack': 'encoder' if bank < 6 else 'decoder', 'bank': bank % 6,
                    'development_positions': len(x), 'validation_positions': len(vx),
                    'rank32_singular_over_largest': float(singular[RANK-1]/singular[0]),
                    'development_rank32_centered_input_energy_fraction': float(np.sum(singular[:RANK]**2)/np.sum(singular**2)),
                    'validation_F32_map': mapped, 'validation_F64_map': f64, 'validation_mean_only': control, 'validation_identity': identity,
                    'F32_vs_F64_prediction_relative_L2': sensitivity, 'input_eligibility': bool(eligible),
                    'map_path': str(path), 'map_sha256': sha(path), 'map_archive_bytes': path.stat().st_size,
                    'map_F32_parameter_bytes': 4*(768*RANK + RANK*768 + 2*768), 'active_coefficients_per_bank': 2*768*RANK})
                print(json.dumps({'bank': bank, 'input_eligibility': bool(eligible), 'validation': mapped}), flush=True)
        result['gates'] = {'all24_trace_hashes_and_pairing_keys_fresh_exact': True, 'development18_validation6_split_exact': True,
                           'all12_rank32_condition_gates': True, 'mean_only_control_exact': True, 'all_predictions_finite': True, 'BLAS_one_thread': True}
        result['input_eligibility_ALL12'] = all(b['input_eligibility'] for b in result['banks'])
        result['decision'] = 'eligible_only_for_new_actual_function_response_and_output_alignment_screen' if result['input_eligibility_ALL12'] else 'reject_this_fixed_rank32_input_interface_before_selector_or_model_construction'
        result['resource'] = {'main_seconds': time.monotonic()-start, 'maximum_checked_rss_bytes': peak,
                              'map_F32_parameter_bytes_ALL12': sum(b['map_F32_parameter_bytes'] for b in result['banks']),
                              'active_coefficients_ALL12': sum(b['active_coefficients_per_bank'] for b in result['banks'])}
        result['scope'] = 'ONE predeclared rank32 affine PCR per bank, trained only18 consumed development books. Six consumed validation books are calibration holdout, not new original-relative quality. No rank/hyperparameter search. Input hidden prediction alone cannot prove function-response preservation/output alignment, usefulness/new384 selector/artifact, accepted rate or physical DRAM. Prior403 identity diagnostics were observed; validation map errors first observed404. No source-weight read/model inference/network/GPU/T4 here.'
        guard(); M.write(args.out, result); print(json.dumps({'sha256': sha(args.out), 'decision': result['decision'], 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update({'failure_stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start, 'maximum_checked_rss_bytes': peak})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
