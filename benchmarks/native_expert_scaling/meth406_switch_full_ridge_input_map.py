"""ONE fixed full-basis ridge input interface on retained paired calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits, threadpool_info
import meth324_switch_reference as M
import meth405_switch_full_basis_inputs as P
import meth404_switch_rank32_input_map as F

PROTOCOL = M.DOC / 'METH_406_SWITCH_FULL_RIDGE_INPUT_MAP_PROTOCOL_20261004.md'
RAW = M.DOC / 'meth405_switch_full_basis_inputs_result.json'
OUT = M.ROOT / 'results/native_expert_scaling/meth406_switch_full_ridge_input_map'
RIDGE_RELATIVE_SINGULAR_SCALE = 1e-5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; stage = 'bindings'
    result = {'experiment': 'METH-406-one-full768-affine-ridge-input-map', 'banks': [], 'relative_ridge_singular_scale': RIDGE_RELATIVE_SINGULAR_SCALE}

    def guard():
        nonlocal peak
        peak = max(peak, psutil.Process().memory_info().rss)
        assert peak <= 1 << 30 and time.monotonic()-start <= 120, 'full_ridge_2min_1GiB'

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); guard()
        return h.hexdigest()

    try:
        result['bindings'] = {}
        for path in (Path(__file__), PROTOCOL, RAW, Path(P.__file__), Path(F.__file__), Path(M.__file__)):
            M.committed(path); result['bindings'][str(path)] = sha(path)
        assert sha(RAW) == '1fb5c222807da2ad8f8e52e8f3d2d5fc64664431edc2332070305fa46cb64788'
        prior = json.loads(RAW.read_text(encoding='utf-8')); assert all(prior['gates'].values()) and len(prior['pairs']) == 96
        arrays = {n: [[] for _ in range(12)] for n in (256, 128)}
        result['paired_input_keys'] = []
        for index, pair in enumerate(prior['pairs']):
            assert (pair['book'], pair['case']) == divmod(index, 4)
            assert pair['prospective_split'] == ('development' if pair['book'] < 18 else 'validation')
            assert P.pairing_key(pair['source_ids'], pair['decoder_ids']) == pair['pairing_sha256']
            result['paired_input_keys'].append(pair['pairing_sha256'])
            for n, path, expected in ((256, pair['source256_trace_path'], pair['source256_trace_sha256']),
                                       (128, pair['source128_capture']['trace_path'], pair['source128_capture']['trace_sha256'])):
                assert sha(path) == expected
                for bank, values in enumerate(P.read_trace(path, n)): arrays[n][bank].append(values)
        assert len(set(result['paired_input_keys'])) == 96
        OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime'] = {'numpy': np.__version__, 'BLAS': threadpool_info(), 'arithmetic': 'F64 SVD/ridge fit, F32 matrix inference'}
            assert all(v['num_threads'] == 1 for v in threadpool_info())
            for bank in range(12):
                stage = f'full_ridge_bank_{bank}'; guard(); positions = 4*(29 if bank < 6 else 14); cut = 18*positions
                x = np.concatenate(arrays[256][bank]).astype(np.float64)
                y = np.concatenate(arrays[128][bank]).astype(np.float64)
                xm = x[:cut].mean(axis=0); ym = y[:cut].mean(axis=0)
                u, singular, vt = np.linalg.svd(x[:cut]-xm, full_matrices=False); guard()
                assert len(singular) == 768 and singular[0] > 0 and np.isfinite(singular).all()
                old = np.asarray(prior['banks'][bank]['singular_value_ratios'])
                assert np.allclose(singular/singular[0], old, rtol=1e-9, atol=1e-14), 'development_spectrum_bridge'
                penalty = (RIDGE_RELATIVE_SINGULAR_SCALE*singular[0])**2
                gain = singular/(singular**2+penalty)
                matrix = vt.T @ ((u.T @ (y[:cut]-ym))*gain[:, None])
                prediction = ym + (x[cut:]-xm) @ matrix
                matrix32 = matrix.astype(np.float32); xm32 = xm.astype(np.float32); ym32 = ym.astype(np.float32)
                prediction32 = ym32 + (x[cut:].astype(np.float32)-xm32) @ matrix32
                assert np.isfinite(prediction32).all()
                measured = F.metrics(prediction32.astype(np.float64), y[cut:], ym)
                mean_only = F.metrics(np.broadcast_to(ym, y[cut:].shape), y[cut:], ym)
                assert abs(mean_only['squared_error_over_validation_mean_only_error']-1) <= 1e-12
                sensitivity = float(np.linalg.norm(prediction32-prediction)/max(np.linalg.norm(prediction), 1e-30))
                eligible = bool(measured['squared_error_over_validation_mean_only_error'] <= .5
                    and measured['relative_L2_quantiles'][2] <= .25 and measured['relative_L2_quantiles'][3] <= .5 and sensitivity <= 1e-5)
                path = OUT / f'bank{bank}.full_ridge.npz'; np.savez(path, matrix=matrix32, input_mean=xm32, output_mean=ym32)
                result['banks'].append({'stack': 'encoder' if bank < 6 else 'decoder', 'bank': bank % 6, 'dimension': 768,
                    'development_positions': cut, 'validation_positions': 6*positions, 'ridge_penalty': float(penalty),
                    'effective_ridge_degrees_of_freedom': float(np.sum(singular**2/(singular**2+penalty))),
                    'development_spectrum_bridge_passed': True, 'validation_F32_map': measured, 'validation_F64_map': F.metrics(prediction, y[cut:], ym),
                    'validation_mean_only': mean_only, 'validation_identity': F.metrics(x[cut:], y[cut:], ym),
                    'F32_vs_F64_prediction_relative_L2': sensitivity, 'input_eligibility': eligible,
                    'map_path': str(path), 'map_sha256': sha(path), 'map_archive_bytes': path.stat().st_size,
                    'map_F32_parameter_bytes': 4*(768*768+2*768), 'active_matrix_coefficients': 768*768})
                print(json.dumps({'bank': bank, 'input_eligibility': eligible, 'validation': measured}), flush=True)
        result['gates'] = {'all96_trace_hashes_and_pairing_keys_fresh_exact': True, 'development18_validation6_split_exact': True,
                           'all12_development_spectra_exact405': True, 'ONE_fixed_full_ridge_variant': True,
                           'mean_only_control_exact': True, 'all_predictions_finite': True, 'BLAS_one_thread': True}
        result['input_eligibility_ALL12'] = all(b['input_eligibility'] for b in result['banks'])
        result['decision'] = 'eligible_only_for_new_actual_function_response_and_output_alignment_screen' if result['input_eligibility_ALL12'] else 'reject_this_fixed_full_affine_ridge_interface_before_selector_or_model_construction'
        result['resource'] = {'main_seconds': time.monotonic()-start, 'maximum_checked_rss_bytes': peak,
                              'map_F32_parameter_bytes_ALL12': sum(b['map_F32_parameter_bytes'] for b in result['banks']),
                              'active_matrix_coefficients_ALL12': sum(b['active_matrix_coefficients'] for b in result['banks'])}
        result['scope'] = 'ONE predeclared full768 affine ridge per bank, penalty=(1e-5*smax)^2, fit ONLY18 consumed development books/all4 cases. Six consumed validation books, not new whole quality. Prior405 rank/spectra observed; these new map errors first406. Fixed local bounds unchanged404/405. No regularization/hyperparameter search/validation selection. No actual source function/output map/new384 selector/artifact/usefulness/accepted-rate/DRAM/other-family proof. No source-weight read/model inference/network/GPU/T4.'
        guard(); M.write(args.out, result); print(json.dumps({'sha256': sha(args.out), 'decision': result['decision'], 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update({'failure_stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start, 'maximum_checked_rss_bytes': peak})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
