"""Paired output-benefit/preservation bound and exposure diagnostic, no fit."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import meth324_switch_reference as M
import meth422_switch_function_gradient_contract as R
import meth432_switch_output_bound_math as H

PROTOCOL = M.DOC/'METH_432_SWITCH_OUTPUT_BOUND_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth432_switch_output_bound'
RECORDS = {
    'meth418_switch_function_capture_result.json': '4ebb37ed40d788eb85168d028b3e0b538cd7d65e009293cd32d5fb0eb117d829',
    'meth420_switch_function_gradient_result.failure.json': '17f60d043cff598cec0d3e85dfa4d23bdad67b83635422891e92729528056427',
    'meth431_switch_additive_pilot_result.json': '648f37d9b0708488d1491b6af91776efb0bf062084f6cffb317e4af580ccb9e4'}


def distribution(logits):
    p = np.stack([R.probability(row) for row in logits])
    z = logits.astype(np.float64); z -= np.max(z, axis=1, keepdims=True)
    lp = z-np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    assert np.max(np.abs(np.sum(p, axis=1)-1)) <= 5e-13 and np.max(np.abs(np.exp(lp)-p)) <= 5e-13
    return p, lp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; result = {'experiment': 'METH-432-paired-output-benefit-bound-and-exposure', 'book_metrics': []}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        size = sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        assert peak <= 2 << 30 and time.monotonic()-start <= 120 and size <= 16 << 20, 'diagnostic_120sec_2GiB_16MiB'
    def sha(p):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(p).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        result['helper_sha256'] = {}
        for p in (Path(__file__), Path(H.__file__), PROTOCOL, Path(M.__file__), Path(R.__file__), Path(R.C.__file__)):
            M.committed(p); result['helper_sha256'][str(p)] = sha(p)
        records = {}; fresh = {}
        for name, expected in RECORDS.items():
            p = M.DOC/name; M.committed(p); assert sha(p) == expected; records[name] = json.loads(p.read_text(encoding='utf-8'))
            for path, expected in records[name]['helper_sha256'].items():
                if path in fresh: assert fresh[path] == expected; continue
                M.committed(Path(path)); assert sha(path) == expected; fresh[path] = expected
            for a in records[name].get('output_inventory', []):
                assert sha(a['path']) == a['sha256'] and Path(a['path']).stat().st_size == a['bytes']
        prior = records['meth418_switch_function_capture_result.json']; baseline = records['meth420_switch_function_gradient_result.failure.json']; pilot = records['meth431_switch_additive_pilot_result.json']
        assert all(prior['gates'].values()) and len(baseline['baselines']) == 384 and len(pilot['updates']) == 6240 and sum(pilot['capacity_pilot_gates'].values()) == 2
        for a in baseline['baselines']: assert sha(a['archive_path']) == a['archive_sha256'] and a['complete_native_states_and_ALL32128_logits_byte_exact']
        for p, expected in pilot['preserved_original_binary_sha256'].items(): assert sha(p) == expected
        result['retained_record_sha256'] = RECORDS; result['full418_inventory_and384_420_archives_fresh_exact'] = True
        result['source_payload_hashes_inherited431_not_reread_for_output_diagnostic'] = {n: a['sha256'] for n, a in pilot['artifacts'].items()}
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_daemons', []).append(p.pid); continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'; OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'CPU_affinity': [0], 'BLAS': threadpool_info(), 'GPU': False}
            stage = 'independent_analytic_fixture'; result['fixture'] = H.fixture(guard, OUT/'fixture.npz')
            captures = {a['label']: a for a in prior['captures']}; archives = {a['label']: a for a in baseline['baselines']}
            h256 = hashlib.sha256(); h128 = hashlib.sha256(); position_metrics = []; keys = []; labels = []; books = []; validation = []
            for bi in range(24):
                stage = f'paired_book{bi}'; logits = {n: [] for n in (256, 128)}
                for ci in range(4):
                    pair = [captures[f'teacher.n{n}.book{bi}.case{ci}'] for n in (256, 128)]
                    assert pair[0]['pairing_sha256'] == pair[1]['pairing_sha256'] and pair[0]['decoder_ids'] == pair[1]['decoder_ids']
                    for n, item in zip((256, 128), pair):
                        a = archives[item['label']]; assert not a['qualification_only_not_paired_training'] and a['source_capture_sha256'] == item['capture_sha256']
                        assert item['prospective_split'] == ('development' if bi < 18 else 'validation')
                        with np.load(a['archive_path'], allow_pickle=False) as z:
                            assert np.array_equal(z['source_ids'], item['source_ids']) and np.array_equal(z['decoder_ids'], item['decoder_ids'])
                            assert z['logits'].shape == (14, 32128); logits[n].append(z['logits'].copy())
                        if n == 128:
                            data = Path(item['capture_path']).read_bytes(); assert data[:8] == b'SWFUN001' and struct.unpack_from('<6I', data, 8) == (768, 3072, 32128, 128, 11, 1)
                            assert len(data) == 32+14*R.C.DTYPE.itemsize; rows = np.frombuffer(data, dtype=R.C.DTYPE, offset=32)
                            assert np.array_equal(rows['id'], item['decoder_ids']); labels.extend(rows['expert'].tolist())
                    keys.extend([f'book{bi}.case{ci}.position{j}:{pair[0]["pairing_sha256"]}' for j in range(14)]); books.extend([bi]*14)
                a = np.concatenate(logits[256]); b = np.concatenate(logits[128]); p, lp = distribution(a); r, lr = distribution(b)
                h256.update(p.tobytes()); h128.update(r.tobytes()); base256 = -np.sum(p*lp, axis=1); base128 = -np.sum(r*lp, axis=1)
                zero_ce = np.stack([np.asarray([R.numpy_loss(a[i], p[i]), R.numpy_loss(a[i], r[i])]) for i in range(56)])
                assert np.max(np.abs(zero_ce-np.stack([base256, base128], axis=1))) <= 1e-10
                ideal = H.book_metrics(p, r, lp, lr, 1.)
                position_metrics.append(np.stack([base256, base128, .5*(base256+base128), ideal['ce256'], ideal['ce128'], ideal['mix'], ideal['kl256'], ideal['kl128']], axis=1))
                result['book_metrics'].append({'book': bi, 'original256_CE': float(np.mean(base256)), 'original128_CE': float(np.mean(base128)),
                    'original_mix_CE': float(np.mean(.5*(base256+base128))), 'unconstrained_mix_CE': float(np.mean(ideal['mix'])), 'unconstrained_kl256': float(np.mean(ideal['kl256'])),
                    'unconstrained_gain': float(np.mean(.5*(base256+base128)-ideal['mix']))})
                if bi >= 18: validation.append((p, r, lp, lr))
                del logits, a, b, ideal, zero_ce; guard()
            assert len(keys) == 1344 and keys == pilot['data']['keys'] and h256.hexdigest() == pilot['data']['paired_probability256_sha256'] and h128.hexdigest() == pilot['data']['paired_probability128_sha256']
            result['paired_probability_sha256'] = {'256': h256.hexdigest(), '128': h128.hexdigest()}; positions = np.concatenate(position_metrics); del position_metrics; gc.collect()
            np.savez(OUT/'paired_position_metrics.npz', metrics=positions, books=np.asarray(books), source128_labels=np.asarray(labels), actual_classifier_ids=np.asarray(pilot['classifier']['actual_ids']))
            stage = 'validation_constrained_output_bound'; bound, constrained = H.solve(validation, .02, .05, guard)
            val_base = np.mean(positions[1008:, :3], axis=0); assert np.max(np.abs(val_base-[pilot['no_added_validation'][k] for k in ('teacher256_CE', 'teacher128_CE', 'equal_mixture_CE')])) <= 1e-10
            all_metrics = np.concatenate([np.stack([m[k] for k in ('ce256', 'ce128', 'mix', 'kl256', 'kl128')], axis=1) for m in constrained])
            np.savez(OUT/'constrained_validation_metrics.npz', metrics=all_metrics, book_t=np.asarray(bound['book_t']), lambda_global=np.asarray(bound['lambda_global']), mu_books=np.asarray(bound['mu_books']))
            result['output_bound'] = bound | {'baseline_CE256': float(val_base[0]), 'baseline_CE128': float(val_base[1]), 'baseline_mixture_CE': float(val_base[2]),
                'unconstrained_mixture_CE': float(np.mean(positions[1008:, 5])), 'attained_constrained_CE128': float(np.mean(all_metrics[:, 1])),
                'attained_mixture_gain': float(val_base[2]-bound['primal_CE']), 'certified_maximum_mixture_gain_upper': float(val_base[2]-bound['dual_lower_CE']),
                'attained_relative_mixture_gain': float((val_base[2]-bound['primal_CE'])/val_base[2]),
                'attained_relative_teacher128_gain': float((val_base[1]-np.mean(all_metrics[:, 1]))/val_base[1])}
            stage = 'exposure'; ids = np.asarray(pilot['classifier']['actual_ids']); labels = np.asarray(labels); counts_dev = np.bincount(ids[:1008], minlength=128); counts_val = np.bincount(ids[1008:], minlength=128)
            teacher_dev = np.bincount(labels[:1008], minlength=128); teacher_val = np.bincount(labels[1008:], minlength=128)
            assert teacher_dev.tolist() == pilot['data']['teacher_source128_function_exposure_dev'] and teacher_val.tolist() == pilot['data']['teacher_source128_function_exposure_val']
            assert counts_dev.tolist() == pilot['classifier']['development_exposure'] and counts_val.tolist() == pilot['classifier']['validation_exposure']
            assert (3*counts_dev).tolist() == pilot['factor_update_exposure']['real'] == pilot['factor_update_exposure']['adapter']
            mask = np.asarray(pilot['real_validation']['gate_mask'], bool)[1008:]; on = np.bincount(ids[1008:][mask], minlength=128)
            assert mask.sum() == 21 and on.sum() == 21
            result['exposure'] = {'actual_ids_dev': counts_dev.tolist(), 'actual_ids_val': counts_val.tolist(), 'teacher_labels_dev': teacher_dev.tolist(), 'teacher_labels_val': teacher_val.tolist(), 'real_hard_gate_on_val': on.tolist(),
                'unseen_actual_ids_val_positions': int(np.sum(counts_val[counts_dev == 0])), 'unseen_actual_ids_val_distinct': int(np.sum((counts_val > 0) & (counts_dev == 0))),
                'actual_ids_with_at_least2_val_positions': int(np.sum(counts_val >= 2)), 'actual_ids_with_at_least2_hard_gate_on_val_positions': int(np.sum(on >= 2)),
                'teacher_labels_unseen_dev_val_positions': int(np.sum(teacher_val[teacher_dev == 0])),
                'actual_classifier_dev_accuracy': float(np.mean(ids[:1008] == labels[:1008])), 'actual_classifier_val_accuracy': float(np.mean(ids[1008:] == labels[1008:])),
                'factor_update_exposure_dev': (3*counts_dev).tolist()}
            result['apparatus_gates'] = {'fresh_bound_record_and_full_archive_hashes': True, 'Decimal80_known_dual_fixture_and_negative_controls': True,
                'all1344_probability_hashes_and_data_keys_exact431': True, 'original_CE_exact431': True, 'primal_feasible_and_duality_gap_at_most1e_8': True,
                'exact_factor_exposure_and_hard_gate_identity431': True}
            ob = result['output_bound']; feasible = ob['attained_relative_mixture_gain'] >= .01 and ob['attained_relative_teacher128_gain'] >= .01
            impossible = ob['certified_maximum_mixture_gain_upper'] < .01*float(val_base[2])-1e-8
            result['diagnostic_decisions'] = {'both_prediction_gains_attainable_under_output_preservation_bounds': feasible,
                'mixture_gain_target_certifiably_impossible_on_consumed_validation': impossible,
                'at_least8_ids_have2_validation_positions_before_gate': result['exposure']['actual_ids_with_at_least2_val_positions'] >= 8,
                'at_least8_ids_have2_validation_positions_under_frozen431_gate': result['exposure']['actual_ids_with_at_least2_hard_gate_on_val_positions'] >= 8}
            result['decision'] = 'output_objective_feasible_eligible_for_NEW_frozen_checkpoint_function_vs_selector_diagnostic' if feasible else 'output_gain_bound_infeasible_close_prediction_objective' if impossible else 'bound_does_not_resolve_all_gain_conditions_no_fit_license'
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*.npz'))]
        result['resource'] = {'seconds': time.monotonic()-start, 'peak_bytes': peak, 'bytes_hashed': hashed, 'output_bytes': sum(a['bytes'] for a in result['output_inventory'])}
        result['scope'] = 'Consumed paired native-teacher probability outputs and fixed431 exposure only; output-distribution oracle has no model/routing/functions/precision or execution cost restriction, so feasible output benefit is not transferable capacity or achieved generation/task quality. All arbitrary per-position distributions are allowed. Equal56 positions/book. No fit/threshold/gate/route/rank/new corpus/GPU/T4/C export/quality/rate/LUT/DRAM/another-family/~100B claim. Existing431 and426 recipes remain closed.'
        guard(); M.write(args.out, result); print(json.dumps({'apparatus_gates': result['apparatus_gates'], 'diagnostic_decisions': result['diagnostic_decisions'], 'output_bound': result['output_bound'], 'resource': result['resource'], 'sha256': M.digest(args.out)}), flush=True)
    except BaseException as error:
        result.update(failure_stage=stage, error=repr(error), seconds=time.monotonic()-start, peak_bytes=peak, bytes_hashed=hashed)
        if OUT.exists(): result['partial_output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': M.digest(p)} for p in sorted(OUT.glob('*.npz'))]
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
