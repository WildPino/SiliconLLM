"""Relative-anchor weighting and local spectrum diagnosis; no model forward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_info, threadpool_limits

import meth444_switch_function_contrast_geometry as Q

M, R = Q.M, Q.R
PROTOCOL = M.DOC/'METH_445_SWITCH_RELATIVE_CONTRAST_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth445_switch_relative_contrast'
PARENT = M.DOC/'meth444_switch_function_contrast_result.json'
PARENT_SHA = '9fbb825172be8d21a4b40d2c619a9906972d6d0c3bbe1519b487879f28d47545'
GLOBAL_RANKS = (32, 64, 128, 256)
LOCAL_RANKS = (32, 64, 96, 127)


def eigensystem(covariance):
    values, basis = np.linalg.eigh(covariance)
    values, basis = values[::-1], basis[:, ::-1].copy()
    for j in range(len(values)):
        if basis[int(np.argmax(np.abs(basis[:, j]))), j] < 0:
            basis[:, j] *= -1
    scale = max(float(np.trace(covariance)), 1)
    assert np.min(values) >= -1e-10*scale
    assert R.relative(basis.T @ basis, np.eye(len(values))) <= 1e-10
    assert R.relative(covariance @ basis, basis*values) <= 1e-10
    assert abs(np.sum(values)-np.trace(covariance)) <= 1e-10*scale
    return values, basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic()
    numeric_start = None
    peak = hashed = 0
    stage = 'bindings'
    result = {'experiment': 'METH-445-relative-anchor-and-local-private-spectrum', 'balanced_global': {}}

    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info()
        peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        output = sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        elapsed = time.monotonic()-start
        assert peak <= 1536 << 20 and output <= 16 << 20, '1536MiB_16MiB'
        assert elapsed <= 120 and ((numeric_start is None and elapsed <= 60) or
               (numeric_start is not None and time.monotonic()-numeric_start <= 60)), 'admission60_numeric60_total120'

    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(4 << 20):
                h.update(block)
                hashed += len(block)
                guard()
        return h.hexdigest()

    try:
        result['helper_sha256'] = {}
        for path in (Path(__file__), PROTOCOL, Path(Q.__file__), Path(M.__file__), Path(R.__file__)):
            M.committed(path)
            result['helper_sha256'][str(path)] = sha(path)
        M.committed(PARENT)
        assert sha(PARENT) == PARENT_SHA
        parent = json.loads(PARENT.read_text(encoding='utf-8'))
        assert all(parent['apparatus_gates'].values()) and parent['decision']['passing_fixed_ranks'] == []
        for path, expected in parent['helper_sha256'].items():
            M.committed(Path(path))
            assert sha(path) == expected
        for item in parent['output_inventory']:
            assert sha(item['path']) == item['sha256']
        result['retained_parent_sha256'] = PARENT_SHA
        result['retained_parent_output_inventory'] = parent['output_inventory']
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['name', 'cmdline']):
            if process.pid in own:
                continue
            name = (process.info['name'] or '').lower()
            argv = process.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_daemons', []).append(process.pid)
                continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_job', process.pid, name)
        psutil.Process().cpu_affinity([0])
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        assert psutil.disk_usage(str(M.ROOT)).free >= 1 << 30
        geometry_path = next(Path(a['path']) for a in parent['output_inventory'] if Path(a['path']).name == 'geometry.npz')
        with np.load(geometry_path, allow_pickle=False) as archive:
            features = archive['features']
            old_means = archive['means']
            old_contrasts = archive['contrasts']
            old_covariance = archive['covariance']
            old_basis = archive['basis']
            old_energy = archive['private_energy']
            keys = archive['keys'].tolist()
        assert features.shape == (24, 128, 768) and keys == parent['data']['keys']
        numeric_start = time.monotonic()
        assert numeric_start-start <= 60, 'admission60_seconds'
        OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime'] = {'numpy': np.__version__, 'torch_import_only': torch.__version__, 'CPU_affinity': [0], 'BLAS': threadpool_info(), 'GPU': False}
            stage = 'exact_retained_geometry_reconstruction'
            means = np.mean(features.astype(np.float64), axis=1)
            contrasts = features.astype(np.float64)-means[:, None, :]
            energy = np.sum(contrasts**2, axis=(1, 2))
            development = contrasts[:18].reshape(18*128, 768)
            R.C.exact(means, old_means)
            R.C.exact(contrasts, old_contrasts)
            R.C.exact(energy, old_energy)
            R.C.exact(development.T @ development/len(development), old_covariance)
            assert np.isfinite(contrasts).all() and np.all(energy > 0)
            for rank in GLOBAL_RANKS:
                residual = contrasts-(contrasts @ old_basis[:, :rank]) @ old_basis[:, :rank].T
                retained = 1-np.sum(residual**2, axis=(1, 2))/energy
                expected = np.asarray(parent['ranks'][str(rank)]['per_anchor_private_energy_retained'])
                assert np.max(np.abs(retained-expected)) <= 1e-12
            shares = energy[:18]/np.sum(energy[:18])
            result['norm_diagnosis'] = {'development_private_energy_shares': shares.tolist(),
                'book0_plus14_position0_energy_share': float(shares[0]+shares[14]),
                'interpretation': 'old_absolute_Frobenius_objective_is_dominated_by_two_position0_anchors; no retrospective gate change'}
            guard()
            stage = 'equal_relative_anchor_development_covariance'
            normalized = (contrasts[:18]/np.sqrt(energy[:18])[:, None, None]).reshape(18*128, 768)
            covariance = normalized.T @ normalized/18
            independent = sum(contrasts[i].T @ contrasts[i]/energy[i] for i in range(18))/18
            assert R.relative(covariance, independent) <= 1e-10 and abs(np.trace(covariance)-1) <= 1e-10
            values, basis = eigensystem(covariance)
            for rank in GLOBAL_RANKS:
                coefficients = contrasts @ basis[:, :rank]
                residual = contrasts-coefficients @ basis[:, :rank].T
                tail = np.sum(residual**2, axis=(1, 2))
                alternate = energy-np.sum(coefficients**2, axis=(1, 2))
                assert np.max(np.abs(tail-alternate)/energy) <= 1e-10
                retained = 1-tail/energy
                gates = {'macro_dev_relative_private_energy_ge0_95': bool(np.mean(retained[:18]) >= .95),
                    'macro_val_relative_private_energy_ge0_95': bool(np.mean(retained[18:]) >= .95),
                    'minimum_val_anchor_relative_private_energy_ge0_90': bool(np.min(retained[18:]) >= .90)}
                result['balanced_global'][str(rank)] = {'per_anchor_relative_energy_retained': retained.tolist(),
                    'macro_development_energy_retained': float(np.mean(retained[:18])),
                    'macro_validation_energy_retained': float(np.mean(retained[18:])), 'gates': gates, 'all_geometry_gates': all(gates.values())}
                guard()
            stage = 'all24_local_oracle_spectra'
            grams = []
            local_values = []
            local_basis = []
            local_retained = {r: [] for r in LOCAL_RANKS}
            r95 = []
            for i in range(24):
                z = contrasts[i]
                assert np.linalg.norm(np.sum(z, axis=0))/np.linalg.norm(z) <= 1e-10
                gram = z @ z.T
                eig, axes = eigensystem(gram)
                assert abs(eig[-1])/energy[i] <= 1e-10, 'centered128_rank_at_most127'
                fractions = np.cumsum(np.maximum(eig, 0))/energy[i]
                assert abs(fractions[126]-1) <= 1e-10
                needed = int(np.searchsorted(fractions, .95)+1)
                assert 1 <= needed <= 127
                r95.append(needed)
                for rank in LOCAL_RANKS:
                    local_retained[rank].append(float(fractions[rank-1]))
                grams.append(gram)
                local_values.append(eig)
                local_basis.append(axes)
                guard()
            np.savez(OUT/'relative_and_local_geometry.npz', balanced_covariance=covariance, balanced_eigenvalues=values,
                     balanced_basis=basis, local_grams=np.asarray(grams), local_eigenvalues=np.asarray(local_values),
                     local_function_axes=np.asarray(local_basis), keys=np.asarray(keys))
            result['local_oracle'] = {'per_anchor_minimum_rank_for0_95_Frobenius_energy': r95,
                'development_rank95_range': [min(r95[:18]), max(r95[:18])], 'validation_rank95_range': [min(r95[18:]), max(r95[18:])],
                'fixed_rank_per_anchor_energy_retained': {str(r): v for r, v in local_retained.items()},
                'scope': 'best separate linear subspace fitted to ALL128 responses at EACH input, including validation; no generalization or posterior-quality claim'}
            result['apparatus_gates'] = {'all_parent_outputs_fresh_exact': True, 'retained_means_contrasts_energy_covariance_byte_exact': True,
                'all_old_rank_energy_curves_reproduced1e_minus12': True, 'relative_covariance_two_forms_trace_eigensystem_qualified': True,
                'all_local_grams_eigen_trace_centered_rank127_qualified': True, 'all_projection_tail_two_forms_qualified': True,
                'no_model_forward_source_mapping_or_new_head': True}
        result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource'] = {'seconds': time.monotonic()-start, 'admission_seconds': numeric_start-start,
            'algebra_and_reporting_seconds': time.monotonic()-numeric_start, 'peak_bytes': peak, 'bytes_hashed': hashed,
            'output_bytes': sum(p['bytes'] for p in result['output_inventory']), 'optimizer_updates': 0, 'model_forward_calls': 0}
        passing = [rank for rank in GLOBAL_RANKS if result['balanced_global'][str(rank)]['all_geometry_gates']]
        result['decision'] = {'passing_balanced_fixed_ranks': passing,
            'next': 'require_native_posterior_checks_before_any_available_common_function_fit' if passing else
                    'relative_weighting_alone_insufficient_compare_local_rank_and_input_coverage_before_new_representation'}
        result['scope'] = 'Pure algebra on immutable444 responses; equal-relative-anchor global PCA fits18dev only; local spectra fit EACH anchor including validation as oracle. No new source values/captures/native heads/quality/rate/LUT/DRAM/training. Frobenius oracle ranks are not KL-optimal/nonlinear compression limits or useful expert counts. No runtime mean/coefficients/state-dependent basis yet. Goal remains incomplete;440/444 original outcomes retained.'
        guard()
        M.write(args.out, result)
        print(json.dumps({'apparatus': result['apparatus_gates'], 'balanced': {r: {'dev': v['macro_development_energy_retained'], 'val': v['macro_validation_energy_retained'], 'gates': v['gates']} for r, v in result['balanced_global'].items()},
            'local_rank95': r95, 'decision': result['decision'], 'resource': result['resource'], 'sha256': M.digest(args.out)}), flush=True)
    except BaseException as error:
        result.update(failure_stage=stage, error=repr(error), seconds=time.monotonic()-start, peak_bytes=peak, bytes_hashed=hashed)
        if OUT.exists():
            result['partial_output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': M.digest(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'), result)
        raise


if __name__ == '__main__':
    main()
