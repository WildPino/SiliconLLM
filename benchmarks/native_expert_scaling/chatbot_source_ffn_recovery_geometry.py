"""Saved-only input support and learned-correction geometry; no model calls."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_source_ffn_local import read, receipt
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    names = dict(capture='chatbot_source_ffn_local_capture_result_20261009.json',
        calibration='chatbot_source_ffn_local_calibrate_result_20261009.json',
        recovery='chatbot_source_ffn_local_recovery_result_repair1_20261009.json')
    documents = {name: DOC / filename for name, filename in names.items()}
    data = {name: read(path) for name, path in documents.items()}
    assert data['capture']['decision'] == 'SOURCE_FFN_OPERANDS_PASS'
    assert data['recovery']['effective_final_updates'] == 512
    files = [Path(__file__), Path(sys.executable), B / 'chatbot_falcon_usability.py',
        B / 'chatbot_falcon_usability_launch.py', B / 'chatbot_source_ffn_local.py',
        DOC / 'CHATBOT_SOURCE_FFN_RECOVERY_GEOMETRY_PROTOCOL_20261009.md',
        SITE / 'torch/__init__.py', SITE / 'numpy/__init__.py', SITE / 'psutil/__init__.py']
    files += list((SITE / 'torch').glob('_C*.pyd'))
    files += [ROOT / name for name in ('benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py', 'docs/research/RESEARCH_INDEX.md')]
    for path in documents.values():
        receipt(path)
        files += [path, path.with_suffix('.terminal.json')]
    for row in data['capture']['records']:
        for site in row['sites']:
            for name in ('x', 'y'):
                item = site[name]
                assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
                files.append(Path(item['path']))
    for row in data['calibration']['records']:
        trial = next(t for t in row['trials'] if (t['alpha'], t['beta']) == (row['selected']['alpha'], row['selected']['beta']))
        for item in trial['FIT'] + row['selected']['DEV']:
            files.append(Path(item['output']['path']))
    for row in data['recovery']['records']:
        for item in row['after']:
            files.append(Path(item['output']['path']))
    files = list(dict.fromkeys(p.resolve() for p in files))
    write(a.out, dict(schema='SOURCE_FFN_LOCAL_BINDING_V1', phase='geometry',
        python=str(Path(sys.executable).resolve()), worker_path=str(Path(__file__).resolve()),
        documents={name: str(path) for name, path in documents.items()},
        sites=[0, 23], rank_relative_cutoff=1e-10, energy_fraction=.99,
        limits=dict(seconds=120, reserve_seconds=20, OS_bytes=2 << 30,
            GPU_allocated_bytes=0, GPU_reserved_bytes=0, output_bytes=8 << 20),
        runtime_binding_scope='Saved x/y/before/after packets, successful result receipts, diagnostic code/Python/selected runtime and foreign hashes; no model weights or full DLL-tree.',
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(files))), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = read(a.binding)
    assert b['phase'] == 'geometry'
    a.directory.mkdir(exist_ok=False)
    completed = []
    try:
        import torch
        import psutil
        assert torch.__version__ == '2.6.0+cu124' and sys.version_info[:3] == (3, 12, 10)
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)

        def guard():
            assert time.monotonic() - start < b['limits']['seconds'] - b['limits']['reserve_seconds']
            assert proc.memory_info().peak_wset < b['limits']['OS_bytes']
            assert not torch.cuda.is_initialized() and not proc.children(recursive=True)

        def load(item):
            p = Path(item['path'])
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
            value = torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.bfloat16).reshape(item['shape']).double()
            assert value.ndim == 2 and value.shape[1] == 2048 and torch.isfinite(value).all()
            return value

        capture, calibration, recovery = [read(b['documents'][name]) for name in ('capture', 'calibration', 'recovery')]
        for site in b['sites']:
            guard()
            cr = next(r for r in calibration['records'] if r['site'] == site)
            rr = next(r for r in recovery['records'] if r['site'] == site)
            trial = next(t for t in cr['trials'] if (t['alpha'], t['beta']) == (cr['selected']['alpha'], cr['selected']['beta']))
            before = {r['id']: r for r in trial['FIT'] + cr['selected']['DEV']}
            after = {r['id']: r for r in rr['after']}
            rows = []
            xs, ys = [], []
            for row in capture['records']:
                packet = next(v for v in row['sites'] if v['site'] == site)
                x, y = load(packet['x']), load(packet['y'])
                old, new = load(before[row['id']]['output']), load(after[row['id']]['output'])
                assert old.shape == new.shape == y.shape == x.shape
                residual, correction = y - old, new - old
                r2, c2, dot = residual.square().sum(), correction.square().sum(), (residual * correction).sum()
                before_error = float((r2 / y.square().sum()).sqrt())
                after_error = float(((new - y).square().sum() / y.square().sum()).sqrt())
                assert abs(before_error - before[row['id']]['relative_L2']) < 1e-12
                assert abs(after_error - after[row['id']]['relative_L2']) < 1e-12
                identity = ((new-y).square().sum() - (r2+c2-2*dot)).abs() / r2
                assert float(identity) < 1e-12
                amplitude = dot / c2.clamp_min(1e-30)
                oracle_error = float(((residual-amplitude*correction).square().sum()/y.square().sum()).sqrt())
                rows.append(dict(id=row['id'], split=row['split'], input=x, source_output=y,
                    before_error=before_error, after_error=after_error,
                    correction_relative_to_initial_error=float((c2/r2).sqrt()),
                    correction_residual_cosine=float(dot/(r2*c2).sqrt().clamp_min(1e-30)),
                    oracle_correction_amplitude=float(amplitude), oracle_after_error=oracle_error,
                    squared_error_identity_relative_defect=float(identity)))
                if row['split'] == 'FIT':
                    xs.append(x)
                    ys.append(y)
            support = {}
            for name, fit in (('input', torch.cat(xs)), ('source_output', torch.cat(ys))):
                _, singular, basis = torch.linalg.svd(fit, full_matrices=False)
                energy = singular.square()
                rank = int((singular > singular[0]*b['rank_relative_cutoff']).sum())
                energy_rank = int(torch.searchsorted(energy.cumsum(0), b['energy_fraction']*energy.sum()))+1
                projectors = dict(numerical_span=basis[:rank], energy99_span=basis[:energy_rank])
                sr = dict(FIT_rows=fit.shape[0], width=fit.shape[1], numerical_rank=rank,
                    energy99_rank=energy_rank, stable_rank=float(energy.sum()/energy[0]),
                    singular_values=[float(v) for v in singular], DEV=[])
                for row in rows:
                    if row['split'] == 'DEV':
                        value = row[name]
                        sr['DEV'].append(dict(id=row['id'], outside_FIT={key: float(((value-(value@q.T)@q).square().sum()/value.square().sum()).sqrt()) for key,q in projectors.items()}))
                support[name] = sr
                guard()
            for row in rows:
                del row['input'], row['source_output']
            completed.append(dict(site=site, correction_geometry=rows, support_geometry=support))
            print(json.dumps(dict(stage='site_complete', site=site, elapsed_seconds=time.monotonic()-start)), flush=True)
        result = dict(schema='SOURCE_FFN_LOCAL_RESULT_V1', phase='geometry',
            decision='SOURCE_FFN_SAVED_GEOMETRY_COMPLETE', freeze=a.freeze, binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid, create_time=proc.create_time()), cases=4, records=completed,
            source_forwards=0, source_generations=0, optimizer_updates=0, native_runs=0, reserved_queries=0,
            GPU_allocated_peak=0, GPU_reserved_peak=0, elapsed_seconds=time.monotonic()-start,
            quality_admission=False, native_admission=False,
            scope='Descriptive F64 saved-data geometry only. Input/output subspace energy is not knowledge capacity or a nonlinear error bound. Oracle amplitudes use actual source targets and cannot be deployed.')
        write(a.directory/'geometry.json', result)
        write(a.out, result)
        guard()
    except BaseException as exc:
        write(a.directory/'first_failure.json', dict(fault=repr(exc), completed=completed, elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('bind', 'worker'), default='worker')
    p.add_argument('--binding', type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    bind(a) if a.mode == 'bind' else worker(a)
