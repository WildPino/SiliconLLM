"""Saved-only mean/variation recovery and a FIT-only constant correction control."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
from chatbot_source_ffn_recovery_geometry import DOC, SITE, read, receipt, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    oldpath = DOC / 'chatbot_source_ffn_recovery_geometry_binding_20261009.json'
    oldresult = DOC / 'chatbot_source_ffn_recovery_geometry_result_20261009.json'
    old = read(oldpath)
    receipt(oldresult)
    assert sha(oldpath) == read(oldresult)['binding_sha256']
    for item in old['inputs']:
        p = Path(item['path'])
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
    files = [Path(item['path']) for item in old['inputs']]
    files += [oldpath, oldresult, oldresult.with_suffix('.terminal.json'), Path(__file__),
        DOC / 'CHATBOT_SOURCE_FFN_CORRECTION_CENTERING_PROTOCOL_20261009.md']
    files = list(dict.fromkeys(p.resolve() for p in files))
    b = {key: value for key, value in old.items() if key != 'inputs'}
    b.update(phase='centering', worker_path=str(Path(__file__).resolve()),
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    write(a.out, b)
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(files))), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = read(a.binding)
    assert b['phase'] == 'centering'
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
            assert time.monotonic()-start < b['limits']['seconds']-b['limits']['reserve_seconds']
            assert proc.memory_info().peak_wset < b['limits']['OS_bytes']
            assert not torch.cuda.is_initialized() and not proc.children(recursive=True)

        def load(item):
            p = Path(item['path'])
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
            value = torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.bfloat16).reshape(item['shape']).double()
            assert value.ndim == 2 and value.shape[1] == 2048 and torch.isfinite(value).all()
            return value

        def components(prediction, target):
            error = prediction-target
            norm = target.square().sum()
            mean_error = error.mean(0)
            mean_energy = error.shape[0]*mean_error.square().sum()
            varying_energy = (error-mean_error).square().sum()
            identity = float((error.square().sum()-mean_energy-varying_energy).abs()/norm)
            assert identity < 1e-12
            return dict(relative_L2=float((error.square().sum()/norm).sqrt()),
                mean_error_squared_normalized=float(mean_energy/norm),
                variation_error_squared_normalized=float(varying_energy/norm),
                centered_relative_L2=float((varying_energy/(target-target.mean(0)).square().sum().clamp_min(1e-30)).sqrt()),
                identity_relative_defect=identity)

        capture, calibration, recovery = [read(b['documents'][name]) for name in ('capture', 'calibration', 'recovery')]
        for site in b['sites']:
            guard()
            cr = next(r for r in calibration['records'] if r['site'] == site)
            rr = next(r for r in recovery['records'] if r['site'] == site)
            trial = next(t for t in cr['trials'] if (t['alpha'], t['beta']) == (cr['selected']['alpha'], cr['selected']['beta']))
            before = {r['id']: r for r in trial['FIT']+cr['selected']['DEV']}
            after = {r['id']: r for r in rr['after']}
            packets = []
            for row in capture['records']:
                packet = next(v for v in row['sites'] if v['site'] == site)
                y, old, new = load(packet['y']), load(before[row['id']]['output']), load(after[row['id']]['output'])
                assert y.shape == old.shape == new.shape
                packets.append((row, y, old, new))
            constant = torch.stack([(y-old).mean(0) for row,y,old,new in packets if row['split'] == 'FIT']).mean(0)
            path = a.directory / f'site{site:02d}.FIT_case_balanced_constant.f64'
            with path.open('xb') as stream:
                stream.write(constant.numpy().tobytes()); stream.flush(); os.fsync(stream.fileno())
            rows = []
            for row,y,old,new in packets:
                first, last, control = [components(value, y) for value in (old, new, old+constant)]
                assert abs(first['relative_L2']-before[row['id']]['relative_L2']) < 1e-12
                assert abs(last['relative_L2']-after[row['id']]['relative_L2']) < 1e-12
                assert abs(control['variation_error_squared_normalized']-first['variation_error_squared_normalized']) < 1e-12
                improvement = first['relative_L2']**2-last['relative_L2']**2
                mean_gain = first['mean_error_squared_normalized']-last['mean_error_squared_normalized']
                varying_gain = first['variation_error_squared_normalized']-last['variation_error_squared_normalized']
                assert abs(improvement-mean_gain-varying_gain) < 1e-12
                rows.append(dict(id=row['id'], split=row['split'],
                    source_mean_energy_fraction=float(y.shape[0]*y.mean(0).square().sum()/y.square().sum()),
                    before=first, learned=last, FIT_constant_control=control,
                    normalized_squared_error_reduction=improvement,
                    mean_gain_fraction_of_reduction=mean_gain/improvement if improvement != 0 else None,
                    variation_gain_fraction_of_reduction=varying_gain/improvement if improvement != 0 else None))
            completed.append(dict(site=site, cases=rows, constant=dict(path=str(path.resolve()),
                sha256=sha(path), bytes=path.stat().st_size, dtype='F64', shape=[2048])))
            print(json.dumps(dict(stage='site_complete', site=site, elapsed_seconds=time.monotonic()-start)), flush=True)
        result = dict(schema='SOURCE_FFN_LOCAL_RESULT_V1', phase='centering',
            decision='SOURCE_FFN_CENTERING_DIAGNOSTIC_COMPLETE', freeze=a.freeze, binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid, create_time=proc.create_time()), cases=4, records=completed,
            source_forwards=0, source_generations=0, optimizer_updates=0, native_runs=0, reserved_queries=0,
            GPU_allocated_peak=0, GPU_reserved_peak=0, elapsed_seconds=time.monotonic()-start,
            quality_admission=False, native_admission=False,
            scope='Saved-data mean/variation decomposition and FIT-only constant output correction control; no deployed model or broad/causal capacity certificate.')
        write(a.directory/'centering.json', result)
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
