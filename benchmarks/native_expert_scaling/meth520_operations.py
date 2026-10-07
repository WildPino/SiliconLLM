"""520 immutable geometry/runtime apparatus; no source-function evaluations."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
BIND = DOC / 'meth520_binding.json'


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


class Meter:
    def __init__(self, kind, seconds, memory):
        self.start = time.monotonic()
        self.proc = psutil.Process()
        self.proc.cpu_affinity([10])
        self.kind, self.seconds, self.memory = kind, seconds, memory
        self.peak = self.hashed = 0
        self.raw = BIND if kind == 'binding' else DOC / f'meth520_{kind}_result.json'
        assert not self.raw.exists() and not self.raw.with_suffix('.failure.json').exists()
        self.r = {'kind': kind, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'process_instance': {'pid': self.proc.pid, 'create_time_unix': self.proc.create_time()},
                  'argv': sys.argv, 'gates': {}, 'new_model_native_capture_calls': 0,
                  'scope': 'Retained single-bank logical screen, not latency, DRAM or fresh quality.'}
        self.timer = threading.Timer(seconds, self.deadline)
        self.timer.daemon = True
        self.timer.start()

    def deadline(self):
        try:
            self.r.update(fault='hard deadline', resource=self.resources())
            if not self.raw.with_suffix('.failure.json').exists():
                write(self.raw.with_suffix('.failure.json'), self.r)
        finally:
            os._exit(124)

    def guard(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        assert self.peak <= self.memory and time.monotonic() - self.start <= self.seconds

    def resources(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        return {'seconds': time.monotonic() - self.start, 'OS_peak_bytes': self.peak,
                'bytes_hashed': self.hashed, 'limits': [self.seconds, self.memory]}

    def digest(self, path, offset=0, count=None):
        p = Path(path)
        s = p.stat()
        length = s.st_size - offset if count is None else count
        assert offset >= 0 and length >= 0 and offset + length <= s.st_size
        h = hashlib.sha256()
        with p.open('rb') as f:
            f.seek(offset)
            remaining = length
            while remaining:
                data = f.read(min(8 << 20, remaining))
                assert data
                h.update(data)
                remaining -= len(data)
                self.hashed += len(data)
                self.guard()
        assert (p.stat().st_size, p.stat().st_mtime_ns) == (s.st_size, s.st_mtime_ns)
        return h.hexdigest()

    def fail(self):
        self.timer.cancel()
        self.r.update(traceback=traceback.format_exc(), resource=self.resources())
        p = self.raw.with_suffix('.failure.json')
        if not p.exists():
            write(p, self.r)


class Context(Meter):
    def __init__(self, kind, binding_sha):
        super().__init__(kind, 900, 1536 << 20)
        try:
            self.admit(kind, binding_sha)
        except BaseException:
            self.fail()
            raise

    def admit(self, kind, binding_sha):
        self.out = ROOT / 'results/native_expert_scaling' / ('meth520_geometry' if kind == 'main' else 'meth520_audit')
        assert not self.out.exists()
        self.out.mkdir()
        assert self.digest(BIND) == binding_sha
        self.b = json.loads(BIND.read_bytes())
        assert sys.version == self.b['python'] and str(Path(sys.executable).resolve()) == self.b['executable']
        for v in self.b['catalog']:
            assert self.digest(v['path']) == v['sha256'], v['path']
        self.dictionary_stat()
        v = self.b['dictionary_prefix']
        assert self.digest(v['path'], v['offset'], v['bytes']) == v['sha256']
        for rel, preserved_sha in self.b['preserved'].items():
            assert self.digest(ROOT / rel) == preserved_sha
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
        assert all(s[:3] == ' M ' and s[3:] in self.b['preserved'] for s in status), status
        assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
        self.r.update(binding_sha256=binding_sha, source_freeze=self.b['freeze_head'],
                      execution_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        self.r['gates']['fresh_used_geometry_features_runtime_science_and_fixed_dictionary_prefix_SHA'] = True

    def dictionary_stat(self):
        b = self.b['dictionary_file']
        s = Path(b['path']).stat()
        assert (s.st_size, s.st_mtime_ns) == (b['bytes'], b['mtime_ns'])

    def numpy(self):
        import numpy as np
        import threadpoolctl as tp
        for key, version in self.b['packages'].items():
            assert importlib.metadata.version(key) == version
        self.pool = tp.threadpool_limits(1)
        known = {str(Path(v['path']).resolve()) for v in self.b['catalog']}
        pools = tp.threadpool_info()
        assert all(v['num_threads'] == 1 and str(Path(v['filepath']).resolve()) in known for v in pools)
        np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
        self.r['runtime'] = {'pools': pools, 'affinity': self.proc.cpu_affinity(), 'Python': sys.version}
        return np

    def data(self, key):
        return self.b['data'][key]['path']

    def finish(self, value):
        self.r.update(value)
        self.dictionary_stat()
        for rel, preserved_sha in self.b['preserved'].items():
            assert self.digest(ROOT / rel) == preserved_sha
        inventory = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': self.digest(p)}
                     for p in sorted(self.out.iterdir()) if p.is_file()]
        size = sum(v['bytes'] for v in inventory)
        assert size <= 32 << 20
        self.r.update(output_inventory=inventory, ended_compute_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      resource_before_serialization=self.resources())
        write(self.raw, self.r)
        terminal = {'result_sha256': self.digest(self.raw), 'process_instance': self.r['process_instance'],
                    'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    **self.resources(), 'output_bytes_before_terminal': size}
        write(self.out / 'terminal_resource.json', terminal)
        self.guard()
        self.timer.cancel()
        print(json.dumps({'result_sha256': terminal['result_sha256'], 'gates': self.r['gates'],
                          'views': value['views'], 'eligibility': value['eligibility'], 'resource': terminal}), flush=True)


def wire(np, path, magic, width, reserved, dtype, count):
    dtype = np.dtype(dtype)
    p = Path(path)
    assert dtype.itemsize == width and p.stat().st_size == 24 + count * width
    with p.open('rb') as f:
        assert f.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
    return np.memmap(p, mode='r', offset=24, dtype=dtype, shape=(count,))


def load_geometry(np, ctx):
    N, L = 17540, 513
    uid = wire(np, ctx.data('uid'), b'M493U001', 180, 0,
               [('m', '<u4', (13,)), ('hash', 'u1', (128,))], N)
    m = uid['m']
    occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
    feat = wire(np, ctx.data('features'), b'M495FEA1', 6148, 512,
                [('phi', '<f4', (512,)), ('q', '<i2', (512,)), ('alpha', '<f4'), ('l', '<f4', (768,))], N)
    assert np.array_equal(m[:, 0], np.arange(N)) and np.all(m[:, 2] == 11) and np.all(m[:, 3] < 128) and np.all(m[:, 6] == 2)
    dev, val = (m[:, 4] & 5) != 0, (m[:, 4] & 2) != 0
    assert (int(dev.sum()), int(val.sum())) == (11721, 5819) and np.all(dev ^ val)
    assert np.array_equal(occ[:, 0], np.arange(19962)) and np.all(occ[:, 3] == 128) and np.all(occ[:, 8] == 11) and np.all(occ[:, 12] == 1)
    assert np.array_equal(np.bincount(occ[:, 1], minlength=N), m[:, 7])
    for oi, mi in [(14, 1), (11, 3), (15, 12), (16, 10)]:
        assert np.array_equal(occ[:, oi], m[occ[:, 1], mi])
    ids = np.flatnonzero(dev)
    codes, alpha = feat['q'][ids], feat['alpha'][ids]
    assert np.isfinite(alpha).all() and np.all(alpha > 0) and codes.min() >= -32767 and codes.max() <= 32767
    h = np.ones((len(ids), L), '<f8')
    h[:, :512] = codes.astype('<f8') * alpha.astype('<f8')[:, None]
    assert np.isfinite(h).all()
    p = Path(ctx.data('geometry'))
    assert p.stat().st_size == 20475956
    with p.open('rb') as f:
        assert f.read(20) == struct.pack('<8sIII', b'M494GEO1', 768, 512, 513)
        f.seek(20 + 3 * 768 * 768 * 8 + 513 * 513 * 8)
        kreg = np.fromfile(f, '<f8', L * L).reshape(L, L)
    assert np.isfinite(kreg).all() and np.array_equal(kreg, kreg.T)
    return uid, occ, ids, codes, alpha, h, kreg


