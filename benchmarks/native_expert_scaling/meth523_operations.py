"""523 immutable source/runtime apparatus; folding reference and retained labels."""
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
BIND = DOC / 'meth523_binding.json'


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


class Meter:
    def __init__(self, kind, seconds, memory):
        self.start = time.monotonic()
        self.proc = psutil.Process()
        self.proc.cpu_affinity([10])
        assert self.proc.cpu_affinity() == [10]
        self.kind, self.seconds, self.memory = kind, seconds, memory
        self.peak = self.hashed = 0
        self.last_output_check = -1.
        self.raw = BIND if kind == 'binding' else DOC / f'meth523_{kind}_result.json'
        assert not self.raw.exists() and not self.raw.with_suffix('.failure.json').exists()
        self.r = {'kind': kind, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'process_instance': {'pid': self.proc.pid, 'create_time_unix': self.proc.create_time()},
                  'argv': sys.argv, 'gates': {}, 'new_model_native_capture_calls': 0,
                  'scope': 'Retained single-bank logical screen, not latency, DRAM or fresh quality.'}
        own = {self.proc.pid, *(p.pid for p in self.proc.parents())}; foreign = []
        for p in psutil.process_iter(['name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): continue
            if name.startswith(('python', 'clang')) or (name.startswith('meth') and name.endswith('.exe')):
                foreign.append({'pid': p.pid, 'name': name})
        assert not foreign, ('foreign_science', foreign)
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
        if time.monotonic() - self.last_output_check >= 1:
            self.last_output_check = time.monotonic()
            assert output_bytes() <= 2 << 30

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
        if hasattr(self, 'out'):
            self.r['partial_outputs'] = [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(self.out.iterdir()) if p.is_file()]
        p = self.raw.with_suffix('.failure.json')
        if not p.exists():
            write(p, self.r)


class Context(Meter):
    def __init__(self, kind, binding_sha):
        super().__init__(kind, 900 if kind == 'main' else 1200, 1536 << 20)
        try:
            self.admit(kind, binding_sha)
        except BaseException:
            self.fail()
            raise

    def admit(self, kind, binding_sha):
        self.out = ROOT / 'results/native_expert_scaling' / ('meth523_fold' if kind == 'main' else 'meth523_audit')
        assert not self.out.exists()
        self.out.mkdir()
        assert self.digest(BIND) == binding_sha
        self.b = json.loads(BIND.read_bytes())
        stat = Path(self.b['payload']['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (self.b['payload_stat']['bytes'], self.b['payload_stat']['mtime_ns'])
        assert sys.version == self.b['python'] and str(Path(sys.executable).resolve()) == self.b['executable']
        for v in self.b['catalog']:
            assert self.digest(v['path']) == v['sha256'], v['path']
        for v in self.b['source_extents']:
            assert self.digest(v['path'], v['offset'], v['bytes']) == v['sha256']
        for rel, preserved_sha in self.b['preserved'].items():
            assert self.digest(ROOT / rel) == preserved_sha
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
        assert all(s[:3] == ' M ' and s[3:] in self.b['preserved'] for s in status), status
        assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
        self.r.update(binding_sha256=binding_sha, source_freeze=self.b['freeze_head'],
                      execution_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        self.r['gates']['fresh_used_input_hidden_labels_source_extents_runtime_and_frozen_science_SHA'] = True

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
        stat = Path(self.b['payload']['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (self.b['payload_stat']['bytes'], self.b['payload_stat']['mtime_ns'])
        for v in self.b['source_extents']:
            assert self.digest(v['path'], v['offset'], v['bytes']) == v['sha256']
        for rel, preserved_sha in self.b['preserved'].items():
            assert self.digest(ROOT / rel) == preserved_sha
        inventory = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': self.digest(p)}
                     for p in sorted(self.out.iterdir()) if p.is_file()]
        size = sum(v['bytes'] for v in inventory)
        assert size <= 2 << 30
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
                          'eligibility': value['eligibility'], 'decision': value['decision'], 'resource': terminal}), flush=True)


def wire(np, path, magic, width, reserved, dtype, count):
    dtype = np.dtype(dtype)
    p = Path(path)
    assert dtype.itemsize == width and p.stat().st_size == 24 + count * width
    with p.open('rb') as f:
        assert f.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
    return np.memmap(p, mode='r', offset=24, dtype=dtype, shape=(count,))



def output_bytes():
    paths = []
    for name in ('meth523_fold', 'meth523_audit'):
        folder = ROOT / 'results/native_expert_scaling' / name
        if folder.exists(): paths.extend(p for p in folder.iterdir() if p.is_file())
    for pattern in ('meth523*.json', 'RETENTION_523*.json', 'ADMISSION_523*.json'):
        paths.extend(DOC.glob(pattern))
    paths.extend((ROOT / 'results/native_expert_scaling').glob('meth523*_resource.json'))
    return sum(p.stat().st_size for p in paths)
