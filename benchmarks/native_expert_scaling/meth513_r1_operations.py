"""513 immutable metadata/resource apparatus; no index or certificate mathematics."""
import datetime
import hashlib
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
BIND = DOC / 'meth513_r1_binding.json'


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


class Context:
    def __init__(self, kind, binding_sha):
        self.start = time.monotonic()
        self.proc = psutil.Process()
        self.proc.cpu_affinity([10])
        self.peak = self.hashed = 0
        self.kind = kind
        self.raw = DOC / f'meth513_{kind}_result.json'
        self.out = ROOT / 'results/native_expert_scaling' / ('meth513_index' if kind == 'main' else 'meth513_audit')
        assert not self.raw.exists() and not self.raw.with_suffix('.failure.json').exists() and not self.out.exists()
        self.out.mkdir()
        self.r = {'kind': kind, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'process_instance': {'pid': self.proc.pid, 'create_time_unix': self.proc.create_time()},
                  'argv': sys.argv, 'gates': {}, 'native_model_source_function_calls': 0,
                  'scope': 'Finite retained-data geometry/logical work; no native timing/DRAM/fresh quality promotion.'}
        self.timer = threading.Timer(177, self.deadline)
        self.timer.daemon = True
        self.timer.start()
        assert self.digest(BIND) == binding_sha
        self.b = json.loads(BIND.read_bytes())
        assert sys.version == self.b['python'] and str(Path(sys.executable).resolve()) == self.b['executable']
        for v in self.b['catalog']:
            assert self.digest(v['path']) == v['sha256']
        for rel, sha in self.b['preserved'].items():
            assert self.digest(ROOT / rel) == sha
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
        assert all(v[:3] == ' M ' and v[3:] in self.b['preserved'] for v in status)
        assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
        self.r.update(binding_sha256=binding_sha, execution_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), source_freeze=self.b['freeze_head'])
        self.r['gates']['fresh_actual_used_input_runtime_source_SHA_before_NumPy'] = True

    def deadline(self):
        try:
            self.r.update(fault='hard177s remaining deadline', resource=self.resources())
            write(self.raw.with_suffix('.failure.json'), self.r)
        finally:
            os._exit(124)

    def guard(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        assert self.peak <= 2 << 30 and time.monotonic() - self.start <= 177

    def resources(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        return {'seconds': time.monotonic() - self.start, 'OS_peak_bytes': self.peak,
                'bytes_hashed': self.hashed, 'limits': [177, 2 << 30]}

    def digest(self, path):
        p = Path(path)
        stat = p.stat()
        h = hashlib.sha256()
        with p.open('rb') as f:
            while data := f.read(8 << 20):
                h.update(data)
                self.hashed += len(data)
                self.guard()
        assert (p.stat().st_size, p.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
        return h.hexdigest()

    def numpy(self):
        import importlib.metadata
        import numpy as np
        import threadpoolctl as tp
        for k, v in self.b['packages'].items():
            assert importlib.metadata.version(k) == v
        self.pool = tp.threadpool_limits(1)
        assert all(v['num_threads'] == 1 for v in tp.threadpool_info())
        known = {str(Path(v['path']).resolve()) for v in self.b['catalog']}
        for v in tp.threadpool_info():
            assert str(Path(v['filepath']).resolve()) in known
        np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
        self.r['runtime'] = {'pools': tp.threadpool_info(), 'affinity': self.proc.cpu_affinity(), 'Python': sys.version}
        return np

    def data(self, key):
        return self.b['data'][key]['path']

    def finish(self, value):
        self.r.update(value)
        inventory = []
        for p in sorted(self.out.iterdir()):
            if p.is_file():
                inventory.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': self.digest(p)})
        size = sum(v['bytes'] for v in inventory)
        assert size <= (64 << 20 if self.kind == 'main' else 16 << 20)
        self.r.update(output_inventory=inventory, ended_compute_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), resource_before_serialization=self.resources())
        write(self.raw, self.r)
        terminal = {'result_sha256': self.digest(self.raw), 'process_instance': self.r['process_instance'],
                    **self.resources(), 'output_bytes': size, 'new_native_model_source_function_calls': 0}
        write(self.out / 'terminal_resource.json', terminal)
        self.guard()
        self.timer.cancel()
        print(json.dumps({'result_sha256': terminal['result_sha256'], 'gates': self.r['gates'],
                          'views': value['views'], 'eligibility': value['eligibility'], 'resource': terminal}), flush=True)
        return self.r

    def fail(self):
        self.timer.cancel()
        self.r.update(traceback=traceback.format_exc(), resource=self.resources())
        destination = self.raw.with_suffix('.failure.json')
        if not destination.exists():
            write(destination, self.r)


def wire(np, path, magic, width, reserved, dtype, count):
    p = Path(path)
    dtype = np.dtype(dtype)
    assert p.stat().st_size == 24 + count * dtype.itemsize
    with p.open('rb') as f:
        assert f.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
    return np.memmap(p, mode='r', offset=24, dtype=dtype, shape=(count,))
