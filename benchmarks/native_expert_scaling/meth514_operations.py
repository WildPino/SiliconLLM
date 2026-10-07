"""Bounded, immutable apparatus for the actual256 retained-state head screen."""
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
BIND = DOC / 'meth514_binding.json'


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def manifest(path):
    data = Path(path).read_bytes()
    assert data[:8] == b'SWI8C001'
    cfg = struct.unpack_from('<13If', data, 8)
    nf, nt = struct.unpack_from('<2I', data, 64)
    cursor = 72

    def string():
        nonlocal cursor
        size = struct.unpack_from('<I', data, cursor)[0]
        cursor += 4
        value = data[cursor:cursor + size].decode('utf8')
        cursor += size
        return value

    files = [string() for _ in range(nf)]
    tensors = {}
    for _ in range(nt):
        name = string()
        assert name not in tensors
        tensors[name] = struct.unpack_from('<5I3Q', data, cursor)
        cursor += 44
    assert cursor == len(data) and nf == 1 and nt == 6392
    assert cfg[:9] == (768, 3072, 12, 64, 12, 12, 256, 64, 32128)
    return cfg, files, tensors


class Context:
    def __init__(self, kind):
        self.start = time.monotonic()
        self.proc = psutil.Process()
        self.proc.cpu_affinity([10])
        self.kind = kind
        self.peak = self.hashed = 0
        self.raw = DOC / f'meth514_{kind}_result.json'
        self.out = ROOT / 'results/native_expert_scaling' / f'meth514_{kind}'
        assert not self.raw.exists() and not self.raw.with_suffix('.failure.json').exists() and not self.out.exists()
        self.out.mkdir()
        self.r = {'experiment': 'METH514 actual256 fixed8 source-head applicability', 'kind': kind,
                  'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': sys.argv,
                  'process_instance': {'pid': self.proc.pid, 'create_time_unix': self.proc.create_time()},
                  'gates': {}, 'new_whole_model_native_compiler_corpus_calls': 0}
        self.timer = threading.Timer(300, self.deadline)
        self.timer.daemon = True
        self.timer.start()

    def deadline(self):
        try:
            self.r.update(fault='hard300s deadline', resource=self.resources())
            write(self.raw.with_suffix('.failure.json'), self.r)
        finally:
            os._exit(124)

    def guard(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        assert self.peak <= 2 << 30 and time.monotonic() - self.start <= 300

    def resources(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        return {'seconds': time.monotonic() - self.start, 'OS_peak_bytes': self.peak,
                'bytes_hashed': self.hashed, 'limits': [300, 2 << 30, 256 << 20]}

    def digest(self, path, offset=0, count=None):
        path = Path(path)
        stat = path.stat()
        size = stat.st_size - offset if count is None else count
        assert 0 <= offset <= offset + size <= stat.st_size
        h = hashlib.sha256()
        with path.open('rb') as stream:
            stream.seek(offset)
            remaining = size
            while remaining:
                data = stream.read(min(remaining, 8 << 20))
                assert data
                h.update(data)
                self.hashed += len(data)
                remaining -= len(data)
                self.guard()
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
        return h.hexdigest()

    def admit(self, binding_sha):
        assert self.digest(BIND) == binding_sha
        self.b = json.loads(BIND.read_bytes())
        assert sys.version == self.b['python'] and str(Path(sys.executable).resolve()) == self.b['executable']
        for row in self.b['catalog']:
            assert self.digest(row['path']) == row['sha256'], row['path']
        for row in self.b['scientific']:
            path = Path(row['path'])
            assert path.read_bytes().replace(b'\r\n', b'\n') == subprocess.check_output(
                ['git', 'show', 'HEAD:' + path.relative_to(ROOT).as_posix()], cwd=ROOT).replace(b'\r\n', b'\n')
        for relative, sha in self.b['preserved'].items():
            assert self.digest(ROOT / relative) == sha
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
        assert all(row[:3] == ' M ' and row[3:] in self.b['preserved'] for row in status)
        assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
        source = self.b['source_head']
        stat = Path(source['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (source['file_bytes'], source['file_mtime_ns'])
        assert self.digest(source['path'], source['offset'], source['bytes']) == source['sha256']
        self.r.update(binding_sha256=binding_sha, source_freeze=self.b['freeze_head'],
                      execution_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        self.r['gates']['fresh_used_metadata_runtime_wires_AND_actual256_F32_head_extent_SHA'] = True
        return self.b

    def numpy(self):
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        import numpy as np
        import threadpoolctl as tp
        for name, version in self.b['packages'].items():
            assert importlib.metadata.version(name) == version
        self.pool = tp.threadpool_limits(1)
        known = {str(Path(row['path']).resolve()).lower() for row in self.b['catalog']}
        for row in tp.threadpool_info():
            assert row['num_threads'] == 1 and str(Path(row['filepath']).resolve()).lower() in known
        np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
        self.r['runtime'] = {'pools': tp.threadpool_info(), 'affinity': self.proc.cpu_affinity(), 'Python': sys.version}
        return np

    def finish(self, value):
        self.r.update(value, ended_compute_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      resource_before_serialization=self.resources())
        write(self.raw, self.r)
        own_size = self.raw.stat().st_size
        peer = DOC / 'meth514_main_result.json'
        combined = own_size + (peer.stat().st_size if self.kind == 'audit' else 0)
        assert combined <= (256 << 20) - (1 << 20)
        source = self.b['source_head']
        stat = Path(source['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (source['file_bytes'], source['file_mtime_ns'])
        terminal = {'result_sha256': self.digest(self.raw), **self.resources(), 'process_instance': self.r['process_instance'],
                    'result_output_bytes': own_size, 'combined_raw_output_bytes': combined,
                    'new_whole_model_native_compiler_corpus_calls': 0}
        write(self.out / 'terminal_resource.json', terminal)
        self.guard()
        self.timer.cancel()
        print(json.dumps({'sha256': terminal['result_sha256'], 'resource': terminal,
                          'gates': self.r['gates'], 'summary': value['summary']}), flush=True)

    def fail(self):
        self.timer.cancel()
        self.r.update(traceback=traceback.format_exc(), resource=self.resources())
        path = self.raw.with_suffix('.failure.json')
        if not path.exists():
            write(path, self.r)
