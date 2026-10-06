"""Small retained-data admission and resource receipts; no model or compiler calls."""
import ctypes
import datetime
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
BIND = DOC / 'meth507_binding.json'

def write(path, value):
    # Serialize before creating the exclusive destination: never leave an empty RAW.
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf8')
    with Path(path).open('xb') as stream:
        stream.write(payload)

class Context:
    def __init__(self, kind):
        assert kind in ('main', 'audit')
        self.start = time.monotonic()
        self.proc = psutil.Process()
        self.proc.cpu_affinity([10])
        self.kind = kind
        self.out = ROOT / 'results/native_expert_scaling' / ('meth507_' + kind)
        self.out.mkdir(exist_ok=False)
        self.destination = DOC / ('meth507_' + kind + '_result.json')
        assert not self.destination.exists()
        self.peak = self.hashed = 0
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.progress = (self.out / 'progress.jsonl').open('x', encoding='utf8')
        self.fatal = (self.out / 'fatal.log').open('x', encoding='utf8')
        faulthandler.enable(self.fatal)
        self.r = {'experiment': 'METH507 retained common-prefix diagnosis',
                  'kind': kind, 'argv': sys.argv, 'process_instance': {
                      'pid': self.proc.pid, 'create_time_unix': self.proc.create_time()},
                  'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'native_model_compiler_calls': 0, 'affinity': [10], 'gates': {}}
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def guard(self):
        info = self.proc.memory_info()
        self.peak = max(self.peak, info.rss, getattr(info, 'peak_wset', 0))
        assert self.peak <= 2 << 30, ('OS_peak', self.peak)
        assert time.monotonic() - self.start <= 180, '180s_deadline'
        size = sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        if self.destination.exists():
            size += self.destination.stat().st_size
        assert size <= 32 << 20, ('output_bytes', size)

    def watch(self):
        while not self.closed.wait(1):
            try:
                self.guard()
            except BaseException:
                self.fail()
                os._exit(124)

    def digest(self, path):
        path = Path(path)
        before = path.stat()
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            while data := stream.read(8 << 20):
                sha.update(data)
                self.hashed += len(data)
                self.guard()
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        return sha.hexdigest()

    def exact(self, row):
        assert Path(row['path']).stat().st_size == row['bytes']
        assert self.digest(row['path']) == row['sha256'], row['path']

    def admit(self, binding_sha):
        assert self.digest(BIND) == binding_sha
        b = json.loads(BIND.read_bytes())
        assert sys.version == b['python'] and str(Path(sys.executable).resolve()) == b['executable']
        for name, version in b['packages'].items():
            assert importlib.metadata.version(name) == version
        for row in b['catalog']:
            self.exact(row)
        self.catalog = {str(Path(v['path']).resolve()).lower() for v in b['catalog']}
        for row in b['scientific']:
            path = Path(row['path'])
            assert path.read_bytes().replace(b'\r\n', b'\n') == subprocess.check_output(
                ['git', 'show', 'HEAD:' + path.relative_to(ROOT).as_posix()], cwd=ROOT).replace(b'\r\n', b'\n')
        for rel, sha in b['preserved'].items():
            assert self.digest(ROOT / rel) == sha
        assert self.digest(ROOT / 'benchmarks/phase60/engine.c') == b['engine_sha256']
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
        assert all(line[:3] == ' M ' and line[3:] in b['preserved'] for line in status)
        assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
        self.r.update(binding_sha256=binding_sha, source_freeze=b['freeze_head'],
                      execution_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        self.r['gates']['frozen_selected_data_and_small_runtime_before_numpy'] = True
        return b

    def modules(self):
        assert not any(n == 'torch' or n.startswith(('torch.', 'pyarrow')) for n in sys.modules)
        python_files = set()
        for module in tuple(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).is_file():
                path = str(Path(filename).resolve()).lower()
                assert path in self.catalog, ('unbound_Python_module', path)
                python_files.add(path)
        windows = str(Path(os.environ['WINDIR']).resolve()).lower() + '\\'
        images = []
        for mapping in self.proc.memory_maps(grouped=True):
            path = Path(mapping.path)
            if path.suffix.lower() not in ('.dll', '.pyd', '.exe') or not path.is_file():
                continue
            resolved = str(path.resolve()).lower()
            assert resolved in self.catalog or resolved.startswith(windows), ('unbound_loaded_image', resolved)
            images.append({'path': str(path), 'bytes': path.stat().st_size, 'system_windows_image': resolved.startswith(windows)})
        self.r['loaded_python_files'] = sorted(python_files)
        self.r['loaded_images'] = images
        self.r['gates']['actual_numpy_only_modules_and_images'] = True

    def log(self, **value):
        self.progress.write(json.dumps({'seconds': time.monotonic()-self.start, **value})+'\n')
        self.progress.flush()

    def finish(self, value):
        self.modules()
        self.guard()
        self.progress.flush()
        self.fatal.flush()
        assert (self.out/'fatal.log').stat().st_size == 0
        self.r['gates']['zero_native_model_calls_and_empty_fatal_log'] = True
        self.r.update(value)
        self.r['resource'] = {'seconds': time.monotonic()-self.start, 'OS_peak_bytes': self.peak,
                              'bytes_hashed': self.hashed, 'limits': {'seconds':180,'OS_peak_bytes':2<<30,'output_bytes':32<<20}}
        self.r['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(self.destination, self.r)
        self.guard()
        print(json.dumps({'result':str(self.destination),'sha256':self.digest(self.destination),
                          'gates':self.r['gates'],'resource':self.r['resource'],
                          'summary':self.r.get('summary')}), flush=True)

    def fail(self):
        with self.lock:
            path = self.destination.with_suffix('.failure.json')
            if not path.exists():
                write(path, {**self.r,'traceback':traceback.format_exc(),
                             'seconds':time.monotonic()-self.start,'OS_peak_bytes':self.peak,'bytes_hashed':self.hashed})

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)
        self.progress.close()
        self.fatal.close()

