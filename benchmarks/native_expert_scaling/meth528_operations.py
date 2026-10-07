"""528 immutable source/runtime apparatus; variable source-atom comparison."""
import datetime
import ctypes
from ctypes import wintypes
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
BIND = DOC / 'meth528_binding.json'


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
        self.compiler_peak = 0
        self.compiler_owned = {}
        self.last_output_check = -1.
        self.raw = BIND if kind == 'binding' else DOC / f'meth528_{kind}_result.json'
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
            self.stop_compiler()
            if hasattr(self, 'out'):
                self.r['partial_outputs'] = [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(self.out.iterdir()) if p.is_file()]
            self.r.update(fault='hard deadline', resource=self.resources())
            if not self.raw.with_suffix('.failure.json').exists():
                write(self.raw.with_suffix('.failure.json'), self.r)
        finally:
            os._exit(124)

    def guard(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        assert self.peak + self.compiler_peak <= self.memory and time.monotonic() - self.start <= self.seconds
        if time.monotonic() - self.last_output_check >= 1:
            self.last_output_check = time.monotonic()
            assert output_bytes() <= 2 << 30

    def resources(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        return {'seconds': time.monotonic() - self.start, 'OS_peak_bytes': self.peak,
                'compiler_family_peak_bytes': self.compiler_peak, 'parent_plus_compiler_peak_bound': self.peak + self.compiler_peak,
                'bytes_hashed': self.hashed, 'limits': [self.seconds, self.memory]}

    def compile(self, argv, label):
        log = self.out / (label + '.log')
        class Memory(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        getmem = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        getmem.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]; getmem.restype = wintypes.BOOL
        with log.open('xb') as f:
            p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000)
            gettimes = ctypes.WinDLL('kernel32', use_last_error=True).GetProcessTimes
            gettimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4; gettimes.restype = wintypes.BOOL
            stamps = [wintypes.FILETIME() for _ in range(4)]
            assert gettimes(wintypes.HANDLE(int(p._handle)), *[ctypes.byref(s) for s in stamps])
            ticks = (stamps[0].dwHighDateTime << 32) | stamps[0].dwLowDateTime
            creation = (ticks - 116444736000000000)/10000000.; self.compiler_owned[p.pid] = creation
            peak = 0
            while p.poll() is None:
                mem = Memory(); mem.cb = ctypes.sizeof(mem); assert getmem(wintypes.HANDLE(int(p._handle)), ctypes.byref(mem), mem.cb)
                peak = max(peak, mem.PeakWorkingSetSize); self.compiler_peak = max(self.compiler_peak, peak)
                try: assert not psutil.Process(p.pid).children(recursive=True), 'Use direct integrated frontend/LLD, no subprocess tree'
                except psutil.NoSuchProcess: pass
                self.guard(); time.sleep(.01)
        # Retained process handle exposes the actual OS peak AFTER termination.
        mem = Memory(); mem.cb = ctypes.sizeof(mem); assert getmem(wintypes.HANDLE(int(p._handle)), ctypes.byref(mem), mem.cb)
        peak = max(peak, mem.PeakWorkingSetSize); self.compiler_peak = max(self.compiler_peak, peak)
        instance = dict(pid=p.pid, create_time_unix=creation, OS_peak_bytes=peak, label=label)
        compiler = self.r.setdefault('compiler', dict(commands=[], observed_instances=[]))
        compiler['commands'].append(dict(argv=argv, exit_code=p.returncode, instance=instance, log=str(log)))
        compiler['observed_instances'].append(instance); compiler['conservative_family_peak_bytes'] = self.compiler_peak
        assert p.returncode == 0, log.read_text(encoding='utf8', errors='replace')
        self.guard(); return log

    def stop_compiler(self):
        for pid, creation in self.compiler_owned.items():
            try:
                p = psutil.Process(pid)
                if abs(p.create_time() - creation) < .002: p.kill()
            except psutil.Error: pass

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
        self.stop_compiler()
        self.r.update(traceback=traceback.format_exc(), resource=self.resources())
        if hasattr(self, 'out'):
            self.r['partial_outputs'] = [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(self.out.iterdir()) if p.is_file()]
        p = self.raw.with_suffix('.failure.json')
        if not p.exists():
            write(p, self.r)


class Context(Meter):
    def __init__(self, kind, binding_sha):
        super().__init__(kind, 240 if kind == 'main' else 300, 512 << 20)
        try:
            self.admit(kind, binding_sha)
        except BaseException:
            self.fail()
            raise

    def admit(self, kind, binding_sha):
        self.out = ROOT / 'results/native_expert_scaling' / ('meth528_atoms' if kind == 'main' else 'meth528_audit')
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
        self.r['gates']['fresh_used_input_anchors_source_extents_runtime_and_frozen_science_SHA'] = True

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
    for name in ('meth528_atoms', 'meth528_audit'):
        folder = ROOT / 'results/native_expert_scaling' / name
        if folder.exists(): paths.extend(p for p in folder.iterdir() if p.is_file())
    for pattern in ('meth528*.json', 'RETENTION_528*.json', 'ADMISSION_528*.json'):
        paths.extend(DOC.glob(pattern))
    paths.extend((ROOT / 'results/native_expert_scaling').glob('meth528*_resource.json'))
    return sum(p.stat().st_size for p in paths)
