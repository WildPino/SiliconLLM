"""521 immutable admission, owned child observations and bounded receipts."""
import ctypes
from ctypes import wintypes
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
BIND = DOC / 'meth521_binding.json'


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
        own = {self.proc.pid, *(p.pid for p in self.proc.parents())}
        foreign = []
        for p in psutil.process_iter(['name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): continue
            if name.startswith(('python', 'clang')) or (name.startswith('meth') and name.endswith('.exe')):
                foreign.append({'pid': p.pid, 'name': name})
        assert not foreign, ('foreign_scientific_processes', foreign)
        self.kind, self.seconds, self.memory = kind, seconds, memory
        self.peak = self.hashed = 0
        self.child = None
        self.child_peak = 0
        self.descendants = {}
        self.last_output_check = -1.
        self.raw = BIND if kind == 'binding' else DOC / f'meth521_{kind}_result.json'
        assert not self.raw.exists() and not self.raw.with_suffix('.failure.json').exists()
        self.r = {'kind': kind, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'process_instance': {'pid': self.proc.pid, 'create_time_unix': self.proc.create_time()},
                  'argv': sys.argv, 'gates': {}, 'commands': [], 'new_model_native_capture_calls': 0,
                  'scope': 'New source-function information and fixed one-bank transfer; no whole-model quality/rate.'}
        self.timer = threading.Timer(seconds, self.deadline)
        self.timer.daemon = True
        self.timer.start()

    def deadline(self):
        try:
            if self.child is not None and self.child.poll() is None:
                self.child.kill()
                self.child.wait(timeout=10)
            self.r.update(fault='hard deadline', resource=self.resources())
            if not self.raw.with_suffix('.failure.json').exists():
                write(self.raw.with_suffix('.failure.json'), self.r)
        finally:
            os._exit(124)

    def guard(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        assert self.peak + self.child_peak <= self.memory and time.monotonic() - self.start <= self.seconds
        if time.monotonic() - self.last_output_check >= 1:
            self.last_output_check = time.monotonic()
            assert output_bytes() <= 2 << 30

    def resources(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        return {'seconds': time.monotonic() - self.start, 'OS_peak_bytes': self.peak + self.child_peak,
                'parent_OS_peak_bytes': self.peak, 'conservative_observed_child_group_peak_bytes': self.child_peak,
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
        if self.child is not None and self.child.poll() is None:
            self.child.kill()
            self.child.wait(timeout=10)
        self.r.update(traceback=traceback.format_exc(), resource=self.resources())
        if hasattr(self, 'out'):
            self.r['partial_outputs'] = [{'path': str(p), 'bytes': p.stat().st_size}
                                        for p in sorted(self.out.iterdir()) if p.is_file()]
        p = self.raw.with_suffix('.failure.json')
        if not p.exists():
            write(p, self.r)


class Context(Meter):
    def __init__(self, kind, binding_sha):
        super().__init__(kind, 900, 2 << 30)
        try:
            self.admit(kind, binding_sha)
        except BaseException:
            self.fail()
            raise

    def admit(self, kind, binding_sha):
        self.out = ROOT / 'results/native_expert_scaling' / ('meth521_transfer' if kind == 'main' else 'meth521_audit')
        assert not self.out.exists()
        self.out.mkdir()
        assert self.digest(BIND) == binding_sha
        self.b = json.loads(BIND.read_bytes())
        assert sys.version == self.b['python'] and str(Path(sys.executable).resolve()) == self.b['executable']
        for v in self.b['catalog']:
            assert self.digest(v['path']) == v['sha256'], v['path']
        self.payload_stat()
        for v in self.b['source_extents']:
            assert self.digest(v['path'], v['offset'], v['bytes']) == v['sha256'], v['name']
        for rel, preserved_sha in self.b['preserved'].items():
            assert self.digest(ROOT / rel) == preserved_sha
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
        assert all(s[:3] == ' M ' and s[3:] in self.b['preserved'] for s in status), status
        assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
        self.r.update(binding_sha256=binding_sha, source_freeze=self.b['freeze_head'],
                      execution_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        self.r['gates']['fresh_ALL_inputs_labels_plan_fixed_bank_source512_extents_runtime_compiler_science_SHA'] = True

    def payload_stat(self):
        b = self.b['payload']
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

    def run(self, argv, label, expected=0):
        record = {'label': label, 'argv': list(map(str, argv)), 'started_utc': utc(),
                  'observer_scope': 'OS peak by retained handle; module/descendant sampling, not complete loader history'}
        self.r['commands'].append(record)
        with (self.out / (label + '.stdout')).open('xb') as so, (self.out / (label + '.stderr')).open('xb') as se:
            child = subprocess.Popen(record['argv'], cwd=ROOT, stdout=so, stderr=se)
            self.child = child
            times = [wintypes.FILETIME() for _ in range(4)]
            get_times = ctypes.WinDLL('kernel32', use_last_error=True).GetProcessTimes
            get_times.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            get_times.restype = wintypes.BOOL
            assert get_times(wintypes.HANDLE(int(child._handle)), *[ctypes.byref(v) for v in times])
            creation = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) / 1e7 - 11644473600
            record['process_instance'] = {'pid': child.pid, 'create_time_unix': creation}
            get_mem = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
            get_mem.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
            get_mem.restype = wintypes.BOOL
            descendants = {}; modules = set(); races = 0; next_sample = 0.; own_peak = 0; group_peak = 0
            while True:
                v = Memory(); v.cb = ctypes.sizeof(v)
                assert get_mem(wintypes.HANDLE(int(child._handle)), ctypes.byref(v), v.cb)
                own_peak = max(own_peak, int(v.PeakWorkingSetSize))
                if child.poll() is None and time.monotonic() >= next_sample:
                    try:
                        proc = psutil.Process(child.pid)
                        modules.update(v.path for v in proc.memory_maps() if v.path.lower().endswith(('.dll', '.exe')))
                        for p in proc.children(recursive=True):
                            try:
                                key = (p.pid, p.create_time()); m = p.memory_info()
                                previous = descendants.get(key, {}).get('OS_peak_bytes', 0)
                                descendants[key] = {'pid': key[0], 'create_time_unix': key[1],
                                                    'OS_peak_bytes': max(previous, m.peak_wset), 'executable': p.exe()}
                                modules.update(v.path for v in p.memory_maps() if v.path.lower().endswith(('.dll', '.exe')))
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                races += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        races += 1
                    next_sample = time.monotonic() + 1.
                group_peak = max(group_peak, own_peak + sum(v['OS_peak_bytes'] for v in descendants.values()))
                self.child_peak = max(self.child_peak, group_peak)
                self.guard()
                if child.poll() is not None:
                    break
                time.sleep(.05)
            self.child = None
        record.update(returncode=child.returncode, ended_utc=utc(), OS_peak_bytes=own_peak,
                      conservative_child_group_peak_bytes=group_peak, observed_modules=sorted(modules),
                      descendant_process_peaks=list(descendants.values()), observer_exit_races=races)
        assert child.returncode == expected, (label, child.returncode)
        print(json.dumps({'command_terminal': label, 'exit': child.returncode,
                          'seconds': self.resources()['seconds']}), flush=True)
        return (self.out / (label + '.stdout')).read_text().splitlines()

    def finish(self, value):
        self.r.update(value)
        self.payload_stat()
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
                          'decision': value['decision'], 'resource': terminal}), flush=True)


class Memory(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                      'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                      'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def output_bytes():
    paths = []
    for name in ('meth521_transfer', 'meth521_audit'):
        folder = ROOT / 'results/native_expert_scaling' / name
        if folder.exists():
            paths.extend(p for p in folder.iterdir() if p.is_file())
    for pattern in ('meth521*.json', 'ADMISSION_521*.json', 'RETENTION_521*.json'):
        paths.extend(DOC.glob(pattern))
    paths.extend((ROOT / 'results/native_expert_scaling').glob('meth521*_resource.json'))
    return sum(p.stat().st_size for p in paths)


def wire(np, path, magic, width, reserved, dtype, count):
    dtype = np.dtype(dtype)
    p = Path(path)
    assert dtype.itemsize == width and p.stat().st_size == 24 + count * width
    with p.open('rb') as f:
        assert f.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
    return np.memmap(p, mode='r', offset=24, dtype=dtype, shape=(count,))
