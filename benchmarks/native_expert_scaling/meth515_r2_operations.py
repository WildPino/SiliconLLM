"""Finite515 artifact/compiler/child admission, timing and terminal receipts."""
import ctypes
from ctypes import wintypes
import datetime
import hashlib
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
BIND = DOC / 'meth515_r1_binding.json'


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


class Memory(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
        ['peak_wset', 'wset', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile']]


def terminal_memory(child):
    query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    query.restype = wintypes.BOOL
    memory = Memory()
    memory.cb = ctypes.sizeof(memory)
    assert query(wintypes.HANDLE(int(child._handle)), ctypes.byref(memory), memory.cb)
    return {'peak_working_set_bytes': int(memory.peak_wset), 'working_set_bytes': int(memory.wset)}


class Context:
    def __init__(self, kind):
        self.start = time.monotonic()
        self.proc = psutil.Process()
        self.proc.cpu_affinity([10])
        self.kind = kind
        self.peak = self.hashed = self.combined_peak = 0
        self.raw = DOC / f'meth515_{kind}_result.json'
        self.out = ROOT / 'results/native_expert_scaling' / f'meth515_{kind}'
        assert not self.raw.exists() and not self.raw.with_suffix('.failure.json').exists() and not self.out.exists()
        self.out.mkdir()
        self.r = {'kind': kind, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': sys.argv,
                  'process_instance': {'pid': self.proc.pid, 'create_time_unix': self.proc.create_time()},
                  'commands': [], 'gates': {}, 'new_original_donor_calls': 0}
        self.timer = threading.Timer(284, self.deadline)
        self.timer.daemon = True
        self.timer.start()

    def deadline(self):
        for v in self.r['commands']:
            if v.get('returncode') is None:
                try:
                    p = psutil.Process(v['process_instance']['pid'])
                    if abs(p.create_time() - v['process_instance']['create_time_unix']) < .002:
                        p.kill()
                except psutil.NoSuchProcess:
                    pass
        try:
            self.r.update(fault='hard284s remaining audit deadline', seconds=time.monotonic() - self.start, bytes_hashed=self.hashed)
            write(self.raw.with_suffix('.failure.json'), self.r)
        finally:
            os._exit(124)

    def guard(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        assert max(self.peak, self.combined_peak) <= 2 << 30 and time.monotonic() - self.start <= 284

    def resources(self):
        self.guard()
        return {'seconds': time.monotonic() - self.start, 'controller_OS_peak_bytes': self.peak,
                'conservative_controller_plus_child_peak_bytes': max(self.peak, self.combined_peak), 'bytes_hashed': self.hashed,
                'limits': [284, 2 << 30, 64 << 20]}

    def digest(self, path):
        p = Path(path)
        s = p.stat()
        h = hashlib.sha256()
        with p.open('rb') as stream:
            while data := stream.read(8 << 20):
                h.update(data)
                self.hashed += len(data)
                self.guard()
        assert (p.stat().st_size, p.stat().st_mtime_ns) == (s.st_size, s.st_mtime_ns)
        return h.hexdigest()

    def admit(self, sha):
        assert self.digest(BIND) == sha
        self.b = json.loads(BIND.read_bytes())
        assert sys.version == self.b['python'] and str(Path(sys.executable).resolve()) == self.b['executable']
        for v in self.b['catalog']:
            assert self.digest(v['path']) == v['sha256'], v['path']
        for v in self.b['scientific']:
            p = Path(v['path'])
            assert p.read_bytes().replace(b'\r\n', b'\n') == subprocess.check_output(['git', 'show', 'HEAD:' + p.relative_to(ROOT).as_posix()], cwd=ROOT).replace(b'\r\n', b'\n')
        for rel, preserved_sha in self.b['preserved'].items():
            assert self.digest(ROOT / rel) == preserved_sha
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
        assert all(v[:3] == ' M ' and v[3:] in self.b['preserved'] for v in status)
        assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
        self.r.update(binding_sha256=sha, source_freeze=self.b['freeze_head'], execution_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        self.r['gates']['all_fresh_used_runtime_source_compiler_reference_AND_full_payload_SHA'] = True
        return self.b

    def run(self, argv, label, env, native=True):
        out, err = self.out / (label + '.stdout'), self.out / (label + '.stderr')
        with out.open('xb') as stdout, err.open('xb') as stderr:
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env)
            p = psutil.Process(child.pid)
            instance = {'pid': child.pid, 'create_time_unix': p.create_time()}
            v = {'label': label, 'argv': argv, 'native': native, 'process_instance': instance, 'returncode': None,
                 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            self.r['commands'].append(v)
            try:
                p.cpu_affinity([0, 2, 4, 6, 8, 10] if native else [10])
                v['affinity_readback'] = p.cpu_affinity()
                while child.poll() is None:
                    self.guard()
                    try:
                        self.combined_peak = max(self.combined_peak, self.proc.memory_info().rss + p.memory_info().peak_wset)
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(.05)
                child.wait()
                v.update(returncode=child.returncode, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), terminal_memory=terminal_memory(child))
                self.combined_peak = max(self.combined_peak, self.peak + v['terminal_memory']['peak_working_set_bytes'])
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        v.update(stdout_sha256=self.digest(out), stderr_sha256=self.digest(err))
        assert child.returncode == 0, v
        self.guard()
        return out

    def finish(self, value):
        self.r.update(value)
        inventory = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': self.digest(p)} for p in sorted(self.out.iterdir()) if p.is_file()]
        assert sum(v['bytes'] for v in inventory) < (64 << 20) - (1 << 20)
        self.r.update(output_inventory=inventory, resource_before_serialization=self.resources(), ended_compute_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(self.raw, self.r)
        terminal = {'result_sha256': self.digest(self.raw), **self.resources(), 'process_instance': self.r['process_instance'],
                    'output_bytes': sum(v['bytes'] for v in inventory) + self.raw.stat().st_size}
        write(self.out / 'terminal_resource.json', terminal)
        self.guard()
        self.timer.cancel()
        print(json.dumps({'result_sha256': terminal['result_sha256'], 'gates': self.r['gates'], 'summary': value['summary'], 'resource': terminal}), flush=True)

    def fail(self):
        self.timer.cancel()
        self.r.update(traceback=traceback.format_exc(), seconds=time.monotonic() - self.start, bytes_hashed=self.hashed)
        p = self.raw.with_suffix('.failure.json')
        if not p.exists():
            write(p, self.r)
