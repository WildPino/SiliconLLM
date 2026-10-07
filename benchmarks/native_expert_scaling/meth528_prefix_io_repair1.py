"""Prefix certificate provenance/resource/wire apparatus; no energy/proof math."""
import ctypes
import datetime
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from meth528_operations import Meter, ROOT, DOC, BIND, write, output_bytes

MANIFEST = DOC / 'meth528_prefix_manifest_repair1_20261007.json'
OLD = ROOT / 'results/native_expert_scaling/meth528_atoms'


class PrefixContext(Meter):
    def __init__(self, kind, manifest_sha, freeze):
        super().__init__(kind, 180 if kind == 'prefix_bound_repair1' else 120, 512 << 20)
        try:
            self.out = ROOT / 'results/native_expert_scaling' / ('meth528_' + kind)
            assert not self.out.exists()
            self.out.mkdir()
            assert self.digest(MANIFEST) == manifest_sha
            self.manifest = json.loads(MANIFEST.read_bytes())
            assert self.digest(BIND) == self.manifest['original_binding_sha256']
            self.b = json.loads(BIND.read_bytes())
            assert sys.version == self.b['python'] and str(Path(sys.executable).resolve()) == self.b['executable']
            assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == freeze
            self.assert_foreign()
            for v in self.manifest['files']:
                assert Path(v['path']).stat().st_size == v['bytes'] and self.digest(v['path']) == v['sha256'], v['path']
            self.extents()
            assert not any(Path(self.manifest['empty_cache']).iterdir())
            self.r.update(source_freeze=freeze, manifest_sha256=manifest_sha,
                          original_binding_sha256=self.manifest['original_binding_sha256'],
                          original528_resource_gate=False, prefix_parents=list(range(1, 13)),
                          excluded_frontier_parent=13, source_response_function_calls=0,
                          scope='Necessary original fixed full-domain rejection from independently verified original prefix; no full528 pass, source response replay or chatbot qualification.')
            self.r['gates']['fresh_frozen_used_inputs_runtime_compiler_partial_and_foreign_SHA'] = True
        except BaseException:
            self.fail()
            raise

    def guard(self):
        self.peak = max(self.peak, self.proc.memory_info().peak_wset)
        assert self.peak + self.compiler_peak <= self.memory and time.monotonic() - self.start <= self.seconds
        if time.monotonic() - self.last_output_check >= 1:
            self.last_output_check = time.monotonic()
            extra = 0
            for name in ('meth528_prefix_bound', 'meth528_prefix_energy_audit', 'meth528_prefix_bound_repair1', 'meth528_prefix_energy_audit_repair1'):
                folder = ROOT / 'results/native_expert_scaling' / name
                if folder.exists():
                    extra += sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
            assert extra <= 465 << 20 and output_bytes() + extra <= 2 << 30

    def assert_foreign(self):
        for rel, sha in self.b['preserved'].items():
            assert self.digest(ROOT / rel) == sha
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines()
        assert all(s[:3] == ' M ' and s[3:] in self.b['preserved'] for s in status), status
        assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()

    def extents(self):
        stat = Path(self.b['payload']['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (self.b['payload_stat']['bytes'], self.b['payload_stat']['mtime_ns'])
        for v in self.manifest['source_extents']:
            assert self.digest(v['path'], v['offset'], v['bytes']) == v['sha256']

    def data(self, key):
        assert key in self.manifest['data_keys']
        return self.b['data'][key]['path']

    def numpy(self):
        import numpy as np
        import threadpoolctl as tp
        for key, version in self.b['packages'].items():
            assert importlib.metadata.version(key) == version
        self.pool = tp.threadpool_limits(1)
        known = {str(Path(v['path']).resolve()) for v in self.manifest['files']}
        pools = tp.threadpool_info()
        assert all(v['num_threads'] == 1 and str(Path(v['filepath']).resolve()) in known for v in pools)
        np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
        self.r['runtime'] = dict(pools=pools, affinity=self.proc.cpu_affinity(), Python=sys.version)
        return np

    def finish(self, value):
        self.r.update(value)
        self.extents()
        self.assert_foreign()
        assert not any(Path(self.manifest['empty_cache']).iterdir())
        inventory = [dict(path=str(p), bytes=p.stat().st_size, sha256=self.digest(p)) for p in sorted(self.out.iterdir()) if p.is_file()]
        self.r.update(output_inventory=inventory, ended_compute_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), resource_before_serialization=self.resources())
        write(self.raw, self.r)
        terminal = dict(result_sha256=self.digest(self.raw), process_instance=self.r['process_instance'], ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **self.resources())
        write(self.out / 'terminal_resource.json', terminal)
        self.guard()
        self.timer.cancel()
        print(json.dumps(dict(result_sha256=terminal['result_sha256'], decision=value['decision'], gates=self.r['gates'], resource=terminal)), flush=True)


def build(ctx, source, stem):
    compiler = ctx.b['compiler']
    obj, dll = ctx.out / (stem + '.o'), ctx.out / (stem + '.dll')
    base = [compiler['path'], '--sysroot=' + compiler['sysroot'], '-target', compiler['target'], '-fintegrated-cc1', '-fuse-ld=lld', '--rtlib=compiler-rt', '--unwindlib=libunwind']
    cc_plan = ctx.compile([*base, '-###', '-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math', '-c', str(source), '-o', str(obj)], stem + '_frontend_plan')
    cc_commands = [shlex.split(line.strip()) for line in cc_plan.read_text(encoding='utf8').splitlines() if line.strip().startswith('"')]
    assert len(cc_commands) == 1 and Path(cc_commands[0][0]).resolve() == Path(compiler['path']).resolve()
    assert '-cc1' in cc_commands[0] and str(obj) in cc_commands[0] and str(source) in cc_commands[0]
    ctx.compile(cc_commands[0], stem + '_direct_cc1')
    plan = ctx.compile([*base, '-###', '-shared', str(obj), '-o', str(dll), '-lm'], stem + '_link_plan')
    commands = [shlex.split(line.strip()) for line in plan.read_text(encoding='utf8').splitlines() if line.strip().startswith('"')]
    assert len(commands) == 1 and Path(commands[0][0]).resolve() == Path(compiler['path']).parent / 'ld.lld.exe'
    assert str(obj) in commands[0] and str(dll) in commands[0]
    ctx.compile(commands[0], stem + '_linker')
    return dll, ctx.digest(dll)


def pointer(a):
    return None if a is None else ctypes.c_void_p(a.ctypes.data)


def save(np, folder, name, value):
    with (folder / (name + '.npy')).open('xb') as f:
        np.save(f, value, allow_pickle=False)
