"""Bound source screen with a held Windows worker handle through actual exit."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/native_expert_scaling'))
from chatbot_falcon_usability import sha, write
import psutil


class Memory(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
       'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
       'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def main(args):
    start = time.monotonic()
    proc = psutil.Process()
    proc.cpu_affinity([11])
    assert sha(args.binding) == args.binding_sha
    b = json.loads(args.binding.read_bytes())
    assert b['schema'] in ('FALCON_USABILITY_BINDING_V1', 'FALCON_SCAN_BRIDGE_BINDING_V1','HYBRID_USABILITY_BINDING_V1','HYBRID_PILOT_BINDING_V1','HYBRID_PILOT_AUDIT_BINDING_V1','HYBRID_NATIVE_BINDING_V1','HYBRID_NATIVE_AUDIT_BINDING_V1','HYBRID_COMMON_BANK_BINDING_V1','HYBRID_COMMON_CORE_BINDING_V1','HYBRID_TRANSFER_CAPTURE_BINDING_V1','HYBRID_TRANSFER_ADOPTION_BINDING_V1','HYBRID_RECOVERY_BINDING_V1','HYBRID_RECOVERY_AUDIT_BINDING_V1','HYBRID_STATE_EVALUATION_BINDING_V1','HYBRID_GROUP_SUM_BINDING_V1','HYBRID_SHARED_PRIVATE_INIT_BINDING_V1')
    OS_cap=b.get('limits',{}).get('OS_bytes',4<<30)
    seconds_cap=b.get('limits',{}).get('seconds',600)
    output_cap=b.get('limits',{}).get('output_bytes',256<<20)
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    assert not args.directory.exists()
    log = args.out.with_suffix('.worker.log')
    terminal = args.out.with_suffix('.terminal.json')
    assert not any(p.exists() for p in (args.out, log, terminal, args.out.with_suffix('.launcher_failure.json')))
    record = dict(freeze=args.freeze, binding_sha256=args.binding_sha, launcher_pid=proc.pid)
    p = None
    peak = 0
    handle = None
    getmem = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    getmem.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    getmem.restype = wintypes.BOOL
    def memory():
        nonlocal peak
        if handle is not None:
            m = Memory()
            m.cb = ctypes.sizeof(m)
            assert getmem(handle, ctypes.byref(m), m.cb)
            peak = max(peak, m.PeakWorkingSetSize)
    def check_inputs():
        for item in b['inputs']:
            assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256'], item['path']
    def guard():
        assert time.monotonic() - start <= seconds_cap, 'family deadline'
        assert peak + proc.memory_info().peak_wset <= OS_cap, 'family OS cap'
        assert log.stat().st_size <= 4 << 20, 'log cap'
        if args.directory.exists():
            assert sum(v.stat().st_size for v in args.directory.iterdir() if v.is_file()) <= output_cap, 'output cap'
    try:
        own = {proc.pid, *(v.pid for v in proc.parents())}
        for other in psutil.process_iter(['name', 'cmdline']):
            name = (other.info['name'] or '').lower()
            argv = other.info['cmdline'] or []
            if other.pid in own:
                continue
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                continue
            assert not name.startswith(('python', 'clang', 'engine')), ('overlap', other.pid, name)
        check_inputs()
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='6',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='6', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        argv = [b['python'], '-I', '-S', '-B', '-X', 'utf8', b['worker_path'], '--binding', str(args.binding.resolve()),
                  '--binding-sha', args.binding_sha, '--freeze', args.freeze,
                  '--directory', str(args.directory.resolve()), '--out', str(args.out.resolve())]
        record['command'] = argv
        with log.open('xb') as stream:
            p = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, creationflags=8)
            handle = wintypes.HANDLE(int(p._handle))
            record['worker_pid'] = p.pid
            position = 0
            while p.poll() is None:
                memory()
                guard()
                try:
                    children=psutil.Process(p.pid).children(recursive=True)
                    for child in children:
                        try:
                            name=child.name().lower()
                        except psutil.NoSuchProcess:
                            continue
                        assert name in b.get('allowed_worker_children',[]), ('unexpected worker descendant',name)
                        if name=='conhost.exe':
                            assert Path(child.exe().removeprefix('\\\\?\\')).resolve()==Path(b['allowed_system_child_path']).resolve()
                except psutil.NoSuchProcess:
                    pass
                with log.open('rb') as f:
                    f.seek(position)
                    chunk = f.read()
                    boundary = chunk.rfind(b'\n') + 1
                    if boundary:
                        print(chunk[:boundary].decode('utf8', errors='replace'), end='', flush=True)
                        position += boundary
                time.sleep(.1)
        memory()
        record.update(exit_code=p.returncode, worker_OS_peak_through_exit=peak,
                     launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        guard()
        assert p.returncode == 0, log.read_text(errors='replace')
        check_inputs()
        result = json.loads(args.out.read_bytes())
        if b['schema'] == 'FALCON_USABILITY_BINDING_V1':
            assert result['schema'] == 'FALCON_USABILITY_RESULT_V1' and result['total'] == 16
        elif b['schema'] == 'HYBRID_USABILITY_BINDING_V1':
            assert result['schema'] == 'HYBRID_USABILITY_RESULT_V1' and result['total'] == 16
        elif b['schema'] == 'HYBRID_PILOT_BINDING_V1':
            assert result['schema'] == 'HYBRID_PILOT_RESULT_V1' and result['complete_learner_available']
        elif b['schema'] == 'HYBRID_PILOT_AUDIT_BINDING_V1':
            assert result['schema'] == 'HYBRID_PILOT_AUDIT_RESULT_V1'
        elif b['schema'] == 'HYBRID_NATIVE_BINDING_V1':
            assert result['schema']==('HYBRID_PACKED_EXPORT_RESULT_V1' if b['phase']=='export' else 'HYBRID_NATIVE_RESULT_V1')
        elif b['schema'] == 'HYBRID_NATIVE_AUDIT_BINDING_V1':
            assert result['schema']=='HYBRID_NATIVE_AUDIT_RESULT_V1'
        elif b['schema'] == 'HYBRID_COMMON_BANK_BINDING_V1':
            assert result['schema']=='HYBRID_COMMON_BANK_RESULT_V1'
        elif b['schema'] == 'HYBRID_COMMON_CORE_BINDING_V1':
            assert result['schema']=='HYBRID_COMMON_CORE_RESULT_V1' and result['core_rows']==3132 and result['completed_modules_cases']==72
        elif b['schema'] == 'HYBRID_TRANSFER_CAPTURE_BINDING_V1':
            assert result['schema']=='HYBRID_TRANSFER_CAPTURE_RESULT_V1' and result['cases']==160 and result['reserved_queries']==0
        elif b['schema'] == 'HYBRID_TRANSFER_ADOPTION_BINDING_V1':
            assert result['schema']=='HYBRID_TRANSFER_ADOPTION_RESULT_V1' and result['cases']==160
        elif b['schema'] == 'HYBRID_RECOVERY_BINDING_V1':
            assert result['schema']=='HYBRID_RECOVERY_RESULT_V1' and result['new_updates']==512
        elif b['schema'] == 'HYBRID_RECOVERY_AUDIT_BINDING_V1':
            assert result['schema']=='HYBRID_RECOVERY_AUDIT_RESULT_V1' and 0<=result['checkpoint_boundary']<=512
        elif b['schema'] == 'HYBRID_STATE_EVALUATION_BINDING_V1':
            assert result['schema']=='HYBRID_STATE_EVALUATION_RESULT_V1' and result['student_cases']==160 and result['boundary']==286
        elif b['schema'] == 'HYBRID_GROUP_SUM_BINDING_V1':
            assert result['schema']=='HYBRID_GROUP_SUM_RESULT_V1' and result['completed_layers']==12 and result['operand_rows']==3132
        elif b['schema'] == 'HYBRID_SHARED_PRIVATE_INIT_BINDING_V1':
            assert result['schema']=='HYBRID_SHARED_PRIVATE_INIT_RESULT_V1' and result['total_parameters']==259669760 and result['parameter_tensors']==283
        else:
            assert result['schema'] == 'FALCON_SCAN_CAPTURE_RESULT_V1' and result['packet_count'] == 12
        assert result['process_instance']['pid'] == p.pid
        record.update(elapsed_seconds=time.monotonic()-start, result_sha256=sha(args.out),
                decision=result['decision'], resource_gates=True,
                binding_scope=b['runtime_binding_scope'], output_files=[dict(path=str(v.resolve()),
                  bytes=v.stat().st_size, sha256=sha(v)) for v in sorted(args.directory.iterdir()) if v.is_file()])
        write(terminal, record)
        print(json.dumps(dict(terminal=str(terminal), decision=result['decision'], exit_code=p.returncode, OS_peak=peak)), flush=True)
    except BaseException as error:
        if p is not None:
            if p.poll() is None:
                p.kill()
                p.wait()
            memory()
            record.update(exit_code=p.returncode, worker_OS_peak_through_exit=peak)
        record.update(fault=repr(error), elapsed_seconds=time.monotonic()-start)
        write(args.out.with_suffix('.launcher_failure.json'), record)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--binding-sha', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
