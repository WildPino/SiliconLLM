"""Bounded acquisition/audit launcher with a held Windows process handle."""
import argparse
import ctypes
from ctypes import wintypes
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_interaction_launch import Memory,sha,tree,write_once,FOREIGN
import psutil


def main(args):
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([11])
    log=args.out.with_suffix('.worker.log');terminal=args.out.with_suffix('.terminal.json')
    assert not args.directory.exists()
    assert not any(p.exists() for p in (args.out,terminal,log,args.out.with_suffix('.failure.json'),args.out.with_suffix('.launcher_failure.json')))
    r=dict(scope='DIRECTIONAL_INFORMATION_THROUGH_EXIT_NO_OLD_RESPONSE_REPLAY',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),launcher_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),gates={})
    p=None;worker_peak=0;handle=None;limits=None
    getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];getmem.restype=wintypes.BOOL
    def memory():
        nonlocal worker_peak
        if handle is not None:
            value=Memory();value.cb=ctypes.sizeof(value);assert getmem(handle,ctypes.byref(value),value.cb)
            worker_peak=max(worker_peak,value.PeakWorkingSetSize)
    def guard():
        assert time.monotonic()-start<=limits['family_seconds'],'family deadline'
        assert proc.memory_info().peak_wset+worker_peak<=limits['OS_bytes'],'conservative family OS peaks'
        assert not log.exists() or log.stat().st_size<=limits['log_bytes'],'worker log cap'
        if args.directory.exists():
            assert sum(v.stat().st_size for v in args.directory.iterdir() if v.is_file())<=limits['output_bytes'],'output cap'
    def preserved():
        assert all(sha(ROOT/rel)==want for rel,want in FOREIGN.items())
        assert not any(v.is_file() for v in (ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    def binding_check(binding):
        for item in binding['inputs']:
            assert str(Path(item['path']).resolve())==item['resolved_path']
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path'];guard()
        for root in binding['capture_runtime_roots']:
            assert tree(root['path'])==(root['tree_sha256'],root['files'],root['bytes']),root['path']
            assert (ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'/root['view_name']).samefile(root['path']);guard()
        for view in binding['interaction_view']:
            assert (ROOT/'results/native_expert_scaling/chatbot_interaction_runtime/site'/view['view_name']).samefile(view['path'])
    try:
        assert sys.version_info[:3]==(3,12,10) and psutil.__version__=='7.2.2'
        assert sha(args.binding)==args.binding_sha
        binding=json.loads(args.binding.read_bytes());assert binding['schema']=='QWEN_DIRECTIONAL_BINDING_V1'
        job=binding['job'];limits=job['limits'];r['job']=job
        assert Path(sys.executable).resolve()==Path(binding['python']).resolve()
        own={proc.pid,*(v.pid for v in proc.parents())}
        for other in psutil.process_iter(['name','cmdline']):
            name=(other.info['name'] or '').lower();argv=other.info['cmdline'] or []
            if other.pid in own:continue
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
            assert not name.startswith(('python','clang','meth')),('overlap',other.pid,name)
        preserved();binding_check(binding);r['gates']['inputs_runtime_and_view_before']=True
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',
            TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        argv=[binding['python'],'-I','-S','-B','-X','utf8','-X',
            'pycache_prefix='+str(ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache'),job['worker_path'],
            '--binding',str(args.binding.resolve()),'--binding-sha',args.binding_sha,'--freeze',args.freeze,
            '--directory',str(args.directory.resolve()),'--out',str(args.out.resolve())]
        r['command']=argv;r['creationflags']=8
        gettimes=ctypes.WinDLL('kernel32',use_last_error=True).GetProcessTimes
        gettimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;gettimes.restype=wintypes.BOOL
        with log.open('xb') as stream:
            p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            handle=wintypes.HANDLE(int(p._handle));stamps=[wintypes.FILETIME() for _ in range(4)]
            assert gettimes(handle,*[ctypes.byref(v) for v in stamps])
            ticks=(stamps[0].dwHighDateTime<<32)|stamps[0].dwLowDateTime
            r['worker_instance']=dict(pid=p.pid,creation_FILETIME=ticks,create_time_unix=(ticks-116444736000000000)/1e7)
            position=0
            while p.poll() is None:
                memory();guard()
                try:
                    descendants=psutil.Process(p.pid).children(recursive=True)
                    if descendants:
                        r['unexpected_descendants']=[v.as_dict(attrs=['pid','create_time','name','exe','cmdline','ppid']) for v in descendants]
                        raise AssertionError('unexpected descendants; owned instances retained')
                except psutil.NoSuchProcess:pass
                with log.open('rb') as progress:
                    progress.seek(position);chunk=progress.read();boundary=chunk.rfind(b'\n')+1
                    if boundary:print(chunk[:boundary].decode('utf8',errors='replace'),end='',flush=True);position+=boundary
                time.sleep(.1)
        memory();r.update(actual_worker_exit_code=p.returncode,worker_OS_peak_through_exit=worker_peak,worker_peak_sampled_AFTER_exit=True)
        guard();assert p.returncode==0,log.read_text(errors='replace')
        raw=json.loads(args.out.read_bytes());assert raw['schema']==job['result_schema']
        assert raw['process_instance']['pid']==p.pid and abs(raw['process_instance']['create_time_unix']-r['worker_instance']['create_time_unix'])<.002
        assert raw['decision'] in job['accepted_decisions']
        assert raw['new_original_BF16_full_forwards']==0 and raw['procedure_gates'] and all(raw['procedure_gates'].values())
        files=sorted(v for v in args.directory.iterdir() if v.is_file())
        r['output_manifest']=[dict(path=str(v.resolve()),bytes=v.stat().st_size,sha256=sha(v)) for v in files]
        binding_check(binding);preserved();guard()
        r['gates'].update(actual_exit_and_procedure=True,through_exit_resources=True,inputs_runtime_and_foreign_after=True)
        r.update(result_sha256=sha(args.out),log_sha256=sha(log),decision=raw['decision'],ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,
            conservative_family_peak_bytes=proc.memory_info().peak_wset+worker_peak,
            resource_scope='Worker OS peak includes actual exit; launcher final receipt/stdout tail outside last snapshot')
        write_once(terminal,r);guard()
        print(json.dumps(dict(terminal=str(terminal),worker_exit=p.returncode,decision=raw['decision'],worker_OS_peak_through_exit=worker_peak)),flush=True)
    except BaseException as error:
        if p is not None:
            if p.poll() is None:p.kill();p.wait()
            memory();r.update(actual_worker_exit_code=p.returncode,worker_OS_peak_through_exit=worker_peak,worker_peak_sampled_AFTER_exit=True)
        r.update(fault=repr(error),elapsed_seconds=time.monotonic()-start,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.launcher_failure.json'),r);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--binding-sha',required=True)
    parser.add_argument('--freeze',required=True);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
