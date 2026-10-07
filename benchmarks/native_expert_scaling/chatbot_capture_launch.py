"""Bound capture launch, direct interpreter and held Windows exit/peak handle."""
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
    r=dict(scope='ORIGINAL_SOURCE_CAPTURE_LAUNCH_THROUGH_EXIT',source_freeze=args.freeze,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),launcher_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),gates={})
    p=None;worker_peak=0;handle=None
    getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];getmem.restype=wintypes.BOOL
    def memory():
        nonlocal worker_peak
        if handle is not None:
            m=Memory();m.cb=ctypes.sizeof(m);assert getmem(handle,ctypes.byref(m),m.cb)
            worker_peak=max(worker_peak,m.PeakWorkingSetSize)
    def guard():
        assert time.monotonic()-start<=1200,'whole family capture deadline including full runtime seals'
        assert proc.memory_info().peak_wset+worker_peak<=12<<30,'family OS peaks'
        assert not log.exists() or log.stat().st_size<=2<<20,'worker log cap'
    def preserved():
        assert all(sha(ROOT/rel)==want for rel,want in FOREIGN.items())
        assert not any(p.is_file() for p in (ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    def binding_check(b):
        for entry in b['inputs']:
            assert str(Path(entry['path']).resolve())==entry['resolved_path'],'logical source/input mapping changed'
            assert Path(entry['path']).stat().st_size==entry['bytes'] and sha(entry['path'])==entry['sha256'],entry['path']
            guard()
        t=b['runtime_tree'];assert tree(t['path'])==(t['tree_sha256'],t['files'],t['bytes'])
        for v in b['interaction_view']:
            assert (ROOT/'results/native_expert_scaling/chatbot_interaction_runtime/site'/v['view_name']).samefile(v['path'])
        guard()
    try:
        assert sys.version_info[:3]==(3,12,10) and psutil.__version__=='7.2.2'
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes());assert b['schema']=='QWEN_ORIGINAL_CAPTURE_BINDING_V1'
        assert Path(sys.executable).resolve()==Path(b['python']).resolve()
        own={proc.pid,*(i.pid for i in proc.parents())}
        for other in psutil.process_iter(['name','cmdline']):
            name=(other.info['name'] or '').lower();cmd=other.info['cmdline'] or []
            if other.pid in own:continue
            if name=='pythonw.exe' and len(cmd)==2 and Path(cmd[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
            assert not name.startswith(('python','clang','meth')),('overlap',other.pid,name)
        preserved();binding_check(b);r['gates']['frozen_inputs_runtime_and_view']=True
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',
            TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
        argv=[b['python'],'-I','-S','-B','-X','utf8','-X',
            'pycache_prefix='+str(ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache'),
            str(ROOT/'benchmarks/native_expert_scaling/chatbot_source_capture.py'),
            '--binding',str(args.binding.resolve()),'--binding-sha',args.binding_sha,'--freeze',args.freeze,
            '--directory',str(args.directory.resolve()),'--out',str(args.out.resolve())]
        r['command']=argv;r['creationflags']=8
        gettimes=ctypes.WinDLL('kernel32',use_last_error=True).GetProcessTimes
        gettimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;gettimes.restype=wintypes.BOOL
        with log.open('xb') as stream:
            p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            handle=wintypes.HANDLE(int(p._handle));stamps=[wintypes.FILETIME() for _ in range(4)]
            assert gettimes(handle,*[ctypes.byref(s) for s in stamps])
            ticks=(stamps[0].dwHighDateTime<<32)|stamps[0].dwLowDateTime
            r['worker_instance']=dict(pid=p.pid,creation_FILETIME=ticks,create_time_unix=(ticks-116444736000000000)/1e7)
            log_position=0
            while p.poll() is None:
                memory();guard()
                try:
                    descendants=psutil.Process(p.pid).children(recursive=True)
                    if descendants:
                        details=[]
                        for child in descendants:
                            try:
                                details.append(child.as_dict(attrs=['pid','create_time','name','exe','cmdline','ppid']))
                            except psutil.NoSuchProcess:
                                details.append(dict(pid=child.pid,instance='exited before detail read'))
                        r['unexpected_descendants']=details
                        raise AssertionError('unexpected worker descendant; identities retained')
                except psutil.NoSuchProcess:pass
                # Forward complete progress lines; no repeated application reads.
                with log.open('rb') as progress:
                    progress.seek(log_position);chunk=progress.read()
                    boundary=chunk.rfind(b'\n')+1
                    if boundary:
                        print(chunk[:boundary].decode('utf8',errors='replace'),end='',flush=True)
                        log_position+=boundary
                time.sleep(.1)
        memory();r.update(actual_worker_exit_code=p.returncode,worker_OS_peak_through_exit=worker_peak,worker_peak_sampled_AFTER_exit=True)
        guard();assert p.returncode==0,log.read_text(errors='replace')
        raw=json.loads(args.out.read_bytes())
        assert raw['process_instance']['pid']==p.pid and abs(raw['process_instance']['create_time_unix']-r['worker_instance']['create_time_unix'])<.002
        assert raw['decision']=='ORIGINAL_SOURCE_CALIBRATION_AVAILABLE_NOT_CONVERTER_QUALIFIED' and all(raw['gates'].values())
        for case in raw['conversations']:
            assert sha(case['binary_path'])==case['binary_SHA256'] and sha(case['journal_path'])==case['journal_SHA256']
        binding_check(b);preserved();guard()
        r['gates'].update(final_worker_exit_and_capture_pass=True,through_exit_resources=True,inputs_runtime_and_foreign_preserved=True)
        r.update(result_sha256=sha(args.out),log_sha256=sha(log),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,
            conservative_family_peak_bytes=proc.memory_info().peak_wset+worker_peak,
            resource_scope='Worker OS peak includes actual exit; launcher final report/stdout tail outside its last snapshot',
            limits=dict(worker_seconds=300,family_seconds=1200,family_OS_peak_bytes=12<<30,GPU_allocated=9<<30,GPU_reserved=10<<30,capture_payload_bytes=2<<30))
        write_once(terminal,r);guard();print(json.dumps(dict(terminal=str(terminal),worker_exit=p.returncode,worker_OS_peak_through_exit=worker_peak)),flush=True)
    except BaseException as error:
        if p is not None:
            if p.poll() is None:p.kill();p.wait()
            memory();r.update(actual_worker_exit_code=p.returncode,worker_OS_peak_through_exit=worker_peak,worker_peak_sampled_AFTER_exit=True)
        r.update(fault=repr(error),elapsed_seconds=time.monotonic()-start,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(args.out.with_suffix('.launcher_failure.json'),r);raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
