"""Frozen offline worker launch; Windows handle retains final OS peak through exit."""
import argparse
import ctypes
from ctypes import wintypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'results/native_expert_scaling/chatbot_interaction_runtime/site'))
import psutil

FOREIGN={
 'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
 'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
 'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
EXT={'.py','.pyd','.dll','.json','.pem'}
NAMES={'METADATA','WHEEL','RECORD'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def tree(path):
    path=Path(path)
    if path.is_file(): return sha(path),1,path.stat().st_size
    files=sorted((p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                  (p.suffix in EXT or p.name in NAMES)),key=lambda p:p.relative_to(path).as_posix())
    h=hashlib.sha256(); total=0
    for p in files:
        n=p.stat().st_size; total+=n
        h.update((p.relative_to(path).as_posix()+'\0'+str(n)+'\0'+sha(p)+'\n').encode('utf8'))
    return h.hexdigest(),len(files),total


def write_once(path,value):
    raw=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
    assert len(raw)<=2<<20
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


class Memory(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in (
        'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
        'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]


def main(args):
    start=time.monotonic(); proc=psutil.Process(); proc.cpu_affinity([11])
    assert sys.version_info[:3]==(3,12,10) and psutil.__version__=='7.2.2'
    out=args.out; terminal=out.with_suffix('.terminal.json'); log=out.with_suffix('.worker.log')
    assert not any(p.exists() for p in (out,terminal,log,out.with_suffix('.failure.json')))
    r=dict(scope='FINAL_OS_RESOURCE_AND_INPUT_BINDING_NO_MODEL_CALLS',source_commit=args.freeze,
           started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
           launcher_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),gates={})
    p=None
    worker_peak=0
    def guard():
        assert time.monotonic()-start<=120, 'whole family deadline'
        assert proc.memory_info().peak_wset+worker_peak<=512<<20, 'conservative family OS peaks'
        assert not log.exists() or log.stat().st_size<=2<<20
    def preserve():
        assert all(sha(ROOT/rel)==want for rel,want in FOREIGN.items())
        assert not any(v.is_file() for v in (ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    try:
        assert sha(args.binding)==args.binding_sha
        b=json.loads(args.binding.read_bytes())
        assert b['schema']=='QWEN_INTERACTION_BINDING_V1'
        assert Path(sys.executable).resolve()==Path(b['python']).resolve()
        own={proc.pid,*(v.pid for v in proc.parents())}
        for other in psutil.process_iter(['name','cmdline']):
            name=(other.info['name'] or '').lower(); argv=other.info['cmdline'] or []
            if other.pid in own: continue
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): continue
            assert not name.startswith(('python','clang','meth')),('overlap',other.pid,name)
        preserve()
        for item in b['inputs']:
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],item['path']
            guard()
        for item in b['runtime_roots']:
            assert tree(item['path'])==(item['tree_sha256'],item['files'],item['bytes']),item['path']
            view=ROOT/'results/native_expert_scaling/chatbot_interaction_runtime/site'/item['view_name']
            assert view.samefile(Path(item['path'])),item['view_name']
            guard()
        r['gates']['all_frozen_inputs_and_runtime_trees']=True
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',
                 TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        argv=[b['python'],'-I','-S','-B','-X','utf8','-X',
              'pycache_prefix='+str(ROOT/'results/native_expert_scaling/meth511_runtime/empty_cache'),
              str(ROOT/'benchmarks/native_expert_scaling/chatbot_interaction.py'),
              '--binding',str(args.binding.resolve()),'--binding-sha',args.binding_sha,
              '--fixtures',str(ROOT/'benchmarks/native_expert_scaling/chatbot_interaction_goldens.json'),
              '--freeze',args.freeze,'--out',str(out.resolve())]
        getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]; getmem.restype=wintypes.BOOL
        gettimes=ctypes.WinDLL('kernel32',use_last_error=True).GetProcessTimes
        gettimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4; gettimes.restype=wintypes.BOOL
        with log.open('xb') as stream:
            p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            handle=wintypes.HANDLE(int(p._handle)); stamps=[wintypes.FILETIME() for _ in range(4)]
            assert gettimes(handle,*[ctypes.byref(v) for v in stamps])
            ticks=(stamps[0].dwHighDateTime<<32)|stamps[0].dwLowDateTime
            r['worker_instance']=dict(pid=p.pid,creation_FILETIME=ticks,create_time_unix=(ticks-116444736000000000)/1e7)
            r['command']=argv; r['creationflags']=8
            while p.poll() is None:
                m=Memory(); m.cb=ctypes.sizeof(m); assert getmem(handle,ctypes.byref(m),m.cb)
                worker_peak=max(worker_peak,m.PeakWorkingSetSize)
                try: assert not psutil.Process(p.pid).children(recursive=True),'unexpected worker descendant'
                except psutil.NoSuchProcess: pass
                guard(); time.sleep(.01)
        m=Memory(); m.cb=ctypes.sizeof(m); assert getmem(handle,ctypes.byref(m),m.cb)
        worker_peak=max(worker_peak,m.PeakWorkingSetSize)
        r.update(actual_worker_exit_code=p.returncode,worker_OS_peak_through_exit=worker_peak,
                 worker_peak_sampled_AFTER_exit=True)
        guard()
        assert p.returncode==0,log.read_text(errors='replace')
        raw=json.loads(out.read_bytes())
        assert raw['process_instance']['pid']==p.pid and abs(raw['process_instance']['create_time_unix']-r['worker_instance']['create_time_unix'])<.002
        assert raw['decision']=='PLAIN_QWEN_INTERACTION_QUALIFIED_PIPELINE_NOT_QUALIFIED' and all(raw['gates'].values())
        for item in b['inputs']:
            assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
        for item in b['runtime_roots']:
            assert tree(item['path'])==(item['tree_sha256'],item['files'],item['bytes'])
            guard()
        preserve(); guard()
        r['gates'].update(final_through_exit_worker_peak_and_family_budget=True,
                           inputs_and_runtime_unchanged_after_worker=True,
                           foreign_and_cache_preserved=True,worker_actual_exit0_and_raw_contract_pass=True)
        r.update(result_sha256=sha(out),log_sha256=sha(log),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset,
                 conservative_family_peak_bytes=proc.memory_info().peak_wset+worker_peak,
                 resource_scope='Worker OS peak through actual exit; launcher final report/short stdout tail not covered by its last snapshot.',
                 limits_seconds_family_peak=[120,512<<20])
        write_once(terminal,r); guard()
        print(json.dumps(dict(terminal=str(terminal),worker_exit=p.returncode,worker_OS_peak_through_exit=worker_peak,
                              conservative_family_peak_bytes=r['conservative_family_peak_bytes'],gates=r['gates'])),flush=True)
    except BaseException as error:
        if p is not None and p.poll() is None: p.kill(); p.wait()
        r.update(fault=repr(error),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                 worker_OS_peak_observed=worker_peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write_once(out.with_suffix('.launcher_failure.json'),r)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--binding-sha',required=True)
    p.add_argument('--freeze',required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
