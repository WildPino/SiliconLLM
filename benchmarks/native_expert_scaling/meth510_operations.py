"""510 bounded native receipts; exact retained inputs, never a whole-model replay."""
import ctypes
from ctypes import wintypes
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
import psutil
import meth507_operations as base
from meth490_r1_operations import Memory
from meth508_operations import manifest

ROOT,DOC,write=base.ROOT,base.DOC,base.write
BIND=DOC/'meth510_binding.json'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

class Context(base.Context):
    def __init__(self,kind):
        assert kind in ('main','audit') and sys.flags.optimize==0
        self.start=time.monotonic();self.proc=psutil.Process();self.proc.cpu_affinity([10]);self.kind=kind
        self.out=ROOT/'results/native_expert_scaling'/('meth510_'+kind);self.out.mkdir(exist_ok=False)
        self.destination=DOC/('meth510_'+kind+'_result.json');assert not self.destination.exists() and not self.destination.with_suffix('.failure.json').exists()
        self.peak=self.hashed=0;self.child=None;self.native_peak=0;self.closed=threading.Event();self.lock=threading.Lock()
        self.progress=(self.out/'progress.jsonl').open('x',encoding='utf8');self.fatal=(self.out/'fatal.log').open('x',encoding='utf8');faulthandler.enable(self.fatal)
        self.r={'experiment':'METH510 native top8 source head parity and paired cost','kind':kind,'argv':sys.argv,
                'process_instance':{'pid':self.proc.pid,'create_time_unix':self.proc.create_time()},'started_utc':stamp(),'affinity':[10],'commands':[],'gates':{}}
        self.mem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        self.mem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];self.mem.restype=wintypes.BOOL
        self.times=ctypes.WinDLL('kernel32',use_last_error=True).GetProcessTimes
        self.times.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;self.times.restype=wintypes.BOOL
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def guard(self):
        m=self.proc.memory_info();self.peak=max(self.peak,m.rss,m.peak_wset)
        assert self.peak<=2<<30 and time.monotonic()-self.start<=360
        size=sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        if self.destination.exists():size+=self.destination.stat().st_size
        assert size<=512<<20,('output_bytes',size)
        if self.child is not None:
            v=Memory();v.cb=ctypes.sizeof(v)
            assert self.mem(wintypes.HANDLE(int(self.child._handle)),ctypes.byref(v),v.cb)
            self.native_peak=max(self.native_peak,int(v.PeakWorkingSetSize));assert self.native_peak<=1<<30
    def watch(self):
        while not self.closed.wait(.5):
            try:self.guard()
            except BaseException:
                self.fail()
                child=self.child
                if child is not None and child.poll() is None:child.kill();child.wait()
                os._exit(124)
    def admit(self,sha):
        assert self.digest(BIND)==sha;b=json.loads(BIND.read_bytes());self.binding=b
        assert sys.version==b['python'] and str(Path(sys.executable).resolve())==b['executable']
        for name,version in b['packages'].items():assert importlib.metadata.version(name)==version
        for row in b['catalog']:self.exact(row)
        self.catalog={str(Path(v['path']).resolve()).lower() for v in b['catalog']}
        for row in b['scientific']:
            p=Path(row['path']);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
        for rel,digest in b['preserved'].items():assert self.digest(ROOT/rel)==digest
        assert self.digest(ROOT/'benchmarks/phase60/engine.c')==b['engine_sha256']
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status),status
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        self.extent_stats=[]
        for row in b['extents']:
            p=Path(row['path']);s=p.stat();h=hashlib.sha256();left=row['bytes']
            assert s.st_size==row['file_bytes']
            with p.open('rb') as f:
                f.seek(row['offset'])
                while left:
                    data=f.read(min(left,8<<20));assert data;h.update(data);left-=len(data);self.hashed+=len(data);self.guard()
            assert h.hexdigest()==row['sha256'];self.extent_stats.append((p,s.st_size,s.st_mtime_ns))
        self.r.update(binding_sha256=sha,source_freeze=b['freeze_head'],execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        self.r['gates']['ALL_frozen_actual_runtime_compiler_wires_509_oracle_and_head_extents']=True
        return b
    def scientific_processes(self,exclude):
        foreign=[];kept=[]
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in exclude:continue
            name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                kept.append({'pid':p.pid,'create_time_unix':p.create_time(),'cpu_seconds':sum(p.cpu_times()[:2])});continue
            if name.startswith(('python','clang','llama','ollama')) or (name.startswith('meth') and name.endswith('.exe')):
                foreign.append({'pid':p.pid,'create_time_unix':p.create_time(),'name':name})
        return foreign,kept
    def run(self,argv,label,expected=0,timing=False):
        assert self.kind=='main';ancestors={self.proc.pid,*(p.pid for p in self.proc.parents())}
        if timing:
            foreign,kept=self.scientific_processes(ancestors);assert not foreign,('foreign_before_timing',foreign)
        record={'label':label,'argv':list(map(str,argv)),'start_utc':stamp(),'expected_exit':expected,'timing':timing,'observed_modules':[],'descendants':[]}
        self.r['commands'].append(record)
        with (self.out/(label+'.stdout')).open('xb') as so,(self.out/(label+'.stderr')).open('xb') as se:
            child=subprocess.Popen(record['argv'],cwd=ROOT,stdout=so,stderr=se,env={**os.environ,**self.binding['runtime_environment']})
            self.child=child;ft=[wintypes.FILETIME() for _ in range(4)]
            assert self.times(wintypes.HANDLE(int(child._handle)),*[ctypes.byref(v) for v in ft])
            record['process_instance']={'pid':child.pid,'create_time_unix':((ft[0].dwHighDateTime<<32)|ft[0].dwLowDateTime)/1e7-11644473600}
            modules=set();desc={};observations=[];samples=0;last=0.;races=0
            try:
                while child.poll() is None:
                    self.guard()
                    if time.monotonic()-last>=.10:
                        try:
                            group=[psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]
                            for p in group:
                                try:
                                    modules.update(v.path for v in p.memory_maps() if v.path.lower().endswith(('.dll','.exe','.pyd')))
                                    if p.pid!=child.pid:
                                        mi=p.memory_info();desc[(p.pid,p.create_time())]={'pid':p.pid,'create_time_unix':p.create_time(),'path':p.exe(),'OS_peak_bytes':mi.peak_wset}
                                except (psutil.NoSuchProcess,psutil.AccessDenied):races+=1
                            if timing:
                                foreign,daemons=self.scientific_processes(ancestors|{p.pid for p in group});samples+=1
                                assert not foreign,('foreign_during_timing',foreign)
                                observations.append({'utc':stamp(),'preserved_daemons':daemons})
                        except psutil.NoSuchProcess:races+=1
                        last=time.monotonic()
                    time.sleep(.02)
                child.wait();self.guard()
            finally:
                if child.poll() is None:child.kill();child.wait()
                record.update(returncode=child.returncode,end_utc=stamp(),observer_exit_races=races,native_peak_bound_so_far=self.native_peak)
                self.child=None
        record.update(observed_modules=sorted(modules),descendants=list(desc.values()),timing_samples=samples,timing_observations=observations)
        assert races<=16,('observer_exit_races',label,races)
        assert all(v['OS_peak_bytes']<=1<<30 for v in desc.values()),'compiler_descendant_peak'
        for path in modules:
            p=Path(path).resolve();key=str(p).lower()
            if key.startswith(str(Path(os.environ['WINDIR']).resolve()).lower()+'\\'):continue
            if p.parent==self.out.resolve() and p.name in ('meth510.exe','libomp.dll'):continue
            assert key in self.catalog,('unbound_child_image',path)
        if timing:
            foreign,endkept=self.scientific_processes(ancestors);assert not foreign and samples>0
            record['timing_start_daemons']=kept;record['timing_end_daemons']=endkept
            for first in kept:
                last_kept=[v for v in endkept if (v['pid'],v['create_time_unix'])==(first['pid'],first['create_time_unix'])]
                assert len(last_kept)==1 and last_kept[0]['cpu_seconds']==first['cpu_seconds'],'preserved_daemon_CPU_during_timing'
            record['scope']='No foreign scientific process in 100ms samples; bound daemon has zero CPU delta. Other OS background work remains; not a final whole-model rate certificate.'
        self.log(command_terminal=label,returncode=child.returncode);assert child.returncode==expected,(label,child.returncode)
        return (self.out/(label+'.stdout')).read_text(encoding='utf8').splitlines()
    def finish(self,value):
        self.modules();self.guard();assert (self.out/'fatal.log').stat().st_size==0
        for p,size,mtime in self.extent_stats:assert (p.stat().st_size,p.stat().st_mtime_ns)==(size,mtime)
        for rel,sha in self.binding['preserved'].items():assert self.digest(ROOT/rel)==sha
        self.r.update(value);self.r['ended_compute_utc']=stamp();self.r['gates']['terminal_empty_fatal_and_unchanged_inputs_foreign_bytes']=True
        write(self.destination,self.r);digest=self.digest(self.destination);self.log(terminal_result_serialized_and_hashed=True);self.guard()
        terminal={'scope':'Through result serialization, SHA and terminal guard; process exit follows. Native peak sampled via retained Win32 process handles including terminal query.',
                  'result_sha256':digest,'seconds':time.monotonic()-self.start,'OS_peak_bytes':self.peak,'native_peak_bytes':self.native_peak,
                  'bytes_hashed':self.hashed,'process_instance':self.r['process_instance'],'summary':value['summary'],'gates':self.r['gates']}
        write(self.out/'terminal_resource.json',terminal);self.guard()
        print(json.dumps({'result':str(self.destination),'sha256':digest,'terminal_resource':terminal,'summary':value['summary']}),flush=True)
