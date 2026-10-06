"""Operational guards/receipts only; no fitting, target or probability mathematics."""
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
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
BIND=DOC/'meth490_r1_binding.json'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
    with Path(path).open('xb') as f:f.write((json.dumps(value,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())

class Memory(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]

class Context:
    def __init__(self,out,raw,seconds,bytes_limit):
        assert os.name=='nt' and sys.flags.optimize==0 and not out.exists() and not raw.exists() and not raw.with_suffix('.failure.json').exists()
        self.out,self.raw,self.seconds,self.bytes_limit=out,raw,seconds,bytes_limit
        self.start=time.monotonic();self.phase='binding';self.peak=self.child_peak=self.hashed=0
        self.cache={};self.child=None;self.descendants={};self.done=threading.Event();self.lock=threading.Lock()
        self.r={'experiment':'METH490 separate affine normalized conditional root mass','start_utc':stamp(),
                'process_instance':{'pid':os.getpid(),'create_time_unix':psutil.Process().create_time()},
                'argv':sys.argv,'commands':[],'gates':{},'isolation_waits':[],'numerical_imports':False}
        assert ctypes.sizeof(Memory)==72
        out.mkdir();self.fatal=(out/'fatal_native.log').open('xb');faulthandler.enable(file=self.fatal,all_threads=True)
        self.progress=(out/'progress.jsonl').open('x',encoding='utf8')
        parent=psutil.Process();parent.cpu_affinity([0]);assert parent.cpu_affinity()==[0]
        self.mem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        self.mem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];self.mem.restype=wintypes.BOOL
        self.times=ctypes.WinDLL('kernel32',use_last_error=True).GetProcessTimes
        self.times.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;self.times.restype=wintypes.BOOL
        self.watchdog=threading.Timer(max(.001,seconds-(time.monotonic()-self.start)),self.deadline)
        self.watchdog.daemon=True;self.watchdog.start()
        self.watcher=threading.Thread(target=self.observe,daemon=True);self.watcher.start()
    def log(self,**v):
        self.progress.write(json.dumps({'phase':self.phase,'seconds':time.monotonic()-self.start,**v})+'\n');self.progress.flush()
    def process_memory(self,child):
        v=Memory();v.cb=ctypes.sizeof(v);assert self.mem(wintypes.HANDLE(int(child._handle)),ctypes.byref(v),v.cb)
        return int(v.PeakWorkingSetSize)
    def observe(self):
        while not self.done.wait(.1):
            try:self.guard()
            except BaseException as e:self.abort(e);return
    def guard(self):
        m=psutil.Process().memory_info();self.peak=max(self.peak,m.rss,getattr(m,'peak_wset',0))
        observed=self.child
        if observed is not None:
            try:
                group=self.process_memory(observed)
                try:
                    for p in psutil.Process(observed.pid).children(recursive=True):
                        try:
                            m=p.memory_info();key=(p.pid,p.create_time());previous=self.descendants.get(key,{}).get('OS_peak_bytes',0)
                            value=max(m.rss,getattr(m,'peak_wset',0),previous);group+=value
                            self.descendants[key]={'pid':key[0],'create_time_unix':key[1],'OS_peak_bytes':value,'executable':p.exe()}
                        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                self.child_peak=max(self.child_peak,group)
            except (OSError,AssertionError):
                if observed.poll() is None:raise
        assert self.peak+self.child_peak<=self.bytes_limit,('combined_host_bound',self.peak,self.child_peak)
        assert time.monotonic()-self.start<=self.seconds,('wall_bound',self.phase)
        assert sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())+getattr(self,'inherited_output_bytes',0)<=64<<20,'combined_output64MiB'
    def deadline(self):self.abort(RuntimeError('METH490 hard wall deadline'))
    def abort(self,error):
        self.fail(error)
        if self.child is not None and self.child.poll() is None:
            self.child.kill();self.child.wait()
        os._exit(3)
    def fail(self,error):
        with self.lock:
            target=self.raw.with_suffix('.failure.json')
            if not target.exists() and not self.raw.exists():
                value={**self.r,'phase':self.phase,'end_utc':stamp(),'traceback':''.join(traceback.format_exception(error)),
                       'resource':self.resources(),'partial_outputs':[{'path':str(p),'bytes':p.stat().st_size} for p in sorted(self.out.iterdir()) if p.is_file()]}
                write(target,value)
    def resources(self):return {'wall_seconds':time.monotonic()-self.start,'parent_peak_bytes':self.peak,'native_peak_bytes':self.child_peak,'bytes_hashed':self.hashed,'wall_limit_seconds':self.seconds,'combined_host_limit_bytes':self.bytes_limit}
    def digest(self,path):
        p=Path(path);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns)
        if key not in self.cache:
            h=hashlib.sha256()
            with p.open('rb') as f:
                while data:=f.read(4<<20):h.update(data);self.hashed+=len(data);self.guard()
            assert p.stat().st_size==s.st_size and p.stat().st_mtime_ns==s.st_mtime_ns
            self.cache[key]=h.hexdigest()
        return self.cache[key]
    def exact(self,v):
        p=Path(v['path']);assert p.stat().st_size==v['bytes'] and self.digest(p)==v['sha256'],str(p)
    def head(self,p):
        p=Path(p);relative=p.relative_to(ROOT).as_posix()
        assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT).replace(b'\r\n',b'\n'),relative
    def quiet(self,label):
        began=time.monotonic();observations=[];last=None
        while True:
            own={os.getpid(),*(p.pid for p in psutil.Process().parents())};foreign=[];kept=[]
            for p in psutil.process_iter(['name','cmdline']):
                if p.pid in own or (self.child is not None and p.pid==self.child.pid):continue
                name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
                if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():kept.append(p.pid);continue
                if name.startswith(('python','clang')) or (name.startswith('meth') and name.endswith('.exe')):foreign.append({'pid':p.pid,'name':name})
            if not foreign:break
            self.guard();assert time.monotonic()-began<=240,'quiet240s'
            if foreign!=last:
                observations.append({'utc':stamp(),'foreign':foreign});print(json.dumps({'quiet':label,'foreign':foreign}),flush=True);last=foreign
            time.sleep(1)
        if observations:self.r['isolation_waits'].append({'label':label,'seconds':time.monotonic()-began,'observations':observations})
        self.r['preserved_daemons']=kept
    def admit(self,sha):
        self.quiet('initial');assert len(sha)==64 and self.digest(BIND)==sha
        self.head(BIND);b=json.loads(BIND.read_bytes());self.r['binding_sha256']=sha;self.inherited_output_bytes=b['inherited_output_bytes']
        assert sys.version==b['runtime']['python'] and str(Path(sys.executable).resolve())==b['runtime']['executable']
        for name,v in b['runtime']['packages'].items():assert importlib.metadata.version(name)==v['version']
        for v in b['catalog']:self.exact(v)
        for v in b['scientific']+b['records']:self.head(v['path'])
        assert hashlib.sha256((ROOT/'benchmarks/phase60/engine.c').read_bytes()).hexdigest()==b['engine_sha256']
        for rel,sha in b['preserved'].items():assert self.digest(ROOT/rel)==sha
        source=json.loads(Path(b['qualified_source_retention']['path']).read_bytes())
        assert all(source['gates'].values()) and source['unique_inputs_audited']==238872 and source['queries_audited']==387036
        assert source['raw_sha256']==b['qualified_source_raw']['sha256']
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status),status
        self.r['source_freeze']=b['freeze_head'];self.r['execution_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.r['gates']['frozen_source_supervision_runtime_and_scientific_files']=True
        return b
    def run(self,argv,label,expected=0):
        self.quiet(label);record={'label':label,'argv':list(map(str,argv)),'start_utc':stamp(),'descendant_process_peaks':[]}
        self.r['commands'].append(record)
        with (self.out/(label+'.stdout')).open('xb') as so,(self.out/(label+'.stderr')).open('xb') as se:
            child=subprocess.Popen(record['argv'],cwd=ROOT,stdout=so,stderr=se);self.descendants={};self.child=child
            times=[wintypes.FILETIME() for _ in range(4)];assert self.times(wintypes.HANDLE(int(child._handle)),*[ctypes.byref(v) for v in times])
            creation=((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)/1e7-11644473600
            record['process_instance']={'pid':child.pid,'create_time_unix':creation};started=time.monotonic()
            descendants={};modules=set();next_sample=started;exit_races=0
            while child.poll() is None:
                self.guard();assert time.monotonic()-started<=120,'child120s'
                if time.monotonic()>=next_sample:
                    try:
                        proc=psutil.Process(child.pid);modules.update(v.path for v in proc.memory_maps() if v.path.lower().endswith(('.dll','.exe')))
                        for p in proc.children(recursive=True):
                            try:
                                m=p.memory_info();key=(p.pid,p.create_time());descendants[key]={'pid':key[0],'create_time_unix':key[1],'OS_peak_bytes':max(m.rss,getattr(m,'peak_wset',0)),'executable':p.exe()}
                            except (psutil.NoSuchProcess,psutil.AccessDenied):exit_races+=1
                    except (psutil.NoSuchProcess,psutil.AccessDenied):exit_races+=1
                    next_sample=time.monotonic()+5
                time.sleep(.025)
            self.child_peak=max(self.child_peak,self.process_memory(child));self.child=None
        descendants.update(self.descendants)
        record.update(returncode=child.returncode,end_utc=stamp(),observer_exit_races=exit_races,descendant_process_peaks=list(descendants.values()),observed_modules=sorted(modules))
        self.log(command_terminal=label,returncode=child.returncode)
        assert child.returncode==expected,(label,child.returncode)
        return (self.out/(label+'.stdout')).read_text().splitlines()
    def finish(self,value):
        self.phase='terminal';self.log(terminal=True);self.guard();self.done.set();self.watchdog.cancel();self.watcher.join(timeout=1)
        for rel,sha in self.binding['preserved'].items():assert self.digest(ROOT/rel)==sha
        assert self.digest(ROOT/'benchmarks/phase60/engine.c')==self.binding['engine_sha256']
        inventory=[{'path':str(p),'bytes':p.stat().st_size,'sha256':self.digest(p)} for p in sorted(self.out.iterdir()) if p.is_file()]
        result={**self.r,**value,'end_utc':stamp(),'resource':self.resources(),'output_inventory':inventory}
        assert result['resource']['wall_seconds']<=self.seconds
        write(self.raw,result);self.progress.close();faulthandler.disable();self.fatal.close();return result
