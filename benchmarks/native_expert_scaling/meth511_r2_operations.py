"""511 staged bounds, frozen file identities and exact native process receipts."""
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

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
BIND=DOC/'meth511_r2_binding.json'
LIMITS={'cohort':(360,2<<30,64<<20),'native':(2400,2<<30,14<<30),'donor':(3600,24<<30,3<<30),'audit':(1200,4<<30,256<<20)}
ENV={'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false'}
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf8')
    with Path(path).open('xb') as f:f.write(data)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(8<<20):h.update(data)
    return h.hexdigest()
def worker(row):
    assert row['worker_physical_cores']==6
    assert [v['slot'] for v in row['worker_affinity']]==[0,1,2]
    assert [v['actual_mask'] for v in row['worker_affinity']]==[1,4,16]
    assert all(v['group']==0 for v in row['worker_affinity'])
    assert len({v['windows_thread_id'] for v in row['worker_affinity']})==3
    assert all(v['actual_mask']==[1,4,16][v['slot']] and v['group']==0 for v in row['worker_binding_events'])
    assert set(v['slot'] for v in row['worker_binding_events'])=={0,1,2}
class Memory(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]

class Context:
    def __init__(self,kind):
        assert kind in LIMITS and not sys.flags.optimize and sys.flags.isolated and sys.dont_write_bytecode
        self.kind=kind;self.start=time.monotonic();self.proc=psutil.Process();self.proc.cpu_affinity([10])
        self.out=ROOT/'results/native_expert_scaling'/('meth511_r2_'+kind);self.out.mkdir(exist_ok=False)
        self.destination=DOC/('meth511_r2_'+kind+'_result.json');assert not self.destination.exists() and not self.destination.with_suffix('.failure.json').exists()
        self.peak=self.native_peak=self.hashed=0;self.child=None;self.handles=[];self.rows=[];self.closed=threading.Event();self.lock=threading.Lock()
        self.progress=(self.out/'progress.jsonl').open('x',encoding='utf8');self.fatal=(self.out/'fatal.log').open('x',encoding='utf8');faulthandler.enable(self.fatal)
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];self.kernel.CreateFileW.restype=wintypes.HANDLE
        self.kernel.CloseHandle.argtypes=[wintypes.HANDLE];self.kernel.CloseHandle.restype=wintypes.BOOL
        self.kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;self.kernel.GetProcessTimes.restype=wintypes.BOOL
        self.mem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        self.mem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];self.mem.restype=wintypes.BOOL
        self.r={'experiment':'METH511 fresh whole fixed8 source readout','kind':kind,'argv':sys.argv,'started_utc':stamp(),
            'process_instance':{'pid':self.proc.pid,'create_time_unix':self.proc.create_time()},'affinity':[10],'commands':[],'gates':{},'native_calls':0,'model_calls':0}
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def guard(self):
        mi=self.proc.memory_info();self.peak=max(self.peak,mi.rss,mi.peak_wset);seconds,peak,output=LIMITS[self.kind]
        assert self.peak<=peak and time.monotonic()-self.start<=seconds,('resource',self.kind,self.peak,time.monotonic()-self.start)
        size=sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        if self.destination.exists():size+=self.destination.stat().st_size
        assert size<=output,('output_bytes',size)
        child=self.child
        if child is not None:
            v=Memory();v.cb=ctypes.sizeof(v);assert self.mem(wintypes.HANDLE(int(child._handle)),ctypes.byref(v),v.cb)
            self.native_peak=max(self.native_peak,int(v.PeakWorkingSetSize));assert self.native_peak<=4<<30
    def watch(self):
        while not self.closed.wait(.5):
            try:self.guard()
            except BaseException:
                self.fail();child=self.child
                if child is not None and child.poll() is None:child.kill();child.wait()
                os._exit(124)
    def digest(self,path):
        p=Path(path);before=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            while data:=f.read(8<<20):h.update(data);self.hashed+=len(data);self.guard()
        assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
        return h.hexdigest()
    def exact(self,row):
        assert Path(row['path']).stat().st_size==row['bytes'] and self.digest(row['path'])==row['sha256'],row['path']
    def retain(self,row):
        p=Path(row['path']);h=self.kernel.CreateFileW(str(p),0x80000000,1,None,3,0x80,None)
        assert h not in (None,ctypes.c_void_p(-1).value),('write_deny_open',str(p),ctypes.get_last_error())
        self.handles.append(h);s=p.stat();assert (s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns']),('identity_stat',str(p));self.rows.append(row)
    def admit(self,digest):
        assert self.digest(BIND)==digest;self.binding=b=json.loads(BIND.read_bytes())
        assert sys.version==b['python'] and str(Path(sys.executable).resolve())==b['executable']
        for k,v in b['packages'].items():assert importlib.metadata.version(k)==v
        assert sys.pycache_prefix==b['runtime_empty_cache'] and not any(Path(sys.pycache_prefix).rglob('*'))
        for name in ['torch','pyarrow','pandas','sklearn','scipy']:assert name not in sys.modules
        for row in b['catalog']:self.retain(row)
        self.catalog={str(Path(r['path']).resolve()).lower() for r in b['catalog']}
        for row in b['scientific']:
            self.exact(row);p=Path(row['path']);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
        for rel,v in b['preserved'].items():assert self.digest(ROOT/rel)==v
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status),status
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        os.environ.update(ENV)
        self.r.update(binding_sha256=digest,source_freeze=b['freeze_head'],execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            input_admission_scope='Fresh full SHA of runtime/source/candidate/corpus/ledgers; original donor ZIP whole SHA retained from acquisition with current stat/write-deny binding. All3320 loaded original F32 tensors receive fresh canonical SHA before first donor inference. Small scientific source is freshly SHA checked.')
        self.r['gates']['bound_runtime_full_payload_donor_corpus_and_committed_sources']=True
        self.modules();self.guard();return b
    def import_result(self,kind,digest):
        p=DOC/('meth511_r2_'+kind+'_result.json');assert self.digest(p)==digest;r=json.loads(p.read_bytes());assert r['binding_sha256']==self.r['binding_sha256']
        try:assert abs(psutil.Process(r['process_instance']['pid']).create_time()-r['process_instance']['create_time_unix'])>=.002
        except psutil.NoSuchProcess:pass
        for row in r['output_inventory']:self.exact(row)
        self.r[kind+'_sha256']=digest;return r
    def modules(self):
        forbidden=('pyarrow','pandas','sklearn','scipy','datasets','bitsandbytes','accelerate')
        assert not any(n==x or n.startswith(x+'.') for n in sys.modules for x in forbidden)
        if self.kind!='donor':assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
        if self.kind in ('native','donor'):assert not any(n in ('duckdb','_duckdb') or n.startswith('duckdb.') for n in sys.modules)
        files=set()
        for m in tuple(sys.modules.values()):
            p=getattr(m,'__file__',None)
            if p and Path(p).is_file():
                key=str(Path(p).resolve()).lower();assert key in self.catalog,('unbound_Python_module',key);files.add(key)
        images=[];windows=str(Path(os.environ['WINDIR']).resolve()).lower()+'\\'
        for v in self.proc.memory_maps():
            p=Path(v.path)
            if p.suffix.lower() not in ('.dll','.pyd','.exe') or not p.is_file():continue
            key=str(p.resolve()).lower();assert key in self.catalog or key.startswith(windows),('unbound_image',key)
            images.append({'path':str(p),'system_windows_image':key.startswith(windows)})
        self.r['loaded_python_files']=sorted(files);self.r['loaded_images']=images
    def scientific_processes(self,exclude):
        allowed={(r['pid'],r['create_time_unix']):r for r in self.binding['idle_processes']};kept=[];foreign=[]
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in exclude:continue
            name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
            identity=(p.pid,p.create_time()) if p.pid in {k[0] for k in allowed} else None
            publisher=name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve()
            if identity in allowed or publisher:
                if identity in allowed:
                    row=allowed[identity];assert p.exe()==row['exe'] and p.name()==row['name']
                    assert sorted((q.pid,q.create_time()) for q in p.children(recursive=True))==sorted(tuple(v) for v in row['descendants'])
                kept.append({'pid':p.pid,'create_time_unix':p.create_time(),'exe':p.exe(),'cpu_seconds':sum(p.cpu_times()[:2]),'bound_idle_app':identity in allowed})
            elif name.startswith(('python','clang','llama','ollama','nvcc','nvidia-smi')) or (name.startswith('meth') and name.endswith('.exe')):
                foreign.append({'pid':p.pid,'create_time_unix':p.create_time(),'name':name,'exe':p.exe()})
        assert {r['pid'] for r in kept if r['bound_idle_app']}=={r['pid'] for r in self.binding['idle_processes']}
        now={(r['pid'],r['create_time_unix']):r['cpu_seconds'] for r in kept}
        if not hasattr(self,'idle_CPU'):self.idle_CPU=now
        assert now==self.idle_CPU,'idle_process_CPU_or_identity_change';return foreign,kept
    def run_native(self,argv,label):
        assert self.kind=='native';ancestors={self.proc.pid,*(p.pid for p in self.proc.parents())}
        foreign,kept=self.scientific_processes(ancestors);assert not foreign,('foreign_before',foreign)
        record={'label':label,'argv':list(map(str,argv)),'start_utc':stamp(),'idle_start':kept,'expected_exit':0};self.r['commands'].append(record)
        with (self.out/(label+'.stdout')).open('xb') as so,(self.out/(label+'.stderr')).open('xb') as se:
            child=subprocess.Popen(record['argv'],cwd=ROOT,stdout=so,stderr=se,env={**os.environ,**ENV,**self.binding['native_runtime_environment']});self.child=child;self.r['native_calls']+=1
            ft=[wintypes.FILETIME() for _ in range(4)];assert self.kernel.GetProcessTimes(wintypes.HANDLE(int(child._handle)),*[ctypes.byref(v) for v in ft])
            record['process_instance']={'pid':child.pid,'create_time_unix':((ft[0].dwHighDateTime<<32)|ft[0].dwLowDateTime)/1e7-11644473600}
            modules=set();observations=[];races=0;last=self.start;gaps=[]
            try:
                while child.poll() is None:
                    self.guard();now=time.monotonic()
                    try:
                        p=psutil.Process(child.pid);desc=p.children(recursive=True);assert not desc,('native_children',[(q.pid,q.create_time()) for q in desc])
                        modules.update(v.path for v in p.memory_maps() if Path(v.path).suffix.lower() in ('.dll','.exe','.pyd'))
                        foreign,kept=self.scientific_processes(ancestors|{child.pid});assert not foreign,('foreign_during',foreign)
                        observations.append({'utc':stamp(),'idle':kept});gaps.append(now-last);last=now
                    except psutil.NoSuchProcess:races+=1
                    time.sleep(.10)
                child.wait();self.guard()
            finally:
                if child.poll() is None:child.kill();child.wait()
                self.guard();record.update(returncode=child.returncode,end_utc=stamp(),native_peak_bound_so_far=self.native_peak,observer_exit_races=races)
                self.child=None
        foreign,kept=self.scientific_processes(ancestors);assert not foreign
        windows=str(Path(os.environ['WINDIR']).resolve()).lower()+'\\'
        for p in modules:assert str(Path(p).resolve()).lower() in self.catalog or str(Path(p).resolve()).lower().startswith(windows),('unbound_child_image',p)
        record.update(observed_modules=sorted(modules),descendants=[],timing_observations=observations,idle_end=kept,
            maximum_intermediate_sample_gap_seconds=max(gaps[1:],default=0),scope='100ms target sampled scientific topology and exact idle CPU counters; other OS work remains; no exhaustive trace.')
        assert child.returncode==0 and (self.out/(label+'.stderr')).stat().st_size==0,(label,child.returncode)
        self.log(command_terminal=label);return [json.loads(v) for v in (self.out/(label+'.stdout')).read_text(encoding='utf8').splitlines()]
    def log(self,**v):self.progress.write(json.dumps({'seconds':time.monotonic()-self.start,**v})+'\n');self.progress.flush()
    def finish(self,value):
        self.modules();self.guard();assert (self.out/'fatal.log').stat().st_size==0
        for row in self.rows:
            s=Path(row['path']).stat();assert (s.st_size,s.st_mtime_ns)==(row['bytes'],row['mtime_ns'])
        for rel,v in self.binding['preserved'].items():assert self.digest(ROOT/rel)==v
        inventory=[]
        for p in sorted(self.out.iterdir()):
            if p.is_file() and p.name not in ('progress.jsonl','fatal.log'):
                inventory.append({'path':str(p),'bytes':p.stat().st_size,'sha256':self.digest(p)})
        self.r.update(value,output_inventory=inventory,ended_compute_utc=stamp());self.r['gates']['terminal_input_stat_write_deny_and_empty_fatal']=True
        write(self.destination,self.r);digest=self.digest(self.destination);self.log(terminal_serialized_and_hashed=True);self.guard()
        terminal={'result_sha256':digest,'process_instance':self.r['process_instance'],'seconds':time.monotonic()-self.start,'OS_peak_bytes':self.peak,'native_OS_peak_bytes':self.native_peak,'bytes_hashed':self.hashed,'limits':LIMITS[self.kind],'scope':'Through serialization, full output inventory SHA and terminal guard; process exit follows.'}
        write(self.out/'terminal_resource.json',terminal);self.guard()
        print(json.dumps({'result':str(self.destination),'sha256':digest,'terminal':terminal,'summary':value.get('summary')}),flush=True)
    def fail(self):
        with self.lock:
            p=self.destination.with_suffix('.failure.json')
            if not p.exists():write(p,{**self.r,'traceback':traceback.format_exc(),'seconds':time.monotonic()-self.start,'OS_peak_bytes':self.peak,'native_OS_peak_bytes':self.native_peak})
    def close(self):
        self.closed.set();self.thread.join(timeout=2)
        for h in self.handles:assert self.kernel.CloseHandle(h)
        self.progress.close();self.fatal.close()
