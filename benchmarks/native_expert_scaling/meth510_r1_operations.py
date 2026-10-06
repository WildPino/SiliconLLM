"""Prospective510 operational resume: seven closed calls reused; only first paired inquiry."""
import ctypes
from ctypes import wintypes
import faulthandler
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import psutil
import meth510_operations as base

ROOT,DOC,write,manifest=base.ROOT,base.DOC,base.write,base.manifest
BIND=DOC/'meth510_r1_binding.json'
class Context(base.Context):
    def __init__(self,kind):
        assert kind in ('main','audit') and sys.flags.optimize==0
        self.start=time.monotonic();self.proc=psutil.Process();self.proc.cpu_affinity([10]);self.kind=kind
        self.out=ROOT/'results/native_expert_scaling'/('meth510_r1_'+kind);self.out.mkdir(exist_ok=False)
        self.destination=DOC/('meth510_r1_'+kind+'_result.json');assert not self.destination.exists() and not self.destination.with_suffix('.failure.json').exists()
        self.peak=self.hashed=0;self.child=None;self.native_peak=0;self.closed=threading.Event();self.lock=threading.Lock()
        self.progress=(self.out/'progress.jsonl').open('x',encoding='utf8');self.fatal=(self.out/'fatal.log').open('x',encoding='utf8');faulthandler.enable(self.fatal)
        self.r={'experiment':'METH510-R1 unchanged head; closed calls reused; idle process identity guard','kind':kind,'argv':sys.argv,
                'process_instance':{'pid':self.proc.pid,'create_time_unix':self.proc.create_time()},'started_utc':base.stamp(),'affinity':[10],'commands':[],'gates':{}}
        self.mem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        self.mem.argtypes=[wintypes.HANDLE,ctypes.POINTER(base.Memory),wintypes.DWORD];self.mem.restype=wintypes.BOOL
        self.times=ctypes.WinDLL('kernel32',use_last_error=True).GetProcessTimes
        self.times.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;self.times.restype=wintypes.BOOL
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def guard(self):
        # Take the process reference atomically before a concurrent terminal clear.
        m=self.proc.memory_info();self.peak=max(self.peak,m.rss,m.peak_wset)
        assert self.peak<=2<<30 and time.monotonic()-self.start<=360
        size=sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        if self.destination.exists():size+=self.destination.stat().st_size
        assert size<=512<<20,('output_bytes',size)
        child=self.child
        if child is not None:
            v=base.Memory();v.cb=ctypes.sizeof(v);assert self.mem(wintypes.HANDLE(int(child._handle)),ctypes.byref(v),v.cb)
            self.native_peak=max(self.native_peak,int(v.PeakWorkingSetSize));assert self.native_peak<=1<<30
    def admit(self,sha):
        base.BIND=BIND;b=super().admit(sha)
        self.first_fault=json.loads(Path(b['resume']['original_failure']['path']).read_bytes());self.reused=[]
        self.r['original_first_fault_sha256']=b['resume']['original_failure']['sha256']
        assert len(self.first_fault['commands'])==7
        assert not (ROOT/'results/native_expert_scaling/meth510_main/head_outputs.bin').exists()
        return b
    def run(self,argv,label,expected=0,timing=False):
        if label!='paired_head':
            assert self.kind=='main' and not timing
            record=self.first_fault['commands'][len(self.reused)];assert record['label']==label and record['returncode']==expected
            old=ROOT/'results/native_expert_scaling/meth510_main'
            if label=='compile':
                assert list(map(str,argv))[:-1]==record['argv'][:-1]
                shutil.copyfile(old/'meth510.exe',Path(argv[-1]))
                assert self.digest(Path(argv[-1]))==self.binding['resume']['compiled_binary']['sha256']
            else:assert list(map(str,argv))[1:]==record['argv'][1:]
            for suffix in ('stdout','stderr'):shutil.copyfile(old/(label+'.'+suffix),self.out/(label+'.'+suffix))
            retained={**record,'retained_from_first_fault':True};self.r['commands'].append(retained);self.reused.append(label)
            self.log(reused_terminal_original_command=label)
            return (self.out/(label+'.stdout')).read_text(encoding='utf8').splitlines()
        assert len(self.reused)==7 and self.digest(self.out/'head_inputs.bin')==self.binding['resume']['packed_inputs']['sha256']
        return super().run(argv,label,expected,timing)
    def scientific_processes(self,exclude):
        allowed={(v['pid'],v['create_time_unix']):v for v in self.binding['idle_processes']};foreign=[];kept=[]
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in exclude:continue
            name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                kept.append({'pid':p.pid,'create_time_unix':p.create_time(),'cpu_seconds':sum(p.cpu_times()[:2])});continue
            identity=(p.pid,p.create_time()) if p.pid in {v[0] for v in allowed} else None
            if identity in allowed:
                row=allowed[identity];assert p.name()==row['name'] and p.exe()==row['exe']
                descendants=sorted((v.pid,v.create_time()) for v in p.children(recursive=True))
                assert descendants==sorted(tuple(v) for v in row['descendants']),'idle_app_descendant_change'
                kept.append({'pid':p.pid,'create_time_unix':p.create_time(),'cpu_seconds':sum(p.cpu_times()[:2]),'bound_idle_app':True})
            elif name.startswith(('python','clang','llama','ollama')) or (name.startswith('meth') and name.endswith('.exe')):
                foreign.append({'pid':p.pid,'create_time_unix':p.create_time(),'name':name})
        assert {v['pid'] for v in kept if v.get('bound_idle_app')}=={v['pid'] for v in self.binding['idle_processes']}
        # Every intermediate sample must equal the first CPU counter, not only ends.
        if not hasattr(self,'idle_start_CPU'):self.idle_start_CPU={v['pid']:v['cpu_seconds'] for v in kept}
        assert all(v['cpu_seconds']==self.idle_start_CPU[v['pid']] for v in kept),'idle_process_CPU_delta'
        return foreign,kept
