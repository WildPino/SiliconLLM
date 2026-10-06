"""495 resource/receipt orchestration only; no learning or candidate arithmetic."""
from pathlib import Path
import time
import psutil
import meth494_operations as base

ROOT,DOC,write,stamp=base.ROOT,base.DOC,base.write,base.stamp
base.operations.BIND=DOC/'meth495_binding.json'
class Context(base.Context):
    def __init__(self,out,raw):
        super().__init__(out,raw,1200,1536<<20)
        self.r['experiment']='METH495 fixed hybrid weighted function and Cartesian control learning'
    def admit(self,sha): return base.operations.Context.admit(self,sha)
    def deadline(self): self.abort(RuntimeError('METH495 hard wall deadline'))
    def guard(self):
        info=psutil.Process().memory_info(); self.peak=max(self.peak,info.rss,getattr(info,'peak_wset',0))
        child=self.child
        if child is not None:
            try:
                group=self.process_memory(child)
                try:
                    for p in psutil.Process(child.pid).children(recursive=True):
                        try:
                            key=(p.pid,p.create_time()); info=p.memory_info(); previous=self.descendants.get(key,{}).get('OS_peak_bytes',0)
                            peak=max(info.rss,getattr(info,'peak_wset',0),previous); group+=peak
                            self.descendants[key]={'pid':key[0],'create_time_unix':key[1],'OS_peak_bytes':peak,'executable':p.exe()}
                        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                self.child_peak=max(self.child_peak,group)
            except (OSError,AssertionError):
                if child.poll() is None: raise
        assert self.peak+self.child_peak<=self.bytes_limit and time.monotonic()-self.start<=self.seconds
        total=0
        for foldername in ('meth495_weighted_hybrid_fit','meth495_retention'):
            folder=ROOT/'results/native_expert_scaling'/foldername
            if folder.exists():
                size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()); assert size<=768<<20; total+=size
        for pattern in ('meth495*.json','RETENTION_495*.json','ADMISSION_495*.json'): total+=sum(p.stat().st_size for p in DOC.glob(pattern))
        total+=sum(p.stat().st_size for p in (ROOT/'results/native_expert_scaling').glob('meth495*windows_terminal.json'))
        assert total<=1<<30,'all495_outputs1GiB'
