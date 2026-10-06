"""R4 donor-only recovery; compiled/native observations are immutable and never replayed."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import psutil
import meth506_operations as previous
ROOT,DOC,write,stamp=previous.ROOT,previous.DOC,previous.write,previous.stamp
BIND=DOC/'meth506_r4_2_binding.json'
PREP,PREPRAW=previous.PREP,previous.PREPRAW
NATIVE=previous.OUT
OUT=ROOT/'results/native_expert_scaling/meth506_r4_2_quality'
AUDIT=ROOT/'results/native_expert_scaling/meth506_r4_2_retention'
RAW=DOC/'meth506_r4_2_result.json'

FIXED_FOLDERS=tuple(ROOT/'results/native_expert_scaling'/n for n in [
    'meth506_artifact','meth506_r2_artifact','meth506_r3_artifact',
    'meth506_whole','meth506_retention','meth506_r4_quality'])

def fixed_snapshot():
    return tuple(sorted((str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns)
                        for d in FIXED_FOLDERS if d.exists()
                        for p in d.iterdir() if p.is_file()))

class Context(previous.Context):
    def __init__(self,out,raw):
        began=time.monotonic()
        self.fixed_outputs=fixed_snapshot()
        self.fixed_output_bytes=sum(v[1] for v in self.fixed_outputs)
        self.output_guard_lock=threading.Lock()
        self.last_extra_size=-1.
        super().__init__(out,raw);self.start=began
        self.r['experiment']='METH506-R4.2 original4.57.6 F32 donor-only recovery'
        self.r['new_compile_native_calls']=0
        self.r['immutable_output_charge']={'files':len(self.fixed_outputs),'bytes':self.fixed_output_bytes,
            'scope':'Charge immutable completed namespaces once; all bound input SHAs still validated before imports. Recheck complete path/size/mtime inventory at terminal. Dynamic current quality/audit files and metadata scanned each second.'}
    def guard(self):
        # Same memory/deadline predicates as506, without enumerating completed outputs.
        m=psutil.Process().memory_info();self.peak=max(self.peak,m.rss,getattr(m,'peak_wset',0))
        child=self.child
        if child is not None:
            try:self.child_peak=max(self.child_peak,self.process_memory(child))
            except (OSError,AssertionError):
                if child.poll() is None:raise
        assert self.peak+self.child_peak<=self.bytes_limit,('combined_OS_peak',self.peak,self.child_peak)
        assert time.monotonic()-self.start<=self.seconds,('wall_bound',self.phase)
        now=time.monotonic()
        if now-self.last_extra_size>=1 and self.output_guard_lock.acquire(blocking=False):
            try:
                if now-self.last_extra_size>=1:
                    self.last_extra_size=now
                    extra=sum(p.stat().st_size for d in [OUT,AUDIT] if d.exists() for p in d.iterdir() if p.is_file())
                    metadata=sum(p.stat().st_size for p in DOC.glob('meth506*.json'))
                    assert extra+self.fixed_output_bytes+metadata<=24<<30
            finally:self.output_guard_lock.release()
    def finish(self,value):
        assert fixed_snapshot()==self.fixed_outputs,'immutable_completed_output_inventory_changed'
        self.r['gates']['immutable_output_path_size_mtime_terminal_revalidation']=True
        return super().finish(value)
    def admit(self,sha):
        self.quiet('initial');assert self.digest(BIND)==sha;self.head(BIND);b=json.loads(BIND.read_bytes());self.binding=b;self.r['binding_sha256']=sha
        assert sys.version==b['runtime']['python'] and str(Path(sys.executable).resolve())==b['runtime']['executable']
        for name,v in b['runtime']['packages'].items():assert importlib.metadata.version(name)==v['version']
        for v in b['catalog']:self.exact(v)
        for v in b['scientific']+b['records']:self.head(v['path'])
        self.module_catalog={str(Path(v['path']).resolve()).lower():v for v in b['catalog']}
        for rel,sha in b['preserved'].items():assert self.digest(ROOT/rel)==sha
        assert self.digest(ROOT/'benchmarks/phase60/engine.c')==b['engine_sha256']
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines();assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status)
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        self.r['source_freeze']=b['freeze_head'];self.r['execution_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.r['gates']['actual_original4p57p6_runtime_and_ALL_immutable_native_files_before_import']=True
        return b
    def run(self,*args,**kwargs):raise RuntimeError('506-R4 forbids compile/native subprocess replay')
