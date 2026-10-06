"""Actual-input admission, bounded native execution and full terminal receipts."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
import meth494_operations as base
ROOT,DOC,write,stamp=base.ROOT,base.DOC,base.write,base.stamp
BIND=DOC/'meth498_binding.json'
class Context(base.Context):
    def __init__(self,out,raw):
        self.last_size_check=-1.;super().__init__(out,raw,900,1536<<20)
        self.r['experiment']='METH498 minimum-source-prior-displacement readout and physical evaluation'
    def deadline(self):self.abort(RuntimeError('METH498 hard wall deadline'))
    def guard(self,force=False):
        info=psutil.Process().memory_info();self.peak=max(self.peak,info.rss,getattr(info,'peak_wset',0));child=self.child
        if child is not None:
            try:
                group=self.process_memory(child)
                try:
                    for p in psutil.Process(child.pid).children(recursive=True):
                        try:
                            key=(p.pid,p.create_time());info=p.memory_info();previous=self.descendants.get(key,{}).get('OS_peak_bytes',0)
                            peak=max(info.rss,getattr(info,'peak_wset',0),previous);group+=peak
                            self.descendants[key]={'pid':key[0],'create_time_unix':key[1],'OS_peak_bytes':peak,'executable':p.exe()}
                        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                self.child_peak=max(self.child_peak,group)
            except (OSError,AssertionError):
                if child.poll() is None:raise
        now=time.monotonic();assert self.peak+self.child_peak<=self.bytes_limit and now-self.start<=self.seconds
        if force or now-self.last_size_check>=1:
            total=0
            for name,limit in (('meth498_minimum_prior_fit',1024<<20),('meth498_retention',256<<20)):
                folder=ROOT/'results/native_expert_scaling'/name
                if folder.exists():size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file());assert size<=limit;total+=size
            for pattern in ('meth498*.json','RETENTION_498*.json','ADMISSION_498*.json'):total+=sum(p.stat().st_size for p in DOC.glob(pattern))
            total+=sum(p.stat().st_size for p in (ROOT/'results/native_expert_scaling').glob('meth498*windows_terminal.json'))
            assert total<=1280<<20;self.last_size_check=now
    def admit(self,sha):
        self.quiet('initial');assert self.digest(BIND)==sha;self.head(BIND);b=json.loads(BIND.read_bytes())
        self.r['binding_sha256']=sha;self.inherited_output_bytes=0
        assert sys.version==b['runtime']['python'] and str(Path(sys.executable).resolve())==b['runtime']['executable']
        for name,v in b['runtime']['packages'].items():assert importlib.metadata.version(name)==v['version']
        for v in b['catalog']:self.exact(v)
        for v in b['scientific']+b['records']:self.head(v['path'])
        for rel,value in b['preserved'].items():assert self.digest(ROOT/rel)==value
        assert self.digest(ROOT/'benchmarks/phase60/engine.c')==b['engine_sha256']
        admission=json.loads(Path(b['source_admission']['path']).read_bytes());assert admission['main_completed'] and all(admission['gates'].values())
        assert admission['decision']=='FINITE_DEVELOPMENT_INTERPOLATION_AND_ALL_CONSUMED_FEATURE_NOVELTY_CERTIFIED'
        rank=json.loads(Path(admission['raw']['path']).read_bytes());assert rank['summary']['development_full_nonempty']==rank['summary']['ALL_full_nonempty']==127
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status),status
        self.r['source_freeze']=b['freeze_head'];self.r['execution_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.r['gates']['ALL_actual_inputs_runtime_native_source_and497_exact_rank_qualified_before_NumPy']=True
        self.guard(force=True);return b
    def finish(self,value):
        self.guard(force=True);result=super().finish(value);self.guard(force=True);return result
