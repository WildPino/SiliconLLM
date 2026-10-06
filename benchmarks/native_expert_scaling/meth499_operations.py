"""Bounded actual-input admission, new native child accounting and terminal receipts."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
import meth494_operations as base
ROOT,DOC,write,stamp=base.ROOT,base.DOC,base.write,base.stamp
BIND=DOC/'meth499_binding.json'
class Context(base.Context):
    def __init__(self,out,raw):
        self.last_size_check=-1.;super().__init__(out,raw,900,1536<<20)
        self.r['experiment']='METH499 unweighted donor-prior functions with explicit variable mass'
    def deadline(self):self.abort(RuntimeError('METH499 hard wall deadline'))
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
            for name,limit in (('meth499_factor_fit',1408<<20),('meth499_retention',128<<20)):
                folder=ROOT/'results/native_expert_scaling'/name
                if folder.exists():size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file());assert size<=limit;total+=size
            for pattern in ('meth499*.json','RETENTION_499*.json','ADMISSION_499*.json'):total+=sum(p.stat().st_size for p in DOC.glob(pattern))
            total+=sum(p.stat().st_size for p in (ROOT/'results/native_expert_scaling').glob('meth499*windows_terminal.json'))
            assert total<=1536<<20;self.last_size_check=now
    def admit(self,sha):
        self.quiet('initial');assert len(sha)==64 and self.digest(BIND)==sha;self.head(BIND);b=json.loads(BIND.read_bytes())
        self.r['binding_sha256']=sha;self.inherited_output_bytes=0
        assert sys.version==b['runtime']['python'] and str(Path(sys.executable).resolve())==b['runtime']['executable']
        for name,v in b['runtime']['packages'].items():assert importlib.metadata.version(name)==v['version']
        for v in b['catalog']:self.exact(v)
        for v in b['scientific']+b['records']:self.head(v['path'])
        for rel,value in b['preserved'].items():assert self.digest(ROOT/rel)==value
        assert self.digest(ROOT/'benchmarks/phase60/engine.c')==b['engine_sha256']
        for name in ('ADMISSION_498_20261006.json','ADMISSION_493_R2_20261006.json','ADMISSION_494_20261006.json','ADMISSION_497_20261006.json'):
            a=json.loads((DOC/name).read_bytes());assert a['main_completed'] and all(a['gates'].values())
        assert json.loads((DOC/'ADMISSION_498_20261006.json').read_bytes())['decision']=='NEXT_SOURCE_PRIOR_TRANSPORT'
        a=json.loads((DOC/'ADMISSION_497_20261006.json').read_bytes());r=json.loads(Path(a['raw']['path']).read_bytes())
        assert r['summary']['development_full_nonempty']==r['summary']['ALL_full_nonempty']==127
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status),status
        self.r['source_freeze']=b['freeze_head'];self.r['execution_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.r['gates']['ALL_actual_references_priors_rank_compiler_runtime_and_scientific_SHA_before_NumPy']=True
        self.guard(force=True);return b
    def finish(self,value):
        self.guard(force=True);result=super().finish(value);self.guard(force=True);return result
