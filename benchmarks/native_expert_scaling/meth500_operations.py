"""METH500 bounds and provenance; no source-function or mask mathematics."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
import meth490_r1_operations as base
ROOT,DOC,write,stamp=base.ROOT,base.DOC,base.write,base.stamp
BIND=DOC/'meth500_binding.json'

class Context(base.Context):
    def __init__(self,out,raw):
        self.last_size=-1.
        super().__init__(out,raw,900,2<<30)
        self.r['experiment']='METH500 ONE four overlapping half-width source experts'
    def deadline(self):self.abort(RuntimeError('METH500 hard900s'))
    def guard(self):
        m=psutil.Process().memory_info();self.peak=max(self.peak,m.rss,getattr(m,'peak_wset',0))
        assert self.child is None and self.peak<=self.bytes_limit and time.monotonic()-self.start<=self.seconds
        now=time.monotonic()
        if now-self.last_size>=1:
            total=0
            for name,cap in [('meth500_overlap',1<<30),('meth500_retention',128<<20)]:
                folder=ROOT/'results/native_expert_scaling'/name
                size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()) if folder.exists() else 0
                assert size<=cap,(name,size);total+=size
            total+=sum(p.stat().st_size for p in DOC.glob('meth500*.json'))
            assert total<=1152<<20;self.last_size=now
    def admit(self,sha):
        self.quiet('initial');assert self.digest(BIND)==sha;self.head(BIND)
        b=json.loads(BIND.read_bytes());self.binding=b;self.r['binding_sha256']=sha
        assert sys.version==b['runtime']['python'] and str(Path(sys.executable).resolve())==b['runtime']['executable']
        for name,v in b['runtime']['packages'].items():assert importlib.metadata.version(name)==v['version']
        for v in b['catalog']:self.exact(v)
        for v in b['scientific']+b['records']:self.head(v['path'])
        a=json.loads((DOC/'ADMISSION_499_20261006.json').read_bytes())
        assert a['main_completed'] and all(a['gates'].values())
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status),status
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        self.r['source_freeze']=b['freeze_head'];self.r['execution_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.r['gates']['actual_source_data_runtime_scientific_SHA_before_NumPy']=True
        return b
    def finish(self,value):
        # Reuse base full output hashing, failure retention and process guards.
        result=super().finish(value)
        terminal={'raw_sha256':self.digest(self.raw),'resource':self.resources(),'scope':'After full RAW serialization.'}
        write(self.out/'terminal_resources.json',terminal);self.guard();self.terminal_resources=terminal
        return result
