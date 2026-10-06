"""Minimal actual-input admission and resource guards; no decomposition mathematics."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
import meth494_operations as base

ROOT,DOC,write,stamp=base.ROOT,base.DOC,base.write,base.stamp
BIND=DOC/'meth496_binding.json'
class Context(base.Context):
    def __init__(self,out,raw):
        self.last_size_check=-1.
        super().__init__(out,raw,300,768<<20)
        self.r['experiment']='METH496 fixed-artifact fitted/coefficient/arithmetic readout error'
    def deadline(self):self.abort(RuntimeError('METH496 hard wall deadline'))
    def guard(self,force=False):
        info=psutil.Process().memory_info(); self.peak=max(self.peak,info.rss,getattr(info,'peak_wset',0))
        assert self.child is None and self.child_peak==0
        now=time.monotonic(); assert self.peak<=self.bytes_limit and now-self.start<=self.seconds
        if force or now-self.last_size_check>=1:
            total=0
            for name in ('meth496_readout_error','meth496_retention'):
                folder=ROOT/'results/native_expert_scaling'/name
                if folder.exists():
                    size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()); assert size<=64<<20; total+=size
            for pattern in ('meth496*.json','RETENTION_496*.json','ADMISSION_496*.json'):total+=sum(p.stat().st_size for p in DOC.glob(pattern))
            total+=sum(p.stat().st_size for p in (ROOT/'results/native_expert_scaling').glob('meth496*windows_terminal.json'))
            assert total<=128<<20; self.last_size_check=now
    def admit(self,sha):
        self.quiet('initial'); assert self.digest(BIND)==sha; self.head(BIND); b=json.loads(BIND.read_bytes())
        self.r['binding_sha256']=sha; self.inherited_output_bytes=0
        assert sys.version==b['runtime']['python'] and str(Path(sys.executable).resolve())==b['runtime']['executable']
        for name,v in b['runtime']['packages'].items():assert importlib.metadata.version(name)==v['version']
        for v in b['catalog']:self.exact(v)
        for v in b['scientific']+b['records']:self.head(v['path'])
        for rel,value in b['preserved'].items():assert self.digest(ROOT/rel)==value
        assert self.digest(ROOT/'benchmarks/phase60/engine.c')==b['engine_sha256']
        admission=json.loads(Path(b['source_admission']['path']).read_bytes())
        assert all(admission['gates'].values()) and admission['main_completed']
        assert admission['decision']=='FIXED_HYBRID_LOCAL_RECIPE_FAIL_INDEPENDENTLY_ADMITTED'
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status),status
        self.r['source_freeze']=b['freeze_head']; self.r['execution_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.r['gates']['ALL_actual_input_runtime_current_helpers_and_source495_admission_frozen']=True
        self.guard(force=True);return b
    def finish(self,value):
        self.guard(force=True); result=super().finish(value); self.guard(force=True);return result
