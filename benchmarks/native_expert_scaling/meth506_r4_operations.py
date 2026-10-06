"""R4 donor-only recovery; compiled/native observations are immutable and never replayed."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import meth506_operations as previous
ROOT,DOC,write,stamp=previous.ROOT,previous.DOC,previous.write,previous.stamp
BIND=DOC/'meth506_r4_binding.json'
PREP,PREPRAW=previous.PREP,previous.PREPRAW
NATIVE=previous.OUT
OUT=ROOT/'results/native_expert_scaling/meth506_r4_quality'
AUDIT=ROOT/'results/native_expert_scaling/meth506_r4_retention'
RAW=DOC/'meth506_r4_result.json'

class Context(previous.Context):
    def __init__(self,out,raw):
        super().__init__(out,raw);self.r['experiment']='METH506-R4 original4.57.6 F32 donor-only recovery'
        self.r['new_compile_native_calls']=0
    def guard(self):
        super().guard()
        extra=sum(p.stat().st_size for d in [OUT,AUDIT] if d.exists() for p in d.iterdir() if p.is_file())
        original=sum(p.stat().st_size for d in [ROOT/'results/native_expert_scaling/meth506_artifact',ROOT/'results/native_expert_scaling/meth506_r2_artifact',PREP,NATIVE,previous.AUDIT] if d.exists() for p in d.iterdir() if p.is_file())
        metadata=sum(p.stat().st_size for p in DOC.glob('meth506*.json'))
        assert extra+original+metadata<=24<<30
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
