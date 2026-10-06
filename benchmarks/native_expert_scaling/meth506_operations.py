"""506 operational bounds and exclusive receipts; no model mathematics."""
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
import meth490_r1_operations as base

ROOT,DOC,write,stamp=base.ROOT,base.DOC,base.write,base.stamp
BIND=DOC/'meth506_r2_binding.json'
OUT=ROOT/'results/native_expert_scaling/meth506_whole'
PREP=ROOT/'results/native_expert_scaling/meth506_r2_artifact'
PREPRAW=DOC/'meth506_r2_preparation_result.json'
AUDIT=ROOT/'results/native_expert_scaling/meth506_retention'
RAW=DOC/'meth506_whole_result.json'

class Context(base.Context):
    def __init__(self,out,raw):
        self.last_size=-1.
        super().__init__(out,raw,3600,24<<30)
        psutil.Process().cpu_affinity([10])
        assert psutil.Process().cpu_affinity()==[10]
        self.r['experiment']='METH506 whole original-I8 zero-only column-WO artifact'
        self.r['observer_and_Python_affinity']=[10]
    def deadline(self):self.abort(RuntimeError('METH506 hard3600s'))
    def guard(self):
        m=psutil.Process().memory_info();self.peak=max(self.peak,m.rss,getattr(m,'peak_wset',0))
        child=self.child
        if child is not None:
            try:self.child_peak=max(self.child_peak,self.process_memory(child))
            except (OSError,AssertionError):
                if child.poll() is None:raise
        assert self.peak+self.child_peak<=self.bytes_limit,('combined_OS_peak',self.peak,self.child_peak)
        assert time.monotonic()-self.start<=self.seconds,('wall_bound',self.phase)
        now=time.monotonic()
        if now-self.last_size>=1:
            size=sum(p.stat().st_size for d in [ROOT/'results/native_expert_scaling/meth506_artifact',PREP,OUT,AUDIT] if d.exists() for p in d.iterdir() if p.is_file())
            size+=sum(p.stat().st_size for p in DOC.glob('meth506*.json'))
            assert size<=24<<30,('all_new_outputs',size)
            self.last_size=now
    def admit(self,sha):
        self.quiet('initial');assert self.digest(BIND)==sha;self.head(BIND)
        b=json.loads(BIND.read_bytes());self.binding=b;self.r['binding_sha256']=sha
        assert sys.version==b['runtime']['python'] and str(Path(sys.executable).resolve())==b['runtime']['executable']
        for name,v in b['runtime']['packages'].items():assert importlib.metadata.version(name)==v['version']
        for v in b['catalog']:self.exact(v)
        for v in b['scientific']+b['records']:self.head(v['path'])
        deriv=json.loads((DOC/'meth506_source_derivation.json').read_bytes())
        self.exact(deriv['old_model']|{'bytes':Path(deriv['old_model']['path']).stat().st_size})
        original=Path(deriv['old_model']['path']).read_text(encoding='utf8')
        for replacement in deriv['model_replacements']:
            assert original.count(replacement['before'])==replacement['occurrences']
            original=original.replace(replacement['before'],replacement['after'])
        assert original==Path(deriv['new_model']['path']).read_text(encoding='utf8')
        assert self.digest(deriv['new_model']['path'])==deriv['new_model']['sha256']
        engine=(ROOT/'benchmarks/phase60/engine.c').read_bytes().replace(b'\r\n',b'\n')
        oldengine=subprocess.check_output(['git','show',deriv['old_engine_revision']+':benchmarks/phase60/engine.c'],cwd=ROOT).replace(b'\r\n',b'\n')
        assert engine.split(b'\n',3)[3]==oldengine.split(b'\n',1)[1]
        assert self.digest(ROOT/'benchmarks/phase60/engine.c')==deriv['engine_new_sha256']
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status),status
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        self.r['source_freeze']=b['freeze_head'];self.r['execution_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.r['gates']['actual_source_runtime_protocol_before_numerical_import']=True
        self.module_catalog={str(Path(v['path']).resolve()).lower():v for v in b['catalog']}
        return b
    def run(self,argv,label,expected=0):
        lines=super().run(argv,label,expected)
        record=self.r['commands'][-1]
        for path in record['observed_modules']:
            p=Path(path).resolve();key=str(p).lower()
            if p.parent==self.out.resolve() and p.name in ['meth506.exe','libomp.dll']:
                self.r.setdefault('built_modules',{})[str(p)]=self.digest(p)
            else:
                assert key in self.module_catalog,('unbound_actual_module',str(p))
                self.exact(self.module_catalog[key])
        record['observed_modules_prebound_or_actual_build']=True
        return lines
    def parent_modules(self):
        modules=sorted({m.path for m in psutil.Process().memory_maps() if m.path.lower().endswith(('.dll','.exe','.pyd'))})
        for path in modules:
            key=str(Path(path).resolve()).lower()
            assert key in self.module_catalog,('unbound_actual_parent_module',path)
            self.exact(self.module_catalog[key])
        self.r['actual_parent_modules']=modules
    def import_diagnostic(self):
        data=(self.out/'fatal_native.log').read_bytes()
        if data:
            text=data.decode('utf8');assert text.count('Windows fatal exception: access violation')==1
            current=text.split('Current thread ',1)[1]
            assert 'line 1293 in create_module' in '\n'.join(current.splitlines()[:4])
            assert 'site-packages\\pyarrow\\__init__.py", line 71 in <module>' in current
            self.r['import_diagnostic']={'bytes':len(data),'sha256':self.digest(self.out/'fatal_native.log'),'scope':'One handled first-chance access-violation diagnostic during PyArrow module import, before cohort/export/native/model; cause unidentified. Not a zero-exception claim.'}
        else:self.r['import_diagnostic']={'bytes':0,'scope':'No faulthandler import diagnostic.'}
        self.import_fatal_bytes=data
    def unchanged_import_diagnostic(self):
        assert (self.out/'fatal_native.log').read_bytes()==self.import_fatal_bytes,'new_fatal_diagnostic_after_import'
    def finish(self,value):
        self.unchanged_import_diagnostic()
        r=super().finish(value)
        write(self.out/'terminal_resources.json',{'raw_sha256':self.digest(self.raw),'resource':self.resources(),'scope':'After full RAW serialization.'})
        self.guard();return r
