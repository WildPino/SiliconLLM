"""METH505 resource/provenance guards, no numerical or layout implementation."""
import importlib.metadata,json,subprocess,sys,time
from pathlib import Path
import psutil
import meth490_r1_operations as base
ROOT,DOC,write,stamp=base.ROOT,base.DOC,base.write,base.stamp
BIND=DOC/'meth505_binding.json'

class Context(base.Context):
    def __init__(self,out,raw):
        self.last_size=-1.
        super().__init__(out,raw,900,4<<30)
        self.r['experiment']='METH505 exact original WI and zero-only column WO'
    def deadline(self):self.abort(RuntimeError('METH505 hard900s'))
    def guard(self):
        m=psutil.Process().memory_info();self.peak=max(self.peak,m.rss,getattr(m,'peak_wset',0))
        child=self.child
        if child is not None:
            try:
                group=self.process_memory(child)
                for p in psutil.Process(child.pid).children(recursive=True):
                    try:
                        m=p.memory_info();key=(p.pid,p.create_time());v=max(m.rss,getattr(m,'peak_wset',0),self.descendants.get(key,{}).get('OS_peak_bytes',0));group+=v
                        self.descendants[key]={'pid':key[0],'create_time_unix':key[1],'OS_peak_bytes':v,'executable':p.exe()}
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                self.child_peak=max(self.child_peak,group)
            except (OSError,AssertionError,psutil.NoSuchProcess):
                if child.poll() is None:raise
        assert self.peak+self.child_peak<=self.bytes_limit and time.monotonic()-self.start<=self.seconds
        now=time.monotonic()
        if now-self.last_size>=1:
            total=0
            for name,cap in [('meth505_exact_sparse',1020<<20),('meth505_retention',4<<20)]:
                folder=ROOT/'results/native_expert_scaling'/name
                size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()) if folder.exists() else 0
                assert size<=cap,(name,size);total+=size
            total+=sum(p.stat().st_size for p in DOC.glob('meth505*.json'))
            assert total<=1<<30;self.last_size=now
    def admit(self,sha):
        self.quiet('initial');assert self.digest(BIND)==sha;self.head(BIND)
        b=json.loads(BIND.read_bytes());self.binding=b;self.r['binding_sha256']=sha
        assert sys.version==b['runtime']['python'] and str(Path(sys.executable).resolve())==b['runtime']['executable']
        for name,v in b['runtime']['packages'].items():assert importlib.metadata.version(name)==v['version']
        for v in b['catalog']:self.exact(v)
        for v in b['scientific']+b['records']:self.head(v['path'])
        a=json.loads((DOC/'ADMISSION_504_20261006.json').read_bytes());assert a['main_completed'] and all(a['gates'].values())
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(v[:3]==' M ' and v[3:] in b['preserved'] for v in status),status
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        self.r['source_freeze']=b['freeze_head'];self.r['execution_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.r['gates']['actual_data_toolchain_runtime_scientific_SHA_before_NumPy']=True
        return b
    def finish(self,value):
        r=super().finish(value)
        write(self.out/'terminal_resources.json',{'raw_sha256':self.digest(self.raw),'resource':self.resources(),'scope':'After full RAW serialization.'});self.guard()
        return r
