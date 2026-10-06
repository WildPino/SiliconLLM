"""Metadata only: exact named external fetch allowed; all native children prohibited."""
import datetime,json,os,time
from pathlib import Path
import psutil
import meth505_r1_operations as base
ROOT,DOC,write=base.ROOT,base.DOC,base.write
base.parent.BIND=DOC/'meth505_r2_binding.json'
FETCH=['-X','utf8','-B','scripts/progressive_ternary/kaggle_run.py','fetch','--account','acct3','--ref','sirwildpino/pqt-src-007-switch-float-20261006-003','--destination','D:/_THINGS/Progetti/SiliconLLM-progressive-runtime/research-artifacts/PQT-SRC-007/remote_003','--out','results/progressive_ternary/PQT-SRC-007/remote_operations_003/kernel_fetch.json']
EXECUTABLES={str(Path(p).resolve()).lower() for p in ['D:/_THINGS/Progetti/SiliconLLM-progressive-runtime/.venv/Scripts/python.exe','C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe']}
class Context(base.Context):
    def __init__(self,out,raw):
        self.last_second_repair_size=-1.
        super().__init__(out,raw);self.r['experiment']='METH505-R2 retained physical inquiry metadata recovery'
    def run(self,*args,**kwargs):raise RuntimeError('METH505-R2 prohibits all native/compile/timing children')
    def quiet(self,label):
        original=json.loads((DOC/'meth505_exact_sparse_result.failure.json').read_bytes())
        end=datetime.datetime.fromisoformat(original['commands'][-1]['end_utc']).timestamp();began=time.monotonic();observations=[]
        while True:
            own={os.getpid(),*(p.pid for p in psutil.Process().parents())};foreign=[];kept=[];downloads=[]
            for p in psutil.process_iter(['name','cmdline']):
                if p.pid in own:continue
                name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
                if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():kept.append(p.pid);continue
                if argv and str(Path(argv[0]).resolve()).lower() in EXECUTABLES and argv[1:]==FETCH:
                    assert p.create_time()>end;downloads.append({'pid':p.pid,'create_time_unix':p.create_time(),'argv':argv,'role':'known external fetch, started after original native cost terminal'});continue
                if name.startswith(('python','clang')) or (name.startswith('meth') and name.endswith('.exe')):foreign.append({'pid':p.pid,'name':name})
            if not foreign:break
            self.guard();assert time.monotonic()-began<=240;observations.append(foreign);time.sleep(1)
        self.r['preserved_daemons']=kept;self.r['permitted_metadata_downloads']=downloads
        if observations:self.r['isolation_waits'].append({'label':label,'seconds':time.monotonic()-began,'observations':observations})
    def guard(self):
        super().guard();now=time.monotonic()
        if now-self.last_second_repair_size>=1:
            folders=['meth505_exact_sparse','meth505_retention','meth505_r1_recovery','meth505_r1_retention','meth505_r2_recovery','meth505_r2_retention']
            total=0
            for name in folders:
                folder=ROOT/'results/native_expert_scaling'/name
                size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()) if folder.exists() else 0
                if name in ['meth505_r2_recovery','meth505_r2_retention']:assert size<=4<<20
                total+=size
            total+=sum(p.stat().st_size for p in DOC.glob('meth505*.json'));assert total<=1<<30
            self.last_second_repair_size=now
