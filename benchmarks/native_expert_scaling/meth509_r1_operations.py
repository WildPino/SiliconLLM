"""509 retained-wire/selected-row operations with inherited bounded monitoring."""
import datetime
import faulthandler
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import psutil
import meth508_operations as previous

ROOT,DOC,write=previous.ROOT,previous.DOC,previous.write
BIND=DOC/'meth509_r1_binding.json'
class Context(previous.Context):
    def __init__(self,kind):
        assert kind in ('main','audit')
        self.start=time.monotonic();self.proc=psutil.Process();self.proc.cpu_affinity([10]);self.kind=kind
        self.out=ROOT/'results/native_expert_scaling'/('meth509_r1_'+kind);self.out.mkdir(exist_ok=False)
        self.destination=DOC/('meth509_r1_'+kind+'_result.json');assert not self.destination.exists()
        self.peak=self.hashed=0;self.closed=threading.Event();self.lock=threading.Lock()
        self.progress=(self.out/'progress.jsonl').open('x',encoding='utf8');self.fatal=(self.out/'fatal.log').open('x',encoding='utf8');faulthandler.enable(self.fatal)
        self.r={'experiment':'METH509-R1 same top8 head; actual508 runtime inclusion','kind':kind,'argv':sys.argv,
                'process_instance':{'pid':self.proc.pid,'create_time_unix':self.proc.create_time()},
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'whole_model_native_compiler_calls':0,'affinity':[10],'gates':{}}
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def admit(self,sha):
        assert self.digest(BIND)==sha;b=json.loads(BIND.read_bytes())
        assert sys.version==b['python'] and str(Path(sys.executable).resolve())==b['executable']
        for name,version in b['packages'].items():assert importlib.metadata.version(name)==version
        for row in b['catalog']:self.exact(row)
        self.catalog={str(Path(v['path']).resolve()).lower() for v in b['catalog']}
        for row in b['scientific']:
            p=Path(row['path']);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
        for rel,digest in b['preserved'].items():assert self.digest(ROOT/rel)==digest
        assert self.digest(ROOT/'benchmarks/phase60/engine.c')==b['engine_sha256']
        status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines()
        assert all(line[:3]==' M ' and line[3:] in b['preserved'] for line in status)
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        self.head_snapshot=self.extent(b['source_head'])
        self.r.update(binding_sha256=sha,source_freeze=b['freeze_head'],execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        self.r['gates']['ALL_actual_used_wires_runtime_original_extent_AND_frozen508_oracle']=True
        return b
