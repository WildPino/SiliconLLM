"""508 extent admission and terminal receipts; reuse frozen507 operational helpers."""
import datetime
import faulthandler
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
import psutil
import meth507_operations as base

ROOT,DOC,write=base.ROOT,base.DOC,base.write
BIND=DOC/'meth508_binding.json'

def manifest(path):
    data=Path(path).read_bytes();assert data[:8]==b'SWI8C001'
    cfg=struct.unpack_from('<13If',data,8);nf,nt=struct.unpack_from('<2I',data,64);offset=72
    def string():
        nonlocal offset
        n=struct.unpack_from('<I',data,offset)[0];offset+=4
        value=data[offset:offset+n].decode('utf8');offset+=n;return value
    files=[string() for _ in range(nf)];tensors={}
    for _ in range(nt):
        name=string();row=struct.unpack_from('<5I3Q',data,offset);offset+=44
        assert name not in tensors;tensors[name]=row
    assert offset==len(data) and nf==1 and nt==3320
    return cfg,files,tensors

class Context(base.Context):
    def __init__(self,kind):
        assert kind in ('main','audit')
        self.start=time.monotonic();self.proc=psutil.Process();self.proc.cpu_affinity([10]);self.kind=kind
        self.out=ROOT/'results/native_expert_scaling'/('meth508_'+kind);self.out.mkdir(exist_ok=False)
        self.destination=DOC/('meth508_'+kind+'_result.json');assert not self.destination.exists()
        self.peak=self.hashed=0;self.closed=threading.Event();self.lock=threading.Lock()
        self.progress=(self.out/'progress.jsonl').open('x',encoding='utf8');self.fatal=(self.out/'fatal.log').open('x',encoding='utf8')
        faulthandler.enable(self.fatal)
        self.r={'experiment':'METH508 source-head fixed-state decomposition','kind':kind,'argv':sys.argv,
                'process_instance':{'pid':self.proc.pid,'create_time_unix':self.proc.create_time()},
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'whole_model_native_compiler_calls':0,
                'affinity':[10],'gates':{}}
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()

    def extent(self,row):
        import hashlib
        p=Path(row['path']);before=p.stat();assert before.st_size==row['file_bytes']
        h=hashlib.sha256();left=row['bytes']
        with p.open('rb') as f:
            f.seek(row['offset'])
            while left:
                data=f.read(min(left,8<<20));assert data;h.update(data);left-=len(data);self.hashed+=len(data);self.guard()
        assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
        assert h.hexdigest()==row['sha256']
        return {'file_bytes':before.st_size,'mtime_ns':before.st_mtime_ns}

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
        self.r['gates']['ALL_frozen_selected_data_runtime_AND_actual_F32_source_extent_before_evaluation']=True
        return b

    def finish(self,value):
        self.modules();self.guard();self.progress.flush();self.fatal.flush()
        assert (self.out/'fatal.log').stat().st_size==0
        self.r['gates']['no_whole_model_native_compiler_calls_and_empty_fatal_log']=True
        self.r.update(value);self.r['resource_before_serialization']={'seconds':time.monotonic()-self.start,'OS_peak_bytes':self.peak,'bytes_hashed':self.hashed}
        self.r['ended_compute_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(self.destination,self.r)
        digest=self.digest(self.destination);self.guard()
        source=Path(value['source_head']['path']).stat()
        assert (source.st_size,source.st_mtime_ns)==(self.head_snapshot['file_bytes'],self.head_snapshot['mtime_ns'])
        self.log(terminal_result_serialized_and_hashed=True)
        self.guard()
        terminal={'scope':'OS peak and elapsed through main/audit result serialization, SHA, terminal guard and final input-stat check; process exit follows.',
                  'result_sha256':digest,'seconds':time.monotonic()-self.start,'OS_peak_bytes':self.peak,'bytes_hashed':self.hashed,
                  'limits':{'seconds':180,'OS_peak_bytes':2<<30,'output_bytes':32<<20},
                  'process_instance':self.r['process_instance'],'gates':self.r['gates'],'summary':value['summary']}
        write(self.out/'terminal_resource.json',terminal);self.guard()
        print(json.dumps({'result':str(self.destination),'sha256':digest,'gates':self.r['gates'],
                          'terminal_resource':terminal,'summary':value['summary']}),flush=True)
