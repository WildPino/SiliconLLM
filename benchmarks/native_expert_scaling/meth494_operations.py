"""Operational limits and direct source extents; no source or prior mathematics."""
import hashlib
import json
from pathlib import Path
import time
import psutil
import meth490_r1_operations as operations

ROOT,DOC,write,stamp=operations.ROOT,operations.DOC,operations.write,operations.stamp
operations.BIND=DOC/'meth494_binding.json'

class Context(operations.Context):
    def __init__(self,out,raw,seconds,bytes_limit):
        super().__init__(out,raw,seconds,bytes_limit)
        self.r['experiment']='METH494 hybrid full-rank donor linear term and Gaussian shared-even prior'
    def deadline(self): self.abort(RuntimeError('METH494 hard wall deadline'))
    def guard(self):
        proc=psutil.Process(); info=proc.memory_info()
        self.peak=max(self.peak,info.rss,getattr(info,'peak_wset',0))
        assert self.child is None and self.child_peak==0
        assert self.peak<=self.bytes_limit and time.monotonic()-self.start<=self.seconds
        total=0
        for name in ('meth494_hybrid_prior','meth494_retention'):
            folder=ROOT/'results/native_expert_scaling'/name
            if folder.exists():
                size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
                assert size<=768<<20; total+=size
        for pattern in ('meth494*.json','RETENTION_494*.json','ADMISSION_494*.json'):
            total+=sum(p.stat().st_size for p in DOC.glob(pattern))
        total+=sum(p.stat().st_size for p in (ROOT/'results/native_expert_scaling').glob('meth494*windows_terminal.json'))
        assert total<=1<<30,'all494_outputs1GiB'
    def extent(self,v):
        p=Path(v['path']); before=p.stat(); assert before.st_size>=v['offset']+v['bytes']
        h=hashlib.sha256(); parts=[]; remaining=v['bytes']
        with p.open('rb') as f:
            f.seek(v['offset'])
            while remaining:
                data=f.read(min(1<<20,remaining)); assert data
                h.update(data); parts.append(data); remaining-=len(data); self.hashed+=len(data); self.guard()
        after=p.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        assert h.hexdigest()==v['sha256'],v['name']
        return b''.join(parts)
    def admit(self,sha):
        b=super().admit(sha)
        self.phase='fresh_ALL_source_bank11_code_scale_router_extents_before_NumPy'
        for v in b['source_extents']: self.extent(v)
        self.r['gates']['ALL512_WI_WO_code_scale_and_router_current_extent_SHA_before_NumPy']=True
        return b
    def finish(self,value):
        self.phase='terminal'; self.log(terminal=True); self.guard()
        for rel,sha in self.binding['preserved'].items(): assert self.digest(ROOT/rel)==sha
        assert self.digest(ROOT/'benchmarks/phase60/engine.c')==self.binding['engine_sha256']
        inventory=[{'path':str(p),'bytes':p.stat().st_size,'sha256':self.digest(p)} for p in sorted(self.out.iterdir()) if p.is_file()]
        result={**self.r,**value,'end_utc':stamp(),'resource':self.resources(),'output_inventory':inventory,
            'terminal_resource_path':str(self.out/'terminal_resources.json')}
        payload=(json.dumps(result,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode()
        assert len(payload)<=8<<20; self.guard()
        with self.raw.open('xb') as f:f.write(payload)
        self.guard(); terminal={'raw_path':str(self.raw),'raw_sha256':self.digest(self.raw),
            'resource':self.resources(),'scope':'After full recordwrite, before small receipt serialization.'}
        write(self.out/'terminal_resources.json',terminal); self.guard()
        self.done.set(); self.watchdog.cancel(); self.watcher.join(timeout=1)
        self.progress.close(); operations.faulthandler.disable(); self.fatal.close(); self.terminal_resources=terminal
        return result
