"""Numbered metadata-only repair; original outputs remain immutable and charged."""
import time
from pathlib import Path
import meth505_operations as parent
ROOT,DOC,write=parent.ROOT,parent.DOC,parent.write
parent.BIND=DOC/'meth505_r1_binding.json'
class Context(parent.Context):
    def __init__(self,out,raw):
        self.last_repair_size=-1.
        super().__init__(out,raw)
        self.r['experiment']='METH505-R1 retained physical inquiry metadata recovery'
    def guard(self):
        super().guard()
        now=time.monotonic()
        if now-self.last_repair_size>=1:
            total=0
            for name,cap in [('meth505_exact_sparse',1020<<20),('meth505_retention',4<<20),('meth505_r1_recovery',8<<20),('meth505_r1_retention',4<<20)]:
                folder=ROOT/'results/native_expert_scaling'/name
                size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()) if folder.exists() else 0
                assert size<=cap;total+=size
            total+=sum(p.stat().st_size for p in DOC.glob('meth505*.json'));assert total<=1<<30
            self.last_repair_size=now
