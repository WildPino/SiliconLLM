"""Numbered source-role schema repair, preserving original apparatus and partials."""
import meth493_operations as inherited
from meth493_operations import ROOT, DOC, write, stamp

inherited.operations.BIND=DOC/'meth493_r1_binding.json'

class Context(inherited.Context):
    def __init__(self,out,raw,seconds,bytes_limit):
        super().__init__(out,raw,seconds,bytes_limit)
        self.r['experiment']='METH493-R1 original472 split mapped to extended479 source role'
    def guard(self):
        inherited.operations.Context.guard(self)
        total=0
        for name in ('meth493_weighted_targets','meth493_retention','meth493_r1_weighted_targets','meth493_r1_retention'):
            folder=ROOT/'results/native_expert_scaling'/name
            if folder.exists(): total+=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        for pattern in ('meth493*.json','RETENTION_493*.json','ADMISSION_493*.json'):
            total+=sum(p.stat().st_size for p in DOC.glob(pattern))
        total+=sum(p.stat().st_size for p in (ROOT/'results/native_expert_scaling').glob('meth493*windows_terminal.json'))
        assert total<=128<<20, 'original_partials_AND_R1_all_outputs128MiB'
