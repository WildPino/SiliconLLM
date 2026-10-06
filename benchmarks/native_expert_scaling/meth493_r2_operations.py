"""Fresh read dependencies, historical provenance retained separately; same hard limits."""
import meth493_operations as inherited
from meth493_operations import ROOT, DOC, write, stamp

inherited.operations.BIND=DOC/'meth493_r2_binding.json'

class Context(inherited.Context):
    def __init__(self,out,raw,seconds,bytes_limit):
        super().__init__(out,raw,seconds,bytes_limit)
        self.r['experiment']='METH493-R2 complete weighted targets with fresh direct dependency catalog'
    def guard(self):
        inherited.operations.Context.guard(self)
        total=0
        for prefix in ('meth493','meth493_r1','meth493_r2'):
            for suffix in ('weighted_targets','retention'):
                folder=ROOT/'results/native_expert_scaling'/(prefix+'_'+suffix)
                if folder.exists(): total+=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        for pattern in ('meth493*.json','RETENTION_493*.json','ADMISSION_493*.json'):
            total+=sum(p.stat().st_size for p in DOC.glob(pattern))
        total+=sum(p.stat().st_size for p in (ROOT/'results/native_expert_scaling').glob('meth493*windows_terminal.json'))
        assert total<=128<<20,'all_original_R1_R2_outputs128MiB'
