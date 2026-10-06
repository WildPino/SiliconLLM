"""Metadata-only donor import admission repair; completed native stage stays R7."""
import ast,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;records=[]
def derive(suffix,replacements,extension='.py'):
    old=B/('meth511_r7_'+suffix+extension);new=B/('meth511_r8_'+suffix+extension);text=old.read_text(encoding='utf8');changes=[]
    for before,after in replacements:
        n=text.count(before);assert n>0,(suffix,before);text=text.replace(before,after);changes.append({'before':before,'after':after,'occurrences':n})
    if extension=='.py':ast.parse(text,filename=str(new))
    with new.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    records.append({'old':str(old),'old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new':str(new),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'replacements':changes})
derive('operations',[
    ('meth511_r7_binding.json','meth511_r8_binding.json'),("('meth511_r7_'+kind)","('meth511_r8_'+kind)"),("'meth511_r7_'","'meth511_r8_'"),
    ("'donor':(3600,24<<30,3<<30)","'donor':(3540,24<<30,3<<30)"),
    ("('meth511_r3_' if kind=='cohort' else 'meth511_r8_')", "('meth511_r3_' if kind=='cohort' else 'meth511_r7_' if kind=='native' else 'meth511_r8_')"),
    ("(self.binding['retained_cohort']['binding_sha256'] if kind=='cohort' else self.r['binding_sha256'])", "(self.binding['retained_cohort']['binding_sha256'] if kind=='cohort' else self.binding['retained_completed_native']['binding_sha256'] if kind=='native' else self.r['binding_sha256'])"),
])
for suffix in ['donor','retention_audit']:
    derive(suffix,[('import meth511_r7_operations as O','import meth511_r8_operations as O')])
derive('windows_terminal',[
    ('meth511_r7_windows_terminal.json','meth511_r8_windows_terminal.json'),
    ("$prefix=if ($kind -eq 'cohort') {'meth511_r3_'} else {'meth511_r7_'}", "$prefix=if ($kind -eq 'cohort') {'meth511_r3_'} elseif ($kind -eq 'native') {'meth511_r7_'} else {'meth511_r8_'}"),
],extension='.ps1')
with (B/'meth511_r8_derivation.json').open('x',encoding='utf8') as f:json.dump(records,f,indent=2);f.write('\n')
