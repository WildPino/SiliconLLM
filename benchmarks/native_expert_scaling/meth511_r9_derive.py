"""Literal auditor count repair; no completed numerical audit/model/C replay."""
import ast,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;records=[]
def derive(suffix,replacements,extension='.py'):
    old=B/('meth511_r8_'+suffix+extension);new=B/('meth511_r9_'+suffix+extension);text=old.read_text(encoding='utf8');changes=[]
    for before,after in replacements:
        n=text.count(before);assert n>0,(suffix,before);text=text.replace(before,after);changes.append({'before':before,'after':after,'occurrences':n})
    if extension=='.py':ast.parse(text,filename=str(new))
    with new.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    records.append({'old':str(old),'old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new':str(new),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'replacements':changes})
derive('operations',[
    ('meth511_r8_binding.json','meth511_r9_binding.json'),("('meth511_r8_'+kind)","('meth511_r9_'+kind)"),("'meth511_r8_'","'meth511_r9_'"),
    ("'audit':(1200,4<<30,256<<20)","'audit':(1180,4<<30,256<<20)"),
    ("else 'meth511_r9_')+kind+'_result.json'", "else 'meth511_r8_' if kind=='donor' else 'meth511_r9_')+kind+'_result.json'"),
    ("if kind=='native' else self.r['binding_sha256'])", "if kind=='native' else self.binding['retained_completed_donor']['binding_sha256'] if kind=='donor' else self.r['binding_sha256'])"),
])
derive('retention_audit',[
    ('import meth511_r8_operations as O','import meth511_r9_operations as O'),
    ('b=ctx.admit(args.binding_sha);prep=', "b=ctx.admit(args.binding_sha);ctx.r['retained_process_instances']=b['pre_audit_fault_instances'];prep="),
    ("assert native['native_calls']==576 and donor['model_calls']==384", "assert native['native_calls']==23 and native['retained_native_calls']==553 and len(native['commands'])==576 and donor['model_calls']==384"),
])
derive('windows_terminal',[
    ('meth511_r8_windows_terminal.json','meth511_r9_windows_terminal.json'),
    ("elseif ($kind -eq 'native') {'meth511_r7_'} else {'meth511_r8_'}", "elseif ($kind -eq 'native') {'meth511_r7_'} elseif ($kind -eq 'donor') {'meth511_r8_'} else {'meth511_r9_'}"),
],extension='.ps1')
with (B/'meth511_r9_derivation.json').open('x',encoding='utf8') as f:json.dump(records,f,indent=2);f.write('\n')
