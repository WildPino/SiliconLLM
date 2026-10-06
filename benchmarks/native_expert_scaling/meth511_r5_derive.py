"""Literal identity-only copies after model server closed; completed cohort unchanged."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;records=[]
for suffix in ['operations','native','donor','retention_audit']:
    old=B/('meth511_r4_'+suffix+'.py');new=B/('meth511_r5_'+suffix+'.py');text=old.read_text(encoding='utf8');changes=[]
    replacements=[('meth511_r4_binding.json','meth511_r5_binding.json'),("('meth511_r4_'+kind)","('meth511_r5_'+kind)"),("'meth511_r4_'","'meth511_r5_'")] if suffix=='operations' else [('import meth511_r4_operations as O','import meth511_r5_operations as O')]
    for before,after in replacements:
        n=text.count(before);assert n>0,(suffix,before);text=text.replace(before,after);changes.append({'before':before,'after':after,'occurrences':n})
    with new.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    records.append({'old':str(old),'old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new':str(new),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'replacements':changes})
with (B/'meth511_r5_derivation.json').open('x',encoding='utf8') as f:json.dump(records,f,indent=2);f.write('\n')
