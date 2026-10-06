"""Metadata-only literal operational511 repair; original scientific stages had zero calls."""
import hashlib
import json
from pathlib import Path
B=Path(__file__).resolve().parent;records=[]
changes={
 'operations':[("BIND=DOC/'meth511_binding.json'","BIND=DOC/'meth511_r1_binding.json'"),
    ("('meth511_'+kind)","('meth511_r1_'+kind)"),("('meth511_'+kind+'_result.json')","('meth511_r1_'+kind+'_result.json')"),
    ("        for k,v in b['packages'].items():assert importlib.metadata.version(k)==v", "        for k,v in b['packages'].items():assert importlib.metadata.version(k)==v\n        assert sys.pycache_prefix==b['runtime_empty_cache'] and not any(Path(sys.pycache_prefix).rglob('*'))")],
 'cohort':[], 'native':[], 'donor':[], 'retention_audit':[]}
for suffix,replacements in changes.items():
    old=B/('meth511_'+suffix+'.py');new=B/('meth511_r1_'+suffix+'.py');source=old.read_text(encoding='utf8');text=source;receipt=[]
    if suffix!='operations':replacements=[('import meth511_operations as O','import meth511_r1_operations as O'),('results/native_expert_scaling/meth511_cohort/cohort.json','results/native_expert_scaling/meth511_r1_cohort/cohort.json')]+replacements
    for before,after in replacements:
        count=text.count(before)
        if before.startswith('results/') and count==0:continue
        assert count>0,(suffix,before);text=text.replace(before,after);receipt.append({'before':before,'after':after,'occurrences':count})
    with new.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    records.append({'old':str(old),'old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new':str(new),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'replacements':receipt})
with (B/'meth511_r1_derivation.json').open('x',encoding='utf8') as f:json.dump(records,f,indent=2);f.write('\n')
