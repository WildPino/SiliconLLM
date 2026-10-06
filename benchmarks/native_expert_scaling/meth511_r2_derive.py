"""Metadata-only R2: canonical donor identity once before inference; matched original native environment."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;records=[]
for suffix in ['operations','cohort','native','donor','retention_audit']:
    old=B/('meth511_r1_'+suffix+'.py');new=B/('meth511_r2_'+suffix+'.py');text=old.read_text(encoding='utf8');changes=[]
    replacements=[]
    if suffix=='operations':
        replacements=[('meth511_r1_binding.json','meth511_r2_binding.json'),("('meth511_r1_'+kind)","('meth511_r2_'+kind)"),
            ("('meth511_r1_'+kind+'_result.json')","('meth511_r2_'+kind+'_result.json')"),
            ("input_admission_scope='Full SHA was freshly computed by metadata binder. This stage requires unchanged size/mtime and retained Win32 read/share-read handles denying writes/deletion; small scientific source is freshly SHA checked.'",
             "input_admission_scope='Fresh full SHA of runtime/source/candidate/corpus/ledgers; original donor ZIP whole SHA retained from acquisition with current stat/write-deny binding. All3320 loaded original F32 tensors receive fresh canonical SHA before first donor inference. Small scientific source is freshly SHA checked.'"),
            ("env={**os.environ,**ENV}","env={**os.environ,**ENV,**self.binding['native_runtime_environment']}")]
    else:
        replacements=[('import meth511_r1_operations as O','import meth511_r2_operations as O'),('results/native_expert_scaling/meth511_r1_cohort/cohort.json','results/native_expert_scaling/meth511_r2_cohort/cohort.json')]
        if suffix=='donor':replacements.extend([
            ("namespace=model.state_dict();assert set(namespace)==set(bound['tensors']);seen=set();shards=[]", "namespace=model.state_dict();assert set(namespace)==set(bound['tensors']);seen=set();shards=[];canonical_hashed_bytes=0"),
            ("assert hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()==expected['sha256'],name;seen.add(name)", "assert hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()==expected['sha256'],name;seen.add(name)\n                canonical_hashed_bytes+=value.numel()*value.element_size()"),
            ("ctx.r['gates']['all3320_actual_original_F32_coefficients7415217408_unique_parameters']=True", "ctx.r['gates']['all3320_actual_original_F32_coefficients7415217408_unique_parameters']=True\n        ctx.r['canonical_F32_bytes_hashed_before_first_inference']=canonical_hashed_bytes")])
    for before,after in replacements:
        n=text.count(before)
        if before.startswith('results/') and n==0:continue
        assert n>0,(suffix,before);text=text.replace(before,after);changes.append({'before':before,'after':after,'occurrences':n})
    with new.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    records.append({'old':str(old),'old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new':str(new),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'replacements':changes})
with (B/'meth511_r2_derivation.json').open('x',encoding='utf8') as f:json.dump(records,f,indent=2);f.write('\n')
