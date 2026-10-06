"""Literal operational reader repair; R2 emitted zero rows/cases/model/native calls."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;records=[]
for suffix in ['operations','cohort','native','donor','retention_audit']:
    old=B/('meth511_r2_'+suffix+'.py');new=B/('meth511_r3_'+suffix+'.py');text=old.read_text(encoding='utf8');changes=[]
    if suffix=='operations':replacements=[('meth511_r2_binding.json','meth511_r3_binding.json'),("('meth511_r2_'+kind)","('meth511_r3_'+kind)"),("('meth511_r2_'+kind+'_result.json')","('meth511_r3_'+kind+'_result.json')")]
    else:replacements=[('import meth511_r2_operations as O','import meth511_r3_operations as O'),('results/native_expert_scaling/meth511_r2_cohort/cohort.json','results/native_expert_scaling/meth511_r3_cohort/cohort.json')]
    if suffix in ('cohort','retention_audit'):
        replacements.append(("'memory_limit':'512MB'","'memory_limit':'1536MiB'"))
        if suffix=='cohort':
            before="        cursor=conn.execute('SELECT file_row_number,text FROM read_parquet(?,file_row_number=true) ORDER BY file_row_number',[b['corpus']['path']])"
            after="        physical=conn.execute('SELECT count(*),min(file_row_number),max(file_row_number),count(DISTINCT file_row_number) FROM read_parquet(?,file_row_number=true)',[b['corpus']['path']]).fetchone()\n        assert physical==(1243,0,1242,1243)\n        selected_sql=','.join(str(row) for row in sorted(wanted));assert all(0<=row<1243 for row in wanted)\n        cursor=conn.execute('SELECT file_row_number,text FROM read_parquet(?,file_row_number=true) WHERE file_row_number IN ('+selected_sql+') ORDER BY file_row_number',[b['corpus']['path']])"
            replacements.extend([(before,after),('                assert row==i;i+=1','                assert row in wanted and row not in texts;i+=1'),('        assert i==1243','        assert i==len(wanted) and set(texts)==wanted')])
        else:
            before="        cursor=conn.execute('SELECT file_row_number,text FROM read_parquet(?,file_row_number=true) ORDER BY file_row_number',[b['corpus']['path']]);count=0"
            after="        physical=conn.execute('SELECT count(*),min(file_row_number),max(file_row_number),count(DISTINCT file_row_number) FROM read_parquet(?,file_row_number=true)',[b['corpus']['path']]).fetchone();assert physical==(1243,0,1242,1243)\n        selected_sql=','.join(str(row) for row in sorted(wanted));assert all(0<=row<1243 for row in wanted)\n        cursor=conn.execute('SELECT file_row_number,text FROM read_parquet(?,file_row_number=true) WHERE file_row_number IN ('+selected_sql+') ORDER BY file_row_number',[b['corpus']['path']]);count=0"
            replacements.extend([(before,after),('                assert row==count;count+=1','                assert row in wanted and row not in texts;count+=1'),('        conn.close();assert count==1243;tok=','        conn.close();assert count==len(wanted) and set(texts)==wanted;tok=')])
    for before,after in replacements:
        n=text.count(before)
        if before.startswith('results/') and n==0:continue
        assert n>0,(suffix,before);text=text.replace(before,after);changes.append({'before':before,'after':after,'occurrences':n})
    with new.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    records.append({'old':str(old),'old_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new':str(new),'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'replacements':changes})
with (B/'meth511_r3_derivation.json').open('x',encoding='utf8') as f:json.dump(records,f,indent=2);f.write('\n')
