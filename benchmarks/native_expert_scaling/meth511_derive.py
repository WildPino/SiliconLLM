"""Metadata-only literal rubric derivation; no imports of numerical packages."""
import ast
import hashlib
import json
from pathlib import Path

B=Path(__file__).resolve().parent
old=B/'meth506_quality.py';text=old.read_text(encoding='utf8');tree=ast.parse(text)
names=['read_output','relative','metrics','edit_distance','lcs','task','paired_interval']
parts=[ast.get_source_segment(text,node) for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
assert len(parts)==len(names)
pure='"""Literal506 numerical rubrics; no Torch or corpus imports."""\nimport struct\nfrom pathlib import Path\nimport numpy as np\nMASK=np.array([1,2,4,5,7,8,10,11])\n\n'+'\n\n'.join(parts)+'\n'
result=(B/'meth506_results.py').read_text(encoding='utf8')
before="task=c['candidate_task'];oc+="
assert result.count(before)==1
result=result.replace('import meth506_quality as Q','import meth511_quality as Q').replace(before,"task=c[arm+'_task'];oc+=")
records=[]
for destination,payload,source,changes in [(B/'meth511_quality.py',pure,old,{'literal_AST_functions':names}),
    (B/'meth511_results.py',result,B/'meth506_results.py',{'import':'meth511_quality','only_numerator_fix':"task=c[arm+'_task']"})]:
    with destination.open('x',encoding='utf8',newline='\n') as f:f.write(payload)
    records.append({'old':str(source),'old_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'new':str(destination),'new_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'changes':changes})
with (B/'meth511_derivation.json').open('x',encoding='utf8') as f:json.dump(records,f,indent=2);f.write('\n')
