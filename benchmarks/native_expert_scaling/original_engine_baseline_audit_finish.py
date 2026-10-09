"""Stored-only stable logaddexp completion; retain first audit script unchanged.

Only the independent log-partition reference is stabilized by subtracting its
maximum before logaddexp reduction. Gate1e-10 unchanged, no model replay.
"""
from pathlib import Path
import hashlib
import json

parent=Path(__file__).with_name('original_engine_baseline_audit.py')
source=parent.read_text()
needle="q=q-np.logaddexp.reduce(q);p=p-np.logaddexp.reduce(p)"
assert source.count(needle)==1
source=source.replace(needle,"q-=q.max();p-=p.max();q=q-np.logaddexp.reduce(q);p=p-np.logaddexp.reduce(p)")
needle="assert max_KL<=1e-10 and max_label_KL<=1e-10"
assert source.count(needle)==1
source=source.replace(needle,"print(json.dumps(dict(reference_case_delta=max_KL,reference_label_delta=max_label_KL)),flush=True);"+needle)
needle="scope='Stored-only independent logaddexp F64 adjudication;not a held experiment family or new inference.')"
assert source.count(needle)==1
source=source.replace(needle,"first_audit_failure=extent(DOC/'original_engine_baseline_first_audit_fault_20261009.json'),"
    "audit_parent=extent(B/'original_engine_baseline_audit.py'),audit_completion=extent(B/'original_engine_baseline_audit_finish.py'),"
    "scope='Stored-only independent stabilized logaddexp F64 adjudication;first unstabilized-reference assertion retained."
    "Same1e-10 gate;no model replay/held experiment resource claim.')")
exec(compile(source,str(parent), 'exec'),dict(__name__='__main__',__file__=str(Path(__file__).resolve())))
