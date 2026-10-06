"""New retention-only metadata freeze; no CUDA/model/control work."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';BASE=ROOT/'benchmarks/native_expert_scaling';DEST=DOC/'meth485_audit_r1_binding.json';assert not DEST.exists()
names=['meth485_retention_audit_r1.py','meth485_audit_r1_windows_terminal.ps1','meth485_finalize_admission_r1.py','meth485_prepare_audit_r1_binding.py']
paths=[BASE/n for n in names]+[DOC/'METH_485_AUDIT_R1_RATIONALE_20261006.json',DOC/'meth485_main_sole_first_registration.json',DOC/'meth485_audit_sole_first_registration.json',DOC/'RETENTION_485_20261006.failure.json']
files=[]
for p in paths:
    rel=p.relative_to(ROOT).as_posix();blob=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT);assert blob.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n')
    files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for n in ('meth485_windows_terminal.json','meth485_audit_windows_terminal.json'):
    p=ROOT/'results/native_expert_scaling'/n;files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
r={'experiment':'METH485 new independent retention R1 metadata binding','original_main_not_replayed':True,'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':files}
with DEST.open('xb') as f:f.write((json.dumps(r,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'binding':str(DEST),'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()}))
