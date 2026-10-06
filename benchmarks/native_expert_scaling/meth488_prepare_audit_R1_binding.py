"""Freeze only numbered admission apparatus; no model or numerical observation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
BASE=ROOT/'benchmarks/native_expert_scaling'
DEST=DOC/'meth488_audit_r1_binding.json'
assert not DEST.exists() and sys.flags.optimize==0

def item(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4<<20):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

paths=[BASE/name for name in ('meth488_retention_audit_R1.py','meth488_audit_r1_windows_terminal.ps1',
                            'meth488_finalize_admission_R1.py','meth488_prepare_audit_R1_binding.py')]
paths+=[DOC/name for name in ('METH_488_R1_RETAINED_PREFIX_ADMISSION_PROTOCOL_20261006.md',
                             'meth488_shared_integer_backend_result.failure.json','meth488_main_sole_first_registration.json',
                             'RETENTION_488_20261006.failure.json','meth488_audit_sole_first_registration.json','meth488_binding.json')]
for p in paths:
    rel=p.relative_to(ROOT).as_posix();head=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
    assert head.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n'),rel
paths+=[ROOT/'results/native_expert_scaling'/name for name in ('meth488_windows_terminal.json','meth488_audit_windows_terminal.json')]
files=[item(p) for p in paths]
expected={'meth488_shared_integer_backend_result.failure.json':'8934a334aaae465aa8c690221900c5c113d72edc43d170af7a5fc64283fb05ad',
          'RETENTION_488_20261006.failure.json':'500c513fccf46c796a0ae316efe679ee11323690de3e5db71c560cacb4871944',
          'meth488_binding.json':'0b644008a14b58388d240a443a0f97852813391b4b738a06af50d4c75e5890c1'}
for v in files:
    if Path(v['path']).name in expected:assert v['sha256']==expected[Path(v['path']).name]
value={'experiment':'METH488-R1 sole independent prefix admission extension','original_main_controls_and_audit_not_replayed':True,
       'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':files,
       'numerical_scope':'ALL11 pre-existing controls and fixed completed96/78 prefixes only; no rate/bootstrap; prior data retained byte-exact.'}
with DEST.open('xb') as f:f.write((json.dumps(value,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],'files':len(files)}))
