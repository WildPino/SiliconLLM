"""Bind staged completion to qualified first data and unchanged executable."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';DEST=DOC/'meth489_binding.json'
assert not DEST.exists() and sys.flags.optimize==0
start=time.monotonic()

def item(path,expected=None):
    p=Path(path);before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4<<20):h.update(b);assert time.monotonic()-start<300
    assert p.stat().st_size==before.st_size and p.stat().st_mtime_ns==before.st_mtime_ns
    v={'path':str(p),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'sha256':h.hexdigest()}
    if expected:assert v['sha256']==expected,str(p)
    return v

def head(path):
    p=Path(path);rel=p.relative_to(ROOT).as_posix()
    assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT).replace(b'\r\n',b'\n'),rel

original_path=DOC/'meth488_binding.json'
item(original_path,'0b644008a14b58388d240a443a0f97852813391b4b738a06af50d4c75e5890c1')
binding=json.loads(original_path.read_bytes())
for v in binding['helpers']+binding['records']+binding['toolchain']+binding['cuda']+binding['runtime']['files']:item(v['path'],v['sha256'])
new_helpers=sorted(BASE.glob('meth489*'))+[DOC/'METH_489_STAGED_COMPLETION_PROTOCOL_20261006.md']
for p in new_helpers:head(p)
binding['helpers']+=list(map(item,new_helpers))
inherited={}
for key,name,digest in [
    ('raw','meth488_shared_integer_backend_result.failure.json','8934a334aaae465aa8c690221900c5c113d72edc43d170af7a5fc64283fb05ad'),
    ('retention','RETENTION_488_R1_20261006.json','ffe322f72b902ffd3932c3f0d7949f51daa463bef0db18c5051f2ca815e67886'),
    ('admission','ADMISSION_488_R1_20261006.json','d0ff3ee1ee6bac081fc38ad16b96957200bd6d3eb86318586c76c35c2bce88b1')]:
    p=DOC/name;head(p);inherited[key]=item(p,digest)
original=json.loads(Path(inherited['raw']['path']).read_bytes())
binding['binary']=item(ROOT/'results/native_expert_scaling/meth488_shared_integer_backend/meth488_shared_integer.exe',original['compile']['binary_sha256'])
binding['inherited']=inherited
binding['records']+=[item(original_path),*inherited.values(),item(DOC/'meth488_audit_r1_binding.json')]
binding['experiment']='METH489 missing-block staged completion and prespecified bootstrap metadata correction'
binding['freeze_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
binding['original_freeze_head']=json.loads(original_path.read_bytes())['freeze_head']
binding['missing_successful_native_blocks']=107
binding['no_compile_or_control_or_completed_block_replay']=True
binding['bootstrap_seed']=485485
binding['preparation_seconds']=time.monotonic()-start
with DEST.open('xb') as f:f.write((json.dumps(binding,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],'missing_blocks':107,'seconds':binding['preparation_seconds']}))
