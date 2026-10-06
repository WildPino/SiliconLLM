"""Pre-first-call wrapper rebind only; main revalidates every inherited input before imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
source=DOC/'meth506_r4_binding.json';dest=DOC/'meth506_r4_1_binding.json';assert not dest.exists()
start=time.monotonic()
def item(p):
    p=Path(p).resolve();data=p.read_bytes();return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
head(source);s=item(source);assert s['sha256']=='d0542cc8a5b36a50f792bc3d79e19abeef86c160b30b38cd9d04b14f1513cb93'
b=json.loads(source.read_bytes());catalog={v['path']:v for v in b['catalog']}
p=ROOT/'benchmarks/native_expert_scaling/meth506_r4_operations.py';head(p);old=catalog[str(p.resolve())];new=item(p);catalog[new['path']]=new
extra=[]
for p in [Path(__file__),DOC/'METH_506_R4_1_GUARD_CADENCE_20261006.md']:
    head(p);v=item(p);catalog[v['path']]=v;extra.append(v)
catalog[s['path']]=s
scientific=[new if v['path']==new['path'] else v for v in b['scientific']]+extra
b.update(catalog=list(catalog.values()),scientific=scientific,records=b['records']+[s],freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         pre_first_call_operations_rebind={'old_operations':old,'new_operations':new,'change':'Output byte inventory cadence1s, same original guard cadence; memory/wall watchdog remain0.1s. Binding path only additionally changed. No quality/cost/data/precision/native changes.','bulk_inputs_rehashed_by_this_wrapper':False,'all_bulk_files_revalidated_by_quality_and_audit_before_import':True,'R4_quality_calls_before_change':0})
with dest.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
assert time.monotonic()-start<=180
print(json.dumps({'binding':item(dest),'seconds':time.monotonic()-start,'scope':'Pre-first-call metadata derivation; every inherited catalog file still revalidated by admitted controller/audit before imports.'}),flush=True)
