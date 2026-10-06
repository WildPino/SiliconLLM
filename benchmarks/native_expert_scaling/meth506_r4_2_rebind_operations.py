"""Metadata-only operational repair after admission-only abort; bulk admission remains."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
source=DOC/'meth506_r4_1_binding.json';dest=DOC/'meth506_r4_2_binding.json';assert not dest.exists()
began=time.monotonic()
def item(p):
    p=Path(p).resolve();data=p.read_bytes();return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
head(source);s=item(source);assert s['sha256']=='01ff407e18f564be016dead88f20add9217a43d7c2438f77e80ee847166f3bd3'
b=json.loads(source.read_bytes());catalog={v['path']:v for v in b['catalog']};old=[];new={}
for name in ['meth506_r4_operations.py','meth506_r4_retention_audit.py','meth506_r4_finalize_admission.py','meth506_r4_windows_terminal.ps1']:
    p=ROOT/'benchmarks/native_expert_scaling'/name;head(p);v=item(p);old.append(catalog[v['path']]);new[v['path']]=v;catalog[v['path']]=v
extra=[]
for p in [Path(__file__),DOC/'METH_506_R4_2_IMMUTABLE_OUTPUT_CHARGE_20261006.md',DOC/'meth506_r4_1_admission_abort.json',DOC/'meth506_r4_1_termination.json',DOC/'meth506_r4_1_tool_receipt.json']:
    head(p);v=item(p);catalog[v['path']]=v;extra.append(v)
catalog[s['path']]=s
b.update(catalog=list(catalog.values()),scientific=[new.get(v['path'],v) for v in b['scientific']]+extra[:2],
         records=[new.get(v['path'],v) for v in b['records']]+[s]+extra[2:],
         freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         pre_first_inference_operations_rebind={'old_sources':old,'new_sources':list(new.values()),
         'change':'Immutable completed output charge once with terminal complete inventory revalidation; dynamic own output guard1s; memory/wall0.1s. Fresh namespaces/metadata paths only; zero prior donor/native calls.',
         'bulk_rehashed_by_derivation':False,'all_bulk_inputs_SHA_revalidated_by_main_and_audit_before_imports':True})
with dest.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
assert time.monotonic()-began<=180
print(json.dumps({'binding':item(dest),'seconds':time.monotonic()-began,'scope':'Operational-only metadata derivation; full SHA admission before numerical imports remains.'}),flush=True)
