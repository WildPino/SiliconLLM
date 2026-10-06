"""Numbered memory-only continuation binding; metadata/hash reads only."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth490_r1_binding.json';assert not DEST.exists();start=time.monotonic()
def item(p,expected=None):
    p=Path(p).resolve();s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(4<<20):h.update(data);assert time.monotonic()-start<=120
    assert p.stat().st_size==s.st_size and p.stat().st_mtime_ns==s.st_mtime_ns
    r={'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()}
    if expected:assert r['sha256']==expected,str(p)
    return r
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
old=DOC/'meth490_binding.json';item(old,'4c339786e131926a750b928a8e610b652c848466343fb80a288d7ac88fd68973');b=json.loads(old.read_bytes())
catalog={v['path']:item(v['path'],v['sha256']) for v in b['catalog']}
inherited={}
for key,name,sha in [('raw','meth490_root_mass_result.failure.json','aabfc0fac58503ef896a155c630faee6668762c7045cce9689807b78ed517b24'),
                     ('retention','RETENTION_490_20261006.json','141c48a7e0ea515e16bc9718918889921bb0e322bef83e26bcc649f98adb0cdb'),
                     ('admission','ADMISSION_490_20261006.json','825988cc2d0ac726e003a7c5246dbdfeb1c7993eb547a689cd4b9b5406450e80')]:
    p=DOC/name;head(p);r=item(p,sha);inherited[key]=r;catalog[r['path']]=r
out=ROOT/'results/native_expert_scaling/meth490_root_mass'
partials=[]
for p in sorted(out.iterdir()):
    if p.is_file():r=item(p);partials.append(r);catalog[r['path']]=r
ret=json.loads(Path(inherited['retention']['path']).read_bytes());assert ret['main_completed'] is False and not ret['numerical_imports'] and all(ret['gates'].values())
assert partials==ret['inherited_partial_inventory']
new=list(sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth490_r1*')))+[DOC/'METH_490_R1_STREAMING_RESUMPTION_PROTOCOL_20261006.md']
for p in new:head(p)
new=list(map(item,new));b['scientific']+=new
records=[item(old),*inherited.values()]
for p in [DOC/'meth490_first_finalizer_fault.json',DOC/'meth490_r1_first_assembly_fault.json',DOC/'meth490_r1_source_derivation.json',
          DOC/'meth490_main_sole_first_registration.json',DOC/'meth490_audit_sole_first_registration.json']:
    records.append(item(p))
auxiliary=[item(ROOT/('results/native_expert_scaling/'+name)) for name in ('meth490_windows_terminal.json','meth490_audit_windows_terminal.json')]
for r in new+records+auxiliary:catalog[r['path']]=r
b.update(experiment='METH490-R1 memory-only staged continuation; preserve first64 updates',inherited=inherited,
         original_out=str(out),inherited_binary=item(out/'meth490_root_mass.exe'),inherited_output_bytes=sum(v['bytes'] for v in partials),
         catalog=list(catalog.values()),freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         inherited_complete_roots=[0,1],remaining_fit_roots=list(range(2,12)),no_compile_control_or_completed_fit_replay=True,
         auxiliary_records=auxiliary,preparation_seconds=time.monotonic()-start)
b['records']+=records
with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],'seconds':b['preparation_seconds'],'remaining_optimizer_updates':320}))
