"""Terminal509-R1 metadata; no shortlist/source-row/whole-model replay."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';start=time.monotonic();proc=psutil.Process()
def item(path,expected=None):
    p=Path(path).resolve();before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(8<<20):h.update(data);assert proc.memory_info().peak_wset<=512<<20 and time.monotonic()-start<=60
    assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':before.st_size,'sha256':h.hexdigest()}
binding=item(DOC/'meth509_r1_binding.json','e60e36b5aa24da575ed4597ad729405cf8910f9a59f69c316ea16c77046aa682')
main=item(DOC/'meth509_r1_main_result.json','e9541ad77a3f4e36b6145202c6c815ad0be33c33781161f8adc6698b26f25e3d')
audit=item(DOC/'meth509_r1_audit_result.json','edd64bb4dec6d0a8cdef945aef2ada597123dd1d1de18bffb9b4cdecf97b6c64')
tools=item(DOC/'meth509_tool_receipts.json');receipts=json.loads(Path(tools['path']).read_bytes())
for key in ('meth509_r1_binding_poll1','meth509_r1_main_poll1','meth509_r1_audit_poll1'):assert receipts['tools'][key]['exit_code']==0
assert receipts['tools']['meth509_main_tool']['exit_code']==1
b=json.loads(Path(binding['path']).read_bytes());m=json.loads(Path(main['path']).read_bytes());a=json.loads(Path(audit['path']).read_bytes())
assert all(m['gates'].values()) and all(a['gates'].values()) and m['summary']==a['summary'] and m['summary']['local_recipe_pass']
assert a['main_sha256']==main['sha256'] and a['binding_sha256']==m['binding_sha256']==binding['sha256']
original_failure=item(b['repair']['original_failure']['path'],b['repair']['original_failure']['sha256']);fault=json.loads(Path(original_failure['path']).read_bytes())
assert 'unbound_Python_module' in fault['traceback'] and 'psutil' in fault['traceback']
assert b['repair']['scientific_main_audit_logic_unchanged'] and b['repair']['original_rank_and_source_row_queries']==0
inventories={};terminal={};processes=[]
for kind,record,rp,key in [('main',m,main,'meth509_r1_main_poll1'),('audit',a,audit,'meth509_r1_audit_poll1')]:
    out=ROOT/'results/native_expert_scaling'/('meth509_r1_'+kind);inv=[item(p) for p in sorted(out.iterdir()) if p.is_file()]
    assert {Path(v['path']).name for v in inv}=={'fatal.log','progress.jsonl','terminal_resource.json'}
    assert (out/'fatal.log').stat().st_size==0 and sum(v['bytes'] for v in inv)+rp['bytes']<=32<<20
    t=json.loads((out/'terminal_resource.json').read_bytes());assert t['result_sha256']==rp['sha256'] and t['gates']==record['gates'] and t['summary']==record['summary']
    assert t['seconds']<=180 and t['OS_peak_bytes']<=2<<30 and json.loads(receipts['tools'][key]['output'])['terminal_resource']==t
    progress=[json.loads(line) for line in (out/'progress.jsonl').read_text(encoding='utf8').splitlines()]
    assert [r['book' if kind=='main' else 'audited_book'] for r in progress[:-1]]==list(range(24)) and progress[-1]['terminal_result_serialized_and_hashed']
    inventories[kind]=inv;terminal[kind]=t
for kind,record in [('original_prequery_fault',fault),('R1_main',m),('R1_audit',a)]:
    own=record['process_instance'];replacement=None;alive=False
    try:p=psutil.Process(own['pid']);replacement=p.create_time();alive=abs(replacement-own['create_time_unix'])<.002
    except psutil.NoSuchProcess:pass
    assert not alive;processes.append({'kind':kind,'original':own,'owned_instance_alive':False,'current_PID_creation_if_reused':replacement})
oldout=ROOT/'results/native_expert_scaling/meth509_main';inventories['original_prequery_fault']=[item(p) for p in sorted(oldout.iterdir()) if p.is_file()]
assert all(v['bytes']==0 for v in inventories['original_prequery_fault'])
preserved={rel:item(ROOT/rel,sha) for rel,sha in b['preserved'].items()};engine=item(ROOT/'benchmarks/phase60/engine.c',b['engine_sha256'])
value={'experiment':'METH509-R1 complete top8 fixed-state verification','source':item(__file__),'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'binding':binding,'main':main,'independent_audit':audit,'tool_receipts':tools,'original_prequery_fault':original_failure,
       'unchanged_scientific_recipe_repair':b['repair'],'summary':m['summary'],'terminal_resource_receipts':terminal,'output_inventory':inventories,
       'owned_process_closure':processes,'preserved_foreign':preserved,'engine':engine,'apparatus_complete':True,'local_recipe_pass':True,
       'goal_complete':False,'scope':'Consumed top8 hybrid winner equality only; no unseen/fresh whole quality/native cost/DRAM/useful-n promotion.',
       'decision':'Select510 one native operator/control/cost inquiry; qualify source C arithmetic and total extra head cost before new whole fresh quality/rate.'}
dest=DOC/'COMPLETION_509_20261006.json';payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf8')
with dest.open('xb') as f:f.write(payload)
print(json.dumps({'completion':item(dest),'summary':value['summary'],'main_seconds':terminal['main']['seconds'],'audit_seconds':terminal['audit']['seconds']}),flush=True)
