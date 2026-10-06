"""508 metadata completion; no readout evaluation or completed-call replay."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import psutil

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';start=time.monotonic();proc=psutil.Process();peak=0
def item(path,expected=None):
    global peak
    p=Path(path).resolve();before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(8<<20):h.update(block);peak=max(peak,proc.memory_info().peak_wset);assert peak<=512<<20 and time.monotonic()-start<=60
    assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':before.st_size,'sha256':h.hexdigest()}
binding=item(DOC/'meth508_binding.json','906499fd54f132c045d2f27bb84f188b9548fa93af51781e8097499734a92e39')
main=item(DOC/'meth508_main_result.json','cf7badf270be0fce0334e4e20f3355a122d05938deb2ccab42fa079e1a1bf241')
audit=item(DOC/'meth508_audit_result.json','e67f3cc1a5bb65ee39cf9364d68e33c9e2a4ce6c2a391c7d1ff02cfd8968e831')
tools=item(DOC/'meth508_tool_receipts.json');toolrecord=json.loads(Path(tools['path']).read_bytes())
for name in ('meth508_binding_poll1','meth508_main_poll1','meth508_audit_poll1'):assert toolrecord['tools'][name]['exit_code']==0
b=json.loads(Path(binding['path']).read_bytes());m=json.loads(Path(main['path']).read_bytes());a=json.loads(Path(audit['path']).read_bytes())
assert all(m['gates'].values()) and all(a['gates'].values()) and m['summary']==a['summary'] and m['summary']['source_state_bridge_qualified']
assert a['main_sha256']==main['sha256'] and a['binding_sha256']==m['binding_sha256']==binding['sha256']
inventories={};terminal={};processes=[]
for kind,record,rp,toolkey in [('main',m,main,'meth508_main_poll1'),('audit',a,audit,'meth508_audit_poll1')]:
    out=ROOT/'results/native_expert_scaling'/('meth508_'+kind);inv=[item(p) for p in sorted(out.iterdir()) if p.is_file()]
    assert {Path(v['path']).name for v in inv}=={'fatal.log','progress.jsonl','terminal_resource.json'}
    assert (out/'fatal.log').stat().st_size==0 and sum(v['bytes'] for v in inv)+rp['bytes']<=32<<20
    t=json.loads((out/'terminal_resource.json').read_bytes());assert t['result_sha256']==rp['sha256'] and t['gates']==record['gates'] and t['summary']==record['summary']
    assert t['seconds']<=180 and t['OS_peak_bytes']<=2<<30
    output=json.loads(toolrecord['tools'][toolkey]['output']);assert output['terminal_resource']==t
    progress=[json.loads(line) for line in (out/'progress.jsonl').read_text(encoding='utf8').splitlines()]
    assert [r['book' if kind=='main' else 'audited_book'] for r in progress[:-1]]==list(range(24)) and progress[-1]['terminal_result_serialized_and_hashed']
    own=record['process_instance'];replacement=None;alive=False
    try:p=psutil.Process(own['pid']);replacement=p.create_time();alive=abs(replacement-own['create_time_unix'])<.002
    except psutil.NoSuchProcess:pass
    assert not alive;processes.append({'kind':kind,'original':own,'owned_instance_alive':False,'current_PID_creation_if_reused':replacement})
    inventories[kind]=inv;terminal[kind]=t
preserved={rel:item(ROOT/rel,sha) for rel,sha in b['preserved'].items()};engine=item(ROOT/'benchmarks/phase60/engine.c',b['engine_sha256'])
d=[c['first_divergence_decomposition'] for c in m['cases'] if c['first_divergence_decomposition']]
terms={}
for key in ('donor_margin','total_pair_perturbation','readout_residual','upstream_scaled_state_effect','donor_arithmetic_bridge_residual'):
    values=[r[key]['float'] for r in d];terms[key]={'minimum':min(values),'median':statistics.median(values),'maximum':max(values)}
value={'experiment':'METH508 completion metadata','source':item(__file__),'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'binding':binding,'main':main,'independent_audit':audit,'tool_receipts':tools,'summary':m['summary'],'pair_term_rollups':terms,
       'readout_abs_exceeds_upstream_cases':sum(abs(r['readout_residual']['float'])>abs(r['upstream_scaled_state_effect']['float']) for r in d),
       'introduced_positions':[{'book':c['book'],'index':c['index'],'step':p['step'],'donor_id':p['donor_id'],'new_id':p['source_head_candidate_state_id']} for c in m['cases'] for p in c['positions'] if p['introduced']],
       'terminal_resource_receipts':terminal,'output_inventory':inventories,'owned_process_closure':processes,
       'preserved_foreign':preserved,'engine':engine,'post_merge':b['post_merge'],'apparatus_complete':True,
       'goal_complete':False,'scope':'Fixed-state source-head benefit only; no full generation/quality/speed/DRAM/useful-n promotion.',
       'next':'ONE proposed top8 original-I8-head shortlist with original F32 row refinement; qualify all996 coverage and hybrid winner before native integration.'}
dest=DOC/'COMPLETION_508_20261006.json';payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf8')
with dest.open('xb') as f:f.write(payload)
print(json.dumps({'completion':item(dest),'summary':value['summary'],'terminal_main_seconds':terminal['main']['seconds'],'terminal_audit_seconds':terminal['audit']['seconds']}),flush=True)
