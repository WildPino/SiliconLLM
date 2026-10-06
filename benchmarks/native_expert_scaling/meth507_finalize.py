"""Metadata/validated-output rollup only; no source linear function or inference."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import psutil

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);peak=0
def item(path, expected=None):
    global peak
    path=Path(path).resolve();before=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(8<<20):
            h.update(data);peak=max(peak,proc.memory_info().peak_wset)
            assert peak<=512<<20 and time.monotonic()-start<=60
    assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(path)
    return {'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}

binding=item(DOC/'meth507_binding.json','59f6ebee73460b58745a6637499ef1eb3f35ecf150db5c31da2f46bc6c4afc09')
main=item(DOC/'meth507_main_result.json','6b21f71416ffda30e9c5b8631aad3408515221b519590e09486a6f92463035ab')
audit=item(DOC/'meth507_audit_result.json','439facabc4a201978065069f776f5355215330dcc8bde3d586f1f46616cd2476')
tools=item(DOC/'meth507_tool_receipts.json');receipt=json.loads(Path(tools['path']).read_bytes())
for key in ('meth507_binding_poll2','meth507_main_poll1','meth507_audit_poll1'):assert receipt['tools'][key]['exit_code']==0
b=json.loads(Path(binding['path']).read_bytes());m=json.loads(Path(main['path']).read_bytes());a=json.loads(Path(audit['path']).read_bytes())
assert all(m['gates'].values()) and all(a['gates'].values()) and m['summary']==a['summary']
assert a['main_sha256']==main['sha256'] and m['binding_sha256']==a['binding_sha256']==binding['sha256']
processes=[];inventories={}
for kind,raw in (('main',m),('audit',a)):
    own=raw['process_instance'];alive=False;replacement=None
    try:
        p=psutil.Process(own['pid']);replacement=p.create_time();alive=abs(replacement-own['create_time_unix'])<.002
    except psutil.NoSuchProcess:pass
    assert not alive
    processes.append({'kind':kind,'original':own,'owned_instance_still_alive':False,'current_PID_creation_if_reused':replacement})
    out=ROOT/'results/native_expert_scaling'/('meth507_'+kind)
    inventory=[item(p) for p in sorted(out.iterdir()) if p.is_file()]
    assert {Path(v['path']).name for v in inventory}=={'fatal.log','progress.jsonl'}
    assert (out/'fatal.log').stat().st_size==0
    progress=[json.loads(line) for line in (out/'progress.jsonl').read_text(encoding='utf8').splitlines()]
    assert [r['book' if kind=='main' else 'audited_book'] for r in progress]==list(range(24))
    assert sum(v['bytes'] for v in inventory)+(main if kind=='main' else audit)['bytes']<=32<<20
    inventories[kind]=inventory
preserved={rel:item(ROOT/rel,sha) for rel,sha in b['preserved'].items()}
engine=item(ROOT/'benchmarks/phase60/engine.c',b['engine_sha256'])
states=[]
for snapshot in (0,1,2,6,12,13):
    values=[x for c in m['cases'] for x in c['encoder_geometry']['relative_l2'][snapshot]]
    states.append({'snapshot':snapshot,'median_relative_l2':statistics.median(values),'maximum_relative_l2':max(values)})
groups=[]
for name,group in [('ALL96',m['cases']),('divergent31',[c for c in m['cases'] if c['divergent']]),('identical65',[c for c in m['cases'] if not c['divergent']])]:
    groups.append({'group':name,'cases':len(group),'encoder_ID_changed_cases':sum(c['encoder_routes']['choice_changes']>0 for c in group),
                   'decoder_ID_changed_cases':sum(c['decoder_routes']['choice_changes']>0 for c in group),
                   'last_aligned_final_decoder_relative_l2_median':statistics.median(c['decoder_geometry']['relative_l2'][-1][-1] for c in group)})
assert all(max(c['encoder_geometry']['max_abs_error'][0])==0 and max(c['encoder_geometry']['max_abs_error'][1])>0 for c in m['cases'])
assert all(row[0]==0 for c in m['cases'] for row in c['decoder_geometry']['max_abs_error'])
unchanged_ids=[{'book':c['book'],'index':c['index'],'step':c['common_prefix_length'],
                'encoder_mass_max_abs_error':c['encoder_routes']['selected_mass_max_abs_error'],
                'decoder_mass_max_abs_error':c['decoder_routes']['selected_mass_max_abs_error']}
               for c in m['cases'] if c['divergent'] and c['encoder_routes']['choice_changes']==c['decoder_routes']['choice_changes']==0]
assert len(unchanged_ids)==5 and all(r['encoder_mass_max_abs_error']>0 and r['decoder_mass_max_abs_error']>0 for r in unchanged_ids)
value={'experiment':'METH507 terminal metadata and validated-output rollup','published_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'binding':binding,'main':main,'independent_audit':audit,'tool_receipts':tools,'summary':m['summary'],
       'metadata_source':item(__file__),'metadata_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'encoder_state_rollup':states,'case_groups':groups,'divergent_unchanged_expert_IDs':unchanged_ids,
       'owned_process_closure':processes,'output_inventory':inventories,'preserved_foreign_files':preserved,'engine':engine,
       'resource_limitations':{'main_audit_reported_peaks_and_seconds_scope':'Recorded before exclusive terminal serialization; later guards also run and satisfy limits. Exact final lifetime peaks/time are unavailable; do not relabel these samples.',
                               'exact_final_lifetime_peak_available':False,'model_native_compiler_calls':0,'metadata_seconds':time.monotonic()-start,'metadata_sampled_peak_bytes':peak},
       'apparatus_complete':True,'goal_complete':False,'decision':'Select proposed508 source-head/state decomposition; no new quality/rate/DRAM/useful-n promotion.'}
payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf8')
dest=DOC/'COMPLETION_507_20261006.json'
with dest.open('xb') as stream:stream.write(payload)
print(json.dumps({'completion':item(dest),'summary':value['summary']}),flush=True)
