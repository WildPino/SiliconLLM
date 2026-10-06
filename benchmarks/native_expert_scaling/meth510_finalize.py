"""510 terminal admission metadata; ZERO compile/native/math replay."""
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
        while data:=f.read(8<<20):h.update(data);assert proc.memory_info().peak_wset<=512<<20 and time.monotonic()-start<=90
    assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':before.st_size,'sha256':h.hexdigest()}
def main():
    binding=item(DOC/'meth510_r1_binding.json','8e95220db76cea3a47b008a470636d25bf171c4ee3bdaa0bff4d6c1abdc58685')
    mainrow=item(DOC/'meth510_r1_main_result.json','56da38caba7ba6ba71e19d5b35a828b70595442468327cae9b011535968ec617')
    auditrow=item(DOC/'meth510_r1_audit_result.json','3f5df20bb6472a56f3a958d249f1459bb1dfc7ac7f09a954a6300563641ec9bc')
    b=json.loads(Path(binding['path']).read_bytes());m=json.loads(Path(mainrow['path']).read_bytes());a=json.loads(Path(auditrow['path']).read_bytes())
    assert all(m['gates'].values()) and all(a['gates'].values()) and m['summary']==a['summary'] and m['summary']['local_recipe_pass']
    assert a['main_sha256']==mainrow['sha256'] and a['binding_sha256']==m['binding_sha256']==binding['sha256']
    tools=item(DOC/'meth510_tool_receipts.json');receipts=json.loads(Path(tools['path']).read_bytes())['tools']
    for key in ('meth510_binding_poll3','meth510_r1_binding_poll1','meth510_r1_main_poll1','meth510_r1_audit_poll2'):assert receipts[key]['exit_code']==0
    assert receipts['meth510_main_poll1']['exit_code']==1
    faultrow=item(b['resume']['original_failure']['path'],b['resume']['original_failure']['sha256']);fault=json.loads(Path(faultrow['path']).read_bytes())
    assert 'foreign_before_timing' in fault['traceback'] and len(fault['commands'])==7 and len(m['commands'])==8
    for old,new in zip(fault['commands'],m['commands'][:7]):assert new=={**old,'retained_from_first_fault':True}
    assert m['commands'][-1]['label']=='paired_head' and not m['commands'][-1].get('retained_from_first_fault')
    inventories={};terminal={}
    for kind,record,rp,key in [('main',m,mainrow,'meth510_r1_main_poll1'),('audit',a,auditrow,'meth510_r1_audit_poll2')]:
        out=ROOT/'results/native_expert_scaling'/('meth510_r1_'+kind);inv=[item(p) for p in sorted(out.iterdir()) if p.is_file()]
        assert sum(v['bytes'] for v in inv)+rp['bytes']<=512<<20 and (out/'fatal.log').stat().st_size==0
        t=json.loads((out/'terminal_resource.json').read_bytes());assert t['result_sha256']==rp['sha256'] and t['gates']==record['gates'] and t['summary']==record['summary']
        assert t['seconds']<=360 and t['OS_peak_bytes']<=2<<30 and t['native_peak_bytes']<=1<<30
        assert json.loads(receipts[key]['output'])['terminal_resource']==t
        progress=[json.loads(line) for line in (out/'progress.jsonl').read_text(encoding='utf8').splitlines()]
        bookkey='book' if kind=='main' else 'audited_book';assert [v[bookkey] for v in progress if bookkey in v]==list(range(24)) and progress[-1]['terminal_result_serialized_and_hashed']
        inventories[kind]=inv;terminal[kind]=t
    originalout=ROOT/'results/native_expert_scaling/meth510_main'
    for row in b['resume']['original_output_inventory']:assert item(row['path'],row['sha256'])==row
    assert not (originalout/'head_outputs.bin').exists() and not (originalout/'paired_head.stdout').exists()
    assert item(ROOT/'results/native_expert_scaling/meth510_r1_main/meth510.exe',b['resume']['compiled_binary']['sha256'])['sha256']==m['binary_sha256']
    assert item(ROOT/'results/native_expert_scaling/meth510_r1_main/head_inputs.bin',b['resume']['packed_inputs']['sha256'])['sha256']==m['input_sha256']
    actual=m['commands'][-1];begin=actual['timing_start_daemons'];end=actual['timing_end_daemons'];assert actual['timing_samples']==len(actual['timing_observations'])>0
    baseline={(v['pid'],v['create_time_unix']):v['cpu_seconds'] for v in begin}
    for rows in [end,*[v['preserved_daemons'] for v in actual['timing_observations']]]:
        assert {(v['pid'],v['create_time_unix']):v['cpu_seconds'] for v in rows}==baseline
    events=item(DOC/'meth510_r1_windows_terminal.json');ev=json.loads(Path(events['path']).read_bytes())
    assert ev['positive_control']['query_available'] and set(ev['known_record_ids'])<=set(v['RecordId'] for v in ev['positive_control']['events'])
    assert all(v['query']['query_available'] and not v['matching_events'] for v in ev['stages'])
    eventtool=item(DOC/'meth510_windows_r1_tool_receipt.json');assert json.loads(Path(eventtool['path']).read_bytes())['tool']['exit_code']==0
    processes=[]
    for label,record in [('original_pre_timing_fault',fault),('R1_main',m),('R1_audit',a),*[(r['label'],r) for r in m['commands']]]:
        own=record['process_instance'];alive=False;replacement=None
        try:p=psutil.Process(own['pid']);replacement=p.create_time();alive=abs(replacement-own['create_time_unix'])<.002
        except psutil.NoSuchProcess:pass
        assert not alive;processes.append({'label':label,'original':own,'owned_instance_alive':False,'current_PID_creation_if_reused':replacement})
    preserved={rel:item(ROOT/rel,sha) for rel,sha in b['preserved'].items()};engine=item(ROOT/'benchmarks/phase60/engine.c',b['engine_sha256'])
    for row in b['scientific']:item(row['path'],row['sha256'])
    value={'experiment':'METH510 native conditional source readout complete','producer':item(__file__),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'binding':binding,'main':mainrow,'independent_audit':auditrow,
        'tool_receipts':tools,'original_pre_timing_fault':faultrow,'original_controls_reused':7,'new_compile_control_calls':0,'sole_new_paired_head_calls':1,
        'summary':m['summary'],'terminal_resource_receipts':terminal,'output_inventory':inventories,'owned_process_closure':processes,
        'Windows_event_receipt':events,'Windows_event_tool':eventtool,'retained_first_UTC_fault':item(DOC/'meth510_first_utc_fault.json'),
        'actual_UTC_helper':item(ROOT/'benchmarks/native_expert_scaling/meth510_terminal_utc_r1.ps1'),
        'metadata_scope':'Standalone metadata FileTime epoch typed as DateTimeOffset before first query; first positive-control fault from DateTime-to-string Kind/fraction loss retained; corrected helper uses actual qualified506 Utc type handling. No numerical replay.',
        'preserved_foreign':preserved,'engine':engine,'apparatus_complete':True,'local_recipe_pass':True,'goal_complete':False,
        'scope':'Exact consumed native finite operator and instrumented paired head cost only; fresh quality, whole rate, useful n, compact core, real DRAM, additional families/scales remain required.',
        'decision':'Select511 prospective new own-state whole teacher/generation/tasks against original F32 donor plus strictly isolated SAMEartifact accepted batch1 rate. Keep506 thresholds unchanged and all510 timing outliers.'}
    dest=DOC/'COMPLETION_510_20261006.json';payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf8')
    with dest.open('xb') as f:f.write(payload)
    print(json.dumps({'completion':item(dest),'summary':value['summary'],'main_seconds':terminal['main']['seconds'],'audit_seconds':terminal['audit']['seconds'],'new_scientific_calls':0}),flush=True)
if __name__=='__main__':main()
