"""Metadata-only source/runtime/compiler/exclusion binding; no numerical imports."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import psutil
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth506_r1_binding.json';assert not DEST.exists() and not DEST.with_suffix('.failure.json').exists()
START=time.monotonic();PROC=psutil.Process();PROC.cpu_affinity([10]);PEAK=HASHED=0
def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset)
    assert PEAK<=512<<20 and time.monotonic()-START<=1800
def item(path,expected=None):
    global HASHED
    p=Path(path).resolve();s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(4<<20):h.update(data);HASHED+=len(data);guard()
    assert (p.stat().st_size,p.stat().st_mtime_ns)==(s.st_size,s.st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()}
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
try:
    catalog={};scientific=[];records=[]
    def add(path,expected=None):
        key=str(Path(path).resolve())
        if key not in catalog:catalog[key]=item(key,expected)
        elif expected:assert catalog[key]['sha256']==expected
        return catalog[key]
    old=json.loads((DOC/'meth500_binding.json').read_bytes());native=json.loads((DOC/'meth499_binding.json').read_bytes());tool=json.loads((DOC/'meth486_binding.json').read_bytes())
    runtime={'python':sys.version,'executable':str(Path(sys.executable).resolve()),'packages':{}}
    assert runtime['executable']==old['runtime']['executable'] and runtime['python']==old['runtime']['python']
    for path,v in old['runtime']['files'].items():add(path,v['sha256'])
    # Pin installed optional imports as well as required dependencies; no installation.
    queue=[dist.metadata['Name'] for dist in metadata.distributions()];seen=set()
    while queue:
        name=queue.pop();name=re.sub('[-_.]+','-',name).lower()
        if name in seen:continue
        seen.add(name);dist=metadata.distribution(name);files=[]
        for rel in dist.files or []:
            if str(rel).lower().endswith(('.py','.pyd','.dll','.exe','.json','.model','.so','metadata','record')):
                p=Path(dist.locate_file(rel))
                if p.is_file():files.append(add(p))
        runtime['packages'][name]={'version':dist.version,'files':files}
        for text in dist.requires or []:
            req=Requirement(text)
            if not req.marker or req.marker.evaluate({'extra':''}):queue.append(req.name)
    print(json.dumps({'phase':'runtime_terminal','seconds':time.monotonic()-START,'hashed_bytes':HASHED,'files':len(catalog)}),flush=True)
    compiler=add(native['compiler']['path'],native['compiler']['sha256'])
    snapshot=[add(v['path'],v['sha256']) for v in native['compiler_snapshot']];assert len(snapshot)==5353
    libomp=add(tool['toolchain'][1]['path'],tool['toolchain'][1]['sha256'])
    # Actual OS closure broad enough to retain compiler/loader/AppCompat imports prospectively.
    os_modules=[]
    for p in sorted(Path(os.environ['SystemRoot']).joinpath('System32').glob('*.dll')):os_modules.append(add(p))
    print(json.dumps({'phase':'compiler_OS_terminal','seconds':time.monotonic()-START,'hashed_bytes':HASHED,'files':len(catalog)}),flush=True)
    names=['meth378_switch_base128_acquisition_result.json','meth379_switch_tensor_binding_result.json','meth380_switch_base128_export_result.json','ADMISSION_505_R2_20261006.json','meth506_source_derivation.json','meth506_quality_derivation.json','meth506_binding.failure.json','meth506_binding_tool_receipt.json']
    for name in names:
        p=DOC/name;head(p);records.append(add(p))
    acquired=json.loads((DOC/names[0]).read_bytes());donor=[add(v['path'],v['sha256']) for v in acquired['files']]
    print(json.dumps({'phase':'donor_envelopes_terminal','seconds':time.monotonic()-START,'hashed_bytes':HASHED,'files':len(catalog)}),flush=True)
    source=ROOT/'results/native_expert_scaling/meth380_switch_w8a8_export'
    payload=add(source/'weights.bin','6bee473e1797332b4650b69a3ba6771128d24e7395d4b961515d844fd71c0cfe')
    manifest=add(source/'manifest.bin','3a4892ad2bc846937a3ad7898048df64b9a26dc391a3d02c110fe75cd614511a')
    corpus_rel='data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet'
    corpus=add(ROOT/corpus_rel,'15f3ecb5e314ce58bbf8eae927b51bf893b9fdabe4a04c8e246547206357a2d5')
    paths=subprocess.check_output(['rg','--files','docs/research','results/native_expert_scaling','benchmarks/donor_adaptation','-g','*.json','-g','*.jsonl'],cwd=ROOT,text=True).splitlines()
    specific=re.compile(re.escape(('pg19:'+corpus_rel+':row=').encode())+rb'(\d+)')
    generic=re.compile(rb'"(?:source_row|corpus_row)"\s*:\s*(\d+)');rows=set();ledgers=[]
    for rel in sorted(paths):
        if Path(rel).name.startswith('meth506') or '/meth506' in rel.replace('\\','/'):continue
        p=ROOT/rel;h=hashlib.sha256();found=set();generic_rows=set();has=False;tail=b''
        with p.open('rb') as f:
            while data:=f.read(4<<20):
                h.update(data);v=tail+data;has|=corpus_rel.encode() in v
                found.update(map(int,specific.findall(v)));generic_rows.update(map(int,generic.findall(v)));tail=v[-512:];guard()
        if has:found.update(generic_rows)
        rows.update(found)
        v={'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'excluded_rows':sorted(found)}
        ledgers.append(v)
        # Every actual source-exposing ledger is pinned again by the scientific main.
        if found:add(p,v['sha256'])
    print(json.dumps({'phase':'source_and_exclusions_terminal','seconds':time.monotonic()-START,'hashed_bytes':HASHED,'files':len(catalog)}),flush=True)
    assert len(rows)>=110 and rows.issubset(range(1243))
    paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth506*'))+[ROOT/'benchmarks/native_expert_scaling'/n for n in ['meth490_r1_operations.py','meth388_switch_thread_binding.h','meth388_switch_three_workers.c','meth328_switch_native_contract.py','meth351_switch_generation_reference.py','meth387_switch_teacher_reference.py','meth387_switch_base128_multi_span_quality.py']]+[ROOT/'benchmarks/phase60/engine.c',ROOT/'benchmarks/phase60/exact_i8_columns.h',DOC/'METH_506_WHOLE_PROTOCOL_20261006.md']
    for p in paths:head(p);scientific.append(add(p))
    for rel,sha in old['preserved'].items():add(ROOT/rel,sha)
    engine=add(ROOT/'benchmarks/phase60/engine.c')
    b={'experiment':'METH506 whole conditional artifact','runtime':runtime,'catalog':list(catalog.values()),'scientific':scientific,'records':records,
       'compiler':compiler,'compiler_snapshot':snapshot,'libomp':libomp,'OS_modules':os_modules,'runtime_environment':tool['runtime_environment']|{'OMP_NUM_THREADS':'3'},
       'source_payload':payload,'source_manifest':manifest,'corpus':corpus,'donor':donor,'donor_directory':str(Path(donor[0]['path']).parent),
       'exclusions':{'rows':sorted(rows),'ledgers':ledgers,'scope':'All retained JSON/JSONL under docs/research, results/native_expert_scaling, benchmarks/donor_adaptation; full book exclusion; source-exposing ledgers pinned in catalog.'},
       'preserved':old['preserved'],'engine_sha256':engine['sha256'],'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},'preparation_seconds':time.monotonic()-START,'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'argv':sys.argv,
       'limits':{'main_audit_seconds_each':3600,'combined_OS_peak_bytes':24<<30,'child_seconds':120,'all_new_outputs_bytes':24<<30},
       'priced_outputs':{'candidate_payload':7541946880,'gen_2arms_96cases_4runs_cap64':9393595392,'teacher_2arms_96cases':701115648,'profile1_2arms_96cases_cap64':2348398848,'donor_references_and_official_heads_cap64':2600000000,'metadata_binding_logs_binary_audit_cap':256<<20},
       'operational_repair':{'revision':'506-R1','original_fault_sha256':'615596c73b78c31c7a279d963dc3722c027406325ba755d23f0ca375091e39b8','metadata_preparation_seconds':1800,'original_seconds':600,'scientific_observations_before_repair':0,'scientific_kernel_cohort_quality_economic_limits_changed':False},
       'new_variable':'All12 bank composition, full core/head/normalized routing and fresh tasks; unchanged505 kernel,3 physical cores0,2,4; observer/Python10','GPU_calls':0,'optimizer_updates':0}
    assert sum(b['priced_outputs'].values())<=24<<30;guard()
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    v=item(DEST);print(json.dumps({'binding':v,'resource':{'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},'excluded_rows':len(rows),'source_ledgers':len(ledgers),'runtime_packages':sorted(runtime['packages']),'process_instance':b['process_instance']}),flush=True)
except BaseException:
    with DEST.with_suffix('.failure.json').open('xb') as f:f.write((json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
