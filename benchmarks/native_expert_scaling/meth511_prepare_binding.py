"""Metadata-only full-file SHA, current exposure ledger scan and actual apparatus prices."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import psutil
import meth511_operations as O

start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);catalog={};hashed=peak=0
def guard():
    global peak
    peak=max(peak,proc.memory_info().peak_wset);assert peak<=512<<20 and time.monotonic()-start<=900
def add(path,expected=None):
    global hashed
    p=Path(path).resolve();key=str(p).lower()
    if key not in catalog:
        before=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            while data:=f.read(8<<20):h.update(data);hashed+=len(data);guard()
        after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p)
        catalog[key]={'path':str(p),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'sha256':h.hexdigest()}
    if expected:assert catalog[key]['sha256']==expected,str(p)
    return catalog[key]
def tree(directory,exclude=()):
    for base,dirs,files in os.walk(directory,followlinks=True):
        dirs[:]=[d for d in dirs if d not in exclude]
        for f in files:add(Path(base)/f)
def main():
    assert sys.flags.isolated and sys.dont_write_bytecode and not O.BIND.exists() and not O.BIND.with_suffix('.failure.json').exists()
    try:
        setup=add(O.DOC/'meth511_runtime_setup_result.json','3c3078b222fe19e843b7e4e3c5cccf5e162e96dbdfd0d3fef390c90e50808e65');s=json.loads(Path(setup['path']).read_bytes())
        old=json.loads((O.DOC/'meth506_r4_2_binding.json').read_bytes());head=json.loads((O.DOC/'meth510_r1_binding.json').read_bytes())
        add(O.DOC/'meth506_r4_2_binding.json');add(O.DOC/'meth510_r1_binding.json')
        packages={n:importlib.metadata.version(n) for n in s['packages_reused_without_reinstallation']};packages['duckdb']=importlib.metadata.version('duckdb');assert packages['duckdb']=='1.4.4'
        assert all(packages[n]==r['version'] for n,r in s['packages_reused_without_reinstallation'].items())
        for n in s['absent_optional_packages']:
            try:importlib.metadata.version(n)
            except importlib.metadata.PackageNotFoundError:continue
            raise AssertionError(('unexpected_optional_distribution',n))
        runtime=Path(sys.executable).parent.parent;tree(runtime)
        base=Path(sys.base_prefix);tree(base/'Lib',exclude=('site-packages',));tree(base/'DLLs')
        for p in base.iterdir():
            if p.is_file():add(p)
        corpus=add(old['corpus']['path'],old['corpus']['sha256']);source=add(old['source_payload']['path'],old['source_payload']['sha256']);smanifest=add(old['source_manifest']['path'],old['source_manifest']['sha256'])
        candidate=add(O.ROOT/'results/native_expert_scaling/meth506_r3_artifact/weights.bin','07d10d07db2d96f5d2f8a86351d06a94637b6067a0cee8de40dcd4490a5eaf39')
        cmanifest=add(O.ROOT/'results/native_expert_scaling/meth506_r3_artifact/manifest.bin','7a89d1a9ab78fc8b6576dce0db47127e85953e706f6929e89c95d822c1d788ef')
        sourcebin=add(O.ROOT/'results/native_expert_scaling/meth506_whole/meth506.exe','a4ad2bd4bb727007578d9b0db24fc1b8ecc23d233d35b06522095407a85fe061')
        candidatebin=add(O.ROOT/'results/native_expert_scaling/meth510_r1_main/meth510.exe','b49c153cc4d1730c2bdbe5ba05f04f499d9a8706f6f63340b445086cd5ff424f')
        for directory in [Path(sourcebin['path']).parent,Path(candidatebin['path']).parent]:add(directory/'libomp.dll','6fc163dd513538a92a187d987438bebc5509afd1f824b20a88dd51463b3d5698')
        donor=[add(r['path'],r['sha256']) for r in old['donor']]
        for r in old['Defender_platform_modules']:add(r['path'])
        reference=add(old['reference_model_code']['path'],old['reference_model_code']['sha256'])
        for name in ['meth379_switch_tensor_binding_result.json','meth380_switch_base128_export_result.json','RETENTION_506_R4_4_20261006.json',
                     'COMPLETION_510_20261006.json','meth510_r1_main_result.json','meth510_r1_audit_result.json']:
            add(O.DOC/name)
        control=add(O.ROOT/'results/native_expert_scaling/meth506_r3_artifact/cohort.json')
        deriv=json.loads((O.ROOT/'benchmarks/native_expert_scaling/meth511_derivation.json').read_bytes())
        for r in deriv:add(r['old'],r['old_sha256']);add(r['new'],r['new_sha256'])
        q=Path(deriv[0]['old']).read_text(encoding='utf8');parts=[ast.get_source_segment(q,n) for n in ast.parse(q).body if isinstance(n,ast.FunctionDef) and n.name in deriv[0]['changes']['literal_AST_functions']]
        assert Path(deriv[0]['new']).read_text(encoding='utf8').endswith('\n\n'.join(parts)+'\n')
        result=Path(deriv[1]['old']).read_text(encoding='utf8').replace('import meth506_quality as Q','import meth511_quality as Q').replace("task=c['candidate_task'];oc+=","task=c[arm+'_task'];oc+=")
        assert result==Path(deriv[1]['new']).read_text(encoding='utf8')
        scientific=[]
        for p in sorted((O.ROOT/'benchmarks/native_expert_scaling').glob('meth511*'))+[O.DOC/'METH_511_WHOLE_PROTOCOL_20261006.md',O.ROOT/'benchmarks/phase60/engine.c']:
            if not p.is_file():continue
            assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(O.ROOT).as_posix()],cwd=O.ROOT).replace(b'\r\n',b'\n')
            scientific.append(add(p))
        for name in ['meth506_quality.py','meth506_results.py','meth506_model.c','meth506_cost_entry.c','meth506_entry.c','meth388_switch_thread_binding.h',
                     'meth510_model.c','meth510_cost_entry.c','meth510_head.h','meth510_entry.c','meth510_control.h']:
            add(O.ROOT/'benchmarks/native_expert_scaling'/name)
        add(O.ROOT/'benchmarks/phase60/exact_i8_columns.h')
        preserved={rel:add(O.ROOT/rel,expected)['sha256'] for rel,expected in {
            'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
            'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
            'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}.items()}
        # Every existing JSON/JSONL exposure ledger in explicit project evidence roots,
        # including ignored/untracked partials and merged donor evidence. No511 self-reference.
        listing=subprocess.check_output(['rg','--files','--hidden','--no-ignore','docs/research','results/native_expert_scaling','benchmarks/donor_adaptation','-g','*.json','-g','*.jsonl'],cwd=O.ROOT,text=True).splitlines()
        ids=re.compile(rb'pg19:data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet:row=(\d+)')
        generic=re.compile(rb'"(?:source_row|corpus_row)"\s*:\s*(\d+)');exclusions=set();ledgers=[]
        for rel in sorted(listing):
            p=O.ROOT/rel
            if 'meth511' in str(p).lower() or '/venv/' in p.as_posix().lower() or '/site-packages/' in p.as_posix().lower():continue
            r=add(p);direct=set();possible=set();associated=False;tail=b''
            with p.open('rb') as f:
                while data:=f.read(4<<20):
                    chunk=tail+data;direct.update(int(m.group(1)) for m in ids.finditer(chunk));possible.update(int(m.group(1)) for m in generic.finditer(chunk))
                    associated|=b'train-00014-of-00023-54b567998cd5eb4b.parquet' in chunk;tail=chunk[-512:];guard()
            rows=direct|(possible if associated else set());assert all(0<=r<1243 for r in rows),(rel,rows)
            exclusions.update(rows);ledgers.append({'file':r,'exposed_rows':sorted(rows),'associated_corpus':associated})
        assert len(exclusions)>=359
        idle=[]
        for row in head['idle_processes']:
            p=psutil.Process(row['pid']);assert p.create_time()==row['create_time_unix'] and p.exe()==row['exe']
            idle.append({**row,'executable':add(p.exe(),row['executable']['sha256']),'cpu_seconds':sum(p.cpu_times()[:2]),
                'descendants':sorted((q.pid,q.create_time()) for q in p.children(recursive=True))})
        generation_wire=36+14*29*768*4+64*14*768*4+64*32128*4+(174+6*64)*12
        teacher_wire=36+14*29*768*4+14*14*768*4+14*32128*4+(174+6*14)*12
        assert generation_wire==12231244 and teacher_wire==3651644
        price={'books':24,'cases':96,'source_tokens':29,'teacher_steps':14,'generation_cap':64,'K':8,'workers':3,
            'native_calls':576,'native_compile_control_replays':0,'donor_high_level_calls':384,
            'max_generation_wire_bytes':generation_wire,'teacher_wire_bytes':teacher_wire,
            'generation_unprofiled_wire_bytes':2*96*4*generation_wire,'generation_profile_wire_bytes':2*96*generation_wire,'teacher_wire_total_bytes':2*96*teacher_wire,
            'donor_saved_arrays_max_bytes':2600000000,'new_total_output_bound_bytes':18<<30,'stage_limits':O.LIMITS,
            'native_mapping_bytes':7541946880,'original_donor_mapping_bytes':sum(r['bytes'] for r in donor if Path(r['path']).suffix=='.bin'),
            'worst_unique_expert_pairs_per_pass':6*(29+64),'expert_pair_code_scale_bytes':2*768*3072+(3072+768)*4,
            'audit_max_head_states':2*96*(14+64),'audit_max_integer_MACs':2*96*(14+64)*32128*768,
            'audit_F64_integer_matrix_bytes':32128*768*8,'physical_RAM_bytes':psutil.virtual_memory().total,
            'disk_free_bytes':shutil.disk_usage(O.ROOT).free,'metadata_binder_limits':[900,512<<20],
            'catalog_files':len(catalog),'catalog_bytes':sum(r['bytes'] for r in catalog.values()),'ledger_files':len(ledgers)}
        assert price['disk_free_bytes']>=2*price['new_total_output_bound_bytes'] and price['physical_RAM_bytes']>=64<<30
        value={'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),'python':sys.version,'executable':str(Path(sys.executable).resolve()),
            'packages':packages,'catalog':list(catalog.values()),'scientific':scientific,'preserved':preserved,'source_payload':source,'source_manifest':smanifest,
            'candidate_payload':candidate,'candidate_manifest':cmanifest,'source_binary':sourcebin,'candidate_binary':candidatebin,'donor':donor,
            'donor_directory':old['donor_directory'],'reference_model_code':reference,'cohort_control':control,'corpus':corpus,
            'exclusions':{'rows':sorted(exclusions),'ledgers':ledgers,'scope':'Explicit all JSON/JSONL evidence roots, including untracked/ignored; associated generic rows conservatively excluded at whole-book level; no donor pretraining exclusion claim.'},
            'idle_processes':idle,'head_extents':head['extents'],'runtime_environment':O.ENV,'runtime_setup':setup,'price':price,
            'preparation_before_serialization':{'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed},'scientific_calls':0}
        O.write(O.BIND,value);guard();print(json.dumps({'binding':add(O.BIND),'price':price,'terminal_seconds':time.monotonic()-start,'terminal_OS_peak_bytes':peak}),flush=True)
    except BaseException:
        O.write(O.BIND.with_suffix('.failure.json'),{'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed,'scientific_calls':0});raise
if __name__=='__main__':main()
