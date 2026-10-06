"""Metadata-only exact510 apparatus price/freeze before any compilation/control."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import psutil
import meth510_operations as O

start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);catalog={};hashed=0;peak=0
def guard():
    global peak
    peak=max(peak,proc.memory_info().peak_wset);assert peak<=512<<20 and time.monotonic()-start<=180
def add(path,expected=None):
    global hashed
    p=Path(path).resolve();key=str(p)
    if key not in catalog:
        before=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            while data:=f.read(8<<20):h.update(data);hashed+=len(data);guard()
        assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
        catalog[key]={'path':key,'bytes':before.st_size,'sha256':h.hexdigest()}
    if expected:assert catalog[key]['sha256']==expected,key
    return catalog[key]
def main():
    assert not O.BIND.exists() and not O.BIND.with_suffix('.failure.json').exists()
    try:
        prev=add(O.DOC/'meth509_r1_binding.json','e60e36b5aa24da575ed4597ad729405cf8910f9a59f69c316ea16c77046aa682');old=json.loads(Path(prev['path']).read_bytes())
        for row in old['catalog']:
            p=Path(row['path']);rel=p.relative_to(O.ROOT).as_posix() if p.is_relative_to(O.ROOT) else None
            if rel is None or rel.startswith(('.venv/','results/native_expert_scaling/meth324_switch_reference/venv/')):add(p,row['sha256'])
        selected=old['selected']
        for row in selected:add(row['wire']['path'],row['wire']['sha256'])
        oracle=add(O.DOC/'meth509_r1_main_result.json','e9541ad77a3f4e36b6145202c6c815ad0be33c33781161f8adc6698b26f25e3d')
        add(O.DOC/'meth509_r1_audit_result.json','edd64bb4dec6d0a8cdef945aef2ada597123dd1d1de18bffb9b4cdecf97b6c64')
        add(O.DOC/'COMPLETION_509_20261006.json','255ec81abb218f37fefc935bd345a7ddcd6680673592860ae1c0ff83e0161f72')
        original=add(O.DOC/'meth506_r3_binding.json');native=json.loads(Path(original['path']).read_bytes())
        compiler=add(native['compiler']['path'],native['compiler']['sha256'])
        for row in native['compiler_snapshot']:add(row['path'],row['sha256'])
        libomp=add(native['libomp']['path'],native['libomp']['sha256'])
        for row in native['Defender_platform_modules']:add(row['path'],row['sha256'])
        add(O.ROOT/'results/native_expert_scaling/meth506_r4_2_main_windows_terminal.json')
        required=['meth507_operations.py','meth508_operations.py','meth490_r1_operations.py','meth506_model.c','meth506_cost_entry.c','meth506_entry.c','meth388_switch_thread_binding.h']
        for name in required:add(O.ROOT/'benchmarks/native_expert_scaling'/name)
        add(O.ROOT/'benchmarks/phase60/exact_i8_columns.h')
        scientific=[]
        files=sorted((O.ROOT/'benchmarks/native_expert_scaling').glob('meth510*'))+[O.ROOT/'benchmarks/phase60/engine.c',O.DOC/'METH_510_NATIVE_HEAD_PROTOCOL_20261006.md']
        assert all(p.is_file() for p in files)
        for p in files:
            rel=p.relative_to(O.ROOT).as_posix()
            assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+rel],cwd=O.ROOT).replace(b'\r\n',b'\n')
            scientific.append(add(p))
        deriv=json.loads((O.ROOT/'benchmarks/native_expert_scaling/meth510_derivation.json').read_bytes())
        for row in deriv:
            add(row['old'],row['old_sha256']);add(row['new'],row['new_sha256']);text=Path(row['old']).read_text(encoding='utf8')
            for change in row['replacements']:
                assert text.count(change['before'])==change['occurrences'];text=text.replace(change['before'],change['after'])
            assert text==Path(row['new']).read_text(encoding='utf8')
        engine=O.ROOT/'benchmarks/phase60/engine.c';before=subprocess.check_output(['git','show','a673b60:benchmarks/phase60/engine.c'],cwd=O.ROOT).decode('utf8').replace('\r\n','\n')
        expected=before.replace('#ifdef SILICON_SWITCH_EXACT_COLUMNS','#ifdef SILICON_SWITCH_HYBRID_HEAD\n#include "../native_expert_scaling/meth510_entry.c"\n#elif defined(SILICON_SWITCH_EXACT_COLUMNS)',1)
        assert expected==engine.read_text(encoding='utf8')
        preserved={k:add(O.ROOT/k,v)['sha256'] for k,v in old['preserved'].items()}
        extent=old['source_head'];add(extent['manifest']['path'],extent['manifest']['sha256']);cfg,files,tensors=O.manifest(extent['manifest']['path'])
        t=tensors['lm_head.weight'];assert tuple(t[:5])==(0,2,32128,768,1)
        extents=[extent]+[{'path':extent['path'],'file_bytes':extent['file_bytes'],'offset':off,'bytes':size,'sha256':sha,'role':role}
           for off,size,sha,role in [(t[5],32128*768,'94fb5e5b185cc39cd792b4759255dd7370fb5c3bb8541085bc4811a873783adc','original_I8_codes'),
                                    (t[6],32128*4,'2949a367de243f66ee7e5cfa9b1e3b80aed64bdf0a4e0af0342a23d10c419530','original_I8_scales')]]
        for row in extents:
            p=Path(row['path']);s=p.stat();assert s.st_size==row['file_bytes'];h=hashlib.sha256();left=row['bytes']
            with p.open('rb') as f:
                f.seek(row['offset'])
                while left:
                    data=f.read(min(left,8<<20));assert data;left-=len(data);h.update(data);hashed_add(len(data));guard()
            assert h.hexdigest()==row['sha256'] and (s.st_size,s.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
        value={'experiment':'METH510 native fixed top8 parity and paired cost','freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),
            'previous509':prev,'oracle509':oracle,'selected':selected,'source_head':extent,'extents':extents,'compiler':compiler,'libomp':libomp,
            'runtime_environment':native['runtime_environment'],'scientific':scientific,'catalog':list(catalog.values()),
            'python':old['python'],'executable':old['executable'],'packages':old['packages'],'preserved':preserved,'engine_sha256':add(engine)['sha256'],
            'price':{'positions':996,'K':8,'warmups':2,'repetitions':3,'paired_pass_orders':['AB','BA','AB','BA','AB'],
                     'head_input_bytes':3067696,'head_output_bytes':256091544,'native_I8_calls':9960,'native_source_rows':39840,'native_source_MACs':30597120,
                     'logical_source_bytes_per_position':24576,'compiled_commands':1,'positive_control_commands':1,'negative_control_commands':5,'paired_native_commands':1,
                     'main_audit_each_limits':{'seconds':360,'parent_OS_peak_bytes':2<<30,'native_OS_peak_bytes':1<<30,'output_bytes':512<<20},
                     'cost_gates':{'mean_paired_extra_seconds_max':.001,'p95_all2988_paired_extra_seconds_max':.002},
                     'catalog_files':len(catalog),'catalog_bytes':sum(v['bytes'] for v in catalog.values()),'extent_bytes':sum(v['bytes'] for v in extents)},
            'preparation_before_serialization':{'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed}}
        O.write(O.BIND,value);guard();print(json.dumps({'binding':add(O.BIND),'price':value['price'],'terminal_seconds':time.monotonic()-start,'terminal_OS_peak_bytes':peak}),flush=True)
    except BaseException:
        import traceback
        O.write(O.BIND.with_suffix('.failure.json'),{'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed});raise
def hashed_add(n):
    global hashed
    hashed+=n
if __name__=='__main__':main()
