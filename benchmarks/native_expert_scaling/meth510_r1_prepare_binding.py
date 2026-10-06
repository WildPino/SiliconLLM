"""510-R1 metadata only; preserve seven closed calls and bind exact idle app instances."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import psutil
import meth510_r1_operations as O

start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);catalog={};peak=hashed=0
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
        original=add(O.DOC/'meth510_binding.json','03090214fb8320069cbd60804484fcee89497b3de9b0e552ebf49c8b5cc8f80c');b=json.loads(Path(original['path']).read_bytes())
        runtime=add(O.DOC/'meth509_r1_binding.json','e60e36b5aa24da575ed4597ad729405cf8910f9a59f69c316ea16c77046aa682');small=json.loads(Path(runtime['path']).read_bytes())
        for row in small['catalog']:
            p=Path(row['path']);rel=p.relative_to(O.ROOT).as_posix() if p.is_relative_to(O.ROOT) else None
            if rel is None or rel.startswith(('.venv/','results/native_expert_scaling/meth324_switch_reference/venv/')):add(p,row['sha256'])
        for row in b['selected']:add(row['wire']['path'],row['wire']['sha256'])
        for row in b['scientific']:add(row['path'],row['sha256'])
        for name in ('meth507_operations.py','meth508_operations.py','meth490_r1_operations.py'):add(O.ROOT/'benchmarks/native_expert_scaling'/name)
        add(b['oracle509']['path'],b['oracle509']['sha256']);add(b['source_head']['manifest']['path'],b['source_head']['manifest']['sha256'])
        scientific=b['scientific'].copy()
        for p in sorted((O.ROOT/'benchmarks/native_expert_scaling').glob('meth510_r1*'))+[O.DOC/'METH_510_R1_OPERATIONAL_RESUME_20261006.md']:
            assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(O.ROOT).as_posix()],cwd=O.ROOT).replace(b'\r\n',b'\n');scientific.append(add(p))
        B=O.ROOT/'benchmarks/native_expert_scaling'
        assert (B/'meth510_r1_retention_audit.py').read_text(encoding='utf8').replace('import meth510_r1_operations as O','import meth510_operations as O').replace('meth510_r1_main_result.json','meth510_main_result.json').replace("results/native_expert_scaling/meth510_r1_main'","results/native_expert_scaling/meth510_main'")==(B/'meth510_retention_audit.py').read_text(encoding='utf8')
        failure=add(O.DOC/'meth510_main_result.failure.json');f=json.loads(Path(failure['path']).read_bytes());assert 'foreign_before_timing' in f['traceback'] and len(f['commands'])==7
        assert [r['label'] for r in f['commands']]==['compile','positive','negative_rows','negative_cols','negative_dtype','negative_source','negative_rounding']
        assert all(r['returncode']==r['expected_exit'] for r in f['commands'])
        out=O.ROOT/'results/native_expert_scaling/meth510_main';assert not (out/'head_outputs.bin').exists() and not (out/'paired_head.stdout').exists()
        inventory=[add(p) for p in sorted(out.iterdir()) if p.is_file()]
        binary=add(out/'meth510.exe','b49c153cc4d1730c2bdbe5ba05f04f499d9a8706f6f63340b445086cd5ff424f');packet=add(out/'head_inputs.bin')
        add(out/'libomp.dll',b['libomp']['sha256']);add(b['libomp']['path'],b['libomp']['sha256'])
        add(O.DOC/'meth510_first_fault_tool_receipts.json');add(O.ROOT/'results/native_expert_scaling/meth506_r4_2_main_windows_terminal.json')
        preserved={rel:add(O.ROOT/rel,sha)['sha256'] for rel,sha in b['preserved'].items()}
        for row in b['extents']:
            p=Path(row['path']);s=p.stat();assert s.st_size==row['file_bytes'];h=hashlib.sha256();left=row['bytes']
            with p.open('rb') as stream:
                stream.seek(row['offset'])
                while left:
                    data=stream.read(min(left,8<<20));assert data;left-=len(data);h.update(data);hashadd(len(data));guard()
            assert h.hexdigest()==row['sha256'] and (s.st_size,s.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
        idle=[]
        for pid,created in [(22920,1791315164.9781234),(31080,1791315166.0894115),(22656,1791315166.0971096)]:
            p=psutil.Process(pid);assert abs(p.create_time()-created)<.002
            idle.append({'pid':pid,'create_time_unix':p.create_time(),'name':p.name(),'exe':p.exe(),'executable':add(p.exe()),'cpu_seconds':sum(p.cpu_times()[:2]),'descendants':sorted((v.pid,v.create_time()) for v in p.children(recursive=True))})
        time.sleep(2)
        for row in idle:
            p=psutil.Process(row['pid']);assert p.create_time()==row['create_time_unix'] and sum(p.cpu_times()[:2])==row['cpu_seconds']
            assert sorted((v.pid,v.create_time()) for v in p.children(recursive=True))==row['descendants']
        value={**b,'experiment':'METH510-R1 same native head; first paired invocation only','freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),
            'scientific':scientific,'catalog':list(catalog.values()),'preserved':preserved,'idle_processes':idle,
            'resume':{'original_binding':original,'original_failure':failure,'original_output_inventory':inventory,'compiled_binary':binary,'packed_inputs':packet,
                      'completed_calls_reused_without_execution':7,'original_paired_head_calls':0,'new_compile_control_calls':0,'new_paired_head_calls':1,
                      'mathematical_main_function':'EXACT original meth510_native_head.main with only original.O rebound','auditor_numeric_source_literal_equal_after_operational_paths':True},
            'preparation_before_serialization':{'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed}}
        value['price']={**b['price'],'R1_new_compile_and_control_commands':0,'R1_new_paired_commands':1,'catalog_files':len(catalog),'catalog_bytes':sum(v['bytes'] for v in catalog.values())}
        O.write(O.BIND,value);guard();print(json.dumps({'binding':add(O.BIND),'price':value['price'],'idle_processes':idle,'terminal_seconds':time.monotonic()-start,'terminal_OS_peak_bytes':peak}),flush=True)
    except BaseException:
        import traceback
        O.write(O.BIND.with_suffix('.failure.json'),{'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed});raise
def hashadd(n):
    global hashed
    hashed+=n
if __name__=='__main__':main()
