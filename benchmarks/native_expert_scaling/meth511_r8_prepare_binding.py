"""Metadata-only bind actual Torch testing sources; retain the completed C stage."""
import ctypes
from ctypes import wintypes
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import psutil
import meth511_r8_operations as O
start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);peak=0;handles=[]
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];k.CreateFileW.restype=wintypes.HANDLE
k.CloseHandle.argtypes=[wintypes.HANDLE];k.CloseHandle.restype=wintypes.BOOL
def guard():
    global peak
    peak=max(peak,proc.memory_info().peak_wset);assert peak<=512<<20 and time.monotonic()-start<=180
def item(path):
    p=Path(path).resolve();s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(8<<20):h.update(data);guard()
    assert (s.st_size,s.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
    return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':h.hexdigest()}
def retain(r):
    h=k.CreateFileW(r['path'],0x80000000,1,None,3,0x80,None);assert h not in (None,ctypes.c_void_p(-1).value),r['path'];handles.append(h)
    s=Path(r['path']).stat();assert (s.st_size,s.st_mtime_ns)==(r['bytes'],r['mtime_ns']);guard()
def closed(instance):
    try:assert abs(psutil.Process(instance['pid']).create_time()-instance['create_time_unix'])>=.002
    except psutil.NoSuchProcess:pass
assert not O.BIND.exists() and not O.BIND.with_suffix('.failure.json').exists()
try:
    previous=item(O.DOC/'meth511_r7_binding.json');assert previous['sha256']=='60992c060ce91a34f0dff6d5dbe2c69633102358c5f69ff1c465f046be4e8a00'
    b=json.loads(Path(previous['path']).read_bytes());catalog={r['path'].lower():r for r in b['catalog']}
    for r in b['catalog']:retain(r)
    fault=item(O.DOC/'meth511_r7_donor_result.failure.json');assert fault['sha256']=='ccf675ff6cea63e8988c8ed60cfde13a2ed522dcc69b271f5f623c6a4209cfeb'
    f=json.loads(Path(fault['path']).read_bytes());assert f['native_calls']==f['model_calls']==0 and "torch\\\\testing\\\\_utils.py" in f['traceback']
    assert f['seconds']+3540<3600;closed(f['process_instance'])
    native=item(O.DOC/'meth511_r7_native_result.json');assert native['sha256']=='278df0c17a7a07000c1bda446f6b090d1f2f27dad8d853464a9ccbf9f6ca681f'
    n=json.loads(Path(native['path']).read_bytes());assert n['binding_sha256']==previous['sha256'] and all(n['gates'].values())
    assert n['native_calls']==23 and n['retained_native_calls']==553 and len(n['commands'])==576;closed(n['process_instance'])
    for command in n['commands']:assert command['returncode']==0;closed(command['process_instance'])
    for row in n['output_inventory']:
        r=item(row['path']);assert (r['bytes'],r['sha256'])==(row['bytes'],row['sha256']);retain(r);catalog[r['path'].lower()]=r
    testing=(O.ROOT/'.venv/Lib/site-packages/torch/testing').resolve();sources=sorted(testing.rglob('*.py'))
    assert len(sources)==105 and sum(p.stat().st_size for p in sources)==4735033
    for p in sources:r=item(p);retain(r);catalog[r['path'].lower()]=r
    for p in [Path(previous['path']),Path(fault['path']),Path(native['path']),O.DOC/'METH_511_R8_TORCH_IMPORT_ADMISSION_20261007.md',
              *sorted((O.ROOT/'benchmarks/native_expert_scaling').glob('meth511_r8*'))]:
        r=item(p);catalog[r['path'].lower()]=r
        if p.name.startswith('meth511_r8') and p.parent.name=='native_expert_scaling':
            assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(O.ROOT).as_posix()],cwd=O.ROOT).replace(b'\r\n',b'\n')
            b['scientific'].append(r)
    deriv=json.loads((O.ROOT/'benchmarks/native_expert_scaling/meth511_r8_derivation.json').read_bytes())
    for row in deriv:
        assert item(row['old'])['sha256']==row['old_sha256'] and item(row['new'])['sha256']==row['new_sha256']
        text=Path(row['old']).read_text(encoding='utf8')
        for v in row['replacements']:assert text.count(v['before'])==v['occurrences'];text=text.replace(v['before'],v['after'])
        assert text==Path(row['new']).read_text(encoding='utf8')
    b.update(catalog=list(catalog.values()),freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),
        retained_completed_native={'raw':native,'binding_sha256':previous['sha256'],'completed_once':True,'total_native_calls':576},
        torch_testing_admission={'previous_binding':previous,'first_pre_inference_fault':fault,'source_root':str(testing),
            'files':len(sources),'bytes':4735033,'no_Torch_code_changed':True,'numerical_calls_repeated':0},
        torch_testing_literal_derivation=deriv)
    b['price']['catalog_files']=len(catalog);b['price']['catalog_bytes']=sum(r['bytes'] for r in catalog.values())
    b['torch_metadata_rebind_resource']={'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'limits':[180,512<<20],'scientific_calls':0}
    O.write(O.BIND,b);r=item(O.BIND);guard();print(json.dumps({'binding':r,'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'native_calls_repeated':0}),flush=True)
except BaseException:
    O.write(O.BIND.with_suffix('.failure.json'),{'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'scientific_calls':0});raise
finally:
    for h in handles:assert k.CloseHandle(h)
