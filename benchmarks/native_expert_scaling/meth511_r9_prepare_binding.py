"""Metadata-only audit count repair; original native/donor data remain authoritative."""
import ctypes
from ctypes import wintypes
import datetime,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import psutil
import meth511_r9_operations as O
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
    previous=item(O.DOC/'meth511_r8_binding.json');assert previous['sha256']=='aefbb88c4db513d1c9d461b456c31f18894cfc22859f14e93acc9ebdf971ba81'
    b=json.loads(Path(previous['path']).read_bytes());catalog={r['path'].lower():r for r in b['catalog']}
    for r in b['catalog']:retain(r)
    fault=item(O.DOC/'meth511_r8_audit_result.failure.json');assert fault['sha256']=='e91cfcdf9883e66af001b4a9ad1d4b693bacf62b4dff28d739b761998d1242fe'
    f=json.loads(Path(fault['path']).read_bytes());assert f['native_calls']==f['model_calls']==0 and "native['native_calls']==576" in f['traceback']
    assert f['seconds']+1180<1200;closed(f['process_instance'])
    donor=item(O.DOC/'meth511_r8_donor_result.json');assert donor['sha256']=='4b73168632b025f3d5977e2f2c6e5fa75c495a4d00777bf1488332785d7ae195'
    d=json.loads(Path(donor['path']).read_bytes());assert d['binding_sha256']==previous['sha256'] and d['model_calls']==384 and all(d['gates'].values());closed(d['process_instance'])
    for row in d['output_inventory']:
        r=item(row['path']);assert (r['bytes'],r['sha256'])==(row['bytes'],row['sha256']);retain(r);catalog[r['path'].lower()]=r
    for p in [Path(previous['path']),Path(fault['path']),Path(donor['path']),O.DOC/'METH_511_R9_AUDIT_CALL_COUNT_20261007.md',
              *sorted((O.ROOT/'benchmarks/native_expert_scaling').glob('meth511_r9*'))]:
        r=item(p);catalog[r['path'].lower()]=r
        if p.name.startswith('meth511_r9') and p.parent.name=='native_expert_scaling':
            assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(O.ROOT).as_posix()],cwd=O.ROOT).replace(b'\r\n',b'\n')
            b['scientific'].append(r)
    deriv=json.loads((O.ROOT/'benchmarks/native_expert_scaling/meth511_r9_derivation.json').read_bytes())
    for row in deriv:
        assert item(row['old'])['sha256']==row['old_sha256'] and item(row['new'])['sha256']==row['new_sha256']
        text=Path(row['old']).read_text(encoding='utf8')
        for v in row['replacements']:assert text.count(v['before'])==v['occurrences'];text=text.replace(v['before'],v['after'])
        assert text==Path(row['new']).read_text(encoding='utf8')
    b.update(catalog=list(catalog.values()),freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),
        retained_completed_donor={'raw':donor,'binding_sha256':previous['sha256'],'completed_once':True,'model_calls':384},
        pre_audit_fault_instances=[{'label':'audit_R8_pre_numerical_fault','pid':f['process_instance']['pid'],
            'create_time_unix':f['process_instance']['create_time_unix'],'start_utc':f['started_utc'],'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}],
        audit_count_repair={'previous_binding':previous,'first_pre_numerical_fault':fault,'native_donor_calls_repeated':0,'literal_derivation':deriv})
    b['price']['catalog_files']=len(catalog);b['price']['catalog_bytes']=sum(r['bytes'] for r in catalog.values())
    b['audit_metadata_rebind_resource']={'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'limits':[180,512<<20],'scientific_calls':0}
    O.write(O.BIND,b);r=item(O.BIND);guard();print(json.dumps({'binding':r,'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'native_donor_calls_repeated':0}),flush=True)
except BaseException:
    O.write(O.BIND.with_suffix('.failure.json'),{'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'scientific_calls':0});raise
finally:
    for h in handles:assert k.CloseHandle(h)
