"""Metadata-only reader rebind: reuse closed R2 SHA after stat/write-deny checks."""
import ctypes
from ctypes import wintypes
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import psutil
import meth511_r3_operations as O
start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);peak=0;handles=[]
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];k.CreateFileW.restype=wintypes.HANDLE
k.CloseHandle.argtypes=[wintypes.HANDLE];k.CloseHandle.restype=wintypes.BOOL
def guard():
    global peak
    peak=max(peak,proc.memory_info().peak_wset);assert peak<=512<<20 and time.monotonic()-start<=180
def item(p):
    p=Path(p).resolve();s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(8<<20):h.update(data);guard()
    assert (s.st_size,s.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
    return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':h.hexdigest()}
assert not O.BIND.exists() and not O.BIND.with_suffix('.failure.json').exists()
try:
    previous=item(O.DOC/'meth511_r2_binding.json');assert previous['sha256']=='bb65b0689873356128cedca247053b86348af1b890fcf5c7b6c06165e8cfd8f6'
    b=json.loads(Path(previous['path']).read_bytes());catalog={r['path'].lower():r for r in b['catalog']}
    for r in b['catalog']:
        h=k.CreateFileW(r['path'],0x80000000,1,None,3,0x80,None);assert h not in (None,ctypes.c_void_p(-1).value),r['path'];handles.append(h)
        s=Path(r['path']).stat();assert (s.st_size,s.st_mtime_ns)==(r['bytes'],r['mtime_ns']),r['path'];guard()
    fault=item(O.DOC/'meth511_r2_cohort_result.failure.json');failure=json.loads(Path(fault['path']).read_bytes());assert failure['native_calls']==failure['model_calls']==0
    try:assert abs(psutil.Process(failure['process_instance']['pid']).create_time()-failure['process_instance']['create_time_unix'])>=.002
    except psutil.NoSuchProcess:pass
    for p in [Path(previous['path']),Path(fault['path']),O.DOC/'METH_511_R3_PROJECTED_READER_20261007.md',
              O.DOC/'METH_511_R2A_PREFLIGHT_PATH_20261007.md',O.DOC/'meth511_r2_preflight_failure.json',
              *sorted((O.ROOT/'benchmarks/native_expert_scaling').glob('meth511_r3*'))]:
        assert p.is_file();r=item(p);catalog[r['path'].lower()]=r
        if p.name.startswith('meth511_r3') and p.parent.name=='native_expert_scaling':
            assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(O.ROOT).as_posix()],cwd=O.ROOT).replace(b'\r\n',b'\n')
            b['scientific'].append(r)
    deriv=json.loads((O.ROOT/'benchmarks/native_expert_scaling/meth511_r3_derivation.json').read_bytes())
    for row in deriv:
        assert item(row['old'])['sha256']==row['old_sha256'] and item(row['new'])['sha256']==row['new_sha256']
        text=Path(row['old']).read_text(encoding='utf8')
        for c in row['replacements']:assert text.count(c['before'])==c['occurrences'];text=text.replace(c['before'],c['after'])
        assert text==Path(row['new']).read_text(encoding='utf8')
    b.update(catalog=list(catalog.values()),freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),
        reader_repair={'previous_binding':previous,'first_query_failure':fault,'rows_returned_before_fault':0,'native_model_calls':0,
            'prior_SHA_admission':'Closed R2 full-SHA receipts reused after current stat and retained read/share-read write-deny handles; no new full input SHA claim.',
            'reader_memory_limit':'1536MiB within unchanged2GiB parent bound','pool':'Exact first96 fixed candidates +2 controls; physical1243 IDs checked by aggregate metadata; no unselected text sort.'},
        reader_literal_derivation=deriv)
    b['price']['catalog_files']=len(catalog);b['price']['catalog_bytes']=sum(r['bytes'] for r in catalog.values());b['price']['reader_internal_memory_bytes']=1536<<20
    b['metadata_reader_rebind_resource']={'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'limits':[180,512<<20],'scientific_calls':0}
    O.write(O.BIND,b);r=item(O.BIND);guard();print(json.dumps({'binding':r,'seconds':time.monotonic()-start,'OS_peak_bytes':peak}),flush=True)
except BaseException:
    O.write(O.BIND.with_suffix('.failure.json'),{'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'scientific_calls':0});raise
finally:
    for h in handles:assert k.CloseHandle(h)
