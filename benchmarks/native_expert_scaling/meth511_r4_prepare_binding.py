"""Metadata-only current idle inventory; exact completed cohort/data receipts reused."""
import ctypes
from ctypes import wintypes
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import psutil
import meth511_r4_operations as O
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
    previous=item(O.DOC/'meth511_r3_binding.json');assert previous['sha256']=='ab63bd873b17b7a7dab390c6c6ca93d816cb9af6715ee056536e81cffa9872e7'
    b=json.loads(Path(previous['path']).read_bytes());catalog={r['path'].lower():r for r in b['catalog']}
    for r in b['catalog']:
        h=k.CreateFileW(r['path'],0x80000000,1,None,3,0x80,None);assert h not in (None,ctypes.c_void_p(-1).value),r['path'];handles.append(h)
        s=Path(r['path']).stat();assert (s.st_size,s.st_mtime_ns)==(r['bytes'],r['mtime_ns']),r['path'];guard()
    fault=item(O.DOC/'meth511_r3_native_result.failure.json');f=json.loads(Path(fault['path']).read_bytes());assert f['native_calls']==f['model_calls']==0
    cohort=item(O.DOC/'meth511_r3_cohort_result.json');assert cohort['sha256']=='90b790c2723e96bc17408f8da4b2285022965276968d61be226b94981059de4c'
    c=json.loads(Path(cohort['path']).read_bytes());assert c['binding_sha256']==previous['sha256'] and all(c['gates'].values())
    for raw in [f,c]:
        try:assert abs(psutil.Process(raw['process_instance']['pid']).create_time()-raw['process_instance']['create_time_unix'])>=.002
        except psutil.NoSuchProcess:pass
    for row in c['output_inventory']:r=item(row['path']);assert r['sha256']==row['sha256'];catalog[r['path'].lower()]=r
    closed=[]
    for row in b['idle_processes']:
        try:p=psutil.Process(row['pid']);assert abs(p.create_time()-row['create_time_unix'])>=.002
        except psutil.NoSuchProcess:pass
        closed.append(row)
    ancestors={proc.pid,*(p.pid for p in proc.parents())};live=[]
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in ancestors:continue
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        if name.startswith(('python','clang','llama','ollama','nvcc','nvidia-smi')) or (name.startswith('meth') and name.endswith('.exe')):
            live.append({'pid':p.pid,'create_time_unix':p.create_time(),'name':p.name(),'exe':p.exe()})
    assert not live,('current_foreign_scientific_processes',live)
    b['idle_processes']=[]
    for p in [Path(previous['path']),Path(fault['path']),Path(cohort['path']),O.DOC/'METH_511_R4_CLOSED_IDLE_APPS_20261007.md',
              *sorted((O.ROOT/'benchmarks/native_expert_scaling').glob('meth511_r4*'))]:
        assert p.is_file();r=item(p);catalog[r['path'].lower()]=r
        if p.name.startswith('meth511_r4') and p.parent.name=='native_expert_scaling':
            assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(O.ROOT).as_posix()],cwd=O.ROOT).replace(b'\r\n',b'\n')
            b['scientific'].append(r)
    deriv=json.loads((O.ROOT/'benchmarks/native_expert_scaling/meth511_r4_derivation.json').read_bytes())
    for row in deriv:
        assert item(row['old'])['sha256']==row['old_sha256'] and item(row['new'])['sha256']==row['new_sha256']
        text=Path(row['old']).read_text(encoding='utf8')
        for v in row['replacements']:assert text.count(v['before'])==v['occurrences'];text=text.replace(v['before'],v['after'])
        assert text==Path(row['new']).read_text(encoding='utf8')
    b.update(catalog=list(catalog.values()),freeze_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),
        retained_cohort={'raw':cohort,'binding_sha256':previous['sha256'],'cohort_sha256':c['cohort_sha256'],'completed_once':True},
        idle_repair={'previous_binding':previous,'first_native_preflight_fault':fault,'proven_closed_idle_instances':closed,'current_live_scientific_processes':live,
            'completed_native_calls':0,'prior_SHA_scope':'Closed full input receipts reused after current stat and retained read/share-read write-deny handles; no new full input SHA claim.'},
        idle_literal_derivation=deriv)
    b['price']['catalog_files']=len(catalog);b['price']['catalog_bytes']=sum(r['bytes'] for r in catalog.values())
    b['idle_metadata_rebind_resource']={'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'limits':[180,512<<20],'scientific_calls':0}
    O.write(O.BIND,b);r=item(O.BIND);guard();print(json.dumps({'binding':r,'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'retained_cohort':cohort['sha256'],'idle_processes':0}),flush=True)
except BaseException:
    O.write(O.BIND.with_suffix('.failure.json'),{'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'scientific_calls':0});raise
finally:
    for h in handles:assert k.CloseHandle(h)
