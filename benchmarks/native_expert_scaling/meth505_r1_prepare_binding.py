"""Bind immutable complete partials and actual observed modules; no numerical observation."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth505_r1_binding.json';FAIL=DEST.with_suffix('.failure.json');assert not DEST.exists() and not FAIL.exists()
START=time.monotonic();PROC=psutil.Process();PROC.cpu_affinity([0]);PEAK=HASHED=0
def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset);assert PEAK<=256<<20 and time.monotonic()-START<=180
def item(path,expected=None):
    global HASHED
    p=Path(path).resolve();s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while d:=f.read(4<<20):h.update(d);HASHED+=len(d);guard()
    assert (p.stat().st_size,p.stat().st_mtime_ns)==(s.st_size,s.st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':h.hexdigest()}
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
try:
    old=json.loads((DOC/'meth505_binding.json').read_bytes());failure=json.loads((DOC/'meth505_exact_sparse_result.failure.json').read_bytes())
    reg=json.loads((DOC/'meth505_first_fault_tool_registration.json').read_bytes());assert reg['actual_exit_code']==1
    assert len(failure['commands'])==3 and all(c['returncode']==0 for c in failure['commands']) and 'unbound_actual_module' in failure['traceback'] and 'apphelp.dll' in failure['traceback']
    catalog={}
    def add(v):
        key=str(Path(v['path']).resolve())
        if key not in catalog:catalog[key]=item(key,v.get('sha256'))
        else:assert not v.get('sha256') or catalog[key]['sha256']==v['sha256']
        return catalog[key]
    for v in old['catalog']:add(v)
    partial={}
    for v in failure['partial_outputs']:
        d=add(v);assert d['bytes']==v['bytes'];partial[Path(d['path']).name]=d
    binary=add(failure['binary']);assert binary['sha256']==partial['meth505_exact_sparse_cpu.exe']['sha256'] and partial['libomp.dll']['sha256']==old['libomp']['sha256']
    prior_known={str(Path(v['path']).resolve()).lower():v for v in old['catalog']};prior_known[str(Path(binary['path']).resolve()).lower()]=binary
    prior_known[str(Path(partial['libomp.dll']['path']).resolve()).lower()]=partial['libomp.dll']
    missing=[];prebound=[]
    for c in failure['commands']:
        for p in c['observed_modules']:
            key=str(Path(p).resolve()).lower()
            if key not in prior_known:missing.append({'stage':c['label'],'file':add({'path':p}),'prospectively_bound_for_original_call':False})
            else:prebound.append({'stage':c['label'],'file':add(prior_known[key])})
    assert len(missing)==1 and missing[0]['stage']=='primal' and Path(missing[0]['file']['path']).name.lower()=='apphelp.dll'
    assert any(v['stage']=='bench' for v in prebound)
    records=list(old['records'])
    for p in [DOC/'meth505_binding.json',DOC/'meth505_builder_tool_registration.json',DOC/'meth505_exact_sparse_result.failure.json',DOC/'meth505_first_fault_tool_registration.json']:
        head(p);records.append(add({'path':str(p)}))
    win=add({'path':str(ROOT/'results/native_expert_scaling/meth505_windows_terminal.json')});windows=json.loads(Path(win['path']).read_bytes());assert windows['query_available'] and not windows['matching_scientific_events'] and windows['query_error'] is None
    scientific=list(old['scientific'])
    for p in sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth505_r1*'))+[DOC/'METH_505_R1_METADATA_REPAIR_PROTOCOL_20261006.md']:
        head(p);scientific.append(add({'path':str(p)}))
    b={**old,'experiment':'METH505-R1 metadata-only retained evidence recovery','catalog':list(catalog.values()),'records':records,'scientific':scientific,'partial':partial,'native_binary':binary,
       'retained_failure':add({'path':str(DOC/'meth505_exact_sparse_result.failure.json')}),'original_windows':win,'prebound_observed':prebound,'unbound_primal_OS_exception':missing[0],
       'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},
       'preparation_seconds':time.monotonic()-START,'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'argv':sys.argv,
       'new_numerical_native_compile_export_or_timing_calls':0,'repair_scope':'No prospective hash claim for original primal apphelp.dll. Actual original cost observed modules were all prospectively bound; retain entire native output/times without replay.'}
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    v=item(DEST);guard();print(json.dumps({'binding':v,'resource':{'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},'process_instance':b['process_instance']}))
except BaseException:
    with FAIL.open('xb') as f:f.write((json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
