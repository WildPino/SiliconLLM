"""Refresh actual metadata evidence and unchanged runtime plus second operational repair."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth505_r2_binding.json';FAIL=DEST.with_suffix('.failure.json');assert not DEST.exists() and not FAIL.exists()
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
    return {'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()}
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
try:
    old=json.loads((DOC/'meth505_r1_binding.json').read_bytes());f=json.loads((DOC/'meth505_r1_result.failure.json').read_bytes());assert not f['numerical_imports'] and not f['commands'] and 'quiet240s' in f['traceback']
    catalog={str(Path(v['path']).resolve()):item(v['path'],v['sha256']) for v in old['catalog']};records=list(old['records']);scientific=list(old['scientific'])
    def add(p):
        key=str(Path(p).resolve())
        if key not in catalog:catalog[key]=item(p)
        return catalog[key]
    for name in ['meth505_r1_binding.json','meth505_r1_builder_tool_registration.json','meth505_r1_result.failure.json','meth505_r1_first_fault_tool_registration.json']:
        p=DOC/name;head(p);records.append(add(p))
    for p in sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth505_r2*'))+[DOC/'METH_505_R2_METADATA_ISOLATION_PROTOCOL_20261006.md']:
        head(p);scientific.append(add(p))
    win=add(ROOT/'results/native_expert_scaling/meth505_r1_windows_terminal.json');w=json.loads(Path(win['path']).read_bytes());assert w['query_available'] and not w['matching_scientific_events'] and w['query_error'] is None
    b={**old,'experiment':'METH505-R2 metadata-only operational recovery','catalog':list(catalog.values()),'records':records,'scientific':scientific,'pre_numerical_R1_failure':add(DOC/'meth505_r1_result.failure.json'),'R1_windows':win,
       'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},
       'preparation_seconds':time.monotonic()-START,'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'argv':sys.argv,
       'new_numerical_native_compile_export_or_timing_calls':0,'second_repair_scope':'Allow only exact named external Kaggle fetch during metadata, require its creation after original benchmark terminal, prohibit all child calls; numerical protocol unchanged.'}
    guard()
    with DEST.open('xb') as stream:stream.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    v=item(DEST);guard();print(json.dumps({'binding':v,'resource':{'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},'process_instance':b['process_instance']}))
except BaseException:
    with FAIL.open('xb') as stream:stream.write((json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
