"""Actual source/runtime/toolchain binding only, no numerical import or observation."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth505_binding.json';FAIL=DEST.with_suffix('.failure.json');assert not DEST.exists() and not FAIL.exists()
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
    catalog={};records=[];scientific=[]
    def add(v):
        p=str(Path(v['path']).resolve())
        if p not in catalog:catalog[p]=item(p,v.get('sha256'))
        else:assert not v.get('sha256') or catalog[p]['sha256']==v['sha256']
        return catalog[p]
    old=json.loads((DOC/'meth500_binding.json').read_bytes());native=json.loads((DOC/'meth499_binding.json').read_bytes());tool=json.loads((DOC/'meth486_binding.json').read_bytes())
    for name in ['meth500_binding.json','meth499_binding.json','meth486_binding.json','ADMISSION_500_20261006.json','meth500_overlap_result.json','ADMISSION_504_20261006.json','meth504_region_result.json','RETENTION_504_20261006.json','meth380_switch_base128_export_result.json']:
        p=DOC/name;head(p);v=add({'path':str(p)});records.append(v)
    for name in ['ADMISSION_500_20261006.json','ADMISSION_504_20261006.json']:
        a=json.loads((DOC/name).read_bytes());assert a['main_completed'] and all(a['gates'].values())
        for k in ['raw','retention']:add(a[k])
    data={k:add(old['data'][k]) for k in ['uid','occurrences','inputs','targets','payload','manifest']}
    raw=json.loads((DOC/'meth500_overlap_result.json').read_bytes());inv={Path(v['path']).name:v for v in raw['output_inventory']}
    data.update({k:add(inv[v]) for k,v in {'hidden':'hidden.npy','codes':'hidden_codes.npy','scales':'hidden_scales.npy'}.items()})
    runtime=old['runtime']
    for files in [runtime['files']]+[v['files'] for v in runtime['packages'].values()]:
        for path,v in files.items():add({'path':path,'sha256':v['sha256']})
    snapshot=[add(v) for v in native['compiler_snapshot']];assert len(snapshot)==5353
    modules=[add(v) for v in native['native_modules']]
    # Common actual OS imports of the compiler and OpenMP runtime, prospectively pinned.
    for name in ['psapi.dll','ucrtbase.dll','advapi32.dll','sechost.dll','rpcrt4.dll','bcrypt.dll','bcryptprimitives.dll','user32.dll','win32u.dll','gdi32.dll','gdi32full.dll','msvcp_win.dll','shell32.dll','ole32.dll','combase.dll','shlwapi.dll','ws2_32.dll','imm32.dll','version.dll']:
        modules.append(add({'path':str(Path('C:/Windows/System32')/name)}))
    libomp=add(tool['toolchain'][1]);compiler=add(native['compiler'])
    paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth505*'))+[ROOT/'benchmarks/native_expert_scaling'/n for n in ['meth490_r1_operations.py','meth374_switch_physical_workers.c','meth374_switch_thread_binding.h']]+[DOC/'METH_505_EXACT_SPARSE_PROTOCOL_20261006.md']
    for p in paths:head(p);v=add({'path':str(p)});scientific.append(v)
    for rel,sha in old['preserved'].items():add({'path':str(ROOT/rel),'sha256':sha})
    add({'path':str(ROOT/'benchmarks/phase60/engine.c'),'sha256':old['engine_sha256']})
    b={'experiment':'METH505 exact original WI/sparse WO physical inquiry','runtime':runtime,'catalog':list(catalog.values()),'records':records,'scientific':scientific,'data':data,
       'compiler':compiler,'compiler_snapshot':snapshot,'native_modules':modules,'libomp':libomp,'runtime_environment':tool['runtime_environment']|{'OMP_NUM_THREADS':'1'},
       'preserved':old['preserved'],'engine_sha256':old['engine_sha256'],'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},'preparation_seconds':time.monotonic()-START,'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'argv':sys.argv,
       'limits':{'seconds_main_audit_each':900,'combined_OS_peak_bytes':4<<30,'child_seconds':120,'all_new_outputs_bytes':1<<30},
       'priced_outputs':{'bank':605949952,'primal':377460824,'timing':841944,'libomp':libomp['bytes'],'binary_and_logs_cap':8<<20,'binding_raw_views_audit_metadata_cap':16<<20},
       'new_variable':'Full original-I8 WI instead of compact approximate I4/rotated WI; exact zero-only original-I8 WO', 'optimizer_updates':0,'GPU_calls':0}
    assert sum(b['priced_outputs'].values())<=1<<30;guard()
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    v=item(DEST);guard();print(json.dumps({'binding':v,'resource':{'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},'process_instance':b['process_instance']}))
except BaseException:
    with FAIL.open('xb') as f:f.write((json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
