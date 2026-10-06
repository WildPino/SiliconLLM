"""Original reference environment + immutable native binding. Metadata only."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth506_r4_binding.json';assert not DEST.exists() and not DEST.with_suffix('.failure.json').exists()
START=time.monotonic();PROC=psutil.Process();PROC.cpu_affinity([10]);PEAK=HASHED=0
def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset);assert PEAK<=512<<20 and time.monotonic()-START<=1800
def item(path,expected=None):
    global HASHED
    p=Path(path).resolve();s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(4<<20):h.update(data);HASHED+=len(data);guard()
    assert (p.stat().st_size,p.stat().st_mtime_ns)==(s.st_size,s.st_mtime_ns)
    if expected:assert h.hexdigest()==expected,str(p)
    return {'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()}
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
try:
    assert Path(sys.executable).resolve()==ROOT/'results/native_expert_scaling/meth324_switch_reference/venv/Scripts/python.exe'
    assert metadata.version('transformers')=='4.57.6' and metadata.version('torch')=='2.6.0+cu124'
    source=DOC/'meth506_r3_binding.json';head(source);assert item(source)['sha256']=='58c1bbd8daa6ef5c398dca46892937de72c833268b4b393e97e11b2e1188392e'
    old=json.loads(source.read_bytes());catalog={}
    for v in old['catalog']:catalog[v['path']]=item(v['path'],v['sha256'])
    def add(p,expected=None):
        key=str(Path(p).resolve())
        if key not in catalog:catalog[key]=item(key,expected)
        elif expected:assert catalog[key]['sha256']==expected
        return catalog[key]
    runtime={'python':sys.version,'executable':str(Path(sys.executable).resolve()),'packages':{}}
    # Actual distributions as resolved in the qualified reference environment; shadowed
    # current5.13.1 files remain bound separately as original native/cohort provenance.
    names=sorted({dist.metadata['Name'].lower() for dist in metadata.distributions()})
    for name in names:
        dist=metadata.distribution(name);files=[]
        for rel in dist.files or []:
            if str(rel).lower().endswith(('.py','.pyd','.dll','.exe','.json','.model','.so','metadata','record','.pth')):
                p=Path(dist.locate_file(rel))
                if p.is_file():files.append(add(p))
        runtime['packages'][name]={'version':dist.version,'files':files}
    add(sys.executable);add(Path(sys.prefix)/'pyvenv.cfg')
    for p in sorted((Path(sys.prefix)/'Lib/site-packages').glob('*.pth')):add(p)
    code=Path(sys.prefix)/'Lib/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py'
    reference_code=add(code,'5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83')
    failure=DOC/'meth506_whole_result.failure.json';head(failure);fv=add(failure,'a678463eaeaf90c961c95b692cf3972791a43ece354fd4281fc7745422b11465');failed=json.loads(failure.read_bytes())
    assert len(failed['commands'])==578 and all(v['returncode']==0 and v['observed_modules_prebound_or_actual_build'] for v in failed['commands'])
    assert all(failed['gates'].values()) and failed['phase']=='fresh-donor-own-states-quality'
    native=ROOT/'results/native_expert_scaling/meth506_whole';native_inventory=[]
    for p in sorted(native.iterdir()):
        if p.is_file():native_inventory.append(add(p))
    records=old['records']+[add(source),fv]
    for name in ['meth506_r3_preparation_result.json','meth506_native_and_reference_fault_tool_receipt.json']:
        p=DOC/name;head(p);records.append(add(p))
    prep=json.loads((DOC/'meth506_r3_preparation_result.json').read_bytes())
    for v in prep['output_inventory']:add(v['path'],v['sha256'])
    original_windows=add(ROOT/'results/native_expert_scaling/meth506_r3_main_windows_terminal.json');w=json.loads(Path(original_windows['path']).read_bytes());assert w['query_available'] and not w['matching_scientific_events']
    scientific=[]
    paths=[Path(v['path']) for v in old['scientific']]+sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth506_r4*'))+[DOC/'METH_506_R4_REFERENCE_RECOVERY_PROTOCOL_20261006.md']
    for p in sorted(set(paths)):head(p);scientific.append(add(p))
    b={**old,'runtime':runtime,'catalog':list(catalog.values()),'scientific':scientific,'records':records,'reference_model_code':reference_code,'immutable_native_inventory':native_inventory,'original_native_windows':original_windows,'original_native_failure':fv,'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},'preparation_seconds':time.monotonic()-START,'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'argv':sys.argv,
       'operational_repair':{'revision':'506-R4','reference_transformers':'4.57.6','failed_reference_transformers':'5.13.1','trained_coefficients_changed':False,'native_calls_replayed':0,'cohort_resampled':False,'quality_economic_thresholds_changed':False,'strict_original_timing_isolation_verified':False}}
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'binding':item(DEST),'reference_model_code':reference_code,'native_files':len(native_inventory),'resource':{'seconds':time.monotonic()-START,'peak_bytes':PEAK,'bytes_hashed':HASHED},'process_instance':b['process_instance']}),flush=True)
except BaseException:
    with DEST.with_suffix('.failure.json').open('xb') as f:f.write((json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
