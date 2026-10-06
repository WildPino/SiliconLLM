"""Metadata-only R2 prospective Defender binding and unchanged scientific inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth506_r2_binding.json';assert not DEST.exists() and not DEST.with_suffix('.failure.json').exists()
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
    source=DOC/'meth506_r1_binding.json';head(source);old=json.loads(source.read_bytes());assert item(source)['sha256']=='a9ce87866544c0e8d855d819973f50f338baee05561242b463d11ac4498f7868'
    changed={str((ROOT/'benchmarks/native_expert_scaling'/name).resolve()) for name in ['meth506_operations.py','meth506_whole.py','meth506_retention_audit.py','meth506_finalize_admission.py']}
    catalog={}
    for v in old['catalog']:
        if v['path'] not in changed:catalog[v['path']]=item(v['path'],v['sha256'])
    print(json.dumps({'phase':'inherited_actual_catalog_terminal','seconds':time.monotonic()-START,'bytes_hashed':HASHED,'files':len(catalog)}),flush=True)
    def add(p):
        key=str(Path(p).resolve())
        if key not in catalog:catalog[key]=item(key)
        return catalog[key]
    defender=[]
    for p in sorted(Path('C:/ProgramData/Microsoft/Windows Defender/Platform/4.18.26080.4-0').glob('*.dll')):defender.append(add(p))
    assert any(Path(v['path']).name=='MpOAV.dll' for v in defender)
    paths=[Path(v['path']) for v in old['scientific']]+sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth506_r2*'))+[DOC/'METH_506_R2_PREPARATION_REPAIR_20261006.md']
    scientific=[]
    for p in sorted(set(paths)):head(p);scientific.append(add(p))
    records=old['records']+[add(source)]
    for name in ['meth506_preparation_result.failure.json','meth506_preparation_fault_tool_receipt.json']:
        p=DOC/name;head(p);records.append(add(p))
    failure=DOC/'meth506_preparation_result.failure.json';assert item(failure)['sha256']=='63a9cd4f6b7b6bcc366a2e9ad97bf96c78be89cf7a1c8ce9347079ae5a860697'
    frozen_partial=[]
    for p in sorted((ROOT/'results/native_expert_scaling/meth506_artifact').iterdir()):
        if p.is_file():frozen_partial.append(add(p))
    windows=ROOT/'results/native_expert_scaling/meth506_prepare_windows_terminal.json';v=json.loads(windows.read_bytes());assert v['query_available'] and not v['matching_scientific_events'];frozen_partial.append(add(windows))
    b={**old,'catalog':list(catalog.values()),'scientific':scientific,'records':records,'Defender_platform_modules':defender,'inherited_failed_preparation_files':frozen_partial,'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'process_instance':{'pid':PROC.pid,'create_time_unix':PROC.create_time()},'preparation_seconds':time.monotonic()-START,'preparation_peak_bytes':PEAK,'preparation_hashed_bytes':HASHED,'argv':sys.argv,
       'operational_repair':{'revision':'506-R2','previous_binding':item(source),'first_preparation_fault':item(failure),'new_preparation_directory':'meth506_r2_artifact','new_preparation_raw':'meth506_r2_preparation_result.json','original_preparation_numerical_imports':True,'original_preparation_cohort_export_compile_native_model_calls':0,'changed_scientific_kernel_cohort_or_quality_economic_gates':False,
                             'import_diagnostic':'One retained Windows first-chance access-violation trace during PyArrow import; Python continued to provenance assertion, exit1, actual Application1000 zero. Cause unidentified; no no-exception claim, no numerical result retroactively qualified.'}}
    guard()
    with DEST.open('xb') as f:f.write((json.dumps(b,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
    d=item(DEST);print(json.dumps({'binding':d,'resource':{'seconds':time.monotonic()-START,'peak_bytes':PEAK,'bytes_hashed':HASHED},'Defender_DLL_count':len(defender),'process_instance':b['process_instance']}),flush=True)
except BaseException:
    with DEST.with_suffix('.failure.json').open('xb') as f:f.write((json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED},indent=2)+'\n').encode())
    raise
