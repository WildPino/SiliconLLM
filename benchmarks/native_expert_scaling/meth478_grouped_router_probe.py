"""ONE complete empirical grouped-router inquiry, with independent saved-source oracle."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import time
START = time.monotonic()
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import faulthandler
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess
import sys
import traceback
import psutil

ROOT = Path.cwd(); DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = ROOT/'results/native_expert_scaling/meth478_grouped_router_probe'
RAW = DOC/'meth478_grouped_router_result.json'
BIND = DOC/'meth478_prospective_bindings.json'
BIND_SHA = '07dc05c3e34df0570880b6c20315008614ffeb43061a109cbcc271a8aa6d6894'
PROTO = DOC/'METH_478_GROUPED_ROUTER_PROTOCOL_20261005.md'
C = ROOT/'benchmarks/native_expert_scaling/meth478_grouped_router.c'
MATH = ROOT/'benchmarks/native_expert_scaling/meth478_grouped_router_math.py'
WINDOWS = ROOT/'benchmarks/native_expert_scaling/meth478_windows_terminal.ps1'
parser = argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
assert args.out.resolve()==RAW.resolve() and not OUT.exists() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists()
OUT.mkdir(); fatal=(OUT/'fatal_native.log').open('xb'); faulthandler.enable(file=fatal,all_threads=True)
progress=(OUT/'progress.jsonl').open('x',encoding='utf8')
parent=psutil.Process(); parent.cpu_affinity([0]); phase='immutable_admission'; commands=[]; peak=hashed=0
process_instance={'pid':parent.pid,'create_time_unix':parent.create_time(),'executable':sys.executable}
start_utc=datetime.fromtimestamp(parent.create_time(),timezone.utc).isoformat()

def checkpoint(**values):
    progress.write(json.dumps({'phase':phase,'seconds':time.monotonic()-START,'pid':os.getpid(),**values})+'\n');progress.flush()
def guard(child=None):
    global peak
    info=parent.memory_info(); value=max(info.rss,getattr(info,'peak_wset',0))
    if child is not None:
        try:
            p=psutil.Process(child.pid); info=p.memory_info(); value+=max(info.rss,getattr(info,'peak_wset',0))
            for q in p.children(recursive=True):
                try: info=q.memory_info();value+=max(info.rss,getattr(info,'peak_wset',0))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        except psutil.NoSuchProcess:pass
    peak=max(peak,value)
    assert time.monotonic()-START<=300 and peak<=2<<30,(phase,peak)
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())+(RAW.stat().st_size if RAW.exists() else 0)<=96<<20
cache={}
def digest(path):
    global hashed
    path=Path(path).resolve();stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
    if key not in cache:
        h=hashlib.sha256()
        with path.open('rb') as f:
            while data:=f.read(4<<20):h.update(data);hashed+=len(data);guard()
        cache[key]=h.hexdigest()
    return cache[key]
def check(item):
    p=Path(item['path']);assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],str(p)
    if 'mtime_ns' in item:assert p.stat().st_mtime_ns==item['mtime_ns'],str(p)
def head(path):
    path=Path(path).resolve();rel=path.relative_to(ROOT).as_posix()
    assert path.read_bytes()==subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters','--path='+rel,'HEAD:'+rel]),rel
def write(path,record):
    with Path(path).open('xb') as f:f.write((json.dumps(record,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode())
def fail_hook(kind,value,tb):
    record={'experiment':'METH478 first grouped-router failure','phase':phase,'traceback':''.join(traceback.format_exception(kind,value,tb)),
            'start_utc':start_utc,'end_utc':datetime.now(timezone.utc).isoformat(),'process_instance':process_instance,'commands':commands,
            'binding_sha256':BIND_SHA,'scientific_sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),C,MATH,PROTO,WINDOWS)},
            'resource':{'seconds':time.monotonic()-START,'OS_peak_combined_bytes':peak,'file_bytes_hashed':hashed},
            'output_inventory':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.iterdir()) if p.is_file()]}
    if not RAW.with_suffix('.failure.json').exists():write(RAW.with_suffix('.failure.json'),record)
    checkpoint(terminal_failure=True);sys.__excepthook__(kind,value,tb)
sys.excepthook=fail_hook;checkpoint()
assert digest(BIND)==BIND_SHA;head(BIND);b=json.loads(BIND.read_bytes())
for p in (Path(__file__),C,MATH,PROTO,WINDOWS):head(p)
assert (ROOT/'.gitattributes').read_bytes()==subprocess.check_output(['git','show','HEAD:.gitattributes'])
assert sys.version==b['runtime']['python'] and Path(sys.executable).resolve()==Path(b['runtime']['executable']).resolve()
assert psutil.__version__==b['runtime']['packages']['psutil']['version'] and str(Path(psutil.__file__).resolve()) in b['runtime']['packages']['psutil']['files']
for path,item in b['runtime']['files'].items():check({'path':path,**item})
for package,item in b['runtime']['packages'].items():
    assert importlib.metadata.version(package)==item['version']
    for path,record in item['files'].items():check({'path':path,**record})
for key in ('records','helpers','scientific_files','source_inventory','compile_assets','system_files'):
    for item in b[key]:check(item)
check(b['compiler']);check(b['preparation_helper'])
check(b['metadata_correction']['helper']);assert digest(b['metadata_correction']['initial_binding_path'])==b['metadata_correction']['initial_binding_sha256']
for item in b['records']+b['helpers']+b['scientific_files']:head(item['path'])
for target in b['targets']:
    artifact=target['artifact'];assert digest(artifact['payload'])==artifact['sha256'] and digest(artifact['manifest'])==artifact['manifest_sha256']
for rel,item in b['preserved_unrelated_files'].items():assert digest(ROOT/rel)==item['sha256']
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).splitlines()==b['tracked_status']
own={parent.pid,*(p.pid for p in parent.parents())};daemons=[]
for p in psutil.process_iter(['name','cmdline']):
    if p.pid in own:continue
    try:
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():daemons.append(p.pid);continue
        assert not(name.startswith('python') or name.startswith('clang') or (name.startswith('meth') and name.endswith('.exe'))),(p.pid,name)
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
assert time.monotonic()-START<=120
gates={'immutable_complete393_source_outputs_both_artifacts_manifests_actual_runtime_assets_frozen':True}
checkpoint(file_bytes_hashed=hashed)

def ledger(path,magic,entries):
    with path.open('xb') as f:
        f.write(magic);f.write(struct.pack('<2I',len(entries),768 if magic==b'M478WGT1' else 96186))
        for item in entries:
            if magic==b'M478WGT1':f.write(struct.pack('<2IQ',item['n'],item['bank'],item['offset']));strings=[item['payload']]
            else:f.write(struct.pack('<6I',item['n'],item['book'],item['case'],item['mode'],item['s'],item['t']));strings=[item['trace'],item['whole']]
            for value in strings:data=value.encode();f.write(struct.pack('<I',len(data)));f.write(data)
weight_ledger=OUT/'weights.ledger.bin';jobs=OUT/'jobs.bin'
ledger(weight_ledger,b'M478WGT1',b['weights']);ledger(jobs,b'M478JOB1',b['tasks'])
def run(argv,label,expected=0):
    before=datetime.now(timezone.utc).isoformat();descendants={};maximum=0
    with (OUT/(label+'.stdout.log')).open('xb') as out,(OUT/(label+'.stderr.log')).open('xb') as err:
        child=subprocess.Popen([str(v) for v in argv],stdout=out,stderr=err)
        process=psutil.Process(child.pid);identity={'pid':child.pid,'create_time_unix':process.create_time(),'executable':str(argv[0])};process.cpu_affinity([0]);assert process.cpu_affinity()==[0]
        try:
            while child.poll() is None:
                guard(child)
                try:
                    info=process.memory_info();maximum=max(maximum,info.rss,getattr(info,'peak_wset',0))
                    for p in process.children(recursive=True):
                        try:
                            key=(p.pid,p.create_time());info=p.memory_info();descendants[key]={'pid':p.pid,'create_time_unix':p.create_time(),'executable':p.exe(),'OS_peak_bytes':max(info.rss,getattr(info,'peak_wset',0),descendants.get(key,{}).get('OS_peak_bytes',0))}
                        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                except psutil.NoSuchProcess:pass
                time.sleep(.025)
        except BaseException:
            if child.poll() is None:child.kill();child.wait()
            raise
    record={'label':label,'argv':[str(v) for v in argv],'returncode':child.returncode,'expected_returncode':expected,'process_instance':identity,
            'start_utc':before,'end_utc':datetime.now(timezone.utc).isoformat(),'OS_peak_bytes':maximum,'descendant_process_peaks':list(descendants.values())}
    commands.append(record);checkpoint(command_terminal=label,returncode=child.returncode)
    assert child.returncode==expected,(label,child.returncode,(OUT/(label+'.stderr.log')).read_bytes())
    return [json.loads(line) for line in (OUT/(label+'.stdout.log')).read_text().splitlines()]

phase='compile_and_numeric_controls'
binary=OUT/'meth478_grouped_router.exe'
run([b['compiler']['path'],'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off',C,'-o',binary],'compile')
controls=run([binary,'controls'],'controls');assert len(controls)==9
toys=((0,1,-1),(10000,9999,-10000),(0,0,0),(1000,999,-1000),(0,-1,-2),(0,0,-1))
with localcontext() as context:
    context.prec=100
    for k,scores in enumerate(toys):
        winner=max(range(3),key=lambda j:scores[j]);assert controls[k]['control']==k and controls[k]['chosen']==winner
        values=[struct.unpack('<f',struct.pack('<f',float((Decimal(v)-Decimal(scores[winner])).exp())))[0] for v in scores]
        total=sum(values);p=struct.unpack('<f',struct.pack('<f',1/total))[0]
        assert controls[k]['denominator']==total and struct.pack('<f',controls[k]['probability'])==struct.pack('<f',p)
def original_dot(a,x):
    left=[0.]*4;right=[0.]*4
    for i in range(0,768,8):
        for k in range(4):left[k]+=a[i+k]*x[i+k];right[k]+=a[i+k+4]*x[i+k+4]
    total=0.
    for k in range(4):total+=left[k]+right[k]
    return total
for k in range(2):
    a=[(i%7-3)*2.**-10 for i in range(768)] if not k else [2.**-100 if i%2 else 2.**80 for i in range(768)]
    x=[(i%5-2)*2.**-9 for i in range(768)] if not k else [2.**-20 if i%2 else -2.**-80 if i%4 else 2.**-80 for i in range(768)]
    actual=original_dot(a,x);assert struct.pack('<d',actual)==struct.pack('<d',controls[6+k]['double_result'])
    assert struct.pack('<f',actual)==struct.pack('<f',controls[6+k]['f32_result'])
assert controls[8]['CPU_affinity_mask']==1 and controls[8]['rounding']==0 and not controls[8]['MXCSR']&0x8040
bad=OUT/'negative_weight_ledger.bin';bad.write_bytes(b'INVALID!'+struct.pack('<2I',24,768))
run([binary,'source',bad,jobs],'negative_weight_ledger',2)
assert (OUT/'negative_weight_ledger.stderr.log').read_bytes()==b'grouped_router_error:weight_ledger_magic\r\n'
gates['independent_Decimal100digits6_ordered_double_dot2_FPU_affinity_and_negative_ledger_controls']=True
phase='ALL_source_score_and_probability_BYTE_admission'
source=run([binary,'source',weight_ledger,jobs],'source');assert len(source)==1 and source[0]['source_queries']==96186 and source[0]['source_score_values']==b['source_score_values'] and source[0]['scores_BYTE_exact'] and source[0]['probability_ID_BYTE_exact']
gates['ALL96186_original_scores_selected_ID_probability_BYTE_before_candidate_geometry']=True

phase='fixed_weight_only_tree_construction'
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)
assert np.__version__==b['runtime']['packages']['numpy']['version'] and threadpoolctl.__version__==b['runtime']['packages']['threadpoolctl']['version']
for module in (np,threadpoolctl):assert str(Path(module.__file__).resolve()) in b['runtime']['packages'][module.__name__]['files']
for pool in threadpoolctl.threadpool_info():assert pool['num_threads']==1 and str(Path(pool['filepath']).resolve()) in b['runtime']['packages']['numpy']['files']
np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
import meth478_grouped_router_math as M
assert Path(M.__file__).resolve()==MATH.resolve()
trees=[]
for weight in b['weights']:
    with Path(weight['payload']).open('rb') as f:f.seek(weight['offset']);data=f.read(weight['bytes'])
    assert hashlib.sha256(data).hexdigest()==weight['sha256']
    weights=np.frombuffer(data,'<f4').reshape(weight['n'],768)
    destination=OUT/f'n{weight["n"]}.bank{weight["bank"]}.tree.bin'
    trees.append({**M.build_tree(weights,weight['n'],weight['bank'],destination,guard),'path':str(destination),'bytes':destination.stat().st_size,'sha256':digest(destination)})
    checkpoint(tree_n=weight['n'],tree_bank=weight['bank'])
assert sum(v['bytes'] for v in trees)==b['expected_tree_bytes']
gates['ONE_fixed_weight_only_balanced_principal_tree_all24_banks_original_weight_roundtrip']=True
phase='complete_native_query_traversal_and_empirical_enclosures'
observations=OUT/'observations.bin';native=run([binary,'candidate',weight_ledger,jobs,OUT,observations],'candidate')
assert native[-1]['terminal'] and native[-1]['queries']==96186 and native[-1]['winner_exact'] and native[-1]['ALL_group_enclosures'] and native[-1]['ALL_probability_intervals'] and native[-1]['ALL_probability_relative_error_le1percent']
assert native[-1]['CPU_affinity_mask']==1 and not native[-1]['MXCSR']&0x8040
assert observations.stat().st_size==b['observation_bytes']
with observations.open('rb') as f:assert f.read(24)==struct.pack('<8sIIQ',b'M478OBS1',208,0,96186)
rows=np.memmap(observations,dtype=M.OBS,mode='r',offset=24,shape=(96186,))
assert np.array_equal(rows['meta'][:,0],np.arange(96186)) and np.isfinite(rows['value']).all()
gates['ALL96186_winner_probability_intervals_group_enclosures_heap_cut_and_wire_complete']=True
phase='complete_work_reporting'
summaries=M.reporting(rows);decisions=[]
for target in summaries['targets']:
    target['work_gates']={'mean_coefficients_le60percent':target['all']['coefficient_ratio_mean']<=.6,
                          'ALL24bank_mode_mean_le80percent':all(v['coefficient_ratio_mean']<=.8 for v in target['bank_modes']),
                          'BOTHmode_p95_le100percent':all(v['coefficient_ratio_p95']<=1 for v in target['modes'].values())}
    target['eligible_for_NEW_C_cost']=all(target['work_gates'].values());decisions.append(target['eligible_for_NEW_C_cost'])
gates['ALL_two_sources_48bank_modes_uniform_queries_vector_scalar_byte_work_reported']=True
for rel,item in b['preserved_unrelated_files'].items():assert digest(ROOT/rel)==item['sha256']
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).splitlines()==b['tracked_status']
assert not ({'torch','transformers','tensorflow','scipy','sklearn','pandas','meth393_switch_router_analysis','meth395_switch_router_integer_screen'}&set(sys.modules))
gates['resource_caps_no_model_capture_fit_query_selection_or_competing_science']=True
guard()
report={'experiment':'METH478 fixed grouped radius router full-witness feasibility','start_utc':start_utc,'end_utc':datetime.now(timezone.utc).isoformat(),
        'process_instance':process_instance,'source_binding_sha256':BIND_SHA,'scientific_sources':{str(p):digest(p) for p in (Path(__file__),C,MATH,PROTO,WINDOWS)},
        'commands':commands,'gates':gates,'trees':trees,'summaries':summaries,'source_oracle':source[0],'native_terminal':native[-1],
        'native_model_commands':0,'native_numeric_commands':4,'geometry_fits':'weight-only fixed tree; no source function/query/label fit',
        'decision':'eligible_target_only_for_NEW_C_cost_after_independent_retention' if any(decisions) else 'REJECT_THIS_GROUPED_RADIUS_TREE_VECTOR_WORK_pending_independent_retention',
        'scope':'Empirical complete consumed393query-witness enclosures; no universal libm/FP certificate, causal n curve, new artifact/quality/rate/physicalDRAM/LUT or other-family claim',
        'preserved_daemons':daemons,'resource':{'seconds_before_raw':time.monotonic()-START,'OS_peak_combined_bytes':peak,'file_bytes_hashed':hashed,'hard_seconds':300,'hard_peak_bytes':2<<30,'hard_new_bytes':96<<20},
        'output_inventory':[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='progress.jsonl']}
write(RAW,report);guard();checkpoint(terminal=True,raw_sha256=digest(RAW),OS_peak_combined_bytes=peak)
print(json.dumps({'raw':str(RAW),'sha256':digest(RAW),'gates':gates,'work':[{k:v for k,v in t.items() if k in ('n','all','work_gates','eligible_for_NEW_C_cost')} for t in summaries['targets']],
                  'seconds':time.monotonic()-START,'OS_peak_combined_bytes':peak}),flush=True)
progress.close();faulthandler.disable();fatal.close()
