"""Metadata/code/runtime freeze; no payload bytes, CUDA loading or model execution."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';BASE=ROOT/'benchmarks/native_expert_scaling'
DEST=DOC/'meth485_binding.json';START=time.monotonic();assert not DEST.exists() and sys.flags.optimize==0

def item(path,expected=None,hash_now=True):
    p=Path(path).resolve();before=p.stat();v={'path':str(p),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns}
    if hash_now:
        h=hashlib.sha256()
        with p.open('rb') as f:
            while b:=f.read(4<<20):h.update(b);assert time.monotonic()-START<300
        assert (p.stat().st_size,p.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
        v['sha256']=h.hexdigest()
        if expected:assert v['sha256']==expected,str(p)
    else:assert expected;v['sha256']=expected;v['inherited_digest_to_refresh_before_controls']=True
    return v

def head(path):
    rel=Path(path).relative_to(ROOT).as_posix();raw=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
    assert Path(path).read_bytes().replace(b'\r\n',b'\n')==raw.replace(b'\r\n',b'\n'),rel

helpers=sorted(BASE.glob('meth485*'))+[ROOT/'benchmarks/phase60/engine.c',BASE/'meth388_switch_three_workers.c',BASE/'meth388_switch_three_workers_cost_entry.c',BASE/'meth388_switch_three_workers_entry.c',BASE/'meth388_switch_thread_binding.h',DOC/'METH_485_SHARED_INTEGER_BACKEND_PROTOCOL_20261006.md',DOC/'meth485_source_derivation.json',DOC/'meth485_preparation_fault_01.json',DOC/'meth485_preparation_fault_02.json',DOC/'METH_485_BINDING_PREPARATION_R1_20261006.md']
for p in helpers:assert p.is_file();head(p)
records=[]
for name,digest in [
 ('meth458_switch_matched_whole_cost_result.json','3762fd4e6c86a6ea7e64f54d4350cc2bd91761f2f856a06bb0a0a442f12d285f'),
 ('RETENTION_458_20261005.json','65ec6408bc1de79d0a3bad869b57de7298285901c5c91ca502330250ea9ea8b0'),
 ('meth387_switch_base128_multi_span_quality_result.json','539275a209c8b8a5680c388ec9130012157a681ed382ba1a4bf49ab8cccb9d4c'),
 ('meth363_switch_all_a16_multi_span_quality_result.json','ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'),
 ('meth382_switch_multi_span_manifest.json','96b3ffd09b0bdd42d7c990006e8744e39b5e1f32b7ae40aaf3b3dca98bf4db78'),
 ('meth362_switch_multi_span_manifest.json','c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'),
 ('meth484_shared_integer_eligibility_result.json','389deaba1ccd0315778388032c1f8e824fe6439d0a7ed3b8947e1eb0f3d001b4'),
 ('RETENTION_484_20261006.json','27d3e0c9f2ec9b7160c4bf7a63685f5aef7e1a8b5e60e8226605877055eaf6dc'),
 ('ADMISSION_484_20261006.json','09af8837c265f7f4dd2cb01f0bf71577bcf17d9cbde393ffc6f78eaca12c8a7b')]:
    head(DOC/name);records.append(item(DOC/name,digest))
prior=json.loads(Path(records[0]['path']).read_bytes())
sources=[];references=[];output_bound=64<<20
for n,num,label,manifestnum in [(128,387,'base128',382),(256,363,'all_a16',362)]:
    old=prior['sources'][str(n)];a=old['artifact'];payload=item(a['payload'],a['sha256'],False)
    assert [payload['bytes'],payload['mtime_ns']]==old['artifact_stat_before']
    qualpath=DOC/f'meth{num}_switch_{label}_multi_span_quality_result.json';quality=json.loads(qualpath.read_bytes())
    cohortpath=DOC/f'meth{manifestnum}_switch_multi_span_manifest.json';cohort=json.loads(cohortpath.read_bytes())
    assert all(quality['gates'].values()) and len(cohort['items'])==len(quality['books'])==24 and len(old['cases'])==96
    refs=[]
    for bi,book in enumerate(quality['books']):
        for ci,c in enumerate(book['cases']):
            qdir=ROOT/f'results/native_expert_scaling/meth{num}_switch_{label}_multi_span_quality';prefix=qdir/f'book{bi}.case{ci}'
            candidates=[x for x in quality['commands'] if len(x['argv'])==10 and x['argv'][4]==str(prefix)]
            assert len(candidates)==1
            teacher=item(str(prefix)+'.0.bin',c['native_output_sha256'],False)
            ref=item(str(prefix)+'.original_reference.npz',c['original_reference_sha256'],False)
            gen=item(str(prefix)+'.original_generation.npz',c['generation']['original_generation_sha256'],False)
            oldcase=old['cases'][4*bi+ci];nativepath=ROOT/f'results/native_expert_scaling/meth458_switch_matched_whole_cost/n{n}.book{bi}.case{ci}.profile0.0.bin'
            natural=item(nativepath,c['generation']['native_generation_sha256'],False)
            assert natural['sha256']==oldcase['modes']['0'][1]['output_sha256']
            refs.append({'book':bi,'case':ci,'teacher':teacher,'natural':natural,'donor_teacher':ref,'donor_generation':gen})
            references.extend([teacher,natural,ref,gen]);output_bound+=teacher['bytes']*2+natural['bytes']*16+(128<<10)
    sources.append({'n':n,'manifest':item(a['manifest'],a['manifest_sha256']),'payload':payload,'quality_path':str(qualpath),'cohort_path':str(cohortpath),
                    'threads':old['threads'],'affinity':old['affinity'],'references':refs,'totals':[old['accepted_cases'],old['accepted_generated_tokens'],old['accepted_prose_tokens']]})
assert output_bound<=12<<30,output_bound
compiler=Path(prior['sources']['128']['compile']['argv'][0]);libomp=Path(prior['sources']['128']['compile']['argv'][-1]).parent/'libomp.dll'
toolchain=[item(compiler,prior['sources']['128']['compile']['compiler_sha256']),item(libomp,prior['sources']['128']['compile']['runtime_sha256'])]
toolchain_support=set(compiler.parent.glob('*.dll'))
for name in ('ld.lld.exe','lld.exe'):
    support=compiler.parent/name
    if support.exists():toolchain_support.add(support)
toolchain.extend(item(p) for p in sorted(toolchain_support) if p.resolve()!=libomp.resolve())
cuda_dir=ROOT/'.venv/Lib/site-packages/torch/lib';cuda_paths=[cuda_dir/name for name in ('cudart64_12.dll','cublasLt64_12.dll','cublas64_12.dll')]+[Path('C:/Windows/System32/nvcuda.dll')]
# Pin the installed driver support modules before CUDA initialization.
cuda_paths+=list(Path('C:/Windows/System32').glob('nv*.dll'))
driverstore=Path('C:/Windows/System32/DriverStore/FileRepository')
for folder in driverstore.glob('nv_dispi.inf_amd64_*'):cuda_paths+=list(folder.rglob('*.dll'))
cuda=[item(p) for p in sorted(set(p.resolve() for p in cuda_paths))]
base=Path(sys.base_prefix);runtime_paths={Path(sys.executable),Path(sys._base_executable),ROOT/'.venv/pyvenv.cfg'}
runtime_paths.update(base.glob('*.dll'));runtime_paths.update((base/'DLLs').glob('*.pyd'));runtime_paths.update((base/'DLLs').glob('*.dll'))
runtime_paths.update(p for p in (base/'Lib').rglob('*.py') if 'site-packages' not in p.parts and '__pycache__' not in p.parts)
versions={}
for name in ('psutil','numpy'):
    dist=importlib.metadata.distribution(name);versions[name]=dist.version
    runtime_paths.update(Path(dist.locate_file(p)).resolve() for p in dist.files if str(p).endswith(('.py','.pyd','.dll','/METADATA','/RECORD')))
for name in ('kernel32.dll','KernelBase.dll','ntdll.dll','msvcrt.dll','ucrtbase.dll','psapi.dll'):runtime_paths.add(Path('C:/Windows/System32')/name)
record={'experiment':'METH485 complete prospective binding','freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'helpers':[item(p) for p in helpers],'records':records,'sources':sources,'toolchain':toolchain,'cuda':cuda,'cuda_dir':str(cuda_dir),
        'runtime':{'executable':str(Path(sys.executable).resolve()),'python':sys.version,'versions':versions,'files':[item(p) for p in sorted(runtime_paths)]},
        'runtime_environment':prior['sources']['128']['runtime_environment'],'complete_output_upper_bound_bytes':output_bound,'preparation_seconds':time.monotonic()-START,
        'no_payload_bytes_or_CUDA_loading':True}
with DEST.open('xb') as f:f.write((json.dumps(record,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'binding':str(DEST),'sha256':item(DEST)['sha256'],'output_upper_bound_bytes':output_bound,'runtime_files':len(runtime_paths),'driver_library_files':len(cuda),'seconds':record['preparation_seconds']}))
