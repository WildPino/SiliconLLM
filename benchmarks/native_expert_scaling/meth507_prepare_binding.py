"""Metadata-only price/freeze selected retained files and a NumPy/stdlib closure."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth507_binding.json'
assert not DEST.exists() and not DEST.with_suffix('.failure.json').exists()
start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([10]);catalog={};hashed=peak=0

def add(path, expected=None):
    global hashed,peak
    path=Path(path).resolve();key=str(path)
    if key not in catalog:
        before=path.stat();h=hashlib.sha256()
        with path.open('rb') as stream:
            while data:=stream.read(8<<20):
                h.update(data);hashed+=len(data)
                peak=max(peak,proc.memory_info().peak_wset)
                assert peak<=512<<20 and time.monotonic()-start<=180
        assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
        catalog[key]={'path':key,'bytes':before.st_size,'sha256':h.hexdigest()}
    if expected:assert catalog[key]['sha256']==expected,key
    return catalog[key]

try:
    raw=add(DOC/'meth506_r4_2_result.json','2bc2a9cfa9772fbc235f4c8e492e093be864784a023ed54cb711322f1c15d476')
    old=json.loads(Path(raw['path']).read_bytes());assert len(old['cases'])==96
    prior=add(DOC/'meth506_r4_2_binding.json','f9fa553e598070d68646694d52369faf91cadba584f9d4dd308f17db53ab8ab3')
    b=json.loads(Path(prior['path']).read_bytes())
    selected=[]
    for ordinal,c in enumerate(old['cases']):
        assert (c['book'],c['index'])==divmod(ordinal,4)
        ref=c['donor_reference'];wire=c['native']['candidate']['generation'][0]['wire']
        assert len({r['wire']['sha256'] for arm in c['native'].values() for r in arm['generation']+arm['profile']})==1
        for row in (ref,wire):assert add(row['path'],row['sha256'])['bytes']==row['bytes']
        selected.append({'book':c['book'],'index':c['index'],'donor':ref,'wire':wire})
    add(DOC/'EVALUATION_506_20261006.json','74c19ab57c628a15fe3b094ee3cc4744963daf9dfa5cbb159e4da2cb5f93ddd8')
    add(DOC/'RETENTION_506_R4_4_20261006.json','0ddd3582f7a56cd2d06594e8c924f207e46225c216512cf294d0540cf18c5524')
    scientific=[]
    paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth507*.py'))+[DOC/'METH_507_RETAINED_DIAGNOSIS_PROTOCOL_20261006.md']
    for path in paths:
        assert path.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+path.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
        scientific.append(add(path))
    assert len(scientific)==5
    for name in ('numpy','psutil'):
        assert metadata.version(name)==b['runtime']['packages'][name]['version']
        for row in b['runtime']['packages'][name]['files']:add(row['path'],row['sha256'])
    base=Path(sys.base_prefix)
    for path in list(base.glob('*.dll'))+list(base.glob('*.exe'))+list((base/'DLLs').iterdir()):
        if path.is_file():add(path)
    for path in (base/'Lib').rglob('*.py'):
        if 'site-packages' not in path.parts:add(path)
    for path in (Path(sys.prefix)/'Lib/site-packages').glob('*.pth'):add(path)
    add(sys.executable);add(Path(sys.prefix)/'pyvenv.cfg')
    for rel,sha in b['preserved'].items():add(ROOT/rel,sha)
    add(ROOT/'benchmarks/phase60/engine.c','d58e3c5fdbe789129c97e3d821b54eb3ffb53352de81a686f3fc186109823e93')
    value={'experiment':'METH507 retained-array first divergence','freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           'python':sys.version,'executable':str(Path(sys.executable).resolve()),'packages':{n:metadata.version(n) for n in ('numpy','psutil')},
           'raw506':raw,'prior_binding506':prior,'selected':selected,'scientific':scientific,'catalog':list(catalog.values()),
           'preserved':b['preserved'],'engine_sha256':'d58e3c5fdbe789129c97e3d821b54eb3ffb53352de81a686f3fc186109823e93',
           'price':{'selected_array_bytes':sum(v['donor']['bytes']+v['wire']['bytes'] for v in selected),'catalog_bytes':sum(v['bytes'] for v in catalog.values()),
                    'catalog_files':len(catalog),'maximum_one_wire_bytes':max(v['wire']['bytes'] for v in selected),
                    'maximum_one_npz_bytes':max(v['donor']['bytes'] for v in selected),
                    'main_audit_each_limits':{'seconds':180,'OS_peak_bytes':2<<30,'output_bytes':32<<20},
                    'new_native_model_compiler_calls':0,'new_cohorts':0},
           'preparation_resource':{'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed},
           'process_instance':{'pid':proc.pid,'create_time_unix':proc.create_time()}}
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf8')
    with DEST.open('xb') as stream:stream.write(payload)
    print(json.dumps({'binding':add(DEST),'price':value['price'],'resource':value['preparation_resource']}),flush=True)
except BaseException:
    payload=(json.dumps({'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,'OS_peak_bytes':peak,'bytes_hashed':hashed},indent=2)+'\n').encode()
    with DEST.with_suffix('.failure.json').open('xb') as stream:stream.write(payload)
    raise
