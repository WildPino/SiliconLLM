"""Post-492 source availability only: fresh byte hashes and NPY headers, no numeric vectors."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import time
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST=DOC/'meth493_weighted_target_metadata.json';FAIL=DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
START=time.monotonic();PROC=psutil.Process();PROC.cpu_affinity([0]);READS=[];PEAK=0;HASHED=0

def guard():
    global PEAK
    PEAK=max(PEAK,PROC.memory_info().peak_wset)
    assert PEAK<=128<<20 and time.monotonic()-START<=30

def descriptor(path,expected=None):
    global HASHED
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(1<<20):h.update(data);HASHED+=len(data);guard()
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    d={'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}
    if expected:assert d['sha256']==expected
    READS.append(d);return d

def save(path,value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode()
    assert len(data)<=256<<10
    with path.open('xb') as f:f.write(data)

try:
    admissionpath=DOC/'ADMISSION_492_20261006.json'
    descriptor(admissionpath,'368a74840f0f3c8d3e656c24cc73247ae4c0eb07e1806c6bc5abfd499f0e09c0')
    admission=json.loads(admissionpath.read_bytes());assert admission['main_completed'] and all(admission['gates'].values())
    oldpath=DOC/'meth472_switch_private_input_probe_result.json'
    descriptor(oldpath,'9fc2efb0acde8f354fffa81e8525dcd30940b07eb8f4bbf7fed49321193ae0b3')
    old=json.loads(oldpath.read_bytes());assert all(old['apparatus_gates'].values())
    retpath=DOC/'RETENTION_472_20261005.json';descriptor(retpath)
    ret=json.loads(retpath.read_bytes());assert ret['raw_sha256']==READS[1]['sha256'] and all(ret['gates'].values())
    inventory={Path(v['path']).name:v for v in old['output_inventory']};assets=[]
    for name,width,shape in (('query_inputs.npy',4732,(19962,)),('reference_functions.npy',3072,(19962,768))):
        path=ROOT/'results/native_expert_scaling/meth472_switch_private_input_probe'/name
        item=descriptor(path,inventory[name]['sha256']);assert item['bytes']==inventory[name]['bytes']
        with path.open('rb') as f:
            assert f.read(8)==b'\x93NUMPY\x01\x00';n=struct.unpack('<H',f.read(2))[0]
            header=ast.literal_eval(f.read(n).decode('ascii'));offset=f.tell()
        assert tuple(header['shape'])==shape and not header['fortran_order']
        assert item['bytes']==offset+19962*width
        if name=='reference_functions.npy':assert header['descr']=='<f4'
        assets.append({**item,'NPY_header':header,'data_offset':offset})
    descriptor(Path(__file__))
    guard()
    save(DEST,{'classification':'POST492_METADATA_SOURCE_AVAILABILITY_NOT_TARGET_COMPILATION_OR_NEW_SCIENCE',
        'assets':assets,'inputs':READS,'source_472_recipe_remains_failed':True,
        'scope':'Cached qualified source references exist and fresh digests match. No multiplication/UID join/feature training/native/quality/cost promotion.',
        'next':'Freeze a complete bank11 weighted-function target compiler and independent verifier before computing any target.',
        'resource':{'wall_seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED,
                    'wall_limit_seconds':30,'host_limit_bytes':128<<20}})
    print(json.dumps({'assets':[{k:v for k,v in a.items() if k!='NPY_header'} for a in assets],
                     'wall_seconds':time.monotonic()-START,'peak_bytes':PEAK,'hashed_bytes':HASHED}))
except BaseException as exc:
    save(FAIL,{'error':str(exc),'traceback':traceback.format_exc(),'inputs':READS,'seconds':time.monotonic()-START})
    raise
