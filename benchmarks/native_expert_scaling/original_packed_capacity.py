"""Convert existing E4M1 byte trits to a runtime-n packed-only original-engine ABI.

Large replication/full-V fixtures are explicitly structural, never useful capacity.
No teacher, tensor training runtime, optimizer or GPU is used.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))

MAGIC = b'E4BPv001'
FIELD = struct.Struct('<64s6I2Q')
FUNCTIONS = ['exp256_ps', 'exp_approx1', 'hsum256', 'dotf', 'matvec', 'silu',
             'softplus', 'reluf', 'acc_add_i8x32', 'matvec_lut_full', 'matvec_lut_rows',
             'build_lut_t3', 'bc_tm', 'ref_t3', 'quant_i8', 'state_reset', 'rmsnorm',
             'topk_sel', 'mlp_moe', 'forward_token']
OBSERVER = '    memcpy(actual_ids[l],idx,sizeof(idx)); memcpy(actual_mass[l],wv,sizeof(wv));\n'


def extent(p):
    return dict(path=str(Path(p).resolve()), bytes=Path(p).stat().st_size, sha256=sha(p))


def raw(p, data):
    with Path(p).open('xb') as f:
        f.write(data)


def legacy_fields(source):
    """Parse offsets, never load the legacy dequantized expert references."""
    with Path(source).open('rb') as f:
        h = struct.unpack('<16I', f.read(64))
    assert h[:11] == (0x45344D31, 1024, 256, 96, 8, 6, 512, 16, 4, 128, 5)
    assert h[11:] == (32, 128, 8, 1, 0)
    fields = []; cursor = 64
    def add(name, shape, dtype=1, keep=True):
        nonlocal cursor
        n = 1
        for d in shape: n *= d
        size = n * (4 if dtype == 1 else 1)
        if keep: fields.append(dict(name=name, shape=shape, dtype=dtype, offset=cursor, bytes=size))
        cursor += size
    add('embed', (1024, 256))
    for l in range(6):
        p = f'layers.{l}.'; add(p+'norm', (256,))
        if l == 5:
            add(p+'qkv', (768, 256)); add(p+'o', (256, 256))
        else:
            for name, shape in [('in_proj',(1024,256)),('conv_w',(512,4)),('conv_b',(512,)),
                ('x_proj',(208,512)),('dt_proj',(512,16)),('dt_b',(512,)),('A_log',(512,96)),
                ('Dskip',(512,)),('out_proj',(256,512))]: add(p+name, shape)
        add(p+'ff_norm', (256,)); add(p+'router', (32,256)); add(p+'router_bias', (32,))
        for name, shape in [('gate_ref',(32,128,256)),('up_ref',(32,128,256)),('down_ref',(32,256,128))]: add(p+name, shape, keep=False)
    add('final_norm', (256,)); add('head', (1024,256))
    for l in range(6):
        p = f'layers.{l}.'
        for name, shape, scale in [('gate',(32,128,256),(32,128)),('up',(32,128,256),(32,128)),('down',(32,256,128),(32,256))]:
            add(p+name+'_q', shape, 2); add(p+name+'_scale', scale)
    assert cursor == Path(source).stat().st_size
    return h, fields


def export(source, path, n=32, v=1024, structural=False):
    import numpy as np
    assert n >= 8 and n <= (2**31-32)//128 and v >= 1 and v <= 2**31-1
    assert (n == 32 and v == 1024) or structural
    # Replication is deliberately restricted to fixtures; no hidden donor transformation.
    assert n % 32 == 0 and (v == 1024 or v == 65537)
    h, entries = legacy_fields(source); table = []; specs = []; cursor = 80 + 110*104
    for e in entries:
        name=e['name'];shape=list(e['shape']);dtype=e['dtype']
        if name.endswith('_q'):
            shape=[n,64,256] if name.endswith('down_q') else [128,n*128];name=name[:-2]+'_code'
        elif name in ('embed','head'):shape=[v,256]
        elif name.endswith(('router','router_bias','gate_scale','up_scale','down_scale')):shape[0]=n
        count=1
        for d in shape:count*=d
        size=count*(4 if dtype==1 else 1)
        table.append(FIELD.pack(name.encode(),dtype,len(shape),*(shape+[0]*(4-len(shape))),cursor,size))
        specs.append((e,cursor,size));cursor+=size
    assert len(table)==110
    header=struct.pack('<16I',1,v,256,96,8,6,512,16,4,128,5,n,128,8,1,110)
    with Path(source).open('rb') as f, Path(path).open('xb') as dest:
        dest.write(MAGIC+header+struct.pack('<Q',cursor)+b''.join(table))
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for e,offset,size in specs:
                name = e['name']; shape = list(e['shape']); dtype = e['dtype']
                arr = np.ndarray(tuple(shape), dtype='<f4' if dtype == 1 else 'i1', buffer=mm, offset=e['offset'])
                if name.endswith('_q'):
                    assert np.all((arr >= -1) & (arr <= 1))
                    if name.endswith('down_q'):
                        code = ((arr[:,:,0::2].astype(np.int16)+1)*3 + arr[:,:,1::2]+1).astype('u1').transpose(0,2,1)
                        shape = [n,64,256]; data = np.tile(code, (n//32,1,1)).tobytes()
                    else:
                        code = ((arr[:,:,0::2].astype(np.int16)+1)*3 + arr[:,:,1::2]+1).astype('u1').reshape(4096,128).T
                        shape = [128,n*128]; data = np.tile(code, (1,n//32)).tobytes()
                elif name in ('embed','head') and v != 1024:
                    shape = [v,256]; data = arr[np.arange(v)%1024].tobytes()
                elif name.endswith(('router','router_bias','gate_scale','up_scale','down_scale')) and n != 32:
                    shape[0] = n; data = np.tile(arr, (n//32,) + (1,)*(arr.ndim-1)).tobytes()
                else:
                    data = arr.tobytes()
                assert len(data)==size and dest.tell()==offset;dest.write(data)
                del arr,data
                if name.endswith('_q'):del code
            assert dest.tell()==cursor
        finally:
            mm.close()
    result = extent(path); result.update(n=n,V=v,structural_fixture=structural,
        replication_factor=n//32,vocabulary_rows='original' if v == 1024 else 'row i copies original i mod1024',
        expert_reference_bytes=0,code_bytes=3*6*n*256*128//2)
    return result


def extract(directory):
    original = (ROOT/'benchmarks/phase60/engine.c').read_text()
    fragments = []; evidence = []
    for name in FUNCTIONS:
        m = re.search(r'^static[^\n]*\b'+name+r'\(', original, re.M); assert m,name
        pos = original.index('{',m.start()); depth=1; end=pos+1
        while depth:
            depth += (original[end]=='{') - (original[end]=='}'); end+=1
        original_body=original[m.start():end]; body=original_body
        if name=='mlp_moe':
            marker='    if(router_time)*router_time+=now_s()-router_start;'
            assert body.count(marker)==1; body=body.replace(marker,OBSERVER+marker)
            assert body.replace(OBSERVER,'') == original_body
            # Original scratch globals belong immediately before this function.
            fragments.append('static int8_t g_xq[D],g_lut[TUP*16],g_hq[HID_E],g_lutd[TDE*16]; static int32_t g_S[HID_E],g_Sd[D];')
        evidence.append(dict(name=name,original_body_sha256=hashlib.sha256(original_body.encode()).hexdigest(),
            compiled_body_sha256=hashlib.sha256(body.encode()).hexdigest(),observer_only=name=='mlp_moe'))
        fragments.append(body)
    p=directory/'original_packed_bodies.h';raw(p,('\n\n'.join(fragments)+'\n').encode())
    value=dict(source=extent(ROOT/'benchmarks/phase60/engine.c'),functions=evidence,header=extent(p));write(directory/'original_bodies.json',value)
    return value


def bind(a):
    import numpy as np
    import numpy._core._multiarray_umath as ext
    import psutil
    compiler=Path(json.loads((DOC/'chatbot_hybrid_engine_probe_binding_20261009.json').read_bytes())['compiler'])
    files=[Path(__file__),B/'original_packed_capacity.c',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
        ROOT/'benchmarks/phase60/engine.c',ROOT/'benchmarks/phase60/e4_export.py',ROOT/'results/phase60/e4_model.bin',
        ROOT/'results/phase57/moe_gran.pt',ROOT/'results/phase55/ids.u16',ROOT/'results/phase55/meta.bin',
        ROOT/'weights/bpe1024.bin',ROOT/'results/native_expert_scaling/engine_e32.exe',
        DOC/'NES_00_ASSET_AND_DISPATCH_20260925.md',DOC/'ORIGINAL_PACKED_CAPACITY_PROTOCOL_20261009.md',
        Path(sys.executable),Path(np.__file__),Path(ext.__file__),Path(psutil.__file__),compiler]
    files += [p for p in compiler.parent.iterdir() if p.name in ('clang-21.exe','ld.lld.exe','lld.exe','libclang-cpp.dll','libLLVM.dll')]
    files += [ROOT/p for p in ('benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md')]
    assert sha(ROOT/'results/phase60/e4_model.bin')=='52086303c95ddab3592ae8285a9420d93f7424c3a17dbcf234ac3769fe0347e1'
    assert sha(ROOT/'results/phase57/moe_gran.pt')=='356478b2f63ace9d6ec429056fee5c0e14e1c0f36253edbca36300eba02d4525'
    _,planned_fields=legacy_fields(ROOT/'results/phase60/e4_model.bin')
    def planned_bytes(n,v):
        total=80+110*104
        for e in planned_fields:
            if e['name'].endswith('_q'):size=e['bytes']*n//32//2
            elif e['name'] in ('embed','head'):size=v*256*4
            elif e['name'].endswith(('router','router_bias','gate_scale','up_scale','down_scale')):size=e['bytes']*n//32
            else:size=e['bytes']
            total+=size
        return total
    projected=dict(small_blob=planned_bytes(32,1024),large_blob=planned_bytes(2304,65537))
    projected['namespace_with_16MiB_misc_reserve']=sum(projected.values())+(16<<20)
    assert projected['namespace_with_16MiB_misc_reserve']<1<<30
    value=dict(schema='ORIGINAL_PACKED_CAPACITY_BINDING_V1',freeze=a.freeze,python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
        compiler=str(compiler),numpy_version=np.__version__,source=str((ROOT/'results/phase60/e4_model.bin').resolve()),
        legacy_executable=str((ROOT/'results/native_expert_scaling/engine_e32.exe').resolve()),
        limits=dict(seconds=120,reserve_seconds=15,OS_bytes=2<<30,output_bytes=1<<30),projected=projected,
        allowed_worker_children=['clang.exe','clang-21.exe','ld.lld.exe','lld.exe','packed_original.exe','engine_e32.exe','conhost.exe'],
        allowed_system_child_path=str((Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve()),
        runtime_binding_scope='Original source/export/trained artifact, actual legacy executable/data, adapter/extractor/runtime entry extents and selected compiler files. Not a complete DLL/dependency-tree hash.',
        criteria=dict(small_logits_bit_exact=True,all_integer_coordinates_exact=True,actual_mass_max_defect=1e-6,
            large_n=2304,large_V=65537,max_input_ID=65536,expert_reference_bytes=0,large_blob_above_512MiB=True,malformed_rejections=2),
        inputs=[extent(p) for p in dict.fromkeys(p.resolve() for p in files)])
    assert a.freeze and not a.out.exists();write(a.out,value);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(value['inputs']))),flush=True)


def worker(a):
    import numpy as np
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());assert b['freeze']==a.freeze and b['schema']=='ORIGINAL_PACKED_CAPACITY_BINDING_V1'
    assert np.__version__==b['numpy_version'];proc=psutil.Process();proc.cpu_affinity([0,2,4,6,8,10]);a.directory.mkdir(exist_ok=False)
    children=[];stage='start';max_child=0
    class Memory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];getmem.restype=wintypes.BOOL
    def guard():
        assert time.monotonic()-start < b['limits']['seconds']-b['limits']['reserve_seconds'],'worker reserve'
        assert proc.memory_info().peak_wset+max_child < b['limits']['OS_bytes'],'worker/direct-child OS cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) < b['limits']['output_bytes'],'namespace cap'
    def run(argv,name,expected=0):
        nonlocal max_child
        guard();log=a.directory/(name+'.log');t=time.monotonic();peak=0
        with log.open('xb') as f:
            p=subprocess.Popen([str(v) for v in argv],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,creationflags=0x08000000)
            def sample():
                nonlocal peak,max_child
                m=Memory();m.cb=ctypes.sizeof(m);assert getmem(wintypes.HANDLE(int(p._handle)),ctypes.byref(m),m.cb);peak=max(peak,m.PeakWorkingSetSize);max_child=max(max_child,peak)
            try:
                while p.poll() is None:sample();guard();time.sleep(.05)
                sample()
            except BaseException:
                if p.poll() is None:p.kill();p.wait()
                sample();raise
            finally:
                rec=dict(command=[str(v) for v in argv],pid=p.pid,exit_code=p.returncode,held_OS_peak=peak,seconds=time.monotonic()-t,log=extent(log));children.append(rec);write(a.directory/(name+'.receipt.json'),rec)
        assert p.returncode==expected,(name,p.returncode,log.read_text(errors='replace'));guard();return rec
    try:
        stage='extract';body=extract(a.directory)
        stage='export_small';small=export(b['source'],a.directory/'original_e32.packed');assert small['bytes']==b['projected']['small_blob'];write(a.directory/'small_export.json',small);guard();print(json.dumps(dict(stage=stage,bytes=small['bytes'])),flush=True)
        stage='export_large';large=export(b['source'],a.directory/'fixture_e2304_v65537.packed',2304,65537,True);assert large['bytes']==b['projected']['large_blob'];write(a.directory/'large_fixture.json',large);guard();print(json.dumps(dict(stage=stage,bytes=large['bytes'])),flush=True)
        stage='compile';exe=a.directory/'packed_original.exe';run([b['compiler'],'-O3','-mavx2','-mfma','-march=znver2',B/'original_packed_capacity.c','-I',a.directory,'-o',exe,'-lm'],'compile')
        ids=ROOT/'results/phase55/ids.u16';count=ids.stat().st_size//2;offset=int(count*.9)
        with ids.open('rb') as f:f.seek(offset*2);tokens=np.frombuffer(f.read(64*2),dtype='<u2').astype('<u4')
        assert tokens.size==64;raw(a.directory/'small_queries.u32',struct.pack('<I',64)+tokens.tobytes())
        raw(a.directory/'large_queries.u32',struct.pack('<I',3)+struct.pack('<3I',0,65535,65536))
        stage='legacy_new_storage_control';run([b['legacy_executable'],'--weights',b['source'],'--threads','1','--mlp','lut','--exp','fast','--pack','byte','--seq','64','--ntok','64','--offset','0','--dumplogits',a.directory/'legacy.f32'],'legacy')
        runs={}
        for name,model in [('small',small),('large',large)]:
            stage=name+'_packed_forward';run([exe,model['path'],a.directory/(name+'_queries.u32'),a.directory/(name+'.f32'),a.directory/(name+'.routes'),a.directory/(name+'.witness'),a.directory/(name+'.native.json')],name)
            runs[name]=json.loads((a.directory/(name+'.native.json')).read_bytes());assert runs[name]['expert_reference_bytes']==0
        assert sha(a.directory/'small.f32')==sha(a.directory/'legacy.f32'),'small logits not bit exact'
        assert large['bytes'] > 512<<20 and runs['large']['max_input_id']==65536 and runs['large']['V']==65537 and runs['large']['n']==2304
        stage='integer_and_mass_audit';_,entries=legacy_fields(b['source']);lookup={e['name']:e for e in entries};integer_checks=0;max_defect=0.
        with Path(b['source']).open('rb') as f:
            mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
            try:
                for name,n in [('small',32),('large',2304)]:
                    selected=list(dict.fromkeys(e for e in (0,31,32,1023,1024,n-1) if 0<=e<n));expected=[]
                    for e in selected:
                        for l in range(6):
                            for organ,length in [('gate',256),('up',256),('down',128)]:
                                entry=lookup[f'layers.{l}.{organ}_q'];arr=np.ndarray(entry['shape'],dtype='i1',buffer=mm,offset=entry['offset'])
                                q=np.arange(length,dtype=np.int64);q=(q%127-63) if organ!='down' else ((q*7)%127-63)
                                expected.append((arr[e%32].astype(np.int64)@q).astype('<i4'));del arr
                    expected=np.concatenate(expected);actual=np.fromfile(a.directory/(name+'.witness'),dtype='<i4');assert np.array_equal(actual,expected);integer_checks+=actual.size
                    buf=(a.directory/(name+'.routes')).read_bytes();assert len(buf)==runs[name]['inputs']*6*64
                    for pos in range(runs[name]['inputs']*6):
                        ids=np.frombuffer(buf,dtype='<u4',count=8,offset=pos*64);mass=np.frombuffer(buf,dtype='<f4',count=8,offset=pos*64+32)
                        assert len(set(ids.tolist()))==8 and np.max(ids)<n and np.all(np.isfinite(mass)) and np.all(mass>=0)
                        defect=abs(float(np.sum(mass,dtype=np.float64))-1);max_defect=max(max_defect,defect);assert defect<=1e-6
            finally:mm.close()
        stage='malformed_contracts';base=bytearray(Path(small['path']).read_bytes()[:80]);bad=bytearray(base);struct.pack_into('<Q',bad,72,2**64-1);raw(a.directory/'bad_extent.packed',bad)
        bad=bytearray(base);struct.pack_into('<I',bad,8+11*4,2**32-1);raw(a.directory/'bad_n.packed',bad)
        for kind in ('extent','n'):
            run([exe,a.directory/('bad_'+kind+'.packed'),a.directory/'small_queries.u32',a.directory/('bad_'+kind+'.f32'),a.directory/('bad_'+kind+'.routes'),a.directory/('bad_'+kind+'.witness'),a.directory/('bad_'+kind+'.native.json')],'reject_'+kind,2)
            assert not (a.directory/('bad_'+kind+'.f32')).exists()
        guard();stage='complete'
        value=dict(schema='ORIGINAL_PACKED_CAPACITY_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),decision='ORIGINAL_PACKED_CAPACITY_PASS',
            source=extent(b['source']),small=small,large_fixture=large,original_bodies=body,compiler=extent(exe),native=runs,children=children,
            small_logits_bit_exact=True,integer_coordinates=integer_checks,integer_all_exact=True,actual_mass_max_defect=max_defect,
            malformed_rejections=2,expert_reference_bytes=0,new_native_inputs=64+64+3,source_calls=0,GPU_calls=0,optimizer_updates=0,
            quality_admission=False,useful_large_n_admission=False,speed_admission=False,
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=max_child,
            resources_scope='Worker held by outer launcher; direct native/compiler child peaks held through exit here; nested linker memory is not separately held. No timing or throughput claim.',
            elapsed_seconds=time.monotonic()-start)
        write(a.out,value);print(json.dumps(dict(stage=stage,decision=value['decision'],integer_coordinates=integer_checks)),flush=True)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(stage=stage,error=repr(error),elapsed_seconds=time.monotonic()-start,children=children))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bind','worker'),default='worker');p.add_argument('--freeze');p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();bind(a) if a.mode=='bind' else worker(a)
