"""Isolated same-parameter LUT/dense batch1 call cost,explicit source cache state."""
import argparse,ctypes,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()

def prepare(a):
    import numpy as np,psutil
    rp=DOC/'ternary_readout_native_result_20261013.json';ap=DOC/'ternary_readout_native_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and r['numeric_all_pass'] and r['additional_all_pass'];pb=json.loads(Path(r['binding']['path']).read_bytes());assert extent(r['binding']['path'])==r['binding'];stp=DOC/'ternary_readout_cost_storage_20261013.json';st=json.loads(stp.read_text(encoding='utf-8-sig'));assert not a.directory.exists() and not a.out.exists() and a.directory.resolve()==Path(st['new_output_directory']).resolve();compiler=Path(pb['compile'][0]['command'][0]);old=B/'ternary_readout_native.c';original=old.read_text();statement='dense[(size_t)t*2048+o]=dotf(O+(size_t)o*3072,x,3072);';assert original.count(statement)==1;body=original.replace('API int tr_history(','API int tr_lut_only(').replace(statement,'')
    tail='''\n#include <cpuid.h>\nAPI int tc_support(int i){unsigned a,b,c,d;if(!__get_cpuid(1,&a,&b,&c,&d))return 0;return i==0?((d>>19)&1):(int)(((b>>8)&255)*8);}\nAPI void tc_flush(const uint8_t*p,size_t n){for(size_t i=0;i<n;i+=64)_mm_clflush(p+i);if(n)_mm_clflush(p+n-1);_mm_mfence();}\nAPI int tc_dense(const float*O,const float*x,float*y,uint16_t*bits){for(int o=0;o<2048;o++){float f=dotf(O+(size_t)o*3072,x,3072);y[o]=f;bits[o]=bf(f);}return 0;}\n'''
    samples=[]
    for site in [0,12,23]:
        fit=[e for e in pb['records'] if e['site']==site and e['split']=='FIT'];small=min(fit,key=lambda e:(e['history'],e['id']));large=max(fit,key=lambda e:(e['history'],e['id']))
        for rec in [small,large]:
            for pos in [0,rec['history']//2,rec['history']-1]:samples.append(dict(site=site,id=rec['id'],position=pos))
    assert len(samples)==18
    paths=[Path(__file__),old,B/'falcon_auxiliary_s6_transport.py',DOC/'TERNARY_READOUT_COST_PROTOCOL_20261013.md',stp,rp,ap,Path(r['binding']['path']),compiler,Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)];nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+list(compiler.parent.glob('*.dll'))+list(compiler.parent.glob('*lld*.exe'))+[Path(e['file']['path']) for e in pb['packets']]+[Path(e['parameters']['path']) for e in pb['exports']]
    for sample in samples:
        e=next(q for q in pb['records'] if q['site']==sample['site'] and q['id']==sample['id']);rr=next(q for q in r['records'] if q['site']==sample['site'] and q['id']==sample['id']);paths+=[Path(e['native']['path']),Path(rr['file']['path'])]
    paths+=[ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)];a.directory.mkdir();src=a.directory/'timed_kernels.c';src.write_text(body+tail,encoding='utf8',newline='\n');dll=a.directory/'timed_kernels.dll';log=a.directory/'compile.log';cmd=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off','-shared',str(src.resolve()),'-o',str(dll.resolve()),'-lm'];co=subprocess.run(cmd,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(co.stdout+co.stderr);assert co.returncode==0;inputs+=[extent(src),extent(dll),extent(log)]
    for e in inputs:assert extent(e['path'])==e
    assert time.monotonic()-START<900 and psutil.Process().memory_info().peak_wset<6<<30 and not psutil.Process().children(recursive=True)
    b=dict(schema='TERNARY_READOUT_COST_BINDING_V1',freeze=a.freeze,python=pb['python'],versions=pb['versions'],parent_report=extent(rp),parent_adjudication=extent(ap),parent_binding=r['binding'],packets=pb['packets'],exports=pb['exports'],records=pb['records'],result_records=r['records'],samples=samples,repetitions=5,modes=['warm_repeat','source_flushed'],compile=dict(command=cmd,source=extent(src),dll=extent(dll),log=extent(log),exit_code=0),source_transform=dict(original=extent(old),renamed_export='tr_lut_only',deleted_statement=statement,remaining_LUT_arithmetic_changed=False),inputs=inputs,output_directory=str(a.directory.resolve()),limits=dict(seconds=900,OS_bytes=6<<30,output_bytes=16<<20),scope='18 fixed FIT source-derived positions/5 alternating paired repetitions/onecore/isolated original F32 and same-p6 LUT call times. Timed wrapper source deletes only original control statement;every measured output required bit-equal archived raw/BF16 scores. Sampled wrapper behavior,not full wrapper history audit. Explicit CLFLUSH initial source eviction,not PMU DRAM bytes,n scaling or model accepted-rate proof.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)

def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity([0]);stage='startup';rows=[];checks=[];receipt=dict(schema='TERNARY_READOUT_COST_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,whole_model_rate_admitted=False)
    def guard():assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    try:
        check();dll=ctypes.CDLL(b['compile']['dll']['path']);lut=dll.tr_lut_only;lut.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*7;lut.restype=ctypes.c_int;dense=dll.tc_dense;dense.argtypes=[ctypes.c_void_p]*4;dense.restype=ctypes.c_int;support=dll.tc_support;support.argtypes=[ctypes.c_int];support.restype=ctypes.c_int;flush=dll.tc_flush;flush.argtypes=[ctypes.c_void_p,ctypes.c_size_t];flush.restype=None;assert support(0)==1 and support(1)==64;freq=ctypes.c_longlong();assert ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq));clock=dict(QPC_frequency=freq.value,perf_counter_resolution=time.get_clock_info('perf_counter').resolution)
        for packet in b['packets']:
            site=packet['site'];guard()
            with np.load(packet['file']['path'],allow_pickle=False) as z:codes=z['codes'].copy();scale=z['scale'].copy()
            with np.load(next(e['parameters']['path'] for e in b['exports'] if e['site']==site),allow_pickle=False) as z:O=z['O'].copy()
            for sample in [s for s in b['samples'] if s['site']==site]:
                stage=f'{site}/{sample["id"]}/{sample["position"]}';guard();e=next(q for q in b['records'] if q['site']==site and q['id']==sample['id']);rr=next(q for q in b['result_records'] if q['site']==site and q['id']==sample['id']);pos=sample['position']
                with np.load(e['native']['path'],allow_pickle=False) as z:X=(z['reconstructed_denominator__normalized'][pos].astype('u4')<<16).view('f4').copy();reference_bits=z['reconstructed_denominator__output'][pos].copy()
                with np.load(rr['file']['path'],allow_pickle=False) as z:reference_lut=z['lut_scores'][pos].copy();reference_dense=z['original_scores'][pos].copy();reference_lut_bits=z['lut_BF16'][pos].copy()
                yl=np.empty(2048,'f8');yd=np.empty(2048,'f4');bl=np.empty(2048,'u2');bd=np.empty(2048,'u2')
                def call(method):
                    if method=='lut':rc=lut(1,codes.ctypes.data,scale.ctypes.data,None,X.ctypes.data,yl.ctypes.data,None,bl.ctypes.data)
                    else:rc=dense(O.ctypes.data,X.ctypes.data,yd.ctypes.data,bd.ctypes.data)
                    assert rc==0
                def verify(method):
                    if method=='lut':assert np.array_equal(yl.view('u8'),reference_lut.view('u8')) and np.array_equal(bl,reference_lut_bits)
                    else:assert np.array_equal(yd.view('u4'),reference_dense.view('u4')) and np.array_equal(bd,reference_bits)
                for method in ['lut','dense']:call(method);verify(method)
                checks.append(dict(**sample,all_raw_and_BF16_bits_exact=True))
                for mode in b['modes']:
                    for rep in range(b['repetitions']):
                        order=['lut','dense'] if rep%2==0 else ['dense','lut']
                        for method in order:
                            guard();call(method);verify(method);evict=0
                            if mode=='source_flushed':
                                t0=time.perf_counter_ns()
                                for arr in ([codes,scale] if method=='lut' else [O]):flush(arr.ctypes.data,arr.nbytes)
                                evict=time.perf_counter_ns()-t0
                            begin=time.perf_counter_ns();call(method);elapsed=time.perf_counter_ns()-begin;verify(method);rows.append(dict(**sample,mode=mode,repetition=rep,method=method,call_nanoseconds=elapsed,flush_nanoseconds=evict,source_bytes=(codes.nbytes+scale.nbytes) if method=='lut' else O.nbytes))
        check();guard();summary={}
        for site in [0,12,23]:
            summary[str(site)]={}
            for mode in b['modes']:
                summary[str(site)][mode]={}
                for method in ['lut','dense']:
                    v=[r['call_nanoseconds'] for r in rows if r['site']==site and r['mode']==mode and r['method']==method];summary[str(site)][mode][method]=dict(median_us=float(np.median(v)/1000),minimum_us=min(v)/1000,maximum_us=max(v)/1000)
        result=dict(schema='TERNARY_READOUT_COST_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,measurements=rows,summary=summary,sample_checks=checks,all_measured_outputs_bit_exact=True,CLFLUSH_supported=True,cache_line_bytes=64,clock=clock,before_after_hashes_exact=True,source_forwards=0,recurrence_calls=0,model_rate_admitted=False,decision='COMPLETE_INPUT_READY_CALL_COST',seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),summary=summary)),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='TERNARY_READOUT_COST_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,whole_model_rate_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as er:
            import traceback
            receipt.update(exit_code=1,error=repr(er),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
