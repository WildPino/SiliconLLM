"""Exact audited integer DLL,including actual BF16 decode/Python input quantizer in call cost."""
import argparse,ctypes,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from integer_readout import quantize,interface
START=time.monotonic()

def prepare(a):
    import numpy as np,psutil
    rp=DOC/'integer_readout_result_20261013.json';ap=DOC/'integer_readout_adjudication_v2_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and r['numeric_all_pass'] and r['additional_all_pass'] and extent(r['binding']['path'])==r['binding'];pb=json.loads(Path(r['binding']['path']).read_bytes());ancestor=json.loads(Path(pb['parent_binding']['path']).read_bytes());oldp=DOC/'ternary_readout_cost_binding_20261013.json';old=json.loads(oldp.read_bytes());reviewp=DOC/'ternary_readout_cost_review_20261013.json';review=json.loads(reviewp.read_bytes());assert review['binding']==extent(oldp) and review['all360_observations_reconciled'];stp=DOC/'integer_readout_cost_storage_20261013.json';st=json.loads(stp.read_text(encoding='utf-8-sig'));assert not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(st['new_output_directory']).resolve();samples=[]
    for site in [0,12,23]:
        rr=[e for e in pb['records'] if e['site']==site and e['split']=='FIT']
        for e in [min(rr,key=lambda q:(q['history'],q['id'])),max(rr,key=lambda q:(q['history'],q['id']))]:
            for pos in [0,e['history']//2,e['history']-1]:samples.append(dict(site=site,id=e['id'],position=pos))
    assert samples==old['samples'];paths=[Path(__file__),B/'integer_readout.py',B/'integer_readout.c',B/'ternary_readout_native.py',B/'ternary_readout_precision.py',B/'falcon_auxiliary_s6_transport.py',DOC/'INTEGER_READOUT_COST_PROTOCOL_20261013.md',stp,rp,ap,Path(r['binding']['path']),Path(pb['parent_binding']['path']),oldp,reviewp,Path(pb['compile']['dll']['path']),Path(old['compile']['dll']['path']),Path(old['compile']['source']['path']),Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)];nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+[Path(e['file']['path']) for e in pb['packets']]+[Path(e['parameters']['path']) for e in ancestor['exports']]
    for s in samples:
        e=next(q for q in pb['records'] if q['site']==s['site'] and q['id']==s['id']);rr=next(q for q in r['records'] if q['site']==s['site'] and q['id']==s['id']);paths+=[Path(e['native']['path']),Path(rr['file']['path'])]
    paths+=[ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)];a.directory.mkdir();assert time.monotonic()-START<900 and psutil.Process().memory_info().peak_wset<6<<30
    b=dict(schema='INTEGER_READOUT_COST_BINDING_V1',freeze=a.freeze,python=pb['python'],versions=pb['versions'],parent_report=extent(rp),parent_adjudication=extent(ap),parent_binding=r['binding'],integer_compile=pb['compile'],baseline_compile=old['compile'],packets=pb['packets'],exports=ancestor['exports'],records=pb['records'],reference_records=r['records'],samples=samples,repetitions=5,modes=['warm_repeat','source_flushed'],methods=['integer','dense'],output_directory=str(a.directory.resolve()),inputs=inputs,limits=dict(seconds=900,OS_bytes=6<<30,output_bytes=16<<20),scope='Exact same full-audited integer DLL/quantizer and source q/scales;18 fixed prior FIT sample positions/5 paired reps/onecore/batch1. Include BF16 decode for both;integer includes literal Python quantize diagnostics and FFI/kernel,all measured integer/scaled/BF16 words exact to archives. Architectural source CLFLUSH not PMU bytes or model-rate/n claim.')
    write(a.out,b);print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)

def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert Path(sys.executable).resolve()==Path(b['python']).resolve() and np.__version__==b['versions']['numpy'] and psutil.__version__==b['versions']['psutil'] and not a.out.exists();proc=psutil.Process();proc.cpu_affinity([0]);stage='startup';rows=[];checks=[];receipt=dict(schema='INTEGER_READOUT_COST_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,whole_model_rate_admitted=False)
    def guard():assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    try:
        check();dll,fn=interface(dict(compile=b['integer_compile']));base=ctypes.CDLL(b['baseline_compile']['dll']['path']);dense=base.tc_dense;dense.argtypes=[ctypes.c_void_p]*4;dense.restype=ctypes.c_int;support=base.tc_support;support.argtypes=[ctypes.c_int];support.restype=ctypes.c_int;flush=base.tc_flush;flush.argtypes=[ctypes.c_void_p,ctypes.c_size_t];flush.restype=None;assert support(0)==1 and support(1)==64;freq=ctypes.c_longlong();assert ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq));clock=dict(QPC_frequency=freq.value,perf_counter_resolution=time.get_clock_info('perf_counter').resolution)
        for packet in b['packets']:
            site=packet['site'];guard()
            with np.load(packet['file']['path'],allow_pickle=False) as z:q=z['q'].copy();scale=z['scale'].copy()
            with np.load(next(e['parameters']['path'] for e in b['exports'] if e['site']==site),allow_pickle=False) as z:O=z['O'].copy()
            for sample in [s for s in b['samples'] if s['site']==site]:
                stage=f'{site}/{sample["id"]}/{sample["position"]}';guard();e=next(q for q in b['records'] if q['site']==site and q['id']==sample['id']);rr=next(q for q in b['reference_records'] if q['site']==site and q['id']==sample['id']);pos=sample['position']
                with np.load(e['native']['path'],allow_pickle=False) as z:input_bits=z['reconstructed_denominator__normalized'][pos].copy();ref_dense=z['reconstructed_denominator__raw_scores'][pos].copy();ref_dense_bits=z['reconstructed_denominator__output'][pos].copy()
                with np.load(rr['file']['path'],allow_pickle=False) as z:ref_r=z['input_I16'][pos].copy();ref_a=z['input_scale'][pos].copy();ref_dots=z['integer_dots'][pos].copy();ref_raw=z['scores'][pos].copy();ref_bits=z['output_BF16'][pos].copy()
                dots=np.empty((1,2048),'i8');raw=np.empty((1,2048),'f8');bits=np.empty((1,2048),'u2');yd=np.empty(2048,'f4');bd=np.empty(2048,'u2');current_r=None;current_a=None
                def call(method):
                    nonlocal current_r,current_a
                    decoded=(input_bits.astype('u4')<<16).view('f4')
                    if method=='integer':
                        current_r,current_a,_=quantize(decoded.astype('f8').reshape(1,3072),np);rc=fn(1,*[v.ctypes.data for v in [q,scale,current_r,current_a,dots,raw,bits]])
                    else:rc=dense(O.ctypes.data,decoded.ctypes.data,yd.ctypes.data,bd.ctypes.data)
                    assert rc==0
                def verify(method):
                    if method=='integer':assert np.array_equal(current_r[0],ref_r) and current_a[0]==ref_a and np.array_equal(dots[0],ref_dots) and np.array_equal(raw[0].view('u8'),ref_raw.view('u8')) and np.array_equal(bits[0],ref_bits)
                    else:assert np.array_equal(yd.view('u4'),ref_dense.view('u4')) and np.array_equal(bd,ref_dense_bits)
                for method in b['methods']:call(method);verify(method)
                checks.append(dict(**sample,all_input_integer_raw_and_BF16_bits_exact=True))
                for mode in b['modes']:
                    for rep in range(5):
                        order=b['methods'] if rep%2==0 else list(reversed(b['methods']))
                        for method in order:
                            guard();call(method);verify(method);evict=0
                            if mode=='source_flushed':
                                start=time.perf_counter_ns()
                                for arr in ([q,scale] if method=='integer' else [O]):flush(arr.ctypes.data,arr.nbytes)
                                evict=time.perf_counter_ns()-start
                            start=time.perf_counter_ns();call(method);elapsed=time.perf_counter_ns()-start;verify(method);rows.append(dict(**sample,mode=mode,repetition=rep,method=method,call_nanoseconds=elapsed,flush_nanoseconds=evict,source_bytes=q.nbytes+scale.nbytes if method=='integer' else O.nbytes))
        check();guard();summary={}
        for site in [0,12,23]:
            summary[str(site)]={}
            for mode in b['modes']:
                summary[str(site)][mode]={}
                for method in b['methods']:
                    v=[r['call_nanoseconds'] for r in rows if r['site']==site and r['mode']==mode and r['method']==method];summary[str(site)][mode][method]=dict(median_us=float(np.median(v)/1000),minimum_us=min(v)/1000,maximum_us=max(v)/1000)
        result=dict(schema='INTEGER_READOUT_COST_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,measurements=rows,summary=summary,sample_checks=checks,all_measured_outputs_bit_exact=True,CLFLUSH_supported=True,cache_line_bytes=64,clock=clock,before_after_hashes_exact=True,source_forwards=0,recurrence_calls=0,model_rate_admitted=False,decision='COMPLETE_INPUT_READY_DECODE_QUANTIZE_CALL_COST',seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),summary=summary)),flush=True)
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
        proc=psutil.Process();receipt=dict(schema='INTEGER_READOUT_COST_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,whole_model_rate_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as er:
            import traceback
            receipt.update(exit_code=1,error=repr(er),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
