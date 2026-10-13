"""State-ready single-token latency of SAME fully audited native branch."""
import argparse,ctypes,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from token_state import interface,arrays,arguments,ptrs
START=time.monotonic()

def prepare(a):
    import numpy as np,psutil
    rp=DOC/'token_state_result_20261013.json';ap=DOC/'token_state_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['report']==extent(rp) and r['numeric_all_pass'] and r['quality_all_pass'] and all(all(q.values()) for q in r['donor_gates'].values());pb=json.loads(Path(r['binding']['path']).read_bytes());op=DOC/'integer_readout_cost_binding_20261013.json';old=json.loads(op.read_bytes());samples=[q for q in old['samples'] if q['site']==0];assert len(samples)==6;sp=DOC/'token_state_cost_storage_20261013.json';st=json.loads(sp.read_text(encoding='utf-8-sig'));assert not a.out.exists() and not a.directory.exists() and a.directory.resolve()==Path(st['new_output_directory']).resolve()
    paths=[Path(__file__),B/'token_state.py',B/'token_state.c',B/'falcon_auxiliary_s6_transport.py',DOC/'TOKEN_STATE_COST_PROTOCOL_20261013.md',sp,rp,ap,Path(r['binding']['path']),op,Path(old['baseline_compile']['dll']['path']),Path(pb['compile']['dll']['path']),Path(pb['table']['path']),Path(pb['core_parameters']['path']),Path(pb['conv_parameters']['path']),Path(pb['readout_packet']['path']),Path(sys.executable),Path(sys.executable).parent/'python312.dll',Path(np.__file__),Path(np._core._multiarray_umath.__file__),Path(np.linalg._umath_linalg.__file__),Path(psutil.__file__)]
    nr=Path(np.__file__).resolve().parent;pr=Path(psutil.__file__).resolve().parent;paths+=list((nr.parent/'numpy.libs').glob('*.dll'))+list(pr.glob('*.pyd'))+[Path(next(q['file']['path'] for q in r['records'] if q['id']==s['id'])) for s in samples]+[ROOT/p for p in ['benchmarks/phase60/engine.c','benchmarks/donor_adaptation/configs/_manifest.json','benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md']];inputs=[extent(p) for p in dict.fromkeys(paths)];a.directory.mkdir();b=dict(schema='TOKEN_STATE_COST_BINDING_V1',freeze=a.freeze,python=pb['python'],versions=pb['versions'],parent_report=extent(rp),parent_adjudication=extent(ap),parent_binding=r['binding'],compile=pb['compile'],support_compile=old['baseline_compile'],table=pb['table'],core_parameters=pb['core_parameters'],conv_parameters=pb['conv_parameters'],readout_packet=pb['readout_packet'],records=pb['records'],reference_records=r['records'],samples=samples,repetitions=5,modes=['warm_ready','source_flushed'],inputs=inputs,output_directory=str(a.directory.resolve()),limits=dict(seconds=900,OS_bytes=6<<30,output_bytes=16<<20),scope='Same audited DLL/state-ready single-token actual-ID step. Prefix regeneration/context setup/pointer arrays outside clock;own row lookup/conv/state/norm/C quantizer/readout/debug outputs/FFI inside.6 fixed FIT positions/5 reps/2 source modes/core0/batch1,all outputs exact to archive. No whole model rate/tail bound/PMU DRAM/useful-n claim.');write(a.out,b);assert time.monotonic()-START<900 and psutil.Process().memory_info().peak_wset<6<<30;print(json.dumps(dict(binding=extent(a.out),seconds=time.monotonic()-START)),flush=True)

def run(a):
    import numpy as np,psutil
    assert extent(a.binding)['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());assert dict(numpy=np.__version__,psutil=psutil.__version__)==b['versions'] and Path(sys.executable).resolve()==Path(b['python']).resolve() and not a.out.exists();proc=psutil.Process();proc.cpu_affinity([0]);contexts=[];stage='startup';rows=[];receipt=dict(schema='TOKEN_STATE_COST_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,model_rate_admitted=False)
    def guard():assert time.monotonic()-START<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        for e in b['inputs']:guard();assert extent(e['path'])==e,e['path']
    try:
        check();dll,create,destroy,block,snap=interface(b);last=dll.ts_last;last.argtypes=[ctypes.c_void_p];last.restype=ctypes.c_uint32;table=np.memmap(b['table']['path'],dtype='u2',mode='r',shape=(65537,6704))
        with np.load(b['core_parameters']['path'],allow_pickle=False) as z:p={k:z[k].copy() for k in ['Ws','Vs','Alog','D','gamma']}
        with np.load(b['conv_parameters']['path'],allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
        with np.load(b['readout_packet']['path'],allow_pickle=False) as z:q=z['q'].copy();scale=z['scale'].copy()
        w=ptrs(list(p.values()));conv=ptrs([c[k] for k in ['cw','cb','db']]);helper=ctypes.CDLL(b['support_compile']['dll']['path']);support=helper.tc_support;support.argtypes=[ctypes.c_int];support.restype=ctypes.c_int;flush=helper.tc_flush;flush.argtypes=[ctypes.c_void_p,ctypes.c_size_t];flush.restype=None;assert support(0)==1 and support(1)==64
        freq=ctypes.c_longlong();assert ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq));clock=dict(QPC_frequency=freq.value,perf_counter_resolution=time.get_clock_info('perf_counter').resolution)
        for sample in b['samples']:
            stage=str(sample);guard();rec=next(q for q in b['records'] if q['id']==sample['id']);ref=next(q for q in b['reference_records'] if q['id']==sample['id']);ids=np.array(rec['input_ids'],'u4');pos=sample['position'];out=arrays(1,np);pre=arrays(max(1,pos),np);args=arguments(out,0,1,np);gold={}
            with np.load(ref['file']['path'],allow_pickle=False) as z:
                for k in out:gold[k]=z[k][pos:pos+1].copy()
            def context():
                ctx=create();assert ctx;contexts.append(ctx)
                if pos:assert block(ctx,pos,ids.ctypes.data,table.ctypes.data,w,conv,q.ctypes.data,scale.ctypes.data,*arguments(pre,0,pos,np))==0
                return ctx
            def verify(ctx):
                assert all(np.array_equal(out[k].view('u1'),gold[k].view('u1')) for k in out) and last(ctx)==ids[pos];state=np.empty((3072,96),'f4');ring=np.empty((3,3584),'u2');seen=np.zeros(1,'u8');assert snap(ctx,state.ctypes.data,ring.ctypes.data,seen.ctypes.data)==0 and seen[0]==pos+1;expected=np.zeros_like(ring)
                for ti in range(max(0,pos-2),pos+1):expected[ti%3]=table[ids[ti],3072:6656]
                assert np.array_equal(ring,expected)
            source_arrays=[table[ids[pos]],q,scale]+list(p.values())+[c[k] for k in ['cw','cb','db']];source_bytes=sum(v.nbytes for v in source_arrays);assert source_bytes==12965728
            for mode in b['modes']:
                for rep in range(5):
                    guard();warm=context();assert block(warm,1,ids[pos:pos+1].ctypes.data,table.ctypes.data,w,conv,q.ctypes.data,scale.ctypes.data,*args)==0;verify(warm);destroy(warm);contexts.remove(warm);ctx=context();evict=0
                    if mode=='source_flushed':
                        start=time.perf_counter_ns()
                        for v in source_arrays:flush(v.ctypes.data,v.nbytes)
                        evict=time.perf_counter_ns()-start
                    start=time.perf_counter_ns();rc=block(ctx,1,ids[pos:pos+1].ctypes.data,table.ctypes.data,w,conv,q.ctypes.data,scale.ctypes.data,*args);elapsed=time.perf_counter_ns()-start;assert rc==0;verify(ctx);destroy(ctx);contexts.remove(ctx);rows.append(dict(**sample,mode=mode,repetition=rep,token_ID=int(ids[pos]),call_nanoseconds=elapsed,flush_nanoseconds=evict,source_bytes=source_bytes,all_output_bits_exact=True,own_prefix_seen_before=pos))
        check();guard();summary={}
        for mode in b['modes']:
            vals=[r['call_nanoseconds'] for r in rows if r['mode']==mode];assert len(vals)==30;summary[mode]=dict(median_us=float(np.median(vals)/1000),minimum_us=min(vals)/1000,maximum_us=max(vals)/1000)
        result=dict(schema='TOKEN_STATE_COST_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,measurements=rows,summary=summary,clock=clock,CLFLUSH_supported=True,cache_line_bytes=64,source_bytes_per_call=12965728,all_measured_outputs_bit_exact=True,before_after_hashes_exact=True,source_model_forwards=0,GPU_calls=0,model_rate_admitted=False,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset,scope=b['scope']);write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),summary=summary)),flush=True)
    except BaseException as er:
        import traceback
        receipt.update(exit_code=1,error=repr(er),stage=stage,traceback=traceback.format_exc());raise
    finally:
        for ctx in contexts:destroy(ctx)
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    for k in ['binding','directory','out']:p.add_argument('--'+k,type=Path)
    for k in ['binding-sha','freeze']:p.add_argument('--'+k)
    a=p.parse_args()
    if a.run:run(a)
    else:
        import psutil
        proc=psutil.Process();receipt=dict(schema='TOKEN_STATE_COST_PREPARATION_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,model_rate_admitted=False)
        try:prepare(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
        except BaseException as er:
            import traceback
            receipt.update(exit_code=1,error=repr(er),traceback=traceback.format_exc());raise
        finally:receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.preparation_terminal.json'),receipt)
