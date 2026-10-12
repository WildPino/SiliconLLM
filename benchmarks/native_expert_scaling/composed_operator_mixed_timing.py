"""Bound after audit: two batch1 ownnorm passes plus retained generator GEMV."""
import argparse,ctypes,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
from composed_operator_mixed import interface,ptrs
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    binding=extent(a.binding);assert binding['sha256']==a.binding_sha;b=json.loads(a.binding.read_bytes());adextent=extent(a.adjudication);assert adextent['sha256']==a.adjudication_sha;ad=json.loads(a.adjudication.read_bytes());r=json.loads(Path(ad['report']['path']).read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['binding']==binding and extent(ad['report']['path'])==ad['report'];assert not a.out.exists() and Path(sys.executable).resolve()==Path(b['python']).resolve();proc=psutil.Process();proc.cpu_affinity([0]);stage='startup';receipt=dict(schema='COMPOSED_OPERATOR_MIXED_TIMING_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,whole_model_timing_admitted=False);sources=[extent(Path(__file__)),extent(DOC/'COMPOSED_OPERATOR_MIXED_TIMING_PROTOCOL_20261013.md'),extent(B/'composed_operator_mixed.py'),extent(a.adjudication.with_suffix('.terminal.json'))];items=[];generator=[]
    def guard():assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+sources+[binding,adextent,ad['report']]+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def snapshot():
        result=[]
        for p in psutil.process_iter(['pid','name','cpu_times']):
            try:q=p.info;tm=q['cpu_times'];result.append(dict(pid=q['pid'],name=q['name'],cpu_seconds=float(tm.user+tm.system) if tm else None))
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        return result
    def raw(e):
        z=(np.fromfile(e['path'],'<u2').astype('<u4')<<16).view('<f4') if e['dtype']=='BF16' else np.fromfile(e['path'],'<f4');return np.ascontiguousarray(z.reshape(e['shape']),dtype='f4')
    try:
        check();dll,fn=interface(b,np);freq=ctypes.c_longlong();assert ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq));qpc_frequency=int(freq.value)
        with np.load(b['parameters']['path'],allow_pickle=False) as z:params={k:z[k].copy() for k in z.files}
        weights=[params[k] for k in ['Wc','Vc','Ws','Vs','Alog','D','gamma','O']];wp=ptrs(weights);ranks=np.array(b['ranks'],'i4');offs=np.array(b['offsets'],'i4');capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];cases={q['id']:q for q in capture};records={q['id']:q for q in r['records']};warm=min((q for q in capture if q['split']=='FIT'),key=lambda q:(q['history'],q['id']));warmups=0
        def call(rec):
            n=rec['history'];f=rec['sites'][23]['fields'];x,bb,cc,dt,gate=[raw(f[k]).reshape(n,-1) for k in ['x','B','C','delta','gate']];y=raw(f['y']).astype('f8');g=gate.astype('f8')*np.exp(-np.logaddexp(0.,-gate.astype('f8')));inv=np.ascontiguousarray(1/np.sqrt(np.mean((y*g)**2,axis=1)+1e-5));floats=[np.empty((n,c),'f4') for c in [384,96,96,384]]+[np.empty((384,96),'f4')]+[np.empty((n,2048),'f4') for _ in range(4)];bits=[np.empty((n,3072),'u2') for _ in range(4)]+[np.empty((n,2048),'u2') for _ in range(4)];times=np.zeros(8,'f8');ip=ptrs([x,bb,cc,dt,gate]);fp=ptrs(floats);qp=ptrs(bits);start=time.perf_counter();rc=fn(n,5,ranks.ctypes.data,offs.ctypes.data,wp,ip,inv.ctypes.data,fp,qp,times.ctypes.data,1);elapsed=time.perf_counter()-start;assert rc==0 and np.isfinite(floats[6]).all()
            with np.load(records[rec['id']]['file']['path'],allow_pickle=False) as z:
                assert np.array_equal(floats[6].view('u4'),z['mixed_decode64_out32__reconstructed_denominator__raw_scores'].view('u4')) and np.array_equal(bits[5],z['mixed_decode64_out32__reconstructed_denominator__output']) and np.array_equal(floats[4].view('u4'),z['final_state'].view('u4'))
            return dict(id=rec['id'],positions=n,native_call_seconds=elapsed,stage_seconds=dict(projection=float(times[0]),scan=float(times[1]),decode=float(times[2]),gate_RMS=float(times[3]),normalize_output=float(times[4])),every_raw_score_output_and_final_state_bit_exact=True)
        for _ in range(3):guard();call(warm);warmups+=1
        before=snapshot()
        for rep in range(2):
            for rec in capture:
                stage=f'operator/{rep}/{rec["id"]}';guard();q=call(rec);q['rep']=rep;items.append(q);print(json.dumps(dict(stage=stage,complete=len(items))),flush=True)
        gfn=dll.retained_generator;gfn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];gfn.restype=ctypes.c_int;n=128;xx=np.ldexp(((np.arange(n*2048,dtype='i4')%31)-15).reshape(n,2048).astype('f4'),-4);out=np.empty((n,6704),'f4')
        assert gfn(n,params['in_proj'].ctypes.data,xx.ctypes.data,out.ctypes.data)==0
        for rep in range(2):
            stage=f'generator/{rep}';guard();start=time.perf_counter();rc=gfn(n,params['in_proj'].ctypes.data,xx.ctypes.data,out.ctypes.data);elapsed=time.perf_counter()-start;assert rc==0 and np.isfinite(out).all();generator.append(dict(rep=rep,positions=n,seconds=elapsed,input_formula='ldexp((arange(128*2048)%31)-15,-4)',input_shape=[128,2048],output_shape=[128,6704],all_coefficients=13729792,checksum=float(out.astype('f8').sum())))
        generator_reference=xx.astype('f8')@params['in_proj'].astype('f8').T;generator_relative=float(np.linalg.norm(out.astype('f8')-generator_reference)/np.linalg.norm(generator_reference));assert generator_relative<=1e-5
        for q in generator:q.update(full_reference_relative=generator_relative,reference_threshold=1e-5)
        after=snapshot();check();guard();totals=[]
        for rep in range(2):
            rr=[q for q in items if q['rep']==rep];positions=sum(q['positions'] for q in rr);seconds=sum(q['native_call_seconds'] for q in rr);st={k:sum(q['stage_seconds'][k] for q in rr) for k in rr[0]['stage_seconds']};totals.append(dict(rep=rep,positions=positions,seconds=seconds,us_per_supplied_position=1e6*seconds/positions,stage_us_per_supplied_position={k:1e6*v/positions for k,v in st.items()},instrumentation_and_call_remainder_us=1e6*(seconds-sum(st.values()))/positions))
        result=dict(schema='COMPOSED_OPERATOR_MIXED_TIMING_V1',source=extent(Path(__file__)),freeze=a.freeze,binding=binding,adjudication=adextent,quality_report=ad['report'],affinity=[0],native_threads=1,QPC_frequency=qpc_frequency,python_perf_counter=vars(time.get_clock_info('perf_counter')),warmup_case=warm['id'],warmup_histories=warmups,reps=2,operator_records=items,operator_totals=totals,retained_generator=generator,source_generator_weight_bytes=params['in_proj'].nbytes,source_out_weight_bytes=params['O'].nbytes,processes_before=before,processes_after=after,all_timed_outputs_match_qualified_artifact=True,inputs_before_after_exact=True,clean_system_isolation_admitted=False,actual_DRAM_bytes_measured=False,whole_model_speed_admitted=False,seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='Actual1-thread batch1 ownnorm mixed_decode64_out32 supplied-source-operand composed operator,all48 histories/two passes/output bit checks. Stage instrumentation and retained generator GEMV on arbitrary vectors separately. Data loading,metrics/checks/hashes outside native call;source inputgen/conv/attention/MLP/head absent from operator time. No end-to-end useful token/s/actual expert DRAM/n admission.')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),totals=totals,generator=generator)),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),stage=stage,traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['binding','adjudication','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['binding-sha','adjudication-sha','freeze']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
