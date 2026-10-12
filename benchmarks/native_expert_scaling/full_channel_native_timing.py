"""Separate after-audit two-pass batch1 timing of all three full-width operators."""
import argparse,ctypes,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path[:0]=[str(B),str(SITE)]
from falcon_auxiliary_s6_transport import extent,write
from full_channel_native import interface,ptrs,raw,buffers
START=time.monotonic();CLOCK=time.perf_counter()


def main(a):
    import numpy as np,psutil
    be=extent(a.binding);ae=extent(a.adjudication);assert be['sha256']==a.binding_sha and ae['sha256']==a.adjudication_sha;b=json.loads(a.binding.read_bytes());ad=json.loads(a.adjudication.read_bytes());assert ad['decision']=='INDEPENDENT_STORED_PASS' and ad['binding']==be and extent(ad['report']['path'])==ad['report'];r=json.loads(Path(ad['report']['path']).read_bytes());assert not a.out.exists() and Path(sys.executable).resolve()==Path(b['python']).resolve();proc=psutil.Process();proc.cpu_affinity([0]);stage='startup';rows=[];generators=[];receipt=dict(schema='FULL_CHANNEL_NATIVE_TIMING_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,whole_model_timing_admitted=False);sources=[extent(Path(__file__)),extent(B/'full_channel_native.py'),extent(DOC/'FULL_CHANNEL_NATIVE_TIMING_PROTOCOL_20261013.md'),extent(a.adjudication.with_suffix('.terminal.json'))]
    def guard():assert time.monotonic()-START<=900 and proc.memory_info().peak_wset<=6<<30 and not proc.children(recursive=True)
    def check():
        for e in b['inputs']+sources+[be,ae,ad['report']]+[q['file'] for q in r['records']]:guard();assert extent(e['path'])==e,e['path']
    def snapshot():
        result=[]
        for p in psutil.process_iter(['pid','name','cpu_times']):
            try:q=p.info;tm=q['cpu_times'];result.append(dict(pid=q['pid'],name=q['name'],cpu_seconds=float(tm.user+tm.system) if tm else None))
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        return result
    try:
        check();dll,fn=interface(b);freq=ctypes.c_longlong();assert ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq));capture=json.loads(Path(b['capture']['path']).read_bytes())['records'];prior={(q['site'],q['id']):q for q in r['records']};warm=min((q for q in capture if q['split']=='FIT'),key=lambda q:(q['history'],q['id']));before=snapshot();warmups=[]
        for packet in b['exports']:
            site=packet['site']
            with np.load(packet['parameters']['path'],allow_pickle=False) as z:params={k:z[k].copy() for k in z.files}
            w=ptrs([params[k] for k in ['Ws','Vs','Alog','D','gamma','O','in_proj']])
            def call(rec):
                n=rec['history'];f=rec['sites'][site]['fields'];inputs=[raw(f[k],np).reshape(n,-1) for k in ['x','B','C','delta','gate']];inv=np.ones(n,'f8');floats,bits=buffers(n,np);times=np.zeros(4,'f8');ip=ptrs(inputs);fp=ptrs(floats);qp=ptrs(bits);start=time.perf_counter();rc=fn(n,1,w,ip,inv.ctypes.data,fp,qp,times.ctypes.data,1);elapsed=time.perf_counter()-start;assert rc==0 and np.isfinite(floats[5]).all()
                with np.load(prior[(site,rec['id'])]['file']['path'],allow_pickle=False) as z:
                    assert np.array_equal(floats[5].view('u4'),z['reconstructed_denominator__raw_scores'].view('u4')) and np.array_equal(bits[3],z['reconstructed_denominator__output']) and np.array_equal(bits[1],z['reconstructed_denominator__normalized']) and np.array_equal(floats[3].view('u4'),z['final_state'].view('u4'))
                return dict(site=site,id=rec['id'],positions=n,native_call_seconds=elapsed,stage_seconds=dict(BC_projection=float(times[0]),scan=float(times[1]),gate_RMS=float(times[2]),normalize_output=float(times[3])),all_qualified_bits_exact=True)
            for _ in range(3):guard();call(warm)
            warmups.append(dict(site=site,id=warm['id'],histories=3))
            for rep in range(2):
                for rec in capture:
                    stage=f'operator/{site}/{rep}/{rec["id"]}';guard();q=call(rec);q['rep']=rep;rows.append(q);print(json.dumps(dict(stage=stage,complete=len(rows))),flush=True)
            gfn=dll.full_generator;gfn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];gfn.restype=ctypes.c_int;n=128;xx=np.ldexp(((np.arange(n*2048,dtype='i4')%31)-15).reshape(n,2048).astype('f4'),-4);out=np.empty((n,6704),'f4');assert gfn(n,params['in_proj'].ctypes.data,xx.ctypes.data,out.ctypes.data)==0
            for rep in range(2):
                stage=f'generator/{site}/{rep}';guard();start=time.perf_counter();rc=gfn(n,params['in_proj'].ctypes.data,xx.ctypes.data,out.ctypes.data);elapsed=time.perf_counter()-start;assert rc==0 and np.isfinite(out).all();generators.append(dict(site=site,rep=rep,positions=n,seconds=elapsed,checksum=float(out.astype('f8').sum())))
            expected=xx.astype('f8')@params['in_proj'].astype('f8').T;err=float(np.linalg.norm(out.astype('f8')-expected)/np.linalg.norm(expected));assert err<=1e-5
            for q in generators:
                if q['site']==site:q.update(final_deterministic_output_F64_reference_relative=err,reference_threshold=1e-5,input_formula='ldexp((arange(128*2048)%31)-15,-4)',source_weight_bytes=params['in_proj'].nbytes)
        after=snapshot();check();guard();totals=[]
        for site in b['sites']:
            for rep in range(2):
                rr=[q for q in rows if q['site']==site and q['rep']==rep];assert len(rr)==48;positions=sum(q['positions'] for q in rr);seconds=sum(q['native_call_seconds'] for q in rr);st={k:sum(q['stage_seconds'][k] for q in rr) for k in rr[0]['stage_seconds']};totals.append(dict(site=site,rep=rep,positions=positions,seconds=seconds,us_per_supplied_position=1e6*seconds/positions,stage_us_per_supplied_position={k:1e6*v/positions for k,v in st.items()},instrumentation_and_call_remainder_us=1e6*(seconds-sum(st.values()))/positions))
        result=dict(schema='FULL_CHANNEL_NATIVE_TIMING_V1',source=extent(Path(__file__)),freeze=a.freeze,binding=be,adjudication=ae,quality_report=ad['report'],native_threads=1,affinity=[0],QPC_frequency=int(freq.value),python_perf_counter=vars(time.get_clock_info('perf_counter')),warmups=warmups,operator_records=rows,operator_totals=totals,retained_generator=generators,processes_before=before,processes_after=after,all_timed_bits_match_qualified_artifact=True,inputs_before_after_exact=True,clean_system_isolation_admitted=False,actual_DRAM_measured=False,whole_model_speed_admitted=False,seconds=time.perf_counter()-CLOCK,OS_peak=proc.memory_info().peak_wset,scope='All3 site fixed full3072x96 supplied-operand native operators,two all48 batch1 passes/site,all normalized/rawscore/output/finalstate bits matched. Retained generator arbitrary-input GEMV separate. No original source hidden/conv/attention/MLP/head/whole useful tokens/s or n/DRAM admission.')
        write(a.out,result);receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(result=extent(a.out),totals=totals,generator=generators)),flush=True)
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
