"""Retained false BF16 smooth controls:actual scalar bits and midpoint coordinates."""
import argparse,ctypes,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
START=time.monotonic()


def main(a):
    import numpy as np,psutil
    proc=psutil.Process();proc.cpu_affinity(list(range(6)));receipt=dict(schema='BF16_ROUNDING_WITNESS_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze,timing_admitted=False)
    def dec(q):return (q.astype('<u4')<<16).view('<f4').astype('f8')
    def bf(q):u=np.asarray(q,dtype='<f4').view('<u4');return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2')
    def load(e):
        with np.load(e['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    try:
        assert not a.out.exists();rp=DOC/'recovered_input_temporal_result_20261013.json';ap=DOC/'recovered_input_temporal_adjudication_20261013.json';r=json.loads(rp.read_bytes());ad=json.loads(ap.read_bytes());assert ad['report']==extent(rp) and ad['decision']=='INDEPENDENT_STORED_NUMERIC_FAILURE';bad=[q for q in ad['numeric_flags'] if not q['passed']];assert bad and all(q['control']=='BF16_smooth_full_source_denominator' for q in bad);b=json.loads(Path(r['binding']['path']).read_bytes());packets={q['site']:q for q in b['exports']};records={(str(q['site'])+'/'+q['id']):q for q in r['records']};paths=[Path(__file__),B/'falcon_auxiliary_s6_transport.py',DOC/'BF16_ROUNDING_WITNESS_PROTOCOL_20261013.md',rp,ap,Path(r['binding']['path']),Path(b['reference_dll']['path'])];paths += [Path(records[q['case']]['file']['path']) for q in bad]+[Path(packets[records[q['case']]['site']]['parameters']['path']) for q in bad];runtime=[q for q in b['inputs'] if q['path'].startswith(str(Path(sys.executable).parent)) or 'numpy' in q['path'].lower() or 'psutil' in q['path'].lower()];inputs=list({q['path']:q for q in runtime+[extent(p) for p in paths]}.values())
        def check():
            for e in inputs:assert time.monotonic()-START<90 and proc.memory_info().peak_wset<2<<30 and extent(e['path'])==e,e['path']
        check();dll=ctypes.CDLL(b['reference_dll']['path']);fn=dll.scalar_output;fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];fn.restype=ctypes.c_int;rows=[]
        for q in bad:
            rec=records[q['case']];z=load(rec['file']);params=load(packets[rec['site']]['parameters']);norm=z['full_source_denominator__normalized'];score=z['full_source_denominator__raw_scores'];actual=z['full_source_denominator__output'];scalar=np.empty_like(score);assert fn(rec['history'],params['O'].ctypes.data,norm.ctypes.data,scalar.ctypes.data)==0 and np.array_equal(scalar.view('u4'),score.view('u4')) and np.array_equal(actual,bf(score));smooth64=dec(norm)@params['O'].astype('f8').T;smooth32=smooth64.astype('f4');expected=bf(smooth32);mask=actual!=expected;coords=np.argwhere(mask);err=float(np.linalg.norm(dec(actual)-dec(expected))/np.linalg.norm(dec(expected)));assert abs(err-q['relative'])<=1e-8;rawerr=float(np.linalg.norm(score.astype('f8')-smooth32.astype('f8'))/np.linalg.norm(smooth32));assert rawerr<=1e-4;points=[]
            for ti,oi in coords:
                av=float(dec(actual[ti,oi:oi+1])[0]);ev=float(dec(expected[ti,oi:oi+1])[0]);mid=(av+ev)/2;sv=float(score[ti,oi]);rv=float(smooth32[ti,oi]);adj=abs(int(actual[ti,oi])-int(expected[ti,oi]))==1;cross=min(sv,rv)<=mid<=max(sv,rv);points.append(dict(position=int(ti),output=int(oi),actual_score=sv,F64_reference=float(smooth64[ti,oi]),F64_cast_F32_reference=rv,actual_BF16_bits=int(actual[ti,oi]),reference_BF16_bits=int(expected[ti,oi]),actual_BF16=av,reference_BF16=ev,midpoint=mid,actual_minus_midpoint=sv-mid,reference_minus_midpoint=rv-mid,adjacent_bins=adj,midpoint_crossing=cross));assert adj and cross
            rows.append(dict(case=q['case'],history=rec['history'],score_words=score.size,score_bit_mismatches=0,BF16_from_actual_scores_bit_mismatches=0,false_flag=q,raw_smooth_relative=rawerr,BF16_smooth_relative=err,BF16_different_words=len(points),all_mismatches_adjacent_midpoint_crossings=True,coordinates=points))
        check();result=dict(schema='BF16_ROUNDING_WITNESS_V1',freeze=a.freeze,report=extent(rp),adjudication=extent(ap),inputs=inputs,records=rows,decision='EXACT_NATIVE_SCALAR_BITS_BF16_REFERENCE_MIDPOINT_CROSSINGS',original_false_numeric_flags_retained=bad,original_threshold_unchanged=1e-4,whole_operator_numeric_admission=False,no_native_history_reruns=True,no_source_forwards=True,seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out,result);assert a.out.stat().st_size<16<<20;receipt.update(exit_code=0,error=None,result=extent(a.out));print(json.dumps(dict(decision=result['decision'],different_words=[q['BF16_different_words'] for q in rows],seconds=result['seconds'])),flush=True)
    except BaseException as e:
        import traceback
        receipt.update(exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-START,OS_peak=proc.memory_info().peak_wset);write(a.out.with_suffix('.terminal.json'),receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--out',required=True,type=Path);main(p.parse_args())
