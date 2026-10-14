"""Every stored generator call: C replay, independent F64 and literal rounding."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),
             str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
import contextual_generator_cost as shared


def main(a):
    import numpy as np
    import psutil
    r=shared.load(a.report);b=shared.load(r['binding']['path'])
    receipt=shared.load(a.report.with_suffix('.terminal.json'))
    assert receipt['exit_code']==0 and receipt['result']==extent(a.report)
    assert extent(r['binding']['path'])==r['binding']
    proc=psutil.Process();proc.cpu_affinity([0]);start=time.monotonic()
    guard=shared.guard_factory(b,start,proc)
    pins=b['inputs']+[r['binding'],extent(a.report),extent(a.report.with_suffix('.terminal.json'))]
    pins += [q['file'] for q in r['records']]
    shared.check(pins,guard)
    assert len(r['records'])==8 and len(r['samples'])==128
    assert r['affinity']==[0] and r['kernel_threads']==1 and r['FP_control']==8064
    assert r['GPU_calls']==r['source_model_forwards']==r['optimizer_updates']==r['RESERVED_queries']==0
    assert r['DRAM']==b['DRAM'] and r['scope']==b['scope']
    def decode(x):return (x.astype('u4')*65536).view('f4')
    def rounding(x):
        bits=np.asarray(x,'f4').view('u4'); high=bits//65536;low=bits%65536
        increment=(low>32768)|((low==32768)&(high%2==1))
        return (high+increment).astype('u2')
    def equal(x,y):
        assert x.shape==y.shape and x.dtype==y.dtype
        assert np.array_equal(x.view('u1'),y.view('u1'))
    def relative(x,y):
        xx=x.astype('f8');yy=y.astype('f8')
        return float(np.sqrt(np.sum((xx-yy)**2))/max(float(np.sqrt(np.sum(yy**2))),1e-30))
    def fields_control(x,y):
        xx,yy=decode(x),decode(y)
        assert np.isfinite(xx).all() and np.isfinite(yy).all()
        return dict(overall=relative(xx,yy),**{k:relative(xx[:,lo:hi],yy[:,lo:hi]) for k,lo,hi in shared.SLICES})
    def statistics(values):
        x=np.array(values,'f8');assert x.size and np.isfinite(x).all() and np.all(x>=0)
        return dict(count=int(x.size),minimum=float(np.min(x)),p10=float(np.quantile(x,.1)),
            median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(np.max(x)))
    dll,support=shared.native(b);owners=shared.weights(b,np);mup=np.array(b['MuP_bits'],'u2')
    checked=0;words=0;max_raw=0.;max_source=0.;max_smooth=0.;addresses=set();controls=[]
    assert len(r['matrix_inventory'])==23 and r['parameter_payload_bytes']==631570432 and r['MuP_bytes']==13408
    for e,inv,w in zip(b['matrices'],r['matrix_inventory'],owners):
        assert {k:inv[k] for k in e}==e and inv['address']>0 and inv['address'] not in addresses
        addresses.add(inv['address']);assert w.nbytes==e['bytes']
    assert r['MuP_address']>0 and r['MuP_address'] not in addresses
    for rec in b['records']:
        guard();h,truth=shared.record_inputs(rec,np)
        stored=[]
        for mode in b['modes']:
            q=next(q for q in r['records'] if q['id']==rec['id'] and q['mode']==mode)
            with np.load(q['file']['path'],allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
            assert set(saved)=={'raw','gg','fields'}
            for k in saved:assert saved[k].shape==(23,16,6704) and saved[k].dtype==np.dtype('f4' if k=='raw' else 'u2')
            stored.append((mode,q,saved))
        for i,w in enumerate(owners):
            guard();matrix=decode(w).astype('f8');assert np.isfinite(matrix).all()
            ref=decode(h[i]).astype('f8')@matrix.T;del matrix
            assert np.isfinite(ref).all()
            ref_fields=rounding(decode(rounding(ref))*decode(mup))
            replay=dict(raw=np.empty((16,6704),'f4'),gg=np.empty((16,6704),'u2'),fields=np.empty((16,6704),'u2'))
            for mode,q,saved in stored:
                guard();raw=saved['raw'][i];gg=saved['gg'][i];fields=saved['fields'][i]
                assert np.isfinite(raw).all()
                error=relative(raw,ref);max_raw=max(max_raw,error);assert error<=b['gates']['raw_relative']
                equal(rounding(raw),gg);equal(rounding(decode(gg)*decode(mup)),fields)
                source=fields_control(fields,truth[i]);smooth=fields_control(fields,ref_fields)
                for key in source:assert abs(source[key]-q['source_controls'][i][key])<=1e-12
                max_source=max(max_source,max(source.values()));max_smooth=max(max_smooth,max(smooth.values()))
                assert max(source.values())<=b['gates']['field_relative']
                assert max(smooth.values())<=b['gates']['field_relative']
                for t in range(16):
                    assert dll.cg_step(w.ctypes.data,h[i,t].ctypes.data,mup.ctypes.data,
                        replay['raw'][t].ctypes.data,replay['gg'][t].ctypes.data,replay['fields'][t].ctypes.data)==0
                for k in replay:equal(replay[k],saved[k][i])
                checked+=16;words+=fields.size
                controls.append(dict(id=rec['id'],mode=mode,site=i+1,raw_relative=error,source=source,smooth=smooth))
            print(json.dumps(dict(id=rec['id'],site=i+1,checked_calls=checked,seconds=time.monotonic()-start)),flush=True)
    assert checked==2944 and words==19736576
    unique=set()
    for s in r['samples']:
        rec=next(q for q in b['records'] if q['id']==s['id'])
        assert s['split']==rec['split'] and s['length_kind']==rec['length_kind']
        assert s['position'] in rec['positions'] and s['mode'] in b['modes']
        assert len(s['site_ns'])==23 and all(isinstance(v,int) and v>0 for v in s['site_ns'])
        assert isinstance(s['whole_ns'],int) and s['whole_ns']>=sum(s['site_ns'])
        assert s['flush_ns']==0 if s['mode']=='resident' else s['flush_ns']>0
        key=(s['mode'],s['id'],s['position']);assert key not in unique;unique.add(key)
    assert len(unique)==128
    exceed=[]
    for mode in b['modes']:
        s=[q for q in r['samples'] if q['mode']==mode];old=r['summary'][mode];assert len(s)==64
        assert statistics([q['whole_ns'] for q in s])==old['whole_ns']
        assert [statistics([q['site_ns'][i] for q in s]) for i in range(23)]==old['per_site_ns']
        assert {q['id']:statistics([v['whole_ns'] for v in s if v['id']==q['id']]) for q in b['records']}==old['per_record_ns']
        assert statistics([q['flush_ns'] for q in s])==old['flush_ns']
        excess=old['whole_ns']['median']>b['recurrent_room_ns'][mode]
        assert old['exceeds_recurrent_room']==excess;exceed.append(excess)
    assert len(r['noop_whole_ns'])==128 and statistics(r['noop_whole_ns'])==r['noop_summary']
    decision='DENSE_CONTEXTUAL_GENERATORS_EXCEED_REMAINING_BUDGET' if any(exceed) else 'GENERATOR_ONLY_WITHIN_RECURRENT_ROOM'
    assert decision==r['decision']
    assert r['last_record_buffers']==dict(input_h_bytes=23*16*2048*2,source_truth_bytes=23*16*6704*2,
        raw_bytes=23*16*6704*4,gg_bytes=23*16*6704*2,field_bytes=23*16*6704*2)
    assert r['operand_ledger']==dict(weight_bytes_per_group=23*6704*2048*2,input_h_bytes_per_group=23*2048*2,
        decoded_input_stack_bytes_per_call=2048*4,FMA_per_group=23*6704*2048,
        decoded_input_logical_read_bytes_per_group=23*6704*2048*4,MuP_logical_read_bytes_per_group=23*6704*2,
        output_bytes_per_group=23*6704*8,
        note='BF coefficient storage, F32 register decode/input stack; logical operands are not DRAM')
    assert r['output_bytes']==sum(q['file']['bytes'] for q in r['records'])
    shared.check(pins,guard);guard()
    write(a.out,dict(schema='CONTEXTUAL_GENERATOR_COST_AUDIT_V1',report=extent(a.report),binding=r['binding'],
        checked_calls=checked,checked_records=8,checked_field_words=words,checked_raw_words=words,checked_GG_words=words,
        independent_F64_products=True,manual_BF_MuP_exact=True,C_replay_every_stored_word_exact=True,
        timing_accounting_exact=True,max_raw_relative=max_raw,max_source_slice_relative=max_source,
        max_smooth_slice_relative=max_smooth,controls=controls,decision='INDEPENDENT_STORED_NUMERIC_AND_ACCOUNTING_PASS',
        budget_decision=decision,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
        independence='Independent F64 products/RNE/statistics; shared mmap/input extraction/native replay/hash helpers. '
                     'No independent wall-clock authentication or DRAM counters. '+b['scope']))
    print(json.dumps(dict(audit=extent(a.out),max_raw=max_raw,max_source=max_source,max_smooth=max_smooth)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    import psutil
    proc=psutil.Process();proc.cpu_affinity([0]);start=time.monotonic()
    receipt=dict(schema='CONTEXTUAL_GENERATOR_COST_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time())
    try:
        assert not a.out.exists();main(a)
        assert time.monotonic()-start<=1800 and proc.memory_info().peak_wset<=4<<30
        receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)
