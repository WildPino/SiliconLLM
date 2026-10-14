"""Stored replay and independent exact integer readout for every timed call."""
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


def main(a):
    import numpy as np
    import psutil
    from row_state import interface
    from token_state import arrays,arguments,ptrs
    r=json.loads(a.report.read_bytes());b=json.loads(Path(r['binding']['path']).read_bytes())
    assert extent(r['binding']['path']) == r['binding']
    assert json.loads(a.report.with_suffix('.terminal.json').read_bytes())['result'] == extent(a.report)
    assert not a.out.exists() and len(r['samples']) == 128 and len(r['records']) == 8
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([0]);contexts=[]
    def guard():
        assert time.monotonic()-start <= b['limits']['seconds']
        assert proc.memory_info().peak_wset <= b['limits']['OS_bytes'] and not proc.children(recursive=True)
    def check():
        pins=b['inputs']+[r['binding'],extent(a.report),extent(a.report.with_suffix('.terminal.json'))]
        pins += [x['file'] for q in r['records'] for x in q['files']]
        for e in pins:
            guard();assert extent(e['path']) == e,e['path']
    def bits(x):
        u=x.astype('f4').view('u4');hi=u>>16;lo=u&65535
        return (hi+((lo>32768)|((lo==32768)&((hi&1)!=0))).astype('u4')).astype('u2')
    def equal(x,y):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.array_equal(x.view('u1'),y.view('u1'))
    def stats(values):
        x=np.array(values,'f8')
        assert len(x) and np.isfinite(x).all() and np.all(x >= 0)
        return dict(count=len(x),minimum=float(x.min()),p10=float(np.quantile(x,.1)),
            median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(x.max()))
    check();dll,create,destroy,tokens,rows,snapshot=interface(b)
    last=dll.ts_last;last.argtypes=[ctypes.c_void_p];last.restype=ctypes.c_uint32
    table=np.memmap(b['table']['path'],'u2','r',shape=(65537,6704))
    checked=0;words=0;prefix_words=0;parameter_bytes=0;names=set()
    try:
        for site,export in enumerate(b['exports']):
            with np.load(export['file']['path'],allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
            assert export['site'] == site
            w=ptrs([p[k] for k in ('Ws','Vs','Alog','D','gamma')]);c=ptrs([p[k] for k in ('cw','cb','db')])
            q64=p['q'].astype('i8');parameter_bytes+=sum(v.nbytes for v in p.values())
            inv=r['parameter_inventory'][site];assert inv['site'] == site
            for entry in inv['arrays']:
                value=p[entry['name']]
                assert entry['bytes'] == value.nbytes and entry['shape'] == list(value.shape)
                assert entry['dtype'] == str(value.dtype) and entry['address'] > 0
                assert entry['address'] not in names;names.add(entry['address'])
            for rec in b['records']:
                tags=np.array(rec['sequence']['consumed_ids'],'u4');n=len(tags);lo=n-16
                assert rec['history'] == n and rec['positions'] == list(range(lo,n))
                item=rec['sequence']['sites'][site];width=2048 if site == 0 else 10800
                trace=np.memmap(item['trace']['path'],'u2','r',shape=(n,width))
                inp=np.ascontiguousarray(table[tags]) if site == 0 else np.ascontiguousarray(trace[:,2048:8752])
                expected=trace if site == 0 else trace[:,8752:]
                for mode in b['modes']:
                    guard();ctx=create();assert ctx;contexts.append(ctx)
                    for first in range(0,lo,64):
                        guard();hi=min(first+64,lo);out=arrays(hi-first,np)
                        assert rows(ctx,hi-first,tags[first:hi].ctypes.data,inp[first:hi].ctypes.data,
                            w,c,p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(out,0,hi-first,np)) == 0
                        equal(out['output'],np.ascontiguousarray(expected[first:hi]));prefix_words+=out['output'].size
                    old=next(x for x in r['records'] if x['id'] == rec['id'] and x['mode'] == mode)
                    assert old['prefix_positions'] == lo and old['prefix_ns'] > 0
                    saved_entry=old['files'][site];assert saved_entry['site'] == site
                    with np.load(saved_entry['file']['path'],allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
                    assert set(saved) == set(arrays(1,np))|{'state','ring','seen','last'}
                    out=arrays(16,np)
                    for t in range(16):
                        position=lo+t
                        assert rows(ctx,1,tags[position:position+1].ctypes.data,inp[position:position+1].ctypes.data,
                            w,c,p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(out,t,t+1,np)) == 0
                    for k,v in out.items():equal(v,saved[k])
                    equal(out['output'],np.ascontiguousarray(expected[lo:]))
                    st=np.empty((3072,96),'f4');rg=np.empty((3,3584),'u2');seen=np.zeros(1,'u8')
                    assert snapshot(ctx,st.ctypes.data,rg.ctypes.data,seen.ctypes.data) == 0
                    finals=dict(state=st,ring=rg,seen=seen,last=np.array([last(ctx)],'u4'))
                    with np.load(item['final']['path'],allow_pickle=False) as z:
                        for k,v in finals.items():equal(v,saved[k]);equal(v,z[k])
                    destroy(ctx);contexts.remove(ctx)
                    # Independent exact wide integer accumulation; no native readout call.
                    norm=(saved['norm'].astype('u4')<<16).view('f4').astype('f8')
                    a0=np.max(np.abs(norm),axis=1)/32767
                    equal(a0,saved['a'])
                    ratio=np.divide(norm,a0[:,None],out=np.zeros_like(norm),where=a0[:,None]!=0)
                    floor=np.floor(ratio);frac=ratio-floor
                    manual=floor+((frac>.5)|((frac==.5)&(floor.astype('i8')%2!=0)))
                    integer=np.clip(manual,-32767,32767).astype('i2');equal(integer,saved['r'])
                    dots=integer.astype('i8')@q64.T;equal(dots,saved['dots'])
                    raw=(dots.astype('f8')*a0[:,None])*p['scale'][None,:]
                    equal(raw,saved['raw']);equal(bits(raw),saved['output']);words+=dots.size;checked+=1
            print(json.dumps(dict(site=site,checked_site_records=checked,seconds=time.monotonic()-start)),flush=True)
        assert checked == 192 and words == 192*16*2048
        assert parameter_bytes == r['parameter_payload_bytes']
        assert r['state_payload_bytes'] == 24*3072*96*4 and r['ring_payload_bytes'] == 24*3*3584*2
        seen_samples=set()
        for s in r['samples']:
            assert len(s['site_ns']) == 24 and all(isinstance(v,int) and v>0 for v in s['site_ns'])
            assert isinstance(s['whole_ns'],int) and s['whole_ns'] >= sum(s['site_ns'])
            rec=next(q for q in b['records'] if q['id'] == s['id'])
            assert s['position'] in rec['positions'] and s['mode'] in b['modes']
            assert s['split'] == rec['split'] and s['length_kind'] == rec['length_kind']
            assert (s['flush_ns'] == 0 if s['mode'] == 'resident' else s['flush_ns'] > 0)
            key=(s['id'],s['mode'],s['position']);assert key not in seen_samples;seen_samples.add(key)
        assert len(seen_samples) == 128
        for mode in b['modes']:
            samples=[s for s in r['samples'] if s['mode'] == mode];old=r['summary'][mode]
            assert len(samples) == 64
            whole=stats([s['whole_ns'] for s in samples]);assert whole == old['whole_ns']
            assert [stats([s['site_ns'][i] for s in samples]) for i in range(24)] == old['per_site_ns']
            assert {rec['id']:stats([s['whole_ns'] for s in samples if s['id'] == rec['id']])
                    for rec in b['records']} == old['per_record_ns']
            assert stats([s['flush_ns'] for s in samples]) == old['flush_ns']
            assert old['remaining_20ms_ns'] == 20_000_000-whole['median']
            assert old['remaining_10ms_ns'] == 10_000_000-whole['median']
        assert len(r['noop_whole_ns']) == 128 and stats(r['noop_whole_ns']) == r['noop_summary']
        assert r['prefix_total_ns'] == sum(q['prefix_ns'] for q in r['records'])
        decision=('RECURRENT_MEDIAN_EXHAUSTS_20MS' if r['summary']['resident']['whole_ns']['median'] >= 20_000_000
                  else 'RECURRENT_MEDIAN_LEAVES_PARTIAL_BUDGET')
        assert r['decision'] == decision and r['GPU_calls'] == r['RESERVED_queries'] == 0
        assert r['affinity'] == [0] and r['kernel_threads'] == 1 and r['DRAM'] == b['DRAM']
        check();guard()
        write(a.out,dict(schema='NATIVE_24_COST_AUDIT_V1',report=extent(a.report),binding=r['binding'],
            checked_site_records=checked,checked_timed_calls=checked*16,checked_integer_words=words,
            checked_prefix_BF_words=prefix_words,all_stored_arrays_exact=True,
            independent_integer_quantization_scaling_BF_exact=True,timing_accounting_exact=True,
            decision='STORED_CONTINUATIONS_AND_ACCOUNTING_PASS',seconds=time.monotonic()-start,
            OS_peak=proc.memory_info().peak_wset,
            scope='Qualified recurrence DLL shared; independent exact readout arithmetic and stored timing accounting. '
                  'No independent wall-clock authentication or DRAM measurement. '+b['scope']))
        print(json.dumps(dict(audit=extent(a.out),checked=checked)),flush=True)
    finally:
        for ctx in contexts:destroy(ctx)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    import psutil
    proc=psutil.Process();start=time.monotonic()
    receipt=dict(schema='NATIVE_24_COST_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time())
    try:main(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)
