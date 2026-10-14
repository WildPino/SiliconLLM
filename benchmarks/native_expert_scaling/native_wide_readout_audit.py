"""Complete candidate replay, independent dot/scales/BF and paired accounting."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'benchmarks/native_expert_scaling'),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from native_wide_readout import load,setup,guard,equal,replay_bridge
from native_24_cost import check_pins,parameter_sets,record_rows,snap
from token_state import arrays,arguments


def main(a):
    import ctypes,numpy as np,psutil
    r=load(a.report);b=load(r['binding']['path']);assert extent(r['binding']['path'])==r['binding']
    terminal=load(a.report.with_suffix('.terminal.json'));assert terminal['exit_code']==0 and terminal['result']==extent(a.report)
    assert not a.out.exists() and len(r['samples'])==128 and len(r['records'])==8
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity([0]);check=lambda:guard(b,start,proc)
    def check_all():
        check_pins(b,check)
        for k in ('adversarial','BF_ties'):assert extent(r[k]['path'])==r[k]
    def BF(x):
        u=x.astype('f4').view('u4');hi=u>>16;lo=u&65535
        return (hi+((lo>32768)|((lo==32768)&((hi&1)!=0))).astype('u4')).astype('u2')
    def stats(values):
        x=np.array(values,'f8');assert len(x) and np.isfinite(x).all() and np.all(x>=0)
        return dict(count=len(x),minimum=float(x.min()),p10=float(np.quantile(x,.1)),median=float(np.median(x)),
                    p90=float(np.quantile(x,.9)),maximum=float(x.max()))
    check_all();apis=setup(b);api,last,read=apis[1];parameters=parameter_sets(b,np)
    with np.load(r['adversarial']['path'],allow_pickle=False) as z:wa,xa,da=[z[k].copy() for k in ('weight','input','dots')]
    assert wa.shape==xa.shape==(64,3072) and wa.dtype==xa.dtype==np.dtype('i2')
    assert np.max(np.abs(wa.astype('i4')))<=364 and np.max(np.abs(xa.astype('i4')))<=32767
    truth=np.array([sum(int(w)*int(x) for w,x in zip(ww,xx)) for ww,xx in zip(wa,xa)],'i8');equal(truth,da)
    control=api[0].ir_dot_controls;control.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*3;control.restype=ctypes.c_int
    replay=np.empty(64,'i8');assert control(64,wa.ctypes.data,xa.ctypes.data,replay.ctypes.data)==0;equal(replay,truth)
    with np.load(r['BF_ties']['path'],allow_pickle=False) as z:
        assert np.all(z['dots']==1);equal(z['raw'][0],z['scale']);equal(BF(z['raw']),z['bits'])
        assert np.array_equal(z['scale'],np.resize(np.array([0.,1.00390625,1.01171875,-1.00390625],'f8'),2048))
    checked=0;words=0
    for site,(p,w,c) in enumerate(parameters):
        assert np.max(np.abs(p['q'].astype('i4')))<=364
        q64=p['q'].astype('i8')
        for rec in b['records']:
            tags,inp,expected=record_rows(b,rec,np);lo=rec['history']-16
            for mode in b['modes']:
                check();record=next(x for x in r['records'] if x['id']==rec['id'] and x['mode']==mode)
                reference=next(x for x in b['reference_records'] if x['id']==rec['id'] and x['mode']==mode)
                assert record['all_arrays_exact'] and record['reference']==reference['files']
                with np.load(reference['files'][site]['file']['path'],allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
                ctx=api[1]();assert ctx
                try:
                    for first in range(0,lo,64):
                        check();hi=min(first+64,lo);out=arrays(hi-first,np)
                        assert api[4](ctx,hi-first,tags[first:hi].ctypes.data,inp[site][first:hi].ctypes.data,
                            w,c,p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(out,0,hi-first,np))==0
                        equal(out['output'],np.ascontiguousarray(expected[site][first:hi]))
                    out=arrays(16,np)
                    for t in range(16):
                        first=lo+t
                        assert api[4](ctx,1,tags[first:first+1].ctypes.data,inp[site][first:first+1].ctypes.data,
                            w,c,p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(out,t,t+1,np))==0
                    for k,v in out.items():equal(v,saved[k])
                    final=snap(api[5],ctx,np);final['last']=np.array([last(ctx)],'u4')
                    for k,v in final.items():equal(v,saved[k])
                finally:api[2](ctx)
                assert np.max(np.abs(saved['r'].astype('i4')))<=32767
                dot=saved['r'].astype('i8')@q64.T;equal(dot,saved['dots'])
                raw=(dot.astype('f8')*saved['a'][:,None])*p['scale'][None,:]
                equal(raw,saved['raw']);equal(BF(raw),saved['output']);checked+=1;words+=dot.size
        print(json.dumps(dict(site=site,checked=checked,seconds=time.monotonic()-start)),flush=True)
    assert checked==192 and words==6291456
    counts=replay_bridge(b,api,last,parameters,np,check);assert counts==r['bridge_counts']
    keys=set()
    for sample in r['samples']:
        rec=next(x for x in b['records'] if x['id']==sample['id']);position=sample['position']
        assert position in rec['positions'] and sample['mode'] in b['modes']
        assert sample['order']==([0,1] if (position-rec['positions'][0])%2==0 else [1,0])
        key=(sample['mode'],sample['id'],position);assert key not in keys;keys.add(key)
        for arm in ('original','wide'):
            s=sample[arm]
            for field in ('site_ns','readout_site_ns'):
                assert len(s[field])==24 and all(isinstance(v,int) and v>0 for v in s[field])
            assert s['whole_ns']>=sum(s['site_ns']) and s['readout_whole_ns']>=sum(s['readout_site_ns'])
            assert (s['flush_ns']==0 if sample['mode']=='resident' else s['flush_ns']>0)
    eligible=[]
    for mode in b['modes']:
        group=[s for s in r['samples'] if s['mode']==mode];old=r['summary'][mode];assert len(group)==64
        for arm in ('original','wide'):
            summaries=dict(whole_ns=stats([s[arm]['whole_ns'] for s in group]),
                per_site_ns=[stats([s[arm]['site_ns'][i] for s in group]) for i in range(24)],
                readout_whole_ns=stats([s[arm]['readout_whole_ns'] for s in group]),
                per_record_ns={k:stats([s[arm]['whole_ns'] for s in group if s['id']==k]) for k in sorted({s['id'] for s in group})})
            assert summaries==old['arms'][arm]
        ratio=old['arms']['wide']['whole_ns']['median']/old['arms']['original']['whole_ns']['median']
        strata={k:old['arms']['wide']['per_record_ns'][k]['median']/old['arms']['original']['per_record_ns'][k]['median']
                for k in old['arms']['wide']['per_record_ns']}
        passed=ratio<=.95 and max(strata.values())<=1.05
        assert ratio==old['median_ratio'] and strata==old['stratum_ratios'] and passed==old['eligible']
        assert old['remaining20ms_ns']==20_000_000-old['arms']['wide']['whole_ns']['median'];eligible.append(passed)
    assert len(r['noop_whole_ns'])==128 and stats(r['noop_whole_ns'])==r['noop_summary']
    assert r['parameter_payload_bytes']==sum(v.nbytes for p,w,c in parameters for v in p.values())==310855680
    assert r['GPU_calls']==r['RESERVED_queries']==0 and r['affinity']==[0] and r['kernel_threads']==1
    assert r['decision']==('EXACT_STORED_ACCELERATION_ELIGIBLE' if all(eligible) else 'EXACT_CANDIDATE_ACCELERATION_REJECTED')
    check_all();check()
    write(a.out,dict(schema='NATIVE_WIDE_READOUT_AUDIT_V1',report=extent(a.report),binding=r['binding'],
        checked_site_records=checked,checked_readout_words=words,bridge_counts=counts,
        all_arrays_and_snapshots_exact=True,independent_integer_and_scale_BF_pass=True,
        paired_accounting_pass=True,decision='ALL_STORED_EXACT_AND_ACCOUNTING_PASS',
        seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
        scope='Candidate recurrence DLL/ABI/buffer/replay helper shared; independent integer/scales/BF and timing summaries. '+b['scope']))
    print(json.dumps(dict(audit=extent(a.out))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    import psutil
    proc=psutil.Process();start=time.monotonic();receipt=dict(schema='NATIVE_WIDE_READOUT_AUDIT_TERMINAL_V1',
        command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try:main(a);receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)
