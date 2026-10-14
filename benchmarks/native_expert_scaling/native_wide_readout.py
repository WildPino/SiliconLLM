"""Exact readout change, paired physical cost, all known native trace compatibility."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0]=[str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write
from native_24_cost import load,check_pins,parameter_sets,record_rows,snap,distribution,PARAM_KEYS
from row_state import interface
from token_state import arrays,arguments
FAMILY=Path('E:/SiliconLLM/results/native_expert_scaling/native_wide_readout_20261014')
SCOPE='Exact accumulation layout only; stored native bridge compatibility and supplied-row cost, not final CPU model quality/TPS/DRAM/general-family admission.'


def equal(x,y):
    assert x.dtype==y.dtype and x.shape==y.shape and (x.view('u1')==y.view('u1')).all()


def guard(b,start,proc):
    assert time.monotonic()-start<=b['limits']['seconds'] and proc.memory_info().peak_wset<=b['limits']['OS_bytes']
    assert not proc.children(recursive=True)
    if FAMILY.exists():assert sum(p.stat().st_size for p in FAMILY.rglob('*') if p.is_file())<=b['limits']['output_bytes']


def setup(b):
    result=[]
    for compile in (b['compile'],b['candidate_compile']):
        api=interface(dict(compile=compile));last=api[0].ts_last
        last.argtypes=[ctypes.c_void_p];last.restype=ctypes.c_uint32
        read=api[0].ir_history;read.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*7;read.restype=ctypes.c_int
        result.append((api,last,read))
    return result


def controls(np):
    rng=np.random.default_rng(0)
    w=rng.integers(-364,365,size=(64,3072),dtype='i2')
    x=rng.integers(-32767,32768,size=(64,3072),dtype='i2')
    w[:8]=364;x[:8]=32767
    w[1]=-364;x[2]=-32767;w[3]=0
    w[4,1::2]=-364;x[4,1::2]=-32767
    x[5,1::2]=-32767;w[6,:1024]=-364;w[6,2048:]=-364
    w[7,np.arange(3072)%16>=2]=0
    return w,x


def paired_summary(samples,np):
    result={}
    for mode in ('resident','parameter_evicted'):
        group=[s for s in samples if s['mode']==mode];arms={}
        for arm in ('original','wide'):
            arms[arm]=dict(whole_ns=distribution([s[arm]['whole_ns'] for s in group],np),
                per_site_ns=[distribution([s[arm]['site_ns'][i] for s in group],np) for i in range(24)],
                readout_whole_ns=distribution([s[arm]['readout_whole_ns'] for s in group],np),
                per_record_ns={key:distribution([s[arm]['whole_ns'] for s in group if s['id']==key],np)
                               for key in sorted({s['id'] for s in group})})
        ratio=arms['wide']['whole_ns']['median']/arms['original']['whole_ns']['median']
        strata={k:arms['wide']['per_record_ns'][k]['median']/arms['original']['per_record_ns'][k]['median']
                for k in arms['wide']['per_record_ns']}
        result[mode]=dict(arms=arms,median_ratio=ratio,stratum_ratios=strata,
            eligible=ratio<=.95 and all(v<=1.05 for v in strata.values()),
            remaining20ms_ns=20_000_000-arms['wide']['whole_ns']['median'])
    return result


def prepare(a):
    import numpy as np,psutil
    assert not FAMILY.exists() and not a.out.exists()
    parent=DOC/'native_24_cost_binding_v2_20261014.json';b=load(parent)
    cp=DOC/'native_24_cost_session_closure_20261014.json';assert load(cp)['decision']=='RECURRENT_COST_SCREEN_QUALIFIED'
    rp=DOC/'native_24_cost_result_20261014.json'; old=load(rp)
    bridge=DOC/'full_width_native_chat_result_20261013.json';r=load(bridge)
    sequences=[x['sequence'] for x in r['tasks']+[r['followup']]+r['records']]
    sequences.extend(c[k] for c in r['prefix_controls'] for k in ('wrapper','continuation'))
    assert len(sequences)==97 and sum(s['seen'] for s in sequences)==24169
    extra=[Path(__file__),B/'native_wide_readout_audit.py',B/'integer_readout_wide.c',
        DOC/'NATIVE_WIDE_READOUT_PROTOCOL_20261014.md',parent,cp,rp,
        DOC/'native_24_cost_adjudication_20261014.json',bridge,
        DOC/'full_width_native_chat_adjudication_20261013.json',
        DOC/'full_width_native_chat_session_closure_20261013.json']
    for p in extra[:4]:
        saved=subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert saved.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n')
    check_pins(b,lambda:None)
    pins=b['inputs']+[extent(p) for p in extra]
    for seq in sequences:
        for item in seq['sites']:
            for key in ('trace','final'):
                e=item[key];actual=extent(e['path'])
                assert all(actual[k]==e[k] for k in ('path','bytes','sha256'));pins.append(actual)
    for rec in old['records']:pins.extend(x['file'] for x in rec['files'])
    pins=list({e['path']:e for e in pins}.values())
    original=(B/'row_state.c').read_text();needle='#include "integer_readout.c"'
    assert original.count(needle)==1
    generated=original.replace(needle,'#include "integer_readout_wide.c"')
    assert generated.replace('#include "integer_readout_wide.c"',needle)==original
    FAMILY.mkdir();src=FAMILY/'row_state_wide.c';src.write_text(generated,encoding='utf8',newline='\n')
    dll=FAMILY/'row_state_wide.dll';log=FAMILY/'compile.log'
    command=[b['compile']['command'][0],'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off',
             '-shared','-I',str(B),str(src),'-o',str(dll),'-lm']
    co=subprocess.run(command,capture_output=True,timeout=45,creationflags=0x08000000)
    log.write_bytes(co.stdout+co.stderr);assert co.returncode==0
    b.update(schema='NATIVE_WIDE_READOUT_BINDING_V1',freeze=a.freeze,
        candidate_compile=dict(command=command,dll=extent(dll),source=extent(src),log=extent(log)),
        parent_binding=extent(parent),parent_result=extent(rp),parent_closure=extent(cp),
        reference_records=old['records'],sequences=sequences,
        inputs=pins+[extent(src),extent(dll),extent(log)],
        output_directory=str(FAMILY/'run'),scope=SCOPE,
        limits=dict(seconds=1800,OS_bytes=2<<30,output_bytes=32<<20))
    check_pins(b,lambda:None);write(a.out,b)
    print(json.dumps(dict(binding=extent(a.out))),flush=True)


def replay_bridge(b,api,last,parameters,np,check):
    counts=dict(sequence_sites=0,positions=0,output_words=0,final_state_words=0)
    _,create,destroy,_,rows,snapshot=api
    for export,(p,w,c) in zip(b['exports'],parameters,strict=True):
        site=export['site']
        for seq in b['sequences']:
            check();rec=dict(sequence=seq,history=seq['seen'])
            tags=np.array(seq['consumed_ids'],'u4');n=len(tags);item=seq['sites'][site]
            trace=np.memmap(item['trace']['path'],'u2','r',shape=(n,2048 if site==0 else 10800))
            if site==0:
                table=np.memmap(b['table']['path'],'u2','r',shape=(65537,6704));inp=np.ascontiguousarray(table[tags]);expected=trace
            else:inp=np.ascontiguousarray(trace[:,2048:8752]);expected=trace[:,8752:]
            ctx=create();assert ctx
            try:
                for first in range(0,n,64):
                    check();hi=min(first+64,n);z=arrays(hi-first,np)
                    assert rows(ctx,hi-first,tags[first:hi].ctypes.data,inp[first:hi].ctypes.data,w,c,
                        p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(z,0,hi-first,np))==0
                    equal(z['output'],np.ascontiguousarray(expected[first:hi]));counts['output_words']+=z['output'].size
                final=snap(snapshot,ctx,np);final['last']=np.array([last(ctx)],'u4')
                with np.load(item['final']['path'],allow_pickle=False) as saved:
                    for k,v in final.items():equal(v,saved[k])
                counts['sequence_sites']+=1;counts['positions']+=n;counts['final_state_words']+=3072*96
            finally:destroy(ctx)
        print(json.dumps(dict(bridge_site=site,**counts)),flush=True)
    assert counts['sequence_sites']==2328 and counts['positions']==580056 and counts['output_words']==1187954688
    return counts


def run(a):
    import numpy as np,psutil
    b=load(a.binding);assert extent(a.binding)['sha256']==a.binding_sha
    assert not a.out.exists() and not Path(b['output_directory']).exists()
    proc=psutil.Process();proc.cpu_affinity([0]);start=time.monotonic();check=lambda:guard(b,start,proc)
    assert Path(sys.executable).resolve()==Path(b['python'])
    assert dict(numpy=np.__version__,psutil=psutil.__version__)==b['versions']
    check_pins(b,check)
    for peer in psutil.process_iter(['pid','name','cmdline']):
        try:
            if peer.pid!=proc.pid and (peer.info['name'] or '').lower().startswith('python'):
                assert not any('native_expert_scaling' in v for v in peer.info['cmdline'] or []),peer.info
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    directory=Path(b['output_directory']);directory.mkdir();apis=setup(b);parameters=parameter_sets(b,np)
    w,x=controls(np);dot=np.empty(64,'i8');control=apis[1][0][0].ir_dot_controls
    control.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*3;control.restype=ctypes.c_int
    assert control(64,w.ctypes.data,x.ctypes.data,dot.ctypes.data)==0
    truth=np.array([sum(int(v)*int(u) for v,u in zip(ww,xx)) for ww,xx in zip(w,x)],'i8');equal(dot,truth)
    control_path=directory/'adversarial.npz';np.savez(control_path,weight=w,input=x,dots=dot)
    tie_q=np.zeros((2048,3072),'i2');tie_q[:,0]=1;tie_x=np.zeros((1,3072),'i2');tie_x[0,0]=1
    tie_scale=np.resize(np.array([0.,1.00390625,1.01171875,-1.00390625],'f8'),2048)
    tie_a=np.ones(1,'f8');tie_d=np.empty((1,2048),'i8');tie_raw=np.empty((1,2048),'f8');tie_bits=np.empty((1,2048),'u2')
    for api,last,readout in apis:
        assert readout(1,tie_q.ctypes.data,tie_scale.ctypes.data,tie_x.ctypes.data,tie_a.ctypes.data,
            tie_d.ctypes.data,tie_raw.ctypes.data,tie_bits.ctypes.data)==0
        assert np.all(tie_d==1);equal(tie_raw[0],tie_scale)
    tie_path=directory/'BF_ties.npz';np.savez(tie_path,scale=tie_scale,dots=tie_d,raw=tie_raw,bits=tie_bits)
    del tie_q,tie_x
    support=ctypes.CDLL(b['support']['path']);evict=support.cost_evict
    evict.argtypes=[ctypes.c_void_p,ctypes.c_size_t];evict.restype=None
    noop=support.cost_noop;noop.argtypes=apis[0][0][4].argtypes;noop.restype=ctypes.c_int
    samples=[];records=[];overhead=[];contexts=[];cpu_before=[t._asdict() for t in psutil.cpu_times(percpu=True)]
    try:
        for mode in b['modes']:
            for rec in b['records']:
                check();tags,inp,expected=record_rows(b,rec,np);lo=rec['history']-16
                outs=[[arrays(16,np) for _ in range(24)] for _ in range(2)]
                direct=[[dict(dots=np.empty((16,2048),'i8'),raw=np.empty((16,2048),'f8'),bits=np.empty((16,2048),'u2'))
                         for _ in range(24)] for _ in range(2)]
                contexts=[[arm[0][1]() for _ in range(24)] for arm in apis];assert all(all(cs) for cs in contexts)
                saved=[]
                old=next(q for q in b['reference_records'] if q['mode']==mode and q['id']==rec['id'])
                for i,item in enumerate(old['files']):
                    with np.load(item['file']['path'],allow_pickle=False) as z:saved.append({k:z[k].copy() for k in z.files})
                    p,wp,cp=parameters[i]
                    for arm,(api,last,readout) in enumerate(apis):
                        for first in range(0,lo,64):
                            check();hi=min(first+64,lo);z=arrays(hi-first,np)
                            assert api[4](contexts[arm][i],hi-first,tags[first:hi].ctypes.data,inp[i][first:hi].ctypes.data,
                                wp,cp,p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(z,0,hi-first,np))==0
                            equal(z['output'],np.ascontiguousarray(expected[i][first:hi]))
                calls=[[[ (contexts[arm][i],1,tags[lo+t:lo+t+1].ctypes.data,inp[i][lo+t:lo+t+1].ctypes.data,
                        wp,cp,p['q'].ctypes.data,p['scale'].ctypes.data,*arguments(outs[arm][i],t,t+1,np))
                        for i,(p,wp,cp) in enumerate(parameters)] for arm in range(2)] for t in range(16)]
                read_calls=[[[ (1,p['q'].ctypes.data,p['scale'].ctypes.data,
                        outs[arm][i]['r'][t:t+1].ctypes.data,outs[arm][i]['a'][t:t+1].ctypes.data,
                        direct[arm][i]['dots'][t:t+1].ctypes.data,direct[arm][i]['raw'][t:t+1].ctypes.data,
                        direct[arm][i]['bits'][t:t+1].ctypes.data)
                        for i,(p,wp,cp) in enumerate(parameters)] for arm in range(2)] for t in range(16)]
                if not overhead:
                    for _ in range(128):
                        begin=time.perf_counter_ns()
                        for call in calls[0][0]:
                            before=time.perf_counter_ns();rc=noop(*call);after=time.perf_counter_ns();assert rc==0
                        overhead.append(time.perf_counter_ns()-begin)
                for t in range(16):
                    check();order=[0,1] if t%2==0 else [1,0]
                    sample=dict(mode=mode,id=rec['id'],position=lo+t,order=order)
                    for arm in order:
                        fb=time.perf_counter_ns()
                        if mode=='parameter_evicted':
                            for p,wp,cp in parameters:
                                for k in PARAM_KEYS:evict(p[k].ctypes.data,p[k].nbytes)
                        flush=time.perf_counter_ns()-fb if mode=='parameter_evicted' else 0
                        times=[];returns=[];begin=time.perf_counter_ns()
                        for call in calls[t][arm]:
                            before=time.perf_counter_ns();rc=apis[arm][0][4](*call);after=time.perf_counter_ns()
                            times.append(after-before);returns.append(rc)
                        elapsed=time.perf_counter_ns()-begin;assert returns==[0]*24
                        sample[('original','wide')[arm]]=dict(whole_ns=elapsed,site_ns=times,flush_ns=flush)
                    for arm in order:
                        times=[];returns=[];begin=time.perf_counter_ns()
                        for call in read_calls[t][arm]:
                            rb=time.perf_counter_ns();rc=apis[arm][2](*call);re=time.perf_counter_ns()
                            times.append(re-rb);returns.append(rc)
                        elapsed=time.perf_counter_ns()-begin;assert returns==[0]*24
                        sample[('original','wide')[arm]].update(readout_site_ns=times,readout_whole_ns=elapsed)
                    samples.append(sample)
                for arm,(api,last,readout) in enumerate(apis):
                    for i,ctx in enumerate(list(contexts[arm])):
                        for k,v in outs[arm][i].items():equal(v,saved[i][k])
                        for k,v in direct[arm][i].items():equal(v,saved[i]['output' if k=='bits' else k])
                        final=snap(api[5],ctx,np);final['last']=np.array([last(ctx)],'u4')
                        for k,v in final.items():equal(v,saved[i][k])
                        api[2](ctx);contexts[arm].remove(ctx)
                contexts=[];records.append(dict(id=rec['id'],mode=mode,reference=old['files'],all_arrays_exact=True))
                print(json.dumps(dict(paired_mode=mode,id=rec['id'],samples=len(samples),seconds=time.monotonic()-start)),flush=True)
        bridge_counts=replay_bridge(b,apis[1][0],apis[1][1],parameters,np,check)
        check_pins(b,check);check();summary=paired_summary(samples,np)
        result=dict(schema='NATIVE_WIDE_READOUT_RESULT_V1',binding=extent(a.binding),samples=samples,
            records=records,summary=summary,adversarial=extent(control_path),BF_ties=extent(tie_path),bridge_counts=bridge_counts,
            noop_whole_ns=overhead,noop_summary=distribution(overhead,np),
            parameter_payload_bytes=sum(v.nbytes for p,w,c in parameters for v in p.values()),
            cpu_times_before=cpu_before,cpu_times_after=[t._asdict() for t in psutil.cpu_times(percpu=True)],
            affinity=proc.cpu_affinity(),kernel_threads=1,DRAM=b['DRAM'],GPU_calls=0,RESERVED_queries=0,
            decision='EXACT_STORED_ACCELERATION_ELIGIBLE' if all(s['eligible'] for s in summary.values()) else 'EXACT_CANDIDATE_ACCELERATION_REJECTED',
            seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,scope=SCOPE)
        write(a.out,result);print(json.dumps(dict(result=extent(a.out),decision=result['decision'],
            ratios={k:v['median_ratio'] for k,v in summary.items()})),flush=True)
    finally:
        for arm,cs in enumerate(contexts):
            for ctx in cs:apis[arm][0][2](ctx)


def close(a):
    import psutil
    r=load(a.report);b=load(r['binding']['path']);ad=load(a.audit)
    assert ad['decision']=='ALL_STORED_EXACT_AND_ACCOUNTING_PASS' and ad['report']==extent(a.report)
    commands=[]
    for path,suffix in ((Path(r['binding']['path']),'.preparation_terminal.json'),(a.report,'.terminal.json'),(a.audit,'.terminal.json')):
        receipt=load(path.with_suffix(suffix));assert receipt['exit_code']==0 and receipt['result']==extent(path)
        if psutil.pid_exists(receipt['pid']):assert abs(psutil.Process(receipt['pid']).create_time()-receipt['creation_time'])>.001
        commands.append(dict(terminal=extent(path.with_suffix(suffix)),pid=receipt['pid'],creation_time=receipt['creation_time'],matching_process_absent=True))
    check_pins(b,lambda:None)
    for k in ('adversarial','BF_ties'):assert extent(r[k]['path'])==r[k]
    expected={e['path']:e['bytes'] for e in [b['candidate_compile'][k] for k in ('dll','source','log')]+[r['adversarial'],r['BF_ties']]}
    actual={str(p.resolve()):p.stat().st_size for p in FAMILY.rglob('*') if p.is_file()};assert actual==expected
    write(a.out,dict(schema='NATIVE_WIDE_READOUT_CLOSURE_V1',report=extent(a.report),audit=extent(a.audit),
        binding=r['binding'],commands=commands,family_files=len(actual),family_bytes=sum(actual.values()),
        all_input_output_hashes_exact=True,decision=r['decision'],goal_status='ACTIVE_INCOMPLETE',scope=SCOPE))
    print(json.dumps(dict(closure=extent(a.out))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','close'])
    for name in ('out','binding','report','audit'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--freeze');parser.add_argument('--binding-sha');a=parser.parse_args()
    import psutil
    proc=psutil.Process();start=time.monotonic();receipt=dict(schema='NATIVE_WIDE_READOUT_TERMINAL_V1',
        phase=a.phase,command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try:
        assert not a.out.exists();{'prepare':prepare,'run':run,'close':close}[a.phase](a)
        assert time.monotonic()-start<=1800 and proc.memory_info().peak_wset<=2<<30
        receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.preparation_terminal.json' if a.phase=='prepare' else '.terminal.json'),receipt)
