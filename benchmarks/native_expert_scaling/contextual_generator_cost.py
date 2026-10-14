"""Actual BF source generator cost; supplied own-history h, not a decoder."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(B), str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent, write

FAMILY = Path('E:/SiliconLLM/results/native_expert_scaling/contextual_generator_cost_20261014')
SCOPE = ('23 sequential actual BF source contextual generators on supplied true own-history h. '
         'No preceding decoder, recurrence, FFN, attention, head or transport. '
         'Cache intervention is not measured DRAM; no end-to-end/useful50/n/general-family admission.')
SLICES = [('gate',0,3072),('x',3072,6144),('B',6144,6400),('C',6400,6656),('dt',6656,6704)]


def load(p):
    return json.loads(Path(p).read_bytes())


def check(pins, guard):
    for e in pins:
        guard()
        assert extent(e['path']) == e, e['path']


def descriptor(e):
    return {k:e[k] for k in ('path','bytes','sha256')}


def decode(x, np):
    return (x.astype('u4') << 16).view('f4')


def bf(x, np):
    u = np.asarray(x, dtype='f4').view('u4')
    hi, lo = u >> 16, u & 65535
    return (hi + ((lo > 32768) | ((lo == 32768) & ((hi & 1) != 0)))).astype('u2')


def relative(x, y, np):
    return float(np.linalg.norm(x.astype('f8')-y.astype('f8'))/max(float(np.linalg.norm(y.astype('f8'))),1e-30))


def controls(fields, truth, np):
    x, y = decode(fields,np), decode(truth,np)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return dict(overall=relative(x,y,np), **{name:relative(x[:,lo:hi],y[:,lo:hi],np) for name,lo,hi in SLICES})


def stats(v, np):
    x = np.asarray(v,'f8')
    assert x.size and np.isfinite(x).all() and (x >= 0).all()
    return dict(count=int(x.size), minimum=float(x.min()), p10=float(np.quantile(x,.1)),
                median=float(np.median(x)), p90=float(np.quantile(x,.9)), maximum=float(x.max()))


def summarize(samples, b, np):
    result = {}
    for mode in b['modes']:
        s = [q for q in samples if q['mode'] == mode]
        whole = stats([q['whole_ns'] for q in s],np)
        result[mode] = dict(whole_ns=whole,
            per_site_ns=[stats([q['site_ns'][i] for q in s],np) for i in range(23)],
            per_record_ns={r['id']:stats([q['whole_ns'] for q in s if q['id']==r['id']],np) for r in b['records']},
            flush_ns=stats([q['flush_ns'] for q in s],np),
            exceeds_recurrent_room=whole['median'] > b['recurrent_room_ns'][mode])
    return result


def weights(b, np):
    owners = []
    with Path(b['source']['path']).open('rb') as f:
        n = int.from_bytes(f.read(8),'little'); header = json.loads(f.read(n)); base = 8+n
    for e in b['matrices']:
        item = header[e['key']]
        assert item['dtype']=='BF16' and item['shape']==[6704,2048]
        assert base+item['data_offsets'][0]==e['offset']
        assert item['data_offsets'][1]-item['data_offsets'][0]==e['bytes']==6704*2048*2
        owners.append(np.memmap(b['source']['path'],dtype='u2',mode='r',offset=e['offset'],shape=(6704,2048)))
    assert len(owners)==23
    return owners


def record_inputs(rec, np):
    inp, truth = [], []
    for i, e in enumerate(rec['traces'],1):
        assert e['site']==i and e['trace']['shape']==[rec['history'],10800]
        trace = np.memmap(e['trace']['path'],dtype='u2',mode='r',shape=(rec['history'],10800))
        inp.append(np.ascontiguousarray(trace[rec['positions'],:2048]))
        truth.append(np.ascontiguousarray(trace[rec['positions'],2048:8752]))
    h, fields = np.stack(inp), np.stack(truth)
    assert np.isfinite(decode(h,np)).all() and np.isfinite(decode(fields,np)).all()
    return h,fields


def native(b):
    dll = ctypes.CDLL(b['compile']['dll']['path'])
    dll.cg_abi.argtypes=[ctypes.c_int];dll.cg_abi.restype=ctypes.c_int
    assert [dll.cg_abi(i) for i in range(4)]==[2048,6704,8,23]
    dll.cg_control.argtypes=[];dll.cg_control.restype=ctypes.c_uint
    assert dll.cg_control()==8064
    for name in ('cg_step','cg_noop'):
        f = getattr(dll,name);f.argtypes=[ctypes.c_void_p]*6;f.restype=ctypes.c_int
    support = ctypes.CDLL(b['support']['path'])
    support.cost_evict.argtypes=[ctypes.c_void_p,ctypes.c_size_t];support.cost_evict.restype=None
    return dll,support


def guard_factory(b,start,proc):
    def guard():
        assert time.monotonic()-start<=b['limits']['seconds'],'phase deadline'
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'OS limit'
        assert not proc.children(recursive=True),'unexpected child'
        if FAMILY.exists():
            assert sum(p.stat().st_size for p in FAMILY.rglob('*') if p.is_file())<=b['limits']['output_bytes']
    return guard


def prepare(a):
    import numpy as np
    import psutil
    assert not FAMILY.exists()
    parent_path = DOC/'native_24_cost_binding_v2_20261014.json'
    parent = load(parent_path)
    full_path = DOC/'full_width_native_chat_binding_20261013.json'
    full = load(full_path)
    lut_path = DOC/'native_bf_activation_lut_session_closure_20261014.json'
    lut = load(lut_path)
    assert lut['decision']=='EXACT_STORED_ACCELERATION_ELIGIBLE'
    assert extent(lut['report']['path'])==lut['report']
    lut_result=load(lut['report']['path'])
    room={mode:20_000_000-lut_result['summary'][mode]['arms']['wide']['whole_ns']['median']
          for mode in ('resident','parameter_evicted')}
    assert room==dict(resident=3182750,parameter_evicted=2976150)
    assert extent(parent['support']['path'])==parent['support']
    source = Path(full['source'])/'model.safetensors'
    prior = next(e for e in full['inputs'] if Path(e['path']).resolve()==source.resolve())
    assert extent(source)==prior
    with source.open('rb') as f:
        n=int.from_bytes(f.read(8),'little');assert 0<n<8<<20
        header=json.loads(f.read(n));base=8+n
    matrices=[]
    for site in range(1,24):
        key=f'model.layers.{site}.mamba.in_proj.weight';item=header[key]
        assert item['dtype']=='BF16' and item['shape']==[6704,2048]
        lo,hi=item['data_offsets'];assert hi-lo==6704*2048*2
        assert key.replace('.weight','.bias') not in header
        matrices.append(dict(site=site,key=key,offset=base+lo,bytes=hi-lo,shape=item['shape'],dtype='BF16'))
    assert all(x['offset']+x['bytes']<=source.stat().st_size for x in matrices)
    assert all(x['offset']+x['bytes']<=y['offset'] or y['offset']+y['bytes']<=x['offset']
               for i,x in enumerate(matrices) for y in matrices[i+1:])
    records=[];paths=[source,parent_path,full_path,lut_path,Path(parent['support']['path'])]
    for rec in parent['records']:
        assert rec['positions']==list(range(rec['history']-16,rec['history']))
        traces=[dict(site=i,trace=rec['sequence']['sites'][i]['trace']) for i in range(1,24)]
        for entry in traces:
            assert extent(entry['trace']['path'])==descriptor(entry['trace'])
            paths.append(Path(entry['trace']['path']))
        records.append({**{k:rec[k] for k in ('id','split','history','positions','length_kind')},'traces':traces})
    frozen=[Path(__file__),B/'contextual_generator_cost_audit.py',B/'contextual_generator_cost.c',
            DOC/'CONTEXTUAL_GENERATOR_COST_PROTOCOL_20261014.md']
    for p in frozen:
        blob=subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n'),str(p)
    compiler=Path(full['compile']['command'][0])
    paths.extend(frozen+[B/'falcon_auxiliary_s6_transport.py',B/'native_24_cost_support.c',
        Path(sys.executable),Path(sys.executable).parent/'python312.dll',compiler,
        DOC/'NATIVE_BF_ACTIVATION_LUT_RESULT_20261014.md',
        DOC/'native_bf_activation_lut_result_20261014.json'])
    for module in (np,psutil):
        paths.extend(p for p in Path(module.__file__).resolve().parent.rglob('*')
                     if p.is_file() and p.suffix in ('.py','.pyd','.dll'))
    paths.extend((Path(np.__file__).resolve().parent.parent/'numpy.libs').glob('*.dll'))
    paths.extend(compiler.parent.glob('*.dll'));paths.extend(compiler.parent.glob('*lld*.exe'))
    paths.extend(Path(m.__file__) for m in tuple(sys.modules.values())
                 if getattr(m,'__file__',None) and Path(m.__file__).is_file())
    paths.extend(ROOT/p for p in ('benchmarks/phase60/engine.c',
        'benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md'))
    pins=[extent(p) for p in dict.fromkeys(p.resolve() for p in paths)]
    FAMILY.mkdir();dll=FAMILY/'generator.dll';log=FAMILY/'compile.log'
    command=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off',
             '-shared',str(B/'contextual_generator_cost.c'),'-o',str(dll),'-lm']
    co=subprocess.run(command,capture_output=True,timeout=45,creationflags=0x08000000)
    log.write_bytes(co.stdout+co.stderr);assert co.returncode==0
    pins.extend([extent(dll),extent(log)])
    b=dict(schema='CONTEXTUAL_GENERATOR_COST_BINDING_V1',freeze=a.freeze,python=str(Path(sys.executable).resolve()),
        versions=dict(numpy=np.__version__,psutil=psutil.__version__),source=prior,matrices=matrices,
        MuP_bits=full['MuP_bits'],records=records,inputs=pins,
        compile=dict(command=command,dll=extent(dll),log=extent(log)),support=parent['support'],
        parent_cost_binding=extent(parent_path),activation_closure=extent(lut_path),affinity=[0],
        modes=['resident','parameter_evicted'],recurrent_room_ns=room,
        limits=dict(seconds=1800,OS_bytes=4<<30,output_bytes=192<<20),
        gates=dict(raw_relative=1e-4,field_relative=.01),output_directory=str(FAMILY/'run'),
        DRAM='UNRESOLVED: exact source operands/cache intervention, no physical traffic counters',scope=SCOPE)
    assert len(b['MuP_bits'])==6704
    native(b);check(pins,lambda:None);write(a.out,b)
    print(json.dumps(dict(binding=extent(a.out),matrix_bytes=sum(e['bytes'] for e in matrices))),flush=True)


def run(a):
    import numpy as np
    import psutil
    b=load(a.binding);assert extent(a.binding)['sha256']==a.binding_sha
    assert str(Path(sys.executable).resolve())==b['python']
    assert dict(numpy=np.__version__,psutil=psutil.__version__)==b['versions']
    prep=load(a.binding.with_suffix('.preparation_terminal.json'))
    assert prep['exit_code']==0 and prep['result']==extent(a.binding)
    directory=Path(b['output_directory']);assert not directory.exists()
    proc=psutil.Process();proc.cpu_affinity(b['affinity']);start=time.monotonic()
    guard=guard_factory(b,start,proc);check(b['inputs'],guard)
    for peer in psutil.process_iter(['pid','name','cmdline']):
        try:
            if peer.pid!=proc.pid and (peer.info['name'] or '').lower().startswith('python'):
                assert not any('native_expert_scaling' in x for x in (peer.info['cmdline'] or [])),peer.info
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    dll,support=native(b);owners=weights(b,np);mup=np.array(b['MuP_bits'],'u2')
    assert all(np.isfinite(decode(w,np)).all() for w in owners)
    assert np.isfinite(decode(mup,np)).all()
    inventory=[dict(**e,address=w.ctypes.data) for e,w in zip(b['matrices'],owners)]
    cpu_before=[p._asdict() for p in psutil.cpu_times(percpu=True)]
    directory.mkdir();samples=[];records=[];overhead=[]
    for mode in b['modes']:
        for rec in b['records']:
            guard();h,truth=record_inputs(rec,np)
            raw=np.empty((23,16,6704),'f4');gg=np.empty((23,16,6704),'u2');fields=np.empty_like(gg)
            calls=[[(w.ctypes.data,h[i,t].ctypes.data,mup.ctypes.data,raw[i,t].ctypes.data,
                     gg[i,t].ctypes.data,fields[i,t].ctypes.data) for i,w in enumerate(owners)] for t in range(16)]
            if not overhead:
                for repeat in range(128):
                    before=time.perf_counter_ns();site_times=[];rcs=[]
                    for call in calls[0]:
                        first=time.perf_counter_ns();rc=dll.cg_noop(*call);last=time.perf_counter_ns()
                        site_times.append(last-first);rcs.append(rc)
                    overhead.append(time.perf_counter_ns()-before)
                    assert rcs==[0]*23
            # Untimed warming touches every actual coefficient and executes every site.
            for call in calls[0]:assert dll.cg_step(*call)==0
            for t,group in enumerate(calls):
                guard();flush_before=time.perf_counter_ns()
                if mode=='parameter_evicted':
                    for w in owners:support.cost_evict(w.ctypes.data,w.nbytes)
                    support.cost_evict(mup.ctypes.data,mup.nbytes)
                flush=time.perf_counter_ns()-flush_before if mode=='parameter_evicted' else 0
                site_times=[];rcs=[];begin=time.perf_counter_ns()
                for call in group:
                    before=time.perf_counter_ns();rc=dll.cg_step(*call);after=time.perf_counter_ns()
                    site_times.append(after-before);rcs.append(rc)
                elapsed=time.perf_counter_ns()-begin;assert rcs==[0]*23
                samples.append(dict(mode=mode,id=rec['id'],split=rec['split'],length_kind=rec['length_kind'],
                    position=rec['positions'][t],whole_ns=elapsed,site_ns=site_times,flush_ns=flush))
            assert np.isfinite(raw).all()
            control=[controls(fields[i],truth[i],np) for i in range(23)]
            assert max(v for c in control for v in c.values())<=b['gates']['field_relative'],control
            p=directory/(rec['id']+'.'+mode+'.npz');np.savez(p,raw=raw,gg=gg,fields=fields)
            records.append(dict(id=rec['id'],mode=mode,file=extent(p),source_controls=control))
            print(json.dumps(dict(mode=mode,id=rec['id'],samples=len(samples),seconds=time.monotonic()-start)),flush=True)
    check(b['inputs'],guard);guard();summary=summarize(samples,b,np)
    decision=('DENSE_CONTEXTUAL_GENERATORS_EXCEED_REMAINING_BUDGET' if any(
        q['exceeds_recurrent_room'] for q in summary.values()) else 'GENERATOR_ONLY_WITHIN_RECURRENT_ROOM')
    r=dict(schema='CONTEXTUAL_GENERATOR_COST_RESULT_V1',binding=extent(a.binding),records=records,samples=samples,
        summary=summary,decision=decision,noop_whole_ns=overhead,noop_summary=stats(overhead,np),
        matrix_inventory=inventory,parameter_payload_bytes=sum(w.nbytes for w in owners),
        MuP_bytes=mup.nbytes,MuP_address=mup.ctypes.data,FP_control=dll.cg_control(),
        last_record_buffers=dict(input_h_bytes=h.nbytes,source_truth_bytes=truth.nbytes,
            raw_bytes=raw.nbytes,gg_bytes=gg.nbytes,field_bytes=fields.nbytes),
        output_bytes=sum(p.stat().st_size for p in directory.iterdir()),
        operand_ledger=dict(weight_bytes_per_group=23*6704*2048*2,input_h_bytes_per_group=23*2048*2,
            decoded_input_stack_bytes_per_call=2048*4,FMA_per_group=23*6704*2048,
            decoded_input_logical_read_bytes_per_group=23*6704*2048*4,
            MuP_logical_read_bytes_per_group=23*6704*2,output_bytes_per_group=23*6704*8,
            note='BF coefficient storage, F32 register decode/input stack; logical operands are not DRAM'),
        cpu_times_before=cpu_before,cpu_times_after=[p._asdict() for p in psutil.cpu_times(percpu=True)],
        affinity=proc.cpu_affinity(),kernel_threads=1,shared_desktop=True,DRAM=b['DRAM'],scope=SCOPE,
        seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
        GPU_calls=0,source_model_forwards=0,optimizer_updates=0,RESERVED_queries=0)
    write(a.out,r);print(json.dumps(dict(result=extent(a.out),decision=decision,
        median_ms={k:v['whole_ns']['median']/1e6 for k,v in summary.items()})),flush=True)


def close(a):
    import psutil
    r=load(a.report);ad=load(a.audit);b=load(r['binding']['path'])
    assert ad['decision']=='INDEPENDENT_STORED_NUMERIC_AND_ACCOUNTING_PASS'
    assert ad['report']==extent(a.report) and ad['checked_calls']==2944 and ad['checked_records']==8
    receipts=[]
    for p,suffix in ((Path(r['binding']['path']),'.preparation_terminal.json'),
                     (a.report,'.terminal.json'),(a.audit,'.terminal.json')):
        t=load(p.with_suffix(suffix));assert t['exit_code']==0 and t['result']==extent(p)
        if psutil.pid_exists(t['pid']):assert abs(psutil.Process(t['pid']).create_time()-t['creation_time'])>.001
        receipts.append(dict(receipt=extent(p.with_suffix(suffix)),pid=t['pid'],creation_time=t['creation_time'],
                             matching_process_absent=True))
    check(b['inputs']+[r['binding']],lambda:None)
    files=[q['file'] for q in r['records']];check(files,lambda:None)
    actual={str(p.resolve()):p.stat().st_size for p in FAMILY.rglob('*') if p.is_file()}
    expected={e['path']:e['bytes'] for e in files+[b['compile']['dll'],b['compile']['log']]}
    assert actual==expected
    write(a.out,dict(schema='CONTEXTUAL_GENERATOR_COST_CLOSURE_V1',binding=r['binding'],report=extent(a.report),
        audit=extent(a.audit),commands=receipts,family_files=len(actual),family_bytes=sum(actual.values()),
        all_inputs_outputs_exact=True,decision='ACTUAL_BF_CONTEXTUAL_GENERATOR_COST_QUALIFIED',
        budget_decision=r['decision'],goal_status='ACTIVE_INCOMPLETE',scope=SCOPE))
    print(json.dumps(dict(closure=extent(a.out))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','close'])
    for name in ('out','binding','report','audit'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--freeze');parser.add_argument('--binding-sha');a=parser.parse_args()
    import psutil
    proc=psutil.Process();proc.cpu_affinity([0]);start=time.monotonic()
    receipt=dict(schema='CONTEXTUAL_GENERATOR_COST_TERMINAL_V1',phase=a.phase,command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time())
    try:
        assert a.out and not a.out.exists()
        {'prepare':prepare,'run':run,'close':close}[a.phase](a)
        assert time.monotonic()-start<=1800 and proc.memory_info().peak_wset<=4<<30
        receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.preparation_terminal.json' if a.phase=='prepare' else '.terminal.json'),receipt)
