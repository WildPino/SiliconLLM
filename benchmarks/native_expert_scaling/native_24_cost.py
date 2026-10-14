"""Bound, state-ready, sequential cost of the admitted 24 recurrent branches."""
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
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(B), str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent, write
from row_state import interface
from token_state import arrays, arguments, ptrs

STEM = 'native_24_cost_20261014'
FAMILY = Path('E:/SiliconLLM/results/native_expert_scaling') / STEM
PARAM_KEYS = ['Ws', 'Vs', 'Alog', 'D', 'gamma', 'cw', 'cb', 'db', 'q', 'scale']
SCOPE = ('Supplied actual causal preconv rows, 24 sequential own-state native branches. '
         'Excludes contextual generators, FFN, attention, head, tokenizer and transport. '
         'No final model/TPS/useful-n/DRAM/general-family admission.')


def load(p):
    return json.loads(Path(p).read_bytes())


def distribution(values, np):
    a = np.asarray(values, dtype='f8')
    assert a.size and np.isfinite(a).all() and (a >= 0).all()
    return dict(count=int(a.size), minimum=float(a.min()), p10=float(np.quantile(a, .1)),
                median=float(np.median(a)), p90=float(np.quantile(a, .9)), maximum=float(a.max()))


def summaries(samples, np):
    result = {}
    for mode in ('resident', 'parameter_evicted'):
        group = [s for s in samples if s['mode'] == mode]
        whole = distribution([s['whole_ns'] for s in group], np)
        result[mode] = dict(whole_ns=whole,
            per_site_ns=[distribution([s['site_ns'][i] for s in group], np) for i in range(24)],
            per_record_ns={name:distribution([s['whole_ns'] for s in group if s['id'] == name], np)
                           for name in sorted({s['id'] for s in group})},
            flush_ns=distribution([s['flush_ns'] for s in group], np),
            remaining_20ms_ns=20_000_000-whole['median'],
            remaining_10ms_ns=10_000_000-whole['median'])
    return result


def guard_factory(binding, start, proc):
    def guard():
        assert time.monotonic()-start <= binding['limits']['seconds'], 'phase deadline'
        assert proc.memory_info().peak_wset <= binding['limits']['OS_bytes'], 'OS limit'
        assert not proc.children(recursive=True), 'unexpected child'
        if FAMILY.exists():
            assert sum(p.stat().st_size for p in FAMILY.rglob('*') if p.is_file()) <= binding['limits']['output_bytes']
    return guard


def check_pins(binding, guard):
    for e in binding['inputs']:
        guard()
        assert extent(e['path']) == e, e['path']


def prepare(a):
    import numpy as np
    import psutil
    assert not a.out.exists() and not FAMILY.exists()
    assert subprocess.check_output(['git', 'rev-parse', a.freeze], cwd=ROOT, text=True).strip() == a.freeze
    rp = DOC/'full_width_native_chat_result_20261013.json'
    ap = DOC/'full_width_native_chat_adjudication_20261013.json'
    cp = DOC/'full_width_native_chat_session_closure_20261013.json'
    r, ad = load(rp), load(ap)
    assert ad['decision'] == 'INDEPENDENT_STORED_QUALITY_PASS' and ad['report'] == extent(rp)
    closure = load(cp)
    assert closure['report'] == extent(rp) and closure['adjudication'] == extent(ap)
    old = load(r['binding']['path'])
    selected = []
    for split in ('FIT', 'DEV'):
        group = [x for x in r['records'] if x['split'] == split]
        short = min(group, key=lambda x:(x['sequence']['seen'], x['id']))
        long = min(group, key=lambda x:(-x['sequence']['seen'], x['id']))
        for kind, item in (('short', short), ('long', long)):
            seq = item['sequence']; n = seq['seen']
            assert n >= 16 and len(seq['sites']) == 24 and n == len(seq['consumed_ids'])
            selected.append(dict(id=item['id'], split=split, length_kind=kind,
                history=n, positions=list(range(n-16,n)), sequence=seq))
    paths = [Path(__file__), B/'native_24_cost_audit.py', B/'native_24_cost_support.c',
        DOC/'NATIVE_24_COST_PROTOCOL_20261014.md', B/'row_state.py', B/'row_state.c',
        B/'token_state.py', B/'token_state.c', B/'full_channel_native.py', B/'full_channel_native.c',
        B/'integer_readout.c', B/'composed_operator_mixed.py', B/'falcon_auxiliary_s6_transport.py',
        rp, ap, cp, Path(r['binding']['path']), DOC/'row_state_binding_20261013.json',
        DOC/'row_state_adjudication_20261013.json', DOC/'row_state_session_closure_20261013.json',
        DOC/'ffn_active_router_session_closure_20261014.json', Path(old['table']['path']),
        Path(old['compile']['dll']['path']), Path(sys.executable), Path(sys.executable).parent/'python312.dll']
    compiler = Path(old['compile']['command'][0]); paths.append(compiler)
    for item in old['exports']: paths.append(Path(item['file']['path']))
    for rec in selected:
        for site in rec['sequence']['sites']:
            paths.extend([Path(site[k]['path']) for k in ('trace','final')])
    for module in (np, psutil):
        paths.extend(p for p in Path(module.__file__).resolve().parent.rglob('*')
                     if p.is_file() and p.suffix in ('.py','.pyd','.dll'))
    paths.extend((Path(np.__file__).resolve().parent.parent/'numpy.libs').glob('*.dll'))
    paths.extend(compiler.parent.glob('*.dll'))
    paths.extend(compiler.parent.glob('*lld*.exe'))
    paths.extend(Path(module.__file__) for module in tuple(sys.modules.values())
                 if getattr(module, '__file__', None) and Path(module.__file__).is_file())
    paths.extend(ROOT/p for p in ('benchmarks/phase60/engine.c',
        'benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py','docs/research/RESEARCH_INDEX.md'))
    frozen = [B/'native_24_cost.py', B/'native_24_cost_audit.py', B/'native_24_cost_support.c',
              DOC/'NATIVE_24_COST_PROTOCOL_20261014.md']
    for p in frozen:
        blob = subprocess.check_output(['git','show', a.freeze+':'+p.relative_to(ROOT).as_posix()], cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n') == p.read_bytes().replace(b'\r\n',b'\n'), str(p)
    failure = DOC/'native_24_cost_binding_20261014.preparation_terminal.json'
    assert load(failure)['exit_code'] == 1 and not FAMILY.exists()
    paths.append(failure)
    for p in paths: assert p.is_file(), str(p)
    inputs = [extent(p) for p in dict.fromkeys(paths)]
    for e in [old['table'], old['compile']['dll']]+[x['file'] for x in old['exports']]:
        assert extent(e['path']) == e
    for rec in selected:
        for site in rec['sequence']['sites']:
            for key in ('trace', 'final'):
                e = site[key]
                assert all(extent(e['path'])[k] == e[k] for k in ('path','bytes','sha256'))
    assert load(DOC/'ffn_active_router_session_closure_20261014.json')['decision'] == 'SOURCE_ACTIVE_ROUTING_ASSAY_QUALIFIED'
    FAMILY.mkdir()
    support = FAMILY/'cost_support.dll'; log = FAMILY/'compile.log'
    command = [str(compiler),'-O3','-mavx2','-shared',str(B/'native_24_cost_support.c'),'-o',str(support)]
    co = subprocess.run(command,capture_output=True,timeout=45,creationflags=0x08000000)
    log.write_bytes(co.stdout+co.stderr); assert co.returncode == 0
    inputs.extend([extent(support),extent(log)])
    binding = dict(schema='NATIVE_24_COST_BINDING_V2', freeze=a.freeze, python=str(Path(sys.executable).resolve()),
        preparation_repair=dict(original_receipt=extent(failure),
            reason='Removed nonexistent root engine.c pin; canonical engine is benchmarks/phase60/engine.c. '
                   'Preflight all paths before hashing. No compilation, family creation or timings in failed preparation.'),
        versions=dict(numpy=np.__version__,psutil=psutil.__version__), compile=old['compile'],
        support=extent(support), support_compile=dict(command=command,log=extent(log)),
        table=old['table'], exports=old['exports'], records=selected, inputs=inputs,
        affinity=[0], modes=['resident','parameter_evicted'], samples_per_record=16,
        output_directory=str(FAMILY/'run'), limits=dict(seconds=1800,OS_bytes=2<<30,output_bytes=512<<20),
        parent_report=extent(rp), parent_audit=extent(ap), parent_closure=extent(cp), scope=SCOPE,
        DRAM='UNRESOLVED: cache-state intervention only; no hardware traffic counters')
    check_pins(binding, lambda:None)
    write(a.out,binding)
    print(json.dumps(dict(binding=extent(a.out), selections=[{k:q[k] for k in ('id','history','positions')} for q in selected])),flush=True)


def parameter_sets(b,np):
    result = []
    for export in b['exports']:
        with np.load(export['file']['path'],allow_pickle=False) as z:
            p = {k:np.ascontiguousarray(z[k]) for k in z.files}
        assert set(p) == set(PARAM_KEYS)
        assert p['q'].shape == (2048,3072) and p['q'].dtype == np.dtype('i2')
        assert all(v.flags.c_contiguous and np.isfinite(v).all() for v in p.values())
        result.append((p,ptrs([p[k] for k in PARAM_KEYS[:5]]),ptrs([p[k] for k in ('cw','cb','db')])))
    assert len(result) == 24
    return result


def record_rows(b,rec,np):
    tags = np.asarray(rec['sequence']['consumed_ids'],dtype='u4')
    table = np.memmap(b['table']['path'],'u2','r',shape=(65537,6704))
    inputs, outputs = [], []
    for site,item in enumerate(rec['sequence']['sites']):
        assert item['site'] == site
        width = 2048 if site == 0 else 10800
        trace = np.memmap(item['trace']['path'],'u2','r',shape=(rec['history'],width))
        inputs.append(np.ascontiguousarray(table[tags]) if site == 0 else np.ascontiguousarray(trace[:,2048:8752]))
        outputs.append(np.array(trace if site == 0 else trace[:,8752:],copy=True))
    return tags,inputs,outputs


def snap(snapshot,ctx,np):
    state=np.empty((3072,96),'f4'); ring=np.empty((3,3584),'u2'); seen=np.zeros(1,'u8')
    assert snapshot(ctx,state.ctypes.data,ring.ctypes.data,seen.ctypes.data) == 0
    return dict(state=state,ring=ring,seen=seen)


def run(a):
    import numpy as np
    import psutil
    b=load(a.binding); assert extent(a.binding)['sha256'] == a.binding_sha
    assert Path(sys.executable).resolve() == Path(b['python'])
    assert dict(numpy=np.__version__,psutil=psutil.__version__) == b['versions']
    assert not a.out.exists() and not Path(b['output_directory']).exists()
    proc=psutil.Process(); proc.cpu_affinity(b['affinity']); start=time.monotonic()
    guard=guard_factory(b,start,proc); check_pins(b,guard)
    # A distinct research worker would confound this feasibility screen.
    for peer in psutil.process_iter(['pid','name','cmdline']):
        try:
            if peer.pid != proc.pid and (peer.info['name'] or '').lower().startswith('python'):
                assert not any('native_expert_scaling' in x for x in (peer.info['cmdline'] or [])), peer.info
        except (psutil.AccessDenied,psutil.NoSuchProcess): pass
    directory=Path(b['output_directory']); directory.mkdir()
    dll,create,destroy,tokens,rows,snapshot=interface(b)
    last=dll.ts_last; last.argtypes=[ctypes.c_void_p];last.restype=ctypes.c_uint32
    support=ctypes.CDLL(b['support']['path']); evict=support.cost_evict
    evict.argtypes=[ctypes.c_void_p,ctypes.c_size_t];evict.restype=None
    noop=support.cost_noop;noop.argtypes=rows.argtypes;noop.restype=ctypes.c_int
    parameters=parameter_sets(b,np)
    inventory=[dict(site=i, arrays=[dict(name=k, address=p[k].ctypes.data,bytes=p[k].nbytes,
        shape=list(p[k].shape),dtype=str(p[k].dtype)) for k in PARAM_KEYS]) for i,(p,w,c) in enumerate(parameters)]
    cpu_before=[p._asdict() for p in psutil.cpu_times(percpu=True)]
    freq=psutil.cpu_freq(); records=[];samples=[];overhead=[];contexts=[];prefix_ns=0
    try:
        for mode in b['modes']:
            for rec in b['records']:
                guard();tags,input_rows,expected=record_rows(b,rec,np);n=rec['history'];lo=n-16
                outputs=[arrays(16,np) for _ in range(24)]; contexts=[create() for _ in range(24)]
                assert all(contexts)
                prefix_begin=time.perf_counter_ns()
                for site,(p,w,c) in enumerate(parameters):
                    for first in range(0,lo,64):
                        guard();hi=min(first+64,lo);z=arrays(hi-first,np)
                        assert rows(contexts[site],hi-first,tags[first:hi].ctypes.data,
                            input_rows[site][first:hi].ctypes.data,w,c,p['q'].ctypes.data,p['scale'].ctypes.data,
                            *arguments(z,0,hi-first,np)) == 0
                        assert np.array_equal(z['output'],expected[site][first:hi])
                prefix_elapsed=time.perf_counter_ns()-prefix_begin;prefix_ns+=prefix_elapsed
                calls=[]
                for t in range(16):
                    position=lo+t
                    calls.append([(contexts[i],1,tags[position:position+1].ctypes.data,
                        input_rows[i][position:position+1].ctypes.data,w,c,p['q'].ctypes.data,
                        p['scale'].ctypes.data,*arguments(outputs[i],t,t+1,np))
                        for i,(p,w,c) in enumerate(parameters)])
                # Real arguments and owners remain alive; no state change in this control.
                if not overhead:
                    for repeat in range(128):
                        begin=time.perf_counter_ns()
                        for call in calls[0]:
                            before=time.perf_counter_ns();rc=noop(*call);after=time.perf_counter_ns()
                            assert rc == 0
                        overhead.append(time.perf_counter_ns()-begin)
                for t,call_set in enumerate(calls):
                    guard();flush_begin=time.perf_counter_ns()
                    if mode == 'parameter_evicted':
                        for p,w,c in parameters:
                            for k in PARAM_KEYS:evict(p[k].ctypes.data,p[k].nbytes)
                    flush_elapsed=time.perf_counter_ns()-flush_begin if mode == 'parameter_evicted' else 0
                    times=[];returns=[];whole_begin=time.perf_counter_ns()
                    for call in call_set:
                        before=time.perf_counter_ns();rc=rows(*call);after=time.perf_counter_ns()
                        times.append(after-before);returns.append(rc)
                    whole_elapsed=time.perf_counter_ns()-whole_begin
                    assert returns == [0]*24
                    samples.append(dict(mode=mode,id=rec['id'],split=rec['split'],length_kind=rec['length_kind'],
                        position=lo+t,site_ns=times,whole_ns=whole_elapsed,flush_ns=flush_elapsed))
                    for i,z in enumerate(outputs):assert np.array_equal(z['output'][t],expected[i][lo+t])
                files=[]
                for i,ctx in enumerate(list(contexts)):
                    final=snap(snapshot,ctx,np); final['last']=np.array([last(ctx)],'u4')
                    with np.load(rec['sequence']['sites'][i]['final']['path'],allow_pickle=False) as z:
                        assert all(np.array_equal(v.view('u1'),z[k].view('u1')) for k,v in final.items())
                    path=directory/f'{mode}.{rec["id"]}.site{i:02d}.npz'
                    np.savez(path,**outputs[i],**final);files.append(dict(site=i,file=extent(path)))
                    destroy(ctx)
                    contexts.remove(ctx)
                contexts=[]
                records.append(dict(mode=mode,id=rec['id'],prefix_positions=lo,prefix_ns=prefix_elapsed,files=files))
                print(json.dumps(dict(mode=mode,id=rec['id'],samples=len(samples),seconds=time.monotonic()-start)),flush=True)
        check_pins(b,guard);guard();s=summaries(samples,np)
        nbytes=sum(x['bytes'] for layer in inventory for x in layer['arrays'])
        result=dict(schema='NATIVE_24_COST_RESULT_V1',binding=extent(a.binding),records=records,samples=samples,
            summary=s,noop_whole_ns=overhead,noop_summary=distribution(overhead,np),prefix_total_ns=prefix_ns,
            parameter_inventory=inventory,parameter_payload_bytes=nbytes,
            state_payload_bytes=24*3072*96*4,ring_payload_bytes=24*3*3584*2,
            context_struct_bytes=24*(3072*96*4+3*3584*2+16),
            input_row_buffer_bytes=sum(v.nbytes for v in input_rows),
            expected_output_buffer_bytes=sum(v.nbytes for v in expected),
            timing_output_buffer_bytes=sum(v.nbytes for z in outputs for v in z.values()),
            operand_ledger=dict(per_site_parameter_payload_bytes=nbytes//24,
                readout_weight_bytes=2048*3072*2,readout_scale_payload_bytes=2048*8,
                readout_input_payload_bytes=3072*2,readout_input_logical_reads_bytes=2048*3072*2,
                state_read_plus_write_bytes=2*3072*96*4,input_row_bytes=6704*2,
                note='Algorithmic operands; input re-read for each output row, cache-line reuse unspecified. Not DRAM.'),
            cpu_times_before=cpu_before,cpu_times_after=[p._asdict() for p in psutil.cpu_times(percpu=True)],
            cpu_frequency=None if freq is None else freq._asdict(),shared_desktop=True,
            affinity=proc.cpu_affinity(),kernel_threads=1,DRAM=b['DRAM'],
            numeric_checks='All prefix/continuation BF and final snapshots exact to qualified bridge.',
            decision=('RECURRENT_MEDIAN_EXHAUSTS_20MS' if s['resident']['whole_ns']['median'] >= 20_000_000
                      else 'RECURRENT_MEDIAN_LEAVES_PARTIAL_BUDGET'),
            seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,scope=SCOPE,
            output_bytes=sum(p.stat().st_size for p in directory.iterdir()),GPU_calls=0,RESERVED_queries=0)
        write(a.out,result);print(json.dumps(dict(result=extent(a.out),summary=s,decision=result['decision'])),flush=True)
    finally:
        for ctx in contexts:destroy(ctx)


def close(a):
    import psutil
    r=load(a.report);ad=load(a.audit);b=load(r['binding']['path'])
    assert ad['decision'] == 'STORED_CONTINUATIONS_AND_ACCOUNTING_PASS' and ad['report'] == extent(a.report)
    assert ad['checked_site_records'] == 192 and ad['checked_timed_calls'] == 3072
    commands=[]
    for p,suffix in ((Path(r['binding']['path']),'.preparation_terminal.json'),
                     (a.report,'.terminal.json'),(a.audit,'.terminal.json')):
        t=load(p.with_suffix(suffix));assert t['exit_code'] == 0 and t['result'] == extent(p)
        if psutil.pid_exists(t['pid']):assert abs(psutil.Process(t['pid']).create_time()-t['creation_time']) > .001
        commands.append(dict(receipt=extent(p.with_suffix(suffix)),pid=t['pid'],creation_time=t['creation_time'],
                             matching_process_absent=True))
    check_pins(b,lambda:None)
    files=[x['file'] for rec in r['records'] for x in rec['files']]
    for e in files:assert extent(e['path']) == e
    actual={str(p.resolve()):p.stat().st_size for p in Path(b['output_directory']).iterdir()}
    assert actual == {e['path']:e['bytes'] for e in files}
    write(a.out,dict(schema='NATIVE_24_COST_CLOSURE_V1',report=extent(a.report),audit=extent(a.audit),
        binding=r['binding'],commands=commands,family_files=len(files)+2,
        family_bytes=sum(p.stat().st_size for p in FAMILY.rglob('*') if p.is_file()),
        all_inputs_outputs_exact=True,decision='RECURRENT_COST_SCREEN_QUALIFIED',goal_status='ACTIVE_INCOMPLETE',scope=SCOPE))
    print(json.dumps(dict(closure=extent(a.out))),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','close'])
    for name in ('binding','out','report','audit'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--binding-sha');parser.add_argument('--freeze');a=parser.parse_args()
    import psutil
    proc=psutil.Process();start=time.monotonic()
    receipt=dict(schema='NATIVE_24_COST_TERMINAL_V1',phase=a.phase,command=sys.orig_argv,
                 pid=proc.pid,creation_time=proc.create_time())
    try:
        assert a.out is not None and not a.out.exists()
        {'prepare':prepare,'run':run,'close':close}[a.phase](a)
        assert time.monotonic()-start <= 1800 and proc.memory_info().peak_wset <= 2<<30
        receipt.update(exit_code=0,error=None,result=extent(a.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.preparation_terminal.json' if a.phase == 'prepare' else '.terminal.json'),receipt)
