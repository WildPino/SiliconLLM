"""Bound CPU scheduling-only assay, full stored audit, and explicit closure."""
import argparse
import ctypes
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['KMP_BLOCKTIME'] = '0'
ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(B), str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent, write
import contextual_generator_cost as cg
from native_24_cost import parameter_sets

STEM = 'physical_worker_organs_20261014'
FAMILY = Path('E:/SiliconLLM/results/native_expert_scaling')/STEM
ARMS = [1, 3, 6]
ORDERS = list(itertools.permutations(ARMS))
SCOPE = ('Supplied actual own-history inputs; sequential23 source BF generators and24 original '
         'integer readouts only. Physical-worker scheduling changes, identical arithmetic. '
         'Excludes convolution/state/norm/input quantization/FFN/attention/router/head/frontend; '
         'no joined decoder, new quality, useful50, useful-n, DRAM or family-scale admission.')


def load(p):
    return json.loads(Path(p).read_bytes())


def exact(x, y, np):
    assert x.dtype == y.dtype and x.shape == y.shape
    assert x.tobytes() == y.tobytes(), 'numeric bit mismatch'


def check(pins, guard):
    for e in pins:
        guard()
        assert extent(e['path']) == e, e['path']


def distribution(values, np):
    v = np.asarray(values, 'f8')
    assert v.size and np.isfinite(v).all() and (v >= 0).all()
    return dict(count=int(v.size), minimum=float(v.min()), median=float(np.median(v)),
                p90=float(np.quantile(v, .9)), maximum=float(v.max()))


def summary(samples, np):
    result = {}
    for mode in ('resident', 'parameter_evicted'):
        result[mode] = {}
        for arm in ARMS:
            rows = [q for q in samples if q['mode'] == mode and q['workers'] == arm]
            result[mode][str(arm)] = {
                key: distribution([q[key] for q in rows], np)
                for key in ('whole_ns', 'generator_ns', 'readout_ns', 'flush_ns')}
            result[mode][str(arm)]['organ_only_remaining_20ms_ns'] = (
                20_000_000-result[mode][str(arm)]['whole_ns']['median'])
    return result


def reference(rec, mode, rb, np):
    row = next(q for q in rb['reference_records'] if q['id'] == rec['id'] and q['mode'] == mode)
    result = {k: [] for k in ('r', 'a', 'dots', 'raw', 'output')}
    assert len(row['files']) == 24
    for site, item in enumerate(row['files']):
        assert item['site'] == site
        with np.load(item['file']['path'], allow_pickle=False) as z:
            for k in result:
                result[k].append(z[k].copy())
    return {k: np.ascontiguousarray(v) for k, v in result.items()}


def outputs(np):
    return dict(g_raw=np.empty((3,23,16,6704), 'f4'),
                g_gg=np.empty((3,23,16,6704), 'u2'),
                g_fields=np.empty((3,23,16,6704), 'u2'),
                r_dots=np.empty((3,24,16,2048), 'i8'),
                r_raw=np.empty((3,24,16,2048), 'f8'),
                r_bits=np.empty((3,24,16,2048), 'u2'))


def function(dll, name, args, result=ctypes.c_int):
    f = getattr(dll, name)
    f.argtypes = args
    f.restype = result
    return f


def run(a, start, proc, guard):
    import numpy as np
    import psutil
    assert subprocess.check_output(['git','branch','--show-current'], cwd=ROOT, text=True).strip() == 'research/native-expert-scaling'
    assert not FAMILY.exists()
    frozen = [Path(__file__), B/'physical_worker_organs.c', B/'physical_worker_organs_audit.py',
              DOC/'PHYSICAL_WORKER_ORGANS_PROTOCOL_20261014.md']
    for p in frozen:
        blob = subprocess.check_output(['git','show',a.freeze+':'+p.relative_to(ROOT).as_posix()], cwd=ROOT)
        assert blob.replace(b'\r\n', b'\n') == p.read_bytes().replace(b'\r\n', b'\n')
    for peer in psutil.process_iter(['pid','name','cmdline']):
        try:
            if peer.pid != proc.pid and (peer.info['name'] or '').lower().startswith('python'):
                assert not any('native_expert_scaling' in x for x in (peer.info['cmdline'] or [])), peer.info
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    gbp = DOC/'contextual_generator_cost_binding_20261014.json'
    rbp = DOC/'native_bf_activation_lut_binding_v2_20261014.json'
    gb, rb = load(gbp), load(rbp)
    parent_closures = [DOC/'contextual_generator_cost_session_closure_20261014.json',
                       DOC/'native_bf_activation_lut_session_closure_20261014.json']
    for p in parent_closures:
        cl = load(p)
        for key in ('report', 'audit'):
            assert extent(cl[key]['path']) == cl[key]
    assert [(r['id'],r['positions']) for r in gb['records']] == [(r['id'],r['positions']) for r in rb['records']]
    compiler = Path(gb['compile']['command'][0])
    paths = frozen+[gbp,rbp,*parent_closures, B/'contextual_generator_cost.c',
        B/'contextual_generator_cost.py', B/'native_24_cost.py', B/'integer_readout.c',
        B/'meth388_switch_thread_binding.h', B/'falcon_auxiliary_s6_transport.py',
        B/'row_state.py', B/'token_state.py', compiler, compiler.parent/'libomp.dll',
        Path(sys.executable), Path(sys.executable).parent/'python312.dll',
        Path(gb['source']['path']), Path(gb['compile']['dll']['path']),
        Path(rb['compile']['dll']['path']), Path(gb['support']['path'])]
    for rec in gb['records']:
        paths.extend(Path(e['trace']['path']) for e in rec['traces'])
    for e in rb['exports']:
        paths.append(Path(e['file']['path']))
    for rec in rb['reference_records']:
        paths.extend(Path(e['file']['path']) for e in rec['files'])
    for module in (np, psutil):
        paths.extend(p for p in Path(module.__file__).parent.rglob('*')
                     if p.is_file() and p.suffix in ('.py','.pyd','.dll'))
    paths.extend((Path(np.__file__).parent.parent/'numpy.libs').glob('*.dll'))
    paths.extend(compiler.parent.glob('*.dll'))
    paths.extend(ROOT/p for p in ('benchmarks/phase60/engine.c',
        'benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py', 'docs/research/RESEARCH_INDEX.md'))
    pins = []
    for p in dict.fromkeys(paths):
        guard(); pins.append(extent(p))
    known = [gb['source'],gb['compile']['dll'],rb['compile']['dll'],gb['support']]
    known += [e['file'] for e in rb['exports']]
    known += [e['trace'] for r in gb['records'] for e in r['traces']]
    known += [e['file'] for r in rb['reference_records'] for e in r['files']]
    by_path = {e['path']: e for e in pins}
    for e in known:
        assert all(by_path[e['path']][k] == e[k] for k in ('path','bytes','sha256'))
    FAMILY.mkdir()
    dll_path, log = FAMILY/'organs.dll', FAMILY/'compile.log'
    command = [str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-ffp-contract=off',
               '-fopenmp','-shared',str(B/'physical_worker_organs.c'),'-o',str(dll_path),'-lm']
    compiled = subprocess.run(command, capture_output=True, timeout=45, creationflags=0x08000000)
    log.write_bytes(compiled.stdout+compiled.stderr)
    assert compiled.returncode == 0, log.read_text(errors='replace')
    pins += [extent(dll_path), extent(log)]
    b = dict(schema='PHYSICAL_WORKER_ORGANS_BINDING_V1',freeze=a.freeze,inputs=pins,
        generator_binding=extent(gbp),readout_binding=extent(rbp),compile=dict(command=command,
        dll=extent(dll_path),log=extent(log)),arms=ARMS,orders=[list(x) for x in ORDERS],
        versions=dict(numpy=np.__version__,psutil=psutil.__version__),python=str(Path(sys.executable).resolve()),
        records=[{k:r[k] for k in ('id','split','length_kind','history','positions')} for r in gb['records']],
        modes=['resident','parameter_evicted'],limits=dict(seconds=3600,OS_bytes=4<<30,output_bytes=2<<30),
        runtime_options=dict(OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0',OPENBLAS_NUM_THREADS='1'),
        scope=SCOPE,DRAM='UNRESOLVED: eviction is a cache intervention, not a DRAM counter.')
    binding_path = DOC/(STEM+'_binding.json');assert not binding_path.exists();write(binding_path,b)
    print(json.dumps(dict(binding=extent(binding_path),phase='bound')), flush=True)
    runtime_directory = os.add_dll_directory(str(compiler.parent))
    dll = ctypes.CDLL(str(dll_path))
    setup = function(dll,'po_setup',[])
    choose = function(dll,'po_threads',[ctypes.c_int])
    team_check = function(dll,'po_team_check',[])
    binding = function(dll,'po_binding',[ctypes.c_int,ctypes.c_int],ctypes.c_uint64)
    generate = function(dll,'po_generator',[ctypes.c_void_p]*6)
    readout = function(dll,'po_readout',[ctypes.c_int]+[ctypes.c_void_p]*7)
    assert setup() == 0
    assert all(choose(n) == 1 for n in (0,2,4,7))
    assert all(choose(n) == 0 for n in ARMS)
    assert generate(None,None,None,None,None,None) == 1
    assert readout(0,None,None,None,None,None,None,None) == 1
    native_g, support = cg.native(gb)
    owners = cg.weights(gb,np)
    mup = np.asarray(gb['MuP_bits'], 'u2')
    parameters = parameter_sets(rb,np)
    initial_binding = [[int(binding(i,j)) for j in range(5)] for i in range(6)]
    assert len({x[0] for x in initial_binding}) == 6
    assert len({x[1] for x in initial_binding}) == 6 and all(x[2] == 0 and x[3] >= 6 for x in initial_binding)
    cpu_before = [q._asdict() for q in psutil.cpu_times(percpu=True)]
    samples, records = [], []
    for mode_index, mode in enumerate(b['modes']):
        for record_index, rec in enumerate(gb['records']):
            guard(); h, truth = cg.record_inputs(rec,np)
            ref = reference(rec,mode,rb,np); out = outputs(np)
            gcalls = [[[tuple(v.ctypes.data for v in (owners[i],h[i,t],mup,
                out['g_raw'][ai,i,t],out['g_gg'][ai,i,t],out['g_fields'][ai,i,t]))
                for i in range(23)] for t in range(16)] for ai in range(3)]
            rcalls = [[[(1,p['q'].ctypes.data,p['scale'].ctypes.data,
                ref['r'][i,t].ctypes.data,ref['a'][i,t:t+1].ctypes.data,
                out['r_dots'][ai,i,t].ctypes.data,out['r_raw'][ai,i,t].ctypes.data,
                out['r_bits'][ai,i,t].ctypes.data) for i,(p,_,_) in enumerate(parameters)]
                for t in range(16)] for ai in range(3)]
            for ai, arm in enumerate(ARMS):
                assert choose(arm) == 0
                for site in range(24):
                    if site: assert generate(*gcalls[ai][0][site-1]) == 0
                    assert readout(*rcalls[ai][0][site]) == 0
            for t in range(16):
                order = ORDERS[(t+record_index*16+mode_index*64)%6]
                for arm in order:
                    guard(); ai = ARMS.index(arm); assert choose(arm) == 0
                    fb = time.perf_counter_ns()
                    if mode == 'parameter_evicted':
                        for w in owners: support.cost_evict(w.ctypes.data,w.nbytes)
                        support.cost_evict(mup.ctypes.data,mup.nbytes)
                        for p,_,_ in parameters:
                            for k in ('q','scale'): support.cost_evict(p[k].ctypes.data,p[k].nbytes)
                    flush = time.perf_counter_ns()-fb if mode == 'parameter_evicted' else 0
                    gt, rt, returned = [], [], []
                    begin = time.perf_counter_ns()
                    for site in range(24):
                        if site:
                            before=time.perf_counter_ns(); rc=generate(*gcalls[ai][t][site-1]);after=time.perf_counter_ns()
                            gt.append(after-before);returned.append(rc)
                        before=time.perf_counter_ns();rc=readout(*rcalls[ai][t][site]);after=time.perf_counter_ns()
                        rt.append(after-before);returned.append(rc)
                    elapsed = time.perf_counter_ns()-begin
                    assert returned == [0]*47
                    samples.append(dict(mode=mode,id=rec['id'],position=rec['positions'][t],
                        workers=arm,order=list(order),whole_ns=elapsed,generator_ns=sum(gt),readout_ns=sum(rt),
                        generator_site_ns=gt,readout_site_ns=rt,flush_ns=flush))
            for key in ('g_raw','g_gg','g_fields'):
                for ai in (1,2): exact(out[key][ai],out[key][0],np)
            for ai in range(3):
                for dest, source in (('r_dots','dots'),('r_raw','raw'),('r_bits','output')):
                    exact(out[dest][ai],ref[source],np)
                for i in range(23):
                    assert max(cg.controls(out['g_fields'][ai,i],truth[i],np).values()) <= gb['gates']['field_relative']
            p = FAMILY/(mode+'.'+rec['id']+'.npz');np.savez(p,**out)
            records.append(dict(id=rec['id'],mode=mode,file=extent(p)))
            print(json.dumps(dict(mode=mode,id=rec['id'],sample_count=len(samples),seconds=time.monotonic()-start)),flush=True)
    assert choose(6) == 0 and team_check() == 0  # Untimed live team metadata, no model call.
    final_binding = [[int(binding(i,j)) for j in range(5)] for i in range(6)]
    assert all(x[1:4] == y[1:4] for x,y in zip(initial_binding,final_binding))
    r = dict(schema='PHYSICAL_WORKER_ORGANS_RESULT_V1',binding=extent(binding_path),records=records,
        samples=samples,summary=summary(samples,np),initial_worker_binding=initial_binding,
        final_worker_binding=final_binding,cpu_before=cpu_before,
        cpu_after=[q._asdict() for q in psutil.cpu_times(percpu=True)],
        process_affinity=proc.cpu_affinity(),negative_contracts_pass=True,FP_control=int(native_g.cg_control()),
        GPU_calls=0,RESERVED_queries=0,scope=SCOPE,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
    report_path = DOC/(STEM+'_result.json');assert not report_path.exists();write(report_path,r)
    from physical_worker_organs_audit import audit
    audited = audit(r,b,gb,rb,np,guard)
    audit_path = DOC/(STEM+'_audit.json');assert not audit_path.exists()
    write(audit_path,dict(report=extent(report_path),binding=extent(binding_path),**audited))
    check(pins,guard)
    for rec in records: assert extent(rec['file']['path']) == rec['file']
    assert extent(report_path) == load(audit_path)['report']
    expected = {e['path']:e['bytes'] for e in [extent(dll_path),extent(log)]+[q['file'] for q in records]}
    actual = {str(p.resolve()):p.stat().st_size for p in FAMILY.rglob('*') if p.is_file()}
    assert actual == expected
    closure_path = DOC/(STEM+'_closure.json');assert not closure_path.exists()
    write(closure_path,dict(schema='PHYSICAL_WORKER_ORGANS_CLOSURE_V1',report=extent(report_path),
        audit=extent(audit_path),binding=extent(binding_path),all_inputs_outputs_exact=True,
        files=len(actual),bytes=sum(actual.values()),decision='EXACT_SCHEDULING_SCREEN_CLOSED',
        scope=SCOPE,goal_status='ACTIVE_INCOMPLETE'))
    runtime_directory.close()
    return dict(binding=extent(binding_path),report=extent(report_path),audit=extent(audit_path),
                closure=extent(closure_path),summary=r['summary'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--freeze',required=True);args=parser.parse_args()
    import psutil
    process=psutil.Process();started=time.monotonic()
    output=DOC/(STEM+'_pipeline.json');terminal=output.with_suffix('.terminal.json')
    owned=output.with_suffix('.owned_process.json')
    assert not output.exists() and not terminal.exists() and not owned.exists()
    receipt=dict(command=sys.orig_argv,pid=process.pid,creation_time=process.create_time(),
        seconds_cap=3600,OS_bytes_cap=4<<30)
    write(owned,receipt)
    def guard():
        assert time.monotonic()-started <= 3600, 'CPU wall cap'
        assert process.memory_info().peak_wset <= 4<<30, 'OS cap'
        if FAMILY.exists(): assert sum(p.stat().st_size for p in FAMILY.rglob('*') if p.is_file()) <= 2<<30, 'output cap'
    try:
        result=run(args,started,process,guard);guard();write(output,result)
        receipt.update(exit_code=0,error=None,result=extent(output))
        print(json.dumps(dict(pipeline=extent(output),summary=result['summary'])),flush=True)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-started,OS_peak=process.memory_info().peak_wset)
        write(terminal,receipt)
