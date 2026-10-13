"""Per-token full-source importance oracle; deliberately not a cheap router."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OPENBLAS_NUM_THREADS'] = '6'
ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader
COUNTS = [32,128,512,2048,4608]


def prepare(args):
    import numpy as np
    import psutil
    assert not args.out.exists() and not args.directory.exists()
    rp,ap,cp = [DOC/(name+'_20261013.json') for name in (
        'ffn_exact_subset_result','ffn_exact_subset_adjudication_v2','ffn_exact_subset_session_closure_v2')]
    r,a,c = [json.loads(p.read_bytes()) for p in (rp,ap,cp)]
    assert a['decision'] == 'INDEPENDENT_STORED_PASS' and a['report'] == extent(rp) and a['all_numeric_pass']
    assert c['report'] == extent(rp) and c['audit'] == extent(ap) and c['all_inputs_and_outputs_hash_verified']
    old = json.loads(Path(r['binding']['path']).read_bytes())
    pins = old['inputs']+[r['binding']]+[x['file'] for x in r['models']+r['records']]
    pins += [extent(p) for p in (rp,ap,cp,Path(__file__),B/'ffn_remainder_oracle_audit.py',
                B/'ffn_remainder_oracle_close.py',DOC/'FFN_REMAINDER_ORACLE_PROTOCOL_20261013.md')]
    unique = {}
    for e in pins:
        if e['path'] in unique: assert unique[e['path']] == e
        unique[e['path']] = e
    for e in unique.values(): assert extent(e['path']) == e,e['path']
    n = sum(x['history'] for x in old['records']); assert n == 22547
    payload = n*3*5*(2048*4+4+8)+n*3*sum(COUNTS[:-1])*2
    assert payload == 3142600860
    binding = dict(schema='FFN_REMAINDER_ORACLE_BINDING_V1',freeze=args.freeze,
        inputs=list(unique.values()),parent_report=extent(rp),parent_audit=extent(ap),parent_closure=extent(cp),
        source=old['source'],records=old['records'],sites=old['sites'],models=old['models'],
        trajectories=old['trajectories'],counts=COUNTS,alpha=old['alpha'],beta=old['beta'],gates=old['gates'],
        versions=dict(numpy=np.__version__,psutil=psutil.__version__),
        output_directory=str(args.directory.resolve()),predicted_array_bytes=payload,
        limits=dict(seconds=1800,audit_seconds=1800,OS_bytes=8 << 30,output_bytes=3 << 30),
        score_audit_rtol=1e-10,score_audit_atol=1e-24,
        score_audit_metric='Frobenius score difference<=rtol*literal score norm+atol; '
                           'independent feature remainder difference/reference feature norm<=1e-10.',
        scope='Per-token energy heuristic with oracle access to ALL source neurons. '
              'Existing64 FIT cells,actual48 own histories/sites0,12,23/full2048 output. '
              'No learned cheap routing,optimal subset,sparsity impossibility,new '
              'whole model,physical ternary bank,DRAM,useful expert-n or CPU TPS admission.')
    write(args.out,binding); print(json.dumps(dict(binding=extent(args.out),predicted_array_bytes=payload)),flush=True)


def run(args):
    import numpy as np
    import psutil
    start = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    assert extent(args.binding)['sha256'] == args.binding_sha
    b = json.loads(args.binding.read_bytes()); assert b['freeze'] == args.freeze
    assert not args.out.exists() and not args.directory.exists() and str(args.directory.resolve()) == b['output_directory']
    assert b['versions'] == dict(numpy=np.__version__,psutil=psutil.__version__)
    output_bytes = 0
    def guard():
        assert time.monotonic()-start <= b['limits']['seconds'] and proc.memory_info().peak_wset <= b['limits']['OS_bytes']
        assert not proc.children(recursive=True) and output_bytes <= b['limits']['output_bytes']
    def check():
        for e in b['inputs']: guard(); assert extent(e['path']) == e,e['path']
    def dec(e):
        x = np.memmap(e['path'],'u2','r',shape=tuple(e['shape']))
        return (x.astype('u4') << 16).view('f4').astype('f8')
    def rel(x,y): return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    check(); args.directory.mkdir(); get,stream = tensor_reader(b['source'],np)
    fixed = json.loads(Path(b['parent_report']['path']).read_bytes()); records,controls = [],[]
    try:
        for index,site in enumerate(b['sites']):
            model = next(m for m in b['models'] if m['site'] == site)
            with np.load(model['file']['path'],allow_pickle=False) as z: m = {k:z[k].copy() for k in z.files}
            U,G,D = [get(f'model.layers.{site}.feed_forward.{name}.weight') for name in ('up_proj','gate_proj','down_proj')]
            energy = np.sum((b['beta']*D)**2,axis=0)
            for rec in b['records']:
                guard(); X = dec(rec['fields'][index]['x']); actual = dec(rec['fields'][index]['output'])
                old_route = next(x for x in b['trajectories'] if x['site'] == site and x['id'] == rec['id'])
                with np.load(old_route['file']['path'],allow_pickle=False) as z: labels = z['labels'].copy()
                u,g = X@U.T,(X@G.T)*b['alpha']; sx = g*np.exp(-np.logaddexp(0,-g))
                uc,gc,s,derivative = [m[k][labels] for k in ('u_center','g_center','s','derivative')]
                af = s*u+uc*derivative*(g-gc)
                rem = uc*(sx-s-derivative*(g-gc))+(u-uc)*(sx-s)
                affine,proxy = af@D.T*b['beta'],(u*sx)@D.T*b['beta']
                score = rem**2*energy
                assert np.isfinite(score).all() and (score >= 0).all()
                ranking = np.argsort(-score,axis=1,kind='stable')
                control = rel(proxy,actual); assert control <= b['gates']['numerical_proxy_each_case_relative']
                controls.append(dict(site=site,id=rec['id'],error=control,pass_gate=True))
                for count in COUNTS:
                    guard(); mask = np.zeros_like(rem)
                    if count < 4608:
                        selected = ranking[:,:count].astype('u2')
                        np.put_along_axis(mask,selected,1,axis=1)
                    else:
                        selected = np.empty((len(X),0),'u2'); mask.fill(1)
                    prediction = affine+(rem*mask)@D.T*b['beta']
                    ep,eb = rel(prediction,proxy),rel(prediction,actual)
                    assert np.isfinite(ep) and np.isfinite(eb)
                    if count == 4608: assert ep <= 1e-10
                    previous = next(x for x in fixed['records'] if x['site'] == site and x['id'] == rec['id'] and x['count'] == count)
                    path = args.directory/f'site{site:02d}.r{count:04d}.{rec["id"]}.npz'
                    # Cap checked before allocation/writing and exact file extent added once.
                    required = len(X)*(2048*4+4+8+(count*2 if count < 4608 else 0))+4096
                    assert output_bytes+required <= b['limits']['output_bytes']
                    np.savez(path,labels=labels.astype('u4'),mass=np.ones(len(X),'f8'),selected=selected,prediction=prediction.astype('f4'))
                    file = extent(path); assert file['bytes'] <= required
                    output_bytes += file['bytes']; guard()
                    records.append(dict(site=site,count=count,id=rec['id'],split=rec['split'],domain=rec['domain'],
                        history=len(X),file=file,error_to_proxy=ep,error_to_BF=eb,
                        fixed_subset_error_to_BF=previous['error_to_BF'],control_only=count == 4608))
            print(json.dumps(dict(site=site,records=len(records),seconds=time.monotonic()-start)),flush=True)
        summary = []
        for site in b['sites']:
            for count in COUNTS:
                group = [x for x in records if x['site'] == site and x['count'] == count]
                fi,dv = [dict(case_mean=sum(x['error_to_BF'] for x in group if x['split'] == split)/24,
                              worst=max(x['error_to_BF'] for x in group if x['split'] == split),
                              fixed_case_mean=sum(x['fixed_subset_error_to_BF'] for x in group if x['split'] == split)/24)
                         for split in ('FIT','DEV')]
                domains = {d:sum(x['error_to_BF'] for x in group if x['split'] == 'DEV' and x['domain'] == d)/
                           sum(x['split'] == 'DEV' and x['domain'] == d for x in group)
                           for d in sorted({x['domain'] for x in group if x['split'] == 'DEV'})}
                summary.append(dict(site=site,count=count,FIT=fi,DEV=dv,DEV_domains=domains,control_only=count == 4608))
        check(); guard(); assert len(records) == 720 and len(controls) == 144
        actual_files = list(args.directory.iterdir()); assert len(actual_files) == 720
        assert sum(p.stat().st_size for p in actual_files) == output_bytes
        result = dict(schema='FFN_REMAINDER_ORACLE_RESULT_V1',binding=extent(args.binding),freeze=args.freeze,
            records=records,controls=controls,summary=summary,output_bytes=output_bytes,
            before_after_hashes_exact=True,seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            source_model_forwards=0,GPU_calls=0,RESERVED_queries=0,optimizer_updates=0,
            decision='COMPLETE_PENDING_INDEPENDENT_AUDIT_NO_CANDIDATE_ADMISSION',scope=b['scope'])
        write(args.out,result); print(json.dumps(dict(result=extent(args.out),summary=summary)),flush=True)
    finally: stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--run',action='store_true')
    for key in ('binding','out','directory'): p.add_argument('--'+key,type=Path)
    p.add_argument('--binding-sha'); p.add_argument('--freeze',required=True); args = p.parse_args()
    import psutil
    start = time.monotonic(); proc = psutil.Process()
    receipt = dict(schema='FFN_REMAINDER_ORACLE_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,
                   creation_time=proc.create_time(),freeze=args.freeze,preparation=args.prepare)
    try:
        if args.prepare: prepare(args)
        else: run(args)
        receipt.update(exit_code=0,error=None,result=extent(args.out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(args.out.with_suffix('.preparation_terminal.json' if args.prepare else '.terminal.json'),receipt)
