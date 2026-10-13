"""FIT-only exact SwiGLU remainder subsets at already qualified affine cells."""
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
    paths = [DOC/'ffn_affine_pilot_result_20261013.json',DOC/'ffn_affine_pilot_adjudication_20261013.json',
             DOC/'ffn_affine_defect_session_closure_20261013.json']
    r,a,c = [json.loads(p.read_bytes()) for p in paths]
    assert a['report'] == extent(paths[0]) and a['all_numeric_pass'] and a['proxy_controls_pass']
    assert c['decision'] == 'AFFINE_1_16_64_REJECTED_NONLINEAR_DEFECT_QUALIFIED'
    assert c['reports'][:2] == [extent(p) for p in paths[:2]]
    old = json.loads(Path(r['binding']['path']).read_bytes())
    models = [m for m in r['models'] if m['count'] == 64]
    trajectories = [x for x in r['records'] if x['count'] == 64]
    inputs = old['inputs']+[r['binding']]+[m['file'] for m in models]+[x['file'] for x in trajectories]
    inputs += [extent(p) for p in paths+[Path(__file__),B/'ffn_exact_subset_audit.py',B/'ffn_exact_subset_close.py',
               DOC/'FFN_EXACT_SUBSET_PROTOCOL_20261013.md']]
    unique = {}
    for e in inputs:
        if e['path'] in unique: assert unique[e['path']] == e
        unique[e['path']] = e
    for e in unique.values(): assert extent(e['path']) == e,e['path']
    n = sum(x['history'] for x in old['records']); assert n == 22547
    array_bytes = n*3*5*(2048*4+4+8)+3*(64*4608*(8+4)+64*8+4608*8)
    assert array_bytes == 2785362780
    binding = dict(schema='FFN_EXACT_SUBSET_BINDING_V1',freeze=args.freeze,
        inputs=list(unique.values()),source=old['source'],parent_report=extent(paths[0]),
        records=old['records'],sites=old['sites'],models=models,trajectories=trajectories,counts=COUNTS,
        alpha=old['alpha_effective_F32'],beta=old['beta_effective_F32'],gates=old['gates'],
        versions=dict(numpy=np.__version__,psutil=psutil.__version__),
        output_directory=str(args.directory.resolve()),predicted_array_bytes=array_bytes,
        limits=dict(seconds=1800,audit_seconds=1800,OS_bytes=8 << 30,output_bytes=3 << 30),
        selection='Per-cell token-weighted FIT mean squared combined exact feature remainder '
                  'times beta-squared down-column norm squared;descending stable source index. '
                  'Empty FIT cells use global token-weighted FIT scores. No DEV selection.',
        scope='All48 actual own histories/three sites/existing64 cells/full2048 outputs. '
              'Implicit affine plus exact selected source neurons. No physical compiled '
              'bank,ternary quantization,scalable routing,whole model,GPU,RESERVED or TPS claim.')
    write(args.out,binding); print(json.dumps(dict(binding=extent(args.out),predicted_array_bytes=array_bytes)),flush=True)


def run(args):
    import numpy as np
    import psutil
    start = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    assert extent(args.binding)['sha256'] == args.binding_sha
    b = json.loads(args.binding.read_bytes())
    assert b['freeze'] == args.freeze and not args.directory.exists() and not args.out.exists()
    assert str(args.directory.resolve()) == b['output_directory']
    assert b['versions'] == dict(numpy=np.__version__,psutil=psutil.__version__)
    def guard():
        assert time.monotonic()-start <= b['limits']['seconds'] and proc.memory_info().peak_wset <= b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
        if args.directory.exists(): assert sum(p.stat().st_size for p in args.directory.iterdir()) <= b['limits']['output_bytes']
    def check():
        for e in b['inputs']: guard(); assert extent(e['path']) == e,e['path']
    def dec(e):
        x = np.memmap(e['path'],'u2','r',shape=tuple(e['shape']))
        return (x.astype('u4') << 16).view('f4').astype('f8')
    def rel(x,y): return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def sigmoid(x): return np.exp(-np.logaddexp(0,-x))
    check(); args.directory.mkdir(); get,stream = tensor_reader(b['source'],np)
    models,records,controls = [],[],[]
    try:
        for index,site in enumerate(b['sites']):
            pm = next(m for m in b['models'] if m['site'] == site)
            with np.load(pm['file']['path'],allow_pickle=False) as z: m = {k:z[k].copy() for k in z.files}
            U,G,D = [get(f'model.layers.{site}.feed_forward.{name}.weight') for name in ('up_proj','gate_proj','down_proj')]
            column_energy = np.sum((b['beta']*D)**2,axis=0)
            def compute(rec):
                X = dec(rec['fields'][index]['x'])
                saved = next(x for x in b['trajectories'] if x['site'] == site and x['id'] == rec['id'])
                with np.load(saved['file']['path'],allow_pickle=False) as z: labels = z['labels'].copy()
                u,g = X@U.T,(X@G.T)*b['alpha']; sx = g*sigmoid(g)
                uc,gc,s,derivative = [m[k][labels] for k in ('u_center','g_center','s','derivative')]
                af = s*u+uc*derivative*(g-gc)
                remainder = uc*(sx-s-derivative*(g-gc))+(u-uc)*(sx-s)
                return labels,af,remainder,u*sx
            sums = np.zeros((64,4608),'f8'); counts = np.zeros(64,'i8')
            for rec in b['records']:
                if rec['split'] != 'FIT': continue
                guard(); labels,af,remainder,feature = compute(rec)
                np.add.at(sums,labels,remainder**2); counts += np.bincount(labels,minlength=64)
            assert counts.sum() == sum(x['history'] for x in b['records'] if x['split'] == 'FIT')
            global_score = sums.sum(axis=0)/counts.sum()*column_energy
            scores = np.tile(global_score,(64,1)); occupied = counts > 0
            scores[occupied] = sums[occupied]/counts[occupied,None]*column_energy
            assert np.isfinite(scores).all() and (scores >= 0).all()
            ranking = np.argsort(-scores,axis=1,kind='stable').astype('i4')
            mp = args.directory/f'site{site:02d}.model.npz'
            np.savez(mp,scores=scores,ranking=ranking,FIT_counts=counts,column_energy=column_energy)
            model = dict(site=site,file=extent(mp),empty_FIT_cells=int((counts == 0).sum()))
            for rec in b['records']:
                guard(); labels,af,remainder,feature = compute(rec)
                affine,proxy = af@D.T*b['beta'],feature@D.T*b['beta']
                actual = dec(rec['fields'][index]['output']); control = rel(proxy,actual)
                assert control <= b['gates']['numerical_proxy_each_case_relative']
                controls.append(dict(site=site,id=rec['id'],error=control,pass_gate=True))
                for count in COUNTS:
                    guard(); prediction = affine.copy()
                    for cell in np.unique(labels):
                        rows = np.flatnonzero(labels == cell); selected = ranking[cell,:count]
                        prediction[rows] += remainder[np.ix_(rows,selected)]@D[:,selected].T*b['beta']
                    identity = rel(prediction,proxy)
                    if count == 4608: assert identity <= 1e-10
                    path = args.directory/f'site{site:02d}.r{count:04d}.{rec["id"]}.npz'
                    np.savez(path,labels=labels.astype('u4'),mass=np.ones(len(labels),'f8'),prediction=prediction.astype('f4'))
                    records.append(dict(site=site,count=count,id=rec['id'],split=rec['split'],domain=rec['domain'],
                        history=len(labels),file=extent(path),error_to_proxy=identity,error_to_BF=rel(prediction,actual)))
            summaries = []
            selected_count = None
            for count in COUNTS:
                group = [x for x in records if x['site'] == site and x['count'] == count]
                fi,dv = [dict(case_mean=sum(x['error_to_BF'] for x in group if x['split'] == split)/24,
                              worst=max(x['error_to_BF'] for x in group if x['split'] == split)) for split in ('FIT','DEV')]
                domains = {d:sum(x['error_to_BF'] for x in group if x['split'] == 'DEV' and x['domain'] == d)/
                           sum(x['split'] == 'DEV' and x['domain'] == d for x in group)
                           for d in sorted({x['domain'] for x in group if x['split'] == 'DEV'})}
                fit_pass = fi['case_mean'] <= .10 and fi['worst'] <= .20
                dev_pass = dv['case_mean'] <= .10 and dv['worst'] <= .20 and max(domains.values()) <= .20
                # All-neuron identity control cannot be admitted as a reduced-cost candidate.
                if count < 4608 and fit_pass and selected_count is None: selected_count = count
                summaries.append(dict(count=count,FIT=fi,DEV=dv,DEV_domains=domains,FIT_eligible=fit_pass,
                                      DEV_eligible=dev_pass,control_only=count == 4608))
            model.update(summaries=summaries,FIT_selected_count=selected_count); models.append(model)
            print(json.dumps(dict(site=site,selected=selected_count,summaries=summaries,seconds=time.monotonic()-start)),flush=True)
        check(); guard(); assert len(records) == 720 and len(controls) == 144
        result = dict(schema='FFN_EXACT_SUBSET_RESULT_V1',binding=extent(args.binding),freeze=args.freeze,
            records=records,models=models,controls=controls,before_after_hashes_exact=True,
            source_model_forwards=0,GPU_calls=0,RESERVED_queries=0,optimizer_updates=0,
            seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            output_bytes=sum(p.stat().st_size for p in args.directory.iterdir()),
            decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',scope=b['scope'])
        write(args.out,result); print(json.dumps(dict(result=extent(args.out))),flush=True)
    finally: stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--run',action='store_true')
    for key in ('binding','out','directory'): p.add_argument('--'+key,type=Path)
    p.add_argument('--binding-sha'); p.add_argument('--freeze',required=True)
    args = p.parse_args()
    import psutil
    start = time.monotonic(); proc = psutil.Process()
    receipt = dict(schema='FFN_EXACT_SUBSET_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,
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
