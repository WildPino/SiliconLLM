"""FIT input sketch chooses IDs; full source reference is diagnostic only."""
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
RANKS = [64,128,256,512]


def select_ids(delta,labels,center,sketch,rank,count,np):
    """ID path has no full U/G operators or full-source neuron activations."""
    z = delta@sketch['encoder'][:,:rank]
    u = center['u_center'][labels]+z@sketch['U_projected'][:,:rank].T
    g = center['g_center'][labels]+z@sketch['G_projected'][:,:rank].T
    sx = g*np.exp(-np.logaddexp(0,-g))
    uc,gc,s,d = [center[k][labels] for k in ('u_center','g_center','s','derivative')]
    rem = uc*(sx-s-d*(g-gc))+(u-uc)*(sx-s)
    scores = rem**2*sketch['column_energy']
    assert np.isfinite(scores).all()
    return np.argsort(-scores,axis=1,kind='stable')[:,:count].astype('u2')


def prepare(args):
    import numpy as np
    import psutil
    assert not args.out.exists() and not args.directory.exists()
    rp,ap,cp = [DOC/(s+'_20261013.json') for s in ('ffn_remainder_oracle_result',
        'ffn_remainder_oracle_adjudication_v2','ffn_remainder_oracle_session_closure_v2')]
    r,a,c = [json.loads(p.read_bytes()) for p in (rp,ap,cp)]
    assert a['all_numeric_pass'] and a['report'] == extent(rp) and c['audit'] == extent(ap)
    assert c['all_inputs_and_outputs_hash_verified'] and not c['candidate_admitted']
    old = json.loads(Path(r['binding']['path']).read_bytes())
    env_path = DOC/'ffn_sketch_router_session_closure_environment_20261014.json'
    env = json.loads(env_path.read_bytes())
    assert env['all_CPU_numerical_and_artifact_pins_current_exact']
    updates = {e['expected']['path']:e for e in env['incidental_GPU_component_drift']}
    counts = {}
    for site in old['sites']:
        passing = [s['count'] for s in r['summary'] if s['site'] == site and not s['control_only']
                   and s['FIT']['case_mean'] <= .10 and s['FIT']['worst'] <= .20]
        counts[str(site)] = min(passing)
    assert counts == {'0':512,'12':2048,'23':32}
    pins = old['inputs']+[r['binding']]+[x['file'] for x in r['records']]
    pins += [extent(p) for p in (rp,ap,cp,env_path,Path(__file__),B/'ffn_active_geometry.py',B/'ffn_active_router_audit.py',
                               B/'ffn_active_router_close.py',DOC/'FFN_ACTIVE_ROUTER_PROTOCOL_20261014.md')]
    unique = {}
    for e in pins:
        if e['path'] in updates:
            assert e == updates[e['path']]['expected']; e = updates[e['path']]['current']
        if e['path'] in unique: assert unique[e['path']] == e
        unique[e['path']] = e
    for e in unique.values(): assert extent(e['path']) == e,e['path']
    n = sum(x['history'] for x in old['records']); assert n == 22547
    model_bytes = 3*(6*2048*2048*8+4*2048*8+2*2048*512*8+2*4608*512*8+
                    4*4608*8+2048*4*8+4*4*8+16*8+16*2048*8+2*16*4608*8+2*16*8)
    trajectories = n*3*4*(2048*4+4+8)+n*4*sum(counts.values())*2
    payload = model_bytes+trajectories
    assert payload < 4 << 30
    binding = dict(schema='FFN_ACTIVE_ROUTER_BINDING_V1',freeze=args.freeze,inputs=list(unique.values()),
        inherited_GPU_pin_updates=list(updates.values()),environment_closure=extent(env_path),sigma_relative_cutoff=1e-12,
        source=old['source'],records=old['records'],sites=old['sites'],models=old['models'],trajectories=old['trajectories'],
        ranks=RANKS,counts=counts,alpha=old['alpha'],beta=old['beta'],gates=old['gates'],
        parent_report=extent(rp),parent_audit=extent(ap),parent_closure=extent(cp),
        versions=dict(numpy=np.__version__,psutil=psutil.__version__),output_directory=str(args.directory.resolve()),
        predicted_array_bytes=payload,limits=dict(seconds=1800,audit_seconds=1800,OS_bytes=8 << 30,output_bytes=4 << 30),
        scope='FIT-only exact source remainder-derivative metric balanced with input second moment. Dual '
              'input encoder/decoder and source-linear sketch ID path. Full output2048 '
              'and exact selected remainder. Reference evaluator still uses implicit full affine/source '
              'operators AFTER ID selection,not a physical reduced-cost kernel. No ternary precision, '
              'actual DRAM,useful large n,own-model quality,CPU TPS or general-family admission.')
    write(args.out,binding); print(json.dumps(dict(binding=extent(args.out),payload=payload,counts=counts)),flush=True)


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
    def route(rec,site):
        old = next(x for x in b['trajectories'] if x['site'] == site and x['id'] == rec['id'])
        with np.load(old['file']['path'],allow_pickle=False) as z: return z['labels'].copy()
    check(); args.directory.mkdir(); get,stream = tensor_reader(b['source'],np)
    parent = json.loads(Path(b['parent_report']['path']).read_bytes()); models,records,controls = [],[],[]
    try:
        for index,site in enumerate(b['sites']):
            old_model = next(x for x in b['models'] if x['site'] == site)
            with np.load(old_model['file']['path'],allow_pickle=False) as z: center = {k:z[k].copy() for k in z.files}
            U,G,D = [get(f'model.layers.{site}.feed_forward.{name}.weight') for name in ('up_proj','gate_proj','down_proj')]
            from ffn_active_geometry import balanced_geometry
            sketch,geometry = balanced_geometry(U,G,D,b['alpha'],b['beta'],center,
                [rec for rec in b['records'] if rec['split'] == 'FIT'],
                lambda rec:dec(rec['fields'][index]['x']),lambda rec:route(rec,site),guard)
            mp = args.directory/f'site{site:02d}.model.npz'; np.savez(mp,**sketch)
            me = extent(mp); output_bytes += me['bytes']; guard()
            count = b['counts'][str(site)]
            for rec in b['records']:
                guard(); X = dec(rec['fields'][index]['x']); actual = dec(rec['fields'][index]['output'])
                labels = route(rec,site); delta = X-center['centers'][labels]
                # All candidate IDs are chosen before full-source diagnostic activations.
                selected_by_rank = {rank:select_ids(delta,labels,center,sketch,rank,count,np) for rank in RANKS}
                u,g = X@U.T,(X@G.T)*b['alpha']; sx = g*np.exp(-np.logaddexp(0,-g))
                uc,gc,s,derivative = [center[k][labels] for k in ('u_center','g_center','s','derivative')]
                affine_feature = s*u+uc*derivative*(g-gc)
                rem = uc*(sx-s-derivative*(g-gc))+(u-uc)*(sx-s)
                affine,proxy = affine_feature@D.T*b['beta'],(u*sx)@D.T*b['beta']
                control = rel(proxy,actual); assert control <= .01
                controls.append(dict(site=site,id=rec['id'],error=control,pass_gate=True))
                oracle = next(x for x in parent['records'] if x['site'] == site and x['id'] == rec['id'] and x['count'] == count)
                with np.load(oracle['file']['path'],allow_pickle=False) as z: oracle_ids = z['selected'].copy()
                for rank in RANKS:
                    guard(); selected = selected_by_rank[rank]; mask = np.zeros(rem.shape,dtype=bool)
                    np.put_along_axis(mask,selected,True,axis=1)
                    prediction = affine+(rem*mask)@D.T*b['beta']
                    overlap = np.zeros_like(mask); np.put_along_axis(overlap,oracle_ids,True,axis=1)
                    recall = float(np.sum(mask&overlap)/(len(X)*count))
                    path = args.directory/f'site{site:02d}.k{rank:04d}.{rec["id"]}.npz'
                    required = len(X)*(2048*4+4+8+count*2)+4096
                    assert output_bytes+required <= b['limits']['output_bytes']
                    np.savez(path,labels=labels.astype('u4'),mass=np.ones(len(X),'f8'),selected=selected,prediction=prediction.astype('f4'))
                    file = extent(path); assert file['bytes'] <= required
                    output_bytes += file['bytes']
                    records.append(dict(site=site,rank=rank,count=count,id=rec['id'],split=rec['split'],domain=rec['domain'],
                        history=len(X),file=file,error_to_BF=rel(prediction,actual),error_to_proxy=rel(prediction,proxy),
                        oracle_error_to_BF=oracle['error_to_BF'],oracle_ID_recall=recall))
            summaries = []; selected_rank = None
            for rank in RANKS:
                group = [x for x in records if x['site'] == site and x['rank'] == rank]
                fi,dv = [dict(case_mean=sum(x['error_to_BF'] for x in group if x['split'] == split)/24,
                              worst=max(x['error_to_BF'] for x in group if x['split'] == split)) for split in ('FIT','DEV')]
                domains = {d:sum(x['error_to_BF'] for x in group if x['split'] == 'DEV' and x['domain'] == d)/
                           sum(x['split'] == 'DEV' and x['domain'] == d for x in group)
                           for d in sorted({x['domain'] for x in group if x['split'] == 'DEV'})}
                fit_pass = fi['case_mean'] <= .10 and fi['worst'] <= .20
                dev_pass = dv['case_mean'] <= .10 and dv['worst'] <= .20 and max(domains.values()) <= .20
                if selected_rank is None and fit_pass: selected_rank = rank
                summaries.append(dict(rank=rank,FIT=fi,DEV=dv,DEV_domains=domains,FIT_eligible=fit_pass,DEV_eligible=dev_pass))
            models.append(dict(site=site,file=me,geometry=geometry,count=count,summaries=summaries,FIT_selected_rank=selected_rank))
            print(json.dumps(dict(site=site,selected=selected_rank,summaries=summaries,seconds=time.monotonic()-start)),flush=True)
        check(); guard(); assert len(records) == 576 and len(models) == 3 and len(controls) == 144
        files = list(args.directory.iterdir()); assert len(files) == 579 and sum(p.stat().st_size for p in files) == output_bytes
        result = dict(schema='FFN_ACTIVE_ROUTER_RESULT_V1',binding=extent(args.binding),freeze=args.freeze,
            models=models,records=records,controls=controls,output_bytes=output_bytes,before_after_hashes_exact=True,
            source_model_forwards=0,GPU_calls=0,RESERVED_queries=0,optimizer_updates=0,
            seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            decision='COMPLETE_PENDING_FULL_AUDIT',scope=b['scope'])
        write(args.out,result); print(json.dumps(dict(result=extent(args.out))),flush=True)
    finally: stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--run',action='store_true')
    for key in ('binding','out','directory'): p.add_argument('--'+key,type=Path)
    p.add_argument('--binding-sha'); p.add_argument('--freeze',required=True); args = p.parse_args()
    import psutil
    start = time.monotonic(); proc = psutil.Process()
    receipt = dict(schema='FFN_ACTIVE_ROUTER_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,
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
