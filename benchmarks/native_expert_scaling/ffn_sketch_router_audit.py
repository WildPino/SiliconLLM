"""Independent FIT moment/eigen/source witnesses and complete sketch predictions."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OPENBLAS_NUM_THREADS'] = '6'
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(ROOT/'benchmarks/native_expert_scaling'),
               str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader


def main():
    import numpy as np
    import psutil
    start = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    rp = DOC/'ffn_sketch_router_result_20261013.json'; tp = rp.with_suffix('.terminal.json')
    out = DOC/'ffn_sketch_router_adjudication_20261013.json'; assert not out.exists()
    r,t = [json.loads(p.read_bytes()) for p in (rp,tp)]
    assert t['exit_code'] == 0 and t['result'] == extent(rp) and extent(r['binding']['path']) == r['binding']
    b = json.loads(Path(r['binding']['path']).read_bytes())
    pins = b['inputs']+[r['binding'],extent(rp),extent(tp)]+[x['file'] for x in r['models']+r['records']]
    def guard():
        assert time.monotonic()-start <= b['limits']['audit_seconds'] and proc.memory_info().peak_wset <= b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
    def check():
        for e in pins: guard(); assert extent(e['path']) == e,e['path']
    def dec(e):
        x = np.memmap(e['path'],'u2','r',shape=tuple(e['shape']))
        return (x.astype('u4') << 16).view('f4').astype('f8')
    def sig(x):
        z = np.exp(-np.abs(x)); return np.where(x >= 0,1/(1+z),z/(1+z))
    def rel(x,y): return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    maximum = {}; checked = 0; eligibility = []
    def compare(key,x,y,tol=1e-10):
        e = rel(x,y); assert np.isfinite(e) and e <= tol,(key,e)
        maximum[key] = max(maximum.get(key,0),e)
    check(); get,stream = tensor_reader(b['source'],np)
    parent = json.loads(Path(b['parent_report']['path']).read_bytes())
    try:
        for index,site in enumerate(b['sites']):
            pm = next(x for x in b['models'] if x['site'] == site)
            with np.load(pm['file']['path'],allow_pickle=False) as z: center = {k:z[k].copy() for k in z.files}
            model = next(x for x in r['models'] if x['site'] == site)
            with np.load(model['file']['path'],allow_pickle=False) as z: m = {k:z[k].copy() for k in z.files}
            assert set(m) == {'moment','covariance','mean','eigenvalues','basis','U_projected','G_projected','column_energy'}
            C = center['centers']
            def route(X): return np.stack([np.sum((X-c)**2,axis=1) for c in C],axis=1).argmin(axis=1)
            fit = []
            for rec in b['records']:
                if rec['split'] == 'FIT':
                    guard(); X = dec(rec['fields'][index]['x']); fit.append(X-C[route(X)])
            fit = np.concatenate(fit); mean = fit.mean(axis=0); moment = fit.T@fit/len(fit)
            centered = fit-mean; covariance = centered.T@centered/len(fit)
            compare('FIT_moment',m['moment'],moment); compare('FIT_mean',m['mean'],mean)
            compare('FIT_covariance',m['covariance'],covariance)
            compare('moment_covariance_mean_identity',m['moment'],m['covariance']+np.outer(m['mean'],m['mean']))
            P,e = m['basis'],m['eigenvalues']; assert P.shape == (2048,2048) and e.shape == (2048,)
            assert np.isfinite(e).all() and np.all(e[:-1] >= e[1:]) and e[-1] >= -1e-10*max(float(e[0]),1e-30)
            for column in range(2048): assert P[np.argmax(np.abs(P[:,column])),column] >= 0
            compare('orthogonality',P.T@P,np.eye(2048)); compare('eigen_residual',moment@P,P*e)
            compare('reconstruction',(P*e)@P.T,moment)
            for key,value in (('orthogonality',rel(P.T@P,np.eye(2048))),
                              ('eigen_residual',rel(moment@P,P*e)),('reconstruction',rel((P*e)@P.T,moment))):
                assert abs(model[key]-value) <= 1e-10
            assert model['FIT_rows'] == len(fit); del fit,centered
            U,G,D = [get(f'model.layers.{site}.feed_forward.{name}.weight') for name in ('up_proj','gate_proj','down_proj')]
            compare('U_projected',m['U_projected'],U@P[:,:512])
            compare('G_projected',m['G_projected'],(G@P[:,:512])*b['alpha'])
            compare('column_energy',m['column_energy'],np.sum((b['beta']*D)**2,axis=0))
            uc,gc = C@U.T,(C@G.T)*b['alpha']; q = sig(gc); sc = gc*q; first = q*(1+gc*(1-q))
            base = (uc*sc)@D.T*b['beta']; count = b['counts'][str(site)]; local = []
            assert model['count'] == count
            for rec in b['records']:
                guard(); X,Y = dec(rec['fields'][index]['x']),dec(rec['fields'][index]['output'])
                labels = route(X); delta = X-C[labels]
                # Literal projected-only ID replay,with no full-source activation input.
                ids_by_rank = {}
                for rank in b['ranks']:
                    Z = delta@P[:,:rank]
                    uh = center['u_center'][labels]+Z@m['U_projected'][:,:rank].T
                    gh = center['g_center'][labels]+Z@m['G_projected'][:,:rank].T
                    sh = gh*np.exp(-np.logaddexp(0,-gh))
                    cu,cg,cs,cd = [center[k][labels] for k in ('u_center','g_center','s','derivative')]
                    rr = cu*(sh-cs-cd*(gh-cg))+(uh-cu)*(sh-cs)
                    ids_by_rank[rank] = np.argsort(-(rr**2*m['column_energy']),axis=1,kind='stable')[:,:count].astype('u2')
                # Independent exact source-reference projection only after IDs are fixed.
                du,dg = delta@U.T,(delta@G.T)*b['alpha']; u,g = X@U.T,(X@G.T)*b['alpha']
                feature = u*(g*sig(g)); lin = sc[labels]*du+uc[labels]*first[labels]*dg
                affine = base[labels]+lin@D.T*b['beta']; rem = feature-uc[labels]*sc[labels]-lin
                proxy = feature@D.T*b['beta']; control = rel(proxy,Y)
                old_control = next(x for x in r['controls'] if x['site'] == site and x['id'] == rec['id'])
                assert control <= .01 and abs(control-old_control['error']) <= 1e-10 and old_control['pass_gate']
                oracle = next(x for x in parent['records'] if x['site'] == site and x['id'] == rec['id'] and x['count'] == count)
                with np.load(oracle['file']['path'],allow_pickle=False) as z: oracle_ids = z['selected'].copy()
                oracle_mask = np.zeros(rem.shape,dtype=bool); np.put_along_axis(oracle_mask,oracle_ids,True,axis=1)
                for rank in b['ranks']:
                    guard(); entry = next(x for x in r['records'] if x['site'] == site and x['rank'] == rank and x['id'] == rec['id'])
                    with np.load(entry['file']['path'],allow_pickle=False) as z: saved = {k:z[k].copy() for k in z.files}
                    assert set(saved) == {'labels','mass','selected','prediction'}
                    assert saved['labels'].dtype == np.dtype('u4') and np.array_equal(labels,saved['labels'])
                    assert saved['mass'].dtype == np.dtype('f8') and saved['mass'].shape == (len(X),) and np.all(saved['mass'] == 1)
                    assert saved['selected'].dtype == np.dtype('u2') and saved['selected'].shape == (len(X),count)
                    assert saved['prediction'].dtype == np.dtype('f4') and saved['prediction'].shape == (len(X),2048)
                    assert np.array_equal(saved['selected'],ids_by_rank[rank])
                    membership = np.zeros(rem.shape,dtype=bool); np.put_along_axis(membership,saved['selected'],True,axis=1)
                    correction = np.zeros_like(affine)
                    for column in range(0,4608,256):
                        correction += (rem[:,column:column+256]*membership[:,column:column+256])@D[:,column:column+256].T
                    prediction = affine+correction*b['beta']; compare('prediction_F32',saved['prediction'].astype('f8'),prediction,1e-6)
                    eb,ep = rel(prediction,Y),rel(prediction,proxy)
                    assert abs(eb-entry['error_to_BF']) <= 1e-9 and abs(ep-entry['error_to_proxy']) <= 1e-9
                    assert entry['oracle_error_to_BF'] == oracle['error_to_BF']
                    recall = float(np.sum(membership&oracle_mask)/(len(X)*count))
                    assert recall == entry['oracle_ID_recall'] and entry['count'] == count and entry['history'] == len(X)
                    assert all(entry[k] == rec[k] for k in ('split','domain'))
                    local.append(dict(rank=rank,split=rec['split'],domain=rec['domain'],error=eb)); checked += 1
            selected = None
            for rank in b['ranks']:
                group = [x for x in local if x['rank'] == rank]
                fi,dv = [dict(case_mean=sum(x['error'] for x in group if x['split'] == split)/24,
                              worst=max(x['error'] for x in group if x['split'] == split)) for split in ('FIT','DEV')]
                domains = {d:sum(x['error'] for x in group if x['split'] == 'DEV' and x['domain'] == d)/
                           sum(x['split'] == 'DEV' and x['domain'] == d for x in group)
                           for d in sorted({x['domain'] for x in group if x['split'] == 'DEV'})}
                fit_pass = fi['case_mean'] <= .10 and fi['worst'] <= .20
                dev_pass = dv['case_mean'] <= .10 and dv['worst'] <= .20 and max(domains.values()) <= .20
                old = next(x for x in model['summaries'] if x['rank'] == rank)
                for k in fi: assert abs(fi[k]-old['FIT'][k]) <= 1e-9 and abs(dv[k]-old['DEV'][k]) <= 1e-9
                for k in domains: assert abs(domains[k]-old['DEV_domains'][k]) <= 1e-9
                assert old['FIT_eligible'] == fit_pass and old['DEV_eligible'] == dev_pass
                if selected is None and fit_pass: selected = rank
                eligibility.append(dict(site=site,rank=rank,FIT_eligible=fit_pass,DEV_eligible=dev_pass))
            assert model['FIT_selected_rank'] == selected
            print(json.dumps(dict(site=site,selected=selected,checked=checked,seconds=time.monotonic()-start)),flush=True)
        assert checked == len(r['records']) == 576 and len(r['models']) == 3 and len(r['controls']) == 144
        assert all(r[k] == 0 for k in ('GPU_calls','source_model_forwards','RESERVED_queries','optimizer_updates'))
        check(); guard()
        result = dict(schema='FFN_SKETCH_ROUTER_AUDIT_V1',source=extent(Path(__file__)),report=extent(rp),terminal=extent(tp),
            binding=r['binding'],checked_predictions=checked,maximum=maximum,literal_selected_IDs_exact=True,
            all_numeric_pass=True,eligibility=eligibility,decision='INDEPENDENT_STORED_PASS',
            seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            scope=b['scope']+' Separate concatenated FIT moments/centered covariance,eigen/source witnesses, '
                  'direct distances,chain-rule reference and256-column output projection. Literal projected ID '
                  'rule shares arithmetic;NumPy BLAS/source reader/BF decoder shared. No physical/model admission.')
        write(out,result); print(json.dumps(dict(adjudication=extent(out))),flush=True)
    finally: stream.close()


if __name__ == '__main__':
    import psutil
    start = time.monotonic(); proc = psutil.Process(); out = DOC/'ffn_sketch_router_adjudication_20261013.json'
    receipt = dict(schema='FFN_SKETCH_ROUTER_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try: main(); receipt.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(out.with_suffix('.terminal.json'),receipt)
