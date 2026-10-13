"""Full stored affine-cell predictions, independent chain-rule action audit."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OPENBLAS_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'benchmarks/native_expert_scaling'),
               str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader


def main(a):
    import numpy as np
    import psutil
    from ffn_affine_pilot import centers
    begin = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    re,te = extent(a.report),extent(a.terminal)
    assert re['sha256'] == a.report_sha and te['sha256'] == a.terminal_sha
    r,t = [json.loads(p.read_bytes()) for p in (a.report,a.terminal)]
    assert t['exit_code'] == 0 and t['result'] == re and not a.out.exists()
    assert extent(r['binding']['path']) == r['binding']
    b = json.loads(Path(r['binding']['path']).read_bytes())
    get,stream = tensor_reader(b['source'],np); stage = 'hashes'; maximum = {}; gates = {}; checked = 0
    receipt = dict(schema='FFN_AFFINE_PILOT_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze)
    def guard():
        assert time.monotonic()-begin <= b['limits']['audit_seconds'] and proc.memory_info().peak_wset <= b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[re,te,r['binding']]+[x['file'] for x in r['models']+r['records']]:
            guard(); assert extent(e['path']) == e,e['path']
    def dec(x): return (x.astype('u4') << 16).view('f4').astype('f8')
    def rel(x,y): return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    def test(key,x,y,tolerance):
        error = rel(x,y); maximum[key] = max(maximum.get(key,0),error)
        gates[key] = bool(gates.get(key,True) and np.isfinite(error) and error <= tolerance)
    def data(rec,index,name):
        e = rec['fields'][index][name]
        return dec(np.memmap(e['path'],'u2','r',shape=tuple(e['shape'])))
    def sig(x):
        out = np.empty_like(x); positive = x >= 0
        out[positive] = 1/(1+np.exp(-x[positive])); ex = np.exp(x[~positive]); out[~positive] = ex/(1+ex)
        return out
    try:
        check(); alpha,beta = b['alpha_effective_F32'],b['beta_effective_F32']; eligibility = []
        for index,site in enumerate(b['sites']):
            U,G,D = [get(f'model.layers.{site}.feed_forward.{name}.weight') for name in ('up_proj','gate_proj','down_proj')]
            fit = np.concatenate([data(rec,index,'x') for rec in b['records'] if rec['split'] == 'FIT'])
            banks = []
            for model in [x for x in r['models'] if x['site'] == site]:
                stage = f'{site}/{model["count"]}/model'; guard()
                with np.load(model['file']['path'],allow_pickle=False) as z: m = {k:z[k].copy() for k in z.files}
                C,it = centers(fit,model['count'],np,guard)
                assert it == model['kmeans_iterations']; test('FIT_centers_replay',C,m['centers'],1e-12)
                uc,gc = C@U.T,C@G.T*alpha; ss = gc*sig(gc); derivative = sig(gc)+gc*sig(gc)*(1-sig(gc))
                for key,value in [('u_center',uc),('g_center',gc),('s',ss),('derivative',derivative),('offset',(uc*ss)@D.T*beta)]:
                    test(key,m[key],value,1e-10)
                v = m['direction']; assert abs(np.linalg.norm(v)-1) <= 1e-12
                action = (ss[0]*(v@U.T)+uc[0]*derivative[0]*(v@G.T)*alpha)@D.T*beta
                test('jacobian_action',m['jacobian_action'],action,1e-10)
                test('finite_difference',m['finite_difference'],action,1e-5)
                banks.append((model,m,ss,derivative))
            del fit
            local = []
            for rec in b['records']:
                stage = f'{site}/{rec["id"]}'; guard()
                X,actual = data(rec,index,'x'),data(rec,index,'output')
                ux,gx = X@U.T,X@G.T*alpha; proxy = (ux*gx*sig(gx))@D.T*beta
                old_control = next(x for x in r['controls'] if x['site'] == site and x['id'] == rec['id'])
                control = rel(proxy,actual); assert abs(control-old_control['error']) <= 1e-10
                assert old_control['pass_gate'] == (control <= b['gates']['numerical_proxy_each_case_relative'])
                for model,m,ss,derivative in banks:
                    entry = next(x for x in r['records'] if x['site'] == site and x['count'] == model['count'] and x['id'] == rec['id'])
                    with np.load(entry['file']['path'],allow_pickle=False) as z: saved = {k:z[k].copy() for k in z.files}
                    C = m['centers']; distances = np.stack([np.sum((X-c)**2,axis=1) for c in C],axis=1)
                    labels = distances.argmin(axis=1)
                    assert np.array_equal(labels.astype('u4'),saved['labels']) and np.all(saved['mass'] == 1)
                    radius = distances[np.arange(len(X)),labels]
                    assert np.allclose(radius,saved['radius_squared'],rtol=1e-10,atol=1e-10)
                    # Separate direct chain-rule computation in input deltas.
                    delta = X-C[labels]; du = delta@U.T; dg = (delta@G.T)*alpha
                    prediction = m['offset'][labels]+(ss[labels]*du+m['u_center'][labels]*derivative[labels]*dg)@D.T*beta
                    test('stored_prediction_F32',saved['prediction'].astype('f8'),prediction,1e-6)
                    eb,ep = rel(prediction,actual),rel(prediction,proxy)
                    assert abs(eb-entry['error_to_BF']) <= 1e-9 and abs(ep-entry['error_to_proxy']) <= 1e-9
                    rr = float(np.sqrt(radius.sum()/max(np.sum(X*X),1e-30)))
                    assert abs(rr-entry['relative_radius_RMS']) <= 1e-10
                    local.append(dict(count=model['count'],split=rec['split'],domain=rec['domain'],error=eb)); checked += 1
            selected = None
            for model,*_ in banks:
                cases = [x for x in local if x['count'] == model['count']]
                fi,dv = [dict(case_mean=sum(x['error'] for x in cases if x['split'] == split)/24,
                              worst=max(x['error'] for x in cases if x['split'] == split)) for split in ('FIT','DEV')]
                domains = {d:sum(x['error'] for x in cases if x['split'] == 'DEV' and x['domain'] == d) /
                           sum(x['split'] == 'DEV' and x['domain'] == d for x in cases)
                           for d in sorted({x['domain'] for x in cases if x['split'] == 'DEV'})}
                fit_pass = fi['case_mean'] <= .10 and fi['worst'] <= .20
                dev_pass = dv['case_mean'] <= .10 and dv['worst'] <= .20 and max(domains.values()) <= .20
                assert fit_pass == model['FIT_eligible'] and dev_pass == model['DEV_eligible']
                for key in fi: assert abs(fi[key]-model['FIT'][key]) <= 1e-9 and abs(dv[key]-model['DEV'][key]) <= 1e-9
                for key in domains: assert abs(domains[key]-model['DEV_domains'][key]) <= 1e-9
                if selected is None and fit_pass: selected = model['count']
                eligibility.append(dict(site=site,count=model['count'],FIT_eligible=fit_pass,DEV_eligible=dev_pass))
            assert all(model['FIT_selected_count'] == selected for model,*_ in banks)
            print(json.dumps(dict(site=site,selected=selected,seconds=time.monotonic()-begin)),flush=True)
        assert checked == 432 and len(r['controls']) == 144 and len(r['models']) == 9
        assert r['all_proxy_controls_pass'] == all(x['pass_gate'] for x in r['controls'])
        gates['source_proxy_fidelity'] = r['all_proxy_controls_pass']
        assert all(r[key] == 0 for key in ('source_model_forwards','GPU_calls','RESERVED_queries','optimizer_updates'))
        check(); guard()
        result = dict(schema='FFN_AFFINE_PILOT_AUDIT_V1',source=extent(Path(__file__)),freeze=a.freeze,
            report=re,terminal=te,binding=r['binding'],checked_predictions=checked,numeric_gates=gates,maximum=maximum,
            all_numeric_pass=all(gates.values()),proxy_controls_pass=r['all_proxy_controls_pass'],eligibility=eligibility,
            decision='INDEPENDENT_STORED_PASS' if all(gates.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',
            seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset,
            scope=b['scope']+' Independent source/chain-rule/distance calculations;FIT kmeans '
                  'replays frozen clustering code and shares NumPy BLAS. No independent clustering optimality claim.')
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(adjudication=extent(a.out),decision=result['decision'],eligibility=eligibility)),flush=True)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage,traceback=traceback.format_exc()); raise
    finally:
        stream.close(); receipt.update(seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('report','terminal','out'): p.add_argument('--'+k,type=Path,required=True)
    for k in ('report-sha','terminal-sha','freeze'): p.add_argument('--'+k,required=True)
    main(p.parse_args())
