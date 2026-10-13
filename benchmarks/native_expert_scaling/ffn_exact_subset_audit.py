"""Recompute FIT selection and all exact-remainder subset predictions."""
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
    rp = DOC/'ffn_exact_subset_result_20261013.json'; tp = rp.with_suffix('.terminal.json')
    out = DOC/'ffn_exact_subset_adjudication_20261013.json'; assert not out.exists()
    r,t = [json.loads(p.read_bytes()) for p in (rp,tp)]
    assert t['exit_code'] == 0 and t['result'] == extent(rp)
    assert extent(r['binding']['path']) == r['binding']; b = json.loads(Path(r['binding']['path']).read_bytes())
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
    def compare(key,x,y,tol):
        error = rel(x,y); assert np.isfinite(error) and error <= tol,(key,error)
        maximum[key] = max(maximum.get(key,0),error)
    check(); get,stream = tensor_reader(b['source'],np)
    try:
        for index,site in enumerate(b['sites']):
            pm = next(x for x in b['models'] if x['site'] == site)
            with np.load(pm['file']['path'],allow_pickle=False) as z: C = z['centers'].copy()
            U,G,D = [get(f'model.layers.{site}.feed_forward.{name}.weight') for name in ('up_proj','gate_proj','down_proj')]
            uc,gc = C@U.T,(C@G.T)*b['alpha']; q = sig(gc)
            sc,first = gc*q,q*(1+gc*(1-q))
            def compute(rec):
                X = dec(rec['fields'][index]['x'])
                distances = np.stack([np.sum((X-c)**2,axis=1) for c in C],axis=1)
                labels = distances.argmin(axis=1)
                delta = X-C[labels]; du,dg = delta@U.T,(delta@G.T)*b['alpha']
                u,g = X@U.T,(X@G.T)*b['alpha']; truth_feature = u*g*sig(g)
                affine_feature = uc[labels]*sc[labels]+sc[labels]*du+uc[labels]*first[labels]*dg
                # Exact feature minus direct chain-rule feature is a separate remainder formula.
                remainder = truth_feature-affine_feature
                return labels,affine_feature,remainder,truth_feature
            energy = b['beta']**2*np.einsum('ij,ij->j',D,D)
            sums = np.zeros((64,4608),'f8'); counts = np.zeros(64,'i8')
            for rec in b['records']:
                if rec['split'] != 'FIT': continue
                guard(); labels,af,rem,feature = compute(rec)
                for cell in np.unique(labels):
                    rows = labels == cell; sums[cell] += np.sum(rem[rows]**2,axis=0); counts[cell] += int(rows.sum())
            global_score = sums.sum(axis=0)/counts.sum()*energy
            scores = np.stack([sums[c]/counts[c]*energy if counts[c] else global_score for c in range(64)])
            ranking = np.argsort(-scores,axis=1,kind='stable').astype('i4')
            model = next(x for x in r['models'] if x['site'] == site)
            with np.load(model['file']['path'],allow_pickle=False) as z: saved = {k:z[k].copy() for k in z.files}
            assert set(saved) == {'scores','ranking','FIT_counts','column_energy'}
            compare('FIT_scores',saved['scores'],scores,1e-10); compare('down_column_energy',saved['column_energy'],energy,1e-12)
            assert np.array_equal(saved['FIT_counts'],counts) and np.array_equal(saved['ranking'],ranking)
            assert model['empty_FIT_cells'] == int((counts == 0).sum())
            local = []
            for rec in b['records']:
                guard(); labels,af,rem,feature = compute(rec)
                actual = dec(rec['fields'][index]['output']); affine,proxy = af@D.T*b['beta'],feature@D.T*b['beta']
                control = rel(proxy,actual)
                old = next(x for x in r['controls'] if x['site'] == site and x['id'] == rec['id'])
                assert abs(control-old['error']) <= 1e-10 and old['pass_gate'] and control <= .01
                for count in b['counts']:
                    guard(); mask = np.zeros((64,4608),'f8')
                    np.put_along_axis(mask,ranking[:,:count],1,axis=1)
                    # Dense masked projection independently verifies the grouped sparse calculation.
                    prediction = affine+(rem*mask[labels])@D.T*b['beta']
                    entry = next(x for x in r['records'] if x['site'] == site and x['id'] == rec['id'] and x['count'] == count)
                    with np.load(entry['file']['path'],allow_pickle=False) as z: trajectory = {k:z[k].copy() for k in z.files}
                    assert set(trajectory) == {'labels','mass','prediction'}
                    assert trajectory['labels'].dtype == np.dtype('u4') and trajectory['mass'].dtype == np.dtype('f8')
                    assert trajectory['prediction'].dtype == np.dtype('f4')
                    assert np.array_equal(labels,trajectory['labels']) and np.all(trajectory['mass'] == 1)
                    assert entry['history'] == len(labels) and all(entry[k] == rec[k] for k in ('split','domain'))
                    compare('prediction_F32',trajectory['prediction'].astype('f8'),prediction,1e-6)
                    eb,ep = rel(prediction,actual),rel(prediction,proxy)
                    assert abs(eb-entry['error_to_BF']) <= 1e-9 and abs(ep-entry['error_to_proxy']) <= 1e-9
                    if count == 4608: assert ep <= 1e-10
                    local.append(dict(count=count,split=rec['split'],domain=rec['domain'],error=eb)); checked += 1
            selected = None
            for count in b['counts']:
                group = [x for x in local if x['count'] == count]
                fi,dv = [dict(case_mean=sum(x['error'] for x in group if x['split'] == split)/24,
                              worst=max(x['error'] for x in group if x['split'] == split)) for split in ('FIT','DEV')]
                domains = {d:sum(x['error'] for x in group if x['split'] == 'DEV' and x['domain'] == d)/
                           sum(x['split'] == 'DEV' and x['domain'] == d for x in group)
                           for d in sorted({x['domain'] for x in group if x['split'] == 'DEV'})}
                fit_pass = fi['case_mean'] <= .10 and fi['worst'] <= .20
                dev_pass = dv['case_mean'] <= .10 and dv['worst'] <= .20 and max(domains.values()) <= .20
                old = next(x for x in model['summaries'] if x['count'] == count)
                assert old['FIT_eligible'] == fit_pass and old['DEV_eligible'] == dev_pass and old['control_only'] == (count == 4608)
                for k in fi: assert abs(fi[k]-old['FIT'][k]) <= 1e-9 and abs(dv[k]-old['DEV'][k]) <= 1e-9
                for k in domains: assert abs(domains[k]-old['DEV_domains'][k]) <= 1e-9
                if count < 4608 and fit_pass and selected is None: selected = count
                eligibility.append(dict(site=site,count=count,FIT_eligible=fit_pass,DEV_eligible=dev_pass,control_only=count == 4608))
            assert model['FIT_selected_count'] == selected
            print(json.dumps(dict(site=site,selected=selected,checked=checked,seconds=time.monotonic()-start)),flush=True)
        assert checked == len(r['records']) == 720 and len(r['controls']) == 144 and len(r['models']) == 3
        assert all(r[k] == 0 for k in ('GPU_calls','source_model_forwards','RESERVED_queries','optimizer_updates'))
        check(); guard()
        result = dict(schema='FFN_EXACT_SUBSET_AUDIT_V1',source=extent(Path(__file__)),report=extent(rp),terminal=extent(tp),
            binding=r['binding'],checked_predictions=checked,maximum=maximum,eligibility=eligibility,
            all_numeric_pass=True,decision='INDEPENDENT_STORED_PASS',seconds=time.monotonic()-start,
            OS_peak=proc.memory_info().peak_wset,
            scope=b['scope']+' Independent direct Euclidean routing,chain-rule deltas,feature '
                  'subtraction,FIT score grouping and dense masked predictions. Shares '
                  'source tensor reader/BF decoding/NumPy BLAS;no selection optimality claim.')
        write(out,result); print(json.dumps(dict(adjudication=extent(out))),flush=True)
    finally: stream.close()


if __name__ == '__main__':
    import psutil
    start = time.monotonic(); proc = psutil.Process(); out = DOC/'ffn_exact_subset_adjudication_20261013.json'
    receipt = dict(schema='FFN_EXACT_SUBSET_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try: main(); receipt.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(out.with_suffix('.terminal.json'),receipt)
