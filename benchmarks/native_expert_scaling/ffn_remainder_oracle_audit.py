"""Literal conditional IDs plus independent full projected prediction replay."""
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
    rp = DOC/'ffn_remainder_oracle_result_20261013.json'; tp = rp.with_suffix('.terminal.json')
    out = DOC/'ffn_remainder_oracle_adjudication_20261013.json'; assert not out.exists()
    r,t = [json.loads(p.read_bytes()) for p in (rp,tp)]
    assert t['exit_code'] == 0 and t['result'] == extent(rp)
    assert extent(r['binding']['path']) == r['binding']; b = json.loads(Path(r['binding']['path']).read_bytes())
    pins = b['inputs']+[r['binding'],extent(rp),extent(tp)]+[x['file'] for x in r['records']]
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
    check(); get,stream = tensor_reader(b['source'],np)
    fixed = json.loads(Path(b['parent_report']['path']).read_bytes()); checked = 0; rows = []
    max_prediction = max_score_difference = max_feature_difference = max_identity = 0.
    try:
        for index,site in enumerate(b['sites']):
            model = next(m for m in b['models'] if m['site'] == site)
            with np.load(model['file']['path'],allow_pickle=False) as z: m = {k:z[k].copy() for k in z.files}
            C = m['centers']
            U,G,D = [get(f'model.layers.{site}.feed_forward.{name}.weight') for name in ('up_proj','gate_proj','down_proj')]
            center_u,center_g = C@U.T,(C@G.T)*b['alpha']
            q = sig(center_g); sc = center_g*q; first = q*(1+center_g*(1-q))
            base = (center_u*sc)@D.T*b['beta']
            energy = b['beta']**2*np.einsum('ij,ij->j',D,D)
            for rec in b['records']:
                guard(); X,Y = dec(rec['fields'][index]['x']),dec(rec['fields'][index]['output'])
                distance = np.stack([np.sum((X-c)**2,axis=1) for c in C],axis=1)
                labels = distance.argmin(axis=1); delta = X-C[labels]
                du,dg = delta@U.T,(delta@G.T)*b['alpha']
                u,g = X@U.T,(X@G.T)*b['alpha']; feature = u*(g*sig(g))
                linear_feature = sc[labels]*du+center_u[labels]*first[labels]*dg
                affine = base[labels]+linear_feature@D.T*b['beta']
                rem = feature-center_u[labels]*sc[labels]-linear_feature
                proxy = feature@D.T*b['beta']
                # Literal producer expression defines its floating-point score tie rule.
                lsx = g*np.exp(-np.logaddexp(0,-g))
                luc,lgc,ls,ld = [m[k][labels] for k in ('u_center','g_center','s','derivative')]
                literal_rem = luc*(lsx-ls-ld*(g-lgc))+(u-luc)*(lsx-ls)
                literal_score = literal_rem**2*np.sum((b['beta']*D)**2,axis=0)
                independent_score = rem**2*energy
                score_difference = float(np.linalg.norm(literal_score-independent_score))
                assert score_difference <= b['score_audit_rtol']*np.linalg.norm(literal_score)+b['score_audit_atol']
                feature_difference = float(np.linalg.norm(literal_rem-rem)/max(np.linalg.norm(feature),1e-30))
                assert feature_difference <= 1e-10
                max_score_difference = max(max_score_difference,score_difference)
                max_feature_difference = max(max_feature_difference,feature_difference)
                ranking = np.argsort(-literal_score,axis=1,kind='stable')
                control = rel(proxy,Y)
                old_control = next(x for x in r['controls'] if x['site'] == site and x['id'] == rec['id'])
                assert np.isfinite(control) and control <= .01 and abs(control-old_control['error']) <= 1e-10 and old_control['pass_gate']
                for count in b['counts']:
                    guard(); entry = next(x for x in r['records'] if x['site'] == site and x['count'] == count and x['id'] == rec['id'])
                    with np.load(entry['file']['path'],allow_pickle=False) as z: saved = {k:z[k].copy() for k in z.files}
                    assert set(saved) == {'labels','mass','selected','prediction'}
                    assert saved['labels'].dtype == np.dtype('u4') and saved['labels'].shape == (len(X),)
                    assert saved['mass'].dtype == np.dtype('f8') and saved['mass'].shape == (len(X),) and np.all(saved['mass'] == 1)
                    assert saved['selected'].dtype == np.dtype('u2') and saved['prediction'].dtype == np.dtype('f4')
                    assert saved['prediction'].shape == (len(X),2048) and np.array_equal(labels,saved['labels'])
                    assert entry['history'] == len(X) and all(entry[k] == rec[k] for k in ('split','domain'))
                    assert entry['control_only'] == (count == 4608)
                    if count < 4608:
                        assert saved['selected'].shape == (len(X),count)
                        assert np.array_equal(saved['selected'],ranking[:,:count])
                        # Separate streaming-by-column projection,not the producer's dense mask.
                        correction = np.zeros_like(affine)
                        # Groups of64 selected slots bound temporary (N,64,2048) memory.
                        for first_slot in range(0,count,64):
                            ids = saved['selected'][:,first_slot:first_slot+64]
                            coefficients = np.take_along_axis(rem,ids,axis=1)
                            correction += np.einsum('ni,oni->no',coefficients,D[:,ids],optimize=True)
                        prediction = affine+correction*b['beta']
                    else:
                        assert saved['selected'].shape == (len(X),0)
                        prediction = affine+rem@D.T*b['beta']
                    discrepancy = rel(saved['prediction'].astype('f8'),prediction)
                    assert np.isfinite(discrepancy) and discrepancy <= 1e-6
                    max_prediction = max(max_prediction,discrepancy)
                    eb,ep = rel(prediction,Y),rel(prediction,proxy)
                    assert abs(eb-entry['error_to_BF']) <= 1e-9 and abs(ep-entry['error_to_proxy']) <= 1e-9
                    if count == 4608: assert ep <= 1e-10; max_identity = max(max_identity,ep)
                    previous = next(x for x in fixed['records'] if x['site'] == site and x['id'] == rec['id'] and x['count'] == count)
                    assert entry['fixed_subset_error_to_BF'] == previous['error_to_BF']
                    rows.append(dict(site=site,count=count,split=rec['split'],domain=rec['domain'],error=eb,fixed_error=previous['error_to_BF']))
                    checked += 1
            print(json.dumps(dict(site=site,checked=checked,seconds=time.monotonic()-start)),flush=True)
        assert checked == len(r['records']) == 720 and len(r['controls']) == 144 and len(r['summary']) == 15
        for summary in r['summary']:
            group = [x for x in rows if x['site'] == summary['site'] and x['count'] == summary['count']]
            for split in ('FIT','DEV'):
                subset = [x for x in group if x['split'] == split]; assert len(subset) == 24
                values = dict(case_mean=sum(x['error'] for x in subset)/24,worst=max(x['error'] for x in subset),fixed_case_mean=sum(x['fixed_error'] for x in subset)/24)
                for k,v in values.items(): assert abs(v-summary[split][k]) <= 1e-9
            for domain,value in summary['DEV_domains'].items():
                subset = [x for x in group if x['split'] == 'DEV' and x['domain'] == domain]
                assert subset and abs(sum(x['error'] for x in subset)/len(subset)-value) <= 1e-9
            assert summary['control_only'] == (summary['count'] == 4608)
        assert all(r[k] == 0 for k in ('GPU_calls','source_model_forwards','RESERVED_queries','optimizer_updates'))
        check(); guard()
        result = dict(schema='FFN_REMAINDER_ORACLE_AUDIT_V1',source=extent(Path(__file__)),report=extent(rp),terminal=extent(tp),binding=r['binding'],
            checked_predictions=checked,all_numeric_pass=True,literal_selected_IDs_exact=True,
            maximum=dict(prediction_F32_relative=max_prediction,independent_score_frobenius_difference=max_score_difference,
                         remainder_feature_relative=max_feature_difference,identity=max_identity),
            decision='INDEPENDENT_STORED_PASS_DIAGNOSTIC_ONLY',seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            scope=b['scope']+' Literal floating score IDs plus independent direct Euclidean routing/chain rule/feature-minus-affine '
                  'remainders/streaming selected-column projection/all metrics. Shared source reader/BF decoder/NumPy primitives. No candidate admission.')
        write(out,result); print(json.dumps(dict(adjudication=extent(out))),flush=True)
    finally: stream.close()


if __name__ == '__main__':
    import psutil
    start = time.monotonic(); proc = psutil.Process(); out = DOC/'ffn_remainder_oracle_adjudication_20261013.json'
    receipt = dict(schema='FFN_REMAINDER_ORACLE_AUDIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try: main(); receipt.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(out.with_suffix('.terminal.json'),receipt)
