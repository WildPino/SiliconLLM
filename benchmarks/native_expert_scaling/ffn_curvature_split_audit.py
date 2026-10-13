"""Full independent chain-rule replay of frozen FFN defect diagnostics."""
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
    rp = DOC/'ffn_curvature_split_20261013.json'
    tp = rp.with_suffix('.terminal.json')
    out = DOC/'ffn_curvature_split_adjudication_20261013.json'
    assert not out.exists()
    r,t = [json.loads(p.read_bytes()) for p in (rp,tp)]
    assert t['exit_code'] == 0 and t['result'] == extent(rp)
    parent = json.loads(Path(r['parent_report']['path']).read_bytes())
    b = json.loads(Path(parent['binding']['path']).read_bytes())
    pins = r['inputs']+[extent(rp),extent(tp),extent(Path(__file__)),
           extent(DOC/'FFN_CURVATURE_SPLIT_AUDIT_PROTOCOL_20261013.md')]
    def guard():
        assert time.monotonic()-start <= 1800 and proc.memory_info().peak_wset <= 8 << 30
        assert not proc.children(recursive=True)
    def check():
        for e in pins: guard(); assert extent(e['path']) == e,e['path']
    def decode(e):
        x = np.memmap(e['path'],'u2','r',shape=tuple(e['shape']))
        return (x.astype('u4') << 16).view('f4').astype('f8')
    def sigmoid(x):
        z = np.exp(-np.abs(x))
        return np.where(x >= 0,1/(1+z),z/(1+z))
    def rel(x,y): return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    check(); get,stream = tensor_reader(b['source'],np)
    maximum = {}; independent = []
    try:
        alpha,beta = b['alpha_effective_F32'],b['beta_effective_F32']
        for index,site in enumerate(b['sites']):
            model = next(x for x in parent['models'] if x['site'] == site and x['count'] == 64)
            with np.load(model['file']['path'],allow_pickle=False) as z: C = z['centers'].copy()
            U,G,D = [get(f'model.layers.{site}.feed_forward.{name}.weight') for name in ('up_proj','gate_proj','down_proj')]
            uc_all,gc_all = C@U.T,(C@G.T)*alpha
            for rec in b['records']:
                guard(); X,Y = decode(rec['fields'][index]['x']),decode(rec['fields'][index]['output'])
                saved = next(x for x in parent['records'] if x['site'] == site and x['count'] == 64 and x['id'] == rec['id'])
                with np.load(saved['file']['path'],allow_pickle=False) as z: labels = z['labels'].copy()
                # Recompute deltas through the source operators, not subtraction of their images.
                delta = X-C[labels]; du,dg = delta@U.T,(delta@G.T)*alpha
                uc,gc = uc_all[labels],gc_all[labels]
                u,g = X@U.T,(X@G.T)*alpha
                q = sigmoid(gc); sx = g*sigmoid(g); sc = gc*q
                first = q*(1+gc*(1-q)); second = q*(1-q)*(2+gc*(1-2*q))
                base = (uc*sc)@D.T*beta
                linear = (sc*du+uc*first*dg)@D.T*beta
                affine = base+linear
                gate = (uc*(sx-sc-first*dg))@D.T*beta
                cross = (du*(sx-sc))@D.T*beta
                truth = (u*sx)@D.T*beta
                bilinear = (du*first*dg)@D.T*beta
                quadratic = affine+(du*first*dg+.5*uc*second*dg**2)@D.T*beta
                energy = max(float(np.sum(truth**2)),1e-30)
                values = dict(exact_remainder_identity_relative=rel(affine+gate+cross,truth),
                    affine_error=rel(affine,truth),
                    gate_remainder_relative=float(np.sqrt(np.sum(gate**2)/energy)),
                    cross_remainder_relative=float(np.sqrt(np.sum(cross**2)/energy)),
                    gate_cross_inner_over_reference_energy=float(np.sum(gate*cross)/energy),
                    retain_exact_cross_error=rel(affine+cross,truth),
                    add_bilinear_error=rel(affine+bilinear,truth),
                    full_quadratic_error=rel(quadratic,truth),
                    full_quadratic_error_to_BF=rel(quadratic,Y),proxy_error_to_BF=rel(truth,Y))
                old = next(x for x in r['records'] if x['site'] == site and x['id'] == rec['id'])
                assert all(old[k] == rec[k] for k in ('split','domain'))
                for k,value in values.items():
                    assert np.isfinite(value)
                    defect = abs(value-old[k]); maximum[k] = max(maximum.get(k,0),defect)
                    assert defect <= 1e-9,(site,rec['id'],k,defect)
                assert values['exact_remainder_identity_relative'] <= 1e-10
                assert abs(values['affine_error']**2-values['gate_remainder_relative']**2-
                           values['cross_remainder_relative']**2-
                           2*values['gate_cross_inner_over_reference_energy']) <= 1e-10
                independent.append(dict(site=site,id=rec['id'],split=rec['split'],domain=rec['domain'],**values))
            print(json.dumps(dict(site=site,checked=len(independent),seconds=time.monotonic()-start)),flush=True)
        fields = tuple(r['summary'][0]['means'])
        assert len(independent) == len(r['records']) == 144 and len(r['summary']) == 6
        for summary in r['summary']:
            selected = [x for x in independent if x['site'] == summary['site'] and x['split'] == summary['split']]
            assert len(selected) == 24
            for k in fields:
                assert abs(sum(x[k] for x in selected)/24-summary['means'][k]) <= 1e-9
                assert abs(max(x[k] for x in selected)-summary['worst'][k]) <= 1e-9
        assert r['GPU_calls'] == r['source_model_forwards'] == 0 and r['all_exact_remainder_identities_pass']
        check(); guard()
        result = dict(schema='FFN_CURVATURE_SPLIT_AUDIT_V1',source=extent(Path(__file__)),report=extent(rp),
            terminal=extent(tp),inputs=pins,checked_cases=144,maximum_metric_absolute_difference=maximum,
            all_numeric_pass=True,decision='INDEPENDENT_STORED_PASS',seconds=time.monotonic()-start,
            OS_peak=proc.memory_info().peak_wset,
            scope='Separate source/delta chain-rule/stable sigmoid replay of all diagnostics, '
                  'identities and summaries. Shares BF decoder/source tensor reader/NumPy BLAS; '
                  'no independent BF CUDA arithmetic, learned candidate or CPU performance claim.')
        write(out,result); print(json.dumps(dict(result=extent(out),decision=result['decision'])),flush=True)
    finally: stream.close()


if __name__ == '__main__':
    import psutil
    start = time.monotonic(); proc = psutil.Process()
    out = DOC/'ffn_curvature_split_adjudication_20261013.json'
    receipt = dict(schema='FFN_CURVATURE_SPLIT_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time())
    try: main(); receipt.update(exit_code=0,error=None,result=extent(out))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(out.with_suffix('.terminal.json'),receipt)
