"""Exact remainder decomposition and quadratic diagnostic at frozen64 cells."""
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


def main():
    import numpy as np
    import psutil
    begin = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    rp = DOC/'ffn_affine_pilot_result_20261013.json'; ap = DOC/'ffn_affine_pilot_adjudication_20261013.json'
    r,a = [json.loads(p.read_bytes()) for p in (rp,ap)]
    assert a['decision'] == 'INDEPENDENT_STORED_PASS' and a['all_numeric_pass'] and a['report'] == extent(rp)
    assert a['proxy_controls_pass'] and extent(r['binding']['path']) == r['binding']
    b = json.loads(Path(r['binding']['path']).read_bytes())
    out = DOC/'ffn_curvature_split_20261013.json'; assert not out.exists()
    models = [m for m in r['models'] if m['count'] == 64]
    inputs = [extent(p) for p in (Path(__file__),DOC/'FFN_CURVATURE_SPLIT_PROTOCOL_20261013.md',rp,ap,Path(r['binding']['path']))]
    inputs += b['inputs']+[m['file'] for m in models]+[rec['file'] for rec in r['records'] if rec['count'] == 64]
    def guard():
        assert time.monotonic()-begin <= 1200 and proc.memory_info().peak_wset <= 8 << 30
        assert not proc.children(recursive=True)
    def check():
        for e in inputs: guard(); assert extent(e['path']) == e,e['path']
    check(); get,stream = tensor_reader(b['source'],np); rows = []
    def dec(x): return (x.astype('u4') << 16).view('f4').astype('f8')
    def sig(x): return np.exp(-np.logaddexp(0,-x))
    def rel(x,y): return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    try:
        alpha,beta = b['alpha_effective_F32'],b['beta_effective_F32']
        for index,site in enumerate(b['sites']):
            model = next(m for m in models if m['site'] == site)
            with np.load(model['file']['path'],allow_pickle=False) as z: m = {key:z[key].copy() for key in z.files}
            U,G,D = [get(f'model.layers.{site}.feed_forward.{name}.weight') for name in ('up_proj','gate_proj','down_proj')]
            for rec in b['records']:
                guard(); Xdesc,Ydesc = rec['fields'][index]['x'],rec['fields'][index]['output']
                X = dec(np.memmap(Xdesc['path'],'u2','r',shape=tuple(Xdesc['shape'])))
                actual = dec(np.memmap(Ydesc['path'],'u2','r',shape=tuple(Ydesc['shape'])))
                saved = next(p for p in r['records'] if p['site'] == site and p['count'] == 64 and p['id'] == rec['id'])
                with np.load(saved['file']['path'],allow_pickle=False) as z: labels = z['labels'].copy()
                u,g = X@U.T,X@G.T*alpha
                uc,gc,sm,sp = [m[k][labels] for k in ('u_center','g_center','s','derivative')]
                du,dg = u-uc,g-gc; sx = g*sig(g)
                feature_affine = sm*u+uc*sp*dg
                affine = feature_affine@D.T*beta; truth = (u*sx)@D.T*beta
                gate_curve = (uc*(sx-sm-sp*dg))@D.T*beta
                cross = (du*(sx-sm))@D.T*beta
                identity = rel(affine+gate_curve+cross,truth)
                assert identity <= 1e-10
                sg = sig(gc); second = sg*(1-sg)*(2+gc*(1-2*sg))
                bilinear = (sp*du*dg)@D.T*beta
                quadratic_gate = (.5*uc*second*dg*dg)@D.T*beta
                denominator = max(float(np.sum(truth*truth)),1e-30)
                row = dict(site=site,id=rec['id'],split=rec['split'],domain=rec['domain'],
                    exact_remainder_identity_relative=identity,
                    affine_error=rel(affine,truth),
                    gate_remainder_relative=float(np.linalg.norm(gate_curve)/max(np.linalg.norm(truth),1e-30)),
                    cross_remainder_relative=float(np.linalg.norm(cross)/max(np.linalg.norm(truth),1e-30)),
                    gate_cross_inner_over_reference_energy=float(np.sum(gate_curve*cross)/denominator),
                    retain_exact_cross_error=rel(affine+cross,truth),
                    add_bilinear_error=rel(affine+bilinear,truth),
                    full_quadratic_error=rel(affine+bilinear+quadratic_gate,truth),
                    full_quadratic_error_to_BF=rel(affine+bilinear+quadratic_gate,actual),
                    proxy_error_to_BF=rel(truth,actual))
                assert all(np.isfinite(value) for value in row.values() if isinstance(value,float))
                energy = (row['gate_remainder_relative']**2+row['cross_remainder_relative']**2+
                          2*row['gate_cross_inner_over_reference_energy'])
                assert abs(energy-row['affine_error']**2) <= 1e-10
                assert abs(row['affine_error']-saved['error_to_proxy']) <= 1e-10
                rows.append(row)
            print(json.dumps(dict(site=site,records=len(rows),seconds=time.monotonic()-begin)),flush=True)
        summary = []
        for site in b['sites']:
            for split in ('FIT','DEV'):
                selected = [x for x in rows if x['site'] == site and x['split'] == split]
                fields = ('affine_error','gate_remainder_relative','cross_remainder_relative',
                          'retain_exact_cross_error','add_bilinear_error','full_quadratic_error','full_quadratic_error_to_BF')
                summary.append(dict(site=site,split=split,means={k:sum(x[k] for x in selected)/24 for k in fields},
                    worst={k:max(x[k] for x in selected) for k in fields}))
        check(); guard(); assert len(rows) == 144
        result = dict(schema='FFN_CURVATURE_SPLIT_V1',source=extent(Path(__file__)),parent_report=extent(rp),parent_audit=extent(ap),
            inputs=inputs,records=rows,summary=summary,all_exact_remainder_identities_pass=True,
            seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset,GPU_calls=0,source_model_forwards=0,
            scope='Exact nonlinear remainder and second-order diagnostic at existing frozen64 '
                  'FIT cells on all known histories. Uses all source neurons/operators,no '
                  'compute reduction/new routing/candidate/model/CPU TPS claim.')
        write(out,result); print(json.dumps(dict(result=extent(out),summary=summary)),flush=True)
    finally: stream.close()


if __name__ == '__main__':
    import psutil
    start = time.monotonic(); proc = psutil.Process()
    receipt = dict(schema='FFN_CURVATURE_SPLIT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,creation_time=proc.create_time())
    try: main(); receipt.update(exit_code=0,error=None,result=extent(DOC/'ffn_curvature_split_20261013.json'))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(DOC/'ffn_curvature_split_20261013.terminal.json',receipt)
