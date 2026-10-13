"""Independent FIT algebra and full stored candidate curve audit; CPU only."""
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
    start = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    re,te = extent(a.report),extent(a.terminal)
    assert re['sha256'] == a.report_sha and te['sha256'] == a.terminal_sha
    r,t = [json.loads(p.read_bytes()) for p in (a.report,a.terminal)]
    assert t['exit_code'] == 0 and t['result'] == re and not a.out.exists()
    assert extent(r['binding']['path']) == r['binding']
    b = json.loads(Path(r['binding']['path']).read_bytes())
    get,stream = tensor_reader(b['source'],np); stage = 'hashes'; gates = {}; maximum = {}; checked = 0
    receipt = dict(schema='TOKEN_CONTEXT_RANK_AUDIT_TERMINAL_V1',command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze)
    def guard():
        assert time.monotonic()-start <= b['limits']['audit_seconds'] and proc.memory_info().peak_wset <= b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
    def check():
        for e in b['inputs']+[re,te,r['binding']]+[m['file'] for m in r['models']]:
            guard(); assert extent(e['path']) == e,e['path']
    def dec(x): return (x.astype('u4')*65536).view('f4').astype('f8')
    def bf(x):
        u = x.astype('f4').view('u4'); lo,hi = u&65535,u>>16
        return (hi+((lo>32768)|((lo==32768)&((hi%2)==1))).astype('u4')).astype('u2')
    def metric(x,y):
        return {k:float(np.linalg.norm(dec(x[:,lo:hi])-dec(y[:,lo:hi])) /
                        max(np.linalg.norm(dec(y[:,lo:hi])),1e-30)) for k,(lo,hi) in b['groups'].items()}
    def test(k,x,y,tol):
        e = float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
        maximum[k] = max(maximum.get(k,0),e); gates[k] = bool(gates.get(k,True) and np.isfinite(e) and e <= tol)
    def fields(row,p):
        raw = dec(row); n = len(row); pad = np.pad(raw[:,3072:6656],((3,0),(0,0)))
        blocks = np.stack([pad[j:j+n] for j in range(4)],axis=2)
        z = dec(bf(np.einsum('tck,ck->tc',blocks,p['cw'],optimize=True)+p['cb']))
        sig = np.empty_like(z); positive = z >= 0
        sig[positive] = 1/(1+np.exp(-z[positive])); ex = np.exp(z[~positive]); sig[~positive] = ex/(1+ex)
        delta = dec(bf(raw[:,6656:]+p['db']))
        soft = np.maximum(delta,0)+np.log1p(np.exp(-np.abs(delta))); soft[delta>20] = delta[delta>20]
        return np.concatenate([row[:,:3072],bf(z*sig),bf(soft)],axis=1)
    def data(rec,index):
        n = len(rec['input_ids']); z = np.memmap(rec['traces'][index]['path'],'u2','r',shape=(n,10800))
        return dec(z[:,:2048]),z[:,2048:8752].copy()
    try:
        check(); mup = dec(np.array(b['MuP_bits'],'u2'))
        for model in r['models']:
            site,method = model['site'],model['method']; index = b['sites'].index(site)
            stage = f'{site}/{method}'; guard()
            with np.load(model['file']['path'],allow_pickle=False) as z: m = {k:z[k].copy() for k in z.files}
            with np.load(b['exports'][index]['file']['path'],allow_pickle=False) as z: p = {k:z[k].copy() for k in ('cw','cb','db')}
            G = get(f'model.layers.{site}.mamba.in_proj.weight')
            fit = [rec for rec in b['records'] if rec['split'] == 'FIT']
            H = np.concatenate([data(rec,index)[0] for rec in fit])
            labels = np.concatenate([np.array(rec['input_ids'],'u4') for rec in fit])
            known = np.unique(labels); means = np.stack([H[labels == token].mean(axis=0) for token in known])
            assert np.array_equal(known,m['known']) and np.array_equal(bf(means),m['prototype'])
            assert np.array_equal(bf(H.mean(axis=0)),m['global_bits'])
            proto,global_mean = dec(m['prototype']),dec(m['global_bits'])
            lookup = dict(zip(known.tolist(),range(len(known)),strict=True))
            mu = proto[np.array([lookup[int(i)] for i in labels])] if method == 'FIT_token_mean' else global_mean
            residual = H-mu; C = residual.T@residual/len(H)
            test('FIT_covariance',m['C'],C,1e-10)
            test('covariance_root',m['sqrtC']@m['sqrtC'],C,1e-8)
            V = m['sqrtC']@m['Q']; test('basis_orthogonal',V.T@V,np.eye(1024),1e-6)
            test('source_factor',m['L'],G@m['sqrtC']@V,1e-6)
            targets = np.concatenate([data(rec,index)[1] for rec in fit])
            w = np.empty(6704)
            for lo,hi in b['groups'].values(): w[lo:hi] = 1/np.sqrt(np.mean(np.sum(dec(targets[:,lo:hi])**2,axis=1)))
            assert np.array_equal(w.view('u8'),m['weight'].view('u8'))
            M = (G*(mup*w)[:,None])@m['sqrtC']; gram = M.T@M
            test('weighted_eigen_identity',gram@V,V*m['metric_eigenvalues'][:1024],1e-6)
            assert np.all(m['metric_eigenvalues'][:-1] >= m['metric_eigenvalues'][1:])
            bias = proto@G.T; global_bias = global_mean@G.T
            local = []; controls = []
            for rec in b['records']:
                guard(); h,ref = data(rec,index); ids = rec['input_ids']
                indices = np.array([lookup.get(i,-1) for i in ids]); seen = indices >= 0
                mu = np.broadcast_to(global_mean,h.shape).copy()
                base = np.broadcast_to(global_bias,(len(h),6704)).copy()
                if method == 'FIT_token_mean': mu[seen] = proto[indices[seen]]; base[seen] = bias[indices[seen]]
                found = next(x for x in r['records'] if x['site'] == site and x['method'] == method and x['id'] == rec['id'])
                assert found['seen_token_positions'] == int(seen.sum()) and found['unknown_positions'] == int((~seen).sum())
                dense = bf(dec(bf(h@G.T))*mup); control = metric(dense,ref)
                for k in control: assert abs(control[k]-found['dense_control'][k]) <= 1e-10
                controls.append(max(control.values()) <= b['dense_source_control_gate'])
                coord = (h-mu)@m['Q']; reference_fields = fields(ref,p)
                for candidate in found['candidates']:
                    rank = candidate['rank']; raw = base.copy() if rank == 0 else base+coord[:,:rank]@m['L'][:,:rank].T
                    pred = bf(dec(bf(raw))*mup); pre,post = metric(pred,ref),metric(fields(pred,p),reference_fields)
                    for key in pre:
                        assert abs(pre[key]-candidate['preconv'][key]) <= 1e-10
                        assert abs(post[key]-candidate['causal'][key]) <= 1e-10
                    passed = bool(max(pre.values()) <= .01 and max(post.values()) <= .01)
                    assert passed == candidate['all_fields_pass']; checked += 1
                    local.append(dict(split=rec['split'],rank=rank,passed=passed))
            selected = next((rank for rank in b['ranks'] if all(x['passed'] for x in local if x['split'] == 'FIT' and x['rank'] == rank)),None)
            dev = bool(selected is not None and all(x['passed'] for x in local if x['split'] == 'DEV' and x['rank'] == selected))
            compact = bool(selected is not None and selected <= b['max_active_rank'] and dev)
            assert (selected,dev,compact) == (model['selected_rank'],model['DEV_selected_pass'],model['compact_candidate'])
            gates['dense_controls'] = bool(gates.get('dense_controls',True) and all(controls))
            print(json.dumps(dict(site=site,method=method,selected_rank=selected,compact=compact,seconds=time.monotonic()-start)),flush=True)
        assert checked == 6*48*5
        assert all(r[k] == 0 for k in ('source_model_forwards','GPU_calls','SSM_calls','RESERVED_queries','optimizer_updates'))
        assert r['all_dense_source_control_pass'] == gates['dense_controls']
        check(); guard()
        verdict = dict(schema='TOKEN_CONTEXT_RANK_AUDIT_V1',report=re,terminal=te,binding=r['binding'],
            source=extent(Path(__file__)),freeze=a.freeze,checked_candidates=checked,numeric_gates=gates,
            maximum=maximum,all_numeric_pass=all(gates.values()),
            compact_candidates=[dict(site=m['site'],method=m['method'],rank=m['selected_rank']) for m in r['models'] if m['compact_candidate']],
            decision='INDEPENDENT_STORED_PASS' if all(gates.values()) else 'INDEPENDENT_STORED_NUMERIC_FAILURE',
            seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset,
            scope=b['scope']+' Independent FIT moments/source-factor/eigen identities and every field curve checked; NumPy BLAS arithmetic shared.')
        write(a.out,verdict); receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(adjudication=extent(a.out),decision=verdict['decision'],compact_candidates=verdict['compact_candidates'])),flush=True)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage,traceback=traceback.format_exc()); raise
    finally:
        stream.close(); receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('report','terminal','out'): p.add_argument('--'+k,type=Path,required=True)
    for k in ('report-sha','terminal-sha','freeze'): p.add_argument('--'+k,required=True)
    main(p.parse_args())
