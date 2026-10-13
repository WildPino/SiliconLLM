"""Necessary shared-residual rank, with prototype and BF rounding allowances."""
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
    assert not (DOC/'token_context_rank_bound_20261013.json').exists()
    rp = DOC/'token_context_rank_result_20261013.json'; ap = DOC/'token_context_rank_adjudication_20261013.json'
    r,a = [json.loads(p.read_bytes()) for p in (rp,ap)]
    assert a['decision'] == 'INDEPENDENT_STORED_PASS' and a['report'] == extent(rp)
    assert extent(r['binding']['path']) == r['binding']
    b = json.loads(Path(r['binding']['path']).read_bytes()); get,stream = tensor_reader(b['source'],np)
    def dec(x): return (x.astype('u4') << 16).view('f4').astype('f8')
    before = [extent(p) for p in [Path(__file__),DOC/'TOKEN_CONTEXT_RANK_BOUND_PROTOCOL_20261013.md',rp,ap,Path(r['binding']['path'])]]
    before += b['inputs']+[m['file'] for m in r['models'] if m['method'] == 'FIT_token_mean']
    for e in before: assert extent(e['path']) == e
    rows = []
    try:
        for model in r['models']:
            if model['method'] != 'FIT_token_mean': continue
            site = model['site']; index = b['sites'].index(site)
            with np.load(model['file']['path'],allow_pickle=False) as z:
                m = {k:z[k].copy() for k in ('known','prototype','C','sqrtC','weight','metric_eigenvalues')}
            G = get(f'model.layers.{site}.mamba.in_proj.weight'); mup = dec(np.array(b['MuP_bits'],'u2'))
            counts = np.zeros(len(m['known']),'i8'); sums = np.zeros((len(counts),2048))
            source_error_energy = 0.; total = 0
            for rec in b['records']:
                if rec['split'] != 'FIT': continue
                n = len(rec['input_ids']); z = np.memmap(rec['traces'][index]['path'],'u2','r',shape=(n,10800))
                ids = np.array(rec['input_ids'],'u4'); positions = np.searchsorted(m['known'],ids)
                assert np.array_equal(m['known'][positions],ids)
                H = dec(z[:,:2048]); np.add.at(sums,positions,H); np.add.at(counts,positions,1)
                for lo in range(0,n,256):
                    assert time.monotonic()-begin <= 1200 and proc.memory_info().peak_wset <= 8 << 30
                    hi = min(n,lo+256); raw = H[lo:hi]@G.T*mup
                    difference = (raw-dec(z[lo:hi,2048:8752]))*m['weight']
                    source_error_energy += float(np.sum(difference**2))
                total += n
            means = sums/counts[:,None]; delta = means-dec(m['prototype'])
            offset = delta@G.T*(mup*m['weight'])
            offset_energy = float(np.sum(counts[:,None]*offset**2)/total)
            M = (G*(mup*m['weight'])[:,None])@m['sqrtC']
            eigen = np.linalg.eigvalsh(M.T@M)[::-1]
            assert np.allclose(eigen,m['metric_eigenvalues'],rtol=1e-8,atol=1e-10)
            eigen = np.maximum(eigen,0)
            tails = np.concatenate([np.cumsum(eigen[::-1])[::-1],[0.]])
            floor = np.sqrt(np.maximum(tails-offset_energy,0))
            u32,uBF = 2.**-24,2.**-8
            eta1 = (1+u32)*(1+uBF)-1; eta = (1+eta1)**2-1
            alpha1 = (1+uBF)*2.**-150+2.**-134
            alpha = float(np.linalg.norm(m['weight']*alpha1*(np.abs(mup)*(1+eta1)+1)))
            tolerance = np.sqrt(5)*b['fit_field_gate']
            source_difference = np.sqrt(source_error_energy/total)
            allowed = float(source_difference+tolerance+eta*(np.sqrt(5)+tolerance+alpha)/(1-eta)+alpha)
            necessary = int(np.flatnonzero(floor <= allowed)[0])
            rows.append(dict(site=site,FIT_positions=total,source_real_to_stored_weighted_RMS=float(source_difference),
                prototype_BF_rounding_energy=offset_energy,eta=eta,alpha=alpha,feasibility_RMS_bound=allowed,
                minimum_necessary_shared_rank=necessary,
                weighted_floor={str(k):float(floor[k]) for k in (0,128,256,512,1024)},
                independent_full_spectrum_verified=True))
            print(json.dumps(rows[-1]),flush=True)
            assert time.monotonic()-begin <= 1200 and proc.memory_info().peak_wset <= 8 << 30
        for e in before: assert extent(e['path']) == e
        out = DOC/'token_context_rank_bound_20261013.json'; assert not out.exists()
        result = dict(schema='TOKEN_CONTEXT_RANK_BOUND_V1',source=extent(Path(__file__)),report=extent(rp),
            audit=extent(ap),inputs=before,rows=rows,seconds=time.monotonic()-begin,
            OS_peak=proc.memory_info().peak_wset,GPU_calls=0,source_model_forwards=0,
            scope='Necessary rank for ANY token-only prototype plus ONE shared linear residual '
                  'operator on this FIT set and actual G,allowing declared F32/BF16 output '
                  'rounding and measured source reference discrepancy. Not a bound on '
                  'token-dependent operators/nonlinear routers/whole-model quality/general source families.')
        write(out,result); print(json.dumps(dict(result=extent(out))),flush=True)
    finally: stream.close()


if __name__ == '__main__':
    import psutil
    start = time.monotonic(); proc = psutil.Process()
    receipt = dict(schema='TOKEN_CONTEXT_RANK_BOUND_TERMINAL_V1',command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time())
    try:
        main(); receipt.update(exit_code=0,error=None,result=extent(DOC/'token_context_rank_bound_20261013.json'))
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(DOC/'token_context_rank_bound_20261013.terminal.json',receipt)
