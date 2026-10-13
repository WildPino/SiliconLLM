"""FIT token prototypes plus output-aware context residual rank, actual own H.

Pure CPU algebra assay. No whole-model substitution or routing-speed claim.
"""
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
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(B),str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write,tensor_reader

SITES = [1,12,23]
RANKS = [0,128,256,512,1024]
GROUPS = dict(gate=(0,3072),x=(3072,6144),B=(6144,6400),C=(6400,6656),delta=(6656,6704))


def decode(x,np):
    return (x.astype('u4') << 16).view('f4').astype('f8')


def bf(x,np):
    u = x.astype('f4').view('u4'); hi,lo = u >> 16,u & 65535
    return (hi+((lo>32768)|((lo==32768)&((hi&1)!=0))).astype('u4')).astype('u2')


def causal(row,p,np):
    raw = decode(row,np); n = len(row)
    pad = np.pad(raw[:,3072:6656],((3,0),(0,0)))
    y = np.broadcast_to(p['cb'],(n,3584)).copy()
    for tap in range(4): y += pad[tap:tap+n]*p['cw'][:,tap]
    y = decode(bf(y,np),np)
    y = bf(y*np.exp(-np.logaddexp(0,-y)),np)
    dt = decode(bf(raw[:,6656:]+p['db'],np),np)
    soft = np.logaddexp(0,dt); soft[dt>20] = dt[dt>20]
    return np.concatenate([row[:,:3072],y,bf(soft,np)],axis=1)


def errors(x,y,np):
    return {name:float(np.linalg.norm(decode(x[:,lo:hi],np)-decode(y[:,lo:hi],np)) /
                       max(np.linalg.norm(decode(y[:,lo:hi],np)),1e-30))
            for name,(lo,hi) in GROUPS.items()}


def prepare(a):
    import numpy as np
    import psutil
    assert not a.out.exists() and not a.directory.exists()
    rp = DOC/'full_width_native_chat_result_20261013.json'
    ap = DOC/'full_width_native_chat_adjudication_20261013.json'
    cp = DOC/'full_width_native_chat_session_closure_20261013.json'
    r,ad,cl = [json.loads(p.read_bytes()) for p in (rp,ap,cp)]
    assert ad['decision'] == 'INDEPENDENT_STORED_QUALITY_PASS' and ad['report'] == extent(rp)
    assert cl['adjudication'] == extent(ap) and cl['decision'] == 'KNOWN_CORPUS_FULL_WIDTH_NATIVE_BACKBONE_QUALIFIED'
    b = json.loads(Path(r['binding']['path']).read_bytes())
    records = [dict(id=x['id'],split=x['split'],domain=x['domain'],
                    input_ids=x['sequence']['consumed_ids'],
                    traces=[x['sequence']['sites'][site]['trace'] for site in SITES]) for x in r['records']]
    assert len(records) == 48 and sum(len(x['input_ids']) for x in records) == 22547
    exports = [b['exports'][site] for site in SITES]
    paths = [Path(__file__),B/'token_context_rank_audit.py',B/'falcon_auxiliary_s6_transport.py',
             DOC/'TOKEN_CONTEXT_RANK_PROTOCOL_20261013.md',rp,ap,cp,Path(r['binding']['path'])]
    paths += [Path(x['path']) for rec in records for x in rec['traces']]
    paths += [Path(e['file']['path']) for e in exports]
    # Source checkpoint/actual resolved NumPy/BLAS/Python binaries from parent.
    source = Path(b['source'])/'model.safetensors'
    paths += [Path(e['path']) for e in b['inputs'] if Path(e['path']).suffix.lower() in ('.dll','.pyd','.exe')]
    paths += [source,Path(np.__file__),Path(psutil.__file__),B/'composed_operator_mixed.py']
    inputs = [extent(p) for p in dict.fromkeys(paths)]
    for e in inputs:
        if e['path'] == str(source.resolve()):
            assert e == next(v for v in b['inputs'] if v['path'] == e['path'])
    binding = dict(schema='TOKEN_CONTEXT_RANK_BINDING_V1',freeze=a.freeze,
        python=str(Path(sys.executable).resolve()),versions=dict(numpy=np.__version__,psutil=psutil.__version__),
        parent_report=extent(rp),parent_audit=extent(ap),parent_closure=extent(cp),
        source=str(source.resolve()),records=records,exports=exports,sites=SITES,
        MuP_bits=b['MuP_bits'],ranks=RANKS,groups=GROUPS,
        methods=['global_center','FIT_token_mean'],unknown_policy='BF16 FIT global mean,ALL unknown DEV rows retained',
        covariance_cutoff_relative=1e-10,fit_field_gate=.01,DEV_field_gate=.01,
        max_active_rank=256,dense_source_control_gate=.01,
        inputs=inputs,output_directory=str(a.directory.resolve()),
        limits=dict(seconds=1800,audit_seconds=1800,OS_bytes=8 << 30,output_bytes=1 << 30),
        scope='Three actual own-history sites1/12/23:FIT-only BF16 token prototypes,weighted '
              'linear residual projection. ALL48 histories/unknown DEV fallback included. '
              'No model substitution/SSM/GPU/RESERVED/CPU cost/useful-n/TPS admission.')
    write(a.out,binding); print(json.dumps(dict(binding=extent(a.out),sites=SITES)),flush=True)


def run(a):
    import numpy as np
    import psutil
    start = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    assert extent(a.binding)['sha256'] == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert not a.out.exists() and not a.directory.exists() and a.directory.resolve() == Path(b['output_directory']).resolve()
    assert dict(numpy=np.__version__,psutil=psutil.__version__) == b['versions']
    get,stream = tensor_reader(b['source'],np); stage = 'hashes'; results = []; models = []
    receipt = dict(schema='TOKEN_CONTEXT_RANK_TERMINAL_V1',command=sys.orig_argv,
                   pid=proc.pid,creation_time=proc.create_time(),freeze=a.freeze)
    def guard():
        assert time.monotonic()-start <= b['limits']['seconds'] and proc.memory_info().peak_wset <= b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
        if a.directory.exists(): assert sum(p.stat().st_size for p in a.directory.iterdir()) <= b['limits']['output_bytes']
    def check():
        for e in b['inputs']: guard(); assert extent(e['path']) == e,e['path']
    def data(rec,index):
        n = len(rec['input_ids']); e = rec['traces'][index]
        assert e['shape'] == [n,10800]
        z = np.memmap(e['path'],'u2','r',shape=(n,10800))
        return decode(z[:,:2048],np),z[:,2048:8752].copy()
    try:
        check(); a.directory.mkdir()
        for index,site in enumerate(SITES):
            G = get(f'model.layers.{site}.mamba.in_proj.weight')
            with np.load(b['exports'][index]['file']['path'],allow_pickle=False) as z:
                p = {k:z[k].copy() for k in ('cw','cb','db')}
            fit = [r for r in b['records'] if r['split'] == 'FIT']
            H = np.concatenate([data(r,index)[0] for r in fit])
            targets = np.concatenate([data(r,index)[1] for r in fit])
            ids = np.concatenate([np.array(r['input_ids'],'u4') for r in fit])
            known,inv = np.unique(ids,return_inverse=True)
            mean = np.zeros((len(known),2048)); np.add.at(mean,inv,H)
            mean /= np.bincount(inv)[:,None]; prototypes = bf(mean,np)
            global_bits = bf(H.mean(axis=0),np); global_mean = decode(global_bits,np)
            weight = np.empty(6704)
            for name,(lo,hi) in GROUPS.items():
                energy = float(np.mean(np.sum(decode(targets[:,lo:hi],np)**2,axis=1)))
                assert energy > 0; weight[lo:hi] = 1/np.sqrt(energy)
            mup = decode(np.array(b['MuP_bits'],'u2'),np)
            for method in b['methods']:
                stage = f'{site}/{method}/FIT'; guard()
                R = H-(decode(prototypes[inv],np) if method == 'FIT_token_mean' else global_mean)
                C = R.T@R/len(R); ev,U = np.linalg.eigh(C); cutoff = ev[-1]*b['covariance_cutoff_relative']
                support = ev > cutoff; assert support.any()
                roots = np.sqrt(np.maximum(ev,0)); roots[~support] = 0
                inverse = np.zeros_like(roots); inverse[support] = 1/roots[support]
                sqrtC = (U*roots)@U.T; invsqrtC = (U*inverse)@U.T
                M = (G*(mup*weight)[:,None])@sqrtC
                metric = M.T@M; vals,V = np.linalg.eigh(metric)
                order = np.argsort(vals)[::-1]; vals = vals[order]; V = V[:,order[:max(RANKS)]]
                Q = invsqrtC@V; L = G@sqrtC@V
                bias = decode(prototypes,np)@G.T if method == 'FIT_token_mean' else None
                global_bias = global_mean@G.T
                path = a.directory/f'site{site:02d}.{method}.npz'
                np.savez(path,known=known,prototype=prototypes,global_bits=global_bits,
                         C=C,sqrtC=sqrtC,Q=Q,L=L,metric_eigenvalues=vals,
                         weight=weight,support=np.array([support.sum()],'u4'))
                model = dict(site=site,method=method,file=extent(path),FIT_tokens=len(H),
                    distinct_FIT_tokens=len(known),covariance_support=int(support.sum()),
                    residual_energy_ratio=float(np.sum(R**2)/np.sum(H**2)),selected_rank=None)
                local = []
                for rec in b['records']:
                    stage = f'{site}/{method}/{rec["id"]}'; guard()
                    h,reference = data(rec,index); tags = np.array(rec['input_ids'],'u4')
                    loc = np.searchsorted(known,tags); seen = loc < len(known)
                    seen[seen] &= known[loc[seen]] == tags[seen]
                    mu = np.broadcast_to(global_mean,h.shape).copy()
                    base = np.broadcast_to(global_bias,(len(h),6704)).copy()
                    if method == 'FIT_token_mean':
                        mu[seen] = decode(prototypes[loc[seen]],np); base[seen] = bias[loc[seen]]
                    residual = h-mu; coordinates = residual@Q
                    dense = bf(decode(bf(h@G.T,np),np)*mup,np)
                    ctrl = errors(dense,reference,np)
                    reference_fields = causal(reference,p,np)
                    row = dict(id=rec['id'],split=rec['split'],domain=rec['domain'],site=site,method=method,
                        history=len(h),seen_token_positions=int(seen.sum()),unknown_positions=int((~seen).sum()),
                        dense_control=ctrl,candidates=[])
                    for rank in RANKS:
                        raw = base.copy() if rank == 0 else base+coordinates[:,:rank]@L[:,:rank].T
                        pred = bf(decode(bf(raw,np),np)*mup,np)
                        pre,post = errors(pred,reference,np),errors(causal(pred,p,np),reference_fields,np)
                        row['candidates'].append(dict(rank=rank,preconv=pre,causal=post,
                            all_fields_pass=bool(max(pre.values()) <= .01 and max(post.values()) <= .01)))
                    local.append(row)
                for rank in RANKS:
                    if all(next(c for c in row['candidates'] if c['rank'] == rank)['all_fields_pass']
                           for row in local if row['split'] == 'FIT'):
                        model['selected_rank'] = rank; break
                selected = model['selected_rank']
                model['DEV_selected_pass'] = bool(selected is not None and all(
                    next(c for c in row['candidates'] if c['rank'] == selected)['all_fields_pass']
                    for row in local if row['split'] == 'DEV'))
                model['compact_candidate'] = bool(selected is not None and selected <= 256 and model['DEV_selected_pass'])
                models.append(model); results += local
                print(json.dumps(dict(site=site,method=method,selected_rank=selected,
                    DEV_selected_pass=model['DEV_selected_pass'],compact=model['compact_candidate'],
                    seconds=time.monotonic()-start)),flush=True)
                del R,C,sqrtC,invsqrtC,M,metric,Q,L,bias,U,V
            del H,targets,mean,prototypes,G
        check(); guard()
        result = dict(schema='TOKEN_CONTEXT_RANK_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
            models=models,records=results,source_model_forwards=0,GPU_calls=0,SSM_calls=0,
            RESERVED_queries=0,optimizer_updates=0,before_after_hashes_exact=True,
            all_dense_source_control_pass=all(max(row['dense_control'].values()) <= b['dense_source_control_gate'] for row in results),
            decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',seconds=time.monotonic()-start,
            OS_peak=proc.memory_info().peak_wset,output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()),
            scope=b['scope'])
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(result=extent(a.out),models=models)),flush=True)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage,traceback=traceback.format_exc()); raise
    finally:
        stream.close(); receipt.update(seconds=time.monotonic()-start,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true'); modes.add_argument('--run',action='store_true')
    for k in ('binding','directory','out'): p.add_argument('--'+k,type=Path)
    for k in ('binding-sha','freeze'): p.add_argument('--'+k,required=k == 'freeze')
    a = p.parse_args()
    if a.run: run(a)
    else: prepare(a)
