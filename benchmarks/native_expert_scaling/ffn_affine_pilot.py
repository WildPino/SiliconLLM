"""FIT-only full-coordinate FFN Taylor cells, exact implicit Jacobian pilot."""
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
SITES = [0,12,23]
COUNTS = [1,16,64]


def dec(x,np): return (x.astype('u4') << 16).view('f4').astype('f8')


def sigmoid(x,np): return np.exp(-np.logaddexp(0,-x))


def assign(X,C,np):
    distance = np.sum(X*X,axis=1)[:,None]+np.sum(C*C,axis=1)[None,:]-2*(X@C.T)
    labels = np.argmin(distance,axis=1)
    return labels,np.maximum(distance[np.arange(len(X)),labels],0)


def centers(X,k,np,guard):
    if k == 1: return X.mean(axis=0,keepdims=True),1
    rng = np.random.default_rng(0); C = [X[int(rng.integers(len(X)))].copy()]
    _,d = assign(X,np.array(C),np)
    for i in range(1,k):
        guard(); assert d.sum() > 0
        idx = int(np.searchsorted(np.cumsum(d),rng.random()*d.sum(),side='right'))
        C.append(X[min(idx,len(X)-1)].copy()); _,new = assign(X,np.array(C[-1:]),np); d = np.minimum(d,new)
    C = np.array(C); previous = None
    for iteration in range(25):
        guard(); labels,d = assign(X,C,np)
        if previous is not None and np.array_equal(labels,previous): return C,iteration+1
        previous = labels.copy(); sums = np.zeros_like(C); np.add.at(sums,labels,X)
        sizes = np.bincount(labels,minlength=k)
        for i in range(k):
            if sizes[i]: C[i] = sums[i]/sizes[i]
            else: C[i] = X[int(np.argmax(d))]; d[np.argmax(d)] = -1
    return C,25


def aggregate(rows,split):
    selected = [r for r in rows if r['split'] == split]
    return dict(case_mean=sum(r['error_to_BF'] for r in selected)/len(selected),
                worst=max(r['error_to_BF'] for r in selected),
                proxy_case_mean=sum(r['error_to_proxy'] for r in selected)/len(selected))


def prepare(a):
    import numpy as np
    import psutil
    assert not a.out.exists() and not a.directory.exists()
    rp = DOC/'ffn_operand_capture_result_20261013.json'; ap = DOC/'ffn_operand_adjudication_20261013.json'
    cp = DOC/'ffn_operand_session_closure_20261013.json'
    r,ad,cl = [json.loads(p.read_bytes()) for p in (rp,ap,cp)]
    assert ad['decision'] == 'INDEPENDENT_STORED_PASS' and ad['report'] == extent(rp)
    assert cl['decision'] == 'ACTUAL_OWN_ALL24_FFN_OPERANDS_QUALIFIED' and cl['audit'] == extent(ap)
    pb = json.loads(Path(r['binding']['path']).read_bytes())
    records = [dict(id=x['id'],split=x['split'],domain=x['domain'],history=x['history'],
        fields=[x['sites'][site] for site in SITES]) for x in r['records']]
    assert len(records) == 48 and sum(x['history'] for x in records) == 22547
    paths = [Path(__file__),B/'ffn_affine_pilot_audit.py',B/'falcon_auxiliary_s6_transport.py',
        DOC/'FFN_AFFINE_PILOT_PROTOCOL_20261013.md',rp,ap,cp,Path(r['binding']['path'])]
    paths += [Path(e['path']) for e in pb['inputs'] if Path(e['path']).suffix.lower() in ('.py','.dll','.pyd','.exe','.safetensors')]
    paths += [Path(e['path']) for rec in records for row in rec['fields'] for e in row.values()]
    inputs = [extent(p) for p in dict.fromkeys(paths)]
    parent_inputs = {e['path']:e for e in pb['inputs']+r['outputs']}
    for e in inputs:
        if e['path'] in parent_inputs: assert e == parent_inputs[e['path']]
    binding = dict(schema='FFN_AFFINE_PILOT_BINDING_V1',freeze=a.freeze,
        python=pb['python'],versions=dict(numpy=np.__version__,psutil=psutil.__version__),
        source=str(Path(pb['source'])/'model.safetensors'),sites=SITES,counts=COUNTS,
        alpha_effective_F32=float(np.float32(pb['gate_multiplier'])),beta_effective_F32=float(np.float32(pb['down_multiplier'])),
        records=records,inputs=inputs,parent_report=extent(rp),parent_audit=extent(ap),parent_closure=extent(cp),
        output_directory=str(a.directory.resolve()),
        gates=dict(numerical_proxy_each_case_relative=.01,FIT_case_mean=.10,FIT_worst=.20,
                   DEV_case_mean=.10,DEV_worst=.20,DEV_domain_case_mean=.20),
        limits=dict(seconds=1800,audit_seconds=1800,OS_bytes=8 << 30,output_bytes=2 << 30),
        routing='Full2048 Euclidean nearest FIT center;one uint32 ID/mass exactly1. '
                'Flat centroid scan is a pilot,not scalable CPU routing.',
        scope='Three actual own-input FFN sites,all48 histories. Full-output affine '
              'Taylor cells with exact implicit Jacobians,not physical compiled bank. '
              'Local eligibility only;no whole-model substitution/GPU/RESERVED/TPS/useful-n claim.')
    write(a.out,binding); print(json.dumps(dict(binding=extent(a.out))),flush=True)


def run(a):
    import numpy as np
    import psutil
    begin = time.monotonic(); proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    assert extent(a.binding)['sha256'] == a.binding_sha
    b = json.loads(a.binding.read_bytes()); assert not a.out.exists() and not a.directory.exists()
    assert a.directory.resolve() == Path(b['output_directory']).resolve()
    assert dict(numpy=np.__version__,psutil=psutil.__version__) == b['versions']
    get,stream = tensor_reader(b['source'],np); stage = 'hashes'; results = []; models = []; controls = []
    receipt = dict(schema='FFN_AFFINE_PILOT_TERMINAL_V1',command=sys.orig_argv,pid=proc.pid,
                   creation_time=proc.create_time(),freeze=a.freeze)
    def guard():
        assert time.monotonic()-begin <= b['limits']['seconds'] and proc.memory_info().peak_wset <= b['limits']['OS_bytes']
        assert not proc.children(recursive=True)
        if a.directory.exists(): assert sum(p.stat().st_size for p in a.directory.iterdir()) <= b['limits']['output_bytes']
    def check():
        for e in b['inputs']: guard(); assert extent(e['path']) == e,e['path']
    def data(rec,index,name):
        e = rec['fields'][index][name]
        return dec(np.memmap(e['path'],'u2','r',shape=tuple(e['shape'])),np)
    def rel(x,y): return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
    try:
        check(); a.directory.mkdir(); alpha,beta = b['alpha_effective_F32'],b['beta_effective_F32']
        for index,site in enumerate(SITES):
            pre = f'model.layers.{site}.feed_forward.'
            U,G,D = [get(pre+name+'.weight') for name in ('up_proj','gate_proj','down_proj')]
            assert U.shape == G.shape == (4608,2048) and D.shape == (2048,4608)
            fit = np.concatenate([data(rec,index,'x') for rec in b['records'] if rec['split'] == 'FIT'])
            bank = []
            for count in COUNTS:
                stage = f'{site}/{count}/centers'; guard(); C,it = centers(fit,count,np,guard)
                uc = C@U.T; gc = (C@G.T)*alpha; sig = sigmoid(gc,np)
                s = gc*sig; derivative = sig+gc*sig*(1-sig)
                offset = (uc*s)@D.T*beta
                farthest = int(np.argmax(np.sum((fit-C[0])**2,axis=1)))
                direction = fit[farthest]-C[0]; assert np.linalg.norm(direction) > 0
                direction /= np.linalg.norm(direction)
                action = (s[0]*(direction@U.T)+uc[0]*derivative[0]*(direction@G.T)*alpha)@D.T*beta
                def evaluate(x):
                    u = x@U.T; g = x@G.T*alpha
                    return (u*g*sigmoid(g,np))@D.T*beta
                finite = (evaluate(C[0]+1e-4*direction)-evaluate(C[0]-1e-4*direction))/(2e-4)
                derivative_error = rel(action,finite); assert derivative_error <= 1e-5
                path = a.directory/f'site{site:02d}.k{count:03d}.npz'
                np.savez(path,centers=C,u_center=uc,g_center=gc,s=s,derivative=derivative,offset=offset,
                         direction=direction,jacobian_action=action,finite_difference=finite)
                model = dict(site=site,count=count,file=extent(path),kmeans_iterations=it,
                             directional_derivative_relative=derivative_error)
                bank.append((model,C,uc,gc,s,derivative))
                print(json.dumps(dict(site=site,centers=count,iterations=it,seconds=time.monotonic()-begin)),flush=True)
            for rec in b['records']:
                stage = f'{site}/{rec["id"]}'; guard(); X = data(rec,index,'x'); actual = data(rec,index,'output')
                ux = X@U.T; gx = (X@G.T)*alpha
                proxy = (ux*(gx*sigmoid(gx,np)))@D.T*beta
                control = rel(proxy,actual)
                controls.append(dict(site=site,id=rec['id'],split=rec['split'],error=control,pass_gate=control <= .01))
                for model,C,uc,gc,s,derivative in bank:
                    labels,radius = assign(X,C,np)
                    # Exact real-arithmetic F(mu)+J(mu)(x-mu),with full output.
                    affine_features = s[labels]*ux+uc[labels]*derivative[labels]*(gx-gc[labels])
                    prediction = affine_features@D.T*beta
                    path = a.directory/f'{rec["id"]}.site{site:02d}.k{model["count"]:03d}.npz'
                    np.savez(path,labels=labels.astype('u4'),mass=np.ones(len(X),'f8'),
                             prediction=prediction.astype('f4'),radius_squared=radius)
                    results.append(dict(site=site,count=model['count'],id=rec['id'],split=rec['split'],domain=rec['domain'],
                        history=len(X),file=extent(path),error_to_BF=rel(prediction,actual),error_to_proxy=rel(prediction,proxy),
                        relative_radius_RMS=float(np.sqrt(radius.sum()/max(np.sum(X*X),1e-30)))))
            selected = None
            for model, *_ in bank:
                local = [r for r in results if r['site'] == site and r['count'] == model['count']]
                fi,dv = aggregate(local,'FIT'),aggregate(local,'DEV')
                domains = {domain:sum(r['error_to_BF'] for r in local if r['split'] == 'DEV' and r['domain'] == domain) /
                           sum(r['split'] == 'DEV' and r['domain'] == domain for r in local)
                           for domain in sorted({r['domain'] for r in local if r['split'] == 'DEV'})}
                fit_pass = fi['case_mean'] <= .10 and fi['worst'] <= .20
                if selected is None and fit_pass: selected = model['count']
                model.update(FIT=fi,DEV=dv,DEV_domains=domains,FIT_eligible=fit_pass,
                    DEV_eligible=dv['case_mean'] <= .10 and dv['worst'] <= .20 and max(domains.values()) <= .20)
                models.append(model)
                print(json.dumps(dict(site=site,count=model['count'],FIT=fi,DEV=dv,FIT_eligible=fit_pass,
                                      DEV_eligible=model['DEV_eligible'],seconds=time.monotonic()-begin)),flush=True)
            for model,*_ in bank: model['FIT_selected_count'] = selected
            del fit,U,G,D,bank
        check(); guard()
        result = dict(schema='FFN_AFFINE_PILOT_RESULT_V1',binding=extent(a.binding),freeze=a.freeze,
            records=results,models=models,controls=controls,all_proxy_controls_pass=all(c['pass_gate'] for c in controls),
            source_model_forwards=0,GPU_calls=0,RESERVED_queries=0,optimizer_updates=0,
            before_after_hashes_exact=True,decision='COMPLETE_PENDING_INDEPENDENT_AUDIT',scope=b['scope'],
            seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset,
            output_bytes=sum(p.stat().st_size for p in a.directory.iterdir()))
        assert len(results) == 432 and len(controls) == 144 and len(models) == 9
        write(a.out,result); receipt.update(exit_code=0,error=None,result=extent(a.out))
        print(json.dumps(dict(result=extent(a.out),models=models,proxy_controls_pass=result['all_proxy_controls_pass'])),flush=True)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage,traceback=traceback.format_exc()); raise
    finally:
        stream.close(); receipt.update(seconds=time.monotonic()-begin,OS_peak=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.terminal.json'),receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true'); m.add_argument('--run',action='store_true')
    for k in ('binding','directory','out'): p.add_argument('--'+k,type=Path)
    for k in ('binding-sha','freeze'): p.add_argument('--'+k,required=k == 'freeze')
    a = p.parse_args()
    if a.prepare: prepare(a)
    else: run(a)
