"""Same failed zero-readout anchor: analytic/complex derivative and stable RMS delta."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import meth437_switch_shared_readout_math as H
M, R, G = H.R.M, H.R, H.G
PROTOCOL = M.DOC/'METH_439_SWITCH_READOUT_FD_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth439_switch_readout_fd'
PRIOR = M.DOC/'meth438_switch_shared_readout_result.failure.json'
PRIOR_SHA = '81e4da2bb5d2f48b3f48f3643da0128fc9d5f038456b217d66d08a168f2b2590'


def context(*args):
    source = inspect.getsource(H.qualify); needle = "    for k in ('C', 'D', 'feature', 'base', 'scores'):"
    assert source.count(needle) == 1
    namespace = vars(H).copy(); exec(source.split(needle)[0]+"    return locals()\n", namespace)
    return namespace['qualify'](*args)


def diagnose(label, feature, base, scores, chosen, buffers, fn, head, params, target, path, expected, reproduce=False):
    c = context(label, feature, base, scores, chosen, buffers, fn, head, params, target, False, path, expected)
    assert np.count_nonzero(c['pv']['C']) == 0
    rng = np.random.default_rng(437+len(scores)); arrays = {}; checks = []; probability = c['probability']; maximum = c['maximum']; anchor_logits = c['anchor_logits']; b = c['b']; nf = c['nf']; hw = c['hw']; scale = c['scale']
    m = np.mean(b*b)+1e-6; r = np.sqrt(m)
    def relative_CE(delta):
        delta = delta-delta[maximum]; return np.log1p(np.sum(probability*np.expm1(delta)))-np.dot(target, delta)
    for k in ('C', 'D', 'feature', 'base', 'scores'):
        value = {'feature': c['f'], 'base': b, 'scores': c['s']}.get(k, c['pp'].get(k)); direction = rng.choice([-1., 1.], size=value.shape)/np.sqrt(value.size)
        def old(t):
            v = value+t*direction; z = c['numpy_path'](v if k == 'feature' else c['f'], v if k == 'base' else b, v if k == 'scores' else c['s'], c['pp'] | {k: v} if k in c['pp'] else c['pp'])
            return relative_CE(z['logits']-anchor_logits)
        velocity = np.float64(c['q']['probability'])*(direction@c['q']['low']) if k == 'C' else direction if k == 'base' else np.zeros_like(b)
        def delta_logits(t):
            delta_post = t*velocity; dm = np.mean(2*b*delta_post+delta_post*delta_post); rn = np.sqrt(m+dm)
            inverse_difference = -dm/(r*rn*(r+rn))
            delta_final = nf*(b*inverse_difference+delta_post/rn)
            return hw@(delta_final*scale)
        analytic = float(np.dot(probability-target, hw@(nf*(velocity/r-b*np.mean(b*velocity)/(r*r*r))*scale)))
        complex_derivative = float(np.imag(relative_CE(delta_logits(1e-20j)))/1e-20)
        old_fd = float((old(1e-4)-old(-1e-4))/(2e-4)); pr = float(np.sum(c['rg'][k]*direction)); pn = float(np.sum(c['ng'][k]*direction))
        old_error = abs(old_fd-pr)/max(abs(pr), 1e-8)
        if reproduce and k == 'C': assert abs(old_error-3.690171289699145e-5) <= 1e-12, ('failed438_not_exact', old_error)
        derivatives = [float((relative_CE(delta_logits(h))-relative_CE(delta_logits(-h)))/(2*h)) for h in (1e-4, 5e-5)]
        richardson = (4*derivatives[1]-derivatives[0])/3
        er = abs(richardson-pr)/max(abs(pr), 1e-8); en = abs(richardson-pn)/max(abs(pn), 1e-8)
        ac = abs(analytic-pr)/max(abs(pr), 1e-8); cc = abs(complex_derivative-pr)/max(abs(pr), 1e-8)
        assert max(er, ac, cc) <= 1e-5 and en <= 1e-3, ('independent_readout_derivative', label, k, er, en, ac, cc)
        arrays['direction_'+k] = direction; arrays['delta_post_velocity_'+k] = velocity
        checks.append(dict(field=k, old_FD=old_fd, old_reference_error=old_error, stable_FD_h=derivatives[0], stable_FD_half_h=derivatives[1],
            richardson=richardson, reference_projection=pr, native_projection=pn, analytic=analytic, complex_step=complex_derivative,
            reference_Richardson_error=er, native_Richardson_error=en, analytic_reference_error=ac, complex_reference_error=cc))
    assert all(v.tobytes() == c['fixed'][k] for k, v in c['offsets'].items())
    for k in c['q']: arrays['native_'+k] = c['native'][k]; arrays['continuation_'+k] = c['npout'][k]; arrays['offset_'+k] = c['offsets'][k]
    for k in c['rg']: arrays['native_gradient_'+k] = c['ng'][k]; arrays['reference_gradient_'+k] = c['rg'][k]
    np.savez(path, **arrays, **{'parameter_'+k: v for k, v in c['pv'].items()}, feature=feature, base=base, scores=scores, target=target, mean=buffers[0], std=buffers[1], finalnorm=fn)
    return dict(label=label, endpoints=c['endpoints'], gradient_errors=c['ge'], FD=checks, exact438_failure_reproduced=reproduce, offsets_fixed=True, zero_post_full_head_exact=True,
        stable_rationalized_RMS_delta_and_head_before_loss_difference=True, fixed_h_and_half_h_Richardson=True, independent_analytic_complex_derivatives=True, archive_path=str(path), approximate_derivative_not_rounding=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; result = {'experiment': 'METH-439-same-zero-readout-FD-diagnosis-no-fit'}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0)); assert peak <= 3 << 30 and time.monotonic()-start <= 180, 'diagnostic_180sec_3GiB'
    def sha(p):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(p).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        result['helper_sha256'] = {}
        for p in (Path(__file__), PROTOCOL, Path(H.__file__), Path(M.__file__), Path(R.__file__), Path(G.__file__), Path(H.L.__file__)):
            M.committed(p); result['helper_sha256'][str(p)] = sha(p)
        M.committed(PRIOR); assert sha(PRIOR) == PRIOR_SHA; parent = json.loads(PRIOR.read_text(encoding='utf-8')); assert parent['updates'] == [] and parent['failure_stage'] == 'adapter_real_initial_composed_contract'
        for p, e in parent['helper_sha256'].items(): M.committed(Path(p)); assert sha(p) == e
        for a in parent['partial_output_inventory']: assert sha(a['path']) == a['sha256']
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): result.setdefault('preserved_daemons', []).append(p.pid); continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        a = parent['artifacts']['256']; assert sha(a['payload']) == a['sha256'] and sha(a['manifest']) == a['manifest_sha256']; result['artifacts'] = {'256': a}
        name, expected = R.C.U.EXPORT[256]; p = M.DOC/name; M.committed(p); assert sha(p) == expected; export = json.loads(p.read_text(encoding='utf-8')); R.B.read_manifest(a['manifest'], export['original_config'], export['tensors'], Path(a['payload']))
        mapped = np.memmap(a['payload'], dtype='u1', mode='r'); entries = export['tensors']; head = G.I8Operator(R.C.tensor(mapped, entries, 'lm_head.weight'), R.C.tensor(mapped, entries, 'lm_head.weight', 'scales')); fn = R.C.tensor(mapped, entries, 'decoder.final_layer_norm.weight')
        prior_path = M.DOC/'meth418_switch_function_capture_result.json'; M.committed(prior_path); assert sha(prior_path) == parent['retained_record_sha256'][prior_path.name]; captures = json.loads(prior_path.read_text(encoding='utf-8')); cap = {x['label']: x for x in captures['captures']}; rows = {}; feature = []
        for bi in range(18):
            for ci in range(4):
                x = cap[f'teacher.n128.book{bi}.case{ci}']; assert sha(x['capture_path']) == x['capture_sha256']; z = np.frombuffer(Path(x['capture_path']).read_bytes(), dtype=R.C.DTYPE, offset=32); feature.append(z['input'].copy())
                if bi == ci == 0: rows[128] = z[0].copy()
        x = cap['teacher.n256.book0.case0']; assert sha(x['capture_path']) == x['capture_sha256']; rows[256] = np.frombuffer(Path(x['capture_path']).read_bytes(), dtype=R.C.DTYPE, offset=32)[0].copy()
        first = rows[256]; trace = Path(x['trace_path']).read_bytes(); assert sha(x['trace_path']) == x['trace_sha256']; dt = np.dtype([('index','<u4'),('phase','<u4'),('input','<f4',(768,)),('scores','<f4',(256,))]); scores = np.frombuffer(trace,dtype=dt,offset=16)[179]['scores'].copy()
        baseline_path = M.DOC/'meth420_switch_function_gradient_result.failure.json'; M.committed(baseline_path); assert sha(baseline_path) == parent['retained_record_sha256'][baseline_path.name]; baseline = json.loads(baseline_path.read_text(encoding='utf-8')); archived = next(v for v in baseline['baselines'] if v['label'] == 'teacher.n256.book0.case0'); assert sha(archived['archive_path']) == archived['archive_sha256']
        with np.load(archived['archive_path'],allow_pickle=False) as z: logits = z['logits'][0].copy()
        values = np.concatenate(feature); buffers = (np.mean(values.astype(np.float64),axis=0).astype(np.float32),np.maximum(np.std(values.astype(np.float64),axis=0),1e-6).astype(np.float32)); target = np.zeros(32128,np.float64); target[(int(np.argmax(logits))+16064)%32128] = 1
        assert psutil.disk_usage(str(M.ROOT)).free >= 2 << 30; OUT.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime'] = {'torch':torch.__version__,'numpy':np.__version__,'CPU_affinity':[0],'BLAS':threadpool_info(),'GPU':False}; stage='same_failed_adapter_anchor'
            result['diagnosis'] = diagnose('same_failed438_adapter',rows[128]['input'],first['post'],scores,int(first['expert']),buffers,fn,head,H.parameters(768,32,437),target,OUT/'same_failed_adapter.npz',logits,True); guard()
        result['prior438_failure_sha256'] = PRIOR_SHA; result['apparatus_gates'] = {'failed438_C_FD_exact_reproduced':True,'ordinary_F64_all_five_fields_match_native':True,'analytic_and_complex_step_all_directions_same_bounds':True,'stable_RMS_delta_and_fixed_Richardson_same_bounds':True,'no_updates_or_scientific_recipe_changed':True}
        result['output_inventory'] = [{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.glob('*.npz')]; assert sum(a['bytes'] for a in result['output_inventory']) <= 16 << 20
        result['resource'] = {'seconds':time.monotonic()-start,'peak_bytes':peak,'bytes_hashed':hashed,'optimizer_updates':0}; result['decision'] = 'numeric_zero_readout_qualification_repair_licensed_ONLY_new_controller_same_fit_configuration'
        guard(); M.write(args.out,result); print(json.dumps({'gates':result['apparatus_gates'],'checks':result['diagnosis']['FD'],'resource':result['resource'],'sha256':M.digest(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,peak_bytes=peak,bytes_hashed=hashed);M.write(args.out.with_suffix('.failure.json'),result);raise


if __name__ == '__main__': main()
