"""Bounded two-arm final-bank retargeting pilot; calibration, not model release."""
import argparse
from collections import OrderedDict
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import meth324_switch_reference as M
import meth422_switch_function_gradient_contract as R
import meth422_switch_function_autograd as G
import meth424_switch_matched_primal_contract as K
import meth426_switch_pilot_math as H

PROTOCOL = M.DOC/'METH_426_SWITCH_FUNCTION_PILOT_PROTOCOL_20261004.md'
OUT = M.ROOT/'results/native_expert_scaling/meth426_switch_function_pilot'
QUAL = OUT/'qualification'
PRIOR424 = M.DOC/'meth424_switch_matched_primal_result.json'
PRIOR424_SHA = '3312dd9dc50492042839a0a557898f704cb1fc435bd3e14745033c0af04d525a'
EPOCHS = 3
BATCH = 32


class ExpertCache:
    def __init__(self, mapped, entries):
        self.mapped = mapped; self.entries = entries; self.cache = OrderedDict(); self.hits = 0; self.misses = 0
    def get(self, identity):
        assert 0 <= identity < 128
        if identity in self.cache:
            self.hits += 1; self.cache.move_to_end(identity); return self.cache[identity]
        self.misses += 1
        if len(self.cache) == 8: self.cache.popitem(last=False)
        ep = f'decoder.block.11.layer.2.mlp.experts.expert_{identity}.'
        ops = tuple(G.I8Operator(R.C.tensor(self.mapped, self.entries, ep+s+'.weight'),
                                  R.C.tensor(self.mapped, self.entries, ep+s+'.weight', 'scales')) for s in ('wi', 'wo'))
        self.cache[identity] = ops; return ops


def probability_rows(logits):
    out = np.empty(logits.shape, np.float64)
    for i in range(len(logits)): out[i] = R.probability(logits[i])
    return out


def load_pairs(prior, baseline, guard):
    cap = {a['label']: a for a in prior['captures']}; old = {a['label']: a for a in baseline['baselines']}
    pre = []; inputs = []; scores = []; old_ids = []; labels = []; books = []; positions = []; keys = []
    logits = {n: np.empty((1344, 32128), np.float32) for n in (256, 128)}; first_rows = {}; cursor = 0
    for bi in range(24):
        for ci in range(4):
            pairs = [cap[f'teacher.n{n}.book{bi}.case{ci}'] for n in (256, 128)]
            assert pairs[0]['pairing_sha256'] == pairs[1]['pairing_sha256'] and pairs[0]['decoder_ids'] == pairs[1]['decoder_ids']
            rows = {}
            for n, item in zip((256, 128), pairs):
                assert item['prospective_split'] == ('development' if bi < 18 else 'validation')
                a = old[item['label']]; assert not a['qualification_only_not_paired_training']
                assert a['source_capture_sha256'] == item['capture_sha256']
                data = Path(item['capture_path']).read_bytes(); assert data[:8] == b'SWFUN001' and len(data) == 32+14*R.C.DTYPE.itemsize
                rows[n] = np.frombuffer(data, dtype=R.C.DTYPE, offset=32).copy()
                assert np.array_equal(rows[n]['position'], np.arange(14))
                with np.load(a['archive_path'], allow_pickle=False) as z:
                    assert np.array_equal(z['source_ids'], item['source_ids']) and np.array_equal(z['decoder_ids'], item['decoder_ids'])
                    for field in ('input', 'up_raw', 'up', 'down', 'probability', 'post', 'final', 'head_input'): R.C.exact(z[field], rows[n][field])
                    assert z['logits'].shape == (14, 32128); logits[n][cursor:cursor+14] = z['logits']
                if bi == ci == 0: first_rows[n] = rows[n][0].copy()
            trace = Path(pairs[0]['trace_path']).read_bytes(); dt = np.dtype([('index', '<u4'), ('phase', '<u4'), ('input', '<f4', (768,)), ('scores', '<f4', (256,))])
            assert trace[:8] == b'SWRTA001' and struct.unpack_from('<2I', trace, 8) == (256, 768)
            traced = np.frombuffer(trace, dtype=dt, offset=16)[179+6*np.arange(14)]
            R.C.exact(traced['input'], rows[256]['input']); assert np.array_equal(np.argmax(traced['scores'], axis=1), rows[256]['expert'])
            pre.append(rows[256]['pre']); inputs.append(rows[256]['input']); scores.append(traced['scores'].copy())
            old_ids.extend(rows[256]['expert'].tolist()); labels.extend(rows[128]['expert'].tolist()); books.extend([bi]*14); positions.extend(range(14))
            keys.extend([f'book{bi}.case{ci}.position{j}:{pairs[0]["pairing_sha256"]}' for j in range(14)])
            cursor += 14; guard()
    assert cursor == 1344
    p256 = probability_rows(logits[256]); guard(); p128 = probability_rows(logits[128]); del logits[128]; gc.collect(); guard()
    data = dict(pre=np.concatenate(pre), input=np.concatenate(inputs), scores=np.concatenate(scores), old_ids=np.asarray(old_ids),
                labels=np.asarray(labels), books=np.asarray(books), positions=np.asarray(positions), keys=keys,
                logits=logits[256], p256=p256, p128=p128, first_rows=first_rows)
    data['dev'] = np.flatnonzero(data['books'] < 18); data['val'] = np.flatnonzero(data['books'] >= 18)
    assert len(data['dev']) == 1008 and len(data['val']) == 336
    return data


def losses(logits, indices, data):
    out = np.empty((len(indices), 3), np.float64)
    for j, i in enumerate(indices):
        z = logits[j].astype(np.float64)
        out[j, 0] = R.numpy_loss(z, data['p256'][i]); out[j, 1] = R.numpy_loss(z, data['p128'][i]); out[j, 2] = .5*(out[j, 0]+out[j, 1])
    assert np.isfinite(out).all(); return out


def optimizer(parameters, lr):
    return torch.optim.Adam(parameters, lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)


def step(loss, opt, parameters):
    assert torch.isfinite(loss).all(), 'nonfinite_loss'
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    opt.step()
    for p in parameters:
        assert torch.isfinite(p).all(), 'nonfinite_coefficient'
        if p in opt.state:
            for field in ('exp_avg', 'exp_avg_sq'): assert torch.isfinite(opt.state[p][field]).all(), 'nonfinite_Adam_state'
    return float(norm)


def fit_affine(name, weight, bias, indices, inputs, labels, binary, result, guard):
    opt = optimizer([weight, bias], .003)
    for epoch in range(EPOCHS):
        order = np.random.default_rng(425+(200 if binary else 100)+epoch).permutation(indices); total = 0.
        for start in range(0, len(order), BATCH):
            ids = order[start:start+BATCH]; opt.zero_grad(set_to_none=True); batch_loss = []
            for i in ids:
                z = H.affine(torch.from_numpy(inputs[i]), weight, bias)
                if binary:
                    value = z[0].to(torch.float64); loss = torch.nn.functional.softplus(-value if labels[i] else value)
                else:
                    target = torch.zeros(128, dtype=torch.float64); target[int(labels[i])] = 1.; loss = G.prediction_loss(z, target)
                batch_loss.append(loss)
            loss = torch.stack(batch_loss).mean(); norm = step(loss, opt, [weight, bias]); total += float(loss.detach())*len(ids)
            result['updates'].append({'stage': name, 'epoch': epoch, 'batch_start': start, 'positions': len(ids), 'loss': float(loss.detach()), 'gradient_norm_before_clip': norm})
            guard()
        result['epochs'].append({'stage': name, 'epoch': epoch, 'positions': len(order), 'mean_loss': total/len(order)})
        print(json.dumps(result['epochs'][-1]), flush=True)


def predict_affine(weight, bias, inputs, binary):
    with torch.no_grad():
        values = np.stack([H.affine(torch.from_numpy(x), weight, bias).numpy() for x in inputs])
    assert np.isfinite(values).all()
    return (values[:, 0] >= 0 if binary else np.argmax(values, axis=1)), values


def forward_arm(arm, index, identity, data, norm, head, factors, cache, permutation=False, removal=False):
    pre = torch.from_numpy(data['pre'][index]); scores = torch.from_numpy(data['scores'][index]); chosen = int(data['old_ids'][index])
    if removal:
        final = G.NativeRMS.apply(pre, norm[1]); hi = final*np.float32(1/np.sqrt(768)); return G.NativeI8.apply(hi, head)
    if arm == 'adapter': return H.adapter_forward(pre, scores, chosen, *norm, head, factors[identity])['logits']
    wi, wo = cache.get((identity+1)%128 if permutation else identity)
    return G.forward(pre, scores, chosen, *norm, wi, wo, head, factors[identity])['logits']


def fit_factors(arm, data, identities, norm, head, factors, cache, result, guard, save):
    opt = optimizer([z for p in factors for z in p.values()], .001)
    counts = np.zeros(128, np.int64)
    for epoch in range(EPOCHS):
        order = np.random.default_rng(425+epoch).permutation(data['dev']); total = 0.
        for sequence, index in enumerate(order):
            identity = int(identities[index]); opt.zero_grad(set_to_none=True)
            z = forward_arm(arm, index, identity, data, norm, head, factors, cache)
            target = torch.from_numpy(.5*(data['p256'][index]+data['p128'][index]))
            loss = G.prediction_loss(z, target); norm_grad = step(loss, opt, list(factors[identity].values()))
            counts[identity] += 1; total += float(loss.detach())
            result['updates'].append({'stage': arm, 'epoch': epoch, 'sequence': sequence, 'data_index': int(index), 'function_id': identity,
                                      'loss': float(loss.detach()), 'gradient_norm_before_clip': norm_grad})
            guard()
            if (sequence+1)%168 == 0: print(json.dumps({'stage': arm, 'epoch': epoch, 'positions_done': sequence+1, 'mean_loss': total/(sequence+1)}), flush=True)
        result['epochs'].append({'stage': arm, 'epoch': epoch, 'positions': len(order), 'mean_loss': total/len(order)})
        save(); print(json.dumps(result['epochs'][-1]), flush=True)
    return counts.tolist()


def forced(arm, indices, identities, data, norm, head, factors, cache, guard, permutation=False, removal=False):
    out = np.empty((len(indices), 32128), np.float32)
    with torch.no_grad():
        for j, i in enumerate(indices):
            out[j] = forward_arm(arm, i, int(identities[i]), data, norm, head, factors, cache, permutation, removal).numpy()
            guard()
    return out


def summarize(values):
    return {'teacher256_CE': float(np.mean(values[:, 0])), 'teacher128_CE': float(np.mean(values[:, 1])), 'equal_mixture_CE': float(np.mean(values[:, 2]))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; hashed = 0; stage = 'bindings'; state = {}
    result = {'experiment': 'METH-426-stable-relative-CE-FD-same-bounded-final-bank-pilot', 'updates': [], 'epochs': [], 'qualifications': []}
    def guard():
        nonlocal peak
        mi = psutil.Process().memory_info(); peak = max(peak, mi.rss, getattr(mi, 'peak_wset', 0))
        size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
        assert peak <= 4 << 30 and time.monotonic()-start <= 1800 and size <= 256 << 20, 'pilot_30min_4GiB_256MiB'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while block := f.read(4 << 20): h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    def save():
        if not state or not OUT.exists(): return
        arrays = {}
        for arm in ('real', 'adapter'):
            for k in ('A', 'B', 'C', 'D'): arrays[arm+'_'+k] = np.stack([p[k].detach().numpy() for p in state[arm]])
            arrays[arm+'_gate_weight'] = state[arm+'_gate'][0].detach().numpy(); arrays[arm+'_gate_bias'] = state[arm+'_gate'][1].detach().numpy()
        arrays['classifier_weight'] = state['classifier'][0].detach().numpy(); arrays['classifier_bias'] = state['classifier'][1].detach().numpy()
        np.savez(OUT/'pilot_checkpoint.npz', **arrays)
    try:
        helpers = [Path(__file__), Path(H.__file__), PROTOCOL, PRIOR424, Path(K.__file__), Path(K.L.__file__), Path(G.__file__), Path(R.__file__), Path(R.C.__file__), Path(R.B.__file__), Path(M.__file__)]
        result['helper_sha256'] = {}
        for p in helpers: M.committed(p); result['helper_sha256'][str(p)] = sha(p)
        first_failure = M.DOC/'meth425_switch_function_pilot_result.failure.json'; M.committed(first_failure)
        assert sha(first_failure) == 'e990e280464f0497227a4d3ed4f09d5038c36944153eac1a6e0a5839e02846e9'
        failed = json.loads(first_failure.read_text(encoding='utf-8'))
        assert len(failed['updates']) == 0 and 'adapter.tiny' in failed['error'] and 'scores' in failed['error']
        for a in failed['partial_output_inventory']: assert sha(a['path']) == a['sha256']
        for p, expected in failed['helper_sha256'].items(): M.committed(Path(p)); assert sha(p) == expected
        legacy_math = Path(H.__file__).with_name('meth425_switch_pilot_math.py'); M.committed(legacy_math)
        assert Path(H.__file__).read_bytes().split(b'def qualify_adapter(')[0] == legacy_math.read_bytes().split(b'def qualify_adapter(')[0]
        result['retained425_failure_sha256'] = sha(first_failure)
        result['native_and_training_math_unchanged425'] = True
        assert sha(PRIOR424) == PRIOR424_SHA; qualified = json.loads(PRIOR424.read_text(encoding='utf-8')); assert all(qualified['gates'].values())
        for p, expected in qualified['helper_sha256'].items(): M.committed(Path(p)); assert sha(p) == expected
        for a in qualified['output_inventory']: assert sha(a['path']) == a['sha256']
        assert sha(R.PRIOR) == R.PRIOR_SHA; prior = json.loads(R.PRIOR.read_text(encoding='utf-8')); assert all(prior['gates'].values())
        for p, expected in prior['helper_sha256'].items(): M.committed(Path(p)); assert sha(p) == expected
        R.C.source_identity()
        for a in prior['output_inventory']: assert sha(a['path']) == a['sha256'] and Path(a['path']).stat().st_size == a['bytes']
        old_path = M.DOC/'meth420_switch_function_gradient_result.failure.json'; assert sha(old_path) == K.RECORDS[old_path.name]
        baseline = json.loads(old_path.read_text(encoding='utf-8')); assert len(baseline['baselines']) == 384
        for a in baseline['baselines']: assert sha(a['archive_path']) == a['archive_sha256'] and a['complete_native_states_and_ALL32128_logits_byte_exact']
        preserved_binaries = {
            str(M.ROOT/'results/native_expert_scaling/meth374_switch_physical_workers_contract/meth374_switch_physical_workers.exe'): '97965a35900cc0483e60f3e2e51f95da90af600e1fa326af8ec34f7c2a3dacd0',
            str(M.ROOT/'results/native_expert_scaling/meth389_switch_three_workers_contract/meth389_switch_three_workers.exe'): '7783d25e7dd51c6f8ad43750d5d7139e622ba74b852a71aee484ffd445860a7b'}
        for p, expected in preserved_binaries.items(): assert sha(p) == expected
        result['preserved_original_binary_sha256'] = preserved_binaries
        result['inherited_forward'] = qualified['inherited_forward']; result['source424_qualified_sha256'] = PRIOR424_SHA
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if p.pid in own: continue
            name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_background_daemons', []).append(p.pid); continue
            assert not (name.startswith('python') or ('meth' in name and name.endswith('.exe'))), ('concurrent_model_job', p.pid, name)
        psutil.Process().cpu_affinity([0]); torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        td = Path(torch.__file__).parent
        result['runtime'] = {'torch': torch.__version__, 'torch_git': torch.version.git_version, 'numpy': np.__version__, 'CPU_affinity': [0], 'threads': 1,
                              'torch_CPU_DLL_sha256': sha(td/'lib/torch_cpu.dll'), 'torch_C_extension_sha256': sha(next(td.glob('_C*.pyd'))), 'GPU': False}
        exports = {}; mapped = {}; entries = {}; initial = {}
        for n, (name, expected) in R.C.U.EXPORT.items():
            stage = f'fresh_source{n}'; p = M.DOC/name; M.committed(p); assert sha(p) == expected
            e = json.loads(p.read_text(encoding='utf-8')); assert all(e['gates'].values()); a = e['artifact']; exports[n] = a
            payload = Path(a['payload']); initial[n] = (payload.stat().st_size, payload.stat().st_mtime_ns)
            assert a == prior['artifacts'][str(n)] and sha(payload) == a['sha256'] and sha(a['manifest']) == a['manifest_sha256']
            R.B.read_manifest(a['manifest'], e['original_config'], e['tensors'], payload)
            mapped[n] = np.memmap(payload, dtype='u1', mode='r'); entries[n] = e['tensors']
        result['artifacts'] = {str(k): v for k, v in exports.items()}
        assert psutil.disk_usage(str(M.ROOT)).free >= 2 << 30; QUAL.mkdir(parents=True)
        with threadpool_limits(limits=1):
            result['runtime']['BLAS'] = threadpool_info(); assert all(x['num_threads'] == 1 for x in threadpool_info())
            stage = 'paired_data'; data = load_pairs(prior, baseline, guard)
            result['data'] = {'development_positions': 1008, 'validation_positions': 336, 'development_books': list(range(18)), 'validation_books': list(range(18, 24)),
                'keys': data['keys'], 'teacher_source128_function_exposure_dev': np.bincount(data['labels'][data['dev']], minlength=128).tolist(),
                'teacher_source128_function_exposure_val': np.bincount(data['labels'][data['val']], minlength=128).tolist(),
                'paired_probability256_sha256': hashlib.sha256(data['p256'].tobytes()).hexdigest(), 'paired_probability128_sha256': hashlib.sha256(data['p128'].tobytes()).hexdigest(),
                'native_teachers_not_original_F32_on_shared_contexts': True, 'natural_cohorts_excluded': True}
            norm = tuple(R.C.tensor(mapped[256], entries[256], k) for k in ('decoder.block.11.layer.2.layer_norm.weight', 'decoder.final_layer_norm.weight'))
            head = G.I8Operator(R.C.tensor(mapped[256], entries[256], 'lm_head.weight'), R.C.tensor(mapped[256], entries[256], 'lm_head.weight', 'scales'))
            cache = ExpertCache(mapped[128], entries[128]); first = data['first_rows'][256]; source_id = int(data['labels'][0]); old_id = int(data['old_ids'][0])
            stage = 'native_no_added_baseline'; ep = f'decoder.block.11.layer.2.mlp.experts.expert_{old_id}.'
            oldops = tuple(G.I8Operator(R.C.tensor(mapped[256], entries[256], ep+k+'.weight'), R.C.tensor(mapped[256], entries[256], ep+k+'.weight', 'scales')) for k in ('wi', 'wo'))
            with torch.no_grad():
                output = G.forward(torch.from_numpy(data['pre'][0]), torch.from_numpy(data['scores'][0]), old_id, *norm, *oldops, head)
                for k, v in output.items(): R.C.exact(v.numpy(), data['logits'][0] if k == 'logits' else first[k])
            del oldops, output; gc.collect(); guard()
            stage = 'affine_numeric_prerequisite'; result['affine_qualification'] = H.qualify_affine()
            stage = 'cross_composed_real_prerequisite'; K.OUT = QUAL  # Only output destination; immutable qualification math/bounds unchanged.
            result['qualifications'].append(K.qualify('real.cross_core256_function128', data['pre'][0], data['scores'][0], old_id, *norm, *cache.get(source_id), head,
                G.corrections(768, 8, 547+1009*source_id, zero=True), .5*(data['p256'][0]+data['p128'][0]), tiny=False)); guard()
            stage = 'adapter_numeric_prerequisite'; tiny = K.tiny_inputs()
            result['qualifications'].append(H.qualify_adapter('adapter.tiny', *tiny[:5], tiny[7], tiny[8], tiny[9], QUAL/'adapter.tiny.npz', tiny=True))
            result['qualifications'].append(H.qualify_adapter('adapter.real', data['pre'][0], data['scores'][0], old_id, *norm, head,
                G.corrections(768, 8, 547+1009*source_id, zero=True), .5*(data['p256'][0]+data['p128'][0]), QUAL/'adapter.real.npz', tiny=False)); guard()
            state['classifier'] = (torch.zeros((128, 768), requires_grad=True), torch.zeros(128, requires_grad=True))
            for arm in ('real', 'adapter'):
                state[arm] = [G.corrections(768, 8, 547+1009*i, zero=True) for i in range(128)]
                state[arm+'_gate'] = (torch.zeros((1, 768), requires_grad=True), torch.tensor([-.001], requires_grad=True))
            assert all(np.array_equal(state['real'][i][k].detach().numpy(), state['adapter'][i][k].detach().numpy()) for i in range(128) for k in ('A', 'B', 'C', 'D'))
            counts = {arm: sum(z.numel() for p in state[arm] for z in p.values()) for arm in ('real', 'adapter')}
            assert counts['real'] == counts['adapter'] == 3145728
            result['initialization'] = {'factor_seeds': [547+1009*i for i in range(128)], 'equal_four_factor_initialization_and_counts': counts,
                'factor_initial_sha256': {arm: hashlib.sha256(b''.join(z.detach().numpy().tobytes() for p in state[arm] for z in p.values())).hexdigest() for arm in ('real', 'adapter')}}
            stage = 'classifier_fit'; fit_affine('classifier', *state['classifier'], data['dev'], data['input'], data['labels'], False, result, guard)
            identities, classifier_scores = predict_affine(*state['classifier'], data['input'], False)
            result['classifier'] = {'development_accuracy': float(np.mean(identities[data['dev']] == data['labels'][data['dev']])),
                'validation_accuracy': float(np.mean(identities[data['val']] == data['labels'][data['val']])), 'actual_ids': identities.tolist(),
                'development_exposure': np.bincount(identities[data['dev']], minlength=128).tolist(), 'validation_exposure': np.bincount(identities[data['val']], minlength=128).tolist()}
            initial_factors = [G.corrections(768, 8, 547+1009*i, zero=True) for i in range(128)]
            stage = 'zero_correction_validation'; zero = forced('real', data['val'], identities, data, norm, head, initial_factors, cache, guard)
            np.save(OUT/'validation.zero_correction.npy', zero); result['zero_correction_validation'] = summarize(losses(zero, data['val'], data)); del zero, initial_factors
            result['factor_update_exposure'] = {}
            for arm in ('real', 'adapter'):
                stage = f'{arm}_factor_fit'; result['factor_update_exposure'][arm] = fit_factors(arm, data, identities, norm, head, state[arm], cache, result, guard, save)
            old_dev = losses(data['logits'][data['dev']], data['dev'], data); old_val = losses(data['logits'][data['val']], data['val'], data)
            result['no_added_validation'] = summarize(old_val); evaluated = {}; gate_masks = {}
            for arm in ('real', 'adapter'):
                stage = f'{arm}_gate_development_labels'
                dev_logits = forced(arm, data['dev'], identities, data, norm, head, state[arm], cache, guard)
                dev_loss = losses(dev_logits, data['dev'], data); del dev_logits
                labels = np.zeros(1344, bool); labels[data['dev']] = (dev_loss[:, 2]+.01 <= old_dev[:, 2]) & (dev_loss[:, 0] <= old_dev[:, 0]+.02)
                result[arm+'_gate_labels'] = {'positive_development_positions': int(np.sum(labels[data['dev']])), 'validation_labels_used': False}
                stage = f'{arm}_gate_fit'; fit_affine(arm+'_gate', *state[arm+'_gate'], data['dev'], data['input'], labels, True, result, guard)
                gate, gate_score = predict_affine(*state[arm+'_gate'], data['input'], True); gate_masks[arm] = gate
                stage = f'{arm}_hard_validation'; val_logits = forced(arm, data['val'], identities, data, norm, head, state[arm], cache, guard)
                hard = np.where(gate[data['val'], None], val_logits, data['logits'][data['val']]); del val_logits
                off = ~gate[data['val']]; R.C.exact(hard[off], data['logits'][data['val'][off]])
                np.save(OUT/f'validation.{arm}.hard.npy', hard); lv = losses(hard, data['val'], data); evaluated[arm] = lv
                ids = identities[data['val']]; mask = gate[data['val']]; by_function = []
                for identity in np.unique(ids[mask]):
                    chosen = mask & (ids == identity); gain = float(np.mean(old_val[chosen, 2]-lv[chosen, 2]))
                    by_function.append({'id': int(identity), 'positions': int(np.sum(chosen)), 'mean_mixture_gain_nats': gain, 'useful_at_least2_positions_and_gain0_01': bool(np.sum(chosen) >= 2 and gain >= .01)})
                result[arm+'_validation'] = summarize(lv) | {'added_positions': int(np.sum(mask)), 'distinct_consulted_added_ids': int(len(np.unique(ids[mask]))),
                    'useful_consulted_added_ids': sum(a['useful_at_least2_positions_and_gain0_01'] for a in by_function), 'per_function': by_function,
                    'gate_scores': gate_score[:, 0].tolist(), 'gate_mask': gate.tolist(), 'per_position_losses': lv.tolist(),
                    'per_book_teacher256_delta': {str(b): float(np.mean(lv[data['books'][data['val']] == b, 0]-old_val[data['books'][data['val']] == b, 0])) for b in range(18, 24)}}
                del hard; guard()
            stage = 'causal_function_ablations'; result['causal_ablations'] = {}
            for label, perm, removal in (('identity_permutation', True, False), ('removed_function', False, True)):
                changed = forced('real', data['val'], identities, data, norm, head, state['real'], cache, guard, perm, removal)
                hard = np.where(gate_masks['real'][data['val'], None], changed, data['logits'][data['val']]); del changed
                np.save(OUT/f'validation.real.{label}.npy', hard)
                result['causal_ablations'][label] = summarize(losses(hard, data['val'], data)); del hard; guard()
            save(); stage = 'decision'
            rv = result['real_validation']; av = result['adapter_validation']; ov = result['no_added_validation']
            gates = {
                'validation_mixture_relative_gain_at_least1percent': (ov['equal_mixture_CE']-rv['equal_mixture_CE'])/max(ov['equal_mixture_CE'], 1e-8) >= .01,
                'validation_teacher128_relative_gain_at_least1percent': (ov['teacher128_CE']-rv['teacher128_CE'])/max(ov['teacher128_CE'], 1e-8) >= .01,
                'validation_teacher256_mean_delta_at_most0_02': rv['teacher256_CE']-ov['teacher256_CE'] <= .02,
                'every_validation_book_teacher256_delta_at_most0_05': max(rv['per_book_teacher256_delta'].values()) <= .05,
                'actual_added_validation_consultation_at_least34_positions': rv['added_positions'] >= 34,
                'at_least8_useful_distinct_added_ids_with2_positions_and0_01_gain': rv['useful_consulted_added_ids'] >= 8,
                'real_beats_equal_parameter_adapter_control_by0_01_nats': av['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01,
                'pretrained_identity_permutation_harms_by0_01_nats': result['causal_ablations']['identity_permutation']['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01,
                'pretrained_function_removal_harms_by0_01_nats': result['causal_ablations']['removed_function']['equal_mixture_CE']-rv['equal_mixture_CE'] >= .01}
            result['capacity_pilot_gates'] = {k: bool(v) for k, v in gates.items()}
            result['apparatus_gates'] = {'fresh_sources_capture_archives_and_424_qualification': True, 'actual_cross_composition_and_adapter_affine_numeric_checks': True,
                'equal_parameter_data_update_control': True, 'exact_no_added_baseline': True, 'one_function_hard_selector_and_no_oracle_in_validation': True,
                'all_fixed_updates_and_final_only_model': True, 'causal_ablations_same_factors_and_selector': True}
            assert sum(x['stage'] == 'classifier' for x in result['updates']) == 96
            for arm in ('real', 'adapter'):
                assert sum(x['stage'] == arm for x in result['updates']) == 3024 and sum(x['stage'] == arm+'_gate' for x in result['updates']) == 96
            result['cache'] = {'capacity': 8, 'hits': cache.hits, 'misses': cache.misses, 'maximum_I64_and_F64_pair_workspace_bytes': 8*2*768*3072*8*2}
        for n, a in exports.items():
            p = Path(a['payload']); assert initial[n] == (p.stat().st_size, p.stat().st_mtime_ns)
        files = sorted(p for p in OUT.rglob('*') if p.is_file()); result['output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
        result['resource_ledger'] = {'original_added_one_bank_coefficients': 603979776, 'original_added_I8_and_scales_bytes': 605945856,
            'correction_trainable_per_arm': 3145728, 'classifier_and_one_gate_trainable': 99201, 'arm_with_selector_F32_coeff_bytes': 12979716,
            'arm_with_selector_values_gradients_two_Adam_moments_bytes': 51918864, 'two_arms_shared_classifier_F32_coeff_bytes': 25565704,
            'active_added_four_factors_F32_coefficient_accesses_bytes': 98304, 'active_extra_classifier_and_gate_F32_coefficient_accesses_bytes': 396804,
            'head_I64_forward_workspace_bytes': 197394432, 'head_F64_backward_workspace_bytes': 197394432,
            'prototype_frozen_teacher_probability_arrays_bytes': 2*1344*32128*8, 'original_C_full_artifact_and_training_workspace_costs_separate': True}
        result['resource'] = {'seconds': time.monotonic()-start, 'maximum_RSS_or_peak_working_set_bytes': peak, 'bytes_hashed': hashed,
            'output_bytes': sum(a['bytes'] for a in result['output_inventory']), 'optimizer_updates': len(result['updates'])}
        passed = all(result['capacity_pilot_gates'].values())
        result['decision'] = 'eligible_for_NEW_native_export_and_excluded_whole_quality_protocol' if passed else 'specified_rank8_hard_selector_pilot_failed_close_recipe_before_sweep'
        result['scope'] = 'Consumed paired calibration/validation only, final decoder bank with fixed original source256 teacher-forced prefixes. Gate-OFF uses exact captured original outputs; offline gate-label/validation controls evaluate forced added branch then select conditional values. ONE function in proposed conditional inference, not a measured combined native execution budget. Native-valued CPU prototype with approximate locally qualified backward, not compiled combined C artifact or fresh donor-relative whole generation/task/rate. Available384 in ONE bank; stored128 functions are not all demonstrated useful. Other11 banks256. Native quantized teacher probabilities, not shared-context original F32 donors. SAME hard selector for causal ablations; adapter control has equal four-factor parameter/update budget and shared classifier, separately trained gate. No GPU/T4/network/new corpus/physicalDRAM/LUT/another family/~100B claim.'
        guard(); M.write(args.out, result); print(json.dumps({'sha256': M.digest(args.out), 'capacity_pilot_gates': result['capacity_pilot_gates'], 'resource': result['resource'], 'decision': result['decision']}), flush=True)
    except BaseException as error:
        if OUT.exists(): save()
        result.update({'failure_stage': stage, 'error': repr(error), 'seconds': time.monotonic()-start, 'maximum_checked_memory_bytes': peak, 'bytes_hashed': hashed})
        if OUT.exists(): result['partial_output_inventory'] = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': M.digest(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
