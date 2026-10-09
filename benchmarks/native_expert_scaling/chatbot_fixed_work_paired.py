"""Paired whole-model recovery: eight private versus two common plus six private.

Adopts actual states and source packets; never regenerates a teacher reply.
Durable A2/A26/B26 boundaries and first-fault state preserve new work.
"""
import argparse
import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    paths = [Path(__file__), Path(sys.executable),
             DOC / 'CHATBOT_FIXED_WORK_PAIRED_PROTOCOL_20261009.md']
    reports = {}
    bindings = {}
    for arm, name in [('A', 'chatbot_broad_pilot'),
                      ('B', 'chatbot_fixed_work_feasibility')]:
        report_path = DOC / (name + '_result_repair1_20261009.json')
        receipt_path = report_path.with_suffix('.terminal.json')
        binding_path = DOC / (name + '_binding_repair1_20261009.json')
        report = json.loads(report_path.read_bytes())
        receipt = json.loads(receipt_path.read_bytes())
        old = json.loads(binding_path.read_bytes())
        assert receipt['exit_code'] == 0 and receipt['resource_gates']
        assert receipt['result_sha256'] == sha(report_path)
        assert receipt['binding_sha256'] == report['binding_sha256'] == sha(binding_path)
        state = report['checkpoint']
        assert Path(state['path']).stat().st_size == state['bytes']
        assert sha(state['path']) == state['sha256']
        paths.extend([report_path, receipt_path, binding_path, Path(state['path'])])
        reports[arm] = report
        bindings[arm] = old
    assert reports['A']['new_updates'] == 24 and reports['A']['final_optimizer_step'] == 318
    assert reports['A']['checkpoint']['sha256'] == '16a85448a706f96719c8d952bbfd5309b8bdf60237f763dfdf101bad1bb9983f'
    assert reports['B']['new_updates'] == 2 and reports['B']['final_private_step'] == 320
    assert reports['B']['final_common_step'] == 2 and reports['B']['actual_exposure_verified']
    assert reports['B']['checkpoint']['sha256'] == 'd6bac2d054cc16d9f736ddf94264abc2051e70f13ee9b0b3e080289a931938b3'
    corpus = Path(bindings['A']['corpus'])
    old_corpus = Path(bindings['A']['old_corpus'])
    records = json.loads(corpus.read_bytes())['records']
    old_records = json.loads(old_corpus.read_bytes())['records']
    retention = [r for r in old_records if r['id'] in bindings['A']['retention_ids']]
    assert len(records) == 48 and len(retention) == 32
    assert sum(len(r['output_ids']) for r in records) == 8808
    assert sum(len(r['output_ids']) for r in retention) == 1103
    fit = [r for r in records if r['split'] == 'FIT']
    domains = sorted({r['domain'] for r in fit})
    buckets = {d: sorted(r['id'] for r in fit if r['domain'] == d) for d in domains}
    order = [buckets[d][i] for i in range(2) for d in domains]
    assert len(domains) == 12 and len(set(order)) == 24 and order == bindings['A']['order']
    longest = min(fit, key=lambda r: (-len(r['student_input_ids']), r['id']))
    assert len(longest['student_input_ids']) == 1507 and longest['id'] == bindings['B']['update_case']
    paths.extend([corpus, old_corpus])
    for rec in records + retention:
        assert rec['student_input_ids'] == rec['input_ids'] + rec['output_ids'][:-1]
        entry = rec['logits']
        assert entry['shape'] == [len(rec['output_ids']), 65537]
        assert Path(entry['path']).stat().st_size == entry['bytes'] and sha(entry['path']) == entry['sha256']
        paths.append(Path(entry['path']))
    for rows in (reports['A']['after']['cases'], reports['A']['retention']['cases'], reports['B']['after']):
        for row in rows:
            entry = row['logits']
            assert Path(entry['path']).stat().st_size == entry['bytes'] and sha(entry['path']) == entry['sha256']
            paths.append(Path(entry['path']))
    immutable = {Path(v['path']).resolve(): v for v in bindings['B']['inputs']}
    runtime_roots = [(SITE / n).resolve() for n in ('torch', 'transformers', 'numpy', 'psutil')]
    source_names = ['chatbot_hybrid_target.py', 'chatbot_hybrid_shared_private_target.py',
                    'chatbot_hybrid_fixed_work_target.py', 'chatbot_target_ssd_storage.py',
                    'chatbot_falcon_ssd_tiles.py', 'chatbot_falcon_usability.py']
    for p, entry in immutable.items():
        if any(p.is_relative_to(r) for r in runtime_roots) or p.name in source_names or p == corpus.resolve():
            assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], str(p)
            paths.append(p)
    old_inputs = {Path(v['path']).resolve(): v for v in bindings['A']['inputs']}
    entry = old_inputs[old_corpus.resolve()]
    assert sha(old_corpus) == entry['sha256'] and old_corpus.stat().st_size == entry['bytes']
    paths.extend(B / n for n in source_names)
    paths.append(B / 'chatbot_falcon_usability_launch.py')
    foreign = {
        'benchmarks/phase60/engine.c': '5f948fc0dcd28b1647a2a2c73067bdc0a76a7d41ae7f34395ede8120aeaa83ce',
        'benchmarks/donor_adaptation/configs/_manifest.json': 'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
        'benchmarks/donor_adaptation/density/build_document_holdout.py': 'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
        'docs/research/RESEARCH_INDEX.md': '99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    for name, digest in foreign.items():
        assert sha(ROOT / name) == digest
        paths.append(ROOT / name)
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    assert a.freeze and not a.out.exists()
    value = dict(schema='FIXED_WORK_PAIRED_BINDING_V1', freeze=a.freeze,
        python=str(Path(sys.executable).resolve()), worker_path=str(Path(__file__).resolve()),
        parent_checkpoints={k: v['checkpoint'] for k, v in reports.items()},
        parent_reports={k: str(DOC / (n + '_result_repair1_20261009.json')) for k, n in
                        [('A', 'chatbot_broad_pilot'), ('B', 'chatbot_fixed_work_feasibility')]},
        corpus=str(corpus.resolve()), old_corpus=str(old_corpus.resolve()),
        broad_ids=[r['id'] for r in records], retention_ids=[r['id'] for r in retention],
        longest_FIT=longest['id'], order=order, actual_worker_updates=dict(A=26, B=24),
        reused_B_updates=2, final_private_step=344, final_common_step=26,
        learning_rate=5e-5, normalized_AQ_dither=.025, clip_norm=1.,
        criteria=dict(B_DEV_case_KL_ratio_max=.95, B_every_domain_ratio_max=1.05,
                      DEV_case_KL_max=1., DEV_label_disagreement_max=.2,
                      domain_DEV_case_KL_max=2., domain_DEV_label_disagreement_max=.35,
                      retention_ratio_max=1.1),
        limits=dict(seconds=1800, reserve_seconds=90, OS_bytes=8 << 30,
                    GPU_allocated_bytes=10 << 30, GPU_reserved_bytes=11 << 30,
                    output_bytes=24 << 30),
        runtime_binding_scope='Actual A318/B320 states and terminal receipts, 48 broad/32 oldDEV source packets, consumed baseline/B2 outputs, unchanged target/storage code and selected runtime files. Original observer-free A forward with an inserted detached route observer; no complete DLL-tree or coordinate-wise matched dither claim.',
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in paths])
    write(a.out, value)
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(paths))), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'FIXED_WORK_PAIRED_BINDING_V1' and b['freeze'] == a.freeze
    a.directory.mkdir(exist_ok=False)
    phase = 'imports'; arm = None; completed = 0; durable = None
    torch = None; snapshot = None; target = None; optimizer = None; updates = []
    optimizer_commit_in_progress = False
    try:
        assert sys.version_info[:3] == (3, 12, 10) and Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from torch.utils.checkpoint import checkpoint
        from transformers.models.falcon_h1 import modeling_falcon_h1 as source_code
        import chatbot_hybrid_target as native
        import chatbot_hybrid_fixed_work_target as variant
        import chatbot_target_ssd_storage as storage
        assert (torch.__version__, transformers.__version__, np.__version__) == ('2.6.0+cu124', '5.13.1', '2.4.6')
        proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

        def guard():
            v = b['limits']
            assert time.monotonic() - start <= v['seconds'] - v['reserve_seconds'], 'worker deadline reserve'
            assert proc.memory_info().peak_wset <= v['OS_bytes'], 'OS cap'
            assert torch.cuda.max_memory_allocated() <= v['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= v['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'unexpected worker child'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= v['output_bytes'], 'output cap'

        def event(stage, **fields):
            print(json.dumps(dict(stage=stage, arm=arm, seconds=time.monotonic() - start, **fields)), flush=True)
            guard()

        generated = storage.install(source_code)
        (a.directory / 'generated_method.py.txt').write_text(generated, encoding='utf8')
        original_forward = textwrap.dedent(inspect.getsource(native.Banks.forward))
        needle = '    mass = torch.softmax(torch.gather(scores, 1, ids), dim=-1)\n'
        inserted = '    if self.routing_observer is not None:\n        self.routing_observer(ids.detach(), mass.detach())\n'
        assert original_forward.count(needle) == 1
        observed_forward = original_forward.replace(needle, needle + inserted)
        assert observed_forward.replace(inserted, '') == original_forward
        namespace = dict(vars(native)); exec(compile(observed_forward, '<observed-original-eight-private>', 'exec'), namespace)
        native.Banks.forward = namespace['forward']
        (a.directory / 'observed_original_eight_forward.py.txt').write_text(observed_forward, encoding='utf8')
        dither = False; original_aq = native.aq63

        def robust_aq(x):
            if not dither or not torch.is_grad_enabled(): return original_aq(x)
            scale = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 63
            noise = torch.empty_like(x).uniform_(-b['normalized_AQ_dither'], b['normalized_AQ_dither'])
            return torch.round(x.detach() * (1 / scale) + noise).clamp(-63, 63), scale

        native.aq63 = robust_aq

        def train_forward(self, ids, positions):
            x = self.embed(ids)
            for block in self.layers:
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True) if self.training and torch.is_grad_enabled() else block(x)
            return self.head(self.final_norm(x)[:, positions])

        class TrainA(native.Target):
            forward = train_forward

        class TrainB(variant.Target):
            forward = train_forward

        records = json.loads(Path(b['corpus']).read_bytes())['records']
        old_records = json.loads(Path(b['old_corpus']).read_bytes())['records']
        retention = [r for r in old_records if r['id'] in b['retention_ids']]
        by_id = {r['id']: r for r in records + retention}
        assert len(by_id) == 80
        reports = {k: json.loads(Path(p).read_bytes()) for k, p in b['parent_reports'].items()}
        phase = 'recorded_start_RNG'
        packet = torch.load(b['parent_checkpoints']['B']['path'], map_location='cpu', weights_only=True, mmap=True)
        assert packet['completed_new_updates'] == 2 and packet['private_optimizer_step'] == 320
        assert packet['common_optimizer_step'] == 2
        paired_cpu_rng = packet['initial_update_CPU_rng'].clone()
        paired_cuda_rng = [v.clone() for v in packet['initial_update_CUDA_rng']]
        del packet; gc.collect()

        def teacher_values(rec):
            bits = np.fromfile(rec['logits']['path'], dtype='<u2').astype('<u4')
            values = (bits << 16).view('<f4').reshape(rec['logits']['shape'])
            assert np.isfinite(values).all()
            return values

        def logp64(values):
            shifted = values.astype(np.float64) - values.max(-1, keepdims=True)
            return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))

        def metrics(values, rec, teacher=None):
            if teacher is None: teacher = teacher_values(rec)
            assert list(values.shape) == rec['logits']['shape'] and np.isfinite(values).all()
            per = []
            for offset in range(0, len(values), 16):
                lp = logp64(teacher[offset:offset + 16]); lq = logp64(values[offset:offset + 16])
                loss = (np.exp(lp) * (lp - lq)).sum(-1)
                assert np.isfinite(loss).all() and loss.min() >= -1e-10
                per.extend(loss.tolist())
            predicted = values.argmax(-1).tolist()
            return dict(KL=float(np.mean(per)), label_KL=per, predicted_ids=predicted,
                disagreements=sum(x != y for x, y in zip(predicted, rec['output_ids'], strict=True)))

        def values(rows):
            n = sum(r['labels'] for r in rows); errors = sum(r['disagreements'] for r in rows)
            return dict(cases=len(rows), labels=n, case_KL=sum(r['KL'] for r in rows) / len(rows),
                label_KL=sum(sum(r['label_KL']) for r in rows) / n, disagreements=errors,
                label_disagreement=errors / n,
                case_disagreement=sum(r['disagreements'] / r['labels'] for r in rows) / len(rows))

        def summarize(rows, broad):
            result = dict(overall=values(rows), domains={d: values([r for r in rows if r['domain'] == d])
                                                       for d in sorted({r['domain'] for r in rows})})
            if broad:
                result.update({s: values([r for r in rows if r['split'] == s]) for s in ('FIT', 'DEV')})
                result['DEV_domains'] = {d: values([r for r in rows if r['domain'] == d and r['split'] == 'DEV'])
                                         for d in result['domains']}
            return result

        def reused(row, stage, exposure=None):
            rec = by_id[row['id']]; entry = row['logits']
            raw = np.fromfile(entry['path'], dtype='<f4').reshape(entry['shape'])
            measured = metrics(raw, rec)
            assert abs(measured['KL'] - row['KL']) <= 1e-4 * max(1., measured['KL'])
            assert measured['predicted_ids'] == row['predicted_ids']
            result = dict(id=rec['id'], domain=rec['domain'], split=rec['split'], labels=len(rec['output_ids']),
                          teacher_forcing_ids=len(rec['student_input_ids']), logits=entry,
                          observation_reused=True, reused_from_stage=stage, **measured)
            if exposure is not None: result['exposure'] = exposure
            return result

        phase = 'parent_outputs_reused'
        baseline_rows = [reused(r, 'parent_A318_after') for r in reports['A']['after']['cases']]
        retention_rows = [reused(r, 'parent_A318_retention') for r in reports['A']['retention']['cases']]
        baseline = dict(broad=summarize(baseline_rows, True), retention=summarize(retention_rows, False),
                        broad_cases=baseline_rows, retention_cases=retention_rows)
        write(a.directory / 'parent_outputs_reused.json', baseline)
        event('parent_outputs_reused', broad_cases=48, retention_cases=32)
        exposure = {}; collection = None; results = {}

        def route_observer(site, k):
            @torch.no_grad()
            def observe(ids, mass):
                if collection is None: return
                stage, identifier = collection; row = exposure.setdefault(stage, {}).setdefault(identifier,
                    dict(private=[[0] * 72 for _ in range(12)], common_positions=[0] * 12,
                         mass_max_defect=[0.] * 12, common_norm2=[0.] * 12))
                count = len(by_id[identifier]['student_input_ids'])
                assert ids.shape == mass.shape == (count, k)
                assert torch.isfinite(mass).all().item() and torch.all(mass > 0).item()
                assert torch.all((ids >= 0) & (ids < 72)).item()
                assert torch.all(ids.sort(-1).values.diff(dim=-1) > 0).item()
                defect = (mass.sum(-1) - 1).abs().max().item(); assert defect <= 1e-6
                row['private'][site] = torch.bincount(ids.flatten(), minlength=72).cpu().tolist()
                row['mass_max_defect'][site] = defect
            return observe

        def common_hook(site):
            @torch.no_grad()
            def observe(module, args, output):
                if collection is None: return
                stage, identifier = collection; row = exposure[stage][identifier]
                assert list(output.shape) == [1, len(by_id[identifier]['student_input_ids']), 512]
                assert torch.isfinite(output).all().item()
                row['common_positions'][site] = 2 * output.shape[1]
                row['common_norm2'][site] = output.double().square().sum().item()
            return observe

        def actual_exposure(stage, rec):
            row = exposure[stage][rec['id']]
            count = len(rec['student_input_ids']); k = 8 if arm == 'A' else 6
            assert all(sum(v) == count * k for v in row['private'])
            assert row['common_positions'] == [count * (0 if arm == 'A' else 2)] * 12
            return row

        def observe(rec, stage, retain):
            nonlocal collection
            teacher = teacher_values(rec)
            tv = torch.from_numpy(teacher).to('cuda'); lp = torch.log_softmax(tv, -1).detach(); prob = lp.exp()
            collection = (stage, rec['id'])
            try:
                logits = target(torch.tensor([rec['student_input_ids']], device='cuda'),
                                torch.tensor(rec['positions'], device='cuda')).squeeze(0)
            finally: collection = None
            assert list(logits.shape) == rec['logits']['shape'] and torch.isfinite(logits).all().item()
            loss = (prob * (lp - torch.log_softmax(logits, -1))).sum(-1).mean()
            assert torch.isfinite(loss).item()
            row = dict(id=rec['id'], domain=rec['domain'], split=rec['split'], labels=len(rec['output_ids']),
                       teacher_forcing_ids=len(rec['student_input_ids']), training_KL_F32=loss.item(),
                       training_dither=dither, exposure=actual_exposure(stage, rec))
            if retain:
                raw = logits.detach().cpu().numpy(); row.update(metrics(raw, rec, teacher))
                path = a.directory / (stage + '.' + rec['id'] + '.logits.f32')
                with path.open('xb') as f:
                    f.write(raw.astype('<f4', copy=False).tobytes()); f.flush(); os.fsync(f.fileno())
                row['logits'] = dict(path=str(path.resolve()), shape=list(raw.shape), bytes=path.stat().st_size, sha256=sha(path))
            return loss, row

        def evaluate(stage, cohort, reuse_rows=None):
            nonlocal phase, dither
            phase = stage; dither = False; target.eval(); rows = []; reuse_rows = reuse_rows or {}
            rng = torch.get_rng_state().clone(); cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
            with torch.no_grad():
                for rec in cohort:
                    guard()
                    if rec['id'] in reuse_rows:
                        row = reused(reuse_rows[rec['id']], 'B2_after', reports['B']['exposure']['after'][rec['id']])
                    else:
                        t0 = time.monotonic(); _, row = observe(rec, stage, True); torch.cuda.synchronize()
                        row['seconds'] = time.monotonic() - t0; row['observation_reused'] = False
                    write(a.directory / (stage + '.' + rec['id'] + '.json'), row); rows.append(row)
                    event(stage, id=rec['id'], KL=row['KL'], differing=row['disagreements'], reused=row['observation_reused'])
            assert torch.equal(rng, torch.get_rng_state())
            assert all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
            result = dict(cases=rows, summary=summarize(rows, cohort is records))
            write(a.directory / (stage + '.json'), result)
            event(stage + '_summary', summary=result['summary'])
            return result

        def same_bits(left, right):
            assert left.dtype == right.dtype == torch.float32 and left.shape == right.shape
            return torch.equal(left.detach().cpu().contiguous().view(torch.int32), right.contiguous().view(torch.int32))

        def cpu_tree(value):
            if isinstance(value, torch.Tensor): return value.detach().to('cpu', copy=True)
            if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, list): return [cpu_tree(v) for v in value]
            if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
            return value

        for arm in ('A', 'B'):
            phase = arm + '_adoption'; completed = 0; updates = []; durable = None; snapshot = None
            adopted = b['parent_checkpoints'][arm]
            old = torch.load(adopted['path'], map_location='cpu', weights_only=True)
            if arm == 'A':
                assert old['candidate_schema'] == 'BROAD_24_TRANSFER_CANDIDATE_V1'
                assert old['completed_broad_updates'] == 24 and old['completed_recovery_updates'] == 286
                target = TrainA(old['config']).to('cuda'); initial_step = 318; common_start = 0
                initial_cpu_rng = paired_cpu_rng; initial_cuda_rng = paired_cuda_rng
                expected_parameters = 254932736; expected_tensors = 211
            else:
                assert old['candidate_schema'] == 'FIXED_WORK_FEASIBILITY_CANDIDATE_V1'
                assert old['target_schema'] == variant.SCHEMA and old['completed_new_updates'] == 2
                assert old['private_optimizer_step'] == 320 and old['common_optimizer_step'] == 2
                target = TrainB(old['config'], 'ternary').to('cuda'); initial_step = 320; common_start = 2
                initial_cpu_rng = old['torch_CPU_rng'].clone(); initial_cuda_rng = [v.clone() for v in old['torch_CUDA_rng']]
                expected_parameters = 259669760; expected_tensors = 283
            config = old['config']; target.load_state_dict(old['model'], strict=True)
            named = list(target.named_parameters()); names = [n for n, _ in named]; pars = [p for _, p in named]
            assert len(names) == expected_tensors and sum(p.numel() for p in pars) == expected_parameters
            assert names == (list(old['model']) if arm == 'A' else old['optimizer_parameter_names'])
            group = old['optimizer']['param_groups'][0]
            optimizer = torch.optim.AdamW(pars, lr=group['lr'], betas=tuple(group['betas']), eps=group['eps'],
                                          weight_decay=group['weight_decay'], foreach=False)
            optimizer.load_state_dict(old['optimizer'])
            assert len(optimizer.state) == expected_tensors
            for index, (name, p) in enumerate(named):
                assert same_bits(p, old['model'][name]), (arm, name)
                slot = optimizer.state[p]; original = old['optimizer']['state'][index]
                assert int(slot['step'].item()) == (common_start if '.banks.common.' in name else initial_step)
                for label in ('exp_avg', 'exp_avg_sq'):
                    assert same_bits(slot[label], original[label]), (arm, name, label)
                assert torch.isfinite(p).all().item() and torch.isfinite(slot['exp_avg']).all().item()
                assert torch.isfinite(slot['exp_avg_sq']).all().item() and torch.all(slot['exp_avg_sq'] >= 0).item()
            for group in optimizer.param_groups:
                assert group['lr'] == b['learning_rate'] and tuple(group['betas']) == (.9, .999)
                assert group['eps'] == 1e-8 and group['weight_decay'] == 0 and group['foreach'] is False
                assert not group['amsgrad'] and not group['maximize']
            prior = {k: v for k, v in old.items() if k not in ('model', 'optimizer', 'torch_CPU_rng', 'torch_CUDA_rng')}
            del old, original; gc.collect()
            torch.set_rng_state(initial_cpu_rng); torch.cuda.set_rng_state_all(initial_cuda_rng)
            handles = []
            for i, layer in enumerate(target.layers):
                bank = layer.banks if arm == 'A' else layer.banks.private
                bank.routing_observer = route_observer(i, 8 if arm == 'A' else 6)
                if arm == 'B': handles.append(layer.banks.common.register_forward_hook(common_hook(i)))
            event('adopted', parameters=expected_parameters, tensors=expected_tensors, private_step=initial_step,
                  common_step=common_start, bit_exact_model_and_moments=True)

            def validate_state(expected_completed=None):
                checked = completed if expected_completed is None else expected_completed
                assert len(optimizer.state) == expected_tensors
                for name, p in named:
                    slot = optimizer.state[p]
                    expected = common_start + checked if '.banks.common.' in name else initial_step + checked
                    assert int(slot['step'].item()) == expected, (name, 'counter')
                    assert torch.isfinite(p).all().item()
                    assert torch.isfinite(slot['exp_avg']).all().item() and torch.isfinite(slot['exp_avg_sq']).all().item()
                    assert torch.all(slot['exp_avg_sq'] >= 0).item()

            def save_state(label):
                validate_state()
                packet = dict(model=cpu_tree(target.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                    optimizer_parameter_names=names, config=config,
                    target_schema='COMPACT_FALCON_TERNARY_TARGET_V1' if arm == 'A' else variant.SCHEMA,
                    candidate_schema='FIXED_WORK_PAIRED_CANDIDATE_V1', arm=arm,
                    completed_worker_updates=completed, completed_paired_updates=completed + (2 if arm == 'B' else 0),
                    private_optimizer_step=initial_step + completed, common_optimizer_step=common_start + completed if arm == 'B' else 0,
                    parent_checkpoint=adopted, parent_metadata=prior, new_updates=list(updates),
                    starting_CPU_rng=initial_cpu_rng, starting_CUDA_rng=initial_cuda_rng,
                    torch_CPU_rng=torch.get_rng_state(), torch_CUDA_rng=torch.cuda.get_rng_state_all(),
                    private_k=8 if arm == 'A' else 6, common_count=0 if arm == 'A' else 2, private_n=72,
                    common_precision='ternary' if arm == 'B' else None, freeze=a.freeze, binding_sha256=a.binding_sha,
                    order=b['order'], training_contract=dict(learning_rate=5e-5, normalized_AQ_dither=.025,
                      clip_norm=1., storage_helper_sha256=sha(B / 'chatbot_target_ssd_storage.py')))
                assert len(packet['model']) == len(packet['optimizer']['state']) == expected_tensors
                for value in packet['model'].values(): assert value.dtype == torch.float32 and np.isfinite(value.numpy()).all()
                for slot in packet['optimizer']['state'].values():
                    assert np.isfinite(slot['exp_avg'].numpy()).all() and np.isfinite(slot['exp_avg_sq'].numpy()).all()
                    assert (slot['exp_avg_sq'].numpy() >= 0).all()
                temporary = a.directory / (label + '.next.pt'); path = a.directory / (label + '.pt')
                with temporary.open('xb') as f:
                    torch.save(packet, f); f.flush(); os.fsync(f.fileno())
                os.replace(temporary, path); del packet; gc.collect()
                return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))

            snapshot = save_state

            def update(identifier):
                nonlocal phase, completed, dither, optimizer_commit_in_progress
                step = completed + 1; phase = arm + '_update_' + str(step); guard(); t0 = time.monotonic()
                target.train(); dither = True; optimizer.zero_grad(set_to_none=True)
                loss, row = observe(by_id[identifier], phase, False); loss.backward()
                assert all(p.grad is not None and p.grad.shape == p.shape and torch.isfinite(p.grad).all().item() for p in pars)
                groups = []
                for i, layer in enumerate(target.layers):
                    bank = layer.banks if arm == 'A' else layer.banks.private
                    branches = [('core', layer.core.parameters()), ('private', bank.parameters())]
                    if arm == 'B': branches.extend([('common', layer.banks.common.parameters()), ('common_down', [layer.banks.common.down])])
                    norms = {}
                    for label, parameters in branches:
                        squared = sum(p.grad.double().square().sum().item() for p in parameters)
                        assert 0 < squared < float('inf'), (i, label, 'gradient path')
                        norms[label] = squared ** .5
                    groups.append(dict(layer=i, **norms))
                norm = torch.nn.utils.clip_grad_norm_(pars, b['clip_norm'], error_if_nonfinite=True).item()
                optimizer_commit_in_progress = True; optimizer.step()
                with torch.no_grad():
                    for layer in target.layers:
                        banks = [layer.banks] if arm == 'A' else [layer.banks.private, layer.banks.common]
                        for bank in banks:
                            for scale in (bank.gate_scale, bank.up_scale, bank.down_scale): scale.clamp_(min=1e-8)
                optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); dither = False
                validate_state(step); completed = step; optimizer_commit_in_progress = False
                row.update(worker_step=step, paired_step=step + (2 if arm == 'B' else 0),
                           private_optimizer_step=initial_step + step,
                           common_optimizer_step=common_start + step if arm == 'B' else 0,
                           gradient_norm_before_clip=norm, layer_gradient_norms=groups,
                           seconds_before_snapshot=time.monotonic() - t0)
                updates.append(row); write(a.directory / (phase + '.json'), row)
                event('update', worker_step=step, paired_step=row['paired_step'], id=identifier,
                      KL=row['training_KL_F32'], interval_seconds=row['seconds_before_snapshot'])
                del loss

            if arm == 'A':
                for _ in range(2): update(b['longest_FIT'])
                phase = 'A_snapshot_2'; t0 = time.monotonic(); durable = save_state('A_boundary_2')
                write(a.directory / 'A_boundary_2.json', dict(checkpoint=durable, worker_updates=completed, snapshot_seconds=time.monotonic() - t0))
                event('durable', updates=completed, checkpoint=durable)
            before = evaluate(arm + '_before', records, {r['id']: r for r in reports['B']['after']} if arm == 'B' else None)
            for identifier in b['order']: update(identifier)
            assert completed == b['actual_worker_updates'][arm]
            phase = arm + '_snapshot_26'; t0 = time.monotonic(); durable = save_state(arm + '_boundary_26')
            snapshot_seconds = time.monotonic() - t0
            write(a.directory / (arm + '_boundary_26.json'), dict(checkpoint=durable, worker_updates=completed, snapshot_seconds=snapshot_seconds))
            event('durable', updates=completed, checkpoint=durable, snapshot_seconds=snapshot_seconds)
            after = evaluate(arm + '_after', records); retained = evaluate(arm + '_retention', retention)
            results[arm] = dict(before=before, after=after, retention=retained, checkpoint=durable,
                new_updates=list(updates), worker_new_updates=completed, reused_updates=2 if arm == 'B' else 0,
                final_private_step=344, final_common_step=26 if arm == 'B' else 0,
                parameters=expected_parameters, tensors=expected_tensors, snapshot_seconds=snapshot_seconds,
                adoption_bit_exact=True, actual_exposure_verified=True)
            write(a.directory / (arm + '_result.json'), results[arm])
            for h in handles: h.remove()
            snapshot = None; target = None; optimizer = None
            del named, names, pars, prior, bank, slot, p, layer, before, after, retained, group
            gc.collect(); torch.cuda.empty_cache(); event('arm_complete', updates=completed)
        c = b['criteria']; gates = {}
        for key, result in results.items():
            post = result['after']['summary']; ret = result['retention']['summary']['overall']
            old = baseline['retention']['overall']
            gates[key] = dict(DEV_absolute_KL=post['DEV']['case_KL'] <= c['DEV_case_KL_max'],
                DEV_absolute_disagreement=post['DEV']['label_disagreement'] <= c['DEV_label_disagreement_max'],
                DEV_every_domain_KL=all(v['case_KL'] <= c['domain_DEV_case_KL_max'] for v in post['DEV_domains'].values()),
                DEV_every_domain_disagreement=all(v['label_disagreement'] <= c['domain_DEV_label_disagreement_max'] for v in post['DEV_domains'].values()),
                old_DEV_KL_retention=ret['case_KL'] <= c['retention_ratio_max'] * old['case_KL'],
                old_DEV_disagreement_retention=ret['label_disagreement'] <= c['retention_ratio_max'] * old['label_disagreement'])
        pa = results['A']['after']['summary']; pb = results['B']['after']['summary']
        preference = dict(B_DEV_KL=pb['DEV']['case_KL'] <= c['B_DEV_case_KL_ratio_max'] * pa['DEV']['case_KL'],
            B_every_domain_KL=all(pb['DEV_domains'][d]['case_KL'] <= c['B_every_domain_ratio_max'] * pa['DEV_domains'][d]['case_KL'] for d in pa['DEV_domains']),
            B_every_domain_disagreement=all(pb['DEV_domains'][d]['label_disagreement'] <= c['B_every_domain_ratio_max'] * pa['DEV_domains'][d]['label_disagreement'] for d in pa['DEV_domains']))
        preferred = all(preference.values()) and gates['B']['old_DEV_KL_retention'] and gates['B']['old_DEV_disagreement_retention']
        phase = 'result'; write(a.directory / 'exposure.json', exposure); guard()
        result = dict(schema='FIXED_WORK_PAIRED_RESULT_V1', decision='PAIRED_B_PREFERENCE_PASS' if preferred else 'PAIRED_B_PREFERENCE_FAIL',
            freeze=a.freeze, binding_sha256=a.binding_sha, process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()),
            arms=results, parent_outputs_reused=baseline, preference_gates=preference, absolute_retention_gates=gates,
            new_optimizer_updates=50, reused_B_updates=2, new_before_cases=94, reused_B_before_cases=2,
            new_final_broad_cases=96, new_retention_cases=64, actual_exposure_verified=True,
            source_calls=0, native_runs=0, reserved_queries=0, quality_admission=False, native_admission=False,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset, elapsed_seconds=time.monotonic() - start,
            scope='One paired finite recovery under equal data/update policy and current eight-function work; F64 full-vocabulary consumed FIT/DEV/retention. No coordinate-wise matched dither, fresh chatbot generation/history/tasks, native speed, useful n or family/scale admission.')
        write(a.out, result); event('complete', decision=result['decision'], preference=preference, gates=gates)
    except BaseException as error:
        fault = dict(fault=repr(error), phase=phase, arm=arm, completed_worker_updates=completed,
                     optimizer_commit_in_progress=optimizer_commit_in_progress,
                     durable_checkpoint=durable, elapsed_seconds=time.monotonic() - start)
        if torch is not None:
            fault.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory / 'first_failure.json', fault)
        if snapshot is not None:
            try:
                optimizer.zero_grad(set_to_none=True)
                write(a.directory / 'failure_state.json', dict(checkpoint=snapshot(arm + '_failure_state'),
                      arm=arm, completed_worker_updates=completed))
            except BaseException as secondary:
                write(a.directory / 'failure_state_fault.json', dict(fault=repr(secondary)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('bind', 'worker'), default='worker')
    parser.add_argument('--binding', type=Path); parser.add_argument('--binding-sha'); parser.add_argument('--freeze')
    parser.add_argument('--directory', type=Path); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); bind(args) if args.mode == 'bind' else worker(args)
