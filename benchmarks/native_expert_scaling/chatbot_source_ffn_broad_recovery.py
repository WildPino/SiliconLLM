"""Fixed-budget broader local recovery using the frozen original Student class."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_source_ffn_local import read, receipt
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def original_student(namespace):
    path = B / 'chatbot_source_ffn_local_recovery.py'
    tree = ast.parse(path.read_bytes())
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'worker')
    body = next(node for node in worker.body if isinstance(node, ast.Try)).body
    students = [node for node in body if isinstance(node, ast.ClassDef) and node.name == 'Student']
    assert len(students) == 1
    node = students[0]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Student'], ast.dump(node, include_attributes=False)


def bind(a):
    capture_path = DOC / 'chatbot_source_ffn_broad_capture_result_20261009.json'
    capture_binding = DOC / 'chatbot_source_ffn_broad_capture_binding_20261009.json'
    capture, old = read(capture_path), read(capture_binding)
    receipt(capture_path)
    assert sha(capture_binding) == capture['binding_sha256'] and capture['decision'] == 'SOURCE_FFN_BROAD_OPERANDS_PASS'
    files = [Path(item['path']) for item in old['inputs']]
    for item in old['inputs']:
        p = Path(item['path'])
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
    for row in capture['records']:
        for site in row['sites']:
            for name in ('x', 'y'):
                item = site[name]
                assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
                files.append(Path(item['path']))
    prior_path = DOC / 'chatbot_source_ffn_input_balance_result_20261009.json'
    initial_path = DOC / 'chatbot_source_ffn_local_calibrate_result_20261009.json'
    prior, initial = read(prior_path), read(initial_path)
    for path in (prior_path, initial_path):
        receipt(path)
        files += [path, path.with_suffix('.terminal.json')]
    def add_extent(item):
        p = Path(item['path'])
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
        files.append(p)
    for rec in prior['records']:
        chosen = rec['selected']
        for name in ('input_scale', 'hidden_scale'):
            add_extent(chosen[name])
        for projection in chosen['projections']:
            for name in ('codes', 'scale'):
                add_extent(projection[name])
        base = next(r for r in initial['records'] if r['site'] == rec['site'])
        trial = next(t for t in base['trials'] if (t['alpha'], t['beta']) == (chosen['alpha'], chosen['beta']))
        for r in trial['FIT']+base['selected']['DEV']:
            add_extent(r['output'])
    files += [capture_path, capture_path.with_suffix('.terminal.json'), capture_binding, Path(__file__),
        B / 'chatbot_source_ffn_local_recovery.py', B / 'chatbot_hybrid_target.py',
        DOC / 'CHATBOT_SOURCE_FFN_BROAD_RECOVERY_PROTOCOL_20261009.md',
        SITE / 'safetensors/__init__.py', SITE / 'safetensors/_safetensors_rust.pyd']
    files = list(dict.fromkeys(p.resolve() for p in files))
    fit = [r for r in capture['records'] if r['split'] == 'FIT']
    domains = sorted({r['domain'] for r in fit})
    by_domain = {d: sorted((r for r in fit if r['domain'] == d), key=lambda r: r['id']) for d in domains}
    assert len(domains) == 12 and all(len(rows) == 2 for rows in by_domain.values())
    schedule, exposure = [], {r['id']: dict(updates=0, rows=0, visited=set()) for r in fit}
    for step in range(1, 257):
        domain = domains[(step-1) % 12]
        visit = (step-1)//12
        rec = by_domain[domain][visit % 2]
        count = rec['labels']
        offset = 0 if count <= 64 else ((visit//2)*64) % count
        stop = min(offset+64, count)
        schedule.append(dict(step=step, id=rec['id'], domain=domain, offset=offset, rows=stop-offset))
        ex = exposure[rec['id']]
        ex['updates'] += 1; ex['rows'] += stop-offset; ex['visited'].update(range(offset, stop))
    exposure = {key: dict(updates=v['updates'], rows=v['rows'], unique_rows=len(v['visited'])) for key, v in exposure.items()}
    b = {key: value for key, value in old.items() if key not in ('inputs', 'retained_capture')}
    b.update(phase='recover', worker_path=str(Path(__file__).resolve()), capture=str(capture_path),
        prior_result=str(prior_path), initial_result=str(initial_path), steps=256, batch=64, lr=5e-5,
        schedule=schedule, expected_exposure_per_site=exposure,
        criteria=dict(DEV_ratio=.90, individual_DEV_ratio=1.05, absolute_DEV_relative_L2=.10,
            DEV_cosine=.99, absolute_centered_DEV_relative_L2=.10),
        limits=dict(seconds=600, reserve_seconds=60, OS_bytes=8 << 30,
            GPU_allocated_bytes=10 << 30, GPU_reserved_bytes=11 << 30, output_bytes=2 << 30),
        runtime_binding_scope='Actual broad x/y/source/selected baseline sectors/original Student AST/new domain-balanced coverage schedule/Python/runtime/foreign hashes; no new inference or loss formula.',
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    write(a.out, b)
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(files))), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = read(a.binding)
    assert b['schema'] == 'SOURCE_FFN_BROAD_BINDING_V1' and b['phase'] == 'recover'
    a.directory.mkdir(exist_ok=False)
    phase, completed, updates = 'startup', [], 0
    torch = None
    try:
        import numpy as np
        import psutil
        import torch
        from torch import nn
        from torch.nn import functional as F
        from safetensors import safe_open
        from chatbot_hybrid_target import aq63
        assert (torch.__version__, np.__version__) == ('2.6.0+cu124', '2.4.6') and sys.version_info[:3] == (3, 12, 10)
        assert Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        ga, db = read(Path(b['source'])/'config.json')['mlp_multipliers']
        Student, student_ast = original_student(dict(torch=torch, nn=nn, F=F, aq63=aq63, ga=ga, db=db))

        def guard():
            lim = b['limits']
            assert time.monotonic()-start <= lim['seconds']-lim['reserve_seconds'], 'time reserve'
            assert proc.memory_info().peak_wset <= lim['OS_bytes'], 'OS cap'
            assert torch.cuda.max_memory_allocated() <= lim['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= lim['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'worker child'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= lim['output_bytes'], 'output cap'

        def event(stage, **fields):
            print(json.dumps(dict(stage=stage, elapsed_seconds=time.monotonic()-start, **fields)), flush=True)
            guard()

        def raw_file(name, array, dtype):
            path = a.directory/name
            with path.open('xb') as stream:
                stream.write(array.tobytes()); stream.flush(); os.fsync(stream.fileno())
            return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path), shape=list(array.shape), dtype=dtype)

        def load(item):
            bits = np.fromfile(item['path'], dtype='<u2').reshape(item['shape'])
            value = torch.from_numpy(bits.copy()).view(torch.bfloat16).to(device='cuda', dtype=torch.float32)
            assert value.ndim == 2 and value.shape[1] == 2048 and torch.isfinite(value).all().item()
            return value

        def metric(actual, expected):
            v, t = actual.double(), expected.double()
            norm = t.square().sum().clamp_min(1e-30)
            error = v-t
            mean_energy = error.shape[0]*error.mean(0).square().sum()
            varying_energy = (error-error.mean(0)).square().sum()
            centered_norm = (t-t.mean(0)).square().sum().clamp_min(1e-30)
            identity = float((error.square().sum()-mean_energy-varying_energy).abs()/norm)
            assert identity < 1e-12
            return dict(relative_L2=float((error.square().sum()/norm).sqrt()),
                cosine=float((v*t).sum()/(v.square().sum()*norm).sqrt().clamp_min(1e-30)),
                centered_relative_L2=float((varying_energy/centered_norm).sqrt()),
                mean_error_squared_normalized=float(mean_energy/norm),
                variation_error_squared_normalized=float(varying_energy/norm),
                source_mean_energy_fraction=float(t.shape[0]*t.mean(0).square().sum()/norm))

        def observed(rec, value, label):
            assert torch.isfinite(value).all().item()
            return dict(id=rec['id'], split=rec['split'], domain=rec['domain'], **metric(value.float(), rec['y']),
                output=raw_file(f"site{site:02d}.{rec['id']}.{label}.bf16",
                    value.contiguous().view(torch.uint16).cpu().numpy().astype('<u2', copy=False), 'BF16'))

        def cpu_tree(value):
            if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
            if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, list): return [cpu_tree(v) for v in value]
            if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
            return value

        def snapshot(site, step, model, opt, history, suffix=None):
            state = cpu_tree(dict(schema='BROAD_SOURCE_FFN_RECOVERY_STATE_V1', site=site, steps=step,
                model=model.state_dict(), optimizer=opt.state_dict(), CPU_RNG=torch.get_rng_state(),
                CUDA_RNG=torch.cuda.get_rng_state(), history=history, binding_sha256=a.binding_sha, freeze=a.freeze))
            assert all(torch.isfinite(v).all().item() for v in state['model'].values())
            assert len(state['optimizer']['state']) == 6
            for slot in state['optimizer']['state'].values():
                assert slot['step'].item() == step and torch.isfinite(slot['exp_avg']).all().item() and torch.isfinite(slot['exp_avg_sq']).all().item()
            path = a.directory / (suffix or f'site{site:02d}.state{step:03d}.pt')
            temp = path.with_suffix('.tmp')
            with temp.open('xb') as stream:
                torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
            temp.replace(path)
            return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))

        capture, prior, old_initial = [read(b[key]) for key in ('capture', 'prior_result', 'initial_result')]
        records = []
        with safe_open(str(Path(b['source'])/'model.safetensors'), framework='pt', device='cpu') as tensors:
            for site in b['sites']:
                phase = f'site{site:02d}'
                selected = next(r for r in prior['records'] if r['site'] == site)['selected']
                S = torch.from_numpy(np.fromfile(selected['input_scale']['path'], dtype='<f4')).to('cuda')
                R = torch.from_numpy(np.fromfile(selected['hidden_scale']['path'], dtype='<f4')).to('cuda')
                assert selected['alpha'] == 0 and torch.equal(S, torch.ones_like(S))
                w = {n: tensors.get_tensor(f'model.layers.{site}.feed_forward.{n}_proj.weight').to(device='cuda', dtype=torch.float32)
                    for n in ('gate', 'up', 'down')}
                transformed = dict(gate=w['gate']/S, up=w['up']*R[:, None]/S, down=w['down']/R)
                scales = {}
                for p in selected['projections']:
                    n = p['name']
                    scales[n] = torch.from_numpy(np.fromfile(p['scale']['path'], dtype='<f4')).to('cuda')
                    code = torch.round(transformed[n]/scales[n][:, None]).clamp(-1, 1).to(torch.int8)
                    pair = ((code[:, 0::2]+1)*3+(code[:, 1::2]+1)).T.contiguous().cpu().numpy().astype('u1', copy=False)
                    assert pair.tobytes() == Path(p['codes']['path']).read_bytes(), ('initial sector', site, n)
                model = Student(transformed, scales)
                assert len(list(model.parameters())) == 6 and sum(p.numel() for p in model.parameters()) == 28322816
                data = [dict(id=r['id'], split=r['split'], domain=r['domain'],
                    **{name: load(next(v for v in r['sites'] if v['site'] == site)[name]) for name in ('x', 'y')}) for r in capture['records']]
                assert len(data) == 48 and sum(r['x'].shape[0] for r in data) == 8808
                fit = [r for r in data if r['split'] == 'FIT']
                assert len(fit) == 24 and sum(r['x'].shape[0] for r in fit) == 4422
                domains = sorted({r['domain'] for r in fit})
                by_domain = {d: sorted((r for r in fit if r['domain'] == d), key=lambda r: r['id']) for d in domains}
                assert len(domains) == 12 and all(len(rows) == 2 for rows in by_domain.values())
                base = next(r for r in old_initial['records'] if r['site'] == site)
                trial = next(t for t in base['trials'] if (t['alpha'], t['beta']) == (selected['alpha'], selected['beta']))
                reused = {r['id']: r for r in trial['FIT']+base['selected']['DEV']}
                initial = []
                for rec in data:
                    guard()
                    if rec['id'] in reused:
                        expected = reused[rec['id']]
                        item = dict(id=rec['id'], split=rec['split'], domain=rec['domain'],
                            **metric(load(expected['output']), rec['y']), output=expected['output'], origin='REUSED_INITIAL_RESPONSE')
                        assert abs(item['relative_L2']-expected['relative_L2']) < 1e-12
                    else:
                        with torch.no_grad():
                            value = model(rec['x']*S)
                        train_value = model(rec['x']*S)
                        assert torch.equal(train_value.view(torch.uint16), value.view(torch.uint16)), ('STE initial bits', site, rec['id'])
                        item = dict(observed(rec, value, 'before'), origin='NEW_INITIAL_RESPONSE')
                        del value, train_value
                    initial.append(item)
                write(a.directory/f'site{site:02d}.initial.json', initial)
                event('initial_complete', site=site, new_cases=44, reused_cases=4)
                opt = torch.optim.AdamW(model.parameters(), lr=b['lr'], betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
                history, priced, price_state = [], [], None
                for step in range(1, b['steps']+1):
                    guard()
                    plan = b['schedule'][step-1]
                    assert plan['step'] == step
                    domain = plan['domain']
                    rec = next(r for r in fit if r['id'] == plan['id'])
                    offset, stop = plan['offset'], plan['offset']+plan['rows']
                    assert rec['domain'] == domain and 0 <= offset < stop <= rec['x'].shape[0] and plan['rows'] <= b['batch']
                    x, y = rec['x'][offset:stop]*S, rec['y'][offset:stop]
                    torch.cuda.synchronize(); tick = time.monotonic(); opt.zero_grad(set_to_none=True)
                    actual = model(x).float()
                    loss = (actual-y).square().sum()/y.square().sum().clamp_min(1e-30)
                    assert torch.isfinite(loss).item()
                    loss.backward(); norms = {}
                    for name, p in model.named_parameters():
                        assert p.grad is not None and torch.isfinite(p.grad).all().item(), ('gradient', site, step, name)
                        norms[name] = p.grad.float().norm().item()
                        assert norms[name] > 0, ('zero gradient', site, step, name)
                    gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
                    opt.step(); updates += 1
                    with torch.no_grad():
                        for n in ('gate', 'up', 'down'): getattr(model, n+'_scale').clamp_(min=1e-8)
                    for p in model.parameters(): assert torch.isfinite(p).all().item()
                    for slot in opt.state.values():
                        assert torch.isfinite(slot['exp_avg']).all().item() and torch.isfinite(slot['exp_avg_sq']).all().item() and slot['step'].item() == step
                    torch.cuda.synchronize(); seconds = time.monotonic()-tick
                    history.append(dict(step=step, id=rec['id'], domain=domain, offset=offset, rows=stop-offset,
                        loss=loss.item(), gradnorm=gradnorm, gradient_norms=norms, seconds=seconds))
                    if step <= 2: priced.append(seconds)
                    if step == 2:
                        price_state = snapshot(site, step, model, opt, history)
                        remaining = b['steps']-2+(b['steps'] if site == 0 else 0)
                        estimate = time.monotonic()-start+remaining*max(priced)+90
                        event('actual_price', site=site, step_seconds=priced, projected_total_seconds=estimate, durable_state=price_state)
                        assert estimate <= b['limits']['seconds']-b['limits']['reserve_seconds'], 'priced total exceeds cap'
                    if step % 32 == 0: event('update', site=site, step=step, loss=loss.item(), step_seconds=seconds)
                    del actual, loss, x, y
                final_state = snapshot(site, b['steps'], model, opt, history)
                exposure = {}
                for rec in fit:
                    visits = [h for h in history if h['id'] == rec['id']]
                    indices = {i for h in visits for i in range(h['offset'], h['offset']+h['rows'])}
                    exposure[rec['id']] = dict(updates=len(visits), rows=sum(h['rows'] for h in visits), unique_rows=len(indices))
                assert exposure == b['expected_exposure_per_site']
                after = []
                with torch.no_grad():
                    for rec in data:
                        guard()
                        item = observed(rec, model(rec['x']*S), 'after')
                        first = next(r for r in initial if r['id'] == rec['id'])
                        item['ratio'] = item['relative_L2']/max(first['relative_L2'], 1e-30)
                        item['centered_ratio'] = item['centered_relative_L2']/max(first['centered_relative_L2'], 1e-30)
                        after.append(item)
                dev_after = [r for r in after if r['split'] == 'DEV']
                dev_before = [r for r in initial if r['split'] == 'DEV']
                ratio = sum(r['relative_L2'] for r in dev_after)/sum(r['relative_L2'] for r in dev_before)
                centered_ratio = sum(r['centered_relative_L2'] for r in dev_after)/sum(r['centered_relative_L2'] for r in dev_before)
                gates = dict(DEV_improvement=ratio <= b['criteria']['DEV_ratio'],
                    every_DEV_retention=all(r['ratio'] <= b['criteria']['individual_DEV_ratio'] for r in dev_after),
                    absolute_DEV_error=all(r['relative_L2'] <= b['criteria']['absolute_DEV_relative_L2'] for r in dev_after),
                    every_DEV_cosine=all(r['cosine'] >= b['criteria']['DEV_cosine'] for r in dev_after),
                    absolute_centered_DEV_error=all(r['centered_relative_L2'] <= b['criteria']['absolute_centered_DEV_relative_L2'] for r in dev_after))
                domain_stats = []
                for domain in domains:
                    first = [r for r in dev_before if r['domain'] == domain]
                    last = [r for r in dev_after if r['domain'] == domain]
                    assert len(first) == len(last) == 2
                    domain_stats.append(dict(domain=domain,
                        before=sum(r['relative_L2'] for r in first)/2, after=sum(r['relative_L2'] for r in last)/2,
                        centered_before=sum(r['centered_relative_L2'] for r in first)/2,
                        centered_after=sum(r['centered_relative_L2'] for r in last)/2))
                projections = []
                parameter_changes = []
                for n in ('gate', 'up', 'down'):
                    scale = getattr(model, n+'_scale').detach()
                    master = getattr(model, n).detach()
                    code = torch.round(master/scale[:, None]).clamp(-1, 1).to(torch.int8)
                    base_code = torch.round(transformed[n]/scales[n][:, None]).clamp(-1, 1).to(torch.int8)
                    parameter_changes.append(dict(name=n, code_elements=code.numel(),
                        changed_trits=int((code != base_code).sum()),
                        master_relative_L2=float(((master.double()-transformed[n].double()).square().sum()/transformed[n].double().square().sum()).sqrt()),
                        scale_relative_L2=float(((scale.double()-scales[n].double()).square().sum()/scales[n].double().square().sum()).sqrt())))
                    pairs = ((code[:, 0::2]+1)*3+(code[:, 1::2]+1)).T.contiguous()
                    assert torch.equal(pairs.T//3-1, code[:, 0::2]) and torch.equal(pairs.T % 3-1, code[:, 1::2])
                    projections.append(dict(name=n,
                        codes=raw_file(f'site{site:02d}.final.{n}.pairs.u8', pairs.cpu().numpy().astype('u1', copy=False), 'U8'),
                        scale=raw_file(f'site{site:02d}.final.{n}.scale.f32', scale.cpu().numpy().astype('<f4', copy=False), 'F32')))
                record = dict(site=site, steps=b['steps'], initial=initial, after=after, DEV_ratio=ratio,
                    centered_DEV_ratio=centered_ratio, domains=domain_stats, gates=gates, history=history, exposure=exposure,
                    price_state=price_state, final_state=final_state, projections=projections,
                    input_scale=selected['input_scale'], trainable_elements=28322816,
                    parameter_changes=parameter_changes,
                    initial_sector_all_pairs_exact=True, new_initial_STE_forward_all_bits_equal=True,
                    reused_initial_cases=4, new_initial_cases=44)
                write(a.directory/f'site{site:02d}.json', record)
                records.append(record); completed.append(site)
                event('site_complete', site=site, DEV_ratio=ratio, centered_DEV_ratio=centered_ratio, gates=gates)
                del model, opt, w, transformed, scales, data, fit, by_domain
        assert updates == 512
        phase = 'result'; guard()
        decision = 'SOURCE_FFN_BROAD_RECOVERY_PASS' if all(all(r['gates'].values()) for r in records) else 'SOURCE_FFN_BROAD_RECOVERY_FAIL'
        write(a.out, dict(schema='SOURCE_FFN_BROAD_RESULT_V1', phase='recover', decision=decision,
            freeze=a.freeze, binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()), cases=48,
            records=records, optimizer_updates=updates, effective_final_updates=512,
            source_forwards=0, source_generations=0, native_runs=0, reserved_queries=0,
            original_Student_AST=student_ast, original_Student_file_sha256=sha(B/'chatbot_source_ffn_local_recovery.py'),
            GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start, quality_admission=False, native_admission=False,
            scope='Fixed256/site from original selected sectors, broader domain-balanced FIT, unchanged loss/Student inference/STE; local consumed DEV evidence, not whole-source/compact chatbot/engine rate/capacity admission.'))
        event('complete', decision=decision)
    except BaseException as error:
        failure = dict(fault=repr(error), phase=phase, completed=completed, optimizer_updates=updates,
            elapsed_seconds=time.monotonic()-start)
        if 'model' in locals() and 'opt' in locals() and opt.state:
            try:
                steps = {int(v['step'].item()) for v in opt.state.values()}
                assert len(steps) == 1 and len(history) == next(iter(steps))
                failure['durable_failure_state'] = snapshot(site, next(iter(steps)), model, opt, history,
                    f'site{site:02d}.failure_state.pt')
            except BaseException as save_error:
                failure['failure_state_fault'] = repr(save_error)
        if torch is not None:
            failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json', failure)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('bind', 'worker'), default='worker')
    p.add_argument('--binding', type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    bind(a) if a.mode == 'bind' else worker(a)
