"""Resume the actual broad site0 state for a fixed additional optimization dose."""
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
from chatbot_source_ffn_broad_recovery import original_student
from chatbot_source_ffn_local import read, receipt
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def original_helpers(namespace):
    path = B / 'chatbot_source_ffn_broad_recovery.py'
    tree = ast.parse(path.read_bytes())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    body = next(n for n in worker.body if isinstance(n, ast.Try)).body
    names = ('load', 'metric', 'cpu_tree', 'snapshot')
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}, {n.name: ast.dump(n, include_attributes=False) for n in nodes}


def bind(a):
    prior_path = DOC / 'chatbot_source_ffn_broad_recovery_result_20261009.json'
    prior_binding = DOC / 'chatbot_source_ffn_broad_recovery_binding_20261009.json'
    prior, old = read(prior_path), read(prior_binding)
    receipt(prior_path)
    assert sha(prior_binding) == prior['binding_sha256'] and prior['optimizer_updates'] == 512
    record = next(r for r in prior['records'] if r['site'] == 0)
    assert record['steps'] == 256 and len(record['history']) == 256
    assert record['final_state']['sha256'] == 'a260302562f5ac40fef1d5043989f7e867b4a1026c468cc13b9e90279f0b2590'
    capture_path = Path(old['capture'])
    capture = read(capture_path)
    receipt(capture_path)
    assert capture['decision'] == 'SOURCE_FFN_BROAD_OPERANDS_PASS' and capture['cases'] == 48
    files = [Path(__file__), Path(sys.executable), B / 'chatbot_falcon_usability.py',
        B / 'chatbot_falcon_usability_launch.py', B / 'chatbot_source_ffn_local.py',
        B / 'chatbot_source_ffn_broad_recovery.py', B / 'chatbot_source_ffn_local_recovery.py',
        B / 'chatbot_hybrid_target.py', prior_path, prior_path.with_suffix('.terminal.json'),
        prior_binding, capture_path, capture_path.with_suffix('.terminal.json'),
        DOC / 'CHATBOT_SOURCE_FFN_DOSE_PROTOCOL_20261009.md',
        Path(old['source'])/'config.json']

    def add(item):
        p = Path(item['path'])
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
        files.append(p)

    add(record['final_state']); add(record['input_scale'])
    for projection in record['projections']:
        for name in ('codes', 'scale'): add(projection[name])
    for row in record['after']: add(row['output'])
    for row in capture['records']:
        packet = next(s for s in row['sites'] if s['site'] == 0)
        for name in ('x', 'y'): add(packet[name])
    relevant = set(p.resolve() for p in files if p.exists())
    for item in old['inputs']:
        p = Path(item['path'])
        if p.resolve() in relevant and p.name != 'chatbot_falcon_usability_launch.py':
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
        if str(SITE.resolve()) in str(p.resolve()) or p.name in ('_manifest.json', 'build_document_holdout.py', 'RESEARCH_INDEX.md'):
            add(item)
    assert sha(B/'chatbot_source_ffn_local_recovery.py') == prior['original_Student_file_sha256']
    fit = [r for r in capture['records'] if r['split'] == 'FIT']
    domains = sorted({r['domain'] for r in fit})
    by_domain = {d: sorted((r for r in fit if r['domain'] == d), key=lambda r: r['id']) for d in domains}
    assert len(domains) == 12 and all(len(r) == 2 for r in by_domain.values())
    schedule = []
    for step in range(257, 1025):
        domain = domains[(step-1) % 12]
        visit = (step-1)//12
        rec = by_domain[domain][visit % 2]
        offset = 0 if rec['labels'] <= 64 else ((visit//2)*64) % rec['labels']
        schedule.append(dict(step=step, id=rec['id'], domain=domain,
            offset=offset, rows=min(offset+64, rec['labels'])-offset))
    files = list(dict.fromkeys(p.resolve() for p in files))
    write(a.out, dict(schema='SOURCE_FFN_DOSE_BINDING_V1', phase='dose',
        python=str(Path(sys.executable).resolve()), worker_path=str(Path(__file__).resolve()),
        source_config=str(Path(old['source'])/'config.json'), prior_result=str(prior_path),
        capture=str(capture_path), state=record['final_state'], site=0, start_step=256, final_step=1024,
        new_updates=768, cases=48, schedule=schedule, lr=5e-5,
        criteria=dict(DEV_ratio=.90, centered_DEV_ratio=.90, individual_DEV_ratio=1.05,
            absolute_DEV_relative_L2=.10, DEV_cosine=.99, absolute_centered_DEV_relative_L2=.10),
        limits=dict(seconds=300, reserve_seconds=45, OS_bytes=8 << 30,
            GPU_allocated_bytes=10 << 30, GPU_reserved_bytes=11 << 30, output_bytes=2 << 30),
        runtime_binding_scope='Actual site0 model/Adam/RNG/history256,48 source x/y,48 retained after256 responses/native fields, original Student/helper AST, config/runtime/foreign hashes; no source weights or full DLL-tree.',
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(files), new_updates=768)), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = read(a.binding)
    assert b['schema'] == 'SOURCE_FFN_DOSE_BINDING_V1' and b['site'] == 0
    a.directory.mkdir(exist_ok=False)
    phase, updates, torch = 'startup', 0, None
    try:
        import numpy as np
        import psutil
        import torch
        from torch import nn
        from torch.nn import functional as F
        from chatbot_hybrid_target import aq63
        assert (torch.__version__, np.__version__) == ('2.6.0+cu124', '2.4.6') and sys.version_info[:3] == (3, 12, 10)
        assert Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        ga, db = read(b['source_config'])['mlp_multipliers']
        Student, student_ast = original_student(dict(torch=torch, nn=nn, F=F, aq63=aq63, ga=ga, db=db))
        helpers, helper_ast = original_helpers(dict(torch=torch, np=np, a=a, os=os, sha=sha))
        load, metric, snapshot = [helpers[name] for name in ('load', 'metric', 'snapshot')]

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

        prior = read(b['prior_result']); record = next(r for r in prior['records'] if r['site'] == 0)
        assert student_ast == prior['original_Student_AST']
        state = torch.load(b['state']['path'], map_location='cpu', weights_only=True)
        assert state['schema'] == 'BROAD_SOURCE_FFN_RECOVERY_STATE_V1' and state['site'] == 0 and state['steps'] == 256
        assert state['binding_sha256'] == prior['binding_sha256'] and state['freeze'] == prior['freeze']
        assert state['history'] == record['history'] and len(state['history']) == 256
        base = {name: value.to('cuda') for name, value in state['model'].items()}
        assert len(base) == 6 and all(v.dtype == torch.float32 and torch.isfinite(v).all().item() for v in base.values())
        model = Student({n: base[n] for n in ('gate', 'up', 'down')}, {n: base[n+'_scale'] for n in ('gate', 'up', 'down')})
        assert sum(p.numel() for p in model.parameters()) == 28322816
        opt = torch.optim.AdamW(model.parameters(), lr=b['lr'], betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
        opt.load_state_dict(state['optimizer'])
        assert len(opt.state) == 6 and all(v['step'].item() == 256 for v in opt.state.values())
        for p in model.parameters():
            assert torch.isfinite(opt.state[p]['exp_avg']).all().item() and torch.isfinite(opt.state[p]['exp_avg_sq']).all().item()
        for projection in record['projections']:
            n = projection['name']; scale = getattr(model, n+'_scale').detach()
            assert scale.cpu().numpy().astype('<f4', copy=False).tobytes() == Path(projection['scale']['path']).read_bytes()
            code = torch.round(getattr(model, n).detach()/scale[:, None]).clamp(-1, 1).to(torch.int8)
            pair = ((code[:, 0::2]+1)*3+(code[:, 1::2]+1)).T.contiguous().cpu().numpy().astype('u1', copy=False)
            assert pair.tobytes() == Path(projection['codes']['path']).read_bytes(), ('state256 native pairs', n)
        torch.set_rng_state(state['CPU_RNG']); torch.cuda.set_rng_state(state['CUDA_RNG'])
        history = list(state['history']); del state
        capture = read(b['capture'])
        data = [dict(id=r['id'], split=r['split'], domain=r['domain'],
            **{name: load(next(s for s in r['sites'] if s['site'] == 0)[name]) for name in ('x', 'y')}) for r in capture['records']]
        assert len(data) == 48 and sum(r['x'].shape[0] for r in data) == 8808
        S = torch.from_numpy(np.fromfile(record['input_scale']['path'], dtype='<f4')).to('cuda')
        assert torch.equal(S, torch.ones_like(S))
        before = record['after']
        assert {r['id'] for r in before} == {r['id'] for r in data}
        phase = 'adopted_state256'; event('adopted', step=256, before_cases_reused=48, all_state_pairs_scales_exact=True)
        priced = []
        for plan in b['schedule']:
            guard()
            step = plan['step']; rec = next(r for r in data if r['id'] == plan['id'])
            assert rec['split'] == 'FIT' and rec['domain'] == plan['domain'] and step == 257+updates
            offset, stop = plan['offset'], plan['offset']+plan['rows']
            assert 0 <= offset < stop <= rec['x'].shape[0] and plan['rows'] <= 64
            x, y = rec['x'][offset:stop]*S, rec['y'][offset:stop]
            torch.cuda.synchronize(); tick = time.monotonic(); opt.zero_grad(set_to_none=True)
            actual = model(x).float(); loss = (actual-y).square().sum()/y.square().sum().clamp_min(1e-30)
            assert torch.isfinite(loss).item()
            loss.backward(); norms = {}
            for name, p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all().item(), ('gradient', step, name)
                norms[name] = p.grad.float().norm().item(); assert norms[name] > 0
            gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
            opt.step(); updates += 1
            with torch.no_grad():
                for n in ('gate', 'up', 'down'): getattr(model, n+'_scale').clamp_(min=1e-8)
            for p in model.parameters(): assert torch.isfinite(p).all().item()
            for slot in opt.state.values():
                assert slot['step'].item() == step and torch.isfinite(slot['exp_avg']).all().item() and torch.isfinite(slot['exp_avg_sq']).all().item()
            torch.cuda.synchronize(); seconds = time.monotonic()-tick
            history.append(dict(step=step, id=rec['id'], domain=rec['domain'], offset=offset, rows=stop-offset,
                loss=loss.item(), gradnorm=gradnorm, gradient_norms=norms, seconds=seconds))
            if updates <= 2: priced.append(seconds)
            if updates == 2:
                price_state = snapshot(0, step, model, opt, history)
                estimate = time.monotonic()-start+(768-2)*max(priced)+90
                event('actual_price', step=step, step_seconds=priced, projected_total_seconds=estimate, durable_state=price_state)
                assert estimate <= b['limits']['seconds']-b['limits']['reserve_seconds'], 'priced total exceeds cap'
            if step % 128 == 0: event('update', step=step, loss=loss.item(), step_seconds=seconds)
            del actual, loss, x, y
        assert updates == 768 and len(history) == 1024 and history[-1]['step'] == 1024
        final_state = snapshot(0, 1024, model, opt, history)
        after = []
        with torch.no_grad():
            for rec in data:
                guard(); value = model(rec['x']*S)
                assert torch.isfinite(value).all().item()
                item = dict(id=rec['id'], split=rec['split'], domain=rec['domain'], **metric(value.float(), rec['y']),
                    output=raw_file(f"site00.{rec['id']}.after.bf16", value.contiguous().view(torch.uint16).cpu().numpy().astype('<u2', copy=False), 'BF16'))
                first = next(r for r in before if r['id'] == rec['id'])
                item['ratio'] = item['relative_L2']/max(first['relative_L2'], 1e-30)
                item['centered_ratio'] = item['centered_relative_L2']/max(first['centered_relative_L2'], 1e-30)
                after.append(item)
        first, last = [[r for r in rows if r['split'] == 'DEV'] for rows in (before, after)]
        ratio = sum(r['relative_L2'] for r in last)/sum(r['relative_L2'] for r in first)
        centered_ratio = sum(r['centered_relative_L2'] for r in last)/sum(r['centered_relative_L2'] for r in first)
        gates = dict(DEV_improvement=ratio <= b['criteria']['DEV_ratio'],
            centered_DEV_improvement=centered_ratio <= b['criteria']['centered_DEV_ratio'],
            every_DEV_retention=all(r['ratio'] <= b['criteria']['individual_DEV_ratio'] for r in last),
            absolute_DEV_error=all(r['relative_L2'] <= b['criteria']['absolute_DEV_relative_L2'] for r in last),
            every_DEV_cosine=all(r['cosine'] >= b['criteria']['DEV_cosine'] for r in last),
            absolute_centered_DEV_error=all(r['centered_relative_L2'] <= b['criteria']['absolute_centered_DEV_relative_L2'] for r in last))
        projections, changes = [], []
        for n in ('gate', 'up', 'down'):
            master, scale = getattr(model, n).detach(), getattr(model, n+'_scale').detach()
            code = torch.round(master/scale[:, None]).clamp(-1, 1).to(torch.int8)
            original = torch.round(base[n]/base[n+'_scale'][:, None]).clamp(-1, 1).to(torch.int8)
            changes.append(dict(name=n, code_elements=code.numel(), changed_trits_from256=int((code != original).sum()),
                master_relative_L2=float(((master.double()-base[n].double()).square().sum()/base[n].double().square().sum()).sqrt()),
                scale_relative_L2=float(((scale.double()-base[n+'_scale'].double()).square().sum()/base[n+'_scale'].double().square().sum()).sqrt())))
            pair = ((code[:, 0::2]+1)*3+(code[:, 1::2]+1)).T.contiguous()
            assert torch.equal(pair.T//3-1, code[:, 0::2]) and torch.equal(pair.T % 3-1, code[:, 1::2])
            projections.append(dict(name=n,
                codes=raw_file(f'site00.final.{n}.pairs.u8', pair.cpu().numpy().astype('u1', copy=False), 'U8'),
                scale=raw_file(f'site00.final.{n}.scale.f32', scale.cpu().numpy().astype('<f4', copy=False), 'F32')))
        phase = 'result'; guard()
        result = dict(schema='SOURCE_FFN_DOSE_RESULT_V1', phase='dose',
            decision='SOURCE_FFN_DOSE_PASS' if all(gates.values()) else 'SOURCE_FFN_DOSE_FAIL',
            freeze=a.freeze, binding_sha256=a.binding_sha, process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()),
            cases=48, site=0, start_step=256, final_step=1024, optimizer_updates=768,
            before=before, after=after, DEV_ratio=ratio, centered_DEV_ratio=centered_ratio, gates=gates,
            history=history, price_state=price_state, final_state=final_state, projections=projections,
            parameter_changes=changes, input_scale=record['input_scale'], original_Student_AST=student_ast,
            original_helper_AST=helper_ast, all_adopted_native_pairs_scales_exact=True,
            source_forwards=0, source_generations=0, native_runs=0, reserved_queries=0,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start, quality_admission=False, native_admission=False,
            scope='Fixed768 NEW updates atsite0 from actual model/Adam/RNG/history256; unchanged Student/loss/data/native work; local consumed DEV evidence only.')
        write(a.directory/'dose.json', result); write(a.out, result)
        event('complete', decision=result['decision'], DEV_ratio=ratio, centered_DEV_ratio=centered_ratio, gates=gates)
    except BaseException as error:
        failure = dict(fault=repr(error), phase=phase, optimizer_updates=updates, elapsed_seconds=time.monotonic()-start)
        if 'model' in locals() and 'opt' in locals() and 'history' in locals():
            try:
                steps = {int(v['step'].item()) for v in opt.state.values()}
                assert len(steps) == 1 and len(history) == next(iter(steps))
                failure['durable_failure_state'] = snapshot(0, next(iter(steps)), model, opt, history, 'site00.failure_state.pt')
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
