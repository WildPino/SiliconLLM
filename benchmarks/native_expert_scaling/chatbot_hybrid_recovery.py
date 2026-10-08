"""Finite balanced whole recovery, reusing adopted teacher bytes and Adam state.

Inference uses the original target arithmetic. Training alone dithers the AQ63
normalized coordinate by a frozen bounded amount. No original donor is loaded.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))

SEED = 20261009
EPOCHS = 4
AMPLITUDE = 0.025


def bind(a):
    corpus_path = ROOT / 'results/native_expert_scaling/chatbot_hybrid_transfer_capture_20261009/corpus.json'
    corpus = json.loads(corpus_path.read_bytes())
    adopted = DOC / 'chatbot_hybrid_transfer_adoption_result_20261009.json'
    assert json.loads(adopted.read_bytes())['decision'] == 'SAVED_CALIBRATION_TRANSPORT_PASS'
    assert json.loads(adopted.with_suffix('.terminal.json').read_bytes())['exit_code'] == 0
    checkpoint = ROOT / 'results/native_expert_scaling/chatbot_hybrid_pilot_repair1_20261008/learner.pt'
    assert sha(checkpoint) == '75b2317efe99fe66fc16f2b0e6df1f5001ef8b9c243c150b87b24e1f433793d9'
    domains = sorted({r['domain'] for r in corpus['records']})
    assert len(domains) == 8 and len(corpus['records']) == 160
    rng = random.Random(SEED)
    orders = []
    for epoch in range(EPOCHS):
        buckets = {d: [r['id'] for r in corpus['records'] if r['split'] == 'FIT' and r['domain'] == d] for d in domains}
        assert all(len(v) == 16 for v in buckets.values())
        for ids in buckets.values():
            rng.shuffle(ids)
        order = []
        for round_index in range(16):
            visits = domains.copy()
            rng.shuffle(visits)
            order.extend(buckets[d][round_index] for d in visits)
        assert len(set(order)) == 128
        orders.append(order)
    files = [Path(__file__), B / 'chatbot_hybrid_target.py', B / 'chatbot_falcon_usability.py',
             B / 'chatbot_falcon_usability_launch.py', Path(sys.executable), corpus_path, checkpoint,
             adopted, adopted.with_suffix('.terminal.json'),
             DOC / 'CHATBOT_HYBRID_RECOVERY_PROTOCOL_20261009.md']
    files += [Path(r['logits']['path']) for r in corpus['records']]
    files += [SITE / 'transformers' / v for v in ('models/falcon_h1/modeling_falcon_h1.py',
              'models/falcon_h1/configuration_falcon_h1.py', 'cache_utils.py', 'utils/import_utils.py')]
    files += [SITE / 'torch' / v for v in ('utils/checkpoint.py', 'optim/adamw.py', 'optim/optimizer.py')]
    foreign = {
        'benchmarks/donor_adaptation/configs/_manifest.json': 'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
        'benchmarks/donor_adaptation/density/build_document_holdout.py': 'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
        'docs/research/RESEARCH_INDEX.md': '99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    assert all(sha(ROOT / k) == v for k, v in foreign.items())
    files += [ROOT / k for k in foreign]
    inputs = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    write(a.out, dict(schema='HYBRID_RECOVERY_BINDING_V1', python=str(Path(sys.executable).resolve()),
          worker_path=str(Path(__file__).resolve()), corpus=str(corpus_path.resolve()),
          checkpoint=str(checkpoint.resolve()), seed=SEED, epochs=EPOCHS, orders=orders,
          normalized_AQ_dither=AMPLITUDE, learning_rate=5e-5, clip_norm=1.0,
          limits=dict(seconds=3600, OS_bytes=16 << 30, GPU_allocated_bytes=11 << 30,
                      GPU_reserved_bytes=12 << 30, output_bytes=12 << 30),
          criteria=dict(FIT_KL_ratio_max=0.50, DEV_KL_ratio_max=0.75,
                        DEV_case_KL_max=1.0, DEV_label_disagreement_max=0.20,
                        domain_DEV_case_KL_max=2.0, domain_DEV_label_disagreement_max=0.35),
          runtime_binding_scope='Checkpoint/adopted full-vocabulary packets/order/code/Python and selected runtime files; exact isolated package versions. Not a full DLL-tree certificate.',
          inputs=inputs))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(inputs))), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'HYBRID_RECOVERY_BINDING_V1'
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    assert sys.version_info[:3] == (3, 12, 10)
    assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
    a.directory.mkdir(exist_ok=False)
    last_snapshot = None
    completed = []
    phase = 'imports'
    try:
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        import tokenizers
        from torch.utils.checkpoint import checkpoint
        import chatbot_hybrid_target as native
        assert torch.__version__ == '2.6.0+cu124' and transformers.__version__ == '5.13.1'
        assert np.__version__ == '2.4.6' and tokenizers.__version__ == '0.22.2' and psutil.__version__ == '7.2.2'
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.manual_seed(b['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))
        runtime = {v.__name__: dict(version=v.__version__, path=v.__file__) for v in (torch, transformers, np, tokenizers, psutil)}
        runtime.update(device=torch.cuda.get_device_name(), tf32=False, dtype='F32',
                       checkpointing='whole blocks, nonreentrant, preserve_rng_state=True',
                       AQ_training_dither=b['normalized_AQ_dither'], AQ_inference='original round/absmax/clip',
                       donor_loaded=False, GPU_bitwise_determinism_claim=False)

        def guard():
            limit = b['limits']
            assert time.monotonic()-start <= limit['seconds']-60, 'worker deadline reserve'
            assert proc.memory_info().peak_wset <= limit['OS_bytes'], 'OS cap'
            assert torch.cuda.max_memory_allocated() <= limit['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= limit['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'worker subprocess'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= limit['output_bytes'], 'output cap'

        def event(stage, **fields):
            print(json.dumps(dict(stage=stage, seconds=time.monotonic()-start, **fields)), flush=True)
            guard()

        def save(path, value):
            with path.open('xb') as f:
                torch.save(value, f)
                f.flush()
                os.fsync(f.fileno())
            return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))

        def cpu_copy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().to('cpu', copy=True)
            if isinstance(value, dict):
                return {k: cpu_copy(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu_copy(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu_copy(v) for v in value)
            return value

        dither_enabled = False
        original_aq = native.aq63

        def robust_aq(x):
            if not dither_enabled or not torch.is_grad_enabled():
                return original_aq(x)
            scale = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 63
            noise = torch.empty_like(x).uniform_(-b['normalized_AQ_dither'], b['normalized_AQ_dither'])
            q = torch.round(x.detach() * (1 / scale) + noise).clamp(-63, 63)
            return q, scale

        # Explicit worker-local training hook; original target file stays intact.
        native.aq63 = robust_aq

        class RecoveryTarget(native.Target):
            def forward(self, ids, positions):
                x = self.embed(ids)
                for block in self.layers:
                    x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True) if self.training and torch.is_grad_enabled() else block(x)
                return self.head(self.final_norm(x)[:, positions])

        old = torch.load(b['checkpoint'], map_location='cpu', weights_only=True)
        assert old['target_schema'] == 'COMPACT_FALCON_TERNARY_TARGET_V1' and len(old['updates']) == 8
        config = old['config']
        target = RecoveryTarget(config).to('cuda')
        target.load_state_dict(old['model'], strict=True)
        parameters = list(target.parameters())
        named = list(target.named_parameters())
        count = sum(p.numel() for p in parameters)
        assert count == 254932736 and len(named) == 211
        assert all(p.dtype == torch.float32 for p in parameters)
        optimizer = torch.optim.AdamW(parameters, lr=b['learning_rate'], weight_decay=0, foreach=False)
        optimizer.load_state_dict(old['optimizer'])
        assert len(optimizer.state) == len(named)
        for group in optimizer.param_groups:
            assert tuple(group['betas']) == (0.9,0.999) and group['eps'] == 1e-8
            assert not group['amsgrad'] and not group['maximize'] and not group['capturable'] and not group['differentiable']
            group.update(lr=b['learning_rate'], weight_decay=0, foreach=False)
        for p in parameters:
            state = optimizer.state[p]
            assert state['exp_avg'].shape == p.shape == state['exp_avg_sq'].shape
            assert int(state['step'].item()) == 8
        moments = sum(v.numel()*v.element_size() for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq'))
        assert moments == count*8
        last_snapshot = dict(model=old['model'], optimizer=old['optimizer'], config=config,
              target_schema=old['target_schema'], completed_recovery_updates=0,
              torch_CPU_rng=torch.get_rng_state(), torch_CUDA_rng=torch.cuda.get_rng_state_all())
        del old
        gc.collect()
        corpus = json.loads(Path(b['corpus']).read_bytes())
        records = corpus['records']
        by_id = {r['id']: r for r in records}
        assert len(by_id) == 160 and sum(len(r['output_ids']) for r in records) == 5746
        longest_id = max((r for r in records if r['split']=='FIT'), key=lambda r:len(r['student_input_ids']))['id']
        support = {label: {d: np.zeros((12,72), dtype=np.int64) for d in sorted({r['domain'] for r in records})} for label in ('before','training','after')}
        collection = None

        def support_hook(site):
            def collect(module, args):
                if collection is None:
                    return
                label, domain = collection
                with torch.no_grad():
                    scores = module.router(args[0].reshape(-1,512))
                    ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :8]
                    counts = torch.bincount(ids.flatten(), minlength=72).cpu().numpy()
                support[label][domain][site] += counts
            return collect

        for i, layer in enumerate(target.layers):
            layer.banks.register_forward_pre_hook(support_hook(i))

        def observe(rec, label, retain=False):
            nonlocal collection
            bits = np.fromfile(rec['logits']['path'], dtype='<u2').astype('<u4')
            bits <<= 16
            teacher = torch.from_numpy(bits.view('<f4').reshape(rec['logits']['shape'])).to('cuda')
            logp = torch.log_softmax(teacher, -1).detach()
            prob = logp.exp()
            ids = torch.tensor([rec['student_input_ids']], device='cuda')
            positions = torch.tensor(rec['positions'], device='cuda')
            collection = (label, rec['domain'])
            try:
                logits = target(ids, positions).squeeze(0)
            finally:
                collection = None
            assert logits.shape == logp.shape and torch.isfinite(logits).all().item(), 'student logits'
            losses = (prob * (logp-torch.log_softmax(logits,-1))).sum(-1)
            loss = losses.mean()
            assert torch.isfinite(loss).item(), 'loss'
            winners = logits.detach().argmax(-1).cpu().tolist()
            row = dict(id=rec['id'], domain=rec['domain'], split=rec['split'], labels=len(rec['output_ids']),
                       input_tokens=len(rec['student_input_ids']), KL=loss.detach().item(),
                       label_KL=losses.detach().cpu().tolist(), predicted_ids=winners,
                       disagreements=sum(x!=y for x,y in zip(winners,rec['output_ids'],strict=True)),
                       training_dither=dither_enabled)
            if retain:
                path = a.directory / (label+'.'+rec['id']+'.logits.f32')
                with path.open('xb') as f:
                    f.write(logits.detach().cpu().numpy().astype('<f4',copy=False).tobytes())
                    f.flush()
                    os.fsync(f.fileno())
                row['logits'] = dict(path=str(path.resolve()), shape=list(logits.shape), bytes=path.stat().st_size, sha256=sha(path))
            return loss, row

        def aggregate(rows):
            def values(chosen):
                n = sum(r['labels'] for r in chosen)
                errors = sum(r['disagreements'] for r in chosen)
                return dict(cases=len(chosen), labels=n, case_KL=sum(r['KL'] for r in chosen)/len(chosen),
                            label_KL=sum(sum(r['label_KL']) for r in chosen)/n,
                            disagreements=errors, label_disagreement=errors/n,
                            case_disagreement=sum(r['disagreements']/r['labels'] for r in chosen)/len(chosen))
            summary = {s: values([r for r in rows if r['split']==s]) for s in ('FIT','DEV')}
            summary['domains'] = {d: {s: values([r for r in rows if r['split']==s and r['domain']==d]) for s in ('FIT','DEV')} for d in support['before']}
            return summary

        def evaluate(label):
            nonlocal phase, dither_enabled
            phase = label
            dither_enabled = False
            target.eval()
            rows = []
            with torch.no_grad():
                for rec in records:
                    t0 = time.monotonic()
                    _, row = observe(rec,label,retain=True)
                    torch.cuda.synchronize()
                    row['seconds'] = time.monotonic()-t0
                    # Each complete observation is durable before a later gate.
                    write(a.directory/(label+'.'+rec['id']+'.json'),row)
                    rows.append(row)
                    event(label,case=row['id'],KL=row['KL'],labels=row['labels'],case_seconds=row['seconds'])
            summary = dict(cases=rows, **aggregate(rows))
            write(a.directory/(label+'.json'),summary)
            return summary

        event('learner_ready',parameters=count,optimizer_moment_bytes=moments,initial_optimizer_step=8,longest_training_case=longest_id,runtime=runtime)
        before = evaluate('before')
        updates = []
        epoch_metrics = []
        first_long_seen = False
        for epoch, order in enumerate(b['orders']):
            epoch_rows = []
            for identifier in order:
                phase = 'training'
                guard()
                step_start = time.monotonic()
                target.train()
                dither_enabled = True
                optimizer.zero_grad(set_to_none=True)
                loss, row = observe(by_id[identifier],'training')
                loss.backward()
                assert all(p.grad is not None and p.grad.shape==p.shape for p in parameters), 'missing gradient'
                inspect = not updates or (identifier==longest_id and not first_long_seen) or len(updates)==511
                gradients = []
                if inspect:
                    for i, layer in enumerate(target.layers):
                        groups = {}
                        for label, pars in (('core',layer.core.parameters()),('experts',layer.banks.parameters()),
                                ('norms',list(layer.input_norm.parameters())+list(layer.ff_norm.parameters()))):
                            total = sum(p.grad.double().square().sum().item() for p in pars)
                            assert 0 < total < float('inf'), (i,label,'gradient')
                            groups[label] = total**.5
                        gradients.append(dict(layer=i,**groups))
                    if identifier==longest_id:
                        first_long_seen = True
                norm = torch.nn.utils.clip_grad_norm_(parameters,b['clip_norm'],error_if_nonfinite=True).item()
                optimizer.step()
                with torch.no_grad():
                    for layer in target.layers:
                        for scale in (layer.banks.gate_scale,layer.banks.up_scale,layer.banks.down_scale):
                            scale.clamp_(min=1e-8)
                dither_enabled = False
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                record = dict(**row,step=len(updates)+1,epoch=epoch+1,gradient_norm_before_clip=norm,
                              layer_gradient_norms=gradients,seconds_before_snapshot=time.monotonic()-step_start)
                # Retain the complete prior boundary until the next complete
                # model/moment/RNG copy succeeds. A fault never replays an update.
                snapshot = dict(model=cpu_copy(target.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
                     config=config,target_schema='COMPACT_FALCON_TERNARY_TARGET_V1',
                     completed_recovery_updates=record['step'],torch_CPU_rng=torch.get_rng_state(),
                     torch_CUDA_rng=torch.cuda.get_rng_state_all(),orders=b['orders'],
                     training_contract=dict(normalized_AQ_dither=b['normalized_AQ_dither'],seed=b['seed'],learning_rate=b['learning_rate']))
                for v in list(snapshot['model'].values())+[v for s in snapshot['optimizer']['state'].values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq')]:
                    assert np.isfinite(v.numpy()).all(), 'nonfinite completed model/moment'
                last_snapshot = snapshot
                record['seconds_with_snapshot'] = time.monotonic()-step_start
                write(a.directory/(f'update_{record["step"]:04d}.json'),record)
                updates.append(record)
                completed.append(record['step'])
                epoch_rows.append(record)
                event('update',step=record['step'],epoch=epoch+1,case=identifier,KL=row['KL'],step_seconds=record['seconds_with_snapshot'],inspected=inspect)
            metric = dict(epoch=epoch+1,online_case_KL=sum(r['KL'] for r in epoch_rows)/128,
                          updates=128,training_seconds=sum(r['seconds_with_snapshot'] for r in epoch_rows))
            epoch_metrics.append(metric)
            write(a.directory/(f'epoch_{epoch+1}.json'),metric)
            event('epoch',**metric)
        assert len(updates)==512 and first_long_seen
        phase = 'checkpoint'
        last_snapshot.update(freeze=a.freeze,binding_sha256=a.binding_sha,updates=updates,
                             checkpoint_selection='fixed final update512, no DEV selection')
        checkpoint_record = save(a.directory/'learner.pt',last_snapshot)
        del snapshot
        last_snapshot = None
        gc.collect()
        after = evaluate('after')
        c = b['criteria']
        gates = dict(complete_512_updates=len(updates)==512,
             FIT_case_KL_recovery=after['FIT']['case_KL']<=c['FIT_KL_ratio_max']*before['FIT']['case_KL'],
             DEV_case_KL_recovery=after['DEV']['case_KL']<=c['DEV_KL_ratio_max']*before['DEV']['case_KL'],
             DEV_absolute_case_KL=after['DEV']['case_KL']<=c['DEV_case_KL_max'],
             DEV_absolute_label_disagreement=after['DEV']['label_disagreement']<=c['DEV_label_disagreement_max'],
             DEV_every_domain_case_KL=all(v['DEV']['case_KL']<=c['domain_DEV_case_KL_max'] for v in after['domains'].values()),
             DEV_every_domain_label_disagreement=all(v['DEV']['label_disagreement']<=c['domain_DEV_label_disagreement_max'] for v in after['domains'].values()),
             routing_support=all(np.count_nonzero(sum(support[label].values())[i])>=8 for label in support for i in range(12)))
        support_record = {label:{domain:rows.tolist() for domain,rows in domains.items()} for label,domains in support.items()}
        write(a.directory/'support.json',support_record)
        plateau = all(epoch_metrics[i]['online_case_KL']>=.99*epoch_metrics[i-1]['online_case_KL'] for i in (2,3))
        result = dict(schema='HYBRID_RECOVERY_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
             process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),runtime=runtime,
             decision='BALANCED_RECOVERY_ELIGIBLE' if all(gates.values()) else 'BALANCED_RECOVERY_FAIL',gates=gates,
             before=before,after=after,updates=updates,epoch_metrics=epoch_metrics,
             late_online_plateau_diagnostic=plateau,checkpoint=checkpoint_record,
             trainable_parameters=count,optimizer_moment_bytes=moments,prior_updates=8,new_updates=len(updates),
             GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
             worker_OS_peak_snapshot=proc.memory_info().peak_wset,elapsed_seconds=time.monotonic()-start,
             native_C_calls=0,source_calls=0,final_chatbot_quality=False,
             limitations='Calibration on original teacher prefixes; related templates; 23 partial replies. No own-history/source-quality/native-parity/rate/useful-n/family admission.')
        write(a.out,result)
        event('complete',decision=result['decision'],gates=gates)
    except BaseException as error:
        write(a.directory/'first_failure.json',dict(fault=repr(error),phase=phase,completed_updates=completed,
               elapsed_seconds=time.monotonic()-start,retained_boundary=None if last_snapshot is None else last_snapshot['completed_recovery_updates']))
        if last_snapshot is not None:
            save(a.directory/'recovery.pt',last_snapshot)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bind', action='store_true')
    p.add_argument('--binding', type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    bind(args) if args.bind else worker(args)
