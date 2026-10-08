"""One complete compact-target initialization and bounded whole-output pilot."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks/native_expert_scaling'))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def main(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'HYBRID_PILOT_BINDING_V1'
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    assert sys.version_info[:3] == (3, 12, 10)
    assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
    a.directory.mkdir(exist_ok=False)
    old = Path(b['resume_directory']) if b.get('resume_directory') else None
    torch = None
    target = None
    optimizer = None
    last_snapshot = None
    completed = []
    runtime = {}
    def cached(name):
        if old and (old/name).is_file():
            return old/name
        return a.directory/name
    try:
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        import psutil
        import numpy as np
        import torch as t
        import transformers
        import tokenizers
        from transformers import AutoTokenizer, FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as source_code
        from chatbot_hybrid_target import Target, basis, init_component, D, L, E, K, H, V
        torch = t
        assert torch.__version__ == '2.6.0+cu124' and transformers.__version__ == '5.13.1'
        assert np.__version__ == '2.4.6' and tokenizers.__version__ == '0.22.2' and psutil.__version__ == '7.2.2'
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))
        runtime = {v.__name__: dict(version=v.__version__, path=v.__file__) for v in (torch, transformers, np, tokenizers, psutil)}
        runtime.update(device=torch.cuda.get_device_name(), source_code=source_code.__file__, tf32=False,
                       student_dtype='F32', source_dtype='BF16', SSD_chunk=16, checkpointing='whole blocks nonreentrant', seed=0)
        def guard():
            limit = b['limits']
            assert time.monotonic()-start <= limit['seconds'], 'worker deadline'
            assert proc.memory_info().peak_wset <= limit['OS_bytes'], 'OS cap'
            assert torch.cuda.max_memory_allocated() <= limit['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= limit['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'worker subprocess'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= limit['output_bytes'], 'output cap'
        def event(stage, **fields):
            print(json.dumps(dict(stage=stage, seconds=time.monotonic()-start, **fields)), flush=True)
            guard()
        def save(path, value):
            assert not path.exists()
            with path.open('xb') as f:
                torch.save(value, f)
                f.flush()
                os.fsync(f.fileno())
            return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))
        def cpu_copy(v):
            if isinstance(v, torch.Tensor):
                return v.detach().to('cpu', copy=True)
            if isinstance(v, dict):
                return {k:cpu_copy(x) for k,x in v.items()}
            if isinstance(v, list):
                return [cpu_copy(x) for x in v]
            if isinstance(v, tuple):
                return tuple(cpu_copy(x) for x in v)
            return v
        cases = json.loads(Path(b['cases_path']).read_bytes())
        assert len(cases['cases']) == 6 and cases['max_new_tokens'] == 8 and cases['context_limit'] == 128
        source_dir = Path(b['source_directory'])
        config = json.loads((source_dir/'config.json').read_bytes())
        tokenizer = AutoTokenizer.from_pretrained(source_dir, local_files_only=True, trust_remote_code=False)
        records = []
        need_teacher = not all(cached(c['id']+'.json').exists() for c in cases['cases']) or not all(
            cached(name+'.initial.pt').exists() for name in ['embed', 'head', 'final_norm']+[f'layers.{i}' for i in range(L)])
        source = None
        if need_teacher:
            source = FalconH1ForCausalLM.from_pretrained(source_dir, local_files_only=True,
                         trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
            assert sum(p.numel() for p in source.parameters()) == b['source_named_elements']
            assert not source_code.is_fast_path_available
            assert not source.config.tie_word_embeddings
            event('source_loaded')
        new_generations = 0
        for case in cases['cases']:
            path = cached(case['id']+'.json')
            if path.exists():
                rec = json.loads(path.read_bytes())
                assert sha(rec['scores']['path']) == rec['scores']['sha256']
            else:
                messages = case.get('history', [])+[dict(role='user', content=case['prompt'])]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
                expected = tokenizer.bos_token+''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
                assert text == expected and ids == tokenizer.encode(text, add_special_tokens=False)
                assert len(ids)+8 <= 128
                with torch.inference_mode():
                    inputs = torch.tensor([ids], device='cuda')
                    generated = source.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        max_new_tokens=8, do_sample=False, use_cache=True, return_dict_in_generate=True, output_scores=True)
                    outputs = generated.sequences[0, len(ids):].tolist()
                    scores = torch.stack(generated.scores).squeeze(1).float().cpu().numpy()
                    assert scores.shape == (len(outputs), V) and np.isfinite(scores).all()
                    assert scores.argmax(1).tolist() == outputs
                    raw = a.directory/(case['id']+'.scores.f32')
                    with raw.open('xb') as f:
                        f.write(scores.astype('<f4', copy=False).tobytes())
                        f.flush()
                        os.fsync(f.fileno())
                    rec = dict(**case, messages=messages, serialized=text, input_ids=ids, output_ids=outputs,
                          output_text=tokenizer.decode(outputs, skip_special_tokens=True), scores=dict(
                          path=str(raw.resolve()), shape=list(scores.shape), bytes=raw.stat().st_size, sha256=sha(raw)))
                    write(path, rec)
                    new_generations += 1
                    del inputs, generated, scores
            assert rec['id'] == case['id'] and rec['split'] == case['split']
            records.append(rec)
            event('supervision_ready', case=case['id'], prefix_ids=len(rec['input_ids']), generated=len(rec['output_ids']))
        target = Target(config).to('cuda')
        count = sum(p.numel() for p in target.parameters())
        assert count == 254932736, count
        assert target.embed.weight.data_ptr() != target.head.weight.data_ptr()
        if cached('basis.pt').exists():
            packet = torch.load(cached('basis.pt'), map_location='cuda', weights_only=True)
            p, info = packet['P'], packet['info']
        else:
            p, info = basis(source)
            save(a.directory/'basis.pt', dict(P=p.cpu(), info=info))
        event('basis_ready', **info)
        components = [('embed', target.embed), ('head', target.head), ('final_norm', target.final_norm)] + [
                      (f'layers.{i}', block) for i, block in enumerate(target.layers)]
        initial = []
        for name, module in components:
            path = cached(name+'.initial.pt')
            if path.exists():
                module.load_state_dict(torch.load(path, map_location='cuda', weights_only=True), strict=True)
            else:
                init_component(name, module, source, p)
                save(path, {k:v.detach().cpu() for k,v in module.state_dict().items()})
            initial.append(dict(component=name, path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path)))
            event('component_ready', component=name)
        write(a.directory/'initial_manifest.json', dict(components=initial, parameters=count, basis=info,
              binding_sha256=a.binding_sha, freeze=a.freeze, source_revision=b['source_revision']))
        del source, p
        gc.collect()
        torch.cuda.empty_cache()
        assert all(p.dtype == torch.float32 for p in target.parameters())
        assert all(torch.isfinite(p).all().item() for p in target.parameters())
        event('complete_target_ready', parameters=count, F32_Adam_array_lower_bytes=count*16,
              matrix_products_per_decode=69632512, teacher_released=True)
        packets = []
        for r in records:
            scores = np.fromfile(r['scores']['path'], dtype='<f4').reshape(r['scores']['shape'])
            logits = torch.tensor(scores, device='cuda')
            logp = torch.log_softmax(logits, -1).detach()
            ids = r['input_ids']+r['output_ids'][:-1]
            packets.append(dict(id=r['id'], split=r['split'], ids=torch.tensor([ids], device='cuda'),
                   positions=torch.arange(len(r['output_ids']), device='cuda')+len(r['input_ids'])-1,
                   logp=logp, prob=logp.exp(), winner=logits.argmax(-1)))
        def observe(packet, grad=False, retain=None):
            logits = target(packet['ids'], packet['positions']).squeeze(0)
            assert logits.shape == packet['logp'].shape and torch.isfinite(logits).all(), 'student logits'
            logq = torch.log_softmax(logits, -1)
            kl = (packet['prob'] * (packet['logp']-logq)).sum(-1).mean()
            assert torch.isfinite(kl), 'loss'
            row = dict(id=packet['id'], split=packet['split'], positions=logits.shape[0],
                       KL=kl.detach().item(), disagreements=(logits.argmax(-1)!=packet['winner']).sum().item())
            if retain:
                raw = a.directory/(retain+'.'+packet['id']+'.logits.f32')
                with raw.open('xb') as f:
                    f.write(logits.detach().cpu().numpy().astype('<f4', copy=False).tobytes())
                    f.flush()
                    os.fsync(f.fileno())
                row['logits'] = dict(path=str(raw.resolve()), shape=list(logits.shape), bytes=raw.stat().st_size, sha256=sha(raw))
            return kl, row
        def evaluate(label):
            existing = cached(label+'.json')
            if existing.exists():
                return json.loads(existing.read_bytes())
            target.eval()
            rows = []
            with torch.no_grad():
                for packet in packets:
                    _, row = observe(packet, retain=label)
                    rows.append(row)
                    event(label, **row)
            summary = dict(cases=rows)
            for split in ('FIT','DEV'):
                chosen = [r for r in rows if r['split']==split]
                n = sum(r['positions'] for r in chosen)
                summary[split] = dict(positions=n, mean_KL=sum(r['KL']*r['positions'] for r in chosen)/n,
                         disagreements=sum(r['disagreements'] for r in chosen),
                         disagreement_fraction=sum(r['disagreements'] for r in chosen)/n)
            write(a.directory/(label+'.json'), summary)
            return summary
        before = evaluate('before')
        optimizer = torch.optim.AdamW(target.parameters(), lr=5e-5, weight_decay=0, foreach=False)
        updates = []
        # Recovery keeps completed updates; emergency checkpoint is produced on a worker fault.
        if old and (old/'recovery.pt').exists():
            recovery = torch.load(old/'recovery.pt', map_location='cuda', weights_only=True)
            target.load_state_dict(recovery['model'], strict=True)
            optimizer.load_state_dict(recovery['optimizer'])
            updates = recovery['updates']
            last_snapshot = cpu_copy(recovery)
            del recovery
        fit = [p for p in packets if p['split']=='FIT']
        for step in range(len(updates), 8):
            step_start = time.monotonic()
            target.train()
            optimizer.zero_grad(set_to_none=True)
            loss, row = observe(fit[step % 4], grad=True)
            loss.backward()
            gradients = []
            for i, layer in enumerate(target.layers):
                groups = {}
                for label, named in (('core', layer.core.named_parameters()), ('experts', layer.banks.named_parameters()),
                       ('norms', list(layer.input_norm.named_parameters())+list(layer.ff_norm.named_parameters()))):
                    total = 0.0
                    for key, par in named:
                        assert par.grad is not None and torch.isfinite(par.grad).all(), (i, label, key, 'gradient')
                        total += par.grad.double().square().sum().item()
                    assert total > 0, (i, label, 'zero gradient')
                    groups[label] = total**.5
                gradients.append(dict(layer=i, **groups))
            for key, par in target.named_parameters():
                assert par.grad is not None and torch.isfinite(par.grad).all(), (key, 'gradient')
            prior = target.head.weight[0].detach().clone()
            norm = torch.nn.utils.clip_grad_norm_(target.parameters(), 1.0, error_if_nonfinite=True).item()
            optimizer.step()
            with torch.no_grad():
                for layer in target.layers:
                    for scale in (layer.banks.gate_scale, layer.banks.up_scale, layer.banks.down_scale):
                        scale.clamp_(min=1e-8)
            assert all(torch.isfinite(p).all().item() for p in target.parameters())
            assert not torch.equal(prior, target.head.weight[0]), 'no readout update'
            moment_bytes = sum(v.numel()*v.element_size() for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq'))
            assert moment_bytes == count*8, moment_bytes
            torch.cuda.synchronize()
            record = dict(step=step+1, **row, gradient_norm_before_clip=norm, layer_gradient_norms=gradients,
                          optimizer_moment_bytes=moment_bytes, seconds=time.monotonic()-step_start)
            write(a.directory/(f'update_{step+1:02d}.json'), record)
            updates.append(record)
            completed.append(f'update_{step+1}')
            # Retain only a completed boundary, never an in-flight partially updated state.
            last_snapshot = dict(model=cpu_copy(target.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
                                 updates=list(updates))
            event('update', step=step+1, KL=row['KL'], gradient_norm=norm, step_seconds=record['seconds'])
        optimizer.zero_grad(set_to_none=True)
        after = evaluate('after')
        checkpoint = save(a.directory/'learner.pt', dict(model={k:v.detach().cpu() for k,v in target.state_dict().items()},
              optimizer=optimizer.state_dict(), updates=updates, freeze=a.freeze, binding_sha256=a.binding_sha,
              config=config, target_schema='COMPACT_FALCON_TERNARY_TARGET_V1'))
        guard()
        gates = dict(complete_finite_gradient_updates=len(updates)==8,
                     FIT_KL_improves=after['FIT']['mean_KL'] < before['FIT']['mean_KL'],
                     DEV_KL_no_more_than_2pct_worse=after['DEV']['mean_KL'] <= before['DEV']['mean_KL']*1.02)
        result = dict(schema='HYBRID_PILOT_RESULT_V1', freeze=a.freeze, binding_sha256=a.binding_sha,
             process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()), runtime=runtime,
             decision='PILOT_RECOVERY_PASS' if all(gates.values()) else 'PILOT_RECOVERY_FAIL', gates=gates,
             complete_learner_available=True, before=before, after=after, updates=updates, checkpoint=checkpoint,
             trainable_parameters=count, initial_basis=info, new_source_generations=new_generations,
             new_updates=len(completed), GPU_allocated_peak=torch.cuda.max_memory_allocated(),
             GPU_reserved_peak=torch.cuda.max_memory_reserved(), worker_OS_peak_snapshot=proc.memory_info().peak_wset,
             elapsed_seconds=time.monotonic()-start, native_C_calls=0, final_chatbot_quality=False,
             matrix_products_per_decode=69632512, flat_router_scope='72-bank pilot only; structured winner/mass/large-n deployment unqualified')
        write(a.out, result)
        event('complete', decision=result['decision'], gates=gates)
    except BaseException as error:
        write(a.directory/'first_failure.json', dict(fault=repr(error), completed=completed,
                  runtime=runtime, elapsed_seconds=time.monotonic()-start))
        if last_snapshot is not None:
            # Retain the actual updated state. Never replay completed teacher calls or updates.
            with (a.directory/'recovery.pt').open('xb') as f:
                torch.save(last_snapshot, f)
                f.flush()
                os.fsync(f.fileno())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--binding-sha', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
