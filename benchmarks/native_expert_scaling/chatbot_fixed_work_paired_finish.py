"""Complete only missing paired B26 observations after the original time cap."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
PREVIOUS = ROOT / 'results/native_expert_scaling/chatbot_fixed_work_paired_20261009'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    original_path = DOC / 'chatbot_fixed_work_paired_binding_20261009.json'
    original = json.loads(original_path.read_bytes())
    receipt_path = DOC / 'chatbot_fixed_work_paired_result_20261009.launcher_failure.json'
    receipt = json.loads(receipt_path.read_bytes())
    fault = json.loads((PREVIOUS / 'first_failure.json').read_bytes())
    assert receipt['exit_code'] == 1 and receipt['binding_sha256'] == sha(original_path)
    assert fault['phase'] == 'B_after' and fault['completed_worker_updates'] == 24
    assert fault['fault'] == "AssertionError('worker deadline reserve')" and not fault['optimizer_commit_in_progress']
    state = json.loads((PREVIOUS / 'B_boundary_26.json').read_bytes())['checkpoint']
    duplicate = json.loads((PREVIOUS / 'failure_state.json').read_bytes())['checkpoint']
    assert state['sha256'] == duplicate['sha256'] == '2384fbafeda4d1b64a7982bc6cae50557af72ba91abd4ea2c6dbe85c61ee5a05'
    files = []
    frozen = DOC / 'chatbot_fixed_work_paired_launcher_frozen_20261009.py.txt'
    for entry in original['inputs']:
        p = Path(entry['path'])
        if p.resolve() == (B / 'chatbot_falcon_usability_launch.py').resolve(): p = frozen
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], str(p)
        files.append(p)
    files.extend(p for p in PREVIOUS.iterdir() if p.is_file())
    files.extend([Path(__file__), B / 'chatbot_falcon_usability_launch.py', original_path, receipt_path,
                  DOC / 'chatbot_fixed_work_paired_result_20261009.worker.log',
                  DOC / 'CHATBOT_FIXED_WORK_PAIRED_FINISH_PROTOCOL_20261009.md'])
    seen = {json.loads(p.read_bytes())['id']: str(p.resolve()) for p in PREVIOUS.glob('B_after.*.json')}
    assert len(seen) == 17 and not list(PREVIOUS.glob('B_retention.*.json'))
    assert len(json.loads((PREVIOUS / 'A_result.json').read_bytes())['new_updates']) == 26
    assert len(list(PREVIOUS.glob('B_update_*.json'))) == 24
    records = json.loads(Path(original['corpus']).read_bytes())['records']
    old = json.loads(Path(original['old_corpus']).read_bytes())['records']
    missing = [r for r in records if r['id'] not in seen] + [r for r in old if r['id'] in original['retention_ids']]
    assert len(missing) == 63
    expected_bytes = sum(len(r['output_ids']) for r in missing) * 65537 * 4
    assert expected_bytes < 4 << 30
    files = list(dict.fromkeys(p.resolve() for p in files))
    assert not a.out.exists() and a.freeze
    value = dict(schema='FIXED_WORK_PAIRED_FINISH_BINDING_V1', freeze=a.freeze,
        python=str(Path(sys.executable).resolve()), worker_path=str(Path(__file__).resolve()),
        original_binding=str(original_path.resolve()), original_directory=str(PREVIOUS),
        checkpoint=state, reused_B_after=seen, new_student_cases=63, new_optimizer_updates=0,
        expected_new_logits_bytes=expected_bytes, original_family_seconds=receipt['elapsed_seconds'],
        criteria=original['criteria'], corpus=original['corpus'], old_corpus=original['old_corpus'],
        retention_ids=original['retention_ids'], limits=dict(seconds=600, reserve_seconds=45, OS_bytes=8 << 30,
          GPU_allocated_bytes=10 << 30, GPU_reserved_bytes=11 << 30, output_bytes=4 << 30),
        runtime_binding_scope='All first-attempt inputs reverified with frozen original launcher, every retained output/fault/checkpoint, unchanged model operators and original prospective criteria; evaluation only, no optimizer or source calls.',
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    write(a.out, value)
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(files), new_cases=63, new_logits_bytes=expected_bytes)), flush=True)


def worker(a):
    start = time.monotonic(); assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes()); assert b['schema'] == 'FIXED_WORK_PAIRED_FINISH_BINDING_V1' and b['freeze'] == a.freeze
    a.directory.mkdir(exist_ok=False); phase = 'imports'; completed = 0
    try:
        import numpy as np
        import psutil
        import torch
        import transformers
        from transformers.models.falcon_h1 import modeling_falcon_h1 as source_code
        import chatbot_hybrid_fixed_work_target as variant
        import chatbot_target_ssd_storage as storage
        assert sys.version_info[:3] == (3, 12, 10)
        assert (torch.__version__, transformers.__version__, np.__version__) == ('2.6.0+cu124', '5.13.1', '2.4.6')
        assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        def guard():
            v = b['limits']; assert time.monotonic() - start <= v['seconds'] - v['reserve_seconds']
            assert proc.memory_info().peak_wset <= v['OS_bytes']
            assert torch.cuda.max_memory_allocated() <= v['GPU_allocated_bytes']
            assert torch.cuda.max_memory_reserved() <= v['GPU_reserved_bytes']
            assert not proc.children(recursive=True)
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= v['output_bytes']
        def event(stage, **fields):
            print(json.dumps(dict(stage=stage, seconds=time.monotonic() - start, **fields)), flush=True); guard()
        (a.directory / 'generated_method.py.txt').write_text(storage.install(source_code), encoding='utf8')
        previous = Path(b['original_directory'])
        records = json.loads(Path(b['corpus']).read_bytes())['records']
        old_records = json.loads(Path(b['old_corpus']).read_bytes())['records']
        retention = [r for r in old_records if r['id'] in b['retention_ids']]
        all_records = {r['id']: r for r in records + retention}
        phase = 'adoption'
        old = torch.load(b['checkpoint']['path'], map_location='cpu', weights_only=True)
        assert old['candidate_schema'] == 'FIXED_WORK_PAIRED_CANDIDATE_V1' and old['target_schema'] == variant.SCHEMA
        assert old['arm'] == 'B' and old['completed_worker_updates'] == 24 and old['completed_paired_updates'] == 26
        assert old['private_optimizer_step'] == 344 and old['common_optimizer_step'] == 26
        target = variant.Target(old['config'], 'ternary').to('cuda'); target.load_state_dict(old['model'], strict=True)
        named = list(target.named_parameters()); assert [n for n, _ in named] == old['optimizer_parameter_names']
        assert len(named) == len(old['optimizer']['state']) == 283 and sum(p.numel() for _, p in named) == 259669760
        for index, (name, p) in enumerate(named):
            assert p.dtype == torch.float32 and torch.isfinite(p).all().item()
            assert torch.equal(p.detach().cpu().contiguous().view(torch.int32), old['model'][name].contiguous().view(torch.int32))
            slot = old['optimizer']['state'][index]
            assert int(slot['step'].item()) == (26 if '.banks.common.' in name else 344)
            assert np.isfinite(slot['exp_avg'].numpy()).all() and np.isfinite(slot['exp_avg_sq'].numpy()).all()
            assert (slot['exp_avg_sq'].numpy() >= 0).all()
        cpu_rng = old['torch_CPU_rng'].clone(); cuda_rng = [v.clone() for v in old['torch_CUDA_rng']]
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
        del old, slot; gc.collect(); target.eval(); event('adopted', counters=[344, 26], optimizer_updates=0)
        active = None; exposure = {}
        def route(site):
            @torch.no_grad()
            def hook(ids, mass):
                rec = all_records[active]; count = len(rec['student_input_ids'])
                assert ids.shape == mass.shape == (count, 6)
                assert torch.isfinite(mass).all().item() and torch.all(mass > 0).item()
                assert torch.all((ids >= 0) & (ids < 72)).item() and torch.all(ids.sort(-1).values.diff(dim=-1) > 0).item()
                defect = (mass.sum(-1) - 1).abs().max().item(); assert defect <= 1e-6
                row = exposure.setdefault(active, dict(private=[[0] * 72 for _ in range(12)], common_positions=[0] * 12,
                                                       mass_max_defect=[0.] * 12, common_norm2=[0.] * 12))
                row['private'][site] = torch.bincount(ids.flatten(), minlength=72).cpu().tolist()
                row['mass_max_defect'][site] = defect
            return hook
        def common(site):
            @torch.no_grad()
            def hook(module, args, output):
                assert torch.isfinite(output).all().item()
                exposure[active]['common_positions'][site] = 2 * output.shape[1]
                exposure[active]['common_norm2'][site] = output.double().square().sum().item()
            return hook
        for site, layer in enumerate(target.layers):
            layer.banks.private.routing_observer = route(site)
            layer.banks.common.register_forward_hook(common(site))
        def values(rows):
            n = sum(r['labels'] for r in rows); err = sum(r['disagreements'] for r in rows)
            return dict(cases=len(rows), labels=n, case_KL=sum(r['KL'] for r in rows) / len(rows),
                label_KL=sum(sum(r['label_KL']) for r in rows) / n, disagreements=err, label_disagreement=err / n,
                case_disagreement=sum(r['disagreements'] / r['labels'] for r in rows) / len(rows))
        def summarize(rows, broad):
            v = dict(overall=values(rows), domains={d: values([r for r in rows if r['domain'] == d]) for d in sorted({r['domain'] for r in rows})})
            if broad:
                v.update({s: values([r for r in rows if r['split'] == s]) for s in ('FIT', 'DEV')})
                v['DEV_domains'] = {d: values([r for r in rows if r['domain'] == d and r['split'] == 'DEV']) for d in v['domains']}
            return v
        def logp64(x):
            shifted = x.astype(np.float64) - x.max(-1, keepdims=True)
            return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
        evaluated = {}
        for stage, cohort in [('B_after', records), ('B_retention', retention)]:
            phase = stage; rows = []
            with torch.no_grad():
                for rec in cohort:
                    guard(); t0 = time.monotonic()
                    if stage == 'B_after' and rec['id'] in b['reused_B_after']:
                        row = json.loads(Path(b['reused_B_after'][rec['id']]).read_bytes())
                        row = dict(row, completion_observation_reused=True)
                    else:
                        active = rec['id']
                        logits = target(torch.tensor([rec['student_input_ids']], device='cuda'),
                                        torch.tensor(rec['positions'], device='cuda')).squeeze(0)
                        assert list(logits.shape) == rec['logits']['shape'] and torch.isfinite(logits).all().item()
                        raw = logits.cpu().numpy()
                        bits = np.fromfile(rec['logits']['path'], dtype='<u2').astype('<u4')
                        teacher = (bits << 16).view('<f4').reshape(rec['logits']['shape']); assert np.isfinite(teacher).all()
                        losses = []
                        for offset in range(0, len(raw), 16):
                            lp = logp64(teacher[offset:offset + 16]); lq = logp64(raw[offset:offset + 16])
                            per = (np.exp(lp) * (lp - lq)).sum(-1)
                            assert np.isfinite(per).all() and per.min() >= -1e-10; losses.extend(per.tolist())
                        winners = raw.argmax(-1).tolist(); support = exposure[rec['id']]
                        assert all(sum(v) == len(rec['student_input_ids']) * 6 for v in support['private'])
                        assert support['common_positions'] == [2 * len(rec['student_input_ids'])] * 12
                        path = a.directory / (stage + '.' + rec['id'] + '.logits.f32')
                        with path.open('xb') as f:
                            f.write(raw.astype('<f4', copy=False).tobytes()); f.flush(); os.fsync(f.fileno())
                        row = dict(id=rec['id'], domain=rec['domain'], split=rec['split'], labels=len(rec['output_ids']),
                            teacher_forcing_ids=len(rec['student_input_ids']), KL=float(np.mean(losses)), label_KL=losses,
                            predicted_ids=winners, disagreements=sum(x != y for x, y in zip(winners, rec['output_ids'], strict=True)),
                            exposure=support, seconds=time.monotonic() - t0, completion_observation_reused=False,
                            logits=dict(path=str(path.resolve()), shape=list(raw.shape), bytes=path.stat().st_size, sha256=sha(path)))
                        completed += 1
                    assert torch.equal(cpu_rng, torch.get_rng_state())
                    assert all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
                    rows.append(row); write(a.directory / (stage + '.' + rec['id'] + '.json'), row)
                    event(stage, id=rec['id'], KL=row['KL'], differing=row['disagreements'], reused=row['completion_observation_reused'])
            evaluated[stage] = dict(cases=rows, summary=summarize(rows, stage == 'B_after'))
            write(a.directory / (stage + '.json'), evaluated[stage])
        assert completed == 63
        result_a = json.loads((previous / 'A_result.json').read_bytes())
        updates_b = [json.loads((previous / (f'B_update_{i}.json')).read_bytes()) for i in range(1, 25)]
        result_b = dict(before=json.loads((previous / 'B_before.json').read_bytes()), after=evaluated['B_after'],
            retention=evaluated['B_retention'], checkpoint=b['checkpoint'], new_updates=updates_b,
            worker_new_updates=24, reused_updates=2, final_private_step=344, final_common_step=26,
            parameters=259669760, tensors=283, adoption_bit_exact=True, actual_exposure_verified=True)
        baseline = json.loads((previous / 'parent_outputs_reused.json').read_bytes())
        c = b['criteria']; gates = {}
        for key, r in [('A', result_a), ('B', result_b)]:
            post = r['after']['summary']; ret = r['retention']['summary']['overall']; parent = baseline['retention']['overall']
            gates[key] = dict(DEV_absolute_KL=post['DEV']['case_KL'] <= c['DEV_case_KL_max'],
                DEV_absolute_disagreement=post['DEV']['label_disagreement'] <= c['DEV_label_disagreement_max'],
                DEV_every_domain_KL=all(v['case_KL'] <= c['domain_DEV_case_KL_max'] for v in post['DEV_domains'].values()),
                DEV_every_domain_disagreement=all(v['label_disagreement'] <= c['domain_DEV_label_disagreement_max'] for v in post['DEV_domains'].values()),
                old_DEV_KL_retention=ret['case_KL'] <= c['retention_ratio_max'] * parent['case_KL'],
                old_DEV_disagreement_retention=ret['label_disagreement'] <= c['retention_ratio_max'] * parent['label_disagreement'])
        pa = result_a['after']['summary']; pb = result_b['after']['summary']
        preference = dict(B_DEV_KL=pb['DEV']['case_KL'] <= c['B_DEV_case_KL_ratio_max'] * pa['DEV']['case_KL'],
            B_every_domain_KL=all(pb['DEV_domains'][d]['case_KL'] <= c['B_every_domain_ratio_max'] * pa['DEV_domains'][d]['case_KL'] for d in pa['DEV_domains']),
            B_every_domain_disagreement=all(pb['DEV_domains'][d]['label_disagreement'] <= c['B_every_domain_ratio_max'] * pa['DEV_domains'][d]['label_disagreement'] for d in pa['DEV_domains']))
        preferred = all(preference.values()) and gates['B']['old_DEV_KL_retention'] and gates['B']['old_DEV_disagreement_retention']
        assert sum(p.stat().st_size for p in a.directory.glob('*.logits.f32')) == b['expected_new_logits_bytes']
        phase = 'result'; guard()
        result = dict(schema='FIXED_WORK_PAIRED_FINISH_RESULT_V1', decision='PAIRED_B_PREFERENCE_PASS' if preferred else 'PAIRED_B_PREFERENCE_FAIL',
            freeze=a.freeze, binding_sha256=a.binding_sha, arms=dict(A=result_a, B=result_b), preference_gates=preference,
            absolute_retention_gates=gates, new_student_cases=63, reused_B_after_cases=17,
            new_optimizer_updates=0, retained_original_new_updates=50, reused_B_updates=2,
            original_family_seconds=b['original_family_seconds'], original_first_fault='worker deadline reserve after 50 complete updates/17 B final observations',
            actual_exposure_verified=True, model_adoption_bit_exact=True, parent_outputs_reused=baseline,
            source_calls=0, native_runs=0, reserved_queries=0, quality_admission=False, native_admission=False,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset, elapsed_seconds=time.monotonic() - start,
            scope='Only63 previously missing B26 output observations; original prospective scientific criteria retained. Complete paired development comparison, not fresh/native/useful-n admission.')
        write(a.out, result); event('complete', decision=result['decision'], preference=preference, gates=gates)
    except BaseException as error:
        write(a.directory / 'first_failure.json', dict(fault=repr(error), phase=phase, new_observations=completed, elapsed_seconds=time.monotonic() - start))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('bind', 'worker'), default='worker')
    p.add_argument('--binding', type=Path); p.add_argument('--binding-sha'); p.add_argument('--freeze')
    p.add_argument('--directory', type=Path); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); bind(a) if a.mode == 'bind' else worker(a)
