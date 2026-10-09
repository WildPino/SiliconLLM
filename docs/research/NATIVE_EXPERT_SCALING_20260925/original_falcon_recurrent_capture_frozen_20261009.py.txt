"""New full-history recurrent operands, without LM-head/answer/training calls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
from original_falcon_whole_recovery import extent, check_inputs, memory_reader
sys.path.insert(0, str(SITE))


def bind(a):
    prior_path = DOC/'chatbot_source_ffn_broad_capture_binding_20261009.json'
    prior = json.loads(prior_path.read_bytes())
    source = Path(prior['source'])
    corpus = Path(prior['corpus'])
    records = json.loads(corpus.read_bytes())['records']
    assert len(records) == 48 and {r['split'] for r in records} == {'FIT', 'DEV'}
    assert all(r['student_input_ids'] == r['input_ids']+r['output_ids'][:-1] for r in records)
    histories = sum(len(r['student_input_ids']) for r in records)
    assert histories == 22547 and max(len(r['student_input_ids']) for r in records) == 1507
    weights = source/'model.safetensors'
    original = {Path(i['path']).resolve(): i for i in prior['inputs']}
    assert extent(weights) == original[weights.resolve()]
    files = [Path(__file__), B/'chatbot_falcon_usability.py', B/'chatbot_falcon_usability_launch.py',
        B/'original_falcon_whole_recovery.py', B/'chatbot_falcon_ssd_tiles.py',
        DOC/'ORIGINAL_FALCON_RECURRENT_CAPTURE_PROTOCOL_20261009.md',
        DOC/'original_falcon_whole_adopted_result_20261009.json',
        DOC/'ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md', prior_path, corpus, weights,
        source/'config.json', Path(sys.executable)]
    files += [SITE/'torch'/p for p in ('__init__.py', '_C.cp312-win_amd64.pyd',
        'lib/torch_cpu.dll', 'lib/torch_cuda.dll', 'cuda/__init__.py')]
    files += [SITE/'transformers'/p for p in ('__init__.py', 'cache_utils.py',
        'models/falcon_h1/modeling_falcon_h1.py', 'models/falcon_h1/configuration_falcon_h1.py')]
    files += [SITE/'numpy/__init__.py', SITE/'psutil/__init__.py', SITE/'safetensors/__init__.py']
    files += [ROOT/p for p in ('benchmarks/phase60/engine.c',
        'benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py', 'docs/research/RESEARCH_INDEX.md')]
    assert shutil.disk_usage(ROOT).free >= 64 << 30
    write(a.out, dict(schema='ORIGINAL_FALCON_RECURRENT_CAPTURE_BINDING_V1',
        python=str(Path(sys.executable).resolve()), worker_path=str(Path(__file__).resolve()),
        source=str(source.resolve()), corpus=str(corpus.resolve()),
        ordered_ids=[r['id'] for r in sorted(records, key=lambda r:(r['split'] != 'FIT', r['domain'], r['id']))],
        source_revision='80ebc50d7799a440b96c93bb6686a3924a09b0cb',
        sites=list(range(24)), cases=48, history_tokens=histories, expected_payload_bytes=histories*24*29792,
        fields=dict(x=dict(width=3072, dtype='BF16'), B=dict(width=256, dtype='BF16'),
            C=dict(width=256, dtype='BF16'), gate=dict(width=3072, dtype='BF16'),
            delta=dict(width=48, dtype='BF16'), y=dict(width=3072, dtype='F32'),
            output=dict(width=2048, dtype='BF16')),
        criteria=dict(all_cases_sites_fields=True, finite=True, exact_field_extents=True,
            source_parameter_identities_versions_unchanged=True, LM_head_calls=0),
        limits=dict(seconds=1800, reserve_seconds=90, OS_bytes=12 << 30,
            GPU_allocated_bytes=10 << 30, GPU_reserved_bytes=11 << 30, output_bytes=20 << 30),
        inputs=[extent(p) for p in dict.fromkeys(files)],
        runtime_binding_scope='Pinned complete source weights/config/base model and principal runtime binaries;'
            '48 retained complete forced histories; no full DLL-tree binding or new source answers.'))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), cases=48, histories=histories,
        payload_bytes=histories*24*29792)), flush=True)


def launch(a):
    import psutil
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'ORIGINAL_FALCON_RECURRENT_CAPTURE_BINDING_V1'
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    proc = psutil.Process(); proc.cpu_affinity([11])
    own = {proc.pid, *(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name', 'cmdline']):
        name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
        if p.pid in own: continue
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): continue
        assert not name.startswith(('python', 'clang', 'gcc', 'engine', 'packed_original')), ('overlap', p.pid, name)
    log = a.out.with_suffix('.worker.log'); terminal = a.out.with_suffix('.terminal.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out, log, terminal, a.out.with_suffix('.launcher_failure.json')))
    worker = None; peak = 0; reader = memory_reader()
    record = dict(freeze=a.freeze, binding_sha256=a.binding_sha, launcher_pid=proc.pid)

    def memory():
        nonlocal peak
        peak = max(peak, reader(worker))

    def guard():
        assert time.monotonic()-start <= b['limits']['seconds'], 'family deadline'
        assert peak+proc.memory_info().peak_wset <= b['limits']['OS_bytes'], 'family OS cap'
        assert log.stat().st_size <= 4 << 20, 'log cap'

    try:
        check_inputs(b)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='6',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='6', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        argv = [b['python'], '-I', '-S', '-B', '-X', 'utf8', b['worker_path'], '--worker',
            '--binding', str(a.binding.resolve()), '--binding-sha', a.binding_sha, '--freeze', a.freeze,
            '--directory', str(a.directory.resolve()), '--out', str(a.out.resolve())]
        record['command'] = argv
        with log.open('xb') as stream:
            worker = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, creationflags=8)
            record.update(worker_pid=worker.pid, worker_creation_time=psutil.Process(worker.pid).create_time())
            offset = 0
            while worker.poll() is None:
                memory(); guard()
                try: assert not psutil.Process(worker.pid).children(recursive=True), 'unexpected descendant'
                except psutil.NoSuchProcess: pass
                with log.open('rb') as f:
                    f.seek(offset); chunk = f.read(); boundary = chunk.rfind(b'\n')+1
                    if boundary: print(chunk[:boundary].decode('utf8', errors='replace'), end='', flush=True); offset += boundary
                time.sleep(.1)
        memory(); guard(); record.update(exit_code=worker.returncode, worker_OS_peak_through_exit=peak,
            launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        assert worker.returncode == 0, log.read_text(errors='replace')[-6000:]
        check_inputs(b); r = json.loads(a.out.read_bytes())
        assert r['schema'] == 'ORIGINAL_FALCON_RECURRENT_CAPTURE_RESULT_V1'
        assert r['cases'] == 48 and r['source_base_forwards'] == 48 and r['sites_per_case'] == 24
        assert r['payload_bytes'] == b['expected_payload_bytes'] and r['LM_head_calls'] == 0
        assert r['source_generations'] == r['optimizer_updates'] == r['native_calls'] == 0
        assert r['GPU_allocated_peak'] <= b['limits']['GPU_allocated_bytes']
        assert r['GPU_reserved_peak'] <= b['limits']['GPU_reserved_bytes']
        outputs = [extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        assert sum(i['bytes'] for i in outputs) <= b['limits']['output_bytes']
        record.update(elapsed_seconds=time.monotonic()-start, result_sha256=sha(a.out),
            decision=r['decision'], output_files=outputs, resource_gates=True)
        write(terminal, record)
        print(json.dumps(dict(terminal=str(terminal), worker_pid=worker.pid, exit_code=worker.returncode,
            seconds=record['elapsed_seconds'], OS_peak=peak, decision=r['decision'])), flush=True)
    except BaseException as error:
        if worker is not None:
            if worker.poll() is None: worker.kill(); worker.wait()
            memory(); record.update(exit_code=worker.returncode, worker_OS_peak_through_exit=peak)
        record.update(fault=repr(error), elapsed_seconds=time.monotonic()-start,
            launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.launcher_failure.json'), record)
        raise


def worker(a):
    start = time.monotonic(); assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes()); a.directory.mkdir(exist_ok=False)
    stage = 'startup'; completed = []; forwards = 0; total_bytes = 0
    torch = None; proc = None
    try:
        assert sys.version_info[:3] == (3, 12, 10)
        assert Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(k) is not None for k in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from transformers import FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        from chatbot_falcon_ssd_tiles import install
        assert (torch.__version__, transformers.__version__, np.__version__, psutil.__version__) == ('2.6.0+cu124', '5.13.1', '2.4.6', '7.2.2')
        proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

        def guard():
            lim = b['limits']
            assert time.monotonic()-start <= lim['seconds']-lim['reserve_seconds'], 'worker reserve'
            assert proc.memory_info().peak_wset <= lim['OS_bytes'], 'worker OS cap'
            assert torch.cuda.max_memory_allocated() <= lim['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= lim['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'unexpected child'
            assert total_bytes <= lim['output_bytes'], 'payload cap'

        def event(**values):
            guard(); print(json.dumps(dict(seconds=time.monotonic()-start, **values)), flush=True)

        def save(site, name, tensor):
            nonlocal total_bytes
            spec = b['fields'][name]
            assert tensor.shape == (1, current['history'], spec['width']), (site, name, tensor.shape)
            assert tensor.dtype == (torch.bfloat16 if spec['dtype'] == 'BF16' else torch.float32)
            assert torch.isfinite(tensor).all().item(), (site, name, 'nonfinite')
            value = tensor[0].detach().cpu().contiguous()
            array = value.view(torch.uint16).numpy().astype('<u2', copy=False) if spec['dtype'] == 'BF16' else value.numpy().astype('<f4', copy=False)
            path = a.directory/f"{current['id']}.site{site:02d}.{name}.{'bf16' if spec['dtype'] == 'BF16' else 'f32'}"
            temp = path.with_suffix(path.suffix+'.tmp')
            with temp.open('xb') as f:
                f.write(array.tobytes()); f.flush(); os.fsync(f.fileno())
            temp.replace(path)
            item = dict(**extent(path), shape=list(array.shape), dtype=spec['dtype'])
            current['sites'][site]['fields'][name] = item; total_bytes += item['bytes']
            assert item['bytes'] == current['history']*spec['width']*(2 if spec['dtype'] == 'BF16' else 4)

        write(a.directory/'source_method.json', install(code))
        stage = 'source_load'
        model = FalconH1ForCausalLM.from_pretrained(b['source'], local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        identities = {name: (id(p), p._version) for name, p in model.named_parameters()}
        assert len(identities) == 411 and sum(p.numel() for p in model.parameters()) == 1554863488
        assert model.config.mamba_chunk_size == 128 and not code.is_fast_path_available
        assert model.config.mamba_d_state == 256 and model.config.mamba_n_groups == 1
        assert model.config.mamba_rms_norm and not model.config.mamba_norm_before_gate
        current = None; staged = {}; head_calls = 0

        def head_hook(module, args):
            nonlocal head_calls
            head_calls += 1
            raise AssertionError('LM head must not be called')

        hooks = [model.lm_head.register_forward_pre_hook(head_hook)]

        def in_observer(site, mixer, module, args, output):
            assert current is not None and site not in staged
            projected = output * mixer.mup_vector
            gate, raw_bc, raw_delta = projected.split([3072, 3584, 48], dim=-1)
            delta = torch.nn.functional.softplus(raw_delta+mixer.dt_bias)
            delta = torch.clamp(delta, mixer.time_step_limit[0], mixer.time_step_limit[1])
            staged[site] = dict(gate=gate, delta=delta)

        def conv_observer(site, mixer, module, args, output):
            assert site in staged
            activated = mixer.act(output[..., :current['history']]).transpose(1, 2)
            xx, bb, cc = activated.split([3072, 256, 256], dim=-1)
            staged[site].update(x=xx, B=bb, C=cc)

        def norm_observer(site, module, args):
            nonlocal stage
            stage = f"{current['id']}/site{site:02d}/norm"
            yy, gate = args
            assert torch.equal(gate, staged[site]['gate'])
            for name in ('x', 'B', 'C', 'gate', 'delta'): save(site, name, staged[site][name])
            save(site, 'y', yy)
            del staged[site]
            guard()

        def output_observer(site, module, args, output):
            save(site, 'output', output)

        for site, layer in enumerate(model.model.layers):
            mixer = layer.mamba
            hooks.append(mixer.in_proj.register_forward_hook(lambda m, args, out, site=site, mixer=mixer: in_observer(site, mixer, m, args, out)))
            hooks.append(mixer.conv1d.register_forward_hook(lambda m, args, out, site=site, mixer=mixer: conv_observer(site, mixer, m, args, out)))
            hooks.append(mixer.norm.register_forward_pre_hook(lambda m, args, site=site: norm_observer(site, m, args)))
            hooks.append(mixer.register_forward_hook(lambda m, args, out, site=site: output_observer(site, m, args, out)))
        records = {r['id']: r for r in json.loads(Path(b['corpus']).read_bytes())['records']}
        event(stage='source_loaded', parameters=len(identities), capture_cases=48)
        for identifier in b['ordered_ids']:
            rec = records[identifier]; stage = identifier
            current = dict(id=identifier, split=rec['split'], domain=rec['domain'], history=len(rec['student_input_ids']),
                input_ids=rec['student_input_ids'], positions=rec['positions'],
                sites=[dict(site=site, fields={}) for site in b['sites']])
            guard()
            with torch.inference_mode():
                ids = torch.tensor([current['input_ids']], device='cuda', dtype=torch.long)
                result = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                forwards += 1
                assert torch.isfinite(result.last_hidden_state).all().item()
            assert not staged and all(set(v['fields']) == set(b['fields']) for v in current['sites'])
            assert current['history']*24*29792 == sum(f['bytes'] for v in current['sites'] for f in v['fields'].values())
            write(a.directory/(identifier+'.json'), current); completed.append(current)
            event(stage='captured_case', id=identifier, split=rec['split'], history=current['history'],
                cases=len(completed), payload_bytes=total_bytes, source_base_forwards=forwards)
            del ids, result; current = None
        for hook in hooks: hook.remove()
        assert {name: (id(p), p._version) for name, p in model.named_parameters()} == identities
        assert head_calls == 0 and total_bytes == b['expected_payload_bytes']
        result = dict(schema='ORIGINAL_FALCON_RECURRENT_CAPTURE_RESULT_V1', freeze=a.freeze,
            binding_sha256=a.binding_sha, decision='COMPLETE_RECURRENT_OPERANDS_PASS',
            cases=len(completed), sites_per_case=24, history_tokens=b['history_tokens'], records=completed,
            payload_bytes=total_bytes, source_base_forwards=forwards, LM_head_calls=head_calls,
            source_generations=0, optimizer_updates=0, native_calls=0, reserved_queries=0,
            source_parameter_identities_versions_unchanged=True,
            full_prefill_scope='One full forced-history base-model call per case, zero initial state; '
                'not a byte-equivalence claim to earlier incremental cached teacher logits.',
            quality_admission=False, speed_admission=False,
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start)
        write(a.out, result); event(stage='complete', decision=result['decision'], result_sha256=sha(a.out))
    except BaseException as error:
        write(a.directory/'first_fault.json', dict(stage=stage, error=repr(error),
            completed_ids=[r['id'] for r in completed], source_base_forwards=forwards,
            payload_bytes=total_bytes, elapsed_seconds=time.monotonic()-start,
            worker_OS_peak_snapshot=None if proc is None else proc.memory_info().peak_wset,
            GPU_allocated_peak=None if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=None if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind', action='store_true'); m.add_argument('--launch', action='store_true'); m.add_argument('--worker', action='store_true')
    p.add_argument('--binding', type=Path); p.add_argument('--binding-sha'); p.add_argument('--freeze')
    p.add_argument('--directory', type=Path); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.bind: bind(args)
    elif args.launch: launch(args)
    else: worker(args)
