"""Actual source residual boundaries for offline original-engine distillation.

No LM head, answer, optimizer or native-engine call. Frozen recurrent operands
qualify the repeated source history; projection is auxiliary supervision only.
"""
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
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
from original_falcon_whole_recovery import extent, check_inputs, memory_reader
sys.path.insert(0, str(SITE))
SITES = [3, 7, 11, 15, 19, 23]
BOUNDARIES = [0, 4, 8, 12, 16, 20, 24]


def bind(a):
    capture = DOC/'original_falcon_recurrent_adopted_result_20261009.json'
    assert sha(capture) == 'bdec178a948e49bbec1d9eb4c01076bec9c182be8347306eb5dc02e0c873c849'
    r = json.loads(capture.read_bytes())
    assert r['cases'] == 48 and r['sites_per_case'] == 24 and r['stored_finite_extent_hash_adoption']
    parent = json.loads(Path(r['adoption_binding']['path']).read_bytes())
    source = Path(parent['source'])
    corpus = Path(parent['corpus'])
    rows = json.loads(corpus.read_bytes())['records']
    adopted = {rec['id']: rec for rec in r['records']}
    assert len(rows) == 48 and sum(len(rec['student_input_ids']) for rec in rows) == 22547
    witnesses = []
    for rec in sorted(rows, key=lambda rec:(rec['split'] != 'FIT', rec['domain'], rec['id'])):
        old = adopted[rec['id']]
        assert rec['student_input_ids'] == rec['input_ids']+rec['output_ids'][:-1] == old['input_ids']
        assert len(rec['student_input_ids']) == old['history']
        witnesses.append(dict(id=rec['id'], split=rec['split'], domain=rec['domain'], history=old['history'],
            outputs=[dict(site=site, **old['sites'][site]['fields']['output']) for site in SITES]))
    assert all(len([rec for rec in rows if rec['split'] == split]) == 24 for split in ('FIT', 'DEV'))
    basis = ROOT/'results/native_expert_scaling/original_falcon_transfer_20261009/basis256.pt'
    transfer_terminal = DOC/'original_falcon_transfer_result_20261009.terminal.json'
    old_outputs = json.loads(transfer_terminal.read_bytes())['output_files']
    assert extent(basis) == next(item for item in old_outputs if Path(item['path']) == basis.resolve())
    files = [Path(__file__), B/'chatbot_falcon_usability.py', B/'chatbot_falcon_usability_launch.py',
        B/'original_falcon_whole_recovery.py', B/'chatbot_falcon_ssd_tiles.py', capture,
        Path(r['adoption_binding']['path']), source/'model.safetensors', source/'config.json',
        source/'source_package.json', corpus, basis, transfer_terminal, Path(sys.executable),
        DOC/'ORIGINAL_HISTORY_BOUNDARIES_PROTOCOL_20261009.md',
        DOC/'original_falcon_recurrent_capture_result_20261009.worker.log',
        DOC/'original_falcon_recurrent_resume_result_20261009.worker.log']
    files += [Path(item['path']) for rec in witnesses for item in rec['outputs']]
    for rec in witnesses:
        for item in rec['outputs']:
            assert extent(item['path']) == {key:item[key] for key in ('path', 'bytes', 'sha256')}
    source_expected = {Path(item['path']).resolve():item for item in parent['inputs']}
    for path in (source/'model.safetensors', source/'config.json', corpus):
        assert extent(path) == source_expected[path.resolve()]
    files += [SITE/'torch'/p for p in ('__init__.py', '_C.cp312-win_amd64.pyd',
        'lib/torch_cpu.dll', 'lib/torch_cuda.dll', 'cuda/__init__.py')]
    files += [SITE/'transformers'/p for p in ('__init__.py', 'cache_utils.py',
        'models/falcon_h1/modeling_falcon_h1.py', 'models/falcon_h1/configuration_falcon_h1.py')]
    files += [SITE/'numpy/__init__.py', SITE/'psutil/__init__.py', SITE/'safetensors/torch.py']
    files += [ROOT/p for p in ('benchmarks/phase60/engine.c', 'benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py', 'docs/research/RESEARCH_INDEX.md')]
    assert shutil.disk_usage(ROOT).free >= 4 << 30
    write(a.out, dict(schema='ORIGINAL_HISTORY_BOUNDARIES_BINDING_V1',
        python=str(Path(sys.executable).resolve()), worker_path=str(Path(__file__).resolve()),
        capture=str(capture.resolve()), source=str(source.resolve()), corpus=str(corpus.resolve()),
        basis=str(basis.resolve()), boundaries=BOUNDARIES, witness_sites=SITES, witnesses=witnesses,
        source_revision='80ebc50d7799a440b96c93bb6686a3924a09b0cb', cases=48, history_tokens=22547,
        raw_payload_bytes=646467584, projected_payload_bytes=161616896,
        criteria=dict(exact_history_outputs=288, source_parameters_unchanged=411, boundaries_per_case=7,
            finite=True, raw_extents_exact=True, projection_orthogonality_max=1e-4,
            projection_relative_RMS=1e-5, LM_head_calls=0),
        limits=dict(seconds=1800, reserve_seconds=120, OS_bytes=12 << 30,
            GPU_allocated_bytes=8 << 30, GPU_reserved_bytes=9 << 30, output_bytes=1 << 30, log_bytes=4 << 20),
        inputs=[extent(path) for path in dict.fromkeys(files)],
        arithmetic='Actual source BF16 zero-state full prefill. Raw h0 multiplied embedding and h4..h24 decoder outputs;'
            'z=h.float()@fixed inherited P.float() CUDA F32, TF32 disabled. CPU F64 energy/moment summaries.',
        scope='Auxiliary teacher residual coordinates; no source-logit transport, new answer labels, optimizer,'
            'native operator changes or retroactive parent resource qualification. Principal binaries, not full DLL tree.'))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(dict.fromkeys(files)))), flush=True)


def launch(a):
    import psutil
    start = time.monotonic(); assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes()); assert b['schema'] == 'ORIGINAL_HISTORY_BOUNDARIES_BINDING_V1'
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    proc = psutil.Process(); proc.cpu_affinity([11]); own = {proc.pid, *(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name', 'cmdline']):
        name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
        if p.pid in own: continue
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): continue
        assert not name.startswith(('python', 'clang', 'gcc', 'engine', 'packed_original', 'native_wide')), ('overlap', p.pid, name)
    log = a.out.with_suffix('.worker.log'); terminal = a.out.with_suffix('.terminal.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out, log, terminal, a.out.with_suffix('.launcher_failure.json')))
    worker = None; peak = 0; reader = memory_reader()
    record = dict(freeze=a.freeze, binding_sha256=a.binding_sha, launcher_pid=proc.pid)
    def memory():
        nonlocal peak
        if worker is not None: peak = max(peak, reader(worker))
    def guard():
        assert time.monotonic()-start <= b['limits']['seconds'], 'family deadline'
        assert peak+proc.memory_info().peak_wset <= b['limits']['OS_bytes'], 'family OS cap'
        assert log.stat().st_size <= b['limits']['log_bytes'], 'log cap'
    try:
        check_inputs(b)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='6', CUBLAS_WORKSPACE_CONFIG=':4096:8')
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
                    f.seek(offset); chunk=f.read(); end=chunk.rfind(b'\n')+1
                    if end: print(chunk[:end].decode('utf8', errors='replace'), end='', flush=True); offset += end
                time.sleep(.1)
        memory(); guard(); record.update(exit_code=worker.returncode, worker_OS_peak_through_exit=peak,
            launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        assert worker.returncode == 0, log.read_text(errors='replace')[-6000:]
        check_inputs(b); r=json.loads(a.out.read_bytes())
        assert r['schema'] == 'ORIGINAL_HISTORY_BOUNDARIES_RESULT_V1'
        assert r['cases'] == r['source_base_forwards'] == 48 and r['exact_history_output_witnesses'] == 288
        assert r['raw_payload_bytes'] == b['raw_payload_bytes'] and r['projected_payload_bytes'] == b['projected_payload_bytes']
        assert r['LM_head_calls'] == r['optimizer_updates'] == r['native_calls'] == r['source_generations'] == 0
        assert r['source_parameter_identities_versions_unchanged']
        assert r['GPU_allocated_peak'] <= b['limits']['GPU_allocated_bytes'] and r['GPU_reserved_peak'] <= b['limits']['GPU_reserved_bytes']
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        assert sum(item['bytes'] for item in outputs) <= b['limits']['output_bytes']
        record.update(elapsed_seconds=time.monotonic()-start, result_sha256=sha(a.out), output_files=outputs,
            decision=r['decision'], resource_gates=True)
        write(terminal, record)
        print(json.dumps(dict(terminal=str(terminal), worker_pid=worker.pid, exit_code=worker.returncode,
            seconds=record['elapsed_seconds'], decision=r['decision'])), flush=True)
    except BaseException as error:
        if worker is not None:
            if worker.poll() is None: worker.kill(); worker.wait()
            memory(); record.update(exit_code=worker.returncode, worker_OS_peak_through_exit=peak)
        record.update(fault=repr(error), elapsed_seconds=time.monotonic()-start,
            launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.launcher_failure.json'), record)
        raise


def energy(array):
    """Uncentered = mean + centered, F64 after the declared saved precision."""
    import numpy as np
    x=array.astype('f8'); total=float(np.sum(x*x)); mean=x.mean(0)
    mean_energy=float(x.shape[0]*np.sum(mean*mean)); centered=float(np.sum((x-mean)**2))
    assert abs(total-mean_energy-centered) <= 1e-10*max(total, 1e-24)
    return dict(total=total, mean=mean_energy, centered=centered)


def worker(a):
    start=time.monotonic(); assert sha(a.binding) == a.binding_sha
    b=json.loads(a.binding.read_bytes()); a.directory.mkdir(exist_ok=False)
    stage='startup'; completed=[]; forwards=0; attempts=0; witness_count=0; raw_bytes=0; projected_bytes=0
    torch=None; proc=None
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
        proc=psutil.Process(); proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        def guard():
            lim=b['limits']
            assert time.monotonic()-start <= lim['seconds']-lim['reserve_seconds'], 'worker reserve'
            assert proc.memory_info().peak_wset <= lim['OS_bytes'], 'worker OS cap'
            assert torch.cuda.max_memory_allocated() <= lim['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= lim['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'unexpected child'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= lim['output_bytes'], 'output cap'
        def event(**values):
            guard(); print(json.dumps(dict(seconds=time.monotonic()-start, **values)), flush=True)
        def raw_file(name, array):
            path=a.directory/name; tmp=path.with_suffix(path.suffix+'.tmp')
            with tmp.open('xb') as f: f.write(array.tobytes()); f.flush(); os.fsync(f.fileno())
            tmp.replace(path)
            return dict(**extent(path), shape=list(array.shape), dtype=str(array.dtype))
        stage='basis'
        P=torch.load(b['basis'], map_location='cpu', weights_only=True)['P'].float().contiguous()
        assert tuple(P.shape) == (2048, 256) and torch.isfinite(P).all()
        pp=P.numpy().astype('f8'); defect=float(np.max(np.abs(pp.T@pp-np.eye(256))))
        assert defect <= b['criteria']['projection_orthogonality_max'], ('basis orthogonality', defect)
        pfile=raw_file('P256.f32', P.numpy().astype('<f4', copy=False)); P=P.to('cuda')
        write(a.directory/'source_method.json', install(code))
        stage='source_load'
        model=FalconH1ForCausalLM.from_pretrained(b['source'], local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        identities={name:(id(p), p._version) for name,p in model.named_parameters()}
        assert len(identities) == 411 and sum(p.numel() for p in model.parameters()) == 1554863488
        assert model.config.mamba_chunk_size == 128 and not code.is_fast_path_available
        current=None; head_calls=0
        def head_hook(module, args):
            nonlocal head_calls
            head_calls += 1; raise AssertionError('LM head must not be called')
        hooks=[model.lm_head.register_forward_pre_hook(head_hook)]
        def save_boundary(boundary, value):
            nonlocal raw_bytes, projected_bytes, stage
            stage=f"{current['id']}/h{boundary}"
            assert boundary not in {item['boundary'] for item in current['boundaries']}
            assert tuple(value.shape) == (1, current['history'], 2048) and value.dtype == torch.bfloat16
            assert torch.isfinite(value).all()
            h=value[0].detach(); z=h.float()@P; assert torch.isfinite(z).all()
            raw=h.cpu().contiguous().view(torch.uint16).numpy().astype('<u2', copy=False)
            zz=z.cpu().numpy().astype('<f4', copy=False)
            hh=(raw.astype('<u4') << 16).view('<f4')
            eh=energy(hh); ez=energy(zz)
            # Orthogonal projection's discarded energy, with measured basis defect.
            discarded=eh['total']-ez['total']; assert discarded >= -1e-4*max(eh['total'], 1e-24)
            item=dict(boundary=boundary,
                h=raw_file(f"{current['id']}.h{boundary:02d}.bf16", raw),
                z=raw_file(f"{current['id']}.h{boundary:02d}.P256.f32", zz),
                source_energy=eh, projected_energy=ez, discarded_energy_difference=discarded,
                retained_energy_fraction=ez['total']/max(eh['total'], 1e-24),
                centered_retained_fraction=ez['centered']/max(eh['centered'], 1e-24))
            raw_bytes += item['h']['bytes']; projected_bytes += item['z']['bytes']
            current['boundaries'].append(item); guard()
        def first_boundary(module, args, kwargs):
            value=kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            save_boundary(0, value)
        hooks.append(model.model.layers[0].register_forward_pre_hook(first_boundary, with_kwargs=True))
        def output_witness(site, module, args, output):
            nonlocal witness_count, stage
            stage=f"{current['id']}/witness{site}"
            assert output.dtype == torch.bfloat16 and tuple(output.shape) == (1, current['history'], 2048)
            observed=output[0].detach().cpu().contiguous().view(torch.uint16).numpy()
            reference=current['_references'][site]
            saved=np.fromfile(reference['path'], dtype='<u2').reshape(current['history'],2048)
            if not np.array_equal(observed, saved):
                fault=raw_file('first_differing_mixer_output.bf16', observed.astype('<u2', copy=False))
                raise AssertionError(('recurrent output differs', current['id'], site, int(np.count_nonzero(observed != saved)), fault))
            witness_count += 1; current['exact_output_witnesses'].append(site)
        for site in SITES:
            hooks.append(model.model.layers[site].mamba.register_forward_hook(
                lambda m,args,out,site=site: output_witness(site,m,args,out)))
            hooks.append(model.model.layers[site].register_forward_hook(
                lambda m,args,out,site=site: save_boundary(site+1,out[0])))
        rows={rec['id']:rec for rec in json.loads(Path(b['corpus']).read_bytes())['records']}
        event(stage='source_loaded', source_parameters=411, basis_orthogonality_max=defect)
        for rec in b['witnesses']:
            current=dict(id=rec['id'], split=rec['split'], domain=rec['domain'], history=rec['history'],
                boundaries=[], exact_output_witnesses=[], _references={item['site']:item for item in rec['outputs']})
            guard(); attempts += 1
            with torch.inference_mode():
                ids=torch.tensor([rows[rec['id']]['student_input_ids']], device='cuda', dtype=torch.long)
                result=model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                forwards += 1; assert torch.isfinite(result.last_hidden_state).all()
            assert [item['boundary'] for item in current['boundaries']] == BOUNDARIES
            assert current['exact_output_witnesses'] == SITES
            del current['_references']; write(a.directory/(rec['id']+'.json'),current); completed.append(current)
            event(stage='captured_case', id=rec['id'], cases=len(completed), raw_bytes=raw_bytes,
                projected_bytes=projected_bytes, exact_output_witnesses=witness_count)
            del ids,result; current=None
        for hook in hooks: hook.remove()
        assert {name:(id(p),p._version) for name,p in model.named_parameters()} == identities
        assert head_calls == 0 and witness_count == 288
        assert raw_bytes == b['raw_payload_bytes'] and projected_bytes == b['projected_payload_bytes']
        result=dict(schema='ORIGINAL_HISTORY_BOUNDARIES_RESULT_V1', freeze=a.freeze, binding_sha256=a.binding_sha,
            decision='TEACHER_RESIDUAL_BOUNDARIES_CAPTURE_PASS', cases=len(completed), records=completed,
            boundaries=BOUNDARIES, history_tokens=b['history_tokens'], basis=pfile, basis_orthogonality_max=defect,
            raw_payload_bytes=raw_bytes, projected_payload_bytes=projected_bytes,
            source_base_attempts=attempts, source_base_forwards=forwards, exact_history_output_witnesses=witness_count,
            LM_head_calls=head_calls, source_generations=0, optimizer_updates=0, native_calls=0, reserved_queries=0,
            source_parameter_identities_versions_unchanged=True, quality_admission=False, speed_admission=False,
            scope='Actual zero-state full-prefill residual boundaries, fixed P256 auxiliary coordinates;'
                'not cached-reply bit parity, source-logit/state transport or completed chatbot conversion.',
            worker_OS_peak_snapshot=proc.memory_info().peak_wset, GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic()-start)
        write(a.out,result); event(stage='complete', result_sha256=sha(a.out))
    except BaseException as error:
        write(a.directory/'first_fault.json', dict(stage=stage,error=repr(error),completed_ids=[rec['id'] for rec in completed],
            source_base_attempts=attempts,source_base_forwards=forwards,exact_output_witnesses=witness_count,
            raw_payload_bytes=raw_bytes,projected_payload_bytes=projected_bytes,elapsed_seconds=time.monotonic()-start,
            worker_OS_peak_snapshot=None if proc is None else proc.memory_info().peak_wset,
            GPU_allocated_peak=None if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=None if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved()))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(); m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true'); m.add_argument('--launch',action='store_true'); m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path); p.add_argument('--binding-sha'); p.add_argument('--freeze')
    p.add_argument('--directory',type=Path); p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.bind: bind(args)
    elif args.launch: launch(args)
    else: worker(args)
