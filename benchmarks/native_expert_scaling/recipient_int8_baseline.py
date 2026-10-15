"""Bounded complete original-recipient cost screen; never a donor-transfer claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path.insert(0, str(SITE))
import psutil

HERE = Path(__file__).resolve().parent
DOC = ROOT / 'docs/research/PIPELINE_REASSESSMENT_20261015'
SOURCE = Path('C:/Users/giosa/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775')
SOURCE_SHA = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
PROFILE = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925/meth374_switch_physical_workers_contract_result.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def env():
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
             HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
             OMP_WAIT_POLICY='ACTIVE', KMP_BLOCKTIME='infinite', KMP_AFFINITY='none',
             OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1',
             OPENBLAS_NUM_THREADS='6', MKL_NUM_THREADS='6', OMP_NUM_THREADS='6')
    return e


def prepare(directory, adopt=None):
    directory.mkdir(parents=True, exist_ok=False)
    os.environ.update({k: v for k, v in env().items() if k in
                      ('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                       'TOKENIZERS_PARALLELISM', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')})
    from transformers import AutoTokenizer
    import numpy as np
    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True)
    prompts = [
        'Explain why metal feels colder than wood at the same room temperature. Give a careful practical explanation.',
        'Descrivi come organizzare una piccola biblioteca di quartiere: libri, prestiti e volontari. Fornisci una guida concreta.',
        'Write a Python function that merges two sorted lists without changing the original lists. Explain its complexity.',
    ]
    cases = []
    for i, prompt in enumerate(prompts):
        ids = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
        # Recent tokenizer APIs may wrap IDs in a BatchEncoding.
        if isinstance(ids, dict) or hasattr(ids, 'keys'):
            ids = ids['input_ids']
        assert isinstance(ids, list) and ids and isinstance(ids[0], int)
        path = directory / f'prompt_{i}.ids'
        np.asarray(ids, dtype='<i4').tofile(path)
        cases.append({'name': f'prompt_{i}', 'prompt': prompt, 'ids': str(path),
                      'tokens': len(ids), 'sha256': sha(path), 'new_tokens': 64})
    base = np.fromfile(cases[0]['ids'], dtype='<i4')
    parity = directory / 'parity.ids'
    assert len(base) >= 32
    base[:32].tofile(parity)
    # Declared context-only cost fixtures, not useful chat or held-out quality.
    seed = tokenizer.encode('A library keeps books for readers. Readers return borrowed books on time. ', add_special_tokens=False)
    for n in (512, 2048):
        p = directory / f'context_{n}.ids'
        np.asarray((seed * (n // len(seed) + 1))[:n], dtype='<i4').tofile(p)
        cases.append({'name': f'context_{n}', 'ids': str(p), 'tokens': n,
                      'sha256': sha(p), 'new_tokens': 64, 'cost_only': True})
    profile = json.loads(PROFILE.read_text())
    cc = Path(profile['compile']['argv'][0])
    assert sha(cc) == profile['compile']['compiler_sha256']
    dll = cc.parent / 'libomp.dll'
    assert sha(dll) == profile['compile']['runtime_sha256']
    files = {Path(__file__).resolve(), HERE / 'recipient_int8_entry.c',
             HERE / 'recipient_int8_audit.py', HERE / 'meth388_switch_thread_binding.h',
             ROOT / 'benchmarks/phase60/engine.c', PROFILE,
             DOC / 'RECIPIENT_INT8_BASELINE_PROTOCOL_20261015.md',
             ROOT / 'benchmarks/donor_adaptation/engine/qwen_export.py',
             ROOT / 'benchmarks/donor_adaptation/engine/donor_engine.c',
             ROOT / 'benchmarks/donor_adaptation/engine/vexpf8.h',
             ROOT / 'benchmarks/donor_adaptation/ternary/t2_rules.py',
             ROOT / 'benchmarks/donor_adaptation/ternary/t1_ternarize.py',
             ROOT / 'benchmarks/donor_adaptation/ternary/t3_rotation.py',
             ROOT / 'benchmarks/donor_adaptation/engine/synth_export.py',
             HERE / 'meth124_centered_factor_cpu.c', cc, dll, Path(sys.executable),
             Path(sys.executable).with_name('python312.dll')}
    for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'model.safetensors'):
        files.add(SOURCE / name)
    for c in cases:
        files.add(Path(c['ids']))
    files.add(parity)
    # Bind Python implementation and native CPU dependencies; no GPU binary claim.
    for package in ('numpy', 'psutil', 'torch', 'transformers', 'safetensors',
                    'tokenizers', 'huggingface_hub'):
        p = SITE / package
        files.update(x for x in p.rglob('*') if x.is_file() and x.suffix in ('.py', '.pyd'))
    torch_lib = SITE / 'torch/lib'
    for name in ('torch_cpu.dll', 'torch_python.dll', 'c10.dll', 'torch.dll', 'libiomp5md.dll'):
        if (torch_lib / name).exists():
            files.add(torch_lib / name)
    for p in (SITE / 'numpy.libs', ROOT / '.venv/Lib/site-packages/numpy.libs'):
        if p.exists():
            files.update(x for x in p.rglob('*.dll'))
    records = []
    for p in sorted(files):
        assert p.is_file(), str(p)
        records.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    source = next(x for x in records if x['path'] == str(SOURCE / 'model.safetensors'))
    assert source['sha256'] == SOURCE_SHA
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    adoption = None
    if adopt is not None:
        old = json.loads((adopt / 'binding.json').read_text())
        fault = json.loads((adopt / 'audit.json.failure.json').read_text())
        assert fault['error_type'] == 'NotImplementedError'
        assert fault['error'] == 'Cannot copy out of meta tensor; no data!'
        allowed = {str(HERE / 'recipient_int8_baseline.py'), str(HERE / 'recipient_int8_audit.py')}
        differences = []
        for x in old['inputs']:
            p = Path(x['path'])
            if p.stat().st_size != x['bytes'] or sha(p) != x['sha256']:
                assert str(p) in allowed, str(p)
                differences.append({'path': str(p), 'old_sha256': x['sha256'], 'new_sha256': sha(p)})
        assert {x['path'] for x in differences} == allowed
        old_cases = {x['name']: x for x in old['cases']}
        assert all(x['sha256'] == old_cases[x['name']]['sha256'] for x in cases)
        assert sha(parity) == sha(Path(old['parity_ids']))
        accepted = []
        names = ['recipient.i8.bin', 'recipient.i8.bin.json', 'recipient.i8.exe']
        exported = json.loads((adopt / 'recipient.i8.bin.json').read_text())
        assert sha(adopt / 'recipient.i8.bin') == exported['sha256']
        for stage in ('export', 'compile'):
            receipt = json.loads((adopt / f'{stage}.terminal.json').read_text())
            assert receipt['exit_code'] == 0
            for channel in ('stdout', 'stderr'):
                assert sha(adopt / f'{stage}.{channel}') == receipt[channel + '_sha256']
            accepted.append(receipt)
            names.extend(f'{stage}.{suffix}' for suffix in ('stdout', 'stderr', 'terminal.json', 'owned.json'))
        inherited = []
        for name in names:
            src, dst = adopt / name, directory / name
            os.link(src, dst)  # Immutable retained artifact adoption; no numerical replay.
            inherited.append({'old_path': str(src), 'path': str(dst),
                              'bytes': dst.stat().st_size, 'sha256': sha(dst)})
        evidence = [adopt / 'binding.json', adopt / 'failure.json',
                    adopt / 'audit.json.failure.json', adopt / 'audit.terminal.json',
                    adopt / 'audit.stderr', adopt / 'audit.owned.json',
                    adopt / 'parity.logits', adopt / 'probe.terminal.json',
                    adopt / 'probe.stdout', adopt / 'probe.stderr', adopt / 'probe.owned.json']
        for p in evidence:
            records.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
        for x in inherited:
            records.append({k: x[k] for k in ('path', 'bytes', 'sha256')})
        adoption = {'from': str(adopt), 'fault_retained': str(adopt / 'audit.json.failure.json'),
                    'explicit_code_changes': differences, 'accepted_commands': accepted,
                    'inherited_artifacts': inherited,
                    'old_probe_sha256': sha(adopt / 'parity.logits'),
                    'scope': 'Reuse sealed source conversion and native executable after CPU reference meta-buffer repair; new probe must exactly replay the retained old probe. No model data/gates changed.'}
    binding = {'schema': 'RECIPIENT_INT8_BASELINE_V1', 'git_head': head,
               'source_sha256': SOURCE_SHA, 'source_directory': str(SOURCE),
               'compiler': str(cc), 'openmp_dll': str(dll), 'python': sys.executable,
               'cases': cases, 'parity_ids': str(parity), 'inputs': records,
               'limits': {'job_seconds': 900, 'total_seconds': 2400,
                          'host_bytes': 8 << 30, 'output_bytes': 2 << 30},
               'cost_only': True, 'quality_and_transfer_admitted': False, 'adoption': adoption}
    write(directory / 'binding.json', binding)
    print(json.dumps({'prepared': str(directory), 'inputs': len(records),
                      'input_bytes': sum(x['bytes'] for x in records),
                      'binding_sha256': sha(directory / 'binding.json')}), flush=True)


def verify(binding):
    for item in binding['inputs']:
        p = Path(item['path'])
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)


def run(directory):
    binding = json.loads((directory / 'binding.json').read_text())
    assert not (directory / 'result.json').exists() and not (directory / 'started.json').exists()
    verify(binding)
    write(directory / 'started.json', {'time': time.time(), 'binding_sha256': sha(directory / 'binding.json')})
    started = time.monotonic()
    commands = []
    child_env = env()
    child_env['PATH'] = str(Path(binding['compiler']).parent) + os.pathsep + child_env['PATH']

    def job(name, argv):
        assert time.monotonic() - started < binding['limits']['total_seconds']
        out = directory / f'{name}.stdout'
        err = directory / f'{name}.stderr'
        t = time.monotonic()
        with out.open('xb') as stdout, err.open('xb') as stderr:
            child = subprocess.Popen(list(map(str, argv)), cwd=ROOT, env=child_env,
                                     stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            proc = psutil.Process(child.pid)
            created = proc.create_time()
            write(directory / f'{name}.owned.json', {'pid': child.pid, 'create_time': created, 'argv': list(map(str, argv))})
            peak = 0
            try:
                while child.poll() is None:
                    try:
                        peak = max(peak, proc.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    assert peak <= binding['limits']['host_bytes'], f'{name}:memory'
                    assert time.monotonic() - t <= binding['limits']['job_seconds'], f'{name}:time'
                    assert time.monotonic() - started <= binding['limits']['total_seconds'], 'family:time'
                    time.sleep(0.1)
            except BaseException:
                if child.poll() is None and proc.create_time() == created:
                    child.kill()
                    child.wait()
                raise
        receipt = {'name': name, 'argv': list(map(str, argv)), 'exit_code': child.returncode,
                   'seconds': time.monotonic() - t, 'peak_rss_observed': peak,
                   'stdout_sha256': sha(out), 'stderr_sha256': sha(err)}
        write(directory / f'{name}.terminal.json', receipt)
        commands.append(receipt)
        print(json.dumps({'stage': name, 'exit_code': child.returncode,
                          'seconds': receipt['seconds']}), flush=True)
        assert child.returncode == 0, f'{name}:exit{child.returncode}'
        return out

    artifact = directory / 'recipient.i8.bin'
    exe = directory / 'recipient.i8.exe'
    py = [binding['python'], '-I', '-S', '-B', '-X', 'utf8']
    # Exporter's imports require the explicit isolated site view.
    boot = ('import runpy,sys;sys.path.insert(0,' + repr(str(SITE)) + ');'
            'p=sys.argv[1];sys.path.insert(0,__import__("os").path.dirname(p));'
            'sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")')
    try:
        adoption = binding.get('adoption')
        if adoption:
            commands.extend(adoption['accepted_commands'])
            print(json.dumps({'adopted': ['export', 'compile'], 'from': adoption['from']}), flush=True)
        else:
            job('export', py + ['-c', boot, ROOT / 'benchmarks/donor_adaptation/engine/qwen_export.py',
                '--model', SOURCE, '--revision', SOURCE.name, '--quant', 'int8', '--rule', 'R8',
                '--load-dtype', 'bfloat16', '--fold', 'none', '--head-ternary', '--threads', '6', '--out', artifact])
        artifact_sha = sha(artifact)
        if not adoption:
            job('compile', [binding['compiler'], '-O3', '-std=c11', '-march=x86-64-v3',
                '-ffp-contract=on', '-fopenmp', '-Wl,--no-insert-timestamp',
                '-DSILICON_QWEN_RECIPIENT_I8', ROOT / 'benchmarks/phase60/engine.c',
                '-o', exe, '-lpsapi', '-lm'])
        logits = directory / 'parity.logits'
        job('probe', [exe, artifact, binding['parity_ids'], '0', '6', 'probe', logits])
        if adoption:
            assert sha(logits) == adoption['old_probe_sha256'], 'native probe replay differs'
        job('audit', py + [HERE / 'recipient_int8_audit.py', '--source', SOURCE / 'model.safetensors',
            '--artifact', artifact, '--artifact-sha', artifact_sha, '--ids', binding['parity_ids'],
            '--logits', logits, '--out', directory / 'audit.json'])
        audited = json.loads((directory / 'audit.json').read_text())
        assert audited['status'] == 'PASS', audited
        results = []
        for repetition in range(3):
            for c in binding['cases']:
                name = f'generate_r{repetition}_{c["name"]}'
                output = job(name, [exe, artifact, c['ids'], str(c['new_tokens']), '6',
                                    'generate', directory / f'{name}.ids'])
                r = json.loads(output.read_text())
                results.append({'repetition': repetition, 'case': c['name'], **r})
        verify(binding)
        assert sha(artifact) == artifact_sha
        output_bytes = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        assert output_bytes <= binding['limits']['output_bytes']
        by_case = {}
        for c in binding['cases']:
            rows = [r for r in results if r['case'] == c['name']]
            assert all(r['native_decode_forwards'] > 0 for r in rows), 'no timed forward'
            rates = [r['native_decode_forwards'] / r['decode_s'] for r in rows]
            by_case[c['name']] = {'rates': rates, 'median': statistics.median(rates),
                                  'worst': min(rates), 'spread': max(rates) / min(rates)}
        budget = all(v['worst'] >= (1000 / 18) and v['spread'] <= 1.10 for v in by_case.values())
        result = {'status': 'BASELINE_COST_ENVELOPE_PASS' if budget else 'BASELINE_COST_ENVELOPE_FAIL',
                  'source_sha256': SOURCE_SHA, 'artifact_sha256': artifact_sha,
                  'artifact_bytes': artifact.stat().st_size, 'executable_sha256': sha(exe),
                  'binding_sha256': sha(directory / 'binding.json'), 'commands': commands,
                  'adoption': adoption,
                  'cases': results, 'cost_summary': by_case, 'seconds': time.monotonic() - started,
                  'output_bytes': output_bytes, 'quality_admitted': False,
                  'donor_transfer_admitted': False, 'goal_complete': False,
                  'scope': 'Complete original-recipient decoded wire and native cost; no memory fusion, fresh chatbot quality, physical DRAM or useful50 admission.'}
        write(directory / 'result.json', result)
        print(json.dumps({'status': result['status'], 'cost_summary': by_case,
                          'seconds': result['seconds']}), flush=True)
    except BaseException as e:
        write(directory / 'failure.json', {'error': repr(e), 'commands': commands,
                                         'seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--adopt', type=Path)
    a = parser.parse_args()
    if a.mode == 'prepare':
        prepare(a.directory.resolve(), a.adopt.resolve() if a.adopt else None)
    else:
        assert a.adopt is None, '--adopt belongs to preparation'
        run(a.directory.resolve())
