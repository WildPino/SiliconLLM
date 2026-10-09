"""Acquire missing source FFN operands on the retained 48-case broad cohort."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_source_ffn_local import read, receipt
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    oldpath = DOC / 'chatbot_source_ffn_local_capture_binding_20261009.json'
    oldresult = DOC / 'chatbot_source_ffn_local_capture_result_20261009.json'
    old, retained = read(oldpath), read(oldresult)
    receipt(oldresult)
    assert sha(oldpath) == retained['binding_sha256'] and retained['decision'] == 'SOURCE_FFN_OPERANDS_PASS'
    superseded = []
    files = []
    launcher = B / 'chatbot_falcon_usability_launch.py'
    for item in old['inputs']:
        p = Path(item['path'])
        if p.resolve() == launcher.resolve():
            superseded.append(item)
        else:
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
        files.append(p)
    records = read(old['corpus'])['records']
    assert len(records) == 48 and sum(len(r['output_ids']) for r in records) == 8808
    assert len({r['domain'] for r in records}) == 12
    assert all(sum(r['split'] == split for r in records) == 24 for split in ('FIT', 'DEV'))
    reused = {r['id']: r for r in retained['records']}
    assert len(reused) == 4 and sum(r['labels'] for r in reused.values()) == 542
    for row in records:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) and row['split'] in ('FIT', 'DEV')
        item = row['logits']
        assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
        files.append(Path(item['path']))
        if row['id'] in reused:
            prior = reused[row['id']]
            assert prior['input_ids'] == row['input_ids'] and prior['output_ids'] == row['output_ids']
            assert prior['source_logits_all_bits_equal'] and prior['labels'] == len(row['output_ids'])
            for site in prior['sites']:
                for name in ('x', 'y'):
                    item = site[name]
                    assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
                    files.append(Path(item['path']))
    files += [oldpath, oldresult, oldresult.with_suffix('.terminal.json'), Path(__file__),
        DOC / 'CHATBOT_SOURCE_FFN_BROAD_CAPTURE_PROTOCOL_20261009.md']
    files = list(dict.fromkeys(p.resolve() for p in files))
    b = {key: value for key, value in old.items() if key not in ('inputs', 'criteria', 'selected_ids')}
    b.update(schema='SOURCE_FFN_BROAD_BINDING_V1', phase='capture', worker_path=str(Path(__file__).resolve()),
        retained_capture=str(oldresult), selected_ids=[r['id'] for r in records], sites=[0, 23], cases=48,
        labels=8808, reused_cases=4, reused_labels=542, new_cases=44, new_labels=8266,
        limits=dict(seconds=1800, reserve_seconds=60, OS_bytes=8 << 30,
            GPU_allocated_bytes=10 << 30, GPU_reserved_bytes=11 << 30, output_bytes=256 << 20),
        superseded_launcher_extents=superseded,
        runtime_binding_scope='Pinned source/original 48 forced packets/source helper/runtime/foreign hashes; original four hidden cases reused. Launcher adds the new broad schema; no source operator changes or full DLL-tree binding.',
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    write(a.out, b)
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(files),
        new_cases=44, new_labels=8266, new_hidden_bytes=135430144)), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = read(a.binding)
    assert b['schema'] == 'SOURCE_FFN_BROAD_BINDING_V1' and b['phase'] == 'capture'
    a.directory.mkdir(exist_ok=False)
    phase = 'startup'
    captured, completed = [], []
    forwards = 0
    torch = None
    try:
        assert sys.version_info[:3] == (3, 12, 10) and Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from transformers import FalconH1ForCausalLM
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        from chatbot_falcon_ssd_tiles import install
        assert (torch.__version__, transformers.__version__, np.__version__) == ('2.6.0+cu124', '5.13.1', '2.4.6')
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        def guard():
            lim = b['limits']
            assert time.monotonic()-start <= lim['seconds']-lim['reserve_seconds'], 'time reserve'
            assert proc.memory_info().peak_wset <= lim['OS_bytes'], 'OS cap'
            assert torch.cuda.max_memory_allocated() <= lim['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= lim['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'worker child'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= lim['output_bytes'], 'output cap'

        def atomic_json(path, value):
            assert not path.exists()
            temp = path.with_suffix('.json.tmp')
            write(temp, value)
            temp.replace(path)

        def raw_file(name, array):
            path = a.directory / name
            assert not path.exists()
            temp = path.with_suffix('.bf16.tmp')
            with temp.open('xb') as stream:
                stream.write(array.tobytes()); stream.flush(); os.fsync(stream.fileno())
            temp.replace(path)
            return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=sha(path), shape=list(array.shape), dtype='BF16')

        def event(stage, **fields):
            print(json.dumps(dict(stage=stage, elapsed_seconds=time.monotonic()-start, **fields)), flush=True)
            guard()

        retained = read(b['retained_capture'])
        reused = {r['id']: r for r in retained['records']}
        records = {r['id']: r for r in read(b['corpus'])['records']}
        write(a.directory/'source_method.json', install(code))
        phase = 'source_load'
        model = FalconH1ForCausalLM.from_pretrained(b['source'], local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        identities = {n: (id(p), p._version) for n, p in model.named_parameters()}
        assert len(identities) == 411 and sum(p.numel() for p in model.parameters()) == 1554863488
        assert model.config.mamba_chunk_size == 128 and not code.is_fast_path_available
        active, samples = None, {}

        def observer(site, module, args, output):
            assert active is not None and args[0].dtype == output.dtype == torch.bfloat16
            assert args[0].shape[-1] == output.shape[-1] == 2048
            for name, tensor in (('x', args[0]), ('y', output)):
                samples[site][name].append(tensor[0, -1].detach().cpu().contiguous().view(torch.uint16).numpy().copy())

        hooks = [model.model.layers[site].feed_forward.register_forward_hook(
            lambda module, args, output, site=site: observer(site, module, args, output)) for site in b['sites']]
        for identifier in b['selected_ids']:
            guard()
            phase = identifier
            rec = records[identifier]
            if identifier in reused:
                row = dict(reused[identifier], packet_origin='REUSED_COMPLETE_FOUR_CASE_CAPTURE',
                    retained_capture_sha256=sha(b['retained_capture']))
                atomic_json(a.directory/(identifier+'.json'), row)
                captured.append(row); completed.append(identifier)
                event('reused_case', id=identifier, completed=len(completed), labels=row['labels'])
                continue
            active = identifier
            samples = {site: dict(x=[], y=[]) for site in b['sites']}
            source_bits = np.fromfile(rec['logits']['path'], dtype='<u2').reshape(rec['logits']['shape'])
            prompt, outputs, past = rec['input_ids'], rec['output_ids'], None
            with torch.inference_mode():
                for index in range(len(outputs)):
                    ids = prompt if index == 0 else [outputs[index-1]]
                    result = model(input_ids=torch.tensor([ids], device='cuda'),
                        attention_mask=torch.ones((1, len(prompt)+index), device='cuda', dtype=torch.long),
                        past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = result.past_key_values
                    logits = result.logits[0, -1]
                    forwards += 1
                    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all().item()
                    bits = logits.view(torch.uint16).cpu().numpy()
                    if not np.array_equal(bits, source_bits[index]):
                        raw_file('first_differing_logits.bf16', bits.astype('<u2', copy=False))
                        raise AssertionError(('source logits mismatch', identifier, index, int((bits != source_bits[index]).sum())))
                    assert all(len(v['x']) == len(v['y']) == index+1 for v in samples.values())
                    del result, logits
                    guard()
            row = dict(id=identifier, split=rec['split'], domain=rec['domain'], labels=len(outputs),
                input_ids=prompt, output_ids=outputs,
                absolute_label_positions=[len(prompt)-1+i for i in range(len(outputs))],
                source_logits_all_bits_equal=True, packet_origin='NEW_MISSING_OPERANDS', sites=[])
            for site in b['sites']:
                packet = dict(site=site)
                for name in ('x', 'y'):
                    packet[name] = raw_file(f'{identifier}.site{site:02d}.{name}.bf16',
                        np.stack(samples[site][name]).astype('<u2', copy=False))
                row['sites'].append(packet)
            atomic_json(a.directory/(identifier+'.json'), row)
            captured.append(row); completed.append(identifier)
            event('captured_case', id=identifier, domain=rec['domain'], split=rec['split'],
                labels=len(outputs), completed=len(completed), new_source_forwards=forwards, logits_all_bits_equal=True)
            del past, samples, source_bits
            active = None
        for hook in hooks:
            hook.remove()
        assert forwards == 8266 and len(captured) == 48
        assert {n: (id(p), p._version) for n, p in model.named_parameters()} == identities
        total = sum(v[name]['bytes'] for r in captured for v in r['sites'] for name in ('x', 'y'))
        assert total == 144310272
        guard()
        result = dict(schema='SOURCE_FFN_BROAD_RESULT_V1', phase='capture', decision='SOURCE_FFN_BROAD_OPERANDS_PASS',
            freeze=a.freeze, binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()), cases=48, records=captured,
            source_forwards=forwards, source_generations=0, reused_cases=4, reused_labels=542,
            new_cases=44, new_labels=8266, hidden_payload_bytes=total, new_hidden_payload_bytes=135430144,
            verified_new_logit_coordinates=forwards*65537, verified_total_logit_coordinates=8808*65537,
            unchanged_parameter_objects_and_versions=411, optimizer_updates=0, native_runs=0, reserved_queries=0,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-start, quality_admission=False, native_admission=False,
            scope='Missing true source FFN operands atsites0/23 onretained48 trajectories;not new reply/quality/rate/conditional capacity evidence.')
        atomic_json(a.directory/'capture.json', result)
        write(a.out, result)
        event('complete', decision=result['decision'])
    except BaseException as error:
        failure = dict(fault=repr(error), phase=phase, completed=completed, source_forwards=forwards,
            records=captured, elapsed_seconds=time.monotonic()-start)
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
