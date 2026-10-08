"""New per-module C-operand core/norm diagnostic, no whole-model replay."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    native = DOC / 'chatbot_hybrid_native_result_repair3_20261008.json'
    n = json.loads(native.read_bytes())
    trace = next(Path(v['path']) for v in n['native_outputs'] if Path(v['path']).name == 'trace.bin')
    source_config = ROOT / 'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008/config.json'
    files = [native, native.with_suffix('.terminal.json'), Path(n['model']['path']), trace,
             trace.parent / 'queries.json', source_config, Path(__file__), B / 'chatbot_hybrid_target.py',
             B / 'chatbot_falcon_usability.py', B / 'chatbot_falcon_usability_launch.py', Path(sys.executable),
             DOC / 'CHATBOT_HYBRID_COMMON_CORE_PROTOCOL_20261009.md']
    files += [SITE / 'transformers/models/falcon_h1' / name for name in
              ('modeling_falcon_h1.py', 'configuration_falcon_h1.py')]
    files += [SITE / 'torch/nn/functional.py']
    foreign = {'benchmarks/donor_adaptation/configs/_manifest.json':
               'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
               'benchmarks/donor_adaptation/density/build_document_holdout.py':
               'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
               'docs/research/RESEARCH_INDEX.md':
               '99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}
    assert all(sha(ROOT / p) == s for p, s in foreign.items())
    files += [ROOT / p for p in foreign]
    binding = dict(schema='HYBRID_COMMON_CORE_BINDING_V1', python=str(Path(sys.executable).resolve()),
                   worker_path=str(Path(__file__).resolve()), model=n['model']['path'], trace=str(trace.resolve()),
                   queries=str((trace.parent / 'queries.json').resolve()), source_config=str(source_config.resolve()),
                   limits=dict(seconds=600, OS_bytes=4 << 30, GPU_allocated_bytes=2 << 30,
                               GPU_reserved_bytes=3 << 30, output_bytes=64 << 20),
                   runtime_binding_scope='Packed model/C operands/source config and selected target/Torch/Transformers files bound; isolated versions, not full native DLL tree.',
                   inputs=[dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    write(a.out, binding)
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(files))), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'HYBRID_COMMON_CORE_BINDING_V1'
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    assert sys.version_info[:3] == (3, 12, 10)
    a.directory.mkdir(exist_ok=False)
    completed = []
    try:
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        import transformers
        from transformers.models.falcon_h1 import modeling_falcon_h1 as source_code
        from chatbot_hybrid_target import SWA, SWA_SITES, target_config, FalconH1Mixer
        from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1RMSNorm
        assert torch.__version__ == '2.6.0+cu124' and np.__version__ == '2.4.6'
        assert psutil.__version__ == '7.2.2' and transformers.__version__ == '5.13.1'
        assert torch.cuda.is_available()
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        proc = psutil.Process()
        proc.cpu_affinity(list(range(6)))
        def guard():
            lim = b['limits']
            assert time.monotonic() - start <= lim['seconds']
            assert proc.memory_info().peak_wset <= lim['OS_bytes']
            assert torch.cuda.max_memory_allocated() <= lim['GPU_allocated_bytes']
            assert torch.cuda.max_memory_reserved() <= lim['GPU_reserved_bytes']
            assert not proc.children(recursive=True)
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= lim['output_bytes']
        dtype = np.dtype([(n, '<f4', 512) for n in ('input', 'core_input', 'core_output', 'ff_input')] + [
            ('scores', '<f4', 72), ('ids', '<u4', 8), ('mass', '<f4', 8), ('ff_output', '<f4', 512), ('output', '<f4', 512)])
        assert dtype.itemsize == 12640
        trace = np.fromfile(b['trace'], dtype=dtype)
        assert len(trace) == 3132
        cases = json.loads(Path(b['queries']).read_bytes())
        assert len(cases) == 6 and sum(len(c['input_ids']) for c in cases) == 261
        model = np.memmap(b['model'], dtype=np.uint8, mode='r')
        magic, *header = struct.unpack_from('<8s16IQ', model)
        assert magic == b'SLH1PK01' and header[:16] == [1,512,12,65537,72,8,128,768,256,48,16,4,128,(1<<5)|(1<<11),212,0x01020304]
        assert header[-1] == len(model) == 425210736
        fields = {}
        cursor = 22128
        for i in range(212):
            r = struct.unpack_from('<64sII4IQQ', model, 80 + 104 * i)
            name = r[0].split(b'\0')[0].decode()
            assert name not in fields and r[7] == cursor
            fields[name] = (r[1], tuple(r[3:3+r[2]]), r[7], r[8])
            cursor += r[8]
        assert cursor == len(model)
        def cuda(name):
            typ, shape, off, size = fields[name]
            assert typ == 1 and int(np.prod(shape)) * 4 == size
            data = np.ndarray(shape, dtype='<f4', buffer=model, offset=off)
            assert np.isfinite(data).all()
            return torch.tensor(data.copy(), device='cuda')
        def metrics(actual, reference):
            assert actual.shape == reference.shape and np.isfinite(actual).all()
            error = np.square(actual.astype(np.float64) - reference).sum(-1)
            energy = np.square(reference.astype(np.float64)).sum(-1)
            assert np.all((energy > 0) | (error == 0))
            rms = np.sqrt(error / np.where(energy == 0, 1, energy))
            return dict(rows=len(rms), RMS_max=float(rms.max()), RMS_mean=float(rms.mean()),
                        material_rows=int(np.count_nonzero(rms > 1e-4))), rms
        config = target_config(json.loads(Path(b['source_config']).read_bytes()))
        assert config.rms_norm_eps == 1e-5 and config.mamba_n_groups == 1
        records = []
        frames = []
        with torch.no_grad():
            for layer in range(12):
                prefix = f'layers.{layer}.core.'
                if layer in SWA_SITES:
                    core = SWA().to('cuda').eval()
                    core.inv_freq.copy_(cuda('rope_inv_freq'))
                else:
                    core = FalconH1Mixer(config, layer).to('cuda').eval()
                    core.register_buffer('mup_vector', torch.ones(1, 1, 2096, device='cuda'), persistent=False)
                    assert not source_code.is_fast_path_available
                state = {name: cuda(prefix + name) for name in core.state_dict()}
                assert {prefix + name for name in state} == {name for name in fields if name.startswith(prefix)}
                core.load_state_dict(state, strict=True)
                del state
                norm1 = FalconH1RMSNorm(512, config.rms_norm_eps).to('cuda').eval()
                norm2 = FalconH1RMSNorm(512, config.rms_norm_eps).to('cuda').eval()
                norm1.weight.copy_(cuda(f'layers.{layer}.input_norm.weight'))
                norm2.weight.copy_(cuda(f'layers.{layer}.ff_norm.weight'))
                offset = 0
                site_records = []
                for case in cases:
                    count = len(case['input_ids'])
                    block = trace[layer::12][offset:offset+count]
                    x = torch.tensor(block['core_input'].copy(), device='cuda')[None]
                    result = core(x)
                    input_norm = norm1(torch.tensor(block['input'].copy(), device='cuda'))
                    # The saved C post-core residual is an F32 addition, independent
                    # of this new CUDA common-input core output.
                    residual = np.add(block['input'], block['core_output'], dtype=np.float32)
                    ff_norm = norm2(torch.tensor(residual, device='cuda'))
                    torch.cuda.synchronize()
                    actuals = {'core': result[0].cpu().numpy(), 'input_norm': input_norm.cpu().numpy(),
                               'ff_norm': ff_norm.cpu().numpy()}
                    references = {'core': block['core_output'], 'input_norm': block['core_input'], 'ff_norm': block['ff_input']}
                    row = dict(layer=layer, kind='SWA' if layer in SWA_SITES else 'SSM', case=case['id'],
                               global_input_start=offset, rows=count, outputs={}, metrics={})
                    row_rms = {}
                    for key, value in actuals.items():
                        path = a.directory / f'layer_{layer:02d}.{case["id"]}.{key}.f32'
                        with path.open('xb') as f:
                            f.write(value.astype('<f4', copy=False).tobytes())
                            f.flush()
                            os.fsync(f.fileno())
                        row['outputs'][key] = dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))
                        row['metrics'][key], row_rms[key] = metrics(value, references[key])
                    for i in range(count):
                        frames.append(dict(layer=layer, case=case['id'], position=i, global_input_row=offset+i,
                                           **{key+'_relative_RMS': float(value[i]) for key, value in row_rms.items()}))
                    write(a.directory / f'layer_{layer:02d}.{case["id"]}.complete.json', row)
                    completed.append(dict(layer=layer, case=case['id']))
                    site_records.append(row)
                    offset += count
                    del x, result, input_norm, ff_norm, actuals, references, residual, block
                    guard()
                assert offset == 261
                summary = dict(layer=layer, kind=site_records[0]['kind'], rows=261,
                               **{key: dict(RMS_max=max(r['metrics'][key]['RMS_max'] for r in site_records),
                                            material_rows=sum(r['metrics'][key]['material_rows'] for r in site_records))
                                  for key in ('core', 'input_norm', 'ff_norm')})
                records.append(summary)
                print(json.dumps(dict(**summary, seconds=time.monotonic()-start)), flush=True)
                del core, norm1, norm2
        assert len(completed) == 72 and len(frames) == 3132
        core_large = sum(r['core']['material_rows'] for r in records)
        norm_large = sum(r[k]['material_rows'] for r in records for k in ('input_norm', 'ff_norm'))
        decision = ('COMMON_INPUT_CORE_DIFFERENCE' if core_large else 'COMMON_INPUT_NORM_DIFFERENCE'
                    if norm_large else 'COMMON_INPUT_CORE_AND_NORM_CLOSE')
        write(a.directory / 'frame_metrics.json', frames)
        report = dict(schema='HYBRID_COMMON_CORE_RESULT_V1', freeze=a.freeze, binding_sha256=a.binding_sha,
                      process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()), decision=decision,
                      layers=records, completed_modules_cases=len(completed), core_rows=3132, norm_rows=6264,
                      material_core_rows=core_large, material_norm_rows=norm_large,
                      frame_metrics=dict(path=str((a.directory/'frame_metrics.json').resolve()),
                                         sha256=sha(a.directory/'frame_metrics.json')),
                      GPU_allocated_peak=torch.cuda.max_memory_allocated(), GPU_reserved_peak=torch.cuda.max_memory_reserved(),
                      worker_OS_peak_snapshot=proc.memory_info().peak_wset, elapsed_seconds=time.monotonic()-start,
                      runtime=dict(torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
                                   device=torch.cuda.get_device_name(), source_code=source_code.__file__, TF32=False,
                                   SSD_chunk=16, initial_state='zero per case/module; continuous sequence', dtype='F32'),
                      whole_target_forwards=0, source_generations=0, bank_evaluations=0, training_updates=0,
                      limits='Common C operands isolate core/norm arithmetic, not original GPU whole trajectory. Does not prove all full-logit causes. Original native parity FAIL remains.')
        write(a.out, report)
        guard()
        print(json.dumps(dict(decision=decision, material_core_rows=core_large, material_norm_rows=norm_large)), flush=True)
    except BaseException as error:
        write(a.directory / 'first_failure.json', dict(fault=repr(error), completed=completed,
                                                     elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('bind', 'worker'), default='worker')
    p.add_argument('--binding', type=Path)
    p.add_argument('--binding-sha')
    p.add_argument('--freeze')
    p.add_argument('--directory', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    bind(a) if a.mode == 'bind' else worker(a)
