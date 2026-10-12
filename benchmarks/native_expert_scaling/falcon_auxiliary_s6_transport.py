"""Exact real-arithmetic generator lift, not a compact pretrained chatbot.

F64 operator assay: actual Falcon-H1 coefficients, deterministic FIT embedding
inputs, independent source and lifted S6 equations. No model loader or training.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
SITE = ROOT / 'results/native_expert_scaling/chatbot_source_runtime/site'
SOURCE = ROOT / 'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
sys.path.insert(0, str(SITE))
BEGAN = time.monotonic()


def extent(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def prepare(a):
    import numpy as np
    import psutil
    adopted = DOC / 'original_falcon_recurrent_adopted_result_20261009.json'
    records = json.loads(adopted.read_bytes())['records']
    fit = [r for r in records if r['split'] == 'FIT']
    r = min(fit, key=lambda r: (r['history'], r['id']))
    assert r['history'] == 58 and len(r['input_ids']) == 58
    files = [Path(__file__), Path(sys.executable), SOURCE / 'config.json', SOURCE / 'model.safetensors',
        adopted, DOC / 'FALCON_AUXILIARY_S6_TRANSPORT_PROTOCOL_20261012.md',
        ROOT / 'benchmarks/native_expert_scaling/original_tensor_learner.py',
        ROOT / 'benchmarks/native_expert_scaling/original_falcon_learner.py',
        ROOT / 'benchmarks/phase60/engine.c',
        SITE / 'transformers/models/falcon_h1/modeling_falcon_h1.py',
        SITE / 'transformers/models/falcon_h1/configuration_falcon_h1.py',
        Path(np.__file__), Path(np._core._multiarray_umath.__file__), Path(psutil.__file__)]
    files += list((SITE / 'psutil').glob('*.pyd'))
    b = dict(schema='FALCON_AUXILIARY_S6_TRANSPORT_BINDING_V1',
        python=str(Path(sys.executable).resolve()), source=str(SOURCE.resolve()), sites=[0, 12, 23],
        input_case=r['id'], input_origin='First13 FIT token embeddings, normalized with each tested site gamma; '
            'these are injected operator inputs, NOT captured intermediate states at sites12/23.',
        input_ids=r['input_ids'][:13], dtype='F64', threads=1,
        gates=dict(component_relative_RMS=1e-10, scalar_relative=1e-10),
        limits=dict(seconds=300, OS_bytes=4 << 30, output_bytes=16 << 20),
        inputs=[extent(p) for p in dict.fromkeys(files)],
        runtime_scope='Principal Python/NumPy/psutil binaries and actual HF source text, not full DLL tree; '
            'CPU-only arithmetic assay. External publisher preserved; no clean timing admission.')
    write(a.out, b)
    print(json.dumps(dict(binding=extent(a.out), seconds=time.monotonic() - BEGAN)), flush=True)


def tensor_reader(path, np):
    """Read only named actual BF16/F32 coefficient ranges; no safetensors loader."""
    stream = Path(path).open('rb')
    n = int.from_bytes(stream.read(8), 'little')
    assert 0 < n < 8 << 20
    header = json.loads(stream.read(n))
    base = 8 + n

    def read(key, rows=None):
        item = header[key]
        shape = item['shape']
        assert item['dtype'] in ('BF16', 'F32')
        width = 2 if item['dtype'] == 'BF16' else 4
        lo, hi = item['data_offsets']
        assert hi - lo == math.prod(shape) * width
        if rows is None:
            stream.seek(base + lo)
            raw = stream.read(hi - lo)
        else:
            assert len(shape) == 2 and all(0 <= i < shape[0] for i in rows)
            chunks = []
            for i in rows:
                stream.seek(base + lo + i * shape[1] * width)
                chunks.append(stream.read(shape[1] * width))
            raw = b''.join(chunks)
            shape = [len(rows), shape[1]]
        assert len(raw) == math.prod(shape) * width
        if width == 2:
            z = (np.frombuffer(raw, '<u2').astype('<u4') << 16).view('<f4')
        else:
            z = np.frombuffer(raw, '<f4')
        result = z.reshape(shape).astype('<f8')
        assert np.isfinite(result).all(), key
        return result
    return read, stream


def silu(x, np):
    # Stable scalar identity, including large injected values.
    return x * np.exp(-np.logaddexp(0.0, -x))


def softplus(x, np):
    return np.logaddexp(0.0, x)


def conv(x, w, bias, np):
    out = np.broadcast_to(bias, x.shape).copy()
    for lag in range(4):
        if lag < len(x):
            out[lag:] += x[:len(x)-lag] * w[:, 3-lag]
    return out


def source_operator(inp, w, c, np):
    u, n, h, p = 3072, 256, 48, 64
    # Source linear -> per-section MuP multipliers -> conv/SiLU.
    projected = (inp * c['ssm_in_multiplier']) @ w['in'].T
    for lo, hi, mul in [(0, u, c['ssm_multipliers'][0]), (u, 2*u, c['ssm_multipliers'][1]),
                       (2*u, 2*u+n, c['ssm_multipliers'][2]),
                       (2*u+n, 2*u+2*n, c['ssm_multipliers'][3]),
                       (2*u+2*n, 2*u+2*n+h, c['ssm_multipliers'][4])]:
        projected[:, lo:hi] *= mul
    gate = projected[:, :u]
    features = silu(conv(projected[:, u:2*u+2*n], w['conv'], w['conv_b'], np), np)
    xx, bb, cc = features[:, :u], features[:, u:u+n], features[:, u+n:]
    delta = softplus(projected[:, -h:] + w['dt_b'], np)
    state = np.zeros((h, p, n))
    ys = []
    for t in range(len(inp)):
        state = np.exp(-np.exp(w['A'])[:, None, None] * delta[t, :, None, None]) * state
        state += delta[t, :, None, None] * xx[t].reshape(h, p, 1) * bb[t, None, None, :]
        ys.append((state * cc[t, None, None, :]).sum(-1).reshape(u)
                  + np.repeat(w['D'], p) * xx[t])
    yy = np.stack(ys)
    gated = yy * silu(gate, np)
    denominator = np.sqrt(np.mean(gated * gated, axis=1, keepdims=True) + c['rms_norm_eps'])
    output = (gated / denominator * w['norm']) @ w['out'].T
    return dict(x=xx, B=bb, C=cc, delta=delta, gate=gate, y=yy,
                denominator=denominator, output=output, projected=projected)


def compile_lift(w, c, np):
    """Actual coefficients into original S6 fields, plus required postgate scalar.

    xx = [source activated x/B/C, SiLU(dt_raw), SiLU(-dt_raw)].
    SiLU(a)-SiLU(-a)=a recovers source pre-softplus dt exactly over the reals.
    Auxiliary channels carry generator features; their output columns are zero.
    """
    d, u, n, h, p = 2048, 3072, 256, 48, 64
    f = u + 2*n + 2*h
    inp = np.zeros((2*f, d))
    sections = [(0, u, c['ssm_multipliers'][1]), (u, u+n, c['ssm_multipliers'][2]),
                (u+n, u+2*n, c['ssm_multipliers'][3])]
    for lo, hi, mul in sections:
        inp[lo:hi] = w['in'][u+lo:u+hi] * (c['ssm_in_multiplier'] * mul)
    dt_weights = w['in'][-h:] * (c['ssm_in_multiplier'] * c['ssm_multipliers'][4])
    inp[u+2*n:u+2*n+h] = dt_weights
    inp[u+2*n+h:f] = -dt_weights
    inp[f:f+u] = w['in'][:u] * (c['ssm_in_multiplier'] * c['ssm_multipliers'][0])
    cw = np.zeros((f, 4)); cb = np.zeros(f)
    cw[:u+2*n] = w['conv']; cb[:u+2*n] = w['conv_b']
    cw[u+2*n:, -1] = 1
    xp = np.zeros((h+2*n, f))
    for j in range(h):
        xp[j, u+2*n+j] = 1; xp[j, u+2*n+h+j] = -1
    for j in range(n):
        xp[h+j, u+j] = 1; xp[h+n+j, u+n+j] = 1
    dp = np.zeros((f, h)); db = np.zeros(f)
    for j in range(u):
        dp[j, j//p] = 1; db[j] = w['dt_b'][j//p]
    aa = np.zeros((f, n)); aa[:u] = np.repeat(w['A'], p)[:, None]
    skip = np.zeros(f); skip[:u] = np.repeat(w['D'], p)
    out = np.zeros((d, f)); out[:, :u] = w['out'] * w['norm'][None, :]
    return dict(in_proj=inp, conv_w=cw, conv_b=cb, x_proj=xp, dt_proj=dp,
                dt_b=db, A_log=aa, Dskip=skip, out_proj=out)


def lifted_operator(inp, w, c, np, remove_negative_dt=False):
    u, n, h = 3072, 256, 48
    f = w['Dskip'].size
    projected = inp @ w['in_proj'].T
    xx, gate = projected[:, :f], projected[:, f:]
    features = silu(conv(xx, w['conv_w'], w['conv_b'], np), np)
    if remove_negative_dt:
        features[:, u+2*n+h:] = 0
    dbl = features @ w['x_proj'].T
    dt = softplus(dbl[:, :h] @ w['dt_proj'].T + w['dt_b'], np)
    bb, cc = dbl[:, h:h+n], dbl[:, h+n:]
    state = np.zeros((f, n)); ys = []
    aa = -np.exp(w['A_log'])
    for t in range(len(inp)):
        state = np.exp(dt[t, :, None] * aa) * state
        state += (dt[t] * features[t])[:, None] * bb[t, None, :]
        ys.append((state * cc[t, None, :]).sum(-1) + w['Dskip'] * features[t])
    yy = np.stack(ys)
    gated = yy * silu(gate, np)
    # Norm is over source PRIMARY channels only, not all auxiliary features.
    denominator = np.sqrt(np.mean(gated[:, :u]**2, axis=1, keepdims=True) + c['rms_norm_eps'])
    unnormalized = gated @ w['out_proj'].T
    return dict(x=features[:, :u], B=bb, C=cc, delta=dt[:, :u:64], gate=gate[:, :u],
                y=yy[:, :u], denominator=denominator, output=unnormalized / denominator,
                output_without_postgate_norm=unnormalized)


def relative(a, b, np):
    return float(np.linalg.norm(a-b) / max(float(np.linalg.norm(b)), 1e-300))


def scalar_witness(source, w, np):
    """Python scalar recurrence with math.fsum, first head/channel, every input.

    Reads the independent source feature operands; does not use compiled fields.
    """
    state = [0.0] * 256; values = []
    for t in range(len(source['x'])):
        dt = float(source['delta'][t, 0]); xx = float(source['x'][t, 0])
        decay = math.exp(-math.exp(float(w['A'][0])) * dt)
        state = [decay*s + dt*xx*float(b) for s, b in zip(state, source['B'][t])]
        values.append(math.fsum(s*float(cc) for s, cc in zip(state, source['C'][t])) + float(w['D'][0])*xx)
    return relative(np.array(values), source['y'][:, 0], np)


def run(a):
    assert extent(a.binding)['sha256'] == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'FALCON_AUXILIARY_S6_TRANSPORT_BINDING_V1'
    assert Path(sys.executable).resolve() == Path(b['python']).resolve()
    assert not a.out.exists() and not a.directory.exists()
    terminal = a.out.with_suffix('.terminal.json')
    assert not terminal.exists()
    import numpy as np
    import psutil
    assert (np.__version__, psutil.__version__) == ('2.4.6', '7.2.2')
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    proc = psutil.Process(); proc.cpu_affinity([0])
    stage = 'before_hash'; stream = None; rows = []
    receipt = dict(schema='FALCON_AUXILIARY_S6_TRANSPORT_TERMINAL_V1', binding=extent(a.binding),
        freeze=a.freeze, command=[sys.executable, *sys.argv], pid=proc.pid, creation_time=proc.create_time(),
        original_engine_modified=False, GPU_or_model_or_optimizer_calls=0,
        process_isolation_admitted=False, timing_control_admitted=False)
    def guard():
        assert time.monotonic()-BEGAN <= b['limits']['seconds'], 'deadline'
        assert proc.memory_info().peak_wset <= b['limits']['OS_bytes'], 'OS cap'
        assert not proc.children(recursive=True), 'unexpected child'
    def check():
        for item in b['inputs']:
            guard(); assert extent(item['path']) == item, item['path']
    try:
        check(); a.directory.mkdir()
        c = json.loads((Path(b['source'])/'config.json').read_bytes())
        assert (c['hidden_size'], c['mamba_d_ssm'], c['mamba_d_state'], c['mamba_n_heads'],
                c['mamba_d_head'], c['mamba_n_groups']) == (2048,3072,256,48,64,1)
        assert c['hidden_act'] == 'silu' and c['mamba_rms_norm'] and not c['mamba_norm_before_gate']
        assert not c['mamba_proj_bias'] and not c['projectors_bias']
        assert c.get('time_step_limit', [0.0, None]) in ([0.0, None], None), 'finite delta clamp unsupported'
        get, stream = tensor_reader(Path(b['source'])/'model.safetensors', np)
        embed = get('model.embed_tokens.weight', b['input_ids']) * c['embedding_multiplier']
        for site in b['sites']:
            stage = f'site{site}'; guard()
            base = f'model.layers.{site}.'
            gamma = get(base+'input_layernorm.weight')
            inp = embed / np.sqrt(np.mean(embed**2, axis=1, keepdims=True)+c['rms_norm_eps']) * gamma
            keys = dict(in_='in_proj.weight', conv='conv1d.weight', conv_b='conv1d.bias',
                        dt_b='dt_bias', A='A_log', D='D', norm='norm.weight', out='out_proj.weight')
            w = {('in' if k == 'in_' else k): get(base+'mamba.'+name) for k, name in keys.items()}
            w['conv'] = w['conv'].reshape(3584,4)
            original = source_operator(inp, w, c, np)
            lifted = compile_lift(w, c, np)
            actual = lifted_operator(inp, lifted, c, np)
            components = {k: dict(relative_RMS=relative(actual[k], original[k], np),
                                 max_abs=float(np.max(np.abs(actual[k]-original[k]))))
                          for k in ('x','B','C','delta','gate','y','denominator','output')}
            witness = scalar_witness(original, w, np)
            ablation = lifted_operator(inp, lifted, c, np, remove_negative_dt=True)
            row = dict(site=site, positions=len(inp), components=components, scalar_recurrence_relative=witness,
                omitted_negative_dt_pair_output_relative=relative(ablation['output'], original['output'], np),
                omitted_postgate_norm_output_relative=relative(actual['output_without_postgate_norm'], original['output'], np),
                compiled_dense_coefficients=sum(z.size for z in lifted.values()),
                compiled_F64_coefficient_bytes=sum(z.nbytes for z in lifted.values()),
                auxiliary_out_columns_exact_zero=bool(np.all(lifted['out_proj'][:,3072:]==0)),
                auxiliary_gate_rows_exact_zero=bool(np.all(lifted['in_proj'][3680+3072:]==0)))
            row['numeric_gates'] = all(v['relative_RMS'] <= b['gates']['component_relative_RMS'] for v in components.values())
            row['numeric_gates'] &= witness <= b['gates']['scalar_relative']
            rows.append(row)
            np.savez(a.directory/f'site{site:02d}.npz', input=inp,
                **{'source_'+k:original[k] for k in ('x','B','C','delta','gate','y','denominator','output')},
                **{'lifted_'+k:actual[k] for k in ('x','B','C','delta','gate','y','denominator','output')},
                ablated_dt_output=ablation['output'], no_norm_output=actual['output_without_postgate_norm'])
            print(json.dumps(dict(site=site, numeric_gates=row['numeric_gates'],
                output_relative=components['output']['relative_RMS'], seconds=time.monotonic()-BEGAN)), flush=True)
            del w, lifted, original, actual, ablation
        stream.close(); stream=None
        stage='after_hash';check()
        outputs=[extent(p) for p in sorted(a.directory.iterdir())]
        assert sum(i['bytes'] for i in outputs) <= b['limits']['output_bytes'];guard()
        assert len(rows)==3 and all(r['numeric_gates'] for r in rows), rows
        result=dict(schema='FALCON_AUXILIARY_S6_TRANSPORT_RESULT_V1', binding=extent(a.binding), freeze=a.freeze,
            decision='EXACT_GENERATOR_LIFT_WITH_REQUIRED_POSTGATE_SCALAR_COMPONENT_QUALIFIED',
            source_input_D=2048, primary_channels=3072, auxiliary_channels=608, lifted_DN=3680,
            N=256, DT=48, input_case=b['input_case'], input_ids=b['input_ids'], input_origin=b['input_origin'],
            records=rows, outputs=outputs, source_active_states=3072*256, naive_lift_states=3680*256,
            original_wide_active_states=1024*96,
            dense_linear_MAC_per_operator=3*3680*2048+(48+2*256)*3680+3680*48,
            source_layer_count=24, compact_original_layer_count=6,
            extension='Primary-only postgate RMS denominator; gamma folded into out_proj columns.',
            engine_extension_implemented=False, compact_generator_conversion=False,
            source_attention_FFN_depth_composition_transferred=False, useful_chatbot_admission=False,
            speed_admission=False, generality_admission=False, inputs_before_after_exact=True,
            seconds=time.monotonic()-BEGAN, OS_peak=proc.memory_info().peak_wset,
            scope='Actual coefficients on39 injected F64 operator positions. Real-arithmetic lift; '
                'not BF16 source/cache parity, quantized original C parity, compact conversion, '
                'held-out chatbot behavior, training dose, or generic donor runtime adoption.')
        write(a.out,result)
        receipt.update(exit_code=0, error=None, inputs_before_after_exact=True, result=extent(a.out), outputs=outputs)
    except BaseException as error:
        receipt.update(exit_code=1,error=repr(error),stage=stage,completed_sites=rows)
        raise
    finally:
        if stream is not None:stream.close()
        receipt.update(seconds=time.monotonic()-BEGAN,OS_peak=proc.memory_info().peak_wset)
        write(terminal,receipt)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--run',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.prepare:prepare(a)
    else:
        assert all((a.binding,a.binding_sha,a.freeze,a.directory))
        run(a)
