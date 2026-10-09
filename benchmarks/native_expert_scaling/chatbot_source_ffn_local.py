"""Acquire true source FFN operands and test a finite diagonal calibration."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def read(path):
    return json.loads(Path(path).read_bytes())


def receipt(path):
    record = read(Path(path).with_suffix('.terminal.json'))
    assert record['exit_code'] == 0 and record['resource_gates'] and record['result_sha256'] == sha(path)
    return record


def bind(a):
    original_path = DOC / 'chatbot_source_ffn_preflight_binding_20261009.json'
    original = read(original_path)
    assert sha(original_path) == read(DOC / 'chatbot_source_ffn_preflight_result_20261009.json')['binding_sha256']
    for item in original['inputs']:
        p = Path(item['path'])
        if p.resolve() != (B / 'chatbot_falcon_usability_launch.py').resolve():
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
    source = Path(original['source'])
    manifest_path = ROOT / 'results/native_expert_scaling/chatbot_source_ffn_preflight_20261009/ffn.manifest.json'
    manifest = read(manifest_path)
    assert sha(manifest['file']['path']) == manifest['file']['sha256'] == '9aa6f319363294aac6ee611def4384e9c63354929c43c6219d8144f698a2a33b'
    largest = max((v for v in manifest['weight_metrics'] if v['projection'] == 'down'), key=lambda v: (v['relative_weight_L2'], -v['site']))
    assert largest['site'] == 23
    records = read(original['corpus'])['records']
    selected = []
    for split in ('FIT', 'DEV'):
        rows = sorted((r for r in records if r['split'] == split), key=lambda r: (len(r['student_input_ids']), r['id']))
        selected += [rows[0], rows[-1]]
    assert [r['id'] for r in selected] == ['broad_fit_explore_instruct_rewriting_008', 'broad_fit_smol_magpie_ultra_022',
        'broad_dev_explore_instruct_rewriting_035', 'broad_dev_smol_magpie_ultra_036']
    assert sum(len(r['output_ids']) for r in selected) == 542
    files = [Path(v['path']) for v in original['inputs']]
    files += [Path(__file__), original_path, manifest_path, Path(manifest['file']['path']),
        DOC / 'CHATBOT_SOURCE_FFN_LOCAL_PROTOCOL_20261009.md', B / 'chatbot_source_ffn_cast_control.py',
        DOC / 'chatbot_source_ffn_cast_result_20261009.json', DOC / 'chatbot_source_ffn_cast_result_20261009.terminal.json']
    cast = read(DOC / 'chatbot_source_ffn_cast_result_20261009.json')
    assert cast['decision'] == 'SOURCE_FFN_F32_CAST_PASS'
    receipt(DOC / 'chatbot_source_ffn_cast_result_20261009.json')
    files += [Path(r['logits']['path']) for r in selected]
    files += [SITE / 'safetensors/__init__.py', SITE / 'safetensors/_safetensors_rust.pyd']
    phase = a.mode.removeprefix('bind-')
    extra = {}
    if phase == 'calibrate':
        capture_path = DOC / 'chatbot_source_ffn_local_capture_result_20261009.json'
        captured = read(capture_path)
        assert captured['decision'] == 'SOURCE_FFN_OPERANDS_PASS' and captured['cases'] == 4 and captured['source_forwards'] == 542
        receipt(capture_path)
        capture_binding = DOC / 'chatbot_source_ffn_local_capture_binding_20261009.json'
        assert sha(capture_binding) == captured['binding_sha256']
        extra['capture'] = str(capture_path)
        files += [capture_path, capture_path.with_suffix('.terminal.json'), capture_binding]
        for row in captured['records']:
            for site in row['sites']:
                for name in ('x', 'y'):
                    item = site[name]
                    assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
                    files.append(Path(item['path']))
    files = list(dict.fromkeys(p.resolve() for p in files))
    write(a.out, dict(schema='SOURCE_FFN_LOCAL_BINDING_V1', phase=phase, python=str(Path(sys.executable).resolve()),
        worker_path=str(Path(__file__).resolve()), source=str(source), corpus=original['corpus'], selected_ids=[r['id'] for r in selected],
        sites=[0, 23], manifest=str(manifest_path), payload=manifest['file']['path'], **extra,
        criteria=dict(identity_relative_L2=1e-5, DEV_relative_error_ratio=.90, individual_DEV_ratio=1.05),
        limits=dict(seconds=600 if phase == 'capture' else 300, reserve_seconds=60 if phase == 'capture' else 30,
            OS_bytes=8 << 30, GPU_allocated_bytes=10 << 30, GPU_reserved_bytes=11 << 30, output_bytes=512 << 20),
        runtime_binding_scope='Pinned donor/four original forced packets/actual FFN sector/source helpers/local mapping/Python/selected runtime/foreign hashes;not full DLL-tree.',
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files]))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(files), phase=phase)), flush=True)


def worker(a):
    start = time.monotonic()
    assert sha(a.binding) == a.binding_sha
    b = read(a.binding)
    assert b['schema'] == 'SOURCE_FFN_LOCAL_BINDING_V1'
    a.directory.mkdir(exist_ok=False)
    phase = 'startup'
    completed = []
    forwards = 0
    torch = None
    try:
        assert sys.version_info[:3] == (3, 12, 10) and Path(sys.executable).resolve() == Path(b['python']).resolve()
        assert all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        assert not any(importlib.util.find_spec(v) is not None for v in ('kernels', 'mamba_ssm', 'causal_conv1d'))
        import numpy as np
        import psutil
        import torch
        from torch.nn import functional as F
        import transformers
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
            assert time.monotonic() - start <= lim['seconds'] - lim['reserve_seconds'], 'time reserve'
            assert proc.memory_info().peak_wset <= lim['OS_bytes'], 'OS cap'
            assert torch.cuda.max_memory_allocated() <= lim['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= lim['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'worker child'
            assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file()) <= lim['output_bytes'], 'output cap'

        def event(stage, **fields):
            print(json.dumps(dict(stage=stage, seconds=time.monotonic()-start, **fields)), flush=True)
            guard()

        def raw_file(name, array, shape, dtype):
            path = a.directory / name
            with path.open('xb') as stream:
                stream.write(array.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path), shape=shape, dtype=dtype)

        def metric(actual, expected):
            v = actual.double()
            t = expected.double()
            e = (v-t).square().sum().item()
            tn = t.square().sum().item()
            vn = v.square().sum().item()
            dot = (v*t).sum().item()
            amplitude = dot/max(vn, 1e-30)
            return dict(relative_L2=(e/max(tn, 1e-30))**.5, cosine=dot/max((vn*tn)**.5, 1e-30),
                oracle_amplitude=amplitude, oracle_relative_L2=((v*amplitude-t).square().sum().item()/max(tn, 1e-30))**.5)

        if b['phase'] == 'capture':
            from transformers import FalconH1ForCausalLM
            from transformers.models.falcon_h1 import modeling_falcon_h1 as code
            from chatbot_falcon_ssd_tiles import install
            write(a.directory/'source_method.json', install(code))
            phase = 'source_load'
            model = FalconH1ForCausalLM.from_pretrained(b['source'], local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
            identities = {n: id(p) for n, p in model.named_parameters()}
            assert len(identities) == 411 and sum(p.numel() for p in model.parameters()) == 1554863488
            assert model.config.mamba_chunk_size == 128 and not code.is_fast_path_available
            active = None
            samples = {}
            hooks = []

            def observer(site, module, args, output):
                assert active is not None and args[0].dtype == output.dtype == torch.bfloat16
                assert args[0].shape[-1] == output.shape[-1] == 2048
                pair = samples[site]
                for name, tensor in (('x', args[0]), ('y', output)):
                    pair[name].append(tensor[0, -1].detach().cpu().contiguous().view(torch.uint16).numpy().copy())

            for site in b['sites']:
                hooks.append(model.model.layers[site].feed_forward.register_forward_hook(
                    lambda module, args, output, site=site: observer(site, module, args, output)))
            records = {r['id']: r for r in read(b['corpus'])['records']}
            captured = []
            for identifier in b['selected_ids']:
                phase = identifier
                rec = records[identifier]
                active = identifier
                samples = {site: dict(x=[], y=[]) for site in b['sites']}
                source_bits = np.fromfile(rec['logits']['path'], dtype='<u2').reshape(rec['logits']['shape'])
                prompt, outputs = rec['input_ids'], rec['output_ids']
                past = None
                with torch.inference_mode():
                    for index in range(len(outputs)):
                        ids = prompt if index == 0 else [outputs[index-1]]
                        result = model(input_ids=torch.tensor([ids], device='cuda'), attention_mask=torch.ones((1, len(prompt)+index), device='cuda', dtype=torch.long),
                            past_key_values=past, use_cache=True, logits_to_keep=1)
                        past = result.past_key_values
                        logits = result.logits[0, -1]
                        forwards += 1
                        assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all().item()
                        bits = logits.view(torch.uint16).cpu().numpy()
                        if not np.array_equal(bits, source_bits[index]):
                            raw_file('first_differing_logits.bf16', bits.astype('<u2', copy=False), [65537], 'BF16')
                            raise AssertionError(('source logits mismatch', identifier, index, int((bits != source_bits[index]).sum())))
                        assert all(len(v['x']) == len(v['y']) == index+1 for v in samples.values())
                        del result, logits
                        guard()
                row = dict(id=identifier, split=rec['split'], domain=rec['domain'], labels=len(outputs),
                    input_ids=prompt, output_ids=outputs, absolute_label_positions=[len(prompt)-1+i for i in range(len(outputs))],
                    source_logits_all_bits_equal=True, sites=[])
                for site in b['sites']:
                    packet = dict(site=site)
                    for name in ('x', 'y'):
                        array = np.stack(samples[site][name]).astype('<u2', copy=False)
                        packet[name] = raw_file(f'{identifier}.site{site:02d}.{name}.bf16', array, [len(outputs), 2048], 'BF16')
                    row['sites'].append(packet)
                write(a.directory/(identifier+'.json'), row)
                captured.append(row)
                completed.append(identifier)
                event('captured_case', id=identifier, labels=len(outputs), logits_all_bits_equal=True)
                del past, samples, source_bits
            active = None
            for hook in hooks:
                hook.remove()
            assert forwards == 542 and len(captured) == 4 and {n:id(p) for n,p in model.named_parameters()} == identities
            assert sum(v[name]['bytes'] for r in captured for v in r['sites'] for name in ('x','y')) == 8880128
            details = dict(records=captured, source_forwards=542, source_generations=0, hidden_payload_bytes=8880128,
                verified_logit_coordinates=542*65537, unchanged_parameter_objects=411)
            decision = 'SOURCE_FFN_OPERANDS_PASS'
        else:
            assert b['phase'] == 'calibrate'
            from safetensors import safe_open
            from chatbot_hybrid_target import aq63, row_scale
            capture = read(b['capture'])
            manifest = read(b['manifest'])
            config = read(Path(b['source'])/'config.json')
            a_gate, b_down = config['mlp_multipliers']
            assert (a_gate, b_down) == (.4419417382415922, .13020833333333331)
            all_sites = []

            def response(x, weights, mode, codes=None, scales=None):
                if mode == 'BF16':
                    h = x.to(torch.bfloat16)
                    u = F.linear(h, weights['up'].to(torch.bfloat16))
                    g = F.linear(h, weights['gate'].to(torch.bfloat16))*a_gate
                    z = u*F.silu(g)
                    return (F.linear(z, weights['down'].to(torch.bfloat16))*b_down).float()
                def linear(h, name):
                    operand, act = aq63(h) if mode in ('AQ63', 'TERNARY_AQ63') else (h, 1.)
                    if mode in ('TERNARY', 'TERNARY_AQ63'):
                        return F.linear(operand, codes[name].float())*scales[name]*act
                    return F.linear(operand, weights[name])*act
                g = linear(x, 'gate')*a_gate
                u = linear(x, 'up')
                return linear(u*F.silu(g), 'down')*b_down

            def observed(identifier, name, raw, reference):
                assert torch.isfinite(raw).all().item()
                output = raw.to(torch.bfloat16)
                bits = output.contiguous().view(torch.uint16).cpu().numpy().astype('<u2', copy=False)
                item = metric(output.float(), reference)
                item['output'] = raw_file(identifier+'.'+name+'.bf16', bits, list(output.shape), 'BF16')
                return item

            with safe_open(str(Path(b['source'])/'model.safetensors'), framework='pt', device='cpu') as tensors:
                for site in b['sites']:
                    phase = f'site{site:02d}'
                    weights = {}
                    for name in ('gate', 'up', 'down'):
                        w = tensors.get_tensor(f'model.layers.{site}.feed_forward.{name}_proj.weight')
                        assert w.dtype == torch.bfloat16
                        weights[name] = w.to(device='cuda', dtype=torch.float32)
                    data = []
                    for rec in capture['records']:
                        packet = next(v for v in rec['sites'] if v['site'] == site)
                        row = dict(id=rec['id'], split=rec['split'])
                        for name in ('x','y'):
                            p = packet[name]
                            bits = np.fromfile(p['path'], dtype='<u2').reshape(p['shape'])
                            row[name] = torch.from_numpy(bits.copy()).view(torch.bfloat16).to(device='cuda', dtype=torch.float32)
                        data.append(row)
                    base_codes, base_scales = {}, {}
                    for name in weights:
                        scale = row_scale(weights[name])
                        code = torch.round(weights[name]/scale[:,None]).clamp(-1,1).to(torch.int8)
                        field = next(v for v in manifest['fields'] if v['site']==site and v['projection']==name and v['field']=='pair_codes')
                        pair = np.fromfile(b['payload'], dtype=np.int8, count=field['bytes'], offset=field['offset']).reshape(field['shape'])
                        actual_pair = ((code[:,0::2]+1)*3+(code[:,1::2]+1)).T.contiguous().cpu().numpy()
                        assert np.array_equal(actual_pair, pair), ('base sector codes', site, name)
                        sf = next(v for v in manifest['fields'] if v['site']==site and v['projection']==name and v['field']=='row_scale')
                        original_scale = np.fromfile(b['payload'], dtype='<f4', count=scale.numel(), offset=sf['offset'])
                        assert np.array_equal(scale.cpu().numpy().view('<u4'), original_scale.view('<u4')), ('base sector scales', site, name)
                        base_codes[name], base_scales[name] = code, scale
                    decomposition = {}
                    for rec in data:
                        identifier = f"site{site:02d}.{rec['id']}"
                        decomposition[rec['id']] = {}
                        for mode in ('BF16','F32','AQ63','TERNARY','TERNARY_AQ63'):
                            value = response(rec['x'], weights, mode, base_codes, base_scales)
                            decomposition[rec['id']][mode] = observed(identifier, mode, value, rec['y'])
                            if mode == 'F32':
                                decomposition[rec['id']][mode]['raw_output_F32'] = raw_file(identifier+'.F32.raw.f32',value.cpu().numpy().astype('<f4',copy=False),list(value.shape),'F32')
                    fit = [r for r in data if r['split']=='FIT']
                    dev = [r for r in data if r['split']=='DEV']
                    input_rms = torch.stack([r['x'].double().square().mean(0) for r in fit]).mean(0).sqrt().clamp_min(1e-12)
                    column_rms = ((weights['gate'].double().square().mean(0)+weights['up'].double().square().mean(0))/2).sqrt().clamp_min(1e-12)
                    hidden_moments = []
                    for rec in fit:
                        z = F.linear(rec['x'], weights['up'])*F.silu(F.linear(rec['x'],weights['gate'])*a_gate)
                        hidden_moments.append(z.double().square().mean(0))
                    hidden_rms = torch.stack(hidden_moments).mean(0).sqrt().clamp_min(1e-12)
                    down_rms = weights['down'].double().square().mean(0).sqrt().clamp_min(1e-12)
                    logS = input_rms.log()-column_rms.log()
                    logR = down_rms.log()-hidden_rms.log()
                    logS -= logS.mean()
                    logR -= logR.mean()
                    trials = []
                    best = None
                    best_loss = float('inf')
                    for alpha in (0., .5, 1.):
                        for beta in (0., .5, 1.):
                            guard()
                            S = (logS*alpha).exp().clamp(1/16,16).float()
                            R = (logR*beta).exp().clamp(1/16,16).float()
                            transformed = dict(gate=weights['gate']/S, up=weights['up']*R[:,None]/S, down=weights['down']/R)
                            scales = {name: row_scale(w) for name,w in transformed.items()}
                            codes = {name:torch.round(w/scales[name][:,None]).clamp(-1,1).to(torch.int8) for name,w in transformed.items()}
                            if alpha == beta == 0:
                                assert all(torch.equal(codes[n],base_codes[n]) and torch.equal(scales[n],base_scales[n]) for n in weights)
                            trial = dict(alpha=alpha,beta=beta,FIT=[],identity_FIT=[])
                            for rec in fit:
                                identifier = f"site{site:02d}.{rec['id']}.a{alpha}.b{beta}"
                                identity = response(rec['x']*S, transformed, 'F32')
                                original = response(rec['x'], weights, 'F32')
                                identity_metric = metric(identity, original)
                                assert identity_metric['relative_L2'] <= b['criteria']['identity_relative_L2'], ('real identity numerical bound',site,alpha,beta)
                                identity_metric['output_F32'] = raw_file(identifier+'.identity.f32', identity.cpu().numpy().astype('<f4',copy=False), list(identity.shape),'F32')
                                trial['identity_FIT'].append(dict(id=rec['id'],**identity_metric))
                                value = response(rec['x']*S, transformed, 'TERNARY_AQ63', codes, scales)
                                trial['FIT'].append(dict(id=rec['id'],**observed(identifier,'quantized',value,rec['y'])))
                            loss = sum(r['relative_L2'] for r in trial['FIT'])/2
                            trial['FIT_case_mean_relative_L2'] = loss
                            trials.append(trial)
                            if loss < best_loss:
                                best_loss = loss
                                best = dict(alpha=alpha,beta=beta,S=S,R=R,weights=transformed,codes=codes,scales=scales)
                            event('FIT_trial', site=site, alpha=alpha, beta=beta, error=loss)
                    selected = dict(alpha=best['alpha'], beta=best['beta'], FIT_case_mean_relative_L2=best_loss, DEV=[])
                    for rec in dev:
                        identifier = f"site{site:02d}.{rec['id']}.selected"
                        identity = response(rec['x']*best['S'], best['weights'], 'F32')
                        identity_metric = metric(identity,response(rec['x'],weights,'F32'))
                        assert identity_metric['relative_L2'] <= b['criteria']['identity_relative_L2']
                        identity_metric['output_F32'] = raw_file(identifier+'.identity.f32',identity.cpu().numpy().astype('<f4',copy=False),list(identity.shape),'F32')
                        value = response(rec['x']*best['S'], best['weights'],'TERNARY_AQ63',best['codes'],best['scales'])
                        item = observed(identifier,'quantized',value,rec['y'])
                        baseline = decomposition[rec['id']]['TERNARY_AQ63']['relative_L2']
                        selected['DEV'].append(dict(id=rec['id'], baseline_relative_L2=baseline, ratio=item['relative_L2']/max(baseline,1e-30),identity=identity_metric,**item))
                    baseline_dev = sum(r['baseline_relative_L2'] for r in selected['DEV'])/2
                    selected_dev = sum(r['relative_L2'] for r in selected['DEV'])/2
                    selected['DEV_case_mean_ratio'] = selected_dev/max(baseline_dev,1e-30)
                    selected['gates'] = dict(DEV_improvement=selected['DEV_case_mean_ratio']<=b['criteria']['DEV_relative_error_ratio'],
                        every_DEV_retention=all(r['ratio']<=b['criteria']['individual_DEV_ratio'] for r in selected['DEV']))
                    selected['input_scale'] = raw_file(f'site{site:02d}.selected.S.f32',best['S'].cpu().numpy().astype('<f4',copy=False),[2048],'F32')
                    selected['hidden_scale'] = raw_file(f'site{site:02d}.selected.R.f32',best['R'].cpu().numpy().astype('<f4',copy=False),[4608],'F32')
                    selected['projections'] = []
                    for name in weights:
                        code,scale = best['codes'][name],best['scales'][name]
                        pair = ((code[:,0::2]+1)*3+(code[:,1::2]+1)).T.contiguous()
                        assert torch.equal(pair.T//3-1,code[:,0::2]) and torch.equal(pair.T%3-1,code[:,1::2])
                        selected['projections'].append(dict(name=name,
                            codes=raw_file(f'site{site:02d}.selected.{name}.pairs.u8',pair.cpu().numpy().astype('u1',copy=False),list(pair.shape),'U8'),
                            scale=raw_file(f'site{site:02d}.selected.{name}.scale.f32',scale.cpu().numpy().astype('<f4',copy=False),list(scale.shape),'F32')))
                    record = dict(site=site, decomposition=decomposition, trials=trials, selected=selected,
                        baseline_sector_all_codes_scales_exact=True, weight_elements=sum(v.numel() for v in weights.values()))
                    write(a.directory/f'site{site:02d}.json',record)
                    all_sites.append(record)
                    completed.append(site)
                    event('site_complete',site=site,alpha=best['alpha'],beta=best['beta'],DEV_ratio=selected['DEV_case_mean_ratio'],gates=selected['gates'])
                    del weights,data,base_codes,base_scales,best,transformed,codes,scales
            details = dict(records=all_sites, source_forwards=0, source_generations=0, evaluated_sites=2,
                FIT_trials=18, capture_binding_sha256=capture['binding_sha256'], local_only=True)
            decision = 'SOURCE_FFN_LOCAL_BALANCE_PASS' if all(all(r['selected']['gates'].values()) for r in all_sites) else 'SOURCE_FFN_LOCAL_BALANCE_FAIL'
        phase = 'result'
        guard()
        write(a.out,dict(schema='SOURCE_FFN_LOCAL_RESULT_V1',phase=b['phase'],decision=decision,freeze=a.freeze,binding_sha256=a.binding_sha,
            process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),cases=4,**details,
            optimizer_updates=0,native_runs=0,reserved_queries=0,GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-start,quality_admission=False,native_admission=False,
            scope='True original forced-trajectory FFN operands or two-site local response/calibration;no whole converted quality/own-history/rate/large-n/family admission.'))
        event('complete',decision=decision)
    except BaseException as error:
        failure = dict(fault=repr(error),phase=phase,completed=completed,source_forwards=forwards,elapsed_seconds=time.monotonic()-start)
        if torch is not None:
            failure.update(GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved())
        write(a.directory/'first_failure.json',failure)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('bind-capture','bind-calibrate','worker'),default='worker')
    parser.add_argument('--binding',type=Path)
    parser.add_argument('--binding-sha')
    parser.add_argument('--freeze')
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    bind(args) if args.mode.startswith('bind-') else worker(args)
