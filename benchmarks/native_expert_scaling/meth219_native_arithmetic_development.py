#!/usr/bin/env python3
"""Consumed-source native-precision screen of the physical METH-211 core."""
import argparse
import gc
import json
import math
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM

import meth214_stored_core_fresh_prediction as Q
import meth127_full_c_parity as N
import meth182_check_group64_ffn as C

L, P = Q.L, Q.P
BANK = P.ROOT / 'benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth126_shared_a_factor_bank.bin'
PRIOR = P.DOC / 'meth214_stored_core_fresh_prediction_result.json'
PRIOR_SHA = '593ada3d37e63d243b6e1ddd133bb8ed0c34a38512e1aadc066236dbc0dc5a90'
ARMS = ('bf16_donor', 'bf16_e1280', 'fp32_native_e1280',
        'stored_fp32_w8a32', 'stored_fp32_w8a8')
MANIFEST_SHA = 'bcd8295be1b2d62dceaf103f985182bad11ac50dd739a63cfb92d11400b5af10'
ANCHORS_SHA = '462eb44c18c83a2df3b27780b7c82e49ecdd2cce8d581a1d1843acef20010409'


def reconstruct(codes, scales):
    rows, width = codes.shape
    assert codes.dtype == torch.int8 and scales.dtype == torch.float16
    assert width % 64 == 0 and scales.shape == (rows, width // 64)
    return (codes.reshape(rows, width // 64, 64).float() *
            scales.float().unsqueeze(-1)).reshape(rows, width)


def quantized_input(x):
    shape = x.shape
    assert shape[-1] % 64 == 0 and x.dtype == torch.float32
    groups = x.reshape(*shape[:-1], shape[-1] // 64, 64)
    maximum = groups.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
    codes = (groups / scale).round().clamp(-127, 127)
    return (codes * scale).reshape(shape)


class StoredFFN(nn.Module):
    def __init__(self, gate, up, down):
        super().__init__()
        for name, value in (('gate', gate), ('up', up), ('down', down)):
            self.register_buffer(name, value)
        self.activation_quantization = False

    def forward(self, x):
        if self.activation_quantization:
            x = quantized_input(x.bfloat16().float())
        hidden = F.silu(F.linear(x, self.gate)) * F.linear(x, self.up)
        if self.activation_quantization:
            hidden = quantized_input(hidden)
        return F.linear(hidden, self.down)


class SwitchExperts(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.enabled = True

    def forward(self, x):
        return self.inner(x) if self.enabled else self.inner.base(x)


def attach_bank(model, device):
    assert P.digest(BANK) == N.BANK_SHA
    size = N.read_bank(BANK, model, device)
    for layer in model.model.layers:
        layer.mlp = SwitchExperts(layer.mlp)
    return [layer.mlp for layer in model.model.layers], size


def load_stored(device):
    config = AutoConfig.from_pretrained(P.M42.MODEL, revision=P.M42.REV, local_files_only=True)
    torch.manual_seed(219219)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32,
        attn_implementation='sdpa').to(device).eval()
    model.config.use_cache = False
    params = dict(model.named_parameters())
    assert len(params) == 290
    used, loaded, ffn = set(), {}, {}
    with safe_open(str(L.CORE), framework='pt', device='cpu') as archive, torch.no_grad():
        assert archive.metadata()['format'] == L.X.FORMAT
        for name, param in params.items():
            organ = P.M24.classify(name, tuple(param.shape))
            if organ == 'ffn':
                keys = (name + '.q8', name + '.scale')
                value = reconstruct(*(archive.get_tensor(k).to(device) for k in keys))
                ffn[name] = value
            elif organ == 'tied_head':
                keys = (name + '.bf16',)
                value = archive.get_tensor(keys[0]).to(device).float()
            else:
                keys = (name,)
                value = archive.get_tensor(name).to(device).float()
            assert value.shape == param.shape and value.dtype == torch.float32
            param.copy_(value)
            assert torch.equal(param, value)
            used.update(keys)
            loaded[organ] = loaded.get(organ, 0) + 1
        codes = archive.get_tensor(L.X.HEAD + '.q').to(device)
        scales = archive.get_tensor(L.X.HEAD + '.scale').to(device)
        assert codes.dtype == torch.int8 and codes.shape == (151936, 896)
        assert scales.dtype == torch.float16 and scales.shape == (151936,)
        proposal = codes.float() * scales.float()[:, None]
        used.update((L.X.HEAD + '.q', L.X.HEAD + '.scale'))
        assert used == set(archive.keys()) and len(used) == 364
    assert loaded == {'tied_head': 1, 'attention': 96, 'ffn': 72, 'control': 121}
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    del params, value, codes, scales
    for li, layer in enumerate(model.model.layers):
        layer.mlp = StoredFFN(*(ffn.pop(f'model.layers.{li}.mlp.{organ}_proj.weight')
                              for organ in ('gate', 'up', 'down')))
    assert not ffn
    wrappers, bank_size = attach_bank(model, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, wrappers, proposal, {
        'parameters_loaded': 290, 'tensors_consumed': 364, 'organs': loaded,
        'every_parameter_copy_equal': True, 'tied_head_pointer_equal': True,
        'source_core_weights_loaded': False, 'factor_bank_sha256': N.BANK_SHA,
        'factor_bank_bytes': bank_size, 'ffn_weight_round_to_bf16': False,
        'core_compute': 'FP32; SDPA; TF32 disabled',
        'experts_compute': 'METH-127 reference: BF16 input/factors/intermediates/gates/residual; FP32 router'}


def component(device, binary):
    """Bind every FFN byte, then check FP32 emulator against existing C executable."""
    assert not binary.exists()
    binary.parent.mkdir(parents=True, exist_ok=True)
    segments = []
    with safe_open(str(L.CORE), framework='pt', device='cpu') as archive, binary.open('xb') as out:
        out.write(struct.pack('<8s4I', b'M182FFN1', 24, 896, 4864, 64))
        for li in range(24):
            for organ in ('gate', 'up', 'down'):
                name = f'model.layers.{li}.mlp.{organ}_proj.weight'
                for suffix in ('.q8', '.scale'):
                    tensor = archive.get_tensor(name + suffix)
                    raw = tensor.numpy().tobytes() if suffix == '.q8' else tensor.view(torch.uint16).numpy().tobytes()
                    segments.append({'name': name + suffix, 'offset': out.tell(),
                                     'bytes': len(raw), 'sha256': P.M17.sha(raw)})
                    out.write(raw)
    with binary.open('rb') as audit:
        assert audit.read(24) == struct.pack('<8s4I', b'M182FFN1', 24, 896, 4864, 64)
        for row in segments:
            assert audit.tell() == row['offset'] and P.M17.sha(audit.read(row['bytes'])) == row['sha256']
        assert audit.read(1) == b''
    assert P.digest(C.VECTORS) == C.VECTORS_SHA
    executable = P.ROOT / 'benchmarks/native_expert_scaling/meth182_group64_ffn_cpu.exe'
    native_path = binary.with_suffix('.check.bin')
    assert not native_path.exists()
    run = subprocess.run([str(executable), str(binary), str(C.VECTORS), str(native_path)],
                         check=True, capture_output=True, text=True, timeout=300)
    timing = json.loads(run.stdout)
    assert timing['threads'] == 6 and timing['tokens'] == 256 and timing['layers'] == 24
    native = native_path.read_bytes()
    assert len(native) == 20 + 16 * 24 * 896 * 4
    assert struct.unpack_from('<8s3I', native) == (b'M182OUT1', 16, 24, 896)
    observed = np.frombuffer(native, dtype='<f4', offset=20).reshape(16, 24, 896)
    assert np.isfinite(observed).all()
    raw = C.VECTORS.read_bytes()
    states = np.frombuffer(raw, dtype='<u2', offset=24).reshape(256, 24, 896)
    errors = []
    with safe_open(str(L.CORE), framework='pt', device='cpu') as archive, torch.inference_mode():
        for li in range(24):
            matrices = []
            for organ in ('gate', 'up', 'down'):
                name = f'model.layers.{li}.mlp.{organ}_proj.weight'
                matrices.append(reconstruct(archive.get_tensor(name + '.q8').to(device),
                                             archive.get_tensor(name + '.scale').to(device)))
            mlp = StoredFFN(*matrices)
            mlp.activation_quantization = True
            x = torch.from_numpy(states[:16, li].copy()).view(torch.bfloat16).to(device).float()
            expected = mlp(x).cpu().numpy().astype(np.float64)
            norms = np.linalg.norm(expected, axis=1)
            assert (norms > 0).all()
            errors.extend((np.linalg.norm(observed[:, li].astype(np.float64)-expected, axis=1)/norms).tolist())
    result = {'binary_sha256': P.digest(binary), 'bytes': binary.stat().st_size,
        'segments': segments, 'executable_sha256': P.digest(executable),
        'c_source_sha256': P.digest(executable.with_suffix('.c')),
        'vectors_sha256': C.VECTORS_SHA, 'output_sha256': P.digest(native_path),
        'timing': timing, 'comparison_rows': len(errors), 'median_relative_l2': float(np.median(errors)),
        'maximum_relative_l2': max(errors),
        'numeric_pass': float(np.median(errors)) <= 1e-4 and max(errors) <= 5e-4,
        'scope': 'Stored M211 FFN bytes on consumed M125 states; approximate FP32 emulator parity; component timing only'}
    return result


def summarize(arms):
    result = {}
    for category in ('pooled', *P.CATEGORIES):
        values = {}
        for arm, cell in arms.items():
            docs = [r for r in cell['document_rows'] if category == 'pooled' or r['category'] == category]
            prompts = [r for r in cell['prompt_rows'] if category == 'pooled' or r['category'] == category]
            assert len(docs) == len(prompts) == (24 if category == 'pooled' else 8)
            values[arm] = {'bpb': sum(r['nats'] for r in docs)/(math.log(2)*sum(r['bytes'] for r in docs)),
                'top1': sum(r['matching'] for r in prompts)/sum(r['positions'] for r in prompts)}
        result[category] = values
    return result


def gates(summary, candidate):
    return {
        'pooled_bpb_vs_all_controls': all(summary['pooled'][candidate]['bpb']-summary['pooled'][c]['bpb'] <= .01 for c in ARMS[:3]),
        'category_bpb_vs_all_controls': all(summary[k][candidate]['bpb']-summary[k][c]['bpb'] <= .02 for k in P.CATEGORIES for c in ARMS[:3]),
        'pooled_top1_vs_both_e1280': all(summary['pooled'][c]['top1']-summary['pooled'][candidate]['top1'] <= .01 for c in ARMS[1:3]),
        'category_top1_vs_both_e1280': all(summary[k][c]['top1']-summary[k][candidate]['top1'] <= .02 for k in P.CATEGORIES for c in ARMS[1:3])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--ffn-binary', required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start, stage, arms, checks = time.monotonic(), 'bindings', {}, {}
    def partial():
        args.out.with_suffix('.partial.json').write_text(json.dumps({'stage': stage, 'arms': arms,
            'component': checks, 'elapsed_seconds': time.monotonic()-start}, indent=2)+'\n', encoding='utf-8')
    try:
        assert P.digest(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        items, parent, _ = Q.bind(P.DOC/'meth213_stored_core_fresh_manifest.json', MANIFEST_SHA,
            P.DOC/'meth213_answerability.json', ANCHORS_SHA)
        device = Q.setup()
        P.MAX_SECONDS = P.M17.MAX_SECONDS = 30*60
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        stage = 'native_ffn_emulator_parity'
        checks = component(device, args.ffn_binary)
        partial()
        print(json.dumps({'stage': stage, 'numeric_pass': checks['numeric_pass'],
                         'maximum_relative_l2': checks['maximum_relative_l2']}), flush=True)
        assert checks['numeric_pass'], 'Native component emulator parity stop'
        gc.collect()
        torch.cuda.empty_cache()
        stage = 'bf16_controls'
        model = AutoModelForCausalLM.from_pretrained(P.M42.MODEL, revision=P.M42.REV,
            dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(device).eval()
        model.config.use_cache = False
        L.D.H.load_centered(model, device, parent['path'])
        wrappers = [layer.mlp for layer in model.model.layers]
        donor_top = {}
        L.D.E.P = P
        with torch.inference_mode():
            for item in items:
                P.M44.set_experts(wrappers, False)
                ids = torch.as_tensor(item['prompt_ids'], device=device)[None]
                donor_top[item['source_id']] = model(ids, use_cache=False).logits.argmax(-1)[0].cpu()
        # The exact donor NLL/match rows are reusable only after every donor choice
        # is regenerated and BF16 E1280 reconciles each previously frozen metric.
        arms[ARMS[0]] = old['arms'][ARMS[0]]
        arms[ARMS[1]] = L.D.E.evaluate(model, wrappers, items, donor_top, ARMS[1], device, start)
        for rows, metric in (('document_rows', 'nats'), ('prompt_rows', 'matching')):
            for row, previous in zip(arms[ARMS[1]][rows], old['arms'][ARMS[1]][rows]):
                assert row['source_id'] == previous['source_id'] and row[metric] == previous[metric]
        partial()
        del model, wrappers
        gc.collect()
        torch.cuda.empty_cache()
        stage = ARMS[2]
        model = AutoModelForCausalLM.from_pretrained(P.M42.MODEL, revision=P.M42.REV,
            dtype=torch.float32, attn_implementation='sdpa', local_files_only=True).to(device).eval()
        model.config.use_cache = False
        wrappers, _ = attach_bank(model, device)
        arms[ARMS[2]] = L.D.E.evaluate(model, wrappers, items, donor_top, ARMS[2], device, start)
        partial()
        print(json.dumps({'arm': stage, 'runtime': P.budget(start, device)}), flush=True)
        del model, wrappers
        gc.collect()
        torch.cuda.empty_cache()
        stage = 'stored_load'
        model, wrappers, proposal, load_record = load_stored(device)
        shortlist = {}
        for arm, enabled in ((ARMS[3], False), (ARMS[4], True)):
            stage = arm
            for wrapper in wrappers:
                wrapper.inner.base.activation_quantization = enabled
            arms[arm] = L.D.E.evaluate(model, wrappers, items, donor_top, arm, device, start)
            partial()
            L.D.H.KS = (64,)
            shortlist[arm] = []
            with torch.inference_mode():
                for item in items:
                    ids = torch.as_tensor(item['prompt_ids'], device=device)[None]
                    hidden = model.model(ids, use_cache=False).last_hidden_state[0]
                    shortlist[arm].append({'source_id': item['source_id'],
                        **L.D.H.score_item(hidden, model.lm_head.weight, proposal)})
                    P.budget(start, device)
            print(json.dumps({'arm': arm, 'runtime': P.budget(start, device)}), flush=True)
        summary = summarize(arms)
        gate_rows = {}
        for arm in ARMS[3:]:
            gate_rows[arm] = {**gates(summary, arm),
                'prompt_k64_inclusion': all(r['misses']['64'] == 0 for r in shortlist[arm]),
                'prompt_exact_rerank': all(r['rerank_mismatches']['64'] == 0 for r in shortlist[arm])}
        decision = ('w8a8_development_pass_freeze_fresh_protocol' if all(gate_rows[ARMS[4]].values()) else
                    'w8a32_only_pass_requires_float_input_native_cost' if all(gate_rows[ARMS[3]].values()) else
                    'native_arithmetic_development_fail_stop_unchanged_precision_path')
        result = {'experiment': 'METH-219-native-arithmetic-development', 'script_sha256': P.digest(Path(__file__)),
            'core_sha256': L.CORE_SHA, 'export_sha256': L.EXPORT_SHA, 'factor_bank_sha256': N.BANK_SHA,
            'manifest_sha256': MANIFEST_SHA, 'anchors_sha256': ANCHORS_SHA, 'prior_result_sha256': PRIOR_SHA,
            'bf16_e1280_reconciliation_exact': True, 'donor_rows_reused_by_binding': True,
            'component': checks, 'load_record': load_record, 'arms': arms, 'summary': summary,
            'shortlist': shortlist, 'gates': gate_rows, 'decision': decision,
            'runtime': {**P.budget(start, device), 'gpu': torch.cuda.get_device_name(device)},
            'scope': 'Consumed-source development, approximate native arithmetic. No fresh quality, generation, task, full C parity/rate, or large-n quality claim'}
        args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'decision': decision, 'summary': summary, 'gates': gate_rows,
                         'runtime': result['runtime']}), flush=True)
    except BaseException as error:
        partial()
        args.out.with_suffix('.failure.json').write_text(json.dumps({'stage': stage, 'error': repr(error),
            'elapsed_seconds': time.monotonic()-start}, indent=2)+'\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
