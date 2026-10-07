"""Offline chatbot source-contract inspection; no tensor values or inference.

This is the first pipeline stage, not a model converter or a readiness claim.
Source revision is a caller declaration; only explicitly read extents are hashed.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_DONOR_PREFLIGHT_PROTOCOL_20261007.md'
FOREIGN = {
    'benchmarks/donor_adaptation/configs/_manifest.json': 'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
    'benchmarks/donor_adaptation/density/build_document_holdout.py': 'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
    'docs/research/RESEARCH_INDEX.md': '99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273',
}
WIDTHS = {'BOOL': 1, 'I8': 1, 'U8': 1, 'I16': 2, 'U16': 2, 'F16': 2,
          'BF16': 2, 'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_write(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf8')
    assert len(data) <= 2 << 20, 'report exceeds2MiB'
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


class Inspection:
    def __init__(self, output):
        self.started = time.monotonic()
        self.proc = psutil.Process()
        self.proc.cpu_affinity([10])
        self.output = output
        self.inputs = []
        self.bytes_read = 0
        self.r = {'scope': 'METADATA_ONLY_NOT_CONVERSION_OR_CHAT_QUALITY',
                  'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'process_instance': {'pid': self.proc.pid, 'create_time_unix': self.proc.create_time()},
                  'new_tensor_value_bytes_read': 0, 'new_model_native_calls': 0,
                  'source_weight_values_freshly_verified': False, 'pipeline_complete': False}
        assert not output.exists() and not output.with_suffix('.failure.json').exists()
        own = {self.proc.pid, *(p.pid for p in self.proc.parents())}
        for p in psutil.process_iter(['name', 'cmdline']):
            name = (p.info['name'] or '').lower()
            if p.pid in own:
                continue
            argv = p.info['cmdline'] or []
            if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                continue
            assert not name.startswith(('python', 'clang', 'meth')), ('foreign_science', p.pid, name)
        self.timer = threading.Timer(60, self.deadline)
        self.timer.daemon = True
        self.timer.start()

    def resources(self):
        return {'seconds': time.monotonic() - self.started,
                'OS_peak_bytes': self.proc.memory_info().peak_wset,
                'bytes_read': self.bytes_read, 'limits_seconds_peak_bytes': [60, 256 << 20]}

    def guard(self):
        assert time.monotonic() - self.started <= 60
        assert self.proc.memory_info().peak_wset <= 256 << 20

    def deadline(self):
        try:
            self.r.update(fault='hard metadata deadline', resource=self.resources(), inputs=self.inputs)
            unique_write(self.output.with_suffix('.failure.json'), self.r)
        finally:
            os._exit(124)

    def extent(self, path, count=None):
        path = Path(path).absolute()
        before = path.stat()
        n = before.st_size if count is None else count
        assert 0 <= n <= before.st_size and n <= 64 << 20
        with path.open('rb') as f:
            data = f.read(n)
        assert len(data) == n
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        self.bytes_read += n
        self.inputs.append({'path': str(path), 'resolved_path': str(path.resolve()),
                            'offset': 0, 'bytes': n, 'file_bytes': before.st_size,
                            'file_mtime_ns': before.st_mtime_ns, 'sha256': sha(data),
                            'whole_file_SHA_verified': n == before.st_size})
        self.guard()
        return data

    def json(self, path):
        return json.loads(self.extent(path))

    def preserve(self):
        for rel, digest in FOREIGN.items():
            assert sha(self.extent(ROOT / rel)) == digest, ('foreign_byte_drift', rel)
        assert not any(p.is_file() for p in (ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))

    def committed(self):
        for path in (Path(__file__).resolve(), PROTOCOL):
            rel = path.relative_to(ROOT).as_posix()
            committed = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT, creationflags=0x08000000)
            actual = self.extent(path)
            assert committed.replace(b'\r\n', b'\n') == actual.replace(b'\r\n', b'\n'), ('unfrozen_source', rel)
        self.r['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, creationflags=0x08000000).decode().strip()


def tensor_headers(ctx, source, config):
    index_path = source / 'model.safetensors.index.json'
    weight_map = ctx.json(index_path)['weight_map'] if index_path.exists() else None
    files = sorted(set(weight_map.values())) if weight_map is not None else ['model.safetensors']
    all_names = set()
    totals = Counter()
    summaries = []
    for name in files:
        assert Path(name).name == name and name.endswith('.safetensors'), 'unsupported index path'
        path = source / name
        before = path.stat()
        with path.open('rb') as f:
            length_bytes = f.read(8)
            assert len(length_bytes) == 8
            header_size = struct.unpack('<Q', length_bytes)[0]
            assert 2 <= header_size <= (64 << 20) - 8 and 8 + header_size <= before.st_size
            raw = f.read(header_size)
        assert len(raw) == header_size
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        header = json.loads(raw)
        header.pop('__metadata__', None)
        ranges = []
        parameters = mtp_parameters = 0
        for key, item in header.items():
            assert key not in all_names and key and isinstance(key, str)
            all_names.add(key)
            if weight_map is not None:
                assert weight_map[key] == name
            shape = item['shape']
            assert isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape)
            count = math.prod(shape)
            begin, end = item['data_offsets']
            assert type(begin) is int and type(end) is int and 0 <= begin <= end
            assert end - begin == count * WIDTHS[item['dtype']]
            ranges.append((begin, end))
            parameters += count
            totals[item['dtype']] += end - begin
            layer = re.match(r'model\.layers\.(\d+)\.', key)
            if layer and int(layer[1]) >= config['num_hidden_layers']:
                mtp_parameters += count
        ranges.sort()
        assert ranges and ranges[0][0] == 0
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        assert 8 + header_size + ranges[-1][1] == before.st_size
        n = header_size + 8
        ctx.bytes_read += n
        ctx.inputs.append({'path': str(path.absolute()), 'resolved_path': str(path.resolve()),
                           'offset': 0, 'bytes': n, 'file_bytes': before.st_size,
                           'file_mtime_ns': before.st_mtime_ns, 'sha256': sha(length_bytes + raw),
                           'whole_file_SHA_verified': False})
        summaries.append({'file': name, 'tensors': len(header), 'named_parameter_elements': parameters,
                          'layers_at_or_beyond_base_depth_elements': mtp_parameters})
        ctx.guard()
    if weight_map is not None:
        assert all_names == set(weight_map)
    return {'files': summaries, 'tensor_names': len(all_names), 'value_bytes_by_dtype': dict(totals),
            'named_parameter_elements': sum(s['named_parameter_elements'] for s in summaries),
            'layers_at_or_beyond_base_depth_elements': sum(s['layers_at_or_beyond_base_depth_elements'] for s in summaries),
            'value_uniqueness_and_aliases_verified': False, 'tensor_values_read_or_hashed': False}


def producer_template(ctx, path):
    """Read bounded GGUF v2/v3 metadata only; never visit tensor value data."""
    path = Path(path)
    before = path.stat()
    fixed = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4),
             5: ('i', 4), 6: ('f', 4), 7: ('?', 1), 10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}
    selected = {}
    with path.open('rb') as f:
        def read(n):
            assert 0 <= n and f.tell() + n <= min(before.st_size, 64 << 20)
            data = f.read(n)
            assert len(data) == n
            return data
        def skip(n):
            assert 0 <= n and f.tell() + n <= min(before.st_size, 64 << 20)
            f.seek(n, 1)
        def number(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize('<' + fmt)))[0]
        def string(keep):
            n = number('Q')
            if keep:
                return read(n).decode('utf8')
            skip(n)
            return None
        def value(kind, keep=False):
            if kind in fixed:
                fmt, n = fixed[kind]
                return number(fmt) if keep else skip(n)
            if kind == 8:
                return string(keep)
            assert kind == 9 and not keep, 'unsupported selected GGUF type'
            subtype, count = number('I'), number('Q')
            assert count <= 2_000_000
            if subtype in fixed:
                skip(fixed[subtype][1] * count)
            else:
                assert subtype == 8, 'nested/unknown GGUF array'
                for i in range(count):
                    string(False)
                    if i % 4096 == 0:
                        ctx.guard()
        assert read(4) == b'GGUF' and number('I') in (2, 3)
        tensors, count = number('Q'), number('Q')
        assert tensors <= 100_000 and count <= 10_000
        keys = set()
        for _ in range(count):
            key = string(True)
            assert key not in keys
            keys.add(key)
            keep = key in ('tokenizer.chat_template', 'general.architecture', 'general.finetune',
                           'tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id')
            result = value(number('I'), keep)
            if keep:
                selected[key] = result
        end = f.tell()
    assert (before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
    ctx.extent(path, end)
    return {'metadata': selected, 'metadata_extent_bytes': end,
            'tensor_values_or_tensor_descriptors_read': False,
            'HF_GGUF_weight_or_ID_equivalence_newly_verified': False}


def family(config):
    model_type = config.get('model_type')
    if model_type == 'qwen2' and config.get('hidden_act') == 'silu':
        return {'classification': 'CAUSAL_DENSE_SWIGLU_GQA',
                'original_channel': 'd_j * SiLU(g_j dot x + gate_bias_j) * (u_j dot x + up_bias_j)',
                'required_core': ['causal GQA', 'QKV biases', 'RoPE', 'RMSNorm', 'KV cache', 'tied or untied full head'],
                'source_router': 'none; added expert router requires a separately learned/qualified conversion',
                'ReLU_omitted_zero_halfspace_rule_applies': False}
    if model_type == 'deepseek_v3' and config.get('hidden_act') == 'silu':
        return {'classification': 'CAUSAL_MOE_SWIGLU_MLA',
                'original_channel': 'd_j * SiLU(g_j dot x) * (u_j dot x)',
                'required_core': ['MLA', 'partial RoPE/YaRN', 'compressed KV cache', 'RMSNorm', 'shared expert', 'untied full head'],
                'source_router': {k: config.get(k) for k in ('scoring_func', 'topk_method', 'norm_topk_prob', 'routed_scaling_factor', 'n_group', 'topk_group', 'num_experts_per_tok')},
                'ReLU_omitted_zero_halfspace_rule_applies': False}
    return {'classification': 'UNIMPLEMENTED_PREFLIGHT_FAMILY_ADAPTER',
            'model_type': model_type, 'ReLU_omitted_zero_halfspace_rule_applies': None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--expected-config-sha', required=True)
    ap.add_argument('--producer-gguf', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    ctx = None
    try:
        assert re.fullmatch('[0-9a-f]{40}', args.revision)
        assert re.fullmatch('[0-9a-f]{64}', args.expected_config_sha)
        ctx = Inspection(args.out.absolute())
        ctx.committed()
        ctx.preserve()
        config = ctx.json(args.source / 'config.json')
        assert ctx.inputs[-1]['sha256'] == args.expected_config_sha
        for key in ('num_hidden_layers', 'hidden_size', 'vocab_size'):
            assert type(config[key]) is int and config[key] > 0
        tokenizer = ctx.json(args.source / 'tokenizer_config.json')
        ctx.extent(args.source / 'tokenizer.json')  # Hash only; no tokenization.
        generation_path = args.source / 'generation_config.json'
        generation = ctx.json(generation_path) if generation_path.exists() else None
        template = tokenizer.get('chat_template')
        producer = producer_template(ctx, args.producer_gguf) if args.producer_gguf else None
        producer_chat = producer['metadata'].get('tokenizer.chat_template') if producer else None
        eos = generation.get('eos_token_id', config.get('eos_token_id')) if generation else config.get('eos_token_id')
        eos = eos if isinstance(eos, list) else [eos]
        assert eos and all(type(v) is int and 0 <= v < config['vocab_size'] for v in eos)
        headers = tensor_headers(ctx, args.source, config)
        ctx.r.update(model_id=args.model_id, declared_revision=args.revision,
                     revision_remote_verified=False, config=config, family=family(config), tensor_inventory=headers,
                     chat_contract={'HF_template_present': bool(template), 'HF_template_sha256': sha(template.encode()) if isinstance(template, str) else None,
                                    'HF_template': template, 'producer_GGUF_candidate': producer,
                                    'producer_template_present': bool(producer_chat),
                                    'producer_template_sha256': sha(producer_chat.encode()) if isinstance(producer_chat, str) else None,
                                    'rendered_messages_or_token_ID_parity_verified': False,
                                    'generation_config_file_present': generation is not None,
                                    'generation_config': generation, 'effective_declared_EOS_IDs': eos,
                                    'EOS_source': 'generation_config.json' if generation and 'eos_token_id' in generation else 'config.json',
                                    'native_stop_policy_equivalence_verified': False},
                     decision='SOURCE_CONTRACT_INSPECTED_PIPELINE_NOT_QUALIFIED',
                     missing_pipeline_gates=['qualified chat serialization/token IDs/stops', 'fresh donor reference in dialogue domain',
                                             'family-specific compact transformation', 'complete exported native artifact',
                                             'fresh own-history chat/task quality', 'SAME-artifact accepted>=50 end-to-end',
                                             'useful larger n/LUT winner and mass/physical DRAM', 'actual additional families/scales'])
        ctx.preserve()
        for item in ctx.inputs:
            st = Path(item['path']).stat()
            assert (st.st_size, st.st_mtime_ns) == (item['file_bytes'], item['file_mtime_ns'])
        ctx.guard()
        ctx.r.update(inputs=ctx.inputs, resource=ctx.resources(),
                     ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        unique_write(ctx.output, ctx.r)
        ctx.timer.cancel()
        print(json.dumps({'out': str(ctx.output), 'report_sha256': sha(ctx.output.read_bytes()),
                          'decision': ctx.r['decision'], 'family': ctx.r['family']['classification'],
                          'named_parameter_elements': headers['named_parameter_elements'], 'resource': ctx.r['resource']}))
    except BaseException:
        if ctx:
            ctx.timer.cancel()
            fault = ctx.output.with_suffix('.failure.json')
            if not fault.exists():
                ctx.r.update(fault='metadata inspection exception', inputs=ctx.inputs,
                             resource=ctx.resources(), exception_type=sys.exc_info()[0].__name__)
                unique_write(fault, ctx.r)
        raise


if __name__ == '__main__':
    main()
