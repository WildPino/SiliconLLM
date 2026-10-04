"""Bounded full original Switch bank export: row I8 scales, original F32 controls."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time

import numpy as np
import psutil
import torch
import meth324_switch_reference as M
import meth379_switch_tensor_binding as T
import meth337_switch_export_recovery as R

PROTOCOL = M.DOC / 'METH_380_SWITCH_W8A8_EXPORT_PROTOCOL_20261004.md'
BOUND = M.DOC / 'meth379_switch_tensor_binding_result.json'
BOUND_SHA = 'e40710575189b2c635226200a83f56c3242e68915bd0901926bbb9efb728dbe2'
QUALIFIED = M.DOC / 'meth334_switch_full_source_result.json'
QUALIFIED_SHA = '155fd1bc3f5894e9450a6ba2e22f4d08e1e974582e3d27bff62e52ee3f8eafaa'
OUT = M.ROOT / 'results/native_expert_scaling/meth380_switch_w8a8_export'
EMBEDDINGS = {'shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'}


def array_digest(array):
    assert array.flags.c_contiguous
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def quantize(array):
    """Every division is F32, nearest-even rounding, symmetric codes [-127,127]."""
    assert array.dtype == np.float32 and array.ndim == 2 and np.isfinite(array).all()
    maximum = np.max(np.abs(array), axis=1)
    scales = np.divide(maximum, np.float32(127.)).astype(np.float32)
    scales[maximum == 0.] = np.float32(1.)
    assert np.isfinite(scales).all() and (scales > 0.).all(), 'unrepresentable_row_scale'
    codes = np.clip(np.rint(np.divide(array, scales[:, None])), -127., 127.).astype(np.int8)
    assert not np.any(codes == -128)
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales)


def primitive_controls():
    array = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                      [127., -127., .5, 1.5, -.5, -1.5, 2.5, -2.5],
                      [13.25, -19.125, .0125, -.0625, 11.375, -7.75, 0., 3.5]], dtype=np.float32)
    codes, scales = quantize(array)
    oracle = []
    for row in array:
        maximum = max(abs(float(v)) for v in row)
        scale = np.float32(maximum / 127.) if maximum else np.float32(1.)
        oracle.append([max(-127, min(127, round(float(np.float32(float(v) / float(scale)))))) for v in row])
    extreme = np.full(4096, 127, dtype=np.int8)
    cancellation = extreme.copy(); cancellation[::2] = -127
    dots = [int(extreme.astype(np.int64) @ extreme.astype(np.int64)),
            int(extreme.astype(np.int64) @ cancellation.astype(np.int64))]
    return {'exact_scalar_nearest_even': bool(np.array_equal(codes, np.array(oracle, dtype=np.int8))),
            'zero_row_scale': float(scales[0]), 'tie_row': codes[1].tolist(),
            'integer_extreme_and_cancellation': dots,
            'passed': bool(np.array_equal(codes, np.array(oracle, dtype=np.int8)))
            and scales[0] == 1. and codes[1].tolist() == [127, -127, 0, 2, 0, -2, 2, -2]
            and dots == [66064384, 0]}


def compressed(name, shape):
    return len(shape) == 2 and name not in EMBEDDINGS and 'relative_attention_bias' not in name and '.router.' not in name


def append_array(stream, array):
    padding = (-stream.tell()) % 64
    stream.write(b'\0' * padding)
    offset = stream.tell()
    data = memoryview(array).cast('B')
    assert stream.write(data) == len(data)
    return offset, len(data), hashlib.sha256(data).hexdigest()


def emit_tensor(stream, name, array, aliases):
    shape = list(array.shape)
    entry = {'shape': shape, 'elements': int(array.size), 'encoding': int(compressed(name, shape)),
             'source_sha256': array_digest(array)}
    if name in EMBEDDINGS and aliases:
        other = next(iter(aliases.values()))
        assert entry['source_sha256'] == other['source_sha256'] and shape == other['shape']
        entry.update({k: other[k] for k in ('offset', 'bytes', 'sha256', 'scale_offset', 'scale_bytes', 'scale_sha256')})
        entry['physical_alias'] = next(iter(aliases))
        aliases[name] = entry
        return entry
    if entry['encoding']:
        codes, scales = R.quantize(array)
        entry['offset'], entry['bytes'], entry['sha256'] = append_array(stream, codes)
        entry['scale_offset'], entry['scale_bytes'], entry['scale_sha256'] = append_array(stream, scales)
    else:
        entry['offset'], entry['bytes'], entry['sha256'] = append_array(stream, array)
        entry.update({'scale_offset': 0, 'scale_bytes': 0, 'scale_sha256': None})
    if name in EMBEDDINGS:
        aliases[name] = entry
    return entry


def manifest(config, entries, payload, path):
    values = [config[k] for k in ('d_model', 'd_ff', 'num_heads', 'd_kv', 'num_layers',
              'num_decoder_layers', 'num_experts', 'expert_capacity', 'vocab_size',
              'relative_attention_num_buckets', 'relative_attention_max_distance',
              'encoder_sparse_step', 'decoder_sparse_step')]
    with path.open('xb') as stream:
        stream.write(b'SWI8A001')
        stream.write(struct.pack('<13IfII', *values, config['layer_norm_epsilon'], 1, len(entries)))
        name = str(payload.resolve()).encode('utf-8')
        stream.write(struct.pack('<I', len(name))); stream.write(name)
        for name, entry in sorted(entries.items()):
            text = name.encode('utf-8'); stream.write(struct.pack('<I', len(text))); stream.write(text)
            shape = entry['shape']
            stream.write(struct.pack('<5I3Q', 0, len(shape), shape[0], shape[1] if len(shape) == 2 else 1,
                         entry['encoding'], entry['offset'], entry['scale_offset'], entry['elements']))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); stage = 'bindings'; maximum_rss = 0
    result = {'experiment': 'METH-380-full-source-all-bank-row-I8-export', 'shards': []}
    def guard():
        nonlocal maximum_rss
        rss = psutil.Process().memory_info().rss; maximum_rss = max(maximum_rss, rss)
        assert rss <= 24 << 30 and time.monotonic() - start <= 2400, 'export_resource_guard'
    try:
        for path in (Path(__file__), PROTOCOL, BOUND, QUALIFIED, T.ACQUISITION, T.HEADERS, Path(T.__file__), Path(M.__file__), Path(R.__file__), Path(R.E.__file__)):
            M.committed(path)
        assert M.digest(BOUND) == BOUND_SHA and M.digest(QUALIFIED) == QUALIFIED_SHA
        bound = json.loads(BOUND.read_text(encoding='utf-8')); qualified = json.loads(QUALIFIED.read_text(encoding='utf-8'))
        assert bound['passed'] and all(qualified['gates'].values()) and torch.__version__ == '2.6.0+cu124'
        assert M.digest(T.HEADERS) == T.HEADERS_SHA and M.digest(T.ACQUISITION) == bound['acquisition_sha256']
        acquisition = json.loads(T.ACQUISITION.read_text(encoding='utf-8')); assert acquisition['passed']
        assert shutil.disk_usage(T.SOURCE).free >= 32 << 30
        torch.set_num_threads(1)
        result.update({'controller_sha256': M.digest(__file__), 'protocol_sha256': M.digest(PROTOCOL),
                       'source_binding_sha256': BOUND_SHA, 'qualified334_sha256': QUALIFIED_SHA,
                       'model': bound['model'], 'revision': bound['revision'], 'primitive_controls': primitive_controls()})
        result['safe_quantizer_equivalence_controls'] = R.equivalence_controls()
        assert result['primitive_controls']['passed'] and result['safe_quantizer_equivalence_controls']['passed']
        stage = 'fresh_source_file_hashes'
        for item in acquisition['files']:
            source = Path(item['path']); assert item['complete'] and source.stat().st_size == item['bytes']
            assert T.file_digest(source, start) == item['sha256']; guard()
        header = next(v for v in json.loads(T.HEADERS.read_text(encoding='utf-8'))['models'] if v['model'] == bound['model'])
        config = json.loads((T.SOURCE / 'config.json').read_text(encoding='utf-8'))
        assert (config['num_experts'], config['d_model'], config['d_ff']) == (128, 768, 3072)
        OUT.mkdir(parents=True); payload = OUT / 'weights.bin'; entries = {}; aliases = {}
        stage = 'source_tensor_quantization'
        with payload.open('xb') as stream:
            for shard in header['shards']:
                shard_start = time.monotonic()
                state = torch.load(T.SOURCE / shard['name'], weights_only=True, mmap=True, map_location='cpu')
                for name, tensor in sorted(state.items()):
                    assert name not in entries and tensor.dtype == torch.float32 and tensor.is_contiguous()
                    array = tensor.detach().numpy(); expected = bound['tensors'][name]
                    assert list(array.shape) == expected['shape'] and np.isfinite(array).all()
                    entry = emit_tensor(stream, name, array, aliases)
                    assert entry['source_sha256'] == expected['sha256']
                    entries[name] = entry
                    assert stream.tell() <= 24 << 30; guard()
                del array, tensor, state; gc.collect(); guard()
                item = {'name': shard['name'], 'target_bytes_so_far': stream.tell(), 'source_tensors_so_far': len(entries),
                        'seconds': time.monotonic() - shard_start}
                result['shards'].append(item); print(json.dumps(item), flush=True)
        assert set(entries) == set(bound['tensors']) and len(entries) == 3320
        assert set(aliases) == EMBEDDINGS
        stage = 'all_serialized_target_readback'
        with payload.open('rb') as stream:
            for name, entry in entries.items():
                stream.seek(entry['offset']); data = stream.read(entry['bytes'])
                assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
                if entry['encoding']:
                    assert entry['bytes'] == entry['elements'] and not np.any(np.frombuffer(data, dtype=np.int8) == -128)
                    stream.seek(entry['scale_offset']); scales = stream.read(entry['scale_bytes'])
                    assert len(scales) == entry['shape'][0] * 4 and hashlib.sha256(scales).hexdigest() == entry['scale_sha256']
                    values = np.frombuffer(scales, dtype='<f4'); assert np.isfinite(values).all() and (values > 0.).all()
                else:
                    assert entry['bytes'] == entry['elements'] * 4
                guard()
        banks = {}
        for name, entry in entries.items():
            match = T.EXPERT.fullmatch(name)
            if not match: continue
            assert entry['encoding'] == 1
            stack, layer, slot, label, organ = match.groups(); key = f'{stack}.block.{layer}.layer.{slot}'
            banks.setdefault(key, {}).setdefault(int(label), {})[organ] = entry
        tuples = {}
        for key, experts in banks.items():
            assert set(experts) == set(range(128))
            values = []
            for label, organs in sorted(experts.items()):
                assert set(organs) == {'wi', 'wo'}
                description = [[organ, organs[organ]['shape'], organs[organ]['sha256'], organs[organ]['scale_sha256']]
                               for organ in ('wi', 'wo')]
                values.append(hashlib.sha256(json.dumps(description, separators=(',', ':')).encode()).hexdigest())
            assert len(set(values)) == 128, 'target_parameter_tuple_collapse'
            tuples[key] = {'labels': 128, 'distinct_code_and_scale_tuples': 128, 'tuple_sha256_by_label': values}
        assert len(tuples) == 12
        spec = OUT / 'manifest.bin'; manifest(config, entries, payload, spec)
        result['tensors'] = entries; result['target_banks'] = tuples
        result['artifact'] = {'payload': str(payload), 'bytes': payload.stat().st_size,
                              'sha256': T.file_digest(payload, start), 'manifest': str(spec), 'manifest_sha256': M.digest(spec),
                              'source_unique_parameters': 7415217408, 'original_experts_per_bank': 128,
                              'f32_embedding_alias_names': sorted(aliases), 'f32_embedding_physical_copies': 1,
                              'separate_I8_head_view_from_same_tied_source': True,
                              'row_I8_tensor_names': sum(v['encoding'] == 1 for v in entries.values()),
                              'f32_tensor_names': sum(v['encoding'] == 0 for v in entries.values())}
        result['original_config'] = config
        result['gates'] = {'primitive_nearest_even_integer_controls': result['primitive_controls']['passed'],
                           'all_actual_source_tensors': True, 'all_serialized_target_readback': True,
                           'all12_banks128_distinct_code_scale_tuples': True, 'original_embedding_controls_exact': True,
                           'safe_quantizer_exact_335_controls': result['safe_quantizer_equivalence_controls']['passed']}
        result['resource'] = {'main_seconds_excluding_imports': time.monotonic() - start,
                              'maximum_checked_rss_bytes': maximum_rss, 'end_rss_bytes': psutil.Process().memory_info().rss}
        result['safe_subnormal_values_bypassed'] = R.SAFE_SUBNORMAL_VALUES
        result['scope'] = 'Common334 operator semantics reused, new source-specific target numerics still required. Actual original7.415B source parameter representation; quantization approximate, no useful-function/quality/native integer execution/LUT or accepted-rate claim. All banks retained; target code/scale tuples distinct is not effective function diversity.'
        result['decision'] = 'eligible_for_separate_native_integer_scaling_and_complete_compact_cpu_cost'
        guard(); M.write(args.out, result)
        print(json.dumps({'sha256': M.digest(args.out), 'artifact': result['artifact'], 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update({'stage': stage, 'error': repr(error), 'main_seconds_excluding_imports': time.monotonic() - start,
                       'maximum_checked_rss_bytes': maximum_rss})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
