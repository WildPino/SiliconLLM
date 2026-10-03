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
import meth327_switch_tensor_binding as T
import meth335_switch_w8a8_export as E

PROTOCOL = M.DOC / 'METH_337_SWITCH_EXPORT_RECOVERY_PROTOCOL_20261003.md'
BOUND = M.DOC / 'meth327_switch_tensor_binding_result.json'
BOUND_SHA = 'e72fc17b5527dc34df5c00ed7c44498b2ee4fc5d338ed8b130fff101a30c20a7'
QUALIFIED = M.DOC / 'meth334_switch_full_source_result.json'
QUALIFIED_SHA = '155fd1bc3f5894e9450a6ba2e22f4d08e1e974582e3d27bff62e52ee3f8eafaa'
OUT = M.ROOT / 'results/native_expert_scaling/meth337_switch_export_recovery'
FAILED = M.DOC / 'meth335_switch_w8a8_export_result.failure.json'
FAILED_SHA = '871be9e23b05755b1a44a3272419ffa8e09a3f701b7d0564585a7c30c30e4e67'
PAYLOAD = M.ROOT / 'results/native_expert_scaling/meth335_switch_w8a8_export/weights.bin'
SAFE_SUBNORMAL_VALUES = 0
EMBEDDINGS = {'shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'}


def array_digest(array):
    assert array.flags.c_contiguous
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def quantize(array):
    """Exact original recipe, with provably-zero subnormal operands bypassed."""
    global SAFE_SUBNORMAL_VALUES
    assert array.dtype == np.float32 and array.ndim == 2 and np.isfinite(array).all()
    absolute_bits = array.view(np.uint32) & np.uint32(0x7fffffff)
    maximum = np.ascontiguousarray(absolute_bits.max(axis=1)).view(np.float32)
    scales = np.divide(maximum, np.float32(127.)).astype(np.float32)
    scales[maximum == 0.] = np.float32(1.)
    assert np.isfinite(scales).all() and (scales > 0.).all(), 'unrepresentable_row_scale'
    # scale>=2*smallest_normal implies |subnormal/scale|<0.5: nearest-even
    # code exactly zero under335. Rows with smaller scales use the full recipe.
    safe = (absolute_bits > 0) & (absolute_bits < np.uint32(0x00800000))
    safe &= (scales >= np.float32(2.) * np.finfo(np.float32).tiny)[:, None]
    count = int(np.count_nonzero(safe)); SAFE_SUBNORMAL_VALUES += count
    if count:
        values = array.copy(); values[safe] = np.float32(0.)
    else:
        values = array
    codes = np.clip(np.rint(np.divide(values, scales[:, None])), -127., 127.).astype(np.int8)
    assert not np.any(codes == -128)
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales)


class ReadonlyComparison:
    def __init__(self, path):
        self.stream = path.open('rb')
        self.calls = 0
    def __enter__(self): return self
    def __exit__(self, *args): self.stream.close()
    def tell(self): return self.stream.tell()
    def write(self, expected):
        actual = self.stream.read(len(expected))
        assert len(actual) == len(expected) and hashlib.sha256(actual).digest() == hashlib.sha256(expected).digest(), 'actual335_payload_mismatch'
        self.calls += 1
        return len(expected)


def equivalence_controls():
    rng = np.random.default_rng(337); cases = []
    for cols in (8, 15, 17, 768, 3072, 4096):
        values = rng.normal(size=(3, cols)).astype(np.float32)
        values[0,0] = 127.; values[0,1:7] = [.5, 1.5, -.5, -1.5, 2.5, -2.5]
        values[1,:] = np.float32(1e-40); values[1,::2] *= np.float32(-1.)
        values[2,0] = 127.; values[2,1:4] = [np.nextafter(np.float32(0.), np.float32(1.)), np.float32(-1e-38), np.float32(1e-38)]
        old_codes, old_scales = E.quantize(values); codes, scales = quantize(values)
        cases.append({'cols': cols, 'exact_old335_codes_and_scales': bool(np.array_equal(codes, old_codes) and np.array_equal(scales, old_scales))})
    return {'seed': 337, 'cases': cases, 'passed': all(v['exact_old335_codes_and_scales'] for v in cases)}


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
        codes, scales = quantize(array)
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
    result = {'experiment': 'METH-337-readonly-complete-export-recovery', 'shards': []}
    def guard():
        nonlocal maximum_rss
        rss = psutil.Process().memory_info().rss; maximum_rss = max(maximum_rss, rss)
        assert rss <= 16 << 30 and time.monotonic() - start <= 1200, 'export_resource_guard'
    try:
        for path in (Path(__file__), PROTOCOL, FAILED, Path(E.__file__), BOUND, QUALIFIED, T.ACQUISITION, T.HEADERS, Path(T.__file__), Path(M.__file__)):
            M.committed(path)
        assert M.digest(FAILED) == FAILED_SHA
        failed = json.loads(FAILED.read_text())
        assert failed['stage'] == 'all_serialized_target_readback' and len(failed['shards']) == 6
        assert failed['shards'][-1]['source_tensors_so_far'] == 6392 and PAYLOAD.stat().st_size == failed['shards'][-1]['target_bytes_so_far'] == 14818015744
        original_stat = PAYLOAD.stat()
        result['preserved335_failure_sha256'] = FAILED_SHA
        result['equivalence_controls'] = equivalence_controls()
        assert result['equivalence_controls']['passed']
        global SAFE_SUBNORMAL_VALUES
        SAFE_SUBNORMAL_VALUES = 0
        assert M.digest(BOUND) == BOUND_SHA and M.digest(QUALIFIED) == QUALIFIED_SHA
        bound = json.loads(BOUND.read_text()); qualified = json.loads(QUALIFIED.read_text())
        assert bound['passed'] and all(qualified['gates'].values()) and torch.__version__ == '2.6.0+cu124'
        assert M.digest(T.HEADERS) == T.HEADERS_SHA and M.digest(T.ACQUISITION) == bound['acquisition_sha256']
        acquisition = json.loads(T.ACQUISITION.read_text()); assert acquisition['passed']
        assert shutil.disk_usage(T.SOURCE).free >= 32 << 30
        torch.set_num_threads(1)
        result.update({'controller_sha256': M.digest(__file__), 'protocol_sha256': M.digest(PROTOCOL),
                       'source_binding_sha256': BOUND_SHA, 'qualified334_sha256': QUALIFIED_SHA,
                       'model': bound['model'], 'revision': bound['revision'], 'primitive_controls': primitive_controls()})
        assert result['primitive_controls']['passed']
        stage = 'fresh_source_file_hashes'
        for item in acquisition['files']:
            source = Path(item['path']); assert item['complete'] and source.stat().st_size == item['bytes']
            assert T.file_digest(source, start) == item['sha256']; guard()
        header = next(v for v in json.loads(T.HEADERS.read_text())['models'] if v['model'] == bound['model'])
        config = json.loads((T.SOURCE / 'config.json').read_text())
        assert (config['num_experts'], config['d_model'], config['d_ff']) == (256, 768, 3072)
        OUT.mkdir(parents=True); payload = PAYLOAD; entries = {}; aliases = {}
        stage = 'source_tensor_reconstruction_and_readonly_payload_verification'
        with ReadonlyComparison(payload) as stream:
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
                    result['verified_source_target_tensors_so_far'] = len(entries)
                    assert stream.tell() <= 24 << 30; guard()
                del array, tensor, state; gc.collect(); guard()
                item = {'name': shard['name'], 'target_bytes_so_far': stream.tell(), 'source_tensors_so_far': len(entries),
                        'seconds': time.monotonic() - shard_start}
                result['shards'].append(item); print(json.dumps(item), flush=True)
            verified_payload_bytes = stream.tell()
        assert verified_payload_bytes == PAYLOAD.stat().st_size
        assert set(entries) == set(bound['tensors']) and len(entries) == 6392
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
            assert set(experts) == set(range(256))
            values = []
            for label, organs in sorted(experts.items()):
                assert set(organs) == {'wi', 'wo'}
                description = [[organ, organs[organ]['shape'], organs[organ]['sha256'], organs[organ]['scale_sha256']]
                               for organ in ('wi', 'wo')]
                values.append(hashlib.sha256(json.dumps(description, separators=(',', ':')).encode()).hexdigest())
            assert len(set(values)) == 256, 'target_parameter_tuple_collapse'
            tuples[key] = {'labels': 256, 'distinct_code_and_scale_tuples': 256, 'tuple_sha256_by_label': values}
        assert len(tuples) == 12
        spec = OUT / 'manifest.bin'; manifest(config, entries, payload, spec)
        result['tensors'] = entries; result['target_banks'] = tuples
        result['artifact'] = {'payload': str(payload), 'bytes': payload.stat().st_size,
                              'sha256': T.file_digest(payload, start), 'manifest': str(spec), 'manifest_sha256': M.digest(spec),
                              'source_unique_parameters': 14664154368, 'original_experts_per_bank': 256,
                              'f32_embedding_alias_names': sorted(aliases), 'f32_embedding_physical_copies': 1,
                              'separate_I8_head_view_from_same_tied_source': True,
                              'new_payload_write_bytes': 0,
                              'row_I8_tensor_names': sum(v['encoding'] == 1 for v in entries.values()),
                              'f32_tensor_names': sum(v['encoding'] == 0 for v in entries.values())}
        assert PAYLOAD.stat().st_mtime_ns == original_stat.st_mtime_ns and PAYLOAD.stat().st_size == original_stat.st_size
        result['safe_subnormal_source_values_bypassed_with_exact_zero_code_proof'] = SAFE_SUBNORMAL_VALUES
        result['prior335_main_seconds'] = failed['main_seconds_excluding_imports']
        result['original_config'] = config
        result['gates'] = {'primitive_nearest_even_integer_controls': result['primitive_controls']['passed'],
                           'all_actual_source_tensors': True, 'all_serialized_target_readback': True,
                           'all12_banks256_distinct_code_scale_tuples': True, 'original_embedding_controls_exact': True, 'exact_recipe_equivalence_controls': True, 'all_actual_existing_bytes_match_expected': True}
        result['resource'] = {'main_seconds_excluding_imports': time.monotonic() - start,
                              'maximum_checked_rss_bytes': maximum_rss, 'end_rss_bytes': psutil.Process().memory_info().rss}
        result['scope'] = 'Actual original14.664B source parameter representation; quantization approximate, no useful-function/quality/native integer execution/LUT or accepted-rate claim. All banks retained; target code/scale tuples distinct is not effective function diversity.'
        result['decision'] = 'eligible_for_separate_native_integer_scaling_and_complete_compact_cpu_cost'
        guard(); M.write(args.out, result)
        print(json.dumps({'sha256': M.digest(args.out), 'artifact': result['artifact'], 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update({'stage': stage, 'error': repr(error), 'main_seconds_excluding_imports': time.monotonic() - start,
                       'maximum_checked_rss_bytes': maximum_rss})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
