#!/usr/bin/env python3
"""Complete descriptor cost of the full-feature U8/two-coefficient LUT proposal.

No tensor payload, model inference, palette training or native timing is read.
Logical accesses are kept separate from physical DRAM traffic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from types import SimpleNamespace

import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
MODELS = ROOT / 'benchmarks/donor_adaptation/density/results'
SOURCE = MODELS / 'strat01_gigachat_source_binding_v1/GigaChat3.1-10B-A1.8B-source-bf16.gguf'
Q4 = MODELS / 'strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf'
PINS = {
    'meth01_gigachat_q4_active_ledger.json': '04df70cc6372d193ddfe0294910b6f30b1efdb248f9fddf18cdb265c54a760fc',
    'meth04_gigachat_lowbit_preflight.json': '99505b980e5973e30da003dddb62f26a0a45c6e29754258724f0ead6b14efcd7',
}
HELPER = ROOT / 'benchmarks/native_expert_scaling/gguf_active_ledger.py'
HELPER_SHA = '3554ff313a1b686eccc2b95371fa3f9b5aa04872558b995bde32c8e6c724ddb5'
TYPES = {0: ('F32', 1, 4), 6: ('Q5_0', 32, 22),
         12: ('Q4_K', 256, 144), 14: ('Q6_K', 256, 210),
         30: ('BF16', 1, 2)}
PALETTE_ENTRIES = 225
PAIR = 2
WEIGHT_BUDGET = 560_000_000
STREAM_BYTES_PER_SECOND = 40_000_000_000
MAX_SECONDS = 120
MAX_RSS = 1 << 30
MAX_HEADER = 16 << 20
FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\r\n') as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + '\n')


def header(path, expected_bytes):
    """Parse only metadata/descriptors/alignment; never read tensor payload."""
    assert path.stat().st_size == expected_bytes, 'model size mismatch'
    digest = hashlib.sha256()
    fields, tensors = {}, {}
    with path.open('rb') as f:
        def read(n):
            assert 0 <= n and f.tell() + n <= MAX_HEADER, 'header resource stop'
            b = f.read(n)
            assert len(b) == n, 'truncated GGUF header'
            digest.update(b)
            return b

        def u32(): return struct.unpack('<I', read(4))[0]
        def u64(): return struct.unpack('<Q', read(8))[0]
        def string(keep=True):
            n = u64()
            assert n <= 1 << 20, 'overlarge string'
            b = read(n)
            return b.decode('utf-8') if keep else None

        def value(kind, keep=True):
            if kind == 8:
                return string(keep)
            if kind == 9:
                subtype, n = u32(), u64()
                assert subtype != 9 and n <= 1 << 20, 'unsupported array'
                for _ in range(n): value(subtype, False)
                return None
            assert kind in FORMATS, 'unsupported metadata type'
            fmt = '<' + FORMATS[kind]
            b = read(struct.calcsize(fmt))
            return struct.unpack(fmt, b)[0] if keep else None

        assert read(4) == b'GGUF' and u32() == 3, 'not GGUF v3'
        nt, nf = u64(), u64()
        assert nt == 414 and nf < 500
        for _ in range(nf):
            name = string()
            assert name not in fields, 'duplicate metadata key'
            kind = u32()
            fields[name] = value(kind, kind != 9)
        alignment = fields.get('general.alignment', 32)
        assert alignment == 32
        for _ in range(nt):
            name, nd = string(), u32()
            assert name not in tensors and 1 <= nd <= 3
            dims = tuple(u64() for _ in range(nd))
            dtype, offset = u32(), u64()
            assert dtype in TYPES and all(0 < d < 1 << 20 for d in dims)
            type_name, block, size = TYPES[dtype]
            elements = math.prod(dims)
            assert dims[0] % block == 0
            tensors[name] = {'shape': dims, 'type': type_name, 'offset': offset,
                             'elements': elements, 'bytes': elements // block * size}
        data_offset = (f.tell() + alignment - 1) // alignment * alignment
        read(data_offset - f.tell())
        assert f.tell() == data_offset
    end = 0
    for t in sorted(tensors.values(), key=lambda t: t['offset']):
        assert t['offset'] % alignment == 0 and t['offset'] == end, 'gap/overlap'
        end += t['bytes']
    assert end + data_offset == expected_bytes, 'header/payload size mismatch'
    return fields, tensors, {'path': str(path), 'file_bytes': expected_bytes,
                            'header_bytes_read': data_offset, 'header_sha256': digest.hexdigest(),
                            'full_payload_hash_rechecked': False}


def aggregate(rows):
    organs = defaultdict(lambda: defaultdict(int))
    for row in rows:
        g = organs[row['organ']]
        for key in ('source_active_bytes', 'q4_active_bytes', 'active_elements',
                    'code_bytes', 'scale_bytes', 'stored_code_bytes', 'stored_scale_bytes',
                    'query_table_entries', 'palette_bytes', 'active_rows', 'input_values'):
            g[key] += row.get(key, 0)
        g['tensor_count'] += 1
    return {k: dict(v) for k, v in sorted(organs.items())}


def price(rows, encoded_organs, table_policy='projection'):
    """Source-derived analytical scenario; counts no physical DRAM accesses."""
    selected = [r for r in rows if r['organ'] in encoded_organs]
    unchanged = [r for r in rows if r['organ'] not in encoded_organs]
    codes = sum(r['code_bytes'] for r in selected)
    scales = sum(r['scale_bytes'] for r in selected)
    palettes = sum(r['palette_bytes'] for r in selected)
    entries = sum(r['query_table_entries'] for r in selected)
    inputs = sum(r['input_values'] for r in selected)
    if table_policy == 'optimistic_common_palettes':
        assert encoded_organs == {'mla', 'routed', 'shared', 'dense0', 'head'}
        # A common Q/KV-A palette, common gate/up/shared input palette and
        # common routed/shared down palette. All head/down inputs still differ.
        saved_pairs = 26 * 768 + 25 * (3 * 768) + 768
        entries -= saved_pairs * PALETTE_ENTRIES
        inputs -= saved_pairs * PAIR
        palettes = (26 * 4 + 25 * 2 + 2 + 1) * PALETTE_ENTRIES * PAIR * 4
    else:
        assert table_policy == 'projection'
    remainder = sum(r['q4_active_bytes'] for r in unchanged)
    weight = codes + scales + palettes + remainder
    return {
        'encoded_organs': sorted(encoded_organs), 'table_policy': table_policy,
        'codes_bytes_per_token': codes, 'scales_bytes_per_token': scales,
        'palette_bytes_stored_and_conservatively_addressed_per_token': palettes,
        'unchanged_q4_bytes_per_token': remainder,
        'total_addressed_weight_bytes_per_token': weight,
        'without_palette_weight_bytes_per_token': weight - palettes,
        'passes_560mb_addressed_weight_gate': weight <= WEIGHT_BUDGET,
        'yardstick_weight_ms_at_40gb_s': weight / STREAM_BYTES_PER_SECOND * 1000,
        'required_weight_gb_s_at_50tok_s': weight * 50 / 1e9,
        'encoded_lookup_adds_per_token': codes,
        'logical_f32_table_gather_bytes_per_token': codes * 4,
        'query_table_entries_per_token': entries,
        'query_table_write_bytes_per_token': entries * 4,
        'query_table_construct_multiply_add_operations': entries * 3,
        'logical_centroid_operand_read_bytes_per_token': entries * 8,
        'distinct_table_input_bytes_per_token': inputs * 4,
        'row_scale_multiplications_per_token': sum(r['active_rows'] for r in selected),
        'largest_single_query_table_bytes': max(r['input_width'] // PAIR * PALETTE_ENTRIES * 4 for r in selected),
        'scope': 'Logical weight/input/LUT accesses and operations; not physical DRAM, cache residency, latency or quality.',
    }


def run(out):
    start = time.monotonic()
    stage = 'bindings'
    try:
        assert sha(HELPER) == HELPER_SHA
        from gguf_active_ledger import expected_names, price_tensor
        records = {}
        for name, pinned in PINS.items():
            assert sha(DOC / name) == pinned, name
            records[name] = json.loads((DOC / name).read_text(encoding='utf-8'))
        a = records['meth01_gigachat_q4_active_ledger.json']
        b = records['meth04_gigachat_lowbit_preflight.json']
        stage = 'headers'
        fs, ts, bs = header(SOURCE, b['source']['bytes'])
        fq, tq, bq = header(Q4, a['file_bytes'])
        wanted = {'general.architecture': 'deepseek2', 'deepseek2.block_count': 26,
                  'deepseek2.leading_dense_block_count': 1, 'deepseek2.expert_count': 64,
                  'deepseek2.expert_used_count': 4, 'deepseek2.vocab_size': 128256}
        assert all(fs[k] == fq[k] == v for k, v in wanted.items())
        assert set(ts) == set(tq) == expected_names(26, 1)
        assert all(ts[k]['shape'] == tq[k]['shape'] for k in ts)
        assert bs['header_bytes_read'] == b['source']['gguf_data_offset']
        bs['previous_full_sha256'] = b['source']['sha256']
        bq['previous_full_sha256'] = '68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb'
        rows = []
        stage = 'tensor_costs'
        encoded = {'mla', 'routed', 'shared', 'dense0', 'head'}
        for name, t in sorted(ts.items()):
            shape = t['shape']
            obj = SimpleNamespace(name=name, shape=shape, n_bytes=t['bytes'], n_elements=t['elements'])
            organ, sb, ae = price_tensor(obj, 64, 4, 128256)
            qt = tq[name]
            _, qb, qae = price_tensor(SimpleNamespace(name=name, shape=shape, n_bytes=qt['bytes'],
                                                      n_elements=qt['elements']), 64, 4, 128256)
            assert ae == qae
            r = {'name': name, 'organ': organ, 'gguf_shape': list(shape), 'source_type': t['type'],
                 'q4_type': qt['type'], 'source_active_bytes': sb, 'q4_active_bytes': qb,
                 'source_stored_bytes': t['bytes'], 'q4_stored_bytes': qt['bytes'], 'active_elements': ae}
            if organ in encoded:
                assert t['type'] == 'BF16' and len(shape) in (2, 3) and shape[0] % PAIR == 0
                input_width, output_width = shape[:2]
                banks = shape[2] if len(shape) == 3 else 1
                active_banks = 4 if organ == 'routed' else banks
                query_banks = 1 if organ == 'routed' and not name.endswith('ffn_down_exps.weight') else active_banks
                assert ae == input_width * output_width * active_banks
                r.update({'input_width': input_width, 'output_width': output_width, 'stored_banks': banks,
                          'active_banks': active_banks, 'distinct_query_inputs': query_banks,
                          'active_rows': output_width * active_banks,
                          'code_bytes': ae // PAIR, 'scale_bytes': output_width * active_banks * 4,
                          'stored_code_bytes': t['elements'] // PAIR,
                          'stored_scale_bytes': output_width * banks * 4,
                          'palette_bytes': PALETTE_ENTRIES * PAIR * 4,
                          'query_table_entries': input_width // PAIR * PALETTE_ENTRIES * query_banks,
                          'input_values': input_width * query_banks})
            rows.append(r)
        groups = aggregate(rows)
        for k, g in groups.items():
            assert g['q4_active_bytes'] == a['organs'][k]['active_payload_bytes']
            assert g['source_active_bytes'] == b['source_organs'][k]['active_bytes']
            assert g['tensor_count'] == a['organs'][k]['tensor_count']
            assert sum(r['source_stored_bytes'] for r in rows if r['organ'] == k) == b['source_organs'][k]['stored_bytes']
            assert sum(r['q4_stored_bytes'] for r in rows if r['organ'] == k) == a['organs'][k]['stored_bytes']
        assert sum(r['q4_active_bytes'] for r in rows) == 1_016_194_144
        assert sum(r['active_elements'] for r in rows if r['organ'] in encoded | {'router'}) == 1_628_078_080
        assert groups['routed']['code_bytes'] + groups['routed']['scale_bytes'] == 296_550_400
        assert groups['routed']['query_table_entries'] * 4 == 92_160_000
        stage = 'complete_scenarios'
        scenarios = {'routed_only': price(rows, {'routed'}),
                     'complete_projection_palettes': price(rows, encoded),
                     'complete_optimistic_common_palettes': price(rows, encoded, 'optimistic_common_palettes')}
        complete = scenarios['complete_projection_palettes']
        head = groups['head']
        no_head_bound = complete['without_palette_weight_bytes_per_token'] - head['code_bytes'] - head['scale_bytes']
        # Diagnostic required code precision ceiling at unchanged full features;
        # no alternate representation is fitted or granted quality eligibility.
        coded_elements = sum(r['active_elements'] for r in rows if r['organ'] in encoded)
        fixed = complete['scales_bytes_per_token'] + complete['unchanged_q4_bytes_per_token']
        code_precision_ceiling = (WEIGHT_BUDGET - fixed) * 8 / coded_elements
        stored = sum((r['stored_code_bytes'] + r['stored_scale_bytes'] + r['palette_bytes'])
                     if r['organ'] in encoded else r['q4_stored_bytes'] for r in rows)
        routed_per_expert = (groups['routed']['stored_code_bytes'] + groups['routed']['stored_scale_bytes']) // 64
        router_per_expert = groups['router']['q4_active_bytes'] // 64
        bias_per_expert = 25 * 4
        parametric = []
        for n in (64, 640, 6400):
            delta = n - 64
            parametric.append({'experts_per_layer': n, 'source_experts_actually_available': 64,
                               'analytical_only_not_instantiated_capacity': n != 64,
                               'encoded_model_payload_bytes': stored + delta * (routed_per_expert + router_per_expert + bias_per_expert),
                               'flat_f32_router_addressed_bytes_per_token': n * router_per_expert,
                               'flat_router_multiply_add_coefficients_per_token': n * router_per_expert // 4,
                               'selected_routed_codes_and_scales_per_token': groups['routed']['code_bytes'] + groups['routed']['scale_bytes'],
                               'complete_addressed_weight_bytes_per_token': complete['total_addressed_weight_bytes_per_token'] + delta * (router_per_expert + bias_per_expert),
                               'selected_experts_per_layer': 4,
                               'query_table_entries_per_token': complete['query_table_entries_per_token']})
        runtime = {'seconds': time.monotonic() - start, 'end_rss_bytes': psutil.Process().memory_info().rss}
        assert runtime['seconds'] < MAX_SECONDS and runtime['end_rss_bytes'] < MAX_RSS
        result = {'experiment': 'METH-300-GigaChat-full-feature-two-coefficient-U8-LUT-cost',
                  'source_revision': '189fff27a1dee68473960c3d5bca53e0e07a3191',
                  'source_bindings': {'bf16': bs, 'q4': bq, 'prior_records': PINS, 'helper_sha256': HELPER_SHA},
                  'format': {'coefficients_per_U8_index': PAIR, 'palette_entries': PALETTE_ENTRIES,
                             'centroid_dtype': 'F32', 'positive_scale_dtype': 'F32', 'scale_scope': 'one per output row'},
                  'tensor_rows': rows, 'organs': groups, 'scenarios': scenarios,
                  'encoded_model_payload_bytes': stored,
                  'diagnostic_zero_head_zero_palette_bound_bytes_per_token': no_head_bound,
                  'diagnostic_max_code_bits_per_original_coefficient_before_palettes': code_precision_ceiling,
                  'parametric_expert_count_ledger': parametric,
                  'gates': {k: v['passes_560mb_addressed_weight_gate'] for k, v in scenarios.items()},
                  'resource': runtime, 'script_sha256': sha(Path(__file__)),
                  'decision': 'reject_two_coefficient_U8_full_feature_LUT_training_as_complete_candidate'
                              if not any(v['passes_560mb_addressed_weight_gate'] for v in scenarios.values())
                              else 'requires_separately_frozen_quality_and_native_cost_screen',
                  'scope': 'Source-header descriptor arithmetic only. No learned palettes, transformed weights, physical DRAM, cache residency, native latency, quality or useful added experts. 40GB/s is a yardstick, not a measured rate bound.'}
        write_new(out, result)
        assert out.stat().st_size < 2_000_000
        print(json.dumps({k: result[k] for k in ('scenarios', 'gates', 'decision', 'resource',
                         'encoded_model_payload_bytes', 'diagnostic_zero_head_zero_palette_bound_bytes_per_token',
                         'diagnostic_max_code_bits_per_original_coefficient_before_palettes')}, ensure_ascii=True))
    except BaseException as error:
        write_new(out.with_suffix('.failure.json'), {'stage': stage, 'error': repr(error),
                  'seconds': time.monotonic() - start, 'script_sha256': sha(Path(__file__))})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    run(args.out)
