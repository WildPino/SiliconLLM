"""Bounded original Ling-mini immutable headers and source-specific active ledger."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time

import psutil
import requests
import meth324_switch_reference as M

MODEL = 'inclusionAI/Ling-mini-2.0'
REVISION = 'a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab'
PROTOCOL = M.DOC / 'METH_407_LING_MINI_SOURCE_HEADERS_PROTOCOL_20261004.md'
OUT = M.ROOT / 'results/native_expert_scaling/meth407_ling_mini_source_headers'
DTYPE_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I64': 8, 'I32': 4, 'I16': 2, 'U8': 1, 'I8': 1, 'BOOL': 1}
LAYER = re.compile(r'^model\.layers\.(\d+)\.')
EXPERT = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
PACKED = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)(?:\.weight)?$')


def analyze(config, tensors):
    d, h, n, k, layers = (config[v] for v in ('hidden_size', 'moe_intermediate_size', 'num_experts', 'num_experts_per_tok', 'num_hidden_layers'))
    assert (d, h, n, k, layers) == (2048, 512, 256, 8, 20)
    assert config['first_k_dense_replace'] == 1 and config['num_nextn_predict_layers'] == 0
    assert config['num_shared_experts'] == 1 and config['moe_shared_expert_intermediate_size'] == 512
    main, extra, banks, categories = {}, {}, defaultdict(dict), defaultdict(list)
    for name, row in tensors.items():
        layer = LAYER.match(name)
        if 'mtp' in name.split('.') or (layer and int(layer[1]) >= layers): extra[name] = row; continue
        main[name] = row
        if layer and '.mlp.experts.' in name:
            banks[int(layer[1])][name] = row; categories['routed'].append(row)
        elif '.mlp.gate.' in name or '.shared_expert_gate.' in name: categories['router_f32'].append(row)
        elif name == 'model.word_embeddings.weight': categories['embedding'].append(row)
        elif name == 'lm_head.weight': categories['head'].append(row)
        elif len(row['shape']) == 2: categories['core_matrices'].append(row)
        else: categories['other_controls_f32'].append(row)
    assert sorted(banks) == list(range(1, layers))
    bank_rows = []
    for layer, rows in sorted(banks.items()):
        packed = [v for name, v in rows.items() if PACKED.fullmatch(name)]
        if packed:
            assert len(rows) == 2 and {tuple(v['shape']) for v in rows.values()} == {(n, 2*h, d), (n, d, h)}
            namespace = 'packed_gate_up_and_down'
        else:
            groups = defaultdict(dict)
            for name, row in rows.items():
                match = EXPERT.fullmatch(name); assert match, ('unclassified_expert', name)
                groups[int(match[2])][match[3]] = row
            assert sorted(groups) == list(range(n)) and len(rows) == 3*n
            for group in groups.values():
                assert set(group) == {'gate_proj', 'up_proj', 'down_proj'}
                assert group['gate_proj']['shape'] == group['up_proj']['shape'] == [h, d] and group['down_proj']['shape'] == [d, h]
            namespace = 'individual_gate_up_down'
        total = sum(v['elements'] for v in rows.values()); assert total == 3*n*d*h
        bank_rows.append({'layer': layer, 'namespace': namespace, 'source_expert_slots': n, 'source_routed_coefficients': total,
                          'decode_selected_parent_count': k, 'decode_active_routed_coefficients': 3*k*d*h})
    assert len(categories['embedding']) == len(categories['head']) == 1
    assert categories['embedding'][0]['shape'] == categories['head'][0]['shape'] == [config['vocab_size'], d]
    assert len(categories['router_f32']) == 2*(layers-1)
    routed_layers = len(banks)
    for layer in range(layers):
        qkv = tensors[f'model.layers.{layer}.attention.query_key_value.weight']; dense = tensors[f'model.layers.{layer}.attention.dense.weight']
        assert qkv['shape'] == [3072,2048] and dense['shape'] == [2048,2048]
    for layer in range(1,layers):
        assert tensors[f'model.layers.{layer}.mlp.gate.weight']['shape'] == [256,2048]
        assert tensors[f'model.layers.{layer}.mlp.gate.expert_bias']['shape'] == [256]
        for kind, shape in (('gate_proj',[512,2048]),('up_proj',[512,2048]),('down_proj',[2048,512])):
            assert tensors[f'model.layers.{layer}.mlp.shared_experts.{kind}.weight']['shape'] == shape
    for kind, shape in (('gate_proj',[5120,2048]),('up_proj',[5120,2048]),('down_proj',[2048,5120])):
        assert tensors[f'model.layers.0.mlp.{kind}.weight']['shape'] == shape
    router = sum(v['elements'] for v in categories['router_f32'])
    core = sum(v['elements'] for v in categories['core_matrices'])
    controls = sum(v['elements'] for v in categories['other_controls_f32'])
    head = categories['head'][0]['elements']; routed = sum(v['decode_active_routed_coefficients'] for v in bank_rows)
    total_main = sum(v['elements'] for v in main.values()); total_extra = sum(v['elements'] for v in extra.values())
    # Same full-width two-U8-indices/eight-weights proposal317. Counts are
    # addressed descriptors, not physical traffic, kernel speed or quality.
    assert all(v['elements'] % 4 == 0 and v['shape'][1] % 8 == 0 for v in categories['core_matrices'])
    coded = core + routed; core_scales = sum(v['shape'][0]*4 for v in categories['core_matrices'])
    scales = core_scales + routed_layers*k*(2*h+d)*4
    palettes = len(categories['core_matrices'])*4096 + routed_layers*k*3*4096
    assert d % 256 == 0
    head_q6 = head//256*210
    ledger = {'full_width_integer_coefficients': coded, 'encoded_two_indices_per8_code_bytes': coded//4,
              'row_scale_bytes': scales, 'palette_bytes': palettes, 'head_Q6_K_bytes': head_q6,
              'all_flat_router_F32_bytes': router*4, 'other_control_F32_bytes': controls*4, 'one_embedding_BF16_row_bytes': d*2}
    ledger['addressed_weight_descriptor_bytes'] = sum(v for key, v in ledger.items() if key.endswith('_bytes'))
    ledger['under560MB_descriptor'] = ledger['addressed_weight_descriptor_bytes'] <= 560000000
    stored_coded = core + sum(v['elements'] for v in categories['routed'])
    stored = {'encoded_matrix_code_bytes': stored_coded//4, 'row_scale_bytes': core_scales+routed_layers*n*(2*h+d)*4,
              'palette_bytes': len(categories['core_matrices'])*4096+routed_layers*n*3*4096, 'head_Q6_K_bytes': head_q6,
              'router_and_control_F32_bytes': (router+controls)*4, 'embedding_BF16_bytes': categories['embedding'][0]['elements']*2}
    stored['total_bytes'] = sum(stored.values())
    return {'main_tensor_name_parameter_count': total_main, 'extra_tensor_name_parameter_count': total_extra,
            'all_tensor_name_parameter_count': total_main+total_extra, 'main_category_parameter_counts': {c:sum(v['elements'] for v in rows) for c,rows in categories.items()},
            'extra_tensors': extra, 'banks': bank_rows, 'full_width_additive_decode_ledger': ledger, 'hypothetical_main_storage': stored,
            'source_uniqueness_and_effective_usefulness_verified': False, 'optional_extra_modules_removed_from_standard_decode_ledger_only': True,
            'main_active_parameters_including_norms_router_head_and_lookup_row': core+routed+router+controls+head+d}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start = time.monotonic(); peak = 0; stage = 'bindings'
    result = {'experiment': 'METH-407-Ling-mini-actual-immutable-headers-source-cost', 'requests': [], 'shards': []}
    bodies = 0

    def guard():
        nonlocal peak
        peak = max(peak, psutil.Process().memory_info().rss)
        assert peak <= 1 << 30 and time.monotonic()-start <= 600 and bodies <= 64 << 20, 'metadata_10min_1GiB_64MiB'

    def get(url, maximum, byte_range=None, total=None):
        nonlocal bodies
        guard(); assert len(result['requests']) < 250
        headers = {'Accept-Encoding': 'identity'}
        if byte_range: headers['Range'] = f'bytes={byte_range[0]}-{byte_range[1]}'
        entry = {'url': url, 'range': byte_range}; result['requests'].append(entry)
        with requests.get(url, headers=headers, stream=True, timeout=(15,30)) as response:
            entry['status'] = response.status_code; response.raise_for_status()
            if byte_range:
                assert response.status_code == 206, ('range_ignored', response.status_code)
                assert response.headers.get('Content-Range') == f'bytes {byte_range[0]}-{byte_range[1]}/{total}', ('range_mismatch', response.headers.get('Content-Range'))
            data = bytearray()
            for block in response.iter_content(1 << 16):
                bodies += len(block); data.extend(block); guard(); assert len(data) <= maximum, 'response_body_limit'
            raw = bytes(data)
            if byte_range: assert len(raw) == byte_range[1]-byte_range[0]+1
            entry.update({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            return raw

    try:
        for p in (Path(__file__), PROTOCOL, Path(M.__file__)): M.committed(p)
        result.update({'controller_sha256': M.digest(__file__), 'protocol_sha256': M.digest(PROTOCOL),
                       'model': MODEL, 'revision': REVISION, 'source_weight_value_payload_downloaded_bytes': 0})
        OUT.mkdir(parents=True); stage = 'api_index_config'
        api_raw = get(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true', 4 << 20)
        (OUT/'api.json').write_bytes(api_raw); api = json.loads(api_raw); assert api['sha'] == REVISION
        siblings = {v['rfilename']: v for v in api['siblings']}
        root = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/'
        cfg_raw = get(root+'config.json', 1 << 20)
        (OUT/'config.json').write_bytes(cfg_raw); config = json.loads(cfg_raw)
        result['config_sha256'] = hashlib.sha256(cfg_raw).hexdigest()
        result['custom_source'] = []
        for name in ('configuration_bailing_moe_v2.py','modeling_bailing_moe_v2.py'):
            raw = get(root+name, 4 << 20); (OUT/name).write_bytes(raw)
            assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == siblings[name]['blobId']
            result['custom_source'].append({'name':name,'sha256':hashlib.sha256(raw).hexdigest(),'path':str(OUT/name),'executed':False})
        index_raw = get(root+'model.safetensors.index.json', 16 << 20)
        (OUT/'model.safetensors.index.json').write_bytes(index_raw); index = json.loads(index_raw)
        for name, raw in (('config.json', cfg_raw), ('model.safetensors.index.json', index_raw)):
            assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == siblings[name]['blobId']
        names = sorted(set(index['weight_map'].values())); assert names and len(names) <= 100
        tensors = {}; dtype_parameters = Counter(); archives = 0
        for number, name in enumerate(names):
            stage = name; source = siblings[name]; lfs = source['lfs']; size = lfs['size']; archives += size
            sha = lfs.get('sha256') or lfs.get('oid'); assert isinstance(sha,str) and re.fullmatch('[0-9a-f]{64}',sha)
            prefix = get(root+name+'?download=true&metadata_range=0-7', 8, (0,7), size)
            length = struct.unpack('<Q',prefix)[0]; assert 2 <= length <= 4 << 20 and 8+length < size
            raw = get(root+name+f'?download=true&metadata_range=8-{7+length}', length, (8,7+length), size)
            saved = OUT/(name+'.header.json'); saved.write_bytes(raw); header = json.loads(raw)
            items = {k:v for k,v in header.items() if k != '__metadata__'}
            assert set(items) == {k for k,v in index['weight_map'].items() if v == name}
            intervals = []
            for tensor, item in items.items():
                assert tensor not in tensors and item['dtype'] in DTYPE_BYTES
                shape = item['shape']; assert all(isinstance(v,int) and v>=0 for v in shape)
                elements = math.prod(shape); a,b = item['data_offsets']; assert 0<=a<=b<=size-8-length
                assert b-a == elements*DTYPE_BYTES[item['dtype']]
                intervals.append((a,b)); dtype_parameters[item['dtype']] += elements
                tensors[tensor] = {'shape':shape, 'dtype':item['dtype'], 'elements':elements, 'shard':name, 'data_offsets':[a,b]}
            position = 0
            for a,b in sorted(intervals): assert a == position; position = b
            assert position+8+length == size
            result['shards'].append({'name':name, 'declared_LFS_size':size, 'declared_LFS_sha256':sha, 'header_bytes':length,
                                      'header_sha256':hashlib.sha256(raw).hexdigest(), 'header_path':str(saved), 'tensors':len(items), 'all_offsets_dtype_sizes_exact':True})
            guard(); print(json.dumps({'shards_done':number+1, 'shards_total':len(names), 'tensors':len(tensors), 'response_bytes':bodies}),flush=True)
        assert set(tensors) == set(index['weight_map'])
        count_bytes = sum(v*DTYPE_BYTES[k] for k,v in dtype_parameters.items()); assert count_bytes == index['metadata']['total_size']
        result.update({'tensors':tensors, 'source_shard_bytes':archives, 'source_tensor_value_bytes':count_bytes,
                       'source_parameters_by_dtype':dict(dtype_parameters), 'source_config':config, 'index_sha256':hashlib.sha256(index_raw).hexdigest()})
        stage = 'main_vs_extra_active_analysis'; analysis = analyze(config,tensors); result['analysis'] = analysis
        assert sum(dtype_parameters.values()) == analysis['all_tensor_name_parameter_count']
        assert analysis['extra_tensor_name_parameter_count'] == 0
        result['parameter_reconciliation'] = {'all_headers':sum(dtype_parameters.values()),
            'main':analysis['main_tensor_name_parameter_count'], 'extra':analysis['extra_tensor_name_parameter_count']}
        result['hardware'] = {'RAM_total_bytes':psutil.virtual_memory().total, 'RAM_available_bytes':psutil.virtual_memory().available}
        result['full_resident_original_BF16_fits_RAM'] = count_bytes <= psutil.virtual_memory().total
        result['full_resident_original_FP32_fits_RAM'] = 4*sum(dtype_parameters.values()) <= psutil.virtual_memory().total
        result['gates'] = {'config_git_blob_bound_to_immutable_revision':True, 'index_git_blob_bound':True, 'ALL_headers_index_offsets_dtype_sizes_exact':True,
                           'ALL19_banks256_source_slot_shapes':True, 'all_parameter_counts_reconciled':True, 'custom_source_git_blobs_exact_not_executed':True}
        assert all(result['gates'].values())
        result['decision'] = 'unchanged_full_width_additive_active_geometry_closed_before_acquisition' if not analysis['full_width_additive_decode_ledger']['under560MB_descriptor'] else 'eligible_only_for_separate_native_active_cost_and_reference_operator_protocol'
        result['resource'] = {'seconds':time.monotonic()-start, 'maximum_checked_rss_bytes':peak, 'response_body_bytes':bodies, 'http_requests':len(result['requests'])}
        result['scope'] = 'Original immutable Ling-mini API/config/index/custom code/ALL actual safetensors headers only; custom code stored, not executed. No source tensor values, unique function/usefulness, original inference/reference parity, weight acquisition/export/native implementation, quality/rate or physical DRAM measured. No MTP declared;19 real banks256/top8 plus shared512/dense first layer. This is another-family metadata applicability, not useful>256/10x. Additive full-width byte ledger hypothetical untrained representation, includes scales/palettes/F32 controls/head; no roofline/cache inference or general donor rejection.'
        guard(); M.write(args.out,result); print(json.dumps({'sha256':M.digest(args.out), 'decision':result['decision'], 'reconciliation':result['parameter_reconciliation'], 'ledger':analysis['full_width_additive_decode_ledger'], 'resource':result['resource']}),flush=True)
    except BaseException as error:
        result.update({'failure_stage':stage, 'error':repr(error), 'seconds':time.monotonic()-start, 'response_body_bytes':bodies})
        M.write(args.out.with_suffix('.failure.json'),result); raise


if __name__ == '__main__': main()
