"""Frozen model-free manifest/source/count admission for exact A16 fanout."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
BASE = ROOT / 'benchmarks/native_expert_scaling'
PROTOCOL = DOC / 'METH_460_SWITCH_COMMON_INPUT_ADMISSION_PROTOCOL_20261005.md'
RAW = DOC / 'meth460_switch_common_input_admission_result.json'
PARENT = DOC / 'meth458_switch_matched_whole_cost_result.json'
PARENT_SHA = '3762fd4e6c86a6ea7e64f54d4350cc2bd91761f2f856a06bb0a0a442f12d285f'
RETENTION_SHA = '65ec6408bc1de79d0a3bad869b57de7298285901c5c91ca502330250ea9ea8b0'
CONFIG_KEYS = ('d', 'ff', 'heads', 'dk', 'enc', 'dec', 'n', 'capacity', 'vocab', 'buckets', 'distance', 'encstep', 'decstep')
PATTERNS = {
    'encoder_self_QKV': 'mv_batch(b->self.q,x,q,c.d,c.d,tokens);mv_batch(b->self.k,x,k,c.d,c.d,tokens);mv_batch(b->self.v,x,v,c.d,c.d,tokens);',
    'cross_KV': 'mv_batch(m->dec[l].cross.k,encoder,a->crossk,c.d,c.d,tokens);mv_batch(m->dec[l].cross.v,encoder,a->crossv,c.d,c.d,tokens);',
    'decoder_self_QKV': 'mv(b->self.q,x,q,c.d,c.d);mv(b->self.k,x,a->selfk+(size_t)position*c.d,c.d,c.d);mv(b->self.v,x,a->selfv+(size_t)position*c.d,c.d,c.d);',
}


def committed(path):
    path = Path(path); rel = path.resolve().relative_to(ROOT).as_posix()
    assert path.read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', f'--path={rel}', f'HEAD:{rel}'], cwd=ROOT), ('physical_HEAD', rel)


def write_new(path, data):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode('utf-8'))


def jobs():
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}; preserved = []
    for process in psutil.process_iter(['name', 'cmdline']):
        if process.pid in own:
            continue
        name = (process.info['name'] or '').lower(); argv = process.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            preserved.append(process.pid); continue
        assert not (name.startswith('python') or (name.startswith('meth') and name.endswith('.exe'))), ('concurrent_job', process.pid, name)
    return preserved


def read_manifest(path, artifact):
    data = Path(path).read_bytes(); assert data[:8] == b'SWI8A001'; offset = 8
    header = struct.unpack_from('<13IfII', data, offset); offset += 64
    config = dict(zip(CONFIG_KEYS, header[:13], strict=True)); config['epsilon'] = header[13]
    files, tensor_count = header[14:]; assert files == 1 and 0 < tensor_count < 20000
    def string(bound):
        nonlocal offset
        size = struct.unpack_from('<I', data, offset)[0]; offset += 4
        assert 0 < size < bound and offset + size <= len(data)
        value = data[offset:offset + size].decode('utf-8'); offset += size; return value
    payload = string(1024); assert Path(payload).resolve() == Path(artifact['payload']).resolve()
    tensors = {}; names = []
    for _ in range(tensor_count):
        name = string(192); entry = struct.unpack_from('<5I3Q', data, offset); offset += 44
        file_index, dims, rows, cols, encoding, code_offset, scale_offset, elements = entry
        assert file_index == 0 and dims in (1, 2) and rows > 0 and cols > 0 and elements == rows * cols and encoding in (0, 1)
        assert code_offset % 64 == 0 and code_offset + elements * (1 if encoding else 4) <= artifact['bytes']
        if encoding:
            assert dims == 2 and scale_offset % 64 == 0 and scale_offset + 4 * rows <= artifact['bytes']
        else:
            assert scale_offset == 0
        assert name not in tensors; tensors[name] = {'dims': dims, 'rows': rows, 'cols': cols, 'encoding': encoding, 'code_offset': code_offset, 'scale_offset': scale_offset}
        names.append(name)
    assert offset == len(data) and names == sorted(names)
    return config, tensors, {'bytes': len(data), 'tensor_count': tensor_count, 'bounded_EOF_exact': True, 'payload_path': payload}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert args.out.resolve() == RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and sys.flags.optimize == 0
    start = time.monotonic(); peak = hashed = 0; stage = 'bindings'
    result = {'experiment': 'METH460-model-free-exact-common-input-A16-fanout-admission', 'helper_sha256': {}, 'sources': {}, 'catalog': []}
    def guard():
        nonlocal peak
        memory = psutil.Process().memory_info(); peak = max(peak, memory.rss, getattr(memory, 'peak_wset', 0))
        assert peak <= 256 << 20 and time.monotonic() - start <= 60, 'model_free_256MiB_60seconds'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(1 << 20):
                h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    try:
        result['git_head_before_execution'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'], cwd=ROOT)
        for path in (Path(__file__), PROTOCOL, PARENT, DOC / 'RETENTION_458_20261005.json'):
            committed(path); result['helper_sha256'][str(path)] = sha(path)
        fault_path = DOC / 'meth459_switch_common_input_admission_result.failure.json'; committed(fault_path)
        assert sha(fault_path) == '16cdb4816a23b03aa63fd0c70148db10cc23c377872447774e621d193020a835'
        fault = json.loads(fault_path.read_text(encoding='utf-8')); result['helper_sha256'][str(fault_path)] = sha(fault_path)
        assert not fault['sources'] and len(fault['catalog']) == 10
        for rel in ('benchmarks/native_expert_scaling/meth459_switch_common_input_admission.py', 'docs/research/NATIVE_EXPERT_SCALING_20260925/METH_459_SWITCH_COMMON_INPUT_ADMISSION_PROTOCOL_20261005.md'):
            path = ROOT / rel; committed(path); assert sha(path) == fault['helper_sha256'][str(path)]; result['helper_sha256'][str(path)] = sha(path)
        assert result['helper_sha256'][str(PARENT)] == PARENT_SHA and result['helper_sha256'][str(DOC / 'RETENTION_458_20261005.json')] == RETENTION_SHA
        parent = json.loads(PARENT.read_text(encoding='utf-8')); retention = json.loads((DOC / 'RETENTION_458_20261005.json').read_text(encoding='utf-8'))
        assert all(parent['apparatus_gates'].values()) and all(retention['gates'].values()) and retention['raw']['sha256'] == PARENT_SHA
        result['preserved_daemons_before'] = jobs()
        qualified = {}
        for name in ('meth374_switch_physical_workers.c', 'meth388_switch_three_workers.c'):
            path = BASE / name; committed(path); observed = sha(path); assert observed == parent['helper_sha256'][str(path)]
            result['helper_sha256'][str(path)] = observed; qualified[name] = path.read_text(encoding='utf-8')
        old = qualified['meth374_switch_physical_workers.c']; new = qualified['meth388_switch_three_workers.c']
        reverse = new.replace('meth388_switch_three_workers', 'meth374_switch_physical_workers').replace('meth388_contract_entry', 'meth374_contract_entry').replace('meth388_forced_entry', 'meth374_forced_entry').replace('meth388_switch_thread_binding.h', 'meth374_switch_thread_binding.h').replace('threads==1||threads==3||threads==6', 'threads==1||threads==6')
        assert reverse == old
        for name, text in qualified.items():
            compact = re.sub(r'\s+', '', text)
            for label, pattern in PATTERNS.items():
                assert compact.count(pattern) == 1, ('qualified_callsite', name, label)
            assert 'floatscale=head_activation_codes(x,codes,cols);' in compact
            assert 'scales[token]=head_activation_codes(x+(size_t)token*cols,codes+(size_t)token*cols,cols);' in compact
            assert 'for(introw=0;row<rows;row++)for(inttoken=0;token<tokens;token++){' in compact
            assert 'y[(size_t)token*rows+row]=(float)(((double)value*(double)w.scales[row])*(double)scales[token]);' in compact
            assert 'fegetround()==FE_TONEAREST' in compact and 'longcode=lrintf(divided);' in compact
            assert 'if(code>32767)code=32767;if(code<-32767)code=-32767;' in compact
        engine = ROOT / 'benchmarks/phase60/engine.c'; committed(engine); assert sha(engine) == parent['preserved_engine_sha256']; result['helper_sha256'][str(engine)] = parent['preserved_engine_sha256']
        engine_text = engine.read_text(encoding='utf-8'); engine_compact = re.sub(r'\s+', '', engine_text)
        assert 'matvec(swa.qkv,xn,xz,3*D,D);' in engine_compact and 'mv_K(swa.qkv,3*D,D,' in engine_compact
        paths = subprocess.check_output(['git', 'ls-files', '--', 'benchmarks/native_expert_scaling/*.c', 'benchmarks/native_expert_scaling/*.h', 'benchmarks/phase60/engine.c'], cwd=ROOT, text=True).splitlines()
        for rel in sorted(paths):
            path = ROOT / rel; physical = path.read_bytes(); blob = subprocess.check_output(['git', 'show', f'HEAD:{rel}'], cwd=ROOT)
            filtered = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', f'--path={rel}', f'HEAD:{rel}'], cwd=ROOT)
            assert physical.decode('utf-8').replace('\r\n', '\n') == blob.decode('utf-8').replace('\r\n', '\n'), ('catalog_canonical_text_HEAD', rel)
            physical_identity = {'physical_equals_filtered_HEAD': physical == filtered, 'canonical_LF_text_exact_HEAD': True,
                                 'raw_HEAD_blob_sha256': hashlib.sha256(blob).hexdigest(), 'filtered_HEAD_sha256': hashlib.sha256(filtered).hexdigest()}
            digest = sha(path)
            text = path.read_text(encoding='utf-8'); hits = [{'line': index + 1, 'text': line[:600]} for index, line in enumerate(text.splitlines()) if re.search(r'\b(?:[A-Za-z0-9_]*fanout[A-Za-z0-9_]*|[A-Za-z0-9_]*qkv[A-Za-z0-9_]*|mv_K|integer_dot4)\b', line, flags=re.IGNORECASE)]
            result['catalog'].append({'path': rel, 'bytes': path.stat().st_size, 'sha256': digest, 'physical_identity': physical_identity, 'name_search_hits': hits}); guard()
        result['existing_paths'] = {'phase60_SWA_fp32_QKV_joint': True, 'phase60_layer_major_multiple_queries': True,
                                    '365_four_queries_integer_dot4': True, 'qualified374_388_common_input_A16_fanout': False,
                                    'absence_scope': 'qualified source callsites and mv/mv_batch bodies only; name catalog is not proof of global repository semantic absence'}
        assert any('integer_dot4' in hit['text'] for item in result['catalog'] if item['path'].endswith('meth365_switch_encoder_batches.c') for hit in item['name_search_hits'])
        stage = 'manifest_geometry_and_counts'
        for n, source in parent['sources'].items():
            assert source['profiler_admissibility']['admitted'] and all(source['profiler_admissibility']['gates'].values())
            artifact = source['artifact']; path = Path(artifact['manifest']); assert sha(path) == artifact['manifest_sha256']
            config, tensors, manifest_info = read_manifest(path, artifact)
            payload = Path(artifact['payload']); payload_stat = [payload.stat().st_size, payload.stat().st_mtime_ns]
            assert payload_stat == source['artifact_stat_before'] and config['n'] == int(n)
            d, ff, enc, dec = (config[key] for key in ('d', 'ff', 'enc', 'dec'))
            assert d == 768 and ff == 3072 and enc == dec == 12 and config['heads'] * config['dk'] == d
            fanouts = []
            for stack, layers in (('encoder', enc), ('decoder', dec)):
                for layer in range(layers):
                    names = [f'{stack}.block.{layer}.layer.0.SelfAttention.{organ}.weight' for organ in ('q', 'k', 'v')]
                    fanouts.append({'kind': stack + '_self_QKV', 'layer': layer, 'input': 'same_selfnorm_x', 'output_layout': 'separate Q/K/V original row order; batch token-major', 'tensors': {name: tensors[name] for name in names}})
                    if stack == 'decoder':
                        names = [f'decoder.block.{layer}.layer.1.EncDecAttention.{organ}.weight' for organ in ('k', 'v')]
                        fanouts.append({'kind': 'cross_KV', 'layer': layer, 'input': 'same_final_encoder_vector_per_source_token', 'output_layout': 'separate cache.crossk/cache.crossv token-major', 'tensors': {name: tensors[name] for name in names}})
                    other = [f'{stack}.block.{layer}.layer.0.SelfAttention.o.weight']
                    if stack == 'decoder':
                        other += [f'decoder.block.{layer}.layer.1.EncDecAttention.{organ}.weight' for organ in ('q', 'o')]
                    assert all(tensors[name]['encoding'] == 1 and tensors[name]['rows'] == tensors[name]['cols'] == d for name in other)
            selected = [entry for group in fanouts for entry in group['tensors'].values()]
            assert all(entry['dims'] == 2 and entry['rows'] == entry['cols'] == d and entry['encoding'] == 1 for entry in selected)
            assert len(selected) == len({entry['code_offset'] for entry in selected}) == 3 * enc + 5 * dec
            dense_layers = {}
            for stack, layers, step in (('encoder', enc, config['encstep']), ('decoder', dec, config['decstep'])):
                dense_layers[stack] = [layer for layer in range(layers) if not (step > 0 and (layer % step == 1 or step == 1))]
                for layer in dense_layers[stack]:
                    prefix = f'{stack}.block.{layer}.layer.{2 if stack == "decoder" else 1}.mlp.'
                    for organ, rows, cols in (('wi', ff, d), ('wo', d, ff)):
                        entry = tensors[prefix + organ + '.weight']; assert entry['rows'] == rows and entry['cols'] == cols and entry['encoding'] == 1
            a, b = len(dense_layers['encoder']), len(dense_layers['decoder']); cases = []; total_generated = 0
            for index, case in enumerate(source['cases']):
                assert case['book'] == index // 4 and case['case'] == index % 4
                rows = [row for mode in case['modes'].values() for row in mode]
                assert len(rows) == 8 and set(case['modes']) == {'0', '1'} and all([row['repetition'] for row in mode] == [-1, 0, 1, 2] for mode in case['modes'].values())
                s = rows[0]['source_tokens']; t = case['generated_tokens']; assert s == 29 and all(row['source_tokens'] == s and row['actual_generated_tokens'] == t for row in rows)
                expected = [
                    {'calls': (4 * enc + 2 * a + 2 * dec) * s, 'code_bytes': ((4 * enc + 2 * dec) * d * d + 2 * a * d * ff) * s,
                     'scale_bytes': ((4 * enc + 2 * dec) * d + a * (ff + d)) * 4 * s, 'f32_bytes': 0},
                    {'calls': (6 * dec + 2 * b) * t, 'code_bytes': (6 * dec * d * d + 2 * b * d * ff) * t,
                     'scale_bytes': (6 * dec * d + b * (ff + d)) * 4 * t, 'f32_bytes': 0}]
                for row in rows:
                    for phase in range(2):
                        assert all(row['counters'][phase][0][key] == value for key, value in expected[phase].items())
                groups = [
                    {'kind': 'encoder_self_QKV', 'invocations': enc, 'queries_per_invocation': s, 'maps': 3},
                    {'kind': 'cross_KV', 'invocations': dec, 'queries_per_invocation': s, 'maps': 2},
                    {'kind': 'decoder_self_QKV', 'invocations': dec * t, 'queries_per_invocation': 1, 'maps': 3}]
                for group in groups:
                    count, queries, maps = (group[key] for key in ('invocations', 'queries_per_invocation', 'maps'))
                    group.update(quantizer_evaluations_before=count * queries * maps, quantizer_evaluations_after=count * queries,
                                 quantizer_evaluations_removed=count * queries * (maps - 1), parallel_regions_before=count * maps,
                                 parallel_regions_after=count, parallel_regions_removed=count * (maps - 1),
                                 unchanged_integer_products=count * queries * maps * d * d,
                                 unchanged_logical_weight_code_bytes=count * queries * maps * d * d)
                case_summary = {'book': case['book'], 'case': case['case'], 'healthy_accepted': case['healthy_accepted'], 'source_tokens': s, 'generated_tokens': t,
                                'groups': groups, 'logical_dense_counters_rederived': expected,
                                'ALL_dense_A16_evaluations_before': expected[0]['calls'] + expected[1]['calls'],
                                'ALL_dense_parallel_regions_before': (4 * enc + 2 * a + 2 * dec) + (6 * dec + 2 * b) * t}
                cases.append(case_summary); total_generated += t; guard()
            aggregate = {key: sum(group[key] for case in cases for group in case['groups']) for key in
                         ('quantizer_evaluations_before', 'quantizer_evaluations_after', 'quantizer_evaluations_removed', 'parallel_regions_before', 'parallel_regions_after', 'parallel_regions_removed', 'unchanged_integer_products', 'unchanged_logical_weight_code_bytes')}
            aggregate.update(ALL_dense_A16_evaluations_before=sum(case['ALL_dense_A16_evaluations_before'] for case in cases),
                             ALL_dense_parallel_regions_before=sum(case['ALL_dense_parallel_regions_before'] for case in cases),
                             ALL_cases=len(cases), ALL_generated_tokens_including_rejected=total_generated)
            aggregate['fraction_ALL_dense_A16_evaluations_removable'] = aggregate['quantizer_evaluations_removed'] / aggregate['ALL_dense_A16_evaluations_before']
            aggregate['fraction_ALL_dense_parallel_regions_removable'] = aggregate['parallel_regions_removed'] / aggregate['ALL_dense_parallel_regions_before']
            aggregate['candidate_integer_product_reduction'] = 0; aggregate['candidate_weight_storage_reduction'] = 0; aggregate['measured_speedup'] = None
            assert len(cases) == 96 and payload_stat == [payload.stat().st_size, payload.stat().st_mtime_ns]
            result['sources'][n] = {'artifact': artifact, 'payload_stat': payload_stat, 'full_payload_hash_fresh_this_experiment': False,
                                    'binary_manifest_sha256': artifact['manifest_sha256'], 'config': config, 'manifest_info': manifest_info,
                                    'fanouts': fanouts, 'dense_layers': dense_layers, 'cases': cases, 'aggregate': aggregate,
                                    'inherited_dense_fraction': source['admitted_decomposition']['whole_cost_fractions']['dense_control_matrices'],
                                    'interpretation': 'exact reusable quantizer/parallel-region sharing is unimplemented in qualified path; milliseconds/DRAM/whole speed benefit unmeasured'}
        stage = 'integer_bound_and_contract'
        weight_max, activation_max = 128, 32767
        bounds = {'I8_absolute_including_minus128': weight_max, 'A16_absolute_excluding_minus32768': activation_max,
                  'pair_MADD_absolute_bound': 2 * weight_max * activation_max,
                  'I32_lane_absolute_bound_cols768': (768 // 8) * weight_max * activation_max,
                  'I32_lane_absolute_bound_cols4096': (4096 // 8) * weight_max * activation_max,
                  'I64_dot_absolute_bound_cols4096': 4096 * weight_max * activation_max}
        assert bounds['pair_MADD_absolute_bound'] < 2 ** 31 and bounds['I32_lane_absolute_bound_cols4096'] < 2 ** 31 and bounds['I64_dot_absolute_bound_cols4096'] < 2 ** 63
        result['integer_bounds'] = bounds
        result['exactness_contract'] = {'identical_input_pointer_values_length_quantizer_and_FE_TONEAREST': True,
            'one_parent_A16_absmax_scale_F32_division_RNE_clamp_shared_const_codes': True,
            'each_original_matrix_keeps_separate_integer_sum_and_F64_row_scale_then_activation_scale_then_F32': True,
            'output_nonalias_and_disjoint_original_offsets_required': True, 'no_FP_sum_reassociation_no_reordered_layer_norm_ReLU_softmax_routes': True,
            'negative_extrema_safe_with_original_integer_dot': True,
            'not_a_new_compiled_numerical_validation': True}
        result['preserved_daemons_after'] = jobs(); guard()
        result['gates'] = {'fresh_frozen_sources_engine_manifest_and_parent_metadata_exact': True,
            'qualified374_388_separate_callsite_redundancy_and_quantizer_contract_exact': True,
            'all36_legal_fanouts_per_source_shapes_I8_scales_offsets_unique': True,
            'ALL1536_inherited_dense_logical_counter_rows_rederived_exact': True,
            'all_cases_including_rejected_charged_in_operation_counts': True,
            'integer_pair_lane_dot_extrema_safe': True, 'existing_SWA_and_multiple_query_paths_distinguished': True,
            'zero_native_compile_run_weight_read_fit_download_GPU_engine_edits': True}
        result['resource'] = {'main_seconds_excluding_imports': time.monotonic() - start, 'peak_working_set_bytes': peak, 'bytes_hashed': hashed}
        result['decision'] = 'admit_ONE_new_exact_A16_common_input_fanout_whole_cost_inquiry_with_future_fixed_quality_cost_gates'
        result['scope'] = 'Model-free algebra and current-source/binary-manifest/count admission only. Existing458 profiles/outputs inherited, not rerun/freshly hashed; source payload values not read/hashed (size+mtime checked). No fresh numeric state/quality/speed/DRAM proof. All-case counts correspond one complete generation per original case, warm/repeats not multiplied into deployment counts. Legal sharing eliminates duplicate activation quantization/OpenMP regions only; coefficient reads/integer products/storage are unchanged, core dimensions untouched, n capacity unaffected. Generic joint QKV exists in phase60 fp32 SWA, token batching in365/layer-major helpers; no novelty/general-family equivalence claimed. f_dense is not quantizer-specific time. Need separately frozen native primal/whole comparison; full transfer/fresh donor quality/SAME50/useful-n/router mass/actualDRAM/families100B remain open.'
        write_new(RAW, result)
        print(json.dumps({'sha256': sha(RAW), 'gates': result['gates'], 'decision': result['decision'],
                          'sources': {n: source['aggregate'] for n, source in result['sources'].items()}, 'integer_bounds': bounds, 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update(stage=stage, error=repr(error), main_seconds_excluding_imports=time.monotonic() - start, peak_working_set_bytes=peak)
        write_new(RAW.with_suffix('.failure.json'), result)
        raise


if __name__ == '__main__':
    main()
