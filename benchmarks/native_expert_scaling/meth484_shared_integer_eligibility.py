"""ONE frozen model-free eligibility inquiry. No payload, GPU or solver imports."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
BASE = ROOT / 'benchmarks/native_expert_scaling'
RAW = DOC / 'meth484_shared_integer_eligibility_result.json'
BIND = DOC / 'meth484_source_binding.json'
KEYS = ('d', 'ff', 'heads', 'dk', 'enc', 'dec', 'n', 'capacity', 'vocab', 'buckets', 'distance', 'encstep', 'decstep')


def exclusive(path, record):
    data = (json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode()
    assert len(data) <= 12 << 20, 'metadata_output_bound'
    with Path(path).open('xb') as stream:
        stream.write(data)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def processes():
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}; preserved = []
    for p in psutil.process_iter(['name', 'cmdline']):
        if p.pid in own:
            continue
        name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            preserved.append(p.pid); continue
        assert not (name.startswith(('python', 'clang')) or (name.startswith('meth') and name.endswith('.exe'))), ('foreign_job', p.pid, name)
    return preserved


def descriptors(path, payload):
    # Separate byte cursor; every field is consumed and exact EOF is mandatory.
    import struct
    data = Path(path).read_bytes(); cursor = 8; assert data[:8] == b'SWI8A001'
    def take(fmt):
        nonlocal cursor
        size = struct.calcsize(fmt); assert cursor + size <= len(data)
        v = struct.unpack_from(fmt, data, cursor); cursor += size; return v
    def string(bound):
        nonlocal cursor
        size, = take('<I'); assert 0 < size < bound and cursor + size <= len(data)
        value = data[cursor:cursor + size].decode('utf-8'); cursor += size
        assert '\x00' not in value; return value
    h = take('<13IfII'); c = dict(zip(KEYS, h[:13], strict=True)); c['epsilon'] = h[13]
    assert h[14] == 1 and 0 < h[15] < 20000
    assert Path(string(1024)).resolve() == Path(payload['path']).resolve()
    assert 0 < c['d'] <= 1024 and 0 < c['ff'] <= 4096 and c['heads'] > 0 and c['heads'] * c['dk'] == c['d']
    assert 0 < c['enc'] <= 24 and 0 < c['dec'] <= 24 and c['n'] > 0 and c['n'] * c['d'] < 1 << 31
    assert c['capacity'] > 0 and 0 < c['vocab'] <= 65536 and c['buckets'] == 32 and c['distance'] == 128 and 0 < c['epsilon'] < 1
    records = {}; names = []
    for _ in range(h[15]):
        name = string(192); f, dims, m, d, encoding, code, scale, elements = take('<5I3Q')
        assert f == 0 and dims in (1, 2) and m > 0 and d > 0 and 0 < elements == m * d < 1 << 31
        assert encoding in (0, 1) and code % 64 == 0 and code + elements * (1 if encoding else 4) <= payload['bytes']
        if encoding:
            assert dims == 2 and d <= 4096 and scale % 64 == 0 and scale + 4 * m <= payload['bytes']
        else:
            assert scale == 0
        assert name not in records
        records[name] = {'name': name, 'file': f, 'dims': dims, 'rows': m, 'cols': d, 'encoding': encoding,
                         'code_offset': code, 'scale_offset': scale, 'elements': elements}
        names.append(name)
    assert cursor == len(data) and names == sorted(names)
    return c, records


def classify(c, records):
    assigned = set(); d, ff = c['d'], c['ff']
    def use(name, m, k, dims, encoding, category, batch=1):
        r = records[name]
        assert name not in assigned and (r['rows'], r['cols'], r['dims'], r['encoding']) == (m, k, dims, encoding), ('shape_codec', name)
        assigned.add(name); r.update(category=category, max_legal_queries_per_operator=batch)
    for name in ('shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'):
        use(name, c['vocab'], d, 2, 0, 'F32_embedding_alias')
    use('lm_head.weight', c['vocab'], d, 2, 1, 'shared_head')
    sparse = {}; dense = {}
    for stack in ('encoder', 'decoder'):
        use(stack + '.final_layer_norm.weight', d, 1, 1, 0, 'F32_control')
        use(stack + '.block.0.layer.0.SelfAttention.relative_attention_bias.weight', c['buckets'], c['heads'], 2, 0, 'F32_control')
        sparse[stack] = []; dense[stack] = []
        for l in range(c['enc' if stack == 'encoder' else 'dec']):
            prefix = f'{stack}.block.{l}.layer.'
            use(prefix + '0.layer_norm.weight', d, 1, 1, 0, 'F32_control')
            for organ in ('q', 'k', 'v', 'o'):
                use(prefix + '0.SelfAttention.' + organ + '.weight', d, d, 2, 1, 'shared_attention',
                    256 if stack == 'encoder' and organ != 'o' else 1)
            if stack == 'decoder':
                use(prefix + '1.layer_norm.weight', d, 1, 1, 0, 'F32_control')
                for organ in ('q', 'k', 'v', 'o'):
                    use(prefix + '1.EncDecAttention.' + organ + '.weight', d, d, 2, 1, 'shared_attention', 256 if organ in ('k', 'v') else 1)
            layer = 1 if stack == 'encoder' else 2; part = prefix + str(layer)
            use(part + '.layer_norm.weight', d, 1, 1, 0, 'F32_control')
            step = c['encstep' if stack == 'encoder' else 'decstep']
            is_sparse = step > 0 and (l % step == 1 or step == 1)
            (sparse if is_sparse else dense)[stack].append(l)
            if is_sparse:
                # Original writer stores router as F32. Never offload this control organ.
                use(part + '.mlp.router.classifier.weight', c['n'], d, 2, 0, 'CPU_F32_router')
                for e in range(c['n']):
                    ep = part + f'.mlp.experts.expert_{e}.'
                    use(ep + 'wi.weight', ff, d, 2, 1, 'CPU_expert')
                    use(ep + 'wo.weight', d, ff, 2, 1, 'CPU_expert')
            else:
                use(part + '.mlp.wi.weight', ff, d, 2, 1, 'shared_dense_FFN')
                use(part + '.mlp.wo.weight', d, ff, 2, 1, 'shared_dense_FFN')
    assert assigned == set(records), ('unclassified_tensors', sorted(set(records) - assigned))
    return sparse, dense


def byte_budgets(records, payload_bytes):
    components = {}; groups = {}; shared = []
    for r in records.values():
        blocks = [('value', r['code_offset'], r['elements'] * (1 if r['encoding'] else 4))]
        if r['encoding']:
            blocks.append(('row_scale', r['scale_offset'], 4 * r['rows']))
        for kind, offset, size in blocks:
            key = (offset, size)
            components.setdefault(key, []).append((r['name'], r['category'], kind))
        if r['category'].startswith('shared_'):
            m, d, q = r['rows'], r['cols'], r['max_legal_queries_per_operator']
            pm, pd = (m + 7) // 8 * 8, (d + 7) // 8 * 8
            b = {'padded_rows': pm, 'padded_cols': pd, 'GPU_weight_bytes': pm * pd,
                 'weight_padding_bytes': pm * pd - m * d,
                 'GPU_operator_scratch_bytes': 4 * q * (pd + 4 * pm),
                 'CPU_operator_arrays_bytes': q * (10 * pd + 28 * pm + 4),
                 'H2D_per_query_bytes': 4 * pd, 'D2H_per_query_bytes': 16 * pm,
                 'CPU_row_scales_bytes': 4 * m,
                 'I32_partial_bound': 16256 * pd, 'I64_reconstruction_intermediate_bound': 6291328 * pd,
                 'original_dot_bound': 128 * 32767 * d}
            assert b['I32_partial_bound'] < 1 << 31 and b['I64_reconstruction_intermediate_bound'] < 1 << 63 and b['original_dot_bound'] < 1 << 53
            r['integer_backend_budget'] = b; shared.append(r)
    end = 0; padding = 0; alias_groups = []
    for (offset, size), users in sorted(components.items()):
        assert offset >= end, ('partial_or_cross_component_overlap', offset, end)
        padding += offset - end; end = offset + size
        cats = {p[1] for p in users}; kinds = {p[2] for p in users}
        assert len(cats) == len(kinds) == 1, ('incompatible_alias', users)
        if len(users) > 1:
            assert cats == {'F32_embedding_alias'} and kinds == {'value'}
            alias_groups.append({'offset': offset, 'bytes': size, 'names': sorted(p[0] for p in users)})
        cat = users[0][1]; g = groups.setdefault(cat, {'value_bytes': 0, 'row_scale_bytes': 0, 'tensor_names': 0})
        g['value_bytes' if users[0][2] == 'value' else 'row_scale_bytes'] += size
    assert end <= payload_bytes; padding += payload_bytes - end
    for r in records.values():
        groups[r['category']]['tensor_names'] += 1
    assert len(alias_groups) == 1 and len(alias_groups[0]['names']) == 3
    totals = {'payload_bytes': payload_bytes, 'unique_component_bytes': sum(size for _, size in components),
              'file_alignment_or_trailing_bytes': padding, 'groups': groups, 'alias_groups': alias_groups,
              'shared_tensor_count': len(shared), 'GPU_shared_weights_bytes': sum(r['integer_backend_budget']['GPU_weight_bytes'] for r in shared),
              'GPU_weight_padding_bytes': sum(r['integer_backend_budget']['weight_padding_bytes'] for r in shared),
              'GPU_max_operator_scratch_bytes': max(r['integer_backend_budget']['GPU_operator_scratch_bytes'] for r in shared),
              'CPU_max_operator_arrays_bytes': max(r['integer_backend_budget']['CPU_operator_arrays_bytes'] for r in shared),
              'CPU_shared_row_scale_bytes': sum(4 * r['rows'] for r in shared)}
    assert totals['unique_component_bytes'] + padding == payload_bytes
    return totals


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert args.out.resolve() == RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and sys.flags.optimize == 0
    start = time.monotonic(); peak = hashed = 0; stage = 'binding'
    out = {'experiment': 'METH484 complete shared integer operator metadata eligibility', 'start_utc': utc(),
           'process_instance': {'pid': os.getpid(), 'create_time_unix': psutil.Process().create_time()}, 'commands': [], 'sources': {}}
    def guard():
        nonlocal peak
        mem = psutil.Process().memory_info(); peak = max(peak, mem.rss, getattr(mem, 'peak_wset', 0))
        assert peak <= 256 << 20 and time.monotonic() - start <= 60, '60seconds_256MiB'
    def sha(path):
        nonlocal hashed
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block := stream.read(1 << 20):
                h.update(block); hashed += len(block); guard()
        return h.hexdigest()
    def verify(item):
        p = Path(item['path']); assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
    try:
        out['head_at_execution'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        rels = [str(BIND.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT))]
        assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *rels], cwd=ROOT)
        binding = json.loads(BIND.read_bytes()); out['binding_sha256'] = sha(BIND)
        assert sys.version == binding['runtime']['python'] and str(Path(sys.executable).resolve()) == binding['runtime']['executable']
        assert psutil.__version__ == binding['runtime']['psutil_version']
        for group in ('helpers', 'source_records', 'catalog'):
            for item in binding[group]:
                verify(item)
        for item in binding['runtime']['files']:
            verify(item)
        for key in ('parent', 'retention'):
            verify(binding[key])
        out['runtime_files_verified'] = len(binding['runtime']['files']); out['catalog_files_verified'] = len(binding['catalog'])
        out['preserved_daemons_before'] = processes()
        parent = json.loads(Path(binding['parent']['path']).read_bytes()); ret = json.loads(Path(binding['retention']['path']).read_bytes())
        assert all(parent['apparatus_gates'].values()) and all(ret['gates'].values()) and ret['raw']['sha256'] == binding['parent']['sha256']
        del ret
        stage = 'original_arithmetic_source'
        texts = {}
        for name in ('meth374_switch_physical_workers.c', 'meth388_switch_three_workers.c'):
            p = BASE / name; assert sha(p) == parent['helper_sha256'][str(p)]; texts[name] = p.read_text(encoding='utf-8')
        a = texts['meth374_switch_physical_workers.c']; b = texts['meth388_switch_three_workers.c']
        b = b.replace('meth388_switch_three_workers', 'meth374_switch_physical_workers').replace('meth388_contract_entry', 'meth374_contract_entry').replace('meth388_forced_entry', 'meth374_forced_entry').replace('meth388_switch_thread_binding.h', 'meth374_switch_thread_binding.h').replace('threads==1||threads==3||threads==6', 'threads==1||threads==6')
        assert a == b
        compact = re.sub(r'\s+', '', a)
        for pattern in ('fegetround()==FE_TONEAREST', 'longcode=lrintf(divided);', 'maximum/32767.f',
                        'if(code>32767)code=32767;if(code<-32767)code=-32767;',
                        '(float)(((double)value*(double)w.scales[row])*(double)scale)',
                        '(float)(((double)value*(double)w.scales[row])*(double)scales[token])',
                        'index%step==1||step==1', 'staticint64_thead_integer_dot'):
            assert pattern in compact, pattern
        assert sha(ROOT / 'benchmarks/phase60/engine.c') == parent['preserved_engine_sha256']
        contracts = []
        for n, label in ((374, 'physical'), (389, 'three')):
            r = json.loads((DOC / f'meth{n}_switch_{label}_workers_contract_result.json').read_bytes())
            assert all(r['gates'].values()) and '-fno-fast-math' in r['compile']['argv'] and '-ffp-contract=off' in r['compile']['argv']
            contracts.append({'method': n, 'inherited_gates': r['gates'], 'compile': r['compile'], 'not_reexecuted': True})
        out['original_arithmetic_contracts'] = contracts
        hits = []
        expression = re.compile(r'cublas|CUDA_R_8I|CUBLAS_COMPUTE_32I|GemmEx|(?:q0|q1|q2).*16384', re.I)
        for item in binding['catalog']:
            for line, text in enumerate(Path(item['path']).read_text(encoding='utf-8').splitlines(), 1):
                if expression.search(text):
                    hits.append({'path': item['relative'], 'line': line, 'text': text[:600]})
            guard()
        out['backend_name_catalog'] = {'scope': binding['catalog_scope'], 'hits': hits,
            'not_a_semantic_global_absence_proof': True, 'untracked_other_branches_external_libraries_excluded': True}
        stage = 'every_tensor_geometry_and_bytes'
        for source_binding in sorted(binding['sources'], key=lambda v: v['n']):
            n = source_binding['n']; source = parent['sources'][str(n)]
            assert source['profiler_admissibility']['admitted'] and all(source['profiler_admissibility']['gates'].values())
            verify(source_binding['manifest']); p = Path(source_binding['payload']['path']); st = p.stat()
            assert [st.st_size, st.st_mtime_ns] == source['artifact_stat_before'] == [source_binding['payload']['bytes'], source_binding['payload']['mtime_ns']]
            c, records = descriptors(source_binding['manifest']['path'], source_binding['payload']); assert c['n'] == n
            sparse, dense = classify(c, records); budget = byte_budgets(records, st.st_size)
            assert budget['GPU_shared_weights_bytes'] <= 1 << 30 and budget['GPU_max_operator_scratch_bytes'] <= 16 << 20
            # Universal integer algebra bounds; no integer vector/GPU arithmetic control run here.
            assert 512 * 128 * 32767 < 1 << 31 and 4096 * 128 * 32767 < 1 << 53
            rows = source['mode_summaries']['1']; fractions = source['admitted_decomposition']['whole_cost_fractions']
            total = rows['ALL_case_mean_full_seconds']; measured = rows['matrix_mean_seconds_by_kind']
            eligible_seconds = measured['dense_control_matrices'] + measured['head_matrix']; f = eligible_seconds / total
            assert math.isclose(f, fractions['dense_control_matrices'] + fractions['head_matrix'], rel_tol=1e-14)
            cost = {'inherited_profile1_aggregate_sum_of_96_case_means_seconds': total,
                    'inherited_dense_plus_head_aggregate_seconds': eligible_seconds, 'eligible_measured_fraction': f,
                    'remaining_measured_fraction': 1 - f, 'constraints': []}
            for metric in ('ordinary', 'prose'):
                rate = rows['accepted_' + metric + '_rate_mean']
                for target in (50, 100):
                    ratio = rate / target; limit = (ratio - (1 - f)) / f
                    cost['constraints'].append({'metric': metric, 'target_accepted_IDs_per_second': target,
                                               'inherited_rate': rate, 'required_whole_cost_ratio': ratio,
                                               'maximum_component_multiplier_at_h0': limit,
                                               'maximum_additional_overhead_fraction_at_r0': ratio - (1 - f),
                                               'equation': '(1-f)+f*r+h <= inherited_rate/target'})
            cases = []; d, ff = c['d'], c['ff']; le, ld = c['enc'], c['dec']; ae, ad = len(dense['encoder']), len(dense['decoder'])
            for index, case in enumerate(source['cases']):
                assert case['book'] == index // 4 and case['case'] == index % 4
                t = case['generated_tokens']; case_rows = [v for vs in case['modes'].values() for v in vs]
                s = case_rows[0]['source_tokens']; assert s == 29 and len(case_rows) == 8
                expected = [((4*le+2*ae+2*ld)*s, ((4*le+2*ld)*d*d+2*ae*d*ff)*s,
                             ((4*le+2*ld)*d+ae*(ff+d))*4*s),
                            ((6*ld+2*ad)*t, (6*ld*d*d+2*ad*d*ff)*t, (6*ld*d+ad*(ff+d))*4*t)]
                for row in case_rows:
                    assert row['source_tokens'] == s and row['actual_generated_tokens'] == t
                    for phase in range(2):
                        counter = row['counters'][phase][0]
                        assert tuple(counter[k] for k in ('calls', 'code_bytes', 'scale_bytes')) == expected[phase] and counter['f32_bytes'] == 0
                    counter = row['counters'][1][3]
                    assert (counter['calls'], counter['code_bytes'], counter['scale_bytes'], counter['f32_bytes']) == (t, t*c['vocab']*d, t*c['vocab']*4, 0)
                calls = expected[0][0] + expected[1][0] + t
                sum_d = (4*le+2*ld)*d*s + ae*(d+ff)*s + 6*ld*d*t + ad*(d+ff)*t + d*t
                sum_m = expected[0][2]//4 + expected[1][2]//4 + c['vocab']*t
                assert d % 8 == ff % 8 == c['vocab'] % 8 == 0, 'this transfer formula assumes zero row/width padding'
                cases.append({'book': case['book'], 'case': case['case'], 'source_tokens': s, 'generated_tokens': t,
                              'eligible_logical_query_maps': calls,
                              'proposed_integer_GEMM_invocations': 3*le+2*ld+(le+2*ae)*s+(6*ld+2*ad+1)*t,
                              'H2D_digit_bytes': 4*sum_d, 'D2H_four_I32_partial_bytes': 16*sum_m})
                guard()
            assert len(cases) == 96
            out['sources'][str(n)] = {'config': c, 'manifest': source_binding['manifest'], 'payload': source_binding['payload'],
                'payload_SHA_refreshed': False, 'payload_values_read': False, 'sparse_layers': sparse, 'dense_layers': dense,
                'tensors': [records[name] for name in sorted(records)], 'byte_budget': budget, 'whole_cost': cost,
                'all_96_cases_including_rejected': cases,
                'aggregate_logical_transfer_and_invocations': {key: sum(row[key] for row in cases) for key in
                    ('eligible_logical_query_maps', 'proposed_integer_GEMM_invocations', 'H2D_digit_bytes', 'D2H_four_I32_partial_bytes')}}
        x, y = (out['sources'][str(n)] for n in (128, 256))
        assert {k:v for k,v in x['config'].items() if k != 'n'} == {k:v for k,v in y['config'].items() if k != 'n'}
        assert x['byte_budget']['GPU_shared_weights_bytes'] == y['byte_budget']['GPU_shared_weights_bytes']
        assert next(v for v in y['whole_cost']['constraints'] if v['metric'] == 'prose' and v['target_accepted_IDs_per_second'] == 50)['maximum_additional_overhead_fraction_at_r0'] > 0
        out['fixed_core_geometry_n_independent'] = True
        out['not_shared_weight_value_equality_or_causal_n_comparison'] = True
        out['integer_identity'] = {'formula': 'u=uint16(q); s=(u>=32768); q0=u&127; q1=(u>>7)&127; q2=(u>>14)-4*s; q=q0+128*q1+16384*q2',
            'partial_absolute_bounds': ['16256*D', '16256*D', '256*D'],
            'signed_digits': [[0,127],[0,127],[-2,1]], 'padding_fourth_column_zero': True,
            'original_CPU_quantizer_and_F64_row_then_query_scale_then_F32_required': True,
            'no_compiled_or_GPU_numeric_equivalence_claim': True}
        out['preserved_daemons_after'] = processes(); guard()
        out['gates'] = {'frozen_sources_runtime_and_qualified_metadata_exact': True,
            'all_tensors_shape_codec_bounds_and_aliases_classified': True,
            'integer_overflow_and_fixed_padding_budgets_eligible': True,
            'eligible_GPU_weights_le1GiB_scratch_le16MiB_n_independent': True,
            'ALL3072_dense_and1536_head_inherited_counter_rows_exact': True,
            'conditional_whole_cost_and_ALL192_case_transfer_counts_exact': True,
            'zero_payload_GPU_solver_native_model_work': True}
        out['decision'] = 'METADATA_ELIGIBLE_FOR_ONE_FROZEN_COMPLETE_INTEGER_BACKEND_CYCLE_NOT_A_SPEED_OR_QUALITY_RESULT'
        out['end_utc'] = utc(); out['resource'] = {'seconds_excluding_imports': time.monotonic()-start, 'peak_working_set_bytes': peak, 'bytes_hashed': hashed}
        exclusive(RAW, out); guard()
        print(json.dumps({'sha256': sha(RAW), 'gates': out['gates'], 'decision': out['decision'], 'resource': out['resource'],
                          'terminal_peak_working_set_bytes': peak, 'terminal_seconds': time.monotonic()-start,
                          'budgets': {n:r['byte_budget'] for n,r in out['sources'].items()},
                          'constraints': {n:r['whole_cost']['constraints'] for n,r in out['sources'].items()}}), flush=True)
    except BaseException as error:
        out.update(stage=stage, error=repr(error), end_utc=utc(), resource={'seconds_excluding_imports': time.monotonic()-start, 'peak_working_set_bytes': peak, 'bytes_hashed': hashed})
        exclusive(RAW.with_suffix('.failure.json'), out); raise


if __name__ == '__main__':
    main()
