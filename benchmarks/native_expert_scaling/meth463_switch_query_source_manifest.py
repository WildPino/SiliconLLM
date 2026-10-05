"""Frozen model-free independent source manifest after METH462 DATA failure."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
PROTOCOL = DOC / 'METH_463_SWITCH_QUERY_SOURCE_MANIFEST_PROTOCOL_20261005.md'
BINDINGS = DOC / 'meth463_prospective_bindings.json'
BINDINGS_SHA = '65309f005dfb17d3c02abcdf3382b4a947b6a2ae365ccbcc2616fb4c99cd6211'
RAW = DOC / 'meth463_switch_query_source_manifest.json'
SOURCE = ROOT / 'results/native_expert_scaling/meth378_switch_base128_source'
CORPUS_REL = 'data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet'
CORPUS_SHA = '15f3ecb5e314ce58bbf8eae927b51bf893b9fdabe4a04c8e246547206357a2d5'
SEED = 'meth463-original128-independent-query-domain-463463'
SPLIT_SEED = 'meth463-book-role-development-validation-463463'
DIRECT = {
    'meth462_switch_query_domain_admission_result.json': 'dbf6114776ae1f6850300c708ddda9f95c6a0071ae91ae848519bf6c82a8f096',
    'RETENTION_462_20261005.json': '783273e98cc525d56ad7508a3058422116c924fd87b97aab072d022fa6e1a1e9',
    'meth382_switch_multi_span_manifest.json': '96b3ffd09b0bdd42d7c990006e8744e39b5e1f32b7ae40aaf3b3dca98bf4db78',
    'meth362_switch_multi_span_manifest.json': 'c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf',
    'meth378_switch_base128_acquisition_result.json': '89add8d24541423dfa61827f785bf125c2062cf1aa6976214f03da94b7c582b7',
    'meth381_switch_base128_contract_result.json': '6ede7a90fbf2716381d31456897eb41b01a0c6ed9010e7ba3b3af91064831a6e',
}
SENTINELS = [32099, 32098, 32097, 32096, 32095]
STARTS = [3, 10, 17, 24]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def committed(path):
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    data = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters',
                                    '--path=' + rel, 'HEAD:' + rel], cwd=ROOT)
    assert Path(path).read_bytes() == data, ('physical_HEAD_identity', rel)


def write_new(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').replace('\n', '\r\n').encode('utf-8')
    assert len(data) <= 8 << 20, ('raw8MiB', len(data))
    with Path(path).open('xb') as stream:
        stream.write(data)


def fields(tokenizer, original):
    assert len(original) == 32
    source = []; target = []; spans = []; previous = 0
    for index, begin in enumerate(STARTS):
        span = original[begin:begin + 2]; spans.append(span)
        source.extend(original[previous:begin]); source.append(SENTINELS[index]); previous = begin + 2
        target.extend([SENTINELS[index], *span])
    source.extend(original[previous:]); source.append(1); target.extend([SENTINELS[4], 1])
    decoder = [0] + target[:-1]
    assert (len(source), len(target), len(decoder)) == (29, 14, 14)
    return {'original_window_ids': original, 'span_starts': STARTS.copy(), 'masked_spans_ids': spans,
            'source_ids': source, 'decoder_ids': decoder, 'target_ids': target,
            'original_window_text': tokenizer.decode(original), 'source_text': tokenizer.decode(source),
            'target_text': tokenizer.decode(target), 'source_ids_sha256': sha(struct.pack('<29i', *source)),
            'target_ids_sha256': sha(struct.pack('<14i', *target))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert args.out.resolve() == RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists()
    assert os.name == 'nt' and sys.flags.optimize == 0
    start = time.monotonic(); peak = hashed = 0; stage = 'bindings_and_imports'
    result = {'experiment': 'METH463-original128-new-independent-book-query-manifest', 'native_or_model_commands': 0,
              'source_binding_sha256': BINDINGS_SHA, 'retained_record_sha256': {}, 'tokenizer_bridge': {},
              'prior_records': [], 'items': [], 'source_only_rejections': [],
              'argv': sys.argv.copy(), 'gates': {}}
    try:
        import psutil
        process = psutil.Process()
        def guard():
            nonlocal peak
            info = process.memory_info(); peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
            assert peak <= 4 << 30 and time.monotonic() - start <= 300, 'hard300s4GiB'
        def digest(path):
            nonlocal hashed
            h = hashlib.sha256()
            with Path(path).open('rb') as stream:
                while block := stream.read(4 << 20):
                    h.update(block); hashed += len(block); guard()
            return h.hexdigest()
        def jobs():
            own = {os.getpid(), *(p.pid for p in process.parents())}; preserved = []
            for p in psutil.process_iter(['name', 'cmdline']):
                if p.pid in own: continue
                name, argv = (p.info['name'] or '').lower(), p.info['cmdline'] or []
                if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                    preserved.append(p.pid); continue
                assert not (name.startswith('python') or (name.startswith('meth') and name.endswith('.exe')) or name == 'clang.exe'), (p.pid, name)
            return preserved
        result['preserved_daemon_pids_before'] = jobs()
        for path in (Path(__file__), PROTOCOL, BINDINGS, *(DOC / name for name in DIRECT)):
            committed(path)
        assert digest(BINDINGS) == BINDINGS_SHA
        bindings = json.loads(BINDINGS.read_text(encoding='utf-8'))
        result['controller_sha256'] = digest(__file__); result['protocol_sha256'] = digest(PROTOCOL)
        result['head_at_execution'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        result['engine_sha256'] = digest(ROOT / 'benchmarks/phase60/engine.c')
        committed(ROOT / 'benchmarks/phase60/engine.c')
        assert bindings['corpus']['path'] == CORPUS_REL and bindings['corpus']['sha256'] == CORPUS_SHA
        assert Path(sys.executable).resolve() == Path(bindings['runtime']['executable']).resolve()
        assert sys.version == bindings['runtime']['python']
        for path, row in bindings['runtime']['files'].items():
            assert Path(path).stat().st_size == row['bytes'] and digest(path) == row['sha256'], path
        for name, row in bindings['runtime']['packages'].items():
            assert metadata.version(name) == row['version'] and digest(row['metadata_path']) == row['sha256'], name
        assert metadata.version('transformers') == '4.57.6'
        result['runtime'] = bindings['runtime']
        for rel, row in bindings['preserved_unrelated_files'].items():
            assert (ROOT / rel).stat().st_size == row['bytes'] and digest(ROOT / rel) == row['sha256'], rel
        assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines() == bindings['tracked_status']
        for name, row in bindings['source_sidefiles'].items():
            assert (SOURCE / name).stat().st_size == row['bytes'] and digest(SOURCE / name) == row['sha256'], name
        result['source_tokenizer_files'] = bindings['source_sidefiles']
        for name, expected in DIRECT.items():
            assert digest(DOC / name) == expected, name
            result['retained_record_sha256'][name] = expected
        old = json.loads((DOC / 'meth382_switch_multi_span_manifest.json').read_text(encoding='utf-8'))
        other = json.loads((DOC / 'meth362_switch_multi_span_manifest.json').read_text(encoding='utf-8'))
        admission = json.loads((DOC / 'meth462_switch_query_domain_admission_result.json').read_text(encoding='utf-8'))
        assert all(admission['gates'].values()) and admission['decision'] == 'existing_data_insufficient_for_fixed_private_INPUT_probe_stop_before_factors_design_new_independent_captures'
        assert all(json.loads((DOC / 'meth381_switch_base128_contract_result.json').read_text(encoding='utf-8'))['gates'].values())
        assert json.loads((DOC / 'meth378_switch_base128_acquisition_result.json').read_text(encoding='utf-8'))['passed'] is True
        assert all(old['gates'].values()) and all(other['gates'].values())
        assert digest(ROOT / CORPUS_REL) == CORPUS_SHA
        result['corpus'] = bindings['corpus']; result['model'] = old['model']; result['revision'] = old['revision']
        result['gates']['strict_source_runtime_and_record_bindings'] = True

        stage = 'current_all_record_exclusions'
        row_pattern = re.compile(re.escape(('pg19:' + CORPUS_REL + ':row=').encode()) + rb'(\d+)')
        generic_pattern = re.compile(rb'"(?:source_row|corpus_row)"\s*:\s*(\d+)')
        list_pattern = re.compile(rb'"excluded_corpus_rows"\s*:\s*\[([\d,\s]*)\]')
        hex_pattern = re.compile(rb'(?<![0-9A-Fa-f])([0-9A-Fa-f]{64})(?![0-9A-Fa-f])')
        excluded = set(); recorded_hashes = set()
        for row in bindings['records']:
            path = ROOT / row['path']; physical = hashlib.sha256(); canonical = hashlib.sha256()
            tail = b''; pending = b''; rows = set(); generic = set(); inherited = set(); has_corpus = False; count = 0; values = set()
            assert path.stat().st_size == row['bytes'], row['path']
            with path.open('rb') as stream:
                while block := stream.read(4 << 20):
                    physical.update(block); hashed += len(block); count += len(block)
                    canonical_chunk = pending + block; pending = canonical_chunk[-1:] if canonical_chunk.endswith(b'\r') else b''
                    if pending: canonical_chunk = canonical_chunk[:-1]
                    canonical.update(canonical_chunk.replace(b'\r\n', b'\n'))
                    data = tail + block; has_corpus |= CORPUS_REL.encode() in data
                    rows.update(int(v) for v in row_pattern.findall(data))
                    generic.update(int(v) for v in generic_pattern.findall(data))
                    for numbers in list_pattern.findall(data): inherited.update(int(v) for v in re.findall(rb'\d+', numbers))
                    values.update(v.decode('ascii').lower() for v in hex_pattern.findall(data))
                    tail = data[-(64 << 10):]; guard()
            canonical.update(pending)
            assert count == row['bytes'] and physical.hexdigest() == row['sha256'] and canonical.hexdigest() == row['canonical_lf_sha256'], row['path']
            assert all(0 <= v < 1243 for v in rows), ('explicit_corpus_row_range', row['path'])
            if has_corpus:
                assert all(0 <= v < 1243 for v in inherited), ('inherited_corpus_row_range', row['path'])
                rows.update(v for v in generic if 0 <= v < 1243); rows.update(inherited)
            excluded.update(rows); recorded_hashes.update(values)
            result['prior_records'].append({**row, 'mentions_corpus': has_corpus, 'excluded_corpus_rows': sorted(rows),
                                           'ambiguous_generic_rows_outside_corpus': sorted(v for v in generic if not 0 <= v < 1243) if has_corpus else [],
                                           'hex64_value_count': len(values), 'hex64_value_set_sha256': sha(('\n'.join(sorted(values)) + '\n').encode())})
        prior_books = old['items'] + other['items']
        assert all(item['corpus_row'] in excluded and item['whole_source_utf8_sha256'] in recorded_hashes for item in prior_books)
        assert set(old['excluded_corpus_rows']).issubset(excluded) and len(excluded) >= 160
        result['excluded_corpus_rows'] = sorted(excluded)
        result['recorded_hex64_values'] = {'count': len(recorded_hashes), 'sha256': sha(('\n'.join(sorted(recorded_hashes)) + '\n').encode()),
                                         'rule': 'conservative membership in ANY recorded hex64; includes all recorded whole-source hashes regardless of field name'}
        result['gates']['all_current_records_and_both_prior_cohorts_excluded'] = True
        result['exclusion_seconds'] = time.monotonic() - start; guard()

        stage = 'original_tokenizer_all382_bridge'
        import transformers
        from transformers import AutoTokenizer
        import pyarrow.parquet as pq
        assert transformers.__version__ == '4.57.6'
        tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
        assert type(tokenizer).__name__ == old['tokenizer_class']
        assert [tokenizer.convert_tokens_to_ids(f'<extra_id_{i}>') for i in range(5)] == SENTINELS
        assert (tokenizer.eos_token_id, tokenizer.pad_token_id) == (1, 0)
        bridge_rows = []
        for book in old['items']:
            assert sha(book['excerpt'].encode()) == book['excerpt_utf8_sha256']
            tokens = tokenizer.encode(book['excerpt'], add_special_tokens=False)
            for case in book['cases']:
                original = tokens[case['excerpt_token_start']:case['excerpt_token_start'] + 32]
                actual = fields(tokenizer, original)
                assert all(actual[key] == case[key] for key in actual), (book['corpus_row'], case['index'])
                bridge_rows.append({'corpus_row': book['corpus_row'], 'case': case['index'],
                                    'source_ids_sha256': actual['source_ids_sha256'], 'target_ids_sha256': actual['target_ids_sha256'],
                                    'decoder_ids_sha256': sha(struct.pack('<14i', *actual['decoder_ids']))})
                guard()
        assert len(bridge_rows) == 96
        result['tokenizer_bridge'] = {'all96_cases': bridge_rows, 'all384_two_token_fields_exact': True,
                                     'tokenizer_class': type(tokenizer).__name__, 'sentinels': SENTINELS, 'eos': 1, 'pad': 0}
        result['gates']['all382_token_field_hash_and_decode_bridge'] = True

        stage = 'one_streamed_corpus_snapshot'
        parquet = pq.ParquetFile(ROOT / CORPUS_REL); assert parquet.metadata.num_rows == 1243
        ranks = sorted((v for v in range(1243) if v not in excluded), key=lambda v: (sha(f'{SEED}|row={v}'.encode()), v))
        assert len(ranks) >= 128
        candidate_order = ranks[:512]; wanted = set(candidate_order); texts = {}; whole_hashes = {}; source_hashes = {}; row_index = 0
        for batch in parquet.iter_batches(batch_size=1, columns=['text']):
            for text in batch.column('text').to_pylist():
                if isinstance(text, str):
                    whole = sha(text.encode()); source_hashes[row_index] = whole
                    if row_index in excluded: whole_hashes[row_index] = whole
                    if row_index in wanted: texts[row_index] = text
                row_index += 1
            guard()
        assert row_index == 1243
        excluded_whole = recorded_hashes | set(whole_hashes.values())
        result['excluded_row_whole_text_sha256'] = [{'corpus_row': v, 'sha256': whole_hashes[v]} for v in sorted(whole_hashes)]
        result['all_corpus_row_source_hashes'] = [{'corpus_row': v, 'sha256': source_hashes[v]} for v in sorted(source_hashes)]
        result['selection'] = {'seed': SEED, 'split_seed': SPLIT_SEED, 'maximum_candidates': 512,
                               'candidate_order': candidate_order, 'admissible_row_rank_count': len(ranks),
                               'source_row_count': row_index, 'parquet_row_groups': parquet.metadata.num_row_groups}

        stage = 'fixed_source_only_selection'
        selected = []; seen_whole = set(); seen_windows = set()
        for rank, row in enumerate(candidate_order):
            source_id = f'pg19:{CORPUS_REL}:row={row}'; text = texts.get(row, ''); whole = source_hashes.get(row)
            reason = None
            if not text or len(text) < 8192: reason = 'less_than8192_characters_or_nontext'
            elif whole in excluded_whole: reason = 'previous_whole_source_text_hash'
            elif whole in seen_whole: reason = 'duplicate_new_whole_source_text_hash'
            if reason:
                result['source_only_rejections'].append({'rank': rank, 'source_id': source_id, 'reason': reason, 'whole_source_utf8_sha256': whole}); continue
            excerpt_start = int(sha((SEED + '|' + source_id).encode()), 16) % (len(text) - 4096)
            excerpt = text[excerpt_start:excerpt_start + 4096]; tokens = tokenizer.encode(excerpt, add_special_tokens=False)
            if len(tokens) < 512 or len(set(tokens)) < 128 or any(v <= 1 or v >= 32000 for v in tokens):
                result['source_only_rejections'].append({'rank': rank, 'source_id': source_id, 'reason': 'token_length_diversity_or_special_id_guard', 'whole_source_utf8_sha256': whole}); continue
            cases = []; quarter = len(tokens) // 4
            for case_index in range(4):
                window_start = case_index * quarter + int(sha(f'{SEED}|{source_id}|window{case_index}'.encode()), 16) % (quarter - 32 + 1)
                original = tokens[window_start:window_start + 32]
                cases.append({'index': case_index, 'excerpt_token_start': window_start, **fields(tokenizer, original)})
            windows = [sha(struct.pack('<32i', *case['original_window_ids'])) for case in cases]
            if len(set(windows)) != 4 or set(windows) & seen_windows:
                result['source_only_rejections'].append({'rank': rank, 'source_id': source_id, 'reason': 'duplicate32_token_window', 'whole_source_utf8_sha256': whole}); continue
            selected.append({'selection_rank': rank, 'source_id': source_id, 'corpus_row': row, 'whole_source_utf8_sha256': whole,
                             'source_characters': len(text), 'excerpt_start_character': excerpt_start, 'excerpt': excerpt,
                             'excerpt_utf8_sha256': sha(excerpt.encode()), 'cases': cases})
            seen_whole.add(whole); seen_windows.update(windows); guard()
            if len(selected) == 128: break
        assert len(selected) == 128, ('source_only_deficit', len(selected))
        role_order = sorted(selected, key=lambda v: (sha((SPLIT_SEED + '|' + v['source_id']).encode()), v['corpus_row']))
        for index, item in enumerate(role_order):
            item['book'] = index; item['role'] = 'development' if index < 64 else 'diagnostic_validation'
        result['items'] = role_order
        assert len({v['corpus_row'] for v in role_order}) == len(seen_whole) == 128
        assert not {v['corpus_row'] for v in role_order} & excluded and not seen_whole & excluded_whole
        assert len(seen_windows) == 512 and sum(len(v['cases']) for v in role_order) == 512
        assert sum(v['role'] == 'development' for v in role_order) == sum(v['role'] == 'diagnostic_validation' for v in role_order) == 64
        result['gates'].update(all128_new_whole_sources_and512_unique_windows=True, immutable64_64_book_roles_and_S29_T14=True,
                               model_free_source_only_selection=True)
        result['preserved_daemon_pids_after'] = jobs(); guard()
        for rel, row in bindings['preserved_unrelated_files'].items():
            assert digest(ROOT / rel) == row['sha256'], rel
        assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).splitlines() == bindings['tracked_status']
        result['gates']['original_engine_and_unrelated_work_preserved'] = True
        result['resource'] = {'main_seconds_including_imports_through_selection': time.monotonic() - start, 'peak_process_bytes': peak,
                              'bytes_hashed': hashed, 'hard_seconds': 300, 'hard_rss_bytes': 4 << 30, 'hard_raw_bytes': 8 << 20}
        result['decision'] = 'new_independent_source_manifest_ready_freeze_separate_native_capture_and_same_data_admission_floors'
        result['scope'] = '128 NEW project source books/64dev64diagnostic-val/4cases each; original128 tokenizer/native S29T14 contract. No donor-pretraining disjointness, statistical independence guarantee, model/capture/geometry/quality/rate/DRAM/useful-n result.'
        write_new(RAW, result); guard()
        print(json.dumps({'sha256': digest(RAW), 'books': len(result['items']), 'contexts': len(seen_windows),
                          'excluded_rows': len(excluded), 'prior_records': len(result['prior_records']),
                          'rejections': len(result['source_only_rejections']), 'gates': result['gates'], 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update(stage=stage, error=repr(error), seconds=time.monotonic() - start, peak_process_bytes=peak)
        failure = RAW.with_suffix('.failure.json')
        if not failure.exists(): write_new(failure, result)
        raise


if __name__ == '__main__':
    main()
