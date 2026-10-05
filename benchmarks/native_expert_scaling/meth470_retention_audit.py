"""Metadata-only METH470 retention; no main/codec/Arrow import or tokenization."""
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time
import psutil

ROOT = Path.cwd()
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
RAW = DOC / 'meth470_switch_development_manifest.json'
TARGET = DOC / 'RETENTION_470_20261005.json'
OUT = ROOT / 'results/native_expert_scaling/meth470_switch_development_manifest'
EVENTS = ROOT / 'results/native_expert_scaling/meth470_windows_terminal.json'
assert not TARGET.exists()
start = time.monotonic()
peak = hashed = 0

def guard():
    global peak
    info = psutil.Process().memory_info()
    peak = max(peak, info.rss, getattr(info, 'peak_wset', 0))
    assert peak <= 512 << 20 and time.monotonic() - start <= 300

def sha(data):
    return hashlib.sha256(data).hexdigest()

def digest(path):
    global hashed
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(4 << 20):
            h.update(data)
            hashed += len(data)
            guard()
    return h.hexdigest()

def head(path):
    rel = path.relative_to(ROOT).as_posix()
    assert path.read_bytes() == subprocess.check_output(
        ['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel])

assert digest(RAW) == '41151e8d80cfa67d0766e6ce2f610ee5834900d08c29f5dd41b74214a28e4dee'
j = json.loads(RAW.read_text(encoding='utf-8'))
assert len(j['gates']) == 12 and all(j['gates'].values()) and j['native_or_model_commands'] == 0
bindings = json.loads((DOC / 'meth470_prospective_bindings.json').read_text(encoding='utf-8'))
assert digest(DOC / 'meth470_prospective_bindings.json') == j['source_binding_sha256']
head(DOC / 'meth470_prospective_bindings.json')
source = ROOT / 'benchmarks/native_expert_scaling/meth470_switch_development_manifest.py'
protocol = DOC / 'METH_470_SWITCH_DEVELOPMENT_MANIFEST_PROTOCOL_20261005.md'
for p, h in ((source, j['controller_sha256']), (protocol, j['protocol_sha256']),
             (ROOT / 'benchmarks/phase60/engine.c', j['engine_sha256'])):
    assert digest(p) == h
    head(p)
for name, h in j['retained_record_sha256'].items():
    assert digest(DOC / name) == h
    head(DOC / name)
for p, v in bindings['runtime']['files'].items():
    assert Path(p).stat().st_size == v['bytes'] and digest(p) == v['sha256']
for v in bindings['runtime']['packages'].values():
    assert digest(v['metadata_path']) == v['sha256']
    for p, row in v['code_files'].items():
        assert Path(p).stat().st_size == row['bytes'] and digest(p) == row['sha256'], p
for name, path in j['actual_loaded_package_paths'].items():
    assert path in bindings['runtime']['packages'][name.partition('.')[0]]['code_files'], (name, path)
for rel, row in bindings['reused_sources'].items():
    assert digest(ROOT / rel) == row['sha256']
    head(ROOT / rel)
prep = bindings['preparation_helper']
assert digest(prep['path']) == prep['sha256']
for name, v in bindings['source_sidefiles'].items():
    p = ROOT / 'results/native_expert_scaling/meth378_switch_base128_source' / name
    assert p.stat().st_size == v['bytes'] and digest(p) == v['sha256']
codec = json.loads((DOC / 'meth467_codec_bindings.json').read_text(encoding='utf-8'))
assert j['codec']['files'] == codec['files'] and j['codec']['tokenizers_version'] == codec['version'] == '0.22.2'
assert j['codec']['bindings_sha256'] == j['retained_record_sha256']['meth467_codec_bindings.json']
assert j['codec']['disallowed_modules_after_import'] == j['codec']['disallowed_modules_after_all_cases'] == []
for p, v in codec['files'].items():
    assert Path(p).stat().st_size == v['bytes'] and digest(p) == v['sha256']
assert digest(ROOT / bindings['corpus']['path']) == bindings['corpus']['sha256']

# Fresh independent exclusion extraction, including canonical bytes and hash membership.
assert len(j['prior_records']) == len(bindings['records']) == 1863
corpus_rel = bindings['corpus']['path']
row_pattern = re.compile(re.escape(('pg19:' + corpus_rel + ':row=').encode()) + rb'(\d+)')
generic_pattern = re.compile(rb'"(?:source_row|corpus_row)"\s*:\s*(\d+)')
list_pattern = re.compile(rb'"excluded_corpus_rows"\s*:\s*\[([\d,\s]*)\]')
hex_pattern = re.compile(rb'(?<![0-9A-Fa-f])([0-9A-Fa-f]{64})(?![0-9A-Fa-f])')
union = set()
all_hashes = set()
for actual, bound in zip(j['prior_records'], bindings['records']):
    assert all(actual[k] == v for k, v in bound.items())
    p = ROOT / bound['path']
    assert p.stat().st_size == bound['bytes']
    physical = hashlib.sha256()
    canonical = hashlib.sha256()
    tail = pending = b''
    rows = set()
    generic = set()
    inherited = set()
    values = set()
    mentions = False
    with p.open('rb') as stream:
        while block := stream.read(4 << 20):
            physical.update(block)
            hashed += len(block)
            canonical_chunk = pending + block
            pending = b'\r' if canonical_chunk.endswith(b'\r') else b''
            if pending:
                canonical_chunk = canonical_chunk[:-1]
            canonical.update(canonical_chunk.replace(b'\r\n', b'\n'))
            data = tail + block
            mentions |= corpus_rel.encode() in data
            rows.update(map(int, row_pattern.findall(data)))
            generic.update(map(int, generic_pattern.findall(data)))
            for numbers in list_pattern.findall(data):
                inherited.update(map(int, re.findall(rb'\d+', numbers)))
            values.update(v.decode('ascii').lower() for v in hex_pattern.findall(data))
            tail = data[-(64 << 10):]
            guard()
    canonical.update(pending)
    assert physical.hexdigest() == bound['sha256'] and canonical.hexdigest() == bound['canonical_lf_sha256']
    assert all(0 <= v < 1243 for v in rows)
    if mentions:
        assert all(0 <= v < 1243 for v in inherited)
        rows.update(v for v in generic if 0 <= v < 1243)
        rows.update(inherited)
    assert actual['mentions_corpus'] == mentions and actual['excluded_corpus_rows'] == sorted(rows)
    assert actual['ambiguous_generic_rows_outside_corpus'] == (sorted(v for v in generic if not 0 <= v < 1243) if mentions else [])
    assert actual['hex64_value_count'] == len(values)
    assert actual['hex64_value_set_sha256'] == sha(('\n'.join(sorted(values)) + '\n').encode())
    union.update(rows)
    all_hashes.update(values)
assert union == set(j['excluded_corpus_rows']) and len(union) == 293
assert j['recorded_hex64_values']['count'] == len(all_hashes)
assert j['recorded_hex64_values']['sha256'] == sha(('\n'.join(sorted(all_hashes)) + '\n').encode())
bridge = j['tokenizer_bridge']['all96_cases']
assert len(bridge) == 96 and j['tokenizer_bridge']['all384_two_token_fields_exact']
old = json.loads((DOC / 'meth382_switch_multi_span_manifest.json').read_text(encoding='utf-8'))
other = json.loads((DOC / 'meth362_switch_multi_span_manifest.json').read_text(encoding='utf-8'))
expected_bridge = {(book['corpus_row'], case['index']): (case['source_ids_sha256'], case['target_ids_sha256'],
                   sha(struct.pack('<14i', *case['decoder_ids']))) for book in old['items'] for case in book['cases']}
assert {(r['corpus_row'], r['case']): (r['source_ids_sha256'], r['target_ids_sha256'], r['decoder_ids_sha256']) for r in bridge} == expected_bridge
previous = json.loads((DOC / 'meth467_switch_rust_query_manifest.json').read_text(encoding='utf-8'))
prior_books = old['items'] + other['items'] + previous['items']
assert all(book['corpus_row'] in union and book['whole_source_utf8_sha256'] in all_hashes for book in prior_books)
assert set(old['excluded_corpus_rows']).issubset(union)
expected_old_roles = [{'book': v['book'], 'role': v['role'], 'source_id': v['source_id'],
                       'whole_source_utf8_sha256': v['whole_source_utf8_sha256']} for v in previous['items']]
assert j['unchanged_original_cohort']['original_book_roles'] == expected_old_roles
assert [v['book'] for v in expected_old_roles] == list(range(128))
assert all(v['role'] == ('development' if v['book'] < 64 else 'diagnostic_validation') for v in expected_old_roles)
assert j['unchanged_original_cohort']['validation_books'] == list(range(64, 128))
assert j['unchanged_original_cohort']['manifest_sha256'] == j['retained_record_sha256']['meth467_switch_rust_query_manifest.json']
assert j['unchanged_original_cohort']['capture_sha256'] == j['retained_record_sha256']['meth469_switch_native_domain_capture_result.json']
assert j['unchanged_original_cohort']['fixed_bank'] == 11
selected = j['items']
assert len(selected) == 64
seed = j['selection']['seed']
assert seed == 'meth470-original128-development-only-query-domain-470470'
assert j['selection']['desired_new_development_books'] == 64
assert j['selection']['new_global_book_range'] == [128, 191]
ranks = sorted((r for r in range(1243) if r not in union), key=lambda r: (sha(f'{seed}|row={r}'.encode()), r))
assert j['selection']['candidate_order'] == ranks[:512]
assert j['selection']['admissible_row_rank_count'] == len(ranks)
accepted_ranks = {v['selection_rank'] for v in selected}
rejected_ranks = {v['rank'] for v in j['source_only_rejections']}
assert len(accepted_ranks) == 64 and len(rejected_ranks) == len(j['source_only_rejections']) == 0
assert not accepted_ranks & rejected_ranks and accepted_ranks | rejected_ranks == set(range(64))
assert [v['selection_rank'] for v in selected] == list(range(64))
prior_windows = {sha(struct.pack('<32i', *case['original_window_ids'])) for book in prior_books for case in book['cases']}
spool = Path(j['corpus_transport']['path'])
assert spool.stat().st_size == j['corpus_transport']['bytes'] and digest(spool) == j['corpus_transport']['sha256']
with spool.open('rb') as stream:
    assert stream.read(16) == struct.pack('<8sII', b'M464TX01', 1243, 56)
    entries = [struct.unpack('<QQQ32s', stream.read(56)) for _ in range(1243)]
    offset = 69624
    chars_total = 0
    for begin, length, charcount, fullsha in entries:
        assert begin == offset and length >= charcount > 0
        data = stream.read(length)
        assert len(data) == length and sha(data) == fullsha.hex()
        assert len(data.decode('utf-8')) == charcount
        offset += length
        chars_total += charcount
        guard()
    assert not stream.read(1) and offset == spool.stat().st_size == 506604735 and chars_total == 506280896
    excluded_whole = {r['sha256'] for r in j['excluded_row_whole_text_sha256']}
    assert {r['corpus_row'] for r in j['excluded_row_whole_text_sha256']} == union
    assert all(entries[r['corpus_row']][3].hex() == r['sha256'] for r in j['excluded_row_whole_text_sha256'])
    assert not {v['whole_source_utf8_sha256'] for v in selected} & (excluded_whole | all_hashes)
    assert len({v['whole_source_utf8_sha256'] for v in selected}) == 64
    for rejection in j['source_only_rejections']:
        row = ranks[rejection['rank']]
        assert rejection['source_id'] == 'pg19:' + corpus_rel + ':row=' + str(row)
        assert rejection['whole_source_utf8_sha256'] == entries[row][3].hex()
        assert rejection['reason'] == 'less_than8192_characters_or_nontext' and entries[row][2] < 8192
    seen = set()
    windows = tokens_total = 0
    for book_index, book in enumerate(selected):
        row = book['corpus_row']
        begin, length, charcount, fullsha = entries[row]
        assert book['book'] == 128 + book_index and book['role'] == 'development_augmentation'
        assert row == ranks[book['selection_rank']] and row not in union
        source_id = 'pg19:' + corpus_rel + ':row=' + str(row)
        assert book['source_id'] == source_id
        stream.seek(begin)
        data = stream.read(length)
        assert sha(data) == fullsha.hex() == book['whole_source_utf8_sha256']
        text = data.decode('utf-8')
        assert len(text) == charcount == book['source_characters'] and len(text) >= 8192
        excerpt_start = int(sha((seed + '|' + source_id).encode()), 16) % (len(text) - 4096)
        assert book['excerpt_start_character'] == excerpt_start and text[excerpt_start:excerpt_start + 4096] == book['excerpt']
        assert sha(book['excerpt'].encode()) == book['excerpt_utf8_sha256']
        tokenbytes = base64.b64decode(book['excerpt_token_ids_base64_le_i32'], validate=True)
        count = book['excerpt_token_count']
        assert len(tokenbytes) == 4 * count and sha(tokenbytes) == book['excerpt_token_ids_sha256']
        tokens = struct.unpack('<' + str(count) + 'i', tokenbytes)
        tokens_total += count
        assert count >= 512 and len(set(tokens)) == book['excerpt_unique_token_count'] >= 128 and all(2 <= v < 32000 for v in tokens)
        quarter = count // 4
        assert len(book['cases']) == 4
        for case_index, case in enumerate(book['cases']):
            token_start = case_index * quarter + int(sha(f'{seed}|{source_id}|window{case_index}'.encode()), 16) % (quarter - 32 + 1)
            original = list(tokens[token_start:token_start + 32])
            assert case['index'] == case_index and case['excerpt_token_start'] == token_start and case['original_window_ids'] == original
            positions = [3, 10, 17, 24]
            sentinels = [32099, 32098, 32097, 32096]
            masked = {pos for p in positions for pos in (p, p + 1)}
            expected_source = []
            for pos, value in enumerate(original):
                if pos in positions:
                    expected_source.append(sentinels[positions.index(pos)])
                elif pos not in masked:
                    expected_source.append(value)
            expected_source.append(1)
            expected_target = [value for p, s in zip(positions, sentinels) for value in (s, original[p], original[p + 1])] + [32095, 1]
            assert case['span_starts'] == positions and case['masked_spans_ids'] == [original[p:p + 2] for p in positions]
            assert case['source_ids'] == expected_source and case['target_ids'] == expected_target and case['decoder_ids'] == [0] + expected_target[:-1]
            assert case['source_ids_sha256'] == sha(struct.pack('<29i', *expected_source))
            assert case['target_ids_sha256'] == sha(struct.pack('<14i', *expected_target))
            key = sha(struct.pack('<32i', *original))
            assert key not in seen and key not in prior_windows
            seen.add(key)
            windows += 1
            guard()
assert windows == len(seen) == 256 and tokens_total == 68495
progress_path = OUT / 'progress.jsonl'
progress_bytes = progress_path.read_bytes()
progress_lines = progress_bytes.splitlines(keepends=True)
progress = [json.loads(v) for v in progress_lines]
assert {v['pid'] for v in progress} == {23952}
terminal = progress[-1]
assert terminal['admitted'] is True and terminal['selected_books'] == 64
assert terminal['raw_bytes'] == RAW.stat().st_size == 2820431
assert terminal['raw_sha256'] == digest(RAW)
assert terminal['seconds'] <= 300 and terminal['final_parent_OS_peak_bytes'] <= 4 << 30
assert (OUT / 'fatal_native.log').stat().st_size == 0
assert len([v for v in progress if 'candidate_rank' in v]) == 64
assert RAW.stat().st_size <= 8 << 20
assert j['resource']['main_seconds_before_raw_write'] <= terminal['seconds']
assert j['resource']['peak_process_bytes_before_raw_write'] <= terminal['final_parent_OS_peak_bytes']
assert terminal['final_bytes_hashed'] == j['resource']['bytes_hashed_before_raw_write'] + RAW.stat().st_size
out_bytes = sum(p.stat().st_size for p in OUT.iterdir())
assert out_bytes == terminal['OUT_bytes_before_terminal_progress_row'] + len(progress_lines[-1])
assert out_bytes <= 2 << 20
assert not RAW.with_suffix('.failure.json').exists()
events = json.loads(EVENTS.read_text(encoding='utf-8'))
assert events['main_pid'] == 23952 and events['query_available'] is True and events['query_error'] is None
assert events['matching_main_events'] == [] and events['returned_event_count'] == 0
assert events['source_process_instance'] == j['process_instance']
qstart = datetime.fromisoformat(events['query_start_utc']).timestamp()
qend = datetime.fromisoformat(events['query_end_utc']).timestamp()
assert qstart <= j['process_instance']['create_time_unix']
assert qstart <= datetime.fromisoformat(j['start_utc']).timestamp() <= progress_path.stat().st_ctime
assert qend >= datetime.fromisoformat(terminal['end_utc']).timestamp()
assert qend >= progress_path.stat().st_mtime
instance = j['process_instance']
pid_reuse = None
try:
    current = psutil.Process(instance['pid'])
    live = {'pid': current.pid, 'create_time_unix': current.create_time(),
            'name': current.name(), 'executable': current.exe()}
    assert not (abs(live['create_time_unix'] - instance['create_time_unix']) < .01
                and live['name'].lower() == instance['name'].lower()
                and Path(live['executable']).resolve() == Path(instance['executable']).resolve()), live
    pid_reuse = live
except psutil.NoSuchProcess:
    pass
own = {psutil.Process().pid, *(p.pid for p in psutil.Process().parents())}
daemons = []
for p in psutil.process_iter(['name', 'cmdline']):
    if p.pid in own:
        continue
    try:
        name = (p.info['name'] or '').lower()
        argv = p.info['cmdline'] or []
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
            daemons.append(p.pid)
            continue
        assert not (name.startswith('python') or (name.startswith('meth') and name.endswith('.exe')) or name == 'clang.exe'), (p.pid, name)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
for rel, v in bindings['preserved_unrelated_files'].items():
    assert digest(ROOT / rel) == v['sha256']
assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines() == bindings['tracked_status']
assert not any(name.partition('.')[0] in {'tokenizers', 'numpy', 'torch', 'transformers', 'pyarrow', 'pandas', 'sklearn'}
               for name in __import__('sys').modules)
audit = {
    'experiment': 'METH470-retained-source-token-role-runtime-provenance-audit',
    'raw_sha256': digest(RAW), 'audit_helper_sha256': digest(__file__),
    'gates': {
        'source_protocol_engine_physicalHEAD_and_direct_records_runtime_exact': True,
        'all36_actual_package_code_native_files_and_loaded_paths_SHA_bound': True,
        'ALL1863_prior_files_physical_canonical_SHA_exclusions_hash_membership_rederived': True,
        'ALL96_original_case_bridge_digests_exact': True,
        'ALL1243_cached_UTF8_rows_offsets_charcounts_SHA_EOF_exact': True,
        'ALL64_selected_actual_source_coordinates_hashes_prior_disjointness_exact': True,
        'ALL256_token_windows_masks_fields_digests_dev_only_roles_rank_prefix_rederived': True,
        'all_old128_roles_and_original64_validation_manifest_capture_records_unchanged': True,
        'actual_terminal_empty_fault_log_process_instance_absent_Windows_query_zero_no_jobs': True,
        'exact_afterwrite_resources_RAW_OUT_terminal_row_and_unrelated3_preserved': True,
    },
    'books': 64, 'contexts': windows, 'excerpt_token_count': tokens_total,
    'source_rows_excluded': len(union), 'raw_bytes': RAW.stat().st_size,
    'output_bytes': {'OUT_only': out_bytes, 'raw_outside_OUT': RAW.stat().st_size,
                     'combined': out_bytes + RAW.stat().st_size,
                     'terminal_progress_row': len(progress_lines[-1])},
    'main_terminal': {'tool_session': 56339, 'tool_exit_code': 0, 'actual_instance': instance,
                      'PID_current_reuse_if_any': pid_reuse, 'terminal_progress': terminal},
    'windows_event_query': events, 'preserved_daemon_pids': daemons,
    'retained_metadata_files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                               for p in [*sorted(OUT.iterdir()), EVENTS,
                                         ROOT / 'benchmarks/native_expert_scaling/meth470_windows_terminal.ps1']],
    'seconds_before_RET_write': time.monotonic() - start, 'OS_peak_bytes_before_RET_write': peak,
    'bytes_hashed_before_RET_write': hashed,
    'hard_audit_seconds': 300, 'hard_audit_OS_peak_bytes': 512 << 20,
    'auxiliary_readout_note': 'An ad hoc display used locale-default cp1252 and failed decoding the valid UTF8 raw; corrected explicit UTF8 display. No scientific main or audit rerun, no raw/source changes.',
    'scope': 'Independent metadata/source/token-array audit; no main/codec/tokenization/Arrow/framework/model/capture/fit rerun. Original96 decode equivalence is the frozen MAIN control; audit independently reconstructs ALL256 native arrays from retained full excerpt codes. Old467/469 roles/raw/RET are SHA-bound; old469 native files are not rehashed here and must be refreshed by separate471 capture. Windows actual invocation query available/zeroevents. Source admission only, no readiness/quality/rate/useful-n result.',
}
guard()
with TARGET.open('xb') as stream:
    stream.write((json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
                 .replace('\n', '\r\n').encode('utf-8'))
guard()
print(json.dumps({'sha256': digest(TARGET), 'gates': audit['gates'],
                  'final_seconds_including_RET_write_hash': time.monotonic() - start,
                  'final_OS_peak_bytes': peak, 'raw_bytes': audit['raw_bytes'],
                  'output_bytes': audit['output_bytes'], 'excerpt_token_count': tokens_total}))
