"""Metadata R1: source469 reuses468 binding; original first helper stays immutable."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import sys
import time

ROOT = Path.cwd()
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
BIND = DOC / 'meth479_prospective_bindings.json'
assert not BIND.exists()
START = time.monotonic()
cache = {}

def item(path, expected=None):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            while data := stream.read(4 << 20):
                h.update(data)
        cache[key] = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': h.hexdigest()}
    if expected is not None:
        assert cache[key]['sha256'] == expected, str(path)
    return cache[key]

def head(path):
    path = Path(path).resolve()
    rel = path.relative_to(ROOT).as_posix()
    assert path.read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel]), rel

old_bind_path = DOC / 'meth478_prospective_bindings.json'
item(old_bind_path, '07dc05c3e34df0570880b6c20315008614ffeb43061a109cbcc271a8aa6d6894')
old = json.loads(old_bind_path.read_bytes())
runtime = old['runtime']
assert sys.version == runtime['python'] and Path(sys.executable).resolve() == Path(runtime['executable']).resolve()
for path, entry in runtime['files'].items():
    item(path, entry['sha256'])
for package in runtime['packages'].values():
    for path, entry in package['files'].items():
        item(path, entry['sha256'])

records = [old_bind_path, DOC / 'METH_479_FIRST_METADATA_ADMISSION_FAULT_20261005.md']
raws = {}
inventories = []
for n, name in ((469, 'meth469_switch_native_domain_capture_result.json'), (471, 'meth471_switch_development_capture_result.json')):
    raw_path = DOC / name
    ret_path = DOC / f'RETENTION_{n}_20261005.json'
    binding_path = DOC / ('meth468_prospective_bindings.json' if n == 469 else 'meth471_prospective_bindings.json')
    records.extend((raw_path, ret_path, binding_path))
    r = json.loads(raw_path.read_bytes())
    ret = json.loads(ret_path.read_bytes())
    assert all(r['gates'].values()) and all(ret['gates'].values())
    item(raw_path, ret['raw_sha256'])
    item(binding_path, r['source_binding_sha256'])
    inventories.extend(ret['files'])
    raws[n] = r
assert len(inventories) == 6224
inventory = [item(v['path'], v['sha256']) for v in inventories]
assert sum(v['bytes'] for v in inventory) == 4566198260 + 2329258130
assert len({v['path'] for v in inventory}) == 6224
for n in (469, 471):
    actual = ROOT / 'results/native_expert_scaling' / ('meth469_switch_native_domain_capture' if n == 469 else 'meth471_switch_development_capture')
    assert {str(p.resolve()) for p in actual.iterdir() if p.is_file()} == {v['path'] for v in inventory if Path(v['path']).parent == actual}
roles = raws[471]['immutable_book_roles']
assert len(roles) == 192 and [v['book'] for v in roles] == list(range(192))
for row in roles:
    book = row['book']
    assert row['role'] == ('development' if book < 64 else 'diagnostic_validation' if book < 128 else 'development_augmentation')
    assert row['split'] == int(64 <= book < 128)
tasks = []
for n in (469, 471):
    r = raws[n]
    assert len(r['cases']) == (512 if n == 469 else 256)
    for case in r['cases']:
        book = case['book']
        role = 0 if book < 64 else 1 if book < 128 else 2
        assert case['role'] == roles[book]['role']
        for mode, label in ((0, 'teacher'), (1, 'natural')):
            row = r['commands'][case['modes'][label]]
            assert row['returncode'] == 0 and not row['negative'] and row['mode'] == label
            trace, whole = Path(row['trace_path']), Path(row['whole_output_path'])
            item(trace, row['trace_sha256'])
            item(whole, row['whole_output_sha256'])
            with whole.open('rb') as stream:
                magic, s, t, d, enc, dec, vocab, count = struct.unpack('<8s7I', stream.read(36))
            assert magic == b'SWR32O01' and (s, d, enc, dec, vocab) == (29, 768, 12, 12, 32128) and count == 6 * (s + t)
            assert (mode == 0 and t == 14) or (mode == 1 and 1 <= t <= 64)
            assert whole.stat().st_size == 36 + 4 * (14 * s * 768 + 14 * t * 768 + t * 32128) + 12 * count
            with trace.open('rb') as stream:
                assert stream.read(16) == struct.pack('<8sII', b'SWRTA001', 128, 768)
                prefix = stream.read(180 * (8 + 4 * (768 + 128)))
            if not mode:
                teacher_prefix = prefix
            else:
                assert prefix == teacher_prefix
            assert trace.stat().st_size == 16 + count * (8 + 4 * (768 + 128))
            tasks.append({'ordinal': len(tasks), 'book': book, 'case': case['case'], 'mode': mode, 'role': role,
                          's': s, 't': t, 'queries': count, 'trace': str(trace), 'whole': str(whole),
                          'trace_sha256': row['trace_sha256'], 'whole_sha256': row['whole_output_sha256']})
assert len(tasks) == 1536 and sum(v['queries'] for v in tasks) == 387036
assert all((v['book'], v['case'], v['mode']) == (i // 8, (i // 2) % 4, i % 2) for i, v in enumerate(tasks))

raw478_path, ret478_path = DOC / 'meth478_grouped_router_result.json', DOC / 'RETENTION_478_20261005.json'
item(raw478_path, '61cae4d4218e0bd9590be7baf1f49de3941d7b0a43eb4f805236e482827c42e4')
item(ret478_path, '1ddd505b00185e53b4cffefac4e96e40e4f65319cd8718849d7e223261a6bd00')
records.extend((raw478_path, ret478_path))
r478 = json.loads(raw478_path.read_bytes())
assert all(json.loads(ret478_path.read_bytes())['gates'].values())
members = [item(v['path'], v['sha256']) for v in r478['trees'] if v['n'] == 128]
assert len(members) == 12 and [v['bank'] for v in r478['trees'] if v['n'] == 128] == list(range(12))
artifact = raws[469]['artifact']
assert artifact == raws[471]['artifact']
item(artifact['payload'], artifact['sha256'])
item(artifact['manifest'], artifact['manifest_sha256'])
export_path = DOC / 'meth380_switch_base128_export_result.json'
records.append(export_path)
export = json.loads(export_path.read_bytes())
assert export['artifact'] == artifact and all(export['gates'].values()) and len(export['tensors']) == 3320
weights = [v for v in old['weights'] if v['n'] == 128]
assert len(weights) == 12 and [v['bank'] for v in weights] == list(range(12))
with Path(artifact['manifest']).open('rb') as stream:
    assert stream.read(8) == b'SWI8A001'
    config = struct.unpack('<13IfII', stream.read(64))
    assert config[0] == 768 and config[6] == 128 and config[14:] == (1, 3320)
    def text():
        size, = struct.unpack('<I', stream.read(4))
        return stream.read(size).decode()
    assert text() == str(Path(artifact['payload']).resolve())
    for name, descriptor in sorted(export['tensors'].items()):
        assert text() == name
        shape = descriptor['shape']
        assert struct.unpack('<5I3Q', stream.read(44)) == (0, len(shape), shape[0], shape[1] if len(shape) == 2 else 1,
                                                         descriptor['encoding'], descriptor['offset'], descriptor['scale_offset'], descriptor['elements'])
    assert not stream.read(1)
for weight in weights:
    descriptor = export['tensors'][weight['name']]
    assert descriptor['shape'] == [128, 768] and descriptor['encoding'] == 0
    assert [weight[k] for k in ('offset', 'bytes', 'sha256')] == [descriptor[k] for k in ('offset', 'bytes', 'sha256')]
    with Path(weight['payload']).open('rb') as stream:
        stream.seek(weight['offset'])
        data = stream.read(weight['bytes'])
    assert hashlib.sha256(data).hexdigest() == weight['sha256']

compile_assets = [item(v['path'], v['sha256']) for v in old['compile_assets']]
system_files = [item(v['path'], v['sha256']) for v in old['system_files']]
helpers = [ROOT / 'benchmarks/phase60/engine.c', ROOT / 'benchmarks/native_expert_scaling/meth479_prepare_bindings.py'] + [ROOT / 'benchmarks/native_expert_scaling' / name for name in (
    'meth393_switch_router_audit.c', 'meth393_switch_router_trace.h', 'meth388_switch_thread_binding.h',
    'meth469_switch_native_domain_capture.py', 'meth471_switch_development_capture.py', 'meth471_retention_audit.py')]
for path in records + helpers:
    head(path)
preserved = old['preserved_unrelated_files']
for rel, value in preserved.items():
    item(ROOT / rel, value['sha256'])
status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines()
assert all(v[:3] == ' M ' and v[3:] in preserved for v in status), status
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
science = [ROOT / 'benchmarks/native_expert_scaling' / name for name in ('meth479_source_router.c', 'meth479_supervision_math.py', 'meth479_windows_terminal.ps1')]
science.append(DOC / 'METH_479_ROUTING_SUPERVISION_PROTOCOL_20261005.md')
bound = (3720 + 3560 + 32) * 248796 + (52 + 72) * 387036 + 5 * 24
assert bound + (8 << 20) <= 2 << 30
binding = {'purpose': 'ONE complete original-source support/native-mass supervision, no fit/candidate claim',
           'head_at_preparation': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'runtime': runtime, 'records': [item(p) for p in records], 'helpers': [item(p) for p in helpers],
           'scientific_files': [item(p) for p in science], 'source_inventory': inventory,
           'compiler': item(old['compiler']['path'], old['compiler']['sha256']), 'compile_assets': compile_assets,
           'system_files': system_files, 'artifact': artifact, 'manifest_entries': 3320, 'weights': weights,
           'membership_sources': members, 'tasks': tasks, 'immutable_book_roles': roles,
           'preserved_unrelated_files': preserved, 'tracked_status': status, 'preparation_helper': item(__file__),
           'query_count': 387036, 'streams': 1536, 'unique_upper_bound': 248796,
           'binary_upper_bound': bound, 'preparation_seconds': time.monotonic() - START}
with BIND.open('xb') as stream:
    stream.write((json.dumps(binding, indent=2) + '\n').replace('\n', '\r\n').encode())
print(json.dumps({'binding': str(BIND), 'sha256': item(BIND)['sha256'], 'streams': len(tasks),
                  'source_files': len(inventory), 'source_bytes': sum(v['bytes'] for v in inventory),
                  'unique_upper_bound': 248796, 'binary_upper_bound': bound, 'seconds': time.monotonic() - START}))
