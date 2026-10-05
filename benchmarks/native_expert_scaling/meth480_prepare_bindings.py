"""Metadata only: pin the complete admitted supervision/runtime and new freezes."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import sys
import time

START = time.monotonic()
ROOT = Path.cwd()
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST = DOC / 'meth480_prospective_bindings.json'
assert not DEST.exists()
cache = {}

def item(path, expected=None):
    p = Path(path).resolve()
    stat = p.stat()
    key = (str(p), stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with p.open('rb') as stream:
            while data := stream.read(4 << 20):
                h.update(data)
                assert time.monotonic() - START <= 180
        cache[key] = {'path': str(p), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': h.hexdigest()}
    if expected:
        assert cache[key]['sha256'] == expected, str(p)
    return cache[key]

def head(path):
    p = Path(path).resolve()
    rel = p.relative_to(ROOT).as_posix()
    assert p.read_bytes() == subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path=' + rel, 'HEAD:' + rel]), rel

prior_path = DOC / 'meth479_prospective_bindings.json'
prior_record = item(prior_path, '0523b1822b84f6e11b40025b3c677116042a06853f4094375d271df73a4016ac')
prior = json.loads(prior_path.read_bytes())
assert sys.version == prior['runtime']['python'] and Path(sys.executable).resolve() == Path(prior['runtime']['executable']).resolve()
for p, v in prior['runtime']['files'].items():
    item(p, v['sha256'])
for package in prior['runtime']['packages'].values():
    for p, v in package['files'].items():
        item(p, v['sha256'])
for key in ('records', 'helpers', 'scientific_files', 'source_inventory', 'compile_assets', 'system_files', 'membership_sources'):
    for v in prior[key]:
        assert item(v['path'], v['sha256'])['bytes'] == v['bytes']
for key in ('compiler', 'preparation_helper'):
    item(prior[key]['path'], prior[key]['sha256'])
for v in prior['records'] + prior['helpers'] + prior['scientific_files']:
    head(v['path'])
artifact = prior['artifact']
item(artifact['payload'], artifact['sha256'])
item(artifact['manifest'], artifact['manifest_sha256'])
records = []
names = ('meth479_r2_routing_supervision_result.json', 'RETENTION_479_R3_20261005.json',
         'meth479_first_scientific_failure_inventory.json', 'meth479_r1_first_failure_inventory.json',
         'meth479_r2_first_audit_failure_inventory.json', 'meth479_routing_supervision_result.failure.json',
         'meth479_r1_routing_supervision_result.failure.json', 'meth479_r2_first_audit_failure.json',
         'METH_479_SUPERVISION_RESULT_20261005.md', 'METH_479_ALGEBRA_REASSESSMENT_AND_NEXT_20261005.md',
         'METH_479_R3_RETENTION_PROTOCOL_20261005.md')
for name in names:
    head(DOC / name)
    records.append(item(DOC / name))
r = json.loads((DOC / names[0]).read_bytes())
ret = json.loads((DOC / names[1]).read_bytes())
assert item(DOC / names[0], 'c03efee574a62743ac214f0c94213f41e2cbb34db0b6db86ddd18ebab53d9200')
assert item(DOC / names[1], '86c57ea27d948a366344466acb4eef86944c65899c1876ec9a630964aac28d3d')
assert all(r['gates'].values()) and all(ret['gates'].values())
assert r['unique_inputs'] == ret['unique_inputs_audited'] == 238872
assert ret['queries_audited'] == 387036 and ret['decision'] == 'SUPERVISION_DATASET_ADMITTED_NOT_A_TRAINED_MODEL_OR_ACCURACY_RESULT'
old_inventory = json.loads((DOC / names[2]).read_bytes())['records']
data_files = {}
source = ROOT / 'results/native_expert_scaling/meth479_routing_supervision'
for name in ('unique_inputs.bin', 'ownership.bin', 'query_links.bin', 'source_fields.bin', 'membership.json'):
    p = source / name
    expected = next(v for v in old_inventory if Path(v['path']).resolve() == p.resolve())
    data_files[name] = item(p, expected['sha256'])
target = next(v for v in r['output_inventory'] if Path(v['path']).name == 'group_targets.bin')
data_files['group_targets.bin'] = item(target['path'], target['sha256'])
for name, magic, width, reserved, count in (
    ('unique_inputs.bin', b'M479UNI1', 3720, 768, 238872), ('ownership.bin', b'M479OWN1', 32, 0, 238872),
    ('query_links.bin', b'M479LNK1', 52, 0, 387036), ('source_fields.bin', b'M479SRC1', 72, 0, 387036),
    ('group_targets.bin', b'M479TGT1', 3560, 127, 238872)):
    v = data_files[name]
    assert v['bytes'] == 24 + count * width
    with Path(v['path']).open('rb') as stream:
        assert stream.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
auxiliary = []
for p, expected in ((ROOT / 'benchmarks/native_expert_scaling/meth479_r3_retention_audit.py', ret['audit_source_sha256']),
                    (ROOT / 'results/native_expert_scaling/meth479_r2_windows_terminal.json', ret['windows_terminal_sha256']),
                    (ROOT / 'results/native_expert_scaling/meth479_windows_terminal.json', ret['original_windows_terminal_sha256'])):
    auxiliary.append(item(p, expected))
head(auxiliary[0]['path'])
scientific = []
for name in ('meth480_transfer_math.py', 'meth480_learned_router.c', 'meth480_windows_terminal.ps1', 'meth480_prepare_bindings.py'):
    p = ROOT / 'benchmarks/native_expert_scaling' / name
    head(p)
    scientific.append(item(p))
proto = DOC / 'METH_480_LEARNED_ROUTER_PROTOCOL_20261005.md'
head(proto)
scientific.append(item(proto))
for rel, v in prior['preserved_unrelated_files'].items():
    assert item(ROOT / rel, v['sha256'])['bytes'] == v['bytes']
status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).splitlines()
assert all(v[:3] == ' M ' and v[3:] in prior['preserved_unrelated_files'] for v in status), status
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
assert (ROOT / '.gitattributes').read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:.gitattributes'])
assert not ({'numpy', 'torch', 'scipy', 'transformers'} & set(sys.modules))
value = {'experiment': 'METH480 metadata-only prospective full supervision/source/runtime binding',
         'prior_binding': prior_record, 'prior': prior, 'records': records, 'data_files': data_files,
         'auxiliary': auxiliary, 'scientific_files': scientific, 'unique_inputs': 238872, 'queries': 387036,
         'development_unique': 159414, 'validation_unique': 79458, 'steps': 32, 'support_prototypes': 2,
         'mass_prototypes': 16, 'new_prediction_bytes': 24 + 238872 * 372, 'model_bytes_per_bank': 835712,
         'parameter_count_per_bank': 3344, 'head_at_preparation': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         'metadata_seconds': time.monotonic() - START, 'file_bytes_hashed': sum(v['bytes'] for v in cache.values())}
with DEST.open('xb') as stream:
    stream.write((json.dumps(value, indent=2) + '\n').replace('\n', '\r\n').encode())
print(json.dumps({'binding': str(DEST), 'sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
                  'seconds': time.monotonic() - START, 'file_bytes_hashed': value['file_bytes_hashed']}), flush=True)
