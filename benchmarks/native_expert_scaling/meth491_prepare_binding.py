"""Fresh metadata binding before the sole mass witness inquiry."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST = DOC / 'meth491_binding.json'
assert not DEST.exists()
START = time.monotonic()

def item(path, expected=None):
    path = Path(path).resolve(); stat = path.stat(); h = hashlib.sha256()
    with path.open('rb') as stream:
        while payload := stream.read(4 << 20):
            h.update(payload)
            assert time.monotonic()-START <= 120
    assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
    result = {'path': str(path), 'bytes': stat.st_size, 'sha256': h.hexdigest()}
    if expected:
        assert result['sha256'] == expected, str(path)
    return result

def head(path):
    path = Path(path)
    assert path.read_bytes().replace(b'\r\n', b'\n') == subprocess.check_output(
        ['git', 'show', 'HEAD:'+path.relative_to(ROOT).as_posix()], cwd=ROOT).replace(b'\r\n', b'\n')

old = DOC / 'meth490_r1_binding.json'
old_record = item(old, 'f63189dd38d4690386eca97509301440bd150731ee47931a180dc7a4c58a5747')
binding = json.loads(old.read_bytes())
catalog = {v['path']: item(v['path'], v['sha256']) for v in binding['catalog']}
mass = {}
for key, name, expected in (
    ('raw', 'meth490_r1_root_mass_result.json', '30760087a7b3866255a91df8ded30d47c96c72099cb9bbc72789d0e7bcbbdaf5'),
    ('retention', 'RETENTION_490_R1_20261006.json', 'bb3c4e7d2c741848564373454088e808a065f8eb55040920bdfcafddb98f497b'),
    ('admission', 'ADMISSION_490_R1_20261006.json', '693541acf5e6b075c9ed2ec38853a595e49c91a4677c840d176cfa56e8f53745')):
    path = DOC / name; head(path); mass[key] = item(path, expected)
raw = json.loads(Path(mass['raw']['path']).read_bytes())
ret = json.loads(Path(mass['retention']['path']).read_bytes())
admission = json.loads(Path(mass['admission']['path']).read_bytes())
assert all(raw['gates'].values()) and all(ret['gates'].values()) and all(admission['gates'].values())
assert admission['main_completed'] and ret['raw']['sha256'] == mass['raw']['sha256']
for record in raw['output_inventory']:
    catalog[record['path']] = item(record['path'], record['sha256'])
mass['diagnostics'] = next(v for v in raw['output_inventory'] if Path(v['path']).name == 'diagnostics.bin')
new = sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth491*')) + [
    DOC / 'METH_491_ROOT_MASS_INTERVAL_PROTOCOL_20261006.md',
    DOC / 'METH_491_INTERVAL_OPERATOR_SOURCES_20261006.md']
for path in new:
    head(path)
new = list(map(item, new))
python_base = Path(next(path for path in binding['runtime']['files'] if path.endswith('python312.dll'))).parent
decimal = [item(python_base / 'DLLs/_decimal.pyd'), item(python_base / 'Lib/decimal.py'),
           item(python_base / 'Lib/_pydecimal.py')]
records = [old_record, *mass.values()]
for value in new + decimal + records:
    catalog[value['path']] = value
binding['scientific'] += new
fault = DOC / 'meth491_first_metadata_patch_fault.json'; head(fault); fault = item(fault)
catalog[fault['path']] = fault
binding['records'] += [old_record, mass['raw'], mass['retention'], mass['admission'], fault]
binding.update(experiment='METH491 fixed small affine mass-interval dual witnesses',
    inherited_output_bytes=0, mass_source=mass, decimal_runtime=decimal, catalog=list(catalog.values()),
    freeze_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    preparation_seconds=time.monotonic()-START, new_native_calls=0, new_optimizer_updates=0)
with DEST.open('xb') as stream:
    stream.write((json.dumps(binding, indent=2, allow_nan=False)+'\n').replace('\n', '\r\n').encode())
print(json.dumps({'binding': str(DEST), 'sha256': item(DEST)['sha256'], 'seconds': binding['preparation_seconds']}))
