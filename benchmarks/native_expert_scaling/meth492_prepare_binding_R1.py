"""Numbered metadata-only origin correction; combined builders90s/256MiB."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import psutil
import _struct

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST = DOC / 'meth492_r1_binding.json'
FAIL = DEST.with_suffix('.failure.json')
assert not DEST.exists() and not FAIL.exists()
FIRST = json.loads((DOC/'meth492_binding.failure.json').read_bytes())
FIRST_REG = json.loads((DOC/'meth492_binding_first_registration.json').read_bytes())
assert FIRST_REG['actual_exit_code'] == 1 and FIRST_REG['no_main_control_or_donor_extrema_started']
assert hashlib.sha256((DOC/'meth492_binding.failure.json').read_bytes()).hexdigest() == FIRST_REG['first_fault_sha256']
LIMIT = 90-FIRST['seconds']; assert LIMIT > 0
assert _struct.__spec__.origin == FIRST_REG['observed_struct_origin'] == 'built-in'
START = time.monotonic()
PROC = psutil.Process(); PROC.cpu_affinity([0])
PEAK = 0; HASHED = 0

def guard():
    global PEAK
    PEAK = max(PEAK, PROC.memory_info().peak_wset)
    assert PEAK <= 256 << 20 and time.monotonic()-START <= LIMIT

def item(path, expected=None):
    global HASHED
    path = Path(path).resolve(); before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as stream:
        while payload := stream.read(4 << 20):
            h.update(payload); HASHED += len(payload); guard()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    result = {'path': str(path), 'bytes': before.st_size, 'sha256': h.hexdigest()}
    if expected:
        assert result['sha256'] == expected, str(path)
    return result

def head(path):
    path = Path(path)
    saved = subprocess.check_output(['git', 'show', 'HEAD:'+path.relative_to(ROOT).as_posix()], cwd=ROOT, timeout=5)
    assert path.read_bytes().replace(b'\r\n', b'\n') == saved.replace(b'\r\n', b'\n')
    guard()

try:
    inherited = item(DOC / 'meth491_binding.json', 'a2e567a26d1583fb2295fbc5c16d5f0a9e7c6fecc8f645b0efbbc310576bdf80')
    binding = json.loads(Path(inherited['path']).read_bytes())
    catalog = {v['path']: item(v['path'], v['sha256']) for v in binding['catalog']}
    records = [inherited]
    for name in ('meth492_binding.failure.json','meth492_binding_first_registration.json'):
        path=DOC/name; head(path); records.append(item(path))
    fault = DOC / 'meth492_first_metadata_patch_fault.json'
    head(fault); records.append(item(fault))
    for name, expected in (
        ('meth491_mass_interval_result.json', '0bf90890796dfe7df62cd6adac4dbc8814b2538fd50c164a745a2bfb4e05a28f'),
        ('RETENTION_491_20261006.json', 'a8377ca820f81a89984929a4756721870ad2050f81e8becb29246477b4871d3c'),
        ('ADMISSION_491_20261006.json', '4810c933c4a2a6de94653fb75935ac625c66e66e24a5ffbdb330e8dbe7158338'),
        ('meth491_admitted_norm_algebra.json', '743baeed9e4d900adbded4833f83c9704bf8874362c9d62e5d4ed250651fa15c')):
        path = DOC / name; head(path); value = item(path, expected); records.append(value)
        parsed = json.loads(path.read_bytes())
        if 'gates' in parsed:
            assert all(parsed['gates'].values())
        for output in parsed.get('output_inventory', []):
            catalog[output['path']] = item(output['path'], output['sha256'])
    new_paths = sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth492*')) + [
        DOC / 'METH_492_SELECTED_AMPLITUDE_ELIGIBILITY_20261006.md',
        DOC / 'METH_492_SELECTED_AMPLITUDE_PROTOCOL_20261006.md',
        DOC / 'METH_492_BINDING_R1_PROTOCOL_20261006.md']
    new = []
    for path in new_paths:
        head(path); new.append(item(path))
    base = Path(next(p for p in binding['runtime']['files'] if p.endswith('python312.dll'))).parent
    arithmetic = [item(base/'Lib/fractions.py'), item(base/'Lib/struct.py'), item(base/'python312.dll')]
    for value in new + records + arithmetic:
        catalog[value['path']] = value
    binding['scientific'] += new
    binding['records'] += records
    binding.update(experiment='METH492 exact constant selected-amplitude eligibility',
        inherited_output_bytes=0, catalog=list(catalog.values()), rational_runtime=arithmetic,
        freeze_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip(),
        preparation_seconds=time.monotonic()-START,
        preparation_peak_bytes=PEAK, preparation_hashed_bytes=HASHED,
        builder_wall_limit_seconds=LIMIT, builder_total_wall_limit_seconds=90,
        first_builder_fault_seconds=FIRST['seconds'], struct_builtin_origin=_struct.__spec__.origin, builder_host_limit_bytes=256 << 20,
        new_native_calls=0, new_optimizer_updates=0)
    guard()
    with DEST.open('xb') as stream:
        stream.write((json.dumps(binding, indent=2, allow_nan=False)+'\n').replace('\n','\r\n').encode())
    print(json.dumps({'binding': str(DEST), 'sha256': item(DEST)['sha256'],
                      'seconds': time.monotonic()-START, 'peak_bytes': PEAK, 'hashed_bytes': HASHED}))
except BaseException as exc:
    with FAIL.open('xb') as stream:
        stream.write((json.dumps({'error': str(exc), 'traceback': traceback.format_exc(),
            'seconds': time.monotonic()-START, 'peak_bytes': PEAK, 'hashed_bytes': HASHED}, indent=2)+'\n').encode())
    raise
