"""Fresh SHA binding of used513 retained geometry and isolated511 runtime only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST = DOC / 'meth513_r1_binding.json'
assert not DEST.exists() and not DEST.with_suffix('.failure.json').exists()
START = time.monotonic()
PROC = psutil.Process()
PROC.cpu_affinity([10])
PEAK = HASHED = 0
CATALOG = {}


def item(path, expected=None):
    global PEAK, HASHED
    p = Path(path).resolve()
    s = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        while d := f.read(8 << 20):
            h.update(d)
            HASHED += len(d)
            PEAK = max(PEAK, PROC.memory_info().peak_wset)
            assert PEAK <= 512 << 20 and time.monotonic() - START <= 180
    assert (p.stat().st_size, p.stat().st_mtime_ns) == (s.st_size, s.st_mtime_ns)
    if expected:
        assert h.hexdigest() == expected, str(p)
    v = {'path': str(p), 'bytes': s.st_size, 'sha256': h.hexdigest()}
    CATALOG[v['path']] = v
    return v


try:
    a = json.loads((DOC / 'ADMISSION_504_20261006.json').read_bytes())
    assert a['main_completed'] and all(a['gates'].values())
    records = [item(DOC / 'ADMISSION_504_20261006.json')]
    for k in ['raw', 'retention', 'binding']:
        records.append(item(a[k]['path'], a[k]['sha256']))
    old = json.loads(Path(a['binding']['path']).read_bytes())
    raw = json.loads(Path(a['raw']['path']).read_bytes())
    data = {k: item(old['data'][k]['path'], old['data'][k]['sha256']) for k in ['uid', 'occurrences', 'hidden', 'codes', 'scales']}
    names = ('seed_codes_', 'centre_dots_', 'row_norm2_', 'query_dots_')
    for v in raw['output_inventory']:
        name = Path(v['path']).name
        if name.startswith(names) or name == 'query_norm2.npy':
            data[name] = item(v['path'], v['sha256'])
    rb = item(DOC / 'meth511_r9_binding.json', '8ac4e645149554677cbd6100334e1ca4d5aad5b3a481dc5e87fa2f9cdbb3987b')
    base = json.loads(Path(rb['path']).read_bytes())
    fallback = base['source_payload']
    fs = Path(fallback['path']).stat()
    retained = next(v for v in base['catalog'] if v['path'] == fallback['path'])
    assert fs.st_size == retained['bytes'] == fallback['bytes'] and fs.st_mtime_ns == retained['mtime_ns']
    runtime = []
    for v in base['catalog']:
        path = v['path'].replace('\\', '/').lower()
        if any(s in path for s in ['/numpy/', '/numpy.libs/', '/psutil/']) or path.endswith('/threadpoolctl.py') or ('/python312/' in path and Path(path).suffix in ['.py', '.pyd', '.dll', '.exe']) or path == base['executable'].replace('\\', '/').lower():
            runtime.append(item(v['path'], v['sha256']))
    assert runtime and any('/numpy.libs/' in v['path'].replace('\\', '/').lower() for v in runtime)
    science = []
    paths = sorted((ROOT / 'benchmarks/native_expert_scaling').glob('meth513*')) + [DOC / 'METH_513_ADAPTIVE_CERTIFICATE_PROTOCOL_20261007.md']
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        assert p.read_bytes().replace(b'\r\n', b'\n') == subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT).replace(b'\r\n', b'\n')
        science.append(item(p))
    for rel, sha in base['preserved'].items():
        item(ROOT / rel, sha)
    assert str(Path(sys.executable).resolve()) == base['executable'] and sys.version == base['python']
    value = {'experiment': 'METH513 fixed64 exact adaptive row-certificate union',
             'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
             'packages': {k: base['packages'][k] for k in ['numpy', 'psutil', 'threadpoolctl']},
             'catalog': list(CATALOG.values()), 'data': data, 'retained504_raw': a['raw'],
             'runtime_files': runtime, 'scientific': science, 'preserved': base['preserved'],
             'retained_complete_original_fallback_payload': {**fallback, 'stat_bytes': fs.st_size, 'stat_mtime_ns': fs.st_mtime_ns, 'full_SHA_scope': 'Previously qualified full SHA retained; current stat equality only, no fresh payload rehash or source-function query.'},
             'process_instance': {'pid': PROC.pid, 'create_time_unix': PROC.create_time()},
             'resource_before_serialization': {'seconds': time.monotonic() - START, 'OS_peak_bytes': PEAK, 'bytes_hashed': HASHED},
             'limits': {'binding': [180, 512 << 20], 'main': [180, 2 << 30, 64 << 20], 'audit': [180, 2 << 30, 16 << 20]},
             'new_model_native_source_function_calls': 0,
             'scope': 'Fresh SHA of used retained geometry only; prior qualified WI/query dots reused. No full payload/runtime-Torch/corpus rehash.'}
    with DEST.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    v = item(DEST)
    print(json.dumps({'binding': v, 'seconds': time.monotonic() - START, 'OS_peak_bytes': PEAK, 'bytes_hashed': HASHED}), flush=True)
except BaseException:
    with DEST.with_suffix('.failure.json').open('x', encoding='utf8') as f:
        json.dump({'traceback': traceback.format_exc(), 'seconds': time.monotonic() - START,
                   'OS_peak_bytes': PEAK, 'bytes_hashed': HASHED}, f, indent=2)
        f.write('\n')
    raise
