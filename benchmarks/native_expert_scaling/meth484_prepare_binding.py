"""Metadata-only preparation; no manifest parsing, tensor values or GPU imports."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
DEST = DOC / 'meth484_source_binding.json'
START = time.monotonic()
assert not DEST.exists() and sys.flags.optimize == 0


def item(path, expected=None):
    p = Path(path).resolve(); before = p.stat(); h = hashlib.sha256()
    with p.open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block); assert time.monotonic() - START < 90
    after = p.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    if expected:
        assert h.hexdigest() == expected, str(p)
    return {'path': str(p), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns, 'sha256': h.hexdigest()}


def head(path):
    p = Path(path); rel = p.relative_to(ROOT).as_posix()
    blob = subprocess.check_output(['git', 'show', 'HEAD:' + rel], cwd=ROOT)
    assert p.read_bytes().replace(b'\r\n', b'\n') == blob.replace(b'\r\n', b'\n'), rel


names = ('meth484_prepare_binding.py', 'meth484_shared_integer_eligibility.py',
         'meth484_retention_audit.py', 'meth484_windows_terminal.ps1', 'meth484_audit_windows_terminal.ps1')
helpers = [ROOT / 'benchmarks/native_expert_scaling' / n for n in names]
helpers.append(DOC / 'METH_484_SHARED_INTEGER_ELIGIBILITY_PROTOCOL_20261006.md')
for p in helpers:
    head(p)
parent = item(DOC / 'meth458_switch_matched_whole_cost_result.json', '3762fd4e6c86a6ea7e64f54d4350cc2bd91761f2f856a06bb0a0a442f12d285f')
retention = item(DOC / 'RETENTION_458_20261005.json', '65ec6408bc1de79d0a3bad869b57de7298285901c5c91ca502330250ea9ea8b0')
r = json.loads(Path(parent['path']).read_bytes())
sources = []
for n, source in r['sources'].items():
    a = source['artifact']; stat = Path(a['payload']).stat()
    assert [stat.st_size, stat.st_mtime_ns] == source['artifact_stat_before']
    sources.append({'n': int(n), 'manifest': item(a['manifest'], a['manifest_sha256']),
                    'payload': {'path': a['payload'], 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                'inherited_458_sha256_not_refreshed': a['sha256']}})
scope = [f'{folder}/*.{ext}' for folder in ('benchmarks/native_expert_scaling', 'benchmarks/phase60') for ext in ('c', 'h', 'py', 'cu')]
paths = sorted(subprocess.check_output(['git', 'ls-files', '--', *scope], cwd=ROOT, text=True).splitlines())
# One batch verifies EVERY catalog's canonical committed text, without 600 git processes.
query = ''.join('HEAD:' + p + '\n' for p in paths).encode()
reply = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT, input=query)
cursor = 0; catalog = []
for rel in paths:
    end = reply.index(b'\n', cursor); header = reply[cursor:end].split(); count = int(header[2])
    assert header[1] == b'blob'; cursor = end + 1; blob = reply[cursor:cursor + count]; cursor += count
    assert reply[cursor:cursor + 1] == b'\n'; cursor += 1
    physical = (ROOT / rel).read_bytes()
    assert physical.replace(b'\r\n', b'\n') == blob.replace(b'\r\n', b'\n'), rel
    catalog.append(dict(item(ROOT / rel), relative=rel, canonical_text_exact_HEAD=True,
                        raw_HEAD_blob_sha256=hashlib.sha256(blob).hexdigest()))
assert cursor == len(reply)
contracts = []
for n in (374, 389):
    label = 'physical' if n == 374 else 'three'
    p = DOC / f'meth{n}_switch_{label}_workers_contract_result.json'; head(p); contracts.append(item(p))
for p in (ROOT / 'benchmarks/native_expert_scaling/meth335_switch_w8a8_export.py',
          DOC / 'METH_483_WHOLE_ALGEBRA_REASSESSMENT_AND_NEXT_20261006.md', ROOT / '.gitattributes'):
    head(p); contracts.append(item(p))
base = Path(sys.base_prefix)
runtime_paths = {Path(sys.executable), Path(sys._base_executable), ROOT / '.venv/pyvenv.cfg'}
runtime_paths.update(base.glob('*.dll'))
runtime_paths.update((base / 'DLLs').glob('*.pyd')); runtime_paths.update((base / 'DLLs').glob('*.dll'))
runtime_paths.update(p for p in (base / 'Lib').rglob('*.py') if 'site-packages' not in p.parts and '__pycache__' not in p.parts)
dist = importlib.metadata.distribution('psutil')
runtime_paths.update(Path(dist.locate_file(p)).resolve() for p in dist.files
                     if str(p).endswith(('.py', '.pyd', '.dll', '/METADATA', '/RECORD')))
runtime = {'executable': str(Path(sys.executable).resolve()), 'python': sys.version,
           'psutil_version': dist.version, 'files': [item(p) for p in sorted(runtime_paths)]}
record = {'experiment': 'METH484 prospective metadata/runtime/source binding; no numeric model work',
          'head_at_preparation': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'helpers': [item(p) for p in helpers], 'parent': parent, 'retention': retention,
          'sources': sources, 'source_records': contracts, 'catalog_scope': scope, 'catalog': catalog,
          'runtime': runtime, 'preparation_seconds': time.monotonic() - START,
          'no_payload_value_read_or_GPU_solver_model_import': True}
with DEST.open('xb') as stream:
    stream.write((json.dumps(record, indent=2, ensure_ascii=False) + '\n').replace('\n', '\r\n').encode())
print(json.dumps({'binding': str(DEST), 'sha256': item(DEST)['sha256'], 'catalog_files': len(catalog),
                  'runtime_files': len(runtime_paths), 'seconds': record['preparation_seconds']}))
