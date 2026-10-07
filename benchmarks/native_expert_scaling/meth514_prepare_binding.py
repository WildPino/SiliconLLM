"""Bind exactly ALL96 actual256 retained generations and original tied head."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil
sys.path.insert(0, str(Path(__file__).resolve().parent))
import meth514_operations as O

START = time.monotonic()
PROC = psutil.Process()
PROC.cpu_affinity([10])
HASHED = PEAK = 0
CATALOG = {}
DEST = O.BIND
assert not DEST.exists() and not DEST.with_suffix('.failure.json').exists()


def item(path, expected=None):
    global HASHED, PEAK
    p = Path(path).resolve()
    stat = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as stream:
        while data := stream.read(8 << 20):
            h.update(data)
            HASHED += len(data)
            PEAK = max(PEAK, PROC.memory_info().peak_wset)
            assert PEAK <= 512 << 20 and time.monotonic() - START <= 180
    assert (p.stat().st_size, p.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
    if expected:
        assert h.hexdigest() == expected, str(p)
    row = {'path': str(p), 'bytes': stat.st_size, 'sha256': h.hexdigest()}
    CATALOG[row['path']] = row
    return row


try:
    runtime_receipt = item(O.DOC / 'meth513_r1_binding.json', 'e149336d994ab20bdf667154c97132be975eac07cbd27268f21e50464afd3c7b')
    previous = json.loads(Path(runtime_receipt['path']).read_bytes())
    assert sys.version == previous['python'] and str(Path(sys.executable).resolve()) == previous['executable']
    runtime = [item(row['path'], row['sha256']) for row in previous['runtime_files']]
    for relative, sha in previous['preserved'].items():
        item(O.ROOT / relative, sha)
    record = item(O.DOC / 'meth363_switch_all_a16_multi_span_quality_result.json', 'ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb')
    raw = json.loads(Path(record['path']).read_bytes())
    assert all(raw['gates'].values()) and raw['artifact']['source_unique_parameters'] == 14664154368
    tensor_record = item(O.DOC / 'meth327_switch_tensor_binding_result.json', raw['original327_binding_sha256'])
    tensor = json.loads(Path(tensor_record['path']).read_bytes())
    cohort = item(O.DOC / 'meth362_switch_multi_span_manifest.json', raw['manifest362_sha256'])
    assert tensor['passed'] and tensor['unique_architecture_parameters_after_confirmed_ties'] == 14664154368
    aliases = ['shared.weight', 'lm_head.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']
    expected = tensor['tensors']['shared.weight']
    assert expected['shape'] == [32128, 768] and expected['bytes'] == 98697216
    assert all(tensor['tensors'][name]['sha256'] == expected['sha256'] for name in aliases)
    spec = item(raw['artifact']['manifest'], raw['artifact']['manifest_sha256'])
    cfg, files, tensors = O.manifest(spec['path'])
    head = tensors['shared.weight']
    assert head[:5] == (0, 2, 32128, 768, 0) and head[6:] == (0, 24674304)
    assert all(tensors[name] == head for name in ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'])
    path = Path(raw['artifact']['payload']).resolve()
    assert Path(files[0]).resolve() == path
    stat = path.stat()
    assert stat.st_size == raw['artifact']['bytes'] == 14818015744
    with path.open('rb') as stream:
        stream.seek(head[5])
        data = stream.read(expected['bytes'])
    assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256']
    HASHED += len(data)
    PEAK = max(PEAK, PROC.memory_info().peak_wset)
    del data
    source_head = {'path': str(path), 'offset': head[5], 'bytes': expected['bytes'], 'sha256': expected['sha256'],
                   'file_bytes': stat.st_size, 'file_mtime_ns': stat.st_mtime_ns, 'shape': [32128, 768],
                   'manifest': spec, 'aliases': aliases, 'unique_original_parameters': 14664154368,
                   'retained_full_payload_sha256': raw['artifact']['sha256'],
                   'full_payload_SHA_scope': 'Retained qualified full SHA; current full size/stat and fresh used F32 extent SHA only.'}
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
    selected = []
    for book, row in enumerate(raw['books']):
        assert len(row['cases']) == 4
        for index, c in enumerate(row['cases']):
            assert c['index'] == index
            prefix = O.ROOT / f'results/native_expert_scaling/meth363_switch_all_a16_multi_span_quality/book{book}.case{index}'
            gen = c['generation']
            native = item(str(prefix) + '.generation.0.bin', gen['native_generation_sha256'])
            donor = item(str(prefix) + '.original_generation.npz', gen['original_generation_sha256'])
            ci, di = gen['native']['generated_ids'], gen['original']['generated_ids']
            prefix_length = 0
            while prefix_length < min(len(ci), len(di)) and ci[prefix_length] == di[prefix_length]:
                prefix_length += 1
            common = min(len(ci), len(di), prefix_length + 1)
            selected.append({'book': book, 'index': index, 'source_id': row['source_id'], 'native': native, 'donor': donor,
                             'candidate_ids': ci, 'donor_ids': di, 'common_positions': common, 'common_prefix_length': prefix_length,
                             'divergent': prefix_length < min(len(ci), len(di))})
    counts = {'cases': len(selected), 'native_positions': sum(len(row['candidate_ids']) for row in selected),
              'common_positions': sum(row['common_positions'] for row in selected), 'first_divergences': sum(row['divergent'] for row in selected)}
    assert counts == {'cases': 96, 'native_positions': 1065, 'common_positions': 949, 'first_divergences': 24}
    science = []
    for p in sorted((O.ROOT / 'benchmarks/native_expert_scaling').glob('meth514*')) + [O.DOC / 'METH_514_ACTUAL256_HEAD_PROTOCOL_20261007.md']:
        assert p.read_bytes().replace(b'\r\n', b'\n') == subprocess.check_output(
            ['git', 'show', 'HEAD:' + p.relative_to(O.ROOT).as_posix()], cwd=O.ROOT).replace(b'\r\n', b'\n')
        science.append(item(p))
    value = {'experiment': 'METH514 frozen ALL96 actual256 fixed8 retained-state screen',
             'freeze_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=O.ROOT, text=True).strip(),
             'python': sys.version, 'executable': str(Path(sys.executable).resolve()), 'packages': previous['packages'],
             'catalog': list(CATALOG.values()), 'runtime_files': runtime, 'scientific': science, 'preserved': previous['preserved'],
             'source_head': source_head, 'source_record': record, 'tensor_record': tensor_record, 'cohort': cohort,
             'selected': selected, 'counts': counts, 'K': 8, 'limits': {'binding': [180, 512 << 20], 'main': [300, 2 << 30], 'audit': [300, 2 << 30], 'combined_output_bytes': 256 << 20},
             'prospective_full_source_head_MACs_per_main_or_audit': (1065 + 949) * 32128 * 768,
             'scope': 'All native states checked against full original source head; donor comparison only on common input histories. Consumed applicability, no new quality/speed/DRAM/n causality.',
             'process_instance': {'pid': PROC.pid, 'create_time_unix': PROC.create_time()},
             'resource_before_serialization': {'seconds': time.monotonic() - START, 'OS_peak_bytes': PEAK, 'bytes_hashed': HASHED}}
    O.write(DEST, value)
    bound = item(DEST)
    print(json.dumps({'binding': bound, 'counts': counts, 'seconds': time.monotonic() - START, 'OS_peak_bytes': PEAK, 'bytes_hashed': HASHED}), flush=True)
except BaseException:
    O.write(DEST.with_suffix('.failure.json'), {'traceback': traceback.format_exc(), 'seconds': time.monotonic() - START,
                                              'OS_peak_bytes': PEAK, 'bytes_hashed': HASHED})
    raise
