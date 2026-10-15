#!/usr/bin/env python3
"""Audit completed storage relocation without running models or editing evidence."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).absolute().parents[1]
CONTROL = Path('E:/SiliconLLM_Artifacts/migrations/20261015')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def historical_document(record):
    """Resolve old handoff pins to its explicitly receipted exact snapshot."""
    receipt_path = ROOT / 'docs/maintenance/handoff_storage_update_20261015.json'
    if record['path'] != 'HANDOFF.md' or not receipt_path.exists():
        return ROOT / record['path']
    update = load(receipt_path)
    require(update['schema'] == 'HANDOFF_STORAGE_UPDATE_V1' and
            update['current_path'] == update['previous_original_path'] == 'HANDOFF.md' and
            update['previous_path'] == 'docs/maintenance/snapshots/HANDOFF_PRE_STORAGE_20261015.md' and
            update['previous_bytes'] == record['bytes'] and
            update['previous_sha256'] == record['sha256'] and
            update['plan_sha256'] == sha(ROOT / 'docs/maintenance/storage_migration_plan_20261015.json'),
            'handoff update custody')
    current = ROOT / update['current_path']
    require(current.stat().st_size == update['current_bytes'] and
            sha(current) == update['current_sha256'], 'updated handoff hash')
    return ROOT / update['previous_path']


def main(output):
    require(not output.exists(), f'output already exists: {output}')
    manifest = load(CONTROL / 'manifest.json')
    require(manifest['status'] == 'COMPLETE', 'migration incomplete')
    plan_path = ROOT / 'docs/maintenance/storage_migration_plan_20261015.json'
    plan = load(plan_path)
    require(sha(plan_path) == manifest['plan_sha256'], 'plan hash')
    require(sha(CONTROL / 'inventory.json') == manifest['inventory_sha256'], 'inventory hash')
    require(sha(CONTROL / 'receipts.jsonl') == manifest['receipts_sha256'], 'receipt hash')
    state = load(CONTROL / 'state.json')
    require(state['phase'] == 'COMPLETE' and
            all(v == 'BACKUP_REMOVED' for v in state['units'].values()), 'cutover state')
    target_root = Path(plan['destination_root'])
    target_device = os.stat(target_root).st_dev
    serial = wintypes.DWORD()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetVolumeInformationW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR,
        wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetVolumeInformationW.restype = wintypes.BOOL
    require(kernel.GetVolumeInformationW(target_root.anchor, None, 0,
                ctypes.byref(serial), None, None, None, 0), 'target volume identity')
    unit_paths = {u['name']: Path(u['relative_path']) for u in plan['units']}
    junctions = []
    for unit, relative in unit_paths.items():
        logical, physical = ROOT / relative, target_root / relative
        require(os.path.isjunction(logical), f'logical junction missing: {logical}')
        require(logical.resolve(strict=True) == physical.resolve(strict=True),
                f'junction target differs: {logical}')
        require(not (Path(manifest['backup_root']) / relative).exists(),
                f'original backup still present: {relative}')
        junctions.append({'unit': unit, 'logical': str(logical), 'physical': str(physical)})
    source_to_target = {}
    target_to_source = {}
    count = length = 0
    with (CONTROL / 'receipts.jsonl').open(encoding='utf-8') as f:
        for line in f:
            receipt = json.loads(line)
            physical = target_root / unit_paths[receipt['unit']] / receipt['relative_path']
            stat = os.stat(physical, follow_symlinks=False)
            require(not stat.st_file_attributes & 0x400 and stat.st_size == receipt['bytes'],
                    f'payload missing or changed length: {physical}')
            # Python's Windows stat file index matches BY_HANDLE_FILE_INFORMATION.
            require(stat.st_ino == receipt['target_file_index'] and
                    stat.st_dev == target_device and serial.value == receipt['target_volume'],
                    f'file identity: {physical}')
            source_id = (receipt['source_volume'], receipt['source_file_index'])
            target_id = (receipt['target_volume'], stat.st_ino)
            require(source_to_target.setdefault(source_id, target_id) == target_id,
                    f'source hardlink group split: {physical}')
            require(target_to_source.setdefault(target_id, source_id) == source_id,
                    f'distinct source files unexpectedly merged: {physical}')
            count += 1
            length += receipt['bytes']
    require(count == manifest['files'] and length == manifest['logical_file_bytes'],
            'global receipt totals')
    nested_links = 0
    for unit, value in load(CONTROL / 'inventory.json')['units'].items():
        for item in value['entries']:
            if item['metadata']['kind'] in ('junction', 'symlink'):
                physical = target_root / unit_paths[unit] / item['relative_path']
                require(os.path.islink(physical) or os.path.isjunction(physical),
                        f'nested link missing: {physical}')
                require(os.readlink(physical) == item['metadata']['raw_target'],
                        f'nested link target changed: {physical}')
                nested_links += 1
    pins = []
    baseline = load(ROOT / 'docs/maintenance/storage_baseline_20261015.json')
    historical = load(ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925/'
                      'HANDOFF_SNAPSHOT_20261015/handoff_verification.json')
    for record in baseline['preserved_files'] + historical['documents']:
        path = historical_document(record)
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
                f'preserved document changed: {path}')
        pins.append(record['path'])
    snapshot = ROOT / ('docs/research/NATIVE_EXPERT_SCALING_20260925/'
                       'HANDOFF_SNAPSHOT_20261015/HANDOFF_PREVIOUS_20261014.md')
    require(snapshot.stat().st_size == historical['snapshot_bytes'] and
            sha(snapshot) == historical['snapshot_sha256'], 'historical snapshot hash')
    result = {'schema': 'STORAGE_RELOCATION_AUDIT_V1', 'status': 'PASS',
              'observed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'manifest_sha256': sha(CONTROL / 'manifest.json'), 'junctions': junctions,
              'file_receipts': count, 'logical_file_bytes': length,
              'distinct_source_files': len(source_to_target),
              'nested_links_preserved': nested_links,
              'preserved_document_checks': pins, 'snapshot_sha256': sha(snapshot),
              'scope': 'Manifest custody, logical redirects, receipt file identities and '
                       'hardlink equivalence, historical document pins. Payload SHA pairs '
                       'are in the migration receipts; this audit does not rehash all tensors. '
                       'No model execution.'}
    handoff_update = ROOT / 'docs/maintenance/handoff_storage_update_20261015.json'
    if handoff_update.exists():
        result['handoff_update_receipt_sha256'] = sha(handoff_update)
        result['handoff_historical_pin_location'] = load(handoff_update)['previous_path']
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'PASS', 'units': len(junctions), 'files': count}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
