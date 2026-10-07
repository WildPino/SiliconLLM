"""519 post-scientific terminal admission; JSON/file/process checks only."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
START = time.monotonic()
PROC = psutil.Process()
HASHED = 0


def item(path, expected=None):
    global HASHED
    p = Path(path).resolve()
    s = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        while data := f.read(8 << 20):
            h.update(data)
            HASHED += len(data)
            assert time.monotonic() - START <= 60 and PROC.memory_info().peak_wset <= 128 << 20
    assert (p.stat().st_size, p.stat().st_mtime_ns) == (s.st_size, s.st_mtime_ns)
    if expected:
        assert h.hexdigest() == expected, str(p)
    return {'path': str(p), 'bytes': s.st_size, 'sha256': h.hexdigest()}


def write(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    bind = item(DOC / 'meth519_binding.json', '759330331caca87bd6c358ae1f7fd00bc31896ab64ed117569141d1caaa899b0')
    primal = item(DOC / 'meth519_main_result.json', '714bf731db9c6791e005cac41821cfd43e2254cb0f86ab5cb44510cf6b19e690')
    audit = item(DOC / 'meth519_audit_result.json', '245bd4c30a240e59663f9cfe6e47375349e9f8660a60e816854837f1239a08cf')
    b, m, a = [json.loads(Path(v['path']).read_bytes()) for v in [bind, primal, audit]]
    assert all(b['gates'].values()) and all(m['gates'].values()) and all(a['gates'].values())
    assert m['binding_sha256'] == a['binding_sha256'] == bind['sha256'] and a['main_sha256'] == primal['sha256']
    windows = item(DOC / 'meth519_windows_terminal.json')
    win = json.loads(Path(windows['path']).read_text(encoding='utf-8-sig'))
    assert len(win['stages']) == 3 and set(win['positive_control_ids']) == {179810, 179791}
    assert all(v['query_available'] and not v['relevant_events'] for v in win['stages'])
    closed = []
    for phase, raw, ref in [('binding', b, bind), ('main', m, primal), ('audit', a, audit)]:
        instance = raw['process_instance']
        try:
            live = psutil.Process(instance['pid'])
            assert abs(live.create_time() - instance['create_time_unix']) > .002
        except psutil.NoSuchProcess:
            pass
        event = next(v for v in win['stages'] if v['kind'] == phase)
        assert event['process_instance'] == instance and event['raw_sha256'] == ref['sha256']
        closed.append({'kind': phase, 'instance': instance, 'closed': True})
    resources = []
    for phase, path, rawsha in [
        ('binding', ROOT / 'results/native_expert_scaling/meth519_binding_resource.json', bind['sha256']),
        ('main', ROOT / 'results/native_expert_scaling/meth519_index/terminal_resource.json', primal['sha256']),
        ('audit', ROOT / 'results/native_expert_scaling/meth519_audit/terminal_resource.json', audit['sha256'])]:
        v = item(path)
        r = json.loads(path.read_bytes())
        assert r.get('binding_sha256', r.get('result_sha256')) == rawsha
        assert r['seconds'] <= r['limits'][0] and r['OS_peak_bytes'] <= r['limits'][1]
        resources.append({'kind': phase, 'receipt': v, 'resource': r})
    for rel, expected in b['preserved'].items():
        item(ROOT / rel, expected)
    payload_stat = Path(b['payload']['path']).stat()
    assert (payload_stat.st_size, payload_stat.st_mtime_ns) == (b['payload']['bytes'], b['payload']['mtime_ns'])
    cache = ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache'
    assert not any(p.is_file() for p in cache.rglob('*'))
    retained = []
    for v in m['output_inventory']:
        retained.append(item(v['path'], v['sha256']))
    assert not a['output_inventory']
    assert m['views'] == a['views'] and m['consumed_books'] == a['consumed_books'] and m['rare_consumed'] == a['rare_consumed']
    assert m['eligibility'] == a['eligibility'] and m['decision'] == a['decision'] == ('NEXT_ACTUAL_LOCAL_CERTIFICATE_BYTE_AND_OPERATOR_COST' if all(m['eligibility'].values()) else 'CLOSE_THIS_LOCAL_MATCHED_RANK32_ENERGY_RECIPE')
    assert len(m['gates']) == 5 and len(a['gates']) == 6
    retention = DOC / 'RETENTION_519_20261007.json'
    write(retention, {'experiment': 'METH519', 'input_binding': bind, 'mathematical_main': primal,
                      'independent_audit': audit, 'large_local_outputs': retained, 'terminal_resource_receipts': resources,
                      'reproducibility_scope': 'Exclusive original namespaces executed once; source/protocol/data SHA retained. Large outputs local, not committed binaries.'})
    retentionref = item(retention)
    admission = DOC / 'ADMISSION_519_20261007.json'
    value = {'experiment': 'METH519', 'main_completed': True, 'independent_audit_completed': True,
             'gates': {'all5_main_matched_local_geometry_capacity_controls': True,
                       'all6_independent_weighted_moment_integer_capacity_cost_controls': True,
                       'actual_exit0_three_instances_closed_and_typed_UTC_zero_faults': True,
                       'foreign_source_stat_empty_cache_and_resource_limits_preserved': True,
                       'ALL_domain_economy_decision_unchanged': True},
             'binding': bind, 'raw': primal, 'audit': audit, 'retention': retentionref, 'windows': windows,
             'closed_instances': closed, 'executor_exit_receipts': [
                 {'kind': 'binding', 'tool_chunk_id': 'e2e541', 'exit_code': 0},
                 {'kind': 'main', 'tool_chunk_id': 'b6b720', 'exit_code': 0},
                 {'kind': 'audit', 'tool_chunk_id': '97f4f2', 'exit_code': 0}],
             'resources': resources, 'views': m['views'], 'rare_consumed': m['rare_consumed'],
             'eligibility': m['eligibility'], 'economic_recipe_eligible': all(m['eligibility'].values()), 'decision': m['decision'],
             'new_model_native_capture_calls': 0, 'mathematical_main_and_independent_audit_runs': [1, 1],
             'first_faults_or_repairs': [], 'physical_DRAM_verified': False,
             'scope': 'Source-derived dev-only local matched rank32 dictionaries and exact necessary capacity/lower byte floors on ALL17540 retained bank11 queries. Main5/audit6 apparatus and terminal admission PASS; economic eligibility separately retained. Possible rows are not sound activation masks; no fresh quality, C latency, DRAM, all-bank/256/large n or goal completion.'}
    write(admission, value)
    # Include all new serialized receipts/metadata in the one combined output gate.
    output = list((ROOT / 'results/native_expert_scaling/meth519_index').iterdir()) + list((ROOT / 'results/native_expert_scaling/meth519_audit').iterdir())
    output += [Path(bind['path']), Path(primal['path']), Path(audit['path']), Path(windows['path']), retention, admission,
               ROOT / 'results/native_expert_scaling/meth519_binding_resource.json']
    size = sum(p.stat().st_size for p in output if p.is_file())
    assert size <= 128 << 20
    print(json.dumps({'admission': item(admission), 'combined_output_bytes': size, 'closed_instances': len(closed),
                      'resource': {'seconds': time.monotonic() - START, 'OS_peak_bytes': PROC.memory_info().peak_wset, 'bytes_hashed': HASHED}}))


if __name__ == '__main__':
    main()
