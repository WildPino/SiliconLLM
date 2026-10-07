"""520 post-scientific terminal admission; JSON/file/process checks only."""
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
    bind = item(DOC / 'meth520_binding.json', 'abcabcd44e8adc2e8dce79300663fccbbe1ca3aadfd70e3a1bfa41ff1a2e128a')
    primal = item(DOC / 'meth520_main_result.json', '5803e189f57d90e3c78695cdbdede9a3ea25ed18a5cbfa48e83bbdc558f4f43b')
    audit = item(DOC / 'meth520_audit_result.json', '8c67ded99b65ff1f9e4b48be1e93c3edcfe7326f4aa6a60968928184b07ed5fb')
    b, m, a = [json.loads(Path(v['path']).read_bytes()) for v in [bind, primal, audit]]
    assert all(b['gates'].values()) and all(m['gates'].values()) and all(a['gates'].values())
    assert m['binding_sha256'] == a['binding_sha256'] == bind['sha256'] and a['main_sha256'] == primal['sha256']
    windows = item(DOC / 'meth520_windows_terminal.json')
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
        ('binding', ROOT / 'results/native_expert_scaling/meth520_binding_resource.json', bind['sha256']),
        ('main', ROOT / 'results/native_expert_scaling/meth520_geometry/terminal_resource.json', primal['sha256']),
        ('audit', ROOT / 'results/native_expert_scaling/meth520_audit/terminal_resource.json', audit['sha256'])]:
        v = item(path)
        r = json.loads(path.read_bytes())
        assert r.get('binding_sha256', r.get('result_sha256')) == rawsha
        assert r['seconds'] <= r['limits'][0] and r['OS_peak_bytes'] <= r['limits'][1]
        resources.append({'kind': phase, 'receipt': v, 'resource': r})
    for rel, expected in b['preserved'].items():
        item(ROOT / rel, expected)
    payload_stat = Path(b['dictionary_file']['path']).stat()
    assert (payload_stat.st_size, payload_stat.st_mtime_ns) == (b['dictionary_file']['bytes'], b['dictionary_file']['mtime_ns'])
    cache = ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache'
    assert not any(p.is_file() for p in cache.rglob('*'))
    retained = []
    for v in m['output_inventory']:
        retained.append(item(v['path'], v['sha256']))
    assert not a['output_inventory']
    assert m['views'] == a['views']
    assert m['eligibility'] == a['eligibility'] and m['decision'] == a['decision'] == ('ELIGIBLE_FOR_SEPARATELY_FROZEN_SOURCE_RESPONSES' if all(m['eligibility'].values()) else 'COMPLEMENT_PLAN_RANK_OR_NUMERICAL_ELIGIBILITY_FAIL')
    assert len(m['gates']) == 5 and len(a['gates']) == 6
    retention = DOC / 'RETENTION_520_20261007.json'
    write(retention, {'experiment': 'METH520', 'input_binding': bind, 'mathematical_main': primal,
                      'independent_audit': audit, 'large_local_outputs': retained, 'terminal_resource_receipts': resources,
                      'reproducibility_scope': 'Exclusive original namespaces executed once; source/protocol/data SHA retained. Large outputs local, not committed binaries.'})
    retentionref = item(retention)
    admission = DOC / 'ADMISSION_520_20261007.json'
    value = {'experiment': 'METH520', 'main_completed': True, 'independent_audit_completed': True,
             'gates': {'all5_main_development_geometry_full_manifest_controls': True,
                       'all6_independent_Fraction_minor_QR_tail_query_plan_controls': True,
                       'actual_exit0_three_instances_closed_and_typed_UTC_zero_faults': True,
                       'foreign_dictionary_stat_empty_cache_and_resource_limits_preserved': True,
                       'ALL_development_rank_stability_eligibility_decision_unchanged': True},
             'binding': bind, 'raw': primal, 'audit': audit, 'retention': retentionref, 'windows': windows,
             'closed_instances': closed, 'executor_exit_receipts': [
                 {'kind': 'binding', 'tool_chunk_id': '288802', 'exit_code': 0},
                 {'kind': 'main', 'tool_chunk_id': '333dc6', 'exit_code': 0},
                 {'kind': 'audit', 'tool_chunk_id': '2c0755', 'exit_code': 0}],
             'resources': resources, 'views': m['views'],
             'eligibility': m['eligibility'], 'source_response_plan_eligible': all(m['eligibility'].values()), 'decision': m['decision'],
             'new_model_native_capture_calls': 0, 'mathematical_main_and_independent_audit_runs': [1, 1],
             'first_faults_or_repairs': [], 'physical_DRAM_verified': False,
             'scope': 'Dev-only pooled rank513 and ALL128 stable complement query manifests. Main5/audit6 and terminal admission PASS; four frozen rank/stability gates retained. No source-response function calls, readout fits, native/model calls, quality/rate, DRAM, all-bank/large n or goal completion.'}
    write(admission, value)
    # Include all new serialized receipts/metadata in the one combined output gate.
    output = list((ROOT / 'results/native_expert_scaling/meth520_geometry').iterdir()) + list((ROOT / 'results/native_expert_scaling/meth520_audit').iterdir())
    output += [Path(bind['path']), Path(primal['path']), Path(audit['path']), Path(windows['path']), retention, admission,
               ROOT / 'results/native_expert_scaling/meth520_binding_resource.json']
    size = sum(p.stat().st_size for p in output if p.is_file())
    assert size <= 32 << 20
    print(json.dumps({'admission': item(admission), 'combined_output_bytes': size, 'closed_instances': len(closed),
                      'resource': {'seconds': time.monotonic() - START, 'OS_peak_bytes': PROC.memory_info().peak_wset, 'bytes_hashed': HASHED}}))


if __name__ == '__main__':
    main()
