"""521 prefix-plus-metric admission; no source, fit or numerical/native replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth521_metric_repair1_operations import BIND, FAULT_SHA, MAIN_SHA, ORIGINAL_BINDING_SHA
from meth521_operations import DOC, ROOT, output_bytes, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repair-binding-sha', required=True); ap.add_argument('--repair-sha', required=True)
    ap.add_argument('--exit-receipts', required=True); args = ap.parse_args(); start = time.monotonic(); proc = psutil.Process(); hashed = 0
    def item(path, expected=None):
        nonlocal hashed
        p = Path(path).resolve(); stat = p.stat(); h = hashlib.sha256()
        with p.open('rb') as f:
            while data := f.read(8 << 20):
                h.update(data); hashed += len(data)
                assert time.monotonic() - start <= 60 and proc.memory_info().peak_wset <= 128 << 20
        assert (p.stat().st_size, p.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
        if expected: assert h.hexdigest() == expected, str(p)
        return {'path': str(p), 'bytes': stat.st_size, 'sha256': h.hexdigest()}
    paths = [('binding', DOC / 'meth521_binding.json', ORIGINAL_BINDING_SHA),
             ('main', DOC / 'meth521_main_result.json', MAIN_SHA),
             ('audit_fault', DOC / 'meth521_audit_result.failure.json', FAULT_SHA),
             ('metric_repair1_binding', BIND, args.repair_binding_sha),
             ('audit_repair1', DOC / 'meth521_audit_repair1_result.json', args.repair_sha)]
    refs = {kind: item(path, sha) for kind, path, sha in paths}
    records = {kind: json.loads(Path(v['path']).read_bytes()) for kind, v in refs.items()}
    assert all(all(v['gates'].values()) for v in records.values())
    b = records['binding']; m = records['main']; a = records['audit_repair1']; fault = records['audit_fault']
    assert 'np.allclose(values[:, :14]' in fault['traceback'] and a['original_audit_fault_sha256'] == FAULT_SHA
    assert a['original_binding_sha256'] == ORIGINAL_BINDING_SHA and a['main_sha256'] == MAIN_SHA
    assert a['binding_sha256'] == args.repair_binding_sha
    winref = item(DOC / 'meth521_metric_repair1_windows_terminal.json')
    win = json.loads(Path(winref['path']).read_text(encoding='utf-8-sig'))
    assert len(win['stages']) == 5 and set(win['positive_control_ids']) == {179810, 179791}
    assert all(v['query_available'] and not v['relevant_events'] for v in win['stages'])
    closed = []
    for kind, raw in records.items():
        stage = next(v for v in win['stages'] if v['kind'] == kind)
        assert stage['process_instance'] == raw['process_instance'] and stage['raw_sha256'] == refs[kind]['sha256']
        for entry in stage['instances']:
            inst = entry['instance']
            try:
                live = psutil.Process(inst['pid']); assert abs(live.create_time() - inst['create_time_unix']) > .002
            except psutil.NoSuchProcess: pass
            closed.append({'kind': entry['kind'], 'instance': inst, 'closed': True})
    resources = []
    for kind, path in [('binding', ROOT / 'results/native_expert_scaling/meth521_binding_resource.json'),
                       ('main', ROOT / 'results/native_expert_scaling/meth521_transfer/terminal_resource.json'),
                       ('metric_repair1_binding', ROOT / 'results/native_expert_scaling/meth521_metric_repair1_binding_resource.json'),
                       ('audit_repair1', ROOT / 'results/native_expert_scaling/meth521_audit_repair1/terminal_resource.json')]:
        ref = item(path); r = json.loads(path.read_bytes())
        assert r.get('binding_sha256', r.get('result_sha256')) == refs[kind]['sha256']
        assert r['seconds'] <= r['limits'][0] and r['OS_peak_bytes'] <= r['limits'][1]
        resources.append({'kind': kind, 'receipt': ref, 'resource': r})
    r = fault['resource']; assert r['seconds'] <= r['limits'][0] and r['OS_peak_bytes'] <= r['limits'][1]
    resources.append({'kind': 'audit_fault', 'receipt': refs['audit_fault'], 'resource': r})
    for rel, sha in b['preserved'].items(): item(ROOT / rel, sha)
    stat = Path(b['payload']['path']).stat(); assert (stat.st_size, stat.st_mtime_ns) == (b['payload']['bytes'], b['payload']['mtime_ns'])
    assert not any(p.is_file() for p in (ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    retained = [item(v['path'], v['sha256']) for v in m['output_inventory'] + a['output_inventory']]
    for v in fault['partial_outputs']:
        ref = item(v['path']); assert ref['bytes'] == v['bytes']; retained.append(ref)
    assert m['decision'] == a['decision'] and m['reports']['outcomes'] == a['reports']['outcomes']
    assert m['source_response_function_calls'] == a['completed_audit_prefix_source_function_calls'] == 53943
    assert a['new_source_response_function_calls'] == a['new_coefficient_solves'] == a['native_calls'] == 0
    exits = json.loads(args.exit_receipts); assert len(exits) == 5
    assert {v['kind']: v['exit_code'] for v in exits} == {'binding': 0, 'main': 0, 'audit_fault': 1, 'metric_repair1_binding': 0, 'audit_repair1': 0}
    retention = DOC / 'RETENTION_521_20261007.json'
    write(retention, {'experiment': 'METH521', 'phases': refs, 'local_outputs': retained,
                      'terminal_resources': resources, 'prefix_witness': 'Frozen audit control flow reaches line203 after all source/equation/physical checks; metric arrays lost on first fault, reconstructed only in repair1.',
                      'scope': 'Original source capture and full independent source prefix once, remaining metric continuation once. Large local binaries not committed.'})
    admission = DOC / 'ADMISSION_521_20261007.json'
    write(admission, {'experiment': 'METH521', 'main_completed': True, 'independent_audit_completed': True,
         'gates': {'all8_main_source_fit_native_metric_controls': True,
                   'independent_full_source_equation_physical_prefix_and_propagated_metric_repair_controls': True,
                   'actual_exit_codes_owned_instances_closed_and_typed_UTC_zero_OS_faults': True,
                   'foreign_source_stat_empty_cache_and_resource_limits_preserved': True,
                   'unchanged_function_routing_thresholds_and_first_metric_fault_retained': True},
         'binding': refs['binding'], 'raw': refs['main'], 'audit': refs['audit_repair1'], 'first_audit_fault': refs['audit_fault'],
         'repair_binding': refs['metric_repair1_binding'], 'retention': item(retention), 'windows': winref,
         'closed_instances': closed, 'executor_exit_receipts': exits, 'resources': resources,
         'source_response_function_calls': {'main': 53943, 'independent_original_audit_prefix': 53943, 'metric_repair1': 0},
         'native_calls': 4, 'compiler_calls': 1, 'reports': m['reports'], 'decision': m['decision'],
         'first_faults_or_repairs': [{'fault': refs['audit_fault'], 'repair_binding': refs['metric_repair1_binding'],
              'repair': refs['audit_repair1'], 'changed_control': 'Algebraic propagation certificate replaces blanket relative tolerance on cancellation-prone energy cross terms.',
              'source_fit_or_native_replay': False}], 'physical_DRAM_verified': False, 'model_calls': 0,
         'scope': 'Fixed-class full-information recipe fails; independent source/equation/native prefix plus metric repair qualified. No fresh corpus/whole quality/useful n/DRAM/family or goal completion.'})
    extra = sum(p.stat().st_size for p in (ROOT / 'results/native_expert_scaling/meth521_audit_repair1').iterdir() if p.is_file())
    size = output_bytes() + extra; assert size <= 2 << 30
    print(json.dumps({'admission': item(admission), 'combined_output_bytes': size, 'closed_instances': len(closed),
                      'resource': {'seconds': time.monotonic() - start, 'OS_peak_bytes': proc.memory_info().peak_wset, 'bytes_hashed': hashed}}))


if __name__ == '__main__': main()
