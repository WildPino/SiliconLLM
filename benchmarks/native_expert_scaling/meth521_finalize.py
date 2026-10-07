"""521 terminal admission; JSON/file/process checks only, no numeric/native replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth521_operations import DOC, ROOT, output_bytes, write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--binding-sha', required=True); ap.add_argument('--main-sha', required=True)
    ap.add_argument('--audit-sha', required=True); ap.add_argument('--exit-receipts', required=True); args = ap.parse_args()
    start = time.monotonic(); proc = psutil.Process(); hashed = 0
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
    bind, primal, audit = [item(DOC / name, sha) for name, sha in (
        ('meth521_binding.json', args.binding_sha), ('meth521_main_result.json', args.main_sha), ('meth521_audit_result.json', args.audit_sha))]
    b, m, a = [json.loads(Path(v['path']).read_bytes()) for v in (bind, primal, audit)]
    assert all(v['gates'] and all(v['gates'].values()) for v in (b, m, a))
    assert m['binding_sha256'] == a['binding_sha256'] == bind['sha256'] and a['main_sha256'] == primal['sha256']
    winref = item(DOC / 'meth521_windows_terminal.json'); win = json.loads(Path(winref['path']).read_text(encoding='utf-8-sig'))
    assert len(win['stages']) == 3 and set(win['positive_control_ids']) == {179810, 179791}
    assert all(v['query_available'] and not v['relevant_events'] for v in win['stages'])
    closed = []
    for phase, raw, ref in (('binding', b, bind), ('main', m, primal), ('audit', a, audit)):
        stage = next(v for v in win['stages'] if v['kind'] == phase)
        assert stage['process_instance'] == raw['process_instance'] and stage['raw_sha256'] == ref['sha256']
        for descriptor in stage['instances']:
            instance = descriptor['instance']
            try:
                live = psutil.Process(instance['pid'])
                assert abs(live.create_time() - instance['create_time_unix']) > .002
            except psutil.NoSuchProcess: pass
            closed.append({'kind': descriptor['kind'], 'instance': instance, 'closed': True})
    resources = []
    for phase, path, sha in (
        ('binding', ROOT / 'results/native_expert_scaling/meth521_binding_resource.json', args.binding_sha),
        ('main', ROOT / 'results/native_expert_scaling/meth521_transfer/terminal_resource.json', args.main_sha),
        ('audit', ROOT / 'results/native_expert_scaling/meth521_audit/terminal_resource.json', args.audit_sha)):
        ref = item(path); r = json.loads(path.read_bytes())
        assert r.get('binding_sha256', r.get('result_sha256')) == sha
        assert r['seconds'] <= r['limits'][0] and r['OS_peak_bytes'] <= r['limits'][1]
        resources.append({'kind': phase, 'receipt': ref, 'resource': r})
    for rel, expected in b['preserved'].items(): item(ROOT / rel, expected)
    stat = Path(b['payload']['path']).stat()
    assert (stat.st_size, stat.st_mtime_ns) == (b['payload']['bytes'], b['payload']['mtime_ns'])
    cache = ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache'; assert not any(p.is_file() for p in cache.rglob('*'))
    retained = [item(v['path'], v['sha256']) for v in m['output_inventory'] + a['output_inventory']]
    assert m['decision'] == a['decision'] and m['reports']['outcomes'] == a['reports']['outcomes']
    assert m['source_response_function_calls'] == a['source_response_function_calls'] == 53943
    assert m['native_calls'] == 4 and m['compiler_calls'] == 1 and a['native_calls'] == a['compiler_calls'] == 0
    exits = json.loads(args.exit_receipts); assert len(exits) == 3 and all(v['exit_code'] == 0 for v in exits)
    retention = DOC / 'RETENTION_521_20261007.json'
    write(retention, {'experiment': 'METH521', 'binding': bind, 'main': primal, 'audit': audit, 'local_outputs': retained,
                      'terminal_resources': resources, 'scope': 'Exclusive capture/main/audit once, code/data/runtime/manifest identities retained. Large local binaries not committed.'})
    admission = DOC / 'ADMISSION_521_20261007.json'
    write(admission, {'experiment': 'METH521', 'main_completed': True, 'independent_audit_completed': True,
         'gates': {'all_main_source_fit_native_and_metric_controls': True,
                   'all_independent53943_source_equations_physical_and_metric_controls': True,
                   'all_owned_instances_exit_closed_and_typed_UTC_zero_faults': True,
                   'foreign_source_stat_empty_cache_and_resource_limits_preserved': True,
                   'frozen_full_information_function_and_routing_decisions_retained': True},
         'binding': bind, 'raw': primal, 'audit': audit, 'retention': item(retention), 'windows': winref,
         'closed_instances': closed, 'executor_exit_receipts': exits, 'resources': resources,
         'source_response_function_calls': {'main': 53943, 'audit': 53943}, 'native_calls': 4, 'compiler_calls': 1,
         'reports': m['reports'], 'decision': m['decision'], 'first_faults_or_repairs': [],
         'physical_DRAM_verified': False, 'model_calls': 0,
         'scope': 'Local fixed-class information test, not fresh corpus/whole quality, useful larger n, physical DRAM or goal completion.'})
    size = output_bytes(); assert size <= 2 << 30
    print(json.dumps({'admission': item(admission), 'combined_output_bytes': size, 'closed_instances': len(closed),
                      'resource': {'seconds': time.monotonic() - start, 'OS_peak_bytes': proc.memory_info().peak_wset, 'bytes_hashed': hashed}}))


if __name__ == '__main__': main()
