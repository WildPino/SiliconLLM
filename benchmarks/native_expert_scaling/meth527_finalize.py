"""527 terminal admission: JSON, hashes and closed process instances only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth527_operations import DOC, ROOT, output_bytes, write


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
        return dict(path=str(p), bytes=stat.st_size, sha256=h.hexdigest())
    refs = [item(DOC / name, sha) for name, sha in (
        ('meth527_binding.json', args.binding_sha), ('meth527_main_result.json', args.main_sha), ('meth527_audit_result.json', args.audit_sha))]
    b, m, a = [json.loads(Path(v['path']).read_bytes()) for v in refs]
    assert all(v['gates'] and all(v['gates'].values()) for v in (b, m, a))
    assert len(m['gates']) == 7 and len(a['gates']) == 8
    assert m['binding_sha256'] == a['binding_sha256'] == refs[0]['sha256'] and a['main_sha256'] == refs[1]['sha256']
    winref = item(DOC / 'meth527_windows_terminal.json'); win = json.loads(Path(winref['path']).read_text(encoding='utf-8-sig'))
    assert len(win['stages']) == 3 and set(win['positive_control_ids']) == {179810, 179791}
    assert all(v['query_available'] and not v['relevant_events'] for v in win['stages'])
    closed = []
    for kind, raw, ref in zip(('binding', 'main', 'audit'), (b, m, a), refs):
        inst = raw['process_instance']; stage = next(v for v in win['stages'] if v['kind'] == kind)
        assert stage['process_instance'] == inst and stage['raw_sha256'] == ref['sha256']
        try:
            live = psutil.Process(inst['pid']); assert abs(live.create_time() - inst['create_time_unix']) > .002
        except psutil.NoSuchProcess: pass
        closed.append(dict(kind=kind, instance=inst, closed=True))
    resources = []
    for kind, path, sha in (
        ('binding', ROOT / 'results/native_expert_scaling/meth527_binding_resource.json', args.binding_sha),
        ('main', ROOT / 'results/native_expert_scaling/meth527_bound_refinement/terminal_resource.json', args.main_sha),
        ('audit', ROOT / 'results/native_expert_scaling/meth527_audit/terminal_resource.json', args.audit_sha)):
        ref = item(path); r = json.loads(path.read_bytes())
        assert r.get('binding_sha256', r.get('result_sha256')) == sha
        assert r['seconds'] <= r['limits'][0] and r['OS_peak_bytes'] <= r['limits'][1]
        resources.append(dict(kind=kind, receipt=ref, resource=r))
    for rel, sha in b['preserved'].items(): item(ROOT / rel, sha)
    for v in b['legacy_output_inventory']: item(v['path'], v['sha256'])
    assert not any(p.is_file() for p in (ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    retained = [item(v['path'], v['sha256']) for v in m['output_inventory'] + a['output_inventory']]
    for key in ('encoding', 'cases', 'resolved_orders', 'newly_resolved_orders', 'coefficient_energy_rows',
                'coefficient_energy_limb_product_terms', 'views', 'eligibility', 'decision', 'constructed_queries',
                'new_router_score_vectors', 'selected_source_integer_dot_checks'):
        assert m[key] == a[key]
    assert a['verified_energy_direct_product_terms'] == 299630592 and a['verified_scalar_rows'] == 325120
    zero = ('source_response_function_calls', 'native_calls', 'model_calls', 'readout_fits',
            'full_source_WI_projection_rows', 'old_native_router_queries_replayed', 'consumed_rows_for_selection', 'old525_geometry_or_queries_recomputed')
    assert all(m[k] == a[k] == 0 for k in zero)
    exits = json.loads(args.exit_receipts); assert len(exits) == 3 and {v['kind'] for v in exits} == {'binding', 'main', 'audit'}
    assert all(v['exit_code'] == 0 for v in exits)
    retention = DOC / 'RETENTION_527_20261007.json'
    write(retention, dict(experiment='METH527', binding=refs[0], main=refs[1], audit=refs[2], local_outputs=retained,
                         terminal_resources=resources, scope='One new row-energy/scalar-bound main and one independent audit. All525/526 outputs unchanged. Original523 time failure and525 scientific closure remain explicit.'))
    admission = DOC / 'ADMISSION_527_20261007.json'
    write(admission, dict(experiment='METH527', main_completed=True, independent_audit_completed=True,
        gates=dict(all7_main_exact_energy_and_outward_scalar_bound_controls=True, all8_independent_original_bits_integer_and_scalar_controls=True,
                   three_actual_exit0_instances_closed_typed_UTC_zero_faults=True, foreign_empty_cache_and_resources_preserved=True,
                   unchanged_frozen_order_eligibility_and_nonselecting_diagnostics_retained=True), binding=refs[0], raw=refs[1], audit=refs[2],
        retention=item(retention), windows=winref, closed_instances=closed, executor_exit_receipts=exits, resources=resources,
        views=m['views'], eligibility=m['eligibility'], decision=m['decision'],
        selected_source_integer_dot_checks=m['selected_source_integer_dot_checks'], constructed_queries=m['constructed_queries'],
        resolved_orders=m['resolved_orders'], newly_resolved_orders=m['newly_resolved_orders'],
        coefficient_energy_rows=m['coefficient_energy_rows'], verified_scalar_rows=a['verified_scalar_rows'], **{k: 0 for k in zero},
        first_faults_or_repairs=[], physical_DRAM_verified=False,
        scope='Bound refinement on unchanged saved geometry only; no query qualification, source-function quality, actual C/DRAM, useful n, fresh whole quality/rate or family/goal claim.'))
    size = output_bytes(); assert size <= 32 << 20
    print(json.dumps(dict(admission=item(admission), combined_output_bytes=size, closed_instances=len(closed),
        resource=dict(seconds=time.monotonic() - start, OS_peak_bytes=proc.memory_info().peak_wset, bytes_hashed=hashed))))


if __name__ == '__main__': main()
