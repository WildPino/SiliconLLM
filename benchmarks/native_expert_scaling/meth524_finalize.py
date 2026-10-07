"""Terminal admission: JSON, file hashes and closed instances only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth524_operations import DOC, ROOT, output_bytes, write


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
        ('meth524_binding.json', args.binding_sha), ('meth524_main_result.json', args.main_sha), ('meth524_audit_result.json', args.audit_sha))]
    b, m, a = [json.loads(Path(v['path']).read_bytes()) for v in refs]
    assert all(v['gates'] and all(v['gates'].values()) for v in (b, m, a))
    assert len(m['gates']) == 6 and len(a['gates']) == 7
    assert m['binding_sha256'] == a['binding_sha256'] == refs[0]['sha256'] and a['main_sha256'] == refs[1]['sha256']
    winref = item(DOC / 'meth524_windows_terminal.json'); win = json.loads(Path(winref['path']).read_text(encoding='utf-8-sig'))
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
        ('binding', ROOT / 'results/native_expert_scaling/meth524_binding_resource.json', args.binding_sha),
        ('main', ROOT / 'results/native_expert_scaling/meth524_tree/terminal_resource.json', args.main_sha),
        ('audit', ROOT / 'results/native_expert_scaling/meth524_audit/terminal_resource.json', args.audit_sha)):
        ref = item(path); r = json.loads(path.read_bytes())
        assert r.get('binding_sha256', r.get('result_sha256')) == sha
        assert r['seconds'] <= r['limits'][0] and r['OS_peak_bytes'] <= r['limits'][1]
        resources.append(dict(kind=kind, receipt=ref, resource=r))
    for rel, sha in b['preserved'].items(): item(ROOT / rel, sha)
    assert not any(p.is_file() for p in (ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    retained = [item(v['path'], v['sha256']) for v in m['output_inventory'] + a['output_inventory']]
    for key in ('cases', 'occupancy', 'views', 'price', 'eligibility', 'decision', 'selected_source_predicate_dot_checks'):
        assert m[key] == a[key]
    zero = ('source_response_function_calls', 'full_signed_WI_projection_rows', 'continuous_source_shadows',
            'candidate_function_vectors', 'native_calls', 'model_calls', 'readout_fits')
    assert all(m[k] == a[k] == 0 for k in zero)
    exits = json.loads(args.exit_receipts); assert len(exits) == 3 and {v['kind'] for v in exits} == {'binding', 'main', 'audit'}
    assert all(v['exit_code'] == 0 for v in exits)
    retention = DOC / 'RETENTION_524_20261007.json'
    write(retention, dict(experiment='METH524', binding=refs[0], main=refs[1], audit=refs[2], local_outputs=retained,
                         terminal_resources=resources, scope='One original dev gain/main and one independent audit, no replay/new functions. Original523 time failure remains explicit upstream.'))
    admission = DOC / 'ADMISSION_524_20261007.json'
    write(admission, dict(experiment='METH524', main_completed=True, independent_audit_completed=True,
        gates=dict(all6_main_gain_tree_route_cost_controls=True, all7_independent_gain_source_route_and_cost_controls=True,
                   three_actual_exit0_instances_closed_typed_UTC_zero_faults=True, foreign_empty_cache_and_resources_preserved=True,
                   unchanged_frozen_depth2_eligibility_decisions_retained=True), binding=refs[0], raw=refs[1], audit=refs[2],
        retention=item(retention), windows=winref, closed_instances=closed, executor_exit_receipts=exits, resources=resources,
        views=m['views'], price=m['price'], eligibility=m['eligibility'], decision=m['decision'],
        selected_source_predicate_dot_checks=m['selected_source_predicate_dot_checks'],
        verified_gain_panels=a['verified_gain_panels'], verified_candidates=a['verified_candidates'],
        maximum_gain_bound_ratio=a['maximum_gain_bound_ratio'], **{k: 0 for k in zero},
        first_faults_or_repairs=[], physical_DRAM_verified=False,
        scope='Necessary finite depth2 information/route/price screen only. No leaf function quality, useful count, physical memory, fresh whole quality/rate or general family/goal claim.'))
    size = output_bytes(); assert size <= 64 << 20
    print(json.dumps(dict(admission=item(admission), combined_output_bytes=size, closed_instances=len(closed),
        resource=dict(seconds=time.monotonic() - start, OS_peak_bytes=proc.memory_info().peak_wset, bytes_hashed=hashed))))


if __name__ == '__main__': main()
