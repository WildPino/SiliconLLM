"""523 terminal admission; only JSON/file/process checks, no function replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meth523_operations import DOC, ROOT, output_bytes, write


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
        ('meth523_binding.json', args.binding_sha), ('meth523_main_result.json', args.main_sha), ('meth523_audit_result.json', args.audit_sha))]
    b, m, a = [json.loads(Path(v['path']).read_bytes()) for v in (bind, primal, audit)]
    assert all(v['gates'] and all(v['gates'].values()) for v in (b, m, a))
    assert m['binding_sha256'] == a['binding_sha256'] == bind['sha256'] and a['main_sha256'] == primal['sha256']
    winref = item(DOC / 'meth523_windows_terminal.json'); win = json.loads(Path(winref['path']).read_text(encoding='utf-8-sig'))
    assert len(win['stages']) == 3 and set(win['positive_control_ids']) == {179810, 179791}
    assert all(v['query_available'] and not v['relevant_events'] for v in win['stages'])
    closed = []
    for kind, raw, ref in (('binding', b, bind), ('main', m, primal), ('audit', a, audit)):
        inst = raw['process_instance']; stage = next(v for v in win['stages'] if v['kind'] == kind)
        assert stage['process_instance'] == inst and stage['raw_sha256'] == ref['sha256']
        try:
            live = psutil.Process(inst['pid']); assert abs(live.create_time() - inst['create_time_unix']) > .002
        except psutil.NoSuchProcess: pass
        closed.append({'kind': kind, 'instance': inst, 'closed': True})
    resources = []
    for kind, path, sha in (
        ('binding', ROOT / 'results/native_expert_scaling/meth523_binding_resource.json', args.binding_sha),
        ('main', ROOT / 'results/native_expert_scaling/meth523_fold/terminal_resource.json', args.main_sha),
        ('audit', ROOT / 'results/native_expert_scaling/meth523_audit/terminal_resource.json', args.audit_sha)):
        ref = item(path); r = json.loads(path.read_bytes())
        assert r.get('binding_sha256', r.get('result_sha256')) == sha
        assert r['seconds'] <= r['limits'][0] and r['OS_peak_bytes'] <= r['limits'][1]
        resources.append({'kind': kind, 'receipt': ref, 'resource': r})
    for rel, sha in b['preserved'].items(): item(ROOT / rel, sha)
    assert not any(p.is_file() for p in (ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    retained = [item(v['path'], v['sha256']) for v in m['output_inventory'] + a['output_inventory']]
    assert m['views'] == a['views'] and m['eligibility'] == a['eligibility'] and m['decision'] == a['decision']
    assert len(m['gates']) == 7 and len(a['gates']) == 7
    assert m['old_native_source_WO_replays'] == a['old_native_source_WO_replays'] == m['native_calls'] == a['native_calls'] == 0
    assert m['new_signed_WI_projection_rows'] == a['new_signed_WI_projection_rows'] == 17540
    assert m['new_continuous_source_shadows'] == a['new_continuous_source_shadows'] == 17540
    exits = json.loads(args.exit_receipts); assert len(exits) == 3 and all(v['exit_code'] == 0 for v in exits)
    retention = DOC / 'RETENTION_523_20261007.json'
    write(retention, {'experiment': 'METH523', 'binding': bind, 'main': primal, 'audit': audit, 'local_outputs': retained,
                      'terminal_resources': resources, 'scope': 'One development-frozen source fold/hinge main and one independent I64/reverse-fold audit. New signed WI information and continuous source shadows counted; no old native WO/model/C replay.'})
    admission = DOC / 'ADMISSION_523_20261007.json'
    write(admission, {'experiment': 'METH523', 'main_completed': True, 'independent_audit_completed': True,
         'gates': {'all7_main_source_fold_hinge_codec_reference_and_report_controls': True,
                   'all7_independent_I64_source_reverse_fold_BYTE_energy_and_report_controls': True,
                   'three_actual_exit0_instances_closed_and_typed_UTC_zero_faults': True,
                   'foreign_empty_cache_runtime_and_resources_preserved': True,
                   'unchanged_single_reference_B512_I16_fidelity_and_coefficient_decisions_retained': True},
         'binding': bind, 'raw': primal, 'audit': audit, 'retention': item(retention), 'windows': winref,
         'closed_instances': closed, 'executor_exit_receipts': exits, 'resources': resources,
         'views': m['views'], 'eligibility': m['eligibility'], 'decision': m['decision'],
         'old_native_source_WO_replays': 0, 'new_signed_WI_projection_rows_each_pass': 17540,
         'new_continuous_source_shadows_each_pass': 17540, 'native_calls': 0, 'model_calls': 0, 'readout_fits': 0,
         'first_faults_or_repairs': [], 'physical_DRAM_verified': False,
         'scope': 'One-bank single-reference source-fold hinge fidelity. Serialized-code arithmetic reference, not actual C/timing/DRAM/fresh quality/useful larger n/other-family or goal completion. No general representation impossibility.'})
    size = output_bytes(); assert size <= 2 << 30
    print(json.dumps({'admission': item(admission), 'combined_output_bytes': size, 'closed_instances': len(closed),
                      'resource': {'seconds': time.monotonic() - start, 'OS_peak_bytes': proc.memory_info().peak_wset, 'bytes_hashed': hashed}}))


if __name__ == '__main__': main()
