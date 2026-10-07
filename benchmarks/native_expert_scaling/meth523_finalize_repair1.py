"""Admit independently audited prefix while marking original main time gate failed."""
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
    ap = argparse.ArgumentParser(); ap.add_argument('--exit-receipts', required=True); args = ap.parse_args()
    start = time.monotonic(); proc = psutil.Process(); hashed = 0
    def item(path, expected=None):
        nonlocal hashed
        p = Path(path).resolve(); stat = p.stat(); h = hashlib.sha256()
        with p.open('rb') as file:
            while data := file.read(8 << 20):
                h.update(data); hashed += len(data)
                assert time.monotonic() - start <= 60 and proc.memory_info().peak_wset <= 128 << 20
        assert (p.stat().st_size, p.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
        if expected: assert h.hexdigest() == expected, str(p)
        return dict(path=str(p), bytes=stat.st_size, sha256=h.hexdigest())
    refs = {}
    for kind, name, sha in (
        ('binding', 'meth523_binding.json', '7cced083d598522bef6b9c382b96b5761e4c26f37130e814241572c5d9603b0e'),
        ('main_fault', 'meth523_main_result.failure.json', 'e8f24213b092c7bf31d3b4203a6d28f853733b72fc7d6927a3c64a6d5ddb84c5'),
        ('main_metadata_repair1', 'meth523_main_metadata_repair1_result.json', '8f0d6fa2dc1ed7be00f0c68f4309994f6e4b818b5f2bf5867ac429bf0f95c1d3'),
        ('main', 'meth523_main_result.json', '0688e17034df2ae58e88dc76eba06b39509ad705cdc7fca77e62d3d0d4c71cd2'),
        ('audit', 'meth523_audit_result.json', 'fe4f6a8195385c9dcdbcd606a526eddf7273381cc7519b608fff0aaa1b6b6bb7')):
        refs[kind] = item(DOC / name, sha)
    raws = {k: json.loads(Path(v['path']).read_bytes()) for k, v in refs.items()}
    b, f, seal, m, a = [raws[k] for k in ('binding', 'main_fault', 'main_metadata_repair1', 'main', 'audit')]
    assert len(f['gates']) == len(m['gates']) == len(a['gates']) == 7 and all(f['gates'].values()) and all(a['gates'].values())
    assert f['views'] == m['views'] == a['views'] and f['eligibility'] == m['eligibility'] == a['eligibility']
    assert f['decision'] == m['decision'] == a['decision'] and m['original_failure'] == refs['main_fault']
    assert not m['original_resource_gate_passed'] and not m['original_main_success'] and m['numerical_prefix_completed']
    assert f['resource']['seconds'] > 900 and f['resource']['OS_peak_bytes'] <= 1536 << 20
    assert m['original_executor_exit'] == dict(tool_chunk_id='2afaca', exit_code=1)
    assert seal['new_numerical_calls'] == 0 and all(seal['gates'].values())
    assert a['main_sha256'] == refs['main']['sha256'] and m['binding_sha256'] == a['binding_sha256'] == refs['binding']['sha256']
    winref = item(DOC / 'meth523_windows_terminal_repair1.json'); win = json.loads(Path(winref['path']).read_text(encoding='utf-8-sig'))
    assert len(win['stages']) == 4 and set(win['positive_control_ids']) == {179810, 179791}
    assert all(v['query_available'] and not v['relevant_events'] for v in win['stages'])
    closed = []
    for kind in ('binding', 'main_fault', 'main_metadata_repair1', 'audit'):
        raw = raws[kind]; inst = raw['process_instance']; stage = next(v for v in win['stages'] if v['kind'] == kind)
        assert stage['process_instance'] == inst and stage['raw_sha256'] == refs[kind]['sha256']
        try:
            live = psutil.Process(inst['pid']); assert abs(live.create_time() - inst['create_time_unix']) > .002
        except psutil.NoSuchProcess: pass
        closed.append(dict(kind=kind, instance=inst, closed=True))
    resources = []
    for kind, path in (
        ('binding', ROOT / 'results/native_expert_scaling/meth523_binding_resource.json'),
        ('main', ROOT / 'results/native_expert_scaling/meth523_fold/terminal_resource.json'),
        ('main_metadata_repair1', ROOT / 'results/native_expert_scaling/meth523_main_metadata_repair1_resource.json'),
        ('audit', ROOT / 'results/native_expert_scaling/meth523_audit/terminal_resource.json')):
        v = item(path); r = json.loads(path.read_bytes())
        assert r.get('binding_sha256', r.get('result_sha256')) == refs[kind]['sha256']
        passed = r['seconds'] <= r['limits'][0] and r['OS_peak_bytes'] <= r['limits'][1]
        assert passed == (kind != 'main'); resources.append(dict(kind=kind, receipt=v, resource=r, original_envelope_passed=passed))
    for rel, sha in b['preserved'].items(): item(ROOT / rel, sha)
    assert not any(p.is_file() for p in (ROOT / 'results/native_expert_scaling/meth511_runtime/empty_cache').rglob('*'))
    retained = [item(v['path'], v['sha256']) for v in m['output_inventory'] + a['output_inventory']]
    exits = json.loads(args.exit_receipts); assert len(exits) == 4
    assert {v['kind']: v['exit_code'] for v in exits} == dict(binding=0, main_fault=1, main_metadata_repair1=0, audit=0)
    retention = DOC / 'RETENTION_523_20261007.json'
    write(retention, dict(experiment='METH523', records=refs, local_outputs=retained, terminal_resources=resources,
        scope='Completed original numerical prefix plus metadata-only sealing and one independent audit. Original time/exit1 remains failed; no main/source/fit/native replay.'))
    admission = DOC / 'ADMISSION_523_20261007.json'
    write(admission, dict(experiment='METH523', main_completed=False, main_numerical_prefix_completed=True, independent_audit_completed=True,
        status='NUMERICAL_PREFIX_AUDITED_ORIGINAL_RESOURCE_GATE_FAILED',
        gates=dict(all7_main_numerical_prefix_and7_independent_audit_controls=True, metadata_only_original_failure_sealed_without_replay=True,
            all4_actual_exit_codes_closed_instances_and_typed_UTC_controls=True, foreign_empty_cache_and_nonmain_resource_envelopes=True,
            original_main_900s_envelope=False, unchanged_original_scientific_decisions=True),
        binding=refs['binding'], raw=refs['main'], audit=refs['audit'], original_failure=refs['main_fault'], metadata_repair=refs['main_metadata_repair1'],
        retention=item(retention), windows=winref, closed_instances=closed, executor_exit_receipts=exits, resources=resources,
        views=m['views'], eligibility=m['eligibility'], decision=m['decision'],
        new_signed_WI_projection_rows_each_scientific_pass=17540, new_continuous_source_shadows_each_scientific_pass=17540,
        old_native_source_WO_replays=0, model_calls=0, native_calls=0, readout_fits=0, physical_DRAM_verified=False,
        scope='Independently verified local source-fold evidence with explicit original time failure. Not a successful original protocol, C/timing/fresh quality/useful-n/DRAM/family/whole-goal pass.'))
    size = output_bytes(); assert size <= 2 << 30
    print(json.dumps(dict(admission=item(admission), combined_output_bytes=size, closed_instances=4,
        resource=dict(seconds=time.monotonic() - start, OS_peak_bytes=proc.memory_info().peak_wset, bytes_hashed=hashed))))


if __name__ == '__main__': main()
