"""Adopt the completed paired evidence after a missing process-instance field.

No model, tensor runtime, optimizer update or observation is executed.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'benchmarks/native_expert_scaling'
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
sys.path.insert(0, str(SITE))


def bind(a):
    original_path = DOC / 'chatbot_fixed_work_paired_finish_binding_20261009.json'
    original = json.loads(original_path.read_bytes())
    result_path = DOC / 'chatbot_fixed_work_paired_finish_result_20261009.json'
    receipt_path = result_path.with_suffix('.launcher_failure.json')
    receipt = json.loads(receipt_path.read_bytes()); source = json.loads(result_path.read_bytes())
    assert receipt['exit_code'] == 0 and receipt['fault'] == "KeyError('process_instance')"
    assert receipt['binding_sha256'] == source['binding_sha256'] == sha(original_path)
    assert sha(result_path) == '239cdda749773f19ebf617417b1aa5ea65140a8e5d8552e45d79d9beba3fce67'
    assert receipt['elapsed_seconds'] <= original['limits']['seconds']
    assert receipt['worker_OS_peak_through_exit'] + receipt['launcher_OS_peak_snapshot'] <= original['limits']['OS_bytes']
    assert source['GPU_allocated_peak'] <= original['limits']['GPU_allocated_bytes']
    assert source['GPU_reserved_peak'] <= original['limits']['GPU_reserved_bytes']
    substitutions = {
        (B / 'chatbot_fixed_work_paired_finish.py').resolve(): DOC / 'chatbot_fixed_work_paired_finish_worker_frozen_20261009.py.txt',
        (B / 'chatbot_falcon_usability_launch.py').resolve(): DOC / 'chatbot_fixed_work_paired_finish_launcher_frozen_20261009.py.txt'}
    files = []
    for entry in original['inputs']:
        path = substitutions.get(Path(entry['path']).resolve(), Path(entry['path']))
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], str(path)
        files.append(path)
    files.extend(p for p in (ROOT / 'results/native_expert_scaling/chatbot_fixed_work_paired_finish_20261009').iterdir() if p.is_file())
    files.extend([Path(__file__), Path(sys.executable), B / 'chatbot_falcon_usability_launch.py',
                  B / 'chatbot_fixed_work_paired_finish.py', original_path, result_path, receipt_path,
                  result_path.with_suffix('.worker.log'), DOC / 'CHATBOT_FIXED_WORK_PAIRED_ADOPTION_PROTOCOL_20261009.md'])
    files = list(dict.fromkeys(p.resolve() for p in files))
    assert not a.out.exists() and a.freeze
    value = dict(schema='FIXED_WORK_PAIRED_ADOPTION_BINDING_V1', freeze=a.freeze,
        python=str(Path(sys.executable).resolve()), worker_path=str(Path(__file__).resolve()),
        source_result=str(result_path.resolve()), source_receipt=str(receipt_path.resolve()),
        completion_binding=str(original_path.resolve()), criteria=original['criteria'],
        corpus=original['corpus'], old_corpus=original['old_corpus'], retention_ids=original['retention_ids'],
        limits=dict(seconds=300, reserve_seconds=30, OS_bytes=1 << 30, output_bytes=16 << 20),
        runtime_binding_scope='All original completion inputs/outputs/faults reverified through frozen executed source copies; exact metadata and stored-row adoption only, no tensor/model/GPU calls.',
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    write(a.out, value)
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(files))), flush=True)


def worker(a):
    start = time.monotonic(); assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes()); assert b['schema'] == 'FIXED_WORK_PAIRED_ADOPTION_BINDING_V1' and b['freeze'] == a.freeze
    a.directory.mkdir(exist_ok=False)
    import psutil
    proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
    source = json.loads(Path(b['source_result']).read_bytes())
    receipt = json.loads(Path(b['source_receipt']).read_bytes())
    assert receipt['exit_code'] == 0 and receipt['fault'] == "KeyError('process_instance')"
    assert source['schema'] == 'FIXED_WORK_PAIRED_FINISH_RESULT_V1'
    assert source['new_student_cases'] == 63 and source['reused_B_after_cases'] == 17
    assert source['new_optimizer_updates'] == 0 and source['retained_original_new_updates'] == 50
    assert source['source_calls'] == source['native_runs'] == source['reserved_queries'] == 0
    records = json.loads(Path(b['corpus']).read_bytes())['records']
    old = json.loads(Path(b['old_corpus']).read_bytes())['records']
    by_id = {r['id']: r for r in records + [r for r in old if r['id'] in b['retention_ids']]}
    bound = {Path(v['path']).resolve(): v for v in b['inputs']}
    def values(rows):
        n = sum(r['labels'] for r in rows); err = sum(r['disagreements'] for r in rows)
        return dict(cases=len(rows), labels=n, case_KL=sum(r['KL'] for r in rows) / len(rows),
            label_KL=sum(sum(r['label_KL']) for r in rows) / n, disagreements=err, label_disagreement=err / n,
            case_disagreement=sum(r['disagreements'] / r['labels'] for r in rows) / len(rows))
    def check_summary(actual, expected):
        assert set(actual) == set(expected)
        for key in actual: assert abs(actual[key] - expected[key]) <= 1e-12 * max(1., abs(expected[key])), key
    for arm, packet in source['arms'].items():
        assert packet['final_private_step'] == 344 and packet['final_common_step'] == (26 if arm == 'B' else 0)
        assert len(packet['new_updates']) == (26 if arm == 'A' else 24)
        for stage in ('before', 'after', 'retention'):
            group = packet[stage]; rows = group['cases']
            assert len(rows) == (32 if stage == 'retention' else 48) and len({r['id'] for r in rows}) == len(rows)
            for row in rows:
                rec = by_id[row['id']]; assert row['labels'] == len(rec['output_ids']) == len(row['label_KL']) == len(row['predicted_ids'])
                assert all(math.isfinite(v) and v >= -1e-10 for v in row['label_KL'])
                assert abs(sum(row['label_KL']) / row['labels'] - row['KL']) < 1e-10
                assert row['disagreements'] == sum(x != y for x, y in zip(row['predicted_ids'], rec['output_ids'], strict=True))
                entry = row['logits']; proof = bound[Path(entry['path']).resolve()]
                assert entry['bytes'] == proof['bytes'] == row['labels'] * 65537 * 4 and entry['sha256'] == proof['sha256']
                support = row['exposure']; positions = len(rec['student_input_ids'])
                assert len(support['private']) == len(support['mass_max_defect']) == 12
                assert all(len(v) == 72 and all(isinstance(n, int) and n >= 0 for n in v) and sum(v) == positions * (6 if arm == 'B' else 8) for v in support['private'])
                assert support['common_positions'] == [positions * (2 if arm == 'B' else 0)] * 12
                assert all(0 <= x <= 1e-6 for x in support['mass_max_defect'])
            check_summary(group['summary']['overall'], values(rows))
            for domain, actual in group['summary']['domains'].items(): check_summary(actual, values([r for r in rows if r['domain'] == domain]))
            if stage != 'retention':
                for split in ('FIT', 'DEV'): check_summary(group['summary'][split], values([r for r in rows if r['split'] == split]))
                for domain, actual in group['summary']['DEV_domains'].items(): check_summary(actual, values([r for r in rows if r['domain'] == domain and r['split'] == 'DEV']))
    c = b['criteria']; pa = source['arms']['A']['after']['summary']; pb = source['arms']['B']['after']['summary']
    preference = dict(B_DEV_KL=pb['DEV']['case_KL'] <= c['B_DEV_case_KL_ratio_max'] * pa['DEV']['case_KL'],
        B_every_domain_KL=all(pb['DEV_domains'][d]['case_KL'] <= c['B_every_domain_ratio_max'] * pa['DEV_domains'][d]['case_KL'] for d in pa['DEV_domains']),
        B_every_domain_disagreement=all(pb['DEV_domains'][d]['label_disagreement'] <= c['B_every_domain_ratio_max'] * pa['DEV_domains'][d]['label_disagreement'] for d in pa['DEV_domains']))
    assert preference == source['preference_gates']
    for arm, packet in source['arms'].items():
        post = packet['after']['summary']; ret = packet['retention']['summary']['overall']; old = source['parent_outputs_reused']['retention']['overall']
        gates = dict(DEV_absolute_KL=post['DEV']['case_KL'] <= c['DEV_case_KL_max'],
            DEV_absolute_disagreement=post['DEV']['label_disagreement'] <= c['DEV_label_disagreement_max'],
            DEV_every_domain_KL=all(v['case_KL'] <= c['domain_DEV_case_KL_max'] for v in post['DEV_domains'].values()),
            DEV_every_domain_disagreement=all(v['label_disagreement'] <= c['domain_DEV_label_disagreement_max'] for v in post['DEV_domains'].values()),
            old_DEV_KL_retention=ret['case_KL'] <= c['retention_ratio_max'] * old['case_KL'],
            old_DEV_disagreement_retention=ret['label_disagreement'] <= c['retention_ratio_max'] * old['label_disagreement'])
        assert gates == source['absolute_retention_gates'][arm]
    assert source['decision'] == ('PAIRED_B_PREFERENCE_PASS' if all(preference.values()) and
        source['absolute_retention_gates']['B']['old_DEV_KL_retention'] and source['absolute_retention_gates']['B']['old_DEV_disagreement_retention'] else 'PAIRED_B_PREFERENCE_FAIL')
    assert not source['quality_admission'] and not source['native_admission']
    assert time.monotonic() - start < b['limits']['seconds'] - b['limits']['reserve_seconds']
    assert proc.memory_info().peak_wset < b['limits']['OS_bytes'] and not proc.children(recursive=True)
    result = dict(source)
    result.update(schema='FIXED_WORK_PAIRED_ADOPTION_RESULT_V1', freeze=a.freeze, binding_sha256=a.binding_sha,
        process_instance=dict(pid=proc.pid, create_time_unix=proc.create_time()),
        source_result=dict(path=b['source_result'], sha256=sha(b['source_result'])),
        source_worker_provenance=dict(pid=receipt['worker_pid'], held_receipt=b['source_receipt'],
          historical_creation_time_available=False, original_process_instance_field_available=False),
        source_finish_family_seconds=receipt['elapsed_seconds'], source_finish_worker_seconds=source['elapsed_seconds'],
        new_student_cases=0, retained_completion_cases=63, new_optimizer_updates=0,
        stored_aggregate_and_exposure_checks=True, metadata_adoption=True, GPU_calls=0,
        source_finish_resources=dict(GPU_allocated_peak=source['GPU_allocated_peak'], GPU_reserved_peak=source['GPU_reserved_peak'],
                                    held_OS_peak=receipt['worker_OS_peak_through_exit']),
        worker_OS_peak_snapshot=proc.memory_info().peak_wset, elapsed_seconds=time.monotonic() - start)
    write(a.directory / 'adoption.json', dict(source_worker_pid=receipt['worker_pid'], new_observations=0,
        new_updates=0, retained_completion_cases=63, aggregate_checks=True, historical_creation_time_available=False))
    write(a.out, result)
    print(json.dumps(dict(stage='adopted', decision=result['decision'], source_worker_pid=receipt['worker_pid'], new_observations=0)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('bind', 'worker'), default='worker')
    p.add_argument('--binding', type=Path); p.add_argument('--binding-sha'); p.add_argument('--freeze')
    p.add_argument('--directory', type=Path); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); bind(a) if a.mode == 'bind' else worker(a)
