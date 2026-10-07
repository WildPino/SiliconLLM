"""Posthoc retained511 diagnosis; no source functions, model, C or numerical replay."""
import datetime
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
START = time.monotonic()
PROC = psutil.Process()
PROC.cpu_affinity([10])
PEAK = 0
DEST = DOC / 'meth512_retained_diagnosis_result.json'
assert not DEST.exists()


def guard():
    global PEAK
    PEAK = max(PEAK, PROC.memory_info().peak_wset)
    assert PEAK <= 512 << 20 and time.monotonic() - START <= 60


def read(name, sha):
    p = DOC / name
    data = p.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha
    guard()
    return json.loads(data), {'path': str(p), 'bytes': len(data), 'sha256': sha}


def ticks(v):
    d = Decimal(str(v)) * 10000000
    assert d == d.to_integral_value()
    return int(d)


def edit(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row = row, [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1,
                           previous[j - 1] + (x != y)))
    return row[-1]


def rational(v):
    return {'numerator': v.numerator, 'denominator': v.denominator,
            'value': float(v)}


INPUTS = {
    'meth511_r8_donor_result.json': '4b73168632b025f3d5977e2f2c6e5fa75c495a4d00777bf1488332785d7ae195',
    'meth511_r9_audit_result.json': '48e95e9fa6d0466c89337fadc77877def862a42165d10a9b5f45c525b30fab27',
    'EVALUATION_511_20261007.json': '2d698cec6755589a0e9e0272fb69999a8064d9a3975b22ca586fa4eab7e38546',
}
raw, receipts = {}, {}
for name, sha in INPUTS.items():
    raw[name], receipts[name] = read(name, sha)
donor, audit, evaluation = [raw[n] for n in INPUTS]
assert donor['summary'] == audit['summary'] == evaluation['summary']
assert evaluation['same_artifact_fresh_quality_AND_warm_accepted_rate_ge50_qualified_in_declared_scope']
assert not evaluation['whole_fixed_recipe_eligible']
assert len(donor['cases']) == 96
KEYS = {'encoder': 'encoder_seconds', 'cross_KV': 'cross_kv_seconds',
        'decoder': 'decode_greedy_seconds', 'total': 'full_generation_seconds'}
rows = []
for ordinal, c in enumerate(donor['cases']):
    assert (c['book'], c['index']) == divmod(ordinal, 4)
    r = {'book': c['book'], 'index': c['index'], 'source_id': c['source_id']}
    r['same_native_generated_IDs'] = c['source_task']['generated_ids'] == c['candidate_task']['generated_ids']
    r['arm'] = {}
    for arm in ['source', 'candidate']:
        task = c[arm + '_task']
        measured = [v for v in c['native'][arm]['generation'] if v['repetition'] >= 0]
        assert [v['repetition'] for v in measured] == [0, 1, 2]
        assert all(v['profile'] == 0 and v['generated_ids'] == task['generated_ids'] for v in measured)
        for v in measured:
            residual = ticks(v[KEYS['total']]) - sum(ticks(v[KEYS[k]]) for k in ['encoder', 'cross_KV', 'decoder'])
            assert abs(residual) <= 3
        sums = {k: sum(ticks(v[key]) for v in measured) for k, key in KEYS.items()}
        normalized = Fraction(edit(task['prose_ids'], c['donor_task']['prose_ids']),
                              max(len(task['prose_ids']), len(c['donor_task']['prose_ids']), 1))
        full = [ticks(v[KEYS['total']]) for v in measured]
        cc = measured[0]['conditional_columns']
        assert len(cc) == 12
        assert all(v['conditional_columns'] == cc for v in measured)
        r['arm'][arm] = {
            'three_repetition_sum_ticks_100ns': sums,
            'mean_seconds': {k: v / 30000000 for k, v in sums.items()},
            'full_repetition_seconds': [v / 10000000 for v in full],
            'full_max_to_min': max(full) / min(full),
            'generated_IDs': task['generated_tokens'], 'prose_IDs': task['prose_tokens'],
            'healthy': task['healthy_complete_nonempty_fields'],
            'generated_IDs_equal_donor': task['generated_ids'] == c['donor_task']['generated_ids'],
            'prose_edit_to_donor': rational(normalized),
            'encoder_nonzero_sum': sum(v['nonzero_sum'] for v in cc[:6]) if arm == 'candidate' else None,
            'decoder_nonzero_sum': sum(v['nonzero_sum'] for v in cc[6:]) if arm == 'candidate' else None,
            'encoder_hidden_density': sum(v['nonzero_sum'] for v in cc[:6]) / (6 * 29 * 3072) if arm == 'candidate' else None,
            'decoder_hidden_density': sum(v['nonzero_sum'] for v in cc[6:]) / (6 * task['generated_tokens'] * 3072) if arm == 'candidate' else None,
            'known_field_exact_accuracy': task['known_field_exact_accuracy'],
        }
        if arm == 'candidate':
            assert all(v['calls'] == 29 for v in cc[:6])
            assert all(v['calls'] == task['generated_tokens'] for v in cc[6:])
        else:
            assert all(v['calls'] == v['nonzero_sum'] == v['maximum'] == 0 for v in cc)
    s, k = r['arm']['source'], r['arm']['candidate']
    r['candidate_to_source_elapsed_ratio'] = k['mean_seconds']['total'] / s['mean_seconds']['total']
    r['component_delta_seconds'] = {key: k['mean_seconds'][key] - s['mean_seconds'][key] for key in KEYS}
    r['candidate_decoder_per_generated_ID_to_source_ratio'] = (
        k['mean_seconds']['decoder'] / k['generated_IDs']
        / (s['mean_seconds']['decoder'] / s['generated_IDs']))
    assert abs(k['prose_edit_to_donor']['value'] - c['normalized_prose_edit']) < 1e-15
    # Profile-only instrumentation is descriptive; never used as warm-clock subtraction.
    profile = c['native']['candidate']['profile'][0]['hybrid_head']
    r['separate_profile_head_seconds'] = profile['scan_seconds'] + profile['refine_seconds']
    rows.append(r)
    guard()


def aggregate(group):
    out = {'cases': len(group), 'arm': {}}
    for arm in ['source', 'candidate']:
        selected = [r['arm'][arm] for r in group]
        sums = {k: sum(v['three_repetition_sum_ticks_100ns'][k] for v in selected) for k in KEYS}
        g = sum(v['generated_IDs'] for v in selected)
        calls = len(group) * 6 * 29
        out['arm'][arm] = {
            'sum_case_mean_seconds': {k: v / 30000000 for k, v in sums.items()},
            'total_generated_IDs': g,
            'accepted_prose_IDs': sum(v['prose_IDs'] for v in selected if v['healthy']),
            'exact_donor_generations': sum(v['generated_IDs_equal_donor'] for v in selected),
            'prose_edit_mean': rational(sum((Fraction(v['prose_edit_to_donor']['numerator'], v['prose_edit_to_donor']['denominator']) for v in selected), Fraction()) / len(selected)),
            'decoder_seconds_per_generated_ID': sums['decoder'] / 30000000 / g,
            'encoder_hidden_density': sum(v['encoder_nonzero_sum'] for v in selected) / (calls * 3072) if arm == 'candidate' else None,
            'decoder_hidden_density': sum(v['decoder_nonzero_sum'] for v in selected) / (6 * g * 3072) if arm == 'candidate' else None,
            'median_within_case_max_min': statistics.median(v['full_max_to_min'] for v in selected),
            'maximum_within_case_max_min': max(v['full_max_to_min'] for v in selected),
        }
    s, c = [out['arm'][arm] for arm in ['source', 'candidate']]
    out['elapsed_ratio'] = c['sum_case_mean_seconds']['total'] / s['sum_case_mean_seconds']['total']
    out['component_delta_seconds'] = {k: c['sum_case_mean_seconds'][k] - s['sum_case_mean_seconds'][k] for k in KEYS}
    return out


books = [aggregate(rows[4 * i:4 * i + 4]) for i in range(24)]
for i, b in enumerate(books):
    b['book'] = i
    b['source_id'] = rows[4 * i]['source_id']
    b['changed_native_generations'] = sum(not r['same_native_generated_IDs'] for r in rows[4 * i:4 * i + 4])
    assert abs(b['elapsed_ratio'] - donor['summary']['matched_cost']['book_ratios'][i]) < 1e-12
groups = {name: aggregate(selected) for name, selected in {
    'all': rows,
    'same_native_generated_IDs': [r for r in rows if r['same_native_generated_IDs']],
    'changed_native_generated_IDs': [r for r in rows if not r['same_native_generated_IDs']],
}.items()}
counts = {'same_native_IDs': groups['same_native_generated_IDs']['cases'],
          'changed_native_IDs': groups['changed_native_generated_IDs']['cases']}
counts.update({key: 0 for key in ['improved_prose_edit', 'worsened_prose_edit', 'equal_prose_edit',
                                 'recovered_exact_donor_generation', 'introduced_donor_generation_difference']})
for r in rows:
    s, c = [r['arm'][arm] for arm in ['source', 'candidate']]
    q, p = [Fraction(v['prose_edit_to_donor']['numerator'], v['prose_edit_to_donor']['denominator']) for v in [s, c]]
    counts['improved_prose_edit' if p < q else 'worsened_prose_edit' if p > q else 'equal_prose_edit'] += 1
    counts['recovered_exact_donor_generation'] += not s['generated_IDs_equal_donor'] and c['generated_IDs_equal_donor']
    counts['introduced_donor_generation_difference'] += s['generated_IDs_equal_donor'] and not c['generated_IDs_equal_donor']
assert counts['same_native_IDs'] == 82 and counts['changed_native_IDs'] == 14
assert [b['book'] for b in books if b['elapsed_ratio'] > 1] == [1, 3, 15, 19, 20, 22]
for arm in ['source', 'candidate']:
    assert abs(groups['all']['arm'][arm]['sum_case_mean_seconds']['total'] - donor['summary']['rates'][arm]['sum_case_mean_seconds']) < 1e-12
source = Path(__file__).read_bytes()
result = {
    'experiment': 'METH512 retained511 same-cohort readout and cost diagnosis',
    'scope': 'Posthoc descriptive only, no new statistical gate or fresh qualification; original511 decisions unchanged.',
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'execution_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'source_sha256': hashlib.sha256(source).hexdigest(), 'input_receipts': receipts,
    'no_C_model_source_function_calls': True, 'counts': counts, 'groups': groups,
    'books': books, 'cases': rows,
    'limits': ['Same-ID subgroups hold generated work fixed but are selected posthoc.',
               'Component clocks diagnose observed elapsed location, not unique causal mechanism or physical DRAM.',
               'Three repetitions and separate profile clocks cannot prove intrinsic per-book kernel regressions.',
               'No whole baseline, donor, auditor or payload observations replayed; no gate relaxed.'],
    'resource': {'process_instance': {'pid': PROC.pid, 'create_time_unix': PROC.create_time()},
                 'seconds_before_serialization': time.monotonic() - START,
                 'OS_peak_bytes_before_serialization': PEAK, 'limits': [60, 512 << 20]},
}
guard()
with DEST.open('x', encoding='utf8') as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write('\n')
guard()
terminal = {'result_sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
            'seconds': time.monotonic() - START, 'OS_peak_bytes': PEAK,
            'process_instance': result['resource']['process_instance'], 'new_C_model_calls': 0}
with DEST.with_suffix('.terminal.json').open('x', encoding='utf8') as f:
    json.dump(terminal, f, indent=2)
    f.write('\n')
print(json.dumps({'counts': counts, 'groups': groups, 'terminal': terminal}), flush=True)
