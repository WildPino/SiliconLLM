"""Read-only routing diagnostics from both exact whole-quality output cohorts."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
import psutil
import meth324_switch_reference as M

PROTOCOL = M.DOC / 'METH_392_SWITCH_ROUTE_AUDIT_PROTOCOL_20261004.md'
NUMERIC = M.DOC / 'meth389_switch_three_workers_contract_result.json'
NUMERIC_SHA = 'fd12cfc2809fb42f52faef0b442781154c30ba27576f264c5f693bc30ec91dd2'
OUTPUTS = M.ROOT / 'results/native_expert_scaling/meth389_switch_three_workers_contract'
QUALITY = {
    256: ('meth363_switch_all_a16_multi_span_quality_result.json', 'ba4f18454cb8788dceacfa7c8d629e6083420c75a9b69318f4277d7a9edfedeb'),
    128: ('meth387_switch_base128_multi_span_quality_result.json', '539275a209c8b8a5680c388ec9130012157a681ed382ba1a4bf49ab8cccb9d4c'),
}


def route_bytes(data, n, source, positions):
    assert len(data) >= 36 and data[:8] == b'SWR32O01'
    s, t, d, enc, dec, vocab, count = struct.unpack_from('<7I', data, 8)
    assert (s, t, d, enc, dec, vocab) == (source, positions, 768, 12, 12, 32128)
    assert count == 6 * (s + t)
    offset = 36 + 4 * ((enc + 2) * s * d + t * (dec + 2) * d + t * vocab)
    assert len(data) == offset + 12 * count
    rows = np.frombuffer(data, dtype=[('expert', '<i4'), ('accepted', '<i4'), ('probability', '<f4')], offset=offset)
    assert len(rows) == count and np.all((rows['expert'] >= 0) & (rows['expert'] < n))
    assert np.all((rows['accepted'] == 0) | (rows['accepted'] == 1))
    p = rows['probability'].astype(np.float64)
    assert np.all(np.isfinite(p)) and np.all((p >= (1. / n) * (1. - 2e-6)) & (p <= 1.))
    return rows[:6 * s].reshape(6, s), rows[6 * s:].reshape(t, 6).T


def summarize(probabilities, counts, n):
    p = np.asarray(probabilities, dtype=np.float64)
    assert len(p) > 0
    amplification = 1. / p
    # If retained selected probability mass is Z, selected expert scales by 1/Z.
    # Every retained candidate has mass <=p. Thus m*p>=1/1.01 is necessary for
    # <=1% relative scaling inflation. This is a lower bound, NOT sufficient.
    minimum = np.minimum(n, np.ceil((1. / 1.01) / p)).astype(np.int64)
    quantiles = [0., .05, .5, .95, 1.]
    return {
        'route_queries': len(p), 'expert_visit_counts': dict(sorted(counts.items())),
        'distinct_selected_experts': len(counts), 'candidate_count': n,
        'quantiles': quantiles,
        'selected_probability_quantiles': np.quantile(p, quantiles).tolist(),
        'singleton_scale_multiplier_quantiles': np.quantile(amplification, quantiles).tolist(),
        'singleton_relative_scale_change_gt1percent_count': int(np.sum(amplification > 1.01)),
        'singleton_relative_scale_change_gt5percent_count': int(np.sum(amplification > 1.05)),
        'necessary_min_candidates_for_1percent_scale_quantiles': np.quantile(minimum, quantiles).tolist(),
        'necessary_min_candidates_for_1percent_scale_counts': dict(sorted(Counter(map(int, minimum)).items())),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start = time.monotonic(); maximum = 0; stage = 'bindings'
    result = {'experiment': 'METH-392-existing-whole-quality-real-routing-diagnostics', 'targets': []}

    def guard():
        nonlocal maximum
        maximum = max(maximum, psutil.Process().memory_info().rss)
        assert maximum <= 2 << 30 and time.monotonic() - start <= 300, 'audit_5min_2GiB'

    try:
        for path in (Path(__file__), PROTOCOL, NUMERIC, Path(M.__file__)): M.committed(path)
        assert M.digest(NUMERIC) == NUMERIC_SHA
        numeric = json.loads(NUMERIC.read_text()); assert all(numeric['gates'].values())
        result.update({'controller_sha256': M.digest(__file__), 'protocol_sha256': M.digest(PROTOCOL),
                       'numeric389_sha256': NUMERIC_SHA, 'controller_head': __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=M.ROOT).decode().strip()})
        for target in numeric['targets']:
            n = target['n']; name, sha = QUALITY[n]; path = M.DOC / name
            M.committed(path); assert M.digest(path) == sha == target['quality_sha256']
            quality = json.loads(path.read_text()); assert all(quality['gates'].values())
            assert len(target['quality_bridges']) == 96 and target['artifact'] == quality['artifact']
            observations = []; modes = {}
            for mode in ('teacher', 'natural'):
                probabilities = [[] for _ in range(12)]; counts = [Counter() for _ in range(12)]; accepted = [0] * 12
                for entry in target['quality_bridges']:
                    bi, ci = entry['book'], entry['case']; expected = quality['books'][bi]['cases'][ci]
                    suffix = '' if mode == 'teacher' else '.generation'
                    output = OUTPUTS / f'n{n}.book{bi}.case{ci}{suffix}.0.bin'
                    data = output.read_bytes(); actual = hashlib.sha256(data).hexdigest()
                    bound = expected['native_output_sha256'] if mode == 'teacher' else expected['generation']['native_generation_sha256']
                    assert actual == bound == entry['teacher_sha256' if mode == 'teacher' else 'natural_sha256']
                    t = len(entry['decoder_ids']) if mode == 'teacher' else len(expected['generation']['native']['generated_ids'])
                    encoder, decoder = route_bytes(data, n, len(entry['source_ids']), t)
                    if not observations:
                        bad = bytearray(data); bad[-4:] = struct.pack('<f', float('nan'))
                        try: route_bytes(bad, n, len(entry['source_ids']), t)
                        except AssertionError: pass
                        else: raise AssertionError('nonfinite_route_negative_not_detected')
                        bad = bytearray(data); bad[-12:-8] = struct.pack('<i', n)
                        try: route_bytes(bad, n, len(entry['source_ids']), t)
                        except AssertionError: pass
                        else: raise AssertionError('out_of_range_route_negative_not_detected')
                        result['negative_probability_and_identity_controls_detected'] = True
                    for bank, rows in enumerate([*encoder, *decoder]):
                        probabilities[bank].extend(map(float, rows['probability']))
                        counts[bank].update(map(int, rows['expert'])); accepted[bank] += int(np.sum(rows['accepted']))
                    observations.append({'book': bi, 'case': ci, 'mode': mode, 'path': str(output), 'bytes': len(data), 'sha256': actual, 'source_tokens': len(entry['source_ids']), 'decoder_positions': t})
                    del data, encoder, decoder; guard()
                banks = []
                for bank in range(12):
                    banks.append({'bank': bank, 'part': 'encoder' if bank < 6 else 'decoder', 'layer': 2 * (bank % 6) + 1,
                                  'accepted_route_queries': accepted[bank], **summarize(probabilities[bank], counts[bank], n)})
                pooled = summarize([p for bank in probabilities for p in bank], sum(counts, Counter()), n)
                # A pooled expert label is NOT a distinct function across layers.
                pooled.pop('distinct_selected_experts'); pooled.pop('expert_visit_counts')
                modes[mode] = {'banks': banks, 'pooled_probability_diagnostics': pooled}
                print(json.dumps({'n': n, 'mode': mode, 'route_queries': pooled['route_queries'], 'selected_probability_quantiles': pooled['selected_probability_quantiles']}), flush=True)
            result['targets'].append({'n': n, 'quality_sha256': sha, 'artifact_identity_from_verified389': target['artifact'], 'observations': observations, 'modes': modes})
        result['gates'] = {'ALL384_complete_outputs_exact_source_quality': len([v for t in result['targets'] for v in t['observations']]) == 384,
                           'all_trace_shapes_identities_probabilities_legal': True, 'negative_probability_and_identity_controls_detected': result['negative_probability_and_identity_controls_detected']}
        assert all(result['gates'].values())
        result['resource'] = {'seconds': time.monotonic() - start, 'maximum_checked_rss_bytes': maximum}
        result['decision'] = 'preserve_full_normalization_in_candidate_search_design_capture_full_scores_before_retrieval_choice'
        result['scope'] = 'Read-only existing exact native quality traces, two different original models/cohorts. Selected IDs/probabilities and descriptive bank coverage; no expert-level usefulness, causal n comparison, full-score ranking/tail reconstruction, quality of a changed router, physical DRAM or timing inference. Singleton scaling is algebraic sensitivity, not measured whole-model harm; candidate lower bound is necessary only. Artifact hashes inherited from freshly verified389, not rehashed here.'
        guard(); M.write(args.out, result)
        print(json.dumps({'sha256': M.digest(args.out), 'gates': result['gates'], 'resource': result['resource']}), flush=True)
    except BaseException as error:
        result.update({'stage': stage, 'error': repr(error), 'seconds': time.monotonic() - start})
        M.write(args.out.with_suffix('.failure.json'), result); raise


if __name__ == '__main__': main()
