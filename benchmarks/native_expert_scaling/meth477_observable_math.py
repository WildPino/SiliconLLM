"""Prospective reporting and high-precision toy controls, no fitting."""
from decimal import Decimal, localcontext
import math

FIELDS = ('kl', 'tv', 'margin_source', 'margin_candidate', 'delta_min', 'delta_max',
          'pmax_source', 'pmax_candidate', 'fisher_half', 'nll_source', 'nll_candidate',
          'delta_nll', 'error_energy', 'reference_energy', 'route_probability', 'range_KL_bound')
TOYS = [([0., 1., -1.], [0., 1. + 2.**-10, -1. - 2.**-10], 1),
        ([10000., 9999., -10000.], [10016., 10015., -9984.], 0),
        ([0., 0., 0.], [0., 0., 0.], 2),
        ([1000., 999., -1000.], [999., 1000., -999.], 1),
        ([0., -1., -2.], [0., -1. + 2.**-20, -2. - 2.**-20], 2),
        ([0., 0., -1.], [0., 2.**-10, -1.], 0)]

def decimal_oracle(source, candidate, label):
    with localcontext() as context:
        context.prec = 100
        s, c = [[Decimal.from_float(float(x)) for x in a] for a in (source, candidate)]
        es, ec = [[(x-max(a)).exp() for x in a] for a in (s, c)]
        ps, pc = [[x/sum(a) for x in a] for a in (es, ec)]
        zs, zc = max(s)+sum(es).ln(), max(c)+sum(ec).ln()
        delta = [y-x for x, y in zip(s, c)]; mu = sum(p*d for p, d in zip(ps, delta))
        kl = sum(p*((x-zs)-(y-zc)) for p, x, y in zip(ps, s, c))
        tv = sum(abs(p-q) for p, q in zip(ps, pc))/2
        def margin(a):
            ordered = sorted(a, reverse=True); return ordered[0]-ordered[1]
        f = sum(p*(d-mu)**2 for p, d in zip(ps, delta))/2
        n0, n1 = zs-s[label], zc-c[label]
        values = [kl, tv, margin(s), margin(c), min(delta), max(delta), max(ps), max(pc), f,
                  n0, n1, n1-n0, Decimal(0), Decimal(0), Decimal(0), (max(delta)-min(delta))**2/8]
        return [float(v) for v in values]

def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual-expected) <= 1e-11 + 1e-10*abs(expected), (actual, expected)

def quantile(values, p):
    ordered = sorted(values); at = (len(ordered)-1)*p; lo = int(at); hi = min(lo+1, len(ordered)-1)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(at-lo)

def summary(records):
    if not records: return {'count': 0}
    result = {'count': len(records), 'argmax_changed': sum(bool(r['flags'] & 128) for r in records),
              'unique_source': sum(bool(r['flags'] & 32) for r in records),
              'margin_certified': sum(bool(r['flags'] & 256) for r in records),
              'negative_raw_KL': sum(r['kl'] < 0 for r in records)}
    for name in ('kl', 'tv', 'margin_source', 'delta_range', 'fisher_half'):
        values = [r['delta_max']-r['delta_min'] if name == 'delta_range' else max(0., r[name]) if name == 'kl' else r[name] for r in records]
        result[name] = {'mean': math.fsum(values)/len(values), 'p50': quantile(values, .5),
                        'p95': quantile(values, .95), 'p99': quantile(values, .99), 'max': max(values)}
    labels = [r for r in records if r['flags'] & 4]
    masked = [r for r in records if r['flags'] & 8]
    for name, group in (('teacher_all_label', labels), ('teacher_masked_content_label', masked)):
        if group:
            result[name] = {'count': len(group), 'source_mean_NLL': math.fsum(r['nll_source'] for r in group)/len(group),
                            'candidate_mean_NLL': math.fsum(r['nll_candidate'] for r in group)/len(group),
                            'mean_delta_NLL': math.fsum(r['delta_nll'] for r in group)/len(group),
                            'source_label_argmax_correct': sum(r['argmax_source'] == r['label'] for r in group),
                            'candidate_label_argmax_correct': sum(r['argmax_candidate'] == r['label'] for r in group)}
    numerator = math.fsum(r['error_energy'] for r in records); denominator = math.fsum(r['reference_energy'] for r in records)
    result['pooled_FFN_RMS'] = math.sqrt(numerator/denominator) if denominator > 0 else None
    result['reference_energy'] = denominator
    return result

def all_summaries(records):
    groups = {'all': records}
    for mode, name in ((0, 'teacher'), (1, 'natural')):
        groups[name] = [r for r in records if r['mode'] == mode]
    for name, bit in (('ready', 1), ('novel_code', 2), ('dominant25_37', 16)):
        for value in (0, 1):
            key = name if value else {'ready': 'fallback', 'novel_code': 'seen_code', 'dominant25_37': 'other_IDs'}[name]
            groups[key] = [r for r in records if bool(r['flags'] & bit) == bool(value)]
            for mode, label in ((0, 'teacher'), (1, 'natural')):
                groups[key + '_' + label] = [r for r in groups[key] if r['mode'] == mode]
    return {'views': {k: summary(v) for k, v in groups.items()},
            'books': {str(b): summary([r for r in records if r['book'] == b]) for b in range(64, 128)},
            'experts': {str(e): {name: summary([r for r in records if r['expert'] == e and (mode is None or r['mode'] == mode)])
                                for mode, name in ((None, 'all'), (0, 'teacher'), (1, 'natural'))} for e in range(128)}}
