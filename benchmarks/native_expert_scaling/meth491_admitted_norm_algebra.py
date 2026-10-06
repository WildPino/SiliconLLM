"""Exact post-hoc elimination of affine bias from admitted fixed witnesses."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import hashlib
import json
from pathlib import Path
from fractions import Fraction
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import struct
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]; DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
OUT = DOC / 'meth491_admitted_norm_algebra.json'; FAIL = OUT.with_suffix('.failure.json')
assert not OUT.exists() and not FAIL.exists()
START = time.monotonic(); READS = []
def read(path, expected):
    path = Path(path); data = path.read_bytes(); sha = hashlib.sha256(data).hexdigest()
    assert sha == expected, str(path)
    READS.append({'path': str(path), 'bytes': len(data), 'sha256': sha})
    assert time.monotonic()-START < 30
    return data
def save(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False)+'\n').replace('\n', '\r\n').encode()
    assert len(data) < 256 << 10
    with path.open('xb') as stream: stream.write(data)

try:
    admission = json.loads(read(DOC / 'ADMISSION_491_20261006.json', '4810c933c4a2a6de94653fb75935ac625c66e66e24a5ffbdb330e8dbe7158338'))
    raw = json.loads(read(admission['raw']['path'], admission['raw']['sha256']))
    ret = json.loads(read(admission['retention']['path'], admission['retention']['sha256']))
    assert all(admission['gates'].values()) and all(ret['gates'].values()) and ret['witnesses_audited'] == 12
    binding = json.loads(read(DOC / 'meth491_binding.json', raw['binding_sha256']))
    known = {v['path']: v for v in binding['catalog'] + raw['output_inventory']}
    oldpath = str(ROOT / 'results/native_expert_scaling/meth490_r1_root_mass/heads.bin')
    payload = read(oldpath, known[oldpath]['sha256'])
    assert payload[:24] == struct.pack('<8sIIQ', b'M490HED1', 3076, 768, 12)
    heads = struct.unpack('<9228f', payload[24:])
    import numpy as np
    import psutil
    assert np.__version__ == binding['runtime']['packages']['numpy']['version']
    import meth491_mass_interval_math as intervals
    primary = intervals.Intervals(raw['eta'])
    near = Context(prec=200, rounding=ROUND_HALF_EVEN)
    down = Context(prec=200, rounding=ROUND_FLOOR); up = Context(prec=200, rounding=ROUND_CEILING)
    eta = Decimal.from_float(raw['eta'])
    def trans(value, name):
        value = getattr(near, name)(value)
        return near.next_minus(value), near.next_plus(value)
    def independent_endpoints(q, right):
        q = Decimal.from_float(float(q)); logq = trans(q, 'ln'); result = []
        for offset in (eta.copy_negate(), eta):
            exp = trans(offset, 'exp')
            complement = (down.subtract(Decimal(1), up.multiply(q, exp[1])),
                          up.subtract(Decimal(1), down.multiply(q, exp[0])))
            assert complement[0] > 0
            result.append((down.subtract(down.add(logq[0], offset), trans(complement[1], 'ln')[1]),
                           up.subtract(up.add(logq[1], offset), trans(complement[0], 'ln')[0])))
        if right: return result
        return [(result[1][1].copy_negate(), result[1][0].copy_negate()),
                (result[0][1].copy_negate(), result[0][0].copy_negate())]
    dtype = np.dtype([('uid', '<u4'), ('q', '<f8', (2,)), ('right', '<u4'), ('rest', '<f8', (3,))])
    desc = binding['mass_source']['diagnostics']
    payload = read(desc['path'], desc['sha256'])
    assert payload[:24] == struct.pack('<8sIIQ', b'M490DIA1', 48, 0, 238872)
    diag = np.frombuffer(payload, dtype, offset=24)
    def decimal_floor(value):
        return str(down.divide(Decimal(value.numerator), Decimal(value.denominator)))
    result = []
    for witness in raw['witnesses']:
        bank = witness['bank']; values = heads[bank*769:(bank+1)*769]
        slopes_norm = sum((abs(Fraction.from_float(v)) for v in values[:768]), Fraction())
        head_norm = slopes_norm + abs(Fraction.from_float(values[768]))
        row = {'bank': bank, 'retained490_raw_F32_slope_L1': decimal_floor(slopes_norm),
               'retained490_raw_F32_total_L1': decimal_floor(head_norm), 'witness_decision': witness['decision']}
        if witness['coefficient_L1_lower_bound'] is not None:
            path = witness['witness_path']
            import io
            with np.load(io.BytesIO(read(path, known[path]['sha256'])), allow_pickle=False) as archive:
                uid = int(archive['uid'][0]); x = archive['input'][0].copy()
            assert diag['uid'][uid] == uid
            right = bool(diag['right'][uid]); q = float(diag['q'][uid, int(right)])
            endpoints = primary.endpoints(q, right)
            independent = independent_endpoints(q, right)
            assert all(lo <= alo <= ahi <= hi for (lo, hi), (alo, ahi) in zip(endpoints, independent))
            L, U = Fraction(endpoints[0][0]), Fraction(endpoints[1][1])
            numerators = list(map(int, witness['residual_numerators']))
            bias = Fraction(numerators[768], 1 << 149)
            maximum_bias_contribution = bias*(U if bias >= 0 else L)
            C = Fraction(Decimal(witness['chosen_C_interval'][0]))
            gap = C - maximum_bias_contribution
            transformed = []
            for index, value in enumerate(x):
                source_value = Fraction.from_float(float(value))
                r = Fraction(numerators[index], 1 << 149) - bias*source_value
                # Independently check the common-denominator algebraic expression.
                n, d = source_value.numerator, source_value.denominator
                check = Fraction(numerators[index]*d - numerators[768]*n, (1 << 149)*d)
                assert r == check; transformed.append(abs(r))
            maximum = max(transformed)
            slope_bound = gap/maximum if gap > 0 and maximum > 0 else None
            row.update(reference_UID=uid, reference_input_role='development-first-fixed-witness-UID',
                reference_logit_outer_interval=[str(endpoints[0][0]), str(endpoints[1][1])],
                exact_bias_residual_numerator=str(numerators[768]),
                bias_eliminated_gap_lower=decimal_floor(gap),
                transformed_slope_residual_Linf=decimal_floor(maximum),
                ideal_slope_L1_necessary_lower_bound=decimal_floor(slope_bound) if slope_bound else None,
                bound_to_retained490_slope_norm_ratio=decimal_floor(slope_bound/slopes_norm) if slope_bound and slopes_norm else None,
                slope_bound_exact_ratio={'numerator': str(slope_bound.numerator), 'denominator': str(slope_bound.denominator)} if slope_bound else None)
        result.append(row)
        assert time.monotonic()-START < 30 and psutil.Process().memory_info().peak_wset <= 256 << 20
    value = {'classification': 'POST_HOC_EXACT_ALGEBRAIC_COROLLARY_OF_ADMITTED_FIXED_WITNESSES',
        'inputs': READS, 'source_helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'no_new_SVD_weights_fit_native_model_or_cohort': True, 'banks': result,
        'scope': 'Necessary ideal raw slope norm bounds after eliminating bias with an existing source constraint; no generic C impossibility or task/rate/robustness promotion.',
        'resource': {'wall_seconds': time.monotonic()-START, 'peak_working_set_bytes': psutil.Process().memory_info().peak_wset,
                     'wall_limit_seconds': 30, 'host_limit_bytes': 256 << 20}}
    save(OUT, value)
    print(json.dumps({'banks': [{k:v for k,v in row.items() if k in ('bank','retained490_raw_F32_slope_L1','ideal_slope_L1_necessary_lower_bound','bound_to_retained490_slope_norm_ratio')} for row in result], 'resource': value['resource']}))
except BaseException as exc:
    save(FAIL, {'error': str(exc), 'traceback': traceback.format_exc(), 'inputs': READS, 'wall_seconds': time.monotonic()-START})
    raise
