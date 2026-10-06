"""Exact rational constant-amplitude eligibility; no donor access at import."""
from fractions import Fraction
import struct

RHO = Fraction(101, 100)
ONE = 0x3f800000

def ratio(bits):
    bits = int(bits)
    assert 0 < bits <= ONE
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    if not exponent:
        return Fraction(fraction, 1 << 149)
    mantissa, power = (1 << 23) | fraction, exponent - 150
    return Fraction(mantissa << power) if power >= 0 else Fraction(mantissa, 1 << -power)

def wire_fraction(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}

def first_at_least(value):
    lo, hi = 1, ONE + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ratio(mid) >= value:
            hi = mid
        else:
            lo = mid + 1
    return lo if lo <= ONE else None

def select(minimum, maximum):
    if minimum is None:
        assert maximum is None
        return {'decision': 'NO_DEVELOPMENT_CELL', 'ideal_constant_exists': None, 'candidate_bits': None}
    low, high = ratio(maximum) / RHO, ratio(minimum) * RHO
    exists = low <= high
    candidate = first_at_least(low) if exists else None
    if candidate is not None and ratio(candidate) > high:
        candidate = None
    decision = ('IDEAL_CONSTANT_EXCLUDED' if not exists else
                'IDEAL_ELIGIBLE_NO_F32' if candidate is None else 'F32_CONSTANT_ELIGIBLE')
    return {'decision': decision, 'ideal_constant_exists': exists, 'candidate_bits': candidate,
            'allowed_constant_interval': [wire_fraction(low), wire_fraction(high)],
            'extrema_ratio': wire_fraction(ratio(maximum) / ratio(minimum))}

def accepted_bits(candidate):
    low, high = ratio(candidate) / RHO, ratio(candidate) * RHO
    first = first_at_least(low)
    last = first_at_least(high)
    if last is None:
        last = ONE
    elif ratio(last) > high:
        last -= 1
    assert first is not None and 1 <= first <= last <= ONE
    return first, last

def controls():
    def bits(value):
        return struct.unpack('<I', struct.pack('<f', value))[0]
    boundary = (bits(10000 / 16384), bits(10201 / 16384))
    # Both p in [.5,1) have integer significands. The allowed c significand
    # lies in [10100050+50/101,10100050+1/2], containing no integer.
    no_float = tuple(0x3f000000 + (n - (1 << 23)) for n in (10000050, 10201051))
    cases = [('one_point', (bits(.5), bits(.5)), 'F32_CONSTANT_ELIGIBLE'),
             ('exact_rho_squared_boundary', boundary, 'F32_CONSTANT_ELIGIBLE'),
             ('one_F32_beyond_boundary', (boundary[0], boundary[1]+1), 'IDEAL_CONSTANT_EXCLUDED'),
             ('ideal_interval_without_F32', no_float, 'IDEAL_ELIGIBLE_NO_F32'),
             ('positive_subnormal', (1, 1), 'F32_CONSTANT_ELIGIBLE'),
             ('probability_one', (ONE, ONE), 'F32_CONSTANT_ELIGIBLE'),
             ('empty_cell', (None, None), 'NO_DEVELOPMENT_CELL'),
             ('validation_only_cell', (None, None), 'NO_DEVELOPMENT_CELL')]
    out = []
    for name, pair, expected in cases:
        result = select(*pair)
        assert result['decision'] == expected, name
        if result['candidate_bits'] is not None:
            bounds = accepted_bits(result['candidate_bits'])
            assert bounds[0] <= pair[0] <= pair[1] <= bounds[1]
        out.append({'control': name, 'input_extrema_bits': list(pair), 'expected': expected, **result})
    assert ratio(1) == Fraction(1, 1 << 149) and ratio(ONE) == 1
    assert ratio(boundary[1])/ratio(boundary[0]) == RHO*RHO
    return out
