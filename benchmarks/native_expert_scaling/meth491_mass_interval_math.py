"""Directed decimal mass intervals and exact F32 dyadic residuals."""
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import struct

class Intervals:
    def __init__(self, eta, precision=100):
        self.near = Context(prec=precision, rounding=ROUND_HALF_EVEN)
        self.down = Context(prec=precision, rounding=ROUND_FLOOR)
        self.up = Context(prec=precision, rounding=ROUND_CEILING)
        exact = Decimal.from_float(float(eta))
        self.eta = exact
        self.minus = self.transcend(exact.copy_negate(), 'exp')
        self.plus = self.transcend(exact, 'exp')
    def transcend(self, value, method):
        rounded = getattr(self.near, method)(value)
        return self.near.next_minus(rounded), self.near.next_plus(rounded)
    def finite(self, q):
        q = Decimal.from_float(float(q))
        assert 0 < q <= 1
        return self.up.multiply(q, self.plus[1]) < 1
    def logit(self, low, high):
        assert 0 < low <= high < 1
        complement_low = self.down.subtract(Decimal(1), high)
        complement_high = self.up.subtract(Decimal(1), low)
        lower = self.down.subtract(self.transcend(low, 'ln')[0],
                                   self.transcend(complement_high, 'ln')[1])
        upper = self.up.subtract(self.transcend(high, 'ln')[1],
                                 self.transcend(complement_low, 'ln')[0])
        return lower, upper
    def endpoints(self, q, right):
        q = Decimal.from_float(float(q))
        assert 0 < q <= 1
        low = self.logit(self.down.multiply(q, self.minus[0]),
                         self.up.multiply(q, self.minus[1]))
        bottom = min(Decimal(1), self.down.multiply(q, self.plus[0]))
        top = min(Decimal(1), self.up.multiply(q, self.plus[1]))
        if top == 1:
            return None
        high = self.logit(bottom, top)
        if right:
            return low, high
        return (high[1].copy_negate(), high[0].copy_negate()), (low[1].copy_negate(), low[0].copy_negate())
    def combination(self, weights, endpoints):
        low = high = Decimal(0)
        for weight, (lower, upper) in zip(weights, endpoints):
            weight = int(weight)
            used = lower if weight >= 0 else upper
            lo, hi = used if weight >= 0 else (used[1], used[0])
            low = self.down.add(low, self.down.multiply(Decimal(weight), lo))
            high = self.up.add(high, self.up.multiply(Decimal(weight), hi))
        return low, high
    def norm_bound(self, lower, maximum_residual_numerator):
        if lower <= 0:
            return None
        if maximum_residual_numerator == 0:
            return 'UNRESTRICTED_INFEASIBILITY'
        numerator = self.down.multiply(lower, Decimal(1 << 149))
        return str(self.down.divide(numerator, Decimal(maximum_residual_numerator)))

def scaled_f32(bits):
    bits = int(bits)
    exponent = (bits >> 23) & 255
    assert exponent != 255
    fraction = bits & 0x7fffff
    value = fraction if exponent == 0 else ((1 << 23) | fraction) << (exponent - 1)
    return -value if bits >> 31 else value

def exact_residual(bits, weights, guard):
    assert bits.shape == (len(weights), 768)
    total = [0] * 769
    for index, (row, weight) in enumerate(zip(bits, weights)):
        weight = int(weight)
        for column, value in enumerate(row):
            total[column] += weight * scaled_f32(value)
        total[768] += weight << 149
        if index % 64 == 0:
            guard()
    return total

def controls(eta):
    # Bit decoder independently compared with float.as_integer_ratio, including extremes.
    values = [0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800000,
              0xbf800000, 0x7f7fffff, 0xff7fffff]
    for bits in values:
        number = float(struct.unpack('<f', struct.pack('<I', bits))[0])
        num, den = number.as_integer_ratio()
        assert scaled_f32(bits) == num * ((1 << 149) // den)
    interval = Intervals(eta)
    assert interval.endpoints(1., True) is None
    zero = Intervals(0.)
    bounds = zero.endpoints(.5, True)
    assert all(low <= 0 <= high for low, high in bounds)
    endpoints = [interval.endpoints(q, True) for q in (.5, .95, .5)]
    weights = [-1, 2, -1]
    low, high = interval.combination(weights, endpoints)
    # Three points x=0,1,2 with these weights have exact affine residual zero.
    assert sum(weights) == sum(w*x for w, x in zip(weights, (0, 1, 2))) == 0
    assert low > 5 and interval.norm_bound(low, 0) == 'UNRESTRICTED_INFEASIBILITY'
    negative = interval.combination([-w for w in weights], endpoints)
    assert negative[1] < 0
    return {'F32_exact_decoder_controls': len(values), 'q1_upper_unbounded': True,
            'eta0_qhalf_interval_contains_zero': True,
            'known_collinear_contradiction_lower': str(low), 'known_collinear_contradiction_upper': str(high),
            'known_collinear_residual_zero': True, 'negative_sign_upper': str(negative[1])}
