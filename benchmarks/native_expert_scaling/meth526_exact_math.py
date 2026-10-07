"""526 exact dyadic products and rational outward Frobenius/projector bounds."""
from fractions import Fraction
import math
import numpy as np


def common_units(values):
    ratios = [float(v).as_integer_ratio() for v in values.flat]
    power = max(den.bit_length() - 1 for num, den in ratios)
    a = np.array([num * ((1 << power) // den) for num, den in ratios], dtype=object).reshape(values.shape)
    bits = max(abs(int(v)).bit_length() for v in a.flat)
    return a, power, bits


def norm_upper(array, power):
    square = sum(int(v) ** 2 for v in array.flat)
    root = math.isqrt(square); ceiling = root + int(root * root < square)
    value = Fraction(ceiling, 1 << power)
    record = dict(square_sum=str(square), denominator_power=power, ceil_integer_root=str(ceiling),
                  upper_rational=[str(value.numerator), str(value.denominator)])
    return value, record


def upward(value):
    rounded = float(value)
    if Fraction.from_float(rounded) < value: rounded = math.nextafter(rounded, math.inf)
    assert math.isfinite(rounded) and Fraction.from_float(rounded) >= value
    return rounded


def certificate(gram, residual, source, qpower, cpower, sigma):
    oq, qr = norm_upper(gram, 2 * qpower)
    rb, rr = norm_upper(residual, cpower + 2 * qpower)
    cnorm, cr = norm_upper(source, cpower)
    assert oq < 1 and sigma > 0
    lower = Fraction.from_float(sigma)
    base = 2 * (rb + cnorm * oq / (1 - oq)) / lower + 2 * oq / (1 - oq)
    upper = 2 * base
    report = dict(orthogonality_upper=upward(oq), rowspace_residual_upper=upward(rb),
        source_Frobenius_upper=upward(cnorm), reused_source_singular_lower=sigma,
        projector_upper_rational=[str(upper.numerator), str(upper.denominator)],
        projector_error_bound=upward(upper))
    return report, dict(orthogonality=qr, rowspace_residual=rr, source_Frobenius=cr)
