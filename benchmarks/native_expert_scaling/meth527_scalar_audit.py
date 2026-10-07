"""Independent scalar outward reductions for the prospective527 bound equations."""
from fractions import Fraction
import math

EPS = 2. ** -53


def gam(n):
    exact = Fraction(n, (1 << 53) - n); x = float(exact)
    return x if Fraction.from_float(x) >= exact else math.nextafter(x, math.inf)


def plus(a, b): return math.nextafter(a + b, math.inf)
def times(a, b): return math.nextafter(a * b, math.inf)
def over(a, b): return math.nextafter(a / b, math.inf)
def minus(a, b): return math.nextafter(a - b, -math.inf)
def square_lower(x): return math.nextafter(x * x, -math.inf)


def certificate(r, yn, c, t, k, hn, projector_bound, orth_bound):
    gsmall, gthree, gdot, gnorm, gproj = [gam(i) for i in (2, 3, 769, 770, 1040)]
    denominator = minus(1., gnorm)
    qnorm2 = plus(1., orth_bound); absolute_norm2 = times(128., qnorm2)
    norm_y = over(yn, denominator); norm_b = over(r, denominator)
    norm_n = over(plus(1., gsmall), denominator)
    norm_a = times(plus(qnorm2, times(gproj, absolute_norm2)), norm_y)
    error_a = plus(times(projector_bound, norm_y), times(times(gproj, absolute_norm2), norm_y))
    error_b = plus(error_a, times(gsmall, plus(norm_y, norm_a)))
    error_r = plus(error_b, times(gnorm, norm_b))
    norm_ph = times(plus(qnorm2, times(gproj, absolute_norm2)), hn)
    error_k = plus(plus(times(projector_bound, hn), times(times(gproj, absolute_norm2), hn)), times(gsmall, plus(hn, norm_ph)))
    error_kn = plus(error_k, times(gnorm, over(k, denominator)))
    error_c = plus(times(hn, error_a), times(gdot, times(hn, norm_a)))
    error_endpoint = plus(plus(plus(plus(error_c, times(r, error_kn)), times(k, error_r)), times(error_r, error_kn)), times(gthree, plus(abs(c), times(r, k))))
    error_t = over(times(2., error_endpoint), 2.)
    lower_k = minus(k, error_kn)
    if lower_k <= 0: return None
    error_n = plus(over(plus(error_k, error_kn), lower_k), times(gsmall, norm_n))
    abs_difference = math.nextafter(abs(t - c), math.inf)
    norm_u = times(over(abs_difference, k), plus(1., gthree))
    error_u = plus(plus(over(plus(error_t, error_c), lower_k), over(times(norm_u, error_kn), lower_k)), times(gthree, norm_u))
    norm_alignment = times(times(norm_n, norm_b), plus(1., gdot))
    error_alignment = plus(plus(plus(times(error_n, norm_b), times(error_b, norm_n)), times(error_n, error_b)), times(gdot, times(norm_n, norm_b)))
    norm_tangent = times(plus(norm_b, times(norm_alignment, norm_n)), plus(1., gthree))
    norm_tangent_norm = times(norm_tangent, plus(1., gnorm))
    error_tangent = plus(plus(plus(plus(error_b, times(norm_alignment, error_n)), times(norm_n, error_alignment)), times(error_n, error_alignment)), times(gthree, plus(norm_b, times(norm_alignment, norm_n))))
    error_tangent_norm = plus(error_tangent, times(gnorm, norm_tangent))
    lower_r = minus(r, error_r); upper_u = plus(norm_u, error_u)
    lower_argument = minus(square_lower(max(lower_r, 0.)), times(upper_u, upper_u))
    if lower_r <= 0 or lower_argument <= 0: return None
    lower_root = max(math.nextafter(math.sqrt(lower_argument), -math.inf), 0.)
    if lower_root <= 0: return None
    # This exact dyadic check makes no assumption about libm's sqrt rounding.
    assert Fraction.from_float(lower_root) ** 2 <= Fraction.from_float(lower_argument)
    upper_root = times(r, plus(1., gthree))
    error_argument = plus(plus(plus(plus(times(times(2., r), error_r), times(error_r, error_r)), times(times(2., norm_u), error_u)), times(error_u, error_u)), times(gthree, plus(times(r, r), times(norm_u, norm_u))))
    error_root = plus(over(error_argument, lower_root), times(EPS, upper_root))
    product_upper = plus(times(norm_alignment, norm_u), times(norm_tangent_norm, upper_root))
    cross_upper = times(product_upper, plus(1., gthree))
    error_cross = plus(plus(plus(plus(plus(plus(times(norm_alignment, error_u), times(norm_u, error_alignment)), times(error_alignment, error_u)), times(norm_tangent_norm, error_root)), times(upper_root, error_tangent_norm)), times(error_tangent_norm, error_root)), times(gthree, product_upper))
    error_distance = plus(plus(plus(times(times(4., r), error_r), times(2., times(error_r, error_r))), times(2., error_cross)), times(gthree, plus(times(2., times(r, r)), times(2., cross_upper))))
    return times(4., error_distance), lower_argument, lower_root


def relative_interval(d, e, yn):
    g = gam(770); yup = over(yn, minus(1., g)); ylo = math.nextafter(yn / plus(1., g), -math.inf)
    dlo, dhi = max(minus(d, e), 0.), plus(d, e)
    rlo = max(math.nextafter(math.sqrt(dlo), -math.inf), 0.); rhi = math.nextafter(math.sqrt(dhi), math.inf)
    assert Fraction.from_float(rlo) ** 2 <= Fraction.from_float(dlo)
    assert Fraction.from_float(rhi) ** 2 >= Fraction.from_float(dhi)
    return [math.nextafter(rlo / yup, -math.inf), over(rhi, ylo)]
