"""Main exact row energies and outward scalar bounds on ORIGINAL525 transients."""
from fractions import Fraction
import math
import numpy as np

U = 2. ** -53


def gamma(n):
    rat = Fraction(n, (1 << 53) - n); value = float(rat)
    return math.nextafter(value, math.inf) if Fraction.from_float(value) < rat else value


def add(a, b): return np.nextafter(np.add(a, b), np.inf)
def mul(a, b): return np.nextafter(np.multiply(a, b), np.inf)
def div(a, b): return np.nextafter(np.divide(a, b), np.inf)
def sub(a, b): return np.nextafter(np.subtract(a, b), -np.inf)
def lowmul(a, b): return np.nextafter(np.multiply(a, b), -np.inf)


def relative_interval(distance2, error, observed_norm):
    g = gamma(770)
    yup = float(div(observed_norm, sub(1., g)))
    ylow = math.nextafter(observed_norm / float(add(1., g)), -math.inf)
    dl = max(float(sub(distance2, error)), 0.); dh = float(add(distance2, error))
    rl = max(math.nextafter(math.sqrt(dl), -math.inf), 0.)
    rh = math.nextafter(math.sqrt(dh), math.inf)
    return [math.nextafter(rl / yup, -math.inf), float(div(rh, ylow))]


def units(norm):
    ratios = [float(v).as_integer_ratio() for v in norm]
    power = max(d.bit_length() - 1 for n, d in ratios)
    v = [n * ((1 << power) // d) for n, d in ratios]
    bits = max(abs(n).bit_length() for n in v)
    assert power <= 64 and bits <= 52 and all(n != 0 for n in v)
    return v, power, bits


def energies(codes, v):
    assert np.all(codes >= -127)
    squares = codes.astype('<i8') ** 2
    weights = [n * n for n in v]; out = [0] * len(codes)
    for limb in range(6):
        coeff = np.array([(n >> (20 * limb)) & ((1 << 20) - 1) for n in weights], '<i8')
        block = squares @ coeff
        assert np.all(block >= 0) and np.all(block < (1 << 44))
        for j, value in enumerate(block): out[j] += int(value) << (20 * limb)
    assert all(0 <= n < (1 << 128) for n in out)
    return out


def norm_up(square, power):
    root = math.isqrt(square); root += int(root * root < square)
    rat = Fraction(root, 1 << power); value = math.ldexp(float(root), -power)
    if Fraction.from_float(value) < rat: value = math.nextafter(value, math.inf)
    return value


def refine(radius, y_norm, panel, h, pb, oq):
    """No old vector/projected coordinate is recreated; only scalar upper bounds."""
    g2, g3, gd, gn, gp = [gamma(n) for n in (2, 3, 769, 770, 1040)]
    den_norm = sub(1., gn); qop = add(1., oq); absolute_projection = mul(128., qop)
    y = div(y_norm, den_norm); bup = div(radius, den_norm)
    nup = div(add(1., g2), den_norm)
    aup = mul(add(qop, mul(gp, absolute_projection)), y)
    da = add(mul(pb, y), mul(mul(gp, absolute_projection), y))
    db = add(da, mul(g2, add(y, aup))); dr = add(db, mul(gn, bup))
    c, t, kn = panel[:, 0], panel[:, 3], panel[:, 6]
    phup = mul(add(qop, mul(gp, absolute_projection)), h)
    dk = add(add(mul(pb, h), mul(mul(gp, absolute_projection), h)), mul(g2, add(h, phup)))
    dkn = add(dk, mul(gn, div(kn, den_norm)))
    dc = add(mul(h, da), mul(gd, mul(h, aup)))
    spread = mul(radius, kn)
    de = add(add(add(add(dc, mul(radius, dkn)), mul(kn, dr)), mul(dr, dkn)), mul(g3, add(np.abs(c), spread)))
    endpoint = mul(2., de); dt = div(endpoint, 2.)
    den_kn = sub(kn, dkn); positive = den_kn > 0
    safe_den = np.where(positive, den_kn, 1.)
    dn = add(div(add(dk, dkn), safe_den), mul(g2, nup))
    abs_difference = np.nextafter(np.abs(np.subtract(t, c)), np.inf)
    uup = mul(div(abs_difference, np.where(kn > 0, kn, 1.)), add(1., g3))
    du = add(add(div(add(dt, dc), safe_den), div(mul(uup, dkn), safe_den)), mul(g3, uup))
    alignup = mul(mul(nup, bup), add(1., gd))
    dal = add(add(add(mul(dn, bup), mul(db, nup)), mul(dn, db)), mul(gd, mul(nup, bup)))
    tangentup = mul(add(bup, mul(alignup, nup)), add(1., g3))
    tnup = mul(tangentup, add(1., gn))
    dw = add(add(add(add(db, mul(alignup, dn)), mul(nup, dal)), mul(dn, dal)), mul(g3, add(bup, mul(alignup, nup))))
    dtn = add(dw, mul(gn, tangentup))
    rlow = sub(radius, dr); ulow = add(uup, du)
    radlow = sub(lowmul(np.maximum(rlow, 0.), np.maximum(rlow, 0.)), mul(ulow, ulow))
    safe = positive & (rlow > 0) & (radlow > 0)
    # A retained certificate records this lower root; the audit verifies its square exactly.
    rootlow = np.maximum(np.nextafter(np.sqrt(np.maximum(radlow, 0.)), -np.inf), 0.)
    safe &= rootlow > 0
    sup = mul(radius, add(1., g3))
    darg = add(add(add(add(mul(mul(2., radius), dr), mul(dr, dr)), mul(mul(2., uup), du)), mul(du, du)), mul(g3, add(mul(radius, radius), mul(uup, uup))))
    ds = add(div(darg, np.where(safe, rootlow, 1.)), mul(U, sup))
    cross_terms = add(mul(alignup, uup), mul(tnup, sup))
    crossup = mul(cross_terms, add(1., g3))
    cross_error = add(add(add(add(add(add(mul(alignup, du), mul(uup, dal)), mul(dal, du)), mul(tnup, ds)), mul(sup, dtn)), mul(dtn, ds)), mul(g3, cross_terms))
    distance_error = add(add(add(mul(mul(4., radius), dr), mul(2., mul(dr, dr))), mul(2., cross_error)), mul(g3, add(mul(2., mul(radius, radius)), mul(2., crossup))))
    bound = mul(4., distance_error)
    assert np.isfinite(bound).all()
    return bound, safe, np.where(safe, radlow, 0.), np.where(safe, rootlow, 0.)
