"""Fixed full-rank readout, immutable development plan and inherited metric contract."""
import hashlib
import math
import struct
from pathlib import Path
import numpy as np
import meth499_math as M
from meth498_math import triangular
from meth521_operations import wire

PAIR = np.dtype([('expert', '<u2'), ('input_owner', '<u2'), ('UID', '<u4')])
RESPONSE = np.dtype([('UID', '<u4'), ('expert', '<u4'), ('input_owner', '<u4'), ('nonzero', '<u4'),
                     ('alpha', '<f4'), ('codes', '<i2', (3072,)), ('F', '<f4', (768,))])


def load(ctx):
    uid = wire(np, ctx.data('uid'), b'M493U001', 180, 0, M.UID, 17540); m = uid['m']
    occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
    ni = wire(np, ctx.data('inputs'), b'M499INP1', 4624, 768, M.NATIVE_INPUT, 17540)
    inputs = ni['core']; ps = ni['source_p']
    feat = wire(np, ctx.data('features'), b'M495FEA1', 6148, 512, M.FEATURE, 17540)
    target = wire(np, ctx.data('unweighted'), b'M499F001', 3072, 768, np.dtype(('<f4', (768,))), 17540)
    y = wire(np, ctx.data('targets'), b'M493Y001', 3072, 768, np.dtype(('<f4', (768,))), 17540)
    dev = (m[:, 4] & 5) != 0; val = (m[:, 4] & 2) != 0
    assert np.array_equal(m[:, 0], np.arange(17540)) and np.all(m[:, 2] == 11) and np.all(m[:, 3] < 128) and np.all(m[:, 6] == 2)
    assert (int(dev.sum()), int(val.sum())) == (11721, 5819) and np.all(dev ^ val)
    assert np.array_equal(inputs['e'], m[:, 3]) and np.all(inputs['accept'] == 1)
    assert ps.tobytes() == m[:, 10].tobytes() and inputs['alpha'].tobytes() == m[:, 12].tobytes()
    assert np.all(np.isfinite(ps) & (ps > 0) & (ps <= 1))
    assert np.array_equal(occ[:, 0], np.arange(19962)) and np.all(occ[:, 1] < 17540)
    assert np.all(occ[:, 3] == 128) and np.all(occ[:, 8] == 11) and np.all(occ[:, 12] == 1)
    for oi, mi in ((14, 1), (11, 3), (15, 12), (16, 10)): assert np.array_equal(occ[:, oi], m[occ[:, 1], mi])
    assert np.array_equal(np.bincount(occ[:, 1], minlength=17540), m[:, 7])
    for k in range(17540):
        xb = inputs['x'][k].tobytes(); qb = inputs['q'][k].tobytes(); ab = inputs['alpha'][k].tobytes(); fb = target[k].tobytes()
        hashes = b''.join(hashlib.sha256(v).digest() for v in (xb, qb, qb + ab, fb))
        assert hashes == uid['hash'][k].tobytes()
        assert np.multiply(target[k], ps[k], dtype=np.float32).tobytes() == y[k].tobytes()
        if k % 256 == 0: ctx.guard()
    pairs = np.load(ctx.data('pairs'), allow_pickle=False); aug = np.load(ctx.data('augmented'), allow_pickle=False)
    prov = np.load(ctx.data('provenance'), allow_pickle=False)
    assert pairs.dtype == PAIR and pairs.shape == (53943,) and aug.dtype == np.dtype('<u4') and aug.shape == (128, 513)
    assert np.array_equal(prov['UID'], np.flatnonzero(dev)) and prov['input_code_pair_SHA'].tobytes() == uid['hash'][dev, :96].tobytes()
    cursor = 0
    for e in range(128):
        own = np.flatnonzero(dev & (m[:, 3] == e)); added = aug[e, len(own):]
        assert np.array_equal(aug[e, :len(own)], own) and len(np.unique(aug[e])) == 513 and np.all(dev[aug[e]])
        part = pairs[cursor:cursor + len(added)]
        assert np.all(part['expert'] == e) and np.array_equal(part['UID'], added)
        assert np.array_equal(part['input_owner'], m[added, 3]) and np.all(part['input_owner'] != e)
        cursor += len(added)
    assert cursor == 53943
    with Path(ctx.data('geometry')).open('rb') as f:
        assert f.read(20) == struct.pack('<8sIII', b'M494GEO1', 768, 512, 513)
        f.seek(20 + 3 * 768 * 768 * 8 + 513 * 513 * 8)
        kreg = np.fromfile(f, '<f8', 513 * 513).reshape(513, 513)
    assert np.array_equal(kreg, kreg.T)
    return uid, occ, inputs, ps, feat, target, y, pairs, aug, kreg


def design(feat, ids):
    h = np.ones((len(ids), 513), '<f8')
    h[:, :512] = feat['q'][ids].astype('<f8') * feat['alpha'][ids, None].astype('<f8')
    return h


def unique(h, rhs, t, source_energy, guard):
    assert h.shape == (len(t), len(t)) and rhs.shape[0] == len(t)
    z = triangular(t, h.T, True, guard); white = M.envelope(t, z, h.T, 3e-12)
    q, r = np.linalg.qr(z); factor = M.envelope(q, r, z, 5e-12)
    orth = float(np.max(np.abs(q.T @ q - np.eye(len(t))))); assert orth <= 5e-11
    v = triangular(r.T, rhs, True, guard); solve = M.envelope(r.T, v, rhs, 3e-12)
    whitened = q @ v; transposed = triangular(t.T, whitened, False, guard)
    back = M.envelope(t.T, transposed, whitened, 3e-12); c = transposed.T
    feasible = M.envelope(h, c.T, rhs, 3e-10)
    diff = h @ c.T - rhs; rms = math.sqrt(float(np.sum(diff * diff)) / source_energy); assert rms <= 1e-7
    assert np.isfinite(c).all()
    return c, {'constraints': len(h), 'whitening_ratio': white, 'QR_ratio': factor, 'orthogonality': orth,
               'solve_ratio': solve, 'backtransform_ratio': back, 'feasibility_ratio': feasible,
               'augmented_F64_fit_RMS': rms, 'unique_fullrank_readout': True,
               'QR_min_abs_diagonal': float(np.min(np.abs(np.diag(r))))}


def controls(native):
    h = np.array([[2., -1., 1.], [0., 3., 1.], [1., 0., 1.]])
    y = np.array([[1., 3.], [2., -1.], [0., 4.]])
    c, record = unique(h, y, np.diag([2., 1., 3.]), float(np.sum(y * y)), lambda: None)
    expected = np.array([[2.5, 1.5, -2.5], [-4., -3., 8.]])
    assert np.max(np.abs(c - expected)) <= 1e-13
    q, scale = M.quant(np.array([[127, -127, 2.5, 3.5, -2.5, -3.5, 0, 1]], '<f4'))
    assert q.tolist() == [[127, -127, 2, 4, -2, -4, 0, 1]] and scale.tolist() == [1.]
    expected_dot = -128 * 32767 * 3072
    assert native[0]['source_full_width_I64'] == expected_dot and native[0]['zero_alpha'] == 1.
    assert native[0]['alpha_bits'] == 0x3f800000 and native[0]['codes'] == [32767, -32767, 2, 4, -2, -4, 0, 1]
    assert native[0]['CPU10'] == 1024 and native[0]['RNE'] == 0 and not native[0]['MXCSR'] & 0x8040
    x = [32765, -19875, 301, -101]
    table = [sum(((p // 3 ** j) % 3 - 1) * x[j] for j in range(4)) for p in range(81)]
    assert native[1] == {'new_table81': table}
    product_bits = [2, 2, 2, 4, 0x80000002, 0x3e000003]
    actual = np.multiply(np.array([7, 9, 10, 14, 0x8000000a, 0x3f000003], '<u4').view('<f4'), np.float32(.25), dtype=np.float32)
    assert actual.view('<u4').tolist() == product_bits and native[2] == {'new_quarter_product_bits': product_bits}
    return {'unique_coefficients': c.tolist(), 'unique_control_record': record, 'I8_codes': q.tolist(), 'I8_scales': scale.tolist(), 'native': native}


def augmented_labels(target, response, pairs, aug, m, e):
    own = np.flatnonzero(((m[:, 4] & 5) != 0) & (m[:, 3] == e))
    at = np.flatnonzero(pairs['expert'] == e); part = response[at]
    assert np.array_equal(part['UID'], pairs['UID'][at]) and np.all(part['expert'] == e)
    assert np.array_equal(part['input_owner'], pairs['input_owner'][at])
    assert np.array_equal(aug[e, len(own):], part['UID'])
    return np.concatenate((target[own], part['F'])).astype('<f8'), len(own)


def decision(outcomes):
    if not (outcomes['unweighted_fit64_ALL_six_RMS_1pct'] and outcomes['weighted_fit64_ALL_six_RMS_1pct']):
        return 'FIXED_CLASS_FULL_SOURCE_INFORMATION_RECIPE_CLOSED'
    return M.decision(outcomes)
