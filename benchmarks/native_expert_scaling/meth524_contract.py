"""Shared dtype, source dimensions and file I/O only; no gain/selection math."""
from pathlib import Path
import numpy as np
from meth524_operations import wire

N, D, H, E = 17540, 768, 3072, 128
SOURCE_BYTES, SOURCE_MACS = 4733952, 4718592
LEAF_BYTES, LEAF_MACS, NODE_BYTES = 1975296, 1376256, 788
UID = np.dtype([('m', '<u4', (13,)), ('hash', 'u1', (128,))])
INPUT = np.dtype([('e', '<u4'), ('accept', '<u4'), ('alpha', '<f4'), ('x', '<f4', (D,)), ('q', '<i2', (D,)), ('p', '<f4')])


def load(ctx):
    m = wire(np, ctx.data('uid'), b'M493U001', 180, 0, UID, N)['m']
    occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
    inputs = wire(np, ctx.data('inputs'), b'M499INP1', 4624, D, INPUT, N)
    assert np.array_equal(m[:, 0], np.arange(N)) and np.all(m[:, 2] == 11) and np.all(m[:, 6] == 2)
    dev = (m[:, 4] & 5) != 0; val = (m[:, 4] & 2) != 0
    assert (dev.sum(), val.sum()) == (11721, 5819) and np.all(dev ^ val)
    assert np.array_equal(inputs['e'], m[:, 3]) and np.all(inputs['accept'] == 1)
    assert inputs['p'].tobytes() == m[:, 10].tobytes() and inputs['alpha'].tobytes() == m[:, 12].tobytes()
    assert np.isfinite(inputs['p']).all() and np.all((inputs['p'] > 0) & (inputs['p'] <= 1))
    assert np.isfinite(inputs['alpha']).all() and np.all(inputs['alpha'] > 0)
    assert np.array_equal(occ[:, 0], np.arange(19962)) and np.all(occ[:, 1] < N) and np.all(occ[:, 12] == 1)
    assert np.all(occ[:, 3] == E) and np.all(occ[:, 8] == 11)
    for oi, mi in ((14, 1), (11, 3), (15, 12), (16, 10)): assert np.array_equal(occ[:, oi], m[occ[:, 1], mi])
    assert np.array_equal(np.bincount(occ[:, 1], minlength=N), m[:, 7])
    counts = np.bincount(m[dev, 3], minlength=E)
    assert counts[0] == 0 and np.all(counts[1:] > 0) and counts.max() == 308
    return m, occ, inputs, dev, counts


def saved(ctx, key, ids, plane=None):
    a = np.load(ctx.data(key), mmap_mode='r', allow_pickle=False)
    if key == 'signed_dots': assert a.shape == (N, H) and a.dtype == np.dtype('<i8')
    else: assert a.shape == (N, 5, D) and a.dtype == np.dtype('<f8')
    out = np.array(a[ids] if plane is None else a[ids, plane], copy=True)
    a._mmap.close(); return out


def source_row(ctx, e, j):
    p = next(v for v in ctx.b['parents'] if v['parent'] == e)['wi']
    with Path(ctx.b['payload']['path']).open('rb') as f:
        f.seek(p['offset'] + j * D); codes = f.read(D)
        f.seek(p['scale_offset'] + 4 * j); scale = f.read(4)
    assert len(codes) == D and len(scale) == 4
    return codes + scale


def domains(m, occ, counts):
    dev = (m[:, 4] & 5) != 0
    for split, mask in (('development', dev), ('consumed_validation', ~dev)):
        yield dict(kind='uid', split=split), np.flatnonzero(mask)
    for label, group in (('0', counts == 0), ('1..4', (counts > 0) & (counts <= 4)), ('5..15', (counts >= 5) & (counts <= 15)), ('>=16', counts >= 16)):
        for split, mask in (('development', dev), ('consumed_validation', ~dev)):
            yield dict(kind='rare', development_class=label, split=split), np.flatnonzero(mask & group[m[:, 3]])
    for e in range(E):
        for split, mask in (('development', dev), ('consumed_validation', ~dev)):
            yield dict(kind='parent', expert=e, split=split), np.flatnonzero(mask & (m[:, 3] == e))
    for book in range(192):
        for mode in range(2):
            yield dict(kind='book', book=book, role=book // 64, mode=mode), occ[(occ[:, 4] == book) & (occ[:, 6] == mode), 1]
    for role in range(3):
        for mode in range(2):
            yield dict(kind='role', role=role, mode=mode), occ[(occ[:, 7] == role) & (occ[:, 6] == mode), 1]
