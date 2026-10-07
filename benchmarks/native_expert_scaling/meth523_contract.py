"""Wire and on-disk array specification; no fitting or transformation mathematics."""
from pathlib import Path
import numpy as np
from meth523_operations import wire

N, D, H, B, E = 17540, 768, 3072, 512, 128
UID = np.dtype([('m', '<u4', (13,)), ('hash', 'u1', (128,))])
INPUT = np.dtype([('e', '<u4'), ('accept', '<u4'), ('alpha', '<f4'), ('x', '<f4', (D,)), ('q', '<i2', (D,)), ('p', '<f4')])
ARMS = ('continuous_source', 'full_fold', 'continuous_hybrid', 'decoded_hybrid', 'physical_reference')
ARRAYS = {
    'signed_dots': ((N, H), '<i8'), 'fold64': ((E, D, D), '<f8'), 'fold_codes': ((E, D, D), '<i2'),
    'fold_scales': ((E, D), '<f4'), 'wi_codes': ((E, B, D), 'i1'), 'wo_codes': ((E, D, B), 'i1'),
    'wi_scales': ((E, B), '<f4'), 'wo_scales': ((E, D), '<f4'), 'hinges': ((E, B), '<u2'),
    'signs': ((E, H), 'u1'), 'scores': ((E, H), '<f8'), 'anchors': ((E,), '<u4'),
    'predictions': ((N, 5, D), '<f8'), 'metrics': ((N, 36), '<f8')}


def load(ctx):
    m = wire(np, ctx.data('uid'), b'M493U001', 180, 0, UID, N)['m']
    occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
    inputs = wire(np, ctx.data('inputs'), b'M499INP1', 4624, D, INPUT, N)
    f = wire(np, ctx.data('unweighted'), b'M499F001', 3072, D, np.dtype(('<f4', (D,))), N)
    y = wire(np, ctx.data('targets'), b'M493Y001', 3072, D, np.dtype(('<f4', (D,))), N)
    h, qh, ah = [np.load(ctx.data(k), mmap_mode='r', allow_pickle=False) for k in ('hidden', 'codes', 'scales')]
    assert h.shape == qh.shape == (N, H) and h.dtype == np.dtype('<f4') and qh.dtype == np.dtype('<i2')
    assert ah.shape == (N,) and ah.dtype == np.dtype('<f4')
    assert np.array_equal(m[:, 0], np.arange(N)) and np.all(m[:, 2] == 11) and np.all(m[:, 6] == 2)
    dev = (m[:, 4] & 5) != 0; val = (m[:, 4] & 2) != 0
    assert (dev.sum(), val.sum()) == (11721, 5819) and np.all(dev ^ val)
    assert np.array_equal(inputs['e'], m[:, 3]) and np.all(inputs['accept'] == 1)
    assert inputs['p'].tobytes() == m[:, 10].tobytes() and inputs['alpha'].tobytes() == m[:, 12].tobytes()
    assert np.array_equal(occ[:, 0], np.arange(19962)) and np.all(occ[:, 1] < N) and np.all(occ[:, 12] == 1)
    assert np.all(occ[:, 3] == E) and np.all(occ[:, 8] == 11)
    for oi, mi in ((14, 1), (11, 3), (15, 12), (16, 10)): assert np.array_equal(occ[:, oi], m[occ[:, 1], mi])
    assert np.array_equal(np.bincount(occ[:, 1], minlength=N), m[:, 7])
    counts = np.bincount(m[dev, 3], minlength=E); assert counts[0] == 0 and np.all(counts[1:] > 0) and counts.max() == 308
    return m, occ, inputs, f, y, h, qh, ah, dev, counts


def weights(ctx, e):
    ans = []
    entry = next(v for v in ctx.b['parents'] if v['parent'] == e)
    with Path(ctx.b['payload']['path']).open('rb') as file:
        for organ, shape in (('wi', (H, D)), ('wo', (D, H))):
            v = entry[organ]; file.seek(v['offset']); raw = file.read(v['bytes']); assert len(raw) == v['bytes']
            file.seek(v['scale_offset']); scales = file.read(v['scale_bytes']); assert len(scales) == v['scale_bytes']
            w = np.frombuffer(raw, dtype='i1').reshape(shape); s = np.frombuffer(scales, dtype='<f4')
            assert np.all(s > 0) and np.isfinite(s).all(); ans.extend((w, s))
    return ans


def create(folder):
    for name, (shape, dtype) in ARRAYS.items():
        p = folder / (name + '.npy'); assert not p.exists()
        with p.open('xb') as file:
            np.lib.format.write_array_header_1_0(file, dict(shape=shape, fortran_order=False, descr=np.dtype(dtype).str))
            file.truncate(file.tell() + int(np.prod(shape)) * np.dtype(dtype).itemsize)


def array(folder, name, mode='r'):
    a = np.load(folder / (name + '.npy'), mmap_mode=mode, allow_pickle=False)
    shape, dtype = ARRAYS[name]; assert a.shape == shape and a.dtype == np.dtype(dtype)
    return a


def put(folder, name, ids, value):
    a = array(folder, name, 'r+'); a[ids] = value; a.flush(); a._mmap.close()


def get(folder, name, ids):
    a = array(folder, name); value = np.array(a[ids], copy=True); a._mmap.close(); return value


def domain_ids(m, occ, counts):
    dev = (m[:, 4] & 5) != 0
    for label, mask in (('development', dev), ('consumed_validation', ~dev)):
        yield dict(kind='uid', split=label), np.flatnonzero(mask)
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
