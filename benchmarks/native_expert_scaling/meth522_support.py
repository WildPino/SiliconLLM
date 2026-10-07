"""Support wire definitions, immutable ownership and shared reporting contract."""
import numpy as np
from meth522_operations import wire

UID = np.dtype([('m', '<u4', (13,)), ('hash', 'u1', (128,))])
METRIC = np.dtype([('selected', '<u4'), ('oracle', '<u4'), ('support', '<u4'), ('omitted', '<u4', (4,)),
                   ('max_kept', 'u1', (4,)), ('source_scale_error', '<f8', (4,)), ('actual_error', '<f8', (4,)),
                   ('reference_energy', '<f8'), ('static_error', '<f8')])
TRACE = np.dtype([('UID', '<u4'), ('expert', '<u2'), ('size', '<u2'), ('accepted', 'u1'), ('reserved', 'u1'),
                  ('checked', '<u2'), ('blocker_UID', '<u4'), ('blocker_union', '<u2'), ('padding', '<u2')])


def load(ctx):
    uid = wire(np, ctx.data('uid'), b'M493U001', 180, 0, UID, 17540); m = uid['m']
    occ = wire(np, ctx.data('occurrences'), b'M493O001', 68, 0, np.dtype(('<u4', (17,))), 19962)
    q = np.load(ctx.data('codes'), mmap_mode='r', allow_pickle=False)
    old = np.load(ctx.data('metrics'), mmap_mode='r', allow_pickle=False)
    assert q.shape == (17540, 3072) and q.dtype == np.dtype('<i2') and old.shape == (17540,) and old.dtype == METRIC
    dev = (m[:, 4] & 5) != 0; val = (m[:, 4] & 2) != 0
    assert np.array_equal(m[:, 0], np.arange(17540)) and np.all(m[:, 2] == 11) and np.all(m[:, 3] < 128) and np.all(m[:, 6] == 2)
    assert (int(dev.sum()), int(val.sum())) == (11721, 5819) and np.all(dev ^ val)
    counts = np.bincount(m[dev, 3], minlength=128); assert counts.max() == 308 and counts[0] == 0 and np.all(counts[1:] > 0)
    assert np.array_equal(occ[:, 0], np.arange(19962)) and np.all(occ[:, 1] < 17540)
    assert np.all(occ[:, 3] == 128) and np.all(occ[:, 8] == 11) and np.all(occ[:, 12] == 1)
    for oi, mi in ((14, 1), (11, 3), (15, 12), (16, 10)): assert np.array_equal(occ[:, oi], m[occ[:, 1], mi])
    assert np.array_equal(np.bincount(occ[:, 1], minlength=17540), m[:, 7])
    card = np.empty(17540, '<u2')
    for start in range(0, 17540, 193):
        part = q[start:start + 193]; assert np.all((part >= 0) & (part <= 32767))
        card[start:start + len(part)] = np.count_nonzero(part, axis=1); ctx.guard()
    assert np.array_equal(card, old['support'])
    return m, occ, q, card, counts


def reports(m, occ, card, counts):
    def metric(ids):
        c = card[ids]; n = len(ids)
        return {'count': n, 'sum_source_support': int(sum(map(int, c))), 'max_source_support': int(c.max()) if n else 0,
                'source_support_above512': int(np.sum(c > 512))}
    dev = (m[:, 4] & 5) != 0
    roles = [{'split': name, **metric(np.flatnonzero(mask))} for name, mask in (('development', dev), ('consumed_validation', ~dev))]
    rare = [{'development_class': name, 'split': role, **metric(np.flatnonzero(mask & group[m[:, 3]]))}
            for name, group in (('0', counts == 0), ('1..4', (counts >= 1) & (counts <= 4)),
                                ('5..15', (counts >= 5) & (counts <= 15)), ('>=16', counts >= 16))
            for role, mask in (('development', dev), ('consumed_validation', ~dev))]
    cells = [{'expert': e, 'split': role, **metric(np.flatnonzero(mask & (m[:, 3] == e)))}
             for e in range(128) for role, mask in (('development', dev), ('consumed_validation', ~dev))]
    views = [{'book': book, 'role': book // 64, 'mode': mode, **metric(occ[(occ[:, 4] == book) & (occ[:, 6] == mode), 1])}
             for book in range(192) for mode in range(2)]
    rm = [{'role': role, 'mode': mode, **metric(occ[(occ[:, 7] == role) & (occ[:, 6] == mode), 1])}
          for role in range(3) for mode in range(2)]
    assert sum(v['count'] for v in views) == sum(v['count'] for v in rm) == 19962
    return {'uid_roles': roles, 'rare': rare, 'cells': cells, 'book_mode': views, 'role_mode': rm}
