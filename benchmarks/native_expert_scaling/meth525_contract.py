"""Shared wire/I/O only. Original F32 router sees x; source WI sees Q16(x)."""
from pathlib import Path
import numpy as np
from meth525_operations import wire
from meth524_contract import load

CANONICAL = np.dtype([('meta', '<u4', (12,)), ('hashes', 'u1', (64,)), ('value', '<f8', (3,)),
                      ('x', '<f4', (768,)), ('scores', '<f4', (128,))])


def read_organ(ctx, key):
    t = ctx.b['organs'][key]
    with Path(ctx.b['payload']['path']).open('rb') as f:
        f.seek(t['offset']); raw = f.read(t['bytes'])
    assert len(raw) == t['bytes']
    return np.frombuffer(raw, dtype='<f4').reshape(t['shape']).copy()


def parent(ctx, e):
    t = next(v for v in ctx.b['parents'] if v['parent'] == e)['wi']
    with Path(ctx.b['payload']['path']).open('rb') as f:
        f.seek(t['offset']); raw = f.read(t['bytes']); f.seek(t['scale_offset']); s = f.read(t['scale_bytes'])
    assert len(raw) == 3072 * 768 and len(s) == 3072 * 4
    w, scale = np.frombuffer(raw, dtype='i1').reshape(3072, 768), np.frombuffer(s, dtype='<f4')
    assert np.all(scale > 0) and np.isfinite(scale).all() and np.all(w >= -127)
    return w, scale


def anchors(ctx, m, inputs, dev):
    anchor = np.load(ctx.data('anchors'), allow_pickle=False); hinges = np.load(ctx.data('hinges'), allow_pickle=False)
    signs = np.load(ctx.data('signs'), allow_pickle=False)
    assert anchor.shape == (128,) and anchor.dtype == np.dtype('<u4') and int(anchor[0]) == 0xffffffff
    assert hinges.shape == (128, 512) and hinges.dtype == np.dtype('<u2') and signs.shape == (128, 3072) and signs.dtype == np.dtype('u1')
    c = wire(np, ctx.data('canonical_scores'), b'M479UNI1', 3720, 768, CANONICAL, 238872)
    cached = np.empty((128, 128), '<f4'); cached[0] = 0
    for e in range(1, 128):
        k = int(anchor[e]); assert k < 17540 and dev[k] and m[k, 3] == e
        row = c[int(m[k, 1])]
        assert row['meta'][7] == e and row['x'].tobytes() == inputs['x'][k].tobytes()
        assert np.float32(row['value'][1]).tobytes() == inputs['p'][k].tobytes() and int(np.argmax(row['scores'])) == e
        assert np.isfinite(row['scores']).all(); cached[e] = row['scores']
        assert len(set(int(v) for v in hinges[e])) == 512 and np.all(hinges[e] < 3072) and np.all(signs[e] <= 1)
    c._mmap.close(); return anchor, hinges, signs, cached
