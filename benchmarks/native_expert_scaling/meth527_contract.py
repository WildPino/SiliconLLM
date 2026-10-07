"""Shared source I/O, energy header and original-panel schema only."""
from pathlib import Path
import struct

ENERGY = struct.Struct('<8s6I')


def norm_bytes(ctx):
    v = ctx.b['organs']['norm']
    with Path(ctx.b['payload']['path']).open('rb') as f:
        f.seek(v['offset']); data = f.read(v['bytes'])
    assert len(data) == 3072
    return data


def wi_bytes(ctx, expert):
    v = next(p['wi'] for p in ctx.b['parents'] if p['parent'] == expert)
    with Path(ctx.b['payload']['path']).open('rb') as f:
        f.seek(v['offset']); data = f.read(v['bytes'])
    assert len(data) == 3072 * 768
    return data


def energy_header(power):
    return ENERGY.pack(b'M527ENR1', 16, 128, 3072, 2 * power, 128, 0)
