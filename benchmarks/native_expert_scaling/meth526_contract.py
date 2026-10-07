"""526 shared I/O only: original organs, retained Q, fixed signed-I512 matrices."""
from pathlib import Path
import struct

MAGIC = b'M526INT1'
HEADER = struct.Struct('<8s6I')
WIDTH = 64
KINDS = {'Q_numerators': 1, 'C_numerators': 2, 'Q_gram_defect': 3,
         'CQ_numerators': 4, 'C_rowspace_residual': 5}


def organ_bytes(ctx, name):
    v = ctx.b['organs'][name]
    with Path(ctx.b['payload']['path']).open('rb') as f:
        f.seek(v['offset']); data = f.read(v['bytes'])
    assert len(data) == v['bytes']
    return data


def write_matrix(path, name, array, power):
    rows, cols = array.shape
    with Path(path).open('xb') as f:
        f.write(HEADER.pack(MAGIC, WIDTH, rows, cols, power, 512, KINDS[name]))
        for row in array:
            encoded = b''.join(int(v).to_bytes(WIDTH, 'little', signed=True) for v in row)
            f.write(encoded)


def read_matrix(path, name, rows, cols, power):
    p = Path(path); assert p.stat().st_size == HEADER.size + rows * cols * WIDTH
    with p.open('rb') as f:
        assert f.read(HEADER.size) == HEADER.pack(MAGIC, WIDTH, rows, cols, power, 512, KINDS[name])
        result = []
        for i in range(rows):
            data = f.read(cols * WIDTH); assert len(data) == cols * WIDTH
            result.append([int.from_bytes(data[j * WIDTH:(j + 1) * WIDTH], 'little', signed=True) for j in range(cols)])
        assert not f.read(1)
    return result
