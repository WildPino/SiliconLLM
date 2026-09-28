"""Frozen shared structural slot plus nine hash-selected content slots."""

import json
from pathlib import Path

import numpy as np

import meth142_token_hash_route_screen as M142


ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth149_recurrent_context_table.json"
TABLE_SHA = "93b6ca77ece874889e83ae368b9f63d78bef8e90a9fbda37cb9e1a871cb0503e"


def load_table():
    import hashlib
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == TABLE_SHA
    record = json.loads(TABLE.read_text(encoding="utf-8"))
    assert record["threshold_sequences"] == 32
    tuples = {(row["token"], row["previous"], row["position"])
              for row in record["tuples"]}
    assert len(tuples) == record["recurrent_tuple_count"] == 75
    return tuples


def shared_mask(tokens, previous, positions, table):
    assert tokens.shape == previous.shape == positions.shape
    return np.fromiter(((int(t), int(u), int(p)) in table
                        for t, u, p in zip(tokens, previous, positions)),
                       dtype=np.bool_, count=tokens.size)


def route(tokens, previous, positions, children, layer_id, table):
    assert tokens.ndim == previous.ndim == positions.ndim == 1
    assert children.shape == (tokens.size, 4)
    assert np.all((children >= 0) & (children < 1280))
    shared = shared_mask(tokens, previous, positions, table)
    c1, c2, c3, c4, c5 = M142.CONSTANTS
    t = tokens.astype(np.uint64)[:, None]
    u = previous.astype(np.uint64)[:, None]
    p = positions.astype(np.uint64)[:, None]
    c = children.astype(np.uint64)
    with np.errstate(over="ignore"):
        z = (M142.SEED ^ (t * c1) ^ (u * c2) ^ (p * c3) ^
             (c * c4) ^ (np.uint64(layer_id) * c5))
        z = z + c1
        z = (z ^ (z >> np.uint64(30))) * c2
        z = (z ^ (z >> np.uint64(27))) * c3
        z = z ^ (z >> np.uint64(31))
        local = np.where(shared[:, None], np.uint64(0),
                         np.uint64(1) + (z % np.uint64(9)))
        grandchildren = c * np.uint64(10) + local
    assert np.array_equal(grandchildren // 10, c)
    assert np.all((grandchildren % 10 == 0) == shared[:, None])
    return grandchildren.astype(np.int64), shared


def golden(table):
    a, shared_a = route(np.array([123]), np.array([45]), np.array([67]),
                        np.array([[89, 89, 89, 89]]), 3, table)
    b, shared_b = route(np.array([11]), np.array([16948]), np.array([7]),
                        np.array([[89, 89, 89, 89]]), 3, table)
    assert int(a[0, 0]) == 899 and not bool(shared_a[0])
    assert int(b[0, 0]) == 890 and bool(shared_b[0])
    return {"content": 899, "structural": 890}
