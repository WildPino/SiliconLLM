#!/usr/bin/env python3
# Phase 63.V - n-gram drafter ASSET builder (offline). magic: "NG01" 0x3130474E
#   Builds a backoff n-gram (orders 1..N) counted on TRAIN ONLY (TS ids.u16 first 90%, same BPE-1024), argmax
#   next-token per context, and serializes a binary the C engine loads. The engine drafter, at each position,
#   packs the last (o-1) tokens -> uint64 key, binary-searches order o (N down to 2), backs off, unigram floor.
#
#   Binary format (little-endian), documented (V-G4 counts these bytes in the resident budget):
#     u32 magic=0x3130474E | u32 N | u32 unigram_argmax
#     for o in 2..N:  u32 count_o ; then count_o records sorted by key: { u64 key ; u16 next ; u16 pad }
#   key(o) = sum_{i=0..o-2} ctx[i] * 1024^i   (ctx = last o-1 tokens, oldest first)  [1024 = vocab]
#
# Run: .venv/Scripts/python.exe benchmarks/phase63/ngram_asset.py --N 5
import os, sys, struct, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "phase62"))
from e5_0_acceptance import build_ngrams          # reuse the exact train-only backoff builder
ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); OUT = os.path.join(ROOT, "results", "phase63"); os.makedirs(OUT, exist_ok=True)
V = 1024; MAGIC = 0x3130474E

def key_of(ctx):                                    # ctx = tuple(last o-1 tokens, oldest first)
    k = 0
    for i, t in enumerate(ctx): k += int(t) * (V ** i)
    return k

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--train-cap", type=int, default=0, help="0 = all TS train")
    a = ap.parse_args()
    ids = np.fromfile(os.path.join(ROOT, "results", "phase55", "ids.u16"), dtype=np.uint16).astype(np.int64)
    ntr = int(len(ids) * 0.9); train = ids[:ntr]
    if a.train_cap: train = train[:a.train_cap]
    orders = list(range(2, a.N + 1))
    tab, ufb = build_ngrams(train, orders)          # tab[o] = {ctx-tuple: argmax_next}; ufb = unigram argmax
    path = os.path.join(OUT, f"ngram_ts_N{a.N}.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<III", MAGIC, a.N, ufb))
        for o in orders:
            items = sorted((key_of(c), nx) for c, nx in tab[o].items())
            f.write(struct.pack("<I", len(items)))
            for k, nx in items: f.write(struct.pack("<QHH", k, nx, 0))
    sz = os.path.getsize(path)
    print(f"n-gram asset N={a.N} on {ntr} TS-train tokens -> {os.path.relpath(path, ROOT)}")
    print(f"  orders {orders} | contexts/order: " + " ".join(f"o{o}={len(tab[o])}" for o in orders))
    print(f"  BYTES = {sz} ({sz/1024:.1f} KB) [V-G4: resident-budget asset]")
    print("STOP. n-gram asset written. No commit.")

if __name__ == "__main__":
    main()
