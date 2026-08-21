#!/usr/bin/env python3
"""D-probe calibration corpus builder (donor-adaptation v2 density programme).

Builds a globally-shuffled mixed-general-text pool from local sources, splits it into
a CALIBRATION half and a disjoint HELD-OUT half, and pins both by SHA-256.

Why global shuffle: the project measured that blocked (file-order) sampling alone cost
+0.0339 BPB = 6.8 sigma on rung-1.  Order is a confound; it is removed here by shuffling
at 8 KiB chunk granularity across every source before any split is taken.

Usage:  python build_calib.py build
        python build_calib.py verify
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "corpus")

CHUNK = 8192          # bytes per shuffle unit
TARGET_TOTAL = 320 * 1024 * 1024
SEED = 20260821

# fraction of the pool each source contributes (mixed *general* text)
MIX = {
    "pg19": 0.40,       # long-form English prose / books
    "wikitext": 0.10,   # encyclopedic prose
    "markdown": 0.25,   # technical documentation (mdn, rust book, k8s, typescript)
    "python": 0.25,     # source code
}


def _read_pg19(budget: int) -> list[bytes]:
    import pyarrow.parquet as pq
    d = os.path.join(REPO, "data", "external", "pg19", "data")
    files = sorted(f for f in os.listdir(d) if f.startswith("train-") and f.endswith(".parquet"))
    out, got = [], 0
    for fn in files:
        pf = pq.ParquetFile(os.path.join(d, fn))
        for batch in pf.iter_batches(batch_size=8, columns=["text"]):
            for t in batch.column("text").to_pylist():
                if not t:
                    continue
                b = t.encode("utf-8", "ignore")
                out.append(b)
                got += len(b)
                if got >= budget:
                    return out
        if got >= budget:
            break
    return out


def _read_wikitext(budget: int) -> list[bytes]:
    import pyarrow.parquet as pq
    import glob
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/train-*.parquet"))
    if not hits:
        return []
    tbl = pq.read_table(hits[0], columns=["text"])
    buf, out, got = [], [], 0
    for t in tbl.column("text").to_pylist():
        buf.append(t)
        if sum(len(x) for x in buf) > 200_000:
            b = "".join(buf).encode("utf-8", "ignore")
            out.append(b); got += len(b); buf = []
            if got >= budget:
                return out
    if buf:
        out.append("".join(buf).encode("utf-8", "ignore"))
    return out


def _read_tree(root: str, exts: tuple[str, ...], budget: int) -> list[bytes]:
    out, got = [], 0
    paths = []
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(exts):
                paths.append(os.path.join(dp, fn))
    paths.sort()
    rng = random.Random(SEED)
    rng.shuffle(paths)                      # document order is itself randomised
    for p in paths:
        try:
            b = open(p, "rb").read()
        except OSError:
            continue
        if not b:
            continue
        out.append(b); got += len(b)
        if got >= budget:
            break
    return out


def build() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    chunks: list[tuple[str, bytes]] = []
    prov: dict[str, int] = {}

    srcs = {
        "pg19": lambda b: _read_pg19(b),
        "wikitext": lambda b: _read_wikitext(b),
        "markdown": lambda b: _read_tree(
            os.path.join(REPO, "data", "external", "markdown_corpus"), (".md", ".markdown", ".mdx"), b),
        "python": lambda b: _read_tree(
            os.path.join(REPO, "data", "external", "the_stack_python"), (".py",), b),
    }

    for name, frac in MIX.items():
        budget = int(TARGET_TOTAL * frac)
        docs = srcs[name](budget)
        got = 0
        for d in docs:
            for i in range(0, len(d), CHUNK):
                c = d[i:i + CHUNK]
                if len(c) < CHUNK // 2:
                    continue
                chunks.append((name, c))
                got += len(c)
                if got >= budget:
                    break
            if got >= budget:
                break
        prov[name] = got
        print(f"  {name:9s} {got/1e6:8.1f} MB from {len(docs)} docs", flush=True)

    # ---- the global shuffle: every chunk from every source, one permutation ----
    rng.shuffle(chunks)

    n = len(chunks)
    half = n // 2
    parts = {"calib": chunks[:half], "heldout": chunks[half:]}
    manifest = {
        "seed": SEED, "chunk_bytes": CHUNK, "mix": MIX,
        "provenance_bytes": prov, "n_chunks": n, "parts": {},
    }
    for pname, plist in parts.items():
        path = os.path.join(OUT, f"{pname}.txt")
        h = hashlib.sha256()
        with open(path, "wb") as f:
            for _src, c in plist:
                f.write(c); h.update(c)
        counts: dict[str, int] = {}
        for s, c in plist:
            counts[s] = counts.get(s, 0) + len(c)
        manifest["parts"][pname] = {
            "path": os.path.relpath(path, REPO).replace("\\", "/"),
            "bytes": os.path.getsize(path),
            "sha256": h.hexdigest(),
            "n_chunks": len(plist),
            "source_bytes": counts,
        }
        print(f"  -> {pname}: {os.path.getsize(path)/1e6:.1f} MB  sha256={h.hexdigest()[:16]}", flush=True)

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest written")


def verify() -> None:
    m = json.load(open(os.path.join(OUT, "manifest.json")))
    ok = True
    for pname, meta in m["parts"].items():
        p = os.path.join(REPO, meta["path"])
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
        good = h.hexdigest() == meta["sha256"]
        ok &= good
        print(f"{pname}: {'OK' if good else 'MISMATCH'} {h.hexdigest()}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"build": build, "verify": verify}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
