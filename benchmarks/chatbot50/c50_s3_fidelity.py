"""C50-S3 fidelity verdict (protocol, "Fidelity rule"): the teacher's production configuration on S2's MMLU panel
against F0's CPU scores of the same panel from S2 (A4's scorer, ctx 4096, ubatch 512), paired per sequence.

PASS iff the CI95 of the mean per-sequence GPU - CPU dBPB lies inside [-TOL, +TOL] and the top-1 agreement is at
least TOP1_MIN.  It reads only finished shards and refuses a partial run.  The rule is frozen in the protocol.

Usage:
    python -I benchmarks/chatbot50/c50_s3_fidelity.py --json benchmarks/chatbot50/results/c50_s3_fidelity_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c50_compare import byte_lengths  # noqa: E402
import c50_a4_checks as A4  # noqa: E402
import c50_s3_teach as T  # noqa: E402

TOL, TOP1_MIN, SEED, DRAWS = 0.002, 0.95, 20261016, 10_000
K = 32
HEAD = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("logp_target", "<f4"), ("lse", "<f8")])
REC = np.dtype([("h", HEAD), ("e", [("id", "<i4"), ("lp", "<f4")], K)])
P0 = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"), ("logp_target", "<f4"),
               ("logp_top1", "<f4"), ("lse", "<f8")])
CPU_REF = T.C / "s2/f0/mmlu.pos.bin"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    a = ap.parse_args()
    cpu = np.fromfile(CPU_REF, dtype=P0)
    rows = [json.loads(x) for x in (T.FID_OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    g = np.concatenate([np.fromfile(T.FID_OUT / f"shard_{r['shard']:06d}.bin", dtype=REC)
                        for r in sorted(rows, key=lambda r: r["shard"])])
    if len(g) != len(cpu):
        raise SystemExit(f"partial run: {len(g)} positions, the panel has {len(cpu)}")
    for f in ("doc", "pos", "target"):
        if not np.array_equal(g["h"][f], cpu[f]):
            raise SystemExit(f"records differ in {f}")
    blen = byte_lengths(Path(A4.TOKENIZER)).astype(np.float64)[cpu["target"]]
    dl = g["h"]["logp_target"].astype(np.float64) - cpu["logp_target"].astype(np.float64)
    docs, inv = np.unique(cpu["doc"], return_inverse=True)
    x = -np.bincount(inv, weights=dl) / np.log(2) / np.bincount(inv, weights=blen)
    rng = np.random.default_rng(SEED)
    boot = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(DRAWS)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    top1 = float(np.mean(g["e"]["id"][:, 0] == cpu["top1"]))
    cpu_bpb = float((-np.bincount(inv, weights=cpu["logp_target"].astype(np.float64)) / np.log(2)
                     / np.bincount(inv, weights=blen)).mean())
    ok = bool(-TOL <= lo and hi <= TOL and top1 >= TOP1_MIN)
    out = {"cell": "C50-S3 fidelity v1", "panel": str(T.FID_IDS), "cpu_reference": str(CPU_REF), "sequences": int(len(docs)),
           "positions": int(len(cpu)), "tol_bpb": TOL, "top1_min": TOP1_MIN,
           "config": json.loads((T.FID_OUT / "launches.jsonl").read_text(encoding="utf-8").splitlines()[-1]).get("config"),
           "dbpb_gpu_minus_cpu": float(x.mean()), "dbpb_ci95": [float(lo), float(hi)], "cpu_bpb_mean": cpu_bpb,
           "mean_abs_dlogp": float(np.abs(dl).mean()), "max_abs_dlogp": float(np.abs(dl).max()), "top1_agreement": top1,
           "wall_s": float(sum(r["wall_s"] for r in rows)), "verdict": "PASS" if ok else "FAIL"}
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
