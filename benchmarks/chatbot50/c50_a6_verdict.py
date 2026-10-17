"""C50-A6 FIT selection: applies the registered rule to the runner's outputs (it chooses nothing itself).

Per arm against R (P0, KV f16, same runtime): paired dBPB over the 72 non-imatrix FIT documents of the A4
panel, document bootstrap (10,000 draws, seed 20261016), and bytes per token at context 2048 =
per-token weight bytes (GGUF check) + KV (20 layers x 2 x 512 values x 2048 positions; f16 2 B, q8_0 34/32 B).
Addendum 2: cost = desk ms per token at context 2048, with IQ3_S routed-expert bytes at the A1-measured
12.4479 GB/s (A1 consequence rule: a failing class is priced at its measured bandwidth), every other byte at
32 GB/s, plus T_FIXED 2 ms; for arms without IQ3_S this orders exactly as bytes do.
FIT candidate = the arm with the lowest desk ms per token whose dBPB upper CI95 <= +0.015 (ROADMAP §5 A6);
none = A6 stops at P0/P1 as the roadmap says.  DEV confirmation and the task panels are a later protocol.

Usage:
    python -I benchmarks/chatbot50/c50_a6_verdict.py --run benchmarks/chatbot50/results/c50_a6_run.json \
        --work E:/.../a6 --out benchmarks/chatbot50/results/c50_a6_verdict.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_compare import byte_lengths  # noqa: E402

SEED, DRAWS, BOUND = 20261016, 10_000, 0.015
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])
CAMP = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
KV_VALUES = 20 * 2 * 512 * 2048
KV_BYTES = {"f16": KV_VALUES * 2, "q8_0": KV_VALUES * 34 / 32}
BW_PLAN, BW_IQ3S, T_FIXED_MS = 32e9, 12.4479e9, 2.0   # BW_IQ3S: results/c50_a1_verdict_2.json, S3_IQ3_S median
N_EXP, K_EXP = 256, 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    work = Path(args.work)
    a2 = json.loads((Path(__file__).resolve().parent / "results" / "c50_a2_gguf_check_q4_k_m.json").read_text(encoding="utf-8"))
    p0_w = next(a["per_token_weight_bytes"] for a in a2["artifacts"] if a["sha256"].startswith("44de7315"))
    meta = [json.loads(x) for x in (CAMP / "a4/inputs/docs_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    blen_tab = byte_lengths(CAMP / "sources/Ling-mini-2.0_a810f64/tokenizer.json")
    if not run["arms"].get("R", {}).get("ok"):
        raise SystemExit("reference arm R missing")
    r = np.fromfile(work / "R" / "docs.pos.bin", dtype=REC)
    docs = np.array([m["index"] for m in meta])
    di = np.searchsorted(docs, r["doc"])
    den = np.bincount(di, weights=blen_tab[r["target"]].astype(np.float64), minlength=len(docs))
    bits_r = -r["logp_target"].astype(np.float64) / math.log(2)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(docs), size=(DRAWS, len(docs)))
    out = {"cell": "C50-A6 FIT selection", "bound_upper_ci95": BOUND, "kv_mb_at_2048": {k: v / 1e6 for k, v in KV_BYTES.items()},
           "bpb_R": float(bits_r.sum() / den.sum()), "arms": {}}
    for a, rec in run["arms"].items():
        if not rec.get("ok"):
            continue
        o = np.fromfile(work / a / "docs.pos.bin", dtype=REC)
        if len(o) != len(r) or not (np.array_equal(o["doc"], r["doc"]) and np.array_equal(o["target"], r["target"])):
            raise SystemExit(f"{a}: positions differ from R")
        num = np.bincount(di, weights=-o["logp_target"].astype(np.float64) / math.log(2) - bits_r, minlength=len(docs))
        st = num[idx].sum(axis=1) / den[idx].sum(axis=1)
        lo, hi = (float(x) for x in np.percentile(st, [2.5, 97.5]))
        w = p0_w if rec["build"] is None else run["builds"][rec["build"]]["per_token_weight_bytes"]
        iq3 = 0.0 if rec["build"] is None else run["builds"][rec["build"]]["bytes_by_type"].get("IQ3_S", 0) * K_EXP / N_EXP
        total = w + KV_BYTES[rec["kv"]]
        ms = 1e3 * ((total - iq3) / BW_PLAN + iq3 / BW_IQ3S) + T_FIXED_MS
        out["arms"][a] = {"dbpb": float(num.sum() / den.sum()), "dbpb_ci95": [lo, hi], "qualifies": hi <= BOUND,
                          "top1_agreement_with_R": float((o["top1"] == r["top1"]).mean()),
                          "weight_mb_per_token": w / 1e6, "kv": rec["kv"],
                          "mb_per_token_at_2048": total / 1e6, "iq3s_mb_per_token": iq3 / 1e6,
                          "desk_ms_per_token_at_2048": ms}
    q = sorted((a for a, v in out["arms"].items() if v["qualifies"]), key=lambda a: out["arms"][a]["desk_ms_per_token_at_2048"])
    out["fit_candidate"] = q[0] if q else None
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for a, v in out["arms"].items():
        print(f"{a:4s} dBPB {v['dbpb']:+.5f} CI ({v['dbpb_ci95'][0]:+.5f}, {v['dbpb_ci95'][1]:+.5f}) "
              f"{v['mb_per_token_at_2048']:.1f} MB/token {v['desk_ms_per_token_at_2048']:.2f} desk ms "
              f"{'qualifies' if v['qualifies'] else ''}")
    print("FIT candidate:", out["fit_candidate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
