"""Audit of R (27 October 2026 review): how much of the recovered document gap is knowledge, and how much is a flatter
distribution? Existing evaluation records only; nothing is run.

R is S4's metric: the share of S2's T0 -> F0 bits gap recovered on the 36 S4 documents. This script first recomputes R
with the verdicts' own functions and stops unless it matches the recorded verdicts (`results/c50_s4_verdict.json`,
`results/c50_t2_n_verdict.json`). Then, per arm and per document category, it reports:
- R_acc: the same share on top-1 accuracy, (acc_X - acc_T0) / (acc_F0 - acc_T0), which a temperature cannot move;
- the mean probability each model gives its own top-1 (confidence);
- where the saved bits sit, by whether T0's and the arm's top-1 are right at the position.

    python -I benchmarks/chatbot50/c50_audit_r_hedging.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_verdict as V  # noqa: E402

C = V.C
RESULTS = Path(__file__).resolve().parent / "results"
T0_PT = C / "s4/verdict/evals/t0"                  # MVE-5's PyTorch T0, the base of R in S4 and T2-N
ARMS = {**{k: C / "s4/verdict/evals" / k for k in ("c48", "c496", "a", "b", "b_shuffle", "b_mask", "a_shuffle")},
        **{k: C / "t2/n/evals" / k for k in ("v0", "e2", "n", "n_shuffle")}}
RECORDED = ("c50_s4_verdict.json", "c50_t2_n_verdict.json")
LN2 = math.log(2)


def positions(arm: Path, docs: list[int]) -> dict[int, np.ndarray]:
    r = np.fromfile(arm / "docs.pos.bin", dtype=V.A4.REC)
    out = {}
    for d in docs:
        s = r[r["doc"] == d]
        out[d] = s[np.argsort(s["pos"])]
    return out


def summary(x: dict, t0: dict, f0: dict, docs: list[int]) -> dict:
    """R_acc, confidence and the saving by flip class, pooled over `docs`."""
    cat = lambda f: np.concatenate([f(d) for d in docs])  # noqa: E731
    right = {k: cat(lambda d, m=m: m[d]["top1"] == m[d]["target"]) for k, m in (("t0", t0), ("x", x), ("f0", f0))}
    acc = {k: float(v.mean()) for k, v in right.items()}
    saved = cat(lambda d: (x[d]["logp_target"].astype(np.float64) - t0[d]["logp_target"].astype(np.float64)) / LN2)
    classes = {"both_right": right["t0"] & right["x"], "t0_wrong_x_right": ~right["t0"] & right["x"],
               "t0_right_x_wrong": right["t0"] & ~right["x"], "both_wrong": ~right["t0"] & ~right["x"]}
    tot = float(saved.sum())
    return {"r_acc": round((acc["x"] - acc["t0"]) / (acc["f0"] - acc["t0"]), 4),
            "acc": {k: round(v, 4) for k, v in acc.items()},
            "confidence_top1": round(float(cat(lambda d: np.exp(x[d]["logp_top1"].astype(np.float64))).mean()), 4),
            "saved_bits": round(tot),
            "saving_by_class": {k: {"positions": round(float(m.mean()), 4),
                                    "share_of_saving": round(float(saved[m].sum()) / tot, 4) if tot else None}
                                for k, m in classes.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
    docs = split["s4"]["docs"]
    blen = V.byte_lengths(Path(V.A4.TOKENIZER))
    meta = {m["index"]: m["category"] for m in map(json.loads, (V.A4_IN / "docs_meta.jsonl").open(encoding="utf-8"))}

    # 1. R as the verdicts compute it, checked against the recorded verdicts before anything else is read
    gap = V.doc_bits(V.LLAMA_T0["a4"], blen, docs)[0] - V.doc_bits(V.LLAMA_F0, blen, docs)[0]
    t0_bits = V.doc_bits(T0_PT, blen, docs)[0]
    bits = {k: V.doc_bits(p, blen, docs)[0] for k, p in ARMS.items()}
    R = {k: V.recovered(t0_bits, b, gap) for k, b in bits.items()}
    recorded = {}
    for f in RECORDED:
        recorded.update(json.loads((RESULTS / f).read_text(encoding="utf-8"))["R"])
    mismatch = {k: (R[k], recorded[k]) for k in R if abs(R[k] - recorded[k]) > 1e-9}
    if mismatch:
        print("STOP: R does not reproduce the recorded verdicts:", mismatch)
        return 1

    # 2. the decomposition, from the per-position records
    t0, f0 = positions(T0_PT, docs), positions(V.LLAMA_F0, docs)
    for d in docs:
        assert np.array_equal(t0[d]["target"], f0[d]["target"]), d
    t0_llama = positions(V.LLAMA_T0["a4"], docs)
    conf = lambda m: round(float(np.concatenate([np.exp(m[d]["logp_top1"].astype(np.float64)) for d in docs]).mean()), 4)  # noqa: E731
    out = {"cell": "C50 review 2026-10-27: R audit", "docs": len(docs), "R_reproduced": True,
           "confidence_top1": {"t0_pytorch": conf(t0), "t0_llama": conf(t0_llama), "f0": conf(f0)}, "arms": {}}
    cats = sorted(set(meta[d] for d in docs))
    for k, p in ARMS.items():
        x = positions(p, docs)
        for d in docs:
            assert np.array_equal(x[d]["target"], t0[d]["target"]), (k, d)
        rec = {"R": round(R[k], 4), **summary(x, t0, f0, docs), "by_category": {}}
        for c in cats:
            dc = [d for d in docs if meta[d] == c]
            i = np.array([docs.index(d) for d in dc])
            rec["by_category"][c] = {"docs": len(dc), "R": round(V.recovered(t0_bits, bits[k], gap, i), 4),
                                     "r_acc": summary(x, t0, f0, dc)["r_acc"]}
        out["arms"][k] = rec

    print(f"R reproduces the recorded verdicts on {len(R)} arms. Confidence in the own top-1: "
          f"T0 {out['confidence_top1']['t0_pytorch']} (llama.cpp {out['confidence_top1']['t0_llama']}), "
          f"F0 {out['confidence_top1']['f0']}")
    print(f"{'arm':10s} {'R':>7s} {'R_acc':>7s} {'conf':>7s} {'both_wrong':>11s} {'fixed':>7s}  " +
          "  ".join(f"{c[:9]:>9s} R/R_acc" for c in cats))
    for k, r in out["arms"].items():
        s = r["saving_by_class"]
        print(f"{k:10s} {r['R']:7.3f} {r['r_acc']:7.3f} {r['confidence_top1']:7.3f} "
              f"{s['both_wrong']['share_of_saving']:11.3f} {s['t0_wrong_x_right']['share_of_saving']:7.3f}  " +
              "  ".join(f"{r['by_category'][c]['R']:7.3f}/{r['by_category'][c]['r_acc']:.3f}" for c in cats))
    print("both_wrong / fixed: share of the arm's saved bits at positions where T0 and the arm are both wrong / where "
          "the arm fixes T0's top-1")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
