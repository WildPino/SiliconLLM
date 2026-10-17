"""C50-A5 part 1 selection: applies the registered rule to the offline report (it chooses nothing itself).

Rule (C50_A5_APPROXIMATE_HEAD_PROTOCOL_20261017.md): among the priced candidates (preview with simulated
4-bit storage, IVF with Q8_0 centroids) whose FIT top-1 recall point estimate is >= 0.999, select the one
with the fewest bytes per token; ties go to the smaller candidate count.  None reaching 0.999 = FAIL.
The float32 preview is not priced (it only separates rank error from storage error).

Registered prediction (scored, not a gate): IVF with exact rerank reaches 0.999 at < 15 MB/token.

Usage:
    python -I benchmarks/chatbot50/c50_a5_select.py --report benchmarks/chatbot50/results/c50_a5_offline_fit.json \
        --out benchmarks/chatbot50/results/c50_a5_select_fit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RECALL = 0.999
PREDICTED_IVF_MB = 15.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    o = rep["oracle"]
    if o["q8k"]["argmax_agreement"] < 0.999 or o["q8k"]["lse_abs_err_median"] > 1e-4:
        raise SystemExit("oracle control not met in the report: VOID")
    cands = []
    for key, ks in rep["preview"].items():
        if not key.endswith("_q4sim"):
            continue
        for k, v in ks.items():
            cands.append({"family": "preview", "config": f"{key} K={k}", "recall": v["recall"],
                          "mb_per_token": v["mb_per_token"], "candidates": int(k)})
    for key, cl in rep["ivf"].items():
        for p, v in cl["probes"].items():
            cands.append({"family": "ivf", "config": f"{key} P={p}", "recall": v["recall"],
                          "mb_per_token": v["mb_per_token"], "candidates": v["mean_candidates"]})
    ok = sorted((c for c in cands if c["recall"] >= RECALL), key=lambda c: (c["mb_per_token"], c["candidates"]))
    best_by_family = {}
    for fam in ("preview", "ivf"):
        f = [c for c in ok if c["family"] == fam]
        best_by_family[fam] = f[0] if f else None
        if not f:  # the closest miss, for the record
            near = max((c for c in cands if c["family"] == fam), key=lambda c: c["recall"])
            best_by_family[fam + "_highest_recall_miss"] = near
    ivf_ok = best_by_family["ivf"]
    out = {
        "cell": "C50-A5 part 1 selection (FIT)", "report": args.report, "recall_threshold": RECALL,
        "exact_head_mb_per_token": rep["exact_head_mb_per_token"],
        "oracle": {"q8k": o["q8k"], "f32": o["f32"]},
        "verdict": "SELECTED" if ok else "FAIL",
        "selected": ok[0] if ok else None,
        "best_by_family": best_by_family,
        "qualifying_count": len(ok), "candidate_count": len(cands),
        "prediction_ivf_under_15mb": {
            "predicted": True,
            "observed": bool(ivf_ok and ivf_ok["mb_per_token"] < PREDICTED_IVF_MB),
            "best_ivf_mb_per_token": ivf_ok["mb_per_token"] if ivf_ok else None},
    }
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("verdict", "selected", "prediction_ivf_under_15mb")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
