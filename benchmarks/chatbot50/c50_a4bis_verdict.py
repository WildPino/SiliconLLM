"""C50-A4-bis verdict: the A4 document gate re-run on fresh documents, with a per-position fidelity gate that
the planted positive must fail first (it chooses nothing beyond the registered rule).

Arms on the same fresh document positions: BF16, P0 (the candidate) and PC (the planted Q2_K).
G1-bis, fidelity: top-1 disagreement with BF16's top-1, pooled over positions (ratio of sums per document),
    document bootstrap 10,000 draws, seed 20261016.  PASS iff upper CI95 <= 0.0820 = 2 x floor_top1, where
    floor_top1 = 0.0410 is the A2b reference floor (llama.cpp R0-R2, a change of build alone) that A3 already
    used in this form.
G1, documents: the A4 G1 unchanged (c50_a4_verdict.docs_gate): paired dBPB upper CI95 <= +0.020.
Outcome:
    INSTRUMENT BLIND   iff PC passes G1-bis: no A4 verdict; design review 2.
    A4 PASS            iff PC fails G1-bis and P0 passes G1-bis and G1 (A4's G2-G5 and G7 stand as recorded;
                       G6 chat is reported only, A4-bis protocol).
    A4 FAIL            otherwise; ROADMAP §5 S0's branch applies (Q5_K_M, then Q6_K).
Reported, not gated: mean |delta log p(target)| in nats, top-1 accuracy, the tail of BF16 log p(target) < -15.

Usage:
    python -I benchmarks/chatbot50/c50_a4bis_verdict.py --inputs E:/.../a4bis/inputs --work E:/.../a4bis \
        --tokenizer tokenizer.json --json verdict.json [--cand p0] [--plant pc]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_a4_verdict import DRAWS, REC, SEED, docs_gate  # noqa: E402

FLOOR_TOP1 = 0.0410            # A2b R0-R2 (C50_A2B_REFERENCE_FLOOR_RESULT_20261017.md)
BOUND_TOP1 = 2 * FLOOR_TOP1    # 0.0820, the A3 form
TAIL = -15.0


def fidelity(inputs: Path, work: Path, cand: str) -> dict:
    meta = [json.loads(x) for x in (inputs / "docs_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    r = np.fromfile(work / "bf16" / "docs.pos.bin", dtype=REC)
    o = np.fromfile(work / cand / "docs.pos.bin", dtype=REC)
    if len(r) != len(o) or not (np.array_equal(r["doc"], o["doc"]) and np.array_equal(r["pos"], o["pos"])
                                and np.array_equal(r["target"], o["target"])):
        raise SystemExit(f"{cand}: arms do not score the same document positions")
    docs = np.array([m["index"] for m in meta])
    d_index = np.searchsorted(docs, r["doc"])
    nd = len(docs)
    dis = (o["top1"] != r["top1"]).astype(np.float64)
    num = np.bincount(d_index, weights=dis, minlength=nd)
    den = np.bincount(d_index, minlength=nd).astype(np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nd, size=(DRAWS, nd))
    stats = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = (float(x) for x in np.percentile(stats, [2.5, 97.5]))
    dl = np.abs(o["logp_target"].astype(np.float64) - r["logp_target"].astype(np.float64))
    tail = r["logp_target"] < TAIL
    dbits = (r["logp_target"].astype(np.float64) - o["logp_target"].astype(np.float64)) / np.log(2)
    cats = np.array([m["category"] for m in meta])
    return {"top1_disagreement": float(num.sum() / den.sum()), "top1_disagreement_ci95": [lo, hi],
            "bound": BOUND_TOP1, "pass": hi <= BOUND_TOP1,
            "by_category": {c: float(num[cats == c].sum() / den[cats == c].sum()) for c in sorted(set(cats))},
            "mean_abs_dlogp_nats": float(dl.mean()), "top1_accuracy": float((o["top1"] == o["target"]).mean()),
            "bf16_top1_accuracy": float((r["top1"] == r["target"]).mean()),
            "tail_positions": int(tail.sum()), "net_dbits_tail": float(dbits[tail].sum()),
            "net_dbits_outside_tail": float(dbits[~tail].sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--cand", default="p0")
    ap.add_argument("--plant", default="pc")
    args = ap.parse_args()
    inputs, work, tok = Path(args.inputs), Path(args.work), Path(args.tokenizer)
    out = {"cell": "C50-A4-bis verdict", "cand": args.cand, "plant": args.plant}
    for role, arm in (("cand", args.cand), ("plant", args.plant)):
        out[role] = {"G1bis_fidelity": fidelity(inputs, work, arm), "G1_documents": docs_gate(inputs, work, tok, arm)}
    c, p = out["cand"], out["plant"]
    if p["G1bis_fidelity"]["pass"]:
        out["verdict"] = "INSTRUMENT BLIND"
    elif c["G1bis_fidelity"]["pass"] and c["G1_documents"]["pass"]:
        out["verdict"] = "A4 PASS"
    else:
        out["verdict"] = "A4 FAIL"
    for role in ("cand", "plant"):
        f, g = out[role]["G1bis_fidelity"], out[role]["G1_documents"]
        print(f"{role:5} {args.cand if role == 'cand' else args.plant:5} "
              f"top-1 disagreement {f['top1_disagreement']:.4f} CI ({f['top1_disagreement_ci95'][0]:.4f}, "
              f"{f['top1_disagreement_ci95'][1]:.4f}) bound {BOUND_TOP1:.4f} {'PASS' if f['pass'] else 'FAIL'}; "
              f"dBPB {g['dbpb']:+.5f} CI ({g['dbpb_ci95'][0]:+.5f}, {g['dbpb_ci95'][1]:+.5f}) {'PASS' if g['pass'] else 'FAIL'}; "
              f"|dlogp| {f['mean_abs_dlogp_nats']:.3f}")
    print("VERDICT", out["verdict"])
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
