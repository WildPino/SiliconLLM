"""C50-A4-bis verdict: A4's document gate, unchanged, on text written after the model's release; the planted
positive must fire first (the script chooses nothing beyond the registered rule).

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_A4BIS_POSTRELEASE_PROTOCOL_20261018.md

G1, documents: c50_a4_verdict.docs_gate unchanged: paired dBPB against BF16, document bootstrap 10,000 draws,
    seed 20261016; PASS iff upper CI95 <= +0.020 (roadmap §2).
Stage "plant" (--plant only): the plant FIRES iff its G1 upper CI95 > +0.020 AND its lower CI95 > 0 (a measured
    loss, not only a wide interval).
Stage "final" (--cand and --plant): the plant must fire, else INSTRUMENT BLIND; then A4 PASS iff the candidate
    passes G1, else A4 FAIL (ROADMAP §5 S0's branch: Q5_K_M, then Q6_K).
Reported, not gated: top-1 disagreement with BF16 (document bootstrap), top-1 accuracy, mean |delta log p(target)|
    in nats, and the tail of BF16 log p(target) < -15.  The 17 October fidelity gate (2 x floor_top1) was withdrawn
    before registration (A4 result, design review step 1).

Usage:
    python -I benchmarks/chatbot50/c50_a4bis_verdict.py --inputs E:/.../a4bis_post/inputs --work E:/.../a4bis_post \
        --tokenizer tokenizer.json --json verdict.json --plant pc [--cand p0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_a4_verdict import DRAWS, REC, SEED, docs_gate  # noqa: E402
from c50_compare import byte_lengths  # noqa: E402

TAIL = -15.0


def fidelity(inputs: Path, work: Path, tokenizer: Path, cand: str) -> dict:
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
    blen = byte_lengths(tokenizer)[r["target"]].astype(np.float64)
    bits_r = -r["logp_target"].astype(np.float64) / np.log(2)
    pcat = cats[d_index]
    return {"bf16_bpb_by_category": {c: float(bits_r[pcat == c].sum() / blen[pcat == c].sum()) for c in sorted(set(cats))},
            "top1_disagreement": float(num.sum() / den.sum()), "top1_disagreement_ci95": [lo, hi],
            "by_category": {c: float(num[cats == c].sum() / den[cats == c].sum()) for c in sorted(set(cats))},
            "mean_abs_dlogp_nats": float(dl.mean()), "top1_accuracy": float((o["top1"] == o["target"]).mean()),
            "bf16_top1_accuracy": float((r["top1"] == r["target"]).mean()),
            "tail_positions": int(tail.sum()), "net_dbits_tail": float(dbits[tail].sum()),
            "net_dbits_outside_tail": float(dbits[~tail].sum())}


def fires(g: dict) -> bool:
    return g["dbpb_ci95"][1] > 0.020 and g["dbpb_ci95"][0] > 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--plant", required=True)
    ap.add_argument("--cand")
    args = ap.parse_args()
    inputs, work, tok = Path(args.inputs), Path(args.work), Path(args.tokenizer)
    out = {"cell": "C50-A4-bis verdict", "stage": "final" if args.cand else "plant", "plant": args.plant, "cand": args.cand}
    roles = [("plant", args.plant)] + ([("cand", args.cand)] if args.cand else [])
    for role, arm in roles:
        out[role] = {"G1_documents": docs_gate(inputs, work, tok, arm), "reported": fidelity(inputs, work, tok, arm)}
    out["plant_fires"] = fires(out["plant"]["G1_documents"])
    if not args.cand:
        out["verdict"] = "PLANT FIRES" if out["plant_fires"] else "PLANT DOES NOT FIRE"
    elif not out["plant_fires"]:
        out["verdict"] = "INSTRUMENT BLIND"
    else:
        out["verdict"] = "A4 PASS" if out["cand"]["G1_documents"]["pass"] else "A4 FAIL"
    for role, arm in roles:
        g, f = out[role]["G1_documents"], out[role]["reported"]
        print(f"{role:5} {arm:5} BPB {g['bpb_bf16']:.4f} -> {g['bpb_p0']:.4f} dBPB {g['dbpb']:+.5f} "
              f"CI ({g['dbpb_ci95'][0]:+.5f}, {g['dbpb_ci95'][1]:+.5f}) {'PASS' if g['pass'] else 'FAIL'}; "
              f"by category {json.dumps({k: round(v, 4) for k, v in g['by_category'].items()})}; "
              f"top-1 disagreement {f['top1_disagreement']:.4f}; |dlogp| {f['mean_abs_dlogp_nats']:.3f}")
    print("VERDICT", out["verdict"])
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
