"""C50-A4 verdict: P0 (Q4_K_M) against the BF16 source, from the two arms' scored outputs.

Gates (C50_A4_QUALITY_PROTOCOL_20261017.md):
  G1 documents   paired dBPB (P0 - BF16) over the 72 non-imatrix FIT documents, document bootstrap
                 (10,000 draws, seed 20261016): upper CI95 <= +0.020 (<= +0.010 recorded as the
                 preferred bound).  Per-category values reported, not gated.
  G2-G6 tasks    (piqa, arc, gsm8k, ifeval, chat) with b = items BF16 right and P0 wrong, c = the
                 reverse: INCONCLUSIVE if BF16 accuracy is not above the panel's ability floor;
                 otherwise PASS iff  b - c <= max(2 * sqrt(b + c), 1)  and  P0 >= floor(0.90 * BF16).
  G7 rollout     FAIL iff >= 3 documents degenerate in P0 and not in BF16.
A4 PASS iff every gate PASSES.  Beside the gates: STRAT-01's PIQA rule ceil(0.98 * BF16).

Usage:
    python -I benchmarks/chatbot50/c50_a4_verdict.py --inputs E:/.../a4/inputs --work E:/.../a4 \
        --bf16 bf16_scores.json --p0 p0_scores.json --tokenizer tokenizer.json [--cand-arm p0|pc] --json verdict.json
(--p0 is the candidate's scores file: P0's for the verdict, the planted positive's for the control)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_compare import byte_lengths  # noqa: E402  (the same byte count as A2b/A3)

SEED, DRAWS = 20261016, 10_000
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])
ABILITY_FLOOR = {"piqa": 0.50, "arc": 0.30, "gsm8k": 0.20, "ifeval": 0.30, "chat": 0.75}


def docs_gate(inputs: Path, work: Path, tokenizer: Path, cand: str) -> dict:
    meta = [json.loads(x) for x in (inputs / "docs_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    r = np.fromfile(work / "bf16" / "docs.pos.bin", dtype=REC)
    o = np.fromfile(work / cand / "docs.pos.bin", dtype=REC)
    if len(r) != len(o) or not (np.array_equal(r["doc"], o["doc"]) and np.array_equal(r["target"], o["target"])):
        raise SystemExit("arms do not score the same document positions")
    blen = byte_lengths(tokenizer)[r["target"]].astype(np.float64)
    docs = np.array([m["index"] for m in meta])
    d_index = np.searchsorted(docs, r["doc"])
    bits_r = -r["logp_target"].astype(np.float64) / math.log(2)
    bits_o = -o["logp_target"].astype(np.float64) / math.log(2)
    nd = len(docs)
    num = np.bincount(d_index, weights=bits_o - bits_r, minlength=nd)
    den = np.bincount(d_index, weights=blen, minlength=nd)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nd, size=(DRAWS, nd))
    stats = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = (float(x) for x in np.percentile(stats, [2.5, 97.5]))
    out = {"documents": nd, "positions": int(len(r)), "bytes": float(den.sum()),
           "bpb_bf16": float(np.bincount(d_index, weights=bits_r, minlength=nd).sum() / den.sum()),
           "bpb_p0": float(np.bincount(d_index, weights=bits_o, minlength=nd).sum() / den.sum()),
           "dbpb": float(num.sum() / den.sum()), "dbpb_ci95": [lo, hi], "pass": hi <= 0.020, "preferred_met": hi <= 0.010,
           "by_category": {}}
    cats = np.array([m["category"] for m in meta])
    for cat in sorted(set(cats)):
        sel = cats == cat
        out["by_category"][cat] = float(num[sel].sum() / den[sel].sum())
    return out


def task_gate(name: str, bf: dict, p0: dict) -> dict:
    a = {i["item"]: bool(i["correct"]) for i in bf["items"]}
    q = {i["item"]: bool(i["correct"]) for i in p0["items"]}
    if a.keys() != q.keys():
        raise SystemExit(f"{name}: arms scored different items")
    n = len(a)
    nb, nq = sum(a.values()), sum(q.values())
    b = sum(a[k] and not q[k] for k in a)
    c = sum(q[k] and not a[k] for k in a)
    out = {"n": n, "bf16_correct": nb, "p0_correct": nq, "bf16_accuracy": nb / n, "p0_accuracy": nq / n,
           "bf16_only": b, "p0_only": c, "ability_floor": ABILITY_FLOOR[name]}
    if nb / n <= ABILITY_FLOOR[name]:
        out["verdict"] = "INCONCLUSIVE_NO_BASELINE_ABILITY"
        return out
    allowed = max(2 * math.sqrt(b + c), 1.0)
    out["allowed_net_loss"] = allowed
    out["relative_floor"] = math.floor(0.90 * nb)
    out["verdict"] = "PASS" if (b - c <= allowed and nq >= math.floor(0.90 * nb)) else "FAIL"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--p0", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--cand-arm", default="p0", help="work subdirectory of the candidate arm (p0, or pc for the planted positive)")
    ap.add_argument("--json")
    args = ap.parse_args()
    bf = json.loads(Path(args.bf16).read_text(encoding="utf-8"))
    p0 = json.loads(Path(args.p0).read_text(encoding="utf-8"))
    out = {"candidate_arm": args.cand_arm,
           "G1_documents": docs_gate(Path(args.inputs), Path(args.work), Path(args.tokenizer), args.cand_arm)}
    g1 = out["G1_documents"]
    print(f"G1 documents: dBPB {g1['dbpb']:+.5f} CI ({g1['dbpb_ci95'][0]:+.5f}, {g1['dbpb_ci95'][1]:+.5f}) "
          f"BF16 {g1['bpb_bf16']:.4f} P0 {g1['bpb_p0']:.4f}  {'PASS' if g1['pass'] else 'FAIL'}"
          f"  preferred<=0.010 {'met' if g1['preferred_met'] else 'not met'}  by category {g1['by_category']}")
    verdicts = [g1["pass"]]
    for g, name in (("G2", "piqa"), ("G3", "arc"), ("G4", "gsm8k"), ("G5", "ifeval"), ("G6", "chat")):
        t = task_gate(name, bf[name], p0[name])
        out[f"{g}_{name}"] = t
        verdicts.append(t["verdict"] == "PASS")
        print(f"{g} {name}: BF16 {t['bf16_correct']}/{t['n']} P0 {t['p0_correct']}/{t['n']} "
              f"(BF16-only {t['bf16_only']}, P0-only {t['p0_only']})  {t['verdict']}")
    out["beside_piqa_strat01_rule"] = {"required": math.ceil(0.98 * out["G2_piqa"]["bf16_correct"]),
                                       "met": out["G2_piqa"]["p0_correct"] >= math.ceil(0.98 * out["G2_piqa"]["bf16_correct"])}
    rb = {i["item"]: i["degenerate"] for i in bf["rollout"]["items"]}
    rq = {i["item"]: i["degenerate"] for i in p0["rollout"]["items"]}
    only = sorted(k for k in rq if rq[k] and not rb[k])
    out["G7_rollout"] = {"n": len(rq), "bf16_degenerate": sum(rb.values()), "p0_degenerate": sum(rq.values()),
                         "p0_only": only, "verdict": "FAIL" if len(only) >= 3 else "PASS"}
    verdicts.append(out["G7_rollout"]["verdict"] == "PASS")
    print(f"G7 rollout: BF16 {sum(rb.values())} P0 {sum(rq.values())} P0-only {len(only)}  {out['G7_rollout']['verdict']}")
    out["verdict"] = "PASS" if all(verdicts) else "NOT PASSED"
    print("VERDICT", out["verdict"])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
