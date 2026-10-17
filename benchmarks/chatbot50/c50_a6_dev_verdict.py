"""C50-A6 part 2 verdict: the FIT candidate C against R on DEV documents 13-32 and the A4 task panels.

Gates (C50_A6_PART2_DEV_TASKS_PROTOCOL_20261017.md):
  D1 DEV documents  paired dBPB (C - R) over the 60 DEV documents, document bootstrap (10,000 draws, seed
                    20261016): upper CI95 <= +0.015 (ROADMAP §5 A6).
  T2-T6 tasks       the A4 rule with R in the reference role and C in the candidate role
                    (c50_a4_verdict.task_gate: ability floor, b - c <= max(2 sqrt(b + c), 1), C >= floor(0.90 R)).
  T7 rollout        FAIL iff >= 3 documents degenerate in C and not in R.
A6 confirmed iff every gate PASSES.  It chooses nothing: the arm under test was fixed by the FIT verdict.

Usage:
    python -I benchmarks/chatbot50/c50_a6_dev_verdict.py --inputs E:/.../a6dev/inputs --work E:/.../a6dev \
        --r R_scores.json --c C_scores.json --json benchmarks/chatbot50/results/c50_a6_dev_verdict.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_a4_verdict import REC, task_gate  # noqa: E402
from c50_compare import byte_lengths  # noqa: E402

SEED, DRAWS, BOUND = 20261016, 10_000, 0.015
TOKENIZER = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/sources/Ling-mini-2.0_a810f64/tokenizer.json")


def docs_gate(inputs: Path, work: Path) -> dict:
    meta = [json.loads(x) for x in (inputs / "docs_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    r = np.fromfile(work / "R" / "docs.pos.bin", dtype=REC)
    o = np.fromfile(work / "C" / "docs.pos.bin", dtype=REC)
    if len(r) != len(o) or not (np.array_equal(r["doc"], o["doc"]) and np.array_equal(r["target"], o["target"])):
        raise SystemExit("arms do not score the same document positions")
    blen = byte_lengths(TOKENIZER)[r["target"]].astype(np.float64)
    docs = np.array([m["index"] for m in meta])
    di = np.searchsorted(docs, r["doc"])
    bits_r = -r["logp_target"].astype(np.float64) / math.log(2)
    bits_o = -o["logp_target"].astype(np.float64) / math.log(2)
    nd = len(docs)
    num = np.bincount(di, weights=bits_o - bits_r, minlength=nd)
    den = np.bincount(di, weights=blen, minlength=nd)
    idx = np.random.default_rng(SEED).integers(0, nd, size=(DRAWS, nd))
    st = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = (float(x) for x in np.percentile(st, [2.5, 97.5]))
    cats = np.array([m["category"] for m in meta])
    return {"documents": nd, "positions": int(len(r)), "bytes": float(den.sum()),
            "bpb_R": float(bits_r.sum() / den.sum()), "bpb_C": float(bits_o.sum() / den.sum()),
            "dbpb": float(num.sum() / den.sum()), "dbpb_ci95": [lo, hi], "pass": hi <= BOUND,
            "top1_agreement": float((r["top1"] == o["top1"]).mean()),
            "by_category": {c: float(num[cats == c].sum() / den[cats == c].sum()) for c in sorted(set(cats))}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--r", required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    rs = json.loads(Path(args.r).read_text(encoding="utf-8"))
    cs = json.loads(Path(args.c).read_text(encoding="utf-8"))
    run = json.loads((Path(__file__).resolve().parent / "results" / "c50_a6_dev_run.json").read_text(encoding="utf-8"))
    out = {"cell": "C50-A6 part 2", "fit_candidate": run["fit_candidate"], "D1_dev_documents": docs_gate(Path(args.inputs), Path(args.work))}
    d = out["D1_dev_documents"]
    print(f"D1 DEV documents ({run['fit_candidate']} vs R): dBPB {d['dbpb']:+.5f} CI ({d['dbpb_ci95'][0]:+.5f}, "
          f"{d['dbpb_ci95'][1]:+.5f}) R {d['bpb_R']:.4f} C {d['bpb_C']:.4f}  {'PASS' if d['pass'] else 'FAIL'}  by category {d['by_category']}")
    verdicts = [d["pass"]]
    for g, name in (("T2", "piqa"), ("T3", "arc"), ("T4", "gsm8k"), ("T5", "ifeval"), ("T6", "chat")):
        t = task_gate(name, rs[name], cs[name])
        t = {k.replace("bf16", "R").replace("p0", "C"): v for k, v in t.items()}
        out[f"{g}_{name}"] = t
        verdicts.append(t["verdict"] == "PASS")
        print(f"{g} {name}: R {t['R_correct']}/{t['n']} C {t['C_correct']}/{t['n']} "
              f"(R-only {t['R_only']}, C-only {t['C_only']})  {t['verdict']}")
    rb = {i["item"]: i["degenerate"] for i in rs["rollout"]["items"]}
    rq = {i["item"]: i["degenerate"] for i in cs["rollout"]["items"]}
    only = sorted(k for k in rq if rq[k] and not rb[k])
    out["T7_rollout"] = {"n": len(rq), "R_degenerate": sum(rb.values()), "C_degenerate": sum(rq.values()),
                         "C_only": only, "verdict": "FAIL" if len(only) >= 3 else "PASS"}
    verdicts.append(out["T7_rollout"]["verdict"] == "PASS")
    print(f"T7 rollout: R {sum(rb.values())} C {sum(rq.values())} C-only {len(only)}  {out['T7_rollout']['verdict']}")
    out["verdict"] = "CONFIRMED" if all(verdicts) else "NOT CONFIRMED"
    print("VERDICT", out["verdict"])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
