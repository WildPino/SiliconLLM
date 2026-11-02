"""Stage 1 (b) verdict: Ling-mini with the routed experts of all 19 MoE layers ternary and trained (it chooses nothing
beyond the registered rule).

Arm tern (s1b_run.json) against part 1's ref.  Points, Delta, CI95 (the same 10,000 draws, seed 20261030), labels and
the chat gate are part 1's (s1_verdict.py).  The stage's criterion is per layer, 1/19 of the 5-point budget; with 19
layers ternary it reads as Delta(tern) against the whole budget.
    "ternary experts survive"   tern FITS: WITHIN SHARE or WITHIN BUDGET, and chat OK;
    "ternary experts closed"    tern is OVER BUDGET (CI95 upper end below -5);
    "undecided"                 otherwise.
Reported, not judged: the document metrics (top-1, KL, dBPB) of tern, tern_ph (post-hoc ternary, no training) and
tern_l10 (layer 10 alone, trained on the original model's inputs), and tern_l10's dBPB against tern's / 19.
--selftest runs planted arms through the rule and fails unless each gets its registered outcome.

Usage:
    python -I benchmarks/stage1/s1b_verdict.py --scores-dir <dir> --work E:/SiliconLLM_Artifacts/stage1/s1a \
        --run-a results/s1a_run.json --run-b results/s1b_run.json --json results/s1b_verdict.json
    python -I benchmarks/stage1/s1b_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_verdict import (CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS, SEED, degenerate_count,  # noqa: E402
                        docs_metrics, label, points, vec)

N_MOE = 19


def decide(arms: dict[str, dict[str, np.ndarray]], degenerate: dict[str, int]) -> dict:
    ref, t = arms["ref"], arms["tern"]
    n = {p: len(ref[p]) for p in PANELS}
    if {p: len(t[p]) for p in PANELS} != n:
        raise SystemExit("tern does not score the same items as ref")
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        boot.append(points(t, idx) - points(ref, idx))
    c = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    chat_ok = (int(t["chat"].sum()) >= int(ref["chat"].sum()) - CHAT_MARGIN
               and degenerate["tern"] <= degenerate["ref"] + DEGEN_MARGIN)
    lab = label(c)
    rec = {"points": points(t), "delta": points(t) - points(ref), "ci95": c, "label": lab, "chat_ok": chat_ok,
           "chat_passes": int(t["chat"].sum()), "degenerate": degenerate["tern"],
           "fits": lab in ("WITHIN SHARE", "WITHIN BUDGET") and chat_ok,
           "panels": {p: 100.0 * float(t[p].mean()) for p in PANELS}}
    outcome = ("ternary experts survive" if rec["fits"] else "ternary experts closed" if lab == "OVER BUDGET"
               else "undecided")
    return {"ref_points": points(ref), "tern": rec, "outcome": outcome}


def selftest() -> int:
    rng = np.random.default_rng(1)
    sizes = {"nq": 600, "tqa": 600, "gsm8k": 100, "arc": 600, "mmlu": 570, "chat": 40}
    ref = {p: rng.random(sizes[p]) < 0.55 for p in PANELS}

    def lose(pts: float, seed: int, chat_drop: int = 0) -> dict:
        r = np.random.default_rng(seed)
        v = {}
        for p in PANELS:
            x = ref[p].copy()
            on = np.flatnonzero(x)
            drop = chat_drop if p == "chat" else int(round(pts / 100 * len(x)))
            x[r.choice(on, size=min(drop, len(on)), replace=False)] = False
            v[p] = x
        return v

    cases = {  # name: (tern, degenerate, outcome)
        "within the share": (lose(0.5, 4, 1), 0, "ternary experts survive"),
        "within the budget": (lose(2.0, 4, 2), 0, "ternary experts survive"),
        "over the budget": (lose(12.0, 4, 8), 0, "ternary experts closed"),
        "near the line": (lose(5.0, 4, 3), 0, "undecided"),
        "within the budget, chat fails": (lose(2.0, 4, 2), DEGEN_MARGIN + 1, "undecided"),
    }
    bad = 0
    for name, (t, dg, outcome) in cases.items():
        out = decide({"ref": ref, "tern": t}, {"ref": 0, "tern": dg})
        ok = out["outcome"] == outcome
        bad += 0 if ok else 1
        print("selftest", "ok  " if ok else "FAIL", name, out["outcome"], out["tern"]["label"],
              [round(x, 2) for x in out["tern"]["ci95"]])
    print("selftest", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scores-dir")
    ap.add_argument("--work")
    ap.add_argument("--run-a")
    ap.add_argument("--run-b")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    work = Path(args.work)
    run_a = json.loads(Path(args.run_a).read_text(encoding="utf-8"))
    run_b = json.loads(Path(args.run_b).read_text(encoding="utf-8"))
    need = {"ref": (run_a, PANELS + ("docs",)), "tern": (run_b, PANELS + ("docs",)),
            "tern_ph": (run_b, ("docs",)), "tern_l10": (run_b, ("docs",))}
    for a, (run, panels) in need.items():
        rec = run["arms"].get(a)
        if rec is None or any(rec["panels"].get(p, {}).get("status") != "COMPLETE" for p in panels):
            raise SystemExit(f"{a}: not every panel COMPLETE")
    arms = {a: {p: vec(json.loads((Path(args.scores_dir) / f"{a}.json").read_text(encoding="utf-8")), p)
                for p in PANELS} for a in ("ref", "tern")}
    out = decide(arms, {a: degenerate_count(work / a) for a in arms})
    out["docs"] = {a: docs_metrics(work, a) for a in ("tern", "tern_ph", "tern_l10")}
    out["one_layer"] = {"tern_l10_dbpb": out["docs"]["tern_l10"]["dbpb"],
                        "tern_dbpb_over_19": out["docs"]["tern"]["dbpb"] / N_MOE}
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    t = out["tern"]
    print(f"tern points {t['points']:6.2f} delta {t['delta']:+6.2f} [{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}] "
          f"{t['label']} chat {t['chat_passes']} ok={t['chat_ok']} degenerate {t['degenerate']}")
    for a, d in out["docs"].items():
        print(f"{a:9s} top1 {d['top1_agreement']:.4f} kl {d['kl_top32_nats']:.4f} dbpb {d['dbpb']:+.4f} "
              f"[{d['dbpb_ci95'][0]:+.4f},{d['dbpb_ci95'][1]:+.4f}]")
    print("one layer", out["one_layer"])
    print("outcome", out["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
