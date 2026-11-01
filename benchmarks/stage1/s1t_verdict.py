"""Stage 1 (a) part 2 verdict: decode-only k = 2 with k2 = 16, untrained and after the short training (it chooses
nothing beyond the registered rule).

Arms (s1t_run.json): top2dm16 (P0, untrained) and, if it ran, trained; against part 1's ref.  Points, Delta, CI95 (the
same 10,000 draws, seed 20261030), labels and the chat gate are part 1's (s1_verdict.py).
    TRAINING HELPS   the CI95 of Delta(trained) - Delta(top2dm16) lies above 0.
Decisions the rule fixes:
    step 2    runs only if top2dm16 does not FIT THE SHARE;
    outcome   "k = 2 fits the share, untrained"   top2dm16 FITS THE SHARE;
              "k = 2 fits the share, trained"     trained FITS THE SHARE;
              "k = 2 fits the budget only"        an arm FITS (within budget, chat OK), none the share;
              "k = 2 does not fit"                neither arm FITS.
--selftest runs planted arms through the rule and fails unless each gets its registered outcome.

Usage:
    python -I benchmarks/stage1/s1t_verdict.py --scores-dir <dir> --work E:/SiliconLLM_Artifacts/stage1/s1a \
        --run-a results/s1a_run.json --run-t results/s1t_run.json --json results/s1t_verdict.json
    python -I benchmarks/stage1/s1t_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_verdict import (CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS, SEED, degenerate_count,  # noqa: E402
                        docs_metrics, label, moe_totals, points, vec)


def decide(arms: dict[str, dict[str, np.ndarray]], degenerate: dict[str, int], untrained: str = "top2dm16") -> dict:
    ref = arms["ref"]
    n = {p: len(ref[p]) for p in PANELS}
    for a, v in arms.items():
        if {p: len(v[p]) for p in PANELS} != n:
            raise SystemExit(f"{a} does not score the same items as ref")
    rng = np.random.default_rng(SEED)
    boot = {a: [] for a in arms}
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        r = points(ref, idx)
        for a, v in arms.items():
            boot[a].append(points(v, idx) - r)
    boot = {a: np.array(b) for a, b in boot.items()}
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    out = {"ref_points": points(ref), "untrained": untrained, "arms": {}}
    for a, v in arms.items():
        c = ci(boot[a])
        chat_ok = (int(v["chat"].sum()) >= int(ref["chat"].sum()) - CHAT_MARGIN
                   and degenerate[a] <= degenerate["ref"] + DEGEN_MARGIN)
        lab = label(c)
        out["arms"][a] = {"points": points(v), "delta": points(v) - points(ref), "ci95": c, "label": lab,
                          "chat_ok": chat_ok, "chat_passes": int(v["chat"].sum()), "degenerate": degenerate[a],
                          "fits": lab in ("WITHIN SHARE", "WITHIN BUDGET") and chat_ok,
                          "fits_share": lab == "WITHIN SHARE" and chat_ok,
                          "panels": {p: 100.0 * float(v[p].mean()) for p in PANELS}}
    U = out["arms"][untrained]
    out["step2_runs"] = not U["fits_share"]
    T = out["arms"].get("trained")
    if T is not None:
        mci = ci(boot["trained"] - boot[untrained])
        T["training"] = {"against": untrained, "margin": T["delta"] - U["delta"], "ci95": mci, "helps": mci[0] > 0}
    if U["fits_share"]:
        out["outcome"] = "k = 2 fits the share, untrained"
    elif T is not None and T["fits_share"]:
        out["outcome"] = "k = 2 fits the share, trained"
    elif U["fits"] or (T is not None and T["fits"]):
        out["outcome"] = "k = 2 fits the budget only"
    else:
        out["outcome"] = "k = 2 does not fit"
    return out


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

    cases = {  # name: (trained or None, untrained, degenerate of trained, outcome, helps, step 2 runs)
        "untrained fits the share": (None, lose(0.5, 5, 1), 0, "k = 2 fits the share, untrained", None, False),
        "trained fits the share": (lose(0.5, 4, 1), lose(8.0, 5, 4), 0, "k = 2 fits the share, trained", True, True),
        "trained fits the budget only": (lose(1.8, 4, 2), lose(8.0, 5, 4), 0, "k = 2 fits the budget only", True, True),
        "neither fits": (lose(9.0, 4, 4), lose(12.0, 5, 6), 0, "k = 2 does not fit", None, True),
        "share fit blocked by the chat gate": (lose(0.5, 4, 1), lose(8.0, 5, 4), DEGEN_MARGIN + 1,
                                               "k = 2 does not fit", True, True),
    }
    bad = 0
    for name, (tr, u, dg, outcome, helps, step2) in cases.items():
        arms = {"ref": ref, "top2dm16": u, **({"trained": tr} if tr is not None else {})}
        out = decide(arms, {"ref": 0, "top2dm16": 0, "trained": dg})
        h = out["arms"]["trained"]["training"]["helps"] if tr is not None else None
        ok = out["outcome"] == outcome and (helps is None or h == helps) and out["step2_runs"] == step2
        bad += 0 if ok else 1
        print("selftest", "ok  " if ok else "FAIL", name, out["outcome"], "helps", h, "step 2", out["step2_runs"])
    print("selftest", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scores-dir")
    ap.add_argument("--work")
    ap.add_argument("--run-a")
    ap.add_argument("--run-t")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    work, recs = Path(args.work), {}
    for f, names in ((args.run_a, ("ref",)), (args.run_t, ("top2dm16", "trained"))):
        run = json.loads(Path(f).read_text(encoding="utf-8"))
        for a in names:
            rec = run["arms"].get(a)
            if rec is None and a == "trained":
                continue
            if rec is None or any(rec["panels"].get(p, {}).get("status") != "COMPLETE" for p in PANELS + ("docs",)):
                raise SystemExit(f"{a}: not every panel COMPLETE")
            recs[a] = rec
    arms = {a: {p: vec(json.loads((Path(args.scores_dir) / f"{a}.json").read_text(encoding="utf-8")), p) for p in PANELS}
            for a in recs}
    out = decide(arms, {a: degenerate_count(work / a) for a in recs})
    for a in arms:
        out["arms"][a]["docs"] = docs_metrics(work, a)
        out["arms"][a]["moe"] = moe_totals(recs[a])
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for a, v in out["arms"].items():
        t = v.get("training", {})
        print(f"{a:8s} points {v['points']:6.2f} delta {v['delta']:+6.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] "
              f"{v['label']:13s} chat {v['chat_passes']} ok={v['chat_ok']} training {t.get('margin', 0):+.2f} "
              f"{t.get('ci95', '')} helps={t.get('helps', '')} top1 {v['docs']['top1_agreement']:.4f} "
              f"kl {v['docs']['kl_top32_nats']:.4f}")
    print("outcome", out["outcome"], "step 2 runs", out["step2_runs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
