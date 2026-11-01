"""Stage 1 (a) part 1b verdict: k = 4 with the routing mass decoupled (it chooses nothing beyond the registered rule).

Arms: top4m16, top4m8, rnd4m16, top4dm16 (s1m_run.json), against part 1's ref, with part 1's top4 and top4d
(s1a_run.json) for the mechanism.  Points, Delta, CI95 (the same 10,000 draws, seed 20261030, for every arm), labels
and the chat gate are part 1's (s1_verdict.py).
    MASS HELPS (arm)  the CI95 of Delta(arm) - Delta(top4) lies above 0 (top4dm16: against top4d).
    Control           top4m16 against rnd4m16, as in part 1.
Decisions the rule fixes:
    k4_form     among top4m16 and top4m8, the one with the higher Delta that FITS THE SHARE (k = 4 everywhere, that
                k2); else top4dm16 if it FITS THE SHARE (decode-only, k2 = 16); else none.
    training_k  2 if k4_form is not none, else 4.
    start       the weighting part 2 starts from at k = 4: the mass arm (top4m16 or top4m8) with the higher Delta if
                its mass helps, else "k2 = k1" (part 1's top4).
--selftest runs planted arms through the rule and fails unless each gets its registered outcome.

Usage:
    python -I benchmarks/stage1/s1m_verdict.py --scores-dir <dir with <arm>.json> --work E:/SiliconLLM_Artifacts/stage1/s1a \
        --run-a benchmarks/stage1/results/s1a_run.json --run-m benchmarks/stage1/results/s1m_run.json --json verdict.json
    python -I benchmarks/stage1/s1m_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_verdict import (BUDGET, CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS, SEED, degenerate_count,  # noqa: E402
                        docs_metrics, label, moe_totals, points, vec)

MASS_ARMS = ("top4m16", "top4m8")
BASE = {"top4m16": "top4", "top4m8": "top4", "top4dm16": "top4d"}
PAIRS = {"top4m16": "rnd4m16"}


def decide(arms: dict[str, dict[str, np.ndarray]], degenerate: dict[str, int]) -> dict:
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
    out = {"ref_points": points(ref), "arms": {}}
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
    for a, b in BASE.items():
        if a in arms and b in arms:
            mci = ci(boot[a] - boot[b])
            out["arms"][a]["mass"] = {"against": b, "margin": out["arms"][a]["delta"] - out["arms"][b]["delta"],
                                      "ci95": mci, "helps": mci[0] > 0}
    for t, c in PAIRS.items():
        if t in arms and c in arms:
            mci = ci(boot[t] - boot[c])
            outcome = ("BEATS CONTROL" if mci[0] > 0 else "ROUTER NO BETTER THAN CHANCE"
                       if out["arms"][c]["ci95"][1] < 0 else "CONTROL UNINFORMATIVE")
            out["arms"][t]["control"] = {"arm": c, "margin": out["arms"][t]["delta"] - out["arms"][c]["delta"],
                                         "ci95": mci, "outcome": outcome}
    A = out["arms"]
    share = [a for a in MASS_ARMS if a in A and A[a]["fits_share"]]
    if share:
        best = max(share, key=lambda a: A[a]["delta"])
        out["k4_form"] = f"everywhere, k2 = {16 if best == 'top4m16' else 8} ({best})"
    elif "top4dm16" in A and A["top4dm16"]["fits_share"]:
        out["k4_form"] = "decode-only, k2 = 16 (top4dm16)"
    else:
        out["k4_form"] = None
    out["training_k"] = 2 if out["k4_form"] else 4
    present = [a for a in MASS_ARMS if a in A]
    lead = max(present, key=lambda a: A[a]["delta"]) if present else None
    out["start"] = lead if lead and A[lead].get("mass", {}).get("helps") else "k2 = k1 (top4)"
    return out


def selftest() -> int:
    rng = np.random.default_rng(1)
    sizes = {"nq": 600, "tqa": 600, "gsm8k": 100, "arc": 600, "mmlu": 570, "chat": 40}
    ref = {p: rng.random(sizes[p]) < 0.55 for p in PANELS}

    def lose(pts: float, seed: int, chat_drop: int = 0) -> dict:
        """ref with about `pts` points removed on every panel; chat loses exactly chat_drop items"""
        r = np.random.default_rng(seed)
        v = {}
        for p in PANELS:
            x = ref[p].copy()
            on = np.flatnonzero(x)
            drop = chat_drop if p == "chat" else int(round(pts / 100 * len(x)))
            x[r.choice(on, size=min(drop, len(on)), replace=False)] = False
            v[p] = x
        return v

    cases = {
        # name: (arms beyond ref, expected k4_form prefix or None, training_k, start)
        "m16 fits the share": ({"top4": lose(7.5, 2, 3), "top4d": lose(4.8, 3, 6), "top4m16": lose(0.6, 4, 1),
                                "top4m8": lose(4.0, 5, 2), "rnd4m16": lose(30, 6, 20), "top4dm16": lose(1, 7, 1)},
                               "everywhere, k2 = 16", 2, "top4m16"),
        "m8 fits, m16 does not": ({"top4": lose(7.5, 2, 3), "top4d": lose(4.8, 3, 6), "top4m16": lose(6.5, 4, 2),
                                   "top4m8": lose(0.5, 5, 1), "rnd4m16": lose(30, 6, 20), "top4dm16": lose(4, 7, 6)},
                                  "everywhere, k2 = 8", 2, "top4m8"),
        "only decode fits": ({"top4": lose(7.5, 2, 3), "top4d": lose(4.8, 3, 6), "top4m16": lose(4.0, 4, 2),
                              "top4m8": lose(6.0, 5, 2), "rnd4m16": lose(30, 6, 20), "top4dm16": lose(0.5, 7, 1)},
                             "decode-only", 2, "top4m16"),
        "helps, nothing fits": ({"top4": lose(7.5, 2, 3), "top4d": lose(4.8, 3, 6), "top4m16": lose(3.5, 4, 2),
                                 "top4m8": lose(6.0, 5, 2), "rnd4m16": lose(30, 6, 20), "top4dm16": lose(4, 7, 8)},
                                None, 4, "top4m16"),
        "does not help": ({"top4": lose(7.5, 2, 3), "top4d": lose(4.8, 3, 6), "top4m16": lose(7.6, 4, 3),
                           "top4m8": lose(7.8, 5, 3), "rnd4m16": lose(30, 6, 20), "top4dm16": lose(5, 7, 8)},
                          None, 4, "k2 = k1 (top4)"),
        # the same top4m16 as the first case, with 4 degenerate rollouts more than ref: the chat gate alone blocks it
        "chat gate blocks a share fit": ({"top4": lose(7.5, 2, 3), "top4d": lose(4.8, 3, 6), "top4m16": lose(0.6, 4, 1),
                                          "top4m8": lose(6.0, 5, 2), "rnd4m16": lose(30, 6, 20), "top4dm16": lose(4, 7, 8)},
                                         None, 4, "top4m16"),
    }
    degen = {"chat gate blocks a share fit": {"top4m16": DEGEN_MARGIN + 1}}
    bad = 0
    for name, (extra, form, tk, start) in cases.items():
        arms = {"ref": ref, **extra}
        out = decide(arms, {a: degen.get(name, {}).get(a, 0) for a in arms})
        got_form = out["k4_form"]
        ok = ((form is None and got_form is None) or (form is not None and got_form is not None and got_form.startswith(form)))
        ok = ok and out["training_k"] == tk and out["start"] == start
        ok = ok and out["arms"]["top4m16"]["control"]["outcome"] == "BEATS CONTROL"
        if name == "chat gate blocks a share fit":
            ok = ok and out["arms"]["top4m16"]["label"] == "WITHIN SHARE" and not out["arms"]["top4m16"]["chat_ok"]
        bad += 0 if ok else 1
        print("selftest", "ok  " if ok else "FAIL", name, got_form, out["training_k"], out["start"],
              {a: round(out["arms"][a]["delta"], 2) for a in extra})
    print("selftest", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scores-dir")
    ap.add_argument("--work")
    ap.add_argument("--run-a")
    ap.add_argument("--run-m")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    work = Path(args.work)
    recs = {}
    for f, names in ((args.run_a, ("ref", "top4", "top4d")), (args.run_m, ("top4m16", "rnd4m16", "top4m8", "top4dm16"))):
        run = json.loads(Path(f).read_text(encoding="utf-8"))
        for a in names:
            rec = run["arms"].get(a)
            if rec is None or any(rec["panels"].get(p, {}).get("status") != "COMPLETE" for p in PANELS + ("docs",)):
                print(f"{a}: not every panel COMPLETE; left out")
                continue
            recs[a] = rec
    if "ref" not in recs:
        raise SystemExit("ref is not complete")
    arms, degenerate = {}, {}
    for a in recs:
        s = json.loads((Path(args.scores_dir) / f"{a}.json").read_text(encoding="utf-8"))
        arms[a] = {p: vec(s, p) for p in PANELS}
        degenerate[a] = degenerate_count(work / a)
    out = decide(arms, degenerate)
    for a in arms:
        out["arms"][a]["docs"] = docs_metrics(work, a)
        out["arms"][a]["moe"] = moe_totals(recs[a])
        if "mass" in recs[a]["panels"].get("nq", {}):
            out["arms"][a]["mass_lines"] = {p: recs[a]["panels"][p].get("mass") for p in recs[a]["panels"]}
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for a, v in out["arms"].items():
        m, c = v.get("mass", {}), v.get("control", {})
        print(f"{a:8s} points {v['points']:6.2f} delta {v['delta']:+6.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] "
              f"{v['label']:13s} chat {v['chat_passes']} ok={v['chat_ok']} "
              f"mass {m.get('margin', 0):+.2f} {m.get('ci95', '')} helps={m.get('helps', '')} {c.get('outcome', '')} "
              f"top1 {v['docs']['top1_agreement']:.4f} kl {v['docs']['kl_top32_nats']:.4f}")
    print("k4_form", out["k4_form"], "training_k", out["training_k"], "start", out["start"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
