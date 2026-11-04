"""Stage 2 test 2c verdict, by the rule registered in S2G_STACKED_ORGANS_PROTOCOL_20261104.md.  It chooses nothing.

Points, Delta, CI95, labels and the chat gate are stage 1's (s1_verdict.py), against ref's 55.19, the same 10,000 paired draws.
Controls, before any label (a failure is VOID):
    C1 floor   each trained arm's first layer: the original weights on the student stream give a block loss at most 1e-3 of
               the layer's post-hoc loss (a MoE layer's sum is not deterministic on the GPU);
    C2 exports every export: layout checked, tensors TQ2_0 and bit-exact, every other tensor equal to P0's, P0's sha256, the
               arm's exact tensor count; every trained arm logged all its layers; each panel's model is the export's sha256;
    C3 plant   ph_both, the known damage: document top-1 < 0.70, KL > 0.10, and it fails the chat gate.
Label of the judged arm, st_both:
    "THE ORGANS WITHOUT A ROUTER FIT THE SHARE"    FITS THE SHARE (CI95 lower end >= -2.5, chat gate);
    "THEY FIT THE BUDGET ALONE"                    FITS (lower end >= -5, chat gate);
    "THEY DO NOT FIT"                              otherwise.
Part 2's first test, margin = Delta(st_both) - Delta(x15), the arm test 3b judged, with its CI95:
    "STAYS ON THE EXPERTS"                   st_both FITS, or the margin's CI95 reaches 0 or above (a tie stays on the plan);
    "THE ORGANS COST MORE THAN THE EXPERTS"  otherwise: the delivery may recommend moving the first test, after test 4's page.
Reported, not judged: st_attn and st_ffn with their labels, the excess of the stack over the sum, the depth arms' documents.

Usage: python -I benchmarks/stage2/s2s_verdict.py --selftest
       python -I benchmarks/stage2/s2s_verdict.py --scores-dir D --work W --run F --results DIR --json F
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
sys.path.insert(0, str(HERE))
from s1_verdict import (CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS, SEED, degenerate_count, docs_metrics,  # noqa: E402
                        label, points, vec)

FULL = ("st_attn", "st_ffn", "st_both")
TRAINED = ("st_attn", "st_ffn", "st_both", "st_attn_hi")
EXPORTS = FULL + ("st_attn_lo", "st_attn_hi", "ph_both")
EXPERTS = "x15"
PLANT_TOP1, PLANT_KL = 0.70, 0.10
FLOOR = 1e-3
TENSORS = {"st_attn": 40, "st_ffn": 60, "st_both": 100, "st_attn_lo": 20, "st_attn_hi": 20, "ph_both": 100}
LAYERS = {"st_attn": 20, "st_ffn": 20, "st_both": 20, "st_attn_hi": 10}


def export_ok(e: dict, arm: str) -> bool:
    return (e["all_exact"] and all(e["layout_checked"].values()) and not e["other_tensors_differ"] and e["p0_sha256_checked"]
            and e["changed"] == TENSORS[arm])


def decide(arms: dict, degenerate: dict, ph: dict, exports: dict, floors: dict, ph_chat: dict, identity: bool = True) -> dict:
    ref = arms["ref"]
    n = {p: len(ref[p]) for p in PANELS}
    for a, v in arms.items():
        if {p: len(v[p]) for p in PANELS} != n:
            raise SystemExit(f"{a} does not score the same items as ref")
    rng = np.random.default_rng(SEED)
    names = list(FULL) + [EXPERTS]
    boot = {a: [] for a in names}
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        r = points(ref, idx)
        for a in names:
            boot[a].append(points(arms[a], idx) - r)
    boot = {a: np.array(b) for a, b in boot.items()}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    rec = {}
    for a in names:
        c = ci(boot[a])
        chat_ok = (int(arms[a]["chat"].sum()) >= int(ref["chat"].sum()) - CHAT_MARGIN
                   and degenerate[a] <= degenerate["ref"] + DEGEN_MARGIN)
        lab = label(c)
        rec[a] = {"delta": points(arms[a]) - points(ref), "ci95": c, "label": lab, "chat_ok": chat_ok,
                  "chat_passes": int(arms[a]["chat"].sum()), "degenerate": degenerate[a],
                  "fits": lab in ("WITHIN SHARE", "WITHIN BUDGET") and chat_ok,
                  "fits_share": lab == "WITHIN SHARE" and chat_ok,
                  "panels": {p: 100.0 * float(arms[a][p].mean()) for p in PANELS}}
    excess = boot["st_both"] - boot["st_attn"] - boot["st_ffn"]
    margin = boot["st_both"] - boot[EXPERTS]
    out = {"arms": rec,
           "excess_over_sum": {"value": rec["st_both"]["delta"] - rec["st_attn"]["delta"] - rec["st_ffn"]["delta"], "ci95": ci(excess)},
           "margin_vs_experts": {"value": rec["st_both"]["delta"] - rec[EXPERTS]["delta"], "ci95": ci(margin)},
           "controls": {"C1": all(abs(c) <= FLOOR * p for c, p in floors.values()),
                        "C2": identity and set(exports) == set(EXPORTS) and all(export_ok(e, a) for a, e in exports.items()),
                        "C3": ph["top1_agreement"] < PLANT_TOP1 and ph["kl_top32_nats"] > PLANT_KL
                        and (ph_chat["passes"] < int(ref["chat"].sum()) - CHAT_MARGIN or ph_chat["degenerate"] > degenerate["ref"] + DEGEN_MARGIN)}}
    if not all(out["controls"].values()):
        out["outcome"] = "VOID"
        return out
    b = rec["st_both"]
    out["outcome"] = ("THE ORGANS WITHOUT A ROUTER FIT THE SHARE" if b["fits_share"] else
                      "THEY FIT THE BUDGET ALONE" if b["fits"] else "THEY DO NOT FIT")
    out["part_2"] = ("STAYS ON THE EXPERTS" if b["fits"] or out["margin_vs_experts"]["ci95"][1] >= 0
                     else "THE ORGANS COST MORE THAN THE EXPERTS")
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
    ok = {"all_exact": True, "layout_checked": {"x": True}, "other_tensors_differ": [], "p0_sha256_checked": True}
    ex = {a: dict(ok, changed=TENSORS[a]) for a in EXPORTS}
    fl = {a: (0.0, 0.5) for a in TRAINED}
    ph = {"top1_agreement": 0.5, "kl_top32_nats": 1.5}
    pc = {"passes": 5, "degenerate": 9}
    deg = {"ref": 0, "st_attn": 0, "st_ffn": 0, "st_both": 0, "x15": 0}

    def arms(both, x15, chat=1):
        return {"ref": ref, "st_attn": lose(both / 2, 3, chat), "st_ffn": lose(both / 2, 4, chat),
                "st_both": lose(both, 5, chat), "x15": lose(x15, 6, 3)}
    stays, costs = "STAYS ON THE EXPERTS", "THE ORGANS COST MORE THAN THE EXPERTS"
    cases = [
        ("share", arms(0.5, 5.0), ph, ex, fl, pc, True, "THE ORGANS WITHOUT A ROUTER FIT THE SHARE", stays),
        ("budget stays whatever the margin", arms(3.0, 3.0), ph, ex, fl, pc, True, None, stays),
        ("experts clearly worse", arms(9.0, 18.0, 8), ph, ex, fl, pc, True, "THEY DO NOT FIT", stays),
        ("do not fit, worse than the experts", arms(14.0, 4.0, 8), ph, ex, fl, pc, True, "THEY DO NOT FIT", costs),
        ("tie stays", arms(9.0, 9.0, 8), ph, ex, fl, pc, True, "THEY DO NOT FIT", stays),
        ("chat gate", arms(0.5, 5.0, 8), ph, ex, fl, pc, True, "THEY DO NOT FIT", None),
        ("floor", arms(0.5, 5.0), ph, ex, dict(fl, st_both=(0.1, 0.5)), pc, True, "VOID", None),
        ("export", arms(0.5, 5.0), ph, dict(ex, st_ffn=dict(ex["st_ffn"], all_exact=False)), fl, pc, True, "VOID", None),
        ("partial arm", arms(0.5, 5.0), ph, dict(ex, st_both=dict(ex["st_both"], changed=60)), fl, pc, True, "VOID", None),
        ("export missing", arms(0.5, 5.0), ph, {k: v for k, v in ex.items() if k != "ph_both"}, fl, pc, True, "VOID", None),
        ("model not the export", arms(0.5, 5.0), ph, ex, fl, pc, False, "VOID", None),
        ("plant silent", arms(0.5, 5.0), {"top1_agreement": 0.85, "kl_top32_nats": 0.2}, ex, fl, pc, True, "VOID", None),
        ("plant passes chat", arms(0.5, 5.0), ph, ex, fl, {"passes": 28, "degenerate": 0}, True, "VOID", None),
    ]
    for name, a, p, e, f, c, ident, want, part in cases:
        got = decide(a, deg, p, e, f, c, ident)
        if (want and got["outcome"] != want) or (part and got.get("part_2") != part):
            print("FAIL", name, got["outcome"], got.get("part_2"), got["margin_vs_experts"], got["arms"]["st_both"]["ci95"])
            return 1
    print(f"selftest PASS, {len(cases)} cases")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("scores-dir", "work", "run", "results", "json"):
        ap.add_argument(f"--{x}", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from s2h_verdict import chat
    run = json.loads(a.run.read_text(encoding="utf-8"))
    for arm, rec in run["arms"].items():
        if any(rec["panels"].get(p, {}).get("status") != "COMPLETE" for p in rec["panel_set"]):
            raise SystemExit(f"{arm}: not every panel COMPLETE")
    x = json.loads((a.results / "s2x_run.json").read_text(encoding="utf-8"))["arms"][EXPERTS]["panels"]
    if len(x) != 7 or any(p.get("status") != "COMPLETE" for p in x.values()):
        raise SystemExit("x15: not every panel COMPLETE")
    names = ["ref"] + list(FULL) + [EXPERTS]
    arms = {n: {p: vec(json.loads((a.scores_dir / f"{n}.json").read_text(encoding="utf-8")), p) for p in PANELS} for n in names}
    deg = {n: degenerate_count(a.work / n) for n in names}
    exports = {n: json.loads((a.results / f"s2s_export_{n}.json").read_text(encoding="utf-8")) for n in EXPORTS}
    identity = all(run["arms"][n]["model_sha256"] == exports[n]["q4_sha256"] for n in EXPORTS)
    floors, logged = {}, True
    for n in TRAINED:
        log = json.loads((a.results / f"s2s_train_{n}.json").read_text(encoding="utf-8"))
        first = log["layers"][min(log["layers"], key=int)]
        floors[n] = (first["carried_only"], first["posthoc"])
        logged = logged and len(log["layers"]) == LAYERS[n]
    docs = {n: docs_metrics(a.work, n) for n in EXPORTS}
    out = decide(arms, deg, docs["ph_both"], exports, floors, chat(a.work, "ph_both"), identity and logged)
    out["docs"] = docs
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "docs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
