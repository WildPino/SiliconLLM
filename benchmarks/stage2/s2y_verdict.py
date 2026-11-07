"""S2L's verdict, by the rules in S2L_ORGANS_FIRST_TEST_PROTOCOL_20261107.md.  It chooses nothing.  Judged on top-1 and panels, never on block loss.

The noise: gap = |Delta(st_both2) - Delta(st_both)|, top-1 gap the same on the documents.  One pair of seeds, and the seeds change the order of the data only.
Grid, as_both against st_both, margin with its paired CI95:
    THE GRID MOVES THE ORGANS   CI95 lower > 0, margin > gap, top-1 gain > top-1 gap
    THE GRID HURTS THE ORGANS   CI95 upper < 0, -margin > gap, top-1 loss > top-1 gap
    WITHIN THE NOISE            otherwise
Stack: stk4's label by stage 1's rule and chat gate, beside the sum of its parts.
st_ffn: recovered = (top-1 of the arm - top-1 of st_ffn) / (1 - top-1 of st_ffn), on ff_nodense, ff_noearly, ff_nolate.
    ONE GROUP CARRIES IT        the best recovers >= 0.5 and the next at most half of it
    IT IS SPREAD                otherwise
Outliers: OUTLIERS WEIGH if recovered(ff_keepout) - recovered(ff_keeprnd) >= 0.15, else OUTLIERS DO NOT CARRY IT.
IQ2_KS, ks_org on the documents:
    IQ2_KS BEATS THE TRAINING   top-1 above st_both's by more than the top-1 gap, and KL under st_both's
    IQ2_KS SITS BETWEEN         top-1 above ph_both's by 0.10 or more
    NO BETTER THAN POST-HOC     otherwise
VOID: an export fails; the st_both control on the tool fails; a trained arm is not lower than its post-hoc on every layer, or did not log its 20 layers; IQ2_KS's imatrix control fails.

Usage: python -I benchmarks/stage2/s2y_verdict.py --selftest
       python -I benchmarks/stage2/s2y_verdict.py --scores-dir D --work W --results DIR --json F
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
from s1_verdict import CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS, SEED, degenerate_count, docs_metrics, label, points, vec  # noqa: E402

GROUPS = ("ff_nodense", "ff_noearly", "ff_nolate")
EXPORTS = ("as_both", "st_both2", "stk4") + GROUPS + ("ff_keepout", "ff_keeprnd", "ks_org")
SUM_STACK = -4.742690058479546 - 6.25


def grid(margin: float, ci: list[float], gap: float, top1: float, top1_gap: float) -> str:
    if ci[0] > 0 and margin > gap and top1 > top1_gap:
        return "THE GRID MOVES THE ORGANS"
    if ci[1] < 0 and -margin > gap and -top1 > top1_gap:
        return "THE GRID HURTS THE ORGANS"
    return "WITHIN THE NOISE"


def groups(rec: dict[str, float]) -> str:
    v = sorted(rec.values(), reverse=True)
    return "ONE GROUP CARRIES IT" if v[0] >= 0.5 and v[1] <= v[0] / 2 else "IT IS SPREAD"


def outliers(keep: float, rnd: float) -> str:
    return "OUTLIERS WEIGH" if keep - rnd >= 0.15 else "OUTLIERS DO NOT CARRY IT"


def ks(t: float, kl: float, both_t: float, both_kl: float, ph_t: float, top1_gap: float) -> str:
    if t > both_t + top1_gap and kl < both_kl:
        return "IQ2_KS BEATS THE TRAINING"
    return "IQ2_KS SITS BETWEEN" if t >= ph_t + 0.10 else "NO BETTER THAN POST-HOC"


def selftest() -> int:
    cases = [(grid(3, [1, 5], 1, 0.05, 0.01), "THE GRID MOVES THE ORGANS"), (grid(1, [0.2, 2], 2, 0.05, 0.01), "WITHIN THE NOISE"), (grid(3, [1, 5], 1, 0.005, 0.01), "WITHIN THE NOISE"),
             (grid(-3, [-5, -1], 1, -0.05, 0.01), "THE GRID HURTS THE ORGANS"), (grid(3, [-1, 5], 1, 0.05, 0.01), "WITHIN THE NOISE"),
             (groups({"a": 0.6, "b": 0.1, "c": 0.2}), "ONE GROUP CARRIES IT"), (groups({"a": 0.6, "b": 0.4, "c": 0.2}), "IT IS SPREAD"), (groups({"a": 0.4, "b": 0.1, "c": 0.1}), "IT IS SPREAD"),
             (outliers(0.3, 0.1), "OUTLIERS WEIGH"), (outliers(0.2, 0.1), "OUTLIERS DO NOT CARRY IT"),
             (ks(0.7, 1.0, 0.63, 1.57, 0.05, 0.01), "IQ2_KS BEATS THE TRAINING"), (ks(0.7, 2.0, 0.63, 1.57, 0.05, 0.01), "IQ2_KS SITS BETWEEN"), (ks(0.1, 9, 0.63, 1.57, 0.05, 0.01), "NO BETTER THAN POST-HOC")]
    ok = all(a == b for a, b in cases)
    print("selftest", "PASS" if ok else "FAIL", len(cases))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("scores-dir", "work", "results", "json"):
        ap.add_argument(f"--{x}", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from s2h_verdict import chat
    names = ["ref", "st_both", "st_both2", "as_both", "stk4"]
    arms = {x: {p: vec(json.loads((a.scores_dir / f"{x}.json").read_text(encoding="utf-8")), p) for p in PANELS} for x in names}
    n = {p: len(arms["ref"][p]) for p in PANELS}
    rng = np.random.default_rng(SEED)
    boot = {x: [] for x in names[1:]}
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        r = points(arms["ref"], idx)
        for x in boot:
            boot[x].append(points(arms[x], idx) - r)
    boot = {x: np.array(v) for x, v in boot.items()}

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    ref = points(arms["ref"])
    delta = {x: points(arms[x]) - ref for x in boot}
    docs = {x: docs_metrics(a.work, x) for x in ("st_both", "ph_both", "st_ffn", "st_both2", "as_both", "stk4", *GROUPS, "ff_keepout", "ff_keeprnd", "ks_org")}
    t1 = {x: d["top1_agreement"] for x, d in docs.items()}
    gap, top1_gap = abs(delta["st_both2"] - delta["st_both"]), abs(t1["st_both2"] - t1["st_both"])
    res = a.results
    exports = {x: json.loads((res / f"s2y_export_{x}.json").read_text(encoding="utf-8")) for x in EXPORTS}
    run = json.loads((res / "s2y_run.json").read_text(encoding="utf-8"))["arms"]
    ok_exports = all(e["all_exact"] and not e["other_tensors_differ"] and run[x]["model_sha256"] == e["q4_sha256"] for x, e in exports.items())
    ok_panels = all(run[x]["panels"].get(p, {}).get("status") == "COMPLETE" for x in EXPORTS for p in run[x]["panel_set"])
    logs = {x: json.loads((res / f"s2y_train_{x}.json").read_text(encoding="utf-8"))["layers"] for x in ("as_both", "st_both2")}
    ok_train = all(len(v) == 20 and all(r["trained"] < r["posthoc"] for r in v.values()) for v in logs.values())
    ok_tool = json.loads((res / "s2y_tool_check.json").read_text(encoding="utf-8"))["equal"]
    ok_ks = json.loads((res / "s2y_ks.json").read_text(encoding="utf-8"))["control"]
    controls = {"exports": ok_exports, "panels": ok_panels, "training": ok_train, "tool": ok_tool, "imatrix": ok_ks}
    margin = delta["as_both"] - delta["st_both"]
    mci = ci(boot["as_both"] - boot["st_both"])
    sci = ci(boot["stk4"])
    deg = {x: degenerate_count(a.work / x) for x in names}
    chat_ok = int(arms["stk4"]["chat"].sum()) >= int(arms["ref"]["chat"].sum()) - CHAT_MARGIN and deg["stk4"] <= deg["ref"] + DEGEN_MARGIN
    rec = {x: (t1[x] - t1["st_ffn"]) / (1 - t1["st_ffn"]) for x in (*GROUPS, "ff_keepout", "ff_keeprnd")}
    out = {"controls": controls, "outcome": "VOID" if not all(controls.values()) else None,
           "noise": {"gap": gap, "top1_gap": top1_gap, "paired_ci95": ci(boot["st_both2"] - boot["st_both"])},
           "delta": delta, "ci95": {x: ci(boot[x]) for x in boot}, "panels": {x: {p: 100.0 * float(arms[x][p].mean()) for p in PANELS} for x in names},
           "degenerate": deg, "chat_passes": {x: int(arms[x]["chat"].sum()) for x in names},
           "grid": {"margin": margin, "ci95": mci, "top1_gain": t1["as_both"] - t1["st_both"], "label": grid(margin, mci, gap, t1["as_both"] - t1["st_both"], top1_gap)},
           "stack": {"delta": delta["stk4"], "ci95": sci, "label": label(sci), "chat_ok": chat_ok, "sum_of_parts": SUM_STACK, "excess": delta["stk4"] - SUM_STACK},
           "st_ffn": {"recovered": rec, "groups": groups({x: rec[x] for x in GROUPS}), "outliers": outliers(rec["ff_keepout"], rec["ff_keeprnd"]),
                      "chat": {x: chat(a.work, x) for x in (*GROUPS, "ff_keepout", "ff_keeprnd")}},
           "iq2_ks": ks(t1["ks_org"], docs["ks_org"]["kl_top32_nats"], t1["st_both"], docs["st_both"]["kl_top32_nats"], t1["ph_both"], top1_gap),
           "docs": docs}
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "docs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
