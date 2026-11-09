"""S2M's verdict, by the rules in S2M_FIRST_STACK_AT_2_BITS_PROTOCOL_20261109.md.  It chooses nothing.  Judged on points, chat and top-1, never on block loss.

ks_org:   stage 1's label and chat gate; its margin over as_both, paired CI95.
Stack:    stage 1's label and chat gate on ks_stk4.  excess = ks_stk4 - bal4 - ks_org, CI95: IT ADDS UP (CI95 holds 0), WORSE THAN ITS PARTS (upper < 0), BETTER THAN ITS PARTS (lower > 0).
          S2L's disproof, as written there: THE COMPOUNDING, NOT THE FORMAT if the excess is WORSE and at most stk4's excess plus S2L's seed gap (-5.70 + 1.90 = -3.80);
          else THE FORMAT KEEPS ITS GAIN IN THE STACK.
Control:  margin = ks_stk4 - ks_all4, paired CI95, against S2L's gap 1.90: BAL'S TRAINING PAYS (lower > 0, margin > gap), IQ2_KS BEATS BAL (the mirror), NOT SHOWN otherwise.
VOID per section: its arms' exports, panels and controls must hold.

Usage: python -I benchmarks/stage2/s2z_verdict.py --selftest
       python -I benchmarks/stage2/s2z_verdict.py --scores-dir D --work W --results DIR --json F
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

GAP, STK4_EXCESS = 1.90, -5.70
LINE = -3.80    # STK4_EXCESS + GAP, written out: in floats the sum is -3.8000000000000003
SECTIONS = {"ks_org": ("ks_org",), "stack": ("ks_stk4", "ks_org"), "control": ("ks_stk4", "ks_all4")}


def excess(ci: list[float]) -> str:
    return "WORSE THAN ITS PARTS" if ci[1] < 0 else "BETTER THAN ITS PARTS" if ci[0] > 0 else "IT ADDS UP"


def disproof(e: float, ci: list[float]) -> str:
    return "THE COMPOUNDING, NOT THE FORMAT" if ci[1] < 0 and e <= LINE else "THE FORMAT KEEPS ITS GAIN IN THE STACK"


def control(m: float, ci: list[float]) -> str:
    if ci[0] > 0 and m > GAP:
        return "BAL'S TRAINING PAYS"
    if ci[1] < 0 and -m > GAP:
        return "IQ2_KS BEATS BAL"
    return "NOT SHOWN"


def selftest() -> int:
    cases = [(excess([-3, -1]), "WORSE THAN ITS PARTS"), (excess([-1, 1]), "IT ADDS UP"), (excess([1, 2]), "BETTER THAN ITS PARTS"),
             (disproof(-5, [-8, -2]), "THE COMPOUNDING, NOT THE FORMAT"), (disproof(-3, [-6, -0.5]), "THE FORMAT KEEPS ITS GAIN IN THE STACK"),
             (disproof(-5, [-9, 0.5]), "THE FORMAT KEEPS ITS GAIN IN THE STACK"), (disproof(-3.80, [-7, -1]), "THE COMPOUNDING, NOT THE FORMAT"),
             (control(3, [1, 5]), "BAL'S TRAINING PAYS"), (control(1.5, [0.2, 3]), "NOT SHOWN"), (control(-3, [-5, -1]), "IQ2_KS BEATS BAL"), (control(3, [-1, 6]), "NOT SHOWN")]
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
    names = ["ref", "ks_org", "ks_stk4", "ks_all4", "bal4", "stk4", "st_attn", "as_both", "st_both"]
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
    deg = {x: degenerate_count(a.work / x) for x in names}
    passes = {x: int(arms[x]["chat"].sum()) for x in names}

    def gate(x):
        return passes[x] >= passes["ref"] - CHAT_MARGIN and deg[x] <= deg["ref"] + DEGEN_MARGIN
    res = a.results
    s2y = json.loads((res / "s2y_run.json").read_text(encoding="utf-8"))["arms"]["ks_org"]
    s2z = json.loads((res / "s2z_run.json").read_text(encoding="utf-8"))["arms"]
    exp = {"ks_org": json.loads((res / "s2y_export_ks_org.json").read_text(encoding="utf-8")),
           **{x: json.loads((res / f"s2z_export_{x}.json").read_text(encoding="utf-8")) for x in ("ks_stk4", "ks_all4")}}
    ks = json.loads((res / "s2z_ks.json").read_text(encoding="utf-8"))

    def done(rec):
        return all(rec["panels"].get(p, {}).get("status") == "COMPLETE" for p in rec["panel_set"])
    ok = {x: e["all_exact"] and not e["other_tensors_differ"] and s2z[x]["model_sha256"] == e["q4_sha256"] and done(s2z[x]) for x, e in exp.items()}
    ok["ks_org"] = ok["ks_org"] and done(s2y) and s2y["model_sha256"] == exp["ks_org"]["q4_sha256"]
    ok["ks_all4"] = ok["ks_all4"] and ks["control"]
    valid = {k: all(ok[x] for x in v) for k, v in SECTIONS.items()}

    oci, eci, cci = ci(boot["ks_org"] - boot["as_both"]), ci(boot["ks_stk4"] - boot["bal4"] - boot["ks_org"]), ci(boot["ks_stk4"] - boot["ks_all4"])
    e, m = delta["ks_stk4"] - delta["bal4"] - delta["ks_org"], delta["ks_stk4"] - delta["ks_all4"]
    out = {"sections_valid": valid, "delta": delta, "ci95": {x: ci(boot[x]) for x in boot}, "panels": {x: {p: 100.0 * float(arms[x][p].mean()) for p in PANELS} for x in names},
           "degenerate": deg, "chat_passes": passes,
           "ks_org": {"label": label(ci(boot["ks_org"])) if valid["ks_org"] else "VOID", "chat_ok": gate("ks_org"),
                      "over_as_both": delta["ks_org"] - delta["as_both"], "over_as_both_ci95": oci},
           "stack": {"label": label(ci(boot["ks_stk4"])) if valid["stack"] else "VOID", "chat_ok": gate("ks_stk4"), "sum_of_parts": delta["bal4"] + delta["ks_org"],
                     "excess": e, "excess_ci95": eci, "excess_label": excess(eci) if valid["stack"] else "VOID", "disproof": disproof(e, eci) if valid["stack"] else "VOID",
                     "stk4_excess_now": delta["stk4"] - delta["bal4"] - delta["st_attn"]},
           "control": {"margin": m, "ci95": cci, "label": control(m, cci) if valid["control"] else "VOID", "ks_all4_label": label(ci(boot["ks_all4"])), "ks_all4_chat_ok": gate("ks_all4")},
           "docs": {x: docs_metrics(a.work, x) for x in ("ks_org", "ks_stk4", "ks_all4", "stk4", "as_both")}}
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("docs", "panels", "ci95")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
