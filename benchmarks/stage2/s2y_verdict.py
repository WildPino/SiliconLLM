"""S2L's verdict, by the rules in S2L_ORGANS_FIRST_TEST_PROTOCOL_20261107.md with amendment 1.  It chooses nothing.  Judged on top-1 and panels, never on block loss.

The noise: gap = |Delta(st_both2) - Delta(st_both)|, top-1 gap the same on the documents.  One pair of seeds, the order of the data only: a single difference, reported as that.
Grid, as_both against sb_q40 (st_both's values in as_both's container): margin with its paired CI95, top-1 gain with its CI95 over documents.
    THE GRID MOVES THE ORGANS   both CI95 lower > 0, margin > gap, top-1 gain > top-1 gap
    THE GRID HURTS THE ORGANS   the mirror
    NOT SHOWN                   otherwise: it closes nothing
Container: Delta(sb_q40) - Delta(st_both), Delta(tq404) - Delta(t4), reported.  Bal: bal4 - tq404, CI95: BAL STANDS IN ITS OWN CONTAINER if lower > 0, else BAL'S MARGIN WAS NOT SHOWN.
Stack: stage 1's label and chat gate on stk4; excess = stk4 - bal4 - st_attn + ref, CI95:
    IT ADDS UP (CI95 holds 0), WORSE THAN ITS PARTS (upper < 0), BETTER THAN ITS PARTS (lower > 0)
st_ffn: what the stack does without a group, not the group's share.  recovered = (arm - st_ffn) / (ref - st_ffn), on top-1 and on chat passes, for ff_nodense, ff_noearly, ff_nolate.
    ONE GROUP MATTERS MOST      the best top-1 recovers >= 0.5, the next at most half of it, and the best's chat recovers >= 0.5
    IT IS SPREAD                otherwise
Outliers, all in BF16, from ff_bf: OUTLIERS WEIGH if top-1 recovered by ff_keepout exceeds both random draws' by 0.15 and its chat passes exceed both; else OUTLIERS DO NOT CARRY IT.
IQ2_KS, against the better of sb_bf and as_both by top-1 (2.19 bits against 2.0):
    IQ2_KS BEATS THE TRAINING   top-1 above it by more than the top-1 gap, KL under it, chat not under it
    IQ2_KS IS CLOSE             top-1 at least its - 0.10
    IQ2_KS IS BELOW             otherwise
VOID is per section: a section reads only if its arms' exports, panels and controls hold.  The tool check holds for all.

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
from s1_verdict import CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS, REC, SEED, degenerate_count, docs_metrics, label, points, vec  # noqa: E402

GROUPS = ("ff_nodense", "ff_noearly", "ff_nolate")
OUT = ("ff_keepout", "ff_keeprnd", "ff_keeprnd2")
SECTIONS = {"grid": ("as_both", "st_both2", "sb_q40"), "stack": ("stk4",), "st_ffn": GROUPS + OUT + ("ff_bf",), "iq2_ks": ("ks_org", "sb_bf", "as_both"), "bal": ("tq40", "tq404")}


def grid(margin: float, ci: list[float], gap: float, top1: float, tci: list[float], top1_gap: float) -> str:
    if ci[0] > 0 and tci[0] > 0 and margin > gap and top1 > top1_gap:
        return "THE GRID MOVES THE ORGANS"
    if ci[1] < 0 and tci[1] < 0 and -margin > gap and -top1 > top1_gap:
        return "THE GRID HURTS THE ORGANS"
    return "NOT SHOWN"


def excess(ci: list[float]) -> str:
    return "WORSE THAN ITS PARTS" if ci[1] < 0 else "BETTER THAN ITS PARTS" if ci[0] > 0 else "IT ADDS UP"


def groups(top1: dict[str, float], chat: dict[str, float]) -> str:
    best = max(top1, key=top1.get)
    v = sorted(top1.values(), reverse=True)
    return "ONE GROUP MATTERS MOST" if v[0] >= 0.5 and v[1] <= v[0] / 2 and chat[best] >= 0.5 else "IT IS SPREAD"


def outliers(keep: float, rnd: list[float], keep_chat: int, rnd_chat: list[int]) -> str:
    return "OUTLIERS WEIGH" if keep - max(rnd) >= 0.15 and keep_chat > max(rnd_chat) else "OUTLIERS DO NOT CARRY IT"


def ks(t: float, kl: float, chat: int, best_t: float, best_kl: float, best_chat: int, top1_gap: float) -> str:
    if t > best_t + top1_gap and kl < best_kl and chat >= best_chat:
        return "IQ2_KS BEATS THE TRAINING"
    return "IQ2_KS IS CLOSE" if t >= best_t - 0.10 else "IQ2_KS IS BELOW"


def bal(ci: list[float]) -> str:
    return "BAL STANDS IN ITS OWN CONTAINER" if ci[0] > 0 else "BAL'S MARGIN WAS NOT SHOWN"


def selftest() -> int:
    up, dn = [0.01, 0.1], [-0.1, -0.01]
    cases = [(grid(3, [1, 5], 1, 0.05, up, 0.01), "THE GRID MOVES THE ORGANS"), (grid(1, [0.2, 2], 2, 0.05, up, 0.01), "NOT SHOWN"), (grid(3, [1, 5], 1, 0.005, up, 0.01), "NOT SHOWN"),
             (grid(3, [1, 5], 1, 0.05, [-0.01, 0.1], 0.01), "NOT SHOWN"), (grid(-3, [-5, -1], 1, -0.05, dn, 0.01), "THE GRID HURTS THE ORGANS"), (grid(3, [-1, 5], 1, 0.05, up, 0.01), "NOT SHOWN"),
             (excess([-3, -1]), "WORSE THAN ITS PARTS"), (excess([-1, 1]), "IT ADDS UP"), (excess([1, 2]), "BETTER THAN ITS PARTS"),
             (groups({"a": 0.6, "b": 0.1, "c": 0.2}, {"a": 0.6, "b": 0, "c": 0}), "ONE GROUP MATTERS MOST"), (groups({"a": 0.6, "b": 0.1, "c": 0.2}, {"a": 0.2, "b": 0, "c": 0}), "IT IS SPREAD"),
             (groups({"a": 0.6, "b": 0.4, "c": 0.2}, {"a": 0.6, "b": 0, "c": 0}), "IT IS SPREAD"), (groups({"a": 0.4, "b": 0.1, "c": 0.1}, {"a": 0.6, "b": 0, "c": 0}), "IT IS SPREAD"),
             (outliers(0.3, [0.1, 0.12], 12, [5, 6]), "OUTLIERS WEIGH"), (outliers(0.3, [0.1, 0.2], 12, [5, 6]), "OUTLIERS DO NOT CARRY IT"), (outliers(0.3, [0.1, 0.1], 5, [5, 6]), "OUTLIERS DO NOT CARRY IT"),
             (ks(0.7, 1.0, 20, 0.63, 1.57, 15, 0.01), "IQ2_KS BEATS THE TRAINING"), (ks(0.7, 2.0, 20, 0.63, 1.57, 15, 0.01), "IQ2_KS IS CLOSE"), (ks(0.7, 1.0, 10, 0.63, 1.57, 15, 0.01), "IQ2_KS IS CLOSE"),
             (ks(0.4, 9, 2, 0.63, 1.57, 15, 0.01), "IQ2_KS IS BELOW"), (bal([1, 3]), "BAL STANDS IN ITS OWN CONTAINER"), (bal([-1, 3]), "BAL'S MARGIN WAS NOT SHOWN")]
    ok = all(a == b for a, b in cases)
    print("selftest", "PASS" if ok else "FAIL", len(cases))
    return 0 if ok else 1


def doc_top1(work: Path, arm: str) -> np.ndarray:
    r = np.fromfile(work / "ref" / "docs.pos.bin", dtype=REC)
    a = np.fromfile(work / arm / "docs.pos.bin", dtype=REC)
    same = r["top1"] == a["top1"]
    return np.array([same[r["doc"] == d].mean() for d in np.unique(r["doc"])])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("scores-dir", "work", "results", "json"):
        ap.add_argument(f"--{x}", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from s2h_verdict import chat
    names = ["ref", "st_both", "st_both2", "as_both", "sb_q40", "stk4", "bal4", "st_attn", "t4", "tq40", "tq404", "bal", "tern"]
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
    docs = {x: docs_metrics(a.work, x) for x in ("st_both", "ph_both", "st_ffn", "st_both2", "as_both", "sb_q40", "stk4", "sb_bf", "ff_bf", *GROUPS, *OUT, "ks_org")}
    t1 = {x: d["top1_agreement"] for x, d in docs.items()}
    gap, top1_gap = abs(delta["st_both2"] - delta["st_both"]), abs(t1["st_both2"] - t1["st_both"])
    res = a.results
    every = sorted({x for v in SECTIONS.values() for x in v})
    run = json.loads((res / "s2y_run.json").read_text(encoding="utf-8"))["arms"]
    exports = {x: json.loads((res / f"s2y_export_{'tq40' if x == 'tq404' else x}.json").read_text(encoding="utf-8")) for x in every}
    arm_ok = {x: e["all_exact"] and not e["other_tensors_differ"] and run[x]["model_sha256"] == e["q4_sha256"]
              and all(run[x]["panels"].get(p, {}).get("status") == "COMPLETE" for p in run[x]["panel_set"]) for x, e in exports.items()}
    logs = {x: json.loads((res / f"s2y_train_{x}.json").read_text(encoding="utf-8"))["layers"] for x in ("as_both", "st_both2")}
    train_ok = all(len(v) == 20 and all(r["trained"] < r["posthoc"] for r in v.values()) for v in logs.values())
    tool_ok = json.loads((res / "s2y_tool_check.json").read_text(encoding="utf-8"))["equal"]
    ks_rec = json.loads((res / "s2y_ks.json").read_text(encoding="utf-8"))
    valid = {k: tool_ok and all(arm_ok[x] for x in v) for k, v in SECTIONS.items()}
    valid["grid"] = valid["grid"] and train_ok
    valid["iq2_ks"] = valid["iq2_ks"] and ks_rec["control"]

    d = doc_top1(a.work, "as_both") - doc_top1(a.work, "sb_q40")
    rd = np.random.default_rng(SEED)
    tci = ci([d[rd.integers(0, len(d), len(d))].mean() for _ in range(DRAWS)])
    margin, mci, gain = delta["as_both"] - delta["sb_q40"], ci(boot["as_both"] - boot["sb_q40"]), t1["as_both"] - t1["sb_q40"]
    deg = {x: degenerate_count(a.work / x) for x in names}
    passes = {x: int(arms[x]["chat"].sum()) for x in names}
    sci, eci = ci(boot["stk4"]), ci(boot["stk4"] - boot["bal4"] - boot["st_attn"])
    chats = {x: chat(a.work, x) for x in (*GROUPS, *OUT, "ff_bf", "ks_org", "sb_bf", "st_ffn")}
    top1 = {x: (t1[x] - t1["st_ffn"]) / (1 - t1["st_ffn"]) for x in GROUPS}
    cr = {x: (chats[x]["passes"] - chats["st_ffn"]["passes"]) / (passes["ref"] - chats["st_ffn"]["passes"]) for x in GROUPS}
    out_rec = {x: (t1[x] - t1["ff_bf"]) / (1 - t1["ff_bf"]) for x in OUT}
    best = max(("sb_bf", "as_both"), key=lambda x: t1[x])
    best_chat = passes["as_both"] if best == "as_both" else chats["sb_bf"]["passes"]
    bci = ci(boot["bal4"] - boot["tq404"])
    out = {"sections_valid": valid, "noise": {"gap": gap, "top1_gap": top1_gap, "paired_ci95": ci(boot["st_both2"] - boot["st_both"])},
           "delta": delta, "ci95": {x: ci(boot[x]) for x in boot}, "panels": {x: {p: 100.0 * float(arms[x][p].mean()) for p in PANELS} for x in names},
           "degenerate": deg, "chat_passes": passes,
           "container": {"st_both_q40_minus_tq2": delta["sb_q40"] - delta["st_both"], "ci95": ci(boot["sb_q40"] - boot["st_both"]), "tern_q40_minus_tq2_k4": delta["tq404"] - delta["t4"],
                         "ci95_tern": ci(boot["tq404"] - boot["t4"]), "tern_q40_minus_tq2_k8": delta["tq40"] - delta["tern"]},
           "bal": {"margin": delta["bal4"] - delta["tq404"], "ci95": bci, "label": bal(bci) if valid["bal"] else "VOID", "k8": delta["bal"] - delta["tq40"], "k8_ci95": ci(boot["bal"] - boot["tq40"])},
           "grid": {"margin": margin, "ci95": mci, "top1_gain": gain, "top1_ci95": tci, "label": grid(margin, mci, gap, gain, tci, top1_gap) if valid["grid"] else "VOID",
                    "against_st_both": delta["as_both"] - delta["st_both"]},
           "stack": {"delta": delta["stk4"], "ci95": sci, "label": label(sci) if valid["stack"] else "VOID", "chat_ok": passes["stk4"] >= passes["ref"] - CHAT_MARGIN and deg["stk4"] <= deg["ref"] + DEGEN_MARGIN,
                     "sum_of_parts": delta["bal4"] + delta["st_attn"], "excess": delta["stk4"] - delta["bal4"] - delta["st_attn"], "excess_ci95": eci, "excess_label": excess(eci)},
           "st_ffn": {"top1_recovered": top1, "chat_recovered": cr, "groups": groups(top1, cr) if valid["st_ffn"] else "VOID", "from_ff_bf": out_rec,
                      "outliers": outliers(out_rec["ff_keepout"], [out_rec["ff_keeprnd"], out_rec["ff_keeprnd2"]], chats["ff_keepout"]["passes"],
                                           [chats["ff_keeprnd"]["passes"], chats["ff_keeprnd2"]["passes"]]) if valid["st_ffn"] else "VOID", "chat": chats},
           "iq2_ks": {"against": best, "label": ks(t1["ks_org"], docs["ks_org"]["kl_top32_nats"], chats["ks_org"]["passes"], t1[best], docs[best]["kl_top32_nats"], best_chat, top1_gap) if valid["iq2_ks"] else "VOID",
                      "bits": 2.1875, "imatrix": {k: v for k, v in ks_rec.items() if k != "tensors"}},
           "docs": docs}
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "docs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
