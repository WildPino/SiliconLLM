"""Stage 2 test 3b verdict, by the rule registered in S2F_EXACT_WHERE_IT_WEIGHS_PROTOCOL_20261104.md.  It chooses nothing: the
judged arm follows from test 2b's verdict, because the head that passes sets the milliseconds left under 24.7 tok/s.

Points, Delta, CI95, labels and the chat gate are stage 1's (s1_verdict.py).  The arm against its control, the same layers
with a random slot: margin = Delta(arm) - Delta(control), and it beats the control iff the CI95 lies above 0.

Usage: python -I benchmarks/stage2/s2x_verdict.py --selftest
       python -I benchmarks/stage2/s2x_verdict.py --scores-dir D --work W --head F --run F --json F
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
from s1_verdict import (CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS, SEED, degenerate_count, docs_metrics,  # noqa: E402
                        label, points, vec)

TERN_DELTA = -8.52
HALF = TERN_DELTA / 2
SLOT_ERR = 1e-3                # the energy-weighted and the mean relative error of a slot read from the context's own file
PLANT_DROP = 0.10              # the planted read's top-1 against tern's
MS_ALL = 3.92                  # one exact slot in every MoE layer of Ling-flash, desk
LAYERS = {"x15": 5, "x9": 11, "x19": 19, "x15k4": 5}


def score(arms: dict, name: str, control: str | None, degenerate: dict) -> dict:
    ref, t = arms["ref"], arms[name]
    n = {p: len(ref[p]) for p in PANELS}
    if {p: len(t[p]) for p in PANELS} != n:
        raise SystemExit(f"{name} does not score the same items as ref")
    rng = np.random.default_rng(SEED)
    boot, marg = [], []
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        boot.append(points(t, idx) - points(ref, idx))
        if control:
            marg.append(points(t, idx) - points(arms[control], idx))
    c = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    chat_ok = (int(t["chat"].sum()) >= int(ref["chat"].sum()) - CHAT_MARGIN
               and degenerate[name] <= degenerate["ref"] + DEGEN_MARGIN)
    lab = label(c)
    delta = points(t) - points(ref)
    rec = {"delta": delta, "ci95": c, "label": lab, "chat_ok": chat_ok, "chat_passes": int(t["chat"].sum()),
           "degenerate": degenerate[name], "fits": lab in ("WITHIN SHARE", "WITHIN BUDGET") and chat_ok,
           "back": delta - TERN_DELTA, "ms": round(MS_ALL * LAYERS.get(name, 0) / 19, 2),
           "panels": {p: 100.0 * float(t[p].mean()) for p in PANELS}}
    if rec["ms"]:
        rec["points_per_ms"] = rec["back"] / rec["ms"]
    if control:
        m = [float(np.percentile(marg, 2.5)), float(np.percentile(marg, 97.5))]
        rec.update(margin=points(t) - points(arms[control]), margin_ci95=m, beats_control=m[0] > 0)
    return rec


def controls_ok(c: dict) -> dict:
    return {"off": c["off_equals_tern"], "replay": c["tern_all_equals_p0_all"],
            "slot_err": c["p0_all_slot_rel_err"] <= SLOT_ERR and c["p0_all_mean_slot_rel_err"] <= SLOT_ERR,
            "plant": c["top1"]["wrong"] <= c["top1"]["tern"] - PLANT_DROP}


def decide(arms: dict, degenerate: dict, untrained: bool, controls: dict) -> dict:
    judged, control = ("x9", "r9") if untrained else ("x15", "r15")
    ok = controls_ok(controls)
    j = score(arms, judged, control, degenerate)
    # Amendment 1: when x15 is judged, x9 and r9 run as reported arms; they change no outcome.
    out = {"judged": judged, "control": control, "controls": ok, "arm": j,
           "control_arm": score(arms, control, None, degenerate),
           "reported": {a: score(arms, a, "r9" if a == "x9" and "r9" in arms else None, degenerate)
                        for a in ("x9", "r9", "x19", "x15k4") if a in arms and a not in (judged, control)}}
    if not all(ok.values()):
        out["outcome"] = "VOID"
    elif j["fits"] and j["beats_control"]:
        out["outcome"] = "EXACT WHERE IT WEIGHS CLOSES THE EXPERTS"
    elif j["delta"] >= HALF and j["beats_control"]:
        out["outcome"] = "HALF COMES BACK"
    else:
        out["outcome"] = "LITTLE COMES BACK"
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
    good = {"off_equals_tern": True, "tern_all_equals_p0_all": True, "p0_all_slot_rel_err": 1e-4, "p0_all_mean_slot_rel_err": 1e-5,
            "top1": {"tern": 0.75, "wrong": 0.4}}
    deg = {"ref": 0, "x15": 0, "r15": 0, "x9": 0, "r9": 0}
    cases = [
        ("closes", dict(ref=ref, x15=lose(1.0, 4, 2), r15=lose(8.0, 5, 6)), False, deg, good, "EXACT WHERE IT WEIGHS CLOSES THE EXPERTS"),
        ("half", dict(ref=ref, x15=lose(2.5, 4, 5), r15=lose(9.0, 5, 8)), False, deg, good, "HALF COMES BACK"),
        ("little", dict(ref=ref, x15=lose(7.0, 4, 6), r15=lose(8.0, 5, 8)), False, deg, good, "LITTLE COMES BACK"),
        ("control as good", dict(ref=ref, x15=lose(1.0, 4, 2), r15=lose(1.0, 5, 2)), False, deg, good, "LITTLE COMES BACK"),
        ("chat fails: half at best", dict(ref=ref, x15=lose(1.0, 4, 8), r15=lose(9.0, 5, 8)), False, deg, good, "HALF COMES BACK"),
        ("the other cell", dict(ref=ref, x9=lose(1.0, 4, 2), r9=lose(8.0, 5, 6)), True, deg, good, "EXACT WHERE IT WEIGHS CLOSES THE EXPERTS"),
        ("reported x9 changes nothing", dict(ref=ref, x15=lose(7.0, 4, 6), r15=lose(8.0, 5, 8), x9=lose(1.0, 6, 2), r9=lose(8.0, 7, 6)), False, deg, good, "LITTLE COMES BACK"),
        ("off differs", dict(ref=ref, x15=lose(1.0, 4, 2), r15=lose(8.0, 5, 6)), False, deg, dict(good, off_equals_tern=False), "VOID"),
        ("replay differs", dict(ref=ref, x15=lose(1.0, 4, 2), r15=lose(8.0, 5, 6)), False, deg, dict(good, tern_all_equals_p0_all=False), "VOID"),
        ("slot error", dict(ref=ref, x15=lose(1.0, 4, 2), r15=lose(8.0, 5, 6)), False, deg, dict(good, p0_all_mean_slot_rel_err=0.01), "VOID"),
        ("plant silent", dict(ref=ref, x15=lose(1.0, 4, 2), r15=lose(8.0, 5, 6)), False, deg, dict(good, top1={"tern": 0.75, "wrong": 0.7}), "VOID"),
    ]
    for name, arms, unt, dg, c, want in cases:
        got = decide(arms, dg, unt, c)["outcome"]
        if got != want:
            print("FAIL", name, got, "!=", want)
            return 1
    print(f"selftest PASS, {len(cases)} cases")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("scores-dir", "work", "head", "run", "json"):
        ap.add_argument(f"--{x}", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    head = json.loads(a.head.read_text(encoding="utf-8"))
    if not head["outcome"].startswith("PASS"):
        raise SystemExit("test 2b did not pass: no milliseconds to spend")
    untrained = head["second_label"] == "THE UNTRAINED PREVIEW PASSES TOO"
    run = json.loads(a.run.read_text(encoding="utf-8"))
    names = [x for x in run["arms"] if all(p.get("status") == "COMPLETE" for p in run["arms"][x]["panels"].values())
             and len(run["arms"][x]["panels"]) == 7]
    arms = {x: {p: vec(json.loads((a.scores_dir / f"{x}.json").read_text(encoding="utf-8")), p) for p in PANELS}
            for x in ["ref"] + names}
    deg = {x: degenerate_count(a.work / x) for x in arms}
    out = decide(arms, deg, untrained, run["controls"])
    out["docs"] = {x: docs_metrics(a.work, x) for x in names}
    print(json.dumps(out, indent=1))
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
