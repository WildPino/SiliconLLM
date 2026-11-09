"""S2M amendment 1's verdict: ks_org at k = 4, as bal4.  It chooses nothing.  Judged on points and chat.

ks_org4:  stage 1's label and chat gate.  Organs at k = 4: margin = ks_org4 - ks_org, paired CI95:
          THE ORGANS LOSE AT k = 4 (upper < 0), THE ORGANS GAIN AT k = 4 (lower > 0), k = 4 DOES NOT MOVE THE ORGANS otherwise.
Excess at k = 4: ks_stk4 - bal4 - ks_org4, CI95: IT ADDS UP / WORSE THAN ITS PARTS (upper < 0) / BETTER THAN ITS PARTS (lower > 0).
          Reported beside it: the same plus top4dm16 (P0 at k = 4 in bal4's form, stage 1), which counts k = 4 once.
VOID: ks_org4's panels complete on ks_org's file, and its copied score panels equal ks_org's byte for byte.

Usage: python -I benchmarks/stage2/s2z_k4_verdict.py --selftest
       python -I benchmarks/stage2/s2z_k4_verdict.py --scores-dir D --work W --results DIR --json F
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
sys.path.insert(0, str(HERE))
from s1_verdict import CHAT_MARGIN, DEGEN_MARGIN, DRAWS, PANELS, SEED, degenerate_count, label, points, vec  # noqa: E402
from s2z_verdict import excess  # noqa: E402


def organs(ci: list[float]) -> str:
    return "THE ORGANS LOSE AT k = 4" if ci[1] < 0 else "THE ORGANS GAIN AT k = 4" if ci[0] > 0 else "k = 4 DOES NOT MOVE THE ORGANS"


def selftest() -> int:
    cases = [(organs([-3, -0.5]), "THE ORGANS LOSE AT k = 4"), (organs([-2, 1]), "k = 4 DOES NOT MOVE THE ORGANS"), (organs([0.5, 2]), "THE ORGANS GAIN AT k = 4"),
             (excess([-3, -1]), "WORSE THAN ITS PARTS"), (excess([-2, 1]), "IT ADDS UP")]
    ok = all(a == b for a, b in cases)
    print("selftest", "PASS" if ok else "FAIL", len(cases))
    return 0 if ok else 1


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("scores-dir", "work", "results", "json"):
        ap.add_argument(f"--{x}", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    names = ["ref", "ks_org4", "ks_org", "ks_stk4", "bal4", "top4dm16"]
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
    deg = {x: degenerate_count(a.work / x) for x in ("ref", "ks_org4", "ks_org", "ks_stk4", "bal4")}
    passes = {x: int(arms[x]["chat"].sum()) for x in names}
    run = json.loads((a.results / "s2z_k4_run.json").read_text(encoding="utf-8"))["arms"]["ks_org4"]
    exp = json.loads((a.results / "s2y_export_ks_org.json").read_text(encoding="utf-8"))
    copied = sorted(x.name for p in ("arc", "mmlu", "docs") for x in (a.work / "ks_org").glob(f"{p}.*"))
    same = {x: (a.work / "ks_org4" / x).exists() and digest(a.work / "ks_org4" / x) == digest(a.work / "ks_org" / x) for x in copied}
    valid = (run["model_sha256"] == exp["q4_sha256"] and all(run["panels"].get(p, {}).get("status") == "COMPLETE" for p in run["panel_set"])
             and len(copied) == 7 and all(same.values()))
    oci = ci(boot["ks_org4"] - boot["ks_org"])
    eci = ci(boot["ks_stk4"] - boot["bal4"] - boot["ks_org4"])
    e1ci = ci(boot["ks_stk4"] - boot["bal4"] - boot["ks_org4"] + boot["top4dm16"])
    e = delta["ks_stk4"] - delta["bal4"] - delta["ks_org4"]
    out = {"valid": valid, "copied_equal": same, "delta": delta, "ci95": {x: ci(boot[x]) for x in boot}, "chat_passes": passes, "degenerate": deg,
           "panels": {x: {p: 100.0 * float(arms[x][p].mean()) for p in PANELS} for x in names},
           "ks_org4": {"label": label(ci(boot["ks_org4"])) if valid else "VOID",
                       "chat_ok": passes["ks_org4"] >= passes["ref"] - CHAT_MARGIN and deg["ks_org4"] <= deg["ref"] + DEGEN_MARGIN,
                       "over_ks_org": delta["ks_org4"] - delta["ks_org"], "over_ks_org_ci95": oci, "organs_at_k4": organs(oci) if valid else "VOID"},
           "excess_k4": {"excess": e, "ci95": eci, "label": excess(eci) if valid else "VOID", "sum_of_parts": delta["bal4"] + delta["ks_org4"],
                         "k4_once": e + delta["top4dm16"], "k4_once_ci95": e1ci, "k4_once_label": excess(e1ci) if valid else "VOID"}}
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("panels", "ci95", "copied_equal")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
