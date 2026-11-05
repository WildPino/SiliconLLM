"""The fourth code, step 2: the verdict by the registered rule (S2K).  Routed experts in bal against ternary, same training, all 19 layers.

Judged at k = 4, the ceiling's routing: margin = points(bal4) - points(t4), paired CI95 over items.
    THE FOURTH CODE PAYS AFTER TRAINING    CI95 lower > 0
    IT IS WORSE AFTER TRAINING             CI95 upper < 0
    NO DIFFERENCE AFTER TRAINING           otherwise
k = 8 (bal against tern), the block loss, top-1 and dBPB are reported beside it.
VOID: the export is not exact, or training did not lower the block loss on every layer against its post-hoc start.

Usage: python -I benchmarks/stage2/s2v_verdict.py --scores-dir D --work W --train F --tern-train F --export F --json F
       python -I benchmarks/stage2/s2v_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
from s1_verdict import DRAWS, PANELS, SEED, degenerate_count, docs_metrics, points, vec  # noqa: E402


def margin(arms: dict, a: str, b: str) -> dict:
    n = {p: len(arms["ref"][p]) for p in PANELS}
    rng = np.random.default_rng(SEED)
    m = []
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        m.append(points(arms[a], idx) - points(arms[b], idx))
    return {"margin": points(arms[a]) - points(arms[b]), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def decide(k4: dict, exact: bool, trained_all_layers: bool) -> str:
    if not (exact and trained_all_layers):
        return "VOID"
    if k4["ci95"][0] > 0:
        return "THE FOURTH CODE PAYS AFTER TRAINING"
    if k4["ci95"][1] < 0:
        return "IT IS WORSE AFTER TRAINING"
    return "NO DIFFERENCE AFTER TRAINING"


def selftest() -> int:
    cases = [(({"ci95": [0.2, 1.0]}, True, True), "THE FOURTH CODE PAYS AFTER TRAINING"), (({"ci95": [-2, -0.1]}, True, True), "IT IS WORSE AFTER TRAINING"),
             (({"ci95": [-1, 1]}, True, True), "NO DIFFERENCE AFTER TRAINING"), (({"ci95": [1, 2]}, False, True), "VOID"), (({"ci95": [1, 2]}, True, False), "VOID")]
    ok = all(decide(*a) == w for a, w in cases)
    print("selftest", "PASS" if ok else "FAIL", len(cases))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("scores-dir", "work", "train", "tern-train", "export", "json"):
        ap.add_argument(f"--{x}", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    names = ["ref", "tern", "t4", "bal", "bal4"]
    arms = {x: {p: vec(json.loads((a.scores_dir / f"{x}.json").read_text(encoding="utf-8")), p) for p in PANELS} for x in names}
    tr = json.loads(a.train.read_text(encoding="utf-8"))["layers"]
    tt = json.loads(a.tern_train.read_text(encoding="utf-8"))["layers"]
    ex = json.loads(a.export.read_text(encoding="utf-8"))["bal_exact"]
    if sorted(tr) != sorted(tt):
        raise SystemExit("the two trainings do not cover the same layers")
    ratio = [tr[i]["rel_form_trained"] / tt[i]["rel_form_trained"] for i in tt]
    k4, k8 = margin(arms, "bal4", "t4"), margin(arms, "bal", "tern")
    moved = all(tr[i]["rel_form_trained"] < tr[i]["rel_form_orig"] for i in tr)
    ref = points(arms["ref"])
    out = {"outcome": decide(k4, ex["exact_to_bf16"], moved), "k4": k4, "k8": k8, "export": ex, "trained_every_layer": moved,
           "delta": {x: points(arms[x]) - ref for x in names[1:]}, "panels": {x: {p: 100.0 * float(arms[x][p].mean()) for p in PANELS} for x in names[1:]},
           "degenerate": {x: degenerate_count(a.work / x) for x in names[1:]}, "block_loss_ratio_mean": float(np.mean(ratio)), "block_loss_ratio_by_layer": ratio,
           "block_loss": {"bal": float(np.mean([tr[i]["rel_form_trained"] for i in tr])), "tern": float(np.mean([tt[i]["rel_form_trained"] for i in tt]))},
           "outlier_groups": sum(sum(tr[i]["outlier_groups"].values()) for i in tr), "seconds": sum(tr[i]["seconds"] for i in tr),
           "docs": {x: docs_metrics(a.work, x) for x in ("tern", "bal")}}
    print(json.dumps(out, indent=1))
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
