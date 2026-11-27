"""S2N amendment 2's table of parts: closed-book recall (nq, tqa) and the other panels, per arm, paired against ref (numbers only).

Usage: python -I benchmarks/stage2/s2n_recall_parts.py --scores-dir E:/SiliconLLM_Artifacts/stage1/s1a/scores --json benchmarks/stage2/results/s2n_recall_parts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1"))
from s1_verdict import DRAWS, PANELS, SEED, points, vec  # noqa: E402

ARMS = ("top4dm16", "bal4", "ks_org", "ks_org4", "ks_stk4", "glob", "e2e_rowx")
RECALL, REST = ("nq", "tqa"), ("gsm8k", "arc", "mmlu", "chat")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    arms = {x: {p: vec(json.loads((a.scores_dir / f"{x}.json").read_text(encoding="utf-8")), p) for p in PANELS} for x in ("ref",) + ARMS}
    n = {p: len(arms["ref"][p]) for p in PANELS}

    def sub(v, idx, ps):
        return float(np.mean([100.0 * (v[p] if idx is None else v[p][idx[p]]).mean() for p in ps]))
    rng = np.random.default_rng(SEED)
    boot = {x: {"all": [], "recall": [], "rest": [], "nq": [], "tqa": []} for x in ARMS}
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        r = {"all": points(arms["ref"], idx), "recall": sub(arms["ref"], idx, RECALL), "rest": sub(arms["ref"], idx, REST),
             "nq": sub(arms["ref"], idx, ("nq",)), "tqa": sub(arms["ref"], idx, ("tqa",))}
        for x in ARMS:
            v = {"all": points(arms[x], idx), "recall": sub(arms[x], idx, RECALL), "rest": sub(arms[x], idx, REST),
                 "nq": sub(arms[x], idx, ("nq",)), "tqa": sub(arms[x], idx, ("tqa",))}
            for k in v:
                boot[x][k].append(v[k] - r[k])
    sets = {"all": PANELS, "recall": RECALL, "rest": REST, "nq": ("nq",), "tqa": ("tqa",)}
    out = {"ref": {p: 100.0 * float(arms["ref"][p].mean()) for p in PANELS}, "arms": {}}
    for x in ARMS:
        out["arms"][x] = {"panels": {p: 100.0 * float(arms[x][p].mean()) for p in PANELS},
                          **{k: {"delta": sub(arms[x], None, ps) - sub(arms["ref"], None, ps),
                                 "ci95": [float(np.percentile(boot[x][k], 2.5)), float(np.percentile(boot[x][k], 97.5))]} for k, ps in sets.items()}}
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for x in ARMS:
        o = out["arms"][x]
        print(f"{x:9s}", " ".join(f"{k} {o[k]['delta']:+6.2f} [{o[k]['ci95'][0]:+6.2f},{o[k]['ci95'][1]:+6.2f}]" for k in sets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
