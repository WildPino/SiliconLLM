"""C50-A1b verdict: timing stability across fresh processes (it chooses nothing beyond the registered order).

Per arm: the 10 admissible per-process median tok/s (bench at context 2048, 128 steps).  Gate (ROADMAP §5 A1b):
max/min of the 10 medians <= 1.06.  A1b PASSES iff some arm passes; the adopted arm is the first passing one in the
registered order B0, B1, B2.  Reported, not gated: mean and min..max of the medians, the per-process within-run
spread (max_step / min_step), and the mean foreign occupancy.

Usage:
    python -I benchmarks/chatbot50/c50_a1b_verdict.py <run.json> --json <verdict.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GATE, ORDER = 1.06, ("B0", "B1", "B2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    out = {"cell": "C50-A1b verdict", "run_status": run.get("verdict_input"), "gate_max_over_min": GATE, "arms": {}}
    if run.get("verdict_input") != "COMPLETE":
        out["verdict"] = "VOID"
        print("VOID:", run.get("verdict_input"))
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 1
    for arm in ORDER:
        ok = [p for p in run["processes"] if p["arm"] == arm and p["status"] == "OK"]
        if not ok:
            continue
        med = [p["median_tok_s"] for p in ok]
        ratio = max(med) / min(med)
        out["arms"][arm] = {"processes": len(ok), "medians_tok_s": med, "mean": sum(med) / len(med),
                            "min": min(med), "max": max(med), "max_over_min": ratio, "pass": len(ok) == 10 and ratio <= GATE,
                            "within_process_max_over_min_step": [p["max_step_s"] / p["min_step_s"] for p in ok],
                            "foreign_pct_mean": sum(p["foreign_pct"] for p in ok) / len(ok),
                            "replaced": sum(1 for p in run["processes"] if p["arm"] == arm and p["status"] == "VOID")}
        a = out["arms"][arm]
        print(f"{arm}: medians {a['min']:.3f}..{a['max']:.3f} tok/s (mean {a['mean']:.3f}) max/min {ratio:.4f} "
              f"{'PASS' if a['pass'] else 'FAIL'}  foreign {a['foreign_pct_mean']:.2f}%  replaced {a['replaced']}")
    passing = [a for a in ORDER if out["arms"].get(a, {}).get("pass")]
    out["adopted_arm"] = passing[0] if passing else None
    out["verdict"] = "PASS" if passing else "FAIL"
    print("VERDICT", out["verdict"], "adopted", out["adopted_arm"])
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
