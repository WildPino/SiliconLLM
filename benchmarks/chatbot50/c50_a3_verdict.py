"""C50-A3 verdict: engine vs llama.cpp R3 under the tolerance derived from the A2b floor.

    top-1 disagreement   <= max(2 * floor_top1, 0.002)
    mean KL(R3 || engine) <= max(2 * floor_KL, 1e-4) nats
    |dBPB| (point)        <= max(2 * floor_dBPB, 0.002), floor_dBPB = largest |dBPB| point estimate
                             over the three A2b floor pairs (A3 addendum 2; the registered CI form
                             failed llama.cpp against itself).  The 95% CI is reported, not gated.

Usage:
    python -I benchmarks/chatbot50/c50_a3_verdict.py <a3_run.json> <a2b_floor.json> [--json verdict.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("floor")
    ap.add_argument("--json")
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    floor = json.loads(Path(args.floor).read_text(encoding="utf-8"))
    out = {"run": args.run, "floor": args.floor}
    if run.get("status") != "COMPLETE" or not isinstance(floor.get("floor"), dict):
        out["verdict"] = f"VOID (run {run.get('status')}, floor {floor.get('floor')})"
        print(out["verdict"])
    else:
        tol = floor["floor"]["a3_tolerance"]
        p = run["pairs"]["R3-ENGINE"]
        floor_dbpb = max(abs(floor["pairs"][k]["dbpb"]) for k in ("R0-R1", "R0-R2", "R0-R3"))
        dbpb_limit = max(2 * floor_dbpb, 0.002)
        out["floor_dbpb"], out["dbpb_ci95_reported"] = floor_dbpb, p["dbpb_ci95"]
        checks = {
            "top1": (p["top1_disagreement"], tol["top1_disagreement"], p["top1_disagreement"] <= tol["top1_disagreement"]),
            "mean_kl": (p["mean_kl_nats"], tol["mean_kl_nats"], p["mean_kl_nats"] <= tol["mean_kl_nats"]),
            "abs_dbpb": (p["dbpb"], dbpb_limit, abs(p["dbpb"]) <= dbpb_limit),
        }
        print(f"(reported) dBPB 95% CI {p['dbpb_ci95']}")
        for k, (v, t, ok) in checks.items():
            print(f"{k}: {v} vs {t}  {'PASS' if ok else 'FAIL'}")
        r0 = run["pairs"].get("R0-ENGINE", {})
        print(f"(reported) R0 vs engine: top1 {r0.get('top1_disagreement')}, KL {r0.get('mean_kl_nats')}, dBPB {r0.get('dbpb')}")
        out["checks"] = {k: {"value": v, "limit": t, "pass": ok} for k, (v, t, ok) in checks.items()}
        out["verdict"] = "PASS" if all(c[2] for c in checks.values()) else "FAIL"
        print("VERDICT", out["verdict"])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
