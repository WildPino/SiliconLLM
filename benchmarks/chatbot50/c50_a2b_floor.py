"""C50-A2b floor and the derived A3 tolerance, from a c50_a2b_floor_run.py output.

Definitions fixed in the protocol before the run:
    floor_top1 = max over pairs (R0,R1), (R0,R2), (R0,R3) of the top-1 disagreement point estimate
    floor_KL   = max over the same pairs of the mean KL(R0 || Rk) point estimate
    A3 tolerance: top-1 <= max(2 * floor_top1, 0.002); mean KL <= max(2 * floor_KL, 1e-4 nats);
                  |dBPB| upper CI <= 0.002 (fixed by the roadmap, not derived)
The repeat R0b must be bit-identical to R0 (negative control); otherwise the floor is VOID.

Usage:
    python -I benchmarks/chatbot50/c50_a2b_floor.py <run.json> [--json floor.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FLOOR_PAIRS = ["R0-R1", "R0-R2", "R0-R3"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json")
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    pairs = run["pairs"]
    out = {"run": args.run, "pairs": {}}
    neg = pairs.get("R0-R0b", {})
    out["negative_control_bit_identical"] = bool(neg.get("bit_identical_logp_target")) and neg.get("top1_disagreement") == 0.0
    missing = [p for p in FLOOR_PAIRS if "top1_disagreement" not in pairs.get(p, {})]
    for p in FLOOR_PAIRS:
        if p in pairs and "top1_disagreement" in pairs[p]:
            q = pairs[p]
            out["pairs"][p] = {k: q.get(k) for k in ("top1_disagreement", "top1_disagreement_ci95", "mean_kl_nats",
                                                     "mean_kl_ci95", "p99_kl_nats", "max_kl_nats", "dbpb", "dbpb_ci95",
                                                     "bit_identical_logp_target", "positions", "documents")}
            print(f"{p}: top1 {q['top1_disagreement']:.5f} {tuple(round(x, 5) for x in q['top1_disagreement_ci95'])}  "
                  f"KL {q.get('mean_kl_nats', float('nan')):.3g} {tuple(round(x, 6) for x in q.get('mean_kl_ci95', ()))}  "
                  f"dBPB {q['dbpb']:+.5f} {tuple(round(x, 5) for x in q['dbpb_ci95'])}")
    print("negative control R0 vs R0b bit-identical:", out["negative_control_bit_identical"])
    if missing or not out["negative_control_bit_identical"]:
        out["floor"] = f"VOID (missing pairs {missing}, negative control {out['negative_control_bit_identical']})"
        print(out["floor"])
    else:
        ft = max(out["pairs"][p]["top1_disagreement"] for p in FLOOR_PAIRS)
        fk = max(out["pairs"][p]["mean_kl_nats"] for p in FLOOR_PAIRS)
        out["floor"] = {"floor_top1": ft, "floor_kl_nats": fk,
                        "a3_tolerance": {"top1_disagreement": max(2 * ft, 0.002), "mean_kl_nats": max(2 * fk, 1e-4),
                                         "abs_dbpb_upper_ci": 0.002}}
        print(f"floor_top1 {ft:.5f}  floor_KL {fk:.3g}  ->  A3 tolerance top1 <= {max(2 * ft, 0.002):.5f}, "
              f"mean KL <= {max(2 * fk, 1e-4):.3g}, |dBPB| upper CI <= 0.002")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
