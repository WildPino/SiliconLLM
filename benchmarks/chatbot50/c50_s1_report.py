"""C50-S1 report: the registered equivalence gate per router format, and the reported statistics.  It chooses nothing.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_S1_CAPACITY_INVARIANCE_PROTOCOL_20261018.md.
    rate of an arm        mean of its OK processes' median tok/s
    c_f                   mean over the --organs m1 processes of format f of router seconds / router bytes per step
    dR_f(m)               (E - 256) * 2048 * 19 * (4 bytes F32 | 2 bytes BF16) per token
    v_pred_f(m)           1 / (1 / v_f(1) + dR_f(m) * c_f)
    rho_f(m)              v_E,f(m) / v_pred_f(m); CI95 from 10,000 bootstrap draws (seed 20261016) resampling each arm's
                          process medians and the m1 organ processes independently
    PASS in format f      iff the CI95 of rho_f lies inside [0.97, 1.03] at m2 and at m4 (tau = 3%)
Reported: raw ratio v_E/v(1); BF16 expansions against the F32 T0 (includes the router upgrade); pool effect v_H/v(1);
router cost per byte from E - H; the router's lap at each m; m6; the selection spread; arm M.

Usage:
    python -I benchmarks/chatbot50/c50_s1_report.py benchmarks/chatbot50/results/c50_s1_run_1.json --json report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPERTS = {"m1": 256, "m2": 536, "m4": 1088, "m6": 1600}
BYTES = {"f32": 4, "bf16": 2}
TAU, DRAWS, SEED = 0.03, 10000, 20261016
GATED_M = ("m2", "m4")


def d_router(m: str, f: str) -> float:
    return (EXPERTS[m] - 256) * 2048 * 19 * BYTES[f]


def ci(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    out = {"cell": "C50-S1 report", "run": args.run, "verdict_input": run.get("verdict_input"),
           "known_positives": run.get("known_positives"), "void_arms": run.get("void_arms", [])}
    if run.get("verdict_input") != "COMPLETE":
        out["verdict"] = "VOID"
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("VOID:", run.get("verdict_input"))
        return 1
    ok = [p for p in run["processes"] if p["status"] == "OK"]
    med = defaultdict(list)
    for p in ok:
        med[p["arm"]].append(p["median_tok_s"])
    med = {k: np.array(v) for k, v in med.items()}
    rng = np.random.default_rng(SEED)

    def boot(arm: str) -> np.ndarray:
        v = med[arm]
        return v[rng.integers(0, len(v), size=(DRAWS, len(v)))].mean(axis=1)

    out["arms"] = {}
    for p_arm, v in sorted(med.items()):
        procs = [p for p in ok if p["arm"] == p_arm]
        out["arms"][p_arm] = {"n": int(len(v)), "rate": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                              "max_over_min": float(v.max() / v.min()), "cv": float(v.std(ddof=1) / v.mean()) if len(v) > 1 else None,
                              "sel_distinct_frac": float(np.mean([p["sel_distinct_frac"] for p in procs])),
                              "sel_copy0_frac": float(np.mean([p["sel_copy0_frac"] for p in procs])),
                              "router_bytes_engine": float(np.mean([p["organ_bytes"]["router"] for p in procs])),
                              "fill_s": float(np.mean([p["fill_s"] for p in procs])),
                              "foreign_pct_max": float(max(p["foreign_pct"] for p in procs))}
        if procs and procs[0].get("organs"):
            out["arms"][p_arm]["router_lap_s"] = float(np.mean([p["organ_mean_s"]["router"] for p in procs]))
    out["replaced"] = sum(1 for p in run["processes"] if p["status"] == "VOID")
    out["formats"] = {}
    for f in ("f32", "bf16"):
        e1 = f"m1/E-{f}/organs=0"
        if e1 not in med:
            out["formats"][f] = {"status": "VOID (known positive K1 failed)" if f == "bf16" else "VOID"}
            continue
        org = [p for p in ok if p["arm"] == f"m1/E-{f}/organs=1"]
        per_byte = np.array([p["organ_mean_s"]["router"] / p["organ_bytes"]["router"] for p in org])
        c_draw = per_byte[rng.integers(0, len(per_byte), size=(DRAWS, len(per_byte)))].mean(axis=1)
        v1 = boot(e1)
        fr = {"c_s_per_byte": float(per_byte.mean()), "c_ci95": ci(c_draw), "v1": float(med[e1].mean()), "by_m": {}}
        for m in ("m2", "m4", "m6"):
            ea, ha = f"{m}/E-{f}/organs=0", f"{m}/H-{f}/organs=0"
            if ea not in med:
                continue
            dr = d_router(m, f)
            ve = boot(ea)
            rho = ve * (1.0 / v1 + dr * c_draw)
            row = {"dR_bytes": dr, "v_E": float(med[ea].mean()),
                   "v_pred": float(1.0 / (1.0 / med[e1].mean() + dr * per_byte.mean())),
                   "rho": float(med[ea].mean() * (1.0 / med[e1].mean() + dr * per_byte.mean())), "rho_ci95": ci(rho),
                   "raw_ratio": float(med[ea].mean() / med[e1].mean()), "raw_ratio_ci95": ci(ve / v1)}
            row["rho_inside_tau"] = bool(1 - TAU <= row["rho_ci95"][0] and row["rho_ci95"][1] <= 1 + TAU)
            if f == "bf16" and "m1/E-f32/organs=0" in med:
                row["against_f32_T0_includes_router_upgrade"] = float(med[ea].mean() / med["m1/E-f32/organs=0"].mean())
            if ha in med:
                vh = boot(ha)
                row["pool_effect"] = float(med[ha].mean() / med[e1].mean())
                row["pool_effect_ci95"] = ci(vh / v1)
                eh = (1.0 / ve - 1.0 / vh) / dr
                row["router_s_per_byte_E_minus_H"] = float((1.0 / med[ea].mean() - 1.0 / med[ha].mean()) / dr)
                row["router_s_per_byte_E_minus_H_ci95"] = ci(eh)
            fr["by_m"][m] = row
        gated = [fr["by_m"].get(m, {}).get("rho_inside_tau") for m in GATED_M]
        fr["verdict"] = "PASS" if all(g is True for g in gated) else "FAIL"
        out["formats"][f] = fr
    if run.get("bank"):
        mb = defaultdict(list)
        for r in run["bank"]:
            if "median_token_s" in r:
                mb[(r["gb"], r["k"])].append(r)
        out["bank"] = {f"{gb:g}GB/k={k}": {"n": len(rs), "median_token_ns": float(np.median([r["median_token_s"] for r in rs])) * 1e9,
                                            "ns_per_row": float(np.median([r["ns_per_row"] for r in rs])),
                                            "gb_s": float(np.median([r["gb_s"] for r in rs]))}
                       for (gb, k), rs in sorted(mb.items())}
        for k in sorted({k for _, k in mb}):
            sizes = sorted(gb for gb, kk in mb if kk == k)
            if len(sizes) > 1:
                lo = out["bank"][f"{sizes[0]:g}GB/k={k}"]["median_token_ns"]
                hi = out["bank"][f"{sizes[-1]:g}GB/k={k}"]["median_token_ns"]
                out["bank"][f"largest_over_smallest_k={k}"] = hi / lo
    verdicts = {f: out["formats"][f].get("verdict", out["formats"][f].get("status")) for f in out["formats"]}
    out["verdict"] = verdicts
    for f, fr in out["formats"].items():
        for m, row in fr.get("by_m", {}).items():
            print(f"{f} {m}: v_E {row['v_E']:.3f} pred {row['v_pred']:.3f} rho {row['rho']:.4f} "
                  f"CI ({row['rho_ci95'][0]:.4f}, {row['rho_ci95'][1]:.4f}) raw {row['raw_ratio']:.4f}"
                  + (f" pool {row['pool_effect']:.4f}" if "pool_effect" in row else ""))
    print("VERDICT", json.dumps(verdicts))
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
