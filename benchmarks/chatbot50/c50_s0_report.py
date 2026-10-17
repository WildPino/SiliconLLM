"""C50-S0 report: v0 of T0 per context, the organ breakdown and its planted control (it chooses nothing).

Per context, from the 10 admissible --organs-off processes (one median tok/s each, 128 steps):
    v0 = mean of the medians; CI95 = mean +- t(0.975, n-1) * sd / sqrt(n); dispersion = max/min and CV.
    desk = 1 / (bytes / 32 GB/s + 2 ms), bytes = the engine's own per-organ byte count (registered desk model).
    sequential fill (the engine has no batched prefill): ctx / fill_s; TTFT of a ctx-token prompt = fill_s.
    RAM: the largest peak working set and peak commit.
Organ breakdown, from the --organs processes: mean ms per organ, its share, bytes and GB/s; the instrumentation
overhead (mean step with organs / without, same context, same rounds) and the lap closure (sum of organs / step).
Planted control (gates the breakdown only, never v0): over 3 pairs at context 512, the median head-time ratio
plant/base >= 1.7 and the median ratio of all other organs' time inside [0.90, 1.12].
The registered predictions are scored, not gated.  The v0 label is the A1b label (STABLE or UNSTABLE).

Usage:
    python -I benchmarks/chatbot50/c50_s0_report.py <run.json> --json <report.json>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ORGANS = ("attention", "kv", "dense", "router", "routed", "shared", "head", "other")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}
DESK_BW, DESK_FIXED_S = 32e9, 0.002
PLANT_HEAD_MIN, PLANT_REST = 1.7, (0.90, 1.12)
PREDICTION = {"v0_2048_over_desk": (0.70, 0.90), "v0_2048_tok_s": (24.0, 31.0), "v0_512_over_v0_2048": (1.05, 1.20)}


def stats(v: list[float]) -> dict:
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    h = T975[n - 1] * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "ci95": [m - h, m + h], "min": min(v), "max": max(v),
            "max_over_min": max(v) / min(v), "cv": sd / m}


def median(v: list[float]) -> float:
    s = sorted(v)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    out = {"cell": "C50-S0 report", "run_status": run.get("verdict_input"), "a1b": run.get("a1b"),
           "mem": run.get("mem"), "prio": run.get("prio"), "contexts": {}}
    if run.get("verdict_input") != "COMPLETE":
        out["status"] = "VOID"
        print("VOID:", run.get("verdict_input"))
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 1
    ok = [p for p in run["processes"] if p["status"] == "OK"]
    label = run["a1b"]["label"]
    for ctx in sorted({p["ctx"] for p in ok if p["phase"] == "rate"}):
        rate = [p for p in ok if p["phase"] == "rate" and p["ctx"] == ctx]
        org = [p for p in ok if p["phase"] == "organs" and p["ctx"] == ctx]
        s = stats([p["median_tok_s"] for p in rate])
        nbytes = sum(rate[0]["organ_bytes"].values())
        desk = 1.0 / (nbytes / DESK_BW + DESK_FIXED_S)
        c = {"v0_tok_s": s, "label": label, "ms_per_token_at_mean": 1e3 / s["mean"], "bytes_per_token": nbytes,
             "desk_tok_s": desk, "v0_over_desk": s["mean"] / desk, "effective_gb_s": nbytes * s["mean"] / 1e9,
             "rounds": [p["round"] for p in rate],
             "sequential_fill_tok_s": stats([ctx / p["fill_s"] for p in rate]),
             "ttft_s_for_ctx_prompt": stats([p["fill_s"] for p in rate]),
             "peak_working_set_gb": max(p["peak_working_set_bytes"] for p in rate) / 1e9,
             "peak_commit_gb": max(p["peak_commit_bytes"] for p in rate) / 1e9,
             "within_process_max_over_min_step": [p["max_step_s"] / p["min_step_s"] for p in rate],
             "foreign_pct_mean": sum(p["foreign_pct"] for p in rate) / len(rate),
             "replaced": sum(1 for p in run["processes"] if p["phase"] == "rate" and p["ctx"] == ctx and p["status"] == "VOID")}
        if org:
            ms = {o: sum(p["organ_mean_s"][o] for p in org) / len(org) * 1e3 for o in ORGANS}
            tot = sum(ms.values())
            c["organs"] = {o: {"ms": ms[o], "share": ms[o] / tot, "mb": rate[0]["organ_bytes"][o] / 1e6,
                               "gb_s": rate[0]["organ_bytes"][o] / (ms[o] / 1e3) / 1e9 if ms[o] > 0 else None}
                           for o in ORGANS}
            paired = [p for p in rate if p["round"] in {q["round"] for q in org}]
            c["organs_processes"] = len(org)
            c["instrumentation_overhead"] = (sum(p["mean_step_s"] for p in org) / len(org)) / \
                                            (sum(p["mean_step_s"] for p in paired) / len(paired))
            c["lap_closure"] = [sum(p["organ_mean_s"].values()) / p["mean_step_s"] for p in org]
        out["contexts"][str(ctx)] = c
        print(f"ctx {ctx}: v0 {s['mean']:.3f} tok/s CI95 [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] "
              f"min..max {s['min']:.3f}..{s['max']:.3f} max/min {s['max_over_min']:.4f} ({label}); "
              f"DESK-MODEL {desk:.2f} -> ratio {c['v0_over_desk']:.3f}; {nbytes / 1e6:.1f} MB/token")
        if org:
            print("   organs ms: " + ", ".join(f"{o} {c['organs'][o]['ms']:.2f}" for o in ORGANS)
                  + f"; overhead {c['instrumentation_overhead']:.4f}")
    base = [p for p in ok if p["phase"] == "plant" and p["kind"] == "s0"]
    plant = [p for p in ok if p["phase"] == "plant" and p["kind"] == "plant"]
    pairs = [(b, q) for b in base for q in plant if b["round"] == q["round"]]
    head = [q["organ_mean_s"]["head"] / b["organ_mean_s"]["head"] for b, q in pairs]
    rest = [sum(q["organ_mean_s"][o] for o in ORGANS if o != "head") / sum(b["organ_mean_s"][o] for o in ORGANS if o != "head")
            for b, q in pairs]
    pc = {"pairs": len(pairs), "head_ratio": head, "rest_ratio": rest,
          "median_head_ratio": median(head) if head else None, "median_rest_ratio": median(rest) if rest else None}
    pc["pass"] = len(pairs) == 3 and pc["median_head_ratio"] >= PLANT_HEAD_MIN and \
        PLANT_REST[0] <= pc["median_rest_ratio"] <= PLANT_REST[1]
    out["planted_control"] = pc
    out["organ_breakdown_status"] = "VALIDATED" if pc["pass"] else "UNVALIDATED (planted control failed)"
    print(f"planted control: head x{pc['median_head_ratio']:.3f}, rest x{pc['median_rest_ratio']:.3f} -> "
          f"{'PASS' if pc['pass'] else 'FAIL'}")
    cx = out["contexts"]
    if "2048" in cx and "512" in cx:
        got = {"v0_2048_over_desk": cx["2048"]["v0_over_desk"], "v0_2048_tok_s": cx["2048"]["v0_tok_s"]["mean"],
               "v0_512_over_v0_2048": cx["512"]["v0_tok_s"]["mean"] / cx["2048"]["v0_tok_s"]["mean"]}
        out["predictions"] = {k: {"value": got[k], "registered": PREDICTION[k],
                                  "hit": PREDICTION[k][0] <= got[k] <= PREDICTION[k][1]} for k in PREDICTION}
        for k, v in out["predictions"].items():
            print(f"prediction {k}: {v['value']:.3f} registered {v['registered']} {'HIT' if v['hit'] else 'MISS'}")
    out["status"] = "COMPLETE"
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
