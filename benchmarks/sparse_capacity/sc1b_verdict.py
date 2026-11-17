#!/usr/bin/env python3
# Sparse Capacity, phase 1b (D-M): the verdict, computed from a run directory without choosing anything.
#   python benchmarks/sparse_capacity/sc1b_verdict.py results/sparse_capacity/f1b_<stamp>
# Rules: docs/research/SPARSE_CAPACITY/F1B_CROSSOVER_PROTOCOL.md, sections 5 and 6 (registered before the run).
import json, os, statistics, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc1b_model as M  # noqa: E402
import sc1b_predict as P  # noqa: E402

PLANT_CPU, PLANT_STR = (0.8, 1.5), (0.8, 1.3)     # P-cpu: a real arm kernel (C2, no unit active), ~1.2 in the shakedown
GHZ_RANGE, GHZ_DRIFT, BUSY_MAX, FLOOR_SLACK = (2.8, 4.7), 0.10, 0.15, 0.9


def main(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    ctl = [json.loads(l) for l in open(os.path.join(run, "controls.jsonl"))]
    rec = [json.loads(l) for l in open(os.path.join(run, "cells.jsonl"))]
    start = next(c for c in ctl if c["kind"] == "controls" and c["when"] == "start")
    end = next((c for c in ctl if c["kind"] == "controls" and c["when"] == "end"), None)
    plants = next(c for c in ctl if c["kind"] == "plants")
    fma_mid = [c for c in ctl if c["kind"] == "fma_mid"]
    rs = list(cfg["r_list"]); hs = list(cfg["h_list"]); P_bank = cfg["P_bank"]
    want = cfg["reps"] * cfg["rounds"]

    ns, dens = defaultdict(list), defaultdict(list)
    nonfinite = [r for r in rec if r["kind"] == "cell_done" and r["nonfinite"]]
    for r in rec:
        if r["kind"] == "sample":
            ns[(r["arm"], r["h"], r["r"])].append(r["ns_per_param"]); dens[(r["arm"], r["h"], r["r"])].append(r.get("density", 0.0))

    # ---- validity
    bad = []
    ram_pk = max(start["ram"][m]["max"] for m in ("read", "write_nt", "rmw"))
    h2d_pk, d2h_pk = start["pcie"]["h2d"]["max"], start["pcie"]["d2h"]["max"]
    bidir_pk = start["pcie"]["bidir_per_dir"]["max"]
    if not PLANT_CPU[0] <= plants["cpu"]["ratio"] <= PLANT_CPU[1]:
        bad.append(f"CPU plant ratio {plants['cpu']['ratio']:.2f}")
    if not PLANT_STR[0] <= plants["str"]["ratio"] <= PLANT_STR[1]:
        bad.append(f"streaming plant ratio {plants['str']['ratio']:.2f}")
    g0 = start["fma6"]["ghz_implied"]
    if not GHZ_RANGE[0] <= g0 <= GHZ_RANGE[1]:
        bad.append(f"FMA plant implies {g0:.2f} GHz")
    if end:
        for lab, g1 in [("end", end["fma6"]["ghz_implied"])] + [(f"round {c['round']} h{c['h']}", c["fma6"]["ghz_implied"])
                                                                 for c in fma_mid]:
            if abs(g1 - g0) / g0 > GHZ_DRIFT:
                bad.append(f"FMA plant drifted {g0:.2f} -> {g1:.2f} GHz at {lab} (throttling)")
    else:
        bad.append("no end-of-run controls (run incomplete)")
    if start["cpu_busy"] > BUSY_MAX:
        bad.append(f"CPU busy {start['cpu_busy']:.0%} at start")
    if nonfinite:
        bad.append(f"{len(nonfinite)} cells with non-finite weights")
    floor_viol = []
    for (arm, h, r), xs in ns.items():
        if len(xs) != want:
            bad.append(f"{arm} h={h} r={r}: {len(xs)} samples, want {want}")
        path = M.ARMS[arm][0]
        if path == "cpu":
            # the CPU arm's t charges the per-layer fixed costs pro rata; the floor counts the weight bytes only
            fl = M.floor_cpu(arm, r, statistics.median(dens[(arm, h, r)]), ram_pk)
        else:
            fl = M.floor_str(arm, r, 0.0, ram_pk, h2d_pk, d2h_pk)
        if min(xs) < FLOOR_SLACK * fl:
            floor_viol.append(f"{arm} h={h} r={r}: {min(xs):.3f} < {FLOOR_SLACK} x floor {fl:.3f} ns/param")
    if floor_viol:
        bad.append(f"{len(floor_viol)} cells under their byte floor")

    # ---- curves and r*
    def curve(arm, h, f):
        return [f(ns[(arm, h, r)]) for r in rs]
    rows = []
    for pair, (ca, sa) in M.PAIRS.items():
        for h in hs:
            if any((x, h, r) not in ns for x in (ca, sa) for r in rs):
                continue
            med = M.crossover(rs, curve(ca, h, statistics.median), curve(sa, h, statistics.median))
            # streaming at its best = the faster of the measured minimum and the IDEAL streamer (review obj. 2-3):
            # measured peak bandwidths, no GPU compute, only active rows moved in the backward
            # the per-token density MEASURED on the CPU arm of the pair (the S arm runs the same weights and tokens)
            ideal = [M.desk_str(sa, h, r, statistics.median(dens[(ca, h, r)]), ram_pk, h2d_pk, bidir_pk, ideal=True)
                     for r in rs]
            s_best = [min(a, b) for a, b in zip(curve(sa, h, min), ideal)]
            lo = M.crossover(rs, curve(ca, h, max), s_best)                 # CPU at its worst, streaming at its best
            lo_meas = M.crossover(rs, curve(ca, h, max), curve(sa, h, min))
            hi = M.crossover(rs, curve(ca, h, min), curve(sa, h, max))
            gains = {T: M.bank_gain(rs, curve(ca, h, statistics.median), curve(sa, h, statistics.median),
                                    M.regime_mean_r(pair, T)) for T in (16384, 131072)} if pair in M.REGIME_BYTES else {}
            rows.append(dict(pair=pair, h=h, med=med, lo=lo, lo_meas=lo_meas, hi=hi, gains=gains,
                             cpu=curve(ca, h, statistics.median), str=curve(sa, h, statistics.median),
                             ideal=ideal, dens=statistics.median(dens[(ca, h, rs[0])])))

    print(f"RUN {run}\n  head {cfg['git_head']}  mode {cfg['mode']}  link {start['pcie'].get('link')}")
    print(f"  RAM rmw {start['ram']['rmw']['med']:.1f} GB/s  read {start['ram']['read']['med']:.1f}  "
          f"H2D {start['pcie']['h2d']['med']:.1f}  D2H {start['pcie']['d2h']['med']:.1f}  "
          f"bidir {start['pcie']['bidir_per_dir']['med']:.1f}  FMA {g0:.2f} GHz  "
          f"plants cpu {plants['cpu']['ratio']:.2f} str {plants['str']['ratio']:.2f}")
    print("\n  median ns per parameter per step, CPU / streaming, by r")
    print(f"  {'pair':<16} {'h':>4} " + " ".join(f"{r:>11}" for r in rs))
    for o in rows:
        print(f"  {o['pair']:<16} {o['h']:>4} " + " ".join(f"{c:5.2f}/{s:<5.2f}" for c, s in zip(o["cpu"], o["str"])))
    print(f"\n  {'pair':<16} {'h':>4} {'r* med':>7} {'[lo':>6} {'hi]':>6} {'lo meas':>8} {'dens@1':>7}"
          f"  step gain at T=16k / 128k (reading, not gate)")
    for o in rows:
        g = " / ".join(f"{100 * v:5.1f}%" for v in o["gains"].values()) if o["gains"] else "-"
        print(f"  {o['pair']:<16} {o['h']:>4} {P.fmt_r(o['med']):>7} {P.fmt_r(o['lo']):>6} {P.fmt_r(o['hi']):>6} "
              f"{P.fmt_r(o['lo_meas']):>8} {o['dens']:7.3f}  {g}")
    gpu = defaultdict(list)
    for r in rec:
        if r["kind"] == "sample" and "gpu_busy_bwd" in r:
            gpu[(r["arm"], r["h"])].append((r["gpu_busy_fwd"] + r["gpu_busy_bwd"]) / r["t"])
    print("\n  streaming: GPU-busy share of the step (median over r and samples; near 1 = compute-bound, not PCIe)")
    print("  " + "  ".join(f"{a} h{h} {statistics.median(v):.2f}" for (a, h), v in sorted(gpu.items())))
    print("\n  desk 2 (measured bandwidths, recorded before the arms):")
    d2 = next((c for c in ctl if c["kind"] == "desk2"), None)
    if d2:
        P.print_table([dict(x, add=tuple(x["add"]), ovl=tuple(x["ovl"])) for x in d2["table"]], title="  DESK 2")
    for v in floor_viol[:20]:
        print("  FLOOR", v)

    # ---- the registered gate
    gate = [o for o in rows if o["pair"] in M.GATE_PAIRS and o["h"] in M.GATE_H]
    complete = len(gate) == len(M.GATE_PAIRS) * len(M.GATE_H)
    angle = [o for o in gate if (o["lo"][1] == ">" or (o["lo"][1] == "=" and o["lo"][0] >= 2.0))]
    stream = complete and all(o["hi"][1] == "<" for o in gate)
    if bad or not complete:
        verdict = "VOID"
    elif angle:
        verdict = "ANGOLO CPU"
    elif stream:
        verdict = "STREAMING"
    else:
        verdict = "GRIGIO"
    print("\nVALIDITY:", "ok" if not bad else "; ".join(bad))
    print("GATE cells (pair, h, r* lo):", [(o["pair"], o["h"], P.fmt_r(o["lo"])) for o in gate])
    print("VERDICT:", verdict, "" if not angle else f"(r* lo >= 2 in: {[(o['pair'], o['h']) for o in angle]})")
    if verdict in ("STREAMING", "GRIGIO"):
        print("  -> the method is streaming; D-M becomes engineering: the question is whether the programme goes on for D-R")


if __name__ == "__main__":
    main(sys.argv[1])
