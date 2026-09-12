#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E33 -- is `down` the whole carve locality cost, and does a coarser carve pay the predicted 1.25x?

Briefs, both pushed BEFORE this file ran:
  docs/research/donor_adaptation/briefs/BRIEF_E33_IS_DOWN_THE_WHOLE_COST.md            (2fa83e9)
  docs/research/donor_adaptation/briefs/BRIEF_E33_ADDENDUM_TWO_GATES_CANNOT_FIRE.md
The addendum records that G-E33C (as written) is arithmetically unsatisfiable -- the router is
CHARGED and shrinks with E -- and that G-E33A was anchored to an E26 absolute E26 itself disowned.
Nothing here may contradict either document.

THIS PROBE EXISTS TO TRY TO KILL A CLAIM OF MINE.  E31 s6.1 -- corrected once already the same
day -- says gate/up are group-major and therefore ALREADY contiguous, that the fine granularity
is `down` at PT_BLK=64 (one cache line per selected neuron), and that coarsening the carve from
E=256 to E=16 is worth 1.25x at S15 and 1.14x at T10.  That is arithmetic on a measured table,
not a measurement.  Here it fires end-to-end or it does not.

EVERY ANCHOR IS READ FROM THE SOURCE PROBE'S JSON, never from a constant I typed from memory.

  python e33_down_locality.py --dir D:/_ktmp/e33 --reps 3
  python e33_down_locality.py --build-only      # export + check G-E33C', take no timing
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e26_carve_cost import cpu_busy, IDLE_BAR                        # noqa: E402
from e28_kernel_transfer import bench, winpath                       # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e33_down_locality.json")
PT_BLK = 64                       # donor_engine.c:638 -- the constant the whole claim rests on
ANCHOR_TOL = 0.10
CHARGE_TOL = 0.020                # G-E33C' (addendum s1): 2.0% on charged weights

SHAPES = {"S15": dict(D=1536, F=8960, L=28, V=151936, ntok=120),
          "T10": dict(D=4096, F=14336, L=48, V=32768, ntok=40)}
#        tag           shape   E     k     (k/E fixed at 1/4 within a shape)
ARMS = [("S15-DENSE",  "S15",    0,   0),
        ("S15-E256",   "S15",  256,  64),
        ("S15-E64",    "S15",   64,  16),
        ("S15-E16",    "S15",   16,   4),
        ("T10-E256",   "T10",  256,  64),
        ("T10-E16",    "T10",   16,   4)]

VERDICT_CELL = ("S15-E16", "S15-E256")            # brief s4, named before the run
BANDS = [(1.18, "LOCALITY-CONFIRMED"), (1.06, "LOCALITY-PARTIAL"), (0.0, "LOCALITY-ABSENT")]
DESK = {"S15": 3.830 / 3.057, "T10": 3.508 / 3.065}   # E31 s6.1's corrected desk model


def log(*a):
    print(*a, flush=True)


def anchors():
    """E26's carved rate and ratio, and E28's dense rate -- from the files, not from memory."""
    e26 = json.load(open(os.path.join(RES, "e26_carve_cost.json")))
    e28 = json.load(open(os.path.join(RES, "e28_kernel_transfer.json")))
    return {"e26_S15_K64": e26["arms"]["S15-K64"]["mean_tok_s"],
            "e26_S15_DENSE": e26["arms"]["S15-DENSE"]["mean_tok_s"],
            "e26_S15_K64_ratio_vs_dense": e26["arms"]["S15-K64"]["ratio_vs_dense"],
            "e28_S15_PACKED": e28["arms"]["S15-PACKED"]["mean_tok_s"]}


def runs_of(shape, E):
    """The two contiguous runs a carve group produces, in BYTES -- the quantities E31 s6.1's
    claim is entirely about.  gate/up: GSZ rows of D/2 bytes, group-major, so ONE run.
    down: MK_PACKED_T at PT_BLK, so GSZ * PT_BLK bytes inside each output block."""
    s = SHAPES[shape]
    if not E:
        return None, None
    gsz = s["F"] // E
    return gsz * (s["D"] // 2), gsz * PT_BLK


def build(d, tag, shape, E, k, seed=1234):
    out = os.path.join(d, "e33_%s.bin" % tag.lower().replace("-", "_"))
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  %-11s %s" % (tag, os.path.basename(out)))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", shape,
           "--out", out, "--codes", "mixed", "--seed", str(seed)]
    if E:
        cmd += ["--carve", str(E), "--carve-k", str(k)]
    log("  build %-11s %s" % (tag, " ".join(cmd[3:])))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(r.stdout.decode(errors="replace")[-2000:])
        log(r.stderr.decode(errors="replace")[-2000:])
        raise SystemExit("synth_export failed for " + tag)
    for ln in r.stdout.decode(errors="replace").splitlines():
        if "GATE V3" in ln or "--carve" in ln or ln.startswith("wrote"):
            log("        " + ln.strip())
    log("        [%.0fs]" % (time.time() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e33")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    d = winpath(a.dir)
    os.makedirs(d, exist_ok=True)
    engine = a.engine if os.path.isabs(a.engine) else os.path.join(HERE, a.engine)
    t0 = time.time()
    A = anchors()

    log("== E33: is `down` the whole carve locality cost? ==")
    log("  E31 s6.1's desk model, the thing this probe may kill:")
    for tag, shape, E, k in ARMS:
        g, dn = runs_of(shape, E)
        if g:
            log("     %-11s GSZ %4d   gate/up run %9d B   down run %8d B"
                % (tag, SHAPES[shape]["F"] // E, g, dn))
    log("     predicted S15-E16/S15-E256 = %.3f ;  T10-E16/T10-E256 = %.3f"
        % (DESK["S15"], DESK["T10"]))
    log("  anchors READ FROM FILE: E26 S15-K64 %.3f (ratio_vs_dense %.4f), E28 S15-PACKED %.2f"
        % (A["e26_S15_K64"], A["e26_S15_K64_ratio_vs_dense"], A["e28_S15_PACKED"]))

    out = {"brief": "briefs/BRIEF_E33_IS_DOWN_THE_WHOLE_COST.md (2fa83e9)",
           "addendum": "briefs/BRIEF_E33_ADDENDUM_TWO_GATES_CANNOT_FIRE.md",
           "PT_BLK": PT_BLK, "desk_model": DESK, "threads": a.threads, "reps": a.reps,
           "bands": BANDS, "verdict_cell": VERDICT_CELL, "anchors_from_file": A,
           "engine": os.path.basename(engine), "arms": {}}

    log("")
    paths, metas = {}, {}
    for tag, shape, E, k in ARMS:
        paths[tag] = build(d, tag, shape, E, k)
        metas[tag] = json.load(open(paths[tag] + ".json"))

    # ---- G-E33C (as written, EXPECTED TO FAIL -- addendum s1) and G-E33C' (decides)
    log("")
    g_c, as_written_ok, amended_ok = {}, True, True
    for shape in ("S15", "T10"):
        act = dict((t, metas[t]["active_weights_per_token"])
                   for t, s, E, k in ARMS if s == shape and E)
        ref = act["%s-E256" % shape]
        dev = dict((t, v / ref - 1.0) for t, v in act.items())
        ident = len(set(act.values())) == 1
        within = all(abs(x) <= CHARGE_TOL for x in dev.values())
        as_written_ok = as_written_ok and ident
        amended_ok = amended_ok and within
        g_c[shape] = {"active_weights_per_token": act, "dev_vs_E256": dev,
                      "identical": ident, "within_2pct": within}
        for t in act:
            log("  G-E33C  %-11s charges %12d  (%+.3f%% vs E256)" % (t, act[t], 100 * dev[t]))
        log("          -> %s as written (identical) | %s amended (<=%.1f%%)"
            % ("PASS" if ident else "FAIL", "PASS" if within else "FAIL", 100 * CHARGE_TOL))
    out["G_E33C_as_written"] = {"per_shape": g_c, "fires": as_written_ok,
                                "note": "expected FAIL: the router is charged and shrinks with E"
                                        " -- addendum s1, a defect of my brief, not of the box"}
    out["G_E33C_amended"] = {"tol": CHARGE_TOL, "fires": amended_ok}
    if not amended_ok:
        out["VOID"] = "G-E33C': a carved arm charges more than 2% away from E256"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E33C' VOID -- see " + OUT)

    if a.build_only:
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("")
        log("  --build-only: artifacts exported and G-E33C' checked, no timing taken")
        return 0

    busy, peak = cpu_busy(4)
    log("")
    log("  contention witness: %.1f%% mean / %.0f%% peak (bar %.0f%% on the MEAN)"
        % (busy, peak, IDLE_BAR))
    if busy > IDLE_BAR:
        raise SystemExit("the box is at %.1f%% -- a contended timing is not a timing.  STOP."
                         % busy)
    out["cpu_busy_mean_pct"] = [busy]
    out["cpu_busy_peak_pct"] = [peak]

    rates = dict((t, []) for t, _, _, _ in ARMS)
    log("")
    for r in range(a.reps):                       # reps OUTERMOST -- E28's law, and E31's fix
        for tag, shape, E, k in ARMS:
            v = bench(engine, paths[tag], SHAPES[shape]["ntok"], a.threads, [])
            rates[tag].append(v)
            log("  rep %d  %-11s %7.2f tok/s" % (r + 1, tag, v))
        b2, p2 = cpu_busy(2)
        out["cpu_busy_mean_pct"].append(b2)
        out["cpu_busy_peak_pct"].append(p2)
        log("         box %.1f%% mean / %.0f%% peak" % (b2, p2))

    for tag, shape, E, k in ARMS:
        v = rates[tag]
        m = sum(v) / len(v)
        g, dn = runs_of(shape, E)
        out["arms"][tag] = {"rates": v, "mean_tok_s": m, "median_tok_s": sorted(v)[len(v) // 2],
                            "spread": (max(v) - min(v)) / m, "shape": shape, "E": E, "k": k,
                            "gate_run_bytes": g, "down_run_bytes": dn,
                            "active_weights_per_token": metas[tag]["active_weights_per_token"]}
    worst = max(out["arms"][t]["spread"] for t, _, _, _ in ARMS)
    out["worst_spread"] = worst
    log("")
    log("  worst spread across arms: %.1f%%" % (100 * worst))

    # ---- anchors: A (informational + ratio decides), B (decides)
    dense = out["arms"]["S15-DENSE"]["mean_tok_s"]
    carved = out["arms"]["S15-E256"]["mean_tok_s"]
    a_abs = carved / A["e26_S15_K64"] - 1.0
    a_rat = (carved / dense) / A["e26_S15_K64_ratio_vs_dense"] - 1.0
    b_dev = dense / A["e28_S15_PACKED"] - 1.0
    out["G_E33A_absolute_informational"] = {
        "measured": carved, "e26": A["e26_S15_K64"], "rel_dev": a_abs, "tol": ANCHOR_TOL,
        "fires": bool(abs(a_abs) <= ANCHOR_TOL),
        "note": "informational only -- E26 disowned its absolutes (addendum s2)"}
    out["G_E33A_ratio_decides"] = {
        "measured_ratio": carved / dense, "e26_ratio": A["e26_S15_K64_ratio_vs_dense"],
        "rel_dev": a_rat, "tol": ANCHOR_TOL, "fires": bool(abs(a_rat) <= ANCHOR_TOL)}
    out["G_E33B"] = {"measured": dense, "e28": A["e28_S15_PACKED"], "rel_dev": b_dev,
                     "tol": ANCHOR_TOL, "fires": bool(abs(b_dev) <= ANCHOR_TOL)}
    log("")
    log("  G-E33A  abs   S15-E256 %.2f vs E26's %.2f  (%+.1f%%)  -> %s   [informational]"
        % (carved, A["e26_S15_K64"], 100 * a_abs,
           "PASS" if out["G_E33A_absolute_informational"]["fires"] else "FAIL"))
    log("  G-E33A' ratio S15-E256/S15-DENSE %.4f vs E26's %.4f  (%+.1f%%)  -> %s   [DECIDES]"
        % (carved / dense, A["e26_S15_K64_ratio_vs_dense"], 100 * a_rat,
           "PASS" if out["G_E33A_ratio_decides"]["fires"] else "FAIL"))
    log("  G-E33B        S15-DENSE %.2f vs E28's %.2f  (%+.1f%%)  -> %s   [DECIDES]"
        % (dense, A["e28_S15_PACKED"], 100 * b_dev, "PASS" if out["G_E33B"]["fires"] else "FAIL"))
    if not (out["G_E33A_ratio_decides"]["fires"] and out["G_E33B"]["fires"]):
        out["VOID"] = "a deciding anchor did not fire"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("")
        log("  VOID -- a deciding anchor did not fire.  No verdict is quotable from this run.")
        return 2

    # ---- the verdict, raw and byte-normalised (addendum s1)
    def cell(num, den, shape):
        raw = out["arms"][num]["mean_tok_s"] / out["arms"][den]["mean_tok_s"]
        f = (out["arms"][num]["active_weights_per_token"] /
             float(out["arms"][den]["active_weights_per_token"]))
        return {"raw": raw, "charge_factor": f, "normalised": raw * f,
                "band_raw": next(n for lo, n in BANDS if raw >= lo),
                "band_normalised": next(n for lo, n in BANDS if raw * f >= lo),
                "desk": DESK[shape]}

    num, den = VERDICT_CELL
    s15 = cell(num, den, "S15")
    t10 = cell("T10-E16", "T10-E256", "T10")
    e64 = out["arms"]["S15-E64"]["mean_tok_s"]
    lo_, hi_ = out["arms"][den]["mean_tok_s"], out["arms"][num]["mean_tok_s"]
    frac = (e64 - lo_) / (hi_ - lo_) if hi_ != lo_ else None
    out["verdict"] = {"cell": "%s / %s" % (num, den), "S15": s15, "T10": t10,
                      "name": s15["band_raw"],
                      "bands_disagree": s15["band_raw"] != s15["band_normalised"],
                      "reported_band": (s15["band_normalised"]
                                        if s15["band_raw"] != s15["band_normalised"]
                                        else s15["band_raw"]),
                      "T10_smaller_than_S15": bool(t10["raw"] < s15["raw"]),
                      "E64_fraction_of_the_way": frac}
    log("")
    log("  VERDICT CELL  %s / %s" % (num, den))
    log("     raw         %.3f  -> %s      (desk model said %.3f)"
        % (s15["raw"], s15["band_raw"], DESK["S15"]))
    log("     normalised  %.3f  -> %s      (x%.5f, removes the charged-router confound)"
        % (s15["normalised"], s15["band_normalised"], s15["charge_factor"]))
    if out["verdict"]["bands_disagree"]:
        log("     !! THE TWO LAND IN DIFFERENT BANDS -- the NORMALISED band is the finding")
    log("     T10  raw %.3f / norm %.3f (desk %.3f); smaller than S15's? %s  -- prediction 3"
        % (t10["raw"], t10["normalised"], DESK["T10"], out["verdict"]["T10_smaller_than_S15"]))
    if frac is not None:
        log("     E64 sits %.0f%% of the way from E256 to E16 (desk model said 77%%) -- pred 4"
            % (100 * frac))

    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
