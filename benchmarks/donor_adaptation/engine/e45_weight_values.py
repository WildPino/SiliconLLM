#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 -- does the engine's speed depend on the WEIGHT VALUES?

Brief:    BRIEF_E45_DOES_SPEED_DEPEND_ON_THE_WEIGHTS.md   (f2c1485)
Addendum: same file, ADDENDUM B                           (d50e59e)
Both pushed before this file existed.  Nothing here may contradict them, and
nothing here decides anything the brief did not register.

THE GAP.  Every throughput number this programme has published at 10B scale was
measured on SYNTHETIC weights (E36, E39, E40 all say so in their own briefs).
E37 measured speed on REAL weights, but only real ones -- its `speed_arms` hold
no synthetic arm.  So the inference that joins the two halves -- that engine
speed does not depend on what the weights actually ARE -- has never been
measured.  E45 measures it on the matched pair E37 built and never raced.

THE DESIGN, from the brief section 3 and addendum B section B.2:
  * two arms, NF (real) and SYN (synthetic), identical bytes, different sha256;
  * INTERLEAVED NF, SYN, NF, SYN, ... so drift in occupancy, DVFS and thermals
    cancels WITHIN a pair instead of loading onto one arm;
  * the statistic is the per-pair ratio NF/SYN, reported as median and spread,
    never as a bare number;
  * K256 is PRIMARY and value-only (every group selected, so the router that
    differs between the files cannot change which memory is touched);
  * K16 is SECONDARY and is value+selection.  It may not override the primary.

  python e45_weight_values.py                 # everything, ~10-15 min
  python e45_weight_values.py --pairs 7
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e44_interval import Occupancy, one_rep, log                     # noqa: E402
from e28_kernel_transfer import winpath                              # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e45_weight_values.json")

# E37's own engine and its own bench length.  G-E45a has to reproduce a gap E37
# measured, so it has to be measured the way E37 measured it.
ENGINE = os.path.join(HERE, "donor_engine_e26.exe")
NTOK, THREADS = 40, 6

# E37's speed table, read from BRIEF_E37 / e37_sparsity_cost.json.  These are the
# known-positive the planted control must resolve -- they are NOT a target, and no
# cell below is compared against them for agreement.
E37_DENSE_NF, E37_K16 = 30.75, 89.83

# addendum B, B.3
G45A_MEDIAN_MIN, G45A_EVERY_PAIR_MIN, G45A_PAIRS = 2.0, 1.5, 3
# brief section 4, G-E45b
MIN_PAIRS = 5
# brief section 4, G-E45c
BROKEN_FLOOR = 0.05


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar(path):
    p = path + ".json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def rel_spread(v):
    return (max(v) - min(v)) / (sum(v) / len(v)) if v else 0.0


def identity_check(nf, syn):
    """addendum B, B.4 -- equal bytes, DIFFERENT content.  Racing a file against
    itself returns r = 1 with a tiny spread and looks exactly like the registered
    prediction, so this runs before anything is timed and aborts on failure."""
    log("== identity of the pair (addendum B, B.4) ==")
    rows = []
    for tag, p in (("NF", nf), ("SYN", syn)):
        if not os.path.exists(p):
            raise SystemExit("missing artifact %s -- E45 races what E37 built, it builds "
                             "nothing.  STOP." % p)
        sc = sidecar(p)
        t0 = time.time()
        h = sha256(p)
        rows.append({"arm": tag, "path": p, "bytes": os.path.getsize(p), "sha256": h,
                     "sha256_sidecar": sc.get("sha256"),
                     "zero_fraction": sc.get("mean_ternary_zero_fraction"),
                     "carve_router_kind": sc.get("carve_router_kind"),
                     "carve_k_in_file": sc.get("carve_k_in_file")})
        log("  %-4s %13d bytes  sha %s  zerofrac %.10f  router %s  [%.0fs]"
            % (tag, rows[-1]["bytes"], h[:16], rows[-1]["zero_fraction"] or float("nan"),
               rows[-1]["carve_router_kind"], time.time() - t0))
    if rows[0]["bytes"] != rows[1]["bytes"]:
        raise SystemExit("the two artifacts are NOT the same size (%d vs %d) -- they are not a "
                         "matched pair and the ratio would price the shape, not the values.  "
                         "STOP." % (rows[0]["bytes"], rows[1]["bytes"]))
    if rows[0]["sha256"] == rows[1]["sha256"]:
        raise SystemExit("the two artifacts are BYTE-IDENTICAL -- this would be a file raced "
                         "against itself, and it would return r = 1 for the one reason that "
                         "means nothing.  STOP.")
    for r in rows:
        if r["sha256_sidecar"] and r["sha256_sidecar"] != r["sha256"]:
            raise SystemExit("%s: the file on disk does not match its own sidecar sha256 -- it "
                             "has been rewritten since E37 measured it.  STOP." % r["arm"])
    log("  equal bytes, different sha256, sidecars agree with the files : FIRES")
    return rows


def race(occ, arms, pairs, label):
    """Interleaved A, B, A, B, ...  Returns per-arm rate lists, per-pair ratios and
    the occupancy read across each cell.  The interleaving IS the design."""
    rates = dict((t, []) for t, _, _ in arms)
    dts = dict((t, []) for t, _, _ in arms)
    walls = dict((t, []) for t, _, _ in arms)
    occs = dict((t, []) for t, _, _ in arms)
    log("  %s -- %d interleaved pairs of %s" % (label, pairs, " / ".join(t for t, _, _ in arms)))
    for i in range(pairs):
        line = []
        for tag, w, flags in arms:
            occ.sample()                       # close the previous window
            v, dt, wall = one_rep(ENGINE, w, NTOK, THREADS, flags)
            o = occ.sample()                   # occupancy across THIS cell only
            rates[tag].append(v)
            dts[tag].append(dt)
            walls[tag].append(wall)
            occs[tag].append(o)
            line.append("%s %7.2f tok/s (occ %4.1f%%, load %5.1fs)" % (tag, v, o, wall - dt))
        a, b = arms[0][0], arms[1][0]
        r = rates[a][-1] / rates[b][-1]
        log("    pair %d  %s   ->  %s/%s = %.4f" % (i + 1, "  ".join(line), a, b, r))
    ratios = [rates[arms[0][0]][i] / rates[arms[1][0]][i] for i in range(pairs)]
    return rates, dts, walls, occs, ratios


def arm_summary(rates, dts, walls, occs, tag):
    v = rates[tag]
    return {"rates": v, "median_tok_s": statistics.median(v), "min_tok_s": min(v),
            "max_tok_s": max(v), "rel_spread": rel_spread(v),
            "engine_dt_s": dts[tag], "wall_s": walls[tag],
            "load_s": [walls[tag][i] - dts[tag][i] for i in range(len(v))],
            "occupancy_pct": occs[tag]}


def verdict_e45c(ratios):
    """brief section 4, G-E45c, applied verbatim.  `r` is the median per-pair ratio and
    `s` is the spread of the per-pair ratios -- max minus min, in ratio units, which is
    the scale |r - 1| is on."""
    r = statistics.median(ratios)
    s = max(ratios) - min(ratios)
    d = abs(r - 1.0)
    if d < s:
        v = "BRIDGE HOLDS"
    elif d > 2.0 * s and d > BROKEN_FLOOR:
        v = "BRIDGE BROKEN"
    else:
        v = "INCONCLUSIVE"
    return {"median_ratio": r, "spread": s, "abs_r_minus_1": d, "verdict": v,
            "ratios": list(ratios)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e37")
    ap.add_argument("--pairs", type=int, default=MIN_PAIRS)
    ap.add_argument("--control-pairs", type=int, default=G45A_PAIRS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.pairs < MIN_PAIRS:
        raise SystemExit("G-E45b registers >= %d interleaved pairs.  %d is below the gate's own "
                         "number and the gate is not tunable downward.  STOP."
                         % (MIN_PAIRS, a.pairs))
    d = winpath(a.dir)
    nf = os.path.join(d, "e37_carved_nf.bin")
    syn = os.path.join(d, "e37_carved_syn.bin")
    dense = os.path.join(d, "e37_dense_nf.bin")
    if not os.path.exists(ENGINE):
        raise SystemExit("missing %s -- G-E45a reproduces a gap E37 measured, so it must be "
                         "measured with E37's engine.  STOP." % ENGINE)

    log("=" * 78)
    log("E45 -- does the engine's speed depend on the weight VALUES?")
    log("  engine %s   ntok %d   threads %d" % (os.path.basename(ENGINE), NTOK, THREADS))
    log("  THIS SCRIPT PRINTS THE GATES THE BRIEF REGISTERED.  It mints no rate and it")
    log("  may not be quoted for an absolute tok/s -- E43 stands, E44 addendum A did not")
    log("  repeal it.  Absolute rates below carry their dispersion and nothing else.")
    log("=" * 78)

    out = {"brief": "BRIEF_E45_DOES_SPEED_DEPEND_ON_THE_WEIGHTS.md",
           "brief_commits": ["f2c1485", "d50e59e"],
           "engine": os.path.basename(ENGINE), "ntok": NTOK, "threads": THREADS,
           "pairs": a.pairs, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    out["identity"] = identity_check(nf, syn)

    occ = Occupancy()
    occ.sample()

    # ---------------------------------------------------------------- G-E45a
    log("")
    log("== G-E45a -- PLANTED CONTROL.  Nothing below counts until this fires ==")
    log("  known positive: E37 read DENSE-NF %.2f and K16 %.2f tok/s, a %.2fx gap."
        % (E37_DENSE_NF, E37_K16, E37_K16 / E37_DENSE_NF))
    log("  the instrument must resolve it: median K16/DENSE >= %.1f and every pair > %.1f"
        % (G45A_MEDIAN_MIN, G45A_EVERY_PAIR_MIN))
    cr, cd, cw, co, cratios = race(
        occ, [("K16", nf, ["--carve-k", "16"]), ("DENSE", dense, [])],
        a.control_pairs, "G-E45a")
    cmed = statistics.median(cratios)
    a_fires = cmed >= G45A_MEDIAN_MIN and min(cratios) > G45A_EVERY_PAIR_MIN
    log("  median K16/DENSE %.4f  (min %.4f, max %.4f)"
        % (cmed, min(cratios), max(cratios)))
    log("  G-E45a : %s" % ("FIRES" if a_fires else "*** FAILS ***"))
    out["G_E45a"] = {"ratios": cratios, "median_ratio": cmed,
                     "median_min": G45A_MEDIAN_MIN, "every_pair_min": G45A_EVERY_PAIR_MIN,
                     "fires": bool(a_fires),
                     "K16": arm_summary(cr, cd, cw, co, "K16"),
                     "DENSE": arm_summary(cr, cd, cw, co, "DENSE"),
                     "e37_reference": {"dense_nf": E37_DENSE_NF, "k16": E37_K16}}
    if not a_fires:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E45a did NOT fire.  The harness has not been shown to resolve a "
                         "speed difference that is really there, so its null on NF vs SYN "
                         "would be worthless.  No null from an instrument that has not fired.  "
                         "STOP.")

    # ------------------------------------------------------- the two real arms
    plan = [("K256", ["--carve-k", "256"], "PRIMARY -- value only, the router cannot "
                                           "change which memory is touched"),
            ("K16", ["--carve-k", "16"], "SECONDARY -- value + selection, may not "
                                         "override the primary")]
    out["arms"] = {}
    for tag, flags, why in plan:
        log("")
        log("== %s  (%s) ==" % (tag, why))
        rr, dd, ww, oo, ratios = race(
            occ, [("NF", nf, flags), ("SYN", syn, flags)], a.pairs, tag)
        vd = verdict_e45c(ratios)
        log("  NF  median %7.2f tok/s  (min %.2f, max %.2f, rel spread %.4f)"
            % (statistics.median(rr["NF"]), min(rr["NF"]), max(rr["NF"]),
               rel_spread(rr["NF"])))
        log("  SYN median %7.2f tok/s  (min %.2f, max %.2f, rel spread %.4f)"
            % (statistics.median(rr["SYN"]), min(rr["SYN"]), max(rr["SYN"]),
               rel_spread(rr["SYN"])))
        log("  per-pair ratio NF/SYN : median %.4f   spread (max-min) %.4f   |r-1| %.4f"
            % (vd["median_ratio"], vd["spread"], vd["abs_r_minus_1"]))
        log("  G-E45c on %s : %s" % (tag, vd["verdict"]))
        out["arms"][tag] = {"why": why, "flags": flags,
                            "NF": arm_summary(rr, dd, ww, oo, "NF"),
                            "SYN": arm_summary(rr, dd, ww, oo, "SYN"),
                            "G_E45c": vd}

    # ---------------------------------------------------------------- G-E45b
    every = []
    for tag in out["arms"]:
        every.append(len(out["arms"][tag]["G_E45c"]["ratios"]) >= MIN_PAIRS)
    b_fires = all(every)
    log("")
    log("== G-E45b -- the ratio is reported with dispersion or not at all ==")
    log("  %d pairs per arm (>= %d), median and min-max printed for every arm : %s"
        % (a.pairs, MIN_PAIRS, "FIRES" if b_fires else "*** FAILS ***"))
    out["G_E45b"] = {"pairs": a.pairs, "min_pairs": MIN_PAIRS, "fires": bool(b_fires)}

    # ------------------------------------------------------------- the report
    prim = out["arms"]["K256"]["G_E45c"]
    sec = out["arms"]["K16"]["G_E45c"]
    log("")
    log("=" * 78)
    log("E45 REGISTERED VERDICT (addendum B, B.2: read from K256)")
    log("  K256  PRIMARY   r = %.4f   s = %.4f   |r-1| = %.4f   -> %s"
        % (prim["median_ratio"], prim["spread"], prim["abs_r_minus_1"], prim["verdict"]))
    log("  K16   secondary r = %.4f   s = %.4f   |r-1| = %.4f   -> %s   (value+selection)"
        % (sec["median_ratio"], sec["spread"], sec["abs_r_minus_1"], sec["verdict"]))
    if prim["verdict"] == "BRIDGE HOLDS" and sec["verdict"] == "BRIDGE BROKEN":
        log("  the two do NOT contradict each other: addendum B B.2 registered this case in")
        log("  advance -- it localises the effect in the ROUTER, not in the values.")
    log("")
    log("  This is measured at 1.5B.  Brief section 5: E45 MAY NOT claim the result composes")
    log("  to 10B by itself, and MAY NOT mint any headline rate.")
    log("=" * 78)
    out["verdict"] = {"primary_arm": "K256", "primary": prim, "secondary": sec}
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.isdir(RES):
        os.makedirs(RES)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
