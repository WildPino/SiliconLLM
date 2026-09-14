#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E63 Part B -- the 10 B cell at one byte per weight, MEASURED.  Idle box only.

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E63_THE_TEN_BILLION_CELL_AT_ONE_BYTE.md`
Part A (`e63_ten_billion_at_one_byte.py`) closed first and is a PRECONDITION: G-E63a FIRES,
G-E63b FIRES, G-E63c EQUIVALENT.  This runner refuses to start if that JSON does not say so.
A rate on a path whose arithmetic has not been proved is not a measurement of anything.

WHY THIS IS A SEPARATE FILE.  E40 addendum A: a gate that fires is doing its job and is not
re-run to a pass.  Part A is cheap and deterministic and Part B is expensive and needs a quiet
machine; keeping them in one process would make "just re-run the whole thing" the easy move.

WHAT THIS RUNNER REFUSES TO DO.

  * It refuses to report a rate from a contended box.  `OCC_BAR = 4.39%` FOREIGN occupancy
    (E52), measured as system-minus-this-engine -- the magnitude, not the threshold, is what
    `G-E44b` got wrong, and Part A's first pass got it wrong again by passing child_ticks = 0.
    Every cell over the bar and the sweep is `VOID`, reported as void, not adjusted.

  * It refuses to turn one number into a point.  Five interleaved repetitions per arm, paired by
    repetition index, median + bootstrap CI.  A timing without dispersion is not a timing.

  * It refuses to divide bytes by a bandwidth (SPEED_LEDGER 59.2, and 61.6 is the cautionary
    case where I broke that rule while auditing for it).  Every rate here is measured; the bands
    and the byte ratios appear only as COMPARATORS beside a measured ratio.

  * It refuses to adjudicate `G-E61c`.  E61 read 1.0947 in a zone its brief deliberately left
    unadjudicated and the probe recorded the repeat as owed.  This runner prints the repeat and
    the two reference readings and stops there.  The verdict stays E61's.
"""
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e44_interval as E44                                          # noqa: E402

OUTDIR = os.path.join(HERE, "results")
OUTFILE = os.path.join(OUTDIR, "e63_part_b.json")
# ADDENDUM C: the interim paired ratio writes to its OWN file, so a contended reading can never
# be mistaken for Part B's output.  --interim also stops after sweep 1: G-E61c's repeat and the
# 3 B cells are absolute readings and a contended box cannot answer them at all.
INTERIM_FILE = os.path.join(OUTDIR, "e63_interim_ratio.json")
PART_A = os.path.join(OUTDIR, "e63_part_a.json")
BRIEF = "BRIEF_E63_THE_TEN_BILLION_CELL_AT_ONE_BYTE.md"

ENGINE = os.path.join(HERE, "donor_engine_e63.exe")
THREADS = 6

E1 = os.path.join("D:", os.sep, "_ktmp", "e1")
E36 = os.path.join("D:", os.sep, "_ktmp", "e36")
E60 = os.path.join("D:", os.sep, "_ktmp", "e60")
E62 = os.path.join("D:", os.sep, "_ktmp", "e62")
E63 = os.path.join("D:", os.sep, "_ktmp", "e63")

# ---------------------------------------------------------------- registered constants
OCC_BAR = 4.39                # E52.  Necessary, not sufficient.
DESK_MODEL = 36.6             # SPEED_LEDGER 60.4's lower figure -- the number G-E63d judges
REPS = 5                      # ">= 5 interleaved repetitions", brief section 4
NTOK = 120                    # decode tokens per repetition
BOOT = 20000
CI = (2.5, 97.5)
SEED = 63

# E63's comparators for G-E63e.  NOT divisors: they are printed BESIDE a measured ratio.
#
# ADDENDUM B.  The brief registered G-E63e against "the byte ratio (2.0)".  2.0 is the FFN's OWN
# ratio, not the token's.  The A10B-K3 charged token is 928,251,904 weights (G-E36C, zero
# tolerance) of which the carved FFN is 106,168,320 = 11.4%; attention alone is 72.3%, and
# EVERYTHING outside the FFN stays packed at half a byte in both files.  So the format change
# moves the token's charged bytes from 464.13 MB to 517.21 MB -- a ratio of 1.1144, not 2.0.
#
# SPEED_LEDGER 59.2's bands are inapplicable for the same reason and are NOT printed: the int8
# artefact is a MIXTURE, 88.6% of it still at half a byte, and no single band describes a
# mixture.  Manufacturing one is exactly what 59.2 forbids and what 61.6 did before being
# corrected.
A10B_CHARGED = 928251904
A10B_FFN_CHARGED = 106168320
CHARGED_MB_PACKED = A10B_CHARGED * 0.5 / 1e6
CHARGED_MB_INT8 = ((A10B_CHARGED - A10B_FFN_CHARGED) * 0.5 + A10B_FFN_CHARGED * 1.0) / 1e6
CHARGED_RATIO = CHARGED_MB_INT8 / CHARGED_MB_PACKED          # 1.1144

# E61's two reference readings for the owed repeat.  Reported, never adjudicated here.
E61C_READ = 1.0947
E8_G_Z5 = 1.0029
E61C_NULL = (0.97, 1.05)
E61C_REFUTES = 1.10

# ---------------------------------------------------------------- the arms
# G-E63d: the pair the whole experiment exists for, interleaved in ONE sweep.
PAIR_10B = [
    ("A10B_PACKED", os.path.join(E36, "e36_a10b.bin"),         ["--carve-k", "3"]),
    ("A10B_INT8",   os.path.join(E63, "e63_a10b_i8.bin"),      ["--carve-k", "3"]),
]
# G-E61c's owed repeat: 05b fp32, mvacc 4 against mvacc 1, same sweep, same binary.
PAIR_E61C = [
    ("05B_F32_M4", os.path.join(E1, "qwen25-05b_f32.bin"), ["--mvacc", "4"]),
    ("05B_F32_M1", os.path.join(E1, "qwen25-05b_f32.bin"), ["--mvacc", "1"]),
]
# The 3 B speed cells E62 could not take (it was quality-only on a dirty box).
# DESCRIPTIVE AND UNGATED, by E14 section 6: a post-hoc metric may not be promoted to a gate.
CELLS_3B = [
    ("3B_F32",    os.path.join(E62, "qwen25-3b_f32.bin"),    []),
    ("3B_INT8",   os.path.join(E62, "qwen25-3b_i8.bin"),     []),
    ("3B_PACKED", os.path.join(E62, "qwen25-3b_packed.bin"), []),
]


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def boot_ci(values, stat=median, b=BOOT, ci=CI, seed=SEED):
    """Percentile bootstrap over repetition indices."""
    rnd = random.Random(seed)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(b):
        draws.append(stat([values[rnd.randrange(n)] for _ in range(n)]))
    draws.sort()
    lo = draws[max(0, min(b - 1, int(round(ci[0] / 100.0 * b))))]
    hi = draws[max(0, min(b - 1, int(round(ci[1] / 100.0 * b))))]
    return (lo, hi)


def boot_ratio_ci(num, den, b=BOOT, ci=CI, seed=SEED):
    """PAIRED bootstrap: one index draw feeds both arms, so the shared box state stays paired."""
    rnd = random.Random(seed)
    n = min(len(num), len(den))
    if n < 2:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(b):
        idx = [rnd.randrange(n) for _ in range(n)]
        d = median([den[i] for i in idx])
        draws.append(median([num[i] for i in idx]) / d if d > 0 else float("nan"))
    draws = [x for x in draws if x == x]
    draws.sort()
    m = len(draws)
    if m < 2:
        return (float("nan"), float("nan"))
    return (draws[max(0, int(round(ci[0] / 100.0 * m)))],
            draws[min(m - 1, int(round(ci[1] / 100.0 * m)))])


# ============================================================ the decision functions
def admissible(cells):
    """E52's bar, on FOREIGN occupancy: system minus this engine's own ticks.

    `cells` = [{"name":..., "foreign":...}, ...].  Necessary, not sufficient -- of the 7.50
    points between E44's two runs, load explained 2.33.  Over the bar => the sweep is VOID and
    is reported as void.
    """
    if not cells:
        return "VOID -- no cell measured"
    over = ["%s=%.2f%%" % (c["name"], c["foreign"]) for c in cells if c["foreign"] >= OCC_BAR]
    if over:
        return "VOID -- over OCC_BAR=%.2f%%: %s" % (OCC_BAR, ", ".join(over))
    return "ADMISSIBLE"


def arm_foreign(cells):
    """Mean foreign occupancy per arm, and the ASYMMETRY between the two arms.

    E61's planted null went out of band because contamination was asymmetric by 8.29 points
    between the arms it compared.  A paired ratio is only paired if both arms were contended
    alike, so the asymmetry is reported beside every ratio -- and at E52's k = 0.262% of rate
    per point, it converts directly into how much of the ratio the box could have manufactured.
    """
    by = {}
    for c in cells:
        by.setdefault(c["name"], []).append(c["foreign"])
    means = dict((k, sum(v) / len(v)) for k, v in by.items())
    asym = (max(means.values()) - min(means.values())) if len(means) >= 2 else 0.0
    return means, asym, asym * 0.00262      # E52: fraction of rate the asymmetry can explain


def g_e63d(lo, hi, bar=DESK_MODEL):
    """The measurement the programme has been projecting.  Bands fixed in the brief, section 4."""
    if lo != lo or hi != hi:
        return "DEAD -- no CI"
    if lo >= bar:
        return "DESK-MODEL-HELD"
    if hi < bar:
        return "DESK-MODEL-OPTIMISTIC"
    return "DESK-MODEL-UNRESOLVED"


def g_e63e(ratio, lo, hi):
    """DESCRIPTIVE by registration: no prediction of mine about this curve has survived contact
    (E31 twice, E33 once, E61 once), so it has no pass line and states comparators only.

    The comparator is the TOKEN's charged byte ratio (addendum B), not the FFN's.  CHARGED, not
    moved: the carved FFN gathers, so its 11.4% costs more than it is charged (E26), and that is
    the whole content of the locality question -- any gap between the measured ratio and
    1/CHARGED_RATIO is the gathered-weight penalty plus whatever the kernel change is worth.
    """
    if ratio != ratio:
        return "DEAD -- no ratio"
    return ("measured %.4f  [%.4f, %.4f] | charged-byte comparator 1/%.4f = %.4f "
            "(token: %.2f -> %.2f MB; the FFN's own ratio 2.0 is NOT the token's) | "
            "moved != charged: the carved 11.4%% gathers"
            % (ratio, lo, hi, CHARGED_RATIO, 1.0 / CHARGED_RATIO,
               CHARGED_MB_PACKED, CHARGED_MB_INT8))


def part_a_is_closed():
    """A rate on an unproved path measures nothing.  Part A must have closed, in the file."""
    if not os.path.exists(PART_A):
        return False, "no %s -- Part A has not run" % PART_A
    with open(PART_A, "r") as fh:
        a = json.load(fh)
    want = {"G-E63a": "FIRES", "G-E63b": "FIRES", "G-E63c": "EQUIVALENT"}
    bad = ["%s=%r" % (k, a.get(k)) for k, v in sorted(want.items()) if a.get(k) != v]
    if bad:
        return False, "Part A did not close: " + ", ".join(bad)
    return True, "Part A closed: " + ", ".join("%s %s" % (k, a[k]) for k in sorted(want))


# ============================================================ measurement
def rep(name, weights, flags):
    """One --bench repetition, with the occupancy split measured AROUND it."""
    sp = E44.Split()
    tok_s, dt, wall, child = E44.one_rep(ENGINE, weights, NTOK, THREADS, flags)
    sysb, foreign = sp.close(child)
    r = {"name": name, "tok_s": tok_s, "engine_dt_s": dt, "wall_s": wall,
         "sys": sysb, "foreign": foreign}
    r.update(sp.record())
    return r


def sweep(arms, reps=REPS, tag=""):
    """Interleaved: every arm gets repetition i before any arm gets repetition i+1."""
    by = dict((a[0], []) for a in arms)
    cells = []
    for i in range(reps):
        for name, wp, flags in arms:
            r = rep(name, wp, flags)
            by[name].append(r["tok_s"])
            cells.append(r)
            log("     %-14s rep %d/%d  %8.3f tok/s   foreign %5.2f%%  (wall %.1f s)"
                % (name, i + 1, reps, r["tok_s"], r["foreign"], r["wall_s"]))
    return by, cells


# ============================================================ self-test
def selftest():
    n = [0]

    def ok(cond, what):
        n[0] += 1
        if not cond:
            raise SystemExit("SELFTEST FAILED: %s" % what)

    ok(median([3, 1, 2]) == 2, "median, odd")
    ok(median([4, 1, 2, 3]) == 2.5, "median, even")
    ok(median([5]) == 5, "median, one")

    lo, hi = boot_ci([10.0] * 8)
    ok(abs(lo - 10.0) < 1e-9 and abs(hi - 10.0) < 1e-9, "bootstrap of a constant is a point")
    lo, hi = boot_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    ok(lo <= 3.0 <= hi and lo < hi, "bootstrap brackets the median and has width")

    lo, hi = boot_ratio_ci([20.0] * 6, [10.0] * 6)
    ok(abs(lo - 2.0) < 1e-9 and abs(hi - 2.0) < 1e-9, "paired ratio bootstrap of constants")
    ok(boot_ratio_ci([1.0], [1.0])[0] != boot_ratio_ci([1.0], [1.0])[0], "one rep gives nan")

    # -- admissibility is a REFUSAL, and the bar is exclusive
    ok(admissible([{"name": "x", "foreign": 1.0}]) == "ADMISSIBLE", "clean cell is admissible")
    ok(admissible([{"name": "x", "foreign": OCC_BAR}]).startswith("VOID"), "the bar is exclusive")
    ok(admissible([{"name": "x", "foreign": 1.0},
                   {"name": "y", "foreign": 9.0}]).startswith("VOID"), "ANY cell over voids it")
    ok("y=9.00%" in admissible([{"name": "x", "foreign": 1.0}, {"name": "y", "foreign": 9.0}]),
       "VOID names the offending cell")
    ok(admissible([]).startswith("VOID"), "no cell is VOID, not admissible")

    # -- G-E63d's three bands, and the boundaries
    ok(g_e63d(37.0, 40.0) == "DESK-MODEL-HELD", "CI entirely above the desk model")
    ok(g_e63d(DESK_MODEL, 40.0) == "DESK-MODEL-HELD", "lower bound exactly at the bar HOLDS")
    ok(g_e63d(30.0, 34.0) == "DESK-MODEL-OPTIMISTIC", "CI entirely below")
    ok(g_e63d(30.0, 40.0) == "DESK-MODEL-UNRESOLVED", "CI spanning the bar")
    ok(g_e63d(30.0, DESK_MODEL) == "DESK-MODEL-UNRESOLVED", "upper bound exactly at the bar")
    ok(g_e63d(float("nan"), 1.0).startswith("DEAD"), "no CI is DEAD")
    ok(len(set([g_e63d(37, 40), g_e63d(30, 34), g_e63d(30, 40)])) == 3, "three distinct verdicts")

    # -- G-E63e states comparators and never a pass
    s = g_e63e(0.65, 0.62, 0.68)
    ok("measured 0.6500" in s, "G-E63e reports the measured ratio")
    ok("0.8974" in s, "G-E63e prints the TOKEN's charged comparator, not the FFN's")
    ok("PASS" not in s.upper() and "FAIL" not in s.upper(), "G-E63e has no pass line")
    ok(abs(CHARGED_RATIO - 1.1144) < 1e-4, "the charged ratio is the token's, not 2.0")
    ok(abs(CHARGED_MB_PACKED - 464.13) < 0.01, "packed charged MB/token")
    ok(abs(CHARGED_MB_INT8 - 517.21) < 0.01, "int8 charged MB/token")
    ok(A10B_CHARGED == 671088640 + 134217728 + 16777216 + A10B_FFN_CHARGED,
       "the decomposition sums to E36's charged figure")

    # -- the arm-asymmetry reporter
    m, a, e = arm_foreign([{"name": "A", "foreign": 2.0}, {"name": "A", "foreign": 4.0},
                           {"name": "B", "foreign": 3.0}, {"name": "B", "foreign": 3.0}])
    ok(abs(m["A"] - 3.0) < 1e-9 and abs(m["B"] - 3.0) < 1e-9, "per-arm means")
    ok(abs(a) < 1e-9, "symmetric arms give zero asymmetry")
    m, a, e = arm_foreign([{"name": "A", "foreign": 12.0}, {"name": "B", "foreign": 2.0}])
    ok(abs(a - 10.0) < 1e-9, "asymmetry is the spread between arm means")
    ok(abs(e - 0.0262) < 1e-9, "E52 converts points into a fraction of rate")
    ok(arm_foreign([{"name": "A", "foreign": 1.0}])[1] == 0.0, "one arm has no asymmetry")

    # -- the precondition really is a precondition
    okc, why = part_a_is_closed()
    ok(isinstance(okc, bool) and isinstance(why, str), "part_a_is_closed returns a reason")

    # -- the artefacts this sweep needs
    for nm, path, _ in PAIR_10B + PAIR_E61C:
        ok(os.path.exists(path), "%s exists (%s)" % (nm, path))
    ok(os.path.exists(ENGINE), "engine exists")

    log("   selftest: %d checks passed" % n[0])
    return n[0]


def save(out):
    os.makedirs(OUTDIR, exist_ok=True)
    out["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(OUTFILE, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)


# ============================================================ main
def main():
    t0 = time.time()
    global REPS, OUTFILE
    interim = "--interim" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--reps" and i + 1 < len(sys.argv):
            REPS = int(sys.argv[i + 1])
    if interim:
        OUTFILE = INTERIM_FILE
    okc, why = part_a_is_closed()
    log("== E63 %s" % ("INTERIM paired ratio (addendum C) -- G-E63d is NOT attempted"
                       if interim else
                       "Part B -- the 10 B cell at one byte per weight, MEASURED"))
    log("   %s" % why)
    if not okc:
        log("   STOP.  A rate on a path whose arithmetic has not been proved measures nothing.")
        return 1
    selftest()

    out = {"brief": BRIEF, "part": ("INTERIM (addendum C)" if interim else "B"),
           "engine": os.path.basename(ENGINE), "precondition": why,
           "non_promotion": ("This reading may NOT adjudicate G-E63d in either direction. "
                             "G-E63d is answered by one clean sweep on an idle box and by "
                             "nothing else; predictions 4 and 5 are scored against that sweep, "
                             "not this one.  G-E63d remains OWED.") if interim else None,
           "constants": {"OCC_BAR": OCC_BAR, "DESK_MODEL": DESK_MODEL, "REPS": REPS,
                         "NTOK": NTOK, "THREADS": THREADS, "BOOT": BOOT, "CI": list(CI)},
           "scope": "Absolute tok/s carries +-5% (instrument).  RATIOS do not.  No rate here is "
                    "bytes divided by a bandwidth (SPEED_LEDGER 59.2).",
           "sweeps": {}, "cells": []}
    save(out)

    # ------------------------------------------------------------ sweep 1: G-E63d / G-E63e
    log("")
    log("-- sweep 1: G-E63d -- A10B carved, one byte per weight against half a byte, k=3")
    log("   %d interleaved reps x %d tokens, paired median + %d-draw bootstrap CI." % (REPS, NTOK, BOOT))
    by, cells = sweep(PAIR_10B)
    out["sweeps"]["A10B"] = by
    out["cells"].extend(cells)
    save(out)

    adm = admissible(cells)
    out["admissibility_A10B"] = adm
    pk, i8 = by["A10B_PACKED"], by["A10B_INT8"]
    m_pk, m_i8 = median(pk), median(i8)
    ci_i8 = boot_ci(i8)
    ci_pk = boot_ci(pk)
    ratio = m_i8 / m_pk if m_pk > 0 else float("nan")
    ci_ratio = boot_ratio_ci(i8, pk)
    fmeans, fasym, fexplains = arm_foreign(cells)
    out["A10B_foreign"] = {"per_arm_mean": fmeans, "asymmetry_points": fasym,
                           "fraction_of_ratio_E52_can_explain": fexplains}
    out["A10B"] = {"median_packed": m_pk, "median_int8": m_i8,
                   "ci_packed": list(ci_pk), "ci_int8": list(ci_i8),
                   "ratio_int8_over_packed": ratio, "ci_ratio": list(ci_ratio),
                   "min_max_packed": [min(pk), max(pk)], "min_max_int8": [min(i8), max(i8)]}
    vd = g_e63d(ci_i8[0], ci_i8[1]) if adm == "ADMISSIBLE" else "VOID -- " + adm
    ve = g_e63e(ratio, ci_ratio[0], ci_ratio[1])
    out["G-E63d"] = vd
    out["G-E63e"] = ve
    save(out)
    log("   packed  median %8.3f  CI [%.3f, %.3f]  range %.3f..%.3f"
        % (m_pk, ci_pk[0], ci_pk[1], min(pk), max(pk)))
    log("   int8    median %8.3f  CI [%.3f, %.3f]  range %.3f..%.3f"
        % (m_i8, ci_i8[0], ci_i8[1], min(i8), max(i8)))
    log("   admissibility: %s" % adm)
    log("   foreign per arm: %s   asymmetry %.2f points -> E52 says the box could "
        "manufacture at most %.4f of the ratio"
        % (", ".join("%s %.2f%%" % (k, v) for k, v in sorted(fmeans.items())),
           fasym, fexplains))
    if interim:
        out["wall_min"] = round((time.time() - t0) / 60.0, 1)
        save(out)
        log("")
        log("== INTERIM reading (addendum C) -- NOT a Part B result")
        log("   G-E63d  : %s" % vd)
        log("   ratio   : %s" % ve)
        log("   E36 measured the packed artefact at 49.96 tok/s (run 2: 51.50) on a CLEAN box;")
        log("   %.4f x 49.96 = %.2f tok/s is an ESTIMATE with both inputs named, not a "
            "measurement." % (ratio, ratio * 49.96))
        log("   G-E63d REMAINS OWED.  %.1f min.  results: %s" % (out["wall_min"], OUTFILE))
        return 0

    # ------------------------------------------------------------ sweep 2: G-E61c's repeat
    log("")
    log("-- sweep 2: G-E61c's owed repeat -- 05b fp32, mvacc 4 / mvacc 1, on a quiet box")
    log("   REPORTED, NOT ADJUDICATED (E40 addendum A): E61's OUT-OF-BAND verdict stands.")
    by2, cells2 = sweep(PAIR_E61C)
    out["sweeps"]["E61C"] = by2
    out["cells"].extend(cells2)
    m4, m1 = median(by2["05B_F32_M4"]), median(by2["05B_F32_M1"])
    r61 = m4 / m1 if m1 > 0 else float("nan")
    ci61 = boot_ratio_ci(by2["05B_F32_M4"], by2["05B_F32_M1"])
    out["E61C_repeat"] = {"median_m4": m4, "median_m1": m1, "ratio": r61, "ci": list(ci61),
                          "admissibility": admissible(cells2),
                          "E61_reading": E61C_READ, "E8_G_Z5": E8_G_Z5,
                          "registered_null_band": list(E61C_NULL),
                          "registered_refutes_at": E61C_REFUTES,
                          "verdict": "NOT ADJUDICATED HERE -- E61's OUT-OF-BAND stands"}
    save(out)
    log("   repeat %.4f  CI [%.4f, %.4f]   (E61 read %.4f, E8's G-Z5 read %.4f; "
        "registered null %.2f-%.2f, refutes at >= %.2f)"
        % (r61, ci61[0], ci61[1], E61C_READ, E8_G_Z5, E61C_NULL[0], E61C_NULL[1], E61C_REFUTES))
    log("   admissibility: %s" % admissible(cells2))

    # ------------------------------------------------------------ sweep 3: the 3 B cells
    log("")
    log("-- sweep 3: the 3 B speed cells E62 could not take.  DESCRIPTIVE AND UNGATED.")
    have = [c for c in CELLS_3B if os.path.exists(c[1])]
    if len(have) < len(CELLS_3B):
        miss = [c[0] for c in CELLS_3B if not os.path.exists(c[1])]
        log("   missing: %s -- reported as missing, not silently dropped" % ", ".join(miss))
        out["missing_3B"] = miss
    if have:
        by3, cells3 = sweep(have, reps=3)
        out["sweeps"]["3B"] = by3
        out["cells"].extend(cells3)
        out["3B"] = dict((k, {"median": median(v), "ci": list(boot_ci(v)),
                              "min_max": [min(v), max(v)]}) for k, v in by3.items())
        out["admissibility_3B"] = admissible(cells3)
        save(out)
        for k in sorted(by3):
            s = out["3B"][k]
            log("   %-12s median %8.3f  CI [%.3f, %.3f]" % (k, s["median"], s["ci"][0], s["ci"][1]))
        log("   admissibility: %s" % out["admissibility_3B"])

    out["wall_min"] = round((time.time() - t0) / 60.0, 1)
    save(out)

    # ------------------------------------------------------------ the reading
    log("")
    log("== gates")
    log("   G-E63d  the 10 B carved cell at one byte : %s" % vd)
    log("   G-E63e  the locality pair (descriptive)  : %s" % ve)
    log("")
    log("   Absolute tok/s carries +-5%; the ratio does not.  No rate above is bytes / bandwidth.")
    log("   %.1f min.  results: %s" % (out["wall_min"], OUTFILE))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    sys.exit(main())
