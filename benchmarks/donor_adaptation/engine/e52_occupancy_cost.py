# -*- coding: utf-8 -*-
"""E52 -- what does a busy box cost, and where should OCC_BAR be?

Brief: BRIEF_E52_WHAT_DOES_A_BUSY_BOX_COST.md, pushed before this file existed.

This measures the INSTRUMENT, not the engine.  No number it prints may be quoted as a speed
result for any arm.  The one arm it runs is the standing one so the y-axis is in familiar units.

    python e52_occupancy_cost.py            # the run
    python e52_occupancy_cost.py --selftest # decision functions only, no engine
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e44_interval as E44                                                   # noqa: E402

ENGINE = os.path.join(HERE, "donor_engine.exe")
WEIGHTS = "D:/_ktmp/e40/e40_r128.bin"
FLAGS = ["--carve-k", "3"]
NTOK = 300
THREADS = 6

LEVELS = [0, 1, 2, 3, 5, 8]        # busy single-core processes
REPS = 3
OUT = os.path.join(HERE, "results", "e52_occupancy_cost.json")

# --- the constants the brief fixed, before any cell existed -------------------------------
G52A_ZERO_MAX = 5.0        # section 4: L=0 must read below this
G52B_DRIFT_TOL = 2.3       # E44 run 2's measured rep-to-rep dispersion at 3.8% foreign
G52C_SPREAD_TOL = 6.0      # E49 addendum C's rule, in percent
G52C_MIN_LEVELS = 3
G52D_COST_BUDGET = G52B_DRIFT_TOL / 2.0     # "at most HALF the dispersion at zero load"


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


# ---- decision functions.  They decide; the measuring code never does. ---------------------
def g_e52a(med_by_level):
    """The knob must work and the meter must see it.  Control: nothing is read if this fails."""
    levels = sorted(med_by_level)
    f = [med_by_level[l] for l in levels]
    if f[0] > G52A_ZERO_MAX:
        return ("FAILS", "L=0 reads %.1f%% foreign, above the %.1f%% this control requires -- "
                         "the box was not quiet and nothing here is readable"
                         % (f[0], G52A_ZERO_MAX))
    if any(b < a for a, b in zip(f, f[1:])):
        return ("FAILS", "foreign occupancy is not monotone in L: %s -- the meter cannot see "
                         "load it was deliberately given" % ["%.1f" % x for x in f])
    return ("FIRES", "L=0 reads %.1f%% and the median rises monotonically to %.1f%% at L=%d"
                     % (f[0], f[-1], levels[-1]))


def g_e52b(open_rate, close_rate):
    """The drift witness.  E51 A.4 is why this exists."""
    d = 100.0 * abs(close_rate - open_rate) / open_rate
    if d <= G52B_DRIFT_TOL:
        return ("CLEAN", d, "the bracketing L=0 cells agree to %.2f%%, inside the %.1f%% "
                            "dispersion this instrument shows at zero load" % (d, G52B_DRIFT_TOL))
    return ("DRIFT-CONTAMINATED", d,
            "the bracketing L=0 cells differ by %.2f%%, beyond %.1f%% -- this run cannot "
            "separate load from time, and NO BAR is set from it" % (d, G52B_DRIFT_TOL))


def spread_pct(v):
    return 100.0 * (max(v) - min(v)) / statistics.median(v)


def g_e52c(cells):
    """cells: {L: {"rate": [...], "foreign": [...]}} -> (verdict, k, r0, rows).

    A level whose reps disperse past the tolerance takes no verdict and leaves the fit.
    """
    rows, pts = [], []
    for l in sorted(cells):
        r, f = cells[l]["rate"], cells[l]["foreign"]
        sp = spread_pct(r)
        keep = sp <= G52C_SPREAD_TOL
        rows.append({"L": l, "rate": statistics.median(r), "foreign": statistics.median(f),
                     "spread": sp, "kept": keep})
        if keep:
            pts.append((statistics.median(f), statistics.median(r)))
    if len(pts) < G52C_MIN_LEVELS:
        return ("NOT COMPUTABLE", None, None, rows)
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return ("NOT COMPUTABLE", None, None, rows)
    slope = (n * sxy - sx * sy) / den
    r0 = (sy - slope * sx) / n
    if r0 <= 0:
        return ("NOT COMPUTABLE", None, None, rows)
    k = -100.0 * slope / r0          # percent of rate lost per point of foreign occupancy
    return ("FIT", k, r0, rows)


def g_e52d(k):
    """The bar, by the rule section 4 fixed before the data."""
    if k is None or k <= 0:
        return (None, "k is not positive or not computable -- no bar is derivable, and the "
                      "present OCC_BAR stands until something measures it")
    bar = G52D_COST_BUDGET / k
    return (bar, "cost budget %.2f%% of rate / k %.3f%% per point = %.2f%% foreign"
                 % (G52D_COST_BUDGET, k, bar))


# ---- self-test: every decision function, in every direction -------------------------------
def selftest():
    rows = []

    def chk(name, got, want, why):
        ok = (got == want) if not isinstance(want, float) else abs(got - want) < 1e-6
        rows.append(ok)
        log("  %-6s %-22s expected %-22s %-6s %s"
            % (name, str(got)[:22], str(want)[:22], "FIRES" if ok else "FAILS", why))

    chk("A1", g_e52a({0: 2.0, 1: 9.0, 2: 17.0, 8: 60.0})[0], "FIRES", "monotone and quiet at 0")
    chk("A2", g_e52a({0: 7.0, 1: 9.0, 2: 17.0})[0], "FAILS", "the box was not quiet at L=0")
    chk("A3", g_e52a({0: 2.0, 1: 17.0, 2: 9.0})[0], "FAILS",
        "not monotone -- the meter cannot see load it was given")

    chk("B1", g_e52b(110.0, 109.0)[0], "CLEAN", "0.91% is inside 2.3%")
    chk("B2", g_e52b(110.0, 100.0)[0], "DRIFT-CONTAMINATED",
        "9.1% -- E51 A.4's failure mode, refused in code")

    tight = {0: {"rate": [110.0, 110.5, 111.0], "foreign": [2.0, 2.1, 2.2]},
             2: {"rate": [105.0, 105.5, 106.0], "foreign": [17.0, 17.1, 17.2]},
             5: {"rate": [98.0, 98.5, 99.0], "foreign": [40.0, 40.1, 40.2]}}
    v, k, r0, rr = g_e52c(tight)
    chk("C1", v, "FIT", "three tight levels fit")
    # the slope is -0.3148 tok/s per point on an intercept of 111.06, so k is 0.283 PERCENT
    # per point -- not 0.31, which is the raw slope.  This check caught that arithmetic.
    chk("C2", round(k, 3), 0.283, "slope -0.3148 tok/s/pt over r0 111.06 = 0.283%% of rate/pt")
    chk("C2b", round(r0, 2), 111.06, "and the intercept the percentage is taken against")
    loose = {0: {"rate": [110.0, 90.0, 130.0], "foreign": [2.0, 2.1, 2.2]},
             2: {"rate": [105.0, 85.0, 125.0], "foreign": [17.0, 17.1, 17.2]},
             5: {"rate": [98.0, 80.0, 118.0], "foreign": [40.0, 40.1, 40.2]}}
    chk("C3", g_e52c(loose)[0], "NOT COMPUTABLE",
        "every level disperses past 6%% -- nothing survives to fit")
    two = dict(tight)
    two[2] = loose[2]
    chk("C4", g_e52c(two)[0], "NOT COMPUTABLE", "two surviving levels is not three")

    chk("D1", round(g_e52d(1.0)[0], 3), 1.15, "budget 1.15 / k 1.0")
    chk("D2", round(g_e52d(0.5)[0], 3), 2.3, "a cheaper box buys a looser bar")
    chk("D3", g_e52d(None)[0], None, "no k, no bar -- the present one stands")
    chk("D4", g_e52d(-0.4)[0], None, "a NEGATIVE cost sets no bar either")

    bad = rows.count(False)
    log("\n  E52 gate self-test : %s  (%d checks)"
        % ("ALL FIRE" if not bad else "%d FAILED" % bad, len(rows)))
    return 1 if bad else 0


# ---- measurement.  It measures and prints; it decides nothing. ----------------------------
BURN = ("import time\n"
        "t=time.time()+%f\n"
        "x=0.0\n"
        "while time.time()<t:\n"
        "    x+=1.0000001\n")


def cell(load, seconds=30.0):
    """One (rate, foreign) pair at `load` busy single-core processes."""
    procs = []
    try:
        for _ in range(load):
            procs.append(subprocess.Popen([sys.executable, "-c", BURN % seconds],
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL))
        if load:
            time.sleep(1.5)                       # let them get going
        sp = E44.Split()
        rate, dt, wall, child = E44.one_rep(ENGINE, WEIGHTS, NTOK, THREADS, FLAGS)
        sysp, foreign = sp.close(child)
        return rate, sysp, foreign
    finally:
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()

    log("=" * 84)
    log("E52 -- what does a busy box cost, and where should the bar be?")
    log("  This measures the INSTRUMENT.  No number here is a speed result for any arm.")
    log("=" * 84)
    log("== E52 gate self-test -- every decision function in every direction ==")
    if selftest():
        raise SystemExit("a decision function did not fire.  Nothing is measured.  STOP.")
    log("")
    if a.selftest:
        return

    out = {"brief": "BRIEF_E52_WHAT_DOES_A_BUSY_BOX_COST.md", "engine": ENGINE,
           "levels": LEVELS, "reps": a.reps, "ntok": NTOK, "threads": THREADS}

    # --- the drift witness opens ----------------------------------------------------------
    log("== the drift witness opens (L=0, outside the rotation) ==")
    r_open, s_open, f_open = cell(0)
    log("   open   L=0   %7.2f tok/s   system %5.1f%%   foreign %5.1f%%"
        % (r_open, s_open, f_open))
    log("")

    # --- the rotated sweep ----------------------------------------------------------------
    log("== the sweep.  The level order ROTATES each repetition so level and time are not")
    log("   the same axis -- E51 A.4 is why (this box's level drifts down under load).")
    cells = {l: {"rate": [], "foreign": [], "system": []} for l in LEVELS}
    for rep in range(a.reps):
        order = LEVELS[rep % len(LEVELS):] + LEVELS[:rep % len(LEVELS)]
        log("   rep %d/%d  order %s" % (rep + 1, a.reps, order))
        for l in order:
            r, sy, f = cell(l)
            cells[l]["rate"].append(r)
            cells[l]["foreign"].append(f)
            cells[l]["system"].append(sy)
            log("      L=%-2d  %7.2f tok/s   system %5.1f%%   foreign %5.1f%%" % (l, r, sy, f))
    log("")

    # --- the drift witness closes ---------------------------------------------------------
    r_close, s_close, f_close = cell(0)
    log("== the drift witness closes ==")
    log("   close  L=0   %7.2f tok/s   system %5.1f%%   foreign %5.1f%%"
        % (r_close, s_close, f_close))
    log("")

    # --- gates ----------------------------------------------------------------------------
    med_f = {l: statistics.median(cells[l]["foreign"]) for l in LEVELS}
    va, wa = g_e52a(med_f)
    log("  G-E52a (control) : %s" % va)
    log("     %s" % wa)
    out["G_E52a"] = {"verdict": va, "why": wa, "foreign_median": med_f}
    if va != "FIRES":
        json.dump(out, open(OUT, "w"), indent=1)
        raise SystemExit("the control did not fire.  Nothing below it is read.  STOP.")
    log("")

    vb, d, wb = g_e52b(r_open, r_close)
    log("  G-E52b (drift)   : %s" % vb)
    log("     %s" % wb)
    out["G_E52b"] = {"verdict": vb, "drift_pct": d, "why": wb,
                     "open": r_open, "close": r_close}
    log("")

    vc, k, r0, rows = g_e52c(cells)
    log("  %-4s %9s %10s %9s  %s" % ("L", "tok/s", "foreign", "spread", "in the fit?"))
    for r in rows:
        log("  %-4d %8.2f  %8.1f%%  %8.1f%%  %s"
            % (r["L"], r["rate"], r["foreign"], r["spread"],
               "yes" if r["kept"] else "NO -- disperses past %.0f%%" % G52C_SPREAD_TOL))
    log("")
    log("  G-E52c (cost)    : %s" % vc)
    if vc == "FIT":
        log("     rate = %.2f * (1 - %.4f/100 * foreign)   ->  k = %.3f%% of rate per point"
            % (r0, k, k))
    out["G_E52c"] = {"verdict": vc, "k_pct_per_point": k, "r0": r0, "rows": rows}
    log("")

    bar, wd = g_e52d(k if vc == "FIT" else None)
    if vb != "CLEAN":
        log("  G-E52d (the bar) : NOT SET -- the run is %s and the brief forbids setting a bar"
            % vb)
        log("     from a run that cannot separate load from time.  The derivable value would")
        log("     have been: %s" % wd)
        out["G_E52d"] = {"bar": None, "why": "drift-contaminated run", "would_have_been": wd}
    else:
        log("  G-E52d (the bar) : %s"
            % ("%.2f%% foreign occupancy" % bar if bar is not None else "NOT SET"))
        log("     %s" % wd)
        if bar is not None:
            log("     present OCC_BAR is %.1f%%.  This rule was fixed before the data and its"
                % E44.OCC_BAR)
            log("     answer is taken whichever way it points.")
        out["G_E52d"] = {"bar": bar, "why": wd, "previous": E44.OCC_BAR}
    log("")

    out["cells"] = cells
    json.dump(out, open(OUT, "w"), indent=1)
    log("wrote " + OUT)


if __name__ == "__main__":
    main()
