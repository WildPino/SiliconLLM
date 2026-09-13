"""E53 -- a real exponential instead of a libm call.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E53_A_REAL_EXPONENTIAL_INSTEAD_OF_A_LIBM_CALL.md

Phases, in the order the brief fixes them.  Nothing downstream runs if an upstream
gate does not pass.

  c0  the REFACTOR did not move the libm arm.  donor_engine_e53.exe --fexp libm must
      reproduce NATS_TOTAL to every digit.  This is not in the brief; it is added by
      addendum A because the brief's design forced a refactor of the two call sites,
      and a refactor that silently changed the control arm would make every later
      comparison meaningless.  It is a regression control with a known-positive
      answer already in hand, which is the strongest kind.
  c1  continuous parity, libm vs poly, bar 1e-3 relative (G-E49a/G-E51a's bar) with
      the same planted control.
  c2  DISCRETE parity: A1 teacher-forced greedy 160/160 with E6's planted controls.
      A faster engine that answers differently has not been sped up.
  d   speed, interleaved and rotated, under OCC_BAR = 4.39 (E52), with E49 addendum
      C's dispersion rule and an E52-style drift witness.

The kernel's own gates G-E53a and G-E53b live in e53_vexpf8_test.c, because they are
a property of the kernel and not of the engine.

Usage:
  python e53_fexp.py --selftest
  python e53_fexp.py --phase c0
  python e53_fexp.py                (all phases, in order)
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e51_math_errno as E51        # E6 greedy machinery + the dispersion rule
import e44_interval as E44          # foreign-occupancy meter + OCC_BAR

OUT = os.path.join(HERE, "results", "e53_fexp.json")

BASE = os.path.join(HERE, "donor_engine_e52.exe")   # frozen pre-E53, == e50 == donor_engine.exe
E53  = os.path.join(HERE, "donor_engine_e53.exe")
THREADS = 6

# ---- c0 / c1 ---------------------------------------------------------------------
S15      = E51.S15
IDS      = E51.IDS
SEQLEN   = E51.SEQLEN
PARITY_K = E51.PARITY_K
CONTROL_K = E51.CONTROL_K
# The pre-E53 value, already published (E49, E51, INDEX).  c0 compares against THIS,
# not against a number produced in the same session, so the control cannot drift with me.
C0_KNOWN = 124963.9517608703
G53C1_REL_BAR = 1e-3            # G-E49a / G-E51a's bar, unchanged
G53C1_CONTROL_MIN = 1e-3        # the k=3 control must exceed it (E51 read 1.593e-01)

# ---- c2 --------------------------------------------------------------------------
G53C2_A1_REQUIRED = E51.G51B_A1_REQUIRED        # 160
G53C2_CONTROL_MAX = E51.G51B_CONTROL_MAX        # 0.90

# ---- d ---------------------------------------------------------------------------
R128       = E51.R128
R128_FLAGS = E51.R128_FLAGS
WINDOWS    = [40, 160, 320, 640, 1280]
REPS       = 5                  # the brief's ">= 5 reps"
SPREAD_TOL = E51.G51C_SPREAD_TOL        # 6%, E49 addendum C
DRIFT_TOL  = 2.3                # E52's G-E52b constant, and E44 run 2's measured dispersion
DRIFT_WINDOW = 160
OCC_BAR    = E44.OCC_BAR        # 4.39, derived by G-E52d


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


# ==================================================================================
# decision functions.  Pure, so the self-test can plant on them.
# ==================================================================================
def g_e53c0(nats_new, known):
    """The refactor must not have moved the libm arm.  Bit-identity or nothing."""
    if nats_new == known:
        return "IDENTICAL", ("libm after the refactor reads %.10f, the pre-E53 value to every"
                             " digit" % nats_new)
    d = abs(nats_new - known) / known
    return "*** MOVED ***", ("libm after the refactor reads %.10f against the pre-E53 %.10f,"
                             " a relative %.3e.  The control arm changed, so no comparison"
                             " against it means anything." % (nats_new, known, d))


def g_e53c1(n_libm, n_poly, n_control):
    """Continuous parity, with the planted control that must fire."""
    rel = abs(n_poly - n_libm) / abs(n_libm)
    crel = abs(n_control - n_libm) / abs(n_libm)
    if crel < G53C1_CONTROL_MIN:
        return ("CONTROL DID NOT FIRE",
                "the k=3 control reads %.3e relative, under the %.0e it must exceed -- the"
                " instrument cannot see a changed model, so its null means nothing"
                % (crel, G53C1_CONTROL_MIN), rel, crel)
    if rel <= G53C1_REL_BAR:
        return ("PARITY HOLDS",
                "poly vs libm %.3e relative, inside the %.0e bar; control fires at %.3e"
                % (rel, G53C1_REL_BAR, crel), rel, crel)
    return ("*** PARITY BROKEN ***",
            "poly vs libm %.3e relative, ABOVE the %.0e bar (control %.3e).  The poly arm is"
            " a different model.  No speed is measured." % (rel, G53C1_REL_BAR, crel), rel, crel)


def g_e53c2(a1, controls):
    """Discrete parity.  a1 = tokens matching out of 160; controls = list of the same."""
    if a1 < G53C2_A1_REQUIRED:
        return ("*** FAILS ***",
                "A1 matches %d/%d.  The poly arm answers differently.  A faster engine that"
                " answers differently has not been sped up." % (a1, G53C2_A1_REQUIRED))
    bad = [c for c in controls if c >= G53C2_A1_REQUIRED * G53C2_CONTROL_MAX]
    if bad:
        return ("CONTROL DID NOT FIRE",
                "a planted control matched %s of %d -- the comparison cannot distinguish a"
                " changed model, so 160/160 proves nothing" % (bad, G53C2_A1_REQUIRED))
    return ("PASS", "A1 %d/%d, planted controls %s -- reproducing E6 to the token"
            % (a1, G53C2_A1_REQUIRED, controls))


def g_e53e(open_rate, close_rate):
    """The drift witness, E52's G-E52b rule with E52's constant."""
    d = 100.0 * abs(close_rate - open_rate) / ((open_rate + close_rate) / 2.0)
    if d <= DRIFT_TOL:
        return "CLEAN", ("the bracketing cells agree to %.2f%%, inside the %.1f%% this"
                         " instrument shows at zero load" % (d, DRIFT_TOL)), d
    return "DRIFT-CONTAMINATED", ("the bracketing cells differ by %.2f%%, above %.1f%% -- the"
                                  " cells are printed and the fit is NOT read as a difference"
                                  " between arms" % (d, DRIFT_TOL)), d


def g_e53d(rows):
    """rows: list of dicts with n, ms_libm, ms_poly, sp_libm, sp_poly.
    Judges each window by E49 addendum C's dispersion rule, then fits both arms."""
    judged = []
    for r in rows:
        if max(r["sp_libm"], r["sp_poly"]) > 100.0 * SPREAD_TOL:
            r["verdict"] = "UNRESOLVABLE"
        else:
            r["verdict"] = "JUDGED"
            judged.append(r)
    if len(judged) < 2:
        return ("C50 NOT COMPUTABLE", judged,
                "%d window(s) survived the dispersion rule and a straight line needs two."
                "  That is the instrument declining to extrapolate, NOT a measurement of the"
                " kernel." % len(judged), None)
    fits = {}
    for arm in ("libm", "poly"):
        xs = [r["n"] / 2.0 for r in judged]          # --bench N decodes at 1..N (E46)
        ys = [r["ms_" + arm] for r in judged]
        nn = float(len(xs))
        mx, my = sum(xs) / nn, sum(ys) / nn
        den = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        a = my - b * mx
        fits[arm] = {"a": a, "b": b, "C50": (20.0 - a) / b if b > 0 else float("nan")}
    return ("FIT", judged,
            "libm  a=%.4f b=%.6f C50=%.0f   ||   poly  a=%.4f b=%.6f C50=%.0f"
            % (fits["libm"]["a"], fits["libm"]["b"], fits["libm"]["C50"],
               fits["poly"]["a"], fits["poly"]["b"], fits["poly"]["C50"]), fits)


# ==================================================================================
def selftest():
    """Every decision function must FIRE on a known-positive before its nulls count."""
    ok = [0]

    def chk(name, cond):
        ok[0] += 1
        log("  %-46s %s" % (name, "fires" if cond else "*** DOES NOT FIRE ***"))
        if not cond:
            raise SystemExit("self-test failed: " + name)

    v, _ = g_e53c0(C0_KNOWN, C0_KNOWN)
    chk("C0-1 identical is IDENTICAL", v == "IDENTICAL")
    v, _ = g_e53c0(C0_KNOWN + 1e-9, C0_KNOWN)
    chk("C0-2 one part in 1e14 is MOVED", v.startswith("***"))
    v, _ = g_e53c0(C0_KNOWN * 1.001, C0_KNOWN)
    chk("C0-3 a real change is MOVED", v.startswith("***"))

    v, _, rel, _ = g_e53c1(100.0, 100.0, 120.0)
    chk("C1-1 identical arms hold", v == "PARITY HOLDS" and rel == 0.0)
    v, _, _, _ = g_e53c1(100.0, 100.00005, 120.0)
    chk("C1-2 5e-7 relative holds", v == "PARITY HOLDS")
    v, _, _, _ = g_e53c1(100.0, 100.5, 120.0)
    chk("C1-3 5e-3 relative BREAKS", v.startswith("***"))
    v, _, _, _ = g_e53c1(100.0, 100.0, 100.00001)
    chk("C1-4 a dead control is caught", v == "CONTROL DID NOT FIRE")

    v, _ = g_e53c2(160, [3, 10])
    chk("C2-1 160/160 with live controls passes", v == "PASS")
    v, _ = g_e53c2(159, [3, 10])
    chk("C2-2 159/160 FAILS", v.startswith("***"))
    v, _ = g_e53c2(160, [3, 150])
    chk("C2-3 a control that matches is caught", v == "CONTROL DID NOT FIRE")

    v, _, d = g_e53e(100.0, 101.0)
    chk("E-1 1%% drift is CLEAN", v == "CLEAN" and abs(d - 0.995) < 0.01)
    v, _, _ = g_e53e(100.0, 105.0)
    chk("E-2 5%% drift is CONTAMINATED", v == "DRIFT-CONTAMINATED")
    v, _, _ = g_e53e(100.0, 97.6)
    chk("E-3 the rule is symmetric in sign", v == "DRIFT-CONTAMINATED")

    # d: a clean synthetic pair where poly's slope is 25% lower
    rows = [{"n": n, "ms_libm": 8.0 + 0.008 * (n / 2.0), "ms_poly": 8.0 + 0.006 * (n / 2.0),
             "sp_libm": 1.0, "sp_poly": 1.0} for n in WINDOWS]
    v, judged, why, fits = g_e53d([dict(r) for r in rows])
    chk("D-1 a clean sweep FITS", v == "FIT" and len(judged) == len(WINDOWS))
    chk("D-2 the fit recovers the planted slopes",
        abs(fits["libm"]["b"] - 0.008) < 1e-9 and abs(fits["poly"]["b"] - 0.006) < 1e-9)
    chk("D-3 the fit recovers the planted intercept", abs(fits["libm"]["a"] - 8.0) < 1e-9)
    noisy = [dict(r) for r in rows]
    for r in noisy[1:]:
        r["sp_poly"] = 40.0
    v, judged, why, fits = g_e53d(noisy)
    chk("D-4 one resolvable window declines to extrapolate", v == "C50 NOT COMPUTABLE")
    half = [dict(r) for r in rows]
    for r in half[2:]:
        r["sp_libm"] = 40.0
    v, judged, why, fits = g_e53d(half)
    chk("D-5 two survivors still fit", v == "FIT" and len(judged) == 2)
    edge = [dict(r) for r in rows]
    for r in edge:
        r["sp_libm"] = 100.0 * SPREAD_TOL
    v, judged, why, fits = g_e53d(edge)
    chk("D-6 exactly at the tolerance is JUDGED", v == "FIT" and len(judged) == len(WINDOWS))

    log("")
    log("  %d of %d fire." % (ok[0], ok[0]))


# ==================================================================================
def nats(engine, flags):
    cmd = ([engine, "--weights", S15, "--threads", str(THREADS), "--seqlen", str(SEQLEN)]
           + list(flags) + ["--bpb", IDS])
    txt, dt = E51.run(cmd, "bpb")
    m = E51.RE_NATS.search(txt)
    if not m:
        raise SystemExit("could not parse NATS_TOTAL")
    cfg = re.search(r"^CONFIG .*$", txt, re.M)
    return float(m.group(1)), int(m.group(2)), dt, (cfg.group(0) if cfg else "NO CONFIG LINE")


def phase_c0(out):
    log("== c0 -- the refactor must not have moved the libm arm ==")
    log("  compared against the PUBLISHED pre-E53 value, not one measured beside it.")
    n, npred, dt, cfg = nats(E53, PARITY_K + ["--fexp", "libm"])
    log("  %s" % cfg)
    log("  e53 --fexp libm   NATS_TOTAL %.10f  n=%d  (%.0f s)" % (n, npred, dt))
    log("  pre-E53 published NATS_TOTAL %.10f" % C0_KNOWN)
    v, why = g_e53c0(n, C0_KNOWN)
    log("")
    log("  c0 : %s" % v)
    log("     %s" % why)
    log("")
    out["c0"] = {"verdict": v, "why": why, "nats": n, "known": C0_KNOWN, "config": cfg}
    return not v.startswith("***")


def phase_c1(out):
    log("== G-E53c1 -- continuous parity, libm vs poly ==")
    nl, npred, t1, cfg_l = nats(E53, PARITY_K + ["--fexp", "libm"])
    log("  %s" % cfg_l)
    log("  libm   k=256   NATS_TOTAL %.10f  n=%d  (%.0f s)" % (nl, npred, t1))
    np_, _, t2, cfg_p = nats(E53, PARITY_K + ["--fexp", "poly"])
    log("  %s" % cfg_p)
    log("  poly   k=256   NATS_TOTAL %.10f         (%.0f s)" % (np_, t2))
    nc, _, t3, _ = nats(E53, CONTROL_K + ["--fexp", "libm"])
    log("  CTRL   k=3     NATS_TOTAL %.10f         (%.0f s)" % (nc, t3))
    v, why, rel, crel = g_e53c1(nl, np_, nc)
    log("")
    log("  G-E53c1 : %s" % v)
    log("     %s" % why)
    log("")
    out["G_E53c1"] = {"verdict": v, "why": why, "rel": rel, "control_rel": crel,
                      "nats_libm": nl, "nats_poly": np_, "nats_control": nc,
                      "config_libm": cfg_l, "config_poly": cfg_p}
    return v == "PARITY HOLDS"


def e6_stage(engine, flags, n_new, prompt_ids, tag):
    """E51's e6_engine_stage, with flags threaded through.  The SCORING is E51's
    (score_against_ref, against E6's stored HuggingFace reference) so the two
    experiments cannot drift apart; only the command construction is local,
    because the extra flag is the only thing that differs."""
    arms = {}
    for name, wp, hf, role in E51.E6_ARMS:
        if not os.path.exists(wp):
            log("  SKIP %s -- %s not on disk" % (name, wp))
            continue
        rec = {"weights": wp, "hf": hf, "role": role, "runs": []}
        for i in range(len(prompt_ids)):
            pfx = os.path.join(E51.TMP, "%s_%s_p%d" % (tag, name, i))
            ids = os.path.join(E51.TMP, "p%d.bin" % i)
            base = [engine, "--weights", wp, "--threads", str(THREADS)] + list(flags)
            r1 = E51.parse_gen(E51.run(base + ["--generate", ids, str(n_new), pfx], "gen")[0])
            r2 = E51.parse_gen(E51.run(base + ["--generate", ids, str(n_new), pfx + "_b"],
                                       "gen b")[0])
            gd = (open(pfx + ".ids.bin", "rb").read() ==
                  open(pfx + "_b.ids.bin", "rb").read())
            for f in (pfx + "_b.ids.bin", pfx + "_b.prefill.bin", pfx + ".prefill.bin"):
                if os.path.exists(f):
                    os.remove(f)
            rec["runs"].append({"prompt": i, "ids": r1["ids"], "G_D": bool(gd),
                                "decode_toks": r1["decode_toks"]})
            log("    %-8s %s p%d  G-D %s  decode %.2f tok/s"
                % (tag, name, i, "PASS" if gd else "FAIL", r1["decode_toks"]))
        arms[name] = rec
    return arms


def phase_c2(out):
    log("== G-E53c2 -- DISCRETE parity.  The argmax is not a mean. ==")
    log("  Scored against E6's stored HuggingFace reference, not against the libm arm:")
    log("  the claim is 160/160 vs HF, and two of my own arms agreeing proves nothing.")
    e6 = json.load(open(os.path.join(E51.E6_RES, "engine.json")))
    ref = json.load(open(os.path.join(E51.E6_RES, "ref.json")))
    n_new = e6["n_new"]
    E51.write_prompt_ids(e6["prompt_ids"])
    res_dir = os.path.join(HERE, "results", "e53")
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    arms = e6_stage(E53, ["--fexp", "poly"], n_new, e6["prompt_ids"], "e53poly")
    json.dump({"n_new": n_new, "threads": THREADS, "arms": arms},
              open(os.path.join(res_dir, "engine_e53poly.json"), "w"), indent=1)
    res = E51.score_against_ref(arms, ref, n_new)
    log("")
    log("  %-4s %-12s %-12s %s" % ("arm", "E6", "E53 poly", "role"))
    for name, wp, hf, role in E51.E6_ARMS:
        if name in res:
            e6m = 160 if name == "A1" else E51.E6_CONTROL_MATCHED[name]
            log("  %-4s %-12s %-12s %s"
                % (name, "%d/160" % e6m, "%d/160" % res[name]["matched"], role))
    controls = sorted(res[k]["matched"] for k in res if k != "A1")
    v, why = g_e53c2(res["A1"]["matched"], controls)
    log("")
    log("  G-E53c2 : %s" % v)
    log("     %s" % why)
    log("")
    out["G_E53c2"] = {"verdict": v, "why": why, "a1": res["A1"]["matched"],
                      "controls": controls, "summary": res}
    return v == "PASS"


def one_bench(engine, flags, n):
    """One repetition, with the FOREIGN occupancy meter G-E44b2 defines."""
    sp = E44.Split()
    cmd = [engine, "--weights", R128, "--threads", str(THREADS)] + R128_FLAGS \
        + list(flags) + ["--bench", str(n)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sout, serr = p.communicate()
    child = E44._process_times(int(p._handle))
    if p.returncode != 0:
        sys.stderr.write(serr.decode("utf-8", "replace")[-1500:])
        raise SystemExit("engine failed: " + " ".join(cmd))
    occ, foc = sp.close(child)
    m = E44.RE_BENCH.search(sout.decode("utf-8", "replace"))
    if not m:
        raise SystemExit("no BENCH line: " + " ".join(cmd))
    return float(m.group(3)), occ, foc


def phase_d(out):
    log("== G-E53d -- the speed, interleaved and rotated, under OCC_BAR = %.2f (E52) ==" % OCC_BAR)
    log("  %d windows x %d reps x 2 arms.  The ARM ORDER alternates each repetition so arm")
    log("  and time are not the same axis -- E51 A.4 is why, E52 is the design that worked.")
    log("")

    r_open, _, foc_open = one_bench(E53, ["--fexp", "libm"], DRIFT_WINDOW)
    log("  drift witness opens  libm n=%d  %.2f tok/s  foreign %.1f%%"
        % (DRIFT_WINDOW, r_open, foc_open))
    log("")

    cells = {}
    breached = []
    for n in WINDOWS:
        for arm in ("libm", "poly"):
            cells[(n, arm)] = []
        for rep in range(REPS):
            order = ("libm", "poly") if rep % 2 == 0 else ("poly", "libm")
            for arm in order:
                rate, occ, foc = one_bench(E53, ["--fexp", arm], n)
                cells[(n, arm)].append(rate)
                if foc == foc and foc > OCC_BAR:
                    breached.append((n, arm, rep + 1, foc))
                log("    n=%-5d %-4s rep %d/%d  %7.2f tok/s  system %5.1f%%  foreign %5.1f%%%s"
                    % (n, arm, rep + 1, REPS, rate, occ, foc,
                       "   <-- ABOVE THE BAR" if (foc == foc and foc > OCC_BAR) else ""))
        log("")

    r_close, _, foc_close = one_bench(E53, ["--fexp", "libm"], DRIFT_WINDOW)
    log("  drift witness closes libm n=%d  %.2f tok/s  foreign %.1f%%"
        % (DRIFT_WINDOW, r_close, foc_close))
    log("")

    dv, dwhy, drift = g_e53e(r_open, r_close)
    log("  G-E53e (drift)   : %s" % dv)
    log("     %s" % dwhy)
    log("")
    if breached:
        log("  OCCUPANCY: %d cell(s) read above the %.2f%% bar: %s"
            % (len(breached), OCC_BAR, breached[:8]))
        log("     Those cells are recorded and NOT citable (E52 / G-E44b2).")
        log("")

    rows = []
    for n in WINDOWS:
        r = {"n": n}
        for arm in ("libm", "poly"):
            v = cells[(n, arm)]
            med = statistics.median(v)
            r["ms_" + arm] = 1000.0 / med
            r["rate_" + arm] = med
            r["sp_" + arm] = 100.0 * (max(v) - min(v)) / med
            r["all_" + arm] = v
        r["delta_pct"] = 100.0 * (r["rate_poly"] / r["rate_libm"] - 1.0)
        rows.append(r)

    log("  n      mean pos   libm tok/s (sp)    poly tok/s (sp)    delta   in the fit?")
    v, judged, why, fits = g_e53d(rows)
    for r in rows:
        log("  %-6d %-10.0f %7.2f (%4.1f%%)     %7.2f (%4.1f%%)   %+6.2f%%  %s"
            % (r["n"], r["n"] / 2.0, r["rate_libm"], r["sp_libm"],
               r["rate_poly"], r["sp_poly"], r["delta_pct"], r["verdict"]))
    log("")
    log("  G-E53d : %s" % v)
    log("     %s" % why)
    log("")
    out["G_E53e"] = {"verdict": dv, "why": dwhy, "drift_pct": drift,
                     "open": r_open, "close": r_close}
    out["G_E53d"] = {"verdict": v, "why": why, "rows": rows, "fits": fits,
                     "breached": breached, "occ_bar": OCC_BAR, "reps": REPS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--phase", default=None, choices=["c0", "c1", "c2", "d"])
    a = ap.parse_args()

    log("E53 -- a real exponential instead of a libm call")
    log("brief: BRIEF_E53_A_REAL_EXPONENTIAL_INSTEAD_OF_A_LIBM_CALL.md")
    log("")
    log("== self-test.  Every decision function must fire on a known-positive. ==")
    selftest()
    log("")
    if a.selftest:
        return

    if not os.path.isdir(os.path.dirname(OUT)):
        os.makedirs(os.path.dirname(OUT))
    out = {"brief": "BRIEF_E53_A_REAL_EXPONENTIAL_INSTEAD_OF_A_LIBM_CALL.md",
           "base": BASE, "e53": E53, "threads": THREADS, "windows": WINDOWS, "reps": REPS}

    phases = [a.phase] if a.phase else ["c0", "c1", "c2", "d"]
    for ph in phases:
        if ph == "c0":
            if not phase_c0(out) and not a.phase:
                json.dump(out, open(OUT, "w"), indent=1)
                raise SystemExit("c0 failed -- the refactor moved the control arm.  STOP.")
        elif ph == "c1":
            if not phase_c1(out) and not a.phase:
                json.dump(out, open(OUT, "w"), indent=1)
                raise SystemExit("G-E53c1 did not hold -- no speed is measured.  STOP.")
        elif ph == "c2":
            if not phase_c2(out) and not a.phase:
                json.dump(out, open(OUT, "w"), indent=1)
                raise SystemExit("G-E53c2 did not pass -- the poly arm answers differently."
                                 "  No speed is measured.  STOP.")
        elif ph == "d":
            phase_d(out)
        json.dump(out, open(OUT, "w"), indent=1)
    log("wrote " + OUT)


if __name__ == "__main__":
    main()
