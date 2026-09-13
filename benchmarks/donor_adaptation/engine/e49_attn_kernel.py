#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E49 -- the attention kernel was never switched on.

Brief: BRIEF_E49_THE_ATTENTION_KERNEL_WAS_NEVER_SWITCHED_ON.md  (fbb977e)
Pushed before this file existed.  Every gate is transcribed from section 4.

donor_engine.c:202 is `static int g_attn=ATTN_SERIAL;` and no runner from E26
onward passes --attn, so E36..E48 all measured SCALAR attention.  --sweep6 sets
every arm to ATTN_AVX4, which is the whole of E48's "2.5x disagreement" with E46.

PARITY FIRST.  avx4 is NOT bit-identical to serial -- E4 published NATS_TOTAL
166667.2449386003 against 166667.1361128952 -- so bit-identity is the wrong test
and end-to-end parity is the right one (Phase 60's law).  If G-E49a fails, every
speed number here is void.

  python e49_attn_kernel.py --selftest
  python e49_attn_kernel.py --phase parity
  python e49_attn_kernel.py --phase speed
  python e49_attn_kernel.py --phase ladder
  python e49_attn_kernel.py                  # parity, then speed, then ladder
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e44_interval import Occupancy, one_rep, log                     # noqa: E402
from e46_context_cost import fit                                     # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e49_attn_kernel.json")
ENGINE = os.path.join(HERE, "donor_engine_e26.exe")

# ---- section 4 -------------------------------------------------------------------
R128 = "D:/_ktmp/e40/e40_r128.bin"
R128_FLAGS = ["--carve-k", "3"]
S15 = "D:/_ktmp/e37/e37_carved_nf.bin"
# ADDENDUM A.  The control was e37_dense_nf.bin and it COULD NOT FIRE: E37's own gate
# G_E37A measured dense against carved-at-k=E at 2.801851617384443e-07 with a 1e-4
# tolerance -- they are equivalent BY CONSTRUCTION.  The replacement is one artifact and
# one flag: E37 measured K256 at 3.4757066520304316 BPB and K3 at 4.029398350226611,
# 0.5537 apart = 5,537x this gate's own bar.  The parity PAIR is pinned to k=256 so both
# kernels are compared at one fixed operating point.
PARITY_K = ["--carve-k", "256"]                    # full carve = E37's kE point
CONTROL_K = ["--carve-k", "3"]                     # the target operating point
IDS = "D:/_ktmp/e1/ids_qwen25-15b_tq.bin"
SEQLEN = 512                                        # e1_bpb_through_engine's own slicing

WINDOWS = [40, 160, 640, 1280, 2560]
HEADLINE_WINDOW = 40                                # the context every published number used
LADDER_WINDOW = 1280
KERNELS = ["serial", "avx4"]
LADDER = ["serial", "serial2", "serial3", "ilp4", "avx1", "avx4"]
REPS = 5
LADDER_REPS = 3
THREADS = 6

# G-E49a.  BPB = NATS/(ln2*bytes), so dBPB = BPB * |dNATS|/NATS.  The brief's bar is
# |dBPB| < 1e-4; with BPB bounded above by 5 on this programme's worst artifact (4.07,
# E37), that bar is IMPLIED by a relative NATS difference below 2e-5.  The mapping is
# written here rather than assumed so it can be audited.
G49A_BPB_BAR = 1e-4
G49A_BPB_UPPER_BOUND = 5.0
G49A_REL_BAR = G49A_BPB_BAR / G49A_BPB_UPPER_BOUND          # 2e-05
G49A_CONTROL_MIN = 1e-3     # the planted control must be at least this far apart
G49C_BAND = 0.05            # the +-5% the ledger already carries
TARGET_TOK_S = 50.0

RE_NATS = re.compile(r"NATS_TOTAL ([0-9.eE+-]+)\s*\nN_PREDICTED (\d+)")


def nats(weights, kernel, flags=()):
    cmd = [ENGINE, "--weights", weights, "--threads", str(THREADS), "--seqlen", str(SEQLEN),
           "--attn", kernel] + list(flags) + ["--bpb", IDS]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True)
    dt = time.time() - t0
    if r.returncode != 0:
        log(r.stderr.decode(errors="replace")[-1200:])
        raise SystemExit("engine failed: " + " ".join(cmd))
    m = RE_NATS.search(r.stdout.decode(errors="replace"))
    if not m:
        log(r.stdout.decode(errors="replace")[-1200:])
        raise SystemExit("no NATS_TOTAL in engine output")
    return float(m.group(1)), int(m.group(2)), dt


def rel(a, b):
    return abs(a - b) / abs(b) if b else float("inf")


def g_e49a(n_serial, n_avx4, n_control):
    """PARITY, with its planted control.  The null does NOT count unless the same
    harness is shown to FIRE on a difference that is really there."""
    r_pair = rel(n_avx4, n_serial)
    r_ctrl = rel(n_control, n_serial)
    fired = bool(r_ctrl >= G49A_CONTROL_MIN)
    passes = bool(r_pair <= G49A_REL_BAR)
    if not fired:
        v = ("VOID", "the planted control did not fire: the harness cannot see a difference "
                     "that IS there, so its null says nothing")
    elif passes:
        v = ("PARITY HOLDS", "avx4 and serial agree to %.3e relative, inside the %.0e that "
                             "|dBPB| < 1e-4 implies" % (r_pair, G49A_REL_BAR))
    else:
        v = ("PARITY FAILS", "avx4 differs from serial by %.3e relative -- every speed number "
                             "in E49 is void" % r_pair)
    return {"nats_serial": n_serial, "nats_avx4": n_avx4, "nats_control": n_control,
            "rel_pair": r_pair, "rel_control": r_ctrl, "rel_bar": G49A_REL_BAR,
            "control_min": G49A_CONTROL_MIN, "control_fired": fired,
            "implied_dbpb_upper": r_pair * G49A_BPB_UPPER_BOUND,
            "verdict": v[0], "reason": v[1]}


def c50(a_ms, b_ms):
    if b_ms <= 0:
        return None
    return (1.0 / TARGET_TOK_S - a_ms / 1000.0) / (b_ms / 1000.0)


def g_e49c(rate_serial_40, rate_avx4_40):
    """Two-sided and registered as such: either the published headline survives inside
    the band the ledger already carries, or E49 restates it."""
    d = (rate_avx4_40 - rate_serial_40) / rate_serial_40
    inside = bool(abs(d) <= G49C_BAND)
    return {"rate_serial": rate_serial_40, "rate_avx4": rate_avx4_40, "delta": d,
            "band": G49C_BAND, "headline_survives": inside,
            "verdict": ("THE HEADLINE SURVIVES -- the kernel is invisible at context 20"
                        if inside else
                        "THE HEADLINE IS RESTATED -- the kernel moves it by more than the "
                        "band the ledger carries")}


def selftest():
    log("== E49 gate self-test -- every decision function in every direction ==")
    ok = [True]

    def check(name, got, want, note):
        good = (got == want)
        ok[0] = ok[0] and good
        log("  %-4s %-26s expected %-26s %s   %s"
            % (name, str(got), str(want), "FIRES" if good else "*** FAILS ***", note))

    N = 166667.1361128952
    # G-E49a, all three outcomes
    check("A1", g_e49a(166667.2449386003, N, N * 1.5)["verdict"], "PARITY HOLDS",
          "E4's OWN two published values: 6.5e-10 apart, and a control 50% away")
    check("A2", g_e49a(N, N * 1.01, N * 1.5)["verdict"], "PARITY FAILS",
          "1% apart is 500x the bar -> the kernels do not compute the same thing")
    check("A3", g_e49a(N, N, N * (1 + 1e-9))["verdict"], "VOID",
          "a control that does NOT fire voids the null, however good the null looks")
    check("A4", g_e49a(N, N * (1 + 1.9e-5), N * 1.5)["verdict"], "PARITY HOLDS",
          "1.9e-5 is inside the 2e-5 the |dBPB| < 1e-4 bar implies")
    check("A5", g_e49a(N, N * (1 + 2.1e-5), N * 1.5)["verdict"], "PARITY FAILS",
          "2.1e-5 is outside it -- the bar is a bar")
    # the mapping from the brief's dBPB bar to the relative bar actually used
    check("A6", "%.0e" % G49A_REL_BAR, "2e-05",
          "1e-4 BPB over a BPB bounded by 5 is 2e-5 relative")
    # G-E49c, both directions
    check("C1", g_e49c(112.7, 115.0)["headline_survives"], True,
          "+2.0% at context 20 -> inside the band the ledger already carries")
    check("C2", g_e49c(112.7, 135.0)["headline_survives"], False,
          "+19.8% -> the headline is restated, not defended")
    check("C3", g_e49c(112.7, 100.0)["headline_survives"], False,
          "the gate must also fire when the kernel makes it WORSE")
    # C50, against both published numbers
    check("D1", "%.0f" % c50(8.327, 0.020307), "575", "E46's own C50, serial")
    check("D2", "%.0f" % c50(8.107, 0.0081821), "1454", "E48's own C50, avx4 under sweep6")
    log("  E49 gate self-test : %s" % ("ALL FIRE" if ok[0] else "*** SOMETHING FAILED ***"))
    if not ok[0]:
        raise SystemExit("a decision function does not behave as registered.  STOP.")


def phase_parity(out):
    log("")
    log("== G-E49a -- PARITY.  Nothing else in E49 counts until this passes. ==")
    for p in (S15, IDS):
        if not os.path.exists(p):
            raise SystemExit("missing %s.  STOP." % p)
    ns, np_, ts = nats(S15, "serial", PARITY_K)
    log("  S15 k=256 serial   NATS_TOTAL %.10f  n=%d  (%.1f s)" % (ns, np_, ts))
    na, _, ta = nats(S15, "avx4", PARITY_K)
    log("  S15 k=256 avx4     NATS_TOTAL %.10f          (%.1f s)" % (na, ta))
    nc, _, tc = nats(S15, "serial", CONTROL_K)
    log("  CONTROL k=3 serial NATS_TOTAL %.10f          (%.1f s)" % (nc, tc))
    g = g_e49a(ns, na, nc)
    log("")
    log("  planted control : %s   (%.3e apart, bar %.0e)"
        % ("FIRES" if g["control_fired"] else "*** DOES NOT FIRE ***",
           g["rel_control"], G49A_CONTROL_MIN))
    log("  avx4 vs serial  : %.3e relative -> |dBPB| <= %.2e   (bar %.0e relative)"
        % (g["rel_pair"], g["implied_dbpb_upper"], G49A_REL_BAR))
    log("  G-E49a : %s" % g["verdict"])
    log("  %s" % g["reason"])
    out["G_E49a"] = g
    out["parity_seconds"] = {"serial": ts, "avx4": ta, "control": tc}
    return g


def phase_speed(out, reps, occ):
    log("")
    log("== G-E49b -- the speed, both kernels, PURE (no --sweep6, no --profile) ==")
    cells = dict((k, dict((w, {"window": w, "mean_pos": w / 2.0, "dt_s": [], "rates": [],
                               "occ": []}) for w in WINDOWS)) for k in KERNELS)
    for i in range(reps):
        for w in WINDOWS:
            for k in KERNELS:                       # kernels interleaved at rep level
                occ.sample()
                v, dt, wall = one_rep(ENGINE, R128, w, THREADS,
                                      R128_FLAGS + ["--attn", k])
                o = occ.sample()
                c = cells[k][w]
                c["dt_s"].append(dt)
                c["rates"].append(v)
                c["occ"].append(o)
                log("    %-7s n=%-5d rep %d/%d  %7.2f tok/s  %8.4f ms/tok  occ %4.1f%%"
                    % (k, w, i + 1, reps, v, dt / w * 1e3, o))
    recs = {}
    for k in KERNELS:
        for w in WINDOWS:
            c = cells[k][w]
            c["median_dt_s"] = statistics.median(c["dt_s"])
            c["median_tok_s"] = statistics.median(c["rates"])
            c["ms_per_tok"] = c["median_dt_s"] / w * 1e3
            c["spread"] = (max(c["dt_s"]) - min(c["dt_s"])) / c["median_dt_s"]
            c["median_occ"] = statistics.median(c["occ"])
        a, b = fit([(cells[k][w]["mean_pos"], cells[k][w]["ms_per_tok"]) for w in WINDOWS])
        recs[k] = {"cells": dict((str(w), cells[k][w]) for w in WINDOWS),
                   "a_ms": a, "b_ms_per_pos": b, "c50": c50(a, b)}
        log("")
        log("  %-7s ms/tok = %.4f + %.6f*pos   ->  C50 = %.0f" % (k, a, b, recs[k]["c50"]))
        for w in WINDOWS:
            c = cells[k][w]
            log("     n=%-5d pos %6.1f  %8.4f ms/tok  %7.2f tok/s  spread %5.1f%%  occ %4.1f%%"
                % (w, c["mean_pos"], c["ms_per_tok"], c["median_tok_s"],
                   100 * c["spread"], c["median_occ"]))
    out["G_E49b"] = recs
    log("")
    log("  b(serial)/b(avx4) = %.3f      C50 %.0f -> %.0f"
        % (recs["serial"]["b_ms_per_pos"] / recs["avx4"]["b_ms_per_pos"],
           recs["serial"]["c50"], recs["avx4"]["c50"]))

    gc = g_e49c(recs["serial"]["cells"][str(HEADLINE_WINDOW)]["median_tok_s"],
                recs["avx4"]["cells"][str(HEADLINE_WINDOW)]["median_tok_s"])
    out["G_E49c"] = gc
    log("")
    log("== G-E49c -- does the published headline move?  (at NTOK=%d, mean position %d) =="
        % (HEADLINE_WINDOW, HEADLINE_WINDOW // 2))
    log("  serial %7.2f tok/s   avx4 %7.2f tok/s   delta %+.1f%%  (band +-%.0f%%)"
        % (gc["rate_serial"], gc["rate_avx4"], 100 * gc["delta"], 100 * G49C_BAND))
    log("  %s" % gc["verdict"])
    return recs


def phase_ladder(out, reps, occ):
    log("")
    log("== G-E49d -- the whole ladder at n=%d, so `avx4 is best` is measured =="
        % LADDER_WINDOW)
    res = {}
    for i in range(reps):
        for k in LADDER:
            occ.sample()
            v, dt, wall = one_rep(ENGINE, R128, LADDER_WINDOW, THREADS,
                                  R128_FLAGS + ["--attn", k])
            res.setdefault(k, []).append(v)
            log("    %-8s rep %d/%d  %7.2f tok/s" % (k, i + 1, reps, v))
    tab = dict((k, {"rates": v, "median_tok_s": statistics.median(v)})
               for k, v in res.items())
    base = tab["serial"]["median_tok_s"]
    order = sorted(tab, key=lambda k: -tab[k]["median_tok_s"])
    for k in tab:
        tab[k]["vs_serial"] = tab[k]["median_tok_s"] / base
    log("")
    log("  %-9s %12s %12s" % ("kernel", "tok/s", "vs serial"))
    for k in order:
        log("  %-9s %12.2f %11.3fx" % (k, tab[k]["median_tok_s"], tab[k]["vs_serial"]))
    log("  fastest : %s" % order[0])
    out["G_E49d"] = {"table": tab, "order": order, "fastest": order[0],
                     "avx4_is_fastest": bool(order[0] == "avx4")}
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["parity", "speed", "ladder", "all"], default="all")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.reps < REPS:
        raise SystemExit("section 4 registers %d reps.  STOP." % REPS)

    log("=" * 84)
    log("E49 -- the attention kernel was never switched on")
    log("  donor_engine.c:202 defaults to ATTN_SERIAL and no runner since E26 passes --attn.")
    log("  PARITY FIRST: if G-E49a does not pass, every speed number below is void.")
    log("=" * 84)
    selftest()

    out = {"brief": "BRIEF_E49_THE_ATTENTION_KERNEL_WAS_NEVER_SWITCHED_ON.md",
           "brief_commit": "fbb977e", "engine": os.path.basename(ENGINE),
           "threads": THREADS, "reps": a.reps, "windows": WINDOWS,
           "weights_speed": R128, "flags_speed": R128_FLAGS,
           "weights_parity": S15, "parity_k": PARITY_K, "control_k": CONTROL_K,
           "ids": IDS, "addendum": "A",
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    occ = Occupancy()
    occ.sample()

    def save():
        if not os.path.isdir(RES):
            os.makedirs(RES)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)

    if a.phase in ("parity", "all"):
        g = phase_parity(out)
        save()
        if g["verdict"] != "PARITY HOLDS":
            raise SystemExit("G-E49a returned %s.  Section 4 says every speed number is "
                             "void.  STOP." % g["verdict"])
    if a.phase in ("speed", "all"):
        phase_speed(out, a.reps, occ)
        save()
    if a.phase in ("ladder", "all"):
        phase_ladder(out, LADDER_REPS, occ)
        save()

    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    log("")
    log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
