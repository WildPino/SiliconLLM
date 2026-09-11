#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E26 part B -- what does an ACTIVATED weight cost, in the engine?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E26_WHAT_AN_ACTIVATED_WEIGHT_COSTS.md,
pushed before this file existed.

WHAT THIS MEASURES.  Time only.  The weights are NOISE (synth_export.py) and the router is
random; E3 s4's Gate V1 established that timing on this engine depends on shape and format and
not on values, which is what makes a synthetic shape a legitimate way to price one.  Which k a
real model SURVIVES is E19's and E24's question and nothing here touches it.

THE PLANTED CONTROL, and it is the point of the run.  At k = E the carve keeps every group, so
the carved arm moves the same weights as the dense arm plus the router -- byte-neutral to
-0.47% at T10 and -0.71% at S15, an offset that is REGISTERED, not fitted.  Whatever it loses
BELOW that offset is the carve machinery's own cost: the router matvec, the top-k, the row
list, the row-selected gate/up kernel, and the transposed down kernel's different access
pattern.  That cost is subtracted from every deeper arm's gain before the gain is called a
saving.  Without it, "the carve is fast" and "the transposed layout is fast" are the same
number.

ONE ARTIFACT PER SHAPE.  Every k in the sweep runs off the same file through the engine's
--carve-k, so no cell can be confounded by a different file, a different permutation or a
different random draw.  The dense control is a separate file in the SAME quant==4 container
with FK_DENSE on every layer, so the container is not a variable either.

PROTOCOL.  Idle box, --threads 6, --fuse off everywhere (a carved FFN cannot fuse gate|up),
>= 3 repetitions INTERLEAVED by rep so a thermal drift hits every arm equally, dispersion
printed with every rate.  A contended timing is not a timing -- and the first attempt at this
run PROVED that the hard way: it read S15 dense at 16.21 tok/s where E25 read 29.70 on the
same shape, with dispersions of 20-39%, because the box was running a game and 5.82 of 12
cores were busy.  So idleness is now an INSTRUMENT and not an intention: cpu_busy_pct()
samples the system before the run and after every repetition, the run REFUSES to start on a
busy box, and any sample above the bar marks the whole record contended.

  python e26_carve_cost.py --dir D:/_ktmp/e26 --engine ./donor_engine_e26.exe --reps 3
  python e26_carve_cost.py --smoke                  # S05 only, 1 rep
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_export as SX          # noqa: E402  -- SHAPES and active_weights, one definition

E_GROUPS = 256                     # D0c's partition, and what E19/E23/E24 all measure at
K_GRID = (256, 128, 64, 16, 4)     # 256 is the PLANTED CONTROL: every group kept
SHAPES_FULL = ("S15", "T10")
SHAPES_SMOKE = ("S05",)
NTOK = {"S05": 200, "S15": 120, "T10": 40}
RE_BENCH = re.compile(r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s")


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- the contention witness
# Locale-independent on purpose: the Get-Counter path '\Processor(_Total)\% Processor Time'
# does NOT exist on this box, whose Windows is Italian and localizes counter names.  CIM class
# and property names are not localized.
# THE BAR, DERIVED RATHER THAN TUNED -- and the reasoning is here because the first version of
# this number (12%) was set by intuition and turned out to sit INSIDE this box's noise.
#
# Measured with --selftest on a box with only an IDE and a browser open: the idle floor is
# 11-13% mean with peaks to 28%, and six spinning processes read 67%.  So:
#   * a bar at 12% rejects a genuinely idle desktop for its own background, and
#   * the run this instrument exists to reject had 5.82 of 12 cores = 48% held by a game.
# The bar therefore has to sit between this box's idle floor and that failure.  25% is two
# standard desktop baselines above the floor and half the load that ruined the first attempt.
#
# IT IS A COARSE PRE-FILTER AND NOTHING MORE.  A CPU percentage cannot resolve the few percent
# that separates a good timing from a bad one -- to hold the error under 5% the background would
# have to stay below ~0.3 of a core, which is below the floor of any real desktop.  The
# LOAD-BEARING validity test is the E25 ANCHOR at the bottom of this file: reproduce a rate that
# was measured on an idle box and published, or the record is void whatever this number said.
IDLE_BAR = 25.0          # percent of all logical cores, MEAN; see the derivation above
PS_BUSY = ("$v=@(); 1..%d | %%{ $p = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor"
           " | Where-Object {$_.Name -eq '_Total'}; $v += [int]$p.PercentProcessorTime;"
           " Start-Sleep -Milliseconds 700 }; $m=($v | Measure-Object -Average -Maximum);"
           " \"$($m.Average) $($m.Maximum)\"")


def cpu_busy(samples=6):
    """System-wide CPU busy over `samples` readings ~700 ms apart: (mean, max), -1 if unreadable.

    THE GATE IS THE MEAN, NOT THE MAX, and that is a correction to the first version of this
    instrument rather than a convenience.  What ruins a six-minute timing is SUSTAINED
    contention -- the first attempt had a game holding 5.82 of 12 cores for the whole run.  A
    desktop with an IDE and a browser open never reads zero and spikes to 20% for 700 ms, and
    barring on the max of a few instantaneous samples rejects an idle box for those spikes.  The
    max is still measured and still recorded, because a big spike during a rep is worth seeing.

    Changing a gate in order to pass it is how an instrument stops being one, so this version is
    re-validated on a PLANTED load before it is used -- see `--selftest`.
    """
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        PS_BUSY % samples], capture_output=True)
    try:
        mean_s, max_s = r.stdout.decode(errors="replace").strip().split()
        return float(mean_s), float(max_s)
    except ValueError:
        return -1.0, -1.0


def cpu_busy_pct(samples=3):
    """Back-compat for anything that wants one number: the mean."""
    return cpu_busy(samples)[0]


# The SECOND, independent idleness test, and the stronger one: E25 measured this exact shape in
# this exact container on an idle box and published it.  A CPU percentage is a proxy; reproducing
# a published rate is the thing itself.  Read from e25_rank_cost.json's own S15-PACKED row:
# rates [29.59, 29.44, 30.07], mean 29.70, spread 2.1%.
E25_S15_PACKED = 29.70
ANCHOR_TOL = 0.10


def active(shape, k, head_ternary=True):
    """Active weights/token.  k <= 0 is a DENSE arm (no router, no carve); k == -1 is the same
    weights in the UNTAGGED quant==2 container.  Mirrored from synth_export.active_weights so
    the runner and the writer cannot disagree about what an arm costs."""
    D, F, L, NH, NKV, HD, V, tied, _src = SX.SHAPES[shape]
    if head_ternary:
        tied = 0
    k = max(k, 0)
    return SX.active_weights(D, F, L, NH, NKV, HD, V, E_GROUPS if k else 0, k)


def build(d, shape, carve, seed=1234):
    """carve > 0: the carved quant==4 file.  carve == 0: the matched quant==4 FK_DENSE
    control.  carve == -1: the same shape in the UNTAGGED quant==2 container -- E25's
    continuity control, kept so an E26 number can be read against every earlier one."""
    tag = "%s_%s" % (shape.lower(), "carve%d" % carve if carve > 0 else
                     ("dense" if carve == 0 else "packed"))
    out = os.path.join(d, "e26_%s.bin" % tag)
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  %-14s %s" % (tag, os.path.basename(out)))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", shape,
           "--out", out, "--seed", str(seed)]
    cmd += ["--carve", str(carve)] if carve > 0 else (["--v4"] if carve == 0 else [])
    log("  build %-14s %s" % (tag, " ".join(cmd[2:])))
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(r.stdout.decode(errors="replace")[-2000:])
        log(r.stderr.decode(errors="replace")[-2000:])
        raise SystemExit("synth_export failed for " + tag)
    for ln in r.stdout.decode(errors="replace").splitlines():
        if "GATE V3" in ln:
            log("   " + ln.strip())
    return out


def bench(engine, w, ntok, threads, k=0):
    cmd = [engine, "--weights", w, "--threads", str(threads)]
    if k:
        cmd += ["--carve-k", str(k)]
    cmd += ["--bench", str(ntok)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(r.stderr.decode(errors="replace")[-1500:])
        raise SystemExit("engine failed on " + w)
    m = RE_BENCH.search(r.stdout.decode(errors="replace"))
    if not m:
        log(r.stdout.decode(errors="replace")[-1500:])
        raise SystemExit("could not parse BENCH line")
    return float(m.group(3))


def selftest_witness():
    """THE PLANTED CONTROL ON THE INSTRUMENT ITSELF.

    The gate moved from the max of a few samples to the mean, and a gate that is relaxed in
    order to pass it is not a gate.  So: read the box, then hold 6 of 12 logical cores with
    spinning processes -- the same order of contention the first attempt actually suffered --
    and require the witness to FIRE.  If it does not, the instrument is broken and no timing
    taken with it counts.
    """
    log("== self-test of the contention witness ==")
    m0, p0 = cpu_busy(8)
    log("  quiet:  %.1f%% mean / %.0f%% peak   (bar %.0f%% on the mean)" % (m0, p0, IDLE_BAR))
    # Built line by line so no backslash escape has to survive a shell round-trip.
    spin = "\n".join(["import time", "t = time.time()", "x = 0.0",
                      "while time.time() - t < 30.0:", "    x += 1.0"])
    kids = [subprocess.Popen([sys.executable, "-c", spin],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(6)]
    try:
        m1, p1 = cpu_busy(8)
    finally:
        for k in kids:
            k.kill()
        for k in kids:
            k.wait()
    log("  loaded: %.1f%% mean / %.0f%% peak   (6 spinning processes, 6 of 12 cores)"
        % (m1, p1))
    fires = m1 > IDLE_BAR
    quiet_ok = 0.0 <= m0 <= IDLE_BAR
    log("")
    log("  FIRES ON THE KNOWN-POSITIVE: %s   (reads the quiet box as idle: %s)"
        % ("YES" if fires else "NO -- THE INSTRUMENT IS BROKEN", "yes" if quiet_ok else "no"))
    m2, _ = cpu_busy(6)
    log("  after:  %.1f%% mean  (the load is gone)" % m2)
    return 0 if fires else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e26")
    ap.add_argument("--engine", default=os.path.join(HERE, "donor_engine_e26.exe"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=0, help="0 = per-shape default")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="re-validate the contention witness on a PLANTED load and exit: it "
                         "must read idle, then FIRE while 6 spinning processes hold the box.")
    ap.add_argument("--allow-contended", action="store_true",
                    help="run even though the box is busy.  The record is then stamped "
                         "contended and NO number from it may enter a ledger.")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e26_carve_cost.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest_witness()
    os.makedirs(a.dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    shapes = SHAPES_SMOKE if a.smoke else SHAPES_FULL
    reps = 1 if a.smoke else a.reps
    if reps < 3 and not a.smoke:
        log("REFUSING: a speed claim on this box needs >= 3 repetitions and a dispersion.")
        return 2
    if a.out.endswith(".json") and a.smoke:
        a.out = a.out[:-5] + "_smoke.json"
    busy0, peak0 = cpu_busy(8)
    log("  contention witness: %.1f%% mean / %.0f%% peak of all logical cores busy before the "
        "run (bar %.0f%% on the MEAN)" % (busy0, peak0, IDLE_BAR))
    if busy0 > IDLE_BAR and not a.allow_contended:
        log("")
        log("REFUSING: a contended timing is not a timing.  The first E26 part B attempt read")
        log("  S15 dense at 16.21 tok/s where E25 read 29.50 on the same shape, because the")
        log("  box was busy.  Close what is using the machine and run this again, or pass")
        log("  --allow-contended to produce a record that is explicitly void.")
        return 3
    busy, peaks = [busy0], [peak0]
    t_start = time.time()

    # (tag, shape, k)  -- k == 0 is the dense control, and it is a DIFFERENT file; every k > 0
    # is the same carved file through --carve-k.
    arms = []
    for s in shapes:
        if s == "S15":
            arms.append(("S15-PACKED", s, -1))     # the untagged container, brief s5
        arms.append(("%s-DENSE" % s, s, 0))
        for k in K_GRID:
            arms.append(("%s-K%d" % (s, k), s, k))

    log("== E26 part B: what an activated weight costs ==")
    log("  engine %s   threads %d   reps %d   E=%d"
        % (os.path.basename(a.engine), a.threads, reps, E_GROUPS))
    log("")
    log("  arm            shape     k   %of F   active weights/token   vs its DENSE control")
    for tag, s, k in arms:
        d0 = active(s, 0)
        act = active(s, k)
        log("  %-14s %-6s %4s %6s   %17.4f G   %+7.2f%%"
            % (tag, s, k if k > 0 else "-",
               ("%.1f%%" % (100.0 * k / E_GROUPS)) if k > 0 else "100%",
               act / 1e9, 100.0 * (act - d0) / d0))
    log("")
    paths = {}
    for s in shapes:
        paths[(s, 0)] = build(a.dir, s, 0)
        paths[(s, 1)] = build(a.dir, s, E_GROUPS)
        if any(t[2] == -1 and t[1] == s for t in arms):
            paths[(s, -1)] = build(a.dir, s, -1)
    log("")

    rates = {t: [] for t, _s, _k in arms}
    for rep in range(reps):
        for tag, s, k in arms:
            n = a.tokens or NTOK[s]
            v = bench(a.engine, paths[(s, -1 if k < 0 else (1 if k else 0))], n,
                      a.threads, max(k, 0))
            rates[tag].append(v)
            log("  rep %d  %-14s %7.2f tok/s   (%d tokens)" % (rep + 1, tag, v, n))
        m, pk = cpu_busy(4)
        busy.append(m)
        peaks.append(pk)
        log("  rep %d  contention witness: %.1f%% mean / %.0f%% peak" % (rep + 1, m, pk))
    contended = max(busy) > IDLE_BAR
    log("")
    if contended:
        log("  *** CONTENDED: the witness peaked at %.0f%% against a %.0f%% bar. Every number"
            % (max(busy), IDLE_BAR))
        log("      below is VOID as a timing and none of it may enter a ledger. ***")
        log("")

    rec = {"brief": "BRIEF_E26_WHAT_AN_ACTIVATED_WEIGHT_COSTS.md", "reps": reps,
           "threads": a.threads, "smoke": bool(a.smoke), "E": E_GROUPS,
           "k_grid": list(K_GRID), "arms": {},
           "cpu_busy_mean_pct": busy, "cpu_busy_peak_pct": peaks,
           "idle_bar_pct": IDLE_BAR, "contended": bool(contended),
           "VOID_AS_A_TIMING": bool(contended)}
    log("  arm            active/tok    mean tok/s   spread   vs DENSE   byte prediction   "
        "G weights/s")
    for tag, s, k in arms:
        v = rates[tag]
        mean = sum(v) / len(v)
        spread = (max(v) - min(v)) / mean if mean else 0.0
        c = rates["%s-DENSE" % s]
        cmean = sum(c) / len(c)
        ratio = mean / cmean if cmean else 0.0
        pred = active(s, 0) / float(active(s, k))
        gw = mean * active(s, k) / 1e9
        rec["arms"][tag] = {"shape": s, "k": k, "active_weights_per_token": active(s, k),
                            "rates": v, "mean_tok_s": mean, "spread": spread,
                            "ratio_vs_dense": ratio, "byte_ratio_prediction": pred,
                            "charged_G_weights_per_s": gw}
        log("  %-14s %8.4f G   %9.2f   %5.1f%%   %+8.2f%%   %+8.2f%%   %11.2f"
            % (tag, active(s, k) / 1e9, mean, 100 * spread, 100 * (ratio - 1),
               100 * (pred - 1), gw))
    log("")

    # ---- the planted control, read out explicitly, and the invariant E25 left standing
    for s in shapes:
        ctl = "%s-K%d" % (s, E_GROUPS)
        over = rec["arms"][ctl]["ratio_vs_dense"]
        exp = rec["arms"][ctl]["byte_ratio_prediction"]
        rec.setdefault("overhead", {})[s] = {"measured": over, "byte_expected": exp,
                                             "own_cost": over / exp}
        log("  PLANTED CONTROL %-4s  %s keeps every group: reads %+.2f%%, bytes predict "
            "%+.2f%%" % (s, ctl, 100 * (over - 1), 100 * (exp - 1)))
        log("     -> the carve machinery's OWN cost at this shape is %+.2f%%.  Every deeper"
            % (100 * (over / exp - 1)))
        log("        arm's gain below is gross; net of this is what the bytes bought.")
        for tag, s2, k in arms:
            if s2 != s or k <= 0 or k == E_GROUPS:
                continue
            g = rec["arms"][tag]["ratio_vs_dense"]
            rec["arms"][tag]["ratio_net_of_overhead"] = g / over if over else None
            log("        %-14s gross %+7.2f%%   net of machinery %+7.2f%%   bytes predict "
                "%+7.2f%%" % (tag, 100 * (g - 1), 100 * (g / over - 1),
                              100 * (rec["arms"][tag]["byte_ratio_prediction"] - 1)))
        gws = [rec["arms"]["%s-DENSE" % s]["charged_G_weights_per_s"]] + \
              [rec["arms"]["%s-K%d" % (s, k)]["charged_G_weights_per_s"] for k in K_GRID]
        lo, hi = min(gws), max(gws)
        band = (hi - lo) / (sum(gws) / len(gws))
        rec.setdefault("charged_throughput", {})[s] = {"min": lo, "max": hi,
                                                       "mean": sum(gws) / len(gws),
                                                       "band": band}
        log("     CHARGED THROUGHPUT %-4s  %.2f - %.2f G active weights/s, band %.2f%%  "
            "(E25 read %s)" % (s, lo, hi, 100 * band,
                               "49.59-50.36, band 1.54%" if s == "T10" else
                               "44.41-45.84, band 3.17%" if s == "S15" else "-"))
        log("        A band of a few percent says the engine converts active weights into time")
        log("        at a rate that does not care how they are ARRANGED -- which is what makes")
        log("        2*D*r and 3*D*GSZ*k the right charges.  A band that opens up says a")
        log("        GATHERED weight costs more than a STREAMED one, and every k in every")
        log("        budget table here is optimistic.")
    # ---- THE SECOND IDLENESS TEST, and the stronger one.
    # A CPU percentage is a proxy for "the box was idle".  Reproducing a rate that was measured
    # on an idle box and published IS the thing itself.  E25 ran this exact shape in this exact
    # untagged container and recorded S15-PACKED at 29.70 tok/s (rates 29.59 / 29.44 / 30.07,
    # spread 2.1%).  If this run cannot reproduce that within the +-5% every absolute timing
    # here carries, doubled for safety, then something held the machine and NOTHING in this
    # record is a timing -- whatever the witness said.
    anch = None
    if "S15-PACKED" in rec["arms"]:
        got = rec["arms"]["S15-PACKED"]["mean_tok_s"]
        dev = (got - E25_S15_PACKED) / E25_S15_PACKED
        anch = {"arm": "S15-PACKED", "measured": got, "e25_published": E25_S15_PACKED,
                "rel_dev": dev, "tol": ANCHOR_TOL, "passes": bool(abs(dev) <= ANCHOR_TOL)}
        log("")
        log("  E25 ANCHOR  S15-PACKED reads %.2f tok/s against E25's published %.2f (%+.2f%%, "
            "bar +-%.0f%%) -> %s"
            % (got, E25_S15_PACKED, 100 * dev, 100 * ANCHOR_TOL,
               "PASS" if anch["passes"] else "FAIL"))
        if not anch["passes"]:
            log("  *** THE ANCHOR FAILED.  This run did not reproduce a rate measured on an")
            log("      idle box in the same container at the same shape, so the box was not")
            log("      idle and this record is VOID AS A TIMING regardless of the witness. ***")
            rec["contended"] = True
            rec["VOID_AS_A_TIMING"] = True
    rec["e25_anchor"] = anch

    rec["seconds"] = time.time() - t_start
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (a.out, rec["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
