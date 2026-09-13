#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E44 -- the ONLY surviving question: what is the INTERVAL on a tok/s from this engine?

Brief: `docs/research/donor_adaptation/briefs/BRIEF_E44_WHAT_IS_A_TOK_S_WORTH.md`,
addendum A.  E44 began as a three-arm page-cache residency experiment.  Addendum A
killed that from the program text before a rep was spent: `load()` is `xmalloc` plus a
chunked `fread` with no mapping anywhere, `fread` writes every byte so no page is left
to fault, and `t0` is taken AFTER the read, AFTER `state_init` and AFTER a warm forward.
No file is read inside the timed window, so the page cache cannot explain variance
measured in it.  `G-E44a`, `G-E44c` and `G-E44d` are retired with the arms.

WHAT IS LEFT is `G-E44b`, and it is the whole of E44:

    every rate is reported with dispersion or not at all.
    >= 5 reps, idle box, median AND min-max spread.  A single number may not appear.

WHY THIS FILE EXISTS AT ALL, rather than a shell loop.  E43 measured 40 cells and found
9-22% dispersion on the quietest box this programme has ever had (1.0-13.3% occupancy),
and separately found that a CONTENDED reading was off by 35-41%.  The standing rule --
"a contended timing is not a timing" -- is therefore correct and not self-enforcing: it
lives in a memory file and in a brief, and the operator is the one who has to remember
it at the moment they are impatient.  So the rule is moved INTO the instrument.  This
runner measures the box and REFUSES rather than producing a number it would have to
disown later.

THE GUARD IS ITSELF GATED, because a guard that cannot fire is decoration.  Before any
rep, `--planted-control` deliberately loads the machine with its own short-lived busy
processes and requires the guard to REFUSE.  Only then do the guard's PASSES mean
anything.  This is the planted-control law applied to a guard instead of to a
measurement, and it is the same law: an instrument must be shown to fire on a known
positive before its nulls count.

Occupancy comes from `GetSystemTimes`, not from a performance counter: the counter path
(`\\Processor(_Total)\\% Processor Time`) is LOCALISED and fails outright on this box,
which runs an Italian Windows.  An instrument that silently returns nothing on a
localised machine is exactly the kind of plausible artefact that would turn "the box was
quiet" into an unchecked assumption.

NOTHING HERE MEASURES QUALITY.  No BPB, no parity.  And nothing here may mint a new
headline rate: addendum A says a re-quote replaces a point with a band and names the arm
it came from.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results")
RE_BENCH = re.compile(r"BENCH\s+(\d+)\s+tokens\s+([\d.]+)\s+s\s+([\d.]+)\s+tok/s")

# E43 measured its quiet sessions at 1.0-13.3% occupancy and still saw 9-22% dispersion.
# The bar is set just above that band: a box quieter than this is as quiet as the
# quietest conditions this programme has ever achieved, and a reading taken above it is
# not comparable with anything already published.
OCC_BAR = 15.0
MIN_REPS = 5          # G-E44b's own number, not tunable downward -- see main()


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


class _FT(ctypes.Structure):
    _fields_ = [("lo", wt.DWORD), ("hi", wt.DWORD)]

    def val(self):
        return (self.hi << 32) | self.lo


def _system_times():
    idle, kern, user = _FT(), _FT(), _FT()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern),
                                                 ctypes.byref(user)):
        raise SystemExit("GetSystemTimes failed -- no occupancy instrument, and without one "
                         "'the box was quiet' is an assumption, not a measurement.  STOP.")
    return idle.val(), kern.val(), user.val()


class Occupancy(object):
    """System-wide busy fraction between successive reads.  `kern` INCLUDES idle on
    Windows, which is the documented trap here: busy = 1 - idle/(kern+user)."""

    def __init__(self):
        self.prev = _system_times()

    def sample(self):
        cur = _system_times()
        di = cur[0] - self.prev[0]
        dt = (cur[1] - self.prev[1]) + (cur[2] - self.prev[2])
        self.prev = cur
        if dt <= 0:
            return float("nan")
        return 100.0 * (1.0 - float(di) / float(dt))


def watch(seconds, every=1.0):
    """Occupancy samples over a window.  Returns the list, never a single number."""
    o = Occupancy()
    time.sleep(every)
    out = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        out.append(o.sample())
        time.sleep(every)
    return [x for x in out if x == x]


def summarise(v):
    return (statistics.median(v), min(v), max(v))


BURN = ("import time\n"
        "t=time.time()+%f\n"
        "x=0.0\n"
        "while time.time()<t:\n"
        "    x+=1.0000001\n")


def planted_control(seconds, bar):
    """The guard must REFUSE on a deliberately loaded box, or its passes mean nothing."""
    log("")
    log("  PLANTED CONTROL for the occupancy guard -- loading the box on purpose")
    n = os.cpu_count() or 4
    procs = []
    try:
        for _ in range(n):
            procs.append(subprocess.Popen([sys.executable, "-c", BURN % (seconds + 4.0)],
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL))
        v = watch(seconds)
        med, lo, hi = summarise(v)
        fires = med > bar
        log("     %d busy processes -> occupancy median %.1f%% (min %.1f, max %.1f), bar %.1f%%"
            % (n, med, lo, hi, bar))
        log("     guard REFUSES on a loaded box : %s" % ("FIRES" if fires else "*** FAILS ***"))
    finally:
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                pass
    if not fires:
        raise SystemExit("The occupancy guard did NOT refuse a box loaded on every core.  Its "
                         "passes would be meaningless, so no rate may be taken through it.  STOP.")
    # let the machine settle before anything is measured through the guard
    time.sleep(3.0)
    return {"busy_procs": n, "median": med, "min": lo, "max": hi, "fires": True}


def one_rep(engine, weights, ntok, threads, flags):
    """One engine invocation.  Returns tok/s, the engine's OWN dt, and the wall time.

    wall - dt is everything outside the timed window: the fread of the weight file from
    D: (a USB 3.1 external HDD -- addendum A section 4), the blob allocation, state_init
    and the warm forward.  It is reported because addendum A added it and because no
    rate in this programme has ever carried it."""
    cmd = [engine, "--weights", weights, "--threads", str(threads)] + list(flags) + \
          ["--bench", str(ntok)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True)
    wall = time.time() - t0
    if r.returncode != 0:
        log(r.stderr.decode(errors="replace")[-1200:])
        raise SystemExit("engine failed: " + " ".join(cmd))
    m = RE_BENCH.search(r.stdout.decode(errors="replace"))
    if not m:
        log(r.stdout.decode(errors="replace")[-1200:])
        raise SystemExit("could not parse the BENCH line")
    return float(m.group(3)), float(m.group(2)), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(HERE, "donor_engine.exe"))
    ap.add_argument("--weights", default="D:/_ktmp/e40/e40_r128.bin")
    ap.add_argument("--arm", default="R128",
                    help="the NAME of the arm, recorded beside the number so a band is "
                         "never quoted without saying what it is a band of")
    ap.add_argument("--ntok", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=MIN_REPS)
    ap.add_argument("--occ-bar", type=float, default=OCC_BAR)
    ap.add_argument("--guard-seconds", type=float, default=20.0)
    ap.add_argument("--no-planted-control", action="store_true",
                    help="skip the guard's own control.  Then the guard's passes are "
                         "undemonstrated and the run is marked as such.")
    ap.add_argument("--flags", default="", help="extra engine flags, space separated")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "e44_interval.json"))
    a = ap.parse_args()
    flags = [x for x in a.flags.split(" ") if x]
    os.makedirs(OUTDIR, exist_ok=True)
    t_start = time.time()

    log("== E44 -- the interval on a tok/s ==  arm %s, %d tokens, %d threads"
        % (a.arm, a.ntok, a.threads))
    log("   brief: BRIEF_E44_WHAT_IS_A_TOK_S_WORTH.md addendum A -- G-E44b is all that is left")
    for p in (a.engine, a.weights):
        if not os.path.exists(p):
            raise SystemExit("missing: " + p)
    log("   engine  %s" % a.engine)
    log("   weights %s  (%.2f GB)" % (a.weights, os.path.getsize(a.weights) / 2.0 ** 30))

    if a.reps < MIN_REPS:
        raise SystemExit("G-E44b requires at least %d reps; %d was asked for.  The gate is "
                         "the point of the file and is not negotiable downward.  STOP."
                         % (MIN_REPS, a.reps))

    pc = None
    if not a.no_planted_control:
        pc = planted_control(min(a.guard_seconds, 10.0), a.occ_bar)

    log("")
    log("  GUARD -- is the box quiet enough for a timing to mean anything?")
    v = watch(a.guard_seconds)
    med, lo, hi = summarise(v)
    log("     occupancy over %.0fs: median %.1f%%  (min %.1f, max %.1f)  bar %.1f%%"
        % (a.guard_seconds, med, lo, hi, a.occ_bar))
    rec = {"brief": "BRIEF_E44_WHAT_IS_A_TOK_S_WORTH.md addendum A",
           "gate": "G-E44b -- every rate with dispersion or not at all",
           "arm": a.arm, "engine": a.engine, "weights": a.weights,
           "weights_bytes": os.path.getsize(a.weights),
           "ntok": a.ntok, "threads": a.threads, "reps_requested": a.reps,
           "occ_bar": a.occ_bar, "flags": flags,
           "planted_control": pc,
           "guard": {"median": med, "min": lo, "max": hi, "samples": v}}

    if med > a.occ_bar:
        log("")
        log("     REFUSED.  median %.1f%% is above the %.1f%% bar." % (med, a.occ_bar))
        log("     E43's quiet sessions ran at 1.0-13.3% and STILL dispersed 9-22%; a reading")
        log("     taken above the bar is not comparable with anything already published, and")
        log("     a contended one was measured off by 35-41%.  No rate is produced.")
        log("     Free the machine and run this again -- the command is unchanged.")
        rec["verdict"] = "REFUSED -- box not quiet"
        rec["G_E44b"] = {"fires": False, "reason": "occupancy above bar, no rate taken"}
        rec["seconds"] = time.time() - t_start
        json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
        log("")
        log("  wrote %s" % a.out)
        return 3

    log("     box is quiet enough -- proceeding")
    log("")
    log("  rep    tok/s     engine dt     wall      outside window    occupancy during")
    rates, dts, walls, occs = [], [], [], []
    for i in range(a.reps):
        o = Occupancy()
        o.sample()
        rate, dt, wall = one_rep(a.engine, a.weights, a.ntok, a.threads, flags)
        occ = o.sample()
        rates.append(rate)
        dts.append(dt)
        walls.append(wall)
        occs.append(occ)
        log("  %3d   %7.2f   %8.3f s   %7.2f s   %8.2f s        %5.1f%%"
            % (i + 1, rate, dt, wall, wall - dt, occ))

    rmed, rlo, rhi = summarise(rates)
    spread = 100.0 * (rhi - rlo) / rmed
    omed = statistics.median(occs)
    outside = [w - d for w, d in zip(walls, dts)]
    log("")
    log("  RATE      median %.2f tok/s   min %.2f   max %.2f   spread %.1f%% of median"
        % (rmed, rlo, rhi, spread))
    log("  OUTSIDE   median %.2f s spent loading and warming, which no published rate has"
        % statistics.median(outside))
    log("            ever carried.  D: is a USB 3.1 external HDD (addendum A section 4).")
    log("  OCCUPANCY median %.1f%% during the reps (bar %.1f%%)" % (omed, a.occ_bar))
    log("")
    breached = [i + 1 for i, x in enumerate(occs) if x == x and x > a.occ_bar]
    fires = (len(rates) >= MIN_REPS) and not breached
    log("  G-E44b  >= %d reps, quiet box, median AND spread reported : %s"
        % (MIN_REPS, "FIRES" if fires else "*** FAILS ***"))
    if breached:
        log("     reps above the bar during the run: %s -- the guard passed BEFORE the reps"
            % breached)
        log("     and the box moved under them.  The rate is recorded and NOT citable.")
    log("")
    log("  THE BAND, and it replaces a point rather than adding one:")
    log("     %s reads %.0f-%.0f tok/s (median %.2f, %d reps, spread %.1f%%)."
        % (a.arm, rlo, rhi, rmed, len(rates), spread))
    log("     Addendum A: this may NOT be quoted as a new headline.  It is the same engine")
    log("     measured with an interval, and the arm it came from is named above.")

    rec.update({"rates": rates, "engine_dt": dts, "wall": walls,
                "outside_window": outside, "occ_during": occs,
                "rate_median": rmed, "rate_min": rlo, "rate_max": rhi,
                "spread_pct_of_median": spread,
                "outside_window_median": statistics.median(outside),
                "occ_during_median": omed,
                "G_E44b": {"fires": bool(fires), "reps": len(rates),
                           "min_reps": MIN_REPS, "reps_over_bar": breached},
                "verdict": "band reported with dispersion",
                "seconds": time.time() - t_start})
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (a.out, rec["seconds"]))
    return 0 if fires else 1


if __name__ == "__main__":
    sys.exit(main())
