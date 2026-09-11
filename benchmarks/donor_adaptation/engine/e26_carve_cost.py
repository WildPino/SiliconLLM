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
printed with every rate.  A contended timing is not a timing.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e26")
    ap.add_argument("--engine", default=os.path.join(HERE, "donor_engine_e26.exe"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=0, help="0 = per-shape default")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e26_carve_cost.json"))
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    shapes = SHAPES_SMOKE if a.smoke else SHAPES_FULL
    reps = 1 if a.smoke else a.reps
    if reps < 3 and not a.smoke:
        log("REFUSING: a speed claim on this box needs >= 3 repetitions and a dispersion.")
        return 2
    if a.out.endswith(".json") and a.smoke:
        a.out = a.out[:-5] + "_smoke.json"
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
    log("")

    rec = {"brief": "BRIEF_E26_WHAT_AN_ACTIVATED_WEIGHT_COSTS.md", "reps": reps,
           "threads": a.threads, "smoke": bool(a.smoke), "E": E_GROUPS,
           "k_grid": list(K_GRID), "arms": {}}
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
    rec["seconds"] = time.time() - t_start
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (a.out, rec["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
