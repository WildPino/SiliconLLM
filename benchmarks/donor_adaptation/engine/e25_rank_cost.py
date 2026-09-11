#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E25 part B -- what does the rank actually cost, in the engine?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E25_WHAT_THE_RANK_COSTS_IN_THE_ENGINE.md,
pushed before this file existed.

WHAT THIS MEASURES.  Time only.  The weights are NOISE (synth_export.py) and nothing here says
anything about quality -- E3 s4's Gate V1 established that timing on this engine depends on
shape and format, not on values, which is what makes a synthetic shape a legitimate way to
price one.  Every rank number published by E21/E22/E23 priced the cut's QUALITY, because both
of them computed A.B and installed the dense product; the COST has been arithmetic until now.

THE PLANTED CONTROL, and it is the point of the run.  For a square [D,D] projection the
factored form moves exactly as many weights as the dense one at r = D/2:  2.D.(D/2) = D^2.
The `Rhalf` arms are therefore BYTE-NEUTRAL BY ARITHMETIC, and whatever they lose against the
`R0` arm IS the factored path's own overhead -- the second matvec call, the extra OpenMP
region, the intermediate vector.  That overhead is subtracted from every other rank's gain
before the gain is called a saving.  Without it, a rank speedup and a container speedup are
indistinguishable.

PROTOCOL.  A contended timing is not a timing: idle box, --threads 6, >= 3 repetitions, and the
dispersion printed with every rate.  Reps are INTERLEAVED (rep 1 of every arm, then rep 2, ...)
so a thermal drift hits every arm equally instead of the arms that happen to run last.
--fuse is off everywhere, including the dense controls: a factored q_proj cannot be fused, and
leaving it on for the dense arms would compare 97 OpenMP regions against 225.

  python e25_rank_cost.py --dir D:/_ktmp/e25 --engine ./donor_engine_e25.exe --reps 3
  python e25_rank_cost.py --smoke                       # S05 only, 1 rep, builds nothing big
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

# (tag, shape, rank, tagged) -- rank 0 + tagged False is the untagged container control
ARMS_FULL = [("S15-PACKED", "S15", 0, False),
             ("S15-TAG-R0", "S15", 0, True),
             ("S15-R768", "S15", 768, True),      # planted control: byte-neutral at D/2
             ("S15-R512", "S15", 512, True),
             ("S15-R256", "S15", 256, True),
             ("T10-TAG-R0", "T10", 0, True),
             ("T10-R2048", "T10", 2048, True),    # planted control: byte-neutral at D/2
             ("T10-R512", "T10", 512, True),
             ("T10-R256", "T10", 256, True)]
ARMS_SMOKE = [("S05-TAG-R0", "S05", 0, True),
              ("S05-R448", "S05", 448, True),     # D/2 = 448 at D=896
              ("S05-R256", "S05", 256, True)]
CONTROL_OF = {"S15": "S15-TAG-R0", "T10": "T10-TAG-R0", "S05": "S05-TAG-R0"}
HALF_ARM = {"S15": "S15-R768", "T10": "T10-R2048", "S05": "S05-R448"}
RE_BENCH = re.compile(r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s")


def log(m):
    print(m, flush=True)


def active(shape, rank, head_ternary=True):
    """Active weights/token, with q_proj and o_proj factored at `rank`.  Arithmetic, mirrored
    from synth_export so the two cannot disagree about what an arm costs."""
    D, F, L, NH, NKV, HD, V, tied, _src = SX.SHAPES[shape]
    if head_ternary:
        tied = 0
    a = SX.active_weights(D, F, L, NH, NKV, HD, V)
    if rank:
        a += (rank * (NH * HD + D) * 2 - (NH * HD * D) * 2) * L
    return a


def build(d, tag, shape, rank, tagged, seed=1234):
    out = os.path.join(d, "e25_%s.bin" % tag.lower().replace("-", "_"))
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  %-12s %s" % (tag, os.path.basename(out)))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", shape,
           "--out", out, "--seed", str(seed)]
    if rank:
        cmd += ["--rank", str(rank)]
    elif tagged:
        cmd += ["--tagged"]
    log("  build %-12s %s" % (tag, " ".join(cmd[2:])))
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(r.stdout.decode(errors="replace")[-2000:])
        log(r.stderr.decode(errors="replace")[-2000:])
        raise SystemExit("synth_export failed for " + tag)
    for ln in r.stdout.decode(errors="replace").splitlines():
        if "GATE V3" in ln:
            log("   " + ln.strip())
    return out


def bench(engine, w, ntok, threads):
    r = subprocess.run([engine, "--weights", w, "--threads", str(threads),
                        "--bench", str(ntok)], capture_output=True)
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
    ap.add_argument("--dir", default="D:/_ktmp/e25")
    ap.add_argument("--engine", default=os.path.join(HERE, "donor_engine_e25.exe"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=0, help="0 = per-shape default")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e25_rank_cost.json"))
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    arms = ARMS_SMOKE if a.smoke else ARMS_FULL
    reps = 1 if a.smoke else a.reps
    if reps < 3 and not a.smoke:
        log("REFUSING: a speed claim on this box needs >= 3 repetitions and a dispersion.")
        return 2
    t_start = time.time()

    # per-shape token budget: enough that load time is amortized and a rep is ~10-20 s
    NTOK = {"S05": 200, "S15": 120, "T10": 40}
    log("== E25 part B: the cost of the factored path ==")
    log("  engine %s   threads %d   reps %d" % (os.path.basename(a.engine), a.threads, reps))
    log("")
    log("  arm            shape   rank   active weights/token   vs its R0 control")
    paths = {}
    for tag, shape, rank, tagged in arms:
        ctrl = active(shape, 0)
        act = active(shape, rank)
        log("  %-14s %-6s %5d   %17.4f G   %+7.2f%%"
            % (tag, shape, rank, act / 1e9, 100.0 * (act - ctrl) / ctrl))
    log("")
    for tag, shape, rank, tagged in arms:
        paths[tag] = build(a.dir, tag, shape, rank, tagged)
    log("")

    rates = {t: [] for t, _s, _r, _g in arms}
    for rep in range(reps):
        for tag, shape, rank, tagged in arms:
            n = a.tokens or NTOK[shape]
            v = bench(a.engine, paths[tag], n, a.threads)
            rates[tag].append(v)
            log("  rep %d  %-14s %7.2f tok/s   (%d tokens)" % (rep + 1, tag, v, n))
    log("")

    rec = {"brief": "BRIEF_E25_WHAT_THE_RANK_COSTS_IN_THE_ENGINE.md", "reps": reps,
           "threads": a.threads, "smoke": bool(a.smoke), "arms": {}}
    log("  arm            active/tok    mean tok/s   spread    vs R0      byte prediction")
    for tag, shape, rank, tagged in arms:
        v = rates[tag]
        mean = sum(v) / len(v)
        spread = (max(v) - min(v)) / mean if mean else 0.0
        c = rates[CONTROL_OF[shape]]
        cmean = sum(c) / len(c)
        ratio = mean / cmean if cmean else 0.0
        pred = active(shape, 0) / float(active(shape, rank))
        rec["arms"][tag] = {"shape": shape, "rank": rank, "tagged": tagged,
                            "active_weights_per_token": active(shape, rank),
                            "rates": v, "mean_tok_s": mean, "spread": spread,
                            "ratio_vs_R0": ratio, "byte_ratio_prediction": pred}
        log("  %-14s %8.4f G   %9.2f   %5.1f%%   %+7.2f%%   %+7.2f%%"
            % (tag, active(shape, rank) / 1e9, mean, 100 * spread,
               100 * (ratio - 1), 100 * (pred - 1)))
    log("")

    # ---- the planted control, read out explicitly
    for shape in sorted({s for _t, s, _r, _g in arms}):
        h = HALF_ARM.get(shape)
        if h not in rec["arms"]:
            continue
        over = rec["arms"][h]["ratio_vs_R0"]
        rec.setdefault("overhead", {})[shape] = over
        log("  PLANTED CONTROL %-6s  %s is byte-neutral and reads %+.2f%% vs R0"
            % (shape, h, 100 * (over - 1)))
        log("     -> the factored path's OWN cost at this shape is %+.2f%%; every other rank's"
            % (100 * (over - 1)))
        log("        gain below is gross, and net of this is the saving the bytes bought.")
        for tag, s2, rank, _g in arms:
            if s2 != shape or rank == 0 or tag == h:
                continue
            g = rec["arms"][tag]["ratio_vs_R0"]
            rec["arms"][tag]["ratio_net_of_overhead"] = g / over if over else None
            log("        %-14s gross %+6.2f%%   net of overhead %+6.2f%%   bytes predict %+6.2f%%"
                % (tag, 100 * (g - 1), 100 * (g / over - 1),
                   100 * (rec["arms"][tag]["byte_ratio_prediction"] - 1)))
    rec["seconds"] = time.time() - t_start
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (a.out, rec["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
