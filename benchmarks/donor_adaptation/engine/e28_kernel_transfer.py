#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E28 -- does E13's kernel lever reach the goal's shape?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E28_THE_OTHER_FACTOR.md, pushed at bb48974
before this file existed.

WHAT THIS MEASURES.  Time only.  The weights are NOISE (synth_export.py); E3 s4's Gate V1
established that timing on this engine depends on shape and format and not on values, which is
what makes a synthetic shape a legitimate way to price one.

WHY.  50 tok/s at T10 (10.6032 G active/token) is 530 G active weights/s against a measured
49.9.  That 10.6x is a PRODUCT of two factors, and every probe from E18 to E27 attacked only the
denominator (make the model smaller).  The numerator has been the constant THROUGHPUT_G = 49.9
since E10 -- but E10 s5.2 measured the loop at 0.29 FMA/cycle/core against Zen 2's 2, about 14%,
and declined to explain the remaining 7x.  E13 then found --lutblk: 1.19x-2.00x the packed kernel
at every footprint, sha256-bit-identical, 1.217x end-to-end on the 0.5 B.

THE OBSTRUCTION, read in the source before anything was measured.  donor_engine.c:1437 dies
unless M.quant == 2, so --lut/--lutblk refuse quant == 3 (E25's rank container) and quant == 4
(E26's carve container).  EVERY T10 artifact on disk is one of those two, so the fast kernel has
never run at the goal's shape and cannot with anything that exists.  THROUGHPUT_G was measured in
a container the fast kernel will not load.

THE PLANTED CONTROLS, and there are two.
  G-E28C: --lut and --lutblk are the same bytes permuted with the same accumulate order, so their
  logits must be BYTE-IDENTICAL (sha256, not parity).  If they are not, the layout change is not
  inert and no timing from it is readable.
  G-E28D: S15-LUTBLK / S15-LUT must reproduce E13's DIRECTION on a shape E13 covered.  An
  instrument must fire on a known-positive before its nulls count.

PROTOCOL.  Idle box, --threads 6, --fuse off, >= 3 repetitions INTERLEAVED by rep so a thermal
drift hits every arm equally, dispersion printed with every rate.  A contended timing is not a
timing, and E26 added the rule that a CPU percentage is only a PRE-FILTER: the instrument is a
reproduced published rate, so BOTH anchors below are gates.

  python e28_kernel_transfer.py --dir /d/_ktmp/e28 --engine ./donor_engine_e26.exe --reps 3
  python e28_kernel_transfer.py --smoke        # S15 only, 1 rep, 16 tokens
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RE_BENCH = re.compile(r"BENCH\s+(\d+) tokens\s+([\d.]+) s\s+([\d.]+) tok/s")

# E26's rebuilt witness, imported rather than reimplemented: the bar's derivation and the
# --selftest planted control live in one place and this probe inherits both.
from e26_carve_cost import cpu_busy, IDLE_BAR, log as _e26_log   # noqa: E402

T_START = time.time()
LOG = []


def log(m):
    LOG.append(m)
    print(m, flush=True)


# ---------------------------------------------------------------- the two published anchors
# Read from e25_rank_cost.json's OWN rows, not from any prose about them.
#   S15-PACKED  rates [29.59, 29.44, 30.07]  mean 29.70   (E26 independently read 26.79)
#   T10-TAG-R0  rates [4.76, 4.51, 4.82]     mean 4.6967
E25_S15_PACKED = 29.70
E25_T10_DENSE = 4.6967
ANCHOR_TOL = 0.10

# E25 measured the container itself to be free at S15: S15-PACKED 29.70 vs S15-TAG-R0 29.50,
# 0.7% apart.  That is why G-E28B may compare a quant==2 file against a quant==3 published rate.
CONTAINER_FREE_NOTE = "E25: S15-PACKED 29.70 vs S15-TAG-R0 29.50, 0.7% -- the container is free"

SHAPES = {"S15": dict(D=1536, F=8960, L=28, V=151936, ntok=120),
          "T10": dict(D=4096, F=14336, L=48, V=32768, ntok=40)}

#       tag            shape  engine flags
ARMS = [("S15-PACKED",  "S15", []),
        ("S15-LUT",     "S15", ["--lut"]),
        ("S15-LUTBLK",  "S15", ["--lutblk"]),
        ("T10-PACKED",  "T10", []),
        ("T10-LUT",     "T10", ["--lut"]),
        ("T10-LUTBLK",  "T10", ["--lutblk"])]

VERDICT_CELL = ("T10-LUTBLK", "T10-PACKED")     # named in the brief, before the run


def active(shape):
    """Weights touched by a matvec on every decoded token.  Same accounting as E25/E26:
    the embedding is a gather, not a matvec; the head is included."""
    s = SHAPES[shape]
    D, F, L, V = s["D"], s["F"], s["L"], s["V"]
    if shape == "S15":
        QD, KD = 1536, 256
    else:
        QD, KD = 4096, 1024
    per_layer = D * QD * 2 + D * KD * 2 + 3 * D * F
    return per_layer * L + V * D


def build_t10_packed(d, seed=1234):
    """The ONE new artifact: T10 in the UNTAGGED quant==2 container, same shape and seed as
    e25_t10_tag_r0, so the container is the only difference."""
    out = os.path.join(d, "e28_t10_packed.bin")
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  t10_packed    %s" % os.path.basename(out))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", "T10",
           "--out", out, "--codes", "mixed", "--seed", str(seed)]
    log("  build t10_packed    %s" % " ".join(cmd[2:]))
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(r.stdout.decode(errors="replace")[-2000:])
        log(r.stderr.decode(errors="replace")[-2000:])
        raise SystemExit("synth_export failed for t10_packed")
    for ln in r.stdout.decode(errors="replace").splitlines():
        if "GATE V3" in ln:
            log("   " + ln.strip())
    j = json.load(open(out + ".json"))
    if j.get("quant") != "packed":
        raise SystemExit("t10_packed is %r, not 'packed' -- the exporter flags are wrong"
                         % j.get("quant"))
    return out


def bench(engine, w, ntok, threads, flags):
    cmd = [engine, "--weights", w, "--threads", str(threads)] + list(flags) + \
          ["--bench", str(ntok)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(r.stderr.decode(errors="replace")[-1500:])
        raise SystemExit("engine failed on " + w + " " + " ".join(flags))
    m = RE_BENCH.search(r.stdout.decode(errors="replace"))
    if not m:
        log(r.stdout.decode(errors="replace")[-1500:])
        raise SystemExit("could not parse BENCH line")
    return float(m.group(3))


def logits_sha(engine, w, idf, n, out, threads, flags):
    cmd = [engine, "--weights", w, "--threads", str(threads)] + list(flags) + \
          ["--logits", idf, str(n), out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(r.stderr.decode(errors="replace")[-1500:])
        raise SystemExit("engine --logits failed on " + w + " " + " ".join(flags))
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), os.path.getsize(out)


def spread(v):
    return (max(v) - min(v)) / (sum(v) / len(v)) if v else 0.0


def winpath(p):
    """Git Bash rewrites a /d/... argv entry to D:/... before a native python sees it, but a
    literal in THIS source never passes through the shell.  Normalise both spellings so the
    same path works whether it came from the command line or from a default."""
    if len(p) > 2 and p[0] == "/" and p[2] == "/" and p[1].isalpha():
        return p[1].upper() + ":" + p[2:]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/d/_ktmp/e28")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--s15", default="/d/_ktmp/e25/e25_s15_packed.bin",
                    help="the quant==2 S15 E25 itself measured -- reused, not rebuilt")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-contended", action="store_true")
    ap.add_argument("--gates-only", action="store_true",
                    help="run G-E28C (sha256 identity) and stop. That gate is "
                         "DETERMINISTIC, so per the rule that a contended timing is not "
                         "a timing but a deterministic check is still a check, it is "
                         "legal on a busy box. No rate is recorded in this mode.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    a.dir, a.s15 = winpath(a.dir), winpath(a.s15)
    a.out = a.out or os.path.join(HERE, "results",
                                  "e28_kernel_transfer%s.json" % ("_smoke" if a.smoke else ""))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(a.dir, exist_ok=True)

    arms = [x for x in ARMS if x[1] == "S15"] if a.smoke else ARMS
    reps = 1 if a.smoke else a.reps
    ntok_override = 16 if a.smoke else None

    log("== E28: does E13's kernel lever reach the goal's shape? ==")
    log("  engine %s   threads %d   reps %d%s"
        % (os.path.basename(a.engine), a.threads, reps, "   [SMOKE]" if a.smoke else ""))
    log("")

    # ---------------------------------------------------------------- G-E28E, before anything
    busy0, peak0 = cpu_busy(6)
    log("  contention witness: %.1f%% mean / %.0f%% peak before the run (bar %.0f%% on the MEAN)"
        % (busy0, peak0, IDLE_BAR))
    if busy0 > IDLE_BAR and not (a.allow_contended or a.gates_only):
        raise SystemExit("REFUSING: %.1f%% busy. A contended timing is not a timing." % busy0)
    if a.gates_only:
        log("  --gates-only: no rate will be recorded, so the witness does not gate this run.")

    # ---------------------------------------------------------------- artifacts
    W = {"S15": a.s15}
    if not os.path.exists(W["S15"]):
        raise SystemExit("missing %s -- E28 reuses the file E25 measured" % W["S15"])
    log("  have  s15_packed    %s" % os.path.basename(W["S15"]))
    if not a.smoke:
        W["T10"] = build_t10_packed(a.dir)
    log("")

    rec = {"brief": "BRIEF_E28_THE_OTHER_FACTOR.md (bb48974)",
           "question": "Is 49.9 G active weights/s a property of this engine at this shape, or "
                       "of the container E25 happened to measure it in?",
           "smoke": bool(a.smoke), "threads": a.threads, "reps": reps,
           "anchors": {"S15-PACKED": E25_S15_PACKED, "T10-PACKED": E25_T10_DENSE,
                       "tol": ANCHOR_TOL, "container_note": CONTAINER_FREE_NOTE},
           "obstruction": "donor_engine.c:1437 -- --lut requires M.quant==2, so it refuses "
                          "quant==3 (rank) and quant==4 (carve)",
           "verdict_cell": "%s / %s" % VERDICT_CELL,
           "idle_bar_pct": IDLE_BAR, "cpu_busy_mean_pct": [busy0], "cpu_busy_peak_pct": [peak0]}

    # ---------------------------------------------------------------- G-E28C, the planted control
    # Same bytes permuted, same accumulate order -> the logits have no licence to move a bit.
    log("  -- G-E28C: --lut vs --lutblk must be BYTE-IDENTICAL (sha256, not parity)")
    import numpy as np
    gc = {}
    for shape in (["S15"] if a.smoke else ["S15", "T10"]):
        n = 8
        ids = np.array([101, 202, 303, 404, 505, 606, 707, 808][:n], dtype="<i4")
        ids = ids % SHAPES[shape]["V"]
        idf = os.path.join(a.dir, "_e28_ids_%s.bin" % shape.lower())
        ids.tofile(idf)
        hs = {}
        for flag in ("--lut", "--lutblk"):
            o = os.path.join(a.dir, "_e28_log_%s_%s.bin" % (shape.lower(), flag.strip("-")))
            hs[flag], sz = logits_sha(a.engine, W[shape], idf, n, o, a.threads, [flag])
            try:
                os.remove(o)
            except OSError:
                pass
        same = hs["--lut"] == hs["--lutblk"]
        gc[shape] = {"lut_sha256": hs["--lut"], "lutblk_sha256": hs["--lutblk"],
                     "identical": bool(same), "n_tokens": n, "ids": ids.tolist()}
        log("     %-4s lut %s" % (shape, hs["--lut"][:32]))
        log("     %-4s blk %s   -> %s" % (shape, hs["--lutblk"][:32],
                                          "IDENTICAL" if same else "*** DIFFER ***"))
    rec["G_E28C"] = {"per_shape": gc, "fires": all(v["identical"] for v in gc.values())}
    if not rec["G_E28C"]["fires"]:
        log("  *** G-E28C MISSED.  The blocked layout is NOT inert at this shape, so no timing")
        log("      below is readable as a layout effect.  This is a bug report, not a result. ***")
    log("")

    if a.gates_only:
        rec["gates_only"] = True
        rec["seconds"] = time.time() - T_START
        rec["log"] = LOG
        out = a.out.replace(".json", "_gates.json")
        json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)
        log("  wrote %s  [%.0fs]  -- deterministic gate only, no timing" % (out, rec["seconds"]))
        return

    # ---------------------------------------------------------------- the sweep
    rates = dict((t, []) for t, _, _ in arms)
    for r in range(reps):
        for tag, shape, flags in arms:
            nt = ntok_override or SHAPES[shape]["ntok"]
            v = bench(a.engine, W[shape], nt, a.threads, flags)
            rates[tag].append(v)
            log("  rep %d  %-12s %7.2f tok/s   (%d tokens)" % (r + 1, tag, v, nt))
        m, p = cpu_busy(3)
        rec["cpu_busy_mean_pct"].append(m)
        rec["cpu_busy_peak_pct"].append(p)
        log("  rep %d  contention witness: %.1f%% mean / %.0f%% peak" % (r + 1, m, p))
    log("")

    rec["arms"] = {}
    for tag, shape, flags in arms:
        v = rates[tag]
        mean = sum(v) / len(v)
        act = active(shape)
        rec["arms"][tag] = {"shape": shape, "flags": flags, "rates": v, "mean_tok_s": mean,
                            "spread": spread(v), "active_weights_per_token": act,
                            "charged_G_weights_per_s": act * mean / 1e9}

    log("  arm            shape  kernel      mean tok/s   spread   charged G-w/s")
    for tag, shape, flags in arms:
        x = rec["arms"][tag]
        log("  %-13s %-6s %-11s %9.2f %7.1f%% %13.2f"
            % (tag, shape, " ".join(flags) or "packed", x["mean_tok_s"],
               100 * x["spread"], x["charged_G_weights_per_s"]))
    log("")

    # ---------------------------------------------------------------- the anchors, G-E28A / G-E28B
    anch = {}
    for tag, pub in (("S15-PACKED", E25_S15_PACKED), ("T10-PACKED", E25_T10_DENSE)):
        if tag not in rec["arms"]:
            continue
        got = rec["arms"][tag]["mean_tok_s"]
        dev = (got - pub) / pub
        anch[tag] = {"measured": got, "e25_published": pub, "rel_dev": dev,
                     "tol": ANCHOR_TOL, "passes": bool(abs(dev) <= ANCHOR_TOL)}
        log("  ANCHOR  %-12s reads %.2f against E25's published %.2f (%+.2f%%, bar +-%.0f%%) -> %s"
            % (tag, got, pub, 100 * dev, 100 * ANCHOR_TOL,
               "PASS" if anch[tag]["passes"] else "FAIL"))
    rec["G_E28A"] = anch.get("S15-PACKED")
    rec["G_E28B"] = anch.get("T10-PACKED")
    if any(not v["passes"] for v in anch.values()):
        log("  *** AN ANCHOR FAILED.  This run did not reproduce a rate measured on an idle box")
        log("      at the same shape, so this record is VOID AS A TIMING. ***")
        rec["VOID_AS_A_TIMING"] = True
    else:
        rec["VOID_AS_A_TIMING"] = False

    # ---------------------------------------------------------------- G-E28D, the known-positive
    if "S15-LUTBLK" in rec["arms"] and "S15-LUT" in rec["arms"]:
        kp = rec["arms"]["S15-LUTBLK"]["mean_tok_s"] / rec["arms"]["S15-LUT"]["mean_tok_s"]
        rec["G_E28D"] = {"ratio": kp, "bar": 1.5, "fires": bool(kp >= 1.5),
                         "e13_microbench_512MB": 3.03}
        log("  G-E28D  known-positive  S15-LUTBLK / S15-LUT = %.3f  (bar >= 1.50) -> %s"
            % (kp, "FIRES" if kp >= 1.5 else "DOES NOT FIRE"))
        if kp < 1.5:
            log("     E13's effect does not reappear on a shape E13 covered, so this run cannot")
            log("     speak about a shape it did not cover.")

    # ---------------------------------------------------------------- the verdict cell
    if all(t in rec["arms"] for t in VERDICT_CELL):
        num, den = (rec["arms"][t]["mean_tok_s"] for t in VERDICT_CELL)
        ratio = num / den
        if ratio > 1.30:
            v = "CONTAINER-COSTS"
        elif ratio >= 1.10:
            v = "KERNEL-TRANSFERS"
        else:
            v = "NO-TRANSFER"
        # the brief's prediction 5, computed rather than asserted
        t10_dense_at = rec["arms"]["T10-LUTBLK"]["mean_tok_s"]
        l21_minres = 7.75 * ratio        # E27's best COMPARABLE floor arm, scaled by this lever
        rec["verdict"] = {
            "name": v, "cell": "%s / %s" % VERDICT_CELL, "ratio": ratio,
            "bands": {"CONTAINER-COSTS": "> 1.30", "KERNEL-TRANSFERS": "1.10 - 1.30",
                      "NO-TRANSFER": "< 1.10"},
            "t10_dense_tok_s": t10_dense_at,
            "e27_L21MINRES_scaled_tok_s": l21_minres,
            "still_short_of_50_by": 50.0 / l21_minres}
        log("")
        log("  VERDICT CELL  %s / %s = %.3f  ->  %s" % (VERDICT_CELL[0], VERDICT_CELL[1],
                                                        ratio, v))
        log("     T10 dense on the fast kernel: %.2f tok/s" % t10_dense_at)
        log("     E27's best COMPARABLE floor arm (L21-MINRES, 7.75) scaled by this lever:"
            " %.2f tok/s" % l21_minres)
        log("     STILL SHORT OF 50 tok/s BY %.1fx -- the numerator alone does not close the gap."
            % (50.0 / l21_minres))

    rec["seconds"] = time.time() - T_START
    rec["log"] = LOG
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (a.out, rec["seconds"]))


if __name__ == "__main__":
    main()
