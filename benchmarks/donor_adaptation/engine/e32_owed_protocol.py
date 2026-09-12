#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E32 -- does --lutblk survive its own protocol?  The --seqlen 512 re-run E14 owes.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E32_THE_OWED_PROTOCOL.md, pushed at 1fb4c18
BEFORE this file existed.  Nothing here may contradict it; the result document scores it as
written.

E14 disqualified its own verdict in its own section 1: with no --seqlen the engine treats the
whole ids file as ONE sequence, so E14 scored one 12,288-token sequence rather than 24 documents
of 512, while the bands it read against were drawn from 512-context regimes.  Its section 5 has
carried the owed re-run ever since, E28 and E30 both re-state it, and in the meantime the entire
engine-side story -- 61.64 G-w/s, 33.01 GB/s, the 1.424x shape lever -- was built on the arm that
verdict licenses.

THIS IS A QUALITY MEASUREMENT.  Deterministic, and by the standing law it may be taken on a
loaded box.  No timing is produced here and none may be quoted from this run.

  python e32_owed_protocol.py
"""
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ENGINE = r"D:\_ktmp\e13\donor_engine.exe"      # the E13 build: the one with --lutblk
KT = r"D:\_ktmp\e1"
OUT = os.path.join(HERE, "results", "e32_owed_protocol.json")

VOCAB = 151936
SEQLEN = 512
# E1's byte convention, and the whole point of the probe: not E14's 51,941 over 12,287.
SCORED_BYTES = 51870
N_PRED = 12264
BYTES_PER_TOK = SCORED_BYTES / float(N_PRED)
CHANCE_BPB = math.log(VOCAB, 2) / BYTES_PER_TOK

E1_05B_TQH = 4.531234          # e1_05b.log:30 -- the same artifact, slice and protocol
ANCHOR_TOL = 0.001
E14_12K = {"0.5B": {"A0": 4.629292, "A1": 4.315413082, "A2": 4.616285418, "A3": 4.315413082},
           "1.5B": {"A0": 3.446375376, "A1": 3.429414114, "A2": 3.440384359,
                    "A3": 3.429414114}}
E14_GREEDY_A3_15B = 73 / 160.0   # 45.6% at 12 k -- generation was never contaminated

CELLS = [("0.5B", os.path.join(KT, "qwen25-05b_tqh.bin"),
          os.path.join(KT, "ids_qwen25-05b_tqh.bin")),
         ("1.5B", os.path.join(KT, "qwen25-15b_tqh.bin"),
          os.path.join(KT, "ids_qwen25-15b_tqh.bin"))]
ARMS = [("A0", [],                              "fp32 activations -- the shipped packed path"),
        ("A1", ["--lut"],                       "int8 activations, ONE scale per vector"),
        ("A2", ["--lut", "--lut-group", "32"],  "int8 activations, one scale per 32 channels"),
        ("A3", ["--lutblk"],                    "E13 blocked layout -- what E28/E30 measured")]

CHEAP_MAX, COSTLY_MIN = 0.010, 0.020      # E14's registered bands, drawn from 512 regimes
RANK_BAR = 0.70                           # prediction 5's line, not a gate
THREADS = os.environ.get("D_THREADS", "6")


def log(*a):
    print(*a, flush=True)


def run_arm(weights, ids, flags):
    cmd = [ENGINE, "--weights", weights, "--threads", THREADS, "--seqlen", str(SEQLEN),
           "--bpb", ids] + flags
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("engine failed (%d): %s" % (r.returncode, r.stderr[-2000:]))
    m = re.search(r"NATS_PER_TOKEN\s+([0-9.]+)", r.stdout)
    n = re.search(r"NATS_TOTAL\s+([0-9.]+)", r.stdout)
    p = re.search(r"N_PREDICTED\s+(\d+)", r.stdout)
    if not m:
        raise SystemExit("no NATS_PER_TOKEN in output:\n" + r.stdout[-2000:])
    nats = float(m.group(1))
    return {"nats_per_token": nats, "nats_total": float(n.group(1)) if n else None,
            "n_predicted": int(p.group(1)) if p else None,
            "bpb": nats / math.log(2) / BYTES_PER_TOK,
            "seconds": time.time() - t0, "cmd": " ".join(cmd)}


def main():
    t_start = time.time()
    log("== E32: the --seqlen 512 re-run E14 owes ==")
    log("  protocol: --seqlen %d, E1's convention -- %d scored bytes over %d predictions"
        % (SEQLEN, SCORED_BYTES, N_PRED))
    log("  chance line at THIS denominator: %.6f BPB   (bytes/token %.6f)"
        % (CHANCE_BPB, BYTES_PER_TOK))
    log("  QUALITY ONLY -- deterministic; no timing is produced or quotable from this run.")

    out = {"brief": "briefs/BRIEF_E32_THE_OWED_PROTOCOL.md (1fb4c18)",
           "seqlen": SEQLEN, "scored_bytes": SCORED_BYTES, "n_pred": N_PRED,
           "bytes_per_token": BYTES_PER_TOK, "chance_bpb": CHANCE_BPB,
           "bands": {"cheap_max": CHEAP_MAX, "costly_min": COSTLY_MIN},
           "e14_12k": E14_12K, "threads": int(THREADS), "cells": {}}

    for tag, w, ids in CELLS:
        if not os.path.exists(w):
            raise SystemExit("missing artifact " + w)
        log("")
        log("  ---- %s  %s" % (tag, os.path.basename(w)))
        arms = {}
        for atag, flags, note in ARMS:
            r = run_arm(w, ids, flags)
            arms[atag] = dict(r, flags=flags, note=note)
            log("    %-3s %-34s BPB %.9f   n_pred %s   [%.0fs]"
                % (atag, " ".join(flags) or "(fp32 activations)", r["bpb"],
                   r["n_predicted"], r["seconds"]))
        b = dict((k, v["bpb"]) for k, v in arms.items())
        cell = {"weights": w, "ids": ids, "arms": arms,
                "dBPB": dict((k, b[k] - b["A0"]) for k in ("A1", "A2", "A3")),
                "A0_vs_chance": b["A0"] - CHANCE_BPB,
                "A3_minus_A1": b["A3"] - b["A1"],
                "shift_vs_E14_12k": dict((k, b[k] - E14_12K[tag][k]) for k in b)}
        out["cells"][tag] = cell
        log("    A0 sits %+.6f vs the chance line (%s)"
            % (cell["A0_vs_chance"],
               "BELOW -- BPB means what it usually means" if cell["A0_vs_chance"] < 0
               else "ABOVE -- instrument only, E14 s2's demotion applies"))
        for k in ("A1", "A2", "A3"):
            log("    dBPB(%s) = %+.9f    (E14 at 12k: %+.9f)"
                % (k, cell["dBPB"][k], E14_12K[tag][k] - E14_12K[tag]["A0"]))
        log("    A3 - A1 = %+.3e" % cell["A3_minus_A1"])

    # ---------------------------------------------------------------- gates
    a0_05 = out["cells"]["0.5B"]["arms"]["A0"]["bpb"]
    g_a = {"measured": a0_05, "e1_published": E1_05B_TQH, "abs_diff": abs(a0_05 - E1_05B_TQH),
           "tol": ANCHOR_TOL, "source": "e1_05b.log:30, same artifact, slice and protocol",
           "fires": bool(abs(a0_05 - E1_05B_TQH) <= ANCHOR_TOL)}
    out["G_E32A"] = g_a
    log("")
    log("  G-E32A  A0(0.5B) reads %.6f against E1's %.6f, |diff| %.6f (bar %.3f) -> %s"
        % (a0_05, E1_05B_TQH, g_a["abs_diff"], ANCHOR_TOL,
           "FIRES" if g_a["fires"] else "VOID -- this is not E1's protocol"))

    diffs = dict((t, out["cells"][t]["A3_minus_A1"]) for t, _, _ in CELLS)
    g_b = {"A3_minus_A1": diffs, "source": "E14 measured +0.000e+00 at both cells",
           "fires": all(d == 0.0 for d in diffs.values())}
    out["G_E32B"] = g_b
    log("  G-E32B  A3 - A1 = %s -> %s"
        % (", ".join("%s %+.3e" % (t, d) for t, d in diffs.items()),
           "FIRES" if g_b["fires"] else "VOID -- the arms are not what they are labelled"))

    below = out["cells"]["1.5B"]["A0_vs_chance"] < 0
    out["G_E32C"] = {"A0_1.5B_vs_chance": out["cells"]["1.5B"]["A0_vs_chance"],
                     "fires": bool(below)}
    log("  G-E32C  A0(1.5B) sits %+.6f vs chance -> %s"
        % (out["cells"]["1.5B"]["A0_vs_chance"],
           "FIRES, the verdict cell is readable" if below
           else "does NOT fire -- instrument only, no verdict cell"))

    if not (g_a["fires"] and g_b["fires"]):
        out["VOID"] = ("G-E32A" if not g_a["fires"] else "G-E32B") + " did not fire"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("VOID -- see " + OUT)

    # ---------------------------------------------------------------- verdict
    if below:
        d = out["cells"]["1.5B"]["dBPB"]["A3"]
        name = ("ACTIVATION-CHEAP" if abs(d) <= CHEAP_MAX else
                "MARGINAL" if abs(d) < COSTLY_MIN else "ACTIVATION-COSTLY")
        out["verdict"] = {"cell": "dBPB(A3) at 1.5B, --seqlen 512", "dBPB": d,
                          "abs": abs(d), "name": name,
                          "e14_12k_dBPB": E14_12K["1.5B"]["A3"] - E14_12K["1.5B"]["A0"],
                          "moved_toward_zero": bool(
                              abs(d) < abs(E14_12K["1.5B"]["A3"] - E14_12K["1.5B"]["A0"]))}
        log("")
        log("  VERDICT CELL  dBPB(A3) at 1.5B, 512 context = %+.9f  ->  %s" % (d, name))
        log("     E14 read %+.9f at 12k; this %s toward zero"
            % (out["verdict"]["e14_12k_dBPB"],
               "MOVES" if out["verdict"]["moved_toward_zero"] else "does NOT move"))
        log("")
        log("     RANK PARTNER (mandatory, E14 s3's law): run e14_gn3_greedy.py -- E14 read")
        log("     %.1f%% top-1 agreement A3 vs A0 at 1.5B, and generation was NEVER contaminated"
            % (100 * E14_GREEDY_A3_15B))
        log("     by the --seqlen defect, so that number stands at this protocol already.")
        log("     A cheap BPB with a broken trajectory is reported as exactly that.")
    else:
        out["verdict"] = {"name": "INSTRUMENT-ONLY", "why": "A0 at 1.5B is not below chance"}

    out["seconds"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
