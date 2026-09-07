#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E14 -- what does int8 activation quantization cost in BPB?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E14_ACTIVATION_INT8_COST.md (a6a6607),
section 10 amendment (moves the verdict cell to 1.5B after A0 showed the 0.5B baseline sits above
the chance line) pushed before any treatment arm ran.

The weights are IDENTICAL in every arm: build_tm is a pure transpose of the same packed bytes, so
--lut, --lutblk and the packed default hold bit-identical ternary weights and differ ONLY in how
activations are represented.  There is no weight-side confound to separate out, by construction.

Bands, verbatim from section 4, unchanged by the amendment:
  G-N0  BPB(A3) - BPB(A1) must be EXACTLY 0        (instrument; E13 proved byte-identical logits)
  G-N1  BPB(A2) < BPB(A1) must FIRE                (instrument; E11 measured 1.40e-01 vs 3.10e-02)
  G-N2  dBPB = BPB(A1) - BPB(A0), and A2 likewise
          <= 0.010  ACTIVATION-CHEAP
          0.010-0.020 ACTIVATION-MARGINAL
          >= 0.020  ACTIVATION-COSTLY
        READ AT 1.5B ONLY -- section 10.2.  0.5B is instrument-only.

Env: D_THREADS (default 6), E14_CELLS (default "0.5B,1.5B")
"""
import json
import math
import os
import re
import subprocess
import sys
import time

ENGINE = r"D:\_ktmp\e13\donor_engine.exe"      # the E13 build: has --lutblk
KT = r"D:\_ktmp\e1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e14_activation.json")

VOCAB = 151936
SCORED_BYTES = 51941          # decoded from the slice; identical for both ids files
N_PRED = 12287
BYTES_PER_TOK = SCORED_BYTES / float(N_PRED)
CHANCE_BPB = math.log(VOCAB, 2) / BYTES_PER_TOK

CELLS = [
    ("0.5B", os.path.join(KT, "qwen25-05b_tqh.bin"), os.path.join(KT, "ids_qwen25-05b_tqh.bin")),
    ("1.5B", os.path.join(KT, "qwen25-15b_tqh.bin"), os.path.join(KT, "ids_qwen25-15b_tqh.bin")),
]
ARMS = [
    ("A0", [],                              "fp32 activations -- the baseline, the shipped path"),
    ("A1", ["--lut"],                       "int8 activations, ONE scale per vector"),
    ("A2", ["--lut", "--lut-group", "32"],  "int8 activations, one scale per 32 channels"),
    ("A3", ["--lutblk"],                    "E13 blocked layout -- MUST equal A1 exactly"),
]

CHEAP_MAX, COSTLY_MIN = 0.010, 0.020
WANT = os.environ.get("E14_CELLS", "0.5B,1.5B").split(",")
THREADS = os.environ.get("D_THREADS", "6")


def run_arm(weights, ids, flags):
    cmd = [ENGINE, "--weights", weights, "--bpb", ids, "--threads", THREADS] + flags
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("engine failed (%d): %s" % (r.returncode, r.stderr[-2000:]))
    m = re.search(r"NATS_PER_TOKEN\s+([0-9.]+)", r.stdout)
    n = re.search(r"NATS_TOTAL\s+([0-9.]+)", r.stdout)
    if not m:
        raise SystemExit("no NATS_PER_TOKEN in output:\n" + r.stdout[-2000:])
    nats = float(m.group(1))
    return {"nats_per_token": nats,
            "nats_total": float(n.group(1)) if n else None,
            "bpb": nats / math.log(2) / BYTES_PER_TOK,
            "seconds": time.time() - t0,
            "cmd": " ".join(cmd)}


def main():
    out = {"brief": "briefs/BRIEF_E14_ACTIVATION_INT8_COST.md @ a6a6607 + section 10",
           "scored_bytes": SCORED_BYTES, "n_predicted": N_PRED,
           "bytes_per_token": BYTES_PER_TOK, "chance_bpb": CHANCE_BPB,
           "bands": {"CHEAP_MAX": CHEAP_MAX, "COSTLY_MIN": COSTLY_MIN},
           "verdict_cell": "1.5B", "cells": {}}
    if os.path.exists(OUT):
        try:
            out["cells"] = json.load(open(OUT)).get("cells", {})
        except Exception:
            pass
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    print("chance line = %.6f BPB  (%d scored bytes, %.6f bytes/token)"
          % (CHANCE_BPB, SCORED_BYTES, BYTES_PER_TOK), flush=True)

    for tag, w, ids in CELLS:
        if tag not in WANT:
            continue
        arms = out["cells"].get(tag, {}).get("arms", {})
        print("\n== %s  %s" % (tag, os.path.basename(w)), flush=True)
        for atag, flags, note in ARMS:
            if atag in arms:
                print("   %-3s CACHED  BPB = %.9f" % (atag, arms[atag]["bpb"]), flush=True)
                continue
            r = run_arm(w, ids, flags)
            r["note"] = note
            arms[atag] = r
            print("   %-3s BPB = %.9f  (%.6f nats/tok, %.0fs)  %s"
                  % (atag, r["bpb"], r["nats_per_token"], r["seconds"], note), flush=True)
            out["cells"][tag] = {"weights": w, "ids": ids, "arms": arms}
            json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

        c = out["cells"][tag]
        b = {k: arms[k]["bpb"] for k in arms}
        c["gn0_a3_minus_a1"] = b["A3"] - b["A1"]
        c["gn0_pass"] = (arms["A3"]["nats_per_token"] == arms["A1"]["nats_per_token"])
        c["gn1_a2_minus_a1"] = b["A2"] - b["A1"]
        c["gn1_fires"] = b["A2"] < b["A1"]
        c["dBPB_A1"] = b["A1"] - b["A0"]
        c["dBPB_A2"] = b["A2"] - b["A0"]
        c["baseline_vs_chance"] = b["A0"] - CHANCE_BPB
        print("   G-N0  A3-A1 = %+.3e   %s" % (c["gn0_a3_minus_a1"],
              "PASS (identical)" if c["gn0_pass"] else "*** FAIL ***"), flush=True)
        print("   G-N1  A2-A1 = %+.9f   %s" % (c["gn1_a2_minus_a1"],
              "FIRES" if c["gn1_fires"] else "*** DOES NOT FIRE ***"), flush=True)
        print("   dBPB(A1) = %+.9f   dBPB(A2) = %+.9f   A0 - chance = %+.6f"
              % (c["dBPB_A1"], c["dBPB_A2"], c["baseline_vs_chance"]), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    print("\n---- verdict ----", flush=True)

    def label(d):
        return ("ACTIVATION-CHEAP" if d <= CHEAP_MAX else
                "ACTIVATION-COSTLY" if d >= COSTLY_MIN else "ACTIVATION-MARGINAL")

    for tag in ("0.5B", "1.5B"):
        c = out["cells"].get(tag)
        if not c:
            continue
        readable = c["baseline_vs_chance"] < 0
        print("  %-5s A0 %s chance by %+.6f BPB -> G-N2 %s"
              % (tag, "below" if readable else "ABOVE", c["baseline_vs_chance"],
                 "READABLE" if readable else "NOT READABLE (section 10.2)"))
        print("        G-N0 %s | G-N1 %s | dBPB(A1) %+.9f %s | dBPB(A2) %+.9f %s"
              % ("PASS" if c["gn0_pass"] else "FAIL",
                 "FIRES" if c["gn1_fires"] else "NO",
                 c["dBPB_A1"], label(c["dBPB_A1"]),
                 c["dBPB_A2"], label(c["dBPB_A2"])))
    v = out["cells"].get("1.5B")
    if v:
        out["verdict"] = {"A1": label(v["dBPB_A1"]), "A2": label(v["dBPB_A2"]),
                          "readable": v["baseline_vs_chance"] < 0}
        print("VERDICT (1.5B, section 10.3): A1 %s, A2 %s"
              % (out["verdict"]["A1"], out["verdict"]["A2"]))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
