#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E12 section 9 -- how much of arm Z is the RULE and how much is the DRAW?

E12 returned CONTROL-FAILED because at 3B the mis-specified rule Z beat the real rule F by 1.490
BPB.  Before that can be read as a property of ternarization at scale, arm Z has to be shown to be
a STABLE quantity.  Two facts say it may not be:

  1. T2 (probes/T2_TERNARIZATION_RULE.md section 3) measured arm Z at 1.5B, on the SAME eval slice
     (ids_sha256 a1a48dc9...), the SAME organs and the SAME rule, and got +3.372681.  E12 got
     +4.001257 at that cell.  The two differ ONLY in which per-tensor seed lands on which tensor
     (t2_rules.py seeds 1000+stats["n"], t1_ternarize.apply_arm seeds 1000+rng, off by one).
     0.628 BPB = 126 sigma_seed from an off-by-one in a random seed.
  2. T2 section 4(a) already measured R0 - Z = -0.064 +/- 0.126, ci95 straddling zero, and wrote
     that arm Z "was the wrong control".  t1_ternarize.ternarize's own docstring records the same
     amendment, dated 2026-09-04.  E12 re-adopted a control this programme had already retired.

This measures the dispersion directly: the same arm Z, K draws, at each cell.

BANDS, FIXED BEFORE THE RUN.  s = max - min of dBPB_Z over the K draws at a cell.
  Z-UNSTABLE  s(3B) >= 1.490   the observed Z-F gap is inside the comparator's own spread ->
                               the "control failure" carries NO information about the rule
  Z-NOISY     0.10 <= s(3B) < 1.490
  Z-STABLE    s(3B) <  0.10    arm Z is a real quantity and the 3B inversion needs explaining

REPLICATION GATE.  seed_base=1000 must reproduce e12_scale.json's dBPB_Z at every cell to the last
digit.  If it does not, apply_arm's new seed_base parameter changed behaviour and nothing here
counts.

PREDICTION, ON THE RECORD: Z-UNSTABLE.  Derived from a MEASURED quantity (the 0.628 T2-vs-E12 gap
at 1.5B, two draws of the identical estimand) over a STRUCTURAL factor (3B has 1.5x the layers and
1.2x the width of 1.5B, so more independently-seeded tensors, not fewer).

Env: D_THREADS (default 6), E12Z_K (default 5), E12Z_CELLS (default "0.5B,1.5B,3B")
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import t1_ternarize as T1  # noqa: E402

OUT = os.path.join(C.RESULTS, "e12_zvar.json")
REF = os.path.join(C.RESULTS, "e12_scale.json")

DONORS = [
    ("0.5B", "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    ("1.5B", "Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
    ("3B",   "Qwen/Qwen2.5-3B",   "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"),
]
K = int(os.environ.get("E12Z_K", "5"))
SEED_BASES = [1000 + 1000 * i for i in range(K)]
WANT = os.environ.get("E12Z_CELLS", "0.5B,1.5B,3B").split(",")

UNSTABLE_MIN = 1.490   # the observed |Z - F| at 3B, from e12_scale.json
STABLE_MAX = 0.10


def main():
    torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))
    t_start = time.time()
    ref = json.load(open(REF))["cells"]
    out = {"bands": {"UNSTABLE_MIN": UNSTABLE_MIN, "STABLE_MAX": STABLE_MAX},
           "seed_bases": SEED_BASES, "cells": {}}
    if os.path.exists(OUT):
        try:
            out["cells"] = json.load(open(OUT)).get("cells", {})
        except Exception:
            pass

    for tag, model_id, rev in DONORS:
        if tag not in WANT:
            continue
        if tag in out["cells"] and len(out["cells"][tag]["draws"]) >= K:
            print("== %-5s CACHED" % tag, flush=True)
            continue
        print("== %-5s %s @ %s" % (tag, model_id, rev[:8]), flush=True)
        C.MODEL_ID, C.REVISION = model_id, rev
        t0 = time.time()
        model, tok = C.load_model(dtype=torch.float32)
        model.eval()
        print("   loaded in %.0fs" % (time.time() - t0), flush=True)
        ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
        assert meta["ids_sha256"] == ref[tag]["slice"]["ids_sha256"], "slice drift -- stop"
        base = ref[tag]["arms"]["base"]["bpb"]

        draws = out["cells"].get(tag, {}).get("draws", {})
        for sb in SEED_BASES:
            if str(sb) in draws:
                continue
            ta = time.time()
            restore, n = T1.apply_arm(model, T1.FFN, 0, "random_sign", None, seed_base=sb)
            val = float(C.bpb(model, ids, byts))
            restore()
            draws[str(sb)] = {"bpb": val, "dBPB": val - base, "n_substituted": n,
                              "seconds": time.time() - ta}
            print("   seed_base %-5d  BPB = %.9f   dBPB = %.9f   (%d tensors, %.0fs)"
                  % (sb, val, val - base, n, time.time() - ta), flush=True)
            out["cells"][tag] = {"base": base, "dBPB_F": ref[tag]["dBPB_F"],
                                 "dBPB_Z_e12": ref[tag]["dBPB_Z"], "draws": draws}
            json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

        d = [v["dBPB"] for v in draws.values()]
        c = out["cells"][tag]
        c["min"], c["max"], c["spread"] = min(d), max(d), max(d) - min(d)
        c["mean"] = sum(d) / len(d)
        c["repro_ok"] = abs(draws["1000"]["dBPB"] - ref[tag]["dBPB_Z"]) < 1e-12
        c["n_beating_F"] = sum(1 for x in d if x < ref[tag]["dBPB_F"])
        print("   %-5s spread %.6f over %d draws  [%.6f .. %.6f]   F = %.6f   "
              "draws beating F: %d/%d   repro(seed 1000) %s"
              % (tag, c["spread"], len(d), c["min"], c["max"], ref[tag]["dBPB_F"],
                 c["n_beating_F"], len(d), "OK" if c["repro_ok"] else "*** FAIL ***"), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        del model
        C.MODEL_ID, C.REVISION = DONORS[1][1], DONORS[1][2]

    print("\n---- verdict ----", flush=True)
    bad = [t for t, c in out["cells"].items() if not c.get("repro_ok", True)]
    if bad:
        print("REPLICATION GATE FAILED at %s -- nothing below counts" % ",".join(bad))
        return
    print("replication gate: seed_base 1000 reproduces e12_scale.json at every cell  OK")
    for tag in ("0.5B", "1.5B", "3B"):
        c = out["cells"].get(tag)
        if not c:
            continue
        print("  %-5s  Z draws %.6f .. %.6f  spread %.6f  mean %.6f | F %.6f | beating F %d/%d"
              % (tag, c["min"], c["max"], c["spread"], c["mean"], c["dBPB_F"],
                 c["n_beating_F"], len(c["draws"])))
    s3 = out["cells"].get("3B", {}).get("spread")
    if s3 is not None:
        v = "Z-UNSTABLE" if s3 >= UNSTABLE_MIN else ("Z-STABLE" if s3 < STABLE_MAX else "Z-NOISY")
        print("VERDICT  spread(3B) = %.6f  ->  %s" % (s3, v))
    out["elapsed_s"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("total %.0fs" % out["elapsed_s"])


if __name__ == "__main__":
    main()
