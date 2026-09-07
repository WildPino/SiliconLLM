#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E12 -- does the ternarization cost shrink with donor scale?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E12_TERNARY_COST_VS_SCALE.md (11a9d83).
Bands in section 4 are hard-coded below and the verdict is read off them mechanically.

WHY THIS EXISTS. E10 closed the engine's weight path: the packed kernel is on its ceiling at every
footprint and ledger 19.4's 37/42 GB/s rows are withdrawn. The remaining 7.4x to 50 tok/s is a
property of the MODEL, and the only route this programme has ever had is fewer active weights per
token at TERNARY precision. That route rests on a quality number measured at ONE shape --
probes/T2_TERNARIZATION_RULE.md:4 reads "Donor: Qwen2.5-1.5B". This draws the curve.

It does NOT re-derive the rule. `ternarize()` is imported from t1_ternarize, which is the single
definition the exporter and the parity gate both call. The only thing that varies across cells is
the donor.

Env:
  E12_SMOKE   1 = 2 sequences, 4 layers, 0.5B+1.5B only
  D_THREADS   torch threads (default 6)
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import t1_ternarize as T1  # noqa: E402   the conversion ORACLE, not a copy of it

SMOKE = os.environ.get("E12_SMOKE", "0") == "1"
SUFFIX = "_smoke" if SMOKE else ""
OUT = os.path.join(C.RESULTS, "e12_scale%s.json" % SUFFIX)

# Pinned to the commit hashes already present in the local HF cache.  The 1.5B hash is the SAME one
# common.py has used throughout this programme, so the middle cell is the historical point.
DONORS = [
    ("0.5B", "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    ("1.5B", "Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
    ("3B",   "Qwen/Qwen2.5-3B",   "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"),
]
if SMOKE:
    DONORS = DONORS[:2]

# The arms.  `base` is each donor's OWN fp32 baseline -- t1_ternarize.BASELINE_STANDING is a 1.5B
# constant and is deliberately NOT used here (brief section 3).
ARMS = [
    ("base", (),                0, "bitlinear158", "this donor's own fp32 baseline"),
    ("FA",   T1.FFN + T1.ATTN,  0, "bitlinear158", "PRIMARY -- the engine's conversion"),
    ("F",    T1.FFN,            0, "bitlinear158", "the planted control's COMPARATOR (same organs)"),
    ("Z",    T1.FFN,            0, "random_sign",  "PLANTED CONTROL -- mis-specified rule, SAME organs as F"),
    ("I",    T1.FFN + T1.ATTN,  0, "identity",     "INSTRUMENT CONTROL -- must equal base exactly"),
]

# Bands, verbatim from brief section 4.  r = dBPB(3B) / dBPB(0.5B).
SHRINK_MAX = 0.75
GROW_MIN = 1.25


def vocab_fingerprint(tok):
    """The eval slice is disk-cached and NOT keyed by model, so the three donors only share it if
    they share a tokenizer.  Verified rather than assumed."""
    v = tok.get_vocab()
    h = hashlib.sha256()
    for k in sorted(v, key=lambda s: v[s]):
        h.update(k.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()


def main():
    torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))
    t_start = time.time()
    out = {
        "brief": "docs/research/donor_adaptation/briefs/BRIEF_E12_TERNARY_COST_VS_SCALE.md @ 11a9d83",
        "rule": "imported from t1_ternarize.ternarize -- BitLinear158, the exporter's oracle",
        "smoke": SMOKE, "bands": {"SHRINK_MAX": SHRINK_MAX, "GROW_MIN": GROW_MIN},
        "cells": {}, "tokenizer_fingerprints": {},
    }
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            out["cells"] = prev.get("cells", {})
            out["tokenizer_fingerprints"] = prev.get("tokenizer_fingerprints", {})
        except Exception:
            pass

    for tag, model_id, rev in DONORS:
        if tag in out["cells"] and all(a[0] in out["cells"][tag]["arms"] for a in ARMS):
            print("== %-5s CACHED  dBPB = %.9f" % (tag, out["cells"][tag]["dBPB"]), flush=True)
            continue
        print("== %-5s %s @ %s" % (tag, model_id, rev[:8]), flush=True)
        C.MODEL_ID, C.REVISION = model_id, rev
        t0 = time.time()
        model, tok = C.load_model(dtype=torch.float32)
        model.eval()
        arch = C.arch(model)
        print("   loaded in %.0fs  %s" % (time.time() - t0, json.dumps(arch)), flush=True)

        fp = vocab_fingerprint(tok)
        out["tokenizer_fingerprints"][tag] = fp

        ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
        n_layers = None
        if SMOKE:
            ids, byts = ids[:2], byts[:2]
            n_layers = 4

        arms = {}
        for atag, organs, group, mode, note in ARMS:
            ta = time.time()
            restore, n_sub = ((lambda: None), 0) if not organs else \
                T1.apply_arm(model, organs, group, mode, n_layers)
            val = C.bpb(model, ids, byts)
            if organs:
                restore()
            arms[atag] = {"bpb": float(val), "n_substituted": n_sub, "mode": mode, "note": note,
                          "seconds": time.time() - ta}
            print("   %-5s BPB = %.9f   (%d tensors, %.0fs)" % (atag, val, n_sub, time.time() - ta),
                  flush=True)

        d = arms["FA"]["bpb"] - arms["base"]["bpb"]
        out["cells"][tag] = {
            "model_id": model_id, "revision": rev, "arch": arch, "slice": meta,
            "arms": arms, "dBPB": d,
            "dBPB_F": arms["F"]["bpb"] - arms["base"]["bpb"],
            "dBPB_Z": arms["Z"]["bpb"] - arms["base"]["bpb"],
            "I_minus_base": arms["I"]["bpb"] - arms["base"]["bpb"],
        }
        print("   dBPB(FA) = %.9f   dBPB(F) = %.9f   planted Z = %.9f   I-base = %+.3e"
              % (d, out["cells"][tag]["dBPB_F"], out["cells"][tag]["dBPB_Z"],
                 out["cells"][tag]["I_minus_base"]), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        del model
        C.MODEL_ID, C.REVISION = DONORS[1][1], DONORS[1][2]   # leave the module as we found it

    # ---------------------------------------------------------------- controls, then the verdict
    fps = set(out["tokenizer_fingerprints"].values())
    tok_ok = len(fps) == 1
    print("\nCONTROL  tokenizers identical across donors: %s  (%d distinct)"
          % ("YES" if tok_ok else "NO -- cells are NOT comparable", len(fps)), flush=True)

    for tag in out["cells"]:
        c = out["cells"][tag]
        # Z and F touch the SAME organs with a mis-specified and the real rule respectively, so
        # Z > F is the planted control firing.  Z against FA would compare different organ sets.
        print("CONTROL  %-5s planted Z = %+.6f vs F = %+.6f (SAME organs; Z>F = fires: %s),"
              "  instrument I-base = %+.3e"
              % (tag, c["dBPB_Z"], c["dBPB_F"],
                 "YES" if c["dBPB_Z"] > c["dBPB_F"] else "NO -- CONTROL FAILED",
                 c["I_minus_base"]), flush=True)

    have = [t for t, _, _ in DONORS if t in out["cells"]]
    print("\n%-6s %14s %14s %14s" % ("donor", "BPB fp32", "BPB ternary", "dBPB"))
    for t in have:
        c = out["cells"][t]
        print("%-6s %14.9f %14.9f %14.9f"
              % (t, c["arms"]["base"]["bpb"], c["arms"]["FA"]["bpb"], c["dBPB"]))

    if "0.5B" in out["cells"] and "3B" in out["cells"]:
        d0, d3 = out["cells"]["0.5B"]["dBPB"], out["cells"]["3B"]["dBPB"]
        r = d3 / d0 if d0 != 0 else float("nan")
        if r <= SHRINK_MAX:
            v = "COST-SHRINKS"
        elif r >= GROW_MIN:
            v = "COST-GROWS"
        else:
            v = "COST-FLAT"
        if "1.5B" in out["cells"] and v != "COST-FLAT":
            d1 = out["cells"]["1.5B"]["dBPB"]
            mono = (d0 >= d1 >= d3) if v == "COST-SHRINKS" else (d0 <= d1 <= d3)
            if not mono:
                v = "NON-MONOTONE"
        out["r"], out["verdict"] = r, v
        print("\nr = dBPB(3B)/dBPB(0.5B) = %.4f   ->   %s" % (r, v))
        print("bands: <=%.2f COST-SHRINKS, %.2f-%.2f COST-FLAT, >=%.2f COST-GROWS"
              % (SHRINK_MAX, SHRINK_MAX, GROW_MIN, GROW_MIN))
    else:
        print("\n(no verdict: needs both the 0.5B and 3B cells)")

    out["seconds_total"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s  (%.0fs)" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
