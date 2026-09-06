#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4 -- turn results/e4/{parity.txt,arms.json} into the tables the brief asks for.

BRIEF_E4_ATTENTION_ACCUMULATORS.md, pushed at 25a6d29 before any arm existed.

The quantity under test is the `attention` ORGAN, not tok/s.  At T10 @300 the whole organ is
9.862 of a 323.6 ms token, so even deleting it buys +3.1%.  What it bounds is

    f = rope + attention + norm+glue        (the cost no weight format can remove)
    1000 / f                                (the ceiling with an infinitely fast weight path)

and at T10 @800 attention is 25.385 of the 26.088 ms of f -- 97.3% of it.  E3 measured that
ceiling at 38.3 tok/s, BELOW the 50 tok/s goal, which is the only reason this loop is worth
touching at all.

CONVENTIONS (feedback_charged_vs_moved_bytes -- a ceiling is a denominator):
  K TOUCHED   = L*NH*HD*pos*4     what the dot loop issues (GQA re-reads included)
  V TOUCHED   = L*NH*HD*pos*4     what the A.V loop issues
  UNIQUE      = 2*L*NKV*HD*pos*4  what the cache actually holds
Whether the re-reads reach DRAM or hit L2 is NOT measured here and is not claimed.
"""
import io
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "e4")
FIXED = ("rope", "attention", "norm+glue")
SCORED_BYTES = 51870          # the pinned 24x512 slice, asserted by e1_bpb_through_engine.py:353
SIGMA_SEED = 0.005            # R1, the constant every quality delta is judged against
SHAPES = {"S05": (896, 4864, 24, 14, 2, 64, 151936),
          "T10": (4096, 14336, 48, 32, 8, 128, 32768)}
# E3 s3, for gate G4 -- the baseline must reproduce itself under the new binary.
E3_BASELINE = {("T10", 300): (3.090, 9.862), ("T10", 800): (2.960, 25.385),
               ("S05", 300): (55.700, 1.096), ("S05", 800): (49.280, 2.847)}
ARM_ORDER = ["serial", "serial2", "ilp4", "avx1", "avx4"]


def parity():
    p = os.path.join(RES, "parity.txt")
    if not os.path.exists(p):
        return None
    txt = io.open(p, encoding="utf-8").read()
    out = {}
    pat = (r"== parity (\w+) ==\s*NATS_TOTAL ([0-9.eE+-]+)\s*"
           r"N_PREDICTED (\d+)\s*NATS_PER_TOKEN ([0-9.eE+-]+)")
    for m in re.finditer(pat, txt):
        out[m.group(1)] = (float(m.group(2)), int(m.group(3)), float(m.group(4)))
    return out


def bpb(nats):
    return nats / math.log(2.0) / SCORED_BYTES


def arms():
    p = os.path.join(RES, "arms.json")
    if not os.path.exists(p):
        return []
    R = json.load(open(p, encoding="utf-8"))
    out = []
    for r in R:
        shape = r["label"].split("_")[0]
        D, F, L, NH, NKV, HD, V = SHAPES[shape]
        pos = (r["bench"] + 1) / 2.0
        o = r["median_rep_organs_ms"]
        f = sum(o[k] for k in FIXED)
        k_touch = L * NH * HD * pos * 4.0
        out.append(dict(shape=shape, arm=r.get("attn", "serial"), bench=r["bench"],
                        tok_s=r["median_tok_s"], iqr=r["iqr_tok_s"], organs=o, f=f,
                        attn=o["attention"], wall=r["median_rep_wall_ms_token"],
                        ceiling=1000.0 / f,
                        k_touched=k_touch, touched=2.0 * k_touch,
                        unique=2.0 * L * NKV * HD * pos * 4.0,
                        gbs_touched=(2.0 * k_touch) / (o["attention"] * 1e-3) / 1e9))
    return out


def main():
    P = parity()
    print("## G2 -- parity on the real donor (qwen25-05b_tqh, pinned 24x512, %d scored bytes)\n"
          % SCORED_BYTES)
    if not P:
        print("  parity.txt not present yet\n")
    else:
        base = P.get("serial")
        print("| arm | NATS_TOTAL | BPB | dBPB vs serial | vs sigma_seed=0.005 |")
        print("|---|---|---|---|---|")
        for a in ARM_ORDER:
            if a not in P:
                continue
            n = P[a][0]
            d = bpb(n) - bpb(base[0]) if base else float("nan")
            if base and n == base[0]:
                tag = "**BIT-IDENTICAL**"
            elif abs(d) < SIGMA_SEED:
                tag = "PASS (%.0fx under)" % (SIGMA_SEED / abs(d)) if d else "PASS"
            else:
                tag = "**FAIL**"
            print("| `%s` | %.10f | %.9f | %+.3e | %s |" % (a, n, bpb(n), d, tag))
        print()

    A = arms()
    if not A:
        print("arms.json not present yet")
        return
    for shape in ("T10", "S05"):
        for b in sorted({r["bench"] for r in A if r["shape"] == shape}):
            sel = [r for r in A if r["shape"] == shape and r["bench"] == b]
            sel.sort(key=lambda r: ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99)
            base = next((r for r in sel if r["arm"] == "serial"), None)
            print("### %s, --bench %d\n" % (shape, b))
            print("| arm | tok/s | IQR | **attention ms** | vs serial | f ms | **1000/f** | "
                  "attn GB/s touched |")
            print("|---|---|---|---|---|---|---|---|")
            for r in sel:
                ratio = (r["attn"] / base["attn"]) if base else float("nan")
                print("| `%s` | %.3f | %.3f | **%.3f** | %.3fx | %.3f | **%.1f** | %.1f |"
                      % (r["arm"], r["tok_s"], r["iqr"], r["attn"], ratio, r["f"],
                         r["ceiling"], r["gbs_touched"]))
            print()

    # ---- G4: does the baseline reproduce E3 under the new binary?
    print("## G4 -- the baseline must reproduce itself (E3 s3 vs this binary, --attn serial)\n")
    print("| point | E3 tok/s | now | E3 attention | now | delta |")
    print("|---|---|---|---|---|---|")
    for r in A:
        if r["arm"] != "serial":
            continue
        key = (r["shape"], r["bench"])
        if key not in E3_BASELINE:
            continue
        t3, a3 = E3_BASELINE[key]
        print("| %s @%d | %.3f | %.3f | %.3f | %.3f | %+.1f%% tok/s, %+.1f%% attn |"
              % (r["shape"], r["bench"], t3, r["tok_s"], a3, r["attn"],
                 100.0 * (r["tok_s"] - t3) / t3, 100.0 * (r["attn"] - a3) / a3))
    print()

    # ---- G1: the planted positive
    print("## G1 -- planted positive (`serial2` = the same dot product twice, bit-identical)\n")
    for b in sorted({r["bench"] for r in A if r["shape"] == "T10"}):
        s = next((r for r in A if r["shape"] == "T10" and r["bench"] == b
                  and r["arm"] == "serial"), None)
        s2 = next((r for r in A if r["shape"] == "T10" and r["bench"] == b
                   and r["arm"] == "serial2"), None)
        if not (s and s2):
            continue
        rr = s2["attn"] / s["attn"]
        print("T10 @%d: attention %.3f -> %.3f = **%.2fx**  -> %s"
              % (b, s["attn"], s2["attn"], rr,
                 "FIRES (>=1.7x required)" if rr >= 1.7
                 else "**DOES NOT FIRE -> E4 is VOID**"))
    print()

    # ---- s5 decision rule, judged at T10 @800
    print("## s5 -- the label, judged on the attention organ at T10 @800\n")
    sel = [r for r in A if r["shape"] == "T10" and r["bench"] == 800]
    base = next((r for r in sel if r["arm"] == "serial"), None)
    cand = [r for r in sel if r["arm"] in ("ilp4", "avx1", "avx4")]
    if base and cand:
        best = min(cand, key=lambda r: r["attn"])
        ratio = best["attn"] / base["attn"]
        lab = ("LATENCY-CONFIRMED" if ratio <= 0.50 else
               ("BANDWIDTH-BOUND" if ratio >= 0.90 else "MIXED"))
        print("best arm `%s`: %.3f ms vs %.3f = **%.3fx**  ->  **%s**"
              % (best["arm"], best["attn"], base["attn"], ratio, lab))
        print("f %.3f -> %.3f ms;  ceiling 1000/f %.1f -> **%.1f tok/s**;  "
              "tok/s %.3f -> %.3f (%+.1f%%)"
              % (base["f"], best["f"], base["ceiling"], best["ceiling"],
                 base["tok_s"], best["tok_s"],
                 100.0 * (best["tok_s"] - base["tok_s"]) / base["tok_s"]))
        print("achieved load rate on the attention organ: %.1f GB/s touched "
              "(%.1f GB/s unique) -- the brief predicted the floor would be memory"
              % (best["gbs_touched"], best["unique"] / (best["attn"] * 1e-3) / 1e9))


if __name__ == "__main__":
    main()
