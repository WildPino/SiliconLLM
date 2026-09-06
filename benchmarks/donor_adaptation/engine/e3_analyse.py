#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3 -- turn results/e3/arms.json into the tables brief s7 asks for.

Definitions, stated here because every number below is one of them:

  WEIGHT ORGANS  = qkv_proj + o_proj + ffn + head.  These are the matvecs, and s2.2 of the probe
                   measured them value-independent, so they are the part that extrapolates.
  f              = rope + attention + norm+glue.  The NON-WEIGHT fixed cost.  This is the quantity
                   SPEED_LEDGER s12.2 reserves 1.17 ms for and calls "optimistic for a 10 B".
  rate, two conventions, and they must never be mixed:
      r_w    = active_weights / (time in the WEIGHT ORGANS).   S05: 493,961,216 / 16.659 = 29.65
      r_wall = active_weights / WALL time.                     S05: 493,961,216 / 17.833 = 27.70
  SPEED_LEDGER s12.2's "Delivered: 493,961,216 / 17.833 ms = 27.7 G-weights/s" is r_wall -- the
  fixed cost f is ALREADY INSIDE its denominator.

  budget@50 tok/s.  Each convention has exactly one self-consistent form:
      r_w    * (20 ms - f)   = 29.65 * 18.83 = 558 M
      r_wall * 20 ms         = 27.70 * 20    = 554 M
  INDEX s7's 522 M is `27.7 * (20 - 1.17)`, which subtracts f from the budget AFTER having already
  divided by it -- it charges the fixed cost TWICE.  Both tables are printed below so the two
  conventions can be read side by side and neither is smuggled into the other.

The organ table comes from the median rep (e3_bench.py picks it); ms/token from different reps are
not averaged, because they have different denominators.

The organ table comes from the median rep (e3_bench.py picks it); ms/token from different reps are
not averaged, because they have different denominators.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synth_export import SHAPES, active_weights

WEIGHT_ORGANS = ("qkv_proj", "o_proj", "ffn", "head")
FIXED_ORGANS = ("rope", "attention", "norm+glue")
LEDGER_RATE = 27.7e9
LEDGER_F = 1.17

# brief s5, written before any measurement existed
PREDICTED = {"S05": (56.1, 1.17), "S15": (17.9, 2.7), "S3": (9.0, 4.6),
             "M7": (3.9, 7.0), "Q8": (3.7, 7.9), "T10": (2.6, 9.0)}


def rows(path):
    recs = json.load(open(path, encoding="utf-8"))
    out = []
    for r in recs:
        arm, b = r["label"].rsplit("_b", 1)
        D, F, L, NH, NKV, HD, V, tied, src = SHAPES[arm]
        act = active_weights(D, F, L, NH, NKV, HD, V)
        o = r["median_rep_organs_ms"]
        wms = sum(o[k] for k in WEIGHT_ORGANS)
        f = sum(o[k] for k in FIXED_ORGANS)
        wall = r["median_rep_wall_ms_token"]
        rate_w = act / (wms * 1e-3)          # weights / time in the weight organs
        rate_wall = act / (wall * 1e-3)      # weights / wall time -- the LEDGER's convention
        # KV traffic: K and V, per layer, NKV*HD values of fp32, at every position up to `pos`.
        # --bench N feeds positions 1..N, so the mean position is (N+1)/2.
        pos = (int(b) + 1) / 2.0
        # Two denominators, and they must both be shown (feedback_charged_vs_moved_bytes):
        #   UNIQUE  -- what the cache actually holds.  kcache is [L][maxseq][NKV*HD] fp32.
        #   TOUCHED -- what the loop issues.  It runs over NH heads and each GQA group of NH/NKV
        #              heads re-reads the same kv slice, so the loop reads GQA times the unique
        #              bytes.  Whether those re-reads reach DRAM or hit L2 is NOT measured here.
        kv_bytes = 2.0 * L * NKV * HD * 4.0 * pos
        kv_touched = kv_bytes * (NH / float(NKV))
        out.append(dict(arm=arm, bench=int(b), src=src, act=act,
                        med=r["median_tok_s"], iqr=r["iqr_tok_s"], reps=r["tok_s"],
                        organs=o, wall=wall, D=D, F=F, L=L, NH=NH, NKV=NKV, HD=HD, V=V,
                        wms=wms, f=f, rate=rate_w, rate_wall=rate_wall,
                        kv_bytes=kv_bytes, kv_gbs=kv_bytes / (o["attention"] * 1e-3) / 1e9,
                        kv_touched=kv_touched, gqa=NH / float(NKV),
                        kv_gbs_touched=kv_touched / (o["attention"] * 1e-3) / 1e9,
                        budget=rate_w * max(0.0, 20.0 - f) * 1e-3,
                        budget_wall=rate_wall * 20.0 * 1e-3))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "e3", "arms.json")
    R = rows(path)
    for b in sorted({r["bench"] for r in R}):
        sel = [r for r in R if r["bench"] == b]
        print("\n=== bench %d ===" % b)
        print("| arm | active w/tok | predicted tok/s | measured tok/s | IQR | wall ms/tok | "
              "weight-organ ms | **f** ms | G-w/s | budget@50 (M) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in sel:
            p = PREDICTED[r["arm"]][0]
            print("| `%s` | %.3f B | %.1f | **%.3f** | %.3f | %.3f | %.3f | **%.3f** | %.1f | %.0f |"
                  % (r["arm"], r["act"] / 1e9, p, r["med"], r["iqr"], r["wall"], r["wms"], r["f"],
                     r["rate"] / 1e9, r["budget"] / 1e6))
        print("\n| arm | qkv_proj | rope | attention | o_proj | ffn | head | norm+glue |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sel:
            o = r["organs"]
            print("| `%s` | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
                  % (r["arm"], o["qkv_proj"], o["rope"], o["attention"], o["o_proj"],
                     o["ffn"], o["head"], o["norm+glue"]))
        print("\n| arm | predicted f (brief s5) | measured f | ratio | **f / 1.17** |")
        print("|---|---|---|---|---|")
        for r in sel:
            pf = PREDICTED[r["arm"]][1]
            print("| `%s` | %.2f | **%.3f** | %.2fx | **%.1fx** |"
                  % (r["arm"], pf, r["f"], r["f"] / pf, r["f"] / LEDGER_F))
        print("\n| arm | G-w/s r_w (organs) | G-w/s r_wall (LEDGER) | weight-organ GB/s | GQA | "
              "KV MB/tok unique | KV MB/tok touched | **attn GB/s unique** | attn GB/s touched |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in sel:
            wgbs = (r["act"] * 0.5) / (r["wms"] * 1e-3) / 1e9   # 0.5 bytes per packed weight
            print("| `%s` | %.1f | **%.1f** | %.1f | %.0f | %.1f | %.1f | **%.1f** | %.1f |"
                  % (r["arm"], r["rate"] / 1e9, r["rate_wall"] / 1e9, wgbs, r["gqa"],
                     r["kv_bytes"] / 1e6, r["kv_touched"] / 1e6,
                     r["kv_gbs"], r["kv_gbs_touched"]))
        print("\n| arm | budget@50: r_w x (20 - f) | budget@50: r_wall x 20 | INDEX s7 |")
        print("|---|---|---|---|")
        for r in sel:
            print("| `%s` | **%.0f M** | %.0f M | 522 M |"
                  % (r["arm"], r["budget"] / 1e6, r["budget_wall"] / 1e6))

    print("\n=== brief s6 decision inputs, T10 ===")
    for b in sorted({r["bench"] for r in R}):
        t = [r for r in R if r["arm"] == "T10" and r["bench"] == b]
        if not t:
            continue
        t = t[0]
        # s6's rate rule names the ledger's 27.7, which is r_wall.  Judged in the ledger's own
        # convention, or the comparison is a denominator swap -- the failure mode s12.1 withdrew.
        rw = t["rate_wall"]
        lab = ("RATE-HOLDS" if abs(rw - LEDGER_RATE) <= 0.10 * LEDGER_RATE
               else ("RATE-DEGRADES" if rw < 24.9e9 else "RATE-ABOVE-BAND (s6 has no such bucket)"))
        print("bench %d:  f = %.3f ms -> %s  |  r_wall = %.2f -> %s  (r_w = %.2f)"
              % (b, t["f"], "RESERVATION-HOLDS" if t["f"] <= 2.0 else "RESERVATION-BREAKS",
                 rw / 1e9, lab, t["rate"] / 1e9))
    print("\n=== the goal, as arithmetic ===")
    t10 = [r for r in R if r["arm"] == "T10" and r["bench"] == 300]
    if t10:
        t = t10[0]
        need = t["act"] / 20e-3
        print("dense T10 at 50 tok/s needs %.0f G-w/s; measured %.1f -> %.1fx short"
              % (need / 1e9, t["rate"] / 1e9, need / t["rate"]))
        print("at the measured rate and measured f, the 50 tok/s weight budget is %.0f M "
              "(INDEX s7 uses 522 M)" % (t["budget"] / 1e6))


if __name__ == "__main__":
    main()
