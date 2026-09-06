#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 -- results/e5/{parity.txt,arms.json} against the gates of BRIEF_E5_DECOMPOSE_R.md (c1bdd70).

The quantity under test is `R`, the 81.4% of the attention organ E4 did not touch.  Everything
here is a DIFFERENCE between two arms measured minutes apart in the same sweep; no number from
another sweep enters a ratio (gate G3, from E4 s2.5 -- between-sweep dispersion on this machine
is 5-10% while within-sweep is 0.000-0.015 tok/s).

    R  = organ(serial) - X_serial,  X_serial = organ(serial2) - organ(serial)     [E4's method]
    S  = organ(avx4,sm2) - organ(avx4,none)          softmax pass
    Y  = organ(avx4,av2) - organ(avx4,none)          A.V loop
    P  = R - S - Y                                   everything that is neither loop
    X_avx4 = organ(avx4,none) - R

S and Y fit their own 1x/2x points by construction and prove nothing there.  The 3x arms are the
test: sm3 must land on organ + 2S and av3 on organ + 2Y, within 3% (brief s4, G1).
"""
import io
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "e5")
FIXED = ("rope", "attention", "norm+glue")
# G0 (brief s8.3, pre-registered after run 1's VOID): no E5 arm touches the weight path, so the sum
# of these four organs is invariant across the arms of a cell.  Any measurement whose W exceeds its
# cell's MINIMUM W by more than G0_TOL is a contended timing and is discarded before any difference
# is taken.  Run 1 died of exactly this: +19.4% and +30.4% of W spread inside the T10 cells, with
# the three arms carrying the excess producing NEGATIVE components.
WEIGHT_ORGANS = ("qkv_proj", "o_proj", "ffn", "head")
G0_TOL = 0.05
G0_MIN_SURVIVORS = 2
SCORED_BYTES = 51870
SHAPES = {"S05": (896, 4864, 24, 14, 2, 64, 151936),
          "T10": (4096, 14336, 48, 32, 8, 128, 32768)}
TOL = 0.03            # brief s4: the 3x prediction's tolerance, E4's own
MIN_DELTA_MS = 0.30   # brief s4: a 2x arm that moves less than this at T10 @800 is INCONCLUSIVE
# brief s3, pre-registered before the run.  ms at T10 @800.
PRE = {"S": (0.8, 2.0), "Y": (2.2, 3.5), "P": (4.0, 7.0), "fork": (0.1, 1.0)}
# E4's published numbers at T10 @800, for the drift line ONLY -- never a denominator here.
E4_ORGAN_AVX4, E4_R, E4_X = 12.058, 9.816, 2.242


def bpb(n):
    return n / math.log(2.0) / SCORED_BYTES


def parity():
    p = os.path.join(RES, "parity.txt")
    if not os.path.exists(p):
        return None
    txt = io.open(p, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"== parity (\w+) ==\s*NATS_TOTAL ([0-9.eE+-]+)", txt):
        out[m.group(1)] = m.group(2)          # kept as TEXT: G2 asks for bit-identity
    return out


def load(path):
    """Group every measurement by (shape, bench, attn, attnr), apply G0, then take the median of
    the survivors.  Run 2 is rep-major, so a key carries three separate single-rep records; run 1
    was arm-major and carries one record per key.  Both load; only the first can survive G0 when
    the machine misbehaves, which is the point of the change."""
    raw = {}
    for r in json.load(open(path, encoding="utf-8")):
        shape = r["label"].split("_")[0]
        o = r["median_rep_organs_ms"]
        key = (shape, r["bench"], r.get("attn", "serial"), r.get("attnr", "none"))
        raw.setdefault(key, []).append(
            dict(organs=o, tok_s=r["median_tok_s"], iqr=r["iqr_tok_s"],
                 W=sum(o[k] for k in WEIGHT_ORGANS), attn_ms=o["attention"],
                 f=sum(o[k] for k in FIXED)))
    # G0 is a CELL-level test: the minimum W of the whole cell is the uncontended reference.
    cells = {}
    for (shape, b, at, ar), v in raw.items():
        cells.setdefault((shape, b), []).extend(v)
    floor = {k: min(x["W"] for x in v) for k, v in cells.items()}
    A, G0 = {}, {}
    for (shape, b, at, ar), v in sorted(raw.items()):
        w0 = floor[(shape, b)]
        keep = [x for x in v if x["W"] <= w0 * (1.0 + G0_TOL)]
        G0.setdefault((shape, b), []).append((at, ar, len(v), len(keep),
                                              max(x["W"] for x in v) / w0 - 1.0))
        if len(keep) < min(G0_MIN_SURVIVORS, len(v)):
            continue                      # arm has no clean measurement; cell will be voided
        D, F, L, NH, NKV, HD, V = SHAPES[shape]
        pos = (b + 1) / 2.0
        med = lambda f: sorted(f(x) for x in keep)[len(keep) // 2]
        A[(shape, b, at, ar)] = dict(
            shape=shape, bench=b, attn=at, attnr=ar, n=len(v), n_kept=len(keep),
            tok_s=med(lambda x: x["tok_s"]), iqr=med(lambda x: x["iqr"]),
            attn_ms=med(lambda x: x["attn_ms"]), f=med(lambda x: x["f"]),
            W=med(lambda x: x["W"]), pos=pos, L=L, NH=NH, NKV=NKV, HD=HD,
            expf_calls=L * NH * pos, v_unique=L * NKV * HD * pos * 4.0,
            v_touched=L * NH * HD * pos * 4.0)
    return A, G0


def report_g0(G0):
    print("## G0 -- the weight path is invariant across arms; anything else is the machine "
          "(brief s8.3)\n")
    print("| cell | measurements | kept | discarded | worst `W` excess | verdict |")
    print("|---|---|---|---|---|---|")
    dead = set()
    for (shape, b), rows in sorted(G0.items()):
        n = sum(r[2] for r in rows)
        k = sum(r[3] for r in rows)
        worst = max(r[4] for r in rows)
        starved = [r for r in rows if r[3] < min(G0_MIN_SURVIVORS, r[2])]
        ok = not starved
        if not ok:
            dead.add((shape, b))
        print("| %s @%d | %d | %d | %d | +%.1f%% | %s |"
              % (shape, b, n, k, n - k, 100.0 * worst,
                 "PASS" if ok else "**VOID -- %d arm(s) with no clean measurement: %s**"
                 % (len(starved), ", ".join(r[1] for r in starved))))
    print()
    return dead


def cell(A, shape, b):
    """The decomposition at one (shape, context), or None if the sweep is missing an arm."""
    g = lambda at, ar: A.get((shape, b, at, ar))
    s1, s2, a0 = g("serial", "none"), g("serial2", "none"), g("avx4", "none")
    if not (s1 and s2 and a0):
        return None
    c = dict(shape=shape, bench=b, base=a0, serial=s1, serial2=s2)
    c["X_serial"] = s2["attn_ms"] - s1["attn_ms"]
    c["R"] = s1["attn_ms"] - c["X_serial"]
    c["X_avx4"] = a0["attn_ms"] - c["R"]
    for name, two, three in (("S", "sm2", "sm3"), ("Y", "av2", "av3")):
        t2, t3 = g("avx4", two), g("avx4", three)
        c[name] = (t2["attn_ms"] - a0["attn_ms"]) if t2 else None
        if t2 and t3:
            pred = a0["attn_ms"] + 2.0 * c[name]
            c[name + "_pred"], c[name + "_meas"] = pred, t3["attn_ms"]
            c[name + "_err"] = (t3["attn_ms"] - pred) / pred
    fk = g("avx4", "fork2")
    c["fork"] = (fk["attn_ms"] - a0["attn_ms"]) if fk else None
    if c["S"] is not None and c["Y"] is not None:
        c["P"] = c["R"] - c["S"] - c["Y"]
    return c


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "arms.json")
    A, G0 = load(path)

    P = parity()
    if P:
        print("## G2 -- parity must be BIT-IDENTICAL (every arm is value-preserving by "
              "construction)\n")
        print("| arm | NATS_TOTAL (as printed) | vs `none` |")
        print("|---|---|---|")
        base = P.get("none")
        for a in ("none", "sm2", "sm3", "av2", "av3", "fork2"):
            if a not in P:
                continue
            tag = ("baseline" if a == "none" else
                   ("**BIT-IDENTICAL**" if P[a] == base else
                    "**FAIL -- arm void (dBPB %+.3e)**" % (bpb(float(P[a])) - bpb(float(base)))))
            print("| `%s` | %s | %s |" % (a, P[a], tag))
        print("\nBPB = %.9f on the pinned 24x512 slice (%d scored bytes).\n"
              % (bpb(float(base)), SCORED_BYTES))

    dead = report_g0(G0)
    cells = [c for c in (cell(A, s, b) for s in ("T10", "S05") for b in (300, 800)) if c
             and (c["shape"], c["bench"]) not in dead]
    if not cells:
        print("arms.json has no complete cell yet")
        return
    T = next((c for c in cells if c["shape"] == "T10" and c["bench"] == 800), None)

    # ---------------- G3: drift, named as drift, and kept out of every ratio
    print("## G3 -- the sweep against E4's published numbers (drift line, not an input)\n")
    if T:
        print("| quantity | E4 (2026-09-06, its own sweep) | this sweep | drift |")
        print("|---|---|---|---|")
        for nm, old, new in (("organ, `avx4`", E4_ORGAN_AVX4, T["base"]["attn_ms"]),
                             ("`R`", E4_R, T["R"]), ("`X` after `avx4`", E4_X, T["X_avx4"])):
            print("| %s | %.3f | %.3f | %+.1f%% |" % (nm, old, new, 100.0 * (new - old) / old))
        print("\nEvery number below is a difference between two arms of THIS sweep.  The table "
              "above is reported so the reader can see the machine move; it is not used.\n")

    # ---------------- G1: the 3x points, which the model can miss
    print("## G1 -- the 3x arms, which S and Y are not fitted to (brief s4)\n")
    print("| point | organ `none` | 2x arm | solved | predicted 3x | **measured 3x** | error | "
          "verdict |")
    print("|---|---|---|---|---|---|---|---|")
    fails = []
    for c in cells:
        for nm, lbl in (("S", "sm"), ("Y", "av")):
            if c.get(nm + "_err") is None:
                continue
            ok = abs(c[nm + "_err"]) <= TOL
            small = (c[nm] < MIN_DELTA_MS and c["shape"] == "T10" and c["bench"] == 800)
            v = ("**INCONCLUSIVE** (2x moved %.3f < %.2f ms)" % (c[nm], MIN_DELTA_MS) if small
                 else ("PASS" if ok else "**FAIL**"))
            if not ok and not small:
                fails.append((c["shape"], c["bench"], nm))
            print("| %s @%d `%s` | %.3f | %.3f | %s = %.3f | %.3f | **%.3f** | %+.2f%% | %s |"
                  % (c["shape"], c["bench"], lbl + "2/3", c["base"]["attn_ms"],
                     c["base"]["attn_ms"] + c[nm], nm, c[nm], c[nm + "_pred"],
                     c[nm + "_meas"], 100.0 * c[nm + "_err"], v))
    print()
    if fails:
        print("**%d prediction(s) missed by more than %.0f%% -> the decomposition is not "
              "established and E5 is VOID for those components.**\n" % (len(fails), 100 * TOL))

    # ---------------- the decomposition
    print("## The decomposition of `R`\n")
    print("| point | organ `avx4` | `X` | `S` softmax | `Y` A.V | `P` rest | "
          "S/R | Y/R | P/R | `f` | R/f |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        if c.get("P") is None:
            continue
        R = c["R"]
        print("| %s @%d | %.3f | %.3f | **%.3f** | **%.3f** | **%.3f** | %.1f%% | %.1f%% | "
              "%.1f%% | %.3f | %.1f%% |"
              % (c["shape"], c["bench"], c["base"]["attn_ms"], c["X_avx4"], c["S"], c["Y"],
                 c["P"], 100.0 * c["S"] / R, 100.0 * c["Y"] / R, 100.0 * c["P"] / R,
                 c["base"]["f"], 100.0 * R / c["base"]["f"]))
    print()

    print("## `fork2` -- one extra OpenMP region per layer, priced directly\n")
    print("| point | regions/token | extra ms | us per region | as %% of `P` |")
    print("|---|---|---|---|---|")
    for c in cells:
        if c.get("fork") is None:
            continue
        reg = c["base"]["L"]
        print("| %s @%d | %d | %.3f | %.1f | %s |"
              % (c["shape"], c["bench"], reg, c["fork"], 1000.0 * c["fork"] / reg,
                 ("%.0f%%" % (100.0 * c["fork"] / c["P"])) if c.get("P") else "-"))
    print()

    # ---------------- how the pre-registered predictions did
    if T:
        print("## Brief s3's predictions, scored\n")
        print("| term | pre-registered range (ms) | measured | verdict |")
        print("|---|---|---|---|")
        for k, nm in (("S", "S"), ("Y", "Y"), ("P", "P"), ("fork", "fork")):
            lo, hi = PRE[k]
            v = T.get(nm if nm != "fork" else "fork")
            if v is None:
                continue
            print("| `%s` | %.1f - %.1f | **%.3f** | %s |"
                  % (k, lo, hi, v, "inside" if lo <= v <= hi else "**OUTSIDE -- my estimate "
                     "was wrong, and by %+.0f%%**"
                     % (100.0 * (v - (hi if v > hi else lo)) / (hi if v > hi else lo))))
        print()

    # ---------------- s5: the label
    if T and T.get("P") is not None:
        print("## s5 -- the label, at T10 @800\n")
        R = T["R"]
        parts = [("SOFTMAX-DOMINATED", "S", T["S"]), ("AV-DOMINATED", "Y", T["Y"]),
                 ("OVERHEAD-DOMINATED", "P", T["P"])]
        parts.sort(key=lambda x: -x[2])
        top = parts[0]
        share = top[2] / R
        near = [p for p in parts if abs(p[2] / R - 0.50) <= 0.03]
        lab = top[0] if share >= 0.50 else "SPLIT"
        if len(near) >= 2:
            lab = "SPLIT"
        print("`R` = %.3f ms = `S` %.3f + `Y` %.3f + `P` %.3f;  largest is `%s` at %.1f%% "
              "-> **%s**" % (R, T["S"], T["Y"], T["P"], top[1], 100.0 * share, lab))
        if near and share < 0.50:
            print("(the leading share is within 3 points of the 0.50 boundary -- brief s4 G4: "
                  "the label is withheld until an interleaved run decides it)")
        f = T["base"]["f"]
        print("\n| if this term went to zero | `f` ms | **ceiling 1000/f** |")
        print("|---|---|---|")
        print("| nothing (measured today) | %.3f | **%.1f** |" % (f, 1000.0 / f))
        for _, nm, v in sorted(parts, key=lambda x: -x[2]):
            print("| `%s` | %.3f | **%.1f** |" % (nm, f - v, 1000.0 / (f - v)))
        print("| all of `R` | %.3f | **%.1f** |" % (f - R, 1000.0 / (f - R)))
        print("\n`Y` reads V at %.1f MB unique / %.1f MB touched per token; at the measured "
              "%.3f ms that is %.1f GB/s unique, %.1f GB/s touched (E4 measured this machine's "
              "DRAM at 35.1 GB/s on the same path)."
              % (T["base"]["v_unique"] / 1e6, T["base"]["v_touched"] / 1e6, T["Y"],
                 T["base"]["v_unique"] / (T["Y"] * 1e-3) / 1e9,
                 T["base"]["v_touched"] / (T["Y"] * 1e-3) / 1e9))
        print("`S` runs %.0f scalar expf per token; at the measured %.3f ms on 6 threads that "
              "is %.1f ns per call, %.0f cycles at 3.6 GHz."
              % (T["base"]["expf_calls"], T["S"],
                 T["S"] * 1e6 / T["base"]["expf_calls"] * 6.0,
                 T["S"] * 1e-3 / T["base"]["expf_calls"] * 6.0 * 3.6e9))


if __name__ == "__main__":
    main()
