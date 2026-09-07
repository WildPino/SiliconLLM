#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 run 5 -- results/e5/{parity5.txt,sweep5.json} against the gates of BRIEF_E5_DECOMPOSE_R.md s11.

Every number here is a difference between two arms measured INSIDE THE SAME PROCESS, interleaved
per token under a palindrome schedule (s11.4).  Nothing from another process, and nothing from
another sweep, enters any difference.

    R  = organ(serial) - X_serial,  X_serial = organ(serial2) - organ(serial)     [E4's method]
    S  = organ(sm2) - organ(sm1)     softmax pass, both points inside the wrapped path (s10.3)
    Y  = organ(av2) - organ(av1)     A.V loop, likewise
    P  = R - S - Y                   everything that is neither loop
    X_avx4 = organ(none) - R

S and Y fit their own 1x/2x points by construction and prove nothing there.  The 3x arms are the
test: sm3 must land on sm1 + 2S and av3 on av1 + 2Y, within 3% (brief s4 G1).
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
WEIGHT_ORGANS = ("qkv_proj", "o_proj", "ffn", "head")
# s11.5: G0 is now a WITHIN-PROCESS test.  No arm touches the weight path and all ten arms share
# one process, so W must agree across the arms of a single run.  Same 1% as s10.4, but it now asks
# whether the interleave spread the machine evenly -- not whether the machine held still for hours.
G0_TOL = 0.01
MIN_RUNS = 3          # s11.5: fewer than three surviving runs voids the cell
TOL = 0.03            # brief s4: the 3x prediction's tolerance, E4's own
MIN_DELTA_MS = 0.30   # brief s4: a 2x arm that moves less than this at T10 @800 is INCONCLUSIVE
SCORED_BYTES = 51870
PRE = {"S": (0.8, 2.0), "Y": (2.2, 3.5), "P": (4.0, 7.0), "fork": (0.1, 1.0)}
E4_ORGAN_AVX4, E4_R, E4_X = 12.058, 9.816, 2.242
ARMS = ("serial", "serial2", "none", "sm1", "sm2", "sm3", "av1", "av2", "av3", "fork2")


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def parity():
    p = os.path.join(RES, "parity5.txt")
    if not os.path.exists(p):
        return None
    txt = io.open(p, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"== parity ([\w ()-]+?) ==\s*NATS_TOTAL ([0-9.eE+-]+)", txt):
        out[m.group(1).strip()] = m.group(2)      # kept as TEXT: G2 asks for bit-identity
    return out


def solve(run):
    """One process -> one complete set of components.  Every term is an arm of THIS run."""
    a = {k: run["arms"][k]["organs"]["attention"] for k in ARMS}
    W = {k: sum(run["arms"][k]["organs"][o] for o in WEIGHT_ORGANS) for k in ARMS}
    f = {k: sum(run["arms"][k]["organs"][o] for o in FIXED) for k in ARMS}
    X_serial = a["serial2"] - a["serial"]
    R = a["serial"] - X_serial
    c = dict(shape=run["shape"], bench=run["bench"], run=run["run"],
             cores_busy=run.get("cores_busy"), tok_s=run.get("tok_s"),
             organ=a["none"], R=R, X_serial=X_serial, X_avx4=a["none"] - R,
             f=f["none"], W=W, Wspread=max(W.values()) / min(W.values()) - 1.0,
             S=a["sm2"] - a["sm1"], Y=a["av2"] - a["av1"],
             S_path=a["sm1"] - a["none"], Y_path=a["av1"] - a["none"],
             fork=a["fork2"] - a["none"],
             S_inc=(a["sm2"] - a["sm1"], a["sm3"] - a["sm2"]),
             Y_inc=(a["av2"] - a["av1"], a["av3"] - a["av2"]))
    c["P"] = R - c["S"] - c["Y"]
    for nm, one, three in (("S", "sm1", "sm3"), ("Y", "av1", "av3")):
        pred = a[one] + 2.0 * c[nm]
        c[nm + "_pred"], c[nm + "_meas"] = pred, a[three]
        c[nm + "_err"] = (a[three] - pred) / pred
    return c


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "sweep5.json")
    runs = json.load(open(path, encoding="utf-8"))

    P = parity()
    if P:
        # s12: G2a is bit-identity on the eight arms that HAVE it (--sweep8); G2b is a
        # structural test on the ten-arm sweep, which mixes serial/serial2/avx4 -- arms that
        # differ in summation order BY CONSTRUCTION (that difference is what X measures) and
        # therefore cannot be bit-identical to each other or to a mixture of themselves.
        print("## G2 -- parity of the interleave (brief s12.2)\n")
        print("| run | NATS_TOTAL (as printed) | test | verdict |")
        print("|---|---|---|---|")
        AVX4, SERIAL = 166667.1361128952, 166667.2449386003
        base = "166667.1361128952"        # results/e5/parity4.txt, arm `none`
        span = (SERIAL - AVX4) / math.log(2) / SCORED_BYTES     # E4's number, 3.03e-06 BPB
        g2 = True
        for k, v in P.items():
            if "sweep8" in k:
                ok = (v == base)
                print("| `%s` | %s | G2a bit-identity | %s |"
                      % (k, v, "**BIT-IDENTICAL**" if ok else "**FAIL -- run 5 is void**"))
            else:
                x = float(v)
                d = abs(x - AVX4) / math.log(2) / SCORED_BYTES
                ok = (AVX4 <= x <= SERIAL) and d <= span
                print("| `%s` | %s | G2b between the pure arms, within %.2e BPB | %s (%.2e) |"
                      % (k, v, span, "PASS" if ok else "**FAIL -- run 5 is void**", d))
            g2 = g2 and ok
        print()
        if not g2:
            print("**G2 failed; run 5 is VOID and nothing below is a result.**\n")

    # ---------------- G0, within-process
    print("## G0 -- the weight path across the arms of ONE process (brief s11.5)\n")
    print("| cell | runs | kept | worst `W` spread across arms | verdict |")
    print("|---|---|---|---|---|")
    cells = {}
    for r in runs:
        cells.setdefault((r["shape"], r["bench"]), []).append(solve(r))
    good = {}
    for k in sorted(cells, key=lambda x: (x[0], x[1])):
        v = cells[k]
        keep = [c for c in v if c["Wspread"] <= G0_TOL]
        ok = len(keep) >= MIN_RUNS
        if ok:
            good[k] = keep
        print("| %s @%d | %d | %d | %+.2f%% | %s |"
              % (k[0], k[1], len(v), len(keep), 100.0 * max(c["Wspread"] for c in v),
                 "PASS" if ok else "**VOID -- fewer than %d clean runs**" % MIN_RUNS))
    print()
    if not good:
        print("no cell survives G0; run 5 is VOID.")
        return

    order = [k for k in (("T10", 800), ("T10", 300), ("S05", 800), ("S05", 300)) if k in good]

    def band(k, f):
        v = [f(c) for c in good[k]]
        return med(v), min(v), max(v)

    # ---------------- the reproducibility interval, reported before anything is concluded
    print("## The spread ACROSS the %d-run set -- this is the reproducibility interval (s11.5)\n"
          % MIN_RUNS)
    print("| cell | runs kept | `R` | `S` | `Y` | `P` | organ `none` |")
    print("|---|---|---|---|---|---|---|")
    for k in order:
        cs = " | ".join("%.3f [%.3f-%.3f]" % band(k, lambda c, n=n: c[n])
                        for n in ("R", "S", "Y", "P", "organ"))
        print("| %s @%d | %d | %s |" % (k[0], k[1], len(good[k]), cs))
    print("\nEach cell is `median [min-max]` over the runs that passed G0.  A component whose band "
          "straddles zero, or whose band is wider than the component, decides nothing.\n")

    # ---------------- G1
    print("## G1 -- the 3x arms, which S and Y are not fitted to (brief s4)\n")
    print("| point | 1x arm | 2x arm | solved | predicted 3x | measured 3x | error (median) | "
          "verdict |")
    print("|---|---|---|---|---|---|---|---|")
    fails = []
    for k in order:
        for nm, lbl in (("S", "sm"), ("Y", "av")):
            e = med([c[nm + "_err"] for c in good[k]])
            v0 = med([c[nm] for c in good[k]])
            small = (v0 < MIN_DELTA_MS and k == ("T10", 800))
            ok = abs(e) <= TOL
            verdict = ("**INCONCLUSIVE** (2x moved %.3f < %.2f ms)" % (v0, MIN_DELTA_MS) if small
                       else ("PASS" if ok else "**FAIL**"))
            if not ok and not small:
                fails.append((k, nm))
            print("| %s @%d `%s` | %.3f | %.3f | %s = %.3f | %.3f | **%.3f** | %+.2f%% | %s |"
                  % (k[0], k[1], lbl + "1/2/3",
                     med([c["organ"] + c[nm + "_path"] for c in good[k]]),
                     med([c["organ"] + c[nm + "_path"] + c[nm] for c in good[k]]),
                     nm, v0, med([c[nm + "_pred"] for c in good[k]]),
                     med([c[nm + "_meas"] for c in good[k]]), 100.0 * e, verdict))
    print()
    if fails:
        print("**%d prediction(s) missed by more than %.0f%% -> those components are not "
              "established and E5 is VOID for them.**\n" % (len(fails), 100 * TOL))

    # ---------------- the two increments, which must be equal
    print("## Both increments are one extra pass; if they disagree the arms are wrong (s10.3)\n")
    print("| cell | `S`: 2x-1x / 3x-2x | ratio | `Y`: 2x-1x / 3x-2x | ratio | "
          "`sm1`-`none` | `av1`-`none` |")
    print("|---|---|---|---|---|---|---|")
    for k in order:
        row = []
        for nm in ("S", "Y"):
            i1 = med([c[nm + "_inc"][0] for c in good[k]])
            i2 = med([c[nm + "_inc"][1] for c in good[k]])
            row += ["%.3f / %.3f" % (i1, i2), "%.2fx" % (i2 / i1) if i1 else "-"]
        print("| %s @%d | %s | %s | %s | %s | %+.3f | %+.3f |"
              % (k[0], k[1], row[0], row[1], row[2], row[3],
                 med([c["S_path"] for c in good[k]]), med([c["Y_path"] for c in good[k]])))
    print("\n`1x - none` is the price of the code SHAPE, not of a pass: both compute the same "
          "values.  It is reported, never absorbed into a component.\n")

    # ---------------- the decomposition
    print("## The decomposition of `R`\n")
    print("| cell | organ `none` | `X` | `S` | `Y` | `P` | S/R | Y/R | P/R | `f` | ceiling 1000/f |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        g = lambda n: med([c[n] for c in good[k]])
        R = g("R")
        print("| %s @%d | %.3f | %.3f | **%.3f** | **%.3f** | **%.3f** | %.1f%% | %.1f%% | %.1f%% "
              "| %.3f | **%.1f** |"
              % (k[0], k[1], g("organ"), g("X_avx4"), g("S"), g("Y"), g("P"),
                 100 * g("S") / R, 100 * g("Y") / R, 100 * g("P") / R, g("f"), 1000.0 / g("f")))
    print()

    # ---------------- fork2
    print("## `fork2` -- one extra OpenMP region per layer, priced directly\n")
    print("| cell | `fork2` - `none` | as a share of `P` |")
    print("|---|---|---|")
    for k in order:
        fk = med([c["fork"] for c in good[k]])
        pp = med([c["P"] for c in good[k]])
        print("| %s @%d | %+.3f | %.1f%% |" % (k[0], k[1], fk, 100.0 * fk / pp if pp else 0.0))
    print()

    T = ("T10", 800)
    if T not in good:
        print("`T10` @800 did not survive G0; brief s9.4 -- the cells above stand and no label "
              "is issued.")
        return

    # ---------------- s3: my predictions, scored, never re-fitted
    print("## s3 -- the predictions I registered before any arm existed, at `T10` @800\n")
    print("| quantity | predicted (ms) | measured | verdict |")
    print("|---|---|---|---|")
    for kk, (lo, hi) in PRE.items():
        v = med([c[kk] for c in good[T]])
        inside = lo <= v <= hi
        print("| `%s` | %.1f - %.1f | **%.3f** | %s |"
              % (kk, lo, hi, v, "inside" if inside else
                 "**OUTSIDE -- my estimate was wrong, by %+.0f%%**"
                 % (100.0 * (v - (hi if v > hi else lo)) / (hi if v > hi else lo))))
    print()

    # ---------------- s5: the label
    if any(k == T for k, _ in fails):
        print("## s5 -- the label\n\nWithheld: G1 failed at `T10` @800, so the component it "
              "failed on is not established and neither is `P`, which is a residual of it.")
        return
    print("## s5 -- the label, at T10 @800\n")
    R = med([c["R"] for c in good[T]])
    parts = [("SOFTMAX-DOMINATED", "S", med([c["S"] for c in good[T]])),
             ("AV-DOMINATED", "Y", med([c["Y"] for c in good[T]])),
             ("OVERHEAD-DOMINATED", "P", med([c["P"] for c in good[T]]))]
    parts.sort(key=lambda x: -x[2])
    top = parts[0]
    share = top[2] / R
    near = [p for p in parts if abs(p[2] / R - 0.50) <= 0.03]
    boundary = abs(share - 0.50) <= 0.03 or len(near) >= 2
    lab = "WITHHELD" if boundary else (top[0] if share >= 0.50 else "SPLIT")
    Sv, Yv, Pv = (med([c[n] for c in good[T]]) for n in ("S", "Y", "P"))
    print("`R` = %.3f ms = `S` %.3f + `Y` %.3f + `P` %.3f;  largest is `%s` at %.1f%% -> **%s**"
          % (R, Sv, Yv, Pv, top[1], 100.0 * share, lab))
    if boundary:
        print("(the leading share is within 3 points of the 0.50 boundary -- brief s4 G4: the "
              "label is withheld)")
    f = med([c["f"] for c in good[T]])
    print("\n| if this term went to zero | `f` ms | **ceiling 1000/f** |")
    print("|---|---|---|")
    print("| nothing (measured today) | %.3f | **%.1f** |" % (f, 1000.0 / f))
    for _, nm, v in parts:
        print("| `%s` | %.3f | **%.1f** |" % (nm, f - v, 1000.0 / (f - v)))
    print("| all of `R` | %.3f | **%.1f** |" % (f - R, 1000.0 / (f - R)))
    print("\nEvery absolute tok/s on this machine carries +-5%% (E4 s2.5); the ratios do not.")


if __name__ == "__main__":
    main()
