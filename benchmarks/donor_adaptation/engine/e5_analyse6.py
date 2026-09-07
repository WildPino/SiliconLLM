#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 run 6 -- results/e5/{parity6.txt,sweep6.json} against the gates of BRIEF_E5_DECOMPOSE_R.md s13.

Every arm is on the `avx4` path, every difference is between arms of ONE code family measured
inside ONE process, interleaved per token.  Run 5's X crossed families and came out negative.

    X  = organ(qk2) - organ(qk1)     the Q.K dot loop        3x test: qk1 + 2X == qk3
    S  = organ(sm2) - organ(sm1)     the softmax pass        3x test: sm1 + 2S == sm3
    Y  = organ(av2) - organ(av1)     the A.V loop            3x test: av1 + 2Y == av3
    R  = organ(none) - X             what E4 could not see inside
    P  = R - S - Y                   everything that is neither loop

The d-probe (--sweepd) runs the IDENTICAL `none` code in three neighbourhoods at equal mean
context length, so d = organ(none_iso) - organ(none_hot) prices arm-switching itself.
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
G0_TOL = 0.01         # s11.5, unchanged: W must agree across the arms of one process
MIN_RUNS = 3          # s11.5: fewer than three surviving runs voids the cell
TOL = 0.03            # brief s4: the 3x prediction's tolerance, E4's own
MIN_DELTA_MS = 0.30   # brief s4: a 2x arm that moves less than this at T10 @800 is INCONCLUSIVE
SCORED_BYTES = 51870
PRE = {"S": (0.8, 2.0), "Y": (2.2, 3.5), "P": (4.0, 7.0)}   # s3, never re-fitted
ARMS = ("none", "sm1", "sm2", "sm3", "av1", "av2", "av3", "qk1", "qk2", "qk3")
DARMS = ("none_iso", "none_hot", "none_edge", "sm2_filler")
BASE_NATS = "166667.1361128952"      # results/e5/parity4.txt, pure `none`


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def parity():
    p = os.path.join(RES, "parity6.txt")
    if not os.path.exists(p):
        return None
    txt = io.open(p, encoding="utf-8").read()
    return dict((m.group(1).strip(), m.group(2))
                for m in re.finditer(r"== parity ([^=]+?) ==\s*NATS_TOTAL ([0-9.eE+-]+)", txt))


def solve(run):
    a = {k: run["arms"][k]["organs"]["attention"] for k in ARMS}
    W = {k: sum(run["arms"][k]["organs"][o] for o in WEIGHT_ORGANS) for k in ARMS}
    c = dict(shape=run["shape"], bench=run["bench"], run=run["run"],
             cores_busy=run.get("cores_busy"), tok_s=run.get("tok_s"),
             organ=a["none"], f=sum(run["arms"]["none"]["organs"][o] for o in FIXED),
             Wspread=max(W.values()) / min(W.values()) - 1.0)
    for nm, one, two, three in (("X", "qk1", "qk2", "qk3"), ("S", "sm1", "sm2", "sm3"),
                                ("Y", "av1", "av2", "av3")):
        c[nm] = a[two] - a[one]
        c[nm + "_base"] = a[one]
        c[nm + "_path"] = a[one] - a["none"]          # the price of the code SHAPE, never absorbed
        c[nm + "_inc"] = (a[two] - a[one], a[three] - a[two])
        pred = a[one] + 2.0 * c[nm]
        c[nm + "_pred"], c[nm + "_meas"] = pred, a[three]
        c[nm + "_err"] = (a[three] - pred) / pred
    c["R"] = a["none"] - c["X"]
    c["P"] = c["R"] - c["S"] - c["Y"]
    return c


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "sweep6.json")
    recs = json.load(open(path, encoding="utf-8"))

    # ---------------- G2
    P = parity()
    g2 = True
    if P:
        print("## G2 -- parity (brief s12.2; every run-6 arm is on `avx4`, so all of it is G2a)\n")
        print("| run | NATS_TOTAL (as printed) | verdict |")
        print("|---|---|---|")
        for k, v in P.items():
            ok = (v == BASE_NATS)
            g2 = g2 and ok
            print("| `%s` | %s | %s |"
                  % (k, v, "**BIT-IDENTICAL**" if ok else "**FAIL -- run 6 is void**"))
        print()
        if not g2:
            print("**G2 failed; run 6 is VOID and nothing below is a result.**\n")

    # ---------------- the d-probe, read BEFORE the components it might qualify
    dprobe = {}
    for r in recs:
        if r.get("mode") != "sweepd":
            continue
        a = {k: r["arms"][k]["organs"]["attention"] for k in DARMS}
        dprobe.setdefault((r["shape"], r["bench"]), []).append(
            dict(d=a["none_iso"] - a["none_hot"], edge=a["none_edge"] - a["none_hot"],
                 hot=a["none_hot"], iso=a["none_iso"]))
    if dprobe:
        print("## `d` -- the price of switching arms, measured (brief s13.4)\n")
        print("(`none_iso`, `none_hot` and `none_edge` run the IDENTICAL `none` code at the same "
              "mean context length; only their neighbours differ.)\n")
        print("| cell | runs | `none_hot` | `none_iso` | **`d` = iso - hot** | `edge` - `hot` | "
              "`d` as %% of the organ |")
        print("|---|---|---|---|---|---|---|")
        for k in sorted(dprobe):
            v = dprobe[k]
            d, hot = med([x["d"] for x in v]), med([x["hot"] for x in v])
            print("| %s @%d | %d | %.3f | %.3f | **%+.3f** | %+.3f | %+.1f%% |"
                  % (k[0], k[1], len(v), hot, med([x["iso"] for x in v]), d,
                     med([x["edge"] for x in v]), 100.0 * d / hot if hot else 0.0))
        print()

    # ---------------- G0, within-process
    cells = {}
    for r in recs:
        if r.get("mode") != "sweep6":
            continue
        cells.setdefault((r["shape"], r["bench"]), []).append(solve(r))
    if not cells:
        print("sweep6.json has no --sweep6 records")
        return
    print("## G0 -- the weight path across the arms of ONE process (brief s11.5)\n")
    print("| cell | runs | kept | worst `W` spread across arms | verdict |")
    print("|---|---|---|---|---|")
    good = {}
    for k in sorted(cells):
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
        print("no cell survives G0; run 6 is VOID.")
        return
    order = [k for k in (("T10", 800), ("T10", 300), ("S05", 800), ("S05", 300)) if k in good]

    def band(k, n):
        v = [c[n] for c in good[k]]
        return med(v), min(v), max(v)

    # ---------------- the reproducibility interval
    print("## The spread ACROSS the runs -- this is the reproducibility interval (s11.5)\n")
    print("| cell | runs kept | `X` | `S` | `Y` | `R` | `P` | organ `none` |")
    print("|---|---|---|---|---|---|---|---|")
    for k in order:
        print("| %s @%d | %d | %s |"
              % (k[0], k[1], len(good[k]),
                 " | ".join("%.3f [%.3f-%.3f]" % band(k, n)
                            for n in ("X", "S", "Y", "R", "P", "organ"))))
    print("\nEach cell is `median [min-max]` over the runs that passed G0.\n")

    # ---------------- G5, new in s13.4
    print("## G5 -- a component that is a duration must be positive (brief s13.4)\n")
    print("| cell | `X` | `S` | `Y` | `R` | `P` | verdict |")
    print("|---|---|---|---|---|---|---|")
    neg = set()
    for k in order:
        vals = [(n, med([c[n] for c in good[k]])) for n in ("X", "S", "Y", "R", "P")]
        bad = [n for n, v in vals if v <= 0.0]
        for n in bad:
            neg.add((k, n))
        print("| %s @%d | %s | %s |"
              % (k[0], k[1], " | ".join("%.3f" % v for _, v in vals),
                 "PASS" if not bad else "**FAIL -- %s not a duration**" % ", ".join(bad)))
    print()

    # ---------------- G1
    print("## G1 -- the 3x arms, which X, S and Y are not fitted to (brief s4)\n")
    print("| point | 1x arm | 2x arm | solved | predicted 3x | measured 3x | error (median) | "
          "verdict |")
    print("|---|---|---|---|---|---|---|---|")
    fails = []
    for k in order:
        for nm, lbl in (("X", "qk"), ("S", "sm"), ("Y", "av")):
            e = med([c[nm + "_err"] for c in good[k]])
            v0 = med([c[nm] for c in good[k]])
            small = (v0 < MIN_DELTA_MS and k == ("T10", 800))
            ok = abs(e) <= TOL
            if not ok and not small:
                fails.append((k, nm))
            print("| %s @%d `%s` | %.3f | %.3f | %s = %.3f | %.3f | **%.3f** | %+.2f%% | %s |"
                  % (k[0], k[1], lbl + "1/2/3", med([c[nm + "_base"] for c in good[k]]),
                     med([c[nm + "_base"] + c[nm] for c in good[k]]), nm, v0,
                     med([c[nm + "_pred"] for c in good[k]]),
                     med([c[nm + "_meas"] for c in good[k]]), 100.0 * e,
                     ("**INCONCLUSIVE** (2x moved %.3f < %.2f ms)" % (v0, MIN_DELTA_MS) if small
                      else ("PASS" if ok else "**FAIL**"))))
    print()
    if fails:
        print("**%d prediction(s) missed by more than %.0f%% -> those components are not "
              "established, and so is anything derived from them.**\n" % (len(fails), 100 * TOL))

    # ---------------- both increments, and the code-shape price
    print("## Both increments are one extra pass; if they disagree the arms are wrong (s10.3)\n")
    print("| cell | `X` inc 2-1 / 3-2 | `S` inc | `Y` inc | `qk1`-`none` | `sm1`-`none` | "
          "`av1`-`none` |")
    print("|---|---|---|---|---|---|---|")
    for k in order:
        inc = []
        for nm in ("X", "S", "Y"):
            i1 = med([c[nm + "_inc"][0] for c in good[k]])
            i2 = med([c[nm + "_inc"][1] for c in good[k]])
            inc.append("%.3f / %.3f (%.2fx)" % (i1, i2, i2 / i1 if i1 else 0.0))
        print("| %s @%d | %s | %+.3f | %+.3f | %+.3f |"
              % (k[0], k[1], " | ".join(inc),
                 med([c["X_path"] for c in good[k]]), med([c["S_path"] for c in good[k]]),
                 med([c["Y_path"] for c in good[k]])))
    print()

    # ---------------- the decomposition
    print("## The decomposition of `R`\n")
    print("| cell | organ `none` | `X` | `S` | `Y` | `P` | S/R | Y/R | P/R | `f` | ceiling 1000/f |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        g = lambda n: med([c[n] for c in good[k]])
        R = g("R")
        print("| %s @%d | %.3f | %.3f | **%.3f** | **%.3f** | **%.3f** | %.1f%% | %.1f%% | %.1f%% "
              "| %.3f | **%.1f** |"
              % (k[0], k[1], g("organ"), g("X"), g("S"), g("Y"), g("P"),
                 100 * g("S") / R, 100 * g("Y") / R, 100 * g("P") / R, g("f"), 1000.0 / g("f")))
    print()

    T = ("T10", 800)
    dead = [n for kk, n in fails if kk == T] + [n for kk, n in neg if kk == T]
    if T not in good:
        print("`T10` @800 did not survive G0; brief s9.4 -- the cells above stand, no label.")
        return
    if dead:
        print("## s5 -- the label\n\nWithheld: %s did not survive its gates at `T10` @800, and "
              "`P` is a residual of all three." % ", ".join("`%s`" % n for n in sorted(set(dead))))
        return

    print("## s3 -- the predictions I registered before any arm existed, at `T10` @800\n")
    print("| quantity | predicted (ms) | measured | verdict |")
    print("|---|---|---|---|")
    for kk, (lo, hi) in PRE.items():
        v = med([c[kk] for c in good[T]])
        print("| `%s` | %.1f - %.1f | **%.3f** | %s |"
              % (kk, lo, hi, v, "inside" if lo <= v <= hi else
                 "**OUTSIDE -- my estimate was wrong, by %+.0f%%**"
                 % (100.0 * (v - (hi if v > hi else lo)) / (hi if v > hi else lo))))
    print()

    print("## s5 -- the label, at T10 @800\n")
    R = med([c["R"] for c in good[T]])
    Sv, Yv, Pv = (med([c[n] for c in good[T]]) for n in ("S", "Y", "P"))
    parts = sorted([("SOFTMAX-DOMINATED", "S", Sv), ("AV-DOMINATED", "Y", Yv),
                    ("OVERHEAD-DOMINATED", "P", Pv)], key=lambda x: -x[2])
    share = parts[0][2] / R
    near = [p for p in parts if abs(p[2] / R - 0.50) <= 0.03]
    boundary = abs(share - 0.50) <= 0.03 or len(near) >= 2
    lab = "WITHHELD" if boundary else (parts[0][0] if share >= 0.50 else "SPLIT")
    print("`R` = %.3f ms = `S` %.3f + `Y` %.3f + `P` %.3f;  largest is `%s` at %.1f%% -> **%s**"
          % (R, Sv, Yv, Pv, parts[0][1], 100.0 * share, lab))
    if boundary:
        print("(within 3 points of the 0.50 boundary -- brief s4 G4: the label is withheld)")
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
