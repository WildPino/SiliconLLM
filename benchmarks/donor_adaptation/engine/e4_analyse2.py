#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4 run 2 -- results/e4/arms2.json against the gates of BRIEF s8.3.

`e4_analyse.py` is left exactly as it was when it produced run 1's VOID; this is a separate file so
that the analysis that labelled run 1 cannot be edited after the fact.

Run 2 differs from run 1 in one structural way: the baseline is `serial_e3`, which restores E3's
per-position KV address arithmetic.  Run 1's `serial` hoisted it, which is worth 4-7% and was
credited to nothing -- brief s8.2.  Both comparators are printed:

    vs serial_e3   what the change is actually worth against the engine E3 measured
    vs serial      the accumulators alone, with the address hoisting held constant

G1' is the reason run 2 exists.  X (the dot loop) and R (softmax + the A.V loop) are solved from the
1x and 2x organ times, so they fit those two points by construction and prove nothing.  `serial3` is
a third point they cannot be fitted to: it must land on 3X + R.
"""
import io
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "e4")
FIXED = ("rope", "attention", "norm+glue")
SCORED_BYTES = 51870
SIGMA_SEED = 0.005
SHAPES = {"S05": (896, 4864, 24, 14, 2, 64, 151936),
          "T10": (4096, 14336, 48, 32, 8, 128, 32768)}
# E3 s3: (tok/s, IQR tok/s, attention ms).  G4' is judged against these.
E3 = {("T10", 300): (3.090, 0.005, 9.862), ("T10", 800): (2.960, 0.000, 25.385),
      ("S05", 300): (55.700, 0.790, 1.096), ("S05", 800): (49.280, 1.510, 2.847)}
# brief s8.3, pre-registered before run 2 existed: attention ms for serial3 at T10.
G1P_PREDICTED = {300: 20.083, 800: 53.482}
G1P_TOL = 0.03
ORDER = ["serial_e3", "serial", "serial2", "serial3", "ilp4", "avx1", "avx4"]


def parity():
    p = os.path.join(RES, "parity.txt")
    if not os.path.exists(p):
        return None
    txt = io.open(p, encoding="utf-8").read()
    out = {}
    pat = (r"== parity (\w+) ==\s*NATS_TOTAL ([0-9.eE+-]+)\s*"
           r"N_PREDICTED (\d+)\s*NATS_PER_TOKEN ([0-9.eE+-]+)")
    for m in re.finditer(pat, txt):
        out[m.group(1)] = float(m.group(2))
    return out


def bpb(nats):
    return nats / math.log(2.0) / SCORED_BYTES


def load(path):
    R = json.load(open(path, encoding="utf-8"))
    out = []
    for r in R:
        shape = r["label"].split("_")[0]
        D, F, L, NH, NKV, HD, V = SHAPES[shape]
        pos = (r["bench"] + 1) / 2.0
        o = r["median_rep_organs_ms"]
        k_touch = L * NH * HD * pos * 4.0
        out.append(dict(shape=shape, arm=r.get("attn", "serial"), bench=r["bench"],
                        tok_s=r["median_tok_s"], iqr=r["iqr_tok_s"], reps=r["tok_s"],
                        organs=o, f=sum(o[k] for k in FIXED), attn=o["attention"],
                        wall=r["median_rep_wall_ms_token"],
                        touched=2.0 * k_touch,
                        unique=2.0 * L * NKV * HD * pos * 4.0))
    for r in out:
        r["ceiling"] = 1000.0 / r["f"]
        r["gbs_touched"] = r["touched"] / (r["attn"] * 1e-3) / 1e9
        r["gbs_unique"] = r["unique"] / (r["attn"] * 1e-3) / 1e9
    return out


def pick(A, shape, bench, arm):
    return next((r for r in A if r["shape"] == shape and r["bench"] == bench
                 and r["arm"] == arm), None)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "arms2.json")
    A = load(path)

    P = parity()
    if P:
        print("## G2 -- parity, real donor, pinned 24x512 slice (%d scored bytes)\n" % SCORED_BYTES)
        print("| arm | BPB | dBPB vs serial | vs sigma_seed = 0.005 |")
        print("|---|---|---|---|")
        base = P["serial"]
        for a in ("serial", "serial2", "ilp4", "avx1", "avx4"):
            if a not in P:
                continue
            d = bpb(P[a]) - bpb(base)
            tag = ("**BIT-IDENTICAL**" if P[a] == base else
                   ("PASS, %.0fx under" % (SIGMA_SEED / abs(d)) if abs(d) < SIGMA_SEED
                    else "**FAIL**"))
            print("| `%s` | %.9f | %+.3e | %s |" % (a, bpb(P[a]), d, tag))
        print()

    # ---------------- G4': the baseline must be E3's baseline again
    print("## G4' -- `serial_e3` must reproduce E3 (brief s8.2)\n")
    print("| point | E3 tok/s (IQR) | `serial_e3` | E3 attention | `serial_e3` | verdict |")
    print("|---|---|---|---|---|---|")
    for (shape, b), (t3, iqr3, a3) in sorted(E3.items()):
        r = pick(A, shape, b, "serial_e3")
        if not r:
            continue
        dt = abs(r["tok_s"] - t3)
        band = max(iqr3, 0.02 * t3)      # E3's own dispersion, floored at 2% (IQR is 0.000 at T10)
        print("| %s @%d | %.3f (%.3f) | %.3f | %.3f | %.3f | %s |"
              % (shape, b, t3, iqr3, r["tok_s"], a3, r["attn"],
                 "PASS (%+.1f%%)" % (100.0 * (r["tok_s"] - t3) / t3) if dt <= band
                 else "**FAIL (%+.1f%%)**" % (100.0 * (r["tok_s"] - t3) / t3)))
    print()

    # ---------------- the price of the address arithmetic alone
    print("## The per-position KV address arithmetic, priced on its own\n")
    print("(`serial_e3` - `serial`: identical FP arithmetic, bit-identical logits, "
          "the only difference is recomputing the address at every position)\n")
    print("| point | `serial_e3` attn | `serial` attn | saved | as %% of the organ | tok/s effect |")
    print("|---|---|---|---|---|---|")
    for shape in ("T10", "S05"):
        for b in (300, 800):
            e3, sr = pick(A, shape, b, "serial_e3"), pick(A, shape, b, "serial")
            if not (e3 and sr):
                continue
            print("| %s @%d | %.3f | %.3f | **%.3f ms** | %.1f%% | %.3f -> %.3f (%+.1f%%) |"
                  % (shape, b, e3["attn"], sr["attn"], e3["attn"] - sr["attn"],
                     100.0 * (e3["attn"] - sr["attn"]) / e3["attn"],
                     e3["tok_s"], sr["tok_s"], 100.0 * (sr["tok_s"] - e3["tok_s"]) / e3["tok_s"]))
    print()

    # ---------------- G1': the planted control as a falsifiable model
    print("## G1' -- the planted control, as a model that can miss (brief s8.3)\n")
    print("| point | 1x `serial` | 2x `serial2` | solved X | solved R | R share | "
          "predicted 3X+R | brief s8.3 | **measured `serial3`** | error |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    verdicts = []
    for shape in ("T10", "S05"):
        for b in (300, 800):
            s1, s2, s3 = (pick(A, shape, b, a) for a in ("serial", "serial2", "serial3"))
            if not (s1 and s2 and s3):
                continue
            X = s2["attn"] - s1["attn"]
            Rr = s1["attn"] - X
            pred = 3 * X + Rr
            err = (s3["attn"] - pred) / pred
            pre = G1P_PREDICTED.get(b) if shape == "T10" else None
            ok = abs(err) <= G1P_TOL
            if shape == "T10":
                verdicts.append((b, ok, err, pre, s3["attn"]))
            print("| %s @%d | %.3f | %.3f | %.3f | %.3f | %.1f%% | %.3f | %s | **%.3f** | %+.2f%% |"
                  % (shape, b, s1["attn"], s2["attn"], X, Rr, 100.0 * Rr / s1["attn"], pred,
                     ("%.3f" % pre) if pre else "-", s3["attn"], 100.0 * err))
    print()
    for b, ok, err, pre, meas in verdicts:
        line = ("T10 @%d: within-run prediction missed by %+.2f%% -> %s"
                % (b, 100.0 * err, "**PASS**" if ok else "**FAIL -> E4 stays VOID**"))
        if pre:
            line += ("  |  against the brief's pre-registered %.3f: %.3f measured, %+.2f%% -> %s"
                     % (pre, meas, 100.0 * (meas - pre) / pre,
                        "PASS" if abs(meas - pre) / pre <= G1P_TOL else "**FAIL**"))
        print(line)
    print()

    # ---------------- the arms
    for shape in ("T10", "S05"):
        for b in (300, 800):
            sel = [r for r in A if r["shape"] == shape and r["bench"] == b]
            if not sel:
                continue
            sel.sort(key=lambda r: ORDER.index(r["arm"]) if r["arm"] in ORDER else 99)
            e3b, srb = pick(A, shape, b, "serial_e3"), pick(A, shape, b, "serial")
            print("### %s, --bench %d\n" % (shape, b))
            print("| arm | tok/s | IQR | **attention ms** | vs `serial_e3` | vs `serial` | f ms | "
                  "**1000/f** | GB/s touched | GB/s unique |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for r in sel:
                print("| `%s` | %.3f | %.3f | **%.3f** | %.3fx | %.3fx | %.3f | **%.1f** | "
                      "%.1f | %.1f |"
                      % (r["arm"], r["tok_s"], r["iqr"], r["attn"],
                         r["attn"] / e3b["attn"] if e3b else float("nan"),
                         r["attn"] / srb["attn"] if srb else float("nan"),
                         r["f"], r["ceiling"], r["gbs_touched"], r["gbs_unique"]))
            print()

    # ---------------- s5 label
    print("## s5 -- the label, judged on the attention organ at T10 @800\n")
    base = pick(A, "T10", 800, "serial_e3")
    cand = [r for r in A if r["shape"] == "T10" and r["bench"] == 800
            and r["arm"] in ("ilp4", "avx1", "avx4")]
    if base and cand:
        best = min(cand, key=lambda r: r["attn"])
        ratio = best["attn"] / base["attn"]
        lab = ("LATENCY-CONFIRMED" if ratio <= 0.50 else
               ("BANDWIDTH-BOUND" if ratio >= 0.90 else "MIXED"))
        print("best arm `%s`: %.3f ms vs `serial_e3` %.3f = **%.3fx**  ->  **%s**"
              % (best["arm"], best["attn"], base["attn"], ratio, lab))
        s1, s2 = pick(A, "T10", 800, "serial"), pick(A, "T10", 800, "serial2")
        if s1 and s2:
            X = s2["attn"] - s1["attn"]
            Rr = s1["attn"] - X
            # what the arms did to the DOT LOOP alone, holding R fixed at its measured value
            print("dot loop alone (organ - R, R = %.3f ms measured): %.3f -> %.3f ms = **%.2fx**"
                  % (Rr, s1["attn"] - Rr, max(1e-9, best["attn"] - Rr),
                     (s1["attn"] - Rr) / max(1e-9, best["attn"] - Rr)))
            print("the floor is now `R` itself: %.1f%% of the best arm's organ is softmax + A.V"
                  % (100.0 * Rr / best["attn"]))
        print("f %.3f -> %.3f ms;  **ceiling 1000/f %.1f -> %.1f tok/s**;  "
              "tok/s %.3f -> %.3f (%+.1f%%)"
              % (base["f"], best["f"], base["ceiling"], best["ceiling"],
                 base["tok_s"], best["tok_s"],
                 100.0 * (best["tok_s"] - base["tok_s"]) / base["tok_s"]))
        print("attention organ load rate: %.1f GB/s touched, %.1f GB/s unique "
              "(brief s3 predicted the floor would be memory near 6-8 ms)"
              % (best["gbs_touched"], best["gbs_unique"]))


if __name__ == "__main__":
    main()
