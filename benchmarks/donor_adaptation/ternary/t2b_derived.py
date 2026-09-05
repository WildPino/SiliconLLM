#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2b derived quantities -- the numbers in probes/T2B_ORGAN_COVERAGE.md sections 4, 5 and 6
that are NOT in t2b_organs.json, recomputed from the per-sequence arrays on disk.

Why a separate script: t2b_organs.py wrote the arms, the deltas and the three pre-registered
increments. Reporting exposed three further quantities, all of which are read off the SAME 24
per-sequence BPB values and none of which may be eyeballed:

  1. the ratio Delta(FAH)/Delta(F) with its OWN interval. The label turned on 1.584 vs a 1.60
     bar -- a 1.0% margin -- and a point estimate without an interval cannot say whether that
     margin is resolved. (It is not: the bar is inside the CI.)
  2. the additivity defect, FAH minus the sum of the three single-organ arms.
  3. attention measured alone vs attention on top of a ternary FFN.

Same bootstrap as the runner: 2000 resamples of the 24 sequences, seed 7, byte-weighted, paired
(every arm resampled on the SAME index draw, because the arms are correlated across sequences).

Usage: python t2b_derived.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
import common as C                                                        # noqa: E402

ARMDIR = os.path.abspath(os.path.join(HERE, "..", "density", "results", "t2b_arms"))
ARMS = ["base", "I", "F", "A", "H", "FA", "FAH"]
N_BOOT, SEED = 2000, 7

# weight counts from the config on disk, the same ones BRIEF s4 derived its bar from
D, F_, L, V = 1536, 8960, 28, 151936
NH, NKV, HD = 12, 2, 128
W_ATTN = L * (D * NH * HD + D * NKV * HD * 2 + NH * HD * D)
W_FFN = L * 3 * D * F_
W_HEAD = V * D


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL_ID, revision=C.REVISION)
    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    assert meta["ids_sha256"].startswith("a1a48dc9"), meta["ids_sha256"]
    w = byts.numpy().astype(np.float64)
    assert int(w.sum()) == 51870, w.sum()
    print("slice ids_sha256 %s  bytes %d" % (meta["ids_sha256"][:16], int(w.sum())))

    P = {a: np.load(os.path.join(ARMDIR, a + ".npy")) for a in ARMS}
    for a in ARMS:
        assert P[a].shape == (24,), (a, P[a].shape)

    rng = np.random.default_rng(SEED)
    IDX = [rng.integers(0, 24, 24) for _ in range(N_BOOT)]

    def d(a, b, i=slice(None)):
        """byte-weighted paired delta of arm a vs arm b over sequence index set i"""
        return float(((P[a][i] - P[b][i]) * w[i]).sum() / w[i].sum())

    def report(name, point, boot):
        boot = np.asarray(boot)
        print("  %-34s %+.6f  se %.6f  ci95 [%+.4f, %+.4f]"
              % (name, point, boot.std(ddof=1),
                 np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
        return boot

    print("\n== s5  the ratio the label turned on ==")
    ratio = d("FAH", "base") / d("F", "base")
    rb = np.array([d("FAH", "base", i) / d("F", "base", i) for i in IDX])
    print("  Delta(FAH)/Delta(F)                %.6f  se %.6f  ci95 [%.4f, %.4f]"
          % (ratio, rb.std(ddof=1), np.percentile(rb, 2.5), np.percentile(rb, 97.5)))
    print("  bar 1.60 -> UNIFORM below, DIFFUSE above.  P(resample > 1.60) = %.3f"
          % (rb > 1.60).mean())
    print("  %s" % ("BAR IS INSIDE THE INTERVAL -- near-miss, not a comfortable pass"
                    if np.percentile(rb, 2.5) < 1.60 < np.percentile(rb, 97.5)
                    else "bar is outside the interval"))

    print("\n== s4  additivity ==")
    singles = d("F", "base") + d("A", "base") + d("H", "base")
    print("  sum of single-organ arms           %+.6f" % singles)
    print("  measured FAH                       %+.6f" % d("FAH", "base"))
    report("additivity defect",
           d("FAH", "base") - singles,
           [d("FAH", "base", i) - (d("F", "base", i) + d("A", "base", i) + d("H", "base", i))
            for i in IDX])
    print("  ratio an ADDITIVE model would give %.6f  (would have been DIFFUSE)"
          % (singles / d("F", "base")))

    print("\n== s4  attention alone vs attention on top of a ternary FFN ==")
    print("  A - base                           %+.6f" % d("A", "base"))
    print("  FA - F                             %+.6f" % d("FA", "F"))
    report("(FA-F) - (A-base)",
           d("FA", "F") - d("A", "base"),
           [d("FA", "F", i) - d("A", "base", i) for i in IDX])

    print("\n== s6  damage per weight ==")
    print("  %-10s %12s %8s %10s %12s %7s" % ("organ", "weights", "share", "Delta",
                                              "BPB/100M", "vs FFN"))
    tot = W_ATTN + W_FFN + W_HEAD
    base_rate = d("F", "base") / (W_FFN / 1e8)
    for name, nw, arm in (("FFN", W_FFN, "F"), ("attention", W_ATTN, "A"), ("head", W_HEAD, "H")):
        rate = d(arm, "base") / (nw / 1e8)
        print("  %-10s %11.1fM %7.1f%% %+10.6f %12.4f %6.2fx"
              % (name, nw / 1e6, 100.0 * nw / tot, d(arm, "base"), rate, rate / base_rate))


if __name__ == "__main__":
    main()
