#!/usr/bin/env python3
"""Is G-E55a2 answerable?  A planted control ON THE GATE, not on the engine.

E57 run 1 refused all six arms as UNSTABLE on interquartile widths of 0.56% to 2.29% --
tighter than the 1.43% and 2.34% windows E55 PASSED.  A gate that refuses data four times
tighter than data it previously passed is measuring something other than the data.

This feeds the gate clean Gaussian samples at KNOWN dispersions and known rep counts and
reports P(PASS).  E55 A.2 ran exactly this test on G-E55a, found 38-60%, and registered
G-E55a2 as the repair.  The repair was never subjected to its own test.  This is that test.

Brief: BRIEF_E57 addendum A.5.
"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e57_trained import g_e55a2, iqr_pct                            # noqa: E402

DRAWS = 300
KS = (8, 15, 25, 40, 80)
SIGMAS = (0.001, 0.005, 0.01, 0.03)


def main():
    rng = random.Random(4242)
    print("P(G-E55a2 = PASS) over %d independent draws from a clean Gaussian population." % DRAWS)
    print("If the gate watches DISPERSION the rows must differ.  If it watches REP COUNT")
    print("the rows are identical and the columns are flat.")
    print("")
    print("  true sigma  " + "".join("%7s" % ("k=%d" % k) for k in KS))
    for sigma in SIGMAS:
        row = []
        for k in KS:
            hits = 0
            for _ in range(DRAWS):
                v = [100.0 * (1.0 + rng.gauss(0, sigma)) for _ in range(k)]
                if g_e55a2(v, rng, 60)[0] == "PASS":
                    hits += 1
            row.append(100.0 * hits / DRAWS)
        print("  %8.1f%%    " % (100 * sigma) + "".join("%6.0f%%" % x for x in row))
    print("")
    print("  A 30x range of true dispersion, and the k=15 column this experiment registered")
    print("  sits at 22-28% no matter which row you read.  |IQR(half) - IQR(full)| / IQR(full)")
    print("  is the relative sampling error of the IQR estimator, and that is SCALE-FREE.")
    print("")
    base = [100.0 * (1.0 + rng.gauss(0, 0.002)) for _ in range(160)]
    print("  one population, IQR %.2f%% at n=160, read at increasing k:" % iqr_pct(base))
    for k in KS + (160,):
        st, rel, _ = g_e55a2(base[:k], rng)
        print("    k=%-4d IQR %6.3f%%   median |half-full|/full %3.0f%%   %s"
              % (k, iqr_pct(base[:k]), 100 * rel, st))


if __name__ == "__main__":
    main()
