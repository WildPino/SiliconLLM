# -*- coding: utf-8 -*-
"""E49 addendum C -- the resolvability check, committed so the table can be re-derived."""
CELLS = {
 40:   {"serial":[98.77,121.38,114.79,112.14,112.19], "avx4":[117.26,121.65,126.79,122.73,119.87]},
 160:  {"serial":[96.70,104.60,107.79,85.91,109.36],  "avx4":[119.29,112.89,114.14,100.10,111.76]},
 640:  {"serial":[69.96,66.92,69.71,69.07,69.92],     "avx4":[92.44,88.97,89.89,89.57,92.20]},
 1280: {"serial":[49.11,48.68,47.35,48.59,47.88],     "avx4":[70.71,74.62,74.92,71.75,74.11]},
 2560: {"serial":[29.73,30.17,30.27,30.57,30.49],     "avx4":[54.35,53.13,53.79,54.25,53.42]},
}
FIT = {"serial": (8.2419, 0.019334), "avx4": (8.2468, 0.008144)}

def separable(lo, hi):
    """Two five-rep sets are separable only if their ranges do not overlap.  A gate read on
    medians across overlapping ranges is reading noise -- feedback_gate_vs_measured_dispersion."""
    return max(lo) < min(hi) or max(hi) < min(lo)

def report():
    print("%-6s %-28s %-28s %8s  %s" % ("n","serial [min..max] med","avx4 [min..max] med",
                                        "d-med","separable?"))
    for n in sorted(CELLS):
        s, a = sorted(CELLS[n]["serial"]), sorted(CELLS[n]["avx4"])
        ms, ma = s[2], a[2]
        print("%-6d [%6.2f..%6.2f] %6.2f     [%6.2f..%6.2f] %6.2f     %+6.1f%%  %s"
              % (n, s[0], s[-1], ms, a[0], a[-1], ma, 100*(ma-ms)/ms,
                 "yes" if separable(s, a) else "NO -- ranges overlap"))
    print()
    print("the fit's own estimate of the kernel effect (all five windows, dominated by the")
    print("three whose dispersion is 2.3-5.9%):")
    (as_, bs), (aa, ba) = FIT["serial"], FIT["avx4"]
    for n in sorted(CELLS):
        pos = n / 2.0
        d = (bs - ba) * pos
        print("  n=%-5d pos %6.1f   delta %6.3f ms on %6.3f ms = %+6.1f%%"
              % (n, pos, d, aa + ba * pos, 100.0 * d / (aa + ba * pos)))

if __name__ == "__main__":
    report()
    s, a = CELLS[40]["serial"], CELLS[40]["avx4"]
    assert not separable(s, a), "if this ever passes, addendum C's premise is wrong"
    print()
    print("G-E49c : VOID -- the gate could not answer the question it asked.")
    print("  at NTOK=40 serial's best rep (%.2f) is avx4's median (%.2f)."
          % (max(s), sorted(a)[2]))
