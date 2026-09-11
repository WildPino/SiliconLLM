#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H0 self-test (CPU, free) -- the T4 bundle must be the same object as the repo.

h0_qat.py reimplements the shipped ternary rule so the Kaggle bundle is self-contained.  A
reimplementation is exactly the kind of thing that is 99% right and silently changes a result,
and this programme has already paid for one (E20 run 1).  So it is not trusted; it is CHECKED,
against the real masters rather than against random tensors, because a rule can agree everywhere
except at the ties that matter.

  G-H0b  h0_qat.r3_actsearch == t2_rules.r3_actsearch on every shipped organ, EXACTLY:
         identical codes (integers) and identical scales.
  G-H0c  the assembled TernaryLowRank module, at s = 1 and with no gradient, reproduces
         e22_compose.ternary_factors(..., balanced=True) -- i.e. the trainer's step-0 forward IS
         E22's QO512-TB, not merely built from the same numbers.
  G-H0d  the straight-through gradient is the identity it claims to be: d(ste(w))/dw == 1.

Run this before shipping anything.  Any failure means the bundle may not be uploaded.

Env: D_THREADS (6), H0_BUNDLE (defaults to the smoke bundle)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
TERNDIR = os.path.abspath(os.path.join(HERE, "..", "ternary"))
for _p in (DENSDIR, TERNDIR):
    sys.path.insert(0, _p)

import t2_rules as T2                                       # noqa: E402
torch.set_grad_enabled(True)   # common.py disables it at import; see h0_qat.py docstring
import h0_qat as H0                                         # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))
BUNDLE = os.environ.get("H0_BUNDLE",
                        os.path.join(HERE, "results", "h0", "h0_factors_smoke.npz"))


def log(m):
    print(m, flush=True)


def main():
    z = np.load(BUNDLE)
    organs = sorted({k.rsplit(".", 1)[0] for k in z.files})
    log("== H0 selftest on %s ==" % os.path.basename(BUNDLE))
    log("   %d organs" % len(organs))

    # ---- G-H0b: the reimplemented rule, on the real masters
    bad = []
    for p in organs:
        for fac, rms in (("A", "rms_A"), ("B", "rms_in")):
            W = torch.from_numpy(z[p + "." + fac])
            r = torch.from_numpy(z[p + "." + rms])
            q0, a0 = T2.r3_actsearch(W, r)
            q1, a1 = H0.r3_actsearch(W, r)
            if not (torch.equal(q0, q1) and torch.equal(a0, a1)):
                bad.append("%s.%s codes=%s scales=%s"
                           % (p, fac, torch.equal(q0, q1), torch.equal(a0, a1)))
    log("  G-H0b  reimplemented rule == t2_rules, %d/%d factors%s"
        % (2 * len(organs) - len(bad), 2 * len(organs),
           "" if not bad else "   FAILS: " + "; ".join(bad[:4])))

    # ---- G-H0c: the module's step-0 forward is E22's QO512-TB
    worst_c, worst_k = 0.0, None
    for p in organs:
        A = torch.from_numpy(z[p + ".A"])
        s = torch.from_numpy(z[p + ".s"])
        B = torch.from_numpy(z[p + ".B"])
        ri = torch.from_numpy(z[p + ".rms_in"])
        ra = torch.from_numpy(z[p + ".rms_A"])
        mod = H0.TernaryLowRank(A, s, B, ri, ra, None)
        n_in = B.shape[1]
        x = torch.eye(n_in)                     # the whole operator, not a sample of it
        with torch.no_grad():
            Weff = mod(x).T                     # [n_out, n_in]
        qB, aB = T2.r3_actsearch(B, ri)
        qA, aA = T2.r3_actsearch(A, ra)
        ref = (qA * aA).mm(qB * aB)
        d = float((Weff - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
        if d > worst_c:
            worst_c, worst_k = d, p
    log("  G-H0c  module forward == E22's ternary product, worst relative %.3e at %s  (tol 1e-6)"
        % (worst_c, worst_k))

    # ---- G-H0d: the straight-through gradient is the identity
    w = torch.randn(8, 32, requires_grad=True)
    rms = torch.rand(32) + 0.5
    H0.ste(w, rms).sum().backward()
    gmax = float((w.grad - 1.0).abs().max())
    log("  G-H0d  d(ste(w))/dw == 1, max deviation %.3e" % gmax)

    ok = (not bad) and worst_c <= 1e-6 and gmax == 0.0
    log("")
    log("  %s" % ("ALL SELFTESTS FIRE -- the bundle is the same object as the repo"
                  if ok else "*** SELFTEST FAILED -- DO NOT UPLOAD THE BUNDLE ***"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
