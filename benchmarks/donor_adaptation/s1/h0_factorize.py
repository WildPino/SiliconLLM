#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H0 step 1 (CPU, free) -- build the factored ternary attention the T4 will train.

Plan: docs/research/donor_adaptation/decisions/T4_HEALING_PROPOSAL.md s3 (H0) and s5b.

WHAT THIS EMITS.  For all 28 layers x {q_proj, o_proj} of Qwen2.5-1.5B, the rank-512
activation-weighted factorisation E21 measured, re-parameterised as

        W  ~=  ternarize(A) . diag(s) . ternarize(B)         A [1536,512]  B [512,1536]

with A and B E22's BALANCED factors (A / c and c * B, c = ||A column||), `s` a per-rank learned
scale initialised to ONES, and the two frozen calibration statistics the quantizer needs
(rms_in for B, rms_A for A).  These are the fp32 MASTERS a straight-through trainer updates;
nothing here trains anything.

WHERE s COMES FROM, AND A CLAIM OF MINE THAT THIS FILE REFUTED.  The T4 proposal s5b(2) argued
for pulling A's column norms OUT into an fp32 s, on the grounds that E22's fold puts a large
dynamic range into B.x -- the one intermediate fp16 must hold.  That is measured here and it is
FALSE, in the opposite direction: on L0.q_proj the row-magnitude spread of B.x is 36.7 unscaled
and 22.4 folded, and folded is smaller in 4/4 organs.  The reason is that c = ||A column|| and
||B row|| are near-perfectly reciprocal (corr(log c, log||B row||) = -0.997 .. -0.984), because
A = W H^1/2 Br carries the singular values and B = Br' H^-1/2 carries their inverse.  Folding c
into B UNDOES that, which is also why E22's balanced arm worked and its registered one did not.
Measured on the same four organs, folded is better conditioned on every axis proposed for the
unfolded form: master row-norm spread 15.6 vs 51.7, 2.9 vs 28.0, 4.6 vs 22.4, 4.7 vs 62.8.

So s5b(2) is WITHDRAWN, not softened, and this file keeps E22's fold verbatim.  `s` survives as
something else and something honest: a per-rank LEARNED SCALE initialised to ONES -- the standard
learned-step-size device in QAT, a continuous degree of freedom the ternary codes cannot express.
At s = 1 the initialisation is E22's QO512-TB identically, by construction rather than by
argument.  It is 28,672 floats = 0.03% of the trainable mass, so it cannot be doing the work on
its own, and H0 can measure whether it does anything at all.

The fp16 finding itself -- s5b(1), eager non-finite at layers.0.self_attn.o_proj on a T4, sdpa
finite -- is measured on that card and is untouched by any of this.  Only my inference FROM it
was wrong.

THE PLANTED CONTROL.  The trainer must start from E22's measured QO512-TB (tf 28/160, BPB
2.812226) and not from something merely close to it.  Because s starts at ones and the masters
are E22's own balanced factors, that is now an identity rather than an argument, and G-H0a
asserts it at zero tolerance per organ: identical ternary CODES (integers), identical scales,
and bit-identical assembled products against e22_compose.ternary_factors(balanced=True).
Failing it aborts the file -- a trainer starting anywhere else makes every H0 number
uninterpretable.

Env: D_THREADS (6), H0_LAYERS (comma list, debug), H0_SMOKE (1 = 2 layers, 4 cal seqs)
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
TERNDIR = os.path.abspath(os.path.join(HERE, "..", "ternary"))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
for _p in (DENSDIR, TERNDIR, ENGDIR):
    sys.path.insert(0, _p)

import common as C                                          # noqa: E402
import t2_rules as T2                                       # noqa: E402
import t2b_organs as T2B                                    # noqa: E402
import e22_compose as E22C                                  # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("H0_SMOKE", "0") == "1"

RANK = 512                                  # E21's QO-ACT-512 / E22's QO512
NCAL, SEQCAL, SEEDCAL = (4 if SMOKE else 32), 512, 42424     # E22's calibration slice, unchanged
ORGANS = ("q_proj", "o_proj")
OUTDIR = os.path.join(HERE, "results", "h0")
BUNDLE = os.path.join(OUTDIR, "h0_factors%s.npz" % ("_smoke" if SMOKE else ""))
META = os.path.join(OUTDIR, "h0_factors%s.json" % ("_smoke" if SMOKE else ""))

# E22's measured QO512-TB, the state this initialisation must reproduce (2f6d27d).
E22_TB = {"bpb": 2.812226, "free": 1, "tf": 28}


def log(m):
    print(m, flush=True)


def reparam(W, H, r, rms_in, n_cal):
    """E22's balanced factorisation, plus s = ones and the frozen quantizer statistics.

    Returns (A0, s, B0, rmsA, Ahat, diag, codes) where Ahat = ternarize(A0).diag(s).ternarize(B0):

        A0   = A / c        fp32 master for the output factor   [n_out, r]
        B0   = c * B        fp32 master for the input factor    [r, n_in]   (E22's FOLD)
        s    = ones(r)      per-rank learned scale, trainable, starts at the identity
        rmsA                per-rank input RMS A sees, in closed form from the same H

    c = A.norm(dim=0) is E22's balancing constant and it stays folded into B: measured, that is
    the better-conditioned half on every axis (see the module docstring).  With s = 1 this is
    e22_compose.ternary_factors(..., balanced=True) executed line for line, which is what makes
    G-H0a an identity.
    """
    A, B, d = E22C.factors(W, H, r)
    c = A.norm(dim=0).clamp_min(1e-30)
    A0 = (A / c).float()
    B0 = (c.unsqueeze(1) * B).float()
    s = torch.ones(r, dtype=torch.float32)

    qB, aB = T2.r3_actsearch(B0, rms_in)
    Bt0 = qB * aB

    # A sees s*(Bt0 x); at s = 1 this is exactly E22's rms_A.
    Bd = (s.unsqueeze(1) * Bt0).double()
    rmsA = (torch.diagonal(Bd.mm(H.double()).mm(Bd.T))
            / float(n_cal)).clamp_min(1e-16).sqrt().float()

    qA, aA = T2.r3_actsearch(A0, rmsA)
    At = qA * aA
    Ahat = (At * s.unsqueeze(0)).mm(Bt0)
    d.update({"zero_frac_A": float((qA == 0).float().mean()),
              "zero_frac_B": float((qB == 0).float().mean()),
              "c_spread": float(c.max() / c.min().clamp_min(1e-30))})
    return A0, s, B0, rmsA, Ahat, d, (qA, aA, qB, aB), c.float()


def e22_tb_path(W, H, r, rms_in, n_cal):
    """E22's balanced path, written out so its CODES can be compared, not just its product.

    Line-for-line e22_compose.ternary_factors(..., balanced=True); it returns only the assembled
    matrix, and G-H0a needs the integers underneath it.
    """
    A, B, _ = E22C.factors(W, H, r)
    c = A.norm(dim=0).clamp_min(1e-30)
    Af, Bf = (A / c).float(), (c.unsqueeze(1) * B).float()
    qB, aB = T2.r3_actsearch(Bf, rms_in)
    Bt = qB * aB
    Bd = Bt.double()
    rmsA = (torch.diagonal(Bd.mm(H.double()).mm(Bd.T))
            / float(n_cal)).clamp_min(1e-16).sqrt().float()
    qA, aA = T2.r3_actsearch(Af, rmsA)
    return (qA, aA, qB, aB), (qA * aA).mm(Bt)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t_all = time.time()
    log("== H0 factorize: rank %d, organs %s, %s =="
        % (RANK, ORGANS, "SMOKE" if SMOKE else "full"))

    model, tok = C.load_model()   # eager: E22 derived its H under eager, so exactness needs it
    a = C.arch(model)
    n_layers = a["n_layers"]
    layer_ids = list(range(n_layers))
    if SMOKE:
        layer_ids = [0, n_layers - 1]
    if os.environ.get("H0_LAYERS"):
        layer_ids = [int(x) for x in os.environ["H0_LAYERS"].split(",")]
    log("   D=%d  L=%d  layers=%s" % (a["d_model"], n_layers, layer_ids))

    ids_cal, _, meta_cal = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    log("   calib slice %dx%d seed %d, ids_sha %s"
        % (NCAL, SEQCAL, SEEDCAL, meta_cal["ids_sha256"][:16]))

    log("== calibration pass 1/2: act_rms (t2b_organs.capture) ==")
    t0 = time.time()
    act_rms = T2B.capture(model, ids_cal, n_layers)
    log("   done in %.0fs, %d organs" % (time.time() - t0, len(act_rms)))

    log("== calibration pass 2/2: H = XtX for %d organs ==" % (len(ORGANS) * len(layer_ids)))
    Hs, cnt, hooks = {}, {}, []

    def mk(key):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            Hs[key] = (x.T @ x) if key not in Hs else Hs[key] + (x.T @ x)
            cnt[key] = cnt.get(key, 0) + x.shape[0]
        return f
    for li in layer_ids:
        lay = model.model.layers[li]
        for nm in ORGANS:
            hooks.append(getattr(lay.self_attn, nm).register_forward_hook(mk((li, nm))))
    t0 = time.time()
    with torch.no_grad():
        for i in range(ids_cal.shape[0]):
            model(ids_cal[i:i + 1])
    for h in hooks:
        h.remove()
    n_cal_h = cnt[(layer_ids[0], ORGANS[0])]
    log("   done in %.0fs, %d H matrices, %d tokens each"
        % (time.time() - t0, len(Hs), n_cal_h))

    store, rows = {}, []
    bad_codes = []
    worst_scale, worst_prod, worst_key = 0.0, 0.0, None
    PROD_TOL = 0.0           # an identity, not an approximation: s starts at ones
    t0 = time.time()
    for li in layer_ids:
        for nm in ORGANS:
            key = (li, nm)
            W = C.get_linear(model, li, nm).weight.data
            H = Hs[key]
            A0, s, B0, rmsA, Ahat, d, mine, cvec = reparam(W, H, RANK, act_rms[key], n_cal_h)
            ref, Wtb = e22_tb_path(W, H, RANK, act_rms[key], n_cal_h)

            # ---- G-H0a, three parts, against E22's own path.
            qA, aA, qB, aB = mine
            rqA, raA, rqB, raB = ref
            codes_ok = bool(torch.equal(qA, rqA) and torch.equal(qB, rqB))
            # (ii) aA to round-off; s*aB is the per-ROW identity the whole form rests on
            den_a = raA.abs().clamp_min(1e-30)
            den_b = raB.abs().clamp_min(1e-30)
            sc = max(float(((aA - raA).abs() / den_a).max()),
                     float(((aB - raB).abs() / den_b).max()))
            # (iii) products, relative to the product's own scale
            pr = float((Ahat - Wtb).abs().max() / Wtb.abs().max().clamp_min(1e-30))
            if not codes_ok:
                bad_codes.append("L%d.%s" % (li, nm))
            if sc > worst_scale or pr > worst_prod:
                worst_key = "L%d.%s" % (li, nm)
            worst_scale, worst_prod = max(worst_scale, sc), max(worst_prod, pr)

            relerr = float((Ahat - W).norm() / W.norm())
            # s5b(2), measured rather than asserted, and it went the OTHER WAY.  Compare the
            # fold we keep (B0 = c*B) against the unfolded form the proposal argued for (B0/c),
            # on the two axes that proposal invoked: the row scale of the fp16 intermediate B.x,
            # and the row-norm spread one learning rate has to cover.
            Bunf = B0 / cvec.unsqueeze(1)
            fol, uns = (B0.abs() @ act_rms[key]), (Bunf.abs() @ act_rms[key])
            rng_fol = float(fol.max() / fol.min().clamp_min(1e-30))
            rng_uns = float(uns.max() / uns.min().clamp_min(1e-30))
            nf, nu = B0.norm(dim=1), Bunf.norm(dim=1)
            cond_fol = float(nf.max() / nf.min().clamp_min(1e-30))
            cond_uns = float(nu.max() / nu.min().clamp_min(1e-30))

            p = "L%02d.%s" % (li, nm)
            store[p + ".A"] = A0.numpy()
            store[p + ".s"] = s.numpy()
            store[p + ".B"] = B0.numpy()
            store[p + ".rms_in"] = act_rms[key].numpy()
            store[p + ".rms_A"] = rmsA.numpy()
            rows.append({"layer": li, "organ": nm, "codes_identical": codes_ok,
                         "scale_relmax": sc, "prod_relmax": pr, "rel_err": relerr,
                         "zero_A": d["zero_frac_A"],
                         "zero_B": d["zero_frac_B"],
                         "c_spread": d["c_spread"],
                         "range_Bx_folded_KEPT": rng_fol, "range_Bx_unfolded": rng_uns,
                         "rownorm_spread_folded_KEPT": cond_fol,
                         "rownorm_spread_unfolded": cond_uns})
            log("  %-14s codes %s  scale %.1e  prod %.1e  relerr %.4f  c spread %.1e  "
                "zeroA %.3f zeroB %.3f | B.x range fold %.1f vs unfold %.1f | rownorm "
                "fold %.1f vs unfold %.1f"
                % (p, "SAME" if codes_ok else "DIFF", sc, pr, relerr, d["c_spread"],
                   d["zero_frac_A"], d["zero_frac_B"], rng_fol, rng_uns, cond_fol, cond_uns))
    log("   %d organs in %.0fs" % (len(rows), time.time() - t0))

    gh0a = (not bad_codes) and worst_scale <= PROD_TOL and worst_prod <= PROD_TOL
    n_par = int(sum(v.size for k, v in store.items() if k.endswith((".A", ".B", ".s"))))
    meta = {
        "plan": "decisions/T4_HEALING_PROPOSAL.md s3 (H0), s5b",
        "what": "fp32 masters for ternarize(A).diag(s).ternarize(B) on q_proj/o_proj, rank 512",
        "model": C.MODEL_ID, "revision": C.REVISION, "smoke": SMOKE, "threads": THREADS,
        "rank": RANK, "organs": list(ORGANS), "layers": layer_ids,
        "calib_slice": meta_cal, "calib_tokens": n_cal_h,
        "e22_tb_anchor": E22_TB,
        "G_H0a": {"claim": "the diag(s) form IS E22's QO512-TB: (i) identical ternary codes, "
                           "zero tolerance; (ii) scales and the per-row identity s*aB == aB_e22 "
                           "to fp32 round-off; (iii) assembled products to fp32 round-off",
                  "note": "(iii) cannot be exact -- (A_t*s).B_t0 and A_t.(s*B_t0) are the same "
                          "product in a different summation order; (i) is the structural claim",
                  "prod_tol": PROD_TOL, "organs_with_different_codes": bad_codes,
                  "worst_scale_rel": worst_scale, "worst_prod_rel": worst_prod,
                  "worst_organ": worst_key, "fires": gh0a},
        "trainable_params": n_par,
        "s5b2_withdrawn": {
            "claim_in_proposal": "E22's fold puts a large dynamic range into B.x, the fp16 "
                                 "intermediate, so pull A's column norms out into an fp32 s",
            "measured": "FALSE, and in the opposite direction -- the fold is better on BOTH "
                        "axes in every organ; see rows[].range_Bx_* and rows[].rownorm_*",
            "why": "c = ||A column|| and ||B row|| are near-reciprocal (corr(log,log) ~ -0.99) "
                   "because A = W H^1/2 Br carries the singular values and B = Br' H^-1/2 "
                   "their inverse, so folding c into B BALANCES the pair rather than unbalancing it",
            "kept_instead": "E22's fold verbatim; s survives as a per-rank learned scale "
                            "initialised to ones (0.03% of trainable mass)"},
        "rows": rows,
        "seconds": time.time() - t_all,
    }
    np.savez(BUNDLE, **store)
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    mb = os.path.getsize(BUNDLE) / 1e6
    log("")
    log("  G-H0a (the diag(s) form IS E22's QO512-TB): %s"
        % ("FIRES" if gh0a else "*** FAILS ***"))
    log("    (i)   ternary codes identical, %d/%d organs%s"
        % (len(rows) - len(bad_codes), len(rows),
           "" if not bad_codes else "  DIFFER: " + ",".join(bad_codes)))
    log("    (ii)  scales, worst relative %.3e  (tol %.0e)" % (worst_scale, PROD_TOL))
    log("    (iii) products, worst relative %.3e  (tol %.0e)  worst organ %s"
        % (worst_prod, PROD_TOL, worst_key))
    log("  trainable fp32 masters: %d params" % n_par)
    log("  bundle %s  (%.1f MB)" % (BUNDLE, mb))
    log("  meta   %s" % META)
    log("  total %.0fs" % (time.time() - t_all))
    if not gh0a:
        log("")
        log("  ABORT: the diag(s) form is NOT E22's QO512-TB, so a trainer starting here would")
        log("  NOT start at the measured tf 28/160.  Nothing may be shipped to the T4.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
