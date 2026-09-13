#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 step 0b (CPU, FREE) -- the GO/NO-GO: can a trained router beat STATIC on the REAL donor?

Brief addendum A, consequences 2 and 3, verbatim:
  2. the router lr and the auxiliary coefficient are fixed by a CPU smoke on the REAL donor;
  3. **if no setting makes the revised gate fire on CPU, H1 does not launch** and the T4 hours
     are handed back unspent.

This file is that smoke.  It is the last thing between the brief and the bundle.

WHY A LOCAL PROXY, STATED UP FRONT SO IT IS NOT DISCOVERED LATER.  `G-H1e` as registered is an
END-TO-END comparison: held-out LM loss with the router against held-out LM loss with STATIC.
Running that on CPU costs a forward+backward through 1.5 B fp32 per step -- roughly a minute a
step, so one setting is an hour and a grid is a day.  Instead this trains each layer's router
on a LOCAL objective: reproduce that layer's own UNCARVED FFN output.

  * It is the SAME comparison (router vs STATIC), on the SAME donor, with the SAME rule, the
    SAME E=256 partition and the SAME k -- on real activations, not a toy.
  * It is NOT `G-H1e`.  A local win need not survive composition through 8 layers and an LM
    head; `G-H1e` is still scored end-to-end at the end of the T4 run by h1_qat.py.
  * Its job is only to answer addendum A's question -- does ANY setting move the router on
    this donor, and which -- so the T4 hours are not spent finding out.

WHAT STATIC IS HERE.  Exactly what addendum B registered: the top-k groups chosen ONCE by
global activation mass over the CALIBRATION half, then used for every token of the HELD-OUT
half.  Both halves are real donor activations captured from the frozen slices; the corpus
halves are disjoint by construction and common.get_slice asserts it.

NO ANCHORING.  The per-layer verdicts are printed as measured, every layer, including the ones
that lose.  The summary counts wins; it does not select the layers that agree.

Env: D_THREADS (6)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(HERE, "..", "density")),
           os.path.abspath(os.path.join(HERE, "..", "engine"))):
    sys.path.insert(0, _p)

import common as C                                          # noqa: E402
import h1_qat as H1                                         # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))
OUTDIR = os.path.join(HERE, "results", "h1")
LRS = (3e-4, 1e-3, 3e-3, 1e-2)
AUXS = (0.0, 0.01, 0.1)
STEPS = 300
BATCH = 256                       # tokens per step, drawn from the captured calibration pool


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


@torch.no_grad()
def capture_inputs(model, layers, ids, cap):
    """The FFN input x for each registered layer, over `ids`.  One forward pass, hooks on the
    carved modules themselves so what is captured is exactly what they will be routed on."""
    buf = {li: [] for li in layers}
    hooks = []

    def mk(li):
        def f(mod, inp, out):
            buf[li].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float().clone())
        return f
    for li in layers:
        hooks.append(model.model.layers[li].mlp.register_forward_hook(mk(li)))
    for i in range(ids.shape[0]):
        model(ids[i:i + 1])
    for h in hooks:
        h.remove()
    out = {}
    for li in layers:
        x = torch.cat(buf[li], 0)
        out[li] = x[:cap] if x.shape[0] > cap else x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--stats", required=True,
                    help="h1_actstats.npz -- REQUIRED: without it this is not E37's format")
    ap.add_argument("--layers", default=",".join(str(x) for x in H1.H1_LAYERS))
    ap.add_argument("--k", type=int, default=H1.K_DEFAULT)
    ap.add_argument("--groups", type=int, default=H1.E_GROUPS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cap", type=int, default=4096, help="tokens kept per layer, per half")
    ap.add_argument("--only-layer", type=int, default=None,
                    help="smoke a single layer first (cheap); default is all 8")
    ap.add_argument("--lrs", default=None,
                    help="comma-separated router lrs; default is the full LRS grid")
    ap.add_argument("--auxs", default=None,
                    help="comma-separated aux coefficients; default is the full AUXS grid")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lrs = tuple(float(x) for x in a.lrs.split(",")) if a.lrs else LRS
    auxs = tuple(float(x) for x in a.auxs.split(",")) if a.auxs else AUXS
    os.makedirs(OUTDIR, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",") if x.strip() != ""]
    if a.only_layer is not None:
        layers = [a.only_layer]
    t0 = time.time()

    log("== H1 router smoke (GO/NO-GO) ==  CPU fp32, layers %s, k %d of E %d"
        % (layers, a.k, a.groups))
    # THE TRAP H0 PAID FOR, and this file walks straight into it if left alone: importing
    # common.py runs torch.set_grad_enabled(False) at module scope, so every backward() below
    # would be a no-op and the smoke would report NO-GO -- i.e. it would hand back the T4
    # hours because of an import side effect.  h1_qat.py avoids it by importing nothing from
    # the repo; this file NEEDS common.get_slice, so it re-enables and asserts instead.
    torch.set_grad_enabled(True)
    if not torch.is_grad_enabled():
        raise SystemExit("AUTOGRAD IS DISABLED -- STOP.  This smoke would train nothing and "
                         "report a NO-GO that is an import side effect, not a measurement.")
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    H1.build_qo(model, a.factors, "cpu")
    H1.build_ffn(model, a.labels, a.stats, layers, a.k, a.groups, "cpu")
    mods = dict(H1.ffn_mods(model, layers))
    # Only the router is trained here.  Left on, the experts would accumulate a [8960,1536]
    # gradient per organ per step -- 3 of them per layer, for nothing.
    for p in model.parameters():
        p.requires_grad_(False)
    for _li, _m in mods.items():
        _m.router.requires_grad_(True)

    ids_cal, _, mc = C.get_slice(tok, "calib", 32, 512, 42424)
    ids_ev, _, me = C.get_slice(tok, "heldout", 24, 512, 1234)
    if mc["corpus_sha256"] == me["corpus_sha256"]:
        raise SystemExit("calib and heldout are the same corpus half -- STOP")
    log("   capturing activations (calib %d seqs, heldout %d seqs, cap %d tokens/layer)..."
        % (ids_cal.shape[0], ids_ev.shape[0], a.cap))
    xc = capture_inputs(model, layers, ids_cal[:8], a.cap)
    xe = capture_inputs(model, layers, ids_ev[:8], a.cap)
    log("   captured  %s" % {li: tuple(xc[li].shape) for li in layers})

    # PLANTED CONTROL FOR THIS FILE.  Before any setting is scored, one step must move the
    # router on the real donor.  Without this, "no setting beat STATIC" is indistinguishable
    # from "nothing was ever trained", and that reading would hand back the GPU hours.
    _li0 = layers[0]
    _m0, _x0 = mods[_li0], xc[layers[0]][:BATCH]
    _m0.router.data.zero_()
    _snap = _m0.router.detach().clone()
    _o = torch.optim.AdamW([_m0.router], lr=1e-3, weight_decay=0.0)
    with torch.no_grad():
        _m0.k = _m0.E
        _t0 = _m0(_x0)
        _m0.k = a.k
    _o.zero_grad(set_to_none=True)
    (_m0(_x0) - _t0).square().mean().backward()
    _gr = None if _m0.router.grad is None else float(_m0.router.grad.abs().sum())
    _o.step()
    _mv = float((_m0.router.detach() - _snap).abs().max())
    log("   planted control: router grad %s, moved %.3e  -> %s"
        % ("None" if _gr is None else "%.3e" % _gr, _mv,
           "FIRES" if (_gr and _gr > 0 and _mv > 0) else "*** FAILS ***"))
    if not (_gr and _gr > 0 and _mv > 0):
        raise SystemExit("The router does not move on the real donor.  A NO-GO from this run "
                         "would be a WIRING failure, not addendum A's answer.  STOP.")
    _m0.router.data.zero_()

    # ---- FREEZE-CACHE THE QUANTISED EXPERTS, and prove it changes nothing -----------------
    # The experts are frozen here (only the router trains), so ste() is a CONSTANT -- yet
    # _quant recomputed r3_actsearch every step: a 10-point grid over three [8960,1536]
    # tensors, ~3.3G element-ops per forward.  That was 92% of the cost of this file.
    # Caching is exactly equivalent, and "exactly" is asserted rather than argued: the cached
    # tensors must be torch.equal to what _quant returns.  30x measured on this shape, which
    # is why this run can afford the FULL grid instead of one narrowed on distorted numbers.
    n_cached = 0
    for li in layers:
        m = mods[li]
        ref = m._quant()
        with torch.no_grad():
            cq = (H1.ste(m.gate, m.rms_in), H1.ste(m.up, m.rms_in), H1.ste(m.down, m.rms_h))
        if not all(torch.equal(c, r) for c, r in zip(cq, ref)):
            raise SystemExit("layer %d: the quantisation cache is NOT bit-identical to "
                             "_quant().  Refusing to trade correctness for speed.  STOP." % li)
        m._quant = (lambda t: (lambda: t))(cq)
        n_cached += 1
    log("   quantisation cached on %d layers, bit-identical to _quant() on every one" % n_cached)

    results, rows = {}, []
    routers_out = {}
    for li in layers:
        m = mods[li]
        xtr, xev = xc[li], xe[li]
        with torch.no_grad():
            k0 = m.k
            m.k = m.E
            ttr, tev = m(xtr), m(xev)
            m.k = k0
            denom = float(tev.square().mean())

        # STATIC, per addendum B: top-k once by global mass over the CALIBRATION half.
        m.arm_mass(True)
        with torch.no_grad():
            m(xtr)
        pick = m.static_from_mass()
        m.arm_mass(False)
        m.set_static(pick)
        with torch.no_grad():
            static = float((m(xev) - tev).square().mean())
        m.set_static(None)

        # the untrained router, for reference: at init it is zeros, so top-k is a tie broken by
        # index -- i.e. groups 0..k-1.  Not a control, just the starting point.
        with torch.no_grad():
            m.router.data.zero_()
            init = float((m(xev) - tev).square().mean())

        log("")
        log("   layer %-3d  uncarved power %.5f   STATIC %.6f (%.4f of power)   init %.6f"
            % (li, denom, static, static / denom, init))
        best = None
        for lr in lrs:
            for aux in auxs:
                torch.manual_seed(1717)
                m.router.data.zero_()
                opt = torch.optim.AdamW([m.router], lr=lr, weight_decay=0.0)
                g = torch.Generator().manual_seed(4242)
                for _ in range(a.steps):
                    sel = torch.randint(0, xtr.shape[0], (BATCH,), generator=g)
                    opt.zero_grad(set_to_none=True)
                    out = m(xtr[sel])
                    loss = (out - ttr[sel]).square().mean()
                    if aux > 0 and m._aux is not None:
                        loss = loss + aux * m._aux
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    # ADDENDUM G: evaluate the trained router with the HARD gate, because
                    # STATIC (set_static) has gates of exactly 1.  The first version of this
                    # file scored a SOFT-gated router against a HARD-gated control, which on
                    # the real donor is worth +0.80 BPB of pure gate form (addendum F) and is
                    # WORSE for peaked routers -- i.e. the distortion interacts with `aux`,
                    # the very parameter being chosen.  Training stays soft: a hard gate has
                    # no gradient to the router at all.
                    occ = float(m._occ.max()) if m._occ is not None else float("nan")
                    m.hard_gate = True
                    v = float((m(xev) - tev).square().mean())
                    v_soft = float("nan")
                    m.hard_gate = False
                    v_soft = float((m(xev) - tev).square().mean())
                win = v < static
                if best is None or v < best[0]:
                    best = (v, lr, aux)
                log("      lr %-7g aux %-5g -> %.6f  ratio %.4f  occ_max %.3f  "
                    "(soft %.6f)%s"
                    % (lr, aux, v, v / static, occ, v_soft,
                       "   beats STATIC" if win else ""))
                rows.append({"layer": li, "lr": lr, "aux": aux, "heldout": v,
                             "heldout_soft_DIAGNOSTIC": v_soft,
                             "static": static, "ratio": v / static, "beats": bool(win),
                             "gate": "hard on both arms", "occ_max": occ})
                routers_out["r%d_lr%g_aux%g" % (li, lr, aux)] =                     m.router.detach().cpu().numpy().copy()
        m.router.data.zero_()
        results[li] = {"static": static, "init": init, "power": denom,
                       "best": {"heldout": best[0], "lr": best[1], "aux": best[2]},
                       "best_ratio": best[0] / static}

    # Per (lr, aux), how many layers did it beat STATIC on?  The setting H1 ships is the one
    # that wins on the MOST layers, not the one with the best single layer.
    tally = {}
    for r in rows:
        tally.setdefault((r["lr"], r["aux"]), [0, 0])
        tally[(r["lr"], r["aux"])][0] += int(r["beats"])
        tally[(r["lr"], r["aux"])][1] += 1
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1][0], kv[0][0]))
    log("")
    log("   settings ranked by layers won:")
    for (lr, aux), (w, n) in ranked:
        log("      lr %-7g aux %-5g   %d of %d layers" % (lr, aux, w, n))
    (blr, baux), (bw, bn) = ranked[0]
    any_win = bw > 0
    log("")
    if any_win:
        log("   GO.  Best setting: --router-lr %g --aux %g  (%d of %d layers beat STATIC)"
            % (blr, baux, bw, bn))
        if bw < bn:
            log("   PARTIAL: it does NOT win on every layer.  Recorded as measured; G-H1e is")
            log("   still scored end-to-end and may still fail there.")
    else:
        log("   NO-GO.  No setting beat STATIC on any layer on the REAL donor.")
        log("   Addendum A consequence 3: H1 DOES NOT LAUNCH and the T4 hours go back unspent.")

    rec = {"brief": "briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md addendum A cons. 2 and 3",
           "what": "LOCAL proxy for G-H1e on the REAL donor -- NOT G-H1e itself",
           "model": C.MODEL_ID, "revision": C.REVISION,
           "calib_slice": mc, "heldout_slice": me,
           "layers": layers, "k": a.k, "E": a.groups, "steps": a.steps, "batch": BATCH,
           "cap_tokens": a.cap, "lrs": list(lrs), "auxs": list(auxs),
           "per_layer": {str(k): v for k, v in results.items()}, "grid": rows,
           "tally": {"lr=%g,aux=%g" % k: {"won": v[0], "of": v[1]} for k, v in tally.items()},
           "decision": {"go": bool(any_win), "router_lr": blr, "aux": baux,
                        "layers_won": bw, "layers_total": bn},
           "seconds": time.time() - t0}
    out = a.out or os.path.join(OUTDIR, "h1_router_smoke.json")
    # The trained routers themselves.  The first version of this file threw them away, so when
    # its evaluation turned out to be mis-gated the ONLY way to re-read it was to retrain
    # everything.  Never again: a measurement that cannot be re-scored is a measurement that
    # must be re-run.
    rpath = os.path.splitext(out)[0] + "_routers.npz"
    np.savez(rpath, **routers_out)
    rec["routers_npz"] = rpath
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)
    log("   wrote %s  (%d trained routers)" % (rpath, len(routers_out)))
    log("   wrote %s  [%.0fs]" % (out, rec["seconds"]))
    return 0 if any_win else 1


if __name__ == "__main__":
    sys.exit(main())
