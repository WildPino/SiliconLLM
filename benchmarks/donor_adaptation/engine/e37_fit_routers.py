#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E37 step 1 -- fit the routers on the REAL donor and score them against random (G-E37D).

Brief: docs/research/donor_adaptation/briefs/BRIEF_E37_WHAT_THE_SPEED_COSTS_A_TRAINED_MODEL.md
(3072941) + its addendum (8cc405a), both pushed before this file existed.

WHY THIS STEP EXISTS AT ALL.  qwen_export.py could only ever write carve_common.router_weights
-- a RANDOM matrix.  That is exactly right for E26, which prices the carve and says in its own
docstring that a trained router is E24's object; it is useless for any statement about quality,
because a random router selects random neurons and would measure the damage of randomness
rather than the damage of sparsity.  E37 needs a router that was actually fitted, so this
writes one, and G-E37D is what stops me from believing it works.

NOTHING HERE IS REIMPLEMENTED.  The ridge fit is e23_router.fit_routers, imported; the labels
are e19_carve_rank.load_labels, imported; the calibration slice is common.get_slice with E23's
own pinned (32, 512, 42424).  If any of those drifted, E23/E24's numbers would drift with them.

  python e37_fit_routers.py                # fit, score, write D:/_ktmp/e37/e37_routers_E256.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "ternary")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))

import common as C                          # noqa: E402
import e19_carve_rank as E19                # noqa: E402
import e23_router as E23                    # noqa: E402
import carve_common as CV                   # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e37_routers.json")

HF = E23.HF                                 # "Qwen/Qwen2.5-1.5B" -- E23's own constant
E_GROUPS = 256
NCAL, SEQCAL, SEEDCAL = E23.NCAL, E23.SEQCAL, E23.SEEDCAL      # 32, 512, 42424
NHELD, SEEDHELD = 8, 909090                 # HELD OUT: a different slice, a different seed
VERDICT_K = 3                               # the brief's verdict cell: 3/256 = 1.17%


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e37")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t0 = time.time()

    log("== E37 step 1: fit the routers on the REAL donor ==")
    log("  donor %s, E=%d, calib (%d x %d, seed %d) -- all imported from E23"
        % (HF, E_GROUPS, NCAL, SEQCAL, SEEDCAL))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(HF, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model.eval()
    D = model.config.hidden_size
    L = model.config.num_hidden_layers
    layer_ids = list(range(L))
    log("  loaded: D=%d L=%d F=%d  [%.0fs]" % (D, L, model.config.intermediate_size,
                                               time.time() - t0))

    labs, _ = E19.load_labels(E_GROUPS)
    ids_cal, _, _ = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    if a.smoke:
        ids_cal, layer_ids = ids_cal[:2], layer_ids[:2]

    log("  fitting (ridge in normal-equation form: O(D^2 + D*E) per layer, not O(tokens)) ...")
    t1 = time.time()
    routers, rdiag, mass_acc = E23.fit_routers(model, ids_cal, layer_ids, labs)
    log("  fitted %d layers in %.0fs" % (len(routers), time.time() - t1))

    # ---- write the artifact the exporter consumes.  fit_routers solves XtX R = XtY with
    # XtY [D, E], so R is [D, E] and the engine's router matrix is [E, D]: transpose ONCE,
    # here, where it can be checked, rather than in the exporter where it would be invisible.
    npz = os.path.join(a.dir, "e37_routers_E%d.npz" % E_GROUPS)
    np.savez(npz, **dict(("r%d" % li, np.ascontiguousarray(routers[li].numpy().T,
                                                           dtype=np.float32))
                         for li in layer_ids))
    log("  wrote %s" % npz)

    # ---- G-E37D: on a HELD-OUT slice, does the fitted router beat a seed-matched random one?
    #      Scored as a RANK metric (E14 s3 forbids a SCORE without one): the fraction of the
    #      ORACLE's top-k groups recovered, and the fraction of the oracle's captured mass.
    ids_h, _, _ = C.get_slice(tok, "heldout", NHELD, SEQCAL, SEEDHELD)
    if a.smoke:
        ids_h = ids_h[:1]
    log("  G-E37D on a HELD-OUT slice (%d x %d, seed %d), k=%d of %d"
        % (ids_h.shape[0], ids_h.shape[1], SEEDHELD, VERDICT_K, E_GROUPS))

    ohs, xs, ms = {}, {}, {}
    pend = {}
    hooks = []

    def mk_in(li):
        def f(mod, inp, out):
            pend[li] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        return f

    def mk_out(li):
        def f(mod, inp, out):
            h = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            m = ((h ** 2) @ ohs[li]).clamp_min(0)
            xs[li].append(pend[li].clone())
            ms[li].append(m)
        return f

    for li in layer_ids:
        lab = torch.from_numpy(labs[li].astype(np.int64))
        oh = torch.zeros(len(lab), E_GROUPS)
        oh[torch.arange(len(lab)), lab] = 1.0
        ohs[li], xs[li], ms[li] = oh, [], []
        lay = model.model.layers[li]
        hooks.append(lay.mlp.gate_proj.register_forward_hook(mk_in(li)))
        hooks.append(lay.mlp.down_proj.register_forward_hook(mk_out(li)))
    for i in range(ids_h.shape[0]):
        model(ids_h[i:i + 1])
    for h in hooks:
        h.remove()

    per_layer, fit_rec, rnd_rec, fit_mass, rnd_mass = {}, [], [], [], []
    for li in layer_ids:
        X = torch.cat(xs[li], 0)
        M = torch.cat(ms[li], 0)
        Rf = routers[li]                                       # [D, E]
        Rr = torch.from_numpy(CV.router_weights(D, E_GROUPS, 26, li)).T   # [D, E], seed-matched
        sel_o = M.topk(VERDICT_K, dim=1).indices
        tot_o = M.gather(1, sel_o).sum(1)
        row = {}
        for nm, R, rl, ml in (("fitted", Rf, fit_rec, fit_mass),
                              ("random", Rr, rnd_rec, rnd_mass)):
            sel = (X @ R).topk(VERDICT_K, dim=1).indices
            hit = (sel.unsqueeze(2) == sel_o.unsqueeze(1)).any(2).float().sum(1) / VERDICT_K
            got = M.gather(1, sel).sum(1)
            rec = float(hit.mean())
            mas = float((got / tot_o.clamp_min(1e-30)).mean())
            row[nm] = {"recall_at_k": rec, "mass_fraction_of_oracle": mas}
            rl.append(rec)
            ml.append(mas)
        row["beats_random"] = bool(row["fitted"]["mass_fraction_of_oracle"]
                                   > row["random"]["mass_fraction_of_oracle"])
        per_layer[li] = row

    nbeat = sum(1 for li in layer_ids if per_layer[li]["beats_random"])
    fires = bool(nbeat == len(layer_ids)
                 and float(np.mean(fit_mass)) > float(np.mean(rnd_mass)))
    log("")
    log("  %-8s  recall@%d   mass fraction of oracle" % ("", VERDICT_K))
    log("  fitted    %.4f      %.4f" % (float(np.mean(fit_rec)), float(np.mean(fit_mass))))
    log("  random    %.4f      %.4f" % (float(np.mean(rnd_rec)), float(np.mean(rnd_mass))))
    log("  layers where fitted beats random: %d/%d -> G-E37D %s"
        % (nbeat, len(layer_ids), "FIRES" if fires else "VOID"))

    out = {"brief": "BRIEF_E37 (3072941) + addendum (8cc405a)", "donor": HF, "E": E_GROUPS,
           "calib": [NCAL, SEQCAL, SEEDCAL], "heldout": [int(ids_h.shape[0]), SEQCAL, SEEDHELD],
           "verdict_k": VERDICT_K, "routers_npz": npz, "smoke": bool(a.smoke),
           "G_E37D": {"per_layer": dict((str(k), v) for k, v in per_layer.items()),
                      "fitted_mean_recall": float(np.mean(fit_rec)),
                      "random_mean_recall": float(np.mean(rnd_rec)),
                      "fitted_mean_mass_fraction": float(np.mean(fit_mass)),
                      "random_mean_mass_fraction": float(np.mean(rnd_mass)),
                      "layers_beating_random": nbeat, "n_layers": len(layer_ids),
                      "fires": fires},
           "ridge_diag": dict((str(k), v) for k, v in rdiag.items()),
           "seconds": time.time() - t0}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0 if fires else 2


if __name__ == "__main__":
    sys.exit(main())
