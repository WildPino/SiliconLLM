#!/usr/bin/env python3
"""S1 -- the oracle-on-|h_i| bound, at every size.  BRIEF_S1 Amendment 1 A1.4 point 2.

This is F1's quantity, NOT a BPB: the minimum ACTIVE fraction `a` that an oracle on |h_i| needs,
so that zeroing the rest costs at most `eps` of a reference norm.  F1 reported it at 1.5B only,
and the F1 audit then made two corrections that this module is built around.

------------------------------------------------------------------------------------------
(1) THE REFERENCE OBJECT IS NOT NEUTRAL, AND THE TWO CHOICES ARE NOT INTERCHANGEABLE.

F1 referenced `eps` to the FFN BLOCK's own output -- post-`down_proj`, pre-residual.  The audit
found that on 23 of 28 layers that object is only 4-14% of the residual stream, so a "1% budget"
there is a 0.04-0.14% budget on what the next layer actually reads; re-referenced, mean `a` fell
from 0.973 to 0.865.

    a_block : ||dY||_F / ||W_d h||_F        <= eps      (F1's choice)
    a_resid : ||dY||_F / ||residual_out||_F <= eps      (what the next layer sees)

BOTH are computed and reported side by side.  Neither is presented as "the" reference, and the
measured block-share-of-residual is printed so the reader can convert between them.

(2) IT IS TOKEN-COUNT SENSITIVE.  The audit re-derived a(eps=0.01) at 256 vs 1024 tokens and it
moved 0.044 at layer 26 and 0.020 at layer 3 -- larger than the separations F1 argued over.  So
`a` is computed at SEVERAL token budgets here and the movement between them is reported as the
resolution of the number.  Every published value carries its n.

Usage:
    python s1_oracle.py --model qwen2.5-1.5b
    python s1_oracle.py --model qwen2.5-7b --dtype float16 --device cuda --device-map auto
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import s1_sparsity_bpb as S

ORACLE_EPS = (0.001, 0.01, 0.05)
ORACLE_TOKENS = (256, 512, 1024, 2048)
ORACLE_GRID = 33          # drop-fraction grid; resolution of `a` is 1/(GRID-1) = 0.031


def stage_oracle(key, dtype=torch.float32, device="cpu", device_map=None,
                 eps_list=ORACLE_EPS, token_budgets=ORACLE_TOKENS, grid=ORACLE_GRID):
    t0 = time.time()
    model, tok, meta = S.load(key, dtype=dtype, device=device, device_map=device_map)
    S.log("donor: " + json.dumps(meta))
    ids, byts, smeta = S.get_eval_slice(tok)
    L = len(model.model.layers)
    nmax = max(token_budgets)
    seqlen = int(ids.shape[1])
    nseq = int(np.ceil(nmax / seqlen))

    hs = {li: [] for li in range(L)}      # the SwiGLU intermediate h, per layer
    rs = {li: [] for li in range(L)}      # the decoder layer's OUTPUT hidden state (residual)

    def mk_h(li):
        def pre(mod, args):
            x = args[0]
            h = (mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)).detach()
            hs[li].append(h.reshape(-1, h.shape[-1]).float().cpu())
        return model.model.layers[li].mlp.register_forward_pre_hook(pre)

    def mk_r(li):
        def post(mod, args, out):
            o = out[0] if isinstance(out, tuple) else out
            rs[li].append(o.detach().reshape(-1, o.shape[-1]).float().cpu())
        return model.model.layers[li].register_forward_hook(post)

    hooks = [mk_h(li) for li in range(L)] + [mk_r(li) for li in range(L)]
    S.CTL.mode = "off"                     # no masking here; this measures the donor as it is
    dev = next(model.parameters()).device
    for i in range(0, nseq, S.EVAL_BATCH):
        model(ids[i:i + S.EVAL_BATCH].to(dev))
    for h in hooks:
        h.remove()

    out = {"stage": "oracle", "donor": meta, "slice": smeta, "env": S.env_block(),
           "eps_list": list(eps_list), "token_budgets": list(token_budgets),
           "grid_points": grid, "a_resolution": 1.0 / (grid - 1),
           "n_tokens_available_achieved": int(min(nmax, nseq * seqlen)),
           "REFERENCE_NOTE": (
               "a_block references eps to ||W_d h||_F, the FFN block's own output -- what F1 "
               "used.  a_resid references it to ||residual_out||_F, the hidden state the next "
               "layer actually reads.  They differ by the block's share of the residual, which "
               "is reported per layer.  They are NOT interchangeable."),
           "by_layer": {}}

    fracs = np.linspace(0.0, 1.0, grid)
    for li in range(L):
        H = torch.cat(hs[li], 0)[:nmax]
        R = torch.cat(rs[li], 0)[:nmax]
        hs[li] = rs[li] = None
        Wd = model.model.layers[li].mlp.down_proj.weight.detach().float().cpu()
        rec = {"block_share_of_residual": {}, "a_block": {}, "a_resid": {}}
        for nt in token_budgets:
            if nt > H.shape[0]:
                continue
            h, r = H[:nt], R[:nt]
            nY = float((h @ Wd.T).norm())
            nR = float(r.norm())
            rec["block_share_of_residual"][str(nt)] = nY / max(nR, 1e-30)
            order = torch.argsort(h.abs(), dim=1)          # ascending |h| -> drop the smallest
            errs = np.empty(grid)
            for j, f in enumerate(fracs):
                kd = int(round(float(f) * h.shape[1]))
                if kd == 0:
                    errs[j] = 0.0
                    continue
                m = torch.zeros_like(h, dtype=torch.bool)
                m.scatter_(1, order[:, :kd], True)
                errs[j] = float((torch.where(m, h, torch.zeros_like(h)) @ Wd.T).norm())
            for eps in eps_list:
                for tag, den in (("a_block", nY), ("a_resid", nR)):
                    ok = np.where(errs / max(den, 1e-30) <= eps)[0]
                    drop = float(fracs[ok[-1]]) if len(ok) else 0.0
                    rec[tag].setdefault(str(eps), {})[str(nt)] = 1.0 - drop
        out["by_layer"][str(li)] = rec
        del H, R, Wd
        ref = str(min(1024, nmax))
        S.log("  layer %2d  block/resid=%.4f  a_block(.01)@%s=%.4f  a_resid(.01)@%s=%.4f"
              % (li, rec["block_share_of_residual"][ref], ref, rec["a_block"]["0.01"][ref],
                 ref, rec["a_resid"]["0.01"][ref]))

    # ---- aggregates.  The MEAN OVER LAYERS is the number the F1 audit corrected to ~0.7984;
    # layer 0 is reported separately because 0.836 was layer 0's value, not the mean.
    agg = {}
    for tag in ("a_block", "a_resid"):
        agg[tag] = {}
        for eps in eps_list:
            agg[tag][str(eps)] = {}
            for nt in token_budgets:
                v = [out["by_layer"][str(li)][tag][str(eps)].get(str(nt)) for li in range(L)]
                v = [x for x in v if x is not None]
                if not v:
                    continue
                agg[tag][str(eps)][str(nt)] = {
                    "mean_over_layers": float(np.mean(v)), "sd_over_layers": float(np.std(v)),
                    "min": float(np.min(v)), "max": float(np.max(v)),
                    "layer0": float(v[0]), "n_layers": len(v), "n_tokens": nt}
    out["aggregate"] = agg

    ref = str(min(1024, nmax))
    bs = [out["by_layer"][str(li)]["block_share_of_residual"][ref] for li in range(L)]
    out["block_share_of_residual_summary"] = {
        "at_n_tokens": int(ref), "mean": float(np.mean(bs)), "min": float(np.min(bs)),
        "max": float(np.max(bs)), "n_layers_below_0.15": int(sum(1 for x in bs if x < 0.15)),
        "n_layers": L, "per_layer": [float(x) for x in bs]}

    lo, hi = str(min(token_budgets)), str(max(token_budgets))
    out["token_sensitivity"] = {}
    for tag in ("a_block", "a_resid"):
        out["token_sensitivity"][tag] = {}
        for eps in eps_list:
            a = agg[tag][str(eps)].get(lo)
            b = agg[tag][str(eps)].get(hi)
            if not (a and b):
                continue
            per = [abs(out["by_layer"][str(li)][tag][str(eps)][hi]
                       - out["by_layer"][str(li)][tag][str(eps)][lo]) for li in range(L)]
            out["token_sensitivity"][tag][str(eps)] = {
                "mean_at_%s_tokens" % lo: a["mean_over_layers"],
                "mean_at_%s_tokens" % hi: b["mean_over_layers"],
                "movement_of_the_mean": abs(b["mean_over_layers"] - a["mean_over_layers"]),
                "max_per_layer_movement": float(np.max(per)),
                "argmax_layer": int(np.argmax(per))}

    out["elapsed_s"] = time.time() - t0
    out["peak_rss_gb"] = S.peak_rss_gb()
    pth = os.path.join(S.RESULTS, f"s1_oracle_{key}.json")
    json.dump(out, open(pth, "w"), indent=2, default=float)
    S.log("ORACLE done (%.0fs) -> %s" % (out["elapsed_s"], pth))
    for tag in ("a_block", "a_resid"):
        g = agg[tag]["0.01"].get(ref)
        if g:
            S.log("  %s(eps=0.01) mean over %d layers = %.4f  (layer0 %.4f, min %.4f, max %.4f, "
                  "n_tokens=%s)" % (tag, g["n_layers"], g["mean_over_layers"], g["layer0"],
                                    g["min"], g["max"], ref))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=S.DONOR_KEY)
    ap.add_argument("--dtype", default="float32", choices=list(S.DTYPES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--device-map", default=None)
    a = ap.parse_args()
    stage_oracle(a.model, dtype=S.DTYPES[a.dtype], device=a.device, device_map=a.device_map)
