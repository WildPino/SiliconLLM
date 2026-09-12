#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E41 -- is the PARTITION the lever?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E41_IS_THE_PARTITION_THE_LEVER.md, pushed
BEFORE this file existed.  Nothing here may contradict it.

E38 s7 item 3 owed this.  Every carve in this programme has used ONE label family and nobody has
asked whether it is any good -- while E38's own numbers show a 2.37 BPB swing from the grouping
alone.  E38 proved no router beats the oracle OF A GIVEN PARTITION; a different partition has a
different oracle, so the question is open and it costs a CPU afternoon.

NOTHING IS REIMPLEMENTED.  The masking hook is E38's `Carve`, imported.  The ridge router is
`e23_router.fit_routers`, refit per partition with that partition's own label map -- the same
code E23 and E37 used.  The eval loop, slice and anchors are E38's.

  python e41_partition_lever.py --smoke     # 2 sequences, gates only, ~minutes
  python e41_partition_lever.py             # the measurement
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
sys.path.insert(0, os.path.join(HERE, "..", "ternary"))
sys.path.insert(0, os.path.join(HERE, "..", "density"))

import common as C                          # noqa: E402
import e19_carve_rank as E19                # noqa: E402
import e23_router as E23                    # noqa: E402
from e38_oracle_ceiling import (Carve, bpb_of, HF, REV, E_GROUPS, CHANCE, DENSE_ANCHOR,
                                ANCHOR_TOL, EXPECT_IDS_SHA, NEV, SEQLEN, SEEDEV, NCAL,
                                SEEDCAL, PARITY_TOL, PART_B_SEED)   # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e41_partition_lever.json")

PERM_SEED = 41041
COACT_SEED = 4141
KS = [16, 3]                                # 6.25% = what E40 says 50 tok/s buys at 10 B; 1.17%
SELECTORS = ["oracle", "fitted"]
PARTS = ["D0C", "RAND", "CONC", "STRIPE", "COACT", "PERM"]
VERDICT_K, VERDICT_SEL = 16, "fitted"

# Brief s5.  D0C's own numbers are E38's measurements on this exact slice.
D0C_FITTED_K16, D0C_ORACLE_K16 = 3.597108, 3.449466
BANDS = [(1.005, "PARTITION-IS-THE-LEVER"), (2.50, "PARTITION-HELPS-A-LOT"),
         (3.45, "PARTITION-HELPS"), (3.597, "PARTITION-HELPS-A-LITTLE"),
         (1e9, "PARTITION-IS-NOT-THE-LEVER")]


def log(*a):
    print(*a, flush=True)


def band_of(bpb):
    return next(n for hi, n in BANDS if bpb <= hi)


def equal_groups_from_order(order, F, E):
    """Deal a neuron ORDER into E equal groups, contiguously (group 0 = first F/E of order)."""
    lab = np.empty(F, dtype=np.int64)
    lab[order] = np.arange(F) // (F // E)
    return lab


def stripe_groups_from_order(order, F, E):
    """Deal a neuron ORDER round-robin, so each group draws evenly from every mass rank."""
    lab = np.empty(F, dtype=np.int64)
    lab[order] = np.arange(F) % E
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t0 = time.time()

    log("== E41: is the PARTITION the lever? ==")
    log("  QUALITY ONLY, fp32, a really trained donor.  No speed, no synthetic weights.")
    log("  E38 s7 item 3: every carve here has used ONE label family, never audited.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(HF, revision=REV, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model.eval()
    D = model.config.hidden_size
    F = model.config.intermediate_size
    L = model.config.num_hidden_layers
    GSZ = F // E_GROUPS
    log("  donor %s  D=%d F=%d L=%d  E=%d (group = %d neurons)" % (HF, D, F, L, E_GROUPS, GSZ))

    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", NEV, SEQLEN, SEEDEV)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH -- STOP")
    if a.smoke:
        ids_ev, byts_ev = ids_ev[:2], byts_ev[:2]
    b_tot = float(byts_ev.sum())
    log("  eval slice %d x %d, %d scored bytes, ids_sha %s"
        % (ids_ev.shape[0], SEQLEN, int(b_tot), EXPECT_IDS_SHA[:16]))

    ids_cal, _, _ = C.get_slice(tok, "calib", 2 if a.smoke else NCAL, SEQLEN, SEEDCAL)
    log("  calib slice %d x %d, seed %d -- NEVER the eval one" % (ids_cal.shape[0], SEQLEN,
                                                                  SEEDCAL))

    out = {"brief": "BRIEF_E41 (pre-registered)", "donor": HF, "revision": REV,
           "precision": "fp32", "E": E_GROUPS, "group_neurons": GSZ, "ks": KS,
           "eval_slice": {"part": "heldout", "n": int(ids_ev.shape[0]), "seqlen": SEQLEN,
                          "seed": SEEDEV, "scored_bytes": b_tot, "ids_sha256": EXPECT_IDS_SHA},
           "chance_bpb": CHANCE, "d0c_fitted_k16": D0C_FITTED_K16,
           "d0c_oracle_k16": D0C_ORACLE_K16}

    # ---- G-E41B: the anchor, before anything is masked.
    t = time.time()
    dense = bpb_of(model, ids_ev, b_tot)
    gb = {"bpb": dense, "anchor": DENSE_ANCHOR, "diff": dense - DENSE_ANCHOR, "tol": ANCHOR_TOL,
          "fires": bool(abs(dense - DENSE_ANCHOR) < ANCHOR_TOL or a.smoke),
          "seconds": time.time() - t}
    out["G_E41B"] = gb
    log("")
    log("  G-E41B  dense fp32 = %.9f  vs E22 base %.6f  diff %+.2e -> %s"
        % (dense, DENSE_ANCHOR, dense - DENSE_ANCHOR, "FIRES" if gb["fires"] else "VOID"))
    if not gb["fires"]:
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E41B VOID -- this session is not measuring E22's object. STOP.")

    # ---- per-neuron calibration mass, which three of the partitions are built from.
    # Collected with the SAME hook the selectors use, on the calib slice, never the eval one.
    d0c_np, _ = E19.load_labels(E_GROUPS)
    gD0C = [torch.from_numpy(d0c_np[li].astype(np.int64)) for li in range(L)]
    carve = Carve(model, gD0C, None, None, None)

    neuron_mass = [torch.zeros(F, dtype=torch.float64) for _ in range(L)]
    coact = [torch.zeros(F, dtype=torch.float64) for _ in range(L)]     # mean |activation|
    handles = []

    def mk_stat(li):
        def f(mod, args):
            flat = args[0].reshape(-1, args[0].shape[-1]).float()
            neuron_mass[li] += (flat ** 2).sum(0).double()
            coact[li] += flat.abs().sum(0).double()
            return None
        return f

    for li, lay in enumerate(model.model.layers):
        handles.append(lay.mlp.down_proj.register_forward_pre_hook(mk_stat(li)))
    carve.mode = "off"
    for i in range(ids_cal.shape[0]):
        model(ids_cal[i:i + 1])
    for h in handles:
        h.remove()
    log("  per-neuron calibration statistics collected on %d sequences" % ids_cal.shape[0])

    # ---- the six partitions.
    rng_r = np.random.default_rng(PART_B_SEED)
    rng_p = np.random.default_rng(PERM_SEED)
    rng_c = np.random.default_rng(COACT_SEED)
    G = {"D0C": gD0C, "RAND": [], "CONC": [], "STRIPE": [], "COACT": [], "PERM": []}
    for li in range(L):
        G["RAND"].append(torch.from_numpy(equal_groups_from_order(rng_r.permutation(F), F,
                                                                  E_GROUPS)))
        order = np.argsort(-neuron_mass[li].numpy())          # heaviest first
        G["CONC"].append(torch.from_numpy(equal_groups_from_order(order, F, E_GROUPS)))
        G["STRIPE"].append(torch.from_numpy(stripe_groups_from_order(order, F, E_GROUPS)))
        # COACT: group neurons whose mean-|activation| profile is similar.  A 1-D projection of
        # the co-activation structure -- cheap, and enough to ask whether the AXIS matters.
        key = coact[li].numpy() / max(1.0, float(ids_cal.shape[0]))
        jitter = rng_c.normal(0.0, 1e-12, size=F)             # break exact ties deterministically
        G["COACT"].append(torch.from_numpy(equal_groups_from_order(
            np.argsort(-(key + jitter)), F, E_GROUPS)))
        # PERM: D0C with the group IDs relabelled.  Same neuron SETS, different names.
        pm = torch.from_numpy(rng_p.permutation(E_GROUPS).astype(np.int64))
        G["PERM"].append(pm[gD0C[li]])

    for nm in PARTS:
        bc = torch.bincount(G[nm][0], minlength=E_GROUPS)
        assert bc.min() > 0, "%s left an empty group" % nm
        log("  partition %-6s groups %d, size min/max %d/%d" % (nm, int((bc > 0).sum()),
                                                                int(bc.min()), int(bc.max())))

    # ---- routers, refit PER PARTITION with e23_router.fit_routers (E23/E37's own code).
    routers = {}
    for nm in PARTS:
        t = time.time()
        lab_map = {li: G[nm][li].numpy() for li in range(L)}
        R, _, _ = E23.fit_routers(model, ids_cal, list(range(L)), lab_map)
        routers[nm] = [torch.as_tensor(R[li]).float().T.contiguous() if
                       torch.as_tensor(R[li]).shape[0] == E_GROUPS else
                       torch.as_tensor(R[li]).float().contiguous() for li in range(L)]
        log("  router refit for %-6s [%.0fs]" % (nm, time.time() - t))
    out["router_fit"] = "e23_router.fit_routers, refit per partition, sqrt target"

    # ---- the sweep.
    rows = {}

    def cell(part, sel, k):
        carve.labels = G[part]
        carve.routers = routers[part]
        carve.mode, carve.k = sel, k
        t1 = time.time()
        b = bpb_of(model, ids_ev, b_tot)
        carve.mode = "off"
        r = {"partition": part, "selector": sel, "k": k, "bpb": b, "vs_dense": b - dense,
             "vs_chance": b - CHANCE, "activation_pct": 100.0 * k / E_GROUPS,
             "seconds": time.time() - t1}
        rows["%s_%s_k%d" % (part, sel, k)] = r
        log("    %-6s %-7s k=%-4d  BPB %.6f   vs dense %+.4f   vs chance %+.4f  [%.0fs]"
            % (part, sel, k, b, b - dense, b - CHANCE, time.time() - t1))
        return r

    # ---- G-E41A: inertness at k = E, per partition.
    log("")
    log("  G-E41A  inertness at k = E, every partition")
    ga = {"per_partition": {}}
    for nm in PARTS:
        r = cell(nm, "oracle", E_GROUPS)
        d = abs(r["bpb"] - dense)
        ga["per_partition"][nm] = {"bpb": r["bpb"], "abs_diff": d,
                                   "inert": bool(d < PARITY_TOL)}
    ga["worst"] = max(v["abs_diff"] for v in ga["per_partition"].values())
    ga["fires"] = bool(all(v["inert"] for v in ga["per_partition"].values()))
    out["G_E41A"] = ga
    log("  G-E41A  worst |BPB - dense| at k=E is %.2e -> %s"
        % (ga["worst"], "FIRES" if ga["fires"] else "VOID"))

    log("")
    for k in ([16] if a.smoke else KS):
        for nm in PARTS:
            for sel in SELECTORS:
                if a.smoke and (nm, sel) not in (("D0C", "oracle"), ("PERM", "oracle")):
                    continue
                cell(nm, sel, k)
    out["cells"] = rows

    # ---- G-E41C: the planted control, BIT-EXACT.  PERM is D0C relabelled, so the oracle -- which
    #      ranks neuron SETS by mass -- must return the identical number, not a close one.
    gc = {"per_k": {}}
    for k in ([16] if a.smoke else KS):
        d0 = rows.get("D0C_oracle_k%d" % k)
        pm = rows.get("PERM_oracle_k%d" % k)
        if not (d0 and pm):
            continue
        gc["per_k"][str(k)] = {"d0c": d0["bpb"], "perm": pm["bpb"],
                               "diff": pm["bpb"] - d0["bpb"],
                               "exact": bool(pm["bpb"] == d0["bpb"])}
    gc["fires"] = bool(gc["per_k"] and all(v["exact"] for v in gc["per_k"].values()))
    out["G_E41C"] = gc
    log("")
    log("  G-E41C  PERM oracle == D0C oracle EXACTLY ? -> %s  (%s)"
        % ("FIRES" if gc["fires"] else "VOID",
           " ".join("k%s diff %+.3e" % (k, v["diff"]) for k, v in gc["per_k"].items())))
    if not gc["fires"]:
        log("          permuting group NAMES moved the number: the instrument is selecting on")
        log("          labels, not on neurons.  No partition comparison below this counts.")

    # ---- the verdict: the BEST partition's fitted BPB at k=16.
    cand = {nm: rows["%s_%s_k%d" % (nm, VERDICT_SEL, VERDICT_K)]["bpb"]
            for nm in PARTS if "%s_%s_k%d" % (nm, VERDICT_SEL, VERDICT_K) in rows}
    if cand:
        best = min(cand, key=cand.get)
        out["verdict"] = {"cell": "best %s at k=%d" % (VERDICT_SEL, VERDICT_K), "partition": best,
                          "bpb": cand[best], "name": band_of(cand[best]),
                          "vs_d0c_fitted": cand[best] - D0C_FITTED_K16,
                          "vs_d0c_oracle": cand[best] - D0C_ORACLE_K16,
                          "all": cand}
        log("")
        log("  VERDICT  best %s at k=%d is %s -> %.6f BPB  (%+.4f vs D0C fitted, %+.4f vs D0C "
            "oracle) -> %s" % (VERDICT_SEL, VERDICT_K, best, cand[best],
                               cand[best] - D0C_FITTED_K16, cand[best] - D0C_ORACLE_K16,
                               band_of(cand[best])))

    carve.close()
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
