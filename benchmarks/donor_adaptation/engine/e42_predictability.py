#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E42 -- is PREDICTABILITY the criterion, and does it have headroom?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E42_IS_PREDICTABILITY_THE_CRITERION.md,
pushed at 22653fd BEFORE this file existed.  Nothing here may contradict it.

E41 came out UNRESOLVABLE and left two things owed, which are one probe: the dispersion on the
axis that reversed its ordering was never measured, and the mechanism it DID identify -- gain
from more calibration tracks how PREDICTABLE a grouping is -- was never used as a criterion.

NOTHING IS REIMPLEMENTED.  The masking hook is E38's `Carve`; the ridge router is
`e23_router.fit_routers`; the eval loop, slice and anchors are E38's.  The one new thing is the
`PRED` partition, built in this file and specified in brief s4 before it existed.

  python e42_predictability.py --smoke     # 2 eval seqs, 1 seed, gates only
  python e42_predictability.py             # the measurement
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
                                ANCHOR_TOL, EXPECT_IDS_SHA, NEV, SEQLEN, SEEDEV,
                                PARITY_TOL, PART_B_SEED)             # noqa: E402
from e41_partition_lever import (COACT_SEED, equal_groups_from_order)   # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e42_predictability.json")

K = 16                                      # brief s2: 6.25%, and no k sweep
ARMS = ["D0C", "COACT", "PRED", "RAND"]
RCAL = 32                                   # brief s2: the reference router budget
SEEDS = [42424, 42425, 42426]               # brief s2: the REPLICATED axis
STAT_SEED = 42424                           # partitions are built once, on the reference slice
PRED_KM_SEED = 42042                        # brief s4
PRED_KM_ITERS = 25
E38_D0C_FITTED = 3.597108                   # G-E42A, the known positive
E41_COACT_32 = 3.583800236827143            # the incumbent, addendum B's reading
E41_RAND_32 = 4.535311855                   # G-E42C reference
E41_D0C_ORACLE = 3.449466200847911
GA_TOL, GC_TOL, GD_FRAC = 0.001, 0.01, 0.50

BANDS = [(1.005, "PREDICTABILITY-IS-THE-LEVER"), (2.50, "PREDICTABILITY-HAS-HEADROOM"),
         (3.449466, "PREDICTABILITY-HELPS"), (E41_COACT_32, "PREDICTABILITY-HELPS-A-LITTLE"),
         (1e9, "PREDICTABILITY-IS-NOT-THE-CRITERION")]


def log(*a):
    print(*a, flush=True)


def band_of(b):
    return next(n for hi, n in BANDS if b <= hi)


def balanced_assign(sim, gsz):
    """Assign each row to a column with capacity `gsz`, most-confident rows first.

    `sim` is [F, E] similarity (higher is better).  Confidence is best minus second best, which
    is the margin the assignment would lose by being overruled.  Deterministic.
    """
    F, E = sim.shape
    top2 = np.argpartition(-sim, 1, axis=1)[:, :2]
    b0 = sim[np.arange(F), top2[:, 0]]
    b1 = sim[np.arange(F), top2[:, 1]]
    conf = np.abs(b0 - b1)
    order = np.argsort(-conf)
    rank = np.argsort(-sim, axis=1)          # [F, E], best first
    cap = np.full(E, gsz, dtype=np.int64)
    lab = np.full(F, -1, dtype=np.int64)
    for j in order:
        for c in rank[j]:
            if cap[c] > 0:
                lab[j] = c
                cap[c] -= 1
                break
    assert (lab >= 0).all() and (cap == 0).all()
    return lab


def spherical_kmeans(Wn, E, seed, iters):
    """Wn is [F, D] unit rows.  Returns centroids [E, D], unit."""
    rng = np.random.default_rng(seed)
    F = Wn.shape[0]
    cen = Wn[rng.choice(F, size=E, replace=False)].copy()
    for _ in range(iters):
        sim = Wn @ cen.T                                     # [F, E]
        asg = sim.argmax(1)
        new = np.zeros_like(cen)
        np.add.at(new, asg, Wn)
        nrm = np.linalg.norm(new, axis=1, keepdims=True)
        dead = (nrm[:, 0] < 1e-12)
        if dead.any():                                       # re-seed empty clusters
            new[dead] = Wn[rng.choice(F, size=int(dead.sum()), replace=False)]
            nrm = np.linalg.norm(new, axis=1, keepdims=True)
        cen = new / np.maximum(nrm, 1e-12)
    return cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t0 = time.time()
    seeds = SEEDS[:1] if a.smoke else SEEDS

    log("== E42: is PREDICTABILITY the criterion, and does it have headroom? ==")
    log("  QUALITY ONLY, fp32, a really trained donor.  No speed, no GPU.")
    log("  router budget fixed at %d seqs; the CALIBRATION SEED is the replicated axis: %s"
        % (RCAL, seeds))

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
    ids_stat, _, _ = C.get_slice(tok, "calib", RCAL, SEQLEN, STAT_SEED)
    log("  eval %d x %d (%d bytes); partitions built ONCE on calib %d x %d seed %d"
        % (ids_ev.shape[0], SEQLEN, int(b_tot), ids_stat.shape[0], SEQLEN, STAT_SEED))

    out = {"brief": "BRIEF_E42 (pre-registered at 22653fd)", "donor": HF, "revision": REV,
           "precision": "fp32", "E": E_GROUPS, "group_neurons": GSZ, "k": K, "arms": ARMS,
           "router_budget": RCAL, "seeds": seeds, "stat_seed": STAT_SEED,
           "eval_slice": {"n": int(ids_ev.shape[0]), "seqlen": SEQLEN, "seed": SEEDEV,
                          "scored_bytes": b_tot, "ids_sha256": EXPECT_IDS_SHA},
           "chance_bpb": CHANCE, "e38_d0c_fitted": E38_D0C_FITTED,
           "e41_coact_32": E41_COACT_32, "e41_rand_32": E41_RAND_32}

    dense = bpb_of(model, ids_ev, b_tot)
    out["dense"] = dense
    log("  dense fp32 = %.9f  (E22 base %.6f, diff %+.2e)"
        % (dense, DENSE_ANCHOR, dense - DENSE_ANCHOR))
    if abs(dense - DENSE_ANCHOR) >= ANCHOR_TOL and not a.smoke:
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("anchor VOID -- not measuring E22's object. STOP.")

    # ---------------------------------------------------------------- statistics, one pass.
    # Collected with the same taps e23_router uses: x is the gate_proj input (what the router
    # reads), h is the down_proj input (post-SwiGLU).  We accumulate, per layer:
    #   coact[F]     mean |h| per neuron          -> COACT, E41's criterion
    #   XtX[D,D]     x'x                          -> shared by the PRED ridge
    #   XtA[D,F]     x'|h|                        -> PRED's per-neuron target (brief s4)
    d0c_np, _ = E19.load_labels(E_GROUPS)
    gD0C = [torch.from_numpy(d0c_np[li].astype(np.int64)) for li in range(L)]

    coact = [torch.zeros(F, dtype=torch.float64) for _ in range(L)]
    XtX = [torch.zeros(D, D, dtype=torch.float64) for _ in range(L)]
    XtA = [torch.zeros(D, F, dtype=torch.float64) for _ in range(L)]
    pend, handles = {}, []

    def mk_in(li):
        def f(mod, inp, out):
            pend[li] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        return f

    def mk_out(li):
        def f(mod, inp, out):
            h = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            ah = h.abs()
            coact[li] += ah.sum(0).double()
            x = pend[li]
            XtX[li] += (x.T @ x).double()
            XtA[li] += (x.T @ ah).double()
        return f

    for li, lay in enumerate(model.model.layers):
        handles.append(lay.mlp.gate_proj.register_forward_hook(mk_in(li)))
        handles.append(lay.mlp.down_proj.register_forward_hook(mk_out(li)))
    t = time.time()
    for i in range(ids_stat.shape[0]):
        model(ids_stat[i:i + 1])
    for h in handles:
        h.remove()
    pend.clear()
    log("  statistics collected on %d sequences [%.0fs]" % (ids_stat.shape[0], time.time() - t))

    # ---------------------------------------------------------------- the four partitions.
    rng_r = np.random.default_rng(PART_B_SEED)
    rng_c = np.random.default_rng(COACT_SEED)
    G = {"D0C": gD0C, "COACT": [], "PRED": [], "RAND": []}
    t = time.time()
    for li in range(L):
        G["RAND"].append(torch.from_numpy(equal_groups_from_order(rng_r.permutation(F), F,
                                                                  E_GROUPS)))
        key = coact[li].numpy() / max(1.0, float(ids_stat.shape[0]))
        jitter = rng_c.normal(0.0, 1e-12, size=F)
        G["COACT"].append(torch.from_numpy(equal_groups_from_order(
            np.argsort(-(key + jitter)), F, E_GROUPS)))
        # PRED, brief s4: per-neuron ridge predictor of |h| from x, unit-normalised, spherical
        # k-means into E balanced groups of GSZ.
        A = XtX[li]
        lam = E23.RIDGE_FRAC * float(torch.diagonal(A).mean())
        W = torch.linalg.solve(A + torch.eye(D, dtype=torch.float64) * lam, XtA[li])   # [D, F]
        Wn = W.T.numpy()                                                               # [F, D]
        nrm = np.linalg.norm(Wn, axis=1, keepdims=True)
        Wn = Wn / np.maximum(nrm, 1e-30)
        cen = spherical_kmeans(Wn, E_GROUPS, PRED_KM_SEED + li, PRED_KM_ITERS)
        G["PRED"].append(torch.from_numpy(balanced_assign(Wn @ cen.T, GSZ)))
    log("  partitions built [%.0fs]" % (time.time() - t))
    del XtA, XtX

    # ---- G-E42D: PRED is a real partition and is not COACT under another name.
    gd = {"per_layer_disagree": [], "sizes_ok": True}
    for li in range(L):
        for nm in ARMS:
            bc = torch.bincount(G[nm][li], minlength=E_GROUPS)
            if not (len(bc) == E_GROUPS and int(bc.min()) == GSZ and int(bc.max()) == GSZ):
                gd["sizes_ok"] = False
        # relabelling-invariant disagreement: fraction of neurons whose PRED groupmates differ
        p, c = G["PRED"][li].numpy(), G["COACT"][li].numpy()
        same = 0
        for g in range(E_GROUPS):
            idx = np.where(p == g)[0]
            same += int(np.bincount(c[idx], minlength=E_GROUPS).max())
        gd["per_layer_disagree"].append(1.0 - same / float(F))
    gd["worst_disagree"] = float(min(gd["per_layer_disagree"]))
    gd["fires"] = bool(gd["sizes_ok"] and gd["worst_disagree"] >= GD_FRAC)
    out["G_E42D"] = gd
    log("  G-E42D  all arms %d groups of %d: %s;  PRED vs COACT disagreement >= %.0f%% in every "
        "layer (worst %.1f%%) -> %s" % (E_GROUPS, GSZ, gd["sizes_ok"], 100 * GD_FRAC,
                                        100 * gd["worst_disagree"],
                                        "FIRES" if gd["fires"] else "VOID"))

    carve = Carve(model, gD0C, None, None, None)
    rows, cells = {}, {}

    def cell(part, sel, k, routers=None, tag=""):
        carve.labels = G[part]
        carve.routers = routers
        carve.mode, carve.k = sel, k
        t1 = time.time()
        b = bpb_of(model, ids_ev, b_tot)
        carve.mode = "off"
        log("    %-6s %-7s k=%-4d %s BPB %.6f   vs dense %+.4f   vs chance %+.4f  [%.0fs]"
            % (part, sel, k, tag, b, b - dense, b - CHANCE, time.time() - t1))
        return b

    # ---- G-E42B: inertness at k = E, per partition.
    log("")
    log("  G-E42B  inertness at k = E, every partition")
    gb = {"per_partition": {}}
    for nm in ARMS:
        b = cell(nm, "oracle", E_GROUPS)
        gb["per_partition"][nm] = {"bpb": b, "abs_diff": abs(b - dense),
                                   "inert": bool(abs(b - dense) < PARITY_TOL)}
    gb["worst"] = max(v["abs_diff"] for v in gb["per_partition"].values())
    gb["fires"] = bool(all(v["inert"] for v in gb["per_partition"].values()))
    out["G_E42B"] = gb
    log("  G-E42B  worst |BPB - dense| at k=E is %.2e -> %s"
        % (gb["worst"], "FIRES" if gb["fires"] else "VOID"))

    # ---- the sweep: four arms x three calibration seeds, fitted selector, k=16.
    log("")
    for sd in seeds:
        ids_fit, _, _ = C.get_slice(tok, "calib", RCAL, SEQLEN, sd)
        for nm in ARMS:
            t = time.time()
            lab_map = {li: G[nm][li].numpy() for li in range(L)}
            R, _, _ = E23.fit_routers(model, ids_fit, list(range(L)), lab_map)
            rt = [torch.as_tensor(R[li]).float().T.contiguous() if
                  torch.as_tensor(R[li]).shape[0] == E_GROUPS else
                  torch.as_tensor(R[li]).float().contiguous() for li in range(L)]
            log("  router fit %-6s seed %d on %d seqs [%.0fs]"
                % (nm, sd, ids_fit.shape[0], time.time() - t))
            b = cell(nm, "fitted", K, rt, "seed %d " % sd)
            cells["%s_%d" % (nm, sd)] = {"arm": nm, "seed": sd, "bpb": b,
                                         "vs_dense": b - dense, "vs_chance": b - CHANCE}
    out["cells"] = cells

    for nm in ARMS:
        v = np.array([cells["%s_%d" % (nm, sd)]["bpb"] for sd in seeds], dtype=np.float64)
        rows[nm] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    "min": float(v.min()), "max": float(v.max()), "per_seed": v.tolist()}
    out["arms"] = rows

    # ---- G-E42A: the known positive.  Both G-E42A and G-E42C compare against numbers measured
    #      on the FULL registered slice, so under --smoke (2 sequences) they are NOT APPLICABLE
    #      rather than VOID: a smoke slice cannot reproduce a full-slice anchor by construction.
    d0c_ref = cells["D0C_%d" % SEEDS[0]]["bpb"]
    ga = {"d0c_seed42424": d0c_ref, "e38": E38_D0C_FITTED, "diff": d0c_ref - E38_D0C_FITTED,
          "tol": GA_TOL, "fires": bool(abs(d0c_ref - E38_D0C_FITTED) < GA_TOL),
          "applicable": not a.smoke}
    out["G_E42A"] = ga
    log("")
    log("  G-E42A  D0C at seed %d = %.6f  vs E38 %.6f  diff %+.2e -> %s"
        % (SEEDS[0], d0c_ref, E38_D0C_FITTED, ga["diff"],
           "N/A (smoke: 2-seq slice, not the registered one)" if a.smoke
           else ("FIRES" if ga["fires"] else "VOID")))

    # ---- G-E42C: the null must behave like a null.
    gain = E41_RAND_32 - rows["RAND"]["mean"]
    gc = {"rand_mean": rows["RAND"]["mean"], "e41_rand_32": E41_RAND_32, "gain": gain,
          "tol": GC_TOL, "fires": bool(gain <= GC_TOL), "applicable": not a.smoke}
    out["G_E42C"] = gc
    log("  G-E42C  RAND mean %.6f vs its E41 32-seq reading %.6f -> gain %+.6f (max %.2f) -> %s"
        % (gc["rand_mean"], E41_RAND_32, gain, GC_TOL,
           "N/A (smoke: 2-seq slice, not the registered one)" if a.smoke
           else ("FIRES" if gc["fires"] else "VOID")))

    # ---- the verdict: PRED's mean.
    pm = rows["PRED"]["mean"]
    out["verdict"] = {"cell": "PRED mean fitted BPB at k=%d over %d seeds" % (K, len(seeds)),
                      "bpb": pm, "name": band_of(pm), "sd": rows["PRED"]["sd"],
                      "vs_coact_mean": pm - rows["COACT"]["mean"],
                      "vs_d0c_mean": pm - rows["D0C"]["mean"],
                      "vs_e41_coact_32": pm - E41_COACT_32,
                      "beats_coact_every_seed": bool(all(
                          cells["PRED_%d" % sd]["bpb"] < cells["COACT_%d" % sd]["bpb"]
                          for sd in seeds))}

    # ---- the secondary registered question: is COACT - D0C resolved at 2 sigma?
    dd = np.array([cells["COACT_%d" % sd]["bpb"] - cells["D0C_%d" % sd]["bpb"] for sd in seeds])
    sd_d = float(dd.std(ddof=1)) if len(dd) > 1 else 0.0
    out["secondary"] = {"coact_minus_d0c_per_seed": dd.tolist(), "mean": float(dd.mean()),
                        "sd": sd_d,
                        "resolved_at_2sigma": bool(abs(dd.mean()) > 2.0 * sd_d and sd_d > 0),
                        "e41_addb": -0.013307}

    log("")
    for nm in ARMS:
        r = rows[nm]
        log("  %-6s mean %.6f  sd %.6f  [%.6f .. %.6f]" % (nm, r["mean"], r["sd"], r["min"],
                                                           r["max"]))
    log("")
    log("  VERDICT  PRED mean %.6f (sd %.6f) -> %s   (%+.6f vs COACT mean, %+.6f vs D0C mean, "
        "beats COACT every seed: %s)"
        % (pm, rows["PRED"]["sd"], out["verdict"]["name"], out["verdict"]["vs_coact_mean"],
           out["verdict"]["vs_d0c_mean"], out["verdict"]["beats_coact_every_seed"]))
    log("  SECONDARY  COACT - D0C = %+.6f +- %.6f -> %s   (E41 addendum B read %+.6f)"
        % (out["secondary"]["mean"], sd_d,
           "RESOLVED at 2 sigma" if out["secondary"]["resolved_at_2sigma"]
           else "UNRESOLVED -- E41's ordering stays unresolved, as registered",
           out["secondary"]["e41_addb"]))

    carve.close()
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
