#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E41 ADDENDUM B -- does the partition ordering depend on the router's calibration budget?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E41_IS_THE_PARTITION_THE_LEVER.md,
ADDENDUM B, pushed at fadeb42 BEFORE this file existed.  Nothing here may contradict it.

The defect it repairs is mine.  BRIEF s2 registered the calibration slice as 8 sequences at seed
424242 (e38_oracle_ceiling's NCAL/SEEDCAL, which E38 used only for its `static` selector), while
the band boundary those numbers are compared to -- E38's D0C fitted 3.597108 -- was measured with
the router npz e37_fit_routers.py wrote, fit on e23_router's own 32 sequences at seed 42424.

Only the ROUTER FIT changes here.  The partitions are rebuilt from per-neuron statistics on the
SAME registered 8-sequence slice, so CONC/STRIPE/COACT are bit-identical to run 1's; the model,
the eval slice, `Carve`, and the eval loop are run 1's, imported.

E36's run-2 rule governs the result: this may not promote the registered verdict.

  python e41_addendum_b.py
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
                                SEEDCAL, PART_B_SEED)                # noqa: E402
from e41_partition_lever import (PERM_SEED, COACT_SEED, PARTS, D0C_FITTED_K16, D0C_ORACLE_K16,
                                 band_of, equal_groups_from_order,
                                 stripe_groups_from_order)           # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e41_addendum_b.json")

K = 16                                      # the verdict k, and the only one B re-runs
SEL = "fitted"                              # the verdict selector, and the only one B re-runs
RCAL, RSEED = E23.NCAL, E23.SEEDCAL         # 32, 42424 -- what E38's boundary was fit at
GD_TOL = 0.001                              # G-E41D


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t0 = time.time()

    log("== E41 ADDENDUM B: the router's calibration budget ==")
    log("  partitions from the REGISTERED %d-seq stats (seed %d); routers refit on %d seqs "
        "(seed %d)" % (NCAL, SEEDCAL, RCAL, RSEED))
    log("  only the %d `%s` cells at k=%d are re-run.  E36's run-2 rule governs." %
        (len(PARTS), SEL, K))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(HF, revision=REV, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model.eval()
    F = model.config.intermediate_size
    L = model.config.num_hidden_layers

    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", NEV, SEQLEN, SEEDEV)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH -- STOP")
    b_tot = float(byts_ev.sum())
    ids_stat, _, _ = C.get_slice(tok, "calib", NCAL, SEQLEN, SEEDCAL)      # partitions: run 1's
    ids_fit, _, _ = C.get_slice(tok, "calib", RCAL, SEQLEN, RSEED)         # routers: E38's
    log("  eval %d x %d (%d bytes); partition stats %d seqs; router fit %d seqs"
        % (ids_ev.shape[0], SEQLEN, int(b_tot), ids_stat.shape[0], ids_fit.shape[0]))

    out = {"brief": "BRIEF_E41 ADDENDUM B (pre-registered at fadeb42)", "donor": HF,
           "revision": REV, "precision": "fp32", "E": E_GROUPS, "k": K, "selector": SEL,
           "partition_stats": [NCAL, SEQLEN, SEEDCAL], "router_fit": [RCAL, SEQLEN, RSEED],
           "d0c_fitted_k16_E38": D0C_FITTED_K16, "d0c_oracle_k16_E38": D0C_ORACLE_K16,
           "chance_bpb": CHANCE,
           "run1_fitted_k16": {"D0C": 3.8043462251479396, "RAND": 4.530483853559613,
                               "CONC": 3.8159112061275415, "STRIPE": 4.41523819389279,
                               "COACT": 3.8870031438980375, "PERM": 3.8043462251479396}}

    dense = bpb_of(model, ids_ev, b_tot)
    out["dense"] = dense
    log("  dense fp32 = %.9f  (E22 base %.6f, diff %+.2e)"
        % (dense, DENSE_ANCHOR, dense - DENSE_ANCHOR))
    if abs(dense - DENSE_ANCHOR) >= ANCHOR_TOL:
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("anchor VOID -- not measuring E22's object. STOP.")

    # ---- partitions, rebuilt exactly as run 1 built them.
    d0c_np, _ = E19.load_labels(E_GROUPS)
    gD0C = [torch.from_numpy(d0c_np[li].astype(np.int64)) for li in range(L)]
    carve = Carve(model, gD0C, None, None, None)

    neuron_mass = [torch.zeros(F, dtype=torch.float64) for _ in range(L)]
    coact = [torch.zeros(F, dtype=torch.float64) for _ in range(L)]
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
    for i in range(ids_stat.shape[0]):
        model(ids_stat[i:i + 1])
    for h in handles:
        h.remove()

    rng_r = np.random.default_rng(PART_B_SEED)
    rng_p = np.random.default_rng(PERM_SEED)
    rng_c = np.random.default_rng(COACT_SEED)
    G = {"D0C": gD0C, "RAND": [], "CONC": [], "STRIPE": [], "COACT": [], "PERM": []}
    for li in range(L):
        G["RAND"].append(torch.from_numpy(equal_groups_from_order(rng_r.permutation(F), F,
                                                                  E_GROUPS)))
        order = np.argsort(-neuron_mass[li].numpy())
        G["CONC"].append(torch.from_numpy(equal_groups_from_order(order, F, E_GROUPS)))
        G["STRIPE"].append(torch.from_numpy(stripe_groups_from_order(order, F, E_GROUPS)))
        key = coact[li].numpy() / max(1.0, float(ids_stat.shape[0]))
        jitter = rng_c.normal(0.0, 1e-12, size=F)
        G["COACT"].append(torch.from_numpy(equal_groups_from_order(
            np.argsort(-(key + jitter)), F, E_GROUPS)))
        pm = torch.from_numpy(rng_p.permutation(E_GROUPS).astype(np.int64))
        G["PERM"].append(pm[gD0C[li]])
    log("  partitions rebuilt; D0C/RAND/PERM are seeded, CONC/STRIPE/COACT come from the same")
    log("  %d-sequence statistics run 1 used, so only the router differs." % NCAL)

    # ---- the only thing that changes: the router's calibration budget.
    routers = {}
    for nm in PARTS:
        t = time.time()
        lab_map = {li: G[nm][li].numpy() for li in range(L)}
        R, _, _ = E23.fit_routers(model, ids_fit, list(range(L)), lab_map)
        routers[nm] = [torch.as_tensor(R[li]).float().T.contiguous() if
                       torch.as_tensor(R[li]).shape[0] == E_GROUPS else
                       torch.as_tensor(R[li]).float().contiguous() for li in range(L)]
        log("  router refit for %-6s on %d seqs [%.0fs]" % (nm, ids_fit.shape[0],
                                                            time.time() - t))

    rows = {}
    log("")
    for nm in PARTS:
        carve.labels = G[nm]
        carve.routers = routers[nm]
        carve.mode, carve.k = SEL, K
        t1 = time.time()
        b = bpb_of(model, ids_ev, b_tot)
        carve.mode = "off"
        r1 = out["run1_fitted_k16"][nm]
        rows[nm] = {"bpb": b, "run1": r1, "delta_vs_run1": b - r1, "vs_dense": b - dense,
                    "vs_chance": b - CHANCE, "seconds": time.time() - t1}
        log("    %-6s %s k=%d  BPB %.6f   (run 1 %.6f, %+.6f)   vs chance %+.4f  [%.0fs]"
            % (nm, SEL, K, b, r1, b - r1, b - CHANCE, time.time() - t1))
    out["cells"] = rows

    # ---- G-E41D: the known positive.  D0C at 32/42424 must reproduce E38's 3.597108.
    d = rows["D0C"]["bpb"] - D0C_FITTED_K16
    gd = {"d0c": rows["D0C"]["bpb"], "e38": D0C_FITTED_K16, "diff": d, "tol": GD_TOL,
          "fires": bool(abs(d) < GD_TOL)}
    out["G_E41D"] = gd
    log("")
    log("  G-E41D  D0C refit at %d/%d = %.6f  vs E38 %.6f  diff %+.6f (tol %.3f) -> %s"
        % (RCAL, RSEED, gd["d0c"], D0C_FITTED_K16, d, GD_TOL,
           "FIRES" if gd["fires"] else "VOID"))
    if not gd["fires"]:
        log("          the calibration budget is NOT what separates the two numbers;")
        log("          this table is reported as uninterpretable, not as a correction.")

    best = min(rows, key=lambda nm: rows[nm]["bpb"])
    order_b = sorted(rows, key=lambda nm: rows[nm]["bpb"])
    order_1 = sorted(rows, key=lambda nm: rows[nm]["run1"])
    out["ordering"] = {"addendum_b": order_b, "run1": order_1,
                       "unchanged": bool(order_b == order_1)}
    out["best"] = {"partition": best, "bpb": rows[best]["bpb"], "band": band_of(rows[best]["bpb"]),
                   "crosses_boundary": bool(rows[best]["bpb"] <= D0C_FITTED_K16)}
    log("")
    log("  ordering run 1      : %s" % " < ".join(order_1))
    log("  ordering addendum B : %s   -> %s" % (" < ".join(order_b),
                                                "UNCHANGED" if out["ordering"]["unchanged"]
                                                else "CHANGED"))
    log("  best is %s at %.6f -> band %s" % (best, rows[best]["bpb"], out["best"]["band"]))
    if out["best"]["crosses_boundary"]:
        log("  CROSSES 3.597108 -- per the registered rule the verdict cell is UNRESOLVABLE and")
        log("  E41 does NOT strengthen the T4 ask.")
    else:
        log("  does NOT cross 3.597108 -- the registered verdict PARTITION-IS-NOT-THE-LEVER")
        log("  stands, and B may not promote it.")

    carve.close()
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
