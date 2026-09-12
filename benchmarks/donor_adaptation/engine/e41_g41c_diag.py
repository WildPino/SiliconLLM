#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E41 addendum A -- WHY did G-E41C go VOID at k=16?

G-E41C is E41's planted control: `PERM` is `D0C` with the 256 group IDs relabelled, so the
per-token ORACLE -- which ranks groups by mass and keeps the top k -- must return the IDENTICAL
BPB, bit for bit.  It did at k=3 and it did on the 2-sequence smoke at k=16.  On the registered
24-sequence slice at k=16 it did not: 3.449466 vs 3.448246, a gap of -1.22e-3.

This script does NOT re-run the sweep and does NOT touch e38_oracle_ceiling.py, whose results are
published.  It reproduces the D0C oracle trajectory at k=16 and, at every masked layer, computes
the PERM selection FROM THE SAME ACTIVATIONS, then reports:

  * whether the group-mass vector is bit-identical under relabelling (the accumulation question);
  * how many (layer, token) selections differ, and where the FIRST one is;
  * at each differing row, whether the k-th and (k+1)-th masses are an EXACT tie.

Until the first divergence the two trajectories are identical by construction, so the first row
reported is the true cause of the whole 1.22e-3.

  python e41_g41c_diag.py            # full registered slice
  python e41_g41c_diag.py --seqs 4   # cheaper look
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
from e38_oracle_ceiling import (HF, REV, E_GROUPS, NEV, SEQLEN, SEEDEV,
                                EXPECT_IDS_SHA)     # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e41_g41c_diag.json")
OUT64 = os.path.join(RES, "e41_g41c_diag_f64.json")
PERM_SEED = 41041                           # e41_partition_lever.py, unchanged
K = 16


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seqs", type=int, default=0)
    ap.add_argument("--f64", action="store_true",
                    help="the proposed REPAIR: accumulate group mass in float64")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t0 = time.time()
    acc = torch.float64 if a.f64 else torch.float32

    log("== E41 addendum A: why G-E41C moved at k=16 ==")
    log("  group mass accumulated in %s%s" % (acc, "   <-- PROPOSED REPAIR" if a.f64 else ""))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(HF, revision=REV, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model.eval()
    F = model.config.intermediate_size
    L = model.config.num_hidden_layers

    ids_ev, _, meta_ev = C.get_slice(tok, "heldout", NEV, SEQLEN, SEEDEV)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH -- STOP")
    if a.seqs:
        ids_ev = ids_ev[:a.seqs]
    log("  %d x %d tokens, ids_sha %s" % (ids_ev.shape[0], SEQLEN, EXPECT_IDS_SHA[:16]))

    d0c_np, _ = E19.load_labels(E_GROUPS)
    lab = [torch.from_numpy(d0c_np[li].astype(np.int64)) for li in range(L)]
    rng_p = np.random.default_rng(PERM_SEED)
    pm = [torch.from_numpy(rng_p.permutation(E_GROUPS).astype(np.int64)) for li in range(L)]
    labp = [pm[li][lab[li]] for li in range(L)]

    st = {"mass_bit_identical": True, "mass_max_abs_diff": 0.0, "rows": 0, "diff_rows": 0,
          "exact_tie_at_boundary_rows": 0, "first": None, "per_layer_diff": [0] * L,
          "diff_rows_with_exact_tie": 0}
    state = {"seq": -1}

    def mk(li):
        def f(mod, args):
            h = args[0]
            flat = h.reshape(-1, h.shape[-1])
            src = flat.to(acc) ** 2
            n = flat.shape[0]
            m = torch.zeros(n, E_GROUPS, dtype=acc)
            m.index_add_(1, lab[li], src)
            mp = torch.zeros(n, E_GROUPS, dtype=acc)
            mp.index_add_(1, labp[li], src)
            mp_un = mp[:, pm[li]]                       # column g  <-  mp column pm[g]
            if not torch.equal(mp_un, m):
                st["mass_bit_identical"] = False
                st["mass_max_abs_diff"] = max(st["mass_max_abs_diff"],
                                              float((mp_un - m).abs().max()))
            s1 = m.topk(K, dim=1).indices
            s2 = mp.topk(K, dim=1).indices
            k1 = torch.zeros(n, E_GROUPS, dtype=torch.bool).scatter_(1, s1, True)
            k2 = torch.zeros(n, E_GROUPS, dtype=torch.bool).scatter_(1, s2, True)[:, pm[li]]
            bad = (k1 != k2).any(1)
            nb = int(bad.sum())
            srt = m.sort(dim=1, descending=True).values
            tie = srt[:, K - 1] == srt[:, K]
            st["rows"] += n
            st["exact_tie_at_boundary_rows"] += int(tie.sum())
            if nb:
                st["diff_rows"] += nb
                st["per_layer_diff"][li] += nb
                st["diff_rows_with_exact_tie"] += int((bad & tie).sum())
                if st["first"] is None:
                    r = int(torch.nonzero(bad)[0, 0])
                    top = srt[r, max(0, K - 3):K + 3].tolist()
                    st["first"] = {"seq": state["seq"], "layer": li, "row": r,
                                   "masses_around_boundary": top,
                                   "exact_tie": bool(tie[r]),
                                   "kth": float(srt[r, K - 1]), "kth_plus_1": float(srt[r, K]),
                                   "rel_gap": float((srt[r, K - 1] - srt[r, K])
                                                    / max(1e-30, abs(float(srt[r, K - 1]))))}
            # mask with the D0C selection so the trajectory is D0C's, exactly as the run did
            return (flat.mul(k1[:, lab[li]].to(flat.dtype)).reshape(h.shape),) + tuple(args[1:])
        return f

    handles = [lay.mlp.down_proj.register_forward_pre_hook(mk(li))
               for li, lay in enumerate(model.model.layers)]
    for i in range(ids_ev.shape[0]):
        state["seq"] = i
        model(ids_ev[i:i + 1])
        if st["first"] is not None and st["first"]["seq"] == i:
            log("  first divergence in sequence %d" % i)
    for h in handles:
        h.remove()

    st["seconds"] = time.time() - t0
    log("")
    log("  group mass bit-identical under relabelling: %s  (max abs diff %.3e)"
        % (st["mass_bit_identical"], st["mass_max_abs_diff"]))
    log("  rows (layer x token) examined      : %d" % st["rows"])
    log("  rows whose KEPT SET differs        : %d  (%.3e of rows)"
        % (st["diff_rows"], st["diff_rows"] / max(1, st["rows"])))
    log("  rows with an EXACT tie at boundary : %d" % st["exact_tie_at_boundary_rows"])
    log("  differing rows that ARE exact ties : %d" % st["diff_rows_with_exact_tie"])
    if st["first"]:
        log("  first divergence: seq %d layer %d row %d  exact_tie=%s  rel_gap %.3e"
            % (st["first"]["seq"], st["first"]["layer"], st["first"]["row"],
               st["first"]["exact_tie"], st["first"]["rel_gap"]))
        log("    masses around the boundary: %s" % st["first"]["masses_around_boundary"])
    nz = [(li, c) for li, c in enumerate(st["per_layer_diff"]) if c]
    log("  layers with any divergence: %s" % (nz if nz else "none"))
    st["accum_dtype"] = str(acc)
    op = OUT64 if a.f64 else OUT
    json.dump(st, open(op, "w", encoding="utf-8"), indent=1)
    log("  wrote %s  [%.0fs]" % (op, st["seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
