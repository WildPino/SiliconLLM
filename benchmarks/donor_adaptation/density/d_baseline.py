#!/usr/bin/env python3
"""Baseline + harness planted control for the density probes.

The instrument under test here is the BPB harness itself.  Before any null it produces can
mean anything it must be shown to (a) return *exactly* zero delta under a no-op edit and
(b) FIRE on the smallest edit that could plausibly matter.
"""
from __future__ import annotations

import sys, os, json, copy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

torch.set_num_threads(12)

N_SEQ, SEQ_LEN, SEED = 24, 512, 1234


def main():
    m, tok = C.load_model()
    A = C.arch(m)
    print("arch (read off the artefact):", json.dumps(A), flush=True)

    ids, byts, meta = C.get_slice(tok, "heldout", N_SEQ, SEQ_LEN, SEED)
    print("slice:", json.dumps(meta, indent=1), flush=True)

    with C.Timer("baseline"):
        base, per = C.bpb(m, ids, byts, return_per_seq=True)
    se = C.bootstrap_se(per, byts)
    print(f"BASELINE BPB = {base:.6f}  (slice-resampling SE {se:.4f})", flush=True)

    log = {"arch": A, "slice": meta, "baseline_bpb": base, "slice_se": se,
           "n_threads": torch.get_num_threads(), "controls": []}

    # ---- control 0: no-op edit must give delta EXACTLY 0 (determinism of the harness) ----
    W = C.get_linear(m, 14, "down_proj").weight
    orig = W.detach().clone()
    W.copy_(orig * 1.0)
    b0 = C.bpb(m, ids, byts)
    log["controls"].append({"name": "noop_edit", "expect": "delta == 0",
                            "bpb": b0, "delta": b0 - base,
                            "fired": abs(b0 - base) == 0.0})
    print(f"  control noop_edit: delta = {b0-base:+.9f}  -> {'PASS' if b0==base else 'FAIL'}", flush=True)

    # ---- control 1: minimal significant corruption ----
    # smallest thing that could be wrong: zero the single most-important OUTPUT NEURON of
    # one FFN in one mid layer (1 row of 8960 in 1 layer of 28 = 0.0026% of the weights).
    lin = C.get_linear(m, 14, "down_proj")          # [1536, 8960]
    Wd = lin.weight
    orig_d = Wd.detach().clone()
    col_norm = orig_d.norm(dim=0)
    top = int(col_norm.argmax())
    Wd[:, top] = 0.0
    b1 = C.bpb(m, ids, byts)
    Wd.copy_(orig_d)
    frac = 1.0 / (A["d_ffn"] * A["n_layers"])
    log["controls"].append({
        "name": "kill_top_ffn_neuron_L14", "expect": "delta > 0",
        "neuron": top, "weights_touched_fraction_of_ffn": frac,
        "bpb": b1, "delta": b1 - base, "fired": (b1 - base) > 0})
    print(f"  control kill_top_ffn_neuron(L14,n={top}): delta = {b1-base:+.6f} "
          f"({'FIRED' if b1>base else 'DID NOT FIRE'})", flush=True)

    # ---- control 2: a whole organ zeroed must be catastrophic (upper anchor) ----
    lin = C.get_linear(m, 14, "down_proj")
    orig_d = lin.weight.detach().clone()
    lin.weight.zero_()
    b2 = C.bpb(m, ids, byts)
    lin.weight.copy_(orig_d)
    log["controls"].append({"name": "zero_down_proj_L14", "expect": "delta >> 0",
                            "bpb": b2, "delta": b2 - base, "fired": (b2 - base) > 0.01})
    print(f"  control zero_down_proj_L14: delta = {b2-base:+.6f}", flush=True)

    # ---- control 3: restore must return EXACTLY to baseline ----
    b3 = C.bpb(m, ids, byts)
    log["controls"].append({"name": "restore_exact", "expect": "delta == 0",
                            "bpb": b3, "delta": b3 - base, "fired": b3 == base})
    print(f"  control restore_exact: delta = {b3-base:+.9f} -> {'PASS' if b3==base else 'FAIL'}",
          flush=True)

    print(C.dump("d_baseline.json", log))


if __name__ == "__main__":
    main()
