#!/usr/bin/env python3
"""D1 -- how many weights are actually needed?

Per organ (and per depth) sensitivity to removal, in BPB on the pinned held-out slice.

Two removal modes, and the distinction is the whole point:
  * unstructured  -- zero the smallest |w| inside each (layer, organ) matrix.  The engine
                     CANNOT skip these in bulk; this is the optimistic bound.
  * structured    -- zero whole rows / columns by L2 norm.  This is what a bulk-contiguous
                     >=48 KB read path can actually exploit.  This is the number that counts.

The FFN gets a third, engine-shaped mode: `ffn_neuron`, which removes hidden neuron j from
gate_proj row j, up_proj row j AND down_proj column j together -- the unit the skip path
actually skips.

Every measurement is a paired comparison against the untouched model on the byte-identical
slice, so the harness's own determinism (verified in d_baseline: no-op delta == 0 exactly)
makes the deltas exact; the reported SE is a paired bootstrap over sequences, i.e. how much
of the delta is "which sequences we happened to draw".
"""
from __future__ import annotations

import sys, os, json, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

THREADS = int(os.environ.get("D_THREADS", "10"))
torch.set_num_threads(THREADS)

N_SEQ, SEQ_LEN, SEED = 24, 512, 1234
LEVELS = [0.25, 0.50, 0.75, 0.90, 0.95]
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
DEPTHS = [0, 4, 9, 13, 18, 22, 27]


# ------------------------------------------------------------------ pruning primitives
class Snapshot:
    """Exact save/restore of every weight a probe touches."""
    def __init__(self): self.saved = {}
    def take(self, key, t):
        if key not in self.saved:
            self.saved[key] = t.detach().clone()
    def restore(self, model):
        for (layer, organ), w in self.saved.items():
            C.get_linear(model, layer, organ).weight.copy_(w)
        self.saved = {}


def prune_unstructured(model, layers, organ, frac, snap):
    """Zero the `frac` smallest-|w| entries of each (layer, organ) matrix independently."""
    n_zeroed = n_total = 0
    for L in layers:
        W = C.get_linear(model, L, organ).weight
        snap.take((L, organ), W)
        k = int(round(frac * W.numel()))
        if k > 0:
            thresh = torch.kthvalue(W.abs().flatten(), k).values
            W.masked_fill_(W.abs() <= thresh, 0.0)
        n_zeroed += int((W == 0).sum()); n_total += W.numel()
    return n_zeroed, n_total


def _struct_axis(organ):
    """Which axis the engine can actually drop for this organ.

    out-rows for q/k/v/gate/up (the produced feature disappears);
    in-cols for o_proj/down_proj (the consumed feature disappears).
    """
    return "col" if organ in ("o_proj", "down_proj") else "row"


def prune_structured(model, layers, organ, frac, snap):
    n_zeroed = n_total = 0
    ax = _struct_axis(organ)
    for L in layers:
        W = C.get_linear(model, L, organ).weight     # [out, in]
        snap.take((L, organ), W)
        norms = W.norm(dim=1) if ax == "row" else W.norm(dim=0)
        k = int(round(frac * norms.numel()))
        if k > 0:
            idx = torch.argsort(norms)[:k]
            if ax == "row":
                W[idx, :] = 0.0
            else:
                W[:, idx] = 0.0
        n_zeroed += int((W == 0).sum()); n_total += W.numel()
    return n_zeroed, n_total


def prune_ffn_neuron(model, layers, frac, snap, importance="joint"):
    """Remove whole FFN hidden neurons: gate row j, up row j, down column j, together."""
    n_zeroed = n_total = 0
    for L in layers:
        g = C.get_linear(model, L, "gate_proj").weight
        u = C.get_linear(model, L, "up_proj").weight
        d = C.get_linear(model, L, "down_proj").weight
        for organ, W in (("gate_proj", g), ("up_proj", u), ("down_proj", d)):
            snap.take((L, organ), W)
        score = g.norm(dim=1) * u.norm(dim=1) * d.norm(dim=0)   # joint magnitude
        k = int(round(frac * score.numel()))
        if k > 0:
            idx = torch.argsort(score)[:k]
            g[idx, :] = 0.0; u[idx, :] = 0.0; d[:, idx] = 0.0
        for W in (g, u, d):
            n_zeroed += int((W == 0).sum()); n_total += W.numel()
    return n_zeroed, n_total


# ------------------------------------------------------------------ stats
def paired_se(pa, pb, byts, n_boot=2000, seed=11):
    """Bootstrap SE of the BPB *delta* over sequence resampling (paired)."""
    rng = np.random.default_rng(seed)
    b = byts.numpy()
    na = pa * math.log(2.0) * b
    nb = pb * math.log(2.0) * b
    n = len(b)
    out = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, n, n)
        s = b[k].sum() * math.log(2.0)
        out[i] = nb[k].sum() / s - na[k].sum() / s
    return float(out.std())


# ------------------------------------------------------------------ main
def main():
    m, tok = C.load_model()
    A = C.arch(m)
    ids, byts, meta = C.get_slice(tok, "heldout", N_SEQ, SEQ_LEN, SEED)
    all_layers = list(range(A["n_layers"]))

    t0 = time.time()
    base, base_per = C.bpb(m, ids, byts, return_per_seq=True)
    print(f"baseline BPB {base:.6f}  ({time.time()-t0:.0f}s/eval, threads={THREADS})", flush=True)

    log = {"arch": A, "slice": meta, "baseline_bpb": base, "threads": THREADS,
           "levels": LEVELS, "controls": [], "organ_sweep": [], "depth_sweep": []}
    outp = os.path.join(C.RESULTS, "d1_pruning.json")

    def evaluate(tag, **extra):
        b, per = C.bpb(m, ids, byts, return_per_seq=True)
        se = paired_se(base_per, per, byts)
        rec = {"tag": tag, "bpb": b, "delta": b - base, "paired_se": se, **extra}
        print(f"  {tag:38s} BPB {b:.5f}  d={b-base:+.5f} +-{se:.5f}", flush=True)
        return rec

    # =============================================================== planted controls
    print("== D1 planted controls ==", flush=True)
    snap = Snapshot()

    # (a) 0% prune must be a no-op -> delta EXACTLY 0
    prune_structured(m, all_layers, "down_proj", 0.0, snap)
    r = evaluate("CTRL prune@0pct_downproj", expect="delta == 0")
    r["fired"] = (r["delta"] == 0.0)
    log["controls"].append(r); snap.restore(m)

    # (b) minimal significant corruption: structured 5% on ONE organ, one level below the grid.
    #     If the instrument cannot see the removal of the 5% least-important o_proj columns
    #     across all layers, nothing it reports about 25% means anything.
    nz, nt = prune_structured(m, all_layers, "o_proj", 0.05, snap)
    r = evaluate("CTRL struct@5pct_o_proj", expect="delta > 0", zero_frac=nz / nt)
    r["fired"] = (r["delta"] > 0)
    log["controls"].append(r); snap.restore(m)

    # (c) upper anchor: 100% of one organ must be catastrophic
    nz, nt = prune_structured(m, all_layers, "v_proj", 1.0, snap)
    r = evaluate("CTRL struct@100pct_v_proj", expect="delta >> 0", zero_frac=nz / nt)
    r["fired"] = (r["delta"] > 0.5)
    log["controls"].append(r); snap.restore(m)

    # (d) restore must be exact
    r = evaluate("CTRL restore_exact", expect="delta == 0")
    r["fired"] = (r["delta"] == 0.0)
    log["controls"].append(r)
    C.dump("d1_pruning.json", log)

    # =============================================================== organ sweep
    print("== D1 organ sweep (all 28 layers, one organ at a time) ==", flush=True)
    for organ in ATTN + MLP:
        for mode in ("unstructured", "structured"):
            for f in LEVELS:
                fn = prune_unstructured if mode == "unstructured" else prune_structured
                nz, nt = fn(m, all_layers, organ, f, snap)
                rec = evaluate(f"{organ}/{mode}/{int(f*100)}%",
                               organ=organ, mode=mode, level=f,
                               zero_frac=nz / nt,
                               axis=_struct_axis(organ) if mode == "structured" else "elementwise",
                               params=nt)
                log["organ_sweep"].append(rec); snap.restore(m)
                C.dump("d1_pruning.json", log)

    # engine-shaped joint FFN neuron removal
    for f in LEVELS:
        nz, nt = prune_ffn_neuron(m, all_layers, f, snap)
        rec = evaluate(f"ffn_neuron/structured/{int(f*100)}%",
                       organ="ffn_neuron", mode="structured", level=f,
                       zero_frac=nz / nt, axis="hidden_neuron", params=nt)
        log["organ_sweep"].append(rec); snap.restore(m)
        C.dump("d1_pruning.json", log)

    # =============================================================== depth sweep
    print("== D1 depth sweep (one layer at a time) ==", flush=True)
    for L in DEPTHS:
        for f in (0.50, 0.90):
            nz = nt = 0
            for organ in ATTN:
                a, b_ = prune_structured(m, [L], organ, f, snap); nz += a; nt += b_
            rec = evaluate(f"L{L:02d}/attn/{int(f*100)}%", layer=L, block="attn",
                           level=f, zero_frac=nz / nt)
            log["depth_sweep"].append(rec); snap.restore(m)

            nz, nt = prune_ffn_neuron(m, [L], f, snap)
            rec = evaluate(f"L{L:02d}/mlp/{int(f*100)}%", layer=L, block="mlp",
                           level=f, zero_frac=nz / nt)
            log["depth_sweep"].append(rec); snap.restore(m)
            C.dump("d1_pruning.json", log)

    print("wrote", C.dump("d1_pruning.json", log), flush=True)


if __name__ == "__main__":
    main()
