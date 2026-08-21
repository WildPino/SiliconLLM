#!/usr/bin/env python3
"""D1b -- complete the D1 organ sweep, resuming interrupted work.

`d1_pruning.py` was interrupted partway through the v_proj/structured row (29 of the full grid
recorded, `organ_sweep` stops at v_proj/structured/90%; o_proj/gate_proj/up_proj/down_proj were
never touched).  This script does NOT re-run or duplicate what already succeeded; it REUSES
`d1_pruning.py`'s own primitives unmodified (`prune_unstructured`, `prune_structured`, `Snapshot`,
`_struct_axis`, `paired_se`) and adds exactly one new primitive the task requires that the
original script did not have: `prune_block_structured`, which zeroes whole CONTIGUOUS blocks of
rows/columns at the engine-legal 48 KB granularity (`d0_layout.json`'s own block-size formula,
reused from D0 -- not re-derived here), rather than cherry-picking the globally-smallest-norm
rows/columns from anywhere in the matrix. That is the difference between "structured" (existing,
free selection -- an optimistic bound on what reordering+compaction could achieve) and
"block_structured" (new, fixed contiguous grouping -- what the engine's block-skip path can
actually exploit without any reordering).

PRE-REGISTERED before any sweep number in this session is looked at:
  - noise band: a delta is INCONCLUSIVE iff |delta| < 2 * paired_bootstrap_SE (n_boot=2000,
    the same bootstrap already used throughout D1). Anything outside that band is reported as a
    real (positive) effect; nothing in this sweep is expected to be negative (BPB should not
    improve from deleting weights) so no separate negative-effect criterion is defined.
  - the four planted controls from `d1_pruning.py` are re-run FIRST, in this session, on this
    process. If any does not fire, the script STOPS before running a single sweep point.
"""
from __future__ import annotations

import sys, os, json, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import d1_pruning as D1          # REUSE: prune_unstructured, prune_structured, Snapshot, paired_se, _struct_axis

torch.set_num_threads(D1.THREADS)

BYTES_PER_WEIGHT = 0.5            # engine design point -- same constant as D0 and d1_pruning
BLOCK_BYTES = 49152
LEVELS_B = [0.25, 0.50, 0.75, 0.90]     # exactly the grid the task specifies
NOISE_BAND_SIGMA = 2.0            # PRE-REGISTERED acceptance criterion, fixed before any number below is read
ATTN = D1.ATTN
MLP = D1.MLP
ALL_ORGANS = list(ATTN) + list(MLP)
OUTP = os.path.join(C.RESULTS, "d1_pruning.json")


# ------------------------------------------------------------------ new primitive: block-structured
def block_size_for(model, organ) -> tuple[int, float]:
    """Engine-legal contiguous block size for this organ's structured axis, reusing D0's own
    formula (units_per_48KB = 49152 / (unit_length_bytes)), read off the ACTUAL loaded weight
    shape rather than assumed."""
    W = C.get_linear(model, 0, organ).weight
    ax = D1._struct_axis(organ)
    unit_len = W.shape[1] if ax == "row" else W.shape[0]     # row: in_features; col: out_features
    bytes_per_unit = unit_len * BYTES_PER_WEIGHT
    bs = BLOCK_BYTES / bytes_per_unit
    if abs(bs - round(bs)) > 1e-6:
        raise ValueError(f"{organ}: non-integer engine block size ({bs}) -- cannot block-prune "
                          f"cleanly; this must be reported, not rounded silently")
    return int(round(bs)), bytes_per_unit


def prune_block_structured(model, layers, organ, frac, snap, block_size):
    """Zero whole CONTIGUOUS blocks of `block_size` rows/cols, chosen by lowest block Frobenius
    norm, WITHOUT permuting the matrix first. This is the engine-legal granularity: no reordering
    is assumed, only fixed-position block skip."""
    ax = D1._struct_axis(organ)
    n_zeroed = n_total = 0
    nblocks_seen = None
    for L in layers:
        W = C.get_linear(model, L, organ).weight
        snap.take((L, organ), W)
        n_units = W.shape[0] if ax == "row" else W.shape[1]
        nblocks = n_units // block_size
        if nblocks * block_size != n_units:
            raise ValueError(f"L{L} {organ}: block_size={block_size} does not evenly divide "
                              f"n_units={n_units} -- cannot block-prune cleanly")
        if ax == "row":
            Wb = W.view(nblocks, block_size, W.shape[1])
            block_norm = Wb.norm(dim=(1, 2))
        else:
            Wb = W.view(W.shape[0], nblocks, block_size).permute(1, 0, 2)
            block_norm = Wb.norm(dim=(1, 2))
        k = int(round(frac * nblocks))
        if k > 0:
            idx = torch.argsort(block_norm)[:k].tolist()
            if ax == "row":
                for b in idx:
                    W[b * block_size:(b + 1) * block_size, :] = 0.0
            else:
                for b in idx:
                    W[:, b * block_size:(b + 1) * block_size] = 0.0
        n_zeroed += int((W == 0).sum()); n_total += W.numel()
        nblocks_seen = nblocks
    return n_zeroed, n_total, nblocks_seen


# ------------------------------------------------------------------ main
def main():
    if not os.path.exists(OUTP):
        raise SystemExit(f"expected existing {OUTP} to append to -- not found")
    log = json.load(open(OUTP))
    existing_tags = {r["tag"] for r in log.get("organ_sweep", [])}
    print(f"loaded existing d1_pruning.json: baseline_bpb={log['baseline_bpb']!r}, "
          f"{len(log.get('organ_sweep', []))} organ_sweep records, "
          f"{len(log.get('controls', []))} controls", flush=True)

    import subprocess
    session2 = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO).decode().strip(),
        "git_branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=C.REPO).decode().strip(),
        "command_line": "python " + " ".join(sys.argv),
        "threads": D1.THREADS,
        "levels_this_session": LEVELS_B,
        "noise_band_sigma": NOISE_BAND_SIGMA,
        "preregistration": "INCONCLUSIVE iff |delta| < 2*paired_bootstrap_SE (n_boot=2000). "
                            "Written before any sweep number in this session was computed.",
        "block_structured_mode_definition": "whole contiguous blocks of block_size rows/cols "
            "(fixed position, no reordering), zeroed by lowest block Frobenius norm; block_size "
            "from D0's own 48KB-granularity formula applied to the ACTUAL loaded weight shape.",
        "new_controls_rerun": [],
        "new_organ_sweep": [],
    }

    m, tok = D1.C.load_model()
    A = D1.C.arch(m)
    assert A == log["arch"], f"arch mismatch vs existing artefact: {A} != {log['arch']}"
    ids, byts, meta = C.get_slice(tok, "heldout", D1.N_SEQ, D1.SEQ_LEN, D1.SEED)
    assert meta["ids_sha256"] == log["slice"]["ids_sha256"], "slice mismatch vs existing artefact"
    all_layers = list(range(A["n_layers"]))

    t0 = time.time()
    base, base_per = C.bpb(m, ids, byts, return_per_seq=True)
    print(f"baseline BPB (this session) {base:.6f}  vs artefact {log['baseline_bpb']:.6f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    assert base == log["baseline_bpb"], "baseline BPB does not reproduce bit-exactly this session"
    session2["baseline_bpb_reproduced"] = base

    def evaluate(tag, **extra):
        b, per = C.bpb(m, ids, byts, return_per_seq=True)
        se = D1.paired_se(base_per, per, byts)
        delta = b - base
        rec = {"tag": tag, "bpb": b, "delta": delta, "paired_se": se, **extra}
        band = NOISE_BAND_SIGMA * se
        rec["verdict"] = "INCONCLUSIVE" if abs(delta) < band else "SIGNIFICANT"
        print(f"  {tag:42s} BPB {b:.5f}  d={delta:+.5f} +-{se:.5f}  [{rec['verdict']}]", flush=True)
        return rec

    # =============================================================== re-run the 4 planted controls
    print("== D1b: re-running the 4 planted controls in THIS session ==", flush=True)
    snap = D1.Snapshot()

    D1.prune_structured(m, all_layers, "down_proj", 0.0, snap)
    r = evaluate("CTRL prune@0pct_downproj", expect="delta == 0")
    r["fired"] = (r["delta"] == 0.0)
    session2["new_controls_rerun"].append(r); snap.restore(m)

    nz, nt = D1.prune_structured(m, all_layers, "o_proj", 0.05, snap)
    r = evaluate("CTRL struct@5pct_o_proj", expect="delta > 0", zero_frac=nz / nt)
    r["fired"] = (r["delta"] > 0)
    session2["new_controls_rerun"].append(r); snap.restore(m)

    nz, nt = D1.prune_structured(m, all_layers, "v_proj", 1.0, snap)
    r = evaluate("CTRL struct@100pct_v_proj", expect="delta >> 0", zero_frac=nz / nt)
    r["fired"] = (r["delta"] > 0.5)
    session2["new_controls_rerun"].append(r); snap.restore(m)

    r = evaluate("CTRL restore_exact", expect="delta == 0")
    r["fired"] = (r["delta"] == 0.0)
    session2["new_controls_rerun"].append(r)

    all_fired = all(c["fired"] for c in session2["new_controls_rerun"])
    print(f"controls this session: {'ALL FIRED' if all_fired else 'FAILURE'}", flush=True)
    log["session2_partial"] = session2
    C.dump("d1_pruning.json", log)
    if not all_fired:
        raise SystemExit("STOP: a planted control did not fire on re-run. See "
                          "log['session2_partial']['new_controls_rerun'] in d1_pruning.json. "
                          "Not running any sweep points.")

    # =============================================================== block sizes (per organ)
    print("\n== D1b: engine-legal block sizes (reused D0 formula, read off loaded weights) ==",
          flush=True)
    block_sizes = {}
    for organ in ALL_ORGANS:
        bs, bpu = block_size_for(m, organ)
        block_sizes[organ] = {"block_size_units": bs, "bytes_per_unit": bpu}
        print(f"  {organ:10s} bytes_per_unit={bpu:6.0f}B  block_size={bs} units", flush=True)
    session2["block_sizes"] = block_sizes

    # =============================================================== organ sweep completion
    print("\n== D1b: organ sweep completion (unstructured + block_structured, "
          f"levels={LEVELS_B}) ==", flush=True)
    for organ in ALL_ORGANS:
        bs = block_sizes[organ]["block_size_units"]
        # q_proj/k_proj/v_proj already have unstructured at these levels from session 1 --
        # only block_structured is new for them. All other organs need both modes.
        if organ in ("q_proj", "k_proj", "v_proj"):
            modes = ["block_structured"]
        else:
            modes = ["unstructured", "block_structured"]

        for mode in modes:
            for f in LEVELS_B:
                tag = f"{organ}/{mode}/{int(f*100)}%"
                if tag in existing_tags:
                    print(f"  SKIP {tag} (already present)", flush=True)
                    continue
                if mode == "unstructured":
                    nz, nt = D1.prune_unstructured(m, all_layers, organ, f, snap)
                    rec = evaluate(tag, organ=organ, mode=mode, level=f, zero_frac=nz / nt,
                                    axis="elementwise", params=nt)
                else:
                    nz, nt, nblocks = prune_block_structured(m, all_layers, organ, f, snap, bs)
                    rec = evaluate(tag, organ=organ, mode=mode, level=f, zero_frac=nz / nt,
                                    axis=D1._struct_axis(organ), params=nt,
                                    block_size_units=bs, n_blocks_per_layer=nblocks)
                log["organ_sweep"].append(rec)
                session2["new_organ_sweep"].append(rec)
                snap.restore(m)
                log["session2_partial"] = session2
                C.dump("d1_pruning.json", log)

    log["session2"] = session2
    del log["session2_partial"]
    C.dump("d1_pruning.json", log)
    print("\nwrote", C.dump("d1_pruning.json", log), flush=True)
    print(f"total new records this session: {len(session2['new_organ_sweep'])} sweep + "
          f"{len(session2['new_controls_rerun'])} controls", flush=True)


if __name__ == "__main__":
    main()
