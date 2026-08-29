#!/usr/bin/env python3
"""D0 -- can a STATIC permutation make a DYNAMIC activation pattern contiguous?

Implements `docs/research/donor_adaptation/briefs/BRIEF_D0_COACTIVATION_PERMUTATION.md`
INCLUDING AMENDMENT 1, which reframes the probe.

A1.1 proves that no block-skippable fraction reaches the speed budget: with a dense gate

        FFN_active = (1.0 + (1-s) + (1-s)) / 3 = (3 - 2s)/3

so `FFN_active = 0.02` needs `s = 1.470` and `s <= 1`.  The gate is 33.3% and is never skipped,
because the gate IS the predictor -- you cannot know which entries of h are zero before
computing W_gate x in full.  The question D0 actually settles is therefore:

    does this donor's activation pattern contain cluster structure strong enough to carve the
    FFN into experts -- i.e. is training-free MoEfication viable on this donor?

Every skippable fraction `s` printed by this probe is printed beside `FFN_active = (3-2s)/3`;
the MoE geometry implied by each arm is reported per layer; duplication is reported as a
WEIGHT-INFLATION MULTIPLIER, not only as a count.

Scope note carried from the coordinator (2026-08-28): a separate probe (F1, gate sign
predictor) is attacking the 33.3% gate term itself, on the observation that the floor is a
property of how our engine computes the predictor rather than a law.  That is orthogonal to
clustering, but it means a NULL from D0 does not close the FFN half of the programme and this
probe must not be written as if it did.

--------------------------------------------------------------------------------------------
THE CRUX, AND WHY THIS PROBE HAS A FIT/SCORE SPLIT

Brief §2: "A permutation is static.  An activation pattern is per-token.  We get to choose ONE
neuron ordering per layer and it must serve EVERY token."  An ordering fitted and then scored on
the SAME tokens cannot distinguish structure from memorisation of the calibration draw.  So:

    the co-activation ordering is FITTED on calibration tokens (calib half)
    and SCORED on disjoint tokens (held-out half),

with the in-sample number reported alongside so the generalisation gap is visible rather than
assumed absent.  This is stronger than the brief requires and is the only honest way to answer
a question whose whole content is "does one static choice serve unseen tokens".

--------------------------------------------------------------------------------------------
WHAT THE RANDOM-PERMUTATION ARM DOES AND DOES NOT DESTROY  (control C4 measures all of this)

Two shuffle-based nulls on this programme turned out not to be null; the most recent recovered
BETTER than the real arm, because shuffling destroyed a structural constraint and made the
problem easier.  So, stated before the arm is trusted, and then MEASURED:

  DESTROYED  -- and only this: the alignment between neuron INDEX and co-activation.  Adjacency
                in the layout carries no information after the shuffle.
  PRESERVED exactly:
      * the per-token active count (a permutation of a row's entries),
      * the per-neuron firing-frequency distribution (relabelled, not changed),
      * the ENTIRE co-activation matrix up to relabelling -- P^T C P is similar to C, so its
        spectrum, its rank and its clusterability are bit-identical.
  CONSEQUENCE: the random arm cannot make the problem easier, because the quantity it changes
  (index alignment) is the only quantity the score depends on, and its expected value under a
  uniform permutation is the exact hypergeometric C(N-k, B)/C(N, B).  A deviation of the random
  arm from that closed form is an INSTRUMENT FAULT, not a finding.  C4 checks the invariances
  exactly and C3 checks the closed form.

--------------------------------------------------------------------------------------------
STAGES (separate processes; each writes its own JSON fragment, so a crash never loses another)

    controls   no donor.               peak ~0.8 GB.   synthetic planted +/- controls, C1..C6.
    model      donor resident, fp32.   peak ~9.6 GB.   losslessness control FIRST, then capture,
                                                       then oracle-routed FIDELITY.
    analyse    no donor.               peak ~2.6 GB.   arms, tables, MoE geometry, duplication.

Machine note: the ~10 GB ceiling under which this was first designed has lifted (the 45.7 GB
llama-server is gone; ~63 GB free).  The design was NOT inflated to fill it -- the extra
headroom bought two things that buy science, not comfort: capture of ALL layers instead of 5
(free in compute, since the hook sits on `down_proj`'s input which is computed anyway), and the
fit/score token split above.  Peak is stated per stage and the allocation inventory is emitted
into the JSON so the write-up cannot drift from the code.  With DEEP_LAYERS-only capture
(`D0_ALL_LAYERS=0`) the model stage falls back under 8 GB.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import d0_layout as D0          # REUSE, unmodified: layout_stats / clustered_order /
                                # balanced_labels / routed_error / run_lengths.  The
                                # equivalence of this file's fast index-based block accounting
                                # against D0.layout_stats is itself a fired control (C5), so
                                # the memory-lean path inherits D0's own planted controls.
import d1_pruning as D1         # IMPORT the eval-slice constants, never retype them

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)

# ----------------------------------------------------------------------------- pinned config
# Same donor / same eval slice / same BPB path as D1 and D4 (brief §8).
N_EVAL, SEQ_LEN_EVAL, SEED_EVAL = D1.N_SEQ, D1.SEQ_LEN, D1.SEED          # 24 x 512, seed 1234

# FIT tokens: calib half (D4's slice).  SCORE tokens: held-out half, an independent draw.
# T = 16384 > d_ffn = 8960 with ~1.8x margin -- D4 established that T < d_ffn makes the
# activation second moment rank-deficient; the same margin is honoured here.
N_FIT, SEQ_LEN_FIT, SEED_FIT = 32, 512, 42424
N_SCORE, SEQ_LEN_SCORE, SEED_SCORE = 32, 512, 909001

ALL_LAYERS = os.environ.get("D0_ALL_LAYERS", "1") == "1"
DEEP_LAYERS = [1, 7, 14, 21, 27]           # full arm/MoE/duplication battery
P_MAX = 0.25                               # top-25% ranked indices stored; any p <= this derivable
DENSITIES = [0.05, 0.10, 0.20]             # requested; ACHIEVED = round(p*d_ffn)/d_ffn, reported
P_PRIMARY = 0.10                           # matches the D0b structureless reference (0.9^B)

RANDOM_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]     # brief §7.4 asks >= 5; 8 used
CLUSTER_SEED = 7
LOSSLESS_PERM_SEED = 20260828
E_LIST = [16, 32, 64, 128]                 # MoE expert counts for the A1.3.2 geometry
E_DEPTH = 64                               # the single E used for the all-layer depth sweep
K_LIST = [1, 2, 4, 8, 16]
COVER_TARGETS = [0.90, 0.95, 0.99, 1.00]
FIDELITY_E = [32, 64, 128]
FIDELITY_K = [1, 2, 4, 8]
FIDELITY_TOKENS = 2048                     # oracle-routed relative-L2 error is O(T) FFN forwards
# Fidelity uses the SIZE-BALANCED partitions only.  d0_layout's own comment says why: with
# unbalanced k-means clusters, top-k over E is NOT an activation fraction of k/E, so the
# balanced partition and its matched random NULL are the only pair whose k/E means the same
# thing on both arms.  The unbalanced k-means partition is covered in the geometry section,
# where the true byte fraction is measured rather than assumed.
FIDELITY_PARTITIONS = ("coactivation_balanced", "random_balanced_NULL")

DUP_TOKENS = 2048                          # token subsample for the set-cover duplication calc
BYTES_PER_WEIGHT = D0.BYTES_PER_WEIGHT     # 0.5 B/weight, engine design point, pulled from D0
BLOCK_BYTES = 49152                        # 48 KiB, the rho-law granularity
assert BYTES_PER_WEIGHT == 0.5

MASKDIR = os.path.join(C.RESULTS, "d0_masks")


def FRAG(name):
    return os.path.join(C.RESULTS, f"d0_coactivation.{name}.json")


# ============================================================================ manifest / io
def _resident_heavyweights(top=8):
    """Brief §8: record resident heavyweight processes.  D0 is NOT a timing probe -- it measures
    activation statistics, so contention makes it slower, not wrong.  There is deliberately NO
    quiescence GATE: gating would block the probe for no methodological reason.  The record
    costs nothing and this project has a banked case of unreproducible numbers from an
    unrecorded environment."""
    try:
        ps = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First "
             f"{top} Name,Id,@{{n='WS_GB';e={{[math]::Round($_.WorkingSet64/1GB,2)}}}} "
             "| ConvertTo-Json -Compress"],
            stderr=subprocess.DEVNULL, timeout=60).decode()
        return json.loads(ps)
    except Exception as e:                                   # never let the manifest fail a run
        return {"error": repr(e)}


def _mem_gb():
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "$o=Get-CimInstance Win32_OperatingSystem; "
             "ConvertTo-Json -Compress @{total_gb=[math]::Round($o.TotalVisibleMemorySize/1MB,2);"
             "free_gb=[math]::Round($o.FreePhysicalMemory/1MB,2)}"],
            stderr=subprocess.DEVNULL, timeout=60).decode()
        return json.loads(out)
    except Exception as e:
        return {"error": repr(e)}


def _rss_gb():
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        # GetCurrentProcess returns a 64-bit pseudo-handle.  ctypes defaults restype to
        # c_int, which TRUNCATES it -- the call then fails and silently reports 0.0 GB.
        # Caught because the model stage printed rss_gb 0.0 with 6 GB of weights resident.
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wt.DWORD]
        fn.restype = wt.BOOL
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return {"error": "GetProcessMemoryInfo failed"}
        return {"rss_gb": round(c.WorkingSetSize / 2**30, 3),
                "peak_rss_gb": round(c.PeakWorkingSetSize / 2**30, 3)}
    except Exception as e:
        return {"error": repr(e)}


def env_manifest(stage: str) -> dict:
    import transformers
    return {
        "stage": stage,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=C.REPO).decode().strip(),
        "git_branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                              cwd=C.REPO).decode().strip(),
        "git_dirty_at_launch": bool(subprocess.check_output(["git", "status", "--porcelain"],
                                                            cwd=C.REPO).decode().strip()),
        "command_line": "python " + " ".join(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__, "numpy": np.__version__,
        "transformers": transformers.__version__,
        "torch_num_threads_achieved": torch.get_num_threads(),
        "env_D_THREADS": THREADS,
        "env_D0_ALL_LAYERS": ALL_LAYERS,
        "memory_at_stage_start": _mem_gb(),
        "resident_heavyweight_processes": _resident_heavyweights(),
        "quiescence_gate": "NONE BY DESIGN -- D0 measures activation statistics, not bandwidth.",
        "pinned": {
            "donor": C.MODEL_ID, "revision": C.REVISION,
            "bpb_eval_slice": {"part": "heldout", "n_seq": N_EVAL, "seq_len": SEQ_LEN_EVAL,
                               "seed": SEED_EVAL, "source": "imported from d1_pruning.py"},
            "fit_slice": {"part": "calib", "n_seq": N_FIT, "seq_len": SEQ_LEN_FIT,
                          "seed": SEED_FIT},
            "score_slice": {"part": "heldout", "n_seq": N_SCORE, "seq_len": SEQ_LEN_SCORE,
                            "seed": SEED_SCORE},
            "deep_layers": DEEP_LAYERS, "depth_sweep_E": E_DEPTH,
            "p_max_stored": P_MAX, "densities_requested": DENSITIES, "p_primary": P_PRIMARY,
            "random_seeds": RANDOM_SEEDS, "cluster_seed": CLUSTER_SEED,
            "lossless_perm_seed": LOSSLESS_PERM_SEED,
            "E_list": E_LIST, "k_list": K_LIST, "dup_tokens": DUP_TOKENS,
            "fidelity_tokens": FIDELITY_TOKENS,
            "bytes_per_weight": BYTES_PER_WEIGHT, "block_bytes": BLOCK_BYTES,
        },
    }


def dump_frag(name, obj):
    obj["_mem_at_write"] = _rss_gb()
    p = FRAG(name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return p


# ============================================================================ the A1.1 identity
def ffn_active_dense_gate(s: float) -> float:
    """AMENDMENT 1 §A1.1.  This function exists so that no table in this probe can print `s`
    without printing the number that actually moves the budget.  A row reading `s = 0.66` looks
    like a success; the same row reads `56.2% active` and plainly is not one."""
    return (3.0 - 2.0 * s) / 3.0


def rho_floor_blocks(d_model: float) -> dict:
    """Engine-legal block size in NEURONS, from the D0b formula -- never hardcoded.
    bytes_per_neuron is d_model-driven in every organ (gate/up ROW length and down COLUMN
    length are both d_model); d_ffn cancels out of the formula entirely."""
    bpn_organ = d_model * BYTES_PER_WEIGHT
    bpn_inter = 3.0 * d_model * BYTES_PER_WEIGHT
    return {"d_model": d_model,
            "bytes_per_neuron_per_organ": bpn_organ,
            "bytes_per_neuron_interleaved": bpn_inter,
            "neurons_per_48KB_per_organ": BLOCK_BYTES / bpn_organ,
            "neurons_per_48KB_interleaved": BLOCK_BYTES / bpn_inter}


def block_sizes_for(d_model: float) -> list:
    """Sweep AROUND the engine-legal floor at this donor's width, and ALSO at the 70B-class
    floor (12 per-organ / 4 interleaved) so D0b's structureless reference is comparable at the
    block size it was quoted for.  All derived; none hardcoded to a donor."""
    cand = set()
    for f in (rho_floor_blocks(d_model), rho_floor_blocks(8192)):
        for key in ("neurons_per_48KB_per_organ", "neurons_per_48KB_interleaved"):
            b = int(round(f[key]))
            cand.update([b, max(2, b // 2), b * 2])
    cand.update([4, 8, 16, 32, 64, 128, 140])       # 140 = d_ffn/64, the E_DEPTH expert size
    return sorted(x for x in cand if x >= 2)


# ============================================================================ fast block accounting
def positions_of_active(idx: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """`pos[idx]` as int32, computed ONCE per (arm, density) and reused across every block size.

    This is the single largest transient in the analyse stage and it does NOT depend on the
    block size: recomputing it inside the block loop would allocate a 235 MB int64 temporary
    fourteen times per arm for nothing.  int32 is safe -- d_ffn = 8960 << 2^31.
    """
    return pos[idx].astype(np.int32)


def block_skippable_from_posidx(posidx: np.ndarray, N: int, bs: int) -> float:
    """Block-skippable fraction for active neurons whose LAYOUT POSITIONS are `posidx` [T, k].

    Definition IDENTICAL to D0.layout_stats: the layout is truncated to nb*bs columns (the
    remainder dropped, exactly as `B[:, :nb*bs].reshape(...)` does), a block is FETCHED if any
    of its neurons is active, skippable = 1 - mean fraction of blocks fetched.  C5 asserts the
    two agree to <1e-12.

    Never materialises the [T, N] mask -- that is the O(input) blow-up this probe avoids.
    """
    nb = N // bs
    keep_hi = nb * bs
    T = posidx.shape[0]
    b = posidx // bs                                     # [T, k] int32
    touched = np.zeros((T, nb), dtype=bool)
    rows = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None], b.shape)
    if keep_hi == N:
        touched[rows.ravel(), b.ravel()] = True
    else:
        valid = (posidx < keep_hi).ravel()
        touched[rows.ravel()[valid], b.ravel()[valid]] = True
    return float(1.0 - touched.mean())


def block_skippable_from_idx(idx: np.ndarray, pos: np.ndarray, N: int, bs: int) -> float:
    """Convenience wrapper for one-off calls (the controls); the hot path uses the two
    functions above so `pos[idx]` is paid for once."""
    return block_skippable_from_posidx(positions_of_active(idx, pos), N, bs)


def dense_mask_from_idx(idx: np.ndarray, N: int) -> np.ndarray:
    """Materialise the [T, N] bool mask.  Used ONLY where D0's reused instruments demand a dense
    mask (C5's equivalence check, clustered_order, balanced_labels).  Charged in the inventory."""
    T = idx.shape[0]
    B = np.zeros((T, N), dtype=bool)
    B[np.arange(T, dtype=np.int64)[:, None], idx] = True
    return B


def analytic_iid_skippable(p: float, bs: int) -> float:
    """The structureless reference: for an i.i.d. mask at density p a block of bs neurons is
    skippable iff all bs are inactive -> (1-p)^bs.  D0b's quoted 0.2824 at block 12 and 0.6563
    at block 4 are exactly the p = 0.10 instances of this same closed form (0.9^12, 0.9^4)."""
    return (1.0 - p) ** bs


def hypergeom_skippable(k_active: int, N: int, bs: int) -> float:
    """Exact reference for a RANDOM permutation of a mask with EXACTLY k active per token: a
    block is skippable iff none of its bs slots drew an active neuron, i.e.
    C(N-k, bs)/C(N, bs) = prod_{i<bs} (N-k-i)/(N-i).  This is the number the `random` arm MUST
    land on; a deviation is an instrument fault, not a finding."""
    logp = 0.0
    for i in range(bs):
        if N - k_active - i <= 0:
            return 0.0
        logp += math.log(N - k_active - i) - math.log(N - i)
    return math.exp(logp)


# ============================================================================ orderings (arms)
def order_identity(N: int) -> np.ndarray:
    return np.arange(N, dtype=np.int64)


def order_random(N: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(N).astype(np.int64)


def order_coactivation(B: np.ndarray, E: int, seed: int):
    """REUSED unmodified from d0_layout.clustered_order: centre the [T,N] mask, take a rank-64
    randomized SVD, row-normalise the right singular vectors, k-means into E clusters, then lay
    the neurons out cluster-by-cluster.  Returns (order, labels)."""
    order, lab = D0.clustered_order(torch.from_numpy(B), E, seed)
    return np.asarray(order, dtype=np.int64), np.asarray(lab).astype(np.int64)


def pos_from_order(order: np.ndarray) -> np.ndarray:
    """order[j] = which neuron sits at position j  ->  pos[n] = position of neuron n."""
    pos = np.empty_like(order)
    pos[order] = np.arange(len(order), dtype=order.dtype)
    return pos


def counts_per_expert(idx: np.ndarray, labels: np.ndarray, E: int) -> np.ndarray:
    """[T, E] int32 count of active neurons per expert.  bincount on a flattened key --
    np.add.at is ~50x slower at this size and this is the inner loop of both the geometry and
    the duplication measurement."""
    T, k = idx.shape
    lab_t = labels[idx].astype(np.int64)
    key = (np.arange(T, dtype=np.int64)[:, None] * E + lab_t).ravel()
    return np.bincount(key, minlength=T * E).astype(np.int32).reshape(T, E)


# ============================================================================ STAGE: controls
def _planted_coactivation_mask(N, T, n_groups, groups_per_token, seed):
    """Synthetic pattern with a KNOWN contiguous group structure, then scrambled by a KNOWN
    permutation.  Groups are equal-sized and disjoint; a token activates `groups_per_token`
    WHOLE groups.  Under the planted (unscrambled) order every group is contiguous, so at block
    size = group size the skippable fraction is EXACTLY 1 - groups_per_token/n_groups."""
    rng = np.random.default_rng(seed)
    g = N // n_groups
    N_eff = g * n_groups
    planted_label = np.repeat(np.arange(n_groups), g)                     # in PLANTED order
    idx = np.empty((T, groups_per_token * g), dtype=np.int64)
    base = [np.arange(x * g, (x + 1) * g) for x in range(n_groups)]
    for t in range(T):
        gs = rng.choice(n_groups, groups_per_token, replace=False)
        idx[t] = np.concatenate([base[x] for x in gs])
    sigma = rng.permutation(N_eff)              # sigma[j] = planted-order slot at scrambled j
    inv = pos_from_order(sigma)                 # inv[planted slot] = scrambled index
    idx_scr = inv[idx]                          # active indices in the SCRAMBLED numbering
    lab_scr = planted_label[sigma]              # planted label of each scrambled neuron
    return idx_scr, lab_scr, N_eff, g, groups_per_token / n_groups


def stage_controls():
    out = {"env": env_manifest("controls")}
    print("== D0 stage `controls` -- synthetic, no donor loaded ==", flush=True)
    controls = []
    from sklearn.metrics import adjusted_rand_score

    # ---- C2 planted-POSITIVE: cluster structure that is definitely there ------------------
    # "An instrument that cannot find structure that is definitely there cannot be believed
    #  when it reports none."  This one must be SHOWN firing before any null below counts.
    N, T, G, GPT = 8960, 4096, 64, 6
    idx, lab_true, N_eff, gsize, frac = _planted_coactivation_mask(N, T, G, GPT, seed=4242)
    k_ach = idx.shape[1]
    B = dense_mask_from_idx(idx, N_eff)
    s_id = block_skippable_from_idx(idx, order_identity(N_eff), N_eff, gsize)
    order_co, lab_co = order_coactivation(B, G, CLUSTER_SEED)
    s_co = block_skippable_from_idx(idx, pos_from_order(order_co), N_eff, gsize)
    s_planted = 1.0 - frac                      # what a perfect recovery must report
    ari = float(adjusted_rand_score(lab_true, lab_co))
    del B
    controls.append({
        "name": "C2_planted_positive_coactivation_groups",
        "construction": f"N={N_eff}, T={T}, {G} disjoint contiguous groups of {gsize} neurons, "
                        f"{GPT} WHOLE groups active per token, then scrambled by a known "
                        f"permutation (seed 4242)",
        "density_achieved": k_ach / N_eff, "k_active_achieved": int(k_ach),
        "block_size": gsize,
        "skippable_planted_exact": s_planted,
        "skippable_identity_on_scrambled": s_id,
        "skippable_coactivation_recovered": s_co,
        "recovery_ratio": s_co / s_planted,
        "cluster_ARI_vs_planted_labels": ari,
        "expect": "co-activation arm recovers >=90% of the planted skippable AND ARI>0.90, "
                  "while the identity arm on the scrambled data stays near 0",
        "fired": bool(s_co > 0.90 * s_planted and s_id < 0.05 and ari > 0.90)})

    # ---- C3 planted-NEGATIVE: i.i.d. at matched density ------------------------------------
    rng = np.random.default_rng(99)
    p = P_PRIMARY
    kk = int(round(p * N))
    idx_n = np.stack([rng.choice(N, kk, replace=False) for _ in range(T)]).astype(np.int64)
    Bn = dense_mask_from_idx(idx_n, N)
    o_co_n, lab_co_n = order_coactivation(Bn, 64, CLUSTER_SEED)
    rows = []
    for bs in (4, 12, 21, 64, 140):
        hg = hypergeom_skippable(kk, N, bs)
        v_id = block_skippable_from_idx(idx_n, order_identity(N), N, bs)
        v_rnd = [block_skippable_from_idx(idx_n, pos_from_order(order_random(N, s)), N, bs)
                 for s in RANDOM_SEEDS]
        v_co = block_skippable_from_idx(idx_n, pos_from_order(o_co_n), N, bs)
        rows.append({"block": bs, "analytic_iid_qB": analytic_iid_skippable(p, bs),
                     "analytic_hypergeom": hg,
                     "identity": v_id, "random_mean": float(np.mean(v_rnd)),
                     "random_min": float(np.min(v_rnd)), "random_max": float(np.max(v_rnd)),
                     "coactivation": v_co,
                     "all_arms_at_analytic": bool(abs(v_id - hg) < 0.02
                                                  and abs(np.mean(v_rnd) - hg) < 0.02
                                                  and abs(v_co - hg) < 0.03)})
    controls.append({
        "name": "C3_planted_negative_iid_matched_density",
        "construction": f"N={N}, T={T}, exactly {kk} active/token drawn uniformly",
        "density_achieved": kk / N,
        "expect": "all three arms land together on the analytic hypergeometric value; any arm "
                  "that 'wins' on noise is broken",
        "rows": rows,
        "fired": bool(all(r["all_arms_at_analytic"] for r in rows))})

    # ---- C4 what the random permutation does and does NOT destroy -- MEASURED --------------
    # Stated in the module docstring, checked here, because two shuffle-based nulls on this
    # programme were not null and the most recent BEAT the real arm.
    perm = order_random(N, RANDOM_SEEDS[0])
    ppos = pos_from_order(perm)
    idx_perm = ppos[idx_n]                       # the same mask, relabelled by the permutation
    Ma = dense_mask_from_idx(idx_n, N)
    Mb = dense_mask_from_idx(idx_perm, N)

    # (i) bookkeeping, exact: un-permuting the permuted mask returns the original, elementwise.
    #     This is the step that could actually be buggy (pos_from_order / gather direction), and
    #     every arm in the probe rides on it.
    bookkeeping = bool(np.array_equal(Mb[:, ppos], Ma))
    # (ii) per-token active count, per token, not just k
    cnt_same = bool(np.array_equal(Ma.sum(1), Mb.sum(1)))
    # (iii) per-neuron firing-frequency multiset
    freq_same = bool(np.array_equal(np.sort(Ma.sum(0)), np.sort(Mb.sum(0))))
    # (iv) co-activation second moment: spectrum on a 512-neuron subset, computed from the
    #      ORIGINAL neurons and from those same neurons at their PERMUTED positions
    sub = np.random.default_rng(1).choice(N, 512, replace=False)
    Fa = Ma[:, sub].astype(np.float32)
    Fb = Mb[:, ppos[sub]].astype(np.float32)
    ev_a = np.linalg.eigvalsh((Fa.T @ Fa) / T)
    ev_b = np.linalg.eigvalsh((Fb.T @ Fb) / T)
    spec_d = float(np.abs(ev_a - ev_b).max())
    # (v) the ONE thing that does change, and it moves to the closed form, not away from it
    changed = [{"block": bs,
                "identity": block_skippable_from_idx(idx_n, order_identity(N), N, bs),
                "after_permutation": block_skippable_from_idx(idx_n, ppos, N, bs),
                "analytic_hypergeom": hypergeom_skippable(kk, N, bs)} for bs in (12, 64)]
    inv_ok = {"bookkeeping_unpermute_is_exact_identity": bookkeeping,
              "per_token_active_count_identical": cnt_same,
              "firing_frequency_multiset_identical": freq_same,
              "coactivation_spectrum_max_abs_diff": spec_d,
              "coactivation_spectrum_identical": bool(spec_d < 1e-4),
              "only_index_alignment_changes": changed}
    controls.append({
        "name": "C4_random_permutation_invariances",
        "why": "two shuffle-based nulls on this programme turned out not to be null, and the "
               "most recent recovered BETTER than the real arm because the shuffle destroyed a "
               "structural constraint and made the problem easier.  So what this shuffle does "
               "and does not destroy is stated, then MEASURED, before the arm is trusted.",
        "destroyed": "ONLY the alignment between neuron index and co-activation",
        "preserved_exactly": ["per-token active count",
                              "per-neuron firing-frequency multiset",
                              "the whole co-activation matrix up to relabelling (P^T C P is "
                              "similar to C: same spectrum, same rank, same clusterability)"],
        "consequence": "the random arm cannot make the problem easier here.  The only quantity "
                       "it changes is the one the score depends on, and its expectation under a "
                       "uniform permutation is the exact hypergeometric C(N-k,B)/C(N,B).  C3 "
                       "checks that expectation is met; a deviation would be an instrument "
                       "fault, not a finding.",
        "measured": inv_ok,
        "expect": "all four invariances hold exactly; the shuffle is a null for INDEX ALIGNMENT "
                  "and for nothing else",
        "fired": bool(bookkeeping and cnt_same and freq_same and spec_d < 1e-4)})
    del Ma, Mb, Fa, Fb

    # ---- C5 instrument equivalence: fast index path == D0.layout_stats ---------------------
    eq = []
    for bs in (4, 12, 21, 64, 140):
        st = D0.layout_stats(Bn, 768.0, block_sizes=[bs], max_tokens=T)
        mine = block_skippable_from_idx(idx_n, order_identity(N), N, bs)
        ref = st["blocks"][str(bs)]["block_skippable_fraction"]
        eq.append({"block": bs, "d0_layout_stats": ref, "fast_index_path": mine,
                   "abs_diff": abs(ref - mine)})
    controls.append({
        "name": "C5_fast_path_equals_D0_layout_stats",
        "expect": "agreement <1e-12 between this file's index-based accounting and "
                  "d0_layout.layout_stats, so the memory-lean path inherits D0's own controls",
        "rows": eq, "fired": bool(all(r["abs_diff"] < 1e-12 for r in eq))})
    del Bn

    # ---- C6 the hypergeometric reference itself, against brute force ------------------------
    bf = []
    rng2 = np.random.default_rng(5)
    for bs in (4, 12, 21, 64):
        hg = hypergeom_skippable(kk, N, bs)
        trials, hits = 20000, 0
        for _ in range(trials):
            a = rng2.choice(N, kk, replace=False)
            hits += int(not np.any(a < bs))
        bf.append({"block": bs, "closed_form": hg, "monte_carlo": hits / trials,
                   "abs_diff": abs(hg - hits / trials)})
    controls.append({
        "name": "C6_hypergeometric_reference_vs_monte_carlo",
        "expect": "closed form matches Monte-Carlo within sampling error (<0.01)",
        "rows": bf, "fired": bool(all(r["abs_diff"] < 0.01 for r in bf))})

    # ---- C7 the A1.1 arithmetic, checked against the brief's own table ----------------------
    tbl = [(0.282, 0.812), (0.656, 0.562), (0.950, 0.367), (1.000, 1.0 / 3.0)]
    c7 = [{"s": s, "brief_A1_1": v, "computed": ffn_active_dense_gate(s),
           "abs_diff": abs(v - ffn_active_dense_gate(s))} for s, v in tbl]
    s_needed = (3.0 - 3 * 0.02) / 2.0
    controls.append({
        "name": "C7_amendment1_identity_reproduces_brief_table",
        "rows": c7, "s_required_for_2pct_FFN_active": s_needed,
        "expect": "FFN_active=(3-2s)/3 reproduces A1.1's table to 1e-3 and the 2% budget needs "
                  "s=1.470, which is impossible",
        "fired": bool(all(r["abs_diff"] < 1.5e-3 for r in c7) and abs(s_needed - 1.47) < 1e-9)})

    out["controls"] = controls
    out["all_fired"] = bool(all(c["fired"] for c in controls))
    for c in controls:
        print(f"  {c['name']:48s} -> {'FIRED' if c['fired'] else '*** DID NOT FIRE ***'}",
              flush=True)
    c2 = controls[0]
    print(f"    C2 detail: planted s={c2['skippable_planted_exact']:.4f}  "
          f"identity(scrambled) s={c2['skippable_identity_on_scrambled']:.4f}  "
          f"co-activation s={c2['skippable_coactivation_recovered']:.4f}  "
          f"ARI={c2['cluster_ARI_vs_planted_labels']:.4f}", flush=True)
    print("wrote", dump_frag("controls", out), flush=True)
    if not out["all_fired"]:
        print("\n!!! a control DID NOT FIRE -- STOP and report to the Principal !!!", flush=True)
        sys.exit(2)


# ============================================================================ STAGE: model
def _perm_ffn_(model, perms: dict):
    """Permute FFN hidden neurons in place: gate/up ROWS and down COLUMNS, together.  This is
    mathematically output-preserving; §7.1 makes verifying that the FIRST thing D0 does."""
    for L, p in perms.items():
        mlp = model.model.layers[L].mlp
        assert mlp.gate_proj.bias is None and mlp.up_proj.bias is None, "biased FFN unhandled"
        idx = torch.as_tensor(np.asarray(p), dtype=torch.long)
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[idx].contiguous()
        mlp.up_proj.weight.data = mlp.up_proj.weight.data[idx].contiguous()
        mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, idx].contiguous()


def _capture(model, ids, layers, kmax, want_vals, tag):
    """One forward pass; hooks sit on `down_proj`'s INPUT, which is exactly
    h = silu(gate x) * up x and is computed anyway -- so hooking all 28 layers costs no extra
    FLOPs, only storage.  Stores the RANKED top-kmax |h| indices (int16) so any density
    p <= P_MAX is derivable from one capture without a second forward pass."""
    buf = {L: [] for L in layers}
    hooks = []

    def mk(L):
        def hook(mod, args):
            h = args[0]
            h = h.reshape(-1, h.shape[-1])
            v, i = h.abs().topk(kmax, dim=1)                      # ranked, descending
            buf[L].append((i.to(torch.int16).numpy().copy(),
                           v.to(torch.float16).numpy().copy() if L in want_vals else None))
        return hook

    for L in layers:
        hooks.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L)))
    t0 = time.time()
    for i in range(ids.shape[0]):
        model.model(ids[i:i + 1])                                 # batch 1 keeps the peak flat
        if (i + 1) % 8 == 0:
            print(f"   [{tag}] {i+1}/{ids.shape[0]} seqs  {time.time()-t0:.0f}s  {_rss_gb()}",
                  flush=True)
    for h in hooks:
        h.remove()

    meta = {}
    os.makedirs(MASKDIR, exist_ok=True)
    for L in layers:
        I = np.concatenate([b[0] for b in buf[L]], 0)
        payload = {"idx": I}
        if L in want_vals:
            payload["val"] = np.concatenate([b[1] for b in buf[L]], 0)
        buf[L] = None
        f = os.path.join(MASKDIR, f"{tag}_L{L:02d}.npz")
        np.savez(f, **payload)
        meta[str(L)] = {"T_tokens_achieved": int(I.shape[0]),
                        "kmax_stored_achieved": int(I.shape[1]),
                        "p_max_achieved": I.shape[1] / model.config.intermediate_size,
                        "has_values": L in want_vals,
                        "idx_sha256": hashlib.sha256(np.ascontiguousarray(I).tobytes()).hexdigest(),
                        "bytes_on_disk": os.path.getsize(f)}
        del I, payload
    return meta, time.time() - t0


def stage_model():
    out = {"env": env_manifest("model")}
    print("== D0 stage `model` -- donor resident ==", flush=True)
    t0 = time.time()
    m, tok = C.load_model()                              # fp32, eager -- same path as D1/D4
    A = C.arch(m)
    out["arch_achieved"] = A
    d_model, d_ffn, n_layers = A["d_model"], A["d_ffn"], A["n_layers"]
    print(json.dumps(A, indent=2), flush=True)
    out["mem_after_load"] = _rss_gb()
    print(f"  loaded in {time.time()-t0:.0f}s  {out['mem_after_load']}", flush=True)

    # ---- slices, and the disjointness assertion (brief §8) ---------------------------------
    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", N_EVAL, SEQ_LEN_EVAL, SEED_EVAL)
    ids_fit, _, meta_fit = C.get_slice(tok, "calib", N_FIT, SEQ_LEN_FIT, SEED_FIT)
    ids_sc, _, meta_sc = C.get_slice(tok, "heldout", N_SCORE, SEQ_LEN_SCORE, SEED_SCORE)
    assert meta_fit["corpus_sha256"] != meta_ev["corpus_sha256"], \
        "fit and BPB-eval draw from the SAME corpus half -- disjointness violated"
    assert meta_fit["part"] == "calib" and meta_ev["part"] == "heldout"
    assert meta_fit["ids_sha256"] != meta_sc["ids_sha256"]
    out["bpb_eval_slice"], out["fit_slice"], out["score_slice"] = meta_ev, meta_fit, meta_sc
    out["disjointness_assertion"] = {
        "fit_corpus_sha256": meta_fit["corpus_sha256"],
        "score_corpus_sha256": meta_sc["corpus_sha256"],
        "bpb_eval_corpus_sha256": meta_ev["corpus_sha256"],
        "fit_half": "calib", "score_half": "heldout",
        "fit_vs_bpb_eval_distinct_halves": True,
        "fit_vs_score_distinct_ids_sha256": True,
        "checked_in_code": "assert meta_fit['corpus_sha256'] != meta_ev['corpus_sha256'] and "
                           "meta_fit['ids_sha256'] != meta_sc['ids_sha256']",
        "note": "the co-activation ordering is FITTED on `fit` (calib half) and SCORED on "
                "`score` (held-out half).  The BPB path uses D1's own pinned slice, untouched."}
    dump_frag("model", out)

    # ---- CONTROL C1: identity / losslessness.  RUN FIRST.  §7.1 ----------------------------
    print("\n-- C1 identity/losslessness control (runs FIRST; a failure voids everything) --",
          flush=True)
    rng = np.random.default_rng(LOSSLESS_PERM_SEED)
    perms = {L: rng.permutation(d_ffn) for L in range(n_layers)}
    inv = {L: pos_from_order(p) for L, p in perms.items()}

    with C.Timer("bpb identity"):
        bpb_before = C.bpb(m, ids_ev, byts_ev, batch=1)
    print(f"   BPB(identity)  = {bpb_before:.9f}", flush=True)

    # max|dlogit| measured on a 2-sequence subset: keeping logits for the whole slice would be
    # 24*512*151936*4 = 7.5 GB.  The BPB comparison needs only scalar NLL accumulation and IS
    # done on the full pinned slice.
    sub = ids_ev[:2]
    base_logits = m(sub).logits.float().clone()
    _perm_ffn_(m, perms)
    dmax = float((m(sub).logits.float() - base_logits).abs().max())
    dref = float(base_logits.abs().max())
    del base_logits

    with C.Timer("bpb permuted"):
        bpb_after = C.bpb(m, ids_ev, byts_ev, batch=1)
    print(f"   BPB(permuted)  = {bpb_after:.9f}   delta = {bpb_after-bpb_before:+.3e}", flush=True)
    _perm_ffn_(m, inv)                                   # restore
    with C.Timer("bpb restored"):
        bpb_restore = C.bpb(m, ids_ev, byts_ev, batch=1)

    c1 = {
        "name": "C1_identity_losslessness",
        "construction": f"independent random permutation of all {d_ffn} FFN neurons in ALL "
                        f"{n_layers} layers (seed {LOSSLESS_PERM_SEED}); gate/up rows and down "
                        f"columns permuted together, then inverted in place",
        "bpb_identity": bpb_before, "bpb_permuted": bpb_after, "bpb_restored": bpb_restore,
        "delta_bpb_permuted_minus_identity": bpb_after - bpb_before,
        "delta_bpb_restored_minus_identity": bpb_restore - bpb_before,
        "max_abs_logit_deviation": dmax, "max_abs_logit_magnitude": dref,
        "relative_logit_deviation": dmax / dref,
        "d1_published_baseline_bpb": 0.767595,
        "matches_d1_baseline_to_6dp": bool(abs(bpb_before - 0.767595) < 5e-7),
        "sigma_seed_for_scale": 0.005,
        "expect": "BPB unchanged to fp32 summation noise (|dBPB| < 1e-6, i.e. <2e-4 sigma_seed), "
                  "restore EXACT, max|dlogit| at fp32 round-off relative to logit magnitude",
        "note": "exact bitwise equality is NOT expected and is not the claim: permuting "
                "down_proj's columns changes the fp32 summation ORDER.  The claim is that the "
                "deviation is round-off, which is why max|dlogit| is reported rather than "
                "asserted to be zero.",
        "fired": bool(abs(bpb_after - bpb_before) < 1e-6 and bpb_restore == bpb_before
                      and dmax / dref < 1e-4)}
    out["control_C1_losslessness"] = c1
    print(f"   max|dlogit| = {dmax:.3e}  (|logit|max = {dref:.3f}, rel = {dmax/dref:.2e})",
          flush=True)
    print(f"   C1 -> {'FIRED' if c1['fired'] else '*** DID NOT FIRE ***'}", flush=True)
    dump_frag("model", out)
    if not c1["fired"]:
        print("\n!!! C1 FAILED -- everything else is void.  STOP. !!!", flush=True)
        sys.exit(2)

    # ---- capture ---------------------------------------------------------------------------
    kmax = int(round(P_MAX * d_ffn))
    assert d_ffn < 32767, "int16 index store would overflow"
    layers = list(range(n_layers)) if ALL_LAYERS else list(DEEP_LAYERS)
    print(f"\n-- capture: ranked top-{kmax} (p={kmax/d_ffn:.4f}) |h| indices, {len(layers)} "
          f"layers, hooks on down_proj input (no extra FLOPs) --", flush=True)
    cap = {}
    for tag, ids in (("fit", ids_fit), ("score", ids_sc)):
        meta, secs = _capture(m, ids, layers, kmax, set(DEEP_LAYERS), tag)
        cap[tag] = {"layers": meta, "seconds": secs, "n_seq": int(ids.shape[0])}
        out["capture"] = cap
        dump_frag("model", out)
        print(f"   [{tag}] captured {len(meta)} layers in {secs:.0f}s", flush=True)
    out["capture_kmax_requested"] = kmax
    out["capture_maskdir"] = MASKDIR

    # ---- FIDELITY: oracle-routed relative L2 error of the FFN output ------------------------
    # A1.3.2 mandates the GEOMETRY (mask-only, computed in `analyse`).  This is the stronger
    # secondary question the geometry cannot answer: if you actually only compute the top-k
    # experts, how much of the FFN output survives?  D0.routed_error is REUSED unmodified; its
    # router is an ORACLE (picks experts by ||h restricted||), so it is an UPPER BOUND on any
    # trainable router.  If the oracle cannot recover the output, nothing can.
    print("\n-- fidelity: oracle-routed relative L2 error (upper bound on any router) --",
          flush=True)
    fid = {}
    sc_pref = ids_sc[:max(1, FIDELITY_TOKENS // SEQ_LEN_SCORE)]
    Xsc = {}
    hooks = []

    def mkx(L):
        def hook(mod, args):
            Xsc.setdefault(L, []).append(args[0].detach().reshape(-1, args[0].shape[-1]).clone())
        return hook
    for L in DEEP_LAYERS:
        hooks.append(m.model.layers[L].mlp.register_forward_pre_hook(mkx(L)))
    for i in range(sc_pref.shape[0]):
        m.model(sc_pref[i:i + 1])
    for h in hooks:
        h.remove()
    Xsc = {L: torch.cat(v, 0) for L, v in Xsc.items()}

    k_prim = int(round(P_PRIMARY * d_ffn))
    for L in DEEP_LAYERS:
        z = np.load(os.path.join(MASKDIR, f"fit_L{L:02d}.npz"))
        idx_fit = z["idx"].astype(np.int64)[:, :k_prim]
        Bfit = dense_mask_from_idx(idx_fit, d_ffn)
        rows = []
        for E in FIDELITY_E:
            lab_bal = np.asarray(D0.balanced_labels(torch.from_numpy(Bfit), E, CLUSTER_SEED))
            lab_rnd = lab_bal.copy(); np.random.default_rng(1000 + E).shuffle(lab_rnd)
            for pname, lab in (("coactivation_balanced", lab_bal),
                               ("random_balanced_NULL", lab_rnd)):
                sizes = np.bincount(lab, minlength=E)
                for k in FIDELITY_K:
                    if k >= E:
                        continue
                    err, kept = D0.routed_error(m, L, Xsc[L], lab, k)
                    rows.append({"E": E, "k": k, "partition": pname,
                                 "nominal_activation_fraction": k / E,
                                 "true_activation_fraction_achieved": kept,
                                 "relerr": err,
                                 "cluster_size_min": int(sizes.min()),
                                 "cluster_size_max": int(sizes.max()),
                                 "expert_bytes_per_organ": float(sizes.mean() * BYTES_PER_WEIGHT
                                                                 * d_model),
                                 "engine_legal_48KB": bool(sizes.min() * BYTES_PER_WEIGHT
                                                           * d_model >= BLOCK_BYTES)})
                    print(f"   L{L:02d} E={E:3d} k={k:2d} {pname:22s} "
                          f"act_true={kept:6.3f} relerr={err:.4f}", flush=True)
        fid[str(L)] = {"rows": rows, "n_tokens_achieved": int(Xsc[L].shape[0]),
                       "token_source": "score slice (held-out half)",
                       "labels_fitted_on": "fit slice (calib half)"}
        del Bfit, idx_fit
        out["fidelity"] = fid
        dump_frag("model", out)

    out["mem_peak"] = _rss_gb()
    print("wrote", dump_frag("model", out), flush=True)
    print(f"== stage `model` done in {time.time()-t0:.0f}s, peak {out['mem_peak']} ==", flush=True)


# ============================================================================ MoE geometry
def moe_geometry(idx: np.ndarray, labels: np.ndarray, N: int, d_model: int, E: int) -> dict:
    """A1.3.2: the MoE geometry a partition implies, read off the MEASURED activation pattern.

    For each token the experts that must be LOADED to retain all its active neurons are those
    containing at least one of them; the expert-active fraction is the fraction of FFN weight
    BYTES that reaches the core, = sum(size of loaded experts)/N, because gate row + up row +
    down column all scale with the same neuron count.

    Reported WITHOUT a dense gate -- removing the dense gate is the entire point (A1.2) -- and
    the router's own cost is stated explicitly rather than assumed negligible.

    IDENTITY worth stating: when the partition is "contiguous blocks of B = N/E neurons under
    an arm's ordering", `expert_active_fraction_full_coverage` is exactly `1 - s(B)` from the
    skippable table.  The two mandated tables are the same measurement in two units.
    """
    sizes = np.bincount(labels, minlength=E).astype(np.int64)
    T, k = idx.shape
    cnt = counts_per_expert(idx, labels, E)
    loaded_any = cnt > 0
    n_loaded_full = loaded_any.sum(1)
    bytes_full = (loaded_any * sizes[None, :]).sum(1) / N

    # ORACLE router: activate experts in descending active-count order (an upper bound on any
    # trainable router) and record the cheapest expert-active fraction reaching each target.
    ordr = np.argsort(-cnt, axis=1, kind="stable")
    cum_cov = np.cumsum(np.take_along_axis(cnt, ordr, axis=1), axis=1) / k
    cum_byt = np.cumsum(sizes[ordr], axis=1) / N
    cover = {}
    for tgt in COVER_TARGETS:
        reach = cum_cov >= tgt - 1e-12
        first = np.where(reach.any(1), np.argmax(reach, axis=1), E - 1)
        cover[f"{tgt:.2f}"] = {
            "mean_experts_activated": float((first + 1).mean()),
            "median_experts_activated": float(np.median(first + 1)),
            "expert_active_fraction": float(np.take_along_axis(cum_byt, first[:, None], 1).mean()),
        }
    topk = {str(kk): {"expert_active_fraction": float(cum_byt[:, kk - 1].mean()),
                      "active_neuron_coverage": float(cum_cov[:, kk - 1].mean())}
            for kk in K_LIST if kk < E}

    router_params = d_model * E
    ffn_params = 3 * d_model * N
    return {
        "E_achieved": int(E),
        "cluster_size_min": int(sizes.min()), "cluster_size_max": int(sizes.max()),
        "cluster_size_mean": float(sizes.mean()), "cluster_size_median": float(np.median(sizes)),
        "cluster_size_std": float(sizes.std()),
        "cluster_size_cv": float(sizes.std() / max(1e-9, sizes.mean())),
        "n_empty_clusters": int((sizes == 0).sum()),
        "expert_bytes_per_organ_at_mean_size": float(sizes.mean() * BYTES_PER_WEIGHT * d_model),
        "engine_legal_48KB_at_min_size": bool(sizes.min() * BYTES_PER_WEIGHT * d_model
                                              >= BLOCK_BYTES),
        "experts_needed_full_coverage_mean": float(n_loaded_full.mean()),
        "experts_needed_full_coverage_median": float(np.median(n_loaded_full)),
        "expert_active_fraction_full_coverage": float(bytes_full.mean()),
        "coverage_targets": cover, "fixed_topk_router": topk,
        "router_params": int(router_params),
        "router_frac_of_ffn_weights": float(router_params / ffn_params),
        "router_note": "a top-k router over E experts is a [d_model, E] matvec per token.  Its "
                       "WEIGHT cost is the fraction above; its LATENCY cost is NOT negligible "
                       "in the same way, because it is strictly serial before any expert byte "
                       "can be fetched -- it replaces the dense gate as the predictor.",
        "moe_ffn_active_full_coverage": float(bytes_full.mean() + router_params / ffn_params),
        "dense_gate_floor_for_comparison": ffn_active_dense_gate(1.0),
    }


# ============================================================================ duplication (§5)
def duplication(idx: np.ndarray, labels: np.ndarray, N: int, E: int, k_router: int,
                n_tokens: int, seed: int = 3) -> dict:
    """§5 / A1.3.3.  Apple's documented failure mode, measured rather than feared.

      (a) `experts_touched` -- for each neuron n, how many DISTINCT experts contain a neuron
          co-active with n.  Apple's mechanism stated directly: a highly-active neuron
          co-activates with everything, so it "wants" to be in every bundle.  Routing-free
          diagnostic; broken out by activity decile, which is where Apple says it bites.

      (b) `dup_extra_greedy` -- the BUDGET-relevant number.  Under an ORACLE top-k router,
          neuron n is MISSED on any token where n is active but n's home expert was not loaded.
          To make top-k routing LOSSLESS by duplication, n must be added to at least one loaded
          expert on every missed token: a set-cover over experts, solved greedily.  The mean of
          (1 + dup_extra_greedy) over neurons is the WEIGHT-INFLATION MULTIPLIER, which
          multiplies the very bytes the MoE was adopted to save.
    """
    T_all, kk = idx.shape
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(T_all, min(n_tokens, T_all), replace=False))
    I = np.ascontiguousarray(idx[sel])
    T = I.shape[0]
    lab_t = labels[I]
    rows = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None], lab_t.shape)
    cnt = counts_per_expert(I, labels, E)
    present = cnt > 0

    freq = np.bincount(I.ravel(), minlength=N).astype(np.float64) / T
    dec = np.digitize(freq, np.quantile(freq, np.arange(1, 10) / 10.0))

    # ---- (a) experts touched ---------------------------------------------------------------
    act = np.zeros((T, N), dtype=np.float32)
    act[rows.ravel(), I.ravel()] = 1.0
    n_touch = ((act.T @ present.astype(np.float32)) > 0).sum(1)
    del act

    # ---- (b) oracle top-k routing -> missed -> greedy set cover -----------------------------
    kr = min(k_router, E)
    sel_e = np.argsort(-cnt, axis=1, kind="stable")[:, :kr]
    loaded = np.zeros((T, E), dtype=bool)
    loaded[np.broadcast_to(np.arange(T, dtype=np.int64)[:, None], sel_e.shape).ravel(),
           sel_e.ravel()] = True
    home_loaded = loaded[rows.ravel(), lab_t.ravel()]
    miss = np.flatnonzero(~home_loaded)
    miss_tok, miss_neu = rows.ravel()[miss], I.ravel()[miss]
    o = np.argsort(miss_neu, kind="stable")
    miss_tok, miss_neu = miss_tok[o], miss_neu[o]
    bounds = np.searchsorted(miss_neu, np.arange(N + 1))
    dup_extra = np.zeros(N, dtype=np.int32)
    dup_naive = np.zeros(N, dtype=np.int32)
    loaded_T = np.ascontiguousarray(loaded.T)                          # [E, T]
    for n in range(N):
        a, b = bounds[n], bounds[n + 1]
        if a == b:
            continue
        sub = loaded_T[:, miss_tok[a:b]]                               # [E, m]
        dup_naive[n] = int(sub.any(1).sum())
        remaining = np.ones(sub.shape[1], dtype=bool)
        c = 0
        while remaining.any() and c < E:
            gain = sub[:, remaining].sum(1)
            e = int(np.argmax(gain))
            if gain[e] == 0:
                break
            remaining[np.flatnonzero(remaining)[sub[e, remaining]]] = False
            c += 1
        dup_extra[n] = c

    def dist(v):
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                "frac_zero": float((v == 0).mean())}

    top = dec == 9
    return {
        "E": int(E), "k_router": int(kr), "n_tokens_used_achieved": int(T),
        "n_tokens_available": int(T_all),
        "missed_active_fraction": float(len(miss_neu) / (T * kk)),
        "experts_touched_all": dist(n_touch),
        "experts_touched_top_activity_decile": dist(n_touch[top]),
        "experts_touched_bottom_activity_decile": dist(n_touch[dec == 0]),
        "dup_extra_greedy_all": dist(dup_extra),
        "dup_extra_greedy_top_activity_decile": dist(dup_extra[top]),
        "dup_naive_upper_bound_all": dist(dup_naive),
        "weight_inflation_multiplier_greedy": float((1 + dup_extra).mean()),
        "weight_inflation_multiplier_naive": float((1 + dup_naive).mean()),
        "weight_inflation_note": "multiplier = total neuron-copies / N.  An expert-active "
                                 "fraction f costs f * multiplier in resident-weight terms, so "
                                 "duplication eats directly into the byte budget the MoE was "
                                 "adopted to save.",
        "decile_definition": "decile 9 = most-active tenth of neurons by firing frequency on "
                             "the same token subsample",
    }


# ============================================================================ STAGE: analyse
def _load_idx(tag, L, k):
    z = np.load(os.path.join(MASKDIR, f"{tag}_L{L:02d}.npz"))
    return np.ascontiguousarray(z["idx"][:, :k].astype(np.int64))


def stage_analyse():
    out = {"env": env_manifest("analyse")}
    mod = json.load(open(FRAG("model")))
    ctl = json.load(open(FRAG("controls")))
    out["control_C1_losslessness"] = mod["control_C1_losslessness"]
    out["synthetic_controls"] = ctl["controls"]
    out["all_synthetic_controls_fired"] = ctl["all_fired"]
    for key in ("arch_achieved", "bpb_eval_slice", "fit_slice", "score_slice",
                "disjointness_assertion", "capture", "fidelity"):
        if key in mod:
            out[key] = mod[key]
    A = mod["arch_achieved"]
    d_model, d_ffn, n_layers = A["d_model"], A["d_ffn"], A["n_layers"]

    out["rho_floor"] = {
        "donor_d_model": rho_floor_blocks(d_model),
        "llama3_70b_class_d_model_8192": rho_floor_blocks(8192),
        "note": "D0b: 21.33 neurons/48KB interleaved is a d_model=1536 number, NOT an engine "
                "universal.  At 70B-class width it is 4.0.  Both floors are swept here, so the "
                "D0b structureless reference (0.2824 @ block 12, 0.6563 @ block 4) is compared "
                "at the block size it was quoted for."}
    BLOCKS = block_sizes_for(d_model)
    out["block_sizes_swept"] = BLOCKS
    out["amendment1_identity"] = {
        "formula": "FFN_active = (3 - 2s)/3",
        "s_required_for_2pct_budget": (3.0 - 3 * 0.02) / 2.0,
        "floor_at_perfect_s": ffn_active_dense_gate(1.0),
        "d0b_reference_block12": {"s": 0.2824, "FFN_active": ffn_active_dense_gate(0.2824)},
        "d0b_reference_block4": {"s": 0.6563, "FFN_active": ffn_active_dense_gate(0.6563)}}

    # ---------------------------------------------------------------- deep layers
    layers = {}
    for L in DEEP_LAYERS:
        t0 = time.time()
        rec = {"layer": L, "densities": {}}

        # realism check: what a TEAL/CATS-style GLOBAL magnitude threshold actually gives
        z = np.load(os.path.join(MASKDIR, f"score_L{L:02d}.npz"))
        if "val" in z:
            V = z["val"].astype(np.float32)
            thr = float(np.median(V[:, int(round(P_PRIMARY * d_ffn)) - 1]))
            per_tok = (V >= thr).sum(1)
            rec["global_threshold_realism_check"] = {
                "threshold": thr, "requested_mean_density": P_PRIMARY,
                "achieved_mean_density": float(per_tok.mean() / d_ffn),
                "achieved_density_p10": float(np.percentile(per_tok, 10) / d_ffn),
                "achieved_density_p90": float(np.percentile(per_tok, 90) / d_ffn),
                "capped_at_p_max": float((per_tok >= V.shape[1]).mean()),
                "note": "the top-p masks below fix the density EXACTLY per token; a real "
                        "TEAL/CATS threshold does not.  Reported so the exact-density "
                        "idealisation is visible rather than hidden."}
            del V

        for p in DENSITIES:
            k = int(round(p * d_ffn))
            idx_fit = _load_idx("fit", L, k)
            idx_sc = _load_idx("score", L, k)
            dens = {"p_requested": p, "p_achieved": k / d_ffn, "k_active_achieved": k,
                    "T_fit_achieved": int(idx_fit.shape[0]),
                    "T_score_achieved": int(idx_sc.shape[0]),
                    "arms": {}, "arms_in_sample": {}, "moe": {}}
            p_ach = k / d_ffn

            arms = {"identity": order_identity(d_ffn)}
            for s in RANDOM_SEEDS:
                arms[f"random_s{s}"] = order_random(d_ffn, s)
            Bfit = dense_mask_from_idx(idx_fit, d_ffn)         # fitted on the CALIB half only
            co_labels = {}
            for E in E_LIST:
                o, lab = order_coactivation(Bfit, E, CLUSTER_SEED)
                arms[f"coactivation_E{E}"] = o
                co_labels[E] = lab

            for name, order in arms.items():
                pos = pos_from_order(order)
                pi_sc = positions_of_active(idx_sc, pos)          # paid for once, not per block
                dens["arms"][name] = {
                    str(bs): {"s": (sv := block_skippable_from_posidx(pi_sc, d_ffn, bs)),
                              "FFN_active_dense_gate": ffn_active_dense_gate(sv),
                              "analytic_iid_qB": analytic_iid_skippable(p_ach, bs),
                              "analytic_hypergeom": hypergeom_skippable(k, d_ffn, bs),
                              "blocks_achieved": d_ffn // bs,
                              "neurons_covered_achieved": (d_ffn // bs) * bs}
                    for bs in BLOCKS}
                del pi_sc
                if name == "identity" or name.startswith("coactivation"):
                    pi_fit = positions_of_active(idx_fit, pos)
                    dens["arms_in_sample"][name] = {
                        str(bs): block_skippable_from_posidx(pi_fit, d_ffn, bs)
                        for bs in BLOCKS}
                    del pi_fit

            spread = {}
            for bs in BLOCKS:
                v = np.array([dens["arms"][f"random_s{s}"][str(bs)]["s"] for s in RANDOM_SEEDS])
                hg = hypergeom_skippable(k, d_ffn, bs)
                spread[str(bs)] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                                   "min": float(v.min()), "max": float(v.max()),
                                   "n_seeds": len(RANDOM_SEEDS),
                                   "analytic_hypergeom": hg,
                                   "residual_structure_abs": float(abs(v.mean() - hg))}
            dens["random_seed_spread"] = spread

            verdict = {}
            for bs in BLOCKS:
                s_id = dens["arms"]["identity"][str(bs)]["s"]
                sr = spread[str(bs)]
                best_E = max(E_LIST, key=lambda E: dens["arms"][f"coactivation_E{E}"][str(bs)]["s"])
                best = dens["arms"][f"coactivation_E{best_E}"][str(bs)]["s"]
                ins = dens["arms_in_sample"][f"coactivation_E{best_E}"][str(bs)]
                verdict[str(bs)] = {
                    "identity": s_id, "random_mean": sr["mean"],
                    "random_min": sr["min"], "random_max": sr["max"], "random_std": sr["std"],
                    "coactivation_best": best, "coactivation_best_E": best_E,
                    "coactivation_in_sample": ins,
                    "generalisation_gap_in_minus_out": ins - best,
                    "structureless_ref_qB": analytic_iid_skippable(p_ach, bs),
                    "coact_beats_identity": bool(best > s_id),
                    "coact_clears_random_spread": bool(best > sr["max"]),
                    "coact_beats_structureless_ref": bool(
                        best > analytic_iid_skippable(p_ach, bs)),
                    "coact_FFN_active_dense_gate": ffn_active_dense_gate(best),
                    "identity_FFN_active_dense_gate": ffn_active_dense_gate(s_id)}
            dens["verdict_by_block"] = verdict

            # --- MoE geometry per arm (A1.3.2) -----------------------------------------------
            for E in E_LIST:
                bs = d_ffn // E
                geo = {}
                for nm in ("identity", f"random_s{RANDOM_SEEDS[0]}"):
                    pos = pos_from_order(arms[nm])
                    geo[nm + "_contiguous_blocks"] = moe_geometry(
                        idx_sc, np.minimum(pos // bs, E - 1), d_ffn, d_model, E)
                geo["coactivation_kmeans"] = moe_geometry(idx_sc, co_labels[E], d_ffn, d_model, E)
                lab_bal = np.asarray(D0.balanced_labels(torch.from_numpy(Bfit), E, CLUSTER_SEED))
                geo["coactivation_balanced"] = moe_geometry(idx_sc, lab_bal, d_ffn, d_model, E)
                lab_rnd = lab_bal.copy(); np.random.default_rng(1000 + E).shuffle(lab_rnd)
                geo["random_balanced_NULL"] = moe_geometry(idx_sc, lab_rnd, d_ffn, d_model, E)
                dens["moe"][str(E)] = geo

            # --- duplication (§5 / A1.3.3), primary density only ------------------------------
            if abs(p - P_PRIMARY) < 1e-12:
                # E values taken from E_LIST, never retyped -- a hardcoded (32, 64, 128) here
                # KeyErrors the moment E_LIST changes, which a dry run caught.
                dens["duplication"] = {
                    str(E): {str(kr): duplication(idx_sc, co_labels[E], d_ffn, E, kr, DUP_TOKENS)
                             for kr in (1, 2, 4, 8) if kr < E}
                    for E in E_LIST}
            del Bfit, idx_fit, idx_sc
            rec["densities"][f"{p:.2f}"] = dens

            b12 = str(int(round(rho_floor_blocks(8192)["neurons_per_48KB_per_organ"])))
            v = verdict[b12]
            print(f"  L{L:02d} p={p_ach:.4f} block{b12}  id={v['identity']:.4f} "
                  f"rnd={v['random_mean']:.4f}[{v['random_min']:.4f},{v['random_max']:.4f}] "
                  f"co={v['coactivation_best']:.4f} ref={v['structureless_ref_qB']:.4f} "
                  f"FFN_active={v['coact_FFN_active_dense_gate']:.4f}", flush=True)

        layers[str(L)] = rec
        out["deep_layers"] = layers
        dump_frag("analyse", out)
        print(f"  L{L:02d} done {time.time()-t0:.0f}s  {_rss_gb()}", flush=True)

    # ---------------------------------------------------------------- depth sweep, all layers
    have_all = all(os.path.exists(os.path.join(MASKDIR, f"score_L{L:02d}.npz"))
                   for L in range(n_layers))
    if have_all:
        print(f"\n-- depth sweep: all {n_layers} layers, p={P_PRIMARY}, E={E_DEPTH} --",
              flush=True)
        k = int(round(P_PRIMARY * d_ffn))
        bs = d_ffn // E_DEPTH
        depth = {}
        for L in range(n_layers):
            idx_fit = _load_idx("fit", L, k)
            idx_sc = _load_idx("score", L, k)
            Bfit = dense_mask_from_idx(idx_fit, d_ffn)
            o, lab = order_coactivation(Bfit, E_DEPTH, CLUSTER_SEED)
            s_co = block_skippable_from_idx(idx_sc, pos_from_order(o), d_ffn, bs)
            s_id = block_skippable_from_idx(idx_sc, order_identity(d_ffn), d_ffn, bs)
            v = [block_skippable_from_idx(idx_sc, pos_from_order(order_random(d_ffn, s)),
                                          d_ffn, bs) for s in RANDOM_SEEDS[:5]]
            geo = moe_geometry(idx_sc, lab, d_ffn, d_model, E_DEPTH)
            depth[str(L)] = {
                "block_size": bs, "p_achieved": k / d_ffn,
                "identity": s_id, "random_mean": float(np.mean(v)),
                "random_min": float(np.min(v)), "random_max": float(np.max(v)),
                "coactivation": s_co,
                "structureless_ref_qB": analytic_iid_skippable(k / d_ffn, bs),
                "coact_FFN_active_dense_gate": ffn_active_dense_gate(s_co),
                "expert_active_fraction_full_coverage": geo["expert_active_fraction_full_coverage"],
                "experts_needed_full_coverage_mean": geo["experts_needed_full_coverage_mean"],
                "expert_active_fraction_at_95pct_coverage":
                    geo["coverage_targets"]["0.95"]["expert_active_fraction"],
                "cluster_size_cv": geo["cluster_size_cv"]}
            print(f"  L{L:02d} id={s_id:.4f} rnd={np.mean(v):.4f} co={s_co:.4f} "
                  f"expert_active(full)={geo['expert_active_fraction_full_coverage']:.4f} "
                  f"expert_active(95%)="
                  f"{geo['coverage_targets']['0.95']['expert_active_fraction']:.4f}", flush=True)
            del Bfit, idx_fit, idx_sc
            out["depth_sweep"] = depth
            dump_frag("analyse", out)

    out["allocation_inventory"] = _ALLOCATION_INVENTORY
    out["mem_peak"] = _rss_gb()
    dump_frag("analyse", out)
    final = os.path.join(C.RESULTS, "d0_coactivation.json")
    with open(final, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("wrote", final, flush=True)


# ============================================================================ inventory
_ALLOCATION_INVENTORY = {
    "law": "inventory every O(input) allocation explicitly, INCLUDING the ones that fit",
    "machine_at_design_time": "79.95 GB total.  Designed under a ~11.4 GB free ceiling (a "
                              "45.7 GB llama-server resident); the ceiling later lifted to "
                              "~63 GB free.  The design was NOT inflated to fill it: the extra "
                              "headroom bought all-layer capture and the fit/score split, both "
                              "of which buy science.  D0_ALL_LAYERS=0 returns the model stage "
                              "to <8 GB with no other change.",
    "stage_controls": {
        "items": [
            {"object": "planted mask [4096, 8960] bool", "bytes": 4096 * 8960},
            {"object": "float32 centred copies inside D0.clustered_order",
             "bytes": 2 * 4096 * 8960 * 4},
            {"object": "svd_lowrank workspace q=64", "bytes": 4 * 8960 * 64 * 4},
            {"object": "C4 spectrum sub-block [4096, 512] float32", "bytes": 2 * 4096 * 512 * 4}],
        "peak_gb_estimate": 0.8},
    "stage_model": {
        "items": [
            {"object": "donor weights fp32 (1.54e9 params)", "bytes": 6_170_000_000,
             "note": "fp32 is not a choice -- it is D1/D4's BPB path and comparability "
                     "requires it"},
            {"object": "load transient (embedding 151936x1536, bf16 -> fp32)",
             "bytes": 933_000_000},
            {"object": "per-batch logits, batch=1 x 512 x 151936 fp32", "bytes": 311_164_928,
             "note": "batch forced to 1 for exactly this reason; C.bpb defaults to 4"},
            {"object": "log_softmax copy of the above", "bytes": 311_164_928},
            {"object": "baseline logits kept for max|dlogit|, 2-SEQUENCE subset",
             "bytes": 622_329_856,
             "note": "the whole 24-seq slice would be 7.5 GB; the BPB comparison needs only "
                     "scalar NLL accumulation and IS run on the full slice"},
            {"object": "down_proj column-permute temporary [1536, 8960] fp32",
             "bytes": 55_050_240},
            {"object": "capture buffer, 28 layers x [16384, 2240] int16, ONE token set at a time",
             "bytes": 28 * 16384 * 2240 * 2,
             "note": "2.05 GB -- THE reason the model stage peaks near 9.6 GB.  This is the "
                     "'per-token FFN activation pattern across all layers' blow-up, held as "
                     "ranked int16 indices rather than a dense [T, N] mask (which would be "
                     "28 x 16384 x 8960 bool = 4.1 GB, and 8x that in float32)"},
            {"object": "capture values, 5 deep layers x [16384, 2240] float16",
             "bytes": 5 * 16384 * 2240 * 2},
            {"object": "fidelity X capture, 5 layers x [4096, 1536] fp32",
             "bytes": 5 * 4096 * 1536 * 4}],
        "peak_gb_estimate": 9.6},
    "stage_analyse": {
        "items": [
            {"object": "one layer's ranked idx, fit + score, [16384, k<=1792] int64",
             "bytes": 2 * 16384 * 1792 * 8, "note": "470 MB, one layer at a time"},
            {"object": "one layer's ranked val [16384, 2240] float32", "bytes": 16384 * 2240 * 4,
             "note": "freed immediately after the global-threshold realism check"},
            {"object": "dense fit mask [16384, 8960] bool for D0.clustered_order",
             "bytes": 16384 * 8960, "note": "147 MB; the only place a full mask is materialised"},
            {"object": "float32 centred copies inside D0.clustered_order",
             "bytes": 2 * 16384 * 8960 * 4, "note": "1.17 GB -- the analyse-stage peak driver"},
            {"object": "touched [T, nblocks] bool, worst case nblocks=4480",
             "bytes": 16384 * 4480},
            {"object": "block-id gather (pos[idx]//bs) [T, k] int32, worst case k=1792",
             "bytes": 16384 * 1792 * 4},
            {"object": "duplication act [2048, 8960] float32", "bytes": 2048 * 8960 * 4}],
        "peak_gb_estimate": 2.6},
    "NOT_materialised": [
        "the [T, N] per-token activation mask across ALL layers at once -- the obvious blow-up. "
        "Masks are captured as ranked int16 indices, written to disk per layer, and re-read ONE "
        "LAYER AT A TIME.",
        "logits for the full eval slice (7.5 GB) -- BPB accumulates scalars, and the logit "
        "deviation is measured on a 2-sequence subset.",
        "a second copy of the donor for the losslessness control -- the permutation is applied "
        "and inverted IN PLACE.",
        "the [T, N] mask for the block accounting itself -- block_skippable_from_idx works "
        "straight off the index array."],
    "disk": {
        "masks": "2 token sets x 28 layers x [16384, 2240] int16 = 4.1 GB, plus float16 values "
                 "for 5 deep layers x 2 sets = 0.73 GB.  Written under results/d0_masks/.",
        "note": "these are intermediates, not deliverables; they are regenerable from the "
                "pinned slices and the pinned donor revision."},
}


# ============================================================================ main
if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "inventory":
        print(json.dumps(_ALLOCATION_INVENTORY, indent=2))
        sys.exit(0)
    if stage in ("controls", "all"):
        stage_controls()
    if stage in ("model", "all"):
        stage_model()
    if stage in ("analyse", "all"):
        stage_analyse()
