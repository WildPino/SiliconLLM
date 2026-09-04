#!/usr/bin/env python3
"""S1 -- which bar predicts BPB?  A published, calibration-free FFN sparsity rule, scored in BPB.

Implements `docs/research/donor_adaptation/briefs/BRIEF_S1_WHICH_BAR_PREDICTS_BPB.md`.

----------------------------------------------------------------------------------------------
THE RULE UNDER TEST (brief section 0/2).  Per TOKEN, per LAYER, on the SwiGLU intermediate

        i  =  act(W_g x)  (*)  (W_u x)          [ this is exactly `h`, what W_d consumes ]

        m_p = argmin ||m||_0   s.t.   || m (*) i ||_1  >=  p * || i ||_1

i.e. keep the largest-|i| entries that together hold fraction p of the L1 mass; zero the rest.
ONE GLOBAL p ACROSS ALL LAYERS -- that is their method.  Per-layer fitting is a DIFFERENT
method and is arm B.  Sparsity is the fraction ZEROED, and it is read off the masks.

----------------------------------------------------------------------------------------------
ARMS.  All at MATCHED ACHIEVED SPARSITY, matched at the finest possible granularity: arm A is
run first and its per-(layer, sequence, position) DROP COUNT k is stored; every other arm is
then handed that same k and differs ONLY in WHICH k entries it drops.  So achieved sparsity is
identical across arms BY CONSTRUCTION, and the comparison is of rules, never of budgets.
(This is asserted, not assumed: `sparsity_match_max_abs_dev` must be 0.)

    A  published   keep top-(F-k) by |i|            k from the L1 rule at global p
    B  ours        keep top-(F-k) by  g = W_g x     F1's one-sided pre-activation gate rule
    C  NULL        keep a random (F-k) subset       the floor
    D  oracle      keep top-(F-k) by true |h_i|     the brief's registered ceiling
    D2 oracle+     keep top-(F-k) by |h_i|*||W_d[:,i]||   BUILDER ADDITION, see NOTE_D below
    P  PLANTED     keep the SMALLEST (F-k) by |h_i| -> drops the most critical set.  Control C2.

NOTE_D -- A CONSEQUENCE OF THE BRIEF'S OWN DEFINITIONS, REPORTED NOT HIDDEN:
    the brief's arm D ("top-k by true |h_i|") and arm A ("top-p L1 on i") select by the SAME
    score, because i IS h.  At MATCHED k they are the SAME MASK, hence the same trajectory,
    hence BPB_D == BPB_A EXACTLY.  Arm D as registered is therefore a tautology, not a ceiling.
    It is still run, because an exact tie is a strong end-to-end determinism check on the
    transplant machinery.  D2 is added BESIDE it (never instead of it) as a ceiling that is
    actually above A: the first-order output-error-optimal single-entry drop score, since
    dropping entry i perturbs the block output by h_i * W_d[:, i], whose norm is
    |h_i| * ||W_d[:, i]||.

----------------------------------------------------------------------------------------------
CONTROLS (brief section 3), none optional:
    C1 IDENTITY  p = 1.0 -> BPB must equal the unmodified model's BPB to full precision.
    C2 PLANTED   arm P at matched k -> BPB must degrade sharply.
    C3 NULL      arm C at high sparsity -> must be clearly worse than A and B.
    C4 ACHIEVED  achieved sparsity printed from the masks, per layer and aggregate, beside nominal.

----------------------------------------------------------------------------------------------
ALLOCATION INVENTORY: `_ALLOCATION_INVENTORY` at the bottom; emitted into every JSON so the
write-up cannot drift from the code.

Usage:
    python s1_sparsity_bpb.py selfcheck
    python s1_sparsity_bpb.py calib   --model <key>
    python s1_sparsity_bpb.py run     --model <key>
    python s1_sparsity_bpb.py gen     --model <key> --p 0.99
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DENSITY = os.path.abspath(os.path.join(HERE, "..", "density"))
sys.path.insert(0, DENSITY)
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

torch.set_grad_enabled(False)
torch.set_num_threads(12)

# ------------------------------------------------------------------ pre-registered constants
# The PINNED donor (D0-D4, F1, R2a all used this exact revision).  PRIMARY.
DONOR_KEY = "qwen2.5-1.5b"

MODELS = {
    # key                  repo_id                       revision (None -> resolve & PRINT)
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", None),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B", None),
    "coder-0.5b": ("Qwen/Qwen2.5-Coder-0.5B", "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"),
    "coder-1.5b": ("Qwen/Qwen2.5-Coder-1.5B", "df3ce67c0e24480f20468b6ef2894622d69eb73b"),
    "coder-3b": ("Qwen/Qwen2.5-Coder-3B", "09d9bc5d376b0cfa0100a0694ea7de7232525803"),
    "coder-7b": ("Qwen/Qwen2.5-Coder-7B", "0396a76181e127dfc13e5c5ec48a8cee09938b02"),
    # Amendment 1 A1.3 ladder.  Revisions resolve at download time and are PRINTED as
    # revision_achieved; they are not pinned here because these are not yet on this machine.
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", None),
    "qwen2.5-14b": ("Qwen/Qwen2.5-14B", None),
}

# eval slice -- the SAME pinned held-out span D0/d_baseline scored, so the baseline is comparable
EVAL_PART, EVAL_NSEQ, EVAL_SEQLEN, EVAL_SEED = "heldout", 24, 512, 1234
EVAL_BATCH = 4                   # fixes the batching; arm C's RNG is batch-INDEPENDENT anyway

# the global-p grid.  p = 1.0 is control C1.
P_GRID = (1.0, 0.9999, 0.999, 0.99, 0.98, 0.95, 0.90, 0.80, 0.70, 0.50)
# the reduced grid used on the scale ladder (compute budget)
P_GRID_LADDER = (1.0, 0.999, 0.99, 0.95, 0.90, 0.80)
# p at which the planted control C2 is run (matched budget, worst possible choice)
P_PLANT = 0.99

RNG_NULL = 20260830              # arm C seed base (STATED)
SIGMA_SEED_BPB = 0.005           # the project's noise constant
MASK_CHUNK = 512                 # rows per chunk in the mask builders (allocation control)

ARMS = ("A", "B", "C", "D", "D2")


def log(*a):
    print(*a, flush=True)


def peak_rss_gb():
    """Windows peak working set, read from the OS -- ACHIEVED, not estimated."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC),
                                               wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return c.PeakWorkingSetSize / 2**30
    except Exception:
        return None


BRIEF_REL = "docs/research/donor_adaptation/briefs/BRIEF_S1_WHICH_BAR_PREDICTS_BPB.md"
BRIEF_BRANCH = "origin/research/donor-adaptation"


def prereg_block():
    """Standing project law: the pre-registration is ON THE REMOTE before the run it governs.

    Recorded here from the repository itself, at the moment the run starts, so the manifest
    cannot claim an ordering the git history does not support.
    """
    import subprocess
    repo = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

    def git(*a):
        try:
            return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except Exception as e:
            return f"<{type(e).__name__}: {e}>"

    # the blob must be read as BYTES: text mode would newline-translate and strip, and the
    # hash of a normalised string is not the hash of the artefact.
    try:
        remote_blob = subprocess.run(["git", "-C", repo, "show", f"{BRIEF_BRANCH}:{BRIEF_REL}"],
                                     capture_output=True, timeout=60).stdout
    except Exception:
        remote_blob = b""
    local_path = os.path.join(repo, BRIEF_REL)
    local_txt = open(local_path, "rb").read() if os.path.exists(local_path) else b""
    return {
        "brief_path": BRIEF_REL,
        "branch_checked": BRIEF_BRANCH,
        "brief_on_remote_achieved": bool(remote_blob),
        "brief_commit_on_remote": git("log", "-1", "--format=%H %ci %s", BRIEF_BRANCH,
                                      "--", BRIEF_REL),
        "remote_branch_head": git("log", "-1", "--format=%H %ci %s", BRIEF_BRANCH),
        "brief_sha256_on_remote": hashlib.sha256(remote_blob).hexdigest() if remote_blob else None,
        "brief_sha256_local_worktree": hashlib.sha256(local_txt).hexdigest() if local_txt else None,
        "brief_remote_equals_worktree": bool(remote_blob) and remote_blob == local_txt,
        "brief_bytes_on_remote": len(remote_blob),
        "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_git_status_short": git("status", "--porcelain", "--", "benchmarks/donor_adaptation/s1"),
        "repo_head": git("rev-parse", "HEAD"),
        "NOTE": ("sha256 of the remote blob is computed from `git show` text, which normalises "
                 "nothing; compare it to the worktree hash to confirm the brief that governed "
                 "this run is byte-identical to the one that was pushed."),
    }


def env_block():
    import transformers
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()}


# ================================================================== the rule and the masks
def l1_keep_count(v: torch.Tensor, p: float, chunk: int = MASK_CHUNK) -> torch.Tensor:
    """m_p = argmin ||m||_0  s.t.  ||m (*) v||_1 >= p ||v||_1.   Returns the KEEP count per row.

    v : [N, F] real.  Returns int64 [N].  Chunked over rows so the cumsum buffer is bounded.
    The SORT is done in the input dtype (fp32) -- sorting is order-only, and fp64 cannot
    reorder fp32 values.  The CUMSUM is fp64 on purpose: F is ~9k-19k and an fp32 running sum
    of that many positive terms loses enough precision to move the count near p -> 1.
    """
    N, F = v.shape
    out = torch.empty(N, dtype=torch.int64, device=v.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        srt, _ = torch.sort(v[s:e].abs().float(), dim=-1, descending=True)
        cs = torch.cumsum(srt.to(torch.float64), dim=-1)
        thr = float(p) * cs[:, -1:]
        # smallest m with cs[m-1] >= thr  ==  #{cs < thr} + 1
        out[s:e] = (cs < thr).sum(dim=-1) + 1
    return out.clamp_(1, F)


def _scatter_keep(out_rows, idx, keep_rows, ar):
    out_rows.scatter_(-1, idx, ar < keep_rows.unsqueeze(-1))


def keep_mask_topk(score: torch.Tensor, keep: torch.Tensor,
                   chunk: int = MASK_CHUNK) -> torch.Tensor:
    """Boolean [N, F] keeping the `keep[n]` LARGEST entries of score[n]."""
    N, F = score.shape
    out = torch.zeros((N, F), dtype=torch.bool, device=score.device)
    ar = torch.arange(F, device=score.device).unsqueeze(0)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        idx = torch.argsort(score[s:e], dim=-1, descending=True)
        _scatter_keep(out[s:e], idx, keep[s:e], ar)
    return out


def l1_rule_mask(v: torch.Tensor, p: float, chunk: int = MASK_CHUNK):
    """Arm A in ONE sort: the L1 keep-count AND the top-|v| mask share the same ordering.

    Returns (bool mask [N, F], keep count int64 [N]).  Algebraically identical to
    keep_mask_topk(v.abs(), l1_keep_count(v, p)) -- asserted in the selfcheck -- but it sorts
    once instead of twice, and the sort is the dominant cost of the whole probe.
    """
    N, F = v.shape
    out = torch.zeros((N, F), dtype=torch.bool, device=v.device)
    keep = torch.empty(N, dtype=torch.int64, device=v.device)
    ar = torch.arange(F, device=v.device).unsqueeze(0)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        srt, idx = torch.sort(v[s:e].abs().float(), dim=-1, descending=True)
        cs = torch.cumsum(srt.to(torch.float64), dim=-1)
        k = ((cs < float(p) * cs[:, -1:]).sum(dim=-1) + 1).clamp_(1, F)
        keep[s:e] = k
        _scatter_keep(out[s:e], idx, k, ar)
    return out, keep


# ================================================================== the controller
class Ctl:
    """Global masking state.  One instance; the patched MLP forwards read it."""

    def __init__(self):
        self.mode = "off"           # off | CALIB | A | B | C | D | D2 | P
        self.p = 1.0
        self.row0 = 0               # index of the first SEQUENCE of the current batch
        self.L = 0
        self.n_seq = 0
        self.seq_len = 0
        self.kdrop = None           # int32 [L, n_seq, seq_len] -- arm A's budget, transplanted
        self.store = False          # arm A writes kdrop
        self.use_store = False      # other arms read kdrop
        self.stat_drop = None       # float64 [L] dropped entries
        self.stat_tot = None        # float64 [L] total entries
        self.calib_drop = None      # dict p -> float64 [L]  (CALIB mode, no masking applied)
        self.colnorm = {}           # layer -> [F]  ||W_d[:, i]||_2   (arm D2)
        self.identity_max_dev = 0.0  # C1: max |h_masked - h| seen, ACHIEVED
        self.check_identity = False

    def reset_stats(self):
        self.stat_drop = torch.zeros(self.L, dtype=torch.float64)
        self.stat_tot = torch.zeros(self.L, dtype=torch.float64)
        self.identity_max_dev = 0.0

    def achieved(self):
        tot = float(self.stat_tot.sum())
        per = (self.stat_drop / self.stat_tot.clamp(min=1)).tolist()
        return {"aggregate": float(self.stat_drop.sum() / max(tot, 1.0)),
                "per_layer": per,
                "n_entries_counted": tot}


CTL = Ctl()


def _null_scores(layer: int, row0: int, B: int, T: int, F: int, device=None) -> torch.Tensor:
    """Arm C.  Seeded by (layer, ABSOLUTE sequence index) only -> independent of batching.

    Always drawn on the CPU with an explicit generator and then moved, so the null mask is
    BIT-IDENTICAL on the CPU and the GPU path.  A device-side RNG would make the CPU-vs-GPU
    anchor control of Amendment 1 A1.2 compare two different nulls.
    """
    parts = []
    for b in range(B):
        g = torch.Generator().manual_seed(
            (RNG_NULL * 1000003 + layer * 100003 + (row0 + b)) % (2**63 - 1))
        parts.append(torch.rand((T, F), generator=g))
    out = torch.stack(parts, 0).reshape(B * T, F)
    return out if device is None else out.to(device)


def masked_forward(self, x):
    ctl, li = self._s1_ctl, self._s1_layer
    g = self.gate_proj(x)
    u = self.up_proj(x)
    h = self.act_fn(g) * u                      # == the brief's intermediate `i`
    mode = ctl.mode
    if mode == "off":
        return self.down_proj(h)

    shp = h.shape
    B, T, F = (shp[0], shp[1], shp[2]) if h.dim() == 3 else (1, shp[0], shp[1])
    hf = h.reshape(-1, F)
    N = hf.shape[0]

    if mode == "CALIB":                         # measure the rule, apply nothing
        for p in ctl.calib_drop:
            ctl.calib_drop[p][li] += float(F * N - l1_keep_count(hf, p).sum())
        ctl.stat_tot[li] += float(N) * F
        return self.down_proj(h)

    # ---- the budget k (number DROPPED) and the mask ---------------------------------------
    if not ctl.use_store:
        # arm A computing its OWN budget from the L1 rule.  Because arm A's score is |i| and
        # the L1 rule sorts |i| too, ONE sort serves both -- see l1_rule_mask, which the
        # selfcheck asserts is mask-identical to the naive two-sort composition.
        m, keep = l1_rule_mask(hf, ctl.p)
        if ctl.store:
            ctl.kdrop[li, ctl.row0:ctl.row0 + B, :T] = (
                (F - keep).reshape(B, T).to(device=ctl.kdrop.device, dtype=torch.int32))
    else:
        keep = (F - ctl.kdrop[li, ctl.row0:ctl.row0 + B, :T].reshape(-1)
                .to(device=hf.device, dtype=torch.int64))
        # ---- the score: which entries survive -------------------------------------------
        if mode in ("A", "D"):
            score = hf.abs()
        elif mode == "B":
            score = g.reshape(-1, F)            # signed pre-activation gate: F1's one-sided rule
        elif mode == "C":
            score = _null_scores(li, ctl.row0, B, T, F, device=hf.device)
        elif mode == "D2":
            score = hf.abs() * ctl.colnorm[li].unsqueeze(0)
        elif mode == "P":
            score = -hf.abs()                   # keep the smallest -> DROP the most critical
        else:
            raise ValueError(mode)
        m = keep_mask_topk(score, keep)
    ctl.stat_drop[li] += float(N) * F - float(m.sum())
    ctl.stat_tot[li] += float(N) * F
    hm = hf * m
    if ctl.check_identity:
        ctl.identity_max_dev = max(ctl.identity_max_dev, float((hm - hf).abs().max()))
    return self.down_proj(hm.reshape(shp))


def install(model, ctl):
    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        mlp._s1_layer = li
        mlp._s1_ctl = ctl
        mlp.forward = types.MethodType(masked_forward, mlp)
    ctl.L = len(model.model.layers)
    return ctl


# ================================================================== selfcheck (no model)
def stage_selfcheck():
    t0 = time.time()
    out = {"stage": "selfcheck", "checks": {}}
    rng = np.random.default_rng(20260830)

    # ---- 1. the L1 rule against a brute-force reference on random and adversarial rows
    for name, V in (("gauss", rng.standard_normal((300, 401))),
                    ("heavy_tail", rng.standard_normal((300, 401)) ** 5),
                    ("with_exact_zeros", rng.standard_normal((300, 401)) *
                     (rng.random((300, 401)) > 0.4))):
        v = torch.tensor(V, dtype=torch.float32)
        for p in (0.5, 0.9, 0.99, 1.0):
            got = l1_keep_count(v, p, chunk=64).numpy()
            want = np.empty(300, dtype=np.int64)
            for n in range(300):                       # brute force, definitionally
                a = np.sort(np.abs(v[n].numpy()))[::-1]
                c = np.cumsum(a.astype(np.float64))
                thr = p * c[-1]
                idx = int(np.searchsorted(c, thr, side="left"))   # first index with c >= thr
                want[n] = max(1, min(idx + 1, len(a)))
            bad = int((got != want).sum())
            out["checks"][f"l1_rule_{name}_p{p}"] = {"n_rows": 300, "n_mismatch": bad,
                                                     "mean_keep": float(got.mean())}
            assert bad == 0, (name, p, bad, got[:5], want[:5])

    # ---- 2. the L1 rule is the MINIMISER: no smaller keep-count reaches p of the L1 mass
    v = torch.tensor(rng.standard_normal((64, 401)) ** 3, dtype=torch.float32)
    for p in (0.5, 0.9, 0.99):
        k = l1_keep_count(v, p, chunk=64)
        a = np.sort(np.abs(v.numpy()), axis=1)[:, ::-1].astype(np.float64)
        c = np.cumsum(a, axis=1)
        tot = c[:, -1]
        ki = k.numpy()
        at = c[np.arange(64), ki - 1] / tot
        below = np.where(ki > 1, c[np.arange(64), np.maximum(ki - 2, 0)] / tot, 0.0)
        out["checks"][f"l1_minimal_p{p}"] = {
            "min_mass_at_k": float(at.min()), "max_mass_at_k_minus_1": float(below.max())}
        assert (at >= p - 1e-9).all(), p
        assert (below[ki > 1] < p + 1e-9).all(), p

    # ---- 3. keep_mask_topk keeps exactly `keep` entries and exactly the largest ones
    s = torch.tensor(rng.standard_normal((257, 401)), dtype=torch.float32)
    kp = torch.tensor(rng.integers(1, 402, 257))
    m = keep_mask_topk(s, kp, chunk=64)
    assert (m.sum(-1) == kp).all()
    kept_min = torch.where(m, s, torch.tensor(float("inf"))).min(-1).values
    drop_max = torch.where(~m, s, torch.tensor(float("-inf"))).max(-1).values
    sep = float((kept_min - drop_max)[kp < 401].min())
    out["checks"]["topk_mask"] = {"counts_exact": True, "min_kept_minus_max_dropped": sep}
    assert sep > 0

    # ---- 4. p = 1.0 masks NOTHING that carries value  ->  h * m == h EXACTLY (C1 mechanism)
    for trial, V in enumerate((rng.standard_normal((128, 401)),
                               rng.standard_normal((128, 401)) * (rng.random((128, 401)) > 0.3))):
        v = torch.tensor(V, dtype=torch.float32)
        m = keep_mask_topk(v.abs(), l1_keep_count(v, 1.0, chunk=64), chunk=64)
        dev = float(((v * m) - v).abs().max())
        out["checks"][f"identity_p1_exact_trial{trial}"] = {
            "max_abs_dev": dev, "mean_kept_frac": float(m.float().mean()),
            "n_dropped_all_zero": bool(
                float(v[~m].abs().max()) == 0.0 if bool((~m).any()) else True)}
        assert dev == 0.0, dev

    # ---- 5. arm A == arm D at matched k (NOTE_D), and arm P is its exact complement
    v = torch.tensor(rng.standard_normal((128, 401)) ** 3, dtype=torch.float32)
    k = l1_keep_count(v, 0.9, chunk=64)
    mA = keep_mask_topk(v.abs(), k, chunk=64)
    mD = keep_mask_topk(v.abs(), k, chunk=64)
    mP = keep_mask_topk(-v.abs(), k, chunk=64)
    out["checks"]["armA_equals_armD_at_matched_k"] = bool((mA == mD).all())
    out["checks"]["armP_drops_the_largest"] = {
        "kept_l1_mass_A": float((v.abs() * mA).sum() / v.abs().sum()),
        "kept_l1_mass_P": float((v.abs() * mP).sum() / v.abs().sum())}
    assert bool((mA == mD).all())
    assert float((v.abs() * mP).sum()) < float((v.abs() * mA).sum())

    # ---- 5b. THE FAST PATH MUST NOT CHANGE WHAT IS COMPUTED.
    # l1_rule_mask fuses the L1 count and the top-|i| mask into ONE sort.  A faster selection
    # that resolved TIES or the p-boundary differently would be a DIFFERENT METHOD, so it is
    # checked against the naive two-sort composition -- on smooth data, on heavy-tailed data,
    # on data with many exact zeros, and on heavily TIED data (values quantised to a tiny
    # alphabet, where ties are the rule rather than the exception).
    fast = {}
    cases = {
        "gauss": rng.standard_normal((256, 601)),
        "heavy_tail": rng.standard_normal((256, 601)) ** 7,
        "many_exact_zeros": rng.standard_normal((256, 601)) * (rng.random((256, 601)) > 0.6),
        "heavily_tied_8_levels": np.round(rng.standard_normal((256, 601)) * 2.0) / 2.0,
        "all_ties_one_value": np.ones((256, 601)),
        "single_spike": np.eye(256, 601) * 100.0 + 1e-6,
    }
    for name, V in cases.items():
        v = torch.tensor(V, dtype=torch.float32)
        for p in (0.5, 0.9, 0.99, 0.999, 1.0):
            mf, kf = l1_rule_mask(v, p, chunk=97)              # fused, chunk deliberately odd
            kn = l1_keep_count(v, p, chunk=64)                 # naive, different chunking
            mn = keep_mask_topk(v.abs(), kn, chunk=64)
            same_mask = bool(torch.equal(mf, mn))
            same_keep = bool(torch.equal(kf, kn))
            # what actually matters downstream: the MASKED VECTOR must be bit-identical
            same_out = bool(torch.equal(v * mf, v * mn))
            fast[f"{name}_p{p}"] = {"mask_identical": same_mask, "keep_identical": same_keep,
                                    "masked_vector_bit_identical": same_out,
                                    "n_mask_differences": int((mf != mn).sum()),
                                    "mean_keep": float(kf.double().mean())}
            assert same_keep, (name, p, "keep count differs")
            assert same_out, (name, p, "masked vector differs -- fast path is a DIFFERENT method")
            assert same_mask, (name, p, "mask differs")
    out["checks"]["fast_path_equals_naive"] = fast
    out["checks"]["FAST_PATH_VERDICT"] = (
        "IDENTICAL on %d (distribution, p) cases including all-ties and many-exact-zeros: "
        "mask, keep-count and masked vector all bit-identical to the naive two-sort form"
        % len(fast))

    # ---- 6. arm C's RNG is batch-INDEPENDENT (the transplant is only valid if it is)
    a = _null_scores(3, 0, 4, 8, 16)[8:24]        # sequences 1,2 inside a batch starting at 0
    b = _null_scores(3, 1, 2, 8, 16)              # the same two sequences, batch starting at 1
    out["checks"]["null_rng_batch_independent"] = bool(torch.equal(a, b))
    assert torch.equal(a, b)

    # ---- 7. PLANTED POSITIVE, end to end through the REAL forward path on a toy SwiGLU MLP
    class ToyMLP(torch.nn.Module):
        def __init__(self, D, F):
            super().__init__()
            self.gate_proj = torch.nn.Linear(D, F, bias=False)
            self.up_proj = torch.nn.Linear(D, F, bias=False)
            self.down_proj = torch.nn.Linear(F, D, bias=False)
            self.act_fn = torch.nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    D, F = 32, 401
    torch.manual_seed(7)
    toy = ToyMLP(D, F).eval()
    x = torch.randn(2, 40, D)
    ctl = Ctl()
    ctl.L = 1
    ctl.reset_stats()
    toy._s1_layer = 0
    toy._s1_ctl = ctl
    ref = toy.forward(x)
    toy.forward = types.MethodType(masked_forward, toy)
    ctl.colnorm[0] = toy.down_proj.weight.norm(dim=0)

    ctl.n_seq, ctl.seq_len = 2, 40
    ctl.kdrop = torch.zeros((1, 2, 40), dtype=torch.int32)

    # C1 on the real path
    ctl.mode, ctl.p, ctl.store, ctl.use_store, ctl.check_identity = "A", 1.0, True, False, True
    ctl.reset_stats()
    y1 = toy(x)
    out["checks"]["toy_C1_identity_max_abs_dev"] = float((y1 - ref).abs().max())
    out["checks"]["toy_C1_h_mask_max_abs_dev"] = ctl.identity_max_dev
    out["checks"]["toy_C1_achieved_sparsity"] = ctl.achieved()["aggregate"]
    assert float((y1 - ref).abs().max()) == 0.0

    # arm A at p = 0.9, budget stored; then every other arm at that SAME budget
    ctl.mode, ctl.p, ctl.check_identity = "A", 0.9, False
    ctl.reset_stats()
    _ = toy(x)
    accA = ctl.achieved()
    ctl.store, ctl.use_store = False, True
    res = {}
    for mode in ("A", "B", "C", "D", "D2", "P"):
        ctl.mode = mode
        ctl.reset_stats()
        y = toy(x)
        res[mode] = {"achieved": ctl.achieved()["aggregate"],
                     "rel_out_err": float((y - ref).norm() / ref.norm())}
    out["checks"]["toy_matched_arms"] = res
    out["checks"]["toy_armA_achieved_p0.9"] = accA["aggregate"]
    sp = [res[m]["achieved"] for m in res]
    out["checks"]["toy_sparsity_match_max_abs_dev"] = float(max(sp) - min(sp))
    assert max(sp) - min(sp) == 0.0
    assert res["A"]["rel_out_err"] == res["D"]["rel_out_err"]
    assert res["P"]["rel_out_err"] > 5 * res["A"]["rel_out_err"], res
    assert res["C"]["rel_out_err"] > res["A"]["rel_out_err"], res
    out["checks"]["toy_D2_beats_A_on_output_err"] = bool(
        res["D2"]["rel_out_err"] < res["A"]["rel_out_err"])
    out["checks"]["PLANTED_VERDICT"] = (
        "FIRES: at matched budget the planted mask's output error is %.1fx arm A's"
        % (res["P"]["rel_out_err"] / max(res["A"]["rel_out_err"], 1e-30)))

    out["elapsed_s"] = time.time() - t0
    out["peak_rss_gb"] = peak_rss_gb()
    out["env"] = env_block()
    out["allocation_inventory"] = _ALLOCATION_INVENTORY
    p = os.path.join(RESULTS, "s1_selfcheck.json")
    json.dump(out, open(p, "w"), indent=2, default=float)
    log("SELFCHECK PASS (%.1fs) -> %s" % (out["elapsed_s"], p))
    log(json.dumps({k: v for k, v in out["checks"].items()
                    if k.startswith("toy") or k.startswith("PLANT")}, indent=2, default=float))
    return out


# ================================================================== model plumbing
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load(key, dtype=torch.float32, device="cpu", device_map=None):
    """Load a donor.  `device_map="auto"` shards across both T4s for the 7B point.

    NOTE (Amendment 1 A1.3): weights are loaded in fp16 on GPU and fp32 on CPU and NOTHING
    ELSE -- no 8-bit, no 4-bit.  This probe measures ACTIVATION statistics, and quantising the
    weights would change the very thing under test.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev = MODELS[key]
    kw = {} if rev is None else {"revision": rev}
    tok = AutoTokenizer.from_pretrained(repo, **kw)
    extra = {"device_map": device_map} if device_map else {}
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=dtype, attn_implementation="eager", **kw, **extra)
    if not device_map and device != "cpu":
        model = model.to(device)
    model.eval()
    # ACHIEVED revision: the commit hash of the snapshot actually on disk
    from huggingface_hub import snapshot_download
    snap = snapshot_download(repo, local_files_only=True, **kw)
    meta = {"key": key, "repo_id": repo, "revision_requested": rev,
            "revision_achieved": os.path.basename(os.path.realpath(snap)),
            "snapshot_dir": snap,
            "n_layers_achieved": len(model.model.layers),
            "d_model_achieved": model.model.layers[0].mlp.gate_proj.in_features,
            "d_ffn_achieved": model.model.layers[0].mlp.gate_proj.out_features,
            "vocab_achieved": int(model.get_input_embeddings().weight.shape[0]),
            "hidden_act_achieved": model.config.hidden_act,
            "n_params_achieved": int(sum(p.numel() for p in model.parameters())),
            "dtype_achieved": str(next(model.parameters()).dtype),
            "device_achieved": str(next(model.parameters()).device),
            "device_map_requested": device_map,
            "devices_in_use_achieved": sorted({str(p.device) for p in model.parameters()}),
            "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False))}
    return model, tok, meta


TOK_PROBE = "def f(x):\n    return x**2  # é中文 tokenizer identity probe 12345\n"


def tok_fingerprint(tok):
    ids = tok(TOK_PROBE, add_special_tokens=False)["input_ids"]
    return {"probe_ids": ids,
            "probe_sha16": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
            "vocab_size": int(len(tok))}


def get_eval_slice(tok):
    import common as C
    ids, byts, meta = C.get_slice(tok, EVAL_PART, EVAL_NSEQ, EVAL_SEQLEN, EVAL_SEED)
    # GUARD: the disk cache is keyed by (part, n, len, seed) and NOT by tokenizer.  Re-encoding
    # the cached ids under THIS tokenizer must reproduce them, or the slice is not ours.
    ok = True
    for r in range(ids.shape[0]):
        dec = tok.decode(ids[r, 1:].tolist())
        if tok(dec, add_special_tokens=False)["input_ids"] != ids[r, 1:].tolist():
            ok = False
            break
    meta = dict(meta)
    meta["tokenizer_roundtrip_ok_achieved"] = ok
    meta["tokenizer_fingerprint"] = tok_fingerprint(tok)
    return ids, byts, meta


def bpb_eval(model, ids, byts):
    """BPB on the pinned slice, driving CTL.row0 so the transplanted budget lines up."""
    tot = torch.zeros((), dtype=torch.float64)
    per = []
    dev = next(model.parameters()).device
    for i in range(0, ids.shape[0], EVAL_BATCH):
        CTL.row0 = i
        chunk = ids[i:i + EVAL_BATCH].to(dev)
        out = model(chunk).logits.float()
        lp = torch.nn.functional.log_softmax(out[:, :-1], dim=-1)
        nll = -lp.gather(-1, chunk[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=1).double().cpu()
        per.append(nll)
        tot += nll.sum()
        del out, lp
    per = torch.cat(per)
    return float(tot / (math.log(2.0) * byts.sum())), (per / (math.log(2.0) * byts)).numpy()


def paired_boot(nats_a, nats_b, byts, n_boot=4000, seed=11):
    """SE of the BPB DELTA under resampling of the SEQUENCES.  PAIRED: the same resample
    indexes both arms, so the shared which-sequences-did-we-draw variance cancels.

    This is the correct error bar for an arm-vs-baseline comparison here, because every arm is
    scored on the SAME token slice and the per-sequence log-losses are therefore highly
    correlated across arms.  Comparing two MARGINAL SEs instead would badly overstate the
    uncertainty of the difference and could make a real effect look null.  The marginal SEs are
    reported too (`bpb_marginal_se`), labelled, so the gap between the two is visible.
    """
    rng = np.random.default_rng(seed)
    b = byts.numpy()
    n = len(b)
    ln2 = math.log(2.0)
    v = np.empty(n_boot)
    for j in range(n_boot):
        k = rng.integers(0, n, n)
        s = b[k].sum() * ln2
        v[j] = (nats_a[k].sum() - nats_b[k].sum()) / s
    return float(v.std())


def stats_vs_baseline(per_arm, nats_base, byts, base_bpb, bpb):
    """Both error bars, explicitly labelled, plus the correlation that makes pairing work."""
    import common as C
    ln2 = math.log(2.0)
    nats_a = per_arm * ln2 * byts.numpy()
    b = byts.numpy()
    per_base_bpb = nats_base / (ln2 * b)
    d = bpb - base_bpb
    se_p = paired_boot(nats_a, nats_base, byts)
    se_m = C.bootstrap_se(per_arm, byts)
    with np.errstate(invalid="ignore"):
        corr = float(np.corrcoef(per_arm, per_base_bpb)[0, 1])
    return {"bpb": bpb, "delta_bpb": d,
            "delta_se_PAIRED": se_p,
            "bpb_marginal_se": se_m,
            # PER-SEQUENCE nats, stored so the TOKEN-BUDGET SENSITIVITY of this instrument can
            # be re-derived from this artefact alone, with no extra forward passes.  The F1
            # audit found a(eps) moving by 0.044 between a 256- and a 1024-token budget, which
            # is larger than the cross-size effects this probe is trying to resolve, so the
            # resolution of the instrument has to be published beside every number it produces.
            "per_seq_nats": [float(x) for x in nats_a],
            "n_seq": int(len(b)), "n_tokens": int(len(b)) * EVAL_SEQLEN,
            "baseline_marginal_se": C.bootstrap_se(per_base_bpb, byts),
            "per_seq_corr_with_baseline": corr,
            "delta_over_paired_se": (d / se_p) if se_p > 0 else float("inf"),
            "delta_over_sigma_seed": d / SIGMA_SEED_BPB,
            "exceeds_sigma_seed": bool(abs(d) > SIGMA_SEED_BPB)}


# ================================================================== stage: calib
def stage_calib(key):
    """One unmodified forward.  Maps nominal p -> ACHIEVED sparsity without applying any mask."""
    t0 = time.time()
    model, tok, meta = load(key)
    log("donor: " + json.dumps(meta))
    ids, byts, smeta = get_eval_slice(tok)
    install(model, CTL)
    CTL.reset_stats()
    CTL.mode = "CALIB"
    CTL.calib_drop = {p: torch.zeros(CTL.L, dtype=torch.float64) for p in P_GRID}
    _ = bpb_eval(model, ids, byts)
    CTL.mode = "off"
    tot = CTL.stat_tot
    rows = {}
    for p in P_GRID:
        d = CTL.calib_drop[p]
        rows[str(p)] = {"p_nominal": p,
                        "achieved_sparsity_aggregate": float(d.sum() / tot.sum()),
                        "achieved_sparsity_per_layer": (d / tot).tolist()}
        log("  p=%-8s achieved sparsity = %.6f"
            % (p, rows[str(p)]["achieved_sparsity_aggregate"]))
    out = {"stage": "calib", "donor": meta, "slice": smeta, "env": env_block(),
           "p_grid": list(P_GRID), "by_p": rows, "elapsed_s": time.time() - t0,
           "peak_rss_gb": peak_rss_gb(), "allocation_inventory": _ALLOCATION_INVENTORY}
    pth = os.path.join(RESULTS, f"s1_calib_{key}.json")
    json.dump(out, open(pth, "w"), indent=2, default=float)
    log("CALIB done -> " + pth)
    return out


# ================================================================== stage: run
def stage_run(key, p_grid, do_plant=True, arms=ARMS, resume=True,
              dtype=torch.float32, device="cpu", device_map=None):
    t0 = time.time()
    model, tok, meta = load(key, dtype=dtype, device=device, device_map=device_map)
    log("donor: " + json.dumps(meta))
    ids, byts, smeta = get_eval_slice(tok)
    log("slice: " + json.dumps({k: v for k, v in smeta.items() if k != "tokenizer_fingerprint"}))
    assert smeta["tokenizer_roundtrip_ok_achieved"], "slice is not byte-exact under this tokenizer"

    install(model, CTL)
    L = CTL.L
    CTL.n_seq, CTL.seq_len = int(ids.shape[0]), int(ids.shape[1])
    _dev0 = next(model.parameters()).device
    CTL.kdrop = torch.zeros((L, CTL.n_seq, CTL.seq_len), dtype=torch.int32, device=_dev0)
    for li, layer in enumerate(model.model.layers):
        # colnorm must live on ITS OWN layer's device: device_map="auto" shards layers across
        # both cards, so a single-device table would raise on the second card's layers.
        w = layer.mlp.down_proj.weight
        CTL.colnorm[li] = w.norm(dim=0).float().to(w.device)

    ln2 = math.log(2.0)
    out = {"stage": "run", "donor": meta, "slice": smeta, "env": env_block(),
           "prereg_check": prereg_block(),
           "pre_registered": {"p_grid": list(p_grid), "arms": list(arms),
                              "eval": {"part": EVAL_PART, "n_seq": EVAL_NSEQ,
                                       "seq_len": EVAL_SEQLEN, "seed": EVAL_SEED,
                                       "batch": EVAL_BATCH},
                              "rng_null": RNG_NULL, "p_plant": P_PLANT,
                              "sigma_seed_bpb": SIGMA_SEED_BPB},
           "NOTE_D": ("arm D (top-k by |h_i|) and arm A (top-p L1 on i) select by the SAME score "
                      "because i IS h; at matched k they are the same mask and BPB_D == BPB_A "
                      "exactly.  Run anyway as a determinism check.  D2 is the Builder's "
                      "output-error-optimal ceiling, added beside D, never instead of it."),
           "results": {}, "controls": {}}

    # ---- RESUME.  This probe has been killed twice by session limits; every completed p is a
    # finished artefact and is not recomputed.  The baseline IS recomputed every time and
    # checked against the stored one -- an identical baseline across two separate processes is
    # a free determinism check on the whole forward path.
    outpath = os.path.join(RESULTS, f"s1_run_{key}.json")
    prev = None
    if resume and os.path.exists(outpath):
        try:
            prev = json.load(open(outpath))
            out["results"] = prev.get("results", {})
            out["controls"] = prev.get("controls", {})
            log("RESUME: %d p-values already on disk: %s"
                % (len(out["results"]), sorted(out["results"])))
        except Exception as e:
            log("RESUME: could not read %s (%s) -- starting clean" % (outpath, e))

    # ---- BASELINE: unmodified model, masking machinery installed but mode = off
    CTL.mode = "off"
    base, per_base = bpb_eval(model, ids, byts)
    nats_base = per_base * ln2 * byts.numpy()
    import common as C
    out["baseline_bpb"] = base
    out["baseline_slice_se"] = C.bootstrap_se(per_base, byts)
    out["baseline_per_seq_nats"] = [float(x) for x in nats_base]
    out["slice_bytes_per_seq"] = [float(x) for x in byts.numpy()]
    out["n_seq_achieved"] = int(ids.shape[0])
    out["n_tokens_achieved"] = int(ids.numel())
    log("BASELINE BPB = %.9f  (slice-resampling SE %.4f)" % (base, out["baseline_slice_se"]))
    if prev and "baseline_bpb" in prev:
        out["baseline_reproduced_across_processes"] = {
            "previous": prev["baseline_bpb"], "now": base,
            "delta": base - prev["baseline_bpb"],
            "bitwise_equal": base == prev["baseline_bpb"]}
        log("  baseline vs previous process: delta %+.3e  bitwise_equal=%s"
            % (base - prev["baseline_bpb"], base == prev["baseline_bpb"]))

    # ---- the sweep
    for p in p_grid:
        done = out["results"].get(str(p))
        if done and all(a in done for a in arms):
            log("  p=%-8s  SKIP (already complete on disk, achieved=%.6f)"
                % (p, done["A"]["achieved"]["aggregate"]))
            continue
        rec = {}
        CTL.mode, CTL.p, CTL.store, CTL.use_store = "A", p, True, False
        CTL.check_identity = (p >= 1.0)
        CTL.reset_stats()
        bA, perA = bpb_eval(model, ids, byts)
        accA = CTL.achieved()
        rec["A"] = dict(stats_vs_baseline(perA, nats_base, byts, base, bA),
                        achieved=accA,
                        identity_h_mask_max_abs_dev=(CTL.identity_max_dev if p >= 1.0 else None))
        log("  p=%-8s  A  achieved=%.6f  BPB=%.9f  d=%+.6f  (paired SE %.6f, marginal SE %.4f)"
            % (p, accA["aggregate"], bA, bA - base,
               rec["A"]["delta_se_PAIRED"], rec["A"]["bpb_marginal_se"]))
        CTL.check_identity = False

        CTL.store, CTL.use_store = False, True
        for mode in [m for m in arms if m != "A"]:
            CTL.mode = mode
            CTL.reset_stats()
            b, pr = bpb_eval(model, ids, byts)
            acc = CTL.achieved()
            rec[mode] = dict(stats_vs_baseline(pr, nats_base, byts, base, b), achieved=acc)
            log("  p=%-8s  %-2s achieved=%.6f  BPB=%.9f  d=%+.6f  (paired SE %.6f)"
                % (p, mode, acc["aggregate"], b, b - base, rec[mode]["delta_se_PAIRED"]))
        sp = [rec[m]["achieved"]["aggregate"] for m in rec]
        rec["sparsity_match_max_abs_dev"] = float(max(sp) - min(sp))
        rec["p_nominal"] = p
        out["results"][str(p)] = rec
        json.dump(out, open(os.path.join(RESULTS, f"s1_run_{key}.json"), "w"),
                  indent=2, default=float)      # checkpoint every p

    # ---- C1 IDENTITY
    r1 = out["results"].get("1.0")
    if r1:
        out["controls"]["C1_IDENTITY"] = {
            "p": 1.0, "bpb_masked": r1["A"]["bpb"], "bpb_unmodified": base,
            "delta": r1["A"]["bpb"] - base,
            "bitwise_equal": r1["A"]["bpb"] == base,
            "achieved_sparsity": r1["A"]["achieved"]["aggregate"],
            "h_mask_max_abs_dev_achieved": r1["A"]["identity_h_mask_max_abs_dev"],
            "verdict": "PASS" if r1["A"]["bpb"] == base else "FAIL -- HARNESS IS WRONG, STOP"}
        log("C1 IDENTITY: " + out["controls"]["C1_IDENTITY"]["verdict"])

    # ---- C2 PLANTED, at P_PLANT's budget
    if do_plant and str(P_PLANT) in out["results"] and "C2_PLANTED" not in out["controls"]:
        CTL.mode, CTL.p, CTL.store, CTL.use_store = "A", P_PLANT, True, False
        CTL.reset_stats()
        _ = bpb_eval(model, ids, byts)
        CTL.store, CTL.use_store = False, True
        CTL.mode = "P"
        CTL.reset_stats()
        bP, _ = bpb_eval(model, ids, byts)
        accP = CTL.achieved()
        aref = out["results"][str(P_PLANT)]["A"]
        out["controls"]["C2_PLANTED"] = {
            "p_budget": P_PLANT, "achieved_sparsity": accP["aggregate"],
            "bpb": bP, "delta_vs_baseline": bP - base,
            "delta_vs_armA_same_budget": bP - aref["bpb"],
            "ratio_to_sigma_seed": (bP - base) / SIGMA_SEED_BPB,
            "verdict": "FIRES" if (bP - base) > 20 * SIGMA_SEED_BPB else "DID NOT FIRE"}
        log("C2 PLANTED: BPB=%.6f  d=%+.6f  -> %s"
            % (bP, bP - base, out["controls"]["C2_PLANTED"]["verdict"]))

    out["elapsed_s"] = time.time() - t0
    out["peak_rss_gb"] = peak_rss_gb()
    out["allocation_inventory"] = _ALLOCATION_INVENTORY
    pth = os.path.join(RESULTS, f"s1_run_{key}.json")
    json.dump(out, open(pth, "w"), indent=2, default=float)
    log("RUN done (%.0fs, peak RSS %.2f GB) -> %s"
        % (out["elapsed_s"], out["peak_rss_gb"] or -1, pth))
    return out


# ================================================================== stage: gen
GEN_PROMPTS = [
    "def quicksort(a):\n",
    "The three most important properties of a good compression algorithm are",
    "# Installing the package\n\nTo install, run",
    "It was a bright cold day in April, and the clocks were",
]


def stage_gen(key, p, max_new=110):
    model, tok, meta = load(key)
    install(model, CTL)
    for li, layer in enumerate(model.model.layers):
        CTL.colnorm[li] = layer.mlp.down_proj.weight.norm(dim=0)
    CTL.store, CTL.use_store = False, False
    samples = []
    for mode, pp in (("off", None), ("A", p)):
        CTL.mode, CTL.p = mode, (pp or 1.0)
        CTL.reset_stats()
        for pr in GEN_PROMPTS:
            ids = tok(pr, return_tensors="pt").input_ids
            o = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            samples.append({"mode": mode, "p": pp, "prompt": pr,
                            "text": tok.decode(o[0], skip_special_tokens=True)})
            log("---- [%s p=%s] %s" % (mode, pp, json.dumps(samples[-1]["text"])[:500]))
        if mode != "off":
            samples[-1]["achieved_sparsity_over_generation"] = CTL.achieved()["aggregate"]
            log("achieved sparsity over the generated tokens = %.6f"
                % CTL.achieved()["aggregate"])
    out = {"stage": "gen", "donor": meta, "p": p, "greedy": True, "max_new_tokens": max_new,
           "samples": samples, "env": env_block(), "peak_rss_gb": peak_rss_gb()}
    pth = os.path.join(RESULTS, f"s1_gen_{key}_p{p}.json")
    json.dump(out, open(pth, "w"), indent=2, default=float)
    log("GEN done -> " + pth)
    return out


# ================================================================== allocation inventory
_ALLOCATION_INVENTORY = {
    "policy": "every allocation that scales with the input, including the ones that fit",
    "per_forward_batch": {
        "logits [B, T, V] fp32": "4*512*151936*4 B = 1.24 GB at B=4, T=512, V=151936 -- the "
                                 "LARGEST single allocation in the probe, and it is the LM "
                                 "head's, not the mask's; log_softmax allocates a second one "
                                 "of the same size, so the true transient is ~2.5 GB",
        "h / g / u  [B, T, F] fp32": "4*512*8960*4 B = 73 MB each at the 1.5B donor",
    },
    "per_layer_transient_mask": {
        "sort buffer [chunk, F] fp64": "512*8960*8 B = 37 MB  (l1_keep_count, chunk=512)",
        "argsort index [chunk, F] int64": "512*8960*8 B = 37 MB  (keep_mask_topk)",
        "bool mask [B*T, F]": "2048*8960 B = 18 MB",
        "arm C random scores [B*T, F] fp32": "2048*8960*4 B = 73 MB",
        "note": "chunked over rows at MASK_CHUNK=512 precisely so these do NOT scale with B*T",
    },
    "persistent": {
        "kdrop [L, n_seq, seq_len] int32": "28*24*512*4 B = 1.4 MB -- the transplanted budget",
        "colnorm L x [F] fp32": "28*8960*4 B = 1.0 MB",
        "stat_drop/stat_tot [L] fp64": "448 B",
        "model fp32": "n_params * 4 B; 6.2 GB at 1.5B, 12.4 GB at 3B, 30.5 GB at 7B",
    },
    "NOT_allocated": [
        "no activation cache on disk (unlike F1's capture stage) -- every arm re-runs the forward",
        "no [T, F] float mask is ever materialised; the mask is bool and multiplied in place",
    ],
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["selfcheck", "calib", "run", "gen"])
    ap.add_argument("--model", default=DONOR_KEY)
    ap.add_argument("--p", type=float, default=0.99)
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--no-plant", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--device-map", default=None)
    a = ap.parse_args()
    if a.stage == "selfcheck":
        stage_selfcheck()
    elif a.stage == "calib":
        stage_calib(a.model)
    elif a.stage == "run":
        stage_run(a.model, P_GRID_LADDER if a.ladder else P_GRID,
                  do_plant=not a.no_plant, arms=tuple(a.arms.split(",")),
                  resume=not a.no_resume, dtype=DTYPES[a.dtype], device=a.device,
                  device_map=a.device_map)
    elif a.stage == "gen":
        stage_gen(a.model, a.p)
