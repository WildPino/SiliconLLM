#!/usr/bin/env python3
"""Stage-1 donor footprint-and-throughput inventory (mandate DONOR_MODEL_ADAPTATION.md sec.9 stage 1).

Metadata-only.  Downloads config.json per candidate (a few KB) and nothing else.
Never touches a weight file.

Every figure is computed from a field that is present in the donor's own config.json.
If a needed field is absent the row is marked PARTIAL and the missing field is named.
Nothing is filled in from a remembered model card (project law 5: the artefact is the authority).

Rate constants come from docs/PHASE64_BUDGET.md sec.1 / sec.1b (measured, 3600X reference),
NOT from the naive 42 GB/s of the mandate sec.5.

Usage:
    python donor_inventory.py fetch      # config.json + repo metadata -> configs/
    python donor_inventory.py analyze    # arithmetic -> stdout markdown + inventory.json
    python donor_inventory.py control    # planted-control suite (must FIRE)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, "configs")

# ---------------------------------------------------------------------------
# Candidate list (mandate sec.8.A: >= ~15 real models spanning size + family)
# ---------------------------------------------------------------------------
CANDIDATES = [
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B",
    "microsoft/Phi-3-mini-4k-instruct",
    "allenai/OLMo-2-1124-7B",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen3-30B-A3B",
    "mistralai/Mixtral-8x7B-v0.1",
    "deepseek-ai/DeepSeek-V2-Lite",
    "allenai/OLMoE-1B-7B-0924",
    "openai/gpt-oss-20b",
    "ai21labs/AI21-Jamba-Mini-1.6",
    "Zyphra/Zamba2-2.7B",
    "ibm-granite/granite-4.0-h-small",
    "nvidia/Nemotron-H-8B-Base-8K",
    "tiiuae/Falcon-H1-7B-Base",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "state-spaces/mamba2-2.7b",
]

# ---------------------------------------------------------------------------
# MEASURED rate constants -- docs/PHASE64_BUDGET.md
# ---------------------------------------------------------------------------
# sec.1b(a): proj-GEMV fp32 rate vs matrix size, t6, row-partitioned, x L1-resident.
# The published sec.1b row is KEPT for reference but is NO LONGER the pricing curve.  It was
# re-measured (docs/research/DONOR_PROJ_RATE.md sec.2): the t1 row reproduces at all 8 points and
# the 96 MB anchor reproduces (+4.9%), but the 16-32 MB cache-transition points move by up to 4.6x
# on OpenMP thread PLACEMENT alone, and `engine.c --gemv-sweep` -- the routine that produced them --
# does not reproduce its own mid-region either.  They are placement-conditioned, not machine facts.
RCURVE_MB_PUBLISHED = [4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0]
RCURVE_GBS_PUBLISHED = [187.0, 185.0, 134.4, 60.5, 55.7, 45.5, 45.3, 36.5]

# DONOR_PROJ_RATE.md sec.4 -- directly MEASURED, 4 independent runs, min-of-5 estimator each.
# 32 MB is the aggregate-L3 cliff: between-run sd 22.2%, declared NON-REPORTABLE and never used.
RCURVE_MB = [4.0, 8.0, 16.0, 24.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0]
RCURVE_GBS = [184.8, 193.3, 199.2, 192.2, 41.1, 38.2, 39.0, 39.7, 39.3, 38.2, 38.9, 38.4, 38.3, 38.3]

# ---------------------------------------------------------------------------
# THE TWO MEASURED PROJECTION RATES -- DIFFERENT QUANTITIES, DIFFERENT SLOTS.
# Conflating them is precisely the defect this re-run exists to remove.
#
# (1) PROJ_STREAM_* -- "the real donor stream".  DONOR_PROJ_RATE.md sec.5.2.
#     Qwen2.5-1.5B's 28 x (q,k,v,o) + head allocated contiguously and walked IN LAYER ORDER;
#     1,550,057,472 B/token, byte accounting EXACT against the F1 hand-derivation.  Measured
#     37.74 +/- 0.18 GB/s (3 invocations, cv 1.1-4.4%) = 41.07 ms/token.  It already carries the
#     cost of interleaving the two skinny K/V organs between the square ones -- which is why it
#     sits ~3% under (2).
#     ==> USED FOR: the whole per-token proj+head time term.  Same shape of object as the term
#         models, so it is the correct slot.
# ADOPTED from CONTROLLER_PROJRATE_AUDIT.md: the audit re-derived this INDEPENDENTLY (its own
# harness, its own source file) and measured 39.87 +/- 0.09 where this Builder reported 37.74.
# I under-reported; the audit's figure is the one carried here.  My 37.74 is retained below as the
# conservative floor, so nothing downstream depends on taking the higher number on trust.
PROJ_STREAM_MEAS = 39.87
PROJ_STREAM_SD = 0.09
PROJ_STREAM_BUILDER = 37.74   # this Builder's independent measurement -- kept as the low edge
# Interval spans BOTH independent measurements, not just one lab's error bars.
PROJ_STREAM_LO = PROJ_STREAM_BUILDER - 2 * 0.18            # 37.38 -- Builder's low edge
PROJ_STREAM_HI = PROJ_STREAM_MEAS + 2 * PROJ_STREAM_SD     # 40.05 -- audit's high edge
#
# (2) PROJ_ASYMPTOTE_* -- "the size-swept asymptote".  DONOR_PROJ_RATE.md sec.4.
#     38.84 +/- 0.68 GB/s over 36 measurements (9 sizes >= 64 MB x 4 runs), FLAT across 16x of
#     size from 64 MB to 1024 MB; no downward trend, no further cliff past 96 MB.
#     ==> USED FOR: a SINGLE organ priced by its own size (e.g. the head-alone row of T7).
#         The interval is widened DOWN to 36.0 to span the measured per-organ spread of
#         36.0-39.0 GB/s (DONOR_PROJ_RATE.md sec.5.1, streamed column, 8 real donor organs).
PROJ_ASYMPTOTE_MEAS = 39.18   # ADOPTED from the audit (Builder measured 38.84)
PROJ_ASYMPTOTE_SD = 0.68
PROJ_ORGAN_LO = 36.0                                         # measured min over 8 donor organs
PROJ_ORGAN_HI = PROJ_ASYMPTOTE_MEAS + 2 * PROJ_ASYMPTOTE_SD  # 40.20
#
# RETIRED: PROJ_STREAMED_FLOOR = 34.0.  Never measured -- it was the bottom edge of an ASSERTED
# [34-40] bracket, and standing on that edge is what produced "zero donors pass".  It is 11%
# below the measured donor stream (1).
PROJ_STREAMED_FLOOR_RETIRED = 34.0

# sec.1 / sec.1b(b): ternary LUT path.
# DONOR_PROJ_RATE.md sec.8 re-measured this at DONOR dimensions and the headline changed: the path
# is BANDWIDTH-bound at donor working sets, not compute-bound.  Holding the kernel fixed and
# growing the working set 1.1 MB -> 512 MB drops the rate 3.6x (96.4 -> 27.3 GB/s).  Controller #2's
# "compute-bound by ~16x" was measured at D=256, where the kernel's own ceiling is only 29 GB/s;
# at D=1536 that ceiling is 97 GB/s, so the claim does not transfer across D.
#
# BOTH old endpoints were D=256 artefacts.  The new bracket is asymmetric ON PURPOSE and the
# caveat must stay visible: the LOW end is engine-integrated (it carries dequant, dReLU, per-row
# scale multiply, combine); the HIGH end is kernel-pure at donor shapes.  They are NOT the same
# quantity and this tool CANNOT say where between them an integrated donor lands.
# MEASURED at donor dimensions -- DONOR_PROJ_RATE.md sec.10.  This replaces the old
# [11.398 .. 27.64] bracket, which spanned two quantities NEITHER of which was the answer:
#   - 11.398 was engine-integrated at D=256 AND, it turns out, derived from the dReLU ROW-SKIP
#     path (the published 207 us is the --skip on --exp fast config).  A SiLU-gated donor cannot
#     use that sparsity, so it was wrong twice: wrong D, wrong path.
#   - 27.64 was kernel-pure, i.e. excluding the quant / LUT-build / scale-multiply / dReLU envelope.
# The integrated rate at donor D (D=1536, ffn=8960, L=28, skip=off) measures 27.31 +/- 0.46 ms/token
# = 21.25 +/- 0.36 GB/s over 4 runs (between-run sd 1.7%).  Integration overhead at donor D is only
# +30.5%, not the 2.4x the bracket assumed.  With skip=off the block is branch-free, so the rate does
# NOT depend on the synthetic weights or on activation sparsity.
LUT_DENSE_MEAS = 21.25
LUT_DENSE_SD = 0.36
LUT_DENSE_LO = LUT_DENSE_MEAS - 2 * LUT_DENSE_SD   # 20.53
LUT_DENSE_HI = LUT_DENSE_MEAS + 2 * LUT_DENSE_SD   # 21.97
LUT_DENSE_RETIRED = 2359296 / 207e-6 / 1e9         # 11.398, kept only so its retirement is visible
LUT_MOE_LO = 4.2                        # engine-integrated routed MoE today -- dispatch/gather
                                        # ALREADY INSIDE this number (see EXPERT_DISPATCH_US)
LUT_MOE_HI = 26.4                       # most conservative donor-shape streamed block, 512 MB
                                        # pool, i.i.d. gather (DONOR_PROJ_RATE sec.8.2)
# Legacy names kept so nothing silently reads a stale constant.
LUT_GBS_PESS = LUT_MOE_LO
LUT_GBS_OPT = 17.0                      # kernel-pure t6 at the D=256 SANDBOX shape (64.1b) --
                                        # reproduced at 18.6-18.9 (DONOR_PROJ_RATE sec.8.1)
DENSE_LUT_GBS = LUT_DENSE_MEAS
EXPERT_DISPATCH_US = 8.4     # decomposed overhead per expert call (64.1b).  NOTE: this is ALREADY
                             # contained in LUT_MOE_LO=4.2 (engine-integrated).  The previous
                             # revision added it ON TOP of 4.2 -- a double count.  It is now added
                             # only at the kernel-pure end, where it is genuinely absent.

# sec.1: DRAM cold-stream aggregate ceiling, saturated at 3 threads.
DRAM_AGG_LO = 40.0
DRAM_AGG_HI = 44.0
DRAM_AGG_MID = 42.0

# ---------------------------------------------------------------------------
# AUDIT C12 (BLOCK): PHASE64_BUDGET sec.2 lists FIVE components; this tool charged four.
# scan-recurrence and SWA/attention were absent, and the headroom above the gate is ~2 ms.
# Both are added here in sec.2's own parametric form.
#
#   scan-recur: t = 46 us * (Dn*N*L) / (512*96*6)      at t6, Dn = 2D
#   SWA:        t = 65 us * (D/256) * n_swa            window LOCKED at 128
#
# Verified against the live engine today (DONOR_PROJ_RATE sec.10.1): at D=256/L=6 the E3.5 config
# measures scan-recur 36.5 us and SWA-attn 48.5 us against sec.2's 46 and 65 -- the forms are the
# engine's own and they reproduce.
SCAN_US_ANCHOR = 46.0        # t6, at Dn=512 N=96 L=6
SCAN_ANCHOR_DEN = 512 * 96 * 6
SSM_STATE_N = 96             # Arch-A's SSM state width; a DESIGN VARIABLE for a converted donor
SWA_US_ANCHOR = 65.0         # t6, per attention layer, at D=256, window 128
SWA_WINDOW = 128
# A RETAINED FULL-ATTENTION layer is NOT an SWA layer.  sec.2's 65 us prices a window-128 layer;
# full attention over a context C reads/streams C/128 times as much.  Charging full attention at
# SWA rates would understate it by ~256x at 32K.  The scaling below is a DECLARED LINEAR
# EXTRAPOLATION over 256x and is NOT measured -- it is flagged wherever it is used.
FULL_ATTN_IS_EXTRAPOLATED = True

# ---------------------------------------------------------------------------
# AUDIT F4 (BLOCK): converting an attention layer to SSM does NOT remove weights -- it ADDS them.
# Arch-A's SSM block, from engine.c:205-208, with DN=2D, N=96, DTR=16, CONV=4:
#     in_proj 2*DN*D | conv_w DN*CONV | conv_b DN | x_proj (DTR+2N)*DN | dt_proj DN*DTR
#     dt_b DN | A DN*N | Dskip DN | out_proj D*DN
# At D=1536 that is 15,160,320 weights/layer against attention's q+k+v+o = 5,505,024 -- 2.75x.
# Under the tool's own P61/D9 map those SSM projections stay fp32 (Phase 61 double-fail), so a
# converted layer costs ~+38.6 MB/token MORE than the attention layer it replaced.
# The previous revision charged the conversion its BENEFITS (KV removed, scan instead of attention)
# and NONE of its WEIGHTS.  Corrected here.
SSM_DTR = 16
SSM_CONV = 4


def ssm_block_params(D, N=SSM_STATE_N):
    DN = 2 * D
    return (2 * DN * D + DN * SSM_CONV + DN + (SSM_DTR + 2 * N) * DN
            + DN * SSM_DTR + DN + DN * N + DN + D * DN)


# The glue term is DECLARED, NOT MEASURED (audit F8 item 3): 7 us at L=6 scaled linearly to models
# 6x deeper.  Carried with a x2 interval so it cannot masquerade as a measured quantity.
GLUE_INTERVAL_FACTOR = 2.0

# rho-law granularity escape (mandate sec.2.2): bulk-contiguous >= ~48 KB chunks pay ~2.5x, not 14x.
RHO_SAFE_CHUNK_KB = 48.0

# Precision maps.
B_TERNARY = 0.5              # 4-bit g=2 packed codes, the engine's real packing
B_FP32 = 4.0
B_FP16 = 2.0
SCALE_BYTES_PER_ROW = 4.0    # per-row fp32 scale, one per output row per ternary matrix
PAD_ELEMS = 32               # engine pads row length to a multiple of 32 (MPAD_* in engine.c)

# Sealed gates (mandate S2 / S4).
SKU_A_BYTES = 16 * 1024**3
SKU_B_BYTES = 64 * 1024**3
TOKS_GATE = 10.0
# Declared, not measured: OS/allocator + activations/scratch margin charged to every SKU verdict.
OS_MARGIN_BYTES = 1.0 * 1024**3

GB = 1024.0**3
MB = 1024.0**2
KB = 1024.0


def r_proj(size_bytes: float, resident_ok: bool = False):
    """(lo GB/s, hi GB/s, is_extrapolated, central GB/s) for a fp32 GEMV of `size_bytes`.

    DEFAULT (resident_ok=False) is THE DONOR CASE and returns the STREAMED rate at every size.

    DONOR_PROJ_RATE.md sec.5.1 measured this directly on eight real Qwen2.5-1.5B organs: once an
    organ is streamed, SIZE AND SHAPE STOP MATTERING.  Skinny 256x1536 K/V, square 1536x1536 Q/O,
    the 52.5 MB MLP slabs and the 890 MB head ALL land in 36.0-39.0 GB/s -- an 8% spread.  The
    skinny-matrix and huge-head worries are falsified by measurement, not argued away.

    The resident branch of the old curve (143-185 GB/s at 1.5-9 MB) is 4x MISLEADING for any
    donor: a decode token walks >1 GB of weights, so no organ survives L3 from one layer to the
    next.  Pricing a donor organ off the resident curve overstates it by ~4x.  resident_ok=True
    exists only for an organ genuinely resident for the WHOLE token -- no donor organ is, and
    nothing on the donor path sets it.
    """
    size_mb = size_bytes / MB
    extrap = size_mb > RCURVE_MB[-1]
    if not resident_ok or size_mb >= RCURVE_MB[-1]:
        # streamed: flat, measured, size-independent out to 1024 MB.
        return PROJ_ORGAN_LO, PROJ_ORGAN_HI, extrap, PROJ_ASYMPTOTE_MEAS
    if size_mb <= RCURVE_MB[0]:
        return PROJ_ORGAN_LO, RCURVE_GBS[0], False, RCURVE_GBS[0]
    for i in range(1, len(RCURVE_MB)):
        if size_mb <= RCURVE_MB[i]:
            x0, x1 = RCURVE_MB[i - 1], RCURVE_MB[i]
            y0, y1 = RCURVE_GBS[i - 1], RCURVE_GBS[i]
            opt = y0 + (y1 - y0) * (size_mb - x0) / (x1 - x0)
            return PROJ_ORGAN_LO, opt, False, opt
    raise AssertionError("unreachable")


def _t(nbytes, rate_lo, rate_hi, rate_central=None):
    """(slow us, fast us, central us) for `nbytes` against a rate bracket.

    With no central rate the central TIME is the midpoint of the TIME interval -- the neutral
    choice when the terms are about to be SUMMED.  Deliberately not the midpoint of the RATE
    interval, which would silently favour the fast end.
    """
    if nbytes is None:
        return None, None, None
    t_slow = nbytes / (rate_lo * 1e9) * 1e6
    t_fast = nbytes / (rate_hi * 1e9) * 1e6
    t_ce = nbytes / (rate_central * 1e9) * 1e6 if rate_central else 0.5 * (t_slow + t_fast)
    return t_slow, t_fast, t_ce


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------
def slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def cmd_fetch(args) -> int:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )

    os.makedirs(CFG_DIR, exist_ok=True)
    api = HfApi()
    index = {}
    for repo in CANDIDATES:
        rec = {"repo_id": repo, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            path = hf_hub_download(repo, "config.json", repo_type="model")
        except GatedRepoError as e:
            rec.update(status="UNAVAILABLE", reason="GATED: " + str(e).splitlines()[0][:200])
            index[repo] = rec
            print(f"[GATED]  {repo}")
            continue
        except RepositoryNotFoundError as e:
            rec.update(status="UNAVAILABLE", reason="404/NOT_FOUND: " + str(e).splitlines()[0][:200])
            index[repo] = rec
            print(f"[404]    {repo}")
            continue
        except EntryNotFoundError as e:
            rec.update(status="UNAVAILABLE", reason="NO config.json: " + str(e).splitlines()[0][:200])
            index[repo] = rec
            print(f"[NOCFG]  {repo}")
            continue
        except Exception as e:  # network / auth / anything else -- record, do not work around
            rec.update(status="UNAVAILABLE", reason=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
            index[repo] = rec
            print(f"[ERR]    {repo}: {type(e).__name__}")
            continue

        raw = open(path, "rb").read()
        dest = os.path.join(CFG_DIR, slug(repo) + ".json")
        with open(dest, "wb") as f:
            f.write(raw)

        # resolved revision SHA: the snapshot directory name in the hub cache
        sha = None
        parts = os.path.normpath(path).split(os.sep)
        if "snapshots" in parts:
            sha = parts[parts.index("snapshots") + 1]

        st_total, st_params, lic = None, None, None
        try:
            info = api.model_info(repo, expand=["safetensors", "cardData", "sha"])
            sha = getattr(info, "sha", None) or sha
            if getattr(info, "safetensors", None):
                st_total = info.safetensors.total
                st_params = dict(info.safetensors.parameters or {})
            cd = getattr(info, "cardData", None) or {}
            lic = cd.get("license")
        except Exception as e:
            rec["metadata_warning"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"

        rec.update(
            status="OK",
            revision_sha=sha,
            config_file=os.path.relpath(dest, HERE).replace("\\", "/"),
            config_sha256=hashlib.sha256(raw).hexdigest(),
            config_bytes=len(raw),
            safetensors_total_params=st_total,
            safetensors_params_by_dtype=st_params,
            license=lic,
        )
        index[repo] = rec
        print(f"[OK]     {repo}  sha={sha}  st_total={st_total}")

    with open(os.path.join(CFG_DIR, "_manifest.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"\nmanifest -> {os.path.join(CFG_DIR, '_manifest.json')}")
    return 0


# ---------------------------------------------------------------------------
# config reading with provenance -- the artefact is the authority (law 5)
# ---------------------------------------------------------------------------
class MissingConfigField(Exception):
    """Raised when a figure needs a config key that the donor's config.json does not carry.

    This is the tool's refusal path: it never substitutes a remembered or default value
    for a load-bearing field.
    """


@dataclass
class Reader:
    cfg: dict
    prov: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    partial: list = field(default_factory=list)

    def need(self, key, figure):
        if key not in self.cfg or self.cfg[key] is None:
            raise MissingConfigField(f"{key} (required for: {figure})")
        self.prov.setdefault(figure, []).append(key)
        return self.cfg[key]

    def opt(self, key, figure, default=None, mark_partial=False):
        if key in self.cfg and self.cfg[key] is not None:
            self.prov.setdefault(figure, []).append(key)
            return self.cfg[key]
        if mark_partial:
            self.partial.append(f"{figure}: key '{key}' absent")
        return default

    def note(self, s):
        self.notes.append(s)


@dataclass
class Mat:
    """One weight matrix class.  out x in, `count` copies in the model,
    `per_token` of them read per generated token."""
    name: str
    organ: str          # emb | head | attn_proj | ssm_proj | mlp | expert | norm
    out: int
    inn: int
    count: int
    per_token: float

    @property
    def params(self):
        return self.out * self.inn * self.count


def pad32(n):
    return ((n + PAD_ELEMS - 1) // PAD_ELEMS) * PAD_ELEMS


def mat_bytes(m: Mat, ternary: bool, copies: float):
    """(payload bytes, scale bytes, padding-overhead bytes) for `copies` of matrix m."""
    if not ternary:
        return m.out * m.inn * B_FP32 * copies, 0.0, 0.0
    ideal = m.out * m.inn * B_TERNARY * copies
    padded = m.out * pad32(m.inn) * B_TERNARY * copies
    scales = m.out * SCALE_BYTES_PER_ROW * copies
    return ideal, scales, padded - ideal


# organs that are ternary under the P61/D9 map (map b)
TERNARY_MAP_B = {"mlp", "expert"}
# under map (a) everything is ternary, taken literally
TERNARY_MAP_A = {"emb", "head", "attn_proj", "ssm_proj", "mlp", "expert", "norm"}

DENSE_TRANSFORMER_TYPES = {
    "llama", "qwen2", "qwen3", "mistral", "phi3", "olmo2",
    "qwen3_moe", "mixtral", "olmoe", "gpt_oss", "deepseek_v2",
}


def mamba2_mixer(R: Reader, d_model, d_inner, n_heads, d_state, n_groups, d_conv, prefix):
    """Mamba-2 mixer weight matrices, all shapes read from config keys.

    in_proj : d_model -> 2*d_inner + 2*n_groups*d_state + n_heads   (z, x, B, C, dt)
    conv1d  : depthwise over (d_inner + 2*n_groups*d_state) channels, width d_conv
    A_log, D, dt_bias : n_heads each ; norm : d_inner
    out_proj: d_inner -> d_model
    """
    out = []
    proj_in = 2 * d_inner + 2 * n_groups * d_state + n_heads
    out.append(Mat(f"{prefix}.in_proj", "ssm_proj", proj_in, d_model, 1, 1))
    out.append(Mat(f"{prefix}.out_proj", "ssm_proj", d_model, d_inner, 1, 1))
    conv_ch = d_inner + 2 * n_groups * d_state
    out.append(Mat(f"{prefix}.conv1d", "ssm_proj", conv_ch, d_conv, 1, 1))
    out.append(Mat(f"{prefix}.misc", "norm", 3 * n_heads + d_inner, 1, 1, 1))
    return out


def build_matrices(R: Reader):
    """Return (matrices, meta) where meta carries the layer-type map and MoE facts.

    Everything here is derived from keys present in the donor's own config.json.
    """
    cfg = R.cfg
    mt = cfg.get("model_type")
    if mt is None and "ssm_cfg" in cfg:
        mt = "mamba_ssm"          # state-spaces raw config, identified by its own keys
    mats = []
    meta = {"model_type": mt}

    # ---- state-spaces raw Mamba-2 -----------------------------------------
    if mt == "mamba_ssm":
        D = R.need("d_model", "hidden_size")
        L = R.need("n_layer", "n_layers")
        V = R.need("vocab_size", "vocab")
        mult = R.opt("pad_vocab_size_multiple", "vocab", 1)
        if mult and V % mult:
            V = ((V + mult - 1) // mult) * mult
            R.note(f"vocab padded to {V} by pad_vocab_size_multiple={mult}")
        tie = R.opt("tie_embeddings", "tie", False)
        attn_idx = R.opt("attn_layer_idx", "layer_types", [])
        meta["layer_types"] = ["attn" if i in (attn_idx or []) else "ssm" for i in range(L)]
        meta["layer_types_source"] = "attn_layer_idx"
        d_inner = 2 * D
        d_state = 128
        R.note("mamba2 d_state/headdim/ngroups NOT in config (ssm_cfg={'layer':'Mamba2'}); "
               "Mamba-2 reference defaults d_state=128, headdim=64, ngroups=1 assumed -> row is PARTIAL")
        R.partial.append("ssm mixer: ssm_cfg carries no d_state/headdim/ngroups")
        n_heads = d_inner // 64
        for i in range(L):
            mats += mamba2_mixer(R, D, d_inner, n_heads, d_state, 1, 4, f"L{i}.mixer")
        mats.append(Mat("emb", "emb", V, D, 1, 0))
        if not tie:
            mats.append(Mat("head", "head", V, D, 1, 1))
        else:
            mats.append(Mat("head(tied)", "head", V, D, 0, 1))
        mats.append(Mat("norms", "norm", D, 1, L + 1, L + 1))
        meta.update(hidden_size=D, n_layers=L, vocab=V, tied=bool(tie),
                    n_attn_layers=len(attn_idx or []), kv_kind="none")
        return mats, meta

    # ---- everything else: hidden_size / num_hidden_layers / vocab_size ----
    D = R.need("hidden_size", "hidden_size")
    L = R.need("num_hidden_layers", "n_layers")
    V = R.need("vocab_size", "vocab")
    tie = bool(R.opt("tie_word_embeddings", "tie", False))
    n_heads = R.opt("num_attention_heads", "attn", None)
    n_kv = R.opt("num_key_value_heads", "attn", n_heads)
    hd = R.opt("head_dim", "head_dim") or R.opt("attention_head_dim", "head_dim")
    if hd is None and n_heads:
        hd = D // n_heads
        R.prov.setdefault("head_dim", []).append("hidden_size/num_attention_heads")

    # ---- layer type map ---------------------------------------------------
    types, src = None, None
    if isinstance(cfg.get("layer_types"), list):
        raw = cfg["layer_types"]
        types = ["ssm" if str(t).startswith("mamba") else "attn" for t in raw]
        meta["attn_subtype"] = raw
        src = "layer_types"
    elif isinstance(cfg.get("layers_block_type"), list):
        raw = cfg["layers_block_type"]
        types = ["attn" if t == "hybrid" else ("ssm" if t == "mamba" else str(t)) for t in raw]
        meta["attn_subtype"] = raw
        src = "layers_block_type"
    elif isinstance(cfg.get("hybrid_override_pattern"), str):
        pat = cfg["hybrid_override_pattern"]
        m = {"M": "ssm", "*": "attn", "-": "mlp"}
        types = [m.get(ch, "?") for ch in pat]
        src = "hybrid_override_pattern"
    elif cfg.get("full_attention_interval"):
        k = cfg["full_attention_interval"]
        types = ["attn" if (i + 1) % k == 0 else "linattn" for i in range(L)]
        src = "full_attention_interval"
    elif mt == "falcon_h1":
        if cfg.get("attn_layer_indices") is None:
            types = ["attn+ssm"] * L
            src = "attn_layer_indices=null -> parallel hybrid, attention on every layer (INFERRED)"
            R.note("falcon_h1: attn_layer_indices is null; read as 'no restriction' => attention in "
                   "every layer, parallel with the mamba mixer. INFERENCE, corroborated only by the "
                   "param cross-check.")
        else:
            idx = set(cfg["attn_layer_indices"])
            types = ["attn" if i in idx else "ssm" for i in range(L)]
            src = "attn_layer_indices"
    elif mt in DENSE_TRANSFORMER_TYPES:
        types = ["attn"] * L
        src = "no layer-type key present and model_type is a dense-transformer family"
    if types is None:
        R.partial.append("layer_types: no layer-type map key in config (looked for layer_types, "
                         "layers_block_type, hybrid_override_pattern, full_attention_interval, "
                         "attn_layer_indices)")
        types = ["?"] * L
        src = "UNKNOWN"
    if len(types) != L:
        R.note(f"layer-type map length {len(types)} != num_hidden_layers {L}")
    meta["layer_types"] = types
    meta["layer_types_source"] = src
    n_attn = sum(1 for t in types if "attn" in t and t != "linattn")
    meta["n_attn_layers"] = n_attn
    meta["n_ssm_layers"] = sum(1 for t in types if "ssm" in t)
    meta["n_linattn_layers"] = sum(1 for t in types if t == "linattn")

    # ---- attention projections -------------------------------------------
    kv_kind = "gqa"
    if cfg.get("kv_lora_rank") is not None:          # MLA (DeepSeek-V2 family)
        kv_kind = "mla"
        kvr = R.need("kv_lora_rank", "attn")
        qk_nope = R.need("qk_nope_head_dim", "attn")
        qk_rope = R.need("qk_rope_head_dim", "attn")
        vhd = R.need("v_head_dim", "attn")
        qlr = cfg.get("q_lora_rank")
        qdim = n_heads * (qk_nope + qk_rope)
        if qlr:
            mats.append(Mat("q_a", "attn_proj", qlr, D, n_attn, n_attn))
            mats.append(Mat("q_b", "attn_proj", qdim, qlr, n_attn, n_attn))
        else:
            mats.append(Mat("q", "attn_proj", qdim, D, n_attn, n_attn))
        mats.append(Mat("kv_a", "attn_proj", kvr + qk_rope, D, n_attn, n_attn))
        mats.append(Mat("kv_b", "attn_proj", n_heads * (qk_nope + vhd), kvr, n_attn, n_attn))
        mats.append(Mat("o", "attn_proj", D, n_heads * vhd, n_attn, n_attn))
        meta["kv_elems_per_tok_per_layer"] = kvr + qk_rope
    elif n_attn:
        if n_heads is None:
            raise MissingConfigField("num_attention_heads (required for: attention projections)")
        qdim = n_heads * hd
        kvdim = (n_kv or n_heads) * hd
        adim = R.opt("attention_hidden_size", "attn")     # zamba2
        src_d = adim or D
        mats.append(Mat("q", "attn_proj", qdim, src_d, n_attn, n_attn))
        mats.append(Mat("k", "attn_proj", kvdim, src_d, n_attn, n_attn))
        mats.append(Mat("v", "attn_proj", kvdim, src_d, n_attn, n_attn))
        mats.append(Mat("o", "attn_proj", D, qdim, n_attn, n_attn))
        if cfg.get("attention_bias") or mt == "qwen2":
            if not cfg.get("attention_bias"):
                R.note("qwen2 carries QKV biases with no attention_bias key -- family assumption, "
                       "declared; worth 4e-5 of the total, adjudicated by the cross-check")
            mats.append(Mat("qkv_bias", "norm", qdim + 2 * kvdim, 1, n_attn, n_attn))
        meta["kv_elems_per_tok_per_layer"] = 2 * (n_kv or n_heads) * hd
    else:
        meta["kv_elems_per_tok_per_layer"] = 0
    meta["kv_kind"] = kv_kind
    meta["n_kv_heads"] = n_kv
    meta["head_dim"] = hd

    # ---- linear-attention layers (qwen3_next gated deltanet) --------------
    if meta["n_linattn_layers"]:
        lk = R.need("linear_key_head_dim", "linattn")
        lv = R.need("linear_value_head_dim", "linattn")
        lnk = R.need("linear_num_key_heads", "linattn")
        lnv = R.need("linear_num_value_heads", "linattn")
        nl = meta["n_linattn_layers"]
        qk = lnk * lk
        vv = lnv * lv
        mats.append(Mat("lin_in_qkvz", "ssm_proj", 2 * qk + 2 * vv, D, nl, nl))
        mats.append(Mat("lin_in_ba", "ssm_proj", 2 * lnv, D, nl, nl))
        mats.append(Mat("lin_conv", "ssm_proj", 2 * qk + vv,
                        R.need("linear_conv_kernel_dim", "linattn"), nl, nl))
        mats.append(Mat("lin_out", "ssm_proj", D, vv, nl, nl))
        R.note("linear-attention (gated DeltaNet) layers carry an O(1) recurrent state, not a "
               "growing KV cache; their state is not charged to the KV table.")

    # ---- SSM mixers -------------------------------------------------------
    n_ssmish = meta["n_ssm_layers"] + (L if all("attn+ssm" == t for t in types) else 0)
    if meta["n_ssm_layers"] or any(t == "attn+ssm" for t in types):
        nl = meta["n_ssm_layers"] or sum(1 for t in types if t == "attn+ssm")
        d_state = (R.opt("mamba_d_state", "ssm") or R.opt("ssm_state_size", "ssm")
                   or R.opt("mamba_d_state", "ssm"))
        exp = R.opt("mamba_expand", "ssm") or R.opt("expand", "ssm")
        d_inner = R.opt("mamba_d_ssm", "ssm")
        mh = R.opt("mamba_n_heads", "ssm") or R.opt("mamba_num_heads", "ssm") or R.opt("n_mamba_heads", "ssm")
        mhd = (R.opt("mamba_d_head", "ssm") or R.opt("mamba_head_dim", "ssm")
               or R.opt("mamba_headdim", "ssm"))
        ng = R.opt("mamba_n_groups", "ssm") or R.opt("n_groups", "ssm") or R.opt("mamba_ngroups", "ssm")
        dconv = R.opt("mamba_d_conv", "ssm") or R.opt("conv_kernel", "ssm")
        missing = [k for k, v in [("d_state", d_state), ("n_heads", mh), ("d_head", mhd),
                                  ("n_groups", ng), ("d_conv", dconv)] if v is None]
        if missing:
            R.partial.append("ssm mixer: missing " + ",".join(missing))
        else:
            if d_inner is None:
                d_inner = mh * mhd if (mh and mhd) else (exp * D if exp else None)
            if d_inner is None:
                R.partial.append("ssm mixer: cannot derive d_inner (no mamba_d_ssm / n_heads*d_head / expand)")
            else:
                for m in mamba2_mixer(R, D, d_inner, mh, d_state, ng, dconv, "mixer"):
                    mats.append(Mat(m.name, m.organ, m.out, m.inn, nl, nl))

    # ---- MLP / MoE --------------------------------------------------------
    n_exp = (R.opt("num_experts", "moe") or R.opt("num_local_experts", "moe")
             or R.opt("n_routed_experts", "moe"))
    topk = R.opt("num_experts_per_tok", "moe") or R.opt("experts_per_token", "moe")
    moe_inter = R.opt("moe_intermediate_size", "moe")
    inter = R.opt("intermediate_size", "mlp") or R.opt("ffn_hidden_size", "mlp")
    if moe_inter is None and n_exp:
        moe_inter = inter          # mixtral/olmoe/gpt_oss/granite: intermediate_size IS the expert size
        R.note("expert width read from intermediate_size (config carries no moe_intermediate_size)")
    act = str(R.opt("mlp_hidden_act", "mlp") or R.opt("hidden_act", "mlp") or "silu")
    gated = act not in ("relu2", "relu", "gelu_new_nogate")
    meta["mlp_gated"] = gated
    meta["mlp_act"] = act
    ng_mlp = 3 if gated else 2

    dense_layers = R.opt("first_k_dense_replace", "moe", 0) or 0
    mlp_only = R.opt("mlp_only_layers", "moe", []) or []
    sparse_step = R.opt("decoder_sparse_step", "moe", 1) or 1
    # which layers carry an MLP/MoE block at all
    mlp_layer_idx = [i for i, t in enumerate(types) if t != "?"]
    if "mlp" in types:                 # nemotron_h: only '-' layers have an MLP
        mlp_layer_idx = [i for i, t in enumerate(types) if t == "mlp"]
    n_mlp_blocks = len(mlp_layer_idx)

    if n_exp and topk and moe_inter:
        moe_idx = [i for i in mlp_layer_idx
                   if i >= dense_layers and i not in mlp_only and (i % sparse_step == 0)]
        n_moe = len(moe_idx)
        n_dense = n_mlp_blocks - n_moe
        # routed experts
        mats.append(Mat("expert.gate", "expert", moe_inter, D, n_exp * n_moe, topk * n_moe))
        if gated:
            mats.append(Mat("expert.up", "expert", moe_inter, D, n_exp * n_moe, topk * n_moe))
        mats.append(Mat("expert.down", "expert", D, moe_inter, n_exp * n_moe, topk * n_moe))
        mats.append(Mat("router", "attn_proj", n_exp, D, n_moe, n_moe))
        # shared experts, always active
        n_sh = R.opt("n_shared_experts", "moe", 0) or 0
        sh_w = R.opt("shared_expert_intermediate_size", "moe") or R.opt("shared_intermediate_size", "moe")
        if n_sh:
            sh_w = (sh_w or moe_inter) * n_sh
        if sh_w:
            mats.append(Mat("shared.gate", "expert", sh_w, D, n_moe, n_moe))
            if gated:
                mats.append(Mat("shared.up", "expert", sh_w, D, n_moe, n_moe))
            mats.append(Mat("shared.down", "expert", D, sh_w, n_moe, n_moe))
        if n_dense and inter:
            mats.append(Mat("mlp.gate", "mlp", inter, D, n_dense, n_dense))
            if gated:
                mats.append(Mat("mlp.up", "mlp", inter, D, n_dense, n_dense))
            mats.append(Mat("mlp.down", "mlp", D, inter, n_dense, n_dense))
        meta.update(n_experts=n_exp, top_k=topk, expert_width=moe_inter,
                    n_moe_layers=n_moe, n_dense_mlp_layers=n_dense,
                    shared_expert_width=sh_w or 0,
                    expert_calls_per_token=topk * n_moe + (n_moe if sh_w else 0))
    else:
        if inter is None:
            R.partial.append("mlp: no intermediate_size / ffn_hidden_size")
        else:
            mats.append(Mat("mlp.gate", "mlp", inter, D, n_mlp_blocks, n_mlp_blocks))
            if gated:
                mats.append(Mat("mlp.up", "mlp", inter, D, n_mlp_blocks, n_mlp_blocks))
            mats.append(Mat("mlp.down", "mlp", D, inter, n_mlp_blocks, n_mlp_blocks))
        meta.update(n_experts=0, top_k=0, expert_width=0, n_moe_layers=0,
                    n_dense_mlp_layers=n_mlp_blocks, shared_expert_width=0,
                    expert_calls_per_token=0)

    # ---- embeddings, head, norms -----------------------------------------
    mats.append(Mat("emb", "emb", V, D, 1, 0))
    mats.append(Mat("head", "head", V, D, 0 if tie else 1, 1))
    mats.append(Mat("norms", "norm", D, 1, 2 * L + 1, 2 * L + 1))

    meta.update(hidden_size=D, n_layers=L, vocab=V, tied=tie, mlp_width=inter)
    return mats, meta


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------
CTX_POINTS = [4096, 32768, 131072]


def footprint(mats, ternary_organs):
    tot = {"payload": 0.0, "scales": 0.0, "padding": 0.0}
    by_organ = {}
    for m in mats:
        if m.count == 0:
            continue
        p, s, pad = mat_bytes(m, m.organ in ternary_organs, m.count)
        tot["payload"] += p
        tot["scales"] += s
        tot["padding"] += pad
        by_organ[m.organ] = by_organ.get(m.organ, 0.0) + p + s + pad
    tot["total"] = tot["payload"] + tot["scales"] + tot["padding"]
    return tot, by_organ


def kv_bytes(meta, ctx, bytes_per_elem, cfg):
    """KV cache bytes at `ctx`, attention layers only.  Returns (bytes, note) or (None, reason)."""
    if meta.get("layer_types_source") == "UNKNOWN":
        return None, "UNKNOWN: no layer-type map in config"
    n = meta.get("n_attn_layers", 0)
    e = meta.get("kv_elems_per_tok_per_layer", 0)
    if n == 0:
        return 0.0, "no attention layers"
    if not e:
        return None, "UNKNOWN: kv element count not derivable"
    sub = meta.get("attn_subtype")
    sw = cfg.get("sliding_window")
    if sub and sw:
        total = 0.0
        for t in sub:
            t = str(t)
            if "mamba" in t:
                continue
            eff = min(ctx, sw) if "sliding" in t else ctx
            total += e * eff * bytes_per_elem
        return total, f"per-layer sliding_window={sw} honoured"
    return e * ctx * n * bytes_per_elem, ""


def per_token_bytes(mats, ternary_organs):
    """Bytes actually read per generated token, split by rate class."""
    out = {"lut": 0.0, "lut_dense": 0.0, "lut_moe": 0.0, "proj": 0.0, "head": 0.0, "other": 0.0}
    for m in mats:
        if m.per_token <= 0:
            continue
        p, s, pad = mat_bytes(m, m.organ in ternary_organs, m.per_token)
        b = p + s + pad
        if m.organ in ("mlp", "expert"):
            if m.organ in ternary_organs:
                cls = "lut"
                out["lut_dense" if m.organ == "mlp" else "lut_moe"] += b
            else:
                cls = "proj"
        elif m.organ == "head":
            cls = "head"
        elif m.organ in ("attn_proj", "ssm_proj"):
            cls = "proj"
        else:
            cls = "other"
        out[cls] += b
    return out


def time_model(ptb, meta, kv_b, glue_us, n_att_override=None, attn_mode="full", attn_ctx=None,
               conv_swap_bytes_per_layer=None):
    """Per-token time as a CENTRAL ESTIMATE WITH A PROPAGATED INTERVAL.

    WHAT CHANGED AND WHY (audit F1 / F8).  The previous revision reported a single
    "MEASURED-ONLY" number built as

        meas = t_proj_pess + t_lut_meas + glue + t_kv_pess

    i.e. the PESSIMISTIC CORNER OF EVERY BRACKET SIMULTANEOUSLY -- proj at 34.0 (the low edge of an
    asserted, never-measured [34-40]) and KV at 40.0 (the low edge of [40-44]).  A bracket was
    silently collapsed to a point at its worst edge and that point was reported as a gate verdict.
    It is the sole reason the tool said "zero donors pass".  A one-corner column is not a
    conservative estimate, it is a different quantity, and it is withdrawn here.

    Every term now carries (slow, fast, central).  The reported total is the sum of centrals; the
    reported interval is (sum of slows, sum of fasts) -- fully-correlated worst case, which is
    WIDER than a root-sum-square would be.  That is deliberate: the terms are not independent
    (they share one memory system), so RSS would understate.
    """
    calls = meta.get("expert_calls_per_token", 0)
    projsz = ptb["proj"] + ptb["head"] + ptb["other"]
    # AUDIT F4: charge the weights the conversion ADDS.  n_conv layers lose their attention
    # projections and gain an Arch-A SSM block, which is 2.75x larger at D=1536.
    n_conv_layers = 0
    if n_att_override is not None:
        n_conv_layers = max(0, (meta.get("n_attn_layers", 0) or 0) - n_att_override)
    if n_conv_layers and conv_swap_bytes_per_layer is not None:
        projsz += n_conv_layers * conv_swap_bytes_per_layer

    # --- proj+head: the whole per-token fp32 stream, on the DIRECTLY MEASURED donor-stream rate.
    #     Not a curve lookup.  DONOR_PROJ_RATE.md sec.5.2 timed exactly this object.
    t_proj = _t(projsz, PROJ_STREAM_LO, PROJ_STREAM_HI, PROJ_STREAM_MEAS)

    # --- KV: measured aggregate DRAM stream, central = midpoint of the measured [40-44].
    t_kv = _t(kv_b, DRAM_AGG_LO, DRAM_AGG_HI, DRAM_AGG_MID) if kv_b is not None else (None, None, None)

    # --- ternary MLP / experts.  The bracket is now [engine-integrated .. kernel-pure at donor D]
    #     and the two ends are DIFFERENT QUANTITIES (see the constants block).  The dispatch term
    #     is added ONLY at the kernel-pure end: LUT_MOE_LO=4.2 is engine-integrated and already
    #     contains it.  Adding it to both ends, as the previous revision did, double-counted.
    d_slow, d_fast, d_ce = _t(ptb["lut_dense"], LUT_DENSE_LO, LUT_DENSE_HI)
    m_slow, m_fast, m_ce = _t(ptb["lut_moe"], LUT_MOE_LO, LUT_MOE_HI)
    disp = calls * EXPERT_DISPATCH_US
    m_fast += disp
    m_ce += 0.5 * disp
    t_lut = (d_slow + m_slow, d_fast + m_fast, d_ce + m_ce)

    # --- glue: DECLARED, not measured.  Carried with a x2 interval so it cannot pass for measured.
    g_slow = glue_us * GLUE_INTERVAL_FACTOR
    g_fast = glue_us / GLUE_INTERVAL_FACTOR
    t_glue = (g_slow, g_fast, glue_us)

    # --- AUDIT C12: the two components sec.2 lists and this tool used to omit.
    D = meta["hidden_size"]
    L = meta["n_layers"]
    n_att = meta.get("n_attn_layers", 0) if n_att_override is None else n_att_override
    n_ssm = max(0, L - n_att)
    # scan-recurrence runs on every layer that is NOT attention (a converted donor's SSM layers)
    t_scan_c = SCAN_US_ANCHOR * (2.0 * D * SSM_STATE_N * n_ssm) / SCAN_ANCHOR_DEN
    t_scan = (t_scan_c * 2.0, t_scan_c / 2.0, t_scan_c)      # x2 interval: parametric, not measured
    # attention on the retained layers
    t_att = None
    if attn_mode == "swa":
        t_att_c = SWA_US_ANCHOR * (D / 256.0) * n_att
    elif attn_mode == "full_swaScaled":
        # DECLARED UPPER BOUND ONLY.  Linearly scaling sec.2's window-128 cost by ctx/128 is a 256x
        # extrapolation at 32K and it overstates: the 65 us anchor contains per-layer fixed costs
        # that do NOT scale with window length.  Kept so the bound is visible, never as an estimate.
        span = max(1.0, (attn_ctx or 32768) / float(SWA_WINDOW))
        t_att_c = SWA_US_ANCHOR * (D / 256.0) * n_att * span
    else:
        # retained FULL attention, first-principles COMPUTE model (a DESK MODEL, not measured).
        # Per layer: QK^T then AV over the whole context = 2 passes x ctx x D MACs = 4*ctx*D flops.
        # The KV *traffic* is already charged in the KV term, so this is the compute on top.
        # Rate bracket: 19.2 GFLOP/s = the measured t6 proj-GEMV rate class (memory-bound floor,
        # pessimistic here since attention has ~12x the arithmetic intensity of a batch-1 GEMV);
        # 100 GFLOP/s = a realistic Zen2 t6 fp32 ceiling.  NEITHER IS MEASURED FOR THIS KERNEL.
        c = attn_ctx or 32768
        flops = 4.0 * c * D * n_att
        t_att = (flops / 19.2e9 * 1e6, flops / 100.0e9 * 1e6, flops / 40.0e9 * 1e6)
        t_att_c = t_att[2]
    if attn_mode != "full":
        t_att = (t_att_c * 2.0, t_att_c / 2.0, t_att_c)

    def total(with_kv=True):
        acc = [0.0, 0.0, 0.0]
        parts = [t_proj, t_lut, t_glue, t_scan, t_att] + ([t_kv] if (with_kv and kv_b is not None) else [])
        for pt in parts:
            for i in range(3):
                acc[i] += pt[i]
        return acc

    slow, fast, ce = total(True)
    kslow, kfast, kce = total(False)

    res = {
        # rate provenance, so a reader can see which measured constant priced this row
        "proj_rate_lo": PROJ_STREAM_LO, "proj_rate_hi": PROJ_STREAM_HI,
        "proj_rate_central": PROJ_STREAM_MEAS, "proj_rate_extrapolated": False,
        "proj_rate_source": "DONOR_PROJ_RATE.md sec.5.2 measured donor stream",
        "t_proj_us": t_proj, "t_lut_us": t_lut, "t_kv_us": t_kv, "t_glue_us": glue_us,
        "t_scan_us": t_scan, "t_att_us": t_att, "n_att": n_att, "n_ssm": n_ssm,
        "n_conv_layers": n_conv_layers,
        "conv_swap_bytes_total": (n_conv_layers * conv_swap_bytes_per_layer
                                  if conv_swap_bytes_per_layer else 0),
        "attn_mode": attn_mode, "attn_extrapolated": (attn_mode != "swa"),
        "t_glue_interval_us": t_glue,
        # (slow, fast) retained under the old key names so existing readers do not silently break
        "t_total_us": (slow, fast),
        "t_total_central_us": ce,
        "toks": (1e6 / slow, 1e6 / fast),
        "toks_central": 1e6 / ce,
        "toks_interval": (1e6 / slow, 1e6 / fast),
        # KV-FREE column: isolates "was this donor eliminated SOLELY by its KV traffic?"
        # Owner directive: a donor may NOT be eliminated only because full-native KV does not fit
        # or stream -- that is what the recall tier absorbs.
        "toks_kvfree_central": 1e6 / kce,
        "toks_kvfree_interval": (1e6 / kslow, 1e6 / kfast),
        "kv_unknown": kv_b is None,
    }
    # The withdrawn column, kept under its old name ONLY so that anything still reading it gets
    # the honest central value rather than the retired worst-corner number.
    res["toks_measured_only"] = res["toks_central"]
    res["t_measured_only_us"] = ce
    return res


def analyze_one(repo, cfg, meta_rec):
    R = Reader(cfg)
    row = {"repo": repo, "status": "OK", "partial": [], "notes": [], "provenance": {}}
    try:
        mats, meta = build_matrices(R)
    except MissingConfigField as e:
        row.update(status="REFUSED", error=f"MissingConfigField: {e}")
        return row
    # AUDIT F7 DEFECT, FIXED: the tool never read max_position_embeddings, so `OLMoE-1B-7B` -- a
    # 4096-position model -- was scored at 32K and 128K, and `Nemotron-H-8B-Base-8K` (8K) at both.
    # Read it, record provenance, and label every context figure against the donor's OWN capacity.
    meta["native_ctx"] = R.opt("max_position_embeddings", "native context window",
                               default=None, mark_partial=True)
    row["meta"] = meta
    row["partial"] = R.partial
    row["notes"] = R.notes
    row["provenance"] = {k: sorted(set(v)) for k, v in R.prov.items()}

    # --- params by organ
    organ_params = {}
    for m in mats:
        organ_params[m.organ] = organ_params.get(m.organ, 0) + m.params
    total = sum(organ_params.values())
    row["params_by_organ"] = organ_params
    row["params_total_computed"] = total
    adv = meta_rec.get("safetensors_total_params")
    by_dtype = meta_rec.get("safetensors_params_by_dtype") or {}
    row["params_total_advertised"] = adv
    row["params_dtypes"] = by_dtype
    packed = "quantization_config" in cfg
    if not adv:
        row["params_delta_pct"] = None
        row["params_agree"] = None
        row["crosscheck"] = "NO_METADATA"
        row["partial"].append("cross-check: repo exposes no safetensors parameter total")
    else:
        d = (total - adv) / adv
        row["params_delta_pct"] = d * 100
        row["params_agree"] = abs(d) <= 0.01
        row["crosscheck"] = "AGREE" if row["params_agree"] else "DISAGREE"
        if not row["params_agree"] and meta.get("tied"):
            # A repo may physically store lm_head.weight as a duplicate of the tied embedding.
            # That is a STORAGE fact (visible in the safetensors total), not a logical parameter.
            # Test the hypothesis explicitly rather than adjusting the formula.
            dup = total + meta["vocab"] * meta["hidden_size"]
            if abs((dup - adv) / adv) <= 0.01:
                row["crosscheck"] = "AGREE_TIED_DUPLICATED"
                row["crosscheck_secondary"] = {
                    "hypothesis": "repo stores lm_head.weight as a physical duplicate of the tied "
                                  "embedding; logical parameter count is the computed one",
                    "computed_plus_one_embedding": dup,
                    "delta_pct": (dup - adv) / adv * 100,
                }
                row["params_agree"] = True
        if packed and not row["params_agree"]:
            # The repo ships pre-quantized: its safetensors totals count PACKED STORAGE elements
            # (uint8 blocks + scales), not logical weights, so the headline total is not comparable.
            # Secondary cross-check: the float-dtype subtotal vs the organs the config's own
            # quantization_config says were NOT converted.
            floaty = sum(v for k, v in by_dtype.items() if k.upper() in ("F32", "F16", "BF16", "F64"))
            keep = cfg["quantization_config"].get("modules_to_not_convert", [])
            unpacked_mine = sum(v for k, v in organ_params.items() if k != "expert")
            row["crosscheck"] = "PACKED_STORAGE"
            row["crosscheck_secondary"] = {
                "float_dtype_subtotal": floaty,
                "my_non_expert_organs": unpacked_mine,
                "delta_pct": (unpacked_mine - floaty) / floaty * 100 if floaty else None,
                "modules_to_not_convert": keep,
                "quant_method": cfg["quantization_config"].get("quant_method"),
            }
    if row["crosscheck"] == "DISAGREE":
        row["partial"].append(
            f"CROSS-CHECK DISAGREEMENT {row['params_delta_pct']:+.2f}%: the organ formula is wrong "
            f"for this family; every derived figure in this row inherits that error")

    # --- active params/token
    act = sum(m.out * m.inn * m.per_token for m in mats if m.per_token > 0 and m.organ != "emb")
    row["active_params_per_token"] = act

    # --- footprints
    row["fp_all_ternary"], row["fp_all_ternary_by_organ"] = footprint(mats, TERNARY_MAP_A)
    row["fp_p61", ] = None
    fpb, fpb_org = footprint(mats, TERNARY_MAP_B)
    row["fp_p61_map"] = fpb
    row["fp_p61_map_by_organ"] = fpb_org
    del row["fp_p61", ]

    # --- KV
    row["kv"] = {}
    for ctx in CTX_POINTS:
        for lbl, bpe in (("fp16", B_FP16), ("4bit", 0.5)):
            b, note = kv_bytes(meta, ctx, bpe, cfg)
            row["kv"][f"{ctx}_{lbl}"] = {"bytes": b, "note": note}

    # --- per-token traffic + time (primary model = P61/D9 map)
    ptb_b = per_token_bytes(mats, TERNARY_MAP_B)
    ptb_a = per_token_bytes(mats, TERNARY_MAP_A)
    row["per_token_bytes_p61"] = ptb_b
    row["per_token_bytes_all_ternary"] = ptb_a
    glue = 7.0 * (meta["n_layers"] / 6.0)     # budget sec.2 "norms+glue ~7 us" at L=6, scaled
    row["glue_us"] = glue

    # AUDIT F4: what ONE attention->SSM conversion costs in per-token weight BYTES.
    # Under the P61/D9 map both attention and SSM projections are fp32 (Phase 61 double-fail),
    # so this is a straight params*4 swap -- and it is POSITIVE: the SSM block is BIGGER.
    _na = meta.get("n_attn_layers", 0) or 0
    # NOTE: `mats` holds ONE Mat per organ with a `per_token` layer multiplicity, so the total must
    # be formed with per_token before dividing by the layer count.  (Dividing the bare sum by _na
    # divides by the layer count TWICE -- caught by the sanity check below.)
    _attn_tot = sum(m.out * m.inn * m.per_token for m in mats if m.organ == "attn_proj")
    _attn_pl = (_attn_tot / _na) if _na else 0.0
    _ssm_pl = ssm_block_params(meta["hidden_size"])
    row["conv_attn_params_per_layer"] = _attn_pl
    row["conv_ssm_params_per_layer"] = _ssm_pl
    row["conv_swap_bytes_per_layer"] = (_ssm_pl - _attn_pl) * B_FP32
    row["conv_swap_ratio"] = (_ssm_pl / _attn_pl) if _attn_pl else None
    # SANITY CHECK (fires loudly rather than silently mispricing): for a standard MHA/GQA layer the
    # attention projections are q+k+v+o, which is between 2*D^2 (full MQA) and 4*D^2 (full MHA).
    if _attn_pl and not (1.5 * meta["hidden_size"] ** 2 <= _attn_pl <= 4.5 * meta["hidden_size"] ** 2):
        row["partial"].append(
            f"conv swap: attention params/layer {_attn_pl:.0f} outside [1.5,4.5]*D^2 for "
            f"D={meta['hidden_size']} -- the per-layer normalisation is suspect")

    row["time"] = {}
    nat = meta.get("native_ctx")
    for ctx in CTX_POINTS:
        for lbl in ("fp16", "4bit"):
            kvb = row["kv"][f"{ctx}_{lbl}"]["bytes"]
            tm = time_model(ptb_b, meta, kvb, glue)
            # A context beyond the donor's own max_position_embeddings is NOT natively servable:
            # it needs RoPE extension (a quality cost under S4) or the recall tier.  Label it;
            # do not price it as if the donor supported it.
            tm["beyond_native"] = (nat is not None and ctx > nat)
            tm["native_ctx"] = nat
            row["time"][f"{ctx}_{lbl}"] = tm
    row["time"]["nokv"] = time_model(ptb_b, meta, 0.0, glue)
    # SKU-A AS SEALED: 32K native attention, capped at whatever the donor natively supports, with
    # the recall tier carrying anything beyond.  Owner directive: a donor may NOT be eliminated
    # solely because full-native KV does not fit or stream.
    ctx_eff = min(32768, nat) if nat else 32768
    row["ctx_effective_skuA"] = ctx_eff
    for lbl, bpe in (("fp16", B_FP16), ("4bit", 0.5)):
        b, note = kv_bytes(meta, ctx_eff, bpe, cfg)
        row["kv"][f"skuA_eff_{lbl}"] = {"bytes": b, "note": note, "ctx": ctx_eff}
        # UNCONVERTED baseline: the donor as it ships -- full attention on every attention layer.
        tm = time_model(ptb_b, meta, b, glue, attn_mode="full", attn_ctx=ctx_eff)
        tm["beyond_native"] = False
        tm["native_ctx"] = nat
        row["time"][f"skuA_eff_{lbl}"] = tm
        # CONVERTED: the actual proposal.  The sealed constraint is "attention on a MINORITY of
        # layers", so n_att = L//2 is the most attention the contract permits; the rest become SSM
        # layers, which pay scan-recurrence and hold NO KV.  Retained layers are WINDOWED (SWA-128),
        # which is the only attention cost this engine has ever measured.
        n_att_c = meta["n_layers"] // 2
        n_orig = meta.get("n_attn_layers", 0) or 1
        b_conv = (b / n_orig * n_att_c) if b is not None else None
        tc = time_model(ptb_b, meta, b_conv, glue, n_att_override=n_att_c,
                        attn_mode="swa", attn_ctx=ctx_eff,
                        conv_swap_bytes_per_layer=row["conv_swap_bytes_per_layer"])
        tc["beyond_native"] = False
        tc["native_ctx"] = nat
        tc["n_att_converted"] = n_att_c
        row["time"][f"converted_{lbl}"] = tc

    # all-ternary sensitivity: projections would run the LUT kernel, which is NOT measured for
    # projection shapes.  Priced at the expert LUT bracket and labelled.
    lut_a = ptb_a["lut"] + ptb_a["proj"] + ptb_a["head"] + ptb_a["other"]
    calls = meta.get("expert_calls_per_token", 0)
    # dispatch added ONLY at the kernel-pure end; LUT_MOE_LO is engine-integrated and contains it.
    row["time_all_ternary_nokv_us"] = (
        lut_a / (LUT_MOE_LO * 1e9) * 1e6 + glue,
        lut_a / (LUT_MOE_HI * 1e9) * 1e6 + calls * EXPERT_DISPATCH_US + glue,
    )

    # --- head alone
    hm = [m for m in mats if m.organ == "head"]
    hb_fp32 = sum(m.out * m.inn * B_FP32 for m in hm)
    hb_tern = 0.0
    for m in hm:
        p, s, pad = mat_bytes(m, True, 1)
        hb_tern += p + s + pad
    row["head"] = {
        "bytes_fp32": hb_fp32, "bytes_ternary": hb_tern,
        "ms_fp32": [hb_fp32 / (r * 1e9) * 1e3 for r in r_proj(hb_fp32)[:2]],
        "ms_ternary_lut": [hb_tern / (r * 1e9) * 1e3 for r in (LUT_GBS_PESS, LUT_GBS_OPT)],
        "fp32_rate_extrapolated": r_proj(hb_fp32)[2],
    }

    # --- expert granularity vs the rho-safe 48 KB chunk
    if meta.get("n_experts"):
        ew = meta["expert_width"]
        d = meta["hidden_size"]
        nmat = 3 if meta.get("mlp_gated", True) else 2
        eb = ew * d * nmat * B_TERNARY + (ew * (nmat - 1) + d) * SCALE_BYTES_PER_ROW
        row["expert_chunk_kb"] = eb / KB
        row["rho_safe"] = eb / KB >= RHO_SAFE_CHUNK_KB
    # --- what binds: the dominant per-token term (P61/D9 map, no KV)
    t = row["time"]["nokv"]
    terms = {"proj+head (fp32)": t["t_proj_us"][2], "ternary MLP/experts": t["t_lut_us"][2],
             "glue": t["t_glue_us"]}
    row["dominant_term"] = max(terms, key=terms.get)
    row["term_us_central"] = terms
    row["term_us_pess"] = {k: v for k, v in
                           (("proj+head (fp32)", t["t_proj_us"][0]),
                            ("ternary MLP/experts", t["t_lut_us"][0]),
                            ("glue", t["t_glue_interval_us"][0]))}

    # --- SKU verdicts.  SKU-A judged at its sealed native attention window (32K, S3);
    #     SKU-B at the sealed 128K native contract.
    row["sku"] = {}
    for lbl, budget, ctx in (("A_16GB@32K", SKU_A_BYTES, 32768), ("B_64GB@128K", SKU_B_BYTES, 131072)):
        for pmap in ("p61", "all_ternary"):
            for kvl in ("fp16", "4bit"):
                tot, fits = sku_verdict(row, budget, ctx, kvl, pmap)
                row["sku"][f"{lbl}|{pmap}|kv{kvl}"] = {"bytes": tot, "fits": fits}

    if row["crosscheck"] == "DISAGREE":
        row["status"] = "FORMULA_DISAGREEMENT"
    elif row["partial"]:
        row["status"] = "PARTIAL"
    return row


def sku_verdict(row, budget_bytes, ctx, kv_label="fp16", precision="p61"):
    fp = row["fp_p61_map"] if precision == "p61" else row["fp_all_ternary"]
    kv = row["kv"][f"{ctx}_{kv_label}"]["bytes"]
    if kv is None:
        return None, None
    logits = row["meta"]["vocab"] * B_FP32
    tot = fp["total"] + kv + logits + OS_MARGIN_BYTES
    return tot, tot <= budget_bytes



# ---------------------------------------------------------------------------
# AUDIT C11 + C12: the 2x2 and the retained-attention curve.
# Neither may be collapsed to a headline -- the whole point is that the verdict
# lives in ONE CELL of a 2x2 and moves with a design variable.
# ---------------------------------------------------------------------------
def kv_per_attn_layer(row, ctx, kv_label):
    """KV bytes for ONE attention layer, so converting layers to SSM removes their KV."""
    n = row["meta"].get("n_attn_layers", 0)
    b = row["kv"][f"{ctx}_{kv_label}"]["bytes"]
    if b is None or not n:
        return 0.0
    return b / n


def cmd_curve(args):
    """tok/s as a function of the number of RETAINED ATTENTION LAYERS.

    For a converted donor n_att is a DESIGN VARIABLE, not a constant: every layer not retained as
    attention becomes an SSM layer (scan-recurrence) and, crucially, stops holding a KV cache.
    Reporting one number at one assumed n_att hides exactly the trade the design has to make.
    """
    out = json.load(open(args.json_in))
    ctx = args.ctx
    for repo in (args.donor or ["Qwen/Qwen2.5-1.5B"]):
        row = next((r for r in out["rows"] if r["repo"] == repo), None)
        if row is None or row.get("status") in ("UNAVAILABLE", "REFUSED"):
            print(f"{repo}: unavailable")
            continue
        m = row["meta"]
        L = m["n_layers"]
        ptb = row["per_token_bytes_p61"]
        glue = row["glue_us"]
        swap = row["conv_swap_bytes_per_layer"]
        print(f"    AUDIT F4: each converted layer swaps "
              f"{row['conv_attn_params_per_layer']/1e6:.2f}M attention params for "
              f"{row['conv_ssm_params_per_layer']/1e6:.2f}M Arch-A SSM params "
              f"({row['conv_swap_ratio']:.2f}x) = {swap/1e6:+.1f} MB/token EXTRA")
        print("    fp32 weight traffic per converted layer. Conversion ADDS weight")
        print("    bytes and REMOVES KV bytes -> there is a CROSSOVER in context.")
        print(f"\n=== {repo}: tok/s vs RETAINED ATTENTION LAYERS  (D={m['hidden_size']} L={L} "
              f"ctx={ctx}) ===")
        print("    Every layer NOT retained becomes an SSM layer: it pays scan-recurrence instead of")
        print("    attention, and it stops holding a KV cache.  Two pricings for a retained layer:")
        print("      SWA-128  = the engine's measured windowed cost (sec.2: 65us * D/256 per layer)")
        print("      FULL     = retained FULL attention, first-principles compute model:")
        print("                 4*ctx*D flops/layer over a [19.2 .. 100] GFLOP/s bracket, central 40.")
        print("                 A DESK MODEL -- this kernel has never been measured at donor ctx.")
        print("                 (Scaling sec.2's window-128 cost by ctx/128 instead gives a 256x")
        print("                  extrapolation and ~25x WORSE numbers; that is an upper bound, not")
        print("                  an estimate, and is available as attn_mode=full_swaScaled.)")
        print(f"    {'n_att':>5} {'n_ssm':>5} | {'SWA-128 central':>15} {'interval':>16} {'gate':>9}"
              f" | {'FULL central':>13} {'interval':>16} {'gate':>9}")
        for kvl in ("4bit", "fp16"):
            print(f"  -- KV {kvl} --")
            per = kv_per_attn_layer(row, ctx, kvl)
            for n_att in range(0, L + 1, max(1, L // 14)):
                cells = []
                for mode in ("swa", "full"):
                    t = time_model(ptb, m, per * n_att, glue, n_att_override=n_att,
                                   attn_mode=mode, attn_ctx=ctx,
                                   conv_swap_bytes_per_layer=swap)
                    lo, hi = t["toks"]
                    g = "PASS" if lo >= TOKS_GATE else ("STRADDLE" if t["toks_central"] >= TOKS_GATE else "FAIL")
                    cells.append((t["toks_central"], lo, hi, g))
                a, b = cells
                print(f"    {n_att:5d} {L-n_att:5d} | {a[0]:15.2f} {a[1]:7.2f}..{a[2]:<7.2f} {a[3]:>9}"
                      f" | {b[0]:13.2f} {b[1]:7.2f}..{b[2]:<7.2f} {b[3]:>9}")
            # crossing points
            for mode, label in (("swa", "SWA-128"), ("full", "FULL")):
                cross = None
                for n_att in range(0, L + 1):
                    t = time_model(ptb, m, per * n_att, glue, n_att_override=n_att,
                                   attn_mode=mode, attn_ctx=ctx,
                                   conv_swap_bytes_per_layer=swap)
                    if t["toks"][0] < TOKS_GATE:
                        cross = n_att
                        break
                if cross is None:
                    print(f"    -> {label}: lower bound clears 10 tok/s at EVERY n_att up to {L}")
                elif cross == 0:
                    print(f"    -> {label}: FAILS even at n_att=0 (all-SSM)")
                else:
                    print(f"    -> {label}: lower bound clears 10 tok/s for n_att <= {cross-1} "
                          f"of {L}; FAILS from n_att={cross}")
    return 0


def cmd_grid(args):
    """AUDIT C11: the 2x2 the headline was hiding.

    {4-bit KV, fp16 KV} x {MLP as charged (retired 11.398), MLP measured at donor D}.
    Three constants each flip this verdict; the one that got measured first was the smallest lever.
    """
    out = json.load(open(args.json_in))
    ctx = args.ctx
    for repo in (args.donor or ["Qwen/Qwen2.5-1.5B"]):
        row = next((r for r in out["rows"] if r["repo"] == repo), None)
        if row is None or row.get("status") in ("UNAVAILABLE", "REFUSED"):
            continue
        m = row["meta"]
        ptb = row["per_token_bytes_p61"]
        glue = row["glue_us"]
        n_att = m.get("n_attn_layers", 0)
        print(f"\n=== {repo}: THE 2x2 (ctx={ctx}, n_att={n_att} = unconverted, attention priced SWA-128) ===")
        print("    proj held at the measured donor stream in every cell; only KV precision and the")
        print("    MLP constant move.  4-bit KV DOES NOT EXIST in this engine; fp16 is what is built.")
        print(f"    {'MLP constant':<38} {'KV':>6} {'central':>9} {'interval':>16} {'verdict':>10}")
        for mlp_label, mlp_rate in (
                ("RETIRED 11.398 (integrated D=256,", LUT_DENSE_RETIRED),
                ("MEASURED 21.25 (integrated donor D)", LUT_DENSE_MEAS),
                ("kernel-pure 27.64 (donor D, unreachable)", 27.64)):
            for kvl in ("4bit", "fp16"):
                kvb = row["kv"][f"{ctx}_{kvl}"]["bytes"]
                g_lo, g_hi = LUT_DENSE_LO, LUT_DENSE_HI
                globals()["LUT_DENSE_LO"] = mlp_rate
                globals()["LUT_DENSE_HI"] = mlp_rate
                t = time_model(ptb, m, kvb, glue, n_att_override=n_att,
                               attn_mode="swa", attn_ctx=ctx)
                globals()["LUT_DENSE_LO"], globals()["LUT_DENSE_HI"] = g_lo, g_hi
                lo, hi = t["toks"]
                v = "PASS" if lo >= TOKS_GATE else ("STRADDLE" if t["toks_central"] >= TOKS_GATE else "FAIL")
                print(f"    {mlp_label:<38} {kvl:>6} {t['toks_central']:9.2f} "
                      f"{lo:7.2f}..{hi:<7.2f} {v:>10}")
        print("    (row 3 is shown only to bound the cell -- kernel-pure is not reachable by an engine)")
    return 0



def cmd_ctxsweep(args):
    """AUDIT F4 follow-up: tok/s vs CONTEXT at fixed n_att, and the conversion crossover.

    attention->SSM trades a per-token WEIGHT cost that is CONSTANT in context for a KV traffic
    cost that GROWS with context.  So there is a crossover length, and BELOW IT THE CONVERSION
    MAKES THE MODEL SLOWER.  Reporting a single tok/s at one context hides the whole argument.
    """
    out = json.load(open(args.json_in))
    ctxs = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    for repo in (args.donor or ["Qwen/Qwen2.5-1.5B"]):
        row = next((r for r in out["rows"] if r["repo"] == repo), None)
        if row is None or row.get("status") in ("UNAVAILABLE", "REFUSED"):
            continue
        m = row["meta"]; ptb = row["per_token_bytes_p61"]; glue = row["glue_us"]
        L = m["n_layers"]; na = m.get("n_attn_layers", 0) or 1
        swap = row["conv_swap_bytes_per_layer"]
        print("")
        print(f"=== {repo}: tok/s vs CONTEXT, and the attention->SSM crossover ===")
        print(f"    Each converted layer ADDS {swap/1e6:+.1f} MB/token of fp32 weight traffic "
              f"(constant in ctx)")
        print(f"    and REMOVES that layer's KV traffic (grows linearly in ctx). Retained attention")
        print("    is priced WINDOWED (SWA-128), the only attention cost this engine has measured.")
        for kvl in ("4bit", "fp16"):
            print(f"  -- KV {kvl} --")
            print(f"    {'ctx':>7} | {'unconverted':>11} {'half (n=L/2)':>13} {'all-SSM':>9} | winner")
            prev = None
            cross = None
            for c in ctxs:
                b, _ = kv_bytes(m, c, B_FP16 if kvl == "fp16" else 0.5, {})
                if b is None:
                    continue
                vals = []
                for n_att in (na, L // 2, 0):
                    t = time_model(ptb, m, b / na * n_att, glue, n_att_override=n_att,
                                   attn_mode="swa", attn_ctx=c,
                                   conv_swap_bytes_per_layer=swap)
                    vals.append(t["toks_central"])
                w = "unconverted" if vals[0] >= max(vals[1:]) else ("all-SSM" if vals[2] >= vals[1] else "half")
                if prev is not None and prev == "unconverted" and w != "unconverted" and cross is None:
                    cross = c
                prev = w
                print(f"    {c:>7} | {vals[0]:11.2f} {vals[1]:13.2f} {vals[2]:9.2f} | {w}")
            if cross:
                print(f"    -> CROSSOVER at ctx ~= {cross}: below it converting is a LOSS, above it a win.")
            elif prev == "unconverted":
                print(f"    -> NO CROSSOVER inside {ctxs[0]}..{ctxs[-1]}: converting is a LOSS at every")
                print("       context tested.  The SSM block's extra weight bytes never pay for themselves.")
            else:
                print(f"    -> converting already wins at ctx={ctxs[0]}; crossover is below the swept range.")
    return 0


def cmd_analyze(args):
    man = json.load(open(os.path.join(CFG_DIR, "_manifest.json")))
    rows = []
    for repo in CANDIDATES:
        rec = man.get(repo, {})
        if rec.get("status") != "OK":
            rows.append({"repo": repo, "status": "UNAVAILABLE",
                         "error": rec.get("reason", "not fetched")})
            continue
        cfg = json.load(open(os.path.join(CFG_DIR, slug(repo) + ".json")))
        rows.append(analyze_one(repo, cfg, rec))
    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rate_constants": {
            "source": "docs/research/DONOR_PROJ_RATE.md (measured 2026-08-20, 3600X) + "
                      "docs/PHASE64_BUDGET.md sec.1 (DRAM aggregate)",
            "proj_donor_stream_gbs": [PROJ_STREAM_LO, PROJ_STREAM_MEAS, PROJ_STREAM_HI],
            "proj_donor_stream_note": "DONOR_PROJ_RATE sec.5.2: 28x(q,k,v,o)+head in layer order, "
                                      "1550057472 B/token, 37.74 +/- 0.18 GB/s = 41.07 ms/token",
            "proj_organ_asymptote_gbs": [PROJ_ORGAN_LO, PROJ_ASYMPTOTE_MEAS, PROJ_ORGAN_HI],
            "proj_organ_asymptote_note": "DONOR_PROJ_RATE sec.4: 38.84 +/- 0.68 over 36 "
                                         "measurements, flat 64 MB -> 1024 MB",
            "proj_retired_floor": PROJ_STREAMED_FLOOR_RETIRED,
            "rcurve_mb": RCURVE_MB, "rcurve_gbs": RCURVE_GBS,
            "rcurve_mb_published": RCURVE_MB_PUBLISHED,
            "rcurve_gbs_published": RCURVE_GBS_PUBLISHED,
            "lut_dense_gbs": [LUT_DENSE_LO, LUT_DENSE_HI],
            "lut_dense_note": "LOW = engine-integrated at D=256; HIGH = kernel-pure at donor D. "
                              "DIFFERENT QUANTITIES; the integrated donor rate is UNMEASURED and "
                              "lies somewhere between.",
            "lut_moe_gbs": [LUT_MOE_LO, LUT_MOE_HI],
            "expert_dispatch_us": EXPERT_DISPATCH_US,
            "expert_dispatch_note": "added ONLY at the kernel-pure end; the 4.2 end already "
                                    "contains it (previous revision double-counted)",
            "dram_aggregate_gbs": [DRAM_AGG_LO, DRAM_AGG_MID, DRAM_AGG_HI],
            "glue_interval_factor": GLUE_INTERVAL_FACTOR,
            "os_margin_bytes": OS_MARGIN_BYTES,
            "bracket_policy": "central estimate + fully-correlated interval. The single-corner "
                              "MEASURED-ONLY column (audit F1) is WITHDRAWN.",
        },
        "manifest": man,
        "rows": rows,
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    render(out)
    md_tables(out, args.md_out)
    return 0


def fmt_b(x):
    if x is None:
        return "UNK"
    if x >= GB:
        return f"{x/GB:.2f} GB"
    if x >= MB:
        return f"{x/MB:.1f} MB"
    return f"{x/KB:.1f} KB"


def render(out):
    print("\n=== donor inventory (stage 1) ===")
    for r in out["rows"]:
        if r["status"] in ("UNAVAILABLE", "REFUSED"):
            print(f"\n## {r['repo']}  [{r['status']}] {r.get('error','')}")
            continue
        m = r["meta"]
        print(f"\n## {r['repo']}  [{r['status']}]  {m['model_type']}")
        print(f"  D={m['hidden_size']} L={m['n_layers']} V={m['vocab']} tied={m['tied']} "
              f"attn_layers={m.get('n_attn_layers')} ({m.get('layer_types_source')})")
        print(f"  params computed={r['params_total_computed']/1e9:.3f}B "
              f"advertised={(r['params_total_advertised'] or 0)/1e9:.3f}B "
              f"delta={r['params_delta_pct'] if r['params_delta_pct'] is None else round(r['params_delta_pct'],2)}%"
              f"  [{r['crosscheck']}]"
              f"{'  <<< DISAGREEMENT' if r['crosscheck']=='DISAGREE' else ''}")
        s = r.get("crosscheck_secondary")
        if s and "quant_method" in s:
            print(f"    packed-storage repo ({s['quant_method']}); secondary check on the organs the "
                  f"config says stay float: mine {s['my_non_expert_organs']/1e9:.4f}B vs "
                  f"{s['float_dtype_subtotal']/1e9:.4f}B -> {s['delta_pct']:+.2f}%")
        elif s and "hypothesis" in s:
            print(f"    {s['hypothesis']}; computed+1 embedding = "
                  f"{s['computed_plus_one_embedding']/1e9:.4f}B ({s['delta_pct']:+.3f}%)")
        print(f"  active/token={r['active_params_per_token']/1e6:.1f}M  "
              f"expert_chunk={r.get('expert_chunk_kb') and round(r['expert_chunk_kb'],1)}KB "
              f"rho_safe={r.get('rho_safe')}")
        print(f"  weights RAM: p61-map={fmt_b(r['fp_p61_map']['total'])}  "
              f"all-ternary={fmt_b(r['fp_all_ternary']['total'])}  "
              f"(scales {fmt_b(r['fp_all_ternary']['scales'])}, pad {fmt_b(r['fp_all_ternary']['padding'])})")
        for k, v in r["sku"].items():
            if "|kvfp16" in k:
                print(f"    SKU {k:26s} {fmt_b(v['bytes']):>9s}  {'FIT' if v['fits'] else 'OVER'}")
        h = r["head"]
        print(f"  head alone: fp32 {fmt_b(h['bytes_fp32'])} -> "
              f"{h['ms_fp32'][0]:.1f}..{h['ms_fp32'][1]:.1f} ms/tok"
              f"{' (EXTRAPOLATED rate)' if h['fp32_rate_extrapolated'] else ''}; "
              f"ternary {fmt_b(h['bytes_ternary'])} -> "
              f"{h['ms_ternary_lut'][1]:.1f}..{h['ms_ternary_lut'][0]:.1f} ms/tok (LUT bracket)")
        t = r["time"]["nokv"]
        print(f"  per-token bytes (p61 map): proj+head {fmt_b(t and (r['per_token_bytes_p61']['proj']+r['per_token_bytes_p61']['head']+r['per_token_bytes_p61']['other']))}"
              f"  ternary-LUT {fmt_b(r['per_token_bytes_p61']['lut'])}"
              f"  r(proj)={t['proj_rate_lo']:.0f}-{t['proj_rate_hi']:.0f} GB/s"
              f"{' EXTRAP' if t['proj_rate_extrapolated'] else ''}")
        print(f"  BINDING TERM: {r['dominant_term']}  "
              f"(proj {t['t_proj_us'][0]/1000:.1f} ms, lut {t['t_lut_us'][0]/1000:.1f} ms)")
        print(f"  tok/s (no KV): pess={t['toks'][0]:.2f} opt={t['toks'][1]:.2f} "
              f"MEASURED-ONLY={t['toks_measured_only']:.2f}   "
              f"[all-ternary map: {1e6/r['time_all_ternary_nokv_us'][0]:.2f}.."
              f"{1e6/r['time_all_ternary_nokv_us'][1]:.2f}, UNMEASURED proj rate]")
        for ctx in CTX_POINTS:
            tt = r["time"][f"{ctx}_fp16"]
            t4 = r["time"][f"{ctx}_4bit"]
            kv = r["kv"][f"{ctx}_fp16"]["bytes"]
            kv4 = r["kv"][f"{ctx}_4bit"]["bytes"]
            if kv is None:
                print(f"   ctx {ctx:>6}: KV {r['kv'][f'{ctx}_fp16']['note']}")
            else:
                print(f"   ctx {ctx:>6}: KV fp16={fmt_b(kv):>9s} 4bit={fmt_b(kv4):>9s}  "
                      f"tok/s {tt['toks'][0]:.2f}..{tt['toks'][1]:.2f} "
                      f"(meas-only {tt['toks_measured_only']:.2f}; kv4bit meas-only "
                      f"{t4['toks_measured_only']:.2f})  "
                      f"{'PASS' if tt['toks'][1] >= TOKS_GATE else 'FAIL'}@10tok/s(optimistic)")
        if r["notes"]:
            for n in r["notes"]:
                print("  note: " + n)
        if r["partial"]:
            print("  PARTIAL: " + "; ".join(r["partial"]))
    shortlist(out)


def md_tables(out, path):
    L = []
    W = L.append
    rows = [r for r in out["rows"] if r["status"] not in ("UNAVAILABLE", "REFUSED")]
    bad = [r for r in out["rows"] if r["status"] in ("UNAVAILABLE", "REFUSED")]
    man = out["manifest"]

    W("### T1 — shape, licence and the parameter cross-check\n")
    W("| donor | family | D | L | V | tied | params computed | params advertised (safetensors) | delta | cross-check | licence |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["meta"]
        d = r["params_delta_pct"]
        adv = r["params_total_advertised"]
        adv_s = "—" if not adv else "%.3fB" % (adv / 1e9)
        d_s = "—" if d is None else "%+.2f%%" % d
        lic = man[r["repo"]].get("license") or "—"
        W(f"| `{r['repo']}` | {m['model_type']} | {m['hidden_size']} | {m['n_layers']} | {m['vocab']} | "
          f"{'yes' if m['tied'] else 'no'} | {r['params_total_computed']/1e9:.3f}B | {adv_s} | "
          f"{d_s} | {r['crosscheck']} | {lic} |")
    for r in bad:
        W(f"| `{r['repo']}` | — | | | | | | | | **{r['status']}** | — |")

    W("\n### T2 — operator map (which layers carry attention), read from the config\n")
    W("| donor | attn layers | SSM layers | linear-attn layers | source key | attention fraction |")
    W("|---|---|---|---|---|---|")
    for r in rows:
        m = r["meta"]
        na, nl = m.get("n_attn_layers", 0), m["n_layers"]
        W(f"| `{r['repo']}` | {na} | {m.get('n_ssm_layers',0)} | {m.get('n_linattn_layers',0)} | "
          f"`{m.get('layer_types_source')}` | {100.0*na/nl:.0f}% |")

    W("\n### T3 — resident weight footprint, both precision maps\n")
    W("| donor | (a) all-ternary 0.5 B/w | of which scales | of which 32-pad | (b) P61/D9 map | SKU-A 16GB @32K, 4-bit KV, map (a) | SKU-B 64GB @128K, 4-bit KV, map (a) |")
    W("|---|---|---|---|---|---|---|")
    for r in rows:
        a, b = r["fp_all_ternary"], r["fp_p61_map"]
        sa = r["sku"]["A_16GB@32K|all_ternary|kv4bit"]
        sb = r["sku"]["B_64GB@128K|all_ternary|kv4bit"]
        W(f"| `{r['repo']}` | {fmt_b(a['total'])} | {fmt_b(a['scales'])} | {fmt_b(a['padding'])} | "
          f"{fmt_b(b['total'])} | {fmt_b(sa['bytes'])} {'FIT' if sa['fits'] else '**OVER**'} | "
          f"{fmt_b(sb['bytes'])} {'FIT' if sb['fits'] else '**OVER**'} |")

    W("\n### T4 — KV cache, attention layers only\n")
    W("| donor | KV elems/tok/layer | kind | 4K fp16 | 4K 4-bit | 32K fp16 | 32K 4-bit | 128K fp16 | 128K 4-bit |")
    W("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["meta"]
        c = [fmt_b(r["kv"][f"{x}_{y}"]["bytes"]) for x in CTX_POINTS for y in ("fp16", "4bit")]
        W(f"| `{r['repo']}` | {m.get('kv_elems_per_tok_per_layer')} | {m.get('kv_kind')} | " + " | ".join(c) + " |")

    W("\n### T5 — per-token byte traffic and the three time columns (weights only, no KV)\n")
    W("| donor | active params/tok | proj+head fp32 bytes/tok | r(proj) GB/s | ternary MLP bytes/tok | ternary expert bytes/tok | expert calls/tok | tok/s PESS | tok/s OPT | tok/s **MEASURED-ONLY** |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        t = r["time"]["nokv"]
        p = r["per_token_bytes_p61"]
        W(f"| `{r['repo']}` | {r['active_params_per_token']/1e6:.0f}M | "
          f"{fmt_b(p['proj']+p['head']+p['other'])} | "
          f"{t['proj_rate_central']:.2f} | "
          f"{fmt_b(p['lut_dense'])} | {fmt_b(p['lut_moe'])} | {r['meta'].get('expert_calls_per_token',0)} | "
          f"{t['toks'][0]:.2f} | {t['toks'][1]:.2f} | **{t['toks_central']:.2f}** |")
    W("\nProjection rate is no longer a curve lookup. Every donor is priced on the **measured donor "
      "stream, 37.74 ± 0.18 GB/s** (`DONOR_PROJ_RATE.md` §5.2) — the whole per-token proj+head "
      "traffic timed in layer order, which is the same object this term models. The retired 34.0 "
      "floor was never measured and is 11% below it.\n")

    W("\n### T6 — tok/s including KV-cache read traffic, vs the sealed ≥10 tok/s gate\n")
    W("Reported as **central [interval]**. The single-corner \"MEASURED-ONLY\" column is **withdrawn** "
      "(audit F1): it took the pessimistic edge of every bracket at once and that is what produced "
      "\"zero donors pass\". `nat` = the donor's own `max_position_embeddings`; a context beyond it is "
      "marked `>nat` and is **not natively servable** without RoPE extension or the recall tier.\n")
    W("| donor | nat ctx | 32K central [lo..hi] | 128K central [lo..hi] | UNCONVERTED | **CONVERTED** central [lo..hi] | gate | KV-free |")
    W("|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["meta"]
        nat = m.get("native_ctx")
        cells = []
        for ctx in (32768, 131072):
            t = r["time"][f"{ctx}_4bit"]
            mark = " `>nat`" if t.get("beyond_native") else ""
            cells.append(f"{t['toks_central']:.2f} [{t['toks'][0]:.2f}..{t['toks'][1]:.2f}]{mark}")
        te = r["time"]["converted_4bit"]
        tu = r["time"]["skuA_eff_4bit"]
        gate = te["toks_central"] >= TOKS_GATE
        gate_lo = te["toks"][0] >= TOKS_GATE
        if gate and gate_lo:
            gstr = "**PASS**"
        elif gate:
            gstr = "**STRADDLE**"
        elif te["toks_kvfree_central"] >= TOKS_GATE:
            gstr = "FAIL (KV-only)"
        else:
            gstr = "**FAIL**"
        W(f"| `{r['repo']}` | {nat if nat else '?'} | " + " | ".join(cells) +
          f" | {tu['toks_central']:.2f} | {te['toks_central']:.2f} "
          f"[{te['toks'][0]:.2f}..{te['toks'][1]:.2f}] | {gstr} | {te['toks_kvfree_central']:.2f} |")
    W("\nKV priced at the measured aggregate DRAM stream [40–44 GB/s], central 42; 4-bit KV assumed, "
      "which **does not exist in this engine** (audit F8) — see the fp16 column in the arithmetic doc.\n")
    W("**`gate @SKU-A` reads PASS only when the LOWER BOUND clears 10 tok/s.** A central estimate on "
      "the right side of a gate whose interval straddles it is reported as straddling, not as a pass.\n")
    W("**KV-free central** isolates the owner's directive: a donor may **not** be eliminated solely "
      "because full-native KV does not fit or stream — that is what the recall tier absorbs. Any donor "
      "failing the gate on KV but clearing it KV-free is a *recall-tier* question, not an elimination.\n")

    W("\n### T7 — head alone (mandate §5's named first-class problem)\n")
    W("| donor | V | head bytes fp32 | ms/token fp32 | head bytes ternary | ms/token ternary (LUT bracket) | head share of the fp32 per-token stream |")
    W("|---|---|---|---|---|---|---|")
    for r in rows:
        h = r["head"]
        p = r["per_token_bytes_p61"]
        share = h["bytes_fp32"] / (p["proj"] + p["head"] + p["other"]) * 100
        W(f"| `{r['repo']}` | {r['meta']['vocab']} | {fmt_b(h['bytes_fp32'])} | "
          f"{h['ms_fp32'][1]:.1f}–{h['ms_fp32'][0]:.1f} | "
          f"{fmt_b(h['bytes_ternary'])} | {h['ms_ternary_lut'][1]:.1f}–{h['ms_ternary_lut'][0]:.1f} | {share:.0f}% |")

    W("\n### T8 — MoE granularity vs the ρ-safe 48 KB chunk\n")
    W("| donor | experts | top-k | expert width | expert chunk (ternary+scales) | ≥48 KB? |")
    W("|---|---|---|---|---|---|")
    for r in rows:
        m = r["meta"]
        if not m.get("n_experts"):
            continue
        W(f"| `{r['repo']}` | {m['n_experts']} | {m['top_k']} | {m['expert_width']} | "
          f"{r['expert_chunk_kb']:.0f} KB | {'yes' if r['rho_safe'] else '**NO**'} |")

    W("\n### T9 — reproducibility manifest\n")
    W("| donor | resolved revision SHA | config.json sha256 | bytes | status |")
    W("|---|---|---|---|---|")
    for repo in CANDIDATES:
        rec = man[repo]
        if rec.get("status") != "OK":
            W(f"| `{repo}` | — | — | — | **UNAVAILABLE** — {rec.get('reason','')[:90]} |")
        else:
            W(f"| `{repo}` | `{rec['revision_sha']}` | `{rec['config_sha256'][:32]}…` | "
              f"{rec['config_bytes']} | OK |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nmarkdown tables -> {path}")


def shortlist(out):
    for sku, ctx, budget in (("SKU-A 16GB", 32768, SKU_A_BYTES), ("SKU-B 64GB", 131072, SKU_B_BYTES)):
        print(f"\n\n=== RANKED: {sku}, native attention window {ctx}, 4-bit KV, "
              f"all-ternary footprint map (most permissive) + P61/D9 time model ===")
        print("    tok/s = CENTRAL [lo..hi]. gate PASS requires the LOWER BOUND >= 10.")
        print("    'kvfree' = same model with the KV term removed: a donor that clears the gate")
        print("    there but not with KV is a RECALL-TIER question, not an elimination (owner directive).")
        print(f"{'donor':40s} {'RAM':>9s} {'fit':>5s} {'nat':>7s} {'central':>8s} {'lo..hi':>16s} "
              f"{'kvfree':>8s} {'gate':>18s}  binding")
        cand = []
        for r in out["rows"]:
            if r["status"] in ("UNAVAILABLE", "REFUSED"):
                continue
            k = f"{'A_16GB@32K' if budget==SKU_A_BYTES else 'B_64GB@128K'}|all_ternary|kv4bit"
            v = r["sku"].get(k, {})
            key = "converted_4bit" if budget == SKU_A_BYTES else f"{ctx}_4bit"
            t = r["time"][key]
            cand.append((t["toks_central"], r, v, t))
        for score, r, v, t in sorted(cand, key=lambda x: -x[0]):
            fits = v.get("fits")
            nat = r["meta"].get("native_ctx")
            lo, hi = t["toks"]
            if lo >= TOKS_GATE:
                gate = "PASS" if fits else "PASS(over RAM)"
            elif t["toks_central"] >= TOKS_GATE:
                gate = "STRADDLE"
            elif t["toks_kvfree_central"] >= TOKS_GATE:
                gate = "FAIL(KV-only)"
            else:
                gate = "FAIL"
            print(f"{r['repo']:40s} {fmt_b(v.get('bytes')):>9s} "
                  f"{('FIT' if fits else 'OVER') if fits is not None else 'UNK':>5s} "
                  f"{(str(nat) if nat else '?'):>7s} {t['toks_central']:8.2f} "
                  f"{lo:7.2f}..{hi:<7.2f} {t['toks_kvfree_central']:8.2f} "
                  f"{gate:>18s}  {r['dominant_term']}"
                  f"{'  [' + r['status'] + ']' if r['status'] != 'OK' else ''}")


# ---------------------------------------------------------------------------
# planted control (project law 6.3 -- minimal significant corruption)
# ---------------------------------------------------------------------------
def sandbox_cfg():
    """The project's OWN 8.3M sandbox engine, transcribed from benchmarks/phase60/engine.c:51-75,
    expressed in the qwen3_moe config schema so it goes through the SAME code path as a donor."""
    return {
        "model_type": "qwen3_moe",
        "hidden_size": 256,          # D
        "num_hidden_layers": 6,      # L
        "vocab_size": 1024,          # V
        "num_attention_heads": 8,    # H
        "num_key_value_heads": 8,
        "head_dim": 32,              # HD = D/H
        "num_experts": 32,           # E
        "num_experts_per_tok": 8,    # KTOP
        "moe_intermediate_size": 128,  # HID_E
        "intermediate_size": 1024,   # MLP_HID (dense path, unused by the MoE engine)
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "sliding_window": 128,       # WIN
        "attention_bias": False,
    }


def control_streamed_bytes(cfg):
    R = Reader(cfg)
    mats, meta = build_matrices(R)
    ptb = per_token_bytes(mats, TERNARY_MAP_B)
    return ptb["lut"], meta


def cmd_control(args):
    buf = io.StringIO()

    class _Tee:
        def write(self, x):
            sys.__stdout__.write(x)
            buf.write(x)

        def flush(self):
            sys.__stdout__.flush()

    old, sys.stdout = sys.stdout, _Tee()
    try:
        rc = _control_body()
    finally:
        sys.stdout = old
    log = getattr(args, "log_out", None)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("")
        print("control log -> " + log)
    return rc


def _control_body():
    print("=" * 78)
    print("PLANTED CONTROL -- the tool must FIRE on a known-positive before its nulls count")
    print("=" * 78)
    ok = True

    print("\n[C1] known-positive: the project's own 8.3M sandbox engine")
    print("     target = 2400 KB/token, counted in-engine, docs/SIZING.md")
    b, meta = control_streamed_bytes(sandbox_cfg())
    target = 2400 * KB
    print(f"     predicted streamed ternary bytes/token = {b:.0f} B = {b/KB:.1f} KB")
    print(f"     independent count                      = {target:.0f} B = {target/KB:.1f} KB")
    err = (b - target) / target * 100
    print(f"     error = {err:+.4f}%")
    if abs(err) < 0.01:
        print("     C1 PASS")
    else:
        print("     C1 FAIL -- the formula is wrong; DO NOT tune the tool until it agrees")
        ok = False
    per_expert = b / (meta["top_k"] * meta["n_moe_layers"])
    print(f"     decomposition: {meta['n_moe_layers']} layers x top-{meta['top_k']} x "
          f"{per_expert:.0f} B/expert ({per_expert/KB:.1f} KB: "
          f"{3*256*128*B_TERNARY:.0f} B codes + {(128+128+256)*4} B per-row scales)")

    print("\n[C2] minimal significant perturbation: num_experts_per_tok 8 -> 7 (one field, one step)")
    c = sandbox_cfg()
    c["num_experts_per_tok"] = 7
    b2, _ = control_streamed_bytes(c)
    exp = target * 7 / 8
    print(f"     predicted = {b2:.0f} B = {b2/KB:.1f} KB ; expected {exp:.0f} B = {exp/KB:.1f} KB")
    print(f"     moved by {b2-b:+.0f} B ({(b2-b)/b*100:+.2f}%), expected {exp-target:+.0f} B (-12.50%)")
    if abs(b2 - exp) < 1 and b2 < b:
        print("     C2 PASS -- right direction, right magnitude")
    else:
        print("     C2 FAIL")
        ok = False

    print("\n[C3] refusal: a config missing a field the tool needs must raise a NAMED error")
    for k in ("num_hidden_layers", "hidden_size", "vocab_size"):
        c = sandbox_cfg()
        del c[k]
        try:
            control_streamed_bytes(c)
            print(f"     removed '{k}' -> NO ERROR RAISED  <<< C3 FAIL (silent-zero failure mode)")
            ok = False
        except MissingConfigField as e:
            print(f"     removed '{k}' -> MissingConfigField: {e}   OK")
        except Exception as e:
            print(f"     removed '{k}' -> {type(e).__name__}: {e}  <<< C3 FAIL (unnamed error)")
            ok = False
    print("")
    print("[C4] rev-C terms: scan-recurrence, attention, and the attention->SSM weight swap")
    print("     Project law 6.3: a term is untrusted until its guard is shown to FIRE and to STAY")
    print("     SILENT.  rev C added three terms and shipped none of these; added here.")
    cfg = sandbox_cfg()
    R = Reader(cfg); mats, meta = build_matrices(R)
    ptb = per_token_bytes(mats, TERNARY_MAP_B)
    D = meta["hidden_size"]; L = meta["n_layers"]; na = meta.get("n_attn_layers", 0)
    attn_tot = sum(m.out * m.inn * m.per_token for m in mats if m.organ == "attn_proj")
    attn_pl = attn_tot / na if na else 0.0
    swap = (ssm_block_params(D) - attn_pl) * B_FP32

    def T(n_att, sw=None, mode="swa"):
        return time_model(ptb, meta, 0.0, 0.0, n_att_override=n_att, attn_mode=mode,
                          attn_ctx=32768, conv_swap_bytes_per_layer=sw)

    # --- C4a scan term: zero when nothing is converted, exact when one layer is ---
    t_all = T(L); t_one = T(L - 1)
    unit = SCAN_US_ANCHOR * (2.0 * D * SSM_STATE_N) / SCAN_ANCHOR_DEN
    ok4 = True
    print(f"     C4a NEGATIVE: n_att=L={L} -> scan = {t_all['t_scan_us'][2]:.6f} us "
          f"-> {'silent (exactly 0)' if t_all['t_scan_us'][2] == 0.0 else '** FIRED WRONGLY **'}")
    if t_all["t_scan_us"][2] != 0.0:
        ok4 = False
    d = t_one["t_scan_us"][2] - t_all["t_scan_us"][2]
    print(f"     C4b POSITIVE: n_att=L-1 -> scan moved {d:+.6f} us, predicted {unit:+.6f} "
          f"-> {'FIRES at exactly the predicted size' if abs(d - unit) < 1e-9 else '** WRONG SIZE **'}")
    if abs(d - unit) >= 1e-9:
        ok4 = False

    # --- C4c attention term: zero at n_att=0, exact per-layer step ---
    t0 = T(0); t1 = T(1)
    unit_a = SWA_US_ANCHOR * (D / 256.0)
    print(f"     C4c NEGATIVE: n_att=0 -> attention = {t0['t_att_us'][2]:.6f} us "
          f"-> {'silent (exactly 0)' if t0['t_att_us'][2] == 0.0 else '** FIRED WRONGLY **'}")
    if t0["t_att_us"][2] != 0.0:
        ok4 = False
    da = t1["t_att_us"][2] - t0["t_att_us"][2]
    print(f"     C4d POSITIVE: n_att=1 -> attention moved {da:+.6f} us, predicted {unit_a:+.6f} "
          f"-> {'FIRES at exactly the predicted size' if abs(da - unit_a) < 1e-9 else '** WRONG SIZE **'}")
    if abs(da - unit_a) >= 1e-9:
        ok4 = False

    # --- C4e SSM weight swap (the audit F4 term): 0 when nothing converts, exact when one does ---
    n0 = T(na, swap); n1 = T(na - 1, swap)
    print(f"     C4e NEGATIVE: n_att=n_attn_layers={na} -> swap charged "
          f"{n0['conv_swap_bytes_total']:.0f} B "
          f"-> {'silent (exactly 0)' if n0['conv_swap_bytes_total'] == 0 else '** FIRED WRONGLY **'}")
    if n0["conv_swap_bytes_total"] != 0:
        ok4 = False
    db = n1["conv_swap_bytes_total"] - n0["conv_swap_bytes_total"]
    print(f"     C4f POSITIVE: one layer converted -> swap charged {db:+.0f} B, predicted "
          f"{swap:+.0f} -> {'FIRES at exactly the predicted size' if abs(db - swap) < 1.0 else '** WRONG SIZE **'}")
    if abs(db - swap) >= 1.0:
        ok4 = False
    print(f"     C4g DIRECTION: the swap must be POSITIVE (SSM block is bigger than attention): "
          f"{swap:+.0f} B -> {'correct sign' if swap > 0 else '** WRONG SIGN **'}")
    if swap <= 0:
        ok4 = False
    print("     C4 " + ("PASS -- all three rev-C terms fire and stay silent as required" if ok4 else "FAIL"))
    if not ok4:
        ok = False

    # must-pass direction: the unperturbed config must NOT raise
    try:
        control_streamed_bytes(sandbox_cfg())
        print("     unperturbed config -> no refusal   OK (guard exercised in both directions)")
    except Exception as e:
        print(f"     unperturbed config -> {type(e).__name__} <<< C3 FAIL (guard fires on a good input)")
        ok = False

    print("\n" + ("ALL CONTROLS PASS" if ok else "CONTROL SUITE FAILED"))
    return 0 if ok else 1


DOC = os.path.abspath(os.path.join(HERE, "..", "..", "docs", "research",
                                   "DONOR_STAGE1_ARITHMETIC.md"))


def cmd_doc(args):
    """Regenerate the three machine-generated regions of DONOR_STAGE1_ARITHMETIC.md in place.

    The prose is hand-written and untouched; only the marked regions are replaced, so every
    number in the document is generated, never transcribed.
    """
    import subprocess

    inv = json.load(open(args.json_out, encoding="utf-8"))
    tables = open(args.md_out, encoding="utf-8").read().rstrip()
    control = open(args.control_log, encoding="utf-8").read().rstrip()
    hfv = subprocess.run([sys.executable, "-c",
                          "import huggingface_hub;print(huggingface_hub.__version__)"],
                         capture_output=True, text=True).stdout.strip()
    tool = open(os.path.abspath(__file__), "rb").read()
    rc = inv["rate_constants"]
    env = "\n".join([
        "| item | value |", "|---|---|",
        f"| generated (UTC) | {inv['generated_utc']} |",
        f"| python | {inv['python']} |",
        f"| huggingface_hub | {hfv} |",
        f"| platform | {inv['platform']} |",
        f"| `donor_inventory.py` sha256 | `{hashlib.sha256(tool).hexdigest()}` |",
        f"| `donor_inventory.py` bytes | {len(tool)} |",
        f"| rate-constant source | {rc['source']} |",
        f"| r(size) curve MB | {rc['rcurve_mb']} |",
        f"| r(size) curve GB/s | {rc['rcurve_gbs']} |",
        f"| r asymptote GB/s (extrapolated) | {rc['rcurve_asymptote']} |",
        f"| ternary routed-expert GB/s (engine-integrated) | {rc['lut_gbs_moe_integrated']} |",
        f"| ternary dense-MLP GB/s (engine-integrated) | {rc['lut_gbs_dense_integrated']:.3f} |",
        f"| ternary kernel-pure ceiling GB/s | {rc['lut_gbs_kernel_pure']} |",
        f"| expert dispatch overhead | {rc['expert_dispatch_us']} us/call |",
        f"| DRAM aggregate GB/s (KV) | {rc['dram_aggregate_gbs']} |",
        f"| declared OS/allocator margin | {rc['os_margin_bytes']/1024**3:.1f} GiB |",
    ])

    s = open(args.doc, encoding="utf-8").read()
    for tag, body in (("GENERATED TABLES", tables), ("CONTROL LOG", control), ("ENV", env)):
        a, b = f"<!-- BEGIN {tag} -->", f"<!-- END {tag} -->"
        if a not in s or b not in s:
            print(f"  marker {tag} missing -- skipped")
            continue
        i, j = s.index(a), s.index(b)
        s = s[:i + len(a)] + "\n" + body + "\n" + s[j:]
    open(args.doc, "w", encoding="utf-8").write(s)
    print(f"doc regenerated -> {args.doc}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    a = sub.add_parser("analyze")
    a.add_argument("--json-out", default=os.path.join(HERE, "inventory.json"))
    a.add_argument("--md-out", default=os.path.join(HERE, "tables.md"))
    c = sub.add_parser("control")
    c.add_argument("--log-out", default=os.path.join(HERE, "control.log"))
    d = sub.add_parser("doc")
    d.add_argument("--json-out", default=os.path.join(HERE, "inventory.json"))
    d.add_argument("--md-out", default=os.path.join(HERE, "tables.md"))
    d.add_argument("--control-log", default=os.path.join(HERE, "control.log"))
    d.add_argument("--doc", default=DOC)
    cu = sub.add_parser("curve", help="tok/s vs retained attention layers (audit C12)")
    cu.add_argument("--json-in", default=os.path.join(HERE, "inventory.json"))
    cu.add_argument("--ctx", type=int, default=32768)
    cu.add_argument("--donor", nargs="*")
    gr = sub.add_parser("grid", help="the 2x2 of KV precision x MLP constant (audit C11)")
    gr.add_argument("--json-in", default=os.path.join(HERE, "inventory.json"))
    gr.add_argument("--ctx", type=int, default=32768)
    gr.add_argument("--donor", nargs="*")
    cs = sub.add_parser("ctxsweep", help="tok/s vs context + the conversion crossover (audit F4)")
    cs.add_argument("--json-in", default=os.path.join(HERE, "inventory.json"))
    cs.add_argument("--donor", nargs="*")
    args = p.parse_args()
    if args.cmd == "fetch":
        return cmd_fetch(args)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    if args.cmd == "control":
        return cmd_control(args)
    if args.cmd == "doc":
        return cmd_doc(args)
    if args.cmd == "curve":
        return cmd_curve(args)
    if args.cmd == "grid":
        return cmd_grid(args)
    if args.cmd == "ctxsweep":
        return cmd_ctxsweep(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
