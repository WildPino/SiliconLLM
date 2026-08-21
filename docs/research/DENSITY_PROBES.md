# Density Probes D0–D3 — donor adaptation v2

**Branch `research/donor-adaptation`. Builder, 2026-08-21. No GPU session; everything here is CPU-class.**
Governing document: `docs/research/DONOR_V2_DENSITY_PROGRAMME.md`.
Mid-flight reframe of D0 and D2 by the coordinator (prior-art survey) is recorded in §1.3 and applied.

---

## 0. The apparatus — pinned before any result

### 0.1 Donor

Read off the artefact (`config.json` of the pinned snapshot), never from memory:

| field | value |
|---|---|
| model | `Qwen/Qwen2.5-1.5B` (Apache-2.0) |
| revision | `8faed761d45a263340a0528343f099c05c9a4323` |
| `hidden_size` | 1536 |
| `num_hidden_layers` | 28 |
| `intermediate_size` | 8960 |
| `hidden_act` | `silu` (SwiGLU: `down(silu(gate x) * up x)`) |
| heads / kv-heads | 12 / 2 (GQA) |
| `vocab_size` | 151936, `tie_word_embeddings` = true |

Loaded fp32 on CPU, `attn_implementation="eager"`, `torch 2.12.0+cpu`, `transformers 4.57.6`.

### 0.2 Corpus

`benchmarks/donor_adaptation/density/build_calib.py`. 281 MB of mixed general text drawn from
local sources — PG-19 books 134.0 MB, technical markdown (MDN / Rust book / Kubernetes /
TypeScript) 64.4 MB, Python source 72.3 MB, WikiText-2-raw 10.8 MB — cut into 8 KiB chunks and
**globally shuffled in one permutation across all sources** before any split is taken. The project
measured that blocked (file-order) sampling alone cost +0.0339 BPB = 6.8σ on rung-1; order is a
confound and it is removed here by construction, not by hope.

| half | bytes | sha256 (first 16) |
|---|---|---|
| `calib` | 140 927 488 | `10d4d28102625f39` |
| `heldout` | 140 640 256 | `f46b0310c15faec5` |

### 0.3 The unit and the eval slice

**BPB, byte-normalised**, on the held-out half:
`BPB = Σ NLL(nats) / (ln2 · Σ bytes)` where the byte count is the exact UTF-8 length of the span
covered by the scored tokens. A sequence is admitted only if `decode(ids[1:])` re-encodes to
exactly `ids[1:]`, so the byte count is a property of the span and not an approximation
(0 sequences rejected).

| | |
|---|---|
| slice | 24 sequences × 512 tokens, held-out half, seed 1234 |
| scored bytes | 51 870 |
| bytes/token | 4.229 |
| `ids_sha256` | `a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65` |
| **baseline BPB** | **0.767595** |
| slice-resampling SE | 0.0622 (marginal; **paired** deltas are far tighter — see below) |

### 0.4 The harness's own planted control — it must fire before its nulls mean anything

`d_baseline.py`, all four logged:

| control | expectation | measured | |
|---|---|---|---|
| no-op edit (`W ← W·1.0`) | delta **exactly** 0 | `+0.000000000` | PASS |
| kill the single highest-norm FFN output neuron, L14 (1 row of 8960 in 1 layer of 28 = 0.0013% of FFN weights) | delta > 0 | **`+0.000023`** | **FIRED** |
| zero all of `down_proj` in L14 | delta ≫ 0 | `+0.012762` | fired |
| restore | delta exactly 0 | `+0.000000000` | PASS |

The harness is **bit-deterministic**: identical inputs give identical BPB to the last digit, and it
resolves a perturbation of 1.3 × 10⁻⁵ of the FFN weights at 2.3 × 10⁻⁵ BPB. Every delta below is a
paired comparison on the byte-identical slice, so the 0.0622 marginal SE is *not* the resolution —
it is the answer to a different question ("would another slice give another number"), and it is
reported per-row as a paired bootstrap over sequences.

> **Already a finding, from the control itself:** zeroing an entire mid-depth `down_proj` — 13.8M
> weights, 0.9% of the model — costs **+0.0128 BPB**. The donor is grossly redundant across depth.

---

## 1. D2 — is there a basis in which the WEIGHTS are sparse?  **AXIS DEAD**

`d2_basis.py` → `results/d2_basis.json`.

### 1.1 The instrument and its controls

Metric `frac99`: the fraction of matrix entries needed to carry 99% of the Frobenius energy
(lower = sparser). For orthogonal `U, V` the map `W → UᵀWV` preserves the operator and `‖W‖_F`, so
the only question is whether the *entries* concentrate.

| control | expectation | identity | Hadamard | verdict |
|---|---|---|---|---|
| **planted**: `W = H·S·Hᵀ` with S 1%-sparse | Hadamard ≪ identity | 0.7348 | **0.0074** | **FIRED** (99× |
| **planted negative**: i.i.d. Gaussian | Hadamard ≈ identity | 0.7345 | 0.7345 | **FIRED** |
| **no-op**: random permutation of rows/cols | exactly equal to identity | — | — | **exact on all 12 matrices** |

The detector finds a rotation that is known to be there, does not invent one that is not, and its
no-op is exact. Its "no" is therefore worth something.

### 1.2 The answer on real donor weights

`frac99`, 12 matrices spanning depth and organ. **Lower is sparser. The i.i.d.-Gaussian value is 0.7345.**

| matrix | identity | permutation | block-Hadamard | block-rand-orth | dense rand-orth | own singular |
|---|---|---|---|---|---|---|
| L00.q_proj | **0.6415** | 0.6415 | 0.7078 | 0.6945 | 0.7322 | 0.00039 |
| L00.o_proj | **0.7047** | 0.7047 | 0.7331 | 0.7314 | 0.7345 | 0.00053 |
| L00.gate_proj | **0.7215** | 0.7215 | 0.7343 | 0.7346 | — | 0.00009 |
| L00.down_proj | **0.7298** | 0.7298 | 0.7347 | 0.7347 | — | 0.00011 |
| L13.q_proj | **0.7033** | 0.7033 | 0.7288 | 0.7304 | 0.7343 | 0.00043 |
| L13.o_proj | **0.7013** | 0.7013 | 0.7322 | 0.7331 | 0.7347 | 0.00045 |
| L13.gate_proj | **0.7203** | 0.7203 | 0.7344 | 0.7345 | — | 0.00011 |
| L13.down_proj | **0.7192** | 0.7192 | 0.7348 | 0.7345 | — | 0.00010 |
| L27.q_proj | **0.6529** | 0.6529 | 0.7120 | 0.7298 | 0.7345 | 0.00037 |
| L27.o_proj | **0.6835** | 0.6835 | 0.7289 | 0.7308 | 0.7340 | 0.00043 |
| L27.gate_proj | **0.7251** | 0.7251 | 0.7338 | 0.7347 | — | 0.00011 |
| L27.down_proj | **0.7174** | 0.7174 | 0.7347 | 0.7345 | — | 0.00011 |

> **12 of 12: the identity is the sparsest basis tested.** Every rotation family — structured
> (Hadamard), structured-random, and dense Haar — moves the weights *toward* the i.i.d.-Gaussian
> limit 0.7345 rather than away from it. Block-Hadamard is not better than the random-orthogonal
> null; it is the same as the null, to three decimals.

This is exactly what a Gaussian-ish trained matrix should do, and it says the modest concentration
the weights *do* have (0.64–0.73 vs 0.7345) is a property of the coordinates they were trained in.
Rotating destroys it. The only basis that concentrates is `W`'s own singular basis (`frac99` ≈ 1e-4,
i.e. the diagonal), and that basis costs `U` and `V` — which is not a basis change, it is low-rank,
and it is priced in D3.

### 1.3 What this closes, and what the coordinator's prior-art note already closed

The programme called rotate-for-sparsity "as far as I know, open". It is not: DenoiseRotator,
ProcrustesGPT and RotPruner (2025–26) all learn orthogonal transforms for exactly this. The
coordinator's survey adds the decisive engineering facts — DenoiseRotator reaches only 50% / 2:4,
adds a permanently-resident dense `d²`-per-layer rotation (6.7% of an 8B model), and **its
block-diagonal variant, the only one we could afford per token, gives most of the gain back**
(ppl 7.597 → 8.882 from 1 to 8 blocks). My measurement is the same conclusion from the other end:
the structured rotations we could afford are indistinguishable from a random-orthogonal null.

**Verdict D2 (weights): the axis is DEAD.** No affordable orthogonal basis makes donor weights
sparser than the basis they arrive in. The redirected question — rotate to make *activations*
sparse — is D2b, below.

### 1.4 Singular spectra — what this costs axis C (feeds D3)

| matrix | r for 90% | 95% | 99% | full rank | stable rank | energy in top 1% of ranks | top 10% |
|---|---|---|---|---|---|---|---|
| L00.q_proj | 445 | 607 | 918 | 1536 | 7.4 | 0.317 | 0.665 |
| L00.o_proj | 937 | 1079 | 1262 | 1536 | 101.1 | 0.046 | 0.284 |
| L00.gate_proj | 908 | 1069 | 1288 | 1536 | 94.0 | 0.067 | 0.319 |
| L00.down_proj | 1172 | 1310 | 1462 | 1536 | 210.7 | 0.036 | 0.196 |
| L13.q_proj | 547 | 709 | 1009 | 1536 | 83.0 | 0.122 | 0.547 |
| L13.o_proj | 634 | 785 | 1056 | 1536 | 92.6 | 0.081 | 0.429 |
| L13.gate_proj | 1112 | 1279 | 1461 | 1536 | 100.8 | 0.052 | 0.268 |
| L13.down_proj | 1077 | 1241 | 1436 | 1536 | 86.4 | 0.052 | 0.270 |
| L27.q_proj | 425 | 578 | 883 | 1536 | 64.8 | 0.174 | 0.659 |
| L27.o_proj | 596 | 751 | 1026 | 1536 | 47.8 | 0.184 | 0.509 |
| L27.gate_proj | 1161 | 1310 | 1464 | 1536 | 44.1 | 0.064 | 0.244 |
| L27.down_proj | 1147 | 1306 | 1475 | 1536 | 157.4 | 0.046 | 0.245 |

**The spectra are flat.** 99% of the energy needs 57–96% of the full rank; the top 10% of ranks
carries only 20–66% of the energy, and for every FFN organ it is ≤32%. `q_proj` is the one organ
with real spectral decay (r₉₀ = 425–547 of 1536, stable rank 7.4 at L00). **A rank-26 truncation is
1.7% of the rank and captures well under a third of the energy in every FFN organ** — the flat-prior
expectation for D3 is therefore that r=26 does *not* transfer. D3 measures it in BPB rather than
asserting it from energy.

---

## 2. D0b — re-deriving the rho-law block floor at DONOR dimensions

`d0b_rho_floor_donor.py` → `results/d0b_rho_floor_donor.json`. Reuses `d0_layout.py`'s own
`BYTES_PER_WEIGHT`, `run_lengths` and `layout_stats` unmodified (`import d0_layout as D0`) —
nothing here is a parallel reimplementation.

### 2.0 The transplant this probe checks

`results/d0_layout.json`'s `neurons_per_48KB_interleaved = 21.333...` was **measured at OUR
engine's own D=1536** and had been quoted in a 100B-donor analysis as `B_block >= 22` **without
re-derivation**. `bytes_per_neuron` for the 3-organ interleaved layout is analytically
`3 * d_model * BYTES_PER_WEIGHT` — it scales with `d_model` alone and **does not depend on
`d_ffn`** — so the constant is D=1536-specific, not an engine-universal floor.

### 2.1 What is ANALYTIC (closed-form arithmetic, no model, no code path beyond arithmetic)

Sweep over `d_model`, both the per-organ-streamed and the 3-organ-interleaved layout:

| `d_model` | bytes/neuron (per-organ) | neurons/48KB (per-organ) | bytes/neuron (interleaved) | neurons/48KB (interleaved) |
|---|---|---|---|---|
| 1536 | 768.0 | 64.00 | 2304.0 | **21.33** |
| 2048 | 1024.0 | 48.00 | 3072.0 | 16.00 |
| 4096 | 2048.0 | 24.00 | 6144.0 | 8.00 |
| 5120 | 2560.0 | 19.20 | 7680.0 | 6.40 |
| 8192 | 4096.0 | 12.00 | 12288.0 | **4.00** |

Control: this script's own D=1536 row is bit-identical to `d0_layout.json`'s
`neurons_per_48KB_per_organ` (64.0) and `neurons_per_48KB_interleaved` (21.333...) — **FIRED**
(`reproduces_d0_artefact_check`).

Real Llama-3-70B-class per-matrix shapes (`d_model=8192, d_ffn=28672`, GQA 64/8 heads), read as
`(out_features, in_features)`:

| organ | shape | axis | structured-unit length | bytes/unit | units/48KB |
|---|---|---|---|---|---|
| q_proj | 8192×8192 | row | 8192 | 4096 | 12.00 |
| k_proj | 1024×8192 | row | 8192 | 4096 | 12.00 |
| v_proj | 1024×8192 | row | 8192 | 4096 | 12.00 |
| o_proj | 8192×8192 | col | 8192 | 4096 | 12.00 |
| gate_proj | 28672×8192 | row | 8192 | 4096 | 12.00 |
| up_proj | 28672×8192 | row | 8192 | 4096 | 12.00 |
| down_proj | 8192×28672 | col | 8192 | 4096 | 12.00 |
| **3-organ interleaved FFN neuron** | gate row + up row + down col | — | 3×8192 | **12288** | **4.00** |

Every organ lands on **12.00 units/48KB** (per-organ streamed) because the structured-axis unit
length is `d_model` in every one of the seven organs — row length = `in_features = d_model` for
q/k/v/gate/up, column length = `out_features = d_model` for o/down — **regardless of `d_ffn`**,
the GQA head-count asymmetry, or the FFN expansion ratio (28672/8192 = 3.5×). The interleaved
FFN-neuron cost (gate row + up row + down **column**) is likewise `d_ffn`-free: `d_ffn` cancels
out of the formula entirely, because it only sets how many neurons there are, never how many
bytes one neuron costs. Control: `real_donor_shapes_match_uniform_D_formula` — the real-shape
calculation (4.00) matches the uniform-`D` formula `32768/D` at D=8192 (4.00) exactly —
**FIRED**.

Hand-derivable control at D=1024 (chosen so the answer is checkable by hand before running
anything): expect bytes/neuron 512.0/1536.0, neurons/48KB 96.0/32.0 (per-organ/interleaved).
Code returned exactly that — **FIRED** (`hand_derivable_control`).

### 2.2 What is MEASURED (the actual `d0_layout.py` code path, run on synthetic donor-scale masks)

No donor weights or activations exist in this repo, so nothing below is a measurement of a real
donor's co-activation structure — it is a measurement of the **layout-accounting code path**
(`run_lengths`/`layout_stats`, imported unmodified from `d0_layout.py`) exercised at the donor's
own `d_ffn = 28672`, and is labelled as such throughout.

| control | expectation | measured | verdict |
|---|---|---|---|
| runs planted at length 4 (the donor interleaved block), N=28672, p=0.10 | `mean_run_neurons == 4` exactly | `4.000000` | **FIRED** |
| i.i.d. mask, same density, N=28672 | `mean_run_neurons ~= 1/(1-p) = 1.111` | `1.111001` | **FIRED** |

With both controls firing, the same synthetic i.i.d.-active mask (a deliberately unstructured
null — **not** a claim about real donor co-activation) was pushed through the block accounting
at the donor's two candidate granularities, holding the activation pattern fixed and varying only
block size:

| block size | block-skippable fraction (synthetic i.i.d., p=0.10) |
|---|---|
| 12 (per-organ) | 0.2824 |
| 4 (interleaved) | 0.6563 |

Both match the closed-form binomial-empty-block probability `(1-p)^block_size` exactly
(`0.9^12 = 0.2824`, `0.9^4 = 0.6561` vs measured 0.6563) — the instrument's block accounting is
doing exactly binomial bookkeeping at this scale, as it should on an unstructured null.

### 2.3 Does the 3-organ interleave survive donor shapes?

**Yes, dimensionally.** §2.1 shows the per-neuron byte cost is driven by `d_model` alone in every
organ; the gate/up vs. down asymmetry (`d_ffn ≠ d_model`) never enters the formula, so it cannot
break it. The one **physical** caveat, and it is scale-independent (present identically at
D=1536, not introduced by donor scale): `down_proj`'s "column" is contiguous only if `down_proj`
is stored **transposed** (`[d_ffn, d_model]` instead of `[d_model, d_ffn]`), so that row `j` of
the transposed store equals column `j` of `down_proj`. This is a pre-existing requirement of the
interleaved layout, not a new problem.

### 2.4 Verdict

> **The transplanted `B_block >= 22` is wrong at donor scale by 5.33×.** The correctly re-derived
> engine-legal interleaved block floor at Llama-3-70B-class dimensions (`d_model=8192`) is
> **4 neurons**, not ≥22. `21.33 / 4.00 = 5.333...`, exactly the ratio implied by
> `d_model_donor / d_model_ours = 8192/1536 = 5.333...` (the two formulas are both `32768/D`, so
> the ratio is exact by construction, not a coincidence to re-check).
>
> Direction of the error matters: the donor's true minimum contiguous unit is **smaller** (finer
> granularity, 4 neurons) than the transplanted assumption (22 neurons) — not larger. §2.2's
> synthetic-mask measurement shows this cuts the right way for block-sparsity quality: at the
> *true* finer granularity (block=4) the same activation pattern yields a **higher** skippable
> fraction (0.656) than at the *assumed* coarser granularity (block=12, 0.282). A block-sparsity
> quality analysis that assumed 22-neuron blocks was assuming a more restrictive constraint than
> the engine actually imposes at donor scale — the correction loosens the constraint, it does not
> tighten it. This still needs re-running through any specific quality analysis that used the `22`
> figure, since a smaller legal block also means more block boundaries to route across, which is
> a separate (routing) question this probe does not answer.

All controls fired (`all_measured_controls_fired: true`); git revision `e410243` on
`research/donor-adaptation`.

---

*(D0, D1, D2b, D3 sections appended as each probe lands — see §6 for the running status.)*

---

## 3. D1 — how much quality does sparsity actually cost, per organ?  **DONE**

`results/d1_pruning.json`, 73 organ-sweep records (44 added this session) + 4 planted controls,
revision `e410243`. Baseline BPB **0.7675949677540373**, reproduced bit-identically at the start of
the completion run. Levels {0.25, 0.50, 0.75, 0.90}; modes: unstructured magnitude, row-structured,
and **contiguous block-structured at the engine-legal granularity** (`block_size = 64` units at our
`D = 1536`, read off the D0 formula, not hardcoded).

### 3.1 Controls — all four re-fired in-session

| control | expect | measured Δ BPB | fired |
|---|---|---|---|
| `prune@0pct_downproj` | Δ == 0 | +0.00000 ± 0.00000 | yes |
| `struct@5pct_o_proj` | Δ > 0 | **+4.02442** ± 0.09798 | yes |
| `struct@100pct_v_proj` | Δ ≫ 0 | **+3.65377** ± 0.07791 | yes |
| `restore_exact` | Δ == 0 | +0.00000 ± 0.00000 | yes |

### 3.2 ⚠ Instrument defect found: the block mask quantises, and two points are contaminated

The block mask can only zero whole blocks, so the achievable sparsity levels are quantised to
`k / n_blocks_per_layer`. Where `n_blocks_per_layer` is small this silently rounds the requested level:

| organ | blocks/layer | requested 0.90 → **actually zeroed** |
|---|---|---|
| `k_proj` | **4** | **1.0000** ← 100%, not 90% |
| `v_proj` | **4** | **1.0000** ← 100%, not 90% |
| `q_proj`, `o_proj` | 24 | 0.9167 |
| `gate/up/down_proj` | 140 | 0.9000 |

**`v_proj/block_structured/90%` is bit-identical to the `struct@100pct_v_proj` control** — same BPB
(4.421361169900009), same Δ, same SE — because it *is* that control. Same for `k_proj` at 90%.

**These two points are struck from the sweep.** They are not results; they are the 100% case wearing a
90% label. The probe caught itself only because it records `zero_frac` alongside the requested `level` —
had it recorded the request alone, two "90% sparsity" numbers would have entered the analysis as the
strongest evidence in the table. **Any masking instrument must report the sparsity it ACHIEVED, never
the one it was asked for.**

Also noted: the printed verdict string labels `prune@0pct` and `restore_exact` — both of which expect
Δ == 0 and correctly got exactly 0 — as `[SIGNIFICANT]`. The JSON `fired` field is correct; the log
label is not. A "SIGNIFICANT" tag on a null-expected control is precisely the label that would let a
broken null through later. Cosmetic today, fix before reuse.

### 3.3 The result: contiguous-block sparsity is catastrophic at OUR granularity, at every level

Absolute Δ BPB against σ_seed = 0.005. **Every** block-structured point at **every** organ and **every**
level is far outside noise — the mildest, `k_proj` at 25%, is +0.132 = **26 σ**. The worst are +4.8.

Coarsening penalty, Δ(block) / Δ(unstructured) at equal sparsity, contaminated points excluded:

| s | measured penalty range | dossier Table 1 |
|---|---|---|
| 0.25 | **17.4× – 826×** | (not tabulated) |
| 0.50 | **7.9× – 135×** | 1.6× |
| 0.75 | 2.4× – 10.3× | (between rows) |
| 0.90 | **0.9× – 3.1×** | 2.4× |

**Two findings, and the second matters more than the first.**

1. **Magnitude.** At s = 0.50 the dossier's bound says 1.6×; measured is 7.9–135×. At `q_proj` the ratio
   is 826× at s = 0.25.
2. **Direction.** The measured penalty **falls monotonically with sparsity**; the dossier's Table 1
   **rises** (1.6 → 1.6 → 2.4 → 3.8). The trend has the wrong sign. This is the empirical counterpart of
   the algebraic finding that formula (23)'s multiplicative factor contains no `s` at all: a constant
   can match a falling curve at exactly one point, and at s = 0.90 the measured 0.9–3.1× does bracket
   the claimed 2.4×. **The model is right at one point by coincidence and wrong everywhere else.**

### 3.4 What this does and does NOT establish

**Does not refute the joint solver.** These are *one-shot magnitude* masks with **no reconstruction**.
The entire purpose of a Hessian-weighted layer-wise solve is to beat naive magnitude pruning, so
**D1 measures the floor the solver has to clear, not a verdict on the solver.** Read this way the
numbers are an argument *for* reconstruction, not against it: if naive block-masking costs +0.31 BPB
at gate_proj/25%, there is a great deal of room for a solver to recover.

**Does establish** that the dossier's recommended operating point `s = 0.80` rests on a quality model
that is falsified in both magnitude and direction, and that the quality–speedup product it was derived
from cannot be evaluated from that model. The right operating sparsity is still open.

**Caveat that ties D1 to D0b.** This sweep runs at **our** `D = 1536`, where the engine-legal block is
**64 units**. At donor width the floor is **12** (per-organ) — 5.3× finer in neuron count. Since the
penalty is a coarsening effect, it should be substantially milder at donor scale. **D1's numbers are an
upper bound on the donor-scale cost, not an estimate of it.** The two probes must be read together.

## 6. Status

| probe | state |
|---|---|
| harness + baseline | **done**, controls fired |
| D2 (weight basis) | **done — axis dead** |
| D0 (layout / byte-run-length, reframed) | running |
| D0b (rho-law block floor at donor dimensions) | **done — transplanted B_block>=22 was wrong by 5.33x; correct donor floor is 4 neurons** |
| D1 (per-organ pruning, BPB) | **done — block-sparsity catastrophic at D=1536; Table 1 falsified in magnitude AND direction; 2 points struck for mask quantisation** |
| D2b (activation basis, reframed) | queued |
| D3 (low-rank transfer) | queued |
