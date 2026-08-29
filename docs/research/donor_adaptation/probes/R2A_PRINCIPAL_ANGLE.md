# R2a — the principal-angle pre-screen: R2's ceiling, computed before R2 is built

**Builder** · run 2026-08-22 → 2026-08-28 · brief `briefs/BRIEF_R2A_PRINCIPAL_ANGLE_PRESCREEN.md`
(pre-registered, commit `253112b`) **and its Amendment 1** (the decay arm).
Apparatus: `benchmarks/donor_adaptation/r2/r2a_principal_angle.py`.
Results: `benchmarks/donor_adaptation/r2/results/r2a_{selfcheck,main,curve,decay}.json` + `.log`.

**Nothing in the brief was altered. Arms were ADDED** (`C7_random_gauss`, `C8_no_mixing`,
`favor_d14`, and the `elu1_noscale` / `hedgehog0_noscale` scaling checks), each labelled as a Builder
addition where it appears. **Not committed.**

---

## 0. The verdict, first

The quantity the brief asked for is

```
    residual = || W_v^donor · Z* · (I − P_Z) ||_F  /  || W_v^donor · Z* ||_F
    Z = X A_lin^T ,   Z* = X A_soft^T ,   P_Z = projector onto rowspace(Z) ⊆ R^T
```

an **upper bound on R2's achievable quality**, needing one eigendecomposition per (layer, head, arm)
and no solver, no training, no apparatus.

**At the pinned `T/D = 5.333` on Qwen2.5-1.5B, across all 28 layers and all 12 heads (336 pairs per
arm), from five measurement stages totalling **8 364 rows**:**

| finding | number |
|---|---|
| **`elu(x)+1` is indistinguishable from a mixing built from the WRONG SEQUENCE with its own kernel** | `-0.0008 +/- 0.0078`; better in 38/60 |
| **...and its mixing matrix IS a uniform causal average** | 0.4-7 % away in Frobenius, row cosine `0.997-1.000` |
| **...and it is exactly as far from the donor's real attention as that uniform average is** | `8.4122` vs `8.4252`; cosine `0.0672` vs `0.0695` |
| **The content kernel adds nothing on top of a decay envelope, at any of nine `gamma`** | `-0.00004` to `-0.0006` |
| **`taylor2` DOES have real content structure - its own matched nulls go UP, not down** | `-0.1623 +/- 0.0803`, better in **333/336** |
| **...but on this donor it is not approximating `exp` at all** | **96-99 %** of causal pairs, **99.5-99.97 %** of kernel mass, sit below `z = -1` where the map is *decreasing* |
| **The best mixing found anywhere is a SHORT, CONTENT-FREE exponential envelope** | minimum interior at **28/28 layers**; oracle median **0.4194** |
| **...whose optimal `gamma` is depth-dependent and NOT monotone in depth** | inverted U: 0.42 / 0.64 / 0.38 by depth third; `r = -0.125` |
| **Every arm damages the retrieval rows more than the diffuse ones - no exception** | peaked - diffuse `+0.051` to `+0.137`, rising with envelope length |
| **The `T/D` curve has NOT flattened at `T/D = 21.33`** | `elu1` 0.217 -> 0.628 and still rising |

> ### The ceiling this implies for R2
>
> **At `T/D = 5.333` the best arm in the entire probe leaves 42 % of the donor head-output's
> Frobenius NORM unreachable by ANY `W_v`** - an ideal value projection recovers 82 % of the energy
> and no more - **and that best arm is an oracle** (`gamma` chosen per layer on the data it is
> scored on). The best *pre-registered* feature map leaves 46 %. At `T/D = 21.333` the residual is
> 51 % and still climbing, so **that is a lower bound on the loss at the `T/D >= 24` that
> `BRIEF_R2` Amendment A1.3 requires.**
>
> **This is not a statement about the solver.** No `W_v` - optimal, regularised, joint over the GQA
> group, ridge-to-donor or ridge-to-zero - can do better, because these numbers are the component
> of the target orthogonal to everything the solver can reach. Section 3, check S10 verifies that
> against a brute-force least-squares solve at production shapes to `|diff| = 0.0`.
>
> **Recommendation: do not build the R2 solver.** The reasons are now specific:
>
> 1. **Three of the four registered `phi` are not merely at the free floor - their mixing matrices
>    ARE the free floor** (section 8.1). There is nothing in them for a value-side solve to exploit.
> 2. **The fourth, `taylor2`, has genuine content structure** (matched nulls, 333/336) **but is
>    operating outside the regime its source paper requires** (section 8.9), would ship an
>    `m = 16 513` state, and **loses to plain no-mixing on the retrieval stratum** (149/336).
> 3. **What actually moves the ceiling is the temporal envelope, and the content kernel is worth
>    ~0.0003 on top of it.** That is Amendment 1's question answered in the direction it named as
>    the more valuable one: **the structure lives in the envelope, not in the content kernel** - and
>    an envelope needs no feature map, no `W_q`, no `W_k`, and no value-side solve to obtain.
> 4. **Even the oracle envelope beats plain no-mixing by only `-0.019`.**
>
> **Where the remaining effort belongs, on this evidence: the temporal envelope and its
> depth-dependence - not feature maps, and not the value-side solve R2 was written to build.**

**One thing this pre-screen cannot license.** A *low* residual would NOT have said "R2 works". The
Researcher's reading of Taylor-Calibrate — its own limitations section says the value solve improves
the starting point but does not remove the need for downstream training, because cross-layer
interaction and residual-stream drift are untouched by a layer-local objective — means a good
ceiling is **necessary but not sufficient**. A *high* residual is the decisive direction, and that
is what was measured. **The line closes on layer-local grounds, on evidence rather than argument.**

---

## 1. Donor, and its exact revision

Read from the loaded object and from the local snapshot directory, not from prose:

| | |
|---|---|
| repo | `Qwen/Qwen2.5-1.5B` |
| **revision (pinned; and the snapshot leaf actually on disk)** | **`8faed761d45a263340a0528343f099c05c9a4323`** |
| `config.json` sha256 | `0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77` |
| snapshot dir | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761…` |

### 1.1 ACHIEVED architecture — every field printed from the constructed modules

| field | ACHIEVED | read from |
|---|---|---|
| `n_layers` | **28** | `len(model.model.layers)` |
| `D` | **1536** | `q_proj.in_features` |
| `head_dim` | **128** | `self_attn.head_dim` |
| q / k / v out-features | 1536 / 256 / 256 | the `nn.Linear`s |
| `n_q_heads` / `n_kv_heads` | **12 / 2** | `out_features // head_dim` |
| **GQA group size** | **6** | `num_key_value_groups` |
| `scaling` | **0.08838834764831845** (= `128^-0.5`) | `self_attn.scaling` |
| q / k / v bias present | **True / True / True** | `.bias is not None` |
| o bias | False | |
| `sliding_window` | **None**; `layer_types` = `{full_attention}` | `self_attn.sliding_window` |
| param dtype at run time | **`torch.float32`** | `q_proj.weight.dtype` |
| `rms_norm_eps` | 1e-06 | `input_layernorm.variance_epsilon` |
| `rope_theta` | 1e6 | `model.model.rotary_emb.config` |
| total params | 1 543 714 304 | `sum(p.numel())` |

`v_proj` **has a bias**, so Controller-audit finding 3 binds: the bias survives the commuting step
only if the mixing rows sum to 1. Measured over all 2532 rows of the main sweep, `A_soft` row sums
lie in **[0.9999999999999987, 1.0000000000000013]**, and every `φ` is used row-normalised (§3).

`X` is the **post-RMSNorm** activation entering `v_proj` (audit finding 2), captured by a
forward-pre-hook on `q_proj`, whose positional input is the identical tensor.

### 1.2 Calibration data — pinned by hash

| | |
|---|---|
| corpus | `benchmarks/donor_adaptation/density/corpus/calib.txt`, sha256 `10d4d281…c59c89d0`, 140 933 631 B |
| mix | pg19 0.40 / wikitext 0.10 / markdown 0.25 / python 0.25, globally shuffled at 8 KiB |
| arm slice | `part=calib, n_seq=8, seq_len=1024, seed=20260822`, ids sha256 `6e07b7ec…71e52c54` |
| C5′ slice | `part=calib, n_seq=8, seq_len=1024, seed=20260823`, ids sha256 `c165e205…ca07e3298` |
| **`T` ACHIEVED** | **8192** (`n_seq=8 × L_seq=1024`), read off the arrays |
| **`T/D` ACHIEVED** | **5.3333** |

`A` is **block-diagonal over the 8 calibration sequences** — each sequence attends only within
itself, at a real in-distribution context length of 1024. This is stated because it is what makes
the apparatus affordable (§9) and because it bounds what the numbers mean: they describe 1024-token
attention patterns pooled over 8 sequences, not one 8192-token context.

---

## 2. The `T ≤ D` degeneracy — refusal demonstrated firing

§1 of the brief: if `T ≤ D` then generically `rowspace(Z) = R^T`, `P_Z = I`, and the residual is zero
for every `φ`, including a nonsensical one.

**The instrument was deliberately asked for a `T ≤ D` case, and refused:**

```
S4  deliberately asking for T <= D  (n_seq=1 -> T=1024, D=1536)
S4  REFUSED [layer 3]: T=1024 <= D=1536. rowspace(Z) is generically all of R^T, P_Z = I
    and the residual is exactly 0 for EVERY phi including a nonsensical one.
    No residual is emitted. (brief section 1)
```

It fired again inside the `T/D` curve at the `n_seq=1` point, and that refusal is recorded in
`r2a_curve.json → refusals` rather than as a data point.

**The refusal is not theoretical — here is the flattering number it suppressed.** Bypassing the gate
at `T = 1024 < D = 1536`, same layer, same head:

| `A_lin` at `T ≤ D` | residual it would have reported |
|---|---|
| `elu1` | **0.0482** |
| `hedgehog0` | **0.0495** |
| a deliberately nonsensical **non-causal** all-ones/`L` mixing | 0.8260 |

**`0.048` would have been reported as "the ceiling is excellent, build the solver."** It is an
artefact of underdetermination. (The nonsense arm scores high only because it is rank-1; a
nonsensical *full-rank* mixing at `T ≤ D` would also have scored near zero, which is the general
point and the reason the gate is on `T ≤ D` and not on plausibility.)

---

## 3. The instrument must fire before its nulls mean anything

All checks in `r2a_selfcheck.json`, all fp64.

| # | check | result | verdict |
|---|---|---|---|
| S1 | my `A_soft` vs the model's own `output_attentions` (eager), layer 3, all 12 heads | `max_abs = 1.33e-06` | agrees to the fp32 reference's own noise (model runs fp32, check runs fp64) |
| S2 | the commuting step **with the bias**: `(W_v X + b)Aᵀ` vs `W_v(X Aᵀ) + b` | `1.33e-15`; row-sum dev `5.6e-16` | §1.2 holds on the real donor including the Qwen2 `v` bias |
| S3 | every `φ` row-normalised (audit finding 3) | all four: row sums in `[1−2e−16, 1+2e−16]`, all entries ≥ 0, all finite | binding constraint met |
| **S5** | **C1 IDENTITY** (`A_lin := A_soft`) | **`3.43e-08`** (eigh route); **`0.0` exactly** (QR route) | **fires** |
| **S6** | **PLANTED POSITIVE**: target built from the *linear* side, `Y* := W_plant Z` | **`2.26e-06`** | **fires the whole `φ → A_lin → Z` path**, which C1 is blind to |
| S7 | **transpose trap**: `Z := X A_lin` (no transpose) — same shape, silent bug | correct `0.1983` vs bug `0.1934` | **DID NOT FIRE — reported as a negative result, below** |
| S8 | **`O(1)`-state recurrence vs the `L×L` block form** | `max_abs 7.11e-15`, rel `4.90e-16` | the two are the same computation in fp64 |
| S9 | eigh route vs QR route (normal-equation squaring) | `0.198260` vs `0.198260`; rank 1526 vs 1526 | routes agree |
| S9b | the same cross-check on **240 production rows** (layers 0 and 14, all heads, all arms) | **`max|Δresidual| = 3.65e-08`, `max|Δrank| = 0`** | the fast route is validated at production scale |
| **S10** | **is `residual` really `min_W ||W Z − M||_F / ||M||_F`?** brute-force `numpy.linalg.lstsq` at production shapes (`D=1536, T=8192, d_v=128`) | projector `0.8990651710` vs lstsq `0.8990651710`, **`|diff| = 0.0`**; and none of 20 perturbations of the lstsq solution beats it | **the reported number IS the achievable minimum over all `W_v`, not a proxy for it** |

**S10 is the check that makes the whole document mean what it says.** Everything else verifies that
the pipeline builds the right `Z` and `M`; S10 verifies that the projector quantity really is the
best any value projection could do. It agrees with a brute-force least-squares solve to `0.0` at the
exact shapes used in the sweep.

**C1 is a TAUTOLOGY of the algebra, and is labelled as one.** With `A_lin := A_soft`, `Z = Z*`, so
`rowspace(M) ⊆ rowspace(Z)` by construction and the residual is zero whatever the content. It tests
the code path — orientation, per-head slicing, GQA group assignment, the projector — and nothing
about the science. In the main sweep it returned **exactly `0.0000` on all 336 pairs**; the largest
magnitude anywhere is `6.12e-08`, which is the eigh route's noise floor.

**S7 is a negative result about the instrument, and is reported as one.** Swapping `A_lin` for its
transpose — a genuinely silent bug, since both give a `D×T` `Z` — moves the residual by 0.005. The
reason is structural: `rowspace(X Aᵀ)` and `rowspace(X A)` are both images of the same `rowspace(X)`,
and the residual is dominated by *which* `D`-dimensional subspace of `R^T` you land in rather than by
its orientation. **This instrument cannot detect an orientation error, and no conclusion below rests
on one.** It is also precisely why `C8_no_mixing` had to be added: without it, one cannot separate
"reachable because the mixing is good" from "reachable because `Z` is built from `X` at all."

---

## 4. Residual × layer × `φ`, with every control

`T/D = 5.3333`, all 28 layers × 12 heads = 336 pairs per arm (60 for the subset-layer controls,
which ran on the pre-stated layers 0, 7, 14, 21, 27). Rank tolerance `1e-6` on singular values.

### 4.1 Pooled

| arm | what it is | median | p25 | p75 | min | max | n |
|---|---|---|---|---|---|---|---|
| `C1_identity` (**the zero anchor**) | **tautology**: `A_lin := A_soft`, so `Z = Z*` | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 336 |
| `C8_no_mixing` (**the no-mixing arm**) | *(Builder-added)* `A := I` — the mixing is the identity matrix | **0.4533** | 0.3682 | 0.5122 | 0.0117 | 0.7353 | 336 |
| `taylor2` | Based, 2nd-order Taylor of `exp` | **0.4604** | 0.3480 | 0.5645 | 0.1185 | 0.8018 | 336 |
| `C5p_wrong_sequence` | **strong null**: `A_lin` from an unrelated sequence (**elu1 kernel**; per-`phi` matched versions in §8.8) | **0.5264** | 0.4648 | 0.6115 | 0.2678 | 0.7920 | 60 |
| `C4_wrong_layer` | **strong null**: `A_lin` from another layer (**elu1 kernel**; matched versions in §8.8) | **0.5271** | 0.4650 | 0.6103 | 0.2673 | 0.7921 | 60 |
| `hedgehog0` | **NOT Hedgehog** — a frozen hedgehog-*shaped* map, see the warning below | **0.5407** | 0.4490 | 0.6237 | 0.1751 | 0.7882 | 336 |
| `elu1` | Katharopoulos `elu(x)+1` | **0.5569** | 0.4592 | 0.6257 | 0.1751 | 0.7916 | 336 |
| `C6_causal_uniform` | **the free floor** | **0.5608** | 0.4669 | 0.6330 | 0.1751 | 0.7923 | 336 |
| `favor` | Performer FAVOR+ (**mis-scaled — see §8**) | **0.8584** | 0.7722 | 0.9212 | 0.2983 | 0.9704 | 336 |
| `C7_random_gauss` | *(Builder-added)* random Gaussian `Z` — **pure-dimension floor** | **0.9007** | 0.8996 | 0.9013 | 0.8976 | 0.9028 | 60 |

> **A naming warning, because two different objects in this document have both been called
> "identity".** `C1_identity` is the **zero anchor**: it sets `A_lin := A_soft`, so `Z = Z*` and the
> residual is 0 by construction. `C8_no_mixing` is an **arm**: it sets `A := I`, the identity
> *matrix*, so the operator passes each token's own value through and mixes nothing; its residual is
> 0.4533. They are unrelated. Below, `C1` is only ever called *the zero anchor* or *the tautology*,
> and `C8` only ever *no-mixing*.

**The scale is anchored at both ends.** `C1 = 0.0000` (everything reachable) and
`C7_random_gauss = 0.9007` against the analytic pure-dimension value
`sqrt(1 − rank/T) = sqrt(1 − 1536/8192) = 0.9014`. A residual is only interpretable between those two
anchors — and **the entire spread of every real arm lives in 0.45 … 0.56, the narrow band just below
the free floor.**

> **`hedgehog0` is not Hedgehog, and no reading of this document should treat it as a test of
> Hedgehog.** Hedgehog's central move is a **learned** per-layer, per-head single-layer MLP
> `phi(x) = exp(W^T x + b)` *distilled on the donor's own attention weights* as soft labels. Our arm
> freezes `W = I`, `b = 0` and fits nothing — the brief registered "one Hedgehog-shaped elementwise
> map used *without* fitting", and that is exactly what was run. **Every published Hedgehog number
> is post-distillation.** So `hedgehog0` landing beside `elu(x)+1` is *consistent with* Hedgehog's
> own thesis (that the fixed functional form is what fails and the fitting is what works), not
> counter-evidence to it. The Researcher records the same point in
> `prior_art/LINEAR_MIXING_MECHANISM.md` §7.4.

### 4.2 Paired deltas — the comparison that carries the result

Paired within (layer, head): same layer, same head, same `T/D`, same normalisation, same slice.
Negative = the first arm is better.

| comparison | mean Δ | sd | median Δ | first arm better in |
|---|---|---|---|---|
| `elu1` − `C6_causal_uniform` | **−0.0059** | 0.0121 | −0.0046 | 261/336 |
| `hedgehog0` − `C6_causal_uniform` | **−0.0154** | 0.0193 | −0.0101 | 313/336 |
| `taylor2` − `C6_causal_uniform` | **−0.0899** | 0.0815 | −0.0867 | 285/336 |
| `favor` − `C6_causal_uniform` | **+0.2796** | 0.0813 | +0.2790 | 0/336 |
| `C8_no_mixing` − `C6_causal_uniform` | **−0.1091** | 0.1339 | −0.0862 | 291/336 |
| `elu1` − `C8_no_mixing` | **+0.1032** | 0.1328 | +0.0844 | 51/336 |
| `hedgehog0` − `C8_no_mixing` | **+0.0936** | 0.1331 | +0.0728 | 56/336 |
| `taylor2` − `C8_no_mixing` | **+0.0192** | 0.1515 | −0.0156 | 191/336 |
| `favor` − `C8_no_mixing` | **+0.3887** | 0.1284 | +0.3730 | 0/336 |
| `elu1` − `C5p_wrong_sequence` | **−0.0008** | 0.0078 | −0.0006 | 38/60 |
| `elu1` − `C4_wrong_layer` | **−0.0006** | 0.0060 | −0.0005 | 38/60 |
| `hedgehog0` − `C5p_wrong_sequence` | **−0.0142** | 0.0135 | −0.0128 | 48/60 |
| `hedgehog0` − `C4_wrong_layer` | **−0.0140** | 0.0135 | −0.0129 | 47/60 |
| `taylor2` − `C5p_wrong_sequence` ⚠ | **−0.0976** | 0.0891 | −0.0922 | 49/60 |
| `taylor2` − `C4_wrong_layer` ⚠ | **−0.0974** | 0.0902 | −0.0910 | 49/60 |
| `C6_causal_uniform` − `C5p_wrong_sequence` | **+0.0003** | 0.0005 | +0.0003 | 16/60 |

> **⚠ The two rows marked above are NOT matched controls.** `C5p_wrong_sequence` and
> `C4_wrong_layer` as run in the main sweep were **both built with the `elu1` kernel**
> (`C5P_C4_PHI = "elu1"`), so comparing `taylor2` against them compares two different feature maps
> at once. **Section 8.8 rebuilds both nulls with each arm's own kernel and supersedes these two
> rows.** The matched numbers are `−0.1623` and `−0.1048` over 336 pairs — i.e. the mismatched null
> was *understating* `taylor2`, not flattering it. Every other row in this table is matched, because
> `C6`, `C8` and `C7` involve no feature map at all.

**Read the `elu1` rows.** Against a mixing computed from a *different sequence's* content — real,
causal, correctly normalised, right shape, wrong content — `elu1` is better by **0.0008** on a
residual of 0.56. Against a mixing computed from a *different layer*, by **0.0006**. **The
position↔content correspondence the whole hypothesis rests on is worth less than one part in six
hundred of the quantity being measured.**

### 4.2b The nulls, compared on a fair footing

`C5′`, `C4` and `C7` ran only on the 5 pre-stated subset layers, so the pooled medians in §4.1 are
**not** comparable to them. Restricted to those same 60 (layer, head) pairs:

| arm (subset layers 0, 7, 14, 21, 27 only) | median |
|---|---|
| `C1_identity` | 0.0000 |
| `C8_no_mixing` | **0.4255** |
| `taylor2` | 0.4332 |
| `hedgehog0` | 0.5058 |
| **`C5p_wrong_sequence`** | **0.5264** |
| **`C6_causal_uniform`** | **0.5264** |
| **`C4_wrong_layer`** | **0.5271** |
| `elu1` | 0.5325 |
| `favor` | 0.7924 |
| `C7_random_gauss` | 0.9007 |

> **On identical layers and heads, the free causal floor, a mixing built from the wrong sequence, and
> a mixing built from the wrong layer agree to within 0.0007 — and `elu1` sits marginally above all
> three.** Paired within (layer, head) `elu1` edges ahead by `0.0008`; unpaired on the same rows it
> is 0.006 behind. **Both readings say the same thing: `elu(x)+1` is not distinguishable from a
> mixing that has been severed from the content it is supposed to encode.**

The brief's warning about shuffled nulls was well placed and the replacements behaved as designed:
`C5′` preserves causality, row normalisation, sparsity profile and scale, and destroys only the
position↔content correspondence. **Neither replacement null beat the real arm**, so this programme's
third shuffle-class control failure did not recur.

### 4.3 Per-layer, median [min–max] over 12 heads

| layer | elu1 | taylor2 | favor | hedgehog0 | C6_causal_uniform | C8_no_mixing |
|---|---|---|---|---|---|---|
| 0 | 0.719 [0.329-0.792] | 0.716 [0.293-0.802] | 0.912 [0.461-0.949] | 0.719 [0.329-0.780] | 0.720 [0.329-0.792] | 0.415 [0.072-0.702] |
| 1 | 0.477 [0.175-0.714] | 0.347 [0.119-0.717] | 0.750 [0.298-0.921] | 0.481 [0.175-0.714] | 0.480 [0.175-0.714] | 0.417 [0.291-0.542] |
| 2 | 0.563 [0.216-0.711] | 0.470 [0.143-0.582] | 0.764 [0.434-0.893] | 0.570 [0.216-0.692] | 0.571 [0.217-0.692] | 0.418 [0.243-0.558] |
| 3 | 0.582 [0.278-0.724] | 0.519 [0.128-0.733] | 0.834 [0.479-0.942] | 0.590 [0.288-0.724] | 0.591 [0.289-0.724] | 0.473 [0.287-0.617] |
| 4 | 0.649 [0.541-0.686] | 0.572 [0.348-0.678] | 0.879 [0.782-0.939] | 0.650 [0.542-0.700] | 0.655 [0.547-0.715] | 0.525 [0.359-0.680] |
| 5 | 0.579 [0.311-0.738] | 0.559 [0.156-0.750] | 0.825 [0.562-0.953] | 0.577 [0.318-0.737] | 0.588 [0.321-0.748] | 0.443 [0.125-0.541] |
| 6 | 0.568 [0.265-0.698] | 0.505 [0.201-0.651] | 0.796 [0.480-0.917] | 0.573 [0.266-0.670] | 0.596 [0.266-0.721] | 0.395 [0.246-0.581] |
| 7 | 0.450 [0.267-0.657] | 0.373 [0.202-0.649] | 0.752 [0.564-0.896] | 0.435 [0.269-0.659] | 0.464 [0.269-0.667] | 0.365 [0.150-0.507] |
| 8 | 0.639 [0.319-0.695] | 0.580 [0.236-0.652] | 0.919 [0.624-0.969] | 0.624 [0.278-0.675] | 0.643 [0.338-0.728] | 0.506 [0.285-0.610] |
| 9 | 0.586 [0.411-0.671] | 0.528 [0.278-0.640] | 0.914 [0.775-0.964] | 0.592 [0.418-0.679] | 0.592 [0.422-0.681] | 0.476 [0.279-0.619] |
| 10 | 0.517 [0.359-0.706] | 0.457 [0.326-0.736] | 0.879 [0.788-0.959] | 0.523 [0.362-0.708] | 0.530 [0.364-0.709] | 0.461 [0.072-0.584] |
| 11 | 0.417 [0.268-0.609] | 0.366 [0.209-0.658] | 0.769 [0.590-0.936] | 0.420 [0.275-0.612] | 0.422 [0.280-0.612] | 0.358 [0.057-0.532] |
| 12 | 0.552 [0.397-0.642] | 0.452 [0.163-0.587] | 0.904 [0.685-0.965] | 0.536 [0.250-0.606] | 0.562 [0.387-0.672] | 0.455 [0.272-0.528] |
| 13 | 0.520 [0.341-0.649] | 0.391 [0.333-0.731] | 0.926 [0.783-0.949] | 0.513 [0.343-0.648] | 0.528 [0.345-0.650] | 0.504 [0.353-0.544] |
| 14 | 0.578 [0.437-0.715] | 0.505 [0.349-0.663] | 0.878 [0.549-0.955] | 0.565 [0.411-0.672] | 0.579 [0.431-0.700] | 0.480 [0.195-0.540] |
| 15 | 0.413 [0.257-0.675] | 0.396 [0.238-0.728] | 0.816 [0.524-0.952] | 0.402 [0.258-0.687] | 0.421 [0.260-0.680] | 0.406 [0.012-0.487] |
| 16 | 0.460 [0.187-0.561] | 0.372 [0.157-0.518] | 0.843 [0.515-0.892] | 0.436 [0.178-0.508] | 0.466 [0.189-0.568] | 0.400 [0.247-0.491] |
| 17 | 0.482 [0.216-0.597] | 0.449 [0.144-0.580] | 0.904 [0.529-0.947] | 0.470 [0.195-0.581] | 0.487 [0.218-0.619] | 0.430 [0.236-0.563] |
| 18 | 0.516 [0.346-0.628] | 0.447 [0.200-0.605] | 0.884 [0.660-0.958] | 0.516 [0.333-0.645] | 0.520 [0.354-0.644] | 0.431 [0.057-0.512] |
| 19 | 0.631 [0.285-0.783] | 0.523 [0.288-0.683] | 0.921 [0.751-0.970] | 0.625 [0.288-0.752] | 0.626 [0.288-0.776] | 0.528 [0.066-0.635] |
| 20 | 0.562 [0.449-0.665] | 0.510 [0.334-0.605] | 0.895 [0.819-0.957] | 0.551 [0.436-0.647] | 0.564 [0.452-0.664] | 0.511 [0.175-0.579] |
| 21 | 0.487 [0.308-0.606] | 0.398 [0.213-0.525] | 0.811 [0.579-0.938] | 0.473 [0.308-0.581] | 0.489 [0.313-0.606] | 0.416 [0.271-0.532] |
| 22 | 0.628 [0.493-0.768] | 0.571 [0.346-0.730] | 0.916 [0.797-0.968] | 0.629 [0.467-0.764] | 0.635 [0.488-0.763] | 0.520 [0.300-0.661] |
| 23 | 0.640 [0.560-0.791] | 0.529 [0.400-0.670] | 0.885 [0.847-0.943] | 0.632 [0.540-0.761] | 0.650 [0.566-0.789] | 0.509 [0.397-0.608] |
| 24 | 0.562 [0.419-0.720] | 0.434 [0.312-0.693] | 0.818 [0.723-0.970] | 0.554 [0.407-0.718] | 0.566 [0.424-0.725] | 0.444 [0.360-0.646] |
| 25 | 0.592 [0.429-0.785] | 0.418 [0.174-0.722] | 0.793 [0.638-0.961] | 0.550 [0.360-0.788] | 0.586 [0.427-0.792] | 0.431 [0.304-0.735] |
| 26 | 0.596 [0.546-0.621] | 0.399 [0.323-0.552] | 0.816 [0.784-0.863] | 0.549 [0.535-0.598] | 0.587 [0.546-0.620] | 0.468 [0.410-0.564] |
| 27 | 0.532 [0.476-0.752] | 0.287 [0.246-0.549] | 0.675 [0.629-0.926] | 0.510 [0.473-0.747] | 0.529 [0.481-0.767] | 0.424 [0.336-0.462] |


---

## 5. Entropy stratification — the stratum where a false positive would hide

**Pre-registered strata** (§4 of the brief). Rows are stratified by the **normalised entropy of the
`A_soft` row**, `H_t / log(t+1)`, computed per (layer, head):

- `short_prefix` — positions `t < 32` **within a sequence**, reported separately and never pooled: a
  3-token prefix is trivially peaked and would contaminate the retrieval stratum.
- `peaked_d1` — the lowest decile of normalised entropy among `t >= 32`. **The retrieval-like rows.**
- `mid` — deciles 2-9.
- `diffuse_d10` — the highest decile.

Sizes per head at `T = 8192`: short-prefix 256, peaked 794, mid 6348, diffuse 794.
Median normalised-entropy quantiles: `q10 = 0.2676`, `q50 = 0.4418`, `q90 = 0.6077`.

The residual is computed **per column of `M`** (`R = M - K Z`, then column norms pooled inside each
stratum), so the stratification is exact and costs nothing extra.

| arm | peaked D1 | mid (D2-D9) | diffuse D10 | short-prefix |
|---|---|---|---|---|
| `C1_identity` | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| `elu1` | **0.6486** | 0.5462 | 0.5114 | 0.3017 |
| `taylor2` | **0.5304** | 0.4623 | 0.4016 | 0.3797 |
| `favor` | **0.9389** | 0.8501 | 0.7938 | 0.9455 |
| `hedgehog0` | **0.6395** | 0.5371 | 0.4913 | 0.3009 |
| `C6_causal_uniform` | **0.6630** | 0.5524 | 0.5145 | 0.2985 |
| `C8_no_mixing` | **0.4890** | 0.4487 | 0.4207 | 0.5237 |
| `C5p_wrong_sequence` | **0.6303** | 0.5222 | 0.4960 | 0.2714 |
| `C4_wrong_layer` | **0.6289** | 0.5247 | 0.4959 | 0.2692 |
| `C7_random_gauss` | **0.8781** | 0.9019 | 0.9215 | 0.9153 |

**Every content-bearing arm is worst exactly on the retrieval-like rows**, by `+0.11` to `+0.14`, in
about 300 of 336 (layer, head) pairs. This is the architectural recall cap the Researcher predicted,
visible in pass 1 for no extra compute:

| arm | peaked - diffuse (paired mean) | peaked worse in |
|---|---|---|
| `hedgehog0` | **+0.1374** | 303/336 |
| `favor` | **+0.1369** | 285/336 |
| `C6_causal_uniform` | **+0.1366** | 300/336 |
| `elu1` | **+0.1334** | 300/336 |
| `taylor2` | **+0.1079** | 293/336 |
| `C8_no_mixing` | **+0.0509** | 234/336 |
| `C5p_wrong_sequence` / `C4_wrong_layer` | +0.0892 / +0.0890 | 45/60 |
| **`C7_random_gauss`** | **-0.0572** | **17/60** |

> **`C7_random_gauss` reverses the sign, and that is the control that makes the stratification
> trustworthy.** A `Z` with no relationship to `X` shows *no* peaked-row penalty - it is slightly
> *better* on peaked rows. So the penalty is not an artefact of how the strata were cut, nor of
> peaked rows having smaller norms; it is a property of the relationship between the linearised
> mixing and the target. **A stratification that fires on the real arms and not on the random one is
> a measurement, not a slicing artefact.**

**And the stratification changes the reading of `taylor2`, the only arm above the floor.**
Paired, `n = 336`:

| `taylor2` vs | peaked D1 | mid | diffuse D10 |
|---|---|---|---|
| `C6_causal_uniform` | **-0.1323** (better in 318/336) | -0.0839 (279/336) | -0.1036 (290/336) |
| `C8_no_mixing` | **+0.0433** (better in only **149/336**) | +0.0237 (187/336) | -0.0136 (230/336) |

> **`taylor2` beats the free causal floor everywhere - including, most strongly, on the retrieval
> rows. But on exactly those retrieval rows it LOSES to doing no mixing at all**, on 187 of 336
> pairs, by `+0.043`. The one feature map that clears the floor does so by reshaping *diffuse*
> mixing; on the rows where a fixed-state operator is already known to be capped, it is worse than
> the null operator. **Reported as its own stratum, never averaged into the bulk.**

---

## 6. The `T/D` curve - and it has NOT flattened

`r2a_curve.json`. Layers 0, 14, 27 x heads 0, 5, 11 x 8 arms. The curve is **nested** - verified at
run time, not assumed:

```
slice nesting verified: slice(n) == slice(nmax)[:n] for all n
```

so every point is a prefix of the same draw and the curve is not confounded by resampling.

**`n_seq = 1` (`T = 1024`, `T/D = 0.667`) was REFUSED**, not plotted.

### Median residual by ACHIEVED `T/D`

| arm | 1.333 | 2.667 | 5.333 | 10.667 | **21.333** |
|---|---|---|---|---|---|
| `taylor2` | 0.1615 | 0.2797 | 0.3486 | 0.3987 | **0.5104** |
| `C8_no_mixing` | 0.1893 | 0.3203 | 0.4361 | 0.5395 | **0.5961** |
| `hedgehog0` | 0.2173 | 0.3894 | 0.4916 | 0.5683 | **0.6210** |
| `elu1` | 0.2174 | 0.4186 | 0.5087 | 0.5815 | **0.6283** |
| `C6_causal_uniform` | 0.1961 | 0.4048 | 0.5062 | 0.5855 | **0.6288** |
| `favor` | 0.6865 | 0.6626 | 0.6783 | 0.6860 | 0.6903 |
| `C7_random_gauss` | 0.5056 | 0.7938 | 0.9007 | 0.9523 | 0.9762 |
| **analytic** `sqrt(1 - 1536/T)` | 0.5000 | 0.7906 | 0.9014 | 0.9520 | 0.9763 |
| `C1_identity` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

`C7_random_gauss` tracks the analytic pure-dimension curve to within 0.006 at every point, which
calibrates the whole `T/D` axis independently of the donor.

> **The curve has not flattened.** From `T/D = 10.667` to `21.333` the residual still rises by
> `+0.047` (`elu1`), `+0.111` (`taylor2`), `+0.057` (`C8`). **Quoting the `T/D = 5.333` value as
> "the" residual would be exactly the error D1 was struck two points for**, so it is not quoted
> alone: the honest statement is that **the residual at the `T/D >= 24` that `BRIEF_R2` requires is
> at least the `T/D = 21.333` value, and on the trend, higher.**

### The ordering is stable across `T/D` - paired mean delta vs `C6`

| arm | 1.333 | 2.667 | 5.333 | 10.667 | 21.333 |
|---|---|---|---|---|---|
| `elu1` | +0.0118 | +0.0083 | +0.0029 | +0.0004 | **-0.0002** |
| `hedgehog0` | +0.0152 | -0.0063 | -0.0134 | -0.0109 | -0.0099 |
| `taylor2` | -0.0272 | -0.0936 | -0.1059 | -0.0979 | **-0.0943** |
| `favor` | +0.4475 | +0.2759 | +0.1836 | +0.1262 | +0.0868 |
| `C8_no_mixing` | -0.0321 | -0.0634 | -0.0740 | -0.0636 | **-0.0651** |

Two things survive the whole sweep, and both matter:

1. **`taylor2` holds a `-0.09` to `-0.11` separation from the free floor at every `T/D >= 2.667`.**
   Not a single point.
2. **`elu1` converges *onto* the free floor as `T/D` grows** - `+0.0118 -> -0.0002`. At the largest
   calibration size measured, `elu(x)+1` and a uniform causal average are the same thing to four
   decimal places. **More calibration data does not rescue it; it removes the last of its
   advantage.**

---

## 7. The one arm above the floor: how far above, measured four ways

`taylor2` (Based's 2nd-order Taylor of `exp`) is the only registered `phi` separated from the null
cluster. Four measurements, no conclusion attached to any single one.

### 7.1 Is the separation larger than the head-to-head spread? **No.**

| layer | median `taylor2` | sd over 12 heads | median `C6` | sd | gap | **gap / pooled sd** | `taylor2` better in |
|---|---|---|---|---|---|---|---|
| 0 | 0.7157 | 0.1835 | 0.7200 | 0.1720 | -0.0043 | **-0.02** | 5/12 |
| 1 | 0.3473 | 0.1938 | 0.4801 | 0.1902 | -0.1328 | **-0.69** | 9/12 |
| 2 | 0.4704 | 0.1185 | 0.5711 | 0.1356 | -0.1007 | **-0.79** | 12/12 |
| 3 | 0.5188 | 0.2205 | 0.5905 | 0.1540 | -0.0717 | **-0.38** | 11/12 |
| 4 | 0.5718 | 0.1177 | 0.6555 | 0.0526 | -0.0837 | **-0.92** | 11/12 |
| 5 | 0.5591 | 0.2014 | 0.5881 | 0.1457 | -0.0290 | **-0.17** | 11/12 |
| 6 | 0.5046 | 0.1651 | 0.5960 | 0.1569 | -0.0914 | **-0.57** | 10/12 |
| 7 | 0.3726 | 0.1570 | 0.4636 | 0.1182 | -0.0909 | **-0.65** | 11/12 |
| 8 | 0.5805 | 0.1314 | 0.6426 | 0.1125 | -0.0621 | **-0.51** | 12/12 |
| 9 | 0.5282 | 0.1140 | 0.5919 | 0.0777 | -0.0637 | **-0.65** | 11/12 |
| 10 | 0.4574 | 0.1146 | 0.5296 | 0.0913 | -0.0722 | **-0.70** | 10/12 |
| 11 | 0.3659 | 0.1511 | 0.4225 | 0.1101 | -0.0566 | **-0.43** | 9/12 |
| 12 | 0.4517 | 0.1226 | 0.5621 | 0.0796 | -0.1104 | **-1.07** | 12/12 |
| 13 | 0.3909 | 0.1258 | 0.5277 | 0.0687 | -0.1368 | **-1.35** | 7/12 |
| 14 | 0.5053 | 0.0932 | 0.5795 | 0.0825 | -0.0741 | **-0.84** | 9/12 |
| 15 | 0.3958 | 0.1530 | 0.4212 | 0.1295 | -0.0255 | **-0.18** | 8/12 |
| 16 | 0.3717 | 0.0942 | 0.4656 | 0.0936 | -0.0939 | **-1.00** | 11/12 |
| 17 | 0.4487 | 0.1430 | 0.4868 | 0.1110 | -0.0381 | **-0.30** | 9/12 |
| 18 | 0.4466 | 0.1167 | 0.5200 | 0.0891 | -0.0734 | **-0.71** | 8/12 |
| 19 | 0.5226 | 0.0964 | 0.6262 | 0.1370 | -0.1035 | **-0.87** | 10/12 |
| 20 | 0.5099 | 0.0925 | 0.5638 | 0.0758 | -0.0539 | **-0.64** | 9/12 |
| 21 | 0.3983 | 0.1104 | 0.4888 | 0.0952 | -0.0905 | **-0.88** | 12/12 |
| 22 | 0.5711 | 0.1062 | 0.6351 | 0.0826 | -0.0640 | **-0.67** | 9/12 |
| 23 | 0.5287 | 0.0767 | 0.6501 | 0.0782 | -0.1214 | **-1.57** | 11/12 |
| 24 | 0.4340 | 0.1050 | 0.5659 | 0.0929 | -0.1319 | **-1.33** | 12/12 |
| 25 | 0.4185 | 0.1836 | 0.5862 | 0.1154 | -0.1677 | **-1.09** | 12/12 |
| 26 | 0.3995 | 0.0716 | 0.5868 | 0.0204 | -0.1873 | **-3.56** | 12/12 |
| 27 | 0.2874 | 0.0856 | 0.5286 | 0.0766 | -0.2413 | **-2.97** | 12/12 |

**Median `gap / pooled-sd` = -0.70; it exceeds 1 in magnitude in only 6 of 28 layers.** The
separation of the medians is *smaller than the dispersion across the 12 heads within either arm*.
Anyone quoting "`taylor2` 0.460 vs `C6` 0.561" as a clean separation is quoting a median gap without
the spread beside it.

**The paired framing is the defensible one** - same layer, same head, same `T/D`, same slice - and
paired the effect is consistent: `-0.0899 +/- 0.0815`, better in 285/336, and the per-layer mean is
negative in **28 of 28 layers**. So the direction is solid; the magnitude is small relative to how
much heads differ from one another.

### 7.2 Does it hold in the peaked / retrieval stratum? **Against `C6` yes, against `C8` no.**

Covered in §5: `-0.1323` vs `C6` (better in 318/336), but **`+0.0433` vs `C8_no_mixing`, better in
only 149/336**.

### 7.3 Does it hold across all 28 layers? **Yes, and it grows with depth.**

Per-layer paired mean `taylor2 - C6` is negative in **28/28**. It is near zero at the input
(`layer 0: -0.0095`, better in only 5/12) and largest at the output
(`layer 25: -0.1666`, `26: -0.1718`, `27: -0.2335`, all 12/12).

### 7.4 Where does it sit relative to `C8_no_mixing`? **A coin flip.**

`taylor2 - C8_no_mixing` over all 336 pairs: **mean `+0.0192`, sd `0.1515`, median `-0.0156`,
`taylor2` better in 191/336.** Per layer:

- `taylor2` beats no-mixing in **11 layers**: [1, 12, 13, 16, 17, 20, 21, 24, 25, 26, 27]
- no-mixing beats `taylor2` in **17 layers**: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 18, 19, 22, 23]

The single largest gap is at layer 0, where `taylor2` is **+0.250 worse** than not mixing at all.

> **Stated flatly: at the pinned `T/D`, replacing the donor's softmax mixing with the best of the
> four registered feature maps buys nothing, on average, over replacing it with the identity.**
> Content-based linear mixing clears the *uniform* causal floor; it does not clear the *no-mixing*
> floor. What `taylor2` earns over `C6` it appears to earn by being *less* of a smoother, not by
> being a better model of the donor's attention.

---

## 8. Amendment 1 - the decay arm, and the mechanism behind three null results

**Stage `decay`**, 2520 rows, 4965 s, **`T = 8192`, `D = 1536`, `T/D = 5.3333` - identical to the
main sweep**, same slice, same layers, same heads, same row-normalisation. The comparison is
therefore licensed; nothing below compares across stages at different settings.

`A[t,s]` proportional to `gamma^(t-s) * k(q_t, k_s)`, causal, row-normalised, `k` the same `elu1`
kernel as the `elu1` arm. For every `gamma` there is a **`decayonly`** twin with `k = 1` - the
envelope alone, no content whatsoever. Sliding-window variants `win_wW` / `winonly_wW` were cheap
given the machinery, so they ran too.

### 8.0 ACHIEVED envelope, read back off the constructed `A`

| requested | achieved | how it was read |
|---|---|---|
| `gamma = 0.5, 0.9, 0.95, 0.99, 0.999` | **0.5, 0.9, 0.95, 0.99, 0.999** (exact) | `median_t A[t,t-2]/A[t,t-1]` on the constructed matrix |
| `W = 32, 128, 512` | **32, 128, 512**; `n_nonzero_in_row` = 32, 128, 512 | counted off row 900 of the constructed matrix |

### 8.1 Item 1 - is `decay ~= decayonly` a genuine null, or an artefact of how the two combine?

**It is genuine, and the mechanism is now measured: the content kernel barely changes the matrix at
all.** A diagnostic that never touches the projector - the relative Frobenius distance between the
two mixing matrices themselves, and their median row-wise cosine:

| pair of mixings | layer 0 | layer 14 | layer 27 |
|---|---|---|---|
| `decay_g0.5` vs `decayonly_g0.5` | `0.0036` / cos `1.0000` | `0.0422` / cos `0.9995` | `0.0381` / cos `0.9997` |
| `decay_g0.9` vs `decayonly_g0.9` | `0.0053` / cos `1.0000` | `0.0655` / cos `0.9982` | `0.0559` / cos `0.9986` |
| `decay_g0.99` vs `decayonly_g0.99` | `0.0049` / cos `1.0000` | `0.0720` / cos `0.9972` | `0.0534` / cos `0.9981` |
| `win_w128` vs `winonly_w128` | `0.0051` / cos `1.0000` | `0.0727` / cos `0.9971` | `0.0549` / cos `0.9981` |
| **`elu1` vs `C6_causal_uniform`** | **`0.0043` / cos `1.0000`** | **`0.0711` / cos `0.9969`** | **`0.0487` / cos `0.9980`** |
| `taylor2` vs `C6_causal_uniform` | `0.0451` / cos `0.9995` | **`0.7445` / cos `0.8277`** | **`0.6757` / cos `0.7429`** |
| `elu1` vs `C8_no_mixing` | `0.9963` / cos `0.0443` | `0.9961` / cos `0.0462` | `0.9962` / cos `0.0457` |
| `A_soft` vs `C6_causal_uniform` | `4.2724` / cos `0.1279` | `8.4252` / cos `0.0695` | `2.0934` / cos `0.3234` |
| **`A_soft` vs `elu1`** | **`4.2722` / cos `0.1279`** | **`8.4122` / cos `0.0672`** | **`2.0815` / cos `0.3271`** |

Read the last two rows together:

> **`elu(x)+1` applied to the donor's own `W_q`/`W_k` produces a mixing matrix that is 0.4-7 % away
> from a uniform causal average, with a row cosine of 0.997-1.000 - and it is exactly as far from
> the donor's real attention as the uniform average is** (`8.4122` vs `8.4252`; cosine `0.0672` vs
> `0.0695`). The two distances agree to the third decimal.

**So this is not "different mixings whose row spaces happen to coincide". The mixings are the same
mixing.** `phi(q_t) . phi(k_s)` for these maps varies by only a few percent across `s`, because
`elu(x)+1` produces strictly positive features with a large common component; row-normalisation then
divides that common component out and what remains is a uniform average plus noise. The same holds
for the decay and window arms: the envelope sets the matrix and the content kernel modulates it by
at most 7 %.

**One mechanism accounts for all three null results in this probe** - `elu1 ~= C6`,
`elu1 ~= C5' / C4`, and `decay ~= decayonly`. It also predicts the exception correctly: **`taylor2`
is the one map whose matrix genuinely differs from uniform** (`rel_fro` 0.745 and 0.676 at layers 14
and 27, cosine 0.83 and 0.74), and it is the one map whose residual separates from the floor - and
it separates most at exactly the layers where its matrix differs most. **The matrix-space diagnostic
and the subspace-space residual agree, having been computed by disjoint code paths.**

*Caveat, stated: this diagnostic is one sequence, three layers, three heads. It is a mechanism
check, not a sweep.*

### 8.2 Item 2 - the `gamma` curve, with both limits, on identical (layer, head) pairs

`C8_no_mixing` (`A := I`) is the **`gamma -> 0`** limit of this family and `C6_causal_uniform`
(`A[t,s] = 1/(t+1)`) is the **`gamma -> 1`** limit, so both endpoints were already measured. Every
row below is the **same 60 (layer, head) pairs** - layers 0, 7, 14, 21, 27 - with dispersion beside
every median:

| gamma | decay (content x envelope) | sd | decayonly (envelope alone) | sd | content contributes |
|---|---|---|---|---|---|
| **0 (= `C8_no_mixing`, A := I)** | 0.4255 | 0.1200 | *(same object)* | | - |
| **0.5** | 0.4046 | 0.0747 | 0.4048 | 0.0746 | **-0.0003 +/- 0.0010** |
| **0.9** | 0.4530 | 0.1124 | 0.4533 | 0.1128 | **-0.0006 +/- 0.0015** |
| **0.95** | 0.4610 | 0.1186 | 0.4613 | 0.1191 | **-0.0005 +/- 0.0017** |
| **0.99** | 0.4831 | 0.1248 | 0.4802 | 0.1250 | **-0.0000 +/- 0.0040** |
| **0.999** | 0.5193 | 0.1274 | 0.5151 | 0.1267 | **-0.0004 +/- 0.0077** |
| **1 (= `C6_causal_uniform`)** | 0.5264 | 0.1263 | *(same object)* | | - |

> **The minimum is INTERIOR, not at a boundary.** `gamma = 0.5` (0.4046) beats both the
> `gamma -> 0` limit (`C8`, 0.4255) and the `gamma -> 1` limit (`C6`, 0.5264). **The ladder as first
> swept was quoted at its endpoint, and `gamma = 0.5` being "best in the tested set" was not
> evidence that it is best.** A follow-up stage filling `gamma = 0.1, 0.25, 0.35, 0.7` on **all 28
> layers** is in section 8.6.

**The content column is null at every single `gamma`** - `-0.0003` to `-0.0006`, sd at most
`0.0077`, across five decades of effective window length. Whatever the envelope is doing, the
content kernel adds nothing on top of it anywhere on the curve.

### 8.3 The full decay / window results, with dispersion

All-28-layer arms (n = 336 each, identical layer/head set):

| arm | n | median | **sd** | p25 | p75 |
|---|---|---|---|---|---|
| `C8_no_mixing` | 336 | **0.4533** | 0.1164 | 0.3682 | 0.5122 |
| `decay_g0.99` | 336 | **0.5056** | 0.1194 | 0.4124 | 0.5704 |
| `decayonly_g0.99` | 336 | **0.5031** | 0.1195 | 0.4155 | 0.5729 |
| `taylor2` | 336 | **0.4604** | 0.1469 | 0.3480 | 0.5645 |
| `C6_causal_uniform` | 336 | **0.5608** | 0.1249 | 0.4669 | 0.6330 |
| `elu1` | 336 | **0.5569** | 0.1259 | 0.4592 | 0.6257 |
| `win_w128` | 336 | **0.5588** | 0.1216 | 0.4654 | 0.6280 |
| `winonly_w128` | 336 | **0.5617** | 0.1217 | 0.4701 | 0.6325 |
| `favor_d14` | 336 | **0.6556** | 0.1396 | 0.5510 | 0.7419 |
| `favor` | 336 | **0.8584** | 0.1223 | 0.7722 | 0.9212 |

Subset-layer arms, **every row restricted to the same 60 pairs** (layers 0, 7, 14, 21, 27):

| arm | n | median | **sd** | p25 | p75 |
|---|---|---|---|---|---|
| `decay_g0.5` | 60 | **0.4046** | 0.0747 | 0.3493 | 0.4488 |
| `decayonly_g0.5` | 60 | **0.4048** | 0.0746 | 0.3493 | 0.4494 |
| `C8_no_mixing` | 60 | **0.4255** | 0.1200 | 0.3389 | 0.4855 |
| `decay_g0.9` | 60 | **0.4530** | 0.1124 | 0.3792 | 0.5101 |
| `decayonly_g0.9` | 60 | **0.4533** | 0.1128 | 0.3805 | 0.5090 |
| `decay_g0.95` | 60 | **0.4610** | 0.1186 | 0.3924 | 0.5185 |
| `decayonly_g0.95` | 60 | **0.4613** | 0.1191 | 0.3941 | 0.5183 |
| `taylor2` | 60 | **0.4332** | 0.1686 | 0.2996 | 0.5474 |
| `elu1_noscale` | 60 | **0.4905** | 0.1363 | 0.4378 | 0.5897 |
| `decay_g0.99` | 60 | **0.4831** | 0.1248 | 0.4196 | 0.5475 |
| `decayonly_g0.99` | 60 | **0.4802** | 0.1250 | 0.4170 | 0.5442 |
| `hedgehog0_noscale` | 60 | **0.5041** | 0.1365 | 0.4297 | 0.6238 |
| `hedgehog0` | 60 | **0.5058** | 0.1273 | 0.4553 | 0.5883 |
| `win_w32` | 60 | **0.5134** | 0.1259 | 0.4616 | 0.5960 |
| `winonly_w32` | 60 | **0.5136** | 0.1262 | 0.4630 | 0.5963 |
| `decay_g0.999` | 60 | **0.5193** | 0.1274 | 0.4530 | 0.6051 |
| `decayonly_g0.999` | 60 | **0.5151** | 0.1267 | 0.4556 | 0.5969 |
| `win_w512` | 60 | **0.5276** | 0.1274 | 0.4624 | 0.6147 |
| `winonly_w512` | 60 | **0.5223** | 0.1266 | 0.4650 | 0.6055 |
| `elu1` | 60 | **0.5325** | 0.1273 | 0.4624 | 0.6215 |
| `C6_causal_uniform` | 60 | **0.5264** | 0.1263 | 0.4653 | 0.6114 |
| `C5p_wrong_sequence` | 60 | **0.5264** | 0.1263 | 0.4648 | 0.6115 |

Two things to note. **`favor_d14`, the corrected FAVOR+ (section 11.1), fixes the rank collapse** -
`rank(Z)` rises from 8 to 1533 - **but it is still `+0.0931` worse than the free causal floor, on
293 of 336 pairs.** With the mis-scaling removed it is a fair test, and it fails. And the sliding
window is not the decay envelope: `win_w128` sits at `-0.0023 +/- 0.0087` from `C6`, i.e. on the
floor.

### 8.4 Item 5 - entropy stratification of the decay arms

Same 60 pairs throughout; `peaked - diffuse` is paired within (layer, head), with its sd:

| arm | peaked D1 | mid | diffuse D10 | **peaked - diffuse** (paired) |
|---|---|---|---|---|
| `C8_no_mixing` | **0.4180** | 0.4223 | 0.4237 | **-0.0050 +/- 0.1728** (34/60) |
| `decay_g0.5` | **0.4106** | 0.3951 | 0.4158 | **-0.0183 +/- 0.1797** (32/60) |
| `decay_g0.9` | **0.4962** | 0.4457 | 0.4381 | **+0.0115 +/- 0.1719** (38/60) |
| `decay_g0.95` | **0.5120** | 0.4556 | 0.4425 | **+0.0218 +/- 0.1661** (39/60) |
| `decay_g0.99` | **0.5481** | 0.4790 | 0.4612 | **+0.0412 +/- 0.1645** (41/60) |
| `decay_g0.999` | **0.6174** | 0.5204 | 0.4904 | **+0.0736 +/- 0.1771** (43/60) |
| `win_w32` | **0.5976** | 0.5175 | 0.5047 | **+0.0476 +/- 0.1705** (41/60) |
| `win_w128` | **0.6042** | 0.5263 | 0.5003 | **+0.0613 +/- 0.1698** (43/60) |
| `win_w512` | **0.6228** | 0.5280 | 0.4964 | **+0.0762 +/- 0.1761** (44/60) |
| `taylor2` | **0.4709** | 0.4339 | 0.3685 | **+0.0678 +/- 0.1300** (48/60) |
| `elu1` | **0.6299** | 0.5261 | 0.4950 | **+0.0837 +/- 0.1806** (45/60) |
| `C6_causal_uniform` | **0.6307** | 0.5226 | 0.4962 | **+0.0893 +/- 0.1798** (45/60) |
| `favor_d14` | **0.7154** | 0.6283 | 0.5489 | **+0.1402 +/- 0.1954** (51/60) |
| `elu1_noscale` | **0.5884** | 0.4832 | 0.4573 | **+0.0780 +/- 0.1589** (46/60) |

> **The retrieval penalty grows with how much mixing the operator does.** On these 60 pairs, as
> `gamma` runs `0 -> 0.5 -> 0.9 -> 0.95 -> 0.99 -> 0.999 -> 1` the peaked-minus-diffuse gap runs
> `-0.005 -> -0.018 -> +0.012 -> +0.022 -> +0.041 -> +0.074 -> +0.089`.
>
> **CORRECTION, from the later all-28-layer sweep (section 8.6).** On this 60-pair subset
> `decay_g0.5` showed a *negative* gap (`-0.018`) and I wrote that it was the only arm in the probe
> that did not damage the retrieval rows. **That was a subset artefact.** On the full 336 pairs
> `decay_g0.5`'s gap is **`+0.0531 +/- 0.1264`, positive in 248/336** - the same sign as every other
> content-bearing arm. The monotone *trend* in `gamma` survives the full sweep; the negative sign at
> `gamma = 0.5` does not. **The 60-pair number was quoted at a resolution it could not support, and
> the sd of 0.18 sitting beside it was the warning.** Section 8.6 supersedes this row.
>
> **But note the dispersion.** On 60 pairs the sd of that gap is `0.16-0.18`, so no individual
> `gamma`-to-`gamma` step is resolved on its own. What is resolved is the **monotone trend across
> seven settings**, plus the 336-pair versions of the two endpoints (`C6 +0.1366` in 300/336,
> `C8 +0.0509` in 234/336), which carry far more weight than any single subset row.

### 8.5 Item 4 - the `_noscale` arms: removing the `d^{-1/2}` on `q`

`elu1` and `hedgehog0` have no canonical scaling (section 11.2), so this was a free choice I made,
and it is worth what it moves:

| layer | `elu1` | `elu1_noscale` | delta | `hedgehog0` | `hedgehog0_noscale` | delta |
|---|---|---|---|---|---|---|
| 0 | 0.7193 | 0.7168 | **-0.0040** | 0.7193 | 0.7201 | **+0.0000** |
| 7 | 0.4502 | 0.4140 | **-0.0293** | 0.4354 | 0.4392 | **-0.0082** |
| 14 | 0.5782 | 0.5588 | **-0.0252** | 0.5648 | 0.5836 | **+0.0334** |
| 21 | 0.4874 | 0.4614 | **-0.0402** | 0.4734 | 0.4832 | **-0.0055** |
| 27 | 0.5325 | 0.4765 | **-0.0529** | 0.5101 | 0.4705 | **-0.0448** |

**Removing the scaling helps `elu1` at every layer, and by more at depth**: `-0.0040` at layer 0
rising monotonically to `-0.0529` at layer 27; paired over 60, `-0.0303 +/- 0.0225`, better in
59/60. Against the free floor, `elu1_noscale - C6 = -0.0314 +/- 0.0236` (59/60) - **five times
`elu1`'s own `-0.006`, but still a third of `taylor2`'s `-0.09`, and nowhere near `C8_no_mixing`.**
`hedgehog0_noscale` moves nothing: `-0.0050 +/- 0.0330`, 32/60, a coin flip.

The direction is consistent with the mechanism in 8.1: dividing `q` by `sqrt(128)` shrinks the
pre-activation range, which pushes `elu(x)+1` further into its near-constant regime and *increases*
the common component of the kernel. Removing it restores a little contrast - and buys a little
residual.

### 8.6 Item 2 closed - the `gamma` minimum, located, on all 28 layers

A follow-up stage (`gamma`, **1920 rows, 2178 s**, same `T/D`, same slice, same layers and heads) filled
`gamma = 0.1, 0.25, 0.35, 0.5, 0.7` on **all 28 layers**, with `C8_no_mixing` and
`C6_causal_uniform` supplying the two limits. Median over 12 heads:

| layer | 0 (`C8`) | 0.1 | 0.25 | 0.35 | 0.5 | 0.7 | 0.99 | 1 (`C6`) | **argmin** |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4148 | 0.3951 | 0.3724 | 0.3736 | 0.3979 | 0.4977 | 0.6777 | 0.7200 | **0.25** |
| 1 | 0.4165 | 0.3992 | 0.3762 | 0.3632 | 0.3602 | 0.3478 | 0.4040 | 0.4801 | **0.7** |
| 2 | 0.4181 | 0.4085 | 0.3985 | 0.3978 | 0.4033 | 0.4297 | 0.5108 | 0.5711 | **0.35** |
| 3 | 0.4726 | 0.4691 | 0.4682 | 0.4704 | 0.4782 | 0.4852 | 0.5323 | 0.5905 | **0.25** |
| 4 | 0.5245 | 0.5153 | 0.5035 | 0.4972 | 0.4902 | 0.5118 | 0.5876 | 0.6555 | **0.5** |
| 5 | 0.4426 | 0.4363 | 0.4274 | 0.4176 | 0.4056 | 0.4615 | 0.5276 | 0.5881 | **0.5** |
| 6 | 0.3952 | 0.3845 | 0.3751 | 0.3704 | 0.3661 | 0.4106 | 0.5167 | 0.5960 | **0.5** |
| 7 | 0.3655 | 0.3558 | 0.3431 | 0.3336 | 0.3305 | 0.3490 | 0.3960 | 0.4636 | **0.5** |
| 8 | 0.5059 | 0.4987 | 0.4964 | 0.4994 | 0.5000 | 0.4968 | 0.5787 | 0.6426 | **0.25** |
| 9 | 0.4759 | 0.4633 | 0.4507 | 0.4461 | 0.4457 | 0.4688 | 0.5335 | 0.5919 | **0.5** |
| 10 | 0.4610 | 0.4505 | 0.4386 | 0.4331 | 0.4203 | 0.4165 | 0.4827 | 0.5296 | **0.7** |
| 11 | 0.3578 | 0.3487 | 0.3383 | 0.3337 | 0.3306 | 0.3484 | 0.3710 | 0.4225 | **0.5** |
| 12 | 0.4551 | 0.4515 | 0.4423 | 0.4355 | 0.4306 | 0.4389 | 0.4932 | 0.5621 | **0.5** |
| 13 | 0.5045 | 0.4884 | 0.4639 | 0.4480 | 0.4256 | 0.4033 | 0.4575 | 0.5277 | **0.7** |
| 14 | 0.4804 | 0.4685 | 0.4631 | 0.4590 | 0.4469 | 0.4384 | 0.5111 | 0.5795 | **0.7** |
| 15 | 0.4063 | 0.3914 | 0.3685 | 0.3531 | 0.3361 | 0.3338 | 0.3746 | 0.4212 | **0.7** |
| 16 | 0.4001 | 0.3936 | 0.3876 | 0.3862 | 0.3851 | 0.3776 | 0.4156 | 0.4656 | **0.7** |
| 17 | 0.4298 | 0.4185 | 0.4029 | 0.3916 | 0.3759 | 0.3621 | 0.4201 | 0.4868 | **0.7** |
| 18 | 0.4314 | 0.4225 | 0.4171 | 0.4161 | 0.4086 | 0.3967 | 0.4660 | 0.5200 | **0.7** |
| 19 | 0.5275 | 0.5183 | 0.5065 | 0.5003 | 0.4943 | 0.4955 | 0.5605 | 0.6262 | **0.5** |
| 20 | 0.5108 | 0.4954 | 0.4725 | 0.4632 | 0.4526 | 0.4490 | 0.5136 | 0.5638 | **0.7** |
| 21 | 0.4163 | 0.4061 | 0.3933 | 0.3869 | 0.3815 | 0.3860 | 0.4398 | 0.4888 | **0.5** |
| 22 | 0.5200 | 0.5096 | 0.4971 | 0.4913 | 0.4882 | 0.4990 | 0.5683 | 0.6351 | **0.5** |
| 23 | 0.5095 | 0.5028 | 0.5004 | 0.5010 | 0.5082 | 0.5315 | 0.5870 | 0.6501 | **0.25** |
| 24 | 0.4440 | 0.4387 | 0.4347 | 0.4350 | 0.4399 | 0.4562 | 0.5155 | 0.5659 | **0.25** |
| 25 | 0.4312 | 0.4266 | 0.4288 | 0.4365 | 0.4571 | 0.4761 | 0.5485 | 0.5862 | **0.1** |
| 26 | 0.4677 | 0.4631 | 0.4609 | 0.4624 | 0.4695 | 0.4853 | 0.5462 | 0.5868 | **0.25** |
| 27 | 0.4242 | 0.4201 | 0.4189 | 0.4129 | 0.4229 | 0.4352 | 0.4865 | 0.5286 | **0.35** |

> **The minimum is interior at 28 of 28 layers.** It is never at `gamma -> 0` (`C8`) and never at
> `gamma -> 1` (`C6`). A short-but-nonzero exponential envelope beats both no-mixing and
> uniform-mixing at every layer of this donor.

**The argmin is depth-dependent, and it is NOT monotone in depth:**

| layers | mean argmin `gamma` |
|---|---|
| 0-8 | **0.422** |
| 9-18 | **0.640** |
| 19-27 | **0.378** |

Counts over 28 layers: `gamma = 0.5` (10 layers), `0.7` (9), `0.25` (6), `0.35` (2), `0.1` (1).
Correlation of argmin with layer index: **`r = -0.125`** - i.e. none.

> HGRN's thesis is that the correct decay is **monotonically increasing with depth** - short at the
> bottom, long at the top - and it builds that in structurally via a `cummax` lower bound. **This
> donor's layer-local geometry gives an inverted U instead**: shortest at both ends, longest in the
> middle third. I state the divergence and do not resolve it: HGRN is a from-scratch RNN measured by
> perplexity, this is a converted-donor subspace residual, and the Researcher marks the carry-over
> to a softmax donor as `[X]`.
>
> **The operational consequence is the one HGRN implies and this measurement confirms: a single
> global `gamma` is being compared against 28 different correct answers. The per-layer argmin above
> is the honest reporting unit; a global winner is not.**

**Resolution caveat.** For the 23 non-subset layers the ladder is
`{0, 0.1, 0.25, 0.35, 0.5, 0.7, 0.99, 1}`, so an argmin reported at `0.7` is bracketed by 0.5 and
0.99 and could lie anywhere in `(0.5, 0.99)`. The argmin is located to the ladder's resolution and
no finer.

**The content kernel is null at the four new `gamma` values too** (60 pairs each):
`decay_g0.1 - decayonly_g0.1 = -0.00004 +/- 0.00014`; `0.25: -0.00012 +/- 0.00043`;
`0.35: -0.00019 +/- 0.00065`; `0.7: -0.00047 +/- 0.00126`. **Nine `gamma` values now, all null.**

**Best-`gamma`-per-layer, as an ORACLE.** Choosing `gamma` per layer *on the same data it is scored
on* is an overfit upper bound and is labelled as one. Pooled median **0.4194** (sd 0.0975), paired:

| best-`gamma` (oracle) vs | mean | sd | better in |
|---|---|---|---|
| `C6_causal_uniform` | **-0.1283** | 0.0854 | 329/336 |
| `elu1` | **-0.1225** | 0.0846 | 327/336 |
| `C8_no_mixing` | **-0.0192** | 0.0687 | 260/336 |
| `taylor2` | **-0.0385** | 0.1186 | **184/336** |

Even with `gamma` chosen by oracle, **the envelope family beats `taylor2` on barely more than half
the head-layer pairs**, and beats plain no-mixing by only `-0.019`.

### 8.7 `decay_g0.5` versus `taylor2` is systematic with depth, not noise

`decay_g0.5` wins **24 of 28 layers**; `taylor2` wins only at layers 21, 25, 26 and 27 - the
deepest. The disagreement is **systematic with depth**: the correlation of
`(decay_g0.5 - taylor2)` with layer index is **`r = +0.604`**. Paired over 336 pairs,
`-0.0371 +/- 0.1154`, `decay_g0.5` better in 184/336 - and that near-split is created entirely by
the deep layers, where `taylor2` pulls ahead (layer 27: `+0.1056` in `taylor2`'s favour, layer 0:
`-0.2058` in `decay`'s).

### 8.8 `taylor2` gets MATCHED nulls - and they do not follow it down

The `C5p_wrong_sequence` and `C4_wrong_layer` arms in the main sweep were **all built with the
`elu1` kernel** (`C5P_C4_PHI = "elu1"` in the apparatus). **That is not a matched control for
`taylor2`**, and the gap it leaves is specific: `1 + z + z^2/2` is non-monotone below `z = -1`, so a
subspace-coverage quantity could reward that artifactually. A `nulls` stage (1472 s, same `T/D`)
rebuilt `C5'` and `C4` **with each arm's own feature map**.

**Reproducibility check first.** The newly computed `C5p_elu1` / `C4_elu1` reproduce the original
`C5p_wrong_sequence` / `C4_wrong_layer` at **mean `+0.0000`, sd `0.0000`** over 60 pairs - a
separate run, a separate activation capture, bit-identical numbers.

| arm - its OWN matched null | n | mean | sd | median | arm better in |
|---|---|---|---|---|---|
| **`taylor2` - `C5p_taylor2`** | 336 | **-0.1623** | 0.0803 | -0.1645 | **333/336** |
| **`taylor2` - `C4_taylor2`** | 336 | **-0.1048** | 0.0513 | -0.1059 | **335/336** |
| `favor_d14` - `C5p_favor_d14` | 60 | -0.0419 | 0.0395 | -0.0383 | 45/60 |
| `hedgehog0` - `C5p_hedgehog0` | 60 | -0.0167 | 0.0153 | -0.0146 | 56/60 |
| **`elu1` - `C5p_elu1`** | 60 | **-0.0008** | 0.0078 | -0.0006 | 38/60 |
| `elu1` - `C4_elu1` | 60 | -0.0006 | 0.0060 | -0.0005 | 38/60 |

> **The concern is not borne out. `taylor2`'s own nulls do not follow it down - they go up.**
> `C5p_taylor2` lands **`+0.0538` ABOVE the free causal floor**: the taylor2 kernel computed from
> the *wrong sequence* is worse than a uniform average. Against its matched null `taylor2` gains
> **`-0.1623` on 333 of 336 pairs** - a *larger* separation than against the mismatched elu1-built
> null (`-0.0976`). **The wrong null had been understating `taylor2`, not flattering it.**
>
> The separation is **largest on the retrieval stratum**: `taylor2 - C5p_taylor2` on peaked rows is
> **`-0.1966 +/- 0.1128`, better in 329/336**.
>
> **`taylor2` has genuine, large, content-dependent structure; `elu1` has none** (`-0.0008`, 38/60).
> Both numbers come from the same code path on the same pairs, and they differ by 200x.

**Per layer, all 28:**

| layer | `taylor2` | `C5p_taylor2` | `C4_taylor2` | t2 - C5p | t2 - C4 |
|---|---|---|---|---|---|
| 0 | 0.7157 | 0.7234 | 0.7617 | **-0.0179** | **-0.0311** |
| 1 | 0.3473 | 0.4952 | 0.5491 | **-0.1051** | **-0.1260** |
| 2 | 0.4704 | 0.5866 | 0.5801 | **-0.1252** | **-0.0870** |
| 3 | 0.5188 | 0.6705 | 0.6049 | **-0.1513** | **-0.0856** |
| 4 | 0.5718 | 0.7394 | 0.6465 | **-0.1894** | **-0.0820** |
| 5 | 0.5591 | 0.6622 | 0.6009 | **-0.1580** | **-0.0773** |
| 6 | 0.5046 | 0.6751 | 0.5772 | **-0.1429** | **-0.0735** |
| 7 | 0.3726 | 0.5113 | 0.4708 | **-0.1130** | **-0.0863** |
| 8 | 0.5805 | 0.7409 | 0.6521 | **-0.1753** | **-0.0865** |
| 9 | 0.5282 | 0.6774 | 0.6295 | **-0.1536** | **-0.0959** |
| 10 | 0.4574 | 0.6415 | 0.5979 | **-0.1701** | **-0.1174** |
| 11 | 0.3659 | 0.5203 | 0.4832 | **-0.1298** | **-0.1037** |
| 12 | 0.4517 | 0.6746 | 0.5853 | **-0.2222** | **-0.1331** |
| 13 | 0.3909 | 0.5559 | 0.5523 | **-0.1398** | **-0.1252** |
| 14 | 0.5053 | 0.6983 | 0.6308 | **-0.1763** | **-0.1046** |
| 15 | 0.3958 | 0.5061 | 0.4592 | **-0.0959** | **-0.0560** |
| 16 | 0.3717 | 0.5745 | 0.5352 | **-0.1874** | **-0.1447** |
| 17 | 0.4487 | 0.6112 | 0.5358 | **-0.1761** | **-0.1085** |
| 18 | 0.4466 | 0.5915 | 0.5539 | **-0.1182** | **-0.0892** |
| 19 | 0.5226 | 0.7394 | 0.6549 | **-0.1905** | **-0.1104** |
| 20 | 0.5099 | 0.6560 | 0.6256 | **-0.1745** | **-0.1228** |
| 21 | 0.3983 | 0.5818 | 0.5107 | **-0.1963** | **-0.1247** |
| 22 | 0.5711 | 0.7297 | 0.6531 | **-0.1629** | **-0.0931** |
| 23 | 0.5287 | 0.7313 | 0.6550 | **-0.2013** | **-0.1203** |
| 24 | 0.4340 | 0.5879 | 0.5371 | **-0.1614** | **-0.1095** |
| 25 | 0.4185 | 0.6482 | 0.5528 | **-0.2218** | **-0.1359** |
| 26 | 0.3995 | 0.6548 | 0.5330 | **-0.2343** | **-0.1250** |
| 27 | 0.2874 | 0.5565 | 0.4756 | **-0.2548** | **-0.1797** |

### 8.9 But `taylor2` is NOT approximating `exp` on this donor - the `q.k` range, measured

Hedgehog section 4.1 `[A]` states the precondition: the 2nd-order Taylor map tracks `exp` *"only in
bounded regimes"*, and it verified for BERT that *"the query-key dot products are bounded in regimes
where the second-order Taylor series `exp` approximation maintains monotonicity"*.
`1 + z + z^2/2` has derivative `1 + z`, so it is **decreasing for `z < -1`**, giving *more* weight to
*more dissimilar* pairs. **The Researcher marks the decoder-only case `[X]`: no source read reports
the `q.k` range for any modern decoder-only LLM, and none for a Qwen-family donor.** It is cheap
here, so it was measured. `z = scaling * q.k` over causal pairs:

| layer | head | min z | p0.1 | p50 | max z | **frac(z < -1)** | **kernel-mass frac below -1** |
|---|---|---|---|---|---|---|---|
| 0 | head0 | 138.93 | 142.05 | 396.23 | 686.07 | **0.0000** | **0.0000** |
| 0 | head5 | -70.76 | -25.18 | 175.04 | 2433.20 | **0.0181** | **0.0000** |
| 0 | head11 | -25.29 | -19.85 | 20.67 | 63.92 | **0.0590** | **0.0047** |
| 7 | head0 | -10.41 | -8.92 | -4.35 | 3.71 | **0.9747** | **0.9951** |
| 7 | head5 | -16.58 | -12.53 | -5.96 | 4.25 | **0.9904** | **0.9984** |
| 7 | head11 | -5.87 | -0.16 | 9.08 | 22.48 | **0.0005** | **0.0000** |
| 14 | head0 | -29.01 | -21.63 | -9.84 | 7.38 | **0.9872** | **0.9989** |
| 14 | head5 | -39.98 | -29.74 | -12.14 | 5.51 | **0.9940** | **0.9997** |
| 14 | head11 | -24.01 | -18.87 | -7.96 | 7.09 | **0.9601** | **0.9971** |
| 21 | head0 | -22.60 | -17.48 | -7.57 | 4.09 | **0.9887** | **0.9992** |
| 21 | head5 | -18.17 | -13.68 | -5.60 | 6.42 | **0.9719** | **0.9971** |
| 21 | head11 | -19.71 | -14.92 | -6.93 | 7.39 | **0.9901** | **0.9992** |
| 27 | head0 | -13.09 | -9.10 | 0.40 | 10.56 | **0.3033** | **0.2326** |
| 27 | head5 | -12.96 | -8.56 | 1.50 | 11.52 | **0.1834** | **0.0920** |
| 27 | head11 | -11.91 | -7.01 | 1.72 | 13.20 | **0.1386** | **0.0461** |

> **The precondition fails, and not marginally.** In the middle of the network **96-99 % of causal
> pairs sit below `z = -1`, carrying 99.5-99.97 % of the kernel mass** - the regime where the map is
> *decreasing*. `z` spans `-40` to `+2433` across heads. At layer 0 head 0 the *minimum* `z` is
> `+138.9`, so `z^2/2 > 9600` and the "Taylor approximation of `exp`" is a pure quadratic there too.
>
> **So `taylor2`, on this donor, is not a softmax approximation. Over almost all of its mass it is a
> magnitude kernel dominated by `z^2/2` and nearly symmetric in the sign of `z`.**

**This does not retract 8.8.** The matched nulls establish that the structure `taylor2` finds is
real and content-dependent, and no non-monotonicity artefact survives a control carrying the
identical non-monotonicity with the wrong content. **What it retracts is the interpretation.**
`taylor2`'s advantage cannot be attributed to approximating the donor's softmax more faithfully,
because over 96-99 % of the mass it is not approximating `exp` at all.

And there is no clean relation between the two quantities: the null-separation is **smallest** at
layer 0 (`-0.018`, where `frac(z<-1) = 0.026`) and **largest** at layer 27 (`-0.255`, where
`frac = 0.208`), with the near-saturated layers 14 and 21 (`frac = 0.98`) in between at `-0.176` and
`-0.196`. **Five layers is too few to fit a trend to, and I do not fit one.**

**Consequence for anyone building on this arm:** the one arm in this probe that beats every floor is
operating far outside the regime its source paper requires, by a mechanism nobody has characterised,
and it would ship an `m = 16 513` recurrent state (section 11.6). **That is a reason to investigate
it, not a reason to build on it.**





### 8.10 Entropy stratification of the whole `gamma` family, on all 336 pairs

This supersedes 8.4's 60-pair table. Same strata, same definitions; `peaked - diffuse` paired within
(layer, head), n = 336 for every row:

| arm | peaked D1 | mid | diffuse D10 | **peaked - diffuse** |
|---|---|---|---|---|
| `C8_no_mixing` (`gamma -> 0`) | 0.4890 | 0.4487 | 0.4207 | **+0.0509 +/- 0.1284** (234/336) |
| `decay_g0.1` | 0.4781 | 0.4426 | 0.4085 | **+0.0521 +/- 0.1252** (237/336) |
| `decay_g0.25` | 0.4707 | 0.4287 | 0.4042 | **+0.0522 +/- 0.1229** (239/336) |
| `decay_g0.35` | 0.4643 | 0.4219 | 0.3995 | **+0.0520 +/- 0.1235** (242/336) |
| `decay_g0.5` | 0.4638 | 0.4146 | 0.4059 | **+0.0531 +/- 0.1264** (248/336) |
| `decay_g0.7` | 0.4786 | 0.4285 | 0.4182 | **+0.0610 +/- 0.1303** (250/336) |
| `decay_g0.99` | 0.5774 | 0.4923 | 0.4716 | **+0.0996 +/- 0.1306** (276/336) |
| `C6_causal_uniform` (`gamma -> 1`) | 0.6630 | 0.5524 | 0.5145 | **+0.1366 +/- 0.1432** (300/336) |
| `taylor2` | 0.5304 | 0.4623 | 0.4016 | **+0.1079 +/- 0.1085** (293/336) |
| `elu1` | 0.6486 | 0.5462 | 0.5114 | **+0.1334 +/- 0.1401** (300/336) |

> **Every arm damages the retrieval rows more than the diffuse ones. There is no exception.** The
> penalty is flat at `+0.052` for `gamma <= 0.5`, then climbs monotonically: `+0.061` at 0.7,
> `+0.100` at 0.99, `+0.137` at the uniform limit. **The longer the envelope, the worse the
> retrieval-like rows do relative to the bulk** - and the effect is 2.7x from end to end of the
> family.

Paired on the peaked stratum against `C8_no_mixing` (n = 336):

| arm | mean | sd | better than no-mixing on peaked rows in |
|---|---|---|---|
| `decay_g0.1` | **-0.0068** | 0.0090 | 303/336 |
| `decay_g0.25` | **-0.0120** | 0.0277 | 277/336 |
| `decay_g0.35` | **-0.0124** | 0.0418 | 258/336 |
| `decay_g0.5` | -0.0075 | 0.0641 | 214/336 |
| `decay_g0.7` | +0.0123 | 0.0942 | 165/336 |
| **`taylor2`** | **+0.0433** | 0.1553 | **149/336** |

> **On the retrieval stratum the ordering inverts against the bulk.** In the pooled residual
> `taylor2` beats every envelope arm at the deep layers; on the peaked rows it is the *worst* of
> this set and loses to plain no-mixing on 187 of 336 pairs. **The short envelopes
> (`gamma = 0.25-0.35`) are the only arms that beat no-mixing on retrieval rows**, and they beat it
> by `0.012` - three times smaller than their bulk advantage.
>
> **This is the false negative the stratification was mandated to catch, and it caught it in the
> direction predicted.** An envelope-dominated mixing looks best in the pooled number and is
> nearly indistinguishable from no mixing where retrieval lives.

**On the surface tension with Gated DeltaNet.** GDN's own section 3.2 is headed *"Decay hurts memory
retention"*, with S-NIAH at 4K/8K putting no-decay DeltaNet at 99.0/98.8 against gated at 91.4/91.8
`[T, via the Researcher]`. Our family runs the opposite way: the `gamma -> 1` (no-forgetting) limit
is the *worst* peaked stratum. **These are not the same comparison and neither validates the
other.** GDN's "no decay" arm retains a delta-rule erase term and a content kernel and is trained
end-to-end; our `gamma -> 1` limit is a content-free uniform causal average measured with no
training at all. What both say, and all this probe claims, is that **the envelope's length is a
recall-relevant knob and must never be tuned on a pooled average.**

---

## 9. Rank diagnostics, conditioning, and tolerance sensitivity

Controller-audit finding 9 required the **exact** `hessian_rank_ratio` on `ZZ^T`, never the
`theoretical_rank_ratio` stand-in, and warned that a feature map of dimension `m` could make
`rank(Z)` collapse toward `m`. Measured, tolerance `1e-6` on singular values of `Z`:

| arm | median `rank(Z)` | `rank(Z)/D` | median `rank(Z*)` | median `cond(ZZ^T)` | residual @1e-5 | @1e-6 | @1e-7 |
|---|---|---|---|---|---|---|---|
| `C1_identity` | **1536** (1536 @1e-5) | 1.0000 | 1536 | 2.80e+08 | 0.0000 | **0.0000** | 0.0000 |
| `elu1` | **1536** (1533 @1e-5) | 1.0000 | 1536 | 8.76e+10 | 0.5578 | **0.5569** | 0.5567 |
| `taylor2` | **1536** (1534 @1e-5) | 1.0000 | 1536 | 5.45e+10 | 0.4606 | **0.4604** | 0.4604 |
| `favor` | **8** (8 @1e-5) | 0.0052 | 1536 | 4.27e+04 | 0.8585 | **0.8584** | 0.8578 |
| `hedgehog0` | **1536** (1533 @1e-5) | 1.0000 | 1536 | 8.72e+10 | 0.5408 | **0.5407** | 0.5407 |
| `C6_causal_uniform` | **1536** (1533 @1e-5) | 1.0000 | 1536 | 9.36e+10 | 0.5612 | **0.5608** | 0.5608 |
| `C5p_wrong_sequence` | **1536** (1535 @1e-5) | 1.0000 | 1536 | 2.62e+10 | 0.5266 | **0.5264** | 0.5264 |
| `C4_wrong_layer` | **1536** (1535 @1e-5) | 1.0000 | 1536 | 2.73e+10 | 0.5273 | **0.5271** | 0.5270 |
| `C7_random_gauss` | **1536** (1536 @1e-5) | 1.0000 | 1536 | 6.34e+00 | 0.9007 | **0.9007** | 0.9007 |
| `C8_no_mixing` | **1536** (1536 @1e-5) | 1.0000 | 1536 | 1.53e+07 | 0.4534 | **0.4533** | 0.4533 |

**The finding is the opposite of the audit's worry, and it sharpens the verdict.** For every arm
except `favor`, `rank(Z) = 1536 = D` **exactly** - the mixing is not rank-deficient at all. So

> **the ceiling measured here is NOT a rank ceiling. It is an ANGLE ceiling.** `rowspace(Z)` has the
> full `D` dimensions available to it and still fails to contain the target: the subspace is the
> right *size* and points the wrong *way*. Adding state dimension would not fix this;
> `C7_random_gauss` also has rank 1536 and sits at 0.90.

**Tolerance sensitivity is negligible** - the residual moves by `<= 0.0011` across three decades of
rank tolerance (`elu1`: 0.5578 / 0.5569 / 0.5567). The result is not a truncation artefact. The
*flattering* direction here is a small tolerance (more retained dimensions, lower residual); the
primary `1e-6` is the middle of the ladder and the extremes bracket it tightly.

`cond(ZZ^T)` runs `~9e10` for the content-bearing arms - i.e. `cond(Z) ~ 3e5`, consistent with the
audit's prediction that mixing badly worsens conditioning. This is why the eigh route was validated
against a condition-preserving QR route on 240 production rows (`max|delta| = 3.65e-08`, §3).

---

## 10. Allocation inventory - every `O(T)` and `O(T^2)` object

Per the standing law, including the ones that fit. At the ACHIEVED `D = 1536`, `T = 8192`,
`L_seq = 1024`, `n_seq = 8`, `d_v = 128`, `m = head_dim = 128`, `Hq = 12`, 28 layers, fp64 for all
linear algebra.

| object | shape | size | order | disposition |
|---|---|---|---|---|
| donor weights | 1.544 G params fp32 | **5 888.8 MB** | `O(1)` | resident, whole run |
| `xn_all` post-RMSNorm activations, **all 28 layers** | `[28, 8, 1024, 1536]` fp32 | **1 344.0 MB** | `O(T)` x layers | **materialised, whole run** |
| `xn_alt` (C5' slice), 5 subset layers | `[5, 8, 1024, 1536]` fp32 | 240.0 MB | `O(T)` | materialised |
| `q` post-RoPE, one layer | `[8, 12, 1024, 128]` fp64 | 96.0 MB | `O(T)` | per layer (x3 when C5'/C4 run) |
| `k` post-RoPE, one layer | `[8, 2, 1024, 128]` fp64 | 16.0 MB | `O(T)` | per layer |
| `Zs = X A_soft^T`, one head | `[1536, 8192]` fp64 | 96.0 MB | `O(T)` | per head |
| `M = W_v Z*`, one head | `[128, 8192]` fp64 | 8.0 MB | `O(T)` | per head |
| `Z = X A_lin^T`, one head-arm | `[1536, 8192]` fp64 | 96.0 MB | `O(T)` | per head-arm |
| **`A_soft` / `A_lin` block** | **`[1024, 1024]` fp64** | **8.0 MB** | **`O(L_seq^2)`, NOT `O(T^2)`** | **per sequence, freed** |
| `G = Z Z^T` + eigenvectors | `[1536, 1536]` fp64 x2 | 36.0 MB | **`O(1)` in `T`** | per head-arm |
| `C = M Z^T`, `K` | `[128, 1536]` fp64 x2 | 3.0 MB | `O(1)` in `T` | per head-arm |
| `R = M - K Z` (per-column residuals) | `[128, 8192]` fp64 | 8.0 MB | `O(T)` | per head-arm |
| QR cross-check `Q` (layers 0, 14 only) | `[8192, 1536]` fp64 | 96.0 MB | `O(T)` | transient |

**Predicted peak 7.91 GB; observed RSS of the running process 8.6 GB.**

**Nothing is ever materialised as `T x T`.** A full `A_soft` at `T = 8192` would be **512 MB per
head**, **6.0 GB for one layer's 12 heads**. It is avoided because `A` is block-diagonal over the 8
calibration sequences, so the largest attention matrix in existence at any moment is
`L_seq x L_seq = 1024 x 1024 = 8 MB`.

### 10.1 **Yes, `A_lin` IS materialised densely, and here is what that costs and what it hides**

The brief asks this to be said plainly. **`A_lin` is materialised as a dense `1024 x 1024` block per
sequence.** The operator we would actually ship has an `O(1)`-state recurrence and would never form
it. Three consequences, measured rather than asserted:

1. **Same numbers.** The `O(1)`-state recurrence
   `S_t = sum_{s<=t} x_s phi(k_s)^T`, `z_t = S_t phi(q_t) / (phi(q_t) . n_t)` was implemented and
   compared against the block form for `elu1`: **`max_abs = 7.11e-15`, relative `4.90e-16`** (S8).
   In fp64 they are the same computation.
2. **The fp32 drift the audit warned about is NOT exercised here.** This apparatus is fp64
   throughout, so the running-sum cancellation that afflicts a shipped fp32 recurrence at long `T`
   does not appear. **The measurement is therefore mildly optimistic relative to what would ship**,
   which is the flattering direction - but the result is a *large* residual, so the bias works
   against the conclusion rather than for it.
3. **Cost.** Block route `2.79e10` flops per head-arm; the `O(1)` recurrence would be `3.22e9`,
   **8.7x cheaper**. A naive single-context `T x T` route would be `2.06e11`, 7.4x *worse* than the
   block route.

### 10.2 The shipped-state inflation, confirmed numerically

The commuting step moves the mixing onto `x in R^D` rather than `v in R^{d_v}`. Taken literally, the
recurrent state becomes `R^{D x m}` instead of `R^{d_v x m}`. Measured off the constructed objects:

| form | state shape | fp32 bytes/layer (12 heads) |
|---|---|---|
| literal "mix then project" (what §1.2's wording invites) | `1536 x 128` | **9.00 MB** |
| correct "project then mix" | `128 x 128` | **0.75 MB** |

**12x inflation**, against the engine's measured 16 MB L3 cliff. This reproduces the Controller's
prediction (9.4 MB vs 0.8 MB) to within rounding. **If any operator on this line is ever shipped, it
must project then mix.**

---

## 11. What this measurement does NOT cover, and one apparatus bug it found

### 11.1 `favor` as registered was mis-scaled - the arm is a bug report, not a verdict on Performer

`favor` posted 0.8584, close to the pure-dimension floor, worse than `C6` on **all 336** pairs. Before
reading that as "Performer fails", the mechanism was diagnosed:

| diagnostic (layer 14, head 0) | measured |
|---|---|
| `||q * scaling||^2 / 2` across tokens | 1.31 .. 4.41 |
| **`||k||^2 / 2` across tokens** | **2.33 .. 446.25 (spread 443.9)** |
| key feature mass on the single largest key, as a fraction of the total | **1.0000** |
| numerical rank of the `1024 x 1024` `A_lin` block | **1** (vs `elu1` 1024, `hedgehog0` 1020, `taylor2` 982) |
| median `rank(Z)` at `T = 8192` | **8** of 1536 |

FAVOR+'s map is `phi(x) = exp(w.x - ||x||^2/2)`. I applied the donor's `scaling = d^{-1/2}` to `q`
only and left `k` raw, so `||k||^2/2` spanned 444 in the exponent and the map put **all** its mass on
one key. **The faithful FAVOR+ softmax-kernel approximation requires `d^{-1/4}` on BOTH `q` and `k`**
so that `phi(q').phi(k') ~ exp(q.k/sqrt(d))`.

**This was my error, not Performer's.** A corrected arm `favor_d14` was added and re-run on all 28
layers; see §8. **The `favor` row above should be read only as: an unfitted, un-normalised FAVOR+ map
applied to donor-scale keys collapses to rank ~1 and is unusable.** That is a real and useful
warning, but it is not a measurement of Performer.

### 11.2 `phi` scaling is a free parameter I fixed by fiat, and it is checked, not assumed

`taylor2` inherits the donor's `d^{-1/2}` on the dot product because Based *is* the Taylor expansion
of `exp(q.k/sqrt(d))` - that one is forced. **`elu1` and `hedgehog0` have no canonical scaling**: the
Katharopoulos and Hedgehog maps are applied to `q`, `k` directly and the softmax temperature has no
role once the softmax is gone. I applied `scaling` to `q` and not to `k`. Because that is a choice,
`elu1_noscale` and `hedgehog0_noscale` (no scaling anywhere) were added on the 5 subset layers - §8.

### 11.3 GQA: these are PER-HEAD residuals, and the joint ceiling is provably worse

The donor is GQA with **6 q-heads sharing one `W_v`**. The brief's formula is per-head, and that is
what is reported. The operative problem for R2 is the *joint* one over the sharing group. For that,

```
min_W sum_h ||W Z_h - M_h||^2  >=  sum_h min_W ||W Z_h - M_h||^2  =  sum_h ||M_h (I - P_{Z_h})||^2
```

so **the joint residual is at least the `||M_h||`-weighted RMS of the per-head residuals.** Every
number in this document is therefore a **lower bound** on the ceiling R2 would actually face on this
donor. Indicative RMS over the 6 heads of each kv group (equal-norm approximation, since `||M_h||`
was not retained per row):

| arm | median over (layer, kv-group) of RMS-over-6-heads | median per-head |
|---|---|---|
| `elu1` | **0.5316** | 0.5569 |
| `taylor2` | **0.4637** | 0.4604 |
| `hedgehog0` | **0.5271** | 0.5407 |
| `C6_causal_uniform` | **0.5373** | 0.5608 |
| `C8_no_mixing` | **0.4422** | 0.4533 |

Structurally it is worse still: the joint reachable set is a `<= D`-dimensional subspace inside
`R^{6T}` rather than `R^T`, so the pure-dimension floor for the joint problem is
`sqrt(1 - 1536/49152) = 0.984` rather than 0.901.

### 11.4 Other boundaries, stated

- **One donor, one revision.** Qwen2.5-1.5B only. Nothing here is a claim about larger donors,
  though `D` grows faster than `d_v` at scale, which pushes the same way.
- **`A` is block-diagonal over 8 sequences of 1024 tokens**, not one 8192-token context. The numbers
  describe realistic 1024-token attention pooled over 8 draws. Long-context behaviour is untested.
- **fp64 throughout.** The fp32 recurrence drift of a shipped operator is not exercised (§10.1);
  this biases the measurement *optimistic*.
- **Layer-local only.** By construction this says nothing about cross-layer interaction or
  residual-stream drift, which Taylor-Calibrate's own limitations section names as the reason a
  value-side solve does not remove the need for downstream training.
- **The transpose trap did not fire** (§3): the instrument is insensitive to `A_lin` orientation.
- **No BPB, no MMLU, no downstream evaluation was run here.** Per the brief's Amendment 1 reporting
  rule, if any downstream stage of this line reports MMLU it may never appear inside an average.

### 11.5 Does the no-mixing result replicate a published effect, or merely resemble one?

**Provenance note.** The numbers below are quoted from the Researcher's
`prior_art/LINEAR_MIXING_MECHANISM.md`, which marks them `[T]` — read individually from the papers'
own tables. **I have not opened those papers myself**, so they are attributed here to the
Researcher's transcription, not to my own reading, per the standing rule. *(One correction to the
attribution I was handed: the `1+ELU`-vs-no-attention table is **arXiv:2510.05901 Table 4**, not
Liger's. Liger's corroboration is separate and sits in that paper's Table 6.)*

**Two corrections to the framing I was originally given, both from the Researcher, both of which
weaken the claim I was about to make:**

1. The `1+ELU` / no-attention pair is **arXiv:2510.05901 Table 4**, not Liger Table 4.
2. **That "No Attention" row is not a deletion.** Its section 3.2 `[A]`: *"no attention where we
   return an **all-zeros attention output**"*, applied at inference to an already-converted
   checkpoint. With the residual stream, `Y = X + 0 = X`; **no weights are removed.** It is a
   zero-output diagnostic, and it is **not** the same object as our `C8_no_mixing`, which passes
   each token's own value through (`A := I`, so `out_t = W_v x_t`). **The two controls are not
   interchangeable and I do not treat them as such.**

**And the "linear scores below no-attention" line must not be cited as a ranking.** On Llama3.1-8B
the teacher scores AVG **73.01**; Linear-only **33.96** and no-attention **33.93** `[T]`. Both MMLU
cells are **22.95, below the 25.0 four-way chance floor**, and Linear-only is above chance on **0 of
6 tasks**. **0.62 AVG points between two chance-level models is not an ordering**, and I withdraw
the version of this paragraph that treated it as one.

**What the same table does carry is a real corroboration, and it is the useful part.**
arXiv:2510.05901 Table 4 `[T]` (Mistral-7B, LA-only, one epoch of weights transfer, no fine-tuning),
margins over the zero-output control derived:

| phi activation | AVG | margin over zero-output control (34.40) | alive above chance on |
|---|---|---|---|
| Softmax | 46.94 | **+12.54** | 4 of 6 tasks |
| Exponential | 46.11 | **+11.71** | 4 of 6 tasks |
| ReLU | 36.07 | +1.67 | — |
| **`1 + ELU`** | **35.12** | **+0.72** | — |
| None | 34.81 | +0.41 | — |

> **The instrument works, and `1+ELU` fails it.** Exponential-family kernels clear the control by
> 12 points; `1+ELU` clears it by 0.72.

**Our ordering is the same ordering**, on a different donor, by a different metric, with no training
at all:

| this probe, `T/D = 5.333`, 28 layers x 12 heads | vs the free causal floor `C6` |
|---|---|
| `taylor2` (exponential-family: 2nd-order Taylor of `exp`) | **`-0.0899 +/- 0.0815`**, better in 285/336 |
| `hedgehog0` (frozen, unfitted) | `-0.0154 +/- 0.0193` |
| **`elu1`** | **`-0.0059 +/- 0.0121`** - on the floor |
| `favor` / `favor_d14` (Performer) | `+0.2796` / `+0.0931` - **below** the floor |

**Three independent tables now put the same three families in the same order**: exponential-family
kernels separate; elementwise positive maps (`1+ELU`) sit at the null; **Performer is worst** — ours
at `+0.0931` over the floor even after the mis-scaling was corrected, and Based Table 6 `[T]` has
Performer as its worst feature-map row by a wide margin (SWDE 8.10 vs Taylor's 29.16 at matched
`d'=16`; AR Ppl 8.53 vs 2.07).

> **This is a replication, and it is also a check on our instrument.** An instrument that had ranked
> `elu(x)+1` above the exponential family, or Performer above either, would have been contradicting
> two published tables. It does not. **But see section 8.7 — our `taylor2` turns out not to be
> behaving as an exponential approximation on this donor at all, which complicates the mapping
> between our arm and theirs.**

**Three fences the Researcher's file puts around this, which I adopt:**

1. **`[X]` No conversion result on a Qwen-family donor exists in any source read.** Our donor is
   outside the regime every published conversion number was measured in, and arXiv:2510.05901 §3.4.1
   `[A]` explicitly reports that donor identity is a live, untheorised variable
   (*"we have not observed any successful LA-only conversions of these models ... suggesting that
   they may be particularly hard to convert"*).
2. **`[X]` No dimension-matched kernel-shape comparison exists in the literature.** Based's Table 6
   ordering is monotone in *effective feature dimension*, not kernel identity, and Hedgehog's
   decisive 100.0-vs-17.0 is `O(nd^3)` vs `O(nd^2)`. **R2a is dimension-matched by construction** —
   `rank(Z) = D = 1536` for every arm except the broken `favor` (section 9) — so its kernel-shape
   comparison is one the literature does not have. **But see the limitation in 11.6.**
3. **Every LA-only row in arXiv:2510.05901 Table 1 has MMLU at or below the 25.0 four-way chance
   level** (23.28 / 22.95 / 23.09) while the macro-average understates the MMLU damage by 4.0-4.7x.
   Per Amendment 1's reporting rule, **MMLU may never be reported inside an average** at any
   downstream stage of this line.

### 11.6 The axis this instrument is blind to: state size

Based Theorem 3.1, Schlag's `d_dot` orthogonality bound and Zoology Theorem 4.4 `[A]` all say the
same thing: **for a fixed-state sequence mixer, recall capacity is set by state dimension.** And the
order of a polynomial feature map *is* its dimension — `elu(x)+1` is order 1 (`m = d = 128`), the
2nd-order Taylor map is order 2 (`m = 1 + d + d^2 = 16 513`).

**Our residual cannot see that axis at all.** `rowspace(Z)` lives in `R^T` and has dimension at most
`D = 1536` **whatever `m` is** — measured: `rank(Z) = 1536` for `elu1` (`m = 128`) and for `taylor2`
(`m = 16 513`) alike, because causal masking restores the rank that a low-dimensional kernel would
otherwise lose. So:

- **Good:** this is exactly the dimension-matched kernel-shape comparison the literature lacks.
- **Bad:** it therefore **cannot reproduce the recall-capacity advantage** that Based and Hedgehog
  attribute to the Taylor map. A `taylor2` separation here is evidence about *shape*, and is not
  evidence about the capacity mechanism the papers actually invoke — and a `taylor2` *null* here
  would not refute that mechanism either.
- **And a cost note that does not appear in the residual:** shipping `taylor2` as written means an
  `m = 16 513` recurrent state, 129x `elu1`'s. Our apparatus never pays it, because
  `phi(q).phi(k) = 1 + z + z^2/2` is computed directly from the dot product. **A residual comparison
  between these two arms is not a comparison between two operators we could equally afford.**

---

## 12. The ceiling, stated for the decision

**What the number means.** `residual` is the fraction of the donor head-output's Frobenius norm
lying orthogonal to everything `W_v Z` can produce. It is attained by no solver and beaten by none
(verified against brute-force `lstsq` at production shapes, `|diff| = 0.0`, section 3 S10).
`1 - residual^2` is the fraction of *energy* an ideal `W_v` could match.

At the pinned `T/D = 5.333`, pooled over all 28 layers x 12 heads:

| arm | residual | **energy an ideal `W_v` recovers** | vs free floor `C6` | vs no-mixing `C8` |
|---|---|---|---|---|
| best-`gamma` per layer (**ORACLE**) | **0.4194** | **82.4 %** | `-0.128` | `-0.019` |
| `C8_no_mixing` | 0.4533 | 79.4 % | `-0.109` | - |
| `taylor2` | 0.4604 | 78.8 % | `-0.090` | `+0.019` |
| `decay_g0.5` (content-free twin: 0.4048) | 0.4557 | 79.2 % | `-0.105` | `+0.002` |
| `hedgehog0` | 0.5407 | 70.8 % | `-0.015` | `+0.094` |
| `elu1` | 0.5569 | 69.0 % | `-0.006` | `+0.103` |
| `C6_causal_uniform` (**free floor**) | 0.5608 | 68.6 % | - | `+0.109` |
| `favor_d14` (corrected FAVOR+) | 0.6556 | 57.0 % | `+0.093` | `+0.202` |
| `C7_random_gauss` (**no relationship**) | 0.9007 | 18.9 % | | |
| `C1_identity` (**zero anchor**) | 0.0000 | 100 % | | |

At `T/D = 21.333`, the largest measured and not converged, `taylor2` is at 0.5104 (74 % energy) and
`elu1` at 0.6283 (61 %) against a free floor of 0.6288.

> ### Does this justify building the R2 solver?
>
> **No, and the pre-screen has now also said where the effort should go instead.**
>
> 1. **The registered memoryless feature maps have nothing for a solve to exploit.** `elu1` and
>    `hedgehog0` are not near the uniform-average floor - their matrices *are* it (0.4-7 % away, row
>    cosine 0.997-1.000), and equidistant from the donor's real attention with it. `favor_d14` is
>    below the floor even after its mis-scaling was fixed.
> 2. **`taylor2` is the single genuine exception and it does not survive scrutiny as a basis for
>    building.** Its matched nulls confirm real content dependence (`-0.1623`, 333/336, and largest
>    on the retrieval stratum), but it is not approximating `exp` on this donor (96-99 % of mass
>    below `z = -1`), it would ship a 16 513-dimensional state, and it loses to no-mixing on the
>    retrieval rows.
> 3. **The ceiling is an ANGLE ceiling, not a rank ceiling** (section 9): `rank(Z) = D = 1536`
>    exactly for every arm but the broken `favor`. The subspace is the right size and points the
>    wrong way. **More state dimension does not address it** - and this instrument is blind to the
>    state-size axis anyway (section 11.6).
> 4. **Everything here is a lower bound on what R2 would face**: the GQA joint solve is provably
>    worse (section 11.3), the `T/D` curve rises with calibration size and has not flattened
>    (section 6), and fp64 flatters relative to a shipped fp32 recurrence (section 10.1).
> 5. **A good ceiling would not have sufficed anyway.** Taylor-Calibrate's own limitations section
>    states the value-side solve improves the starting point but does not remove the need for
>    downstream training, because cross-layer interaction and residual-stream drift are untouched by
>    a layer-local objective. This probe is layer-local by construction.
> 6. **And the published ordering replicates here** (section 11.5): exponential-family kernels
>    separate, `1+ELU` sits at the null, Performer is worst - the same three-way order as
>    arXiv:2510.05901 Table 4 and Based Table 6, on a different donor, by a different metric, with
>    no training. That is a check on the instrument as much as a result.
>
> **What the probe found that was not asked for:** the quantity that moves the ceiling is the
> **temporal envelope**, its optimum is **interior at every layer**, **depth-dependent**, and
> **inverted-U in depth** - and the content kernel contributes `~0.0003` on top of it at all nine
> `gamma` tested. Amendment 1 named that outcome in advance as *"worth more than a verdict on R2"*.
>
> **Total cost: 8 364 measured rows across five measurement stages, plus a sixth diagnostic-only
> `selfcheck` stage; 13 990 s = 3.89 hours of one 6-core desktop. No GPU, no training, no solver.**
> `BRIEF_R2` Amendment A1.5 predicted the pre-screen would be the highest value per unit of compute
> on the R2 line. On this evidence it was.

**The one thing that would reopen this**: an arm that beats `C8_no_mixing` on the **peaked**
stratum by more than the `0.012` the short envelopes manage. Nothing tested does.

---

## 13. Reproducibility manifest

| | |
|---|---|
| apparatus | `benchmarks/donor_adaptation/r2/r2a_principal_angle.py` |
| stages | `selfcheck`, `main`, `curve`, `decay`, `gamma`, `nulls` (each `python r2a_principal_angle.py <stage>`) |
| results | `benchmarks/donor_adaptation/r2/results/r2a_{selfcheck,main,curve,decay,gamma,nulls}.{json,log}` |
| donor | `Qwen/Qwen2.5-1.5B` rev **`8faed761d45a263340a0528343f099c05c9a4323`**, `config.json` sha256 `0e8c8aa8…71e66f77` |
| corpus | `benchmarks/donor_adaptation/density/corpus/calib.txt` sha256 `10d4d281…c59c89d0` |
| arm slice | `calib`, `n_seq` per stage, `seq_len=1024`, **seed 20260822**, ids sha256 `6e07b7ec…71e52c54` |
| C5' slice | `calib`, `seq_len=1024`, **seed 20260823**, ids sha256 `c165e205…ca07e3298` |
| **FAVOR+ seed** | **424242**, `m = 256` orthogonal random features |
| `C7_random_gauss` seed | `default_rng(9000 + sequence_index)` |
| planted-positive seed | `default_rng(11)` |
| rank tolerance | primary **`1e-6`** on singular values of `Z`; ladder `1e-5 / 1e-6 / 1e-7` all reported |
| strata | normalised entropy `H_t / log(t+1)`; deciles over `t >= 32`; `t < 32` a separate bucket |
| `C4_wrong_layer` donor layer | `(l + 14) mod 28` — fixed by rule before measurement |
| `C5p_wrong_sequence` construction | `phi(q), phi(k)` from the seed-20260823 slice at the **same** layer, applied to the seed-20260822 slice's `X` |
| `T <= D` | **refused**, not reported |
| python / numpy / torch / transformers | 3.12.10 / 2.4.6 / 2.12.0+cpu / 4.57.6 |
| platform | Windows-11-10.0.26200, AMD Ryzen 5 3600X, 6 torch threads, CPU only |
| precision | fp64 for all linear algebra; donor loaded fp32, `attn_implementation="eager"` |
| wall clock and rows | `selfcheck` 71 s (0 rows, diagnostics only); `main` 4273 s (**2532**); `curve` 1031 s (**360**); `decay` 4965 s (**2520**); `gamma` 2178 s (**1920**); `nulls` 1472 s (**1032**). **Totals: 13 990 s = 3.89 h; 2532+360+2520+1920+1032 = 8 364 rows.** CPU only |
| row counts, derived | `main` 7 arms x 336 + 3 x 60; `curve` 8 x 45 (3 layers x 3 heads x 5 `T`); `decay` 5 x 336 + 14 x 60; `gamma` 5 x 336 + 4 x 60; `nulls` 2 x 336 + 6 x 60. **Every arm present at its full designed count; no arm silently failed to run.** |
| `gamma` ladder | requested `0.1, 0.25, 0.35, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999`; **every one read back off the constructed `A` and exact** |
| window ladder | requested `32, 128, 512`; achieved 32, 128, 512 (`n_nonzero_in_row` counted off the matrix) |
| matched nulls | `C5p_<phi>` / `C4_<phi>` built with the arm's OWN feature map; the elu1-matched pair reproduces the original arms at sd `0.0000` |
| git | branch `research/donor-adaptation`. **NOT COMMITTED — commits are not the Builder's.** |

### Deviations from the brief, all in the strengthening direction

| # | deviation | why |
|---|---|---|
| 1 | added `C7_random_gauss` (random `Z`) | anchors the residual scale at the "no relationship" end; matches analytic `sqrt(1 - rank/T)` to 0.006 |
| 2 | added `C8_no_mixing` (`A := I`) | separates "reachable because the mixing is good" from "reachable because `Z` is built from `X`". It carries the sharpest finding |
| 3 | added `C0` planted positive + transpose trap | the audit's finding 19; C1 is blind to the `phi -> A_lin -> Z` path |
| 4 | added `favor_d14` | `favor` as registered was mis-scaled (§11.1); the corrected map is the honest test |
| 5 | added `elu1_noscale`, `hedgehog0_noscale` | `phi` scaling is unforced for these two maps (§11.2) |
| 6 | `A` block-diagonal over 8 x 1024 rather than one long context | affordability (§10); stated, not hidden |
| 7 | `C5'` / `C4` and the two floors ran on 5 pre-stated layers (0, 7, 14, 21, 27), not all 28 | cost; the layer set was fixed before measurement and never selected on results |

| 8 | `favor` re-run as `favor_d14` after its mis-scaling was found | section 11.1; the original row is kept and labelled |
| 9 | added the `gamma` stage (`0.1-0.7` on all 28 layers) after the first ladder proved unconverged | section 8.6; the endpoint-quoting error the brief names in section 1 |
| 10 | added matched per-`phi` nulls `C5p_<phi>` / `C4_<phi>` | the originals were all elu1-built and were not a control for `taylor2` (section 8.8) |
| 11 | added the `q.k` logit-range diagnostic | Hedgehog's boundedness precondition is `[X]` for decoder-only donors (section 8.9) |
| 12 | added the mixing-geometry diagnostic (Frobenius distance / row cosine between `A` matrices) | to separate a genuine null from an artefact of combination (section 8.1) |

**Nothing in the pre-registered brief was removed, reweighted, or re-scoped.**

### Corrections I made to my own earlier text in this document

| what | where |
|---|---|
| `decay_g0.5` claimed as "the only arm with a negative peaked-diffuse gap" - a 60-pair artefact; on 336 pairs it is `+0.0531` | section 8.4, superseded by 8.10 |
| the null comparison originally quoted 60-pair arms against 336-pair medians | fixed in section 4.2b |
| `taylor2`'s separation originally measured against elu1-built nulls | fixed in section 8.8; the matched null makes the separation **larger** |
| "linear attention scores below no attention" cited as a ranking | withdrawn in section 11.5 - both models are at chance |
| the `1+ELU` / no-attention table attributed to Liger | corrected to arXiv:2510.05901 Table 4 |
| `favor` reported without noting its mis-scaling | section 11.1, and `favor_d14` added |
| **the total row count, stated four ways and none agreeing (10 400 / 10 412 / 8 940 / 8 364)** | **fixed everywhere to the verified 8 364; cause in the note below** |
| the `gamma` stage's own row count (2520) and wall time (4270 s) | corrected to **1920 rows / 2178 s** |
| total wall clock quoted as ~5.1 h | corrected to **3.89 h** |

**What caused the row-count discrepancy, since a number that should be identical in several places
and is not has twice been a symptom here.** It was prose arithmetic, not data. The per-stage figures
for `main`, `curve`, `decay` and `nulls` were transcribed from each stage's printed
`wrote ... [Ns]` completion line and are all correct. **`gamma` is the only stage whose completion
line I never read**: I wrote section 8.6 while it was still running, took its shape from the `decay`
stage sitting above it in my notes (2520 rows), and invented a wall time. `nulls`' row count (1008)
was likewise arithmetic done in prose rather than counted. The two document totals (10 400, 10 412)
were never derived from anything at all — **they do not even equal the sum of my own stated
components (8 940)**, which is the tell I should have caught.

**No evidence is missing and no conclusion is affected.** A per-arm audit of all five result files
confirms every arm ran at its full designed count — 336 rows for a 28-layer arm, 60 for a
5-layer subset arm, 45 for a curve arm — with no partial or absent arm anywhere. Every headline
number in this document was computed directly from the JSON and independently re-verified against
it, with each comparison's `n` read from the data rather than asserted; that check is unaffected by
how many rows the prose claimed in total.


