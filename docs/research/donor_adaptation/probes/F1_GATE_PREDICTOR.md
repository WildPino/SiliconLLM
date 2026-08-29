# F1 — the gate-sign predictor: the construction works, the donor has no sparsity for it to exploit

**Builder** · 2026-08-29 · branch `research/donor-adaptation` @ `3f885c5`
**Brief:** `docs/research/donor_adaptation/briefs/BRIEF_F1_GATE_SIGN_PREDICTOR.md`, including **Amendment 1**.
**Instrument:** `benchmarks/donor_adaptation/f1/f1_gate_predictor.py` (4 stages, all complete).

> **Status: measurement complete, all four stages run, all pre-registered controls reported.**
> **Nothing here is a decision.** The Adapter/Principal owns the verdict.

---

## 0. The one-line result

**The instrument is sound and fires on a known positive (C4, 28/28 layers). The construction under test is
not what fails.** What fails is the brief's unstated premise: that this donor's FFN is sparse.

> **Measured on held-out tokens, at a 1% FFN-output-error budget, this donor needs a mean of `a = 0.9506`
> of its FFN neurons — 95%, not the 2–10% the brief prices.** On 24 of 28 layers `a ≥ 0.95`.
> An **oracle that reads `|h_i|` exactly** — a ceiling no gate-only predictor can reach — still needs
> **`a = 0.836` mean**. The sparsity the brief is built on is not present in the donor at this tolerance.

Consequently the best `FFN_active` any arm reaches at recall 0.99, at target width, charging the predictor,
is **0.7384** — and that is at layer 1, one of only two genuinely sparse layers, at `r = 256`. The mean over
all 28 layers at `r = 64` is **0.9788**. The brief's target is 0.0200 and the dense-gate floor it set out to
break is 0.3333. **The floor was not broken; the measurement never got near it.**

**This contradicts the brief's framing, not its algebra.** §1's cost identity is correct and the closed form
is correct. The premise that `a` is small is what the data refuses.

---

## 1. What `a` is, and against what tolerance — the number the whole brief rests on

### 1.1 Definition, ACHIEVED

This donor is **SwiGLU, not dReLU: there is no natural zero**, so the "active set" is a *choice* and the
instrument makes it explicit (`theta_rule`, `f1_gate_predictor.py:258`).

- `theta(eps)` = the **largest** `θ` such that zeroing every neuron with `g_i ≤ θ` costs at most `eps` of the
  FFN block's output energy, measured as `‖Y − Y(θ)‖_F / ‖Y‖_F` where `Y = h W_dᵀ`, `h = SiLU(W_g x) ⊙ (W_u x)`.
- Fitted on the **first 1024 calibration `X_theta` tokens** (`T_theta_achieved` on disk = 2048, sliced to 1024
  before the matmul).
- **`a_eval` — the ACHIEVED `a` used everywhere in this report — is the fraction of `(token, neuron)` pairs
  with `g > θ` on the 4096 HELD-OUT eval tokens**, i.e. θ is fitted on calibration and `a` reported
  out-of-sample.

**Search procedure ACHIEVED:** a 13-point coarse grid over quantiles of the pooled gate
(`np.linspace(0, 0.98, 13)`), then bisection. `n_refine = 14`, and the loop runs `n_refine + 8` = **22
bisection steps**. Convergence is verifiable per layer from `rel_err_achieved` vs `eps_requested` — see below.

**`eps` swept (ACHIEVED, pre-registered):** `EPS_LADDER = (0.001, 0.01, 0.05)`, `EPS_PRIMARY = 0.01`.

### 1.2 The full `a(eps)` curve — not one point

**Every row is ACHIEVED, read from `f1_analyse.json`.** `rel-err` is the error the bisection actually
converged to; it lands on the requested `eps` to 3–5 significant figures on all 28 layers × 3 ε, so **the
threshold search converged everywhere — a near-1.0 `a` is not a refinement failure.**

| layer | θ(.001) | a(.001) | rel-err | θ(.01) | **a(.01)** | rel-err | θ(.05) | a(.05) | rel-err | **a_eval (.01, held-out)** | a at θ=0 | rel-err at θ=0 | oracle-\|h\| a(.01) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | −4.607 | 0.9999 | 0.00100 | −3.446 | 0.9995 | 0.00996 | −2.304 | 0.9934 | 0.05000 | **0.9995** | 0.2388 | 0.4626 | 0.8359 |
| 1 | −7.781 | 0.8796 | 0.00100 | −5.393 | 0.3360 | 0.01000 | −1.250 | 0.0060 | 0.05000 | **0.3579** | 0.0015 | 0.0523 | 0.1328 |
| 2 | −6.991 | 0.9181 | 0.00100 | −4.779 | 0.4378 | 0.01000 | −1.135 | 0.0250 | 0.05000 | **0.4796** | 0.0069 | 0.0522 | 0.1953 |
| 3 | −7.793 | 0.9955 | 0.00100 | −6.425 | 0.9393 | 0.01000 | −5.182 | 0.7467 | 0.05000 | **0.9628** | 0.0275 | 0.6031 | 0.8164 |
| 4 | −6.968 | 0.9984 | 0.00100 | −5.719 | 0.9747 | 0.01000 | −4.578 | 0.8811 | 0.05000 | **0.9838** | 0.0783 | 0.6948 | 0.8750 |
| 5 | −8.016 | 0.9950 | 0.00100 | −6.637 | 0.9389 | 0.01000 | −5.435 | 0.7527 | 0.05000 | **0.9540** | 0.0132 | 0.8115 | 0.8125 |
| 6 | −5.352 | 1.0000 | 0.00099 | −3.884 | 0.9987 | 0.01000 | −2.814 | 0.9876 | 0.05000 | **0.9981** | 0.1304 | 0.7978 | 0.8906 |
| 7 | −4.616 | 1.0000 | 0.00099 | −3.336 | 0.9994 | 0.00999 | −2.402 | 0.9940 | 0.05000 | **0.9988** | 0.1608 | 0.7896 | 0.8828 |
| 8 | −4.997 | 1.0000 | 0.00099 | −3.694 | 0.9995 | 0.00991 | −2.587 | 0.9932 | 0.05000 | **0.9991** | 0.1615 | 0.8374 | 0.8828 |
| 9 | −4.527 | 1.0000 | 0.00100 | −3.334 | 0.9994 | 0.01000 | −2.394 | 0.9938 | 0.05000 | **0.9990** | 0.1854 | 0.8319 | 0.8789 |
| 10 | −4.822 | 0.9999 | 0.00100 | −3.547 | 0.9988 | 0.01000 | −2.577 | 0.9917 | 0.05000 | **0.9988** | 0.1860 | 0.8753 | 0.8828 |
| 11 | −5.521 | 0.9998 | 0.00100 | −4.057 | 0.9973 | 0.01000 | −2.899 | 0.9841 | 0.04999 | **0.9971** | 0.1814 | 0.8653 | 0.8828 |
| 12 | −5.109 | 0.9999 | 0.00100 | −3.801 | 0.9988 | 0.01000 | −2.635 | 0.9918 | 0.05000 | **0.9985** | 0.2211 | 0.7488 | 0.8789 |
| 13 | −5.137 | 0.9999 | 0.00099 | −3.634 | 0.9988 | 0.01000 | −2.469 | 0.9927 | 0.05000 | **0.9987** | 0.2392 | 0.8149 | 0.8750 |
| 14 | −4.708 | 1.0000 | 0.00100 | −3.468 | 0.9992 | 0.01000 | −2.430 | 0.9938 | 0.05000 | **0.9990** | 0.2557 | 0.7703 | 0.8672 |
| 15 | −4.961 | 1.0000 | 0.00100 | −3.585 | 0.9990 | 0.01000 | −2.523 | 0.9917 | 0.05000 | **0.9989** | 0.2606 | 0.7017 | 0.8711 |
| 16 | −4.828 | 0.9999 | 0.00100 | −3.537 | 0.9988 | 0.01000 | −2.487 | 0.9918 | 0.05000 | **0.9988** | 0.2571 | 0.6941 | 0.8750 |
| 17 | −5.234 | 0.9999 | 0.00100 | −3.772 | 0.9986 | 0.01000 | −2.587 | 0.9910 | 0.05000 | **0.9982** | 0.2627 | 0.6108 | 0.8750 |
| 18 | −5.307 | 0.9999 | 0.00100 | −3.780 | 0.9986 | 0.01000 | −2.603 | 0.9900 | 0.05000 | **0.9983** | 0.2992 | 0.5631 | 0.8711 |
| 19 | −5.329 | 0.9998 | 0.00100 | −3.726 | 0.9976 | 0.01000 | −2.582 | 0.9837 | 0.05000 | **0.9976** | 0.2905 | 0.4900 | 0.8633 |
| 20 | −5.387 | 0.9999 | 0.00100 | −3.846 | 0.9985 | 0.01000 | −2.788 | 0.9878 | 0.05000 | **0.9986** | 0.2536 | 0.5595 | 0.8750 |
| 21 | −5.615 | 0.9998 | 0.00100 | −4.237 | 0.9968 | 0.01000 | −3.151 | 0.9779 | 0.05000 | **0.9970** | 0.1942 | 0.5479 | 0.8789 |
| 22 | −5.846 | 0.9996 | 0.00100 | −4.301 | 0.9951 | 0.01000 | −3.152 | 0.9701 | 0.05000 | **0.9947** | 0.1957 | 0.5079 | 0.8672 |
| 23 | −5.862 | 0.9997 | 0.00100 | −4.234 | 0.9966 | 0.01000 | −3.095 | 0.9759 | 0.05000 | **0.9958** | 0.1869 | 0.4879 | 0.8672 |
| 24 | −5.762 | 0.9997 | 0.00100 | −4.131 | 0.9970 | 0.01000 | −2.911 | 0.9780 | 0.05000 | **0.9950** | 0.1979 | 0.4170 | 0.8555 |
| 25 | −6.102 | 0.9997 | 0.00100 | −4.227 | 0.9977 | 0.01000 | −2.892 | 0.9848 | 0.05000 | **0.9964** | 0.2166 | 0.4108 | 0.8555 |
| 26 | −4.736 | 0.9979 | 0.00100 | −2.664 | 0.9737 | 0.01000 | −1.066 | 0.6511 | 0.05000 | **0.9722** | 0.2014 | 0.0672 | 0.5234 |
| 27 | −5.456 | 0.9962 | 0.00100 | −3.888 | 0.9559 | 0.01000 | −2.680 | 0.8050 | 0.05000 | **0.9510** | 0.1431 | 0.1806 | 0.6875 |

**Mean over 28 layers:** `a(.001)` = 0.993 · **`a(.01)` = 0.951** · `a(.05)` = 0.909 (calib) ·
**`a_eval(.01)` = 0.9506 (held-out)** · **oracle-|h| `a(.01)` = 0.836**.

### 1.3 How to read this — the ε matters exactly as much as you said

- **ε = 0.01 is a *tight* bar, and that is the honest place to read `a`.** It is 1% relative Frobenius error
  in one FFN block's output, before that error propagates through 27 further layers. A per-layer budget
  looser than this is not obviously safe end-to-end, and **this run did not measure end-to-end** (§9).
- **Loosening ε does not rescue the construction.** At ε = 0.05 — five times looser, and a bar nobody has
  shown is survivable — `a` is still 0.909 mean, and ≥ 0.97 on 20 of 28 layers. Only layers 1, 2 and 26 come
  down (0.006, 0.025, 0.651).
- **The θ = 0 "sign rule" is where the sparsity everyone expects lives, and it is unaffordable.** It gives
  `a` = 0.13–0.30 — exactly the regime the engine wants — at a relative output error of **0.41 to 0.88** on
  most layers. That is not a tolerance; that is destroying the layer. **A dReLU-shaped intuition about
  "which neurons are off" does not transfer to a SwiGLU donor**, because SiLU has a negative lobe
  (min −0.2785 at g = −1.2785) and those neurons carry real output energy.
- **The oracle bounds any gate-only rule.** An oracle reading `|h_i| = |SiLU(g_i)·u_i|` exactly — which
  *also* sees `W_u`, so it is strictly stronger than anything F1 proposes — still needs `a = 0.836` mean at
  ε = 0.01. **So even a perfect, free predictor of the gate cannot get `FFN_active` below ≈ 0.84 on this
  donor at this tolerance.** The gap between 0.836 and 0.951 is the price of the negative lobe and of
  ignoring `W_u`; the gap between 0.836 and 0.02 is the price of the premise being wrong.

**Layers 1 and 2 are the exceptions** (`a` = 0.358, 0.480) and they are the only layers where the rest of
this report has any measurement room at all. Two layers out of 28 is not a route.

---

## 2. `S_frac @ recall 0.99, r=64` — the arms, and whether they are distinguishable

### 2.1 What each arm is

Every projector arm is the **same expression** — `ĝ = G U_r U_rᵀ`, a projection of the exact gate onto an
`r`-dimensional subspace of `R^F` — differing **only in which subspace**. That makes them exactly comparable.

| arm | what it is | role |
|---|---|---|
| **`fit_actw`** | `U_r` = top-`r` left singular vectors of `M = W_g H^{1/2}`, `H = XXᵀ` | **the claim** — activation-weighted, closed-form, no training |
| **`fit_plain`** | `U_r` from `M = W_g` (i.e. `H = I`) | does activation weighting earn its complexity? |
| **`C2_random_proj`** | `B` random `[r, D]`, `A` **solved** in closed form `A = W_g H Bᵀ (B H Bᵀ)⁻¹` | **the brief's registered floor**: how much does *any* `r`-dim read of `x` buy? |
| **`C2b_random_out`** | random orthonormal `U_r` in `R^F`, nothing fitted (**Builder addition**) | matched null: same family, same cost, *unfitted* subspace. Without it a win over C2 could be a win of family rather than of fit |
| **`C3_wrong_layer_*`** | `U_r` from layer `(L+14) mod 28`, in the **matching** weighting | real spectrum, wrong correspondence — the strong null that worked in D4 |
| **`C1_full_rank`** | `r = D = 1536` | **tautology** (§6) |
| **`C4_planted`** | a synthetically exactly-rank-32 `W_g`, on this layer's real `X` and real `H` | **known positive** (§6) |

Nulls are **rebuilt fresh per (layer, rank)**, and C3 exists in both weightings so each fitted arm is nulled
against a subspace of its own kind (`null_construction_policy`, verbatim in `f1_analyse.json`).

### 2.2 The resolution of the instrument — and it is the whole story

`S_frac` is a mean over `T_eval × F = 4096 × 8960 = 36,700,160` boolean entries; its **quantisation is
2.7 × 10⁻⁸**. So a 0.001 difference is ~37,000× the quantisation step and is *not* rounding noise.

**But quantisation is the wrong resolution to quote. The binding one is dynamic range.** At micro-recall ρ
with true active fraction `a`:

- a **perfect** predictor needs `S_frac = ρ·a` (arithmetic floor — you must keep at least the actives you claim);
- an **uninformative** predictor needs `S_frac ≈ ρ` (it must keep a ρ-fraction of *everything*).

**The entire span the instrument can resolve is therefore `ρ − ρ·a = ρ(1−a)`.** With `a ≈ 0.999`, that span
is **0.001 wide**. Every arm — fitted and null alike — is crushed against it.

| layer | `a` | floor `.99a` | uninformative (C2b) | **RANGE** | `fit_actw` r=64 | **skill** |
|---|---|---|---|---|---|---|
| 0 | 0.9995 | 0.9895 | 0.9900 | **0.0005** | 0.9895 | +0.975 |
| **1** | **0.3579** | 0.3543 | 0.9887 | **0.6344** | 0.8293 | **+0.251** |
| **2** | **0.4796** | 0.4749 | 0.9886 | **0.5138** | 0.8491 | **+0.272** |
| 3 | 0.9628 | 0.9531 | 0.9898 | 0.0367 | 0.9772 | +0.344 |
| 4 | 0.9838 | 0.9739 | 0.9898 | 0.0159 | 0.9816 | +0.513 |
| 5 | 0.9540 | 0.9444 | 0.9898 | 0.0453 | 0.9780 | +0.260 |
| 6 | 0.9981 | 0.9881 | 0.9900 | 0.0018 | 0.9885 | +0.772 |
| 7 | 0.9988 | 0.9888 | 0.9900 | 0.0012 | 0.9890 | +0.812 |
| 8 | 0.9991 | 0.9891 | 0.9900 | 0.0008 | 0.9893 | +0.852 |
| 9 | 0.9990 | 0.9891 | 0.9900 | 0.0009 | 0.9892 | +0.863 |
| 10 | 0.9988 | 0.9888 | 0.9900 | 0.0012 | 0.9889 | +0.895 |
| 11 | 0.9971 | 0.9871 | 0.9900 | 0.0028 | 0.9874 | +0.909 |
| 12 | 0.9985 | 0.9885 | 0.9900 | 0.0015 | 0.9886 | +0.948 |
| 13 | 0.9987 | 0.9887 | 0.9900 | 0.0012 | 0.9888 | +0.968 |
| 14 | 0.9990 | 0.9890 | 0.9900 | 0.0010 | 0.9890 | +0.945 |
| 15 | 0.9989 | 0.9889 | 0.9900 | 0.0011 | 0.9889 | +0.963 |
| 16 | 0.9988 | 0.9888 | 0.9900 | 0.0012 | 0.9888 | +0.947 |
| 17 | 0.9982 | 0.9882 | 0.9900 | 0.0018 | 0.9883 | +0.953 |
| 18 | 0.9983 | 0.9883 | 0.9900 | 0.0017 | 0.9884 | +0.934 |
| 19 | 0.9976 | 0.9877 | 0.9900 | 0.0023 | 0.9879 | +0.888 |
| 20 | 0.9986 | 0.9886 | 0.9900 | 0.0014 | 0.9888 | +0.851 |
| 21 | 0.9970 | 0.9870 | 0.9900 | 0.0029 | 0.9877 | +0.778 |
| 22 | 0.9947 | 0.9848 | 0.9899 | 0.0052 | 0.9861 | +0.737 |
| 23 | 0.9958 | 0.9859 | 0.9900 | 0.0041 | 0.9870 | +0.710 |
| 24 | 0.9950 | 0.9851 | 0.9899 | 0.0048 | 0.9865 | +0.703 |
| 25 | 0.9964 | 0.9865 | 0.9899 | 0.0034 | 0.9872 | +0.798 |
| **26** | 0.9722 | 0.9625 | 0.9896 | 0.0272 | 0.9751 | **+0.535** |
| **27** | 0.9510 | 0.9415 | 0.9894 | 0.0479 | 0.9682 | **+0.444** |

`skill = (uninformative − fit_actw) / (uninformative − floor)`; 1.0 = perfect, 0 = no better than an
unfitted subspace of the same rank and cost.

> **Spearman ρ(available range, measured skill) = −0.836 over 28 layers (p = 3.1 × 10⁻⁸).**
>
> **The predictor's apparent skill is highest exactly where there is no room to measure it.** On the seven
> layers where the range exceeds 0.01 (layers 1, 2, 3, 4, 5, 26, 27) skill is **0.25–0.54**. On the
> saturated layers it reads 0.7–0.97 — but a skill of 0.97 across a range of 0.001 moves `S_frac` from
> 0.9900 to 0.9895 and is worth nothing.

**Answer to the question as posed: on the ~21 layers where `fit` and `plain` sit within 0.001 of each other,
that 0.001 is not a gap — it is the full width of the measurable interval, and both arms are pinned to its
edge.** Those layers should be read as *no information*, not as *a small effect*.
**On layers 1 and 2 the separation is real and large** (fit 0.829/0.849 vs plain 0.947/0.948 vs C2
0.906/0.936 vs C2b/C3 ≈ 0.989), against a range of 0.63/0.51 — there the instrument genuinely discriminates.

### 2.3 The two layers with dynamic range, full rank sweep

`S_frac` at recall 0.99.

**Layer 1** — `a` = 0.3579, floor 0.3543, uninformative 0.990:

| arm | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 |
|---|---|---|---|---|---|---|
| `fit_actw` | 0.8868 | 0.8729 | 0.8551 | **0.8293** | 0.7894 | **0.7250** |
| `fit_plain` | 0.9714 | 0.9688 | 0.9612 | 0.9473 | 0.9265 | 0.8902 |
| `C2_random_proj` | 0.9490 | 0.9437 | 0.9338 | 0.9059 | 0.8848 | 0.8415 |
| `C2b_random_out` | 0.9887 | 0.9885 | 0.9887 | 0.9887 | 0.9886 | 0.9875 |
| `C3_wrong_layer_actw` | 0.9888 | 0.9890 | 0.9890 | 0.9889 | 0.9884 | 0.9880 |

**Layer 2** — `a` = 0.4796, floor 0.4749, uninformative 0.990:

| arm | r=8 | r=16 | r=32 | r=64 | r=128 | r=256 |
|---|---|---|---|---|---|---|
| `fit_actw` | 0.9107 | 0.8973 | 0.8775 | **0.8491** | 0.8111 | **0.7576** |
| `fit_plain` | 0.9817 | 0.9795 | 0.9674 | 0.9477 | 0.9272 | 0.8942 |
| `C2_random_proj` | 0.9771 | 0.9663 | 0.9454 | 0.9355 | 0.9011 | 0.8663 |
| `C2b_random_out` | 0.9894 | 0.9888 | 0.9886 | 0.9886 | 0.9880 | 0.9888 |
| `C3_wrong_layer_actw` | 0.9891 | 0.9893 | 0.9897 | 0.9895 | 0.9896 | 0.9891 |

Three findings here, all registered:

1. **Activation weighting earns its complexity, clearly, where anything can be measured.** `fit_actw` beats
   `fit_plain` by 0.10–0.12 and beats the registered C2 floor by 0.08–0.12 at every rank.
2. **`fit_plain` is WORSE than the C2 random-projection floor at every rank on both layers.** Plain SVD of
   `W_g` is below the brief's own null. Per §3.3 that is a real negative on the plain arm.
3. **`fit_actw` never approaches the floor.** At `r = 256` on layer 1 it needs `S_frac` = 0.725 against a
   floor of 0.354 — still keeping **twice** the neurons a perfect predictor would, having spent a rank large
   enough to cost 4% of the FFN by itself. The curve is flattening, not converging.

**Per-token recall** (the worst token, not the mean) at the `r=64`, recall-0.99 operating point: layer 1
min 0.9479 / p01 0.9674; layer 2 min 0.9503 / p01 0.9675; layer 15 min 0.9783; layer 27 min 0.9300.
Micro-recall 0.99 conceals tokens losing 5–7% of their actives.

### 2.4 Mean over all 28 layers, `r = 64`, recall 0.99

| arm | mean `S_frac` | `FFN_active` achieved width (n=3) | at target width (n=3) | at target width (n=2) |
|---|---|---|---|---|
| `fit_actw` | 0.9754 | 0.9917 | **0.9788** | 0.6536 |
| `fit_plain` | 0.9845 | 1.0008 | 0.9879 | 0.6597 |
| `C2_random_proj` | 0.9826 | 0.9989 | 0.9860 | 0.6584 |
| `C2b_random_out` | 0.9898 | 1.0061 | 0.9932 | 0.6632 |
| `C3_wrong_layer_actw` | 0.9899 | 1.0062 | 0.9932 | 0.6633 |

At recall **0.999**, `r=64`: `fit_actw` 0.9930, `fit_plain` 0.9975, `C2` 0.9967, `C2b`/`C3` 0.9990.

---

## 3. `eff_rank_actw` vs `eff_rank_plain` — and `H`'s own spectrum beside them

**Definitions, both the same functional.** `eff_rank` = participation ratio of the singular values,
`PR(σ) = (Σσᵢ²)² / Σσᵢ⁴`. It is a *top-heavy* measure: one dominant direction drives it to ≈ 1 regardless of
what the tail does.

- `eff_rank_plain` = PR of `σ(W_g)` — **`W_g` alone**.
- `eff_rank_actw` = PR of `σ(W_g H^{1/2})` — **`W_g` seen through the activation metric**.
- `eff_rank(H^{1/2})` = PR of `σ(H^{1/2})` = PR of `√eig(H)` — **`H` alone**. *Computed post-hoc from the
  cached `H_fit_L*.npy` with `numpy.linalg.eigvalsh`; script in the manifest (§11).*

**The numbers side by side. No prose resolves this; the columns do.**

| layer | `eff_rank_actw` | `eff_rank_plain` | **`eff_rank(H^{1/2})`** | `eig₁/eig₂` of `H` | `eig₁/tr(H)` | `H` rank for 90% energy | `W_g` rank for 99% energy |
|---|---|---|---|---|---|---|---|
| 0 | 3.49 | 734.6 | 20.12 | 9.2 | 0.2161 | 416 | 1288 |
| 1 | 1.11 | 884.6 | 1.97 | 23.1 | **0.7119** | 174 | 1471 |
| 2 | 1.15 | 879.3 | 2.90 | 8.9 | 0.5820 | 276 | 1466 |
| 3 | 1.22 | 926.9 | 3.89 | 5.6 | 0.4961 | 404 | 1468 |
| 4 | 1.35 | 969.7 | 5.74 | 4.8 | 0.4057 | 578 | 1469 |
| 5 | 1.22 | 797.4 | 5.01 | 4.5 | 0.4323 | 531 | 1463 |
| 6 | 2.13 | 912.8 | 17.24 | 3.9 | 0.2291 | 777 | 1469 |
| 7 | 2.51 | 958.4 | 20.36 | 4.8 | 0.2114 | 768 | 1465 |
| 8 | 2.31 | 949.8 | 19.89 | 4.1 | 0.2123 | 787 | 1456 |
| 9 | 2.70 | 970.9 | 22.85 | 4.0 | 0.1975 | 795 | 1464 |
| 10 | 2.55 | 899.7 | 20.56 | 3.6 | 0.2068 | 781 | 1459 |
| 11 | 2.02 | 837.4 | 15.30 | 3.3 | 0.2397 | 716 | 1458 |
| 12 | 2.37 | 912.8 | 18.21 | 4.1 | 0.2217 | 743 | 1463 |
| 13 | 2.68 | 932.5 | 21.86 | 4.2 | 0.2020 | 767 | 1461 |
| 14 | 2.66 | 923.4 | 24.25 | 3.9 | 0.1904 | 802 | 1451 |
| 15 | 2.69 | 890.4 | 18.18 | 3.1 | 0.2162 | 723 | 1459 |
| 16 | 2.77 | 934.8 | 17.09 | 3.4 | 0.2247 | 706 | 1454 |
| 17 | 2.73 | 936.2 | 14.45 | 3.4 | 0.2444 | 656 | 1454 |
| 18 | 2.97 | 961.2 | 15.70 | 3.5 | 0.2343 | 658 | 1461 |
| 19 | 2.54 | 943.0 | 15.73 | 3.3 | 0.2332 | 661 | 1453 |
| 20 | 2.76 | 970.9 | 20.39 | 2.6 | 0.1993 | 682 | 1446 |
| 21 | 2.16 | 967.5 | 15.44 | 2.4 | 0.2298 | 641 | 1449 |
| 22 | 2.06 | 999.5 | 13.35 | 2.9 | 0.2536 | 589 | 1450 |
| 23 | 2.16 | 941.7 | 13.94 | 3.2 | 0.2506 | 598 | 1442 |
| 24 | 2.47 | 999.2 | 14.81 | 3.6 | 0.2451 | 606 | 1450 |
| 25 | 3.39 | 1044.2 | 17.05 | 3.1 | 0.2245 | 610 | 1455 |
| 26 | 3.94 | 918.9 | 17.22 | 2.5 | 0.2168 | 585 | 1461 |
| 27 | 1.90 | 715.7 | 6.92 | 4.1 | 0.3637 | 401 | 1464 |

**What the columns say, in the order that decides it:**

1. **`H` is itself dominated by very few directions.** Its top eigenvalue carries **19.0%–71.2%** of total
   variance, and `eig₁/eig₂` is 2.4–23.1. This is the documented massive-activation structure of this donor
   family, and it is present in every layer.
2. **`eff_rank_actw` (1.1–3.9) tracks `eff_rank(H^{1/2})` (2.0–24.3), not `eff_rank_plain`.** It sits *below*
   `eff_rank(H^{1/2})` everywhere and moves with it — smallest at layers 1–5 where `eig₁/tr(H)` is largest
   (0.41–0.71), largest at layers 6–26 where `eig₁/tr(H)` falls to ≈ 0.20.
3. **`W_g` itself is essentially full rank.** `eff_rank_plain` = 716–1044, and **`W_g` needs a mean of 1452
   of its 1536 singular values to reach 99% of its energy**. Condition number is a benign 16–66 — the
   spectrum is *flat*, not decaying.
4. **`r = 64` captures a mean of 0.1428 of `W_g`'s plain energy** (0.117–0.179) — 14%, at `r = 64` of 1536.

> **Verdict on the alternative explanation: it is the correct one.** The tiny `eff_rank_actw` is inherited
> from `H`'s own spectrum and **says nothing about `W_g` being compressible**. The activation-weighted
> objective `‖(W_g − AB)X‖_F` is easy to make small because `X` lives in a few directions — but making that
> norm small is not the same as ranking `g` correctly on the directions that decide the active set.
> **`eff_rank_actw ≈ 2` must not be quoted as evidence that a rank-2 gate predictor exists.**

The `actw` 90%-energy rank makes the same point without the top-heaviness: it is **1** on layers 1–5 (where
`H` is most concentrated) but jumps to **118–301** on layers 6–27, and the `actw` **99%**-energy rank is
**342–1148**. The recall measurement in §2 agrees with the 99% column, not with the PR column.

---

## 4. Amendment 1 — the IVF-PQ arm

**Parameters ACHIEVED:** `nlist = 95` (rule `round(sqrt(F))`, `F = 8960`), `m = 48` subquantizers of
`dsub = 32`, `2⁸ = 256` codewords each, `kmeans_iters = 25`, seed 20260904. Recall measured against the
**true** active set from the **exact** gate (A1.4 point 2), never against the index's own ranking.

| `nprobe` | reachable frac | **ceiling recall** (all reached kept) | **recall ACTUALLY achieved at the 0.99 row** | `S_frac` | `FFN_active` MACs | `FFN_active` BYTES | target width MACs / BYTES |
|---|---|---|---|---|---|---|---|
| 1 | 0.0007 | 0.0007 | **0.0007** | 0.0007 | 0.0138 | 0.1056 | 0.0056 / 0.0402 |
| 2 | 0.0015 | 0.0015 | **0.0015** | 0.0015 | 0.0147 | 0.1068 | 0.0064 / 0.0411 |
| 4 | 0.0033 | 0.0034 | **0.0034** | 0.0033 | 0.0168 | 0.1095 | 0.0082 / 0.0430 |
| 8 | 0.0071 | 0.0074 | **0.0074** | 0.0071 | 0.0210 | 0.1150 | 0.0121 / 0.0470 |
| 16 | 0.0527 | 0.0540 | **0.0540** | 0.0527 | 0.0675 | 0.1642 | 0.0579 / 0.0930 |
| 32 | 0.2808 | 0.2865 | **0.2865** | 0.2808 | 0.2974 | 0.3993 | 0.2861 / 0.3218 |
| 64 | 0.9031 | 0.9068 | **0.9063** | 0.9024 | 0.9225 | 1.0350 | 0.9081 / 0.9449 |
| **95 = nlist** | 1.0000 | 1.0000 | **0.9900** | 0.9838 | 1.0073 | 1.1300 | 0.9899 / 1.0278 |

> **The cheap-looking rows are cheap because they retrieve almost nothing.** `FFN_active` = 0.0138 at
> `nprobe = 1` is attached to a recall of **0.0007**. **The 0.99 recall target is met at exactly one setting:
> `nprobe = 95 = nlist` — a full scan of every list.** At the recall the brief requires, **the index prunes
> nothing at all**, and its `FFN_active` is 0.99 (MACs) / 1.03 (bytes) at target width — worse than dense.

**Three measured reasons, all ACHIEVED:**

1. **The partition collapsed.** Mean list size is 94.3 (= F/nlist), but the **mean largest list across layers
   is 4494 keys** (max 8842 of 8960 — one list holding 99% of the keys) and the **mean smallest list is
   1.0**. `W_g`'s rows do not cluster.
2. **The coarse quantizer is the wrong router.** Euclidean k-means centroids are used to rank by *inner
   product* — a known MIPS/L2 mismatch. The evidence is that at `nprobe = 1` reachability is 0.0007, i.e.
   ≈ 6 of 8960 keys: tokens are routed to the *tiny* lists, not the giant one.
3. **PQ cannot represent the residuals.** **Reconstruction relative error ACHIEVED = 0.8257 mean
   (0.8162–0.8299)** at 48 × 8 bits. The scores the index ranks by carry ≈ 83% error, so even a reached key
   is ranked almost arbitrarily.

**Index bytes, counted in, not footnoted (A1.4 point 1):** resident **1.5 MB at achieved width (0.09 × the
16 MB L3)**, **8.3 MB at target width (0.50 × L3)**. The structure *is* cache-resident by the banked rule —
**but that rule says an auxiliary structure is free only if cache-resident AND bandwidth-light, and this one
buys no pruning at the required recall, so residency is irrelevant.**

**Build cost (A1.4 point 3):** 19.5 s/layer mean (15–26 s), 28 layers, offline and once. *Run metadata only.*

---

## 5. Cost accounting, with the predictor charged

### 5.1 The byte convention, stated once, applied to every arm

The instrument's `FFN_active` is a fraction of the **dense FFN's weight bytes for one token**:

```
FFN_active = [ predictor_cost + n · |S|/F ] / 3       n = 3 PRIMARY
```

`n = 3` (registered in §1 and primary here): the predictor picks `S`, then the **exact** gate is recomputed
on `S`, so gate, up and down are each read on `S`. `n = 2` (reported beside it): `ĝ` is *used* as the gate and
never corrected — a **different construction** that inherits the predictor's error into the activation
values. It is the accounting behind §4's `(0.01 + 2(1−s))/3` identity.

**Two byte conventions, both reported for every arm:**

- **Convention A — uniform bytes/element.** Every stored element (`W_g`, `W_u`, `W_d`, and the predictor
  factors `A`, `B`) costs the same. `FFN_active = pred/3 + s` where `pred = r(D+F)/(D·F)`. **This is what
  `f1_analyse.json` reports.** It is dtype-free but flatters the predictor.
- **Convention B — engine-realistic.** `W_g/W_u/W_d` ternary-packed at **0.25 B/element** (this engine's
  target), predictor factors `A`, `B` at **fp16 = 2 B/element** (an SVD factor is not ternary), index
  codebooks/centroids fp16, PQ codes 1 B. Then `FFN_active = (8/3)·pred + s` — **the predictor term is 8×
  its convention-A value.** This is the same convention `f1_ann.json`'s `bytes_frac` uses, so the ANN BYTES
  columns above are convention B and the MACs columns are a compute-side view.

### 5.2 The predictor charged — `fit_actw`, target width `D = 8192, F = 28672`, recall 0.99

| `r` | mean `S_frac` | `pred = r(D+F)/(D·F)` | **`FFN_active` conv-A** | **`FFN_active` conv-B** |
|---|---|---|---|---|
| 8 | 0.9805 | 0.00126 | 0.9809 | 0.9838 |
| 16 | 0.9793 | 0.00251 | 0.9801 | 0.9860 |
| 32 | 0.9777 | 0.00502 | 0.9793 | 0.9911 |
| **64** | 0.9754 | 0.01004 | **0.9788** | **1.0022** |
| 128 | 0.9723 | 0.02009 | 0.9790 | 1.0258 |
| 256 | 0.9675 | 0.04018 | 0.9809 | **1.0746** |

**Reference points, all measured or defined:**

| quantity | value |
|---|---|
| brief's target | **0.0200** |
| dense-gate structural floor the brief set out to break | **0.3333** |
| **best `FFN_active` reached anywhere** (conv-A, target width, any arm/rank/layer, recall ≥ 0.99) | **0.7384** — `fit_actw`, `r=256`, **layer 1**, `S_frac` 0.7250 |
| mean over 28 layers, `fit_actw` `r=64` (conv-A) | **0.9788** |
| **`FFN_active` of a FREE, PERFECT predictor** = mean `a` | **0.9506** |
| dense-gate baseline `(1+3a)/3` at measured `a` | **1.2840** |

> **Under convention B the predictor term overtakes its own benefit past `r = 64`:** going from `r=64` to
> `r=256` cuts `S_frac` by 0.0079 while adding 0.0804 of predictor bytes. **The rank sweep has an interior
> optimum around `r = 16–32`, and it sits at `FFN_active ≈ 0.98`.**
>
> Note the last two rows: **`(1+3a)/3 = 1.284 > 1`.** At `a = 0.95` a gate-predicted sparse path costs *more*
> than simply running the FFN dense. **The construction is not merely insufficient here; at the measured
> activation fraction it is a pessimisation.**

### 5.3 The brief's own cost table is internally inconsistent (found by `selfcheck`; brief NOT edited)

Recorded verbatim as `BRIEF_TABLE_DISCREPANCY` in every JSON:

| row | brief prints | `n=3` from §1's own formula | `n=2` |
|---|---|---|---|
| a = 0.10 | 0.103 | 0.10335 | 0.07002 |
| a = 0.05 | 0.053 | 0.05335 | 0.03668 |
| **a = 0.02** | **0.0167** | **0.02335** | 0.01668 |

Rows 1–2 are `n=3`; **row 3 is `n=2`**. Under the registered `n=3` formula the 2% target row is **2.33%, not
1.67% — 40% higher, at exactly the operating point the brief cares about.** Knock-on: §4's "`s = 0.97`
reaches 2%" is 2.33% under `n=3`; a true 0.02 needs `s = 0.975` under `n=2` or **`s = 0.9833` under `n=3`**.

---

## 6. Controls — every pre-registered control, with its verdict

| control | what it tests | ACHIEVED | verdict |
|---|---|---|---|
| **`selfcheck`** algebra | closed form, Eckart–Young, orthonormality | E–Y rel err 0.0 (k=8) / 7.9e−15 (k=32); orthonormality 2.0e−15 / 2.2e−15; `H^{1/2}` roundtrip 1.7e−15; **200/200 random factorisations worse** at both k | **PASS** |
| **`selfcheck`** eigh-shortcut | `eigh(MᵀM)` vs direct SVD | subspace alignment err 3.9e−09, σ rel err 1.0e−15 | **PASS** |
| **direct-SVD cross-check** (in `analyse`, layer 0, k=64) | the shortcut on real data | alignment err **1.62e−09** | **PASS** |
| **C1 full rank** (`r = D = 1536`) | the code path end to end | max \|ĝ−g\| **7.25e−05**, max rel dev **1.44e−06**; recall **1.000000** at every target; mean `S_frac` at recall 1.0 = **0.950618** vs true mean `a` = **0.950618** (identical) | **PASS — and it is a TAUTOLOGY.** `U_D` spans `colspace(W_g)` so `ĝ ≡ g` by construction. **It cannot fail on a correct implementation, so it tests the code path, not the science.** Labelled as such in the artefact (`IS_A_TAUTOLOGY`) |
| **C2 random projection** (registered floor: `B` random, `A` solved) | how much *any* `r`-dim read of `x` buys | mean `S_frac` 0.9826 (r=64). On layers 1–2: 0.906/0.936 — **beats `fit_plain`**, loses to `fit_actw` | **REAL CONTROL, FIRED.** It discriminates: `fit_actw` beats it, `fit_plain` does not |
| **C2b random output subspace** (Builder addition, fresh per layer×rank) | matched null — same family and cost, unfitted subspace | mean `S_frac` **0.9898** (r=64); **0.9875–0.9894 on layers 1–2 across all six ranks, flat in `r`** | **REAL CONTROL, FIRED.** Sits at the uninformative value ρ = 0.99 and **does not improve with rank** — confirming the fitted arms' gains are gains of *fit*, not of *family* |
| **C3 wrong-layer** (offset +14, both weightings) | real spectrum, wrong correspondence | mean `S_frac` **0.9899** (r=64); 0.988–0.990 on layers 1–2, flat in `r` | **REAL CONTROL, FIRED.** Indistinguishable from C2b ⇒ a `W_g` subspace carries **no transferable** information about a different layer's active set |
| **C4 planted positive** (synthetic exactly-rank-32 `W_g`, real `X`, real `H`, active fraction matched per layer) | **the instrument must recover at the planted rank and NOT before** | planted rank ACHIEVED **32/32 on all 28 layers**. rel dev: **r=8: 0.37–0.90 · r=16: 0.22–0.82 · r=32: 2.7e−07–4.7e−07 · r=64: same**. recall at τ=θ: **r=16: 0.8495–0.9999 · r=32: 1.0000 on every layer** | **FIRES, 28/28 layers.** Exact at exactly the planted rank, broken one rank below. **This is the control that makes the nulls meaningful: the instrument demonstrably detects a genuinely low-rank gate. It did not detect one in the donor because the donor does not have one.** |

**Shuffled-`W_g` nulls were not used**, per the brief's explicit instruction (two shuffle nulls on this
programme were not null, and one beat the real arm).

---

## 7. The brief's pre-registered verdicts (§3.3), adjudicated

| pre-registered outcome | measured | verdict |
|---|---|---|
| recall ≥ 0.99 with `FFN_active ≤ 0.10` at `r ≤ 64`, beating C2 by a clear margin | best anywhere = **0.7384** (r=256, layer 1); mean at r=64 = **0.9788** | **NOT MET.** The floor is not broken |
| recall ≥ 0.99 only with `FFN_active > 0.25` → "partial — better than 33.3% but not transformative" | `FFN_active` is **0.98 mean**, and **> 0.3333 on every layer, arm and rank measured** | **Worse than this row.** Not even partial: it never beats the dense-gate floor it was built to beat |
| **C2 does as well as the fitted arms** → "the construction has shown nothing" | C2 **beats `fit_plain`** but **loses to `fit_actw`** by 0.08–0.12 where measurable | **SPLIT.** The activation-weighted arm is genuinely above the registered floor. **The plain-SVD arm is below it and is falsified.** But the margin buys nothing, because the ceiling itself is at 0.95 |

---

## 8. What contradicts the brief

1. **The brief's premise — not its algebra — is what fails.** §1's identity, the closed form and the
   asymmetry argument are all correct and all verified. **`a` is 0.95, not 0.02–0.10.** Everything downstream
   of "the model's actual activation sparsity" is priced against a sparsity this donor does not have at a 1%
   per-layer error budget.
2. **"The floor disappears" does not survive contact.** The brief's claim is that `FFN_active` becomes bounded
   by `a` rather than by 1/3. **That is true — and it is the wrong direction here, because the measured
   bound `a = 0.9506` is nearly three times *worse* than the 0.3333 floor it replaces.** At the measured `a`
   the dense-gate baseline `(1+3a)/3 = 1.284`, so the sparse path is a pessimisation.
3. **`eff_rank_actw ≈ 2` is an artefact of `H`, not a property of `W_g`.** Anyone reading the effective-rank
   column alone would conclude the gate is rank-2 compressible. `W_g` needs **1452 of 1536** singular values
   for 99% of its energy (§3).
4. **Amendment 1's arm is the weaker one, not the stronger one.** The brief judged it "if it works it is
   stronger than the arm the brief was written around". **At the required recall it provides zero pruning**
   (`nprobe` must equal `nlist`), its partition is degenerate (largest list mean 4494 of 8960, smallest 1),
   and its PQ reconstruction error is **0.826**. Amendment A1.5's own warning — that this is the construction
   its author would most like to be true — is the relevant one.
5. **`nlist ∝ √N` was banked for the recall tier and does not transfer to `W_g`'s rows.** The banked law was
   measured on a retrieval corpus; `W_g`'s rows are not a clusterable point set in that sense. **This is a
   second instance of a transplant failing** (cf. `ADAPTER_MEMO_01` §2.2g/§2.2h).
6. **`project_phase58_predictor` transfers more than the brief allowed.** The brief said its trained-gate
   failure "does not transfer" because F1 is untrained. **The failure mode is nevertheless the same, and it
   is not about training: the gate's active set is high-dimensional.** F1's contribution is that it now has a
   *closed-form, controlled* measurement of that, with a positive control that fires.
7. **The brief's §1 cost table is internally inconsistent** (§5.3) — its 2% row uses `n=2` while the rows
   above it use `n=3`. Under the registered `n=3` formula the target row is 2.33%, not 1.67%.

### 8.1 Independent corroboration from D0 — a different construction, the same donor, the same wall

**Not my measurement.** The following is read from another Builder's write-up,
`docs/research/donor_adaptation/probes/D0_COACTIVATION.md`, which is untracked in the same working tree and
which I did not produce or re-run. I flag it because it bears directly on how §1 should be read.

D0 attacks the FFN from the **opposite direction** — an MoE carve with a co-activation partition, not a
low-rank gate predictor — on the **same donor**. Crucially, **D0 does not derive its activation fraction from
a quality budget: it prescribes one**, at `p = 0.10` exactly (`k = round(0.10 × 8960) = 896`, exact per-token
top-k). At that prescribed density it reports:

- reconstruction `relerr` of **0.63–0.97** at layers 7, 14 and 21, with its own summary describing both the
  co-activation carve and its matched null as *"close to emitting zeros"*;
- at the byte budget that actually reaches the 2% target (`E=128, k=2`, 2.04% of FFN bytes including the
  router), an **oracle-routed** carve losing **59.5% (L27), 83.7% (L21), 94.7% (L14)**.

> **Two probes, two unrelated constructions, one conclusion: this donor's FFN does not survive a 10% active
> fraction, let alone 2%.** D0 assumed 10% and measured the damage; F1 asked what fraction a 1% damage budget
> permits and measured 95%. **These are the same finding approached from opposite ends, and they agree.**

That D0's premise (10% is a reasonable operating point) is exactly what F1's threshold rule contradicts is
worth the Adapter's attention: **both probes are costed against an activation fraction that this donor has
never been shown to tolerate.**

---

## 9. What the brief asked that this run does NOT answer

1. **§3.4 — end-to-end held-out BPB against the unmodified donor. NOT RUN.** No BPB number exists in any F1
   artefact. Everything here is the *screen*, and the brief itself says "a layer-wise recall number is not a
   result, it is a screen." No quality claim may be drawn from this document.
2. **The recall-sensitive probe demanded alongside BPB. NOT RUN.** This programme has established that a
   sparsified model can hold perplexity while losing retrieval; nothing here tests retrieval.
3. **Whether ε = 0.01 per layer is the right bar.** `a` is reported across a three-point ε ladder, but
   **no measurement here connects a per-layer Frobenius budget to end-to-end quality.** The whole reading of
   `a` hinges on this and it is unmeasured. **This is the single most load-bearing gap in the report.**
4. **Whether a *different donor* is sparser.** `a` is measured on **Qwen2.5-1.5B only**. The brief prices the
   construction at Llama-3-70B geometry (`D=8192, F=28672`); this run reports that width **arithmetically,
   through the cost formula, never by measuring a model of that width.** Activation sparsity is known to vary
   with scale and with activation function — **a 70B donor, or a ReLU/dReLU-family donor, could give a very
   different `a`, and F1 says nothing about that.** Every column labelled "target width" is geometry
   substituted into a formula, not a measurement.
5. **Prior art (§2).** The brief requires the Researcher to report on Deja Vu and on the closed-form /
   activation-weighted / no-training / recall-first combination **before any novelty claim**. **No prior-art
   search was performed in this run.** No novelty is claimed anywhere in this document.
6. **`nlist` was not swept.** A1.4 asks for `nlist` "if cheap". Only `nprobe` was swept, at the single
   `nlist = 95` from the banked √N rule. Given the degenerate partition, a different `nlist` — or a
   MIPS-appropriate quantizer, or normalised keys — might behave differently. **Untested.**
7. **Rank > 256 untested**, as are `τ` policies other than a global per-layer threshold (e.g. per-token
   top-k, or a per-token adaptive τ). `r = 256` is already cost-dominant under convention B, so this is a
   bounded gap — but it is a gap.
8. **Only the primary ε = 0.01 was carried into the recall arms.** The `a(ε)` ladder exists for the threshold
   rule, but **the arms and controls were measured at ε = 0.01 only.** Whether `fit_actw`'s margin over C2
   grows at ε = 0.05 — where layers 1, 2 and 26 are genuinely sparse — is **not measured**, and it is the
   most promising follow-up this run suggests.
9. **Peak RSS was not instrumented.** The allocation inventory's ~1.6 GB peak for `analyse`/`ann` is verified
   *by construction* (shapes and dtypes, §10) but **not by a measured high-water mark.**

---

## 10. Allocation inventory — verified, not copied

The instrument carries `_ALLOCATION_INVENTORY` and emits it into every JSON. Checked against what was
actually observed on disk and against the loaded array shapes:

| object | inventory claims | **observed** | check |
|---|---|---|---|
| `CACHE_DIR` total | 1.6e9 B | **1,585,457,664 B** (84 files) | ✔ payload 1,585,446,912 + 84 × 128 B `.npy` headers = exact |
| `H_fit` 28 × [D,D] fp64 | 528,482,304 | 28 × [1536,1536] float64 = 528,482,304 | ✔ shape+dtype read from the files |
| `X_theta` 28 × [2048,D] fp32 | 352,321,536 | 28 × [2048,1536] float32 = 352,321,536 | ✔ |
| `X_eval` 28 × [4096,D] fp32 | 704,643,072 | 28 × [4096,1536] float32 = 704,643,072 | ✔ |
| `analyse` subspace bank 2 × 28 × [F,256] fp32 | 513,802,240 | 2 × 28 × 8960 × 256 × 4 = 513,802,240 | ✔ arithmetic |
| `analyse` weights 3 × [F,D] fp32 | 165,150,720 | 3 × 8960 × 1536 × 4 = 165,150,720 | ✔ |
| `G`, `Ghat`, per arm [T_eval,F] fp32 | 146,800,640 each | 4096 × 8960 × 4 = 146,800,640 | ✔ |
| `true_mask` [T_eval,F] bool | 36,700,160 | 4096 × 8960 = 36,700,160 | ✔ |
| `analyse` / `ann` PEAK | 1.6e9 | **not instrumented** | ⚠ consistent with the parts, but no measured high-water mark |
| donor resident during `analyse`/`ann` | none | none — `WeightReader` reads one layer at a time from the pinned safetensors | ✔ by construction |

**The largest persistent allocation in `analyse` is the 0.51 GB subspace bank**, held for the whole stage
because C3 pairs layer `L` with `(L+14) mod 28` in the matching weighting.

---

## 11. Reproducibility manifest

**Donor (ACHIEVED, read off tensor headers, not config prose):**

| field | value |
|---|---|
| `repo_id` | `Qwen/Qwen2.5-1.5B` |
| `revision_pinned` | `8faed761d45a263340a0528343f099c05c9a4323` |
| `config_sha256` | `0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77` |
| `safetensors_bytes` | 3,087,467,144 |
| `n_layers` / `D` / `F` ACHIEVED | **28 / 1536 / 8960** |
| `stored_dtype_achieved` | **BF16** (upcast to fp32 for all arithmetic) |
| `hidden_act` | `silu` |

**Data slices (ACHIEVED, `n_rejected: 0` on both):**

| slice | part | n_seq × len | seed | tokens | ids_sha256 (head) | corpus_sha256 (head) |
|---|---|---|---|---|---|---|
| fit | `calib` | 16 × 1024 | 20260822 | **16384** | `4daef4bc5a08b815…` | `10d4d28102625f39…` |
| eval | **`heldout`** | 4 × 1024 | 20260828 | **4096** | `b300731f54b1ac9e…` | `f46b0310c15faec5…` |
| theta | first 2 fit seqs | 2 × 1024 | 20260822 | 2048 stored, **first 1024 used** | — | — |

**Pre-registered constants ACHIEVED:** ranks (8,16,32,64,128,256) · ε ladder (0.001, 0.01, 0.05), primary
0.01 · recall targets (0.90, 0.95, 0.99, 0.999, 1.0) · `H_ridge_rel` 1e−10 (**0 eigenvalues floored on the
selfcheck `H`**) · `C4_true_rank` 32 · `C3_layer_offset` 14 · `nlist` 95, `m` 48, `dsub` 32, `nbits` 8,
`kmeans_iters` 25 · seeds: fit 20260822, eval 20260828, C2 20260901, C2b 20260902, C4 20260903, ANN 20260904.

**Environment:** Python 3.12.10 · numpy 2.4.6 · torch 2.12.0+cpu (6 threads) · transformers 4.57.6 ·
Windows-11-10.0.26200 · AMD64 Family 23 Model 113 (Zen2, 12 threads).
Repo `research/donor-adaptation` @ `3f885c5`.

**Commands, in order (all four stages complete):**

```
python benchmarks/donor_adaptation/f1/f1_gate_predictor.py selfcheck
python benchmarks/donor_adaptation/f1/f1_gate_predictor.py capture
python benchmarks/donor_adaptation/f1/f1_gate_predictor.py analyse
python benchmarks/donor_adaptation/f1/f1_gate_predictor.py ann
python benchmarks/donor_adaptation/f1/f1_tables.py            # formatter only
```

**The `H`-spectrum columns in §3 are the only numbers here not produced by the instrument.** They were
computed post-hoc from the cached Gram matrices:

```python
import numpy as np, os
CACHE = r'D:\_THINGS\_scratch_f1'
def PR(s): e = np.asarray(s, float)**2; return float(e.sum()**2 / (e*e).sum())
for l in range(28):
    H = np.load(os.path.join(CACHE, f'H_fit_L{l:02d}.npy'))
    w = np.clip(np.linalg.eigvalsh(H)[::-1], 0, None)
    print(l, PR(np.sqrt(w)), w[0]/w[1], w[0]/w.sum(),
          int(np.searchsorted(np.cumsum(w)/w.sum(), 0.90) + 1))
```

**Run metadata — elapsed times. NOT a deliverable, and no performance conclusion is drawn from any of them.**
`selfcheck` 5.9 s · `capture` 379.8 s · `analyse` 3813.6 s · `ann` 1441.5 s (build 19.5 s/layer mean).
`capture` and `analyse` ran while the Owner's `llama-server.exe` was resident at ~18 GB; `ann` ran on a clear
machine. **Any bandwidth or throughput reading from these numbers would be invalid, and none is offered.**

---

## 12. Files left in the working tree (nothing committed, nothing pushed)

| path | size | stage |
|---|---|---|
| `benchmarks/donor_adaptation/f1/f1_gate_predictor.py` | 57,436 B | instrument (pre-existing) |
| `benchmarks/donor_adaptation/f1/f1_tables.py` | 12,149 B | formatter (pre-existing) |
| `benchmarks/donor_adaptation/f1/results/f1_selfcheck.json` | 9,390 B | `selfcheck` |
| `benchmarks/donor_adaptation/f1/results/f1_capture.json` | 6,362 B | `capture` |
| `benchmarks/donor_adaptation/f1/results/f1_capture.log` | 662 B | `capture` |
| `benchmarks/donor_adaptation/f1/results/f1_analyse.json` | 11,685,566 B | `analyse` |
| `benchmarks/donor_adaptation/f1/results/f1_analyse.log` | 5,119 B | `analyse` |
| `benchmarks/donor_adaptation/f1/results/f1_ann.json` | 2,219,038 B | `ann` |
| `benchmarks/donor_adaptation/f1/results/f1_ann.log` | 1,705 B | `ann` |
| `benchmarks/donor_adaptation/f1/results/f1_tables.md` | 244 lines | formatter |
| `docs/research/donor_adaptation/probes/F1_GATE_PREDICTOR.md` | this file | write-up |
| `D:\_THINGS\_scratch_f1\` (**outside the repo on purpose**) | 1,585,457,664 B (84 files) | `capture` cache |

**STOP.** Measurement complete; the verdict is the Adapter/Principal's.
