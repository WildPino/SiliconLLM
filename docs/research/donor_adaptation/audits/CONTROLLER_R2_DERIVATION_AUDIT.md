# Controller audit — Brief R2, the closed-form V-side solve

Adversarial review of `docs/research/donor_adaptation/briefs/BRIEF_R2_CLOSED_FORM_VALUE_SOLVE.md`.
The Controller does not implement, commit, or adjudicate direction. Findings carry a reproduction path.

---

## 2026-08-22 — R2 derivation audit (first pass)

### VERDICT: **BLOCK**

**Not on the algebra — §1.2 is exact and §1.3 is the true minimiser, both verified numerically. The block
is on §1.4, §3.2 and §3.3: the estimator as written fails its own C1 IDENTITY control by ~4 orders of
magnitude, the `<0.25` threshold is justified by a D4 number that D4's own results file contradicts
(wrong-layer recovery was **−1.472**, not ≈0.25), the primary metric is a monotone function of `T/D` and
of a layer set the Adapter chooses, and in synthetic replication the C5 shuffle null recovered **more**
than the real arm.**

Every one of those is fixable and the fix is given below. **The derivation is worth proceeding on. The
pre-registration as written is not.**

Scope: findings 1–12 are algebra and apparatus. Findings 13–20 are the design gate. Findings 21–26 are the
requested audit of `ADAPTER_MEMO_01` §2.2c.

---

## PART A — the derivation

### 1. §1.2, the commuting step — **CORRECT.** The brief does not die at line one.

Verified against source, not a diagram: `transformers` 4.57.6,
`.../transformers/models/llama/modeling_llama.py::LlamaAttention.forward` + `eager_attention_forward`,
and the same functions in `qwen2/modeling_qwen2.py`, `qwen3/modeling_qwen3.py`.

The real operator order is:

```
xn      = RMSNorm(x)                       # ONE norm, shared by q,k,v
q,k,v   = W_q xn , W_k xn , W_v xn
q,k     = apply_rotary_pos_emb(q, k, cos, sin)      # RoPE touches q,k ONLY -- v is untouched
(qwen3:  q,k = q_norm(q), k_norm(k) -- head-dim RMSNorm, again q,k ONLY)
A       = softmax(q k^T * scaling + causal_mask)    # scalars
out     = A v ; concat heads ; W_o
```

Point by point, against what the brief asked to be attacked:

- **RoPE — stays entirely inside `A[t,s]`. Confirmed at source.** `apply_rotary_pos_emb` is called on
  `(query_states, key_states)` and returns only those two. `value_states` never enters it. The
  position-dependence is fully absorbed into the scalar `A[t,s]`, which is exactly what the step requires.
  Same in Llama, Qwen2, Qwen3.
- **Qwen3's `q_norm`/`k_norm`** are head-dim RMSNorms applied to q and k only — same conclusion.
- **Causal mask** is an additive `-inf` pre-softmax; `A` stays a matrix of scalars. No effect on the step.
- **Softmax denominator** is a per-row scalar; `A[t,s]` is a scalar *after* normalisation. No effect.
- **Per-head scaling** (`self.scaling = head_dim**-0.5`) multiplies logits, i.e. lives inside `A`.
- **Sliding window** (Qwen2/Qwen3 `layer_types`) is another additive mask — no effect on the step.
- **Attention sinks** (GPT-OSS-class): an extra logit in the denominator carrying no value vector. It makes
  the row sums of `A` over real tokens **< 1** but leaves them scalars. The step survives; see finding 3
  for the one place this bites.

Numerical confirmation (`benchmarks/donor_adaptation/r2/r2_check.py`, C0): `max|W_v(X A^T) − (W_v X) A^T| = 5.3e-15` in fp64.

**This is exact linear algebra and it holds for the real architecture. Said plainly: §1.2 is right.**

### 2. §1.1 notation puts `x_t` at the wrong point in the block — FLAG, but fatal if implemented literally

The table defines `x_t` as *"hidden state entering the attention block at position `t`"*. In a pre-norm
Llama/Qwen block that is the **residual-stream input**, i.e. **pre-RMSNorm**. But
`v_s = W_v · RMSNorm(x_s)`, not `W_v · x_s`. RMSNorm is a per-token nonlinearity (`x / rms(x)` then a
learned diagonal gain), so `W_v x_s` is simply not the value vector.

This does **not** break §1.2 — RMSNorm acts per token *before* mixing, so it commutes fine — but only if
`X` is redefined:

> **`X := [xn_1 … xn_T]`, the POST-RMSNorm activations**, captured at the input to `v_proj`, not at the
> input to the decoder layer.

D4's `capture_organ_inputs` already hooks module inputs via a forward-pre-hook, so hooking `v_proj` gives
the right tensor for free. **The risk is purely that the Builder reads §1.1 literally and hooks the layer.**
Fix the notation before dispatch.

### 3. The bias term — Qwen2 has `bias=True` on q/k/v, and the step then requires row-sum-1

`Qwen2Attention.__init__`: `self.q_proj = nn.Linear(..., bias=True)`, likewise `k_proj`, `v_proj`
(`o_proj` is `bias=False`). Llama and Qwen3 use `config.attention_bias`, default `False`.

With a bias, `v_s = W_v xn_s + b_v` and

```
out_t = SUM_s A[t,s] (W_v xn_s + b_v) = W_v z_t + b_v * (SUM_s A[t,s])
```

The bias survives as `W_v z_t + b_v` **only if the row sums of `A` are 1**. Measured
(`r2_check2.py` §E, fp64):

| `A` | row-sum range | `max` error of the brief's form |
|---|---|---|
| `A_soft` | [1.000, 1.000] | 1.1e-14 |
| `A_lin`, row-normalised | [1.000, 1.000] | 2.2e-15 |
| `A_lin`, **unnormalised** | [77.7, 34328.4] | **4.0e+04** |

**Pre-register as a hard constraint:** every `φ` in §3.4 is used in its **row-normalised** form
(`A_lin[t,s] = φ(q_t)ᵀφ(k_s) / SUM_{s'<=t} φ(q_t)ᵀφ(k_{s'})`). The raw 2nd-order Taylor and raw FAVOR+ maps
are often written unnormalised; one unnormalised map on a Qwen2 donor silently corrupts the target by four
orders of magnitude. If a sink-bearing donor is ever used, the sink mass needs explicit handling since
`SUM_s A_soft[t,s] < 1` there.

### 4. §1.3 — the equation is **CORRECT**. It is the true ridge minimiser. Verified.

Derived independently. With `W` in `R^{d_v×D}`, `Z, Z*` in `R^{D×T}`:

```
d/dW [ ||W Z − W° Z*||_F^2 + lam*||W||_F^2 ] = 2(WZ − W°Z*)Z^T + 2*lam*W = 0
  =>  W (Z Z^T + lam*I) = W° Z* Z^T
  =>  W = (W° Z* Z^T)(Z Z^T + lam*I)^-1
```

That is the brief's equation, character for character. Orientation, transposes and the side of the inverse
are all right (`Z Zᵀ` is `D×D`; the inverse is on the **right**, correct for the tokens-as-columns
convention).

Numerical check (`r2_check.py`, C1), `D=17, T=64, d_v=6`, fp64, lam=0:

```
max| brief_closed_form − numpy.linalg.lstsq |  = 7.4e-13
err(brief) = 5.877790e+01   err(lstsq) = 5.877790e+01   err(donor) = 7.694919e+01
random perturbations of the solution all increase the error: True
```

**No BLOCK here. §1.3 is not wrong.**

### 5. §1.4 is wrong twice, and the second one is the D4 bug — **BLOCK**

**(a) The brief's transcription of D4 is the PRE-FIX equation.** §1.4 writes D4 as

```
D4:  W' = (W * H_cross)(H_reg)^-1   with H_cross = X_S X^T ,  H = X_S X_S^T
```

The code (`benchmarks/donor_adaptation/density/d4_reconstruction.py::reconstruct_row`, lines 171–199) does
**not** do that:

```python
H_reg = H + lam * torch.eye(D, dtype=torch.float64)
A   = H_reg.index_select(0, S).index_select(1, S)     # H_reg[S,S]
rhs = W_full @ H_reg.index_select(1, S)               # W @ H_reg[:,S]  -- SAME H_reg
```

The cross term is `H_reg[:,S] = (H + lam*I)[:,S]`, i.e. **regularised**, not `H[:,S]`. That fix is the one
the identity control caught, and the docstring says so: *"The earlier version regularised only the system
matrix `(H[S,S]+lambda*I)` while leaving the right-hand side on the unregularised `H[:,S]` … a real
distortion."* **§1.4 compares R2 to the bug, not to the code.**

**(b) D4 is not the special case `Z* = Z`.** Re-deriving D4 in R2's notation: D4 restricts input features to
a surviving set `S`, so `Z = R_Sᵀ X` (`|S|×T`) and `Z* = X` (`D×T`). Then `Z*Zᵀ = X X_Sᵀ = H[:,S]` and
`ZZᵀ = H[S,S]` — exactly D4's pair. So:

> **D4 is the special case `Z = R_S^T Z*` — the two "mixings" are related by a coordinate SELECTION, not
> the case `Z* = Z`. `Z* = Z` is D4's IDENTITY CONTROL (S = full), not D4 itself.**

Not pedantry. It is *why* the lambda convention transfers the way it does: in D4 both matrices are slices of
one object, so regularising once and slicing twice is natural. In R2 they are two genuinely different
objects, and "which one gets the lambda" becomes a real modelling choice the brief made silently — and made
the way that breaks its own control.

### 6. C1 IDENTITY, as pre-registered, **cannot fire** under the brief's estimator — **BLOCK**

Set `A_lin := A_soft`, so `Z = Z*`. The brief's form returns

```
W = W° (Z Z^T) (Z Z^T + lam*I)^-1   !=   W°
```

— shrunk toward zero by exactly the Tikhonov term. §3.3 pre-registers this arm as *"must return
`W_v^donor` to machine precision"* and *"if it does not fire, nothing downstream may be reported."*
**It will not fire.**

Measured (`r2_check.py` C2, `r2_check2.py` §A), fp64, `lam = 1e-4*trace(H)/N` exactly as the brief specifies:

| case | `max_abs_weight_deviation`, brief's form | D4-symmetric form |
|---|---|---|
| toy, well-conditioned (`cond=36`) | **1.03e-03** | 2.40e-14 |
| `Z` from `A_soft` (`cond=3.6e1`) | 7.58e-05 | — |
| `Z` from `A_lin` elu+1 (`cond=7.1e2`) | **6.57e-04** | — |
| `Z` from causal-uniform (`cond=9.2e2`) | **6.93e-04** | — |
| *(D4's actual post-fix identity control, for scale)* | — | **2.98e-08** |

**The brief's identity control lands 4–5 orders of magnitude above the one D4 passed, and it gets worse as
conditioning gets worse.** Since `Z = X A^T` is a heavy smoothing of `X`, `cond(ZZᵀ)` in this toy is already
20–400× `cond(XXᵀ)` (= 2.43); on a real donor it will be worse. Not a rounding artefact — the same failure
shape as D4's original bug, arriving by a different route.

### 7. **Corrected algebra.** Ridge-toward-donor — it is D4's convention and it restores the control

```
   min_W  || W Z − W° Z* ||_F^2  +  lam * || W − W° ||_F^2

       =>  W (Z Z^T + lam*I) = W° Z* Z^T + lam*W°

       =>  W = ( W° ( Z* Z^T + lam*I ) ) ( Z Z^T + lam*I )^-1        <-- USE THIS
```

Properties, all verified in fp64 (`r2_check.py` C3, `r2_check2.py`):

- At `Z* = Z` it returns `W°` **exactly**: `W°(H+lam I)(H+lam I)^-1 = W°`, deviation **2.40e-14**. C1 fires.
- Gradient of the stated objective at the solution: **1.38e-11**. It is the genuine minimiser of a *stated*
  objective, not a fudge — and "shrink toward the donor" is the correct prior for a conversion problem.
- It is *literally* what `reconstruct_row` computes, with `Z*Zᵀ + lam I` playing the role of `H_reg[:,S]`.
  So §1.4's "the solver transfers" claim becomes **true** — but only after this change.
- Fit quality is not sacrificed: `err(sym) = 5.877791e+01` vs `err(brief) = 5.877791e+01` vs
  `err(donor) = 7.694919e+01`. The two estimators differ only in *where they shrink to*.

If the Adapter prefers to keep ridge-to-zero, that is defensible — but then **§3.3 must be rewritten**: C1
becomes "run at lam=0 and require machine precision" (measured 1.71e-14, so it does fire there) plus a
separate lam>0 arm whose expected deviation is *derived in advance* and checked against that derivation.
What is not acceptable is the current combination of ridge-to-zero with a machine-precision identity gate.

### 8. "The solver transfers unchanged" is false as an engineering claim — FLAG

`reconstruct_row(W_full, H, S, lam)` takes **one** matrix and a support set. R2 needs **two** matrices and
no support set. There is no signature under which the existing function computes R2's solve. A new function
is required; what transfers is the *convention* (regularise once, use the same regularised object on both
sides), the Cholesky/lstsq fallback, and the rank diagnostic. State that honestly in the Builder brief so
nobody budgets it as a call-site change.

### 9. Non-symmetry of `H_cross = Z*Zᵀ` — **not a numerical problem.** But the rank diagnostic does not transfer — **BLOCK**

**Non-symmetry is fine, and I looked for this specific failure before concluding that.** In
`reconstruct_row` the cross term appears only as a right-multiplication forming `rhs`; it is never factored.
`torch.linalg.cholesky` is applied to `A = H_reg[S,S]`, which in R2 becomes `ZZᵀ + lam I` — symmetric PSD by
construction. `torch.linalg.eigvalsh` in `hessian_rank_ratio` is likewise applied to `ZZᵀ`, not to the cross
term. **Do not manufacture a concern here.**

**What does break is `theoretical_rank_ratio`.** It returns `min(T,D)/D` and is documented as an *"always
mathematically valid (never optimistic)"* stand-in, empirically validated against the exact rank **on `X`**
(0.9122 vs 0.9143 etc.). That validation does not transfer:

```
rank(Z) = rank(X A_lin^T) <= min( rank(X), rank(A_lin) )
```

and for a feature map of dimension `m`, `A_lin = φ(Q)ᵀφ(K)` is rank <= `m` before masking (`m = d_k = 128`
for elu+1; `m` = number of random features for FAVOR+). Causal masking and row normalisation raise the
*exact* rank back up, but the *effective* rank stays near `m`. `min(T,D)/D` will report `1.0` while the
usable rank is a fraction of `D` — the bound stays technically valid and becomes **operationally vacuous**,
disabling precisely the abort trigger (Failure Mode 1, `rank(H)/N < 0.5`) that the joint-solver adjudication
made mandatory.

> **Required:** run the exact `hessian_rank_ratio` (eigvalsh) on `ZZᵀ` for **every** feature map and **every**
> arm. At `D = 1536` that costs ~0.04 s — the cost argument that justified the stand-in in D4 (`D = 8960`,
> 7.8 s) does not apply here. Report the numerical rank ratio **and** a spectral-decay curve, not just the
> ratio.

### 10. `lambda` is not calibrated for `Z`, and is now a free knob on the primary metric — FLAG

`lam = 1e-4*trace(H)/N` was tuned against `H = XᵀX` for MLP activations. `ZZᵀ` has a very different spectrum
(measured: 20–400× worse conditioning even in a benign toy). `lam` therefore moves `err(SOLVED)` and hence
`recovery` directly, and nothing in §3 pins it. **Pre-register lam as either (a) fixed at the D4 value and
reported alongside the resulting C1 deviation, or (b) selected on a held-out split by a rule stated in
advance — never on the reported metric.** Otherwise it joins `T` and the layer set as a third lever that can
be turned until the gate clears.

### 11. GQA — the closed form **survives**, with one honest caveat. Here is the equation.

For a KV group `g` of `G` query heads sharing one `W_v^{kv}`, each with its own `A^h`:

```
   min_{W_g}  SUM_{h in g} || W_g Z_h  −  W_g° Z*_h ||_F^2

       =>  W_g ( SUM_h Z_h Z_h^T + lam*I ) = SUM_h W° Z*_h Z_h^T   ( + lam*W°  under the finding-7 convention )

       =>  W_g = ( SUM_h W° Z*_h Z_h^T + lam*W° ) ( SUM_h Z_h Z_h^T + lam*I )^-1
```

Still **one** `D×D` solve for the whole group. Verified against a stacked `lstsq` over the concatenated
system (`r2_check2.py` §D, `G=3`): `max diff = 8.3e-14`.

**The caveat, which the brief does not state.** The above minimises the *unweighted* per-head output error.
The quantity that actually matters is the block output error, which weights head `h` by `W_o^h`:

- **MHA (distinct `W_v^h` per head): the weighting is free.** `min_W ||B(WC − M)||_F^2` with `BᵀB`
  invertible reduces to `W CCᵀ = M Cᵀ` — the unweighted normal equation. Verified:
  `max|unweighted − W_o-weighted| = 2.6e-13`.
- **GQA (shared `W_v`): the weighting is NOT free.** Stationarity becomes
  `SUM_h (W_o^hᵀ W_o^h) W (Z_h Z_hᵀ) = SUM_h (W_o^hᵀ W_o^h) W° Z*_h Z_hᵀ`, a **generalised Sylvester
  equation** with `G` terms. No single-inverse solution exists; the exact solve is a Kronecker system of
  dimension `d_v·D` (= 196 608 at Qwen2-1.5B, a 196k×196k dense matrix — **intractable**). Verified
  (`r2_check2.py` §D): the single-inverse answer differs from the true weighted minimiser by
  `max|dW| = 3.46` and is 0.8% worse on the weighted objective in the toy.

> **So: yes, GQA is still linear and still closed-form — for the unweighted objective. Saying "closed-form"
> without saying "unweighted" is exactly the kind of unstated modelling choice this project keeps getting
> caught by. Put it in the brief.** The 0.8% is a random-`W_o` toy and is not a prediction; the real gap is
> unmeasured (deferred-checks list, item 4).

### 12. §1.5 — the §3a analogy is **pattern-matching.** The prediction is right for the wrong reason.

§3a's theorem is about the **support structure of the unknown**: for a ROW-structured mask, every surviving
row already has full support `S`, so the row-separable solve is *algebraically forced* to return `W` or `0`
regardless of `H`. It has **no free parameters left**.

R2 is not that. `W_v` is **fully unconstrained** — all `d_v × D` entries free. Nothing is forced. The error
is in the **regressor** `Z`, not in the unknown's support. That is an errors-in-variables / subspace-overlap
problem, a different object.

**The correct statement of R2's structural limit:**

```
   min_W || W Z − W° Z* ||_F  =  || W° Z* ( I − P_Z ) ||_F ,
        P_Z = orthogonal projector onto rowspace(Z) in R^T
```

so the achievable fit is governed **entirely by how much of `colspace(A_soft Xᵀ)` lies inside
`colspace(A_lin Xᵀ)`** — i.e. by the principal angles between two `<=D`-dimensional subspaces of `R^T`. Not
"no freedom", but "freedom pointing the wrong way".

**This is constructive, and it is the single most useful thing in this audit.** That overlap is computable
**without solving anything, without touching `W_v`, and at a fraction of the cost of the full design**: form
`Z` and `Z*` for one layer, whiten by `ZZᵀ`, take the SVD, read the canonical correlations.

> **Recommendation: run the principal-angle pre-screen FIRST, as its own gate, before the §3 apparatus is
> built at all.** If the canonical correlations are low across the feature-map sweep, R2's answer is already
> known and the arms, controls and splits are unnecessary. If they are high, the same quantity gives a
> **derived upper bound** on `recovery` to check the measured value against — which is exactly the
> "instrument fires on a known positive" standard the brief's own §5 invokes.

§1.5's *prediction* ("degenerates toward a mean predictor if `Z` and `Z*` are weakly related") is correct.
Its *derivation* from §3a is not. Restate it from the projector identity above.

---

## PART B — the pre-registered design (§3)

### 13. The recovery fraction: the denominator is `err(DO-NOTHING)`, and it degenerates in a predictable place

`err = ||Yhat − Y*_donor||_F / ||Y*_donor||_F`, and the CEILING arm **is** the donor, so `err(CEILING) = 0`
identically. Therefore

```
recovery = ( err(DN) − err(SOLVED) ) / err(DN)  =  1 − err(SOLVED)/err(DN)
```

The denominator blows up not when "DN ≈ CEILING" in some abstract sense, but when **`err(DO-NOTHING) → 0`,
i.e. where the linearisation already works.** Two places where that is likely on a real donor:

- **Heads whose `A_soft` is near-uniform / diffuse** — common in early layers. A row-normalised `A_lin` is
  also near-uniform there, so `err(DN)` is small and the ratio is `0/0`-unstable.
- **Sink-dominated heads** (a large fraction of Llama/Qwen heads put most mass on token 0). `A_soft ≈ e_1 1ᵀ`
  is near rank-1; `err(DN)` can be very small or very large depending on whether `φ` happens to reproduce the
  sink, and the ratio is uninformative in both directions.

> **Required:** report `err(DN)` and `err(SOLVED)` as raw numbers **per head and per layer**, never only the
> ratio; and pre-register a floor (e.g. exclude and separately list any head with `err(DN) < 0.05`) rather
> than letting a near-zero denominator create or destroy the headline.

### 14. `recovery >= 0` is guaranteed in-sample — "SOLVED beats DO-NOTHING" is not a finding

`W = W°` is a **feasible point** of the SOLVED optimisation. So `err(SOLVED) <= err(DO-NOTHING)` on the fit
set **by construction** (up to the lam term). The comparison the brief calls *"the comparison that decides
everything"* cannot come out negative in-sample. Only its **magnitude**, and only on a **held-out** split,
carries information.

§3.1 says `err` is computed "on held-out calibration data" — good — but it does not say the solve is fitted
on a disjoint split, and D4's split machinery (`build_calib.py`, the `slice_calib_*` / `slice_heldout_*`
files) exists and must be reused explicitly. **Pre-register: solve on `slice_calib`, evaluate on
`slice_heldout`, report both, and report the in-sample/held-out gap as a number.** A large gap is the
overfitting signature, and finding 15 shows it will be large.

### 15. §3.2 is satisfiable by choosing `T` — the same defect the last gate had. **BLOCK**

`recovery` is a monotone function of `T/D`, and `T` is under the experimenter's control. Measured
(`r2_check2.py` §C, `D = 24`, real elu+1 `A_lin` against real softmax `A_soft`, in-sample):

| `T` | `err(DN)` | `err(SOLVED)` | **recovery** |
|---|---|---|---|
| 12 (`T < D`) | 0.778 | 0.201 | **0.741** |
| 24 (`T = D`) | 0.875 | 0.298 | **0.659** |
| 40 | 0.909 | 0.507 | **0.443** |
| 100 | 0.956 | 0.733 | **0.233** |
| 400 (`T ≈ 17D`) | 0.974 | 0.814 | **0.165** |

**The same method, the same donor, the same feature map spans 0.741 → 0.165 — across the entire
pre-registered verdict table — purely by choosing the calibration length.** At `T <= D` the regression
interpolates and `recovery` is a statement about nothing. §3.4 says only *"state which `T`"*; §3.2 does not
bind it.

> **Required in the pre-registration, before any number exists:**
> - `T` fixed in advance, with `T/D >= 24`. `ADAPTER_MEMO_01` §3 argues for SparseGPT's 24–64× calibration
>   margin and explicitly *"retires any thought of economising on calibration tokens"*; D4 ran at 1.83× and
>   the memo records that as a weakness. At `D = 1536` that is `T >= 36 864`.
> - `recovery` reported at **>=3 values of `T`** so the reader sees the curve, not one point on it.
> - Held-out evaluation mandatory (finding 14).

### 16. §3.2 is also satisfiable by choosing the LAYER SET. **BLOCK**

The threshold reads *"on a majority of converted layers"*. **Which layers get converted is a choice the
Adapter makes** — indeed the whole programme premise (`ADAPTER_MEMO_01` §2.2b) is that attention is retained
on a *minority* of layers, i.e. the converted set is selected. "A majority of a set I choose" is not a gate:
select the easiest 60% of layers and any threshold clears.

> **Required:** pre-register the layer set as **all layers**, report the full per-layer distribution
> (min / median / max), and key the verdict to a **pre-stated** statistic over **all** layers. If a subset
> is genuinely intended, the selection rule must be fixed and stated before measurement and must not
> reference `recovery`.

### 17. §3.2's `<0.25` justification is contradicted by D4's own results file. **BLOCK**

The brief: *"`< 0.25` is set at the level D4 measured for its wrong-layer arm."* From
`benchmarks/donor_adaptation/density/results/d4_reconstruction.json`:

```
/controls/hessian_ablation/real_H/recovery_point        =  0.483
/controls/hessian_ablation/identity_H/recovery_point    =  0.000
/controls/hessian_ablation/shuffled_H/recovery_point    = -1.595
/controls/hessian_ablation/wrong_layer_H/recovery_point = -1.472    <-- the wrong-layer arm
/organ_sweep_summary: recovery_mean 0.445, min 0.122, max 0.885
```

**D4's wrong-layer arm measured −1.472, not ≈0.25.** Wrong structure was *catastrophically worse than doing
nothing*, which is `ADAPTER_MEMO_01` §3's own headline (*"reconstructing with wrong statistics is worse than
deleting the organ entirely"*). Nothing in D4 supports 0.25 as a noise level. The nearest real D4 number to
0.25 is `recovery_min = 0.122` from the *organ sweep* — a genuine-signal arm, not a null.

This is a literature-fabrication-class error committed against our **own** results file, and it points the
favourable way (it sets the failure threshold conveniently low). **The threshold may still end up at 0.25,
but it must be justified by something that exists.** Given findings 17 and 18, the honest construction is:
*set the failure threshold at the measured value of the strongest null arm plus a stated margin* — value
determined after the nulls are measured, but with the **rule** fixed now.

### 18. C5 NULL — the shuffle is not merely weak; it measured **higher than the real arm**. **BLOCK**

Direct answer to the question posed: **yes, shuffling a `T×T` causal attention matrix is worse than D4's
column shuffle, and for a reason that inverts the control's sign.** Permuting a lower-triangular
row-normalised matrix **destroys causality**, producing a *denser* mixing with *higher* effective rank —
which gives `Z` **more** row space and therefore **more** fitting freedom, not less. A null arm that hands
the solver extra degrees of freedom is not a null; it is an upper bound wearing a null's label.

Measured (`r2_check2.py` §B, `D=24, T=400`, everything else held equal):

| arm | `err(DN)` | `err(SOLVED)` | **recovery** |
|---|---|---|---|
| **real `A_lin` (elu+1)** | 0.974 | 0.814 | **0.165** |
| **C5 as designed: shuffled `A_lin` columns** | 1.015 | 0.843 | **0.170** ← *higher than the real arm* |
| causal-uniform (`A[t,s] = 1/t`, zero content) | 1.000 | 0.942 | 0.059 |
| `A_lin` computed from an **unrelated sequence** | 0.999 | 0.946 | 0.053 |
| random causal row-stochastic | 1.004 | 0.957 | 0.047 |

Two things follow. First, **C5 as written would have been reported as "the null recovered as much as the
real arm" and read as a refutation of R2 — when in fact it is a broken control.** Second, there is a
**structural floor of ≈0.05** that *any* causal mixing achieves for free; any measured `recovery` must be
read against that floor, not against zero.

> **Proposed genuinely-null arm — C5': content-severed `A_lin`.** Compute `φ(q), φ(k)` from a **different,
> unrelated calibration sequence** of the same length at the same layer, then apply that `A_lin` to *this*
> sequence's `X`. This preserves causality, row normalisation, sparsity profile, row-entropy distribution
> and scale, and destroys **only** the position↔content correspondence — the one thing the hypothesis claims
> to exploit. Measured recovery 0.053, i.e. it behaves like a null.
>
> **Additional required arm — C6 STRUCTURE-FREE FLOOR: `A_lin := causal uniform`** (`A[t,s] = 1/t`; no `φ`,
> no `W_q`, no `W_k`, no content whatsoever). This is the "any causal smoother" floor. **If SOLVED does not
> beat C6 by a clear margin, the feature map contributed nothing and the whole `φ` sweep of §3.4 is
> measuring the causal mask.** This arm costs almost nothing and is currently missing.
>
> And per §3.3's own rule: **measure and report the residual structure of every null** (row-sum profile,
> row entropy, and the Frobenius overlap `||A_null (elementwise*) A_soft||_F / ||A_soft||_F`) as numbers,
> not as an assumption. That rule was written for exactly this, and C5 as designed violates it.

### 19. C1 IDENTITY — it is a tautology **of the solve**, and that is fine; the problem is it tests none of R2's new code

Direct answer to the question asked: under the corrected (finding-7) estimator, `Z* = Z` implies
`W = W°(H+lam I)(H+lam I)^-1 = W°` **algebraically**, independent of `Z`'s content. But that is precisely
what makes it a valid *implementation* test — D4's identity control was equally forced in theory and still
caught a real bug (`delta = +0.34921`, `max_abs_weight_deviation = 0.6545`, fixed to `5.94e-09` /
`2.98e-08`), because the bug broke the algebraic identity. **So: not a vacuous arm. Keep it. It is the
cheapest correctness gate available and it is the one that catches a transpose error.**

**What it does not test is everything R2 adds.** If `A_lin := A_soft` is injected by substituting the matrix,
then `Z` and `Z*` are produced by the *same* code from the *same* input, so any bug in `φ`, in the causal
masking, in the row normalisation, in per-head slicing, or in GQA group assignment **cancels identically**.
C1 is blind to the entire new code path.

> **Required companion — C0 PLANTED POSITIVE** (the "must fire on a known positive" law). Build the target
> from the *linear* side: draw a random `W_v^plant`, set the target to `Y* := W_v^plant Z` (not `W° Z*`),
> and require the solve to return `W_v^plant` to machine precision. This exercises the full
> `φ → A_lin → Z` path — the code C1 cannot see — and has a known, non-trivial answer. Without it, a null
> result from R2 is indistinguishable from a bug in the feature-map path.
>
> **Also add C2' TRANSPOSE TRAP:** run C0 with `Z` and `Z*` deliberately swapped and confirm the error is
> large. An orientation bug in a `D×T` / `T×D` pipeline is silent whenever the shapes happen to line up.

### 20. Reconstruction error cannot see the recall cap — **FLAG, and strategically the most important one**

`ATTENTION_LINEARISATION_PRIOR_ART.md` §4 / §9 establishes, mechanistically and from the papers' own
framing, that a fixed state of dimension `d'` can hold only `O(d')` exact associative bindings, that this is
**architectural, not a training-recipe artefact**, and — explicitly — that *"a closed-form per-layer solve
(§7, if one existed) would presumably fit the feature map optimally given the state dimension, but would not
remove the state-dimension ceiling itself."* **The Researcher has already written down that R2 cannot fix
the thing that most endangers the donor's usefulness.**

The §3.4 first pass measures **layer-wise Frobenius reconstruction error only**. That metric averages over
all `T` positions and all `d_v` dimensions. Retrieval behaviour lives in a **small number of spiky heads**
placing near-all mass on one distant token — exactly the heads a smooth kernel `φ` cannot reproduce, and
exactly the heads whose contribution to a mean Frobenius error is negligible. **A high `recovery` is
therefore fully compatible with retrieval being destroyed.** §3.4 defers the recall probe to stage 2 and
gates stage 2 on §3.2 — so the design can pass its gate on a metric blind to its own known failure mode, and
discover it only after promotion.

> **Minimum fix, cheap, in the first pass:** stratify every reported number by **`A_soft` row entropy** (a
> proxy for head spikiness), reporting `recovery` separately for the top and bottom entropy deciles. If
> `recovery` is high on diffuse heads and at the floor on spiky heads, that is the recall cap showing up in
> pass 1 for almost no extra compute, and it changes the verdict's meaning entirely.
>
> **And state in §3.2 that a passing `recovery` is NOT evidence about retrieval** — otherwise a 0.6 will be
> read as "the line is real" when the prior art already says the recall ceiling is untouched by it.

### 20b. Scale inventory — every `O(T)` and `O(T^2)` allocation, at a stated `T`

At Qwen2-1.5B geometry (`D = 1536`, 12 q-heads, `n_kv = 2`, `d_k = d_v = 128`, `L = 28`), stated at both
`T = 8192` and the `T >= 36 864` finding 15 requires, because the answer changes character:

| object | shape | `T = 8192` | `T = 36 864` | disposition |
|---|---|---|---|---|
| `X` post-norm, one layer | `D×T` fp32 | 50.3 MB | 226 MB | **stream per layer**; ×28 layers = 1.4 / 6.3 GB if held |
| `A_soft`, **per head** | `T×T` fp32 | **268 MB** | **5.44 GB** | **the dominant term** |
| `A_soft`, all 12 heads, one layer | | 3.2 GB | 65 GB | **must never be materialised** |
| `A_lin`, per head | `T×T` fp32 | 268 MB | 5.44 GB | should never be materialised at all (below) |
| `Z`, `Z*`, per head | `D×T` fp64 | 100.7 MB ×2 | 453 MB ×2 | stream in token blocks |
| `ZZᵀ`, `Z*Zᵀ` accumulators | `D×D` fp64 | **18.9 MB ×2** | 18.9 MB ×2 | **`O(1)` in `T` — the only persistent objects** |
| Cholesky workspace | `D×D` fp64 | 18.9 MB | 18.9 MB | |

**Peak under the streaming schedule I would require: ~130 MB** (`X` block + two `D×D` accumulators + one
head's block of `Z`/`Z*`). **Peak under the naive "materialise `A` per head" schedule: 0.4 GB at
`T = 8192`, 5.9 GB at `T = 36 864`** — and 65 GB if anyone hoists the head loop outside the allocation.

Compute, not just memory: forming `Z = X A_linᵀ` naively is `D·T^2` per head = 1.03e11 flops at `T = 8192`;
×12 heads ×28 layers ×6 arms ×4 feature maps ≈ **8.3e14 flops**, and 20× worse at `T = 36 864`. **The `T^2`
route makes the §3.4 sweep the expensive thing, which destroys §1.6's entire strategic argument ("we can
price a dozen `φ`").**

**Is `T×T` materialisation a validity problem or only a cost problem? Both — and the validity half flatters
the hypothesis.**

- **Same numbers, different accumulation.** For causal row-normalised linear attention the `T×T` product and
  the `O(1)`-state recurrence are an exact reassociation — mathematically identical. In exact arithmetic,
  cost only.
- **In finite precision they are not.** The shipped recurrence carries a running
  `S_t = SUM_{s<=t} φ(k_s) x_sᵀ` whose magnitude grows with `t`; cancellation drift in fp32 at long `T` is a
  documented failure mode of linear attention and is **absent** from the `T×T` form. **The materialised
  measurement is therefore systematically optimistic relative to what we would ship** — the direction this
  project's standing rule says gets more scrutiny, not less. Require the first pass to compute `Z` **both**
  ways for at least one head/layer and report `||Z_matrix − Z_recurrence||`.
- **A third point nobody has stated, which bears on the engine budget.** The commuting trick moves the mixing
  onto `x` in `R^D` rather than `v` in `R^{d_v}`. Implemented *literally*, the recurrent state becomes
  `SUM φ(k_s) x_sᵀ` in `R^{m×D}` instead of the standard `SUM φ(k_s) v_sᵀ` in `R^{m×d_v}` — a
  **`D/d_v = 12×` inflation of the shipped state**, 9.4 MB/layer at fp32 vs 0.8 MB. Against the engine's
  measured 16 MB L3 cliff (`project_probe3_cache`) that is the difference between resident and not. The two
  forms are mathematically identical, so the fix is trivial — **ship "project then mix", never "mix then
  project"** — but §1.2's framing (*"the mixing can therefore be applied to the input"*) invites exactly the
  wrong one, and the Builder brief must say so explicitly.

---

## PART C — audit of `ADAPTER_MEMO_01` §2.2c (requested; new and unaudited)

### 21. The arithmetic reproduces exactly. No objection to the numbers as computed.

Recomputed independently (`D=8192, d_ffn=28672, L=80`, GQA 64/8, `head_dim=128`):

```
q 67.11M + k 8.39M + v 8.39M + o 67.11M = 150.99M attn/layer     -> memo "151 M"   OK
3 x 8192 x 28672                        = 704.64M ffn/layer      -> memo "705 M"   OK
total 855.64M/layer ; shares 17.6% / 82.4%                       -> memo "18/82"   OK

 4/80: active 1.731B, 0.8657 GB/tok, 24.55 tok/s   -> memo 24.5   OK
 6/80: active 2.033B, 1.0167 GB/tok, 20.90 tok/s   -> memo 20.9   OK
 7/80: active 2.184B, 1.0922 GB/tok, 19.46 tok/s   -> memo 19.5   OK
10/80: active 2.637B, 1.3187 GB/tok, 16.11 tok/s   -> memo 16.1   OK
20/80: active 4.147B, 2.0737 GB/tok, 10.25 tok/s   -> memo 10.2   OK

">20 tok/s" -> <=2.125B active -> attn budget 0.997B -> 6.60 layers = 8.25%   -> memo "<=8.3%, <=6.6"  OK
5-trit: 0.2 vs 0.5 B/weight = 2.5x -> 61.3 / 52.2 / 40.3 / 25.6               -> memo table            OK
```

The 12.5% literature floor is faithfully transcribed from `ATTENTION_LINEARISATION_PRIOR_ART.md` (§1b:
tested ratios 50/25/12.5/0%, no elbow, degradation still worsening). **§2.2c's central claim — that the
budget assumed a ratio 1.5× below anything measured — is correct and well-supported. That part stands.**

### 22. The geometry is **68.5B, not 100B**. Headline and table are on different models. **FLAG**

`80 × 855.64M = 68.45B` body; `+2.10B` embeddings (128k vocab, untied) = **70.55B**. The memo calls this
*"a 100B-class geometry"* and reports `active ≈ 1.73 B (1.7% of 100B)` — against the actual body that is
**2.53%**, and the 50 GB / 100B headline in §2.1 is a different model from the one in the table.

The `tok/s` column is *not* affected (it is computed from the absolute active count, correctly). **What is
affected is whether the target closes at 100B.** At the same per-layer geometry a genuine 100B body needs
`L ≈ 117`, and the FFN term scales with `L`:

| | `L = 80` (68.5B — what the table prices) | `L = 117` (100.1B — what the headline claims) |
|---|---|---|
| 4 attn layers (5%) | 1.731B → **24.5 tok/s** | 2.253B → **18.9 tok/s** |
| 10 attn layers (12.5%) | 2.637B → **16.1 tok/s** | 3.159B → **13.5 tok/s** |

> **At a real 100B, §2.2's worked target misses >20 tok/s even at the unvalidated 5% ratio.** The memo's
> conclusion ("the budget does not close at any ratio the literature has validated") is *understated*, not
> overstated — it does not close at the ratio the memo *assumed*, either, once the parameter count matches
> the label. Fix the label or fix the geometry; do not leave both.

### 23. **The table assumes the converted layers' attention weights vanish. Under R2 they do not.** Largest-effect finding in Part C.

§2.2 prices attention as `n_retained × 151M` — the 76 converted layers contribute **zero**. But R2's own
§1.1 says `W_q`, `W_k` are *"reused from the donor, never modified"*, `W_v` is solved (same shape), and `W_o`
is retained. **A converted layer keeps all four projections.** They are ternary, so 0.5 B/weight, but they
are read **every token** unless something makes them activation-sparse — and nothing in the programme has
established activation sparsity on attention projections (Probe-2 and Phase 58.B are FFN/dReLU results).

```
80 layers x 151M (all layers keep q,k,v,o)  = 12.08 B
+ 80 layers x 705M x 2% (FFN)               =  1.13 B
                                     active = 13.21 B  ->  6.60 GB/token  ->  3.2 tok/s
```

> **The attention-layer ratio is nearly irrelevant to the weight-read budget under a projection-reusing
> conversion — 4/80 and 10/80 both give 3.2 tok/s, because the 76 converted layers dominate either way.**
> What R2 buys is the **KV cache**, not the weight stream. That may well be the more valuable purchase (see
> finding 24), but §2.2 / §2.2c prices the wrong quantity, and the "5% vs 12.5%" question the memo now treats
> as the programme's central risk is, on this axis, a rounding error next to the 12.08 B that is unaccounted
> for.
>
> **This is not an argument against R2.** It is an argument that the speed case for R2 must be rebuilt on the
> KV axis, and that "convert attention to a recurrent operator" only reaches the memo's numbers if the
> converted operator's projections are *also* driven to ~2% activation — an unestablished, separate claim.

### 24. KV traffic is absent from the table, and it inverts the ranking

Commit `e9503e3` on this branch is titled *"the KV read path at 128K is the wall"*. The §2.2c table counts
weight bytes only. KV bytes per token for the **retained** attention layers (fp16 K and V, 8 kv-heads × 128):

| context | 4 attn layers | 10 attn layers | weights at 12.5% |
|---|---|---|---|
| 4 096 | 0.067 GB | 0.168 GB | 1.319 GB |
| 32 768 | 0.537 GB | **1.342 GB** | 1.319 GB |
| 131 072 | 2.147 GB | **5.369 GB** | 1.319 GB |

**At 32K, KV already equals the entire weight stream. At 128K it is 4× the weight stream.** The marginal
cost of the 10th retained attention layer is not the 151M weights the table prices — at 128K it is ~0.54 GB
of KV traffic, ~7× larger. **Retaining more attention layers to buy quality is far more expensive than
§2.2c says, and the correct conclusion is *stronger* than the memo's, in the same direction.**

### 25. The model is a bandwidth model fed by a constant measured in a compute-bound regime — the memo flags the constant but not the model

§2.1 records the LUT path as *"compute-bound by ~16×, not bandwidth-bound"*, and §2.4 correctly notes that on
a compute-bound path denser packing is a **loss** (*"+10% compute for −50% bytes … ~10% slower"*). §2.2c then
applies `tok/s = 21.25 GB/s ÷ bytes/token` throughout, and multiplies the whole column by 2.5× for 5-trit
packing.

The memo flags the 5-trit claim as *"not a claim, a conditional that D5 resolves"* — honest, and I do not
call it fabrication. **But the same conditional applies to every other cell in the table, and that is not
flagged.** If the path is still compute-bound at donor width, `tok/s` does not scale with bytes/token at all
and the entire §2.2 / §2.2c table — not just the 5-trit column — is the wrong functional form. Conversely,
if it is bandwidth-bound at donor width, then `21.25 GB/s`, measured *in the compute-bound regime at
`D = 1536`*, is not the right constant either.

> **Either way, `21.25 ÷ bytes` is a model that cannot be correct in both regimes, and the memo uses it in
> both.** State it once, at the top of §2.2: *every tok/s figure in this memo is conditional on D5 finding
> the path bandwidth-bound at donor width; if D5 finds it compute-bound, the table must be rebuilt on a flops
> model, not rescaled.*

### 26. What in §2.2c I checked and found **sound**

- The 12.5% figure, its provenance, and the "no elbow, still worsening" reading — all match the prior-art
  document's table-level `[T]` citations.
- The distinction between Zoology's 6.3% (from-scratch, not donor-converted) and the donor-conversion numbers
  — correctly drawn, and correctly refused as evidence.
- The self-correction naming §2.2's 5% as *"the same shape of error as the D1 bracket collapse"* — that
  reading is right, and the correction was made in the unfavourable direction, which is the right sign.
- The D5 reframe (packing question → "must we out-perform the literature or merely match it") follows from
  the arithmetic as given. Findings 23 and 24 change *which* quantity is decisive, not whether the reframe
  was logically valid on its own terms.

---

## Summary of required changes before code is written

**BLOCK-level (must change):**

1. **Adopt the finding-7 estimator** `W = (W°(Z*Zᵀ + lam I))(ZZᵀ + lam I)^-1`, or rewrite §3.3's C1 as a
   lam=0 test with a derived lam>0 expectation. As written, C1 cannot fire. *(findings 5, 6, 7)*
2. **Rewrite §1.4.** D4's special case is `Z = R_Sᵀ Z*`, not `Z* = Z`; D4's cross term is regularised; the
   solver does not transfer unchanged. *(findings 5, 8)*
3. **Use exact `hessian_rank_ratio` on `ZZᵀ`**, never `theoretical_rank_ratio`. *(finding 9)*
4. **Bind `T` in §3.2** (`T/D >= 24`, >=3 values reported, held-out mandatory). *(findings 14, 15)*
5. **Bind the layer set** to all layers with a pre-stated statistic. *(finding 16)*
6. **Withdraw the `<0.25` justification** — D4's wrong-layer arm was −1.472. *(finding 17)*
7. **Replace C5** with C5' (content-severed `A_lin`) and add **C6** (causal-uniform structure-free floor);
   report the ≈0.05 causal floor and read every result against it. *(finding 18)*
8. **Add C0 PLANTED POSITIVE** — C1 is blind to the entire `φ → A_lin → Z` path. *(finding 19)*

**FLAG-level (must be stated; may not change the design):**

9. Redefine `X` as post-RMSNorm. *(finding 2)*
10. Require row-normalised `φ`; Qwen2 has q/k/v biases. *(finding 3)*
11. State that the GQA closed form solves the **unweighted** objective; the `W_o`-weighted one is a Sylvester
    system with no single-inverse solution. *(finding 11)*
12. Restate §1.5 from the projector identity, not from §3a. *(finding 12)*
13. Report `err(DN)` and `err(SOLVED)` raw, per head and per layer, with a denominator floor. *(finding 13)*
14. State in §3.2 that `recovery` is **blind to the recall cap**, and stratify by `A_soft` row entropy in
    pass 1. *(finding 20)*
15. Fix `lambda` by a pre-stated rule. *(finding 10)*
16. Ship "project then mix"; the literal reading of §1.2 inflates the recurrent state 12×. *(finding 20b)*

**RECOMMENDED, and cheaper than the whole design:**

17. **Run the principal-angle pre-screen first** (finding 12). It answers R2's central question directly,
    yields a derived upper bound on `recovery` to check the apparatus against, and costs one SVD per layer.

---

## Checks I could not run, for scheduling

Everything above is fp64 synthetic at `D <= 24`, `T <= 400`, seconds and a few MB. It verifies **algebra and
control logic**, not donor behaviour. These need a real model and are deferred:

1. **Real `cond(ZZᵀ)` and exact `rank(ZZᵀ)/D`** per feature map, per layer, on the actual donor — the
   finding-9 trigger. Cheap once activations exist (~0.04 s per eigvalsh at `D = 1536`).
2. **The finding-6 identity deviation on real `Z`.** My 6.6e-4 is a benign toy; the real number decides
   whether the brief's estimator is merely imprecise or unusable.
3. **Principal angles / canonical correlations between `colspace(A_lin Xᵀ)` and `colspace(A_soft Xᵀ)`** on
   the real donor, per layer, per `φ` — the finding-12 pre-screen. **Highest value per unit of compute of
   anything in this brief.**
4. **The real size of the GQA `W_o`-weighting gap** (finding 11). My 0.8% is a random-`W_o` toy and is not a
   prediction; it needs the donor's actual `W_o`.
5. **The real null floor.** My ≈0.05 causal-uniform floor is synthetic; on real activations (heavy-tailed,
   strongly mean-shifted — the exact property that made D4's `shuffle_columns` retain 52% of off-diagonal
   mass) it will be **higher**. Must be measured before any threshold is fixed.
6. **`T`-dependence of `recovery` on the real donor** (finding 15). The synthetic curve establishes the
   *mechanism*; the real curve sets the required `T`.
7. **fp32 recurrence vs `T×T` matrix drift** at the production `T` (finding 20b). Needs a real feature map at
   real length.
8. **Per-head entropy stratification** of `err(DN)` / `err(SOLVED)` (finding 20).
9. **D5's `rate(D, threads)` curve** — finding 25 cannot be closed without it. Already commissioned.
10. **Donor identity and revision.** §3.4 says "whichever ≤2B model is already local"; that is not a
    pre-registration. The exact repo id + revision hash must be printed from the loaded object (D1's
    achieved-vs-requested lesson) before the run, not chosen during it.

---

*Controller, 2026-08-22. Reproduction scripts: `benchmarks/donor_adaptation/r2/r2_check.py`, `benchmarks/donor_adaptation/r2/r2_check2.py`
(numpy, fp64, seconds, a few MB). Sources read: `transformers` 4.57.6 `modeling_llama.py` /
`modeling_qwen2.py` / `modeling_qwen3.py`; `benchmarks/donor_adaptation/density/d4_reconstruction.py`;
`benchmarks/donor_adaptation/density/results/d4_reconstruction.json`;
`docs/research/ATTENTION_LINEARISATION_PRIOR_ART.md`;
`docs/research/donor_adaptation/decisions/ADAPTER_MEMO_01_SPEED_BUDGET.md`;
`docs/research/donor_adaptation/audits/CONTROLLER_DENSITY_AUDIT.md`.*
