# Brief R2 — a closed-form solve for attention linearisation (the V-side normal equation)

**Author:** the Adapter / Principal · **Date:** 2026-08-22
**Status:** DERIVATION — sent to the Controller to be broken BEFORE any code is written.
**Owner directive, 2026-08-22:** *"ciò che mi interessa di più è proprio (a), quindi impegno e metodo."*
(a) = converting attention to a recurrent operator **with no training**. This brief is that line's method.

---

## 0. Why this exists

The Researcher's `ATTENTION_LINEARISATION_PRIOR_ART.md` §4(b) returned a **CONFIRMED GAP**:

> Every published linearisation method fits its feature map by gradient descent. **None publishes a
> closed-form solve** — despite several of the losses (LoLCATs' MSE against softmax attention weights,
> MOHAWK's Frobenius matching) being *linear-least-squares-shaped* if the nonlinearity were removed.

That is a gap sitting next to an apparatus we already own. **D4 built, debugged and validated a
regularised layer-wise normal-equation solver, with a working identity control that has already caught
one real bug (the regularisation asymmetry).** This brief claims the two things connect, and states
exactly how.

**I am not asking anyone to believe the derivation. I am asking the Controller to break it first.**

---

## 1. The derivation

### 1.1 Notation

One donor layer, one attention head `h`. Tokens are columns.

| symbol | meaning | shape |
|---|---|---|
| `x_t` | hidden state entering the attention block at position `t` | `D` |
| `X` | `[x_1 … x_T]`, the calibration activations | `D × T` |
| `W_q^h, W_k^h` | query/key projections — **reused from the donor, never modified** | `d_k × D` |
| `W_v^h` | value projection — **the unknown we solve for** | `d_v × D` |
| `W_o` | output projection, `W_o^h` its slice for head `h` | `D × H·d_v` |
| `A_soft^h` | the donor's causal softmax attention matrix | `T × T` |
| `A_lin^h` | the linearised attention matrix induced by a **fixed** feature map `φ` | `T × T` |

### 1.2 The step everything rests on

The head output under any attention matrix `A`:

```
out_t^h = Σ_s A^h[t,s] · v_s^h
        = Σ_s A^h[t,s] · (W_v^h x_s)
        = W_v^h · ( Σ_s A^h[t,s] x_s )          <-- W_v is LINEAR, A[t,s] are SCALARS
        = W_v^h · z_t^h ,   where  z_t^h := Σ_s A^h[t,s] x_s
```

> **The value projection commutes with the attention mixing.** The mixing can therefore be applied to the
> *input* rather than to the values. This is the entire trick, and it is why the problem is closed-form.

Define two mixed-activation matrices, both `D × T`:

```
Z^h   = X (A_lin^h)ᵀ      (the linearised mixing — what the converted model will see)
Z*^h  = X (A_soft^h)ᵀ     (the donor's true mixing — the target)
```

### 1.3 The objective and its solution

Match the converted head's output to the donor's own head output on calibration data:

```
min_{W_v^h}  Σ_t || W_v^h z_t^h  −  W_v^{h,donor} z*_t^h ||²
```

Differentiate, set to zero, add the same Tikhonov term D4 already uses (`λ = 1e-4 · trace(H)/N`):

```
    W_v^h  =  ( W_v^{h,donor} · Z*^h (Z^h)ᵀ ) · ( Z^h (Z^h)ᵀ + λI )⁻¹
```

### 1.4 The claim that makes this cheap

Compare to the equation `d4_reconstruction.py` already implements:

```
D4:   W' = ( W · H_cross ) · ( H_reg )⁻¹      with  H_cross = X_S Xᵀ ,  H = X_S X_Sᵀ
R2:   W' = ( W · H_cross ) · ( H_reg )⁻¹      with  H_cross = Z* Zᵀ  ,  H = Z Zᵀ
```

> **They are the same equation. D4 is the special case `Z* = Z`.** D4 restricted the support of a single
> activation set; R2 relates two *different* mixings of the same inputs. **The solver, the regulariser,
> the conditioning diagnostic and the identity control all transfer unchanged.**

### 1.5 What this does NOT claim

**It does not fit the feature map.** `φ`, `W_q`, `W_k` are fixed inputs to the solve. The solve finds the
best *value-side compensation* for a given, wrong attention matrix.

**This is the §3a theorem again, and I want it stated as a prediction rather than discovered as a
surprise.** D4 established that healing must happen in the **consuming** direction: a ROW-structured
error upstream leaves the downstream solve no freedom. Here `A_lin` is upstream and `W_v` is downstream.
So the derivation carries a built-in failure mode:

> **If `A_lin` mixes so differently from `A_soft` that `Z` and `Z*` are weakly related, the solve
> degenerates toward a mean predictor and recovers nothing.** How weakly-related is too weak is exactly
> what the experiment must measure. **A near-zero result is a real result here, not a failed run.**

### 1.6 The pay-off that is unique to a closed form

Because each evaluation is one linear solve rather than a training run, **the feature map `φ` becomes a
free hyperparameter we can actually sweep.** Every published method must commit to one `φ` and spend GPU
hours fitting it. We can price a dozen. *That*, not the accuracy of any single fit, is the strategic
argument for this line.

---

## 2. Controller: what I want you to attack

**Do not implement anything. Do not help this look plausible. Break it.** In priority order:

1. **§1.2, the commuting step.** Is it actually valid for a real donor's attention block? Check against
   the true operator order in a modern Llama/Qwen-class layer — **RoPE, RMSNorm placement, the causal
   mask, and the softmax denominator**. RoPE in particular is applied to Q and K *after* projection and is
   position-dependent: confirm it lives entirely inside `A^h[t,s]` and therefore does not break the step.
   **If `W_v` does not commute with the mixing in the real architecture, this brief is dead at line one
   and I need to know today.**
2. **§1.3.** Is the normal equation correct as written — orientation, transposes, which side the inverse
   is on? D4's first run died of a regularisation asymmetry (`H[S,S]+λI` as system matrix against an
   **unregularised** `H[:,S]` on the RHS). **Check whether I have reintroduced the same asymmetry here.**
3. **GQA.** 1.5B-class donors use grouped-query attention: one `W_v^{kv}` serves several Q heads, each
   with a *different* `A^h`. The per-head decoupled solve in §1.3 then does not apply. Confirm the joint
   solve over the sharing group is still linear — and if it is not, say so.
4. **The `W_o` variant.** Solving `W_v` and `W_o` jointly is bilinear, hence alternating least squares with
   each half closed-form and still zero gradient training. **Is that alternation guaranteed to descend,
   and can it be seeded so it does not wander?**
5. **Scale.** Per the standing law, **inventory every O(T) and O(T²) allocation on this path.** `A_soft` is
   `T×T` per head per layer; `Z` is `D×T` per head. State what is materialised, what streams, and what
   the peak footprint is at the T you propose. The linear side should never need a `T×T` matrix at all —
   it has an `O(1)`-state recurrence — **confirm that, because if the apparatus materialises `A_lin` as
   `T×T` it is silently measuring something other than what we would ship.**

---

## 3. Pre-registered design, for the Controller to audit before it runs

Registered now, before any number exists.

### 3.1 Arms — the comparison that decides everything

The unconverted donor is **not** the baseline to beat. The baseline is *doing nothing on the value side*:

| arm | `A` used | `W_v` used | role |
|---|---|---|---|
| **CEILING** | `A_soft` | donor | the unconverted donor — defines the best achievable |
| **DO-NOTHING** | `A_lin` | donor | **swap softmax→linear, change nothing else. THE BASELINE.** |
| **SOLVED** | `A_lin` | R2 solve | the claim |
| C1 IDENTITY | `A_soft` | R2 solve | **must return the donor weights to machine precision** |
| C4 WRONG-LAYER | `A_lin` from another layer | R2 solve | must collapse (D4: wrong-layer ≈ noise) |
| C5 NULL | shuffled `A_lin` | R2 solve | must collapse — **and see §3.3** |

**Primary quantity — the recovery fraction:**

```
recovery = ( err(DO-NOTHING) − err(SOLVED) ) / ( err(DO-NOTHING) − err(CEILING) )
```

with `err = || Y_hat − Y*_donor ||_F / || Y*_donor ||_F` on held-out calibration data.

### 3.2 Pre-registered thresholds

| recovery, on a majority of converted layers | verdict |
|---|---|
| **≥ 0.50** | the line is real; proceed to end-to-end BPB and a recall probe |
| 0.25 – 0.50 | partial; report as such, no promotion |
| **< 0.25** | **the closed-form V-side solve does not rescue linearisation. Say so and close it.** |

`< 0.25` is set at the level D4 measured for its *wrong-layer* arm — i.e. at the level where the solve is
not distinguishable from having the wrong structure entirely.

### 3.3 Controls, and the two lessons that must not be re-learned

- **C1 IDENTITY is non-negotiable and runs FIRST.** Set `A_lin := A_soft`, so `Z = Z*`. The solve must
  return `W_v^donor` to machine precision. **This exact control caught D4's regularisation bug. If it
  does not fire, nothing downstream may be reported.**
- **C5 must be VERIFIED null, not assumed null.** D4's `shuffle_columns` arm **retained ~52% of the real
  off-diagonal Frobenius mass** — a null that was not null. **Measure the residual structure of the
  shuffled `A_lin` and report it as a number.** A null arm that has not been measured is not a control.
- **Report ACHIEVED, not REQUESTED, parameters.** D1 silently quantised a requested 90% sparsity to 100%
  and cost us two points. Whatever `φ`, `T`, and layer set are actually used must be printed from the
  objects themselves.
- **Do not print `[SIGNIFICANT]` on arms whose expected value is null**, and do not label a
  zero-by-construction outcome `INCONCLUSIVE`. Both happened in D1/D4.

### 3.4 Scope of the first pass

Feature maps to sweep (all **fixed**, no learned parameters): `elu(x)+1` (Katharopoulos), Based's 2nd-order
Taylor of `exp`, Performer FAVOR+ random features (state the seed), and one Hedgehog-shaped elementwise
map used *without* fitting. Donor: whichever ≤2B open-weights model is already local — **state which and
its exact revision**. Layer-wise reconstruction error only in this pass; end-to-end BPB and a
recall-sensitive probe are stage 2 and only happen if §3.2 clears.

---

## 4. Sequencing

1. **Controller breaks §1 and §2.** No code until it reports.
2. If the algebra survives → I write the Builder brief and dispatch.
3. **In parallel, the Researcher checks one narrow question:** has anyone published *this specific
   construction* — fixed feature map, donor `W_q`/`W_k` reused, value projection solved in closed form
   against the donor's own attention outputs? §4(b) says no closed-form solve exists at all, but
   **novelty is a claim and this project does not promote claims that have only been checked in the
   general.** `[X] not found` is the correct and valued answer.

---

## 5. Standing rules for anyone acting on this

> A literature number does not enter a decision until it has been read **in the paper's own table**.
> An instrument must be shown to **fire on a known positive** before its nulls mean anything.
> A deviation that **flatters** the hypothesis gets more scrutiny, not less — all three of this project's
> past fabrications pointed the favourable way.
> **My own derivation above is subject to all three.**
