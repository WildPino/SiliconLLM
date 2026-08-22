# Brief R2a — the principal-angle pre-screen: R2's ceiling, computed before R2 is built

**Author:** the Adapter / Principal · **Date:** 2026-08-22 · **Status:** pre-registered, awaiting a free machine.
**Queued behind D5** — it loads a donor and would contend with a bandwidth measurement. It is not lower
priority; it is behind an instrument.

---

## 0. Where this came from, and why it runs before the experiment

The Controller's audit of `BRIEF_R2` (`audits/CONTROLLER_R2_DERIVATION_AUDIT.md`) confirmed the algebra and
then pointed out that **R2's central question has a closed-form answer that needs no solver, no apparatus
and no training** — one SVD per layer.

R2 solves `min_{W_v} || W_v Z − W_v^donor Z* ||_F`. The reachable set is `{ W_v Z : W_v }` — every matrix
whose rows lie in the **row space of `Z`**. So the part of the target that **no** `W_v` can ever reach is
fixed by geometry alone:

```
    residual  =  || W_v^donor · Z* · (I − P_Z) ||_F  /  || W_v^donor · Z* ||_F
```

where `P_Z` projects onto `rowspace(Z) ⊆ R^T`. This is the **principal angles between the two subspaces**.

> **`residual` is an upper bound on R2's achievable quality, computed without solving anything.**
> Large residual for every candidate `φ` ⇒ **R2 is bounded away from success and we learn it for the cost
> of a few SVDs.** Small residual ⇒ the ceiling is high and building the solver is justified.
>
> **This is the highest value per unit of compute on the entire (a) line.**

---

## 1. ⚠ The degeneracy that would make this vacuous — read before writing any code

`Z` is `D × T`, so `rowspace(Z)` has dimension **at most `min(D, rank(Z))`** inside `R^T`.

> **If `T ≤ D`, then generically `rowspace(Z)` is all of `R^T`, `P_Z = I`, and the residual is exactly zero
> — for every `φ`, including a nonsensical one.** The pre-screen would report a perfect ceiling and mean
> nothing.

**This is not hypothetical. It is almost certainly what the Controller measured:** recovery ran `0.741` at
`T/D = 0.5` down to `0.165` at `T/D = 17`. **The high number at small `T` is an artifact of
underdetermination, not a good result.** The same family of defect has already cost this project twice
(D4's `rank(H)/N` structurally pinned at 1.000 when `T > D`; D1's requested-vs-achieved sparsity).

**Mandatory, non-negotiable:**

1. **`T/D` is pinned in advance and printed as ACHIEVED**, read off the actual arrays, not the config.
2. **`rank(Z)` and `rank(Z*)` are computed and reported per layer** (numerical rank at a stated tolerance).
3. **The instrument REFUSES to report a residual when `T ≤ D`** — it must stop and say why, not emit a zero.
4. Report the residual as a **curve over `T/D`**, not a single number. If the curve has not flattened by the
   largest `T` affordable, **say so** — an unconverged curve is a result, and quoting its endpoint as the
   answer is the error D1 was struck two points for.

---

## 2. What to compute

For a real donor, per layer, per candidate feature map `φ`:

| symbol | what | note |
|---|---|---|
| `X` | hidden states **entering the attention block**, `D × T` | after the layer's input norm, exactly what `W_q`/`W_k`/`W_v` see |
| `A_soft` | the donor's real causal softmax attention, per head | the target mixing |
| `A_lin` | linearised mixing from `φ`, using the donor's **own** `W_q`, `W_k`, unchanged | the candidate |
| `Z* = X A_softᵀ`, `Z = X A_linᵀ` | `D × T` each | |
| **`residual`** | `‖W_v^donor Z* (I − P_Z)‖_F / ‖W_v^donor Z*‖_F` | **the deliverable** |

Feature maps, all **fixed, zero learned parameters**: `elu(x)+1` (Katharopoulos), Based's 2nd-order Taylor
of `exp`, Performer FAVOR+ random features (**state the seed**), and one Hedgehog-shaped elementwise map
used *without* fitting.

**Donor:** whichever ≤2B open-weights model is already local. **State which and its exact revision hash.**

---

## 3. Controls — and the two this project has already paid for

| control | construction | expected | why it is here |
|---|---|---|---|
| **C1 IDENTITY** | `A_lin := A_soft` | residual **exactly 0** | `rowspace(Z) = rowspace(Z*)` by construction. Tests the code path. **It is a tautology of the algebra — label it as one.** |
| **C6 CAUSAL-UNIFORM** | uniform mixing over the causal prefix | **the free floor** | the Controller measured **≈0.05 recovery that any causal mixing earns for doing nothing.** No result below this floor means anything |
| **C5′ WRONG-SEQUENCE** | `A_lin` from an **unrelated sequence** — real, causal, correctly normalised, wrong content | should be poor | the strong null: structure and constraints intact, correspondence wrong |
| **C4 WRONG-LAYER** | `A_lin` from a different layer | should be poor | D4 measured wrong-layer at **−1.472** recovery — catastrophic, not noise-level |

> **DO NOT use a shuffled `A_lin` as a null.** Measured: shuffled recovered **0.170** against the real arm's
> **0.165** — *better than the real thing.* Shuffling a causal matrix **destroys causality along with the
> content**, yielding a denser, higher-rank mixing with *more* fitting freedom. **A healthy method would
> have looked refuted by its own control.** This is the second shuffle-based null on this programme that
> was not null.

---

## 4. The stratification that stops a false positive

The Researcher established that fixed-state operators carry an **architectural recall cap**, and that a
closed-form solve *"would not remove the state-dimension ceiling."* **Spiky retrieval heads contribute
negligibly to a mean Frobenius quantity** — so a `φ` that destroys retrieval can post an excellent
residual.

**Mandatory: stratify every reported residual by `A_soft` row entropy.** Low-entropy (peaked,
retrieval-like) rows are reported as their **own stratum**, never averaged into the bulk. **If the peaked
stratum is much worse than the bulk, that is the finding**, whatever the mean says.

---

## 5. Deliverable

`docs/research/donor_adaptation/probes/R2A_PRINCIPAL_ANGLE.md`:

- **Residual × layer × `φ` × entropy-stratum**, with the `T/D` curve and every control.
- `rank(Z)`, `rank(Z*)`, `T/D` **achieved**, per layer.
- An explicit statement of **the ceiling this implies for R2**, and whether it justifies building the
  solver at all.
- The full O(T) and O(T²) allocation inventory. **`A_soft` is `T×T` per head per layer — state what is
  materialised, what streams, and the peak footprint.** The linear side has an `O(1)`-state recurrence and
  should never need a `T×T` matrix; **if the apparatus materialises `A_lin` densely, say so**, because then
  it is not measuring the operator we would ship.

**A large residual is a real and valuable result.** It closes the (a) line cheaply and honestly, and this
project would far rather spend a few SVDs learning that than a Builder's week discovering it.

---

## 6. Standing rules

> Report **ACHIEVED**, not requested, parameters — printed from the objects themselves.
> An instrument must **fire on a known positive** before its nulls mean anything.
> A result that **flatters** the hypothesis earns more scrutiny, not less.
> **Nothing here may be reported until C1 fires and the `T > D` refusal is demonstrated working.**
