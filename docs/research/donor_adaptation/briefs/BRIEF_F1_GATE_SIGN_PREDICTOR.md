# Brief F1 — the 33.3% FFN floor is not structural: attacking the gate with a closed-form low-rank predictor

**Author:** the Adapter / Principal · **Date:** 2026-08-28 · **Status:** pre-registered.
**Owner directive, 2026-08-28:** *"ti voglio più inventivo … qui siamo all'avanguardia, quindi la dobbiamo
scrivere noi la letteratura in materia."*

---

## 0. The observation this whole brief rests on

A week of probes has produced a wall, recorded in `ADAPTER_MEMO_01` §2.2d and measured **in our own engine**
(`ENGINE_PLAN.md:56`, E3 CLOSED): gate 100% + up 21.5% + down 12.3% = **44.6% of FFN bytes per token**, with
a structural floor of **33.3%** — because

> *you cannot know which entries of `h` are zero until you have computed `W_g·x` in full.*

I have been treating that as a law. **It is not a law. It is a property of our engine's implementation.**

Look at what the sentence actually says: the gate is expensive because **it is the predictor**, and a
predictor that reads every weight it predicts for saves nothing on itself. **But nothing requires the
predictor to be `W_g` at full width.** We need to know *which neurons survive*, not *what the gate's values
are*.

> **The gate produces `d_ffn` real numbers. The skip decision needs `d_ffn` bits.**
> We are paying full matrix-multiply price for a full-precision quantity in order to extract one bit per
> neuron. **That is the inefficiency, and it is ours.**

---

## 1. The construction

SwiGLU: `h = SiLU(W_g x) ⊙ (W_u x)`, `y = W_d h`, with `W_g, W_u : d_ffn × d` and `W_d : d × d_ffn`.

Factor the gate, in **closed form, with no training**:

```
    W_g  ≈  A · B        A : d_ffn × r ,  B : r × d
    ĝ    =  A (B x)      the predicted gate, cost r(d + d_ffn) instead of d·d_ffn
```

Use `ĝ` **only to choose the active set** `S = { i : ĝ_i > τ }`. Then compute the **exact** gate on `S`
alone, and read only rows `S` of `W_u` and columns `S` of `W_d`.

**Cost accounting**, as a fraction of the dense FFN's bytes:

```
    FFN_active  =  [ r(d + d_ffn)/(d·d_ffn)  +  |S|/d_ffn  +  |S|/d_ffn  +  |S|/d_ffn ] / 3
                     └── predictor ──┘          └── exact gate, up, down on S ──┘
```

At donor width (`d = 8192`, `d_ffn = 28672`) with `r = 64`, the predictor term is **1.0%**. So:

| true active fraction `a` | `FFN_active` with a rank-64 predictor | (for comparison: dense-gate floor) |
|---|---|---|
| 0.10 | **10.3%** | 33.3% |
| 0.05 | **5.3%** | 33.3% |
| 0.02 | **1.67%** | 33.3% |

> **The floor disappears.** With a predictor, `FFN_active` is bounded by the model's actual activation
> sparsity, not by an implementation artefact. The 2% target stops being unreachable-by-construction and
> becomes an empirical question about sparsity — which is the question we wanted to be asking all along.

## 1.1 The asymmetry that makes this tractable, and which the accuracy framing hides

This is **not** a problem of approximating `W_g` well.

| error | consequence |
|---|---|
| **false positive** — predicted active, actually inactive | we compute a neuron we did not need. **Costs a little speed. Zero quality loss.** |
| **false negative** — predicted inactive, actually active | we skip a neuron that mattered. **Costs quality.** |

**The two errors are not symmetric and must never be summed into an "accuracy".** We want **recall ≈ 1** and
we are willing to pay precision for it. Lower `τ` until recall is where we want it; the price is a larger
`|S|`, which is a speed cost that the table above prices exactly.

> **So the quantity to measure is not "how well does rank `r` approximate `W_g`". It is: at what `r` and `τ`
> does recall reach 99% / 99.9%, and how large is `|S|` when it does?** A predictor that is mediocre at
> regression can be excellent at this job.

## 1.2 Why the factorisation should be activation-weighted, not plain SVD

Plain SVD minimises `‖W_g − AB‖_F`, which weights every input direction equally. **We do not care about
directions the data never visits.** The right objective is

```
    min_{A,B}  ‖ (W_g − AB) X ‖_F        X = calibration activations
```

which is a **closed-form** generalised problem (SVD of `W_g X`, or of `W_g H^{1/2}` with `H = XXᵀ`) — the
same activation-weighted family as D4's reconstruction and R2a's principal angles. **We already own this
machinery and its controls.**

---

## 2. Prior art — assume it exists, and find it before claiming anything

**Deja Vu (Liu et al.) predicts contextual sparsity with small learned predictors, and this is the same
problem.** Do not present this construction as novel until the Researcher has reported. What may be
distinct here — and what is worth stating precisely — is the combination of **closed-form,
activation-weighted, no training, and recall-first thresholding with the false-positive/false-negative
asymmetry made explicit**. **That is a claim to check, not to assert**, and this project has already had
one brief built on a "confirmed gap" that was false one fence over.

Note also our own history: `project_phase58_predictor` records that a **trained** predictability gate
FAILED on our model, while finding 86–92% intrinsic predictor-free structure. **That is a negative result
on a different construction and it does not transfer** — but it does mean the Builder must say clearly how
this differs.

---

## 3. Pre-registered design

Registered before any number exists.

### 3.1 The measurement

Per layer, per rank `r ∈ {8, 16, 32, 64, 128, 256}`, on real calibration activations:

1. Compute the donor's **true** gate `g = W_g x` and its true active set at the donor's own operating
   threshold. **State how that threshold is defined and print it as ACHIEVED.**
2. Compute `ĝ` from the activation-weighted rank-`r` factorisation.
3. Sweep `τ` and report the **recall / |S| trade-off curve** — not a single operating point.
4. Report **`FFN_active`** from §1's formula at each point, **beside** the recall. A recall number without
   its `|S|` is meaningless here.

### 3.2 Arms and controls

| arm | construction | role |
|---|---|---|
| **rank-`r` activation-weighted** | `min ‖(W_g − AB)X‖_F` | the claim |
| **rank-`r` plain SVD** | `min ‖W_g − AB‖_F` | does activation weighting earn its complexity? |
| **C1 full-rank** | `r = min(d, d_ffn)` | **must give recall exactly 1.0 and `|S|` exactly the true set.** The tautology control — label it as one, and note that it tests the code path, not the science |
| **C2 random projection** | `B` random, `A` solved | the null: how much does *any* `r`-dimensional read of `x` buy? **This is the floor every other arm must beat** |
| **C3 wrong-layer** | factorisation from another layer's `W_g` | real spectrum, wrong correspondence — the strong null that worked in D4 |
| **C4 planted positive** | a `W_g` synthetically made exactly rank-`r` | **the instrument must recover recall 1.0 at that `r` and not before.** Show it firing |

**Do not use a shuffled `W_g` as a null.** Two shuffle-based nulls on this programme were not null, and the
most recent one *beat the real arm* because shuffling destroyed a structural constraint. C2 and C3 above are
the replacements.

### 3.3 Pre-registered thresholds

| result | verdict |
|---|---|
| recall ≥ 0.99 with `FFN_active ≤ 0.10` at `r ≤ 64`, beating C2 by a clear margin | **the floor is broken.** Proceed to end-to-end BPB |
| recall ≥ 0.99 only with `FFN_active > 0.25` | partial — better than 33.3% but not transformative |
| **C2 random projection does as well as the fitted arms** | **the construction has shown nothing** — any low-dimensional read of `x` would do, and that is a statement about the data, not about `W_g` |

### 3.4 The check that decides whether any of this survives contact with quality

Recall is a proxy. **The real question is end-to-end.** Once the curve exists, take the best operating
point and measure **held-out BPB against the unmodified donor**, plus at least one **recall-sensitive**
probe — because this project has established that a fixed-state or sparsified model can hold perplexity
while losing retrieval, and because **MMLU may never be reported inside an average** on this programme.

**A layer-wise recall number is not a result. It is a screen.**

---

## 4. Why this is worth doing before anything else on the FFN half

`ADAPTER_MEMO_01` §2.2d proved that **no block-skippable fraction, however good, delivers the budget** — the
identity `FFN_active = (3 − 2s)/3` needs `s = 1.47` and `s ≤ 1`. That killed the dense-FFN route and sent
the FFN half to MoE, where §2.2g then found that probe-4 never entered our regime and the scale-free target
is 32× beyond anything measured.

> **This brief attacks the term that made the identity hopeless.** If the gate's `1` becomes `0.01`, the
> identity becomes `FFN_active = (0.01 + 2(1−s))/3` and `s = 0.97` reaches 2%. **That is a demanding but
> ordinary sparsity target rather than an arithmetic impossibility**, and it does not require restructuring
> the donor into an MoE at all.

**And it is cheap**: one activation-weighted SVD per layer per rank. No training, no GPU, the same cost
profile as R2a — which priced six mixings in about twenty minutes of compute.

---

## 5. Standing rules

> Report **ACHIEVED**, not requested, parameters — read off the constructed objects.
> An instrument must **fire on a known positive** before its nulls mean anything — here, C4.
> A result that **flatters** the hypothesis earns more scrutiny, not less. **This one would flatter badly:
> it rescues a route I have watched collapse for a week.** Treat every favourable number accordingly.
> **Never sum a false positive and a false negative into an accuracy.** They have different costs.

---

## AMENDMENT 1 — 2026-08-28 — a second construction, sublinear where the first is linear

**Appended, not edited in place.** Nothing registered above is removed. **One arm is added, and if it
works it is stronger than the arm the brief was written around.**

### A1.1 The FFN is a key-value memory, and we already built a retriever

`h = SiLU(W_g x) ⊙ (W_u x)`, `y = W_d h`. Read that as a lookup rather than a matmul:

- the **rows of `W_g`** are `d_ffn` **key vectors** in `R^d`;
- the **columns of `W_d`** are the corresponding **value vectors**;
- the **active set is the top-`k` keys by inner product with `x`.**

This is the FFN-as-key-value-memory view (Geva et al.), and it is not controversial. **What follows from it
is the part we have not used:**

> **Finding the top-`k` by inner product without computing all `d_ffn` inner products is approximate
> nearest-neighbour search — and this project already built, tuned and measured one.** `project_phase55`
> banks **IVF-PQ recall at ~18 µs**, and `project_recall_scaleup_synthesis` banks the scaling law
> `nlist ∝ √N` together with the rule that an auxiliary structure is free **iff** it is cache-resident and
> bandwidth-light.

**We built a retriever for the recall tier and never pointed it at the FFN's own gate.**

### A1.2 Why this is not a variant of the rank-`r` arm — it is a different complexity class

| | rank-`r` predictor (§1) | **IVF-PQ over `W_g`'s rows** |
|---|---|---|
| what it computes | **all** `d_ffn` predicted scores, cheaply | **only** the retrieved candidates |
| cost in `d_ffn` | **linear** — `r·d_ffn` for the `A` multiply | **sublinear** — `nprobe` lists out of `nlist ∝ √d_ffn` |
| at `d_ffn = 28672`, `r = 64` | 1.8 M MACs for the `A` multiply alone | scans a small fraction of the keys |
| fails by | a direction the low-rank fit does not span | a key that landed in an unprobed list |

> **The rank-`r` construction still touches every neuron. The index does not.** At the active fractions
> this programme needs — 2–5% of 28672 — that is the difference between reading a thin slice of everything
> and reading only what you retrieve.

### A1.3 The asymmetry from §1.1 applies unchanged, and it is what makes ANN acceptable here

Approximate retrieval is normally uncomfortable because it silently misses things. **Here the miss
structure is exactly the one we can afford:**

- a **retrieved-but-inactive** key is a false positive — **speed cost, zero quality loss**;
- a **missed active** key is a false negative — quality loss.

**So `nprobe` is the recall dial**, and it is the same dial as `τ` in §3.1: turn it up until recall is where
we want it and pay in `|S|`. **The measurement, the curves, and every control in §3.2 transfer unchanged** —
this arm slots into the harness you are already building.

### A1.4 What to measure, and the honest cost caveat

Same deliverable shape as §3.1: the **recall / `|S|` / `FFN_active` curve**, this time swept over
`nprobe` (and `nlist`, if cheap), against the same **C2 random-projection floor**, **C3 wrong-layer**, and
**C4 planted-positive** controls.

**Three things that must be counted honestly and would be easy to leave out:**

1. **The index's own bytes.** Codebooks and inverted lists are weights we must read or hold. **They go into
   `FFN_active`, not into a footnote.** Per the banked law, the auxiliary structure is free only if it is
   cache-resident and bandwidth-light — **state whether it is, at donor width, or say that it is not.**
2. **PQ is lossy, and the scores it ranks by are approximate.** Recall must be measured against the
   **true** top-`k` from the exact gate, never against the index's own ranking.
3. **Build cost per layer, once, offline** — state it. It is a conversion-time cost, not an inference-time
   one, but it is not zero and this programme does not hide costs in the word "offline".

### A1.5 Scope, and permission to defer

**If adding this arm would delay the rank-`r` result, do the rank-`r` arm first and report it.** I would
rather have one clean curve than two half-measured ones. **But say so explicitly if you defer** — I want to
know that the comparison is pending rather than assume it was covered.

**And the same warning as §5 applies with more force here:** this construction is the one I would most like
to be true, because it reuses machinery we already own and it is sublinear where everything else has been
linear. **That is precisely why its favourable numbers need the hardest look.**


---

## AMENDMENT 1 — 2026-08-29, the Adapter / Principal

**Appended, not edited in place.** Two corrections, both found by the Builder that ran this brief, and the
second retires the brief's own thesis.

### A1.1 An arithmetic error in §1's cost table — mine

The `0.02` row reads **1.67%**. It is **2.33%**.

`FFN_active = (p + 3a)/3` with `p = r(d + d_ffn)/(d·d_ffn) = 64·(8192+28672)/(8192·28672) = 1.0045%`.
At `a = 0.02`: `(0.010045 + 0.060)/3 = 0.02335`. **1.67% is what you get from `2|S|/d_ffn`** — two organs,
where the two rows above it correctly use three. Verified independently. **The `0.10` and `0.05` rows are
right**; only the target row is wrong, and it is the row the brief leans on.

### A1.2 §1's thesis — "the floor disappears" — is REFUTED by this brief's own run

§1 argued that with a predictor `FFN_active` is bounded by the donor's real activation sparsity rather than
by the 33.3% dense-gate artefact, and that the 2% target therefore *"becomes an empirical question about
sparsity."* **It was the right question. The donor answered no.**

Measured on the donor (`Qwen2.5-1.5B`-class, `silu`), `a` = the fraction of neurons that must stay active for
the FFN block to lose at most `ε` of its output Frobenius energy, θ fitted on calib and reported on 4096
**held-out** tokens, converging exactly on all 28 layers × 3 ε:

| ε | mean `a` |
|---|---|
| 0.001 | 0.993 |
| 0.01 | **0.951** |
| 0.05 | 0.909 |
| **oracle on `\|h_i\|`** (sees `W_u` too — stronger than anything this brief proposes) | **0.836** |
| θ=0, the dReLU sign rule | 0.13–0.30, **at 41–88% relative output error** |

> **Even a perfect oracle can skip only 16.4% of the neurons.** That is a ceiling on the *question*, not a
> limit of our construction, and it sits an order of magnitude away from the `a = 0.02` the cost table was
> built to price. **The 2% target is not reachable on this donor by selection of any kind.**

**And the predictor is a pessimisation, not a shortfall.** Charged, at the measured `a`, the best
`FFN_active` obtained anywhere is **0.7384**; the mean at `r = 64` is **0.9788** under a uniform-byte
convention and **1.0022** when the predictor is charged at fp16 against ternary weights. **Above 1.0 means
the predictor costs more than the dense FFN it was meant to skip.**

### A1.3 What this brief got right, and what to keep

- **The instrument fires.** C4 (planted rank) **fires 28/28** — exact at planted rank 32, rel dev 3e-7,
  broken at r=16. C2, C2b and C3 are real controls. **C1 passes but cannot fail; it is a tautology and is
  labelled one.** The nulls here are earned.
- **`eff_rank_actw ≈ 2` must never be quoted as evidence of a compressible gate.** It is inherited from the
  activation covariance: `eff_rank(H^{1/2})` = 2.0–24.3 tracks it, `H`'s top eigenvalue carries 19–71% of
  its variance, and `W_g` itself needs **1452 of 1536** singular values for 99% energy, with `r=64`
  capturing 14%.
- **A resolution law worth keeping.** The resolvable span is `ρ(1−a)`, so at `a ≈ 0.999` every arm is pinned
  inside a 0.001-wide window. Spearman ρ(available range, measured skill) = **−0.836, p = 3.1e-8**:
  **apparent skill is highest exactly where there is no room for skill.** On the seven layers with a range
  above 0.01, skill falls to 0.25–0.54. **Any future probe reporting a margin must first report the span
  that margin had available.**
- **Amendment 1's ANN arm is dead as specified.** The 0.99 recall target is met at exactly one setting —
  `nprobe = nlist`, a full scan. The cheap rows are cheap because they retrieve nothing (`FFN_active` 0.0138
  at recall 0.0007). Partition collapsed, PQ reconstruction error 0.826, and a Euclidean k-means was used to
  route an inner-product query.

### A1.4 The one gap that could still reopen this

**Nothing here connects a per-layer ε to end-to-end quality, and no BPB was run.** Every number above is
Frobenius energy of one FFN block's output. If the model tolerates far more than ε = 0.05 end-to-end, the
ladder stops too early and `a` at the tolerable ε is unknown.

**That is the only remaining route to a different answer, and it must be measured rather than argued.**
Registered here so that the closure is conditional on it and cannot be quietly promoted to unconditional.
