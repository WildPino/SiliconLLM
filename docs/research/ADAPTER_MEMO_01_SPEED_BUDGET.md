# Adapter Memo 01 — the two levers, and the speed budget that follows

**Author:** the Adapter / Principal · **Date:** 2026-08-21 · **Base revision:** `e410243`
**Status:** decision memo. Fixes the target the Builder's briefs are written against. Not a gate — the
Architect/Owner alone adjudicates sealed gates.

---

## 1. The distinction the v2 programme never made cleanly

`DONOR_V2_DENSITY_PROGRAMME.md` framed the problem as "the donor is 33–66× too dense" and listed four
attacks. It did not separate two mechanisms that behave completely differently on this engine:

| | **Static weight sparsity** | **Dynamic activation sparsity** |
|---|---|---|
| What it is | Weights permanently removed | Neuron's activation is zero *for this token*, so its weights are never loaded |
| When decided | Once, at conversion | Per token, at inference |
| Measured cost, in-house | **Catastrophic.** D1: +0.13 to +4.8 BPB at every organ and level, against σ_seed = 0.005 | **Near-free.** Probe-2: dReLU vs SiLU = **+0.0006 BPB** at 92%/79% sparsity |
| Ceiling | Bounded by how much of the model is genuinely redundant | Bounded only by how peaked the per-token computation is |
| Gets us to 2%? | **No.** 90% already costs +3.5 BPB | **This is the only lever that can** |

**The 2%-per-token target lives entirely on the second lever.** Static pruning is a footprint tool and a
secondary one; it is not the road to speed. Every hour spent pushing static pruning past the point D1
already measured is an hour spent on the wrong axis.

This also re-reads D1 correctly. D1's numbers are not a discouraging result about our programme — they
are the **floor that reconstruction must beat**, measured on the naive method. Brief D4 is measuring the
gap. And D1's block-vs-unstructured penalty is a statement about *weight* masks, which is not the mask
the engine actually exploits.

**Consequence for the reading of `feedback` and prior probes:** Probe-2 and Phase 58.B are not old
results to be cited in passing. They are the *primary* evidence for the whole v2 thesis, and they were
obtained on a model **we trained ourselves with dReLU from scratch**. Whether the same sparsity can be
*induced* in a donor pretrained with SiLU, and at what token cost, is now the highest-value open
question in the programme. It is out with the Researcher.

---

## 2. The speed budget, and where the 2% must actually come from

### 2.1 The headline arithmetic

At **0.5 bytes/weight** (the engine's real packing: two trits per byte via `(w0+1)*3+(w1+1)`), a 100B
donor is **50 GB** of weights. At the measured engine-integrated ternary LUT rate:

> active_bytes_per_token = 50 GB × activation_fraction
> tokens_per_second = 21.25 GB/s ÷ active_bytes_per_token

| activation fraction | active weights | bytes/token | tok/s |
|---|---|---|---|
| 100% (dense donor) | 100 B | 50 GB | **0.43** |
| 10% | 10 B | 5 GB | 4.3 |
| 4% | 4 B | 2 GB | 10.6 |
| **2%** | **2 B** | **1 GB** | **21.3** |
| 1% | 1 B | 0.5 GB | 42.5 |

**⚠ The load-bearing constant, flagged rather than assumed.** 21.25 ± 0.36 GB/s was measured
*engine-integrated at D = 1536*. Whether it holds at donor width D = 8192 is **exactly the class of
transplant that has already been caught twice today** (the `B_block ≥ 22` floor, and the skippable-0.001
column). The LUT path is recorded as **compute-bound by ~16×**, not bandwidth-bound, so its rate is a
function of the kernel's shape and not a property of the DRAM. **This must be re-measured at donor
projection widths before any tok/s figure derived from it is quoted to the Owner.** Until then every
number in this table carries that caveat. → **queued as brief D5.**

### 2.2 Where the 2% has to come from — the part nobody had allocated

A 100B-class geometry (`d_model = 8192`, `d_ffn = 28672`, `L = 80`):

| | per layer | share |
|---|---|---|
| attention (q + k + v + o), GQA 64/8 | 151 M | 18% |
| FFN (gate + up + down) | 705 M | **82%** |
| total | 856 M | |

**If the FFN goes to 2% active and attention stays dense, the model is still at 13% overall** — because
attention, untouched, is 18% of the weights and 100% of them are read every token. Attention alone would
cap us at **3.3 tok/s**.

So the target decomposes, and both halves are mandatory:

> **(a) FFN at ~2% activation** — the activation-sparsity lever, on 82% of the weights.
> **(b) Attention retained on at most a few percent of layers** — everything else converted to a
> recurrent operator with no KV re-read.

Worked target: attention kept on **4 of 80 layers**, FFN at 2% everywhere:

```
attention   4 layers × 151 M                   = 0.60 B
FFN        80 layers × 705 M × 0.02            = 1.13 B
                                        active ≈ 1.73 B  (1.7% of 100B)
                                   bytes/token ≈ 0.87 GB
                                         tok/s ≈ 24
```

**That closes.** It is the first end-to-end arithmetic in this programme that reaches the Owner's ">20
tok/s at 100B" without a step marked as a guess — subject to the §2.1 caveat on the rate constant.

### 2.3 What this says about the sealed constraints

- **S3 ("attention on a minority of layers") is now quantified and it is far more demanding than
  "minority".** 4 of 80 is **5%**, not 49%. The earlier stage-0 finding — full attention retained on
  ≤2 of 28 layers, windowed on ≤18 of 28 — is consistent with this and is now explained by it rather
  than being an isolated measurement.
- The KV-traffic wall found at stage 0 is the same constraint seen from the other side. It is not a
  separate problem to be solved after the density problem; **it is the attention half of the same
  budget.**

---

## 2.4 "Va rifatta la lookuptable" — the packing question, and why it may reverse at donor scale

The Owner's mandate names the lookup table explicitly as a thing that may need redoing. It has a concrete
arithmetic behind it that has not been written down in this programme.

The engine packs **two trits per byte** — `(w0+1)*3 + (w1+1)`, nine states in eight bits — giving
**4 bits/weight, 0.5 bytes/weight**. The information-theoretic floor for ternary is `log2(3) = 1.585`
bits, and the natural dense packing is **five trits per byte** (`3^5 = 243 ≤ 256`), i.e. **1.6 bits/weight,
0.2 bytes/weight**.

> **That is a 2.5× reduction in bytes read per weight.**

On the engine as measured today, that is a **loss**, and the project has already banked why: the LUT path
is **compute-bound by ~16×**, and the related finding on nibble-packing was *"+10% compute for −50% bytes
on a compute-bound path = ~10% slower"*. Denser packing buys footprint, not speed. Correct — **at
`D = 1536`, with a 2.25 MiB working set that is L3-resident.**

**The donor case is not that case.** A 100B donor at 0.5 bytes/weight is **50 GB**, streamed from DRAM.
Nothing about that is L3-resident, and a path that streams tens of gigabytes per second from main memory
is bandwidth-bound almost by definition. **If the path reverts to bandwidth-bound at donor scale, the sign
of the packing trade-off flips and 5-trits-per-byte becomes a ~2.5× speedup rather than a ~10% penalty.**

Compounded with §2.2's worked target, at 0.2 bytes/weight the same 1.73 B active weights become 0.35 GB
per token — **~61 tok/s** on the same rate constant. That is the difference between meeting the Owner's
">20 tok/s" and comfortably exceeding it.

**This is not a claim. It is a conditional that D5 resolves**, because D5's `rate(D, threads)` curve with
working-set sizes and an L3-residency flag is exactly the measurement that says whether the path is still
compute-bound at donor width. Sequence matters:

1. **D5 first** — is the path compute-bound or bandwidth-bound at donor projection widths?
2. **Only if bandwidth-bound**, a follow-on brief on the 5-trit packing: decode cost per byte, whether
   `pshufb` can still serve it (a 243-state decode does not fit a 16-byte shuffle table the way a 9-state
   one does — this is the hard part and it may not be free), and end-to-end parity.
3. The parity rule is absolute and is banked from Phase 60: **kernel-bit-exact does not compose to
   system-correctness; end-to-end parity always.**

Recorded here so that the packing question is not re-derived from scratch later, and so that its
dependency on D5 is explicit rather than assumed.

---

## 3. What is now known, and at what confidence

| finding | status | confidence |
|---|---|---|
| Weight-basis rotation makes weights no sparser (D2) | closed, axis dead | measured, 12/12 matrices, controls fired |
| ρ-floor is `32768/D` → ~4 neurons (interleaved) / 12 (per-organ) at donor width | closed (D0b) | measured, all controls fired; **under Controller audit** |
| Static block-pruning without reconstruction is catastrophic at every level (D1) | closed for the naive baseline | measured; 2 points struck for mask quantisation; **under Controller audit** |
| dReLU activation sparsity is near-free at 92%/79% | banked (probe-2) | measured **on a model we trained ourselves**, not on a donor |
| Block-structuring activation sparsity is free (18%→50% @ BS8) | banked (Phase 58.B) | same caveat |
| No published work solves ternary + contiguous-block jointly | verified | primary-source search; every adjacent cell populated |
| No published work targets CPU/DRAM at tens-of-KB block granularity with both a quality table and a contiguous-read-size table | verified | the two closest (Neuralink, LLM-in-a-Flash) are calibrated to smartphone flash pages, not CPU cache economics |
| **MoEfication physically permutes the FFN so experts are CONTIGUOUS submatrices, losslessly, with no backbone training** | **promoted as the layout mechanism** | primary source, equations in §3.2; the mechanism we need, already published |
| **Conditional computation pays on memory-bound CPUs and largely does not on GPUs** | **corroborated externally** | MoEfication Table 1 `[T]`: at 12.5% activation, **CPU 2.28× vs GPU 1.47×** — measured wall-clock, not projected |
| Router cost when FFN semantics are preserved | −0.3 average at 12.5% activation | MoEfication Table 4 `[T]`, T5-Large. **12.5% is not 2%; T5 is not a 100B donor** |
| **Cheap MoE-ification with a fresh router destroys general-purpose retention** | **measured, and it fails S4 badly** | LLaMA-MoE-v2 Table 1 `[T]`: 7B tokens → **MMLU 67.22 → 37.41**, IFEval 76.53 → 32.72 |
| Co-activation grouping: **1 negative, 1 positive, UNRECONCILED** | changes how D0 must be run | Apple abandoned it (*"worked against our original intention"*); Neuralink reports it works and **cites Apple but never engages the negative**. ⚠ **Corrected 2026-08-22:** I first counted LLaMA-MoE as a second negative — it is not; its losing arm was weight-space k-means on `W_up` rows, a different construction, and untabulated `[X]` |
| Sparse Upcycling as a density tool | **rejected** | each expert is a full FFN copy → capacity scaling at equal-or-greater per-token cost. Wrong direction |
| Joint solver cost at 100B on a T4 | ~15–40 h | three independent anchors |
| **How much reconstruction recovers of D1's loss** | **OPEN — brief D4 running** | the number the programme turns on |
| **Whether donor activation sparsity can be induced, and at what token cost** | **OPEN — Researcher running** | the number the 2% target turns on |
| **Whether 21.25 GB/s holds at donor width** | **OPEN — brief D5 written and pre-registered** | every tok/s figure depends on it |
| **Whether the LUT path is still compute-bound at donor width** | **OPEN — D5 answers it** | decides the sign of the packing trade-off (§2.4) |
| 5-trits-per-byte packing: 0.2 vs 0.5 bytes/weight | **conditional on D5**, not yet a brief | ~2.5× on bytes; a loss today, potentially a 2.5× gain at donor scale |

---

## 3a. A structural finding from D4's pre-registration — healing must happen in the CONSUMING layer

The Builder derived this **before running D4** and recorded it in the pre-registration, which is where a
finding like this has to appear if it is to be trusted:

> D1's masks make `gate_proj`/`up_proj`/`q`/`k`/`v` **ROW-structured** — whole output neurons zeroed — and
> `o_proj`/`down_proj` **COLUMN-structured** — whole input features zeroed. The row-separable
> SparseGPT-style solve only has freedom to act when the surviving set `S` varies *within* a row's
> support. For a ROW-structured organ every kept row already has full support and every dropped row has
> none: **the closed form is mathematically forced to return exactly `W` or exactly `0`, independent of
> `H`.**

**Read what that means for the programme, not just for D4.** Zeroing an output neuron is exactly what the
engine needs — a dead neuron's weights are the ones we skip. And single-layer reconstruction on the organ
that produced that neuron **cannot recover anything, by construction**. Not by failure of the method; by
the shape of the problem.

The recovery, if it exists, has to come from the **consuming** layer: a neuron killed in `gate_proj` is a
dead *input column* of `down_proj`, and there the surviving set does vary within each row, so the solve
has real freedom. **For structured neuron removal, healing belongs in the layer that reads the neuron,
not the layer that writes it.**

Three consequences, recorded now:

1. **The dossier's Tier-2 multi-layer Schur-complement solve is not an optional refinement.** For
   row-structured sparsity it is the *only* place the correction can live. I previously catalogued Tier 2
   as budget spent on an unestablished drift law (§2.2 of the adjudication); that criticism was about the
   *drift justification*, and it stands — but the multi-layer solve turns out to be load-bearing for a
   completely different and better reason than the one the dossier gave for it.
2. **D4's `gate_proj` arm can only return zero recovery**, and that zero is a theorem, not evidence about
   the method. It must not be averaged in with the column-structured organs.
3. **Any future brief that prunes whole neurons and then reconstructs the same layer is malformed.**
   Write it as prune-in-layer-ℓ, reconstruct-in-layer-ℓ+1.

---

## 3b. Amendment to brief D0, issued before D0 is dispatched

`DONOR_V2_DENSITY_PROGRAMME.md` §6 specifies D0 as co-activation clustering of FFN neurons with a
mandatory random-clustering control. The prior art (`STRUCTURED_SPARSITY_PRIOR_ART.md` §5) has changed
what that control means:

> **The random-clustering arm is not a formality — one published attempt at co-activation bundling
> failed outright, and the one success cites that failure without explaining it.** The literature is
> **split and unreconciled**, not stacked against us. If our co-activation arm shows an advantage we are
> resolving an open contradiction; if it shows none, we have replicated Apple. Either is worth reporting.
>
> *(Corrected 2026-08-22: I originally wrote "two published results predict it will lose", counting
> LLaMA-MoE. Wrong — LLaMA-MoE's losing arm was weight-space k-means on `W_up` rows, not co-activation
> grouping, and it is untabulated.)*

And Apple's stated failure mode is a concrete, checkable hypothesis rather than a vague warning: *highly
active neurons get duplicated across bundles.* **D0 must therefore measure and report neuron duplication
across clusters.** A probe reporting only cluster quality cannot distinguish "co-activation clustering
does not help" from "our implementation hit the documented failure mode" — and those two call for
opposite next moves.

---

## 3c. THE SYNTHESIS — the two things that fit our budget each lack what the other has

The Researcher's activation-sparsity pass (`ACTIVATION_SPARSITY_PRIOR_ART.md`, 1345 lines) returns one
dominating fact:

> **Every training-requiring ReLUfication method is three to four orders of magnitude outside our budget.**
> ReLU Strikes Back **30B tokens**; ProSparse **34.6–134B**; Turbo Sparse/dReLU **150B tokens on 64× A800**;
> Q-Sparse **40B**; the 2025 Sparsing Law needs **800B tokens even at 2.4B parameters**. Against
> 90 T4-hours/week, none of these is a candidate at any scale. **The continued-pretraining route to
> activation sparsity is closed for us. Not tight — closed.**

Note especially Turbo Sparse: it is the closest published analogue to our own in-house dReLU result, and
it cost **150B tokens on 64 A800s**. Our +0.0006 BPB at 92%/79% sparsity came free *because we trained
with dReLU from the start*. That is the difference between designing for sparsity and retrofitting it,
priced by someone else.

**What does fit the budget:**

| | cost | sparsity pattern | usable by our engine? |
|---|---|---|---|
| TEAL / CATS (training-free thresholding) | **≈1 GPU-hour or less** (a calibration pass) | **scattered** | **No.** Scattered sparsity is worth ~zero at 48 KB granularity |
| MoEfication permutation | training-free, lossless | **contiguous by construction** | **Yes** — but it needs a grouping, and the grouping is the contested part |

**Read the table.** The cheap method gives sparsity without contiguity. The layout method gives
contiguity without deciding sparsity. **Neither is sufficient; their composition is exactly what is
missing, and it is exactly the gap the searches keep returning empty on.**

> **The candidate route, stated as a hypothesis to be tested and not as a plan:**
> take a donor's *existing* activation sparsity, obtain it cheaply with a training-free threshold method,
> and then **permute the FFN so that the sparsity that already exists becomes contiguous** — MoEfication's
> `W̄₁ = W₁P` applied not to hand-built experts but to whatever firing structure the donor already has.
> Cost: a calibration pass plus an offline permutation. No gradient training of the backbone.

This is the first route in the programme where **both** halves are individually published, training-free,
and inside budget, and where the novelty is the composition rather than a new mechanism we would have to
invent. It also lands squarely on the two verified gaps: nobody targets CPU/DRAM at tens-of-KB block
granularity, and nobody solves ternary + contiguous-block jointly.

**And ternary is not an obstacle to it.** The Researcher surfaced two sources I had not asked for —
**BitNet a4.8** (arXiv:2411.04965) and **Sparse-BitNet** (arXiv:2603.05168) — both showing **ternary
weights compose with induced sparsity without catastrophic interaction**. Neither tests a dense-donor →
ternary+sparse *joint conversion*, so this is permission to proceed, not evidence that our specific
conversion works.

**A framing correction the Researcher made, and it is a good one:** my structured/scattered binary hides
a third tier. GPU tensor-core N:M structuring (Q-Sparse's "Block Q-Sparse", 2:4) *is* structured — at a
granularity roughly **1000× too fine** for a 48 KB read. A paper claiming "structured sparsity" must be
checked for *which* structure before it counts as relevant to us. Three tiers, not two: scattered,
GPU-block, cache-block.

---

## 4. Standing rules this memo re-states, because they were violated today

1. **A measured constant carries the dimensions at which it was measured.** Before reusing one at a new
   scale, rewrite its symbolic derivation and see which dimensions appear inside it. Caught twice today:
   `B_block ≥ 22` and the skippable-0.001 column. §2.1 above flags the third candidate before it bites.
2. **Do not calibrate on the numbers of the document you are auditing.** My own "413 h" came from
   adopting an implausible throughput out of the artefact under review. Every conversion needs an
   external anchor, and at least two independent ones before it enters a verdict.
3. **An instrument must report the parameter it ACHIEVED, not the one it was asked for.** D1's mask
   quantised a requested 90% to 100% on two organs; caught only because `zero_frac` was recorded next to
   `level`.
4. **A planted control must be shown to fire on a known positive before its nulls mean anything** — and
   a control that fires on everything is not a control. This is what the Controller is auditing now.
