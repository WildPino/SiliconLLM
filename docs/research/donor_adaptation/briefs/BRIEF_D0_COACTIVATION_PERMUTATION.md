# Brief D0 — can a STATIC permutation make a DYNAMIC activation pattern contiguous?

**Author:** the Adapter / Principal · **Date:** 2026-08-22 · **Status:** pre-registered, awaiting dispatch
**Assigned to:** the Builder · **Depends on:** nothing. Dispatchable as soon as a Builder is free.
**Pre-registration rule:** written and pushed BEFORE the run. Nothing below is edited after numbers exist;
corrections go in an appended, dated section.

---

## 1. Why this is now the probe that matters

`ADAPTER_MEMO_01_SPEED_BUDGET.md` §3c identified the one route where both halves are published,
training-free, and inside our budget:

> Take the activation sparsity a donor **already has**, obtain it cheaply with a training-free threshold
> method, and then **permute the FFN so that the sparsity that already exists becomes contiguous** —
> MoEfication's `W̄₁ = W₁P` applied not to hand-built experts but to the firing structure the donor
> already possesses.

The first half is bought from the literature (TEAL/CATS, ≈1 GPU-hour). **The second half is the part
nobody has done at our granularity, and D0 is the experiment that says whether it is possible at all.**

If a permutation cannot make a donor's activation pattern contiguous, the route dies here and we have
spent one CPU-day finding out. If it can, we have the mechanism that turns scattered sparsity — worth
approximately zero on this engine — into skipped memory traffic.

## 2. The scientific crux, stated precisely

**A permutation is static. An activation pattern is per-token.** We get to choose *one* neuron ordering
per layer, and it must serve *every* token.

That tension is the whole experiment, and it is exactly where the literature splits:

- **Apple (LLM in a Flash, arXiv:2312.11514)** tried co-activation bundling and reports it **failed**,
  with a named mechanism: *"this resulted in loading highly active neurons multiple times and the
  bundling worked against our original intention."* Highly-active neurons want to be in every group.
- **Neuralink (arXiv:2410.19274)** reports the same idea **works** — 2.37× over llama.cpp — and **cites
  Apple without engaging the negative result.**

**The literature is split and unreconciled.** That is a better position than it first appeared: this is
not a probe re-running a settled question, it is a probe adjudicating an open one.
*(See `STRUCTURED_SPARSITY_PRIOR_ART.md` §5, including my recorded correction — I initially miscounted
LLaMA-MoE as a second negative; its losing arm was weight-space k-means, a different construction.)*

## 3. The question, stated so it can only have one answer

> For a real donor's measured per-token FFN activation pattern, what **block-skippable fraction** is
> achieved at the engine-legal block size, under each of: **no permutation**, a **random permutation**,
> and a **co-activation-derived permutation** — and how does that fraction vary with block size?

Not "does clustering work". The deliverable is a **table of skippable fraction × permutation arm × block
size**, with the no-permutation arm as the number to beat and the random arm as the honest control.

## 4. Arms — one variable, three levels

| arm | what it is | why it is here |
|---|---|---|
| **identity** | no permutation; the donor's native neuron order | **the baseline to beat.** Any claim of benefit is measured against this, not against zero |
| **random** | a random permutation, several seeds | the control the literature predicts may match the co-activation arm. **Report its seed-to-seed spread** — if the co-activation arm sits inside that spread it has shown nothing |
| **co-activation** | neurons grouped by how often they fire together on calibration data, then permuted so groups are adjacent | the arm under test |

Report the i.i.d. reference from D0b alongside: at 90% sparsity a structureless mask gives
**0.2824** skippable at block 12 and **0.6563** at block 4. **Any arm that does not beat the structureless
reference has found no structure**, whatever it beats among the other arms.

## 5. ⚠ Mandatory measurement: neuron duplication

**Apple's failure mode is a concrete, checkable hypothesis, not a vague warning.** D0 must measure and
report, per layer:

- the **duplication factor**: how many groups each neuron would need to belong to for the grouping to
  capture its co-activations, and the distribution of that across neurons;
- the same broken out for the **most-active decile** of neurons specifically, since that is where Apple
  says the mechanism bites.

**A probe that reports only cluster quality cannot distinguish "co-activation grouping does not help"
from "our implementation hit the documented failure mode" — and those two call for opposite next moves.**
This measurement is not optional and its absence invalidates a null result from this probe.

## 6. Pre-registered interpretation — fixed before any number exists

| outcome | reading |
|---|---|
| co-activation beats identity **and** clears the random arm's seed spread | the route in memo §3c is alive; proceed to a joint sparsity+permutation brief |
| co-activation ≈ random, **both** beat identity | the *benefit is from grouping at all*, not from co-activation. Cheaper: use random. Report as a replication of LLaMA-MoE's finding in a different construction |
| co-activation ≈ random ≈ identity | **no exploitable static structure.** The route in §3c dies and we say so. Check §5's duplication numbers to attribute it |
| co-activation **worse** than identity | replication of Apple. §5's duplication numbers should show why; if they do not, suspect the implementation |

**Do not adjust these after seeing numbers.** If an outcome arises that this table does not cover, report
that fact rather than forcing it into a row.

## 7. Planted controls — all must FIRE

1. **⚠ IDENTITY / LOSSLESSNESS — run this FIRST.** A permutation of FFN neurons, with the consuming
   projection permuted correspondingly, is **mathematically output-preserving**. Verify the permuted model
   reproduces the unpermuted model's BPB **exactly**, and report the max absolute logit deviation.
   **If this does not hold, everything else is void and you STOP.**
   *(This control class is why brief D4 was aborted: its identity control caught a regularisation bug that
   would otherwise have produced a full sweep of plausible, meaningless numbers. Put it first.)*
2. **Known-positive: planted block structure.** Construct a synthetic activation pattern with a *known*
   contiguous group structure under a *known* scrambling permutation. The clustering must **recover it**,
   and the measured skippable fraction must rise to the planted value. **An instrument that cannot find
   structure that is definitely there cannot be believed when it reports none.**
3. **Known-negative: i.i.d. pattern.** On a structureless synthetic pattern at matched density, all three
   arms must land together at the analytic `q^B` value. Any arm that "wins" on noise is broken.
4. **Random-arm seed spread.** At least 5 seeds. This defines the width of "no effect" and every
   comparison is read against it, not against a point estimate.

## 8. Constraints and hygiene

- **Same donor, same eval slice, same BPB path as D1/D4.** A different eval path voids comparison.
- **Calibration for the co-activation statistics must be DISJOINT from the eval slice**, asserted in code
  with corpus sha256s recorded — follow the pattern D4 established, it is good.
- **The block size is read from the D0b formula, never hardcoded.** Sweep block sizes around the
  engine-legal floor, and report the ρ-floor for the donor's width alongside.
- **Machine quiescence:** record resident heavyweight processes in the manifest. This is not a timing
  probe so contention does not corrupt it, but the record costs nothing and this project has a banked
  case of unreproducible numbers from unrecorded environment.
- Report **achieved** quantities next to requested ones — density achieved vs threshold requested, group
  sizes achieved vs asked. D1 had two points silently rounded from 90% to 100%.
- Git revision, exact command line, seeds, thread counts in every record. Write incrementally.

## 9. Deliverables — then STOP

- `benchmarks/donor_adaptation/density/results/d0_coactivation.json`: pre-registration, all four control
  outcomes, the skippable × arm × block-size table with seed spreads, the §5 duplication distributions,
  and the reproducibility manifest.
- A section in `docs/research/DENSITY_PROBES.md` in the style of §2/§3.
- Report to the Principal: the table, which controls fired, the duplication numbers, and which row of §6
  the result lands in — **or that it lands in none of them.**

**A failed control is a STOP and a message to the Principal, not a line in a log.**

---

## AMENDMENT 1 — 2026-08-22, the Adapter / Principal

**Appended, not edited in place.** The brief above stands as written. This changes what D0 is *for*, and
adds one mandatory column, because the arithmetic underneath it turned out to be decisive.

### A1.1 The identity that reframes this probe

`ADAPTER_MEMO_01` §2.2d records the measured in-engine FFN cost (`ENGINE_PLAN.md:56`, E3 CLOSED):
**gate 100% + up 21.5% + down 12.3% = 44.6% of FFN weight-bytes per token.** The gate is 100% because it
**is** the predictor — you cannot know which entries of `h` are zero until you have computed `W_gate·x` in
full.

So with a dense gate, for an up/down block-skippable fraction `s`:

```
    FFN_active = ( 1.0 + (1−s) + (1−s) ) / 3  =  (3 − 2s) / 3
```

| `s` | FFN active |
|---|---|
| 0.282 *(D0b structureless ref, block 12)* | 81.2% |
| 0.656 *(D0b structureless ref, block 4)* | 56.2% |
| 0.785 / 0.877 *(what our engine already achieves on up / down)* | ~44.6% |
| 0.950 | 36.7% |
| **1.000 — skip ALL of up and down** | **33.3%** |

Setting `FFN_active = 0.02` and solving gives **`s = 1.470`**. `s` cannot exceed 1.

> **The 2% FFN target is unreachable for ANY block-skippable fraction, including a perfect one. The dense
> gate alone is 33.3% and it is never skipped. So no result D0 can return — however good — delivers the
> speed budget in the framing of §3 above.**

That is not a reason to cancel D0. **It is a reason to stop treating D0's headline number as the answer
and to be explicit about the question it actually settles.**

### A1.2 What D0 is actually for

The escape from the 33.3% floor is not a better skip — it is **removing the full-width gate matvec**, which
means restructuring the donor's dense FFN into an **MoE**: a small router selects a few experts, and the
unselected experts' weights are never read at all.

> **D0's real question is therefore not "how much can we skip" but "does this donor's activation pattern
> contain cluster structure strong enough to carve the FFN into experts" — i.e. is MoEfication viable on
> this donor, without training.** The permutation arms measure exactly that; only the reporting was aimed
> at the wrong target.

This makes D0 **load-bearing for the whole FFN half of the programme**, not a granularity study. It goes out
the moment D5 releases the machine. It is queued behind an instrument, not behind a priority.

### A1.3 Mandatory additional reporting

Everything in §3–§5 above still stands. **Add these, and do not report the skippable table without them:**

1. **Translate every skippable fraction into `FFN_active = (3 − 2s)/3`, and print it beside `s`.** A row
   that reads `s = 0.66` looks like a success; the same row reads `56.2% active` and is plainly not one.
   **Report the number that moves the budget, not the number that flatters the method.**
2. **The implied MoE geometry**, per layer, for each permutation arm: how many clusters, their size
   distribution, how many would have to be activated per token to retain the measured active neurons, and
   the resulting **expert-active fraction** — the MoE analogue of `FFN_active`, this time *without* a dense
   gate, plus the router's own cost stated explicitly rather than assumed negligible.
3. **The duplication cost, already mandated in §5, now becomes budget-relevant rather than diagnostic.**
   If neurons must be replicated across experts to preserve behaviour, that inflates total weights and eats
   directly into the byte budget the MoE was adopted to save. **Report duplication as a weight-inflation
   multiplier, not only as a count.**

### A1.4 Standing caution, unchanged

The 44.6% figure was measured on **our own 8.3M model in our own engine**, not on a donor, and must not be
quoted as a donor number. **The structural argument — the gate is unskippable because it is the
predictor — is architecture-level and does transfer.** And the literature tally on co-activation clustering
remains **one negative (Apple), one positive (Neuralink), unreconciled**, with my own earlier miscount of
LLaMA-MoE already corrected in `STRUCTURED_SPARSITY_PRIOR_ART.md` §2. **This probe adjudicates an open
question; it does not confirm a settled one.**
