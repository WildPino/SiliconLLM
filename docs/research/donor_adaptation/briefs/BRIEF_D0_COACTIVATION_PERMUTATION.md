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
