# Brief D5 — the ternary LUT rate at DONOR projection widths

**Author:** the Adapter / Principal · **Date:** 2026-08-21 · **Status:** pre-registered, awaiting dispatch
**Assigned to:** the Builder (one brief at a time — dispatch after D4 returns)
**Pre-registration rule:** this document is written and committed BEFORE the run. Nothing below may be
edited after numbers exist; corrections go in an appended section with a date.

---

## 1. Why this exists

`ADAPTER_MEMO_01_SPEED_BUDGET.md` §2.1 converts activation fraction into tok/s using **21.25 ± 0.36 GB/s**,
the engine-integrated ternary LUT rate. Every speed claim in the donor programme — including the
"~24 tok/s at 100B" that closes the budget — is that constant multiplied by an activation fraction.

**That constant was measured at our model width `D = 1536`.** Two facts make its transfer to donor width
an open question rather than a formality:

1. **The LUT path is recorded as compute-bound by ~16×, not bandwidth-bound.** A compute-bound rate is a
   property of the kernel's shape — its inner-loop length, its register pressure, its L3 residency — not
   of the DRAM. Widening `D` changes all three. A bandwidth-bound rate would transfer; this one has no
   reason to.
2. **This project has been caught twice in one day transplanting a constant across scale.** `B_block ≥ 22`
   was wrong by 5.33× at donor width (D0b), and the skippable ≈ 0.001 column was a single-density point
   spread across a range. §2.1 of the memo flags this constant as the third candidate. **We flag it before
   it bites rather than after.**

If the rate falls at donor width, every tok/s figure in the programme falls with it and the required
activation fraction tightens. If it rises, the target loosens. Either way the Owner cannot be shown a
tok/s number until this is measured.

## 2. The question, stated so it can only have one answer

> At the projection widths of real donor models, what is the sustained engine-integrated ternary LUT
> rate, in GB/s, as a function of `D` and thread count?

Not "is it still 21.25". The deliverable is a **curve**, `rate(D, threads)`, with the existing D=1536
point reproduced on it as a check.

## 3. Method

- Use the existing C benchmark path (`benchmarks/donor_adaptation/gemv_donor_bench.c` and the engine's
  `--kselftest` / recompilation-by-`-D` route already used for the projection-rate measurement). **Reuse
  the apparatus that produced 21.25 GB/s** — a different harness makes the comparison void.
- Sweep `D ∈ {1536, 2048, 4096, 5120, 8192}` and, separately, the real per-matrix shapes of a
  Llama-3-70B-class geometry (`d_model = 8192`, `d_ffn = 28672`): q/o `8192×8192`, k/v `1024×8192`,
  gate/up `28672×8192`, down `8192×28672`.
- Sweep threads over the range already used in the engine work, including `t6`.
- Report GB/s **and** the working-set size per point, so the L3 crossing is visible. The 16 MB L3 cliff
  (≈100 → 28 GB/s) is a banked measurement of this project; the curve must be read against it.

## 4. Pre-registered interpretation — fixed before any number exists

| outcome | reading |
|---|---|
| rate at `D = 8192` within ±10% of 21.25 GB/s | constant transfers; memo §2.1 caveat is discharged; tok/s table stands |
| rate falls below 21.25 | the memo's tok/s table is **optimistic**; required activation fraction tightens proportionally; the "4 attention layers of 80" allocation must be recomputed |
| rate rises above 21.25 | target loosens; **do not celebrate — check first whether the working set fell inside L3**, which would make the point non-representative of a 50 GB donor streaming from DRAM |

**The third row is the trap.** A wider `D` with a small number of rows can shrink the working set into
cache and produce a flattering rate that a real donor would never see. Every point must report its
working-set size, and any point that is L3-resident must be labelled as such and excluded from the
donor-budget curve.

## 5. Planted controls — the rate must be shown to MOVE before any flat result is believed

1. **Reproduce the banked point.** At `D = 1536` with the same thread count, the harness must return
   21.25 GB/s to within its stated ±0.36. If it does not, STOP — the apparatus has drifted and nothing
   else it says can be trusted.
2. **Known-positive: force a slowdown.** Introduce a deliberate, documented pessimisation (e.g. a stride
   that defeats sequential prefetch, or a working set forced past 16 MB) and show the measured rate
   drops sharply. **A rate meter that returns the same number under a change that must slow it down is
   not measuring throughput.**
3. **Known-positive: force a speedup.** Shrink the working set to be comfortably L3-resident and show the
   rate rises toward the ~100 GB/s regime the L3 probe banked. This bounds the meter from above.
4. **fp32 cross-check.** Measure the fp32 streamed projection rate at the same shapes. The banked value is
   **38.84 ± 0.68 GB/s** (independently audited at 39.87 ± 0.09). If the harness cannot reproduce that,
   its ternary number is not trustworthy either.

Controls 2 and 3 are the minimal-significant-corruption test: they establish that the instrument responds
in both directions before any flat or surprising reading is accepted.

## 6. Constraints

- **CPU only. Zero GPU.** This is an engine measurement.
- No `-ffast-math`. Standing repository rule.
- Record thread placement explicitly — `OMP_PLACES` / `OMP_PROC_BIND` and the thread count — in every
  result record. **The 16–32 MB points in `PHASE64_BUDGET.md` were found unreproducible precisely because
  thread placement was unrecorded and the numbers swung 4.6× on it.** Do not repeat that.
- Every record carries git revision, exact command line, compiler and flags, and the machine's state
  (confirm no heavy background processes).

## 7. Deliverables — then STOP

- `benchmarks/donor_adaptation/results/d5_lut_rate_donor_width.json`: the `rate(D, threads)` table, working
  set per point with an L3-resident flag, all four control outcomes, and the reproducibility manifest.
- A short section in `docs/research/DONOR_PROJ_RATE.md` (the existing home of the projection-rate work).
- Report to the Principal: the curve, which controls fired, whether the D=1536 point reproduced, and an
  explicit statement of whether memo §2.1's caveat is discharged, tightened, or loosened.

**Do not** update the memo's tok/s table yourself — that is the Principal's call once the Controller has
seen the result.

---

## AMENDMENT 1 — 2026-08-22, the Adapter / Principal

**Appended, not edited in place.** The pre-registration above stands as written and as pushed; this
records what was wrong with it and what replaces it.

### A1.1 I withdraw control 1's tolerance, and the reason is my error

Control 1 required reproducing the banked `21.25 ± 0.36 GB/s`. **I took the quoted uncertainty of a
banked number and used it as a tolerance, without checking it was achievable by the instrument in this
run's conditions.** The Builder's diagnosis measured the instrument's own between-run dispersion today at
**8.36%** — roughly ±2 GB/s. A ±0.36 window against that is not a drift detector; **it is a gate that
could not pass.** The `FAIL` it produced was information about my brief, not about the apparatus.

Recorded as a standing rule: *a tolerance is derived from the instrument's demonstrated precision under
the run's own conditions, never inherited from the quoted uncertainty of the value being compared to.*

### A1.2 The instrument defect the diagnosis uncovered, which is larger than D5

The harness reports **`min` of the per-rep times**, i.e. the **maximum** of the per-rep rates. That is a
maximum order statistic: **its expected value rises with the number of repetitions, by construction, with
no change in the underlying distribution.** Measured, same machine, same shape:

| reps | reported GB/s |
|---|---|
| 5 | 22.36 |
| 25 | 23.44 |
| 30 | 25.13 |
| **per-rep mean at 25 reps** | **21.57** |

> **This biases upward every number this harness has ever produced, by an amount that depends on the
> `reps` used for that number — including the banked 21.25 itself, and including §8 and §10 of
> `DONOR_PROJ_RATE.md`.** Numbers taken at different `reps` were never comparable to each other.

**And it very likely explains the whole discrepancy.** The per-rep **mean** at 25 reps is **21.57**,
against a banked **21.25** — a 1.5% difference, well inside normal between-session variation. **The mean
reproduces; the min does not.** That is the signature of an estimator problem, not apparatus drift.

### A1.3 What replaces control 1

**Primary estimator changes to the per-rep MEAN**, which is unbiased in `reps`. Control 1 becomes:

> Run the banked §10.2 protocol (`mlpint`, `OMP_PLACES=cores`, ≥4 independent invocations). Report, for
> each invocation: **`min`, `median`, `mean`, `sd`, `cv`, and `reps`** — all six, every time.
> **PASS** iff the **grand mean of the per-rep means** lands within **±3 × (the between-invocation sd of
> that same mean estimator, measured in this run)** of the banked 21.25 GB/s.

**This is not a weakened gate — it is a gate on a different, better-behaved statistic, and it can still
fail.** If the mean estimator does *not* reproduce 21.25, the estimator hypothesis is wrong and we are
back to a genuine drift question with one candidate explanation eliminated.

`min` stays in the output as a secondary column, solely so the new numbers remain relatable to the old
banked ones. **It may not be used as the reported rate for anything.**

### A1.4 A question the evidence raises that has not been answered

The four matched-protocol draws were **20.203 / 24.762 / 24.763 / 24.755**. The last three agree to
**0.018% — four significant figures.** The first is **18.4% below** them. That is not dispersion; it is
**two distinct machine states**, and three-of-four agreeing that tightly means the measurement is
extremely repeatable *once whatever it is has settled*.

**Determine whether invocation order predicts the rate** — i.e. whether the first invocation after a gap
is systematically low (cold pages, first-touch allocation, frequency/boost ramp) while subsequent ones run
warm. Run enough invocations to answer it, and report rate against invocation index.

**If a warm-up effect is confirmed, the protocol must state explicitly how it is handled** — a discarded
warm-up invocation, or a warm-up loop before timing starts — and that handling becomes part of every
number this harness reports from now on. **It must not be left to whether an operator happened to have run
it recently.**

### A1.5 The deliverable is re-scoped to ratios

D5's decision-relevant outputs are **comparisons**, not absolute rates: fp32-vs-ternary at matched shape,
`rate(D=8192)` vs `rate(D=1536)`, and 5-trit-vs-2-trit packing. **Ratios taken at a fixed `reps` are far
more robust to a biased-but-consistent estimator than absolute GB/s figures are.**

Therefore: **report every D5 result primarily as a ratio against a same-run, same-`reps` reference point,
with the absolute GB/s as a secondary column carrying its estimator and `reps` in the label.** `reps` and
invocation count are **pinned for the whole run** and printed in the header.

### A1.6 What remains undischarged

`ADAPTER_MEMO_01` §2.1's caveat on the 21.25 GB/s constant is **not** lifted by this amendment. It is now
worse-specified than before: the constant is known to be upward-biased by an unquantified amount that
depends on the `reps` behind it. **Until control 1 passes under A1.3, no tok/s figure anywhere in this
programme rests on a verified rate constant** — including every table in `ADAPTER_MEMO_01` §2.2, §2.2c and
§2.2d, all of which are explicitly conditional on it.
