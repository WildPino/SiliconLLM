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
