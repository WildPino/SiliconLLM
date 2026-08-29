# CONTROLLER AUDIT — `DONOR_PROJ_RATE.md` §12.6 (commit `4c318d0`)

**Auditor:** Controller (independent adversarial review).
**Target:** `docs/research/DONOR_PROJ_RATE.md` §12.6, committed `4c318d0` (2026-08-29 01:42:30 +0200),
reversing §12.5 (`d419344`).
**Apparatus:** `benchmarks/donor_adaptation/gemv_donor_bench.c` modes `d5g`/`d5m`/`d5abl`/`d5cd`
(committed `6097131`).
**Artefacts:** `benchmarks/donor_adaptation/results/d5_auditresponse_raw.log`,
`benchmarks/donor_adaptation/results/a1_4_donor_order/invocation_{1..6}.log`,
`benchmarks/donor_adaptation/results/d5_raw.log`.
**Machine state — observed directly, not taken on report.** The audit was begun under the stated
constraint (`llama-server.exe` resident at ~18 GB, no timing runs permitted) and the verdicts below were
reached without any fresh measurement. I was subsequently told the machine was clear and timing runs were
permitted. **I checked, and it is not clear.** `llama-server.exe` is indeed gone, but
`python.exe f1_gate_predictor.py ann` (pid 26112, started 11:32:52) is running an active compute job with a
working set oscillating across the harness's own 1 GB quiescence threshold — sampled at 0.89 GB, 1.35 GB and
1.67 GB within about a minute. **No timing measurement was run for this audit**, and none should be run
until that job finishes. Every number below is derived from the committed artefacts, the committed source,
or arithmetic on them. See A12 for what that observation revealed about the gate itself, and the closing
section for which findings a genuinely clean run could still sharpen — the priority finding, A9, is not one
of them.

---

## VERDICT SUMMARY

| # | Claim | Verdict |
|---|---|---|
| A1 | The `Mpad` discriminator's premise (instructions per charged byte are shape-invariant) | **PASS** |
| A2 | §12.6.1's inference — a 2.12× charged-rate fall shows the path is memory/access-pattern-bound | **BLOCK** |
| A3 | Robustness of the spread to dropping noisy rows (2.02× / 1.91×) | **FLAG** (arithmetic error; estimator choice is the wrong robustness axis) |
| A4 | The ablation "strips the compute, not the memory traffic" | **BLOCK** (it removes one of two loads per iteration) |
| A5 | "Compute is ~16% / ~27% of runtime" | **FLAG** (not derivable from a jump ratio on an out-of-order core) |
| A6 | `d5cd` PASS as the control closing the audit's coverage gap | **BLOCK** (run in the one regime the disputed claim does not live in; structurally blind to the defect §12.6.4 concedes) |
| A7 | A1.4 grand mean 9.71 GB/s, no order effect, fall 2.27×–2.34× | **PASS** on 9.71 and 2.27×; **BLOCK** on 2.34× (unsupported citation); **FLAG** on "no order effect" |
| A8 | The Claim-1 exemption from the byte-accounting defect | **PASS**, narrowed — real for the tok/s arithmetic, false for the mechanistic gloss §12.6.7 attaches to it |
| A9 | "Ablated gate/up reaches only 39.7% of ceiling vs 85.9% at k/v → the access pattern caps the wide shape" | **BLOCK** — the two percentages are computed against non-comparable ceilings, exactly as suspected |
| A10 | Provenance of `results/a1_4_donor_order/` (promoted from outside the tree) | **PASS** on format/content/internal consistency; **FLAG** on commit ordering and on the un-named revision |
| A11 | Process signature — does §12.6 lean the same way the last round did? | **FLAG** — one specific, load-bearing omission, described below |
| A12 | The machine-quiescence gate as a guarantee that a run was uncontended | **FLAG** — it is a one-shot sample taken before the run, and cannot see contention that arrives or oscillates during it |

---

## A1 — **PASS.** The premise survives. Instruction count per charged byte *is* invariant across the `d5m` sweep.

The Principal asked me to attack the premise itself. It holds, and I could not break it.

`matvec_lut_full` (`gemv_donor_bench.c:135-144`) is parameterised on `M`, `Mpad`, `T` **at runtime**. There
is one function body; no shape specialisation is possible at compile time, so the instruction *sequence* is
identical at every point of the sweep. Per inner iteration it consumes exactly 32 bytes of `codes` and
executes a fixed sequence (one 16-B `lut` load, one 32-B `codes` load, broadcast, `pshufb`, `extracti128`,
2×`psrldq`, 4×`cvtepi8_epi32`, 4×`vpaddd`).

The one place this could break is if charged bytes `EB = T·Mpad` diverged from bytes actually loaded by the
loop, `M·T`. In `mode_d5m` (`:1162-1172`) `M` is set to the target `Mpad` and every value
(1024/2048/4096/8192/28672) is a multiple of 32, so `Mpad = (M+31)&~31 = M` and **`EB` is exactly the bytes
the loop loads**, at every point. Per-block overhead (accumulator zeroing, the 32-lane store, the `y`
write-back) amortises over `T·32 = 131072` bytes and is likewise invariant. LUT traffic is `M·T/2 = EB/2`
at every point — a constant fraction.

I also checked the two confounds that could have produced a shape-dependent *effective* rate at fixed
instruction mix:

- **OpenMP load imbalance.** `OMP_PFOR` over `base` with step 32 and `schedule(static)` gives `M/32` =
  32/64/128/256/896 iterations across 6 threads → utilisation 88.9%/97%/97%/99.2%/99.6%. Imbalance is
  **worst at the fastest point**, so it cannot manufacture the fall; if anything it understates it.
- **Frequency droop.** Not excluded — see A3.

**Conclusion:** a purely ALU-throughput-limited `matvec_lut_full` must report flat *charged* GB/s across
`d5m`. It does not. **"ALU-throughput-bound" is dead, and §12.6 is right to bury it.** That is the extent
of what §12.6.1 establishes. What it does *not* establish is A2.

**Reproduction:** `gemv_donor_bench.c:135-144` (kernel), `:1162-1172` (`mode_d5m`, `Ms[5]`), `:1096`
(`Mpad=(M+31)&~31`, `EB=T·Mpad`), `:1121` (`gmean = EB/1e3/m`).

---

## A2 — **BLOCK.** The 2.12× fall is quantitatively the charged-vs-moved artefact §12.6.4 concedes on the same page. Restated in moved bytes, the sweep is flat, and the discriminator returns the *other* pre-registered branch.

This is the central finding of this audit.

§12.6.4 states, correctly and in its own words, that `matvec_lut_full` "reads 32 of every 64 bytes in a
cache line at stride `Mpad` and is charged exactly `EB` bytes regardless of how much DRAM traffic that
pattern actually generates," that the undercount factor is between 1× and 2×, and that it **grows with
`EB`**. §12.6.1's table is then reported, interpreted, and made the headline entirely in charged bytes,
with the correction never applied.

Apply it. The mechanism the document itself endorses predicts the factor from residency alone: a block that
stays resident across one `base`-pass gets its second half from cache (f ≈ 1); a block that does not gets
the line refetched (f ≈ 2). `L3_BYTES` is 16 MiB (`gemv_donor_bench.c:960`).

| `Mpad` | `EB` | resident across a pass? | charged mean | **moved mean (f)** | charged min | **moved min** |
|---|---|---|---|---|---|---|
| 1024 | 4 MiB | yes | 22.73 | **22.73** (1.0) | 28.76 | **28.76** |
| 2048 | 8 MiB | yes | 21.61 | **21.61** (1.0) | 22.45 | **22.45** |
| 4096 | 16 MiB | at the boundary | 15.26 | **21.36** (1.4) | 17.08 | **23.92** |
| 8192 | 32 MiB | no | 11.29 | **22.59** (2.0) | 11.73 | **23.46** |
| 28672 | 112 MiB | no | 10.72 | **21.45** (2.0) | 11.14 | **22.28** |

**In moved bytes the sweep is flat at 21.4–22.7 GB/s across a 28× change in `Mpad`** — a ±3% spread, i.e.
the pre-registered outcome the run itself printed as *"if mean_gbps is flat (±few %) across this row set,
compute-bound survives"* (`gemv_donor_bench.c:1170`).

The 1.4 at the boundary row is the one interpolated value and it is not load-bearing: at f = 1.0 the
sequence is 22.73 / 21.61 / 15.26 / 22.59 / 21.45 and at f = 2.0 it is 22.73 / 21.61 / 30.51 / 22.59 /
21.45. **Under every value of f in the range the document itself endorses, the monotonicity is gone and the
2.12× fall is gone.** The residual is one non-monotonic point at exactly the L3 boundary.

Three consequences:

1. **The discriminator does not discriminate.** The pre-registered dichotomy ("flat → compute-bound
   survives; falls → Claim 2 dead") is not exhaustive. A third branch — *charged rate falls, moved rate
   flat, because the accounting bias tracks residency* — was already on the record when the dichotomy was
   written (see A11), and it is the branch the data lands in. §12.6.1's headline is therefore a second
   measurement of the half-line waste, not independent evidence about what binds the kernel.
2. **The "cliff-then-plateau" shape is the artefact's signature, not corroboration.** §12.6.1 offers
   "a cliff at the L3 boundary followed by a plateau … consistent with a residency/traffic mechanism" as
   support. It is: the residency transition is *precisely* what flips f from 1 to 2. The shape is what the
   accounting bias predicts, so it cannot also be evidence for a separate mechanism.
3. **A flat ~22 GB/s of moved bytes does not restore "compute-bound" either.** 22 GB/s is well under the
   ~38.7 GB/s sequential fp32 figure, and A1 independently kills ALU-throughput-bound. The honest reading
   is that the kernel sustains a roughly constant real DRAM rate that is shape-independent, and that the
   *charged* rate degrades because line utilisation collapses past L3. That is a stronger statement than
   §12.6 makes, and it happens to be one the prior Controller round already made (A11).

**What would settle it, and what cannot be settled here.** Whether f is really 1→2 and not some other
shape-dependent function requires **direct DRAM-traffic measurement via hardware performance counters**,
which nothing in this userspace harness can do — the prior round already flagged this as the one
unreachable check, and §12.6.4 restates it. **FLAG — not resolvable with this harness at any machine
state.** But the burden runs the other way: §12.6 reports a number it has
declared biased by an unknown factor of 1–2× that tracks the swept variable, and draws a mechanistic
conclusion from it anyway. The correction cannot be omitted merely because its exact value is unknown.

**Reproduction:** `results/d5_auditresponse_raw.log`, the five `D5A1CSV,d5m,Mpad_sweep` rows; multiply
`mean_gbps` by f = 1 for `EB` < 16 MiB and f = 2 for `EB` > 16 MiB; compare against §12.6.4's own paragraph
beginning "This section's own two measurements now show the charged rate is NOT a stable multiple of moved
bytes across shapes."

---

## A3 — **FLAG.** The spread is real and if anything understated, but §12.6.1's robustness argument tests the wrong axis, and one of its two ratios is arithmetically wrong.

Verified from the raw rows. Three separate defects, none of which change the direction:

1. **"1.91× fall" is wrong; it is 2.02×.** §12.6.1: *"dropping it too leaves 21.61 -> 11.29 -> 10.72, a
   1.91x fall on three of the cleanest rows."* 21.61/11.29 = 1.914, but the listed set ends at 10.72:
   21.61/10.72 = **2.016**. The quoted ratio silently drops the last listed point. (Anti-flattering — it
   understates.)
2. **"all three of those remaining points have `cv` under 5%"** — after dropping `Mpad=1024` there are
   **four** remaining points, and one of them (`Mpad=4096`, cv 9.77%) is over the line. The sentence
   miscounts its own table.
3. **The robustness axis is estimator choice, not row exclusion — and it was not tested.** The `Mpad=1024`
   row's `cv=24.15%` means its *mean* is contaminated by a stall, not that its shape is untrustworthy. Its
   own `min` is 28.76, and the **identical configuration** re-run in `d5abl` (M=1024, K=8192, pool
   536870912, reps=8, same kernel) reads min 29.15 — **agreeing to 1.4%** while the means differ by 22%
   (22.73 vs 27.81). So the top-of-sweep is ~28.8, and the honest charged fall is 28.76/11.14 = **2.58×**,
   not 2.12×. Dropping the noisy row *understates* the fall; §12.6.1's framing of that exclusion as
   conservative is backwards.

**A genuine, uncosted confound the write-up does not mention:** the sweep is *not* run at a fixed pool.
`pool_bytes` is 536870912 on four rows and **469762048 (448 MiB) on the `Mpad=28672` row** — `nblk` falls
128 → 64 → 32 → 16 → **4**, and the harness's `touch` floor (`gemv_donor_bench.c:1105`) makes that row
stream **7 GiB per rep against ~2 GiB for every other row**, running roughly 7× longer under sustained AVX2
plus DRAM load. §12.6.1's "pool fixed at the same target (512 MB) across every point" is false as written,
and neither the pool composition nor the sustained-load/frequency-droop confound is bounded. It does not
overturn the fall (dropping that row leaves 22.73 → 11.29 = 2.01× on genuinely fixed-pool rows), but the
sweep is not the single-variable experiment it is described as.

**Reproduction:** `results/d5_auditresponse_raw.log`, compare `pool_bytes` across the five `d5m` rows and
against `D5A1CSV,d5abl_real,kv_1024x8192`; `gemv_donor_bench.c:1100-1105`.

---

## A4 — **BLOCK.** The ablation does not keep the loads. It removes one of the two loads per iteration.

§12.6.2 describes `matvec_ablation` as *"identical load … strips the compute, keeps the memory traffic and
access pattern unchanged,"* and the source comment (`gemv_donor_bench.c:1067-1072`) says *"Reads the
IDENTICAL 32 bytes per t from the IDENTICAL address … strips the compute, not the memory traffic."*

The real kernel's inner body issues **two** loads per `t`:

```
__m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut + t*16)));   // 16 B
__m256i idx = _mm256_loadu_si256((const __m256i*)(codes + t*Mpad + base));                 // 32 B
```

`matvec_ablation` (`:1073-1084`) takes `(void)lut;` and issues **one**. The DRAM traffic is indeed
unchanged (the LUT is `T·16` = 64 KB, L2-resident), but **load-port occupancy is halved** — from 2
loads/iteration to 1, on a core that retires 2 loads/cycle. The ablation therefore conflates "remove the
decode ALU work" with "halve the load issue rate."

This matters because the write-up's whole argument is that the *access pattern*, not the decode logic, is
the constraint — and the ablation it uses to isolate that changes the load stream. The two shapes' jumps
(1.195× and 1.362×) cannot be attributed to ALU removal alone. The direction of the finding ("stripping
decode did not buy much") is probably safe, since removing half the loads *and* ~90% of the ALU ops bought
only 1.2–1.4×. But the mechanism attribution in §12.6.2's last sentence is not licensed by this apparatus.

**Fix path (not implemented, per role):** keep the `lut` load and discard its result, or replace it with an
equally-sized dummy load from the same address, so only the `pshufb`/`cvtepi8_epi32`/`vpaddd` chain is
removed. Requires a fresh run on a quiet machine; see the closing section, item 1 — this is the single
re-run I would commission first. **FLAG — open.**

**Reproduction:** `gemv_donor_bench.c:138-139` vs `:1077-1079`.

---

## A5 — **FLAG.** "Compute accounts for ~16% / ~27% of runtime" is not derivable from a jump ratio.

The arithmetic is right (1 − 1/1.195 = 16.3%, 1 − 1/1.362 = 26.6%), though the stated formula
("`ablated_time/real_time = 1/jump`") describes a different quantity than the percentages quoted.

The inference is not. `1 − 1/jump` equals compute's share of runtime **only under strict serialisation of
compute and memory**. On an out-of-order core with memory-level parallelism, compute that fully overlaps a
memory stall contributes 0 to the jump while occupying an arbitrary share of issue slots. So the measured
jump is a **lower bound** on compute's share, not an estimate of it.

The direction is safe (a lower bound of 16–27% is consistent with "compute is not the binding term"), and
the bias runs against the write-up's own conclusion, so this is a FLAG on rigour rather than a BLOCK on the
result. But §12.6.7 item 1 then leans on the number as a *ceiling* — *"The ablation's ~16-27% compute-time
estimate is a ceiling on how much this kernel's decode costs"* — which is exactly the inverted reading. It
is a floor, not a ceiling.

---

## A6 — **BLOCK.** `d5cd` can fail in principle, but not in the regime it is offered as covering, and it is structurally blind to the defect §12.6.4 concedes.

Three separate problems.

**It is run at the one donor shape chosen *because* it is not in the disputed regime.** §12.6.3 says the
control is needed because *"Claim 2's whole evidence base lives"* at donor shapes. Claim 2's evidence base
is the **streamed, past-L3** regime — `gate/up` at `EB` = 112 MiB, `q/o` at 32 MiB, the `D=8192` sweep
point. `d5cd` runs `k/v` at `EB` = 4 MiB, selected in the source comment (`:1140-1142`) precisely because
it *"fits comfortably under the 16 MiB L3 cliff."* The coverage gap the audit identified is over the
past-L3 regime; `d5cd` covers the pre-L3 regime. **The gap is not closed.**

**It cannot detect the bias it is offered as reassurance against.** At `EB` = 4 MiB both arms have f ≈ 1:
the resident condition trivially, and the streamed condition because each 4 MiB block is fully reused
across the 32 `base`-passes within a single touch before the next block is drawn. So the charged/moved bias
is ≈ 1 in *both* directions of this control. A meter that undercounts by 2× at 112 MiB would PASS `d5cd`
unchanged. Calling it *"the first control … that validates the kernel-pure meter moves correctly, in both
directions"* overstates what a same-f comparison can validate.

**Its thresholds are inherited, and the estimator is not.** `mode_d5cd` (`:1146-1161`) is a literal clone of
`d5_control23`'s pass logic — `drop ≥ 2.0×`, `resident ≥ 50` — which is *good* (no post-hoc tuning). But
`control23` computes those from `lut_one_point`'s **min** estimator (`g[1]`), while `d5cd` applies them to
`lut_probe_generic`'s **mean**. The shift is conservative, so this is a note, not a defect. Separately,
`≥2.0×` is a loose bar for an L3-vs-DRAM contrast, and the resident reading (71.71) is 28% under the
harness's own stated target of ~100 GB/s while still printing PASS against the 50 floor.

**Can it fail?** Yes — a meter that timed nothing memory-resident, or a pool that did not exceed cache,
would return ≈1×. So it is not vacuous. A control that would close the actual gap is described in the
closing section, item 2; it needs a synthetic shape pair, not a re-run of this one. But it is a control on the *plumbing*, not on the *accounting*, and
§12.6.3 sells it as the latter.

**Reproduction:** `gemv_donor_bench.c:1140-1161` vs `:1236-1252` (`d5_control23`);
`results/d5_auditresponse_raw.log`, the two `D5A1CSV,d5cd_*` rows (`pool_bytes` 4194304 vs 536870912,
`EB` 4194304 in both).

---

## A7 — **PASS** on 9.71 and 2.27×. **BLOCK** on 2.34×. **FLAG** on "no order effect."

**The arithmetic checks out.** From `invocation_{1..6}.log`, `mean_gbps` = 10.7394, 9.6441, 10.0445,
8.6542, 8.8232, 9.6660.

| quantity | §12.6.6 | recomputed | verdict |
|---|---|---|---|
| six-invocation mean | 9.60 | 9.5952 | ✓ |
| six-invocation sd | 8.07% | 8.07% (sample sd) | ✓ |
| seven-invocation grand mean (+10.38) | **9.71** | **9.7073** | ✓ |
| seven-invocation sd | 7.90% | 7.90% (sample sd) | ✓ |
| cv-clean subset (1,4,5) | 9.41, sd 12.3% | 9.4056, sd 12.31% | ✓ |
| fall vs 22.07 | 2.27× | 2.2735 | ✓ |
| fall vs 22.69 | 2.34× | 2.3374 | arithmetic ✓, **input unsupported** |

**BLOCK — "this session's own `D=1536` reading (22.69, sec.12.6 manifest)" does not exist.** Three checks:

- §12.6's manifest (lines 1198-1203) contains no rate at all, only file paths and build/env pointers.
- **No `D=1536` measurement exists in this session's artefacts.** `results/d5_auditresponse_raw.log` runs
  only `d5m`/`d5abl`/`d5cd`, all at `K=8192`; all six `a1_4_donor_order` logs are `D=8192 HID=28672`. The
  only `D=1536` rows in the repo are in `results/d5_raw.log`, from the 2026-08-22 session.
- **`22.69` occurs exactly once elsewhere in the document, at line 1065** — as §12.5.4's *ternary
  Gweights/s at `D=8192`*. Different quantity, different unit, different `D`.

Whatever its origin, the citation is unverifiable, the section pointer is wrong, and the number collides
verbatim with an unrelated cell. It is one of the two denominators of a headline range, and it is the one
that makes "roughly 2×" round cleanly from both ends. Under the project's own standing rule
(`feedback_literature_fabrication`: a number does not enter a decision until it has been read in its own
source table) this is BLOCK-grade. **The only supported statement is "2.27×", from 22.07.**

**FLAG — "no monotonic order effect" tests one direction only, at n=6.** §12.6.6's argument is that a
cold-start signature predicts invocation 1 *low* and it is *high*, therefore no order effect. That refutes
cold-start; it does not test order. The series 10.74 / 9.64 / 10.04 / 8.65 / 8.82 / 9.67 has first-three
mean 10.14 and last-three mean 9.05 — an 11% **decline**, i.e. a warm-degradation signature, which is the
order effect the stated test is blind to. Spearman ρ = −0.486 on n = 6 (p ≈ 0.33, two-sided): not
significant, so the conclusion is *defensible*, but it is defensible because the test has essentially no
power, not because the effect was excluded. The write-up asserts absence rather than reporting
non-detection. Six invocations at 8% dispersion cannot separate a 10% drift from noise.

**Note, not a finding:** the write-up's honesty about the cv-clean subset being *noisier* (12.3% vs 8.07%)
is exactly the kind of self-undermining detail whose presence I looked for and found. Credit where due.

---

## A8 — **PASS**, narrowed. The Claim-1 exemption is real for the tok/s arithmetic and false for the gloss §12.6.7 puts on it.

The Principal asked whether the exemption is real. It is, for the one use it is claimed for, and the prior
Controller round already granted it on the record (`CONTROLLER_D5_RESULTS_AUDIT.md`, "One thing that does
*not* bite here").

`ADAPTER_MEMO_01`'s tok/s arithmetic is `time_per_token = charged_bytes_per_token / charged_rate`. Code
bytes per token is a property of the packed model, and the measured rate is a true charged rate.
**Wall-clock time per token is correct regardless of DRAM traffic**, at both `D`. The exemption holds.

But §12.6's justification for it is wrong, and the error is worth recording because it will be reused.
The stated reason is *"both the `D=1536` and `D=8192` rates … use the identical charged-byte convention on
both sides of that multiplication."* Identical *convention* is not sufficient — the convention's *bias*
must also match, and §12.6.4's own mechanism says it does not: at `D=1536`/`HID=8960` the gate/up block is
≈6.9 MB (f ≈ 1) and at `D=8192`/`HID=28672` it is 112 MiB (f ≈ 2). **The two endpoints of Claim 1's ratio
sit on opposite sides of exactly the shape-dependence being conceded.** The exemption survives not because
the biases cancel — they do not — but because tok/s never divides by moved bytes at all.

The consequence §12.6 does not draw: **most of Claim 1's ~2× fall is the accounting artefact, i.e. it is
self-inflicted and fixable, not a property of donor width.** So §12.6.7's revised answer is half right and
half not:

- *"the 21.25 GB/s constant … falls to roughly half"* — **true as wall-clock**, and it is the number the
  memo needs.
- *"the donor-width path is memory-bound … access-pattern-bound"* as an explanation of that fall —
  **inherits A2 in full.** In moved bytes the machine delivers the same ~22 GB/s at both widths.

---

## A9 — **BLOCK.** 39.7% and 85.9% are computed against ceilings that are not comparable. Corrected, the gap is ~6 points, not ~46. This is the claim the Principal wanted broken, and it breaks.

§12.6.2's ceiling column is the write-up's own "more telling number," and it is the load-bearing input to
the "free ~2× in our own kernel" hope. It is a **charged-byte numerator over a moved-byte denominator**,
with the numerator's bias differing by ~2× between the two rows.

- Denominator: `fp32_streamed_point` (`gemv_donor_bench.c:1304-1335`) reads `W` row-major and contiguously
  via `matvec`/`dotf`. Charged `wb = M·K·4` = moved bytes exactly. ~38.7 GB/s.
- Numerator: `lut_probe_generic` charged `EB`, biased 1× at `k/v` (`EB` 4 MiB) and ~2× at `gate/up`
  (`EB` 112 MiB) — the document's own §12.6.4 figure.

| shape | `EB` | ablated charged | **ablated moved** | % of 38.7 as published | **% of 38.7, corrected** |
|---|---|---|---|---|---|
| `k/v` | 4 MiB | 33.22 | 33.22 | 85.9% | **85.8%** |
| `gate/up` | 112 MiB | 15.38 | ~30.75 | **39.7%** | **~79.5%** |

**There is no wide-shape anomaly.** Corrected, both shapes sit at ~80–86% of the sequential figure with the
same stripped kernel — a 6-point gap fully attributable to `k/v`'s residual L3 assistance, which §12.6.2
already invokes. The inference *"Something in the access pattern itself — not the decode logic — is capping
it well below what this memory subsystem can deliver sequentially"* is **the half-line waste, counted a
second time**, not an additional constraint sitting on top of it.

Two further non-comparabilities, both smaller and both in the same direction:

- **Estimator mismatch.** The ceiling percentages divide a **mean**-estimator ternary rate
  (`lut_probe_generic`, `:1121`) by a **min**-estimator fp32 rate (`fp32_streamed_point`, `:1328`, `best`).
  On the ablated rows min/mean is 1.068 (`k/v`) and 1.023 (`gate/up`); like-for-like on mins the published
  numbers become 91.7% and 40.6%, and corrected-and-like-for-like, 91.7% vs 81.3%.
- **Access-pattern mismatch in the ceiling itself.** 38.7 GB/s is a hardware-prefetched sequential stream.
  An 8 KB-stride demand-miss walk is limited by outstanding-miss concurrency, not by prefetched streaming
  throughput. "% of the sequential fp32 ceiling" is not a headroom measure for this kernel under any
  correction. §12.6.2 half-acknowledges this by writing "sequential," then reasons as if the shortfall were
  headroom.

**What survives, and it is worth stating because it is the actionable part.** The proposed fix in §12.6.4 —
consume `base` and `base+32` together while the line is hot — is still the right move, and it is worth
roughly **2× on the charged rate at past-L3 shapes** (11.29 → ~22, per A2's table). But:

- it is **not additional** to the accounting artefact — it *is* the accounting artefact;
- it is **bounded by the ~22 GB/s moved-byte plateau**, i.e. ~2.0×, not the 2.16× that 39.7 → 85.9 implies;
- and it buys nothing at `k/v`-class shapes, which are already at f ≈ 1.

So: the "free ~2×" is real as a tok/s lever at gate/up-class shapes, and the evidence offered for it in
§12.6.2 is double-counted. Those are two different statements and §12.6 conflates them.

**Reproduction:** `results/d5_auditresponse_raw.log`, `D5A1CSV,d5abl_ablated,*` rows; `DONOR_PROJ_RATE.md`
line 1047 for the 38.71 figure; `gemv_donor_bench.c:1328` (fp32 `best`/min) vs `:1121` (ternary `gmean`).

---

## A10 — Provenance of `results/a1_4_donor_order/`. **PASS** on format and internal consistency; **FLAG** on commit ordering.

The Media Manager disclosed in `4c318d0` that this directory was promoted from outside the working tree.
Audited on four axes.

**Format — PASS.** All six files are byte-for-byte structurally identical to `mode_d5g`'s output: the
banner block from `main`, the quiescence stanza (`gemv_donor_bench.c:102-110`, ending `-- clean.`), the
`==== d5g: generic six-stat shape probe (D=8192 HID=28672 L=28 nt=6 reps=5) ====` header emitted by
`:1198`, one `D5A1CSV,d5g,…` row with the field order and `%.4f`/`%.2f` precisions of `:1204-1206`, and the
trailing `STOP. No commit, no push.` No field is absent, reordered, or differently formatted. These were
not hand-composed.

**Content — PASS, and this is the strong check.** Every log reports `ws_bytes=9865003008`. That value is
not free text; it is `S.codes_layer · L` from `:1200-1201`, with
`codes_layer = TUP·Mpg·2 + TDN·Mpd` (`:701`). For `D=8192, HID=28672, L=28`:
`TUP = D/2 = 4096`, `Mpg = 28672`, `TDN = HID/2 = 14336`, `Mpd = 8192` →
`(4096·28672·2 + 14336·8192)·28 = 352321536·28 = **9,865,003,008**`. Exact match. `ntok_per_rep=8` is
likewise the `:1203` floor for a 9.87 GB working set. The logs are consistent with the committed harness at
the stated shape to the byte.

**Statistical plausibility — PASS.** The six means (8.65–10.74) bracket the independent `d5g` draw of
10.38 from the previous session and sit on the same plateau as `d5m`'s `Mpad=8192`/`28672` rows
(11.29/10.72) and `d5abl_real,gateup` (11.29) — the engine-integrated figure being modestly below the
kernel-pure one, as §10 already establishes. Nothing in the six is anomalous relative to independently
committed measurements.

**Timestamps — FLAG, non-fatal.** All six carry the identical mtime (2026-08-29 01:33), which is consistent
with a batch copy and *inconsistent* with six sequential in-place runs. That is exactly what was disclosed,
so it corroborates rather than contradicts the MM's account. But it means mtimes carry **no** independent
evidence of when the runs occurred, and there is nothing else in the artefacts that dates them. The claim
that they were produced "during the same clear-machine window as sec.12.6.1-12.6.3" rests on testimony
alone. Two internal details are mildly discordant with a single window: the `a1_4` runs use `reps=5` while
the same window's `d5m`/`d5abl`/`d5cd` all use `reps=8`, and the audit-response log is a single
concatenated background job while these are six separate invocations. Neither is impossible; neither is
corroboration.

**Commit ordering — FLAG.** `d05fe98` (01:36:25) commits `d5_auditresponse_raw.log`; `6097131` (01:41:24)
commits the harness that produced it. **The results were committed five minutes before the apparatus.**
Combined with §12.6's manifest reading "git revision at time of these runs" **without naming a revision**,
there is no commit that pins the source state at run time — no revision containing this harness existed
when the runs were made. Not evidence of anything wrong, but the manifest does not do the job a manifest
exists to do. Reproduction should be pinned to `6097131`.

**Verdict: the artefacts are what they claim to be.** I can reproduce every derived number in §12.6.6 from
them, and their internal arithmetic matches the committed harness exactly. The only thing testimony
carries alone is *when* they were run, and that does not bear on any conclusion in §12.6.

---

## A11 — **FLAG.** One omission, and it is the specific one that would have neutered the headline.

The standing rule says a result that flatters earns more scrutiny, and that the prior round's defects all
leaned the same way. §12.6 is, on the whole, **markedly better disciplined than §12.5**: it reverses the
Principal's own prior conclusion, it self-reports the byte-accounting defect without being pushed a second
time, it flags the 15% inter-invocation gap it could have smoothed, it reports the cv-clean subset being
*noisier* than the full set, it withdraws "block size drives the rate," and it refuses to price 5-trit
rather than reversing into a positive. Several of those cut against its own thesis. I looked for the
last round's signature and mostly did not find it.

One thing is not explicable that way.

`CONTROLLER_D5_RESULTS_AUDIT.md` finding 3 did not merely identify the 1–2× accounting bias. **It applied
it**, in a table, and stated the result: *"Applying the factor to the streamed points collapses the spread
that finding 2 identifies … its actual DRAM byte rate is flat at ~21-23 GB/s."* §12.6.4 cites that finding
by name — *"plausibly the ~2x the audit's finding 3 proposed"* — and then §12.6.1 and §12.6.2 report their
headline numbers uncorrected, without the correction being applied, mentioned, or given a sensitivity
range. The `Mpad` sweep, corrected, **reproduces finding 3's table on new data** (A2). That is a genuine
and useful result. It is reported instead as a 2.12× fall that kills compute-bound, and as a 39.7%
ceiling shortfall that points at a new constraint.

The pre-registration is affected by the same gap. §12.6.1 says the read was *"printed into the run itself,
before the numbers existed."* It is compiled in, but the `D5NOTE` prints **after** the sweep loop
(`gemv_donor_bench.c:1170`, and last in the log), and — more to the point — the dichotomy it registers
("flat → compute-bound survives; falls → dead") was written *after* the same auditor had already put the
third branch on the record. **A pre-registration whose branches were known to be non-exhaustive when it was
written cannot license the conclusion it returns.**

I am not calling this the last round's failure mode — the direction of §12.6's overall reversal argues
against that. I am calling it a single load-bearing omission that runs in the flattering direction, on the
one correction the write-up demonstrably had in hand.

---

## A12 — **FLAG.** The `-- clean.` banner on every artefact in this section certifies less than it appears to. The quiescence gate is a single sample taken before the run.

This finding came out of checking the machine state myself rather than accepting it as reported, and it
applies to §12.5 and §12.6 equally.

`quiescence_gate` (`gemv_donor_bench.c:97-119`) is called **once**, from `main` at `:1462`, before mode
dispatch. It enumerates processes, compares each working set against `QUIESCENCE_THRESHOLD_BYTES` = 1 GB
(`:64`), prints `-- clean.` if none qualify, and is never consulted again. The timed work then runs for
seconds to minutes — the `d5m` sweep's `Mpad=28672` row alone streams ~7 GiB per rep across 8 reps, and each
`a1_4` invocation streams ~9.87 GB per rep across 5 — with no further check.

So the banner certifies *"no process was over 1 GB at the instant the binary started."* It does not
certify that the run was uncontended. Three gaps, all live:

- **Contention arriving mid-run** is invisible. The gate cannot retract a `-- clean.` it already printed.
- **An oscillating working set can walk under the bar at exactly the wrong moment.** The job I observed
  today is a working example: sampled three times in about a minute it read 0.89 / 1.35 / 1.67 GB. Started
  during one of its dips, the harness would print `-- clean.` and then measure against a job contending for
  cores and memory bandwidth throughout. **A banner that says clean on a contended run is worse than a
  refusal**, because it launders the number.
- **Working set is the wrong proxy for the thing being measured.** These are memory-bandwidth
  measurements. A process with a 400 MB working set saturating DRAM passes the gate untouched; five Chrome
  processes at ~500 MB each, as are running now, never register.

This does not invalidate any artefact — I have no evidence that any of these runs was contended. But it
does mean the `-- clean.` line is not the corroboration the write-ups lean on it as, and it is the most
economical available explanation for two things this audit already had to work around: the 22% spread
across three same-invocation-config replicates in A3 (22.73 / 27.81 / 24.16, one of them at `cv` 24.15%
while its own `min` agreed with the others to 1.4%), and the 8-12% between-invocation dispersion §12.6.6
reports as unexplained. §12.6.6 closes by saying whether that dispersion "has a cause (thermal, background
OS scheduling, something else) is not established." A gate that samples once at startup cannot rule out the
second of those, and the write-up treats it as though it had.

**Reproduction:** `gemv_donor_bench.c:97-119` (the gate body — note it returns, and is not re-entered),
`:1462` (the sole call site, before mode dispatch), `:64` (the 1 GB threshold on working set).
Compare against the timed loops at `:1108-1119`, which take no quiescence argument.

---

## WHAT §12.6 GETS RIGHT (recorded so the withdrawals are not over-read)

1. **ALU-throughput-bound is dead** (A1), and the ablation supports it independently of A4's defect.
   §12.5's Claim 2 should stay withdrawn.
2. **The `Mpad` sweep kills the rival LUT-footprint hypothesis** that the prior audit's finding 10 raised
   (`T ≥ 4096` → 8.9-11.3 GB/s). `d5m` holds `T = 4096` — LUT = 64 KB — at every point and still reads
   22.7-28.8 GB/s at small `Mpad`. That is a real contribution the write-up does not claim.
3. **The `q/o` 8.92 outlier is correctly retired** (§12.6.5), on a properly repeated measurement at the
   matching condition. Finding 10's second point is discharged.
4. **The access-pattern fix is the right next move**, and prioritising it over building a 5-trit decoder is
   the correct call — on A9's corrected arithmetic even more clearly than on the published version.
5. **Refusing to price 5-trit** rather than reversing into a positive is the right restraint.

---

## WHAT A FRESH RUN COULD AND COULD NOT SETTLE

Recorded against the possibility that the machine becomes genuinely quiet and someone wants to close the
open items. Nothing here reopens a verdict; the verdicts above were reached from the artefacts and stand.

**A9 — the priority finding — cannot be sharpened by any re-run, and no re-run should be commissioned for
it.** This is worth stating plainly because it is counter-intuitive. A9 is not a dispute about whether
39.7% and 85.9% were measured correctly; I accept both as accurate readings of what the harness computed.
It is a dispute about **what they are ratios of** — a charged-byte numerator over a moved-byte denominator,
with the numerator's bias differing ~2× between the two rows. Re-running `d5abl` on a silent machine
produces the same two quantities in the same two incompatible units, to tighter error bars. **Tighter error
bars on a unit mismatch do not address a unit mismatch.** The finding is closed as it stands.

What *would* settle the residual uncertainty inside A9 — the exact value of the correction factor f, as
opposed to its existence and its direction — is **direct DRAM traffic measurement via hardware performance
counters**, which nothing in this userspace harness can reach at any machine state. The prior Controller
round flagged this as the one unreachable check and §12.6.4 restates it. My A2/A9 corrections are therefore
a sensitivity analysis across the 1-2× range the document itself endorses, not a measurement — and the
point of A2 is that the conclusion flips anywhere in that range, so the exact value is not load-bearing.

Genuinely open, in descending order of value, each needing a quiet machine:

1. **A load-preserving ablation** (A4). Retain the `lut` load and discard its result, so only the
   `pshufb`/`cvtepi8_epi32`/`vpaddd` chain is removed. This separates "remove the decode ALU work" from
   "halve the load issue rate," which the committed `matvec_ablation` conflates. Cheap, and it is the one
   re-run that would materially strengthen §12.6.2's mechanism attribution rather than just its precision.
2. **A `d5cd`-equivalent at a past-L3 shape** (A6) — the control that would actually close the coverage gap
   §12.6.3 claims to close. `gate/up` cannot be forced resident, so this needs a synthetic pair straddling
   the 16 MiB boundary rather than a real organ.
3. **A stride-conflict probe, which no one has proposed and which the `Mpad` sweep cannot separate.** Every
   `Mpad` in `d5m` is a large power of two or a multiple of 4096 (28672 = 7·4096), so every point drives the
   `t`-loop's addresses into a handful of cache sets. Re-running the sweep with each `Mpad` padded by 64 —
   a shape change, not a kernel change — would show how much of the collapse is set-conflict eviction rather
   than capacity. If a large fraction is conflict, the remedy is a padded stride, which is cheaper than
   either the access-pattern rewrite or a 5-trit decoder and is independent of both. **This is the cheapest
   unexplored lever the probe has, and it is not in §12.6.7's list of three.**
4. **An equal-work-per-rep `Mpad` sweep** (A3), to bound the frequency-droop and pool-composition confounds
   the current sweep leaves open: hold `touch·EB` constant across rows instead of letting the top row stream
   7 GiB per rep against ~2 GiB elsewhere, and hold `nblk` genuinely fixed rather than letting it fall
   128 → 4.
5. **A re-run of `d5m`/`d5abl` purely as replication.** Lowest value. It would confirm the artefacts and
   tighten A3's estimator question, but changes no verdict.

Not settleable by any run:

- **The provenance date of `a1_4_donor_order/`** (A10). Rests on testimony; no artefact can date it, and it
  bears on no conclusion.
- **Whether past runs were contended** (A12). The gate did not record enough to reconstruct it. Future runs
  could, if the gate were sampled during the timed window rather than once before it.

## DISPOSITION

**§12.6 must not enter `ADAPTER_MEMO_01` or any decision document in its present form.**

Blocking, in order of consequence:

1. **A9** — the 39.7%/85.9% comparison, and every inference from it, must be withdrawn or restated in one
   currency. Corrected, the wide-shape anomaly does not exist.
2. **A2** — §12.6.1's 2.12× fall must be reported alongside its moved-byte correction, or not reported as
   evidence about the binding constraint. §12.6.7's "access-pattern-bound" answer rests on A2 and A9
   together and does not survive either.
3. **A7 (2.34×)** — strike `22.69` or produce the artefact it came from. Quote 2.27×.
4. **A6** — §12.6.3's claim to have closed the coverage gap must be withdrawn; the control is real but
   covers the wrong regime.
5. **A4** — the ablation's "identical load" description is factually wrong and must be corrected in both
   the document and the source comment.
6. **A12** — the `-- clean.` banner must stop being cited as evidence a run was uncontended. It records a
   single pre-run sample and nothing else.

Passing and safe to carry forward: **A1** (compute-bound is dead), **A8** (Claim 1's tok/s arithmetic is
usable, for the narrower reason given), **A10** (the A1.4 artefacts are genuine), §12.6.6's 9.71 GB/s grand
mean, and the five items under "What §12.6 gets right."

**On the priority question put to me:** finding 7 / A9 is closed, and closed against the write-up. The two
percentages are not computed against comparable ceilings. Corrected to one currency the wide-shape anomaly
disappears (~79.5% vs ~85.8%), and the ~2× it was read as promising is the half-line waste itself rather
than headroom sitting on top of it. **The access-pattern fix is still the right next build** — it is worth
about 2× on the charged rate at past-L3 shapes and nothing at `k/v`-class shapes — but it should be
commissioned on that arithmetic, bounded by the ~22 GB/s moved-byte plateau, and not on §12.6.2's ceiling
column. Before it is built, the stride-conflict probe in the closing section (item 3) is cheaper, has never
been run, and could change what the fix should be.

**No fixes implemented. No commits. No pushes.**
