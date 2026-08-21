# Controller audit — `DONOR_PROJ_RATE.md` (the measurement that flips the ≥10 tok/s gate)

**Role:** Controller, independent adversarial review under §7.1 two-key. **Branch** `research/donor-adaptation`.
**Date** 2026-08-20. **READ-ONLY on code.** Nothing in `benchmarks/` or `docs/` was edited; this file is the
only artefact written. Verify with `git status` — `gemv_donor_bench.c` remains untracked-and-unmodified.

**Subject:** `docs/research/DONOR_PROJ_RATE.md` (Builder), `benchmarks/donor_adaptation/gemv_donor_bench.c`.

**Everything below was re-run from source by me, in my own build, in my own session.** Binaries were built to
`bin/ctrl_donor_bench.exe` and `bin/ctrl_engine_check.exe` so as not to overwrite the Builder's.

```
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp \
      benchmarks/donor_adaptation/gemv_donor_bench.c -o bin/ctrl_donor_bench.exe -lm
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp \
      benchmarks/phase60/engine.c              -o bin/ctrl_engine_check.exe -lm
export OMP_PROC_BIND=close OMP_PLACES=cores
```
clang 21.1.8 x86_64-w64-windows-gnu, AMD Ryzen 5 3600X, 6C/12T, 80 GB, Windows 11.

---

## VERDICT SUMMARY

| # | claim under attack | verdict |
|---|---|---|
| C1 | known-positive gate failed; the failure is in the published table, not the harness | **PASS** — replicated |
| C2 | byte accounting 1,550,057,472 B | **PASS** — re-derived from the config, exact |
| C3 | asymptote 38.84 GB/s, not 34/36 | **PASS** — I measure 39.18 ± 1.05 |
| C4 | 37.74 GB/s on the donor stream | **PASS with FLAG** — I measure 39.87; ±0.18 understates by ~11× |
| C5 | timing methodology (warm buffer / DCE / hoisting) | **PASS** — defect classes excluded, three ways |
| C6 | `ktrunc` planted control fires | **PASS** — I ran it; it fires at 8.925e−03 vs 4.526e−08 |
| C7 | fp32 comparator blind to 1 ULP; float64 + ladder substituted | **PASS** — substitution is adequate |
| C8 | ternary LUT is bandwidth-bound at donor D | **PASS** — replicated, more strongly than reported |
| C9 | `run_expert_rate` replica reproduces | **PASS with FLAG** — reproduces, noisier than reported |
| **C10** | **recommendation #1 (`PROJ_STREAMED_FLOOR 34.0→37.74`)** | **BLOCK — the amendment is a no-op** |
| **C11** | **"34.0 is the sole source of the FAIL verdict"** | **BLOCK — falsified; it is one of three** |
| **C12** | **the decomposition the constant is substituted into** | **BLOCK — omits terms larger than the margin** |
| C13 | 37.74 vs 38.84 for the `PROJ_STREAMED_FLOOR` slot | **FLAG** — neither; wrong quantity for that slot |

**The measurement is sound. The inference from it to a cleared gate is not.**

---

## C1 — PASS. The known-positive failure does point where the Builder says it points.

This was the claim flagged as most self-serving, so I did not take the Builder's interleave table on
trust: I rebuilt `benchmarks/phase60/engine.c` **unmodified** and ran `--gemv-sweep` twice myself.

`bin/ctrl_engine_check.exe --gemv-sweep`, t6 GB/s, verbatim:

| size MB | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|
| **engine.c, my run 1** | 193.5 | 194.0 | 202.6 | 179.2 | 122.8 | 44.3 | 40.6 | 39.4 |
| **engine.c, my run 2** | 191.7 | 192.9 | 198.5 | 206.2 | 162.4 | 42.9 | 40.5 | 38.1 |
| Builder's engine.c (mean 3) | 188.2 | 188.3 | 190.7 | 197.0 | 120.3 | 40.1 | 37.9 | 38.7 |
| *published 2026-07-12 §1b* | *187* | *185* | *134* | *60.5* | *55.7* | *45.5* | *45.3* | *36.5* |

**The reconciliation is sound, and it is not an artefact of the Builder's harness.** `engine.c` — the routine
that *produced* the published table — does not reproduce its own 16/24/32 MB t6 points on this machine
today, in my build, in my session. At 24 MB the published value is 60.5 and the original routine returns
179.2 and 206.2. That is a 3× disagreement produced by the published code itself, with no Builder code in
the path at all.

The alternative hypothesis in the brief — "the harness is simply wrong in a way that happens to also affect
the engine's own routine" — is **excluded by construction**: my `engine.c` build shares no source with
`gemv_donor_bench.c`. Two independently-built binaries from two different source files cannot share a
harness defect. The only shared factor is the machine, which is precisely the Builder's claim (§2.3:
placement-conditioned).

**Does anything in the donor conclusion depend on the 5 non-reproducing points?** No, and I verified this
rather than accepting it. `r_proj()` in `donor_inventory.py` short-circuits at `size_mb > 96` and never
interpolates `RCURVE_GBS` for any donor: the smallest donor proj+head stream in the study is orders above
96 MB (Qwen2.5-1.5B is 1.55 GB). The 16-32 MB region is unreachable for every donor row. **PASS.**

*Two corrections to the Builder's own account, both against its interest and both minor.*
(i) §9 says the 48/64 MB published points "did not reproduce under any placement tried". My 48 MB
runs give 44.3 and 42.9 against a published 45.5 — that is −2.6% and −5.7%, i.e. **inside** the ±8% band.
Only 64 MB genuinely misses (40.5 / 40.6 vs 45.3, −10%). The Builder over-reported its own failure count.
(ii) my 4 MB t1 is 36.0/35.4 vs published 30.5 = +18%, so the Builder's "t1 reproduces at every one of the
8 points, worst +13%" is optimistic; on my run the t1 row also breaks the band at the small end. Neither
touches a donor row.

## C2 — PASS. Byte accounting re-derived independently, exact.

Derived by me from `configs/Qwen__Qwen2.5-1.5B.json` alone, no Builder code, no F1:

```
q  = 1536 · (12·128)     = 2,359,296      o  = (12·128) · 1536 = 2,359,296
k  = 1536 · (2·128)      =   393,216      v                    =   393,216
per layer                = 5,505,024   × 28 = 154,140,672  × 4 B = 616,562,688 B
head (tied) = 151936·1536= 233,373,696                     × 4 B = 933,494,784 B
                                                     TOTAL      = 1,550,057,472 B
```
Matches the harness's C5 print and F1's hand-derivation to the byte. `1550057472 / 37.74e9 = 41.072 ms`.
Ternary term re-derived the same way: `3·1536·8960·28·0.5 = 578,027,520 B` — matches. **PASS.**

Convention note, not a defect: qkv **biases** (2,048/layer, 229,376 B total) are excluded, and the stream is
charged at **fp32** for a **bf16** donor. Both are the tool's declared convention and both are conservative
or neutral. The fp32 choice is load-bearing at 2× and should not be silently revisited later.

## C3 — PASS. The asymptote is not 34 and not 36. I measure 39.18.

`bin/ctrl_donor_bench.exe sweep --reps 5`, t6 GB/s, my run:

| MB | 64 | 96 | 128 | 192 | 256 | 384 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| **t6 GB/s (mine)** | 40.4 | 40.1 | 39.3 | 39.5 | 37.4 | 37.4 | 39.8 | 39.6 | 39.1 |

**Mean over the 9 tail points = 39.18, sd 1.05 (2.7%).** Builder: 38.84 ± 0.68. Agreement 0.9%.
Flat across 16× of size, no further cliff, no downward trend. The published "34-36 GB/s" assertion is
contradicted by the original routine's own last point today (my `engine.c` 96 MB = 38.1-39.4). **PASS**, and
the Builder's figure is if anything 1% conservative relative to mine.

## C4 — PASS on the point estimate, FLAG on the error bar.

`bin/ctrl_donor_bench.exe fullstack --reps 5`, three independent invocations, my build:

| inv | byte accounting | t1 GB/s | **t6 GB/s** | **ms/token** | cv% |
|---|---|---|---|---|---|
| 1 | EXACT MATCH | 26.05 | **39.94** | 38.805 | 1.78 |
| 2 | EXACT MATCH | 26.11 | **39.77** | 38.971 | 1.85 |
| 3 | EXACT MATCH | 25.80 | **39.91** | 38.840 | 5.70 |

**I measure 39.87 ± 0.09 GB/s where the Builder reports 37.74 ± 0.18.** The point estimate is corroborated
and the discrepancy runs **in the conservative direction** — the Builder under-reported. Substituting my own
constant gives a *larger* margin, not a smaller one. On the headline, **PASS.**

**FLAG — the ±0.18 is not an uncertainty on the constant.** My session mean sits **5.6% above** the
Builder's, which is **11× the quoted ±0.18** and 24× the quoted cv. The Builder's ±0.18 is a within-session
repeatability figure (three invocations, minutes apart, same machine state); it is being quoted as though it
bounded the constant. It does not: the between-session term dominates it by an order of magnitude, and
neither session samples machine-to-machine, thermal, or memory-population variation at all.

*Does ±0.18 propagate to a tok/s interval that still clears 10?* Formally yes — 41.07 ± 0.19 ms gives
10.218 ± 0.021 tok/s, putting the gate ~10σ away — **but that computation is meaningless**, because it
propagates the wrong sigma. The honest statement is that the *proj term alone* is not what puts the donor
near the gate; see C12 for what does.

**What else in the 41.07 ms is uncertain, and it is not the 41.07.** The proj term is 42% of the 97.87 ms
total. The remaining 58% is: MLP 50.90 ms on a rate the Builder itself brackets as `[20.93, 50.90]` — a
factor of 2.4; KV 5.87 ms on an **unbuilt** precision; glue 32.7 µs linearly extrapolated from an L=6 anchor
to an L=28 model, unmeasured. Every one of those carries more uncertainty than the constant that was
measured. **The measurement improved the best-known term and left the two worst untouched.**

## C5 — PASS. The classic microbenchmark defects are excluded.

Audited `time_fp32()` (l.119-140) and `mode_fullstack()` (l.265-300) by reading, then falsified by running.

- **Dead-code elimination / hoisting.** `x[p%K] += 1e-9f` poisons the input every pass, so the call cannot be
  hoisted; `y` is a heap pointer that escapes to `free`, so the stores cannot be elided. Verified
  *empirically*, which is stronger than reading the source: the same code path reports 193 GB/s at 4 MB and
  39 GB/s at 1024 MB. An eliminated loop reports neither — it reports both identically and absurdly.
- **Function-pointer inlining.** l.131-132 branch on `kern` *outside* the loop and call `matvec` directly in
  the `else` leg. The Builder's disclosed first-build defect (routing through the pointer, blocking inlining,
  moving the cliff 30%) is genuinely fixed in the source I compiled.
- **Cache warming between repeats — the defect the brief names.** **Structurally impossible in `fullstack`.**
  The working set is a single 1,550,057,472 B contiguous allocation walked once per repetition in layer
  order. Aggregate L3 is 32 MB = **2.06% of the stream**. At most the trailing 2% of one repetition can
  survive into the next, which cannot move a rate by more than ~2%. The `rp==0` warm pass is discarded.
- **Is the "streamed" condition actually achieved?** Verified three independent ways, and this is the
  strongest evidence in the audit. (i) The reported rate is **39.87 GB/s against a machine DRAM aggregate
  ceiling of 40-44 GB/s** (`PHASE64_BUDGET.md` §1). A benchmark accidentally measuring a warm buffer does
  not land at 95% of the DRAM ceiling — it lands where the resident rows land, at **150-200 GB/s**, which the
  same harness reports for the same kernel at 4-24 MB. The number is on the correct side of a physical wall.
  (ii) `fullstack` (39.87) agrees with the 1024 MB sweep point (39.1) to 2%, and with `lm_head` streamed
  (39.9) to 0.1% — three different working sets, one rate. (iii) `organs` RESIDENT vs STREAMED separates
  cleanly (q_proj 167.8 → 40.0); the instrument demonstrably *can* report a warm buffer, and does not here.
- **Timer resolution.** `now_s()` over a `QueryPerformanceCounter`-class source against windows of 38-60 ms,
  and 21 µs at the smallest size. Resolution is not a factor at any reported size.

**PASS.** I could not construct a reading under which the donor-relevant numbers measure a warm buffer.

## C6 — PASS. I ran the `ktrunc` control. It fires.

The brief is right that this is the project's signature failure mode, so I did not accept the Builder's
transcript. `bin/ctrl_donor_bench.exe control`, my build, my run, verbatim:

```
C3  kernel        bytes read   charged   inflation  rel-err     norm-err    caught?
    honest           9.0 MB      9.0 MB   1.00x      1.150e+01   4.526e-09   silent (correct)
    halfrows         4.5 MB      9.0 MB   2.00x      1.150e+01   7.059e-03   CAUGHT
    ktrunc           4.5 MB      9.0 MB   2.00x      6.013e+07   8.925e-03   CAUGHT
    and the reason it matters: honest 162.10 GB/s vs ktrunc 290.56 GB/s (1.79x inflation ...)
```

**`ktrunc` fires at norm-err 8.925e−03 against a fire threshold of 4.526e−08 — 197,000× above threshold**,
while the honest kernel stays silent at 4.526e−09, below it. The negative legs (C1a, C2 δ=0, C3 honest, C4a)
are all silent; the positive legs (C1b, C2 δ≥1e−4, C3 halfrows/ktrunc, C4b, C4c) all fire. The control is
real and it discriminates. **PASS.**

*Difference from the Builder's transcript, immaterial:* my inflation ratio is **1.79×** (162.10 vs 290.56),
the Builder's 2.11× (143.00 vs 301.97). Both are run-to-run noise on a 9 MB L3-resident hot loop; the
norm-err figures — which are pure arithmetic and carry the control — are **bit-identical to the Builder's**.
The quoted "143→302 GB/s (2.11×)" should be treated as one draw, not a constant.

*Scope limit worth recording:* C3 catches a kernel that **reads fewer bytes than charged**. It does not and
cannot catch a kernel that reads all its bytes **from cache instead of DRAM**. That second failure mode is
the one the brief worried about, and it is excluded by C5, not by C3. The Builder's claim that "every rate
in this document is guarded by a check demonstrated to fire on exactly the artefact that would fake it" is
**one artefact too broad** — C3 guards the byte-accounting failure, not the residency failure.

## C7 — PASS. The float64 substitution is adequate; the blindness does not invalidate correctness.

The Builder's honest negative reproduces exactly in my run (C1c: 0/1536 output floats changed; norm-err
4.526e−09 vs threshold 4.526e−08), as does the C2 ladder (first fire at δ=1e−4). These are deterministic
arithmetic, not timing, so bit-identical reproduction is expected and confirms I am running the same test.

**The blindness is physics, and the Builder's diagnosis is correct.** A 1-ULP move of one fp32 weight
(dw = 2.328e−10) against x = 0.02 perturbs one row by 4.657e−12. The row's own fp32 accumulation noise over
K=1536 is ~4.5e−9. The signal is **~1000× below the noise floor of the instrument**. A threshold able to
resolve it would fire on every honest run — the control would be a false-positive generator, which is worse
than blind. No fp32-vs-float64 comparator on this shape can do better; this is not a harness defect.

**Is the substitution adequate?** Yes, because the two replacements together cover the property the control
existed to establish — *that every element of W is actually consumed*, which is the only correctness
property a bandwidth benchmark strictly needs:
1. **C1b establishes it directly.** The float64 reference sees the 1-ULP change at exactly the predicted
   magnitude `dw·x` (agreement 0.000e+00 relative), on exactly row 768 and no other (zero cross-talk across
   the other 1535 rows). Element `W[768][768]` is provably read and provably contributes.
2. **C2 quantifies what the fp32 gate *can* resolve** — δ=1e−4 on one element in 2,359,296 — measured, not
   asserted. That is a real sensitivity figure for the gate that is actually used.
3. **C3/C6 covers the failure mode that would actually inflate a rate.** A 1-ULP corruption is not a
   plausible route to a wrong GB/s; a truncated read is, and that is caught six orders above threshold.

The residual gap — corruptions between 1 ULP and δ=1e−4 pass unseen — is real and is correctly disclosed.
It bounds the *numerical* claim, not the *rate* claim. **PASS.**

## C8 — PASS. Controller #2's compute-bound reading is a D=256 artefact. Adjudicated against #2.

`bin/ctrl_donor_bench.exe lut --reps 3`, my run, **donor attn proj 1536×1536, kernel shape held fixed**,
only the pool size varied:

| pool MB | 1.1 | 3.4 | 15.8 | 31.5 | 63.0 | 127.1 | 255.4 | 511.9 |
|---|---|---|---|---|---|---|---|---|
| **t6 GB/s** | **96.40** | 90.55 | 86.09 | 53.57 | 37.23 | 30.72 | 22.06 | **23.06** |

**96.40 → 23.06 GB/s = a 4.2× fall with the kernel completely unchanged.** The Builder reported 3.6×; I
measure it larger. Nothing about the kernel, the LUT, the block shape or the access pattern differs down
that column — only how many distinct blocks the pool holds. **A compute-bound kernel is physically incapable
of this.** Same shape on the other donor organs: gate/up 101.25 → 31.69 (3.2×), down 74.89 → 26.28 (2.8×).

**Has the Builder changed something else between the two measurements?** I checked for exactly this, since
it is the alternative explanation. It has not: the experiment is *internally* controlled — the falsification
does not compare the Builder's number against Controller #2's number, it compares the Builder's number
against **itself** at a different working-set size. Any confound in kernel, build, flags or machine cancels
down the column. The design is sound and it was pre-declared to be able to come out either way.

The mechanism is the ceiling shift: the kernel's own resident ceiling is **29.3 GB/s at D=256** but
**96.4-101.3 GB/s at D=1536** (my runs), so at D=256 the ceiling sits close enough to the streamed rate
(resident÷streamed = 1.6×) that the path *looks* compute-limited, while at donor D the ratio is 3.6-4.2×.
**Controller #2's "compute-bound by ~16×" is scoped to D=256 and does not transfer. Adjudicated in the
Builder's favour.** The Builder correctly flags its fork/join explanation as HYPOTHESIS; that flag must
survive into any downstream quotation.

## C9 — PASS with FLAG. The `run_expert_rate` known-positive reproduces.

`bin/ctrl_donor_bench.exe lutrepro`, 2 invocations, mine:

| | published 64.1b(2) | Builder | **mine** |
|---|---|---|---|
| t1 GB/s | 7.45 | 7.58, 8.24 | **7.94, 7.92** |
| t6 GB/s | 17.0 | 18.59, 18.92 | **18.56, 16.73** |
| µs/expert t6 | 2.88 | 2.60, 2.64 | **2.648, 2.939** |

Reproduces, bracketing the published values. **PASS** — this is a genuine known-positive in the Builder's
favour and it does what a known-positive must: it demonstrates the apparatus fires correctly on a published
result, which is what licenses treating the §2 divergence as specific to the fp32 cache-transition region.

**FLAG:** my two t6 draws span 16.73-18.56 (11%), where the Builder reported 18.59-18.92 (1.8%). The
Builder's two invocations understate this replica's dispersion; "within ~11% on every figure" is right, but
"same ratio" implies a tightness that a third and fourth draw do not support.

## C10 — BLOCK. Recommended amendment #1 is a no-op. Executing it as written changes nothing.

Follow-up #1 reads: *"Amend `donor_inventory.py`: `PROJ_STREAMED_FLOOR 34.0 → 37.74` (measured, §5.2) and
re-run `analyze`. Restate every downstream 'zero donors pass' claim."*

**`PROJ_STREAMED_FLOOR` is never consulted for any donor in this study.** `r_proj()` (l.107-127):

```python
size_mb = size_bytes / MB
if size_mb > RCURVE_MB[-1]:                                  # 96 MB
    return RCURVE_ASYMPTOTE_LO, RCURVE_ASYMPTOTE_HI, True    # <-- 34.0, 36.0; FLOOR not read
```

Every donor's proj+head stream exceeds 96 MB — Qwen2.5-1.5B's is 1.55 GB, 16× past it — so every donor takes
the short-circuit branch and receives `(RCURVE_ASYMPTOTE_LO, RCURVE_ASYMPTOTE_HI) = (34.0, 36.0)`.
`PROJ_STREAMED_FLOOR` is only reachable on the `<= 96 MB` interpolation paths. Executed against the tool:

```
BASELINE      r_proj(1.550GB) = (34.0, 36.0, True)
rec#1 FLOOR:=37.74  r_proj    = (34.0, 36.0, True)    <-- UNCHANGED
ASYMPTOTE_LO:=37.74 r_proj    = (37.74, 36.0, True)
```

**Applying recommendation #1 exactly as written leaves the FAIL verdict standing, silently.** The risk is
specific and severe: an implementer follows the instruction, re-runs `analyze`, sees no change, and concludes
the measurement did not matter. The constant that actually produces the 34.0 is **`RCURVE_ASYMPTOTE_LO`**.

Note also the third line: moving `ASYMPTOTE_LO` alone yields `(37.74, 36.0)` — **the pessimistic bound now
exceeds the optimistic one**, inverting the bracket and corrupting every `pess..opt` range in the tool.
`RCURVE_ASYMPTOTE_HI` must move with it. **BLOCK until the amendment names the right constants.**

## C11 — BLOCK. "The sole source of the FAIL verdict" is falsified.

Claimed twice, in §0 headline 2 and §5.3: *"`PROJ_STREAMED_FLOOR = 34.0` is 11% low and is **the sole source
of the FAIL verdict**."* I tested this by holding the proj constant at the original 34.0 and moving only the
*other* bracketed constants — the same operation the Builder performed in the opposite direction:

| proj GB/s | MLP ms | KV | total ms | **tok/s** | gate |
|---|---|---|---|---|---|
| **34.0 (unchanged)** | 50.90 integrated | 4-bit | 102.39 | 9.77 | FAIL |
| **34.0 (unchanged)** | **20.99 kernel-pure** | 4-bit | 72.48 | **13.80** | **PASS** |
| **34.0 (unchanged)** | **20.99 kernel-pure** | fp16 | 90.10 | **11.10** | **PASS** |
| 37.74 (measured) | 50.90 integrated | 4-bit | 97.87 | 10.22 | PASS |
| 37.74 (measured) | 50.90 integrated | **fp16** | 115.49 | **8.66** | **FAIL** |
| 39.87 (Controller) | 50.90 integrated | **fp16** | 113.30 | **8.83** | **FAIL** |

**At the untouched 34.0, moving only the MLP constant clears the gate at 13.80 tok/s.** 34.0 is therefore not
the sole source of anything — it is one of at least three constants any one of which flips the verdict, and
it is the *smallest* lever of the three. The Builder's own §8.5 establishes the MLP bracket is a factor of
2.4 wide; a 2.4× bracket on 52% of the budget dominates an 11% correction on 42% of it.

**Conversely, and this is the finding that matters most: the measured proj constant does not flip the gate on
its own either.** Rows 5 and 6 show that the Builder's measured 37.74 — and my larger 39.87 — combined with
the tool's own MLP charge and the **only KV precision that is actually built**, return **8.66 and 8.83 tok/s,
FAIL**. The PASS at 10.22 lives in exactly one cell of the matrix: `{4-bit KV} × {integrated MLP}`. 4-bit KV
is unbuilt (audit F8). **BLOCK on the claim as phrased.** The defensible statement is: *the proj constant
moves one term of a four-term budget in which two other terms remain bracketed more widely than the margin.*

## C12 — BLOCK. The decomposition omits terms larger than the 2.2% margin.

The 10.22 tok/s clears 10 by **2.13 ms of headroom** in a 97.87 ms budget. I audited what the budget contains.
`time_model()` (`donor_inventory.py` l.642-673) computes exactly:

```python
pess = t_proj_pess + t_lut_pess + glue_us + (t_kv_pess or 0.0)
```

**proj + LUT-MLP + KV + glue. That is a weight-streaming model only.** `PHASE64_BUDGET.md` §2 — the project's
own canonical component model, the one this whole budget descends from — lists **five** components. The two
that are missing are compute, not streaming, and they are not small at donor dimensions:

| §2 term | §2 formula | evaluated at D=1536, L=28 |
|---|---|---|
| scan-recur | `46 µs · (Dn·N·L)/(512·96·6)`, Dn=2D | **859 µs (N=64) … 1717 µs (N=128)** |
| SWA | `65 µs · (D/256) · n_swa` | **390 µs per attention-retaining layer** |

The donor programme's own sealed constraint is *"attention on a minority of layers"* — so both terms are
live by construction: the converted layers pay scan-recur, the retained ones pay SWA. Against 2.13 ms of
headroom:

| omitted terms restored | total ms | **tok/s** | gate |
|---|---|---|---|
| none (as reported) | 97.87 | 10.217 | PASS |
| scan-recur only, N=96 (+1.29 ms) | 99.16 | **10.084** | PASS, margin 0.8% |
| scan + 2 SWA layers (+2.07 ms) | 99.94 | **10.006** | **on the line** |
| scan + 4 SWA layers (+2.85 ms) | 100.72 | **9.928** | **FAIL** |
| scan + 7 SWA layers (+4.02 ms) | 101.89 | **9.814** | **FAIL** |

**Two SWA layers and one scan term are enough to take the verdict back.** This is not a defect the Builder
introduced — it inherits the decomposition from F1 and is explicit that it substitutes *only* the proj
constant, which is correct discipline. But it is decisive for what the result may be *reported as*: a
2.2% margin cannot carry a verdict when the model producing it omits terms the project's own budget document
sizes at 1.3-4.0 ms.

Two further unbudgeted-or-unmeasured items inside the same 2.13 ms: **glue** is `7 µs · 28/6 = 32.7 µs`, a
linear extrapolation of an L=6 measurement to a model 4.7× deeper, never measured; and the sum assumes
**zero overlap** between organs (conservative, and the one assumption pointing the other way).

*In fairness, and it should be stated plainly:* my own replication (39.87 GB/s → 10.45 tok/s) buys 2.9 ms of
headroom rather than 2.13, and the MLP charge at 50.90 is pessimistic by up to 2.4×. It is entirely possible
the donor clears 10 comfortably. **But that would then rest on one unmeasured quantity happening to offset
another unmeasured quantity, which is not a measurement.** **BLOCK on the verdict flip, not on the number.**

## C13 — FLAG. 37.74 is the wrong quantity for the `PROJ_STREAMED_FLOOR` slot; so is 38.84.

They are measured on different things and neither matches the slot's semantics.

- **37.74 (§5.2)** is `Qwen2.5-1.5B`'s per-token stream *in that model's layer order*, and the Builder
  explains it sits 3% below the synthetic asymptote precisely *because* the two skinny K/V organs interleave
  between the square ones. **That 3% is a Qwen-specific organ-order penalty.** `PROJ_STREAMED_FLOOR` /
  `RCURVE_ASYMPTOTE_LO` are **model-generic** constants applied to all 18 donors. Installing 37.74 there
  charges every other donor — different n_kv ratios, different head sizes, MoE and hybrid layouts — Qwen's
  interleave penalty. Right number, wrong slot.
- **38.84 / my 39.18 (§4)** is the size-swept synthetic asymptote and *is* the quantity the slot names
  ("the fully-streamed rate for a matrix of this size"). That is the correct constant for the generic slot.
- **Neither is engine-integrated.** Both are kernel-pure. The Builder rigorously flags this caveat for the
  LUT path (§8.5) and then does not apply the same standard to the proj path.

*The one mitigation, and it is a good one:* the analogous integration gap appears genuinely small here.
`PHASE64_BUDGET.md` §1's *engine-integrated* proj-GEMV datum is 11 MB / 272 µs ≈ **40.4 GB/s** — which lands
on the kernel-pure streamed rate (38.8-39.9), not 4.8× below it as the MoE path does. So the proj path
appears to carry little integration overhead. **This should be stated as the argument, in place of the
silence.** It is the strongest available defence of using a kernel-pure proj number and the Builder does not
make it.

**Recommended amendment, if one is made:** set `RCURVE_ASYMPTOTE_LO = 38.8` **and** `RCURVE_ASYMPTOTE_HI` to
the top of the measured band, keep `PROJ_STREAMED_FLOOR` consistent, and record 37.74 separately as the
*Qwen-specific measured stream* — not as a global constant.

---

## VERDICT

**Does `Qwen2.5-1.5B` clear the sealed ≥10 tok/s gate on this evidence?**

# STILL UNDECIDED

The Builder's own §9 framing — *"this settles the bracket question F1 raised; it does not settle the donor
route"* — is the correct one. The §0 headline is not.

**Confidence: high (that it is undecided), and rising in the donor's favour.**

**What is now settled, at high confidence, and it is real work:**
- The fp32 streamed projection rate on the 3600X is **38.8-39.9 GB/s**, measured to 1024 MB and on the
  donor's actual 1.55 GB stream. **It is not 34 and not 36.** Independently replicated by me at 39.18
  (sweep) and 39.87 (fullstack). The published 34-36 asymptote was an assertion and is now falsified.
- There is **no further bandwidth cliff** past 96 MB, and organ shape stops mattering once streamed.
- The `PHASE64_BUDGET.md` §1b **16-32 MB t6 points are not reproducible by `engine.c` itself** and must be
  annotated placement-conditioned. **No donor row depends on them** (verified: `r_proj` short-circuits).
- **Controller #2's compute-bound reading is scoped to D=256** and does not transfer to donor D. Adjudicated
  in the Builder's favour on an internally-controlled experiment I re-ran.
- The harness is a sound instrument: controls fire on known positives, stay silent on known negatives, and
  the byte accounting is exact.

**Why the gate is nonetheless not cleared:**
1. The PASS exists in **one cell** of the assumption matrix — `{4-bit KV} × {integrated MLP}` — and 4-bit KV
   is **unbuilt** (F8). On the only precision that is built, the Builder's own measured constant returns
   **8.66 tok/s, FAIL**; my faster constant returns 8.83, also FAIL (C11).
2. The 2.2% margin is **smaller than terms the project's own §2 budget model contains and this tool omits**
   — scan-recur and SWA, 1.3-4.0 ms against 2.13 ms of headroom (C12).
3. The claim "34.0 is the sole source of the FAIL verdict" is **false**: at 34.0 unchanged, moving only the
   MLP constant gives 13.80 tok/s (C11). Three constants each flip this verdict; the measured one is the
   smallest lever.
4. The prescribed remedy **does not work** (C10).

**To the owner, the defensible sentence is:** *the projection rate is now measured rather than asserted, it
is 14% better than the tool charges, and it removes one of the three reasons the donor was failing — the
donor now sits above the gate on the tool's current decomposition, but that decomposition omits two compute
terms larger than the margin and assumes a KV precision that does not exist yet. Undecided, trending pass.*

**Required before this may be reported as clearing the gate:**
1. Fix the amendment to name `RCURVE_ASYMPTOTE_LO`/`_HI`, not `PROJ_STREAMED_FLOOR` (C10). Use **38.8**, not
   37.74, for the generic slot (C13).
2. Add the §2 scan-recur and SWA terms to `time_model()`, or state explicitly that the tok/s figure is a
   **weight-streaming bound** and not a decode-rate prediction (C12).
3. Retract "sole source of the FAIL verdict" (C11).
4. Requote the error bar as between-session, or drop it (C4).
5. Close the MLP bracket with an engine-integrated measurement at donor dims — the Builder's own follow-up
   #4, which C11 shows is the **highest-value** item on the list, not the fourth.

*No commit, no push. Nothing outside this file was written.*

---

## ADDENDUM — `donor_inventory.py` was rewritten *during* this audit

**Recorded for accuracy about what was audited.** `benchmarks/donor_adaptation/donor_inventory.py` changed on
disk mid-session (mtime 23:10:33, +309/−86 lines) while this review was running. I did not write it — my only
interaction with that file was a read and a read-only `import` for the C10 execution test. The C10 test output
(3-tuples from `r_proj`) confirms it ran against the pre-rewrite version, i.e. **the version
`DONOR_PROJ_RATE.md` describes and whose amendment it recommends.** C10-C13 are audits of that version and of
the recommendation as written; they stand as such.

**The new version, inspected read-only, resolves C10 and C13 and does not resolve C12:**

- **C10 resolved.** `PROJ_STREAMED_FLOOR` is retired to `PROJ_STREAMED_FLOOR_RETIRED` and `r_proj()` is
  restructured to return the streamed rate at every size by default. The no-op path is gone. The fix is
  exactly the one C10 requires — which corroborates the finding rather than superseding it, and means the
  §9 follow-up #1 text is now wrong in a second way: it names a constant that no longer exists.
- **C13 resolved, and correctly.** The generic slot is now `PROJ_ASYMPTOTE_MEAS = 38.84` with
  `PROJ_ORGAN_LO = 36.0` / `PROJ_ORGAN_HI = 40.20`, i.e. the **size-swept asymptote**, not the Qwen-specific
  37.74. That is what C13 recommends. My independent sweep (39.18) and fullstack (39.87) both sit inside
  `[36.0, 40.20]`, so the installed interval is supported by my measurements as well as the Builder's.
- **C11 partly answered.** The LUT bracket is now honestly widened to `[11.398, 27.64]` with the
  kernel-pure-vs-integrated caveat carried in the source comment. This makes C11's point structurally visible
  in the tool: at `LUT_DENSE_HI` the gate clears even at the retired 34.0 proj rate.
- **C12 NOT resolved.** `grep -n "scan\|swa\|SWA"` over the new file returns **nothing** in the timing path,
  and the total is still `t_proj + t_lut + glue + t_kv`. The two `PHASE64_BUDGET.md` §2 compute terms remain
  omitted, and they remain larger than the margin. **This is the one BLOCK that survives the rewrite, and it
  is the one the verdict turns on.**

The verdict above is unchanged: **STILL UNDECIDED**, for the reasons in C11 and C12.

*Housekeeping: my read-only `import` created `benchmarks/donor_adaptation/__pycache__/` (gitignored); it has
been removed. No commit, no push, no source file written by me.*
