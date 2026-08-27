# Controller audit — BRIEF D5 results (`DONOR_PROJ_RATE.md` §12.5)

Independent adversarial review. The Controller logs BLOCK / PASS / FLAG with a reproduction path; he does
not implement fixes, commit, push, or adjudicate direction.

---

## 2026-08-27 — audit of §12.5, commits `6b7e695` (harness + raw log) and `d419344` (write-up)

### VERDICT: **BLOCK**

**Claim 2 — "the donor-width ternary path is compute-bound, therefore 5-trit packing does not pay" — is
not merely unproven; it is contradicted by the write-up's own Part-B table, in which the identical LUT
kernel delivers 8.92 to 29.23 GB/s across shapes (a 3.28× spread that a compute-bound kernel cannot
produce) and reaches 76% of the machine's fp32 DRAM ceiling at a real donor organ (`k/v`, `K=8192`,
fully streamed) — so §12.5.6's second half and §12.5.7's second bullet must not enter any decision
document, and the causal ablation the Builder costed at 30–60 min is now necessary but no longer
sufficient on its own.**

Claims 1 and 3 are a different matter and I want that separated sharply:

| claim | verdict |
|---|---|
| **1** — the 21.25 GB/s constant does not transfer to donor width; it falls ~2× | **PASS WITH FLAGS.** Direction and rough magnitude sound. The specific charge I was asked to test — that the Builder compared a kernel-pure number to an integrated one — is **unfounded**; see finding 1. Flags are about missing artefacts and a dangling cross-reference, not about the number. |
| **2** — compute-bound, therefore 5-trit does not pay | **BLOCK.** Findings 2–6. |
| **3** — L-sensitivity does not bite where it matters | **PASS WITH FLAGS.** The explanation is arithmetically consistent with the measurements (finding 11). The flagging of `D=2048/4096/5120` is honest in substance; only its placement invites misreading (finding 12). |

**I ran no timing and no compute.** Everything below is derived from the committed raw log, the committed
harness source, and shape arithmetic. Cost to the machine: zero. The checks I could not run under that
choice are listed at the end.

---

## FINDING 1 — the "kernel-pure vs integrated" charge is **unfounded**. State this plainly.

**Not a defect.** I was asked to check whether the Builder compared a kernel-pure number to an integrated
one after telling the Principal that is the mistake the Principal made. He did not.

The `10.38 GB/s` donor-width point comes from `mode_d5g` (`gemv_donor_bench.c:1039`), which calls
`mlp_alloc` / `mlp_time` — the **engine-integrated** `MlpStack` path, the identical apparatus behind
control 1 and behind the banked 21.25. It is `L=28`, mean estimator, `skip=0`, same shape family. The
`llama70b_down` ~10.5 GB/s rows in the raw log are kernel-pure and §12.5.5 explicitly refuses to use them
for the transfer question, citing §10's ~30.5% integration overhead. The like-for-like route is the one
the write-up says it is.

That the two routes agree (10.38 integrated at `L=28` vs ~10.5 kernel-pure) is presented in §12.5.6 as
mutual corroboration. It is not — if integration costs ~30.5%, integrated and kernel-pure agreeing to 1%
is a coincidence that wants explaining, not a confirmation. **Right conclusion, wrong supporting sentence.**
Reproduction: `DONOR_PROJ_RATE.md:1136-1139`.

---

## FINDING 2 — **BLOCK.** `matvec_lut_full`'s byte rate varies 3.28× with shape. A compute-bound kernel's cannot.

**This is the finding that blocks Claim 2, and it needs no new measurement.**

`matvec_lut_full` (`gemv_donor_bench.c:135-143`) executes a fixed instruction sequence per byte of `codes`:
per `t`, one 16 B broadcast, one 32 B `loadu`, one `pshufb`, four `cvtepi8_epi32`, four `vpaddd`, two
shifts — covering exactly 32 bytes of codes. **Bytes-per-instruction is a compile-time constant,
independent of `M`, `K`, `Mpad`, `T`.** Therefore a genuinely compute-bound `matvec_lut_full` must deliver
a **shape-invariant GB/s**.

From the committed log, `t6`, all points streamed from a ≥470 MB pool (all past the 16 MiB L3 cliff):

| point | `Mpad` | `T` | LUT footprint | block `EB` | GB/s | % of fp32 ceiling (38.7) |
|---|---|---|---|---|---|---|
| `k/v` 1024×8192 | 1024 | 4096 | 64 KB | 4.0 MiB | **29.23** | **76%** |
| ternary `D=1536` | 5376 | 768 | 12 KB | 3.9 MiB | 23.19 | 60% |
| ternary `D=2048` | 7168 | 1024 | 16 KB | 7.0 MiB | 22.97 | 59% |
| ternary `D=4096` | 14336 | 2048 | 32 KB | 28.0 MiB | 20.66 | 53% |
| ternary `D=5120` | 17920 | 2560 | 40 KB | 43.8 MiB | 20.52 | 53% |
| `gate/up` 28672×8192 | 28672 | 4096 | 64 KB | 112 MiB | 11.05 | 29% |
| ternary `D=8192` | 28672 | 4096 | 64 KB | 112 MiB | 11.34 | 29% |
| `down` 8192×28672 | 8192 | 14336 | 224 KB | 112 MiB | 10.55 | 27% |
| `q/o` 8192×8192 | 8192 | 4096 | 64 KB | 32 MiB | 8.92–11.63 | 23–30% |

**8.92 → 29.23 GB/s is a 3.28× spread on one instruction sequence.** That variation is a property of the
memory hierarchy, not of the ALU. **The kernel is not compute-bound at these shapes.**

The second half is worse for the claim. §12.5.4 states the ternary path sits "at 29-33% of the demonstrated
ceiling at every `D`". That percentage is computed **only over the gate/up-shaped `D`-sweep**. The
write-up's own Part-B table contains a **real donor organ at donor width, fully streamed past L3, running
at 29.23 GB/s = 76% of the fp32 ceiling** — quoted in §12.5.5, never carried into §12.5.4's ceiling
arithmetic. **A kernel at 76% of a streaming ceiling is not a kernel with bus headroom to spare.** The
"29-33% at every `D`" sentence is false as written; the correct range across the measured donor shapes is
**23-76%**.

**Reproduction:** `benchmarks/donor_adaptation/results/d5_raw.log`, rows `D5CSV,organs,*` and
`D5CSV,fp32_vs_ternary,ternary_*` with `threads=6` and the largest pool per shape; shape arithmetic
`EB = (K/2)·((M+31)&~31)`, `Mpad=(M+31)&~31`, `T=K/2`, LUT = `T·16` bytes.

---

## FINDING 3 — **BLOCK.** A candidate mechanism the write-up does not know about: the harness under-counts the ternary arm's DRAM traffic by a shape-dependent factor of 1–2×, while counting the fp32 arm's exactly.

This is my hypothesis, offered as the *likely* explanation of finding 2 and flagged as needing the causal
test — but it is a structural reading of the kernel, not a guess.

The inner loop reads `codes[t*Mpad + base]`, 32 bytes, with `t` striding by `Mpad`. Every `Mpad` in the
sweep is a multiple of 64. So each iteration touches **32 bytes of a 64-byte cache line**; the other half
of that line is consumed by the **next** `base` iteration, which occurs one full pass over the entire
`EB`-byte array later. Therefore:

- `EB` small enough to stay resident across a pass → line reused → DRAM traffic ≈ `1 × EB`.
- `EB` ≫ L3 → line evicted before its second half is wanted → DRAM traffic ≈ **`2 × EB`**.

`lut_one_point` charges `EB × touch` bytes (`gemv_donor_bench.c:987`). **The reported ternary GB/s is a
charged-byte rate whose relationship to actual DRAM traffic varies with `EB`.** `fp32_streamed_point` /
`matvec` read each row fully sequentially with 100% line utilisation, so the fp32 arm's charged bytes
equal its actual bytes.

Applying the factor to the streamed points collapses the spread that finding 2 identifies:

| point | charged GB/s | implied actual DRAM GB/s | useful Gweights/s |
|---|---|---|---|
| ternary `D=1536` (`EB` 3.9 MiB, resident) | 23.19 | ~23.2 | 46.4 |
| ternary `D=8192` (`EB` 112 MiB, no reuse) | 11.34 | **~22.7** | 22.7 |
| `gate/up` (112 MiB) | 11.05 | ~22.1 | 22.1 |
| `down` (112 MiB) | 10.55 | ~21.1 | 21.1 |
| `q/o` (32 MiB) | ~11.0 | ~22.0 | 22.0 |

**Read the last two columns together.** From `D=1536` to `D=8192` the kernel's *useful work* rate halves
(46.4 → 22.7 Gweights/s) while its *actual DRAM byte rate* is flat at ~21-23 GB/s. **A kernel whose real
memory traffic rate is invariant across a 28× change in block size while its useful-work rate halves is
limited by memory traffic, not by compute** — it simply is not at the *sequential* fp32 ceiling, because a
32-of-64-byte strided pattern cannot reach a sequential ceiling.

**Why this is decisive for the decision, not just for the label.** 5-trit packing's entire value is fewer
bytes per weight. The write-up rejects it because DRAM is not saturated. If the binding constraint is
instead **bytes/lines touched below the DRAM ceiling** — line utilisation, L2/L3 fill rate, load-port
throughput, TLB — then **reducing bytes per weight is exactly the right lever and the write-up's conclusion
is backwards.** The evidence available today points that way, not the Builder's way.

**Reproduction:** read `gemv_donor_bench.c:135-143` (stride `Mpad`, 32 B load, 64 B line), compare
`gemv_donor_bench.c:987` (charged `EB·touch`) with `gemv_donor_bench.c:1137-1175` (fp32 arm, sequential
`matvec`). Confirm empirically with `k/v` `EB`=4 MiB (76% of ceiling) vs `gate/up` `EB`=112 MiB (29%) in
the log, at identical `T`, identical LUT footprint and identical instruction mix.

---

## FINDING 4 — **BLOCK.** The two "independent" signatures are three ratios of the same four numbers per `D`.

§12.5.4 and §12.5.6 present `ratio_fp32_over_ternary` and thread-scaling as "two independent, mutually
corroborating signatures". They are not independent in any sense:

- signature 1 = `fp32_t6 / tern_t6`
- signature 2 = `{tern_t6/tern_t1, fp32_t6/fp32_t1}`

Both `t6` values appear in both. Signature 1 is algebraically recoverable from signature 2's inputs. The
write-up's own justification gives the game away: "already sitting in this log at zero additional cost",
"at no additional measurement cost" — **no additional cost means no additional measurement.** If `tern_t6`
at `D=8192` is mis-accounted (findings 2-3), both signatures move together and neither is a check on the
other.

**What else produces each signature:**

*Signature 1 (fp32 byte rate > ternary byte rate).* fp32 moves 8 bytes per weight against ternary's 0.5.
**Any kernel with lower bytes-per-MAC shows a lower byte rate unless it is saturating the bus.** The ratio
being >1 therefore carries no information about bandwidth-boundedness on its own; the only question it can
answer is "is ternary's byte rate at the ceiling", and finding 2 answers that with 76% at `k/v`. The ratio
*growing* with `D` is exactly what finding 3's line-utilisation collapse predicts, with no compute story
required. The arms are also not matched on access pattern: the ternary arm is called with `iid=1`
hardcoded (`gemv_donor_bench.c:1185`, random block selection), the fp32 arm cycles blocks sequentially
(`p%ncopy`, `gemv_donor_bench.c:1168`); and at `D=8192` the two arms stream different pools
(469,762,048 B vs 939,524,096 B). A comparison the write-up calls "matched shape" is unmatched on gather
pattern, pool size and touch count.

*Signature 2 (fp32 caps at 1.5-1.6×).* This is **arithmetically forced and carries no diagnostic content.**
fp32 `t1` is 23.9-25.1 GB/s and the ceiling is ~38.7: `38.7/24.2 = 1.60`. A single Zen2 core already at 62%
of the machine ceiling cannot scale more than 1.6× no matter what binds it. Calling that "the textbook
signature of a kernel that saturates DRAM with 2-3 threads" restates the ceiling measurement.

*Signature 2 (ternary scales 2.4-4.8×).* Two alternatives the write-up does not consider. (a) A
**latency-bound / memory-level-parallelism-bound** kernel scales with threads precisely because more
threads mean more outstanding misses — textbook, and it is *not* compute-bound, and denser packing *does*
help it. (b) The scaling ratios are unstable because the **denominator** is: ternary `t1` runs 6.14 /
5.72 / 8.60 / 6.65 / **2.37** across the sweep and 6.76 / 2.66 / 2.39 / 2.56 across the organs — a 3.6×
spread in single-thread rate that no compute story explains and that finding 2 already flags as a memory
effect. The "2.4-4.8×" range is that instability, not a thread-scaling law.

**Reproduction:** `d5_raw.log`, all `D5CSV,fp32_vs_ternary,*` rows; `gemv_donor_bench.c:1177-1198`.

---

## FINDING 5 — **BLOCK.** "Compute-bound ⇒ 5-trit does not pay" is an assumption, not a result, on *either* branch.

Even granting the compute-bound label, the write-up's costing of the alternative is inherited, not derived.
The pre-registered dichotomy (brief §4, §12.3) says compute-bound → "roughly a 10% loss (fixed per-call
overhead amortised over less work per byte)". **Nothing in D5 measures or derives that 10%.**

It is also mechanically wrong about what a 5-trit kernel would be. The current kernel's speed comes from
`_mm256_shuffle_epi8` indexing a 16-entry LUT with a 4-bit code — 2 trits per byte is what makes a `pshufb`
LUT possible at all. **5 trits per byte has no 4-bit index and cannot use this LUT scheme**; it needs a
different unpack (base-3 extraction) whose cost is not a "fixed per-call overhead" but a per-byte one. The
write-up's decision therefore rests on a cost model for a kernel nobody has written, on the branch it did
not take, in a regime it has mislabelled.

**This does not mean 5-trit pays.** It means D5 has not priced it in either direction, and "does not pay"
is not a finding D5 can carry.

---

## FINDING 6 — **BLOCK-supporting.** Nothing validates the kernel-pure LUT meter in the regime where Claim 2's whole evidence base sits.

Controls 2 and 3 exercise the kernel-pure LUT meter at **1536×1536** — `EB` 1.125 MiB, `T` 768, LUT 12 KB —
a shape used nowhere else in D5 and specifically **not** in the `EB > L3` / LUT ≥ 64 KB regime where every
load-bearing Claim-2 number lives. Control 1 validates the *integrated* meter at `D=1536`. Control 4
validates the *fp32* meter.

**No planted control fires against the kernel-pure LUT meter at a donor shape.** Given that the meter's
charged-byte accounting is precisely what findings 2-3 dispute, that is not a gap in coverage, it is a gap
exactly where the instrument is suspect. The project's own rule applies: an instrument must be shown to
scratch on a known positive *in the regime where its nulls are being believed*.

**Reproduction:** `gemv_donor_bench.c:1069-1086` (controls 2/3 shape), vs `gemv_donor_bench.c:1177-1198`
and `1199-1224` (donor shapes).

---

## FINDING 7 — FLAG. Controls 2 and 3 share one measurement; "all four fire" is three independent measurements, not four.

`resident_t6 = 95.73` is the numerator of control 2's drop ratio **and the entirety of control 3**
(`gemv_donor_bench.c:1078-1084`, both read `g0[1]`). §12.5.2's table presents four independent PASSes.

Neither is satisfiable by construction — both can fail, and control 2's ≥2.0× threshold against a measured
3.46× is a real directional test. But if that single `pool=0` point is wrong, two of four controls fall
together. Reproduction: the two `D5VERDICT,control2/control3` lines, both quoting 95.73.

---

## FINDING 8 — FLAG. Control 1's in-process gate is **satisfiable by construction** — the exact mirror of the defect Amendment 1 was written to fix — and the external verdict it defers to has no artefact.

`gemv_donor_bench.c:1023-1026`:

```c
const char* ext_verdict=getenv("GEMV_D5_A1_VERDICT");
int p = ext_verdict? !strcmp(ext_verdict,"PASS") : 1;
```

With the variable unset, control 1 **always passes**. With it set, it passes if the operator typed `PASS`.
The committed log's `D5SUMMARY,controls,c1=PASS` is an operator-supplied string.

The *design* is defensible and honestly documented: A1.3 requires a between-invocation `sd` over ≥4
separate processes, which one process genuinely cannot compute. **The problem is that the external
computation has no artefact.** §12.5.1 and §12.5.2 cite "14 independent `d5c1` invocations, grand mean
22.07 GB/s, between-invocation sd 4.01%, PASS at ±3σ — **see sec.12.4**". §12.4 contains no such data: it
documents **ten** draws, grand mean **23.66**, min-based, pre-amendment, and closes with "**Control 1,
plainly: FAIL**". **The cross-reference behind the control-1 verdict points at a section that says the
opposite.** No `d5c1` log is committed; `git ls-files benchmarks/donor_adaptation/results/` returns only
`d5_raw.log`.

Partial mitigation, and it matters: this run's own live datum, printed `informational_only`, is
`mean_gbps_t6 = 22.6899` — consistent with the claimed 22.07 and +6.8% against banked. So I have
corroboration that the number is in the right place. I do not have the artefact that makes the verdict
auditable. **Fix is cheap: commit the 14 `d5c1` invocations and repoint the reference.**

---

## FINDING 9 — FLAG. Both halves of Claim 1's load-bearing ratio are prose-only, and the manifest's git revision does not contain the code that produced one of them.

- `22.07` — no artefact (finding 8).
- `10.38` — no artefact. The `d5g` numbers in §12.5.3 (37.34 / 24.54 / 31.31 / 24.75 / 11.39 / **10.38**)
  appear nowhere but that prose table. `grep -rn "10\.38\|37\.34" docs/` returns only `DONOR_PROJ_RATE.md`.
  No command line, no `reps`-per-invocation record, no thread-placement record for those runs beyond the
  claim that they match.
- §12.5.1 records `git revision = ad5cf075...`. **`git show ad5cf07:benchmarks/donor_adaptation/gemv_donor_bench.c | grep -c mode_d5g` → `0`.** `mode_d5g` first exists at `6b7e695`. The recorded revision cannot
  reproduce the §12.5.3 numbers. (This is the normal artefact of running against a dirty tree before
  committing — it is honest, and it is still not reproducible as recorded.)
- `n=1` invocation at the donor end against `n=14` at the reference end. With demonstrated
  between-invocation dispersion of 4.01% (8.36% the day before), a 2.13× effect is far outside noise —
  **the direction and rough size are safe.** The three-significant-figure "2.13×" is not; it should read
  "roughly 2×".
- The two points differ in `D` (1536→8192), in `HID/D` (5.83→3.50) and in per-layer block (19.7 → 336 MiB)
  simultaneously. All three are donor properties, so as an answer to "does the constant transfer to the
  donor" this is legitimate — but §12.5.6's "matched methodology" invites reading it as a width-only
  isolation, which it is not.

**One thing that does *not* bite here, and I want it on the record:** finding 3's byte-accounting problem
does **not** invalidate Claim 1 for its intended use. `ADAPTER_MEMO_01`'s tok/s arithmetic multiplies the
rate by *code bytes per token*, and both 22.07 and 10.38 are charged-code-byte rates. Same accounting on
both sides, same accounting downstream. **Claim 1 is usable.**

---

## FINDING 10 — FLAG. The `q/o` anomaly is load-bearing for §12.5.5, and the write-up picks its lowest of seven rows.

Asked directly: does a §12.5 conclusion depend on the unexplained point? **Yes.** §12.5.5 concludes "block
size (not which axis is `M` vs `K`) being the driver once both are deep in the streamed regime" — and `q/o`
falsifies that inside the same table: 32 MiB → ~11 GB/s while `D=4096` at 28 MiB → 20.66 GB/s. The
write-up flags `q/o` as unexplained and then keeps the block-size reading anyway. **A conclusion is being
carried across a data point that contradicts it.**

Second, smaller point: §12.5.5 quotes `q/o` at **8.92**. The seven `q/o` `t6` rows are 11.63, 11.58, 11.48,
11.07, 10.97, 11.00, **8.92** — the quoted figure is the single lowest, and it is the only one below 10.9.
The anomaly is real and robust at ~11.0 (it never approaches 20), but taking the lowest of seven without
saying so sharpens it. At 11.0 the "`k/v` is 3.3× faster" becomes 2.7×.

Third, and this is why it matters beyond §12.5.5: **the same unexplained regime break — LUT footprint
crossing 64 KB and/or `EB` crossing L3 — is what produces the `D=8192` collapse that Claim 2 rests on.**
`T ≤ 2560` (LUT ≤ 40 KB) → 20.5-23.2 GB/s; `T ≥ 4096` (LUT ≥ 64 KB) → 8.9-11.3 GB/s, with `k/v` the lone
exception at 29.2. The anomaly is not a curiosity beside the result; it is the same phenomenon.

**Reproduction:** `grep "organs,llama70b_qo" d5_raw.log | awk -F, '$7==6'`.

---

## FINDING 11 — PASS. Claim 3's explanation is arithmetically consistent with its numbers.

Checked directly rather than accepted. Per-layer integrated block = `2·EB(HID,D) + EB(D,HID)`:

| `D` | `HID` | per-layer block | vs 16 MiB L3 | measured `L2/L28` ratio |
|---|---|---|---|---|
| 1536 | 5376 | **11.81 MiB** | **UNDER** | **1.522×** |
| 2048 | 7168 | 21.00 MiB | over | not measured |
| 4096 | 14336 | 84.00 MiB | over | not measured |
| 5120 | 17920 | 131.25 MiB | over | not measured |
| 8192 | 28672 | **336.0 MiB** | far over | **1.097×** |

The offered mechanism — at `D=1536` one layer sits *just under* the cliff so layer count decides whether
the working set behaves cache-assisted or cold; at `D=8192` every layer is 21× past the cliff so layer
count barely changes the access pattern — **is exactly what the arithmetic says.** The write-up's own
"12.4 MB" and "352 MB" are the same figures in decimal MB. **This one is right.**

Minor: the write-up mixes decimal MB into a table that is otherwise MiB throughout.

---

## FINDING 12 — FLAG. The unmeasured `D`-sweep rows are honestly flagged in substance, weakly in placement — and the committed raw log still asserts the false premise.

*On the flagging:* §12.5.3 states the dsweep table is "an **upper bound** on the true `L=28` curve, not a
faithful stand-in", names the confirmed overstatement at both ends (~10% and ~52%), and refuses to
interpolate the middle. That refusal is **more** cautious than its own mechanism requires — the mechanism
predicts monotone decline from `D=2048` onward — so the substance is honest, not evasive. Nothing
downstream leans on those rows: Claim 1 uses the `d5g` point, Claim 2 uses kernel-pure rows with no `L`.

*Where a reader would misread:* the table is headed "**Raw sweep**", carries five clean rates and no
per-row marker, and the caveat lands three paragraphs below. A reader lifting the table lifts inflated
numbers. **Recommend marking `D=2048/4096/5120` in-row.**

*The part that is not just presentation:* the committed raw log carries the harness's own printed licence
for `L=2` — "*every layer's working set already exceeds L3 at every D swept, so this is not a shortcut that
changes the regime*" (`gemv_donor_bench.c:1119-1121`). **That is false at `D=1536`, where the layer is
11.81 MiB against a 16 MiB L3 — and `D=1536` is exactly the point where the `L=2` proxy inflates by 1.52×.**
The source comment at `gemv_donor_bench.c:1060-1061` compounds it, justifying the shortcut with "20.6 MB
per layer at `D=1536,HID=8960`" — a figure from the control-1 shape, not the sweep's. The write-up
corrects the conclusion; the harness text and the committed log still state the false premise.

---

## FINDING 13 — FLAG. Estimator discipline: A1.2 compliant, A1.5 partly, **A1.3 breached on 72 of 74 timing rows**, **A1.4 undischarged and not declared so**.

**A1.2 — PASS.** The instrument note is present at `DONOR_PROJ_RATE.md:445` (§8) and `:577` (§10), verbatim,
appended, and it does not retro-edit a single old number. It does what it says. Checked.

**A1.3 — BREACHED, disclosed, immaterial to the conclusions.** A1.3 requires `min`, `median`, `mean`, `sd`,
`cv` **and** `reps` on **every** timing row. Delivered on 2 rows (`D5A1CSV,control1,nt=1/nt=6`) and on the
uncommitted `d5g` rows. Every other timing row — lsens (2), control23 (8), control4 (4), dsweep (10),
fp32_vs_ternary (20), organs (28) = **72 rows** — reports `min`-derived GB/s, `us`, `cv` only. No median,
no mean, no sd; `reps` lives in the header, not the row. `mlp_point` (`:948`) has the plumbing and passes
`NULL,NULL,NULL` to `mlp_time`; `time_fp32` computes `gbps_mean` (`:207`) and every call site discards it.
Compliance is a few lines away and was not taken.

**Impact, stated honestly:** min-of-5 biases upward by ≈1.16σ. Load-bearing rows have `cv` 0.7-2.5%, so the
bias is ~1-3% per arm and partly cancels in ratios — which is A1.5's stated rationale. **This
non-compliance does not by itself move any conclusion, and I am not going to claim it does.** What it does
cost is that the write-up's headline pair (10.38 / 22.07) is mean-based while all of §12.5.4 and §12.5.5 is
min-based, and A1.5's requirement that absolute figures "carry their estimator and `reps` in the label" is
met only in prose, not in the tables.

**A1.4 — UNDISCHARGED, and not listed as open.** A1.4 required determining whether **invocation order
predicts the rate**, and — if a warm-up effect is confirmed — writing the handling into the protocol for
every number the harness reports from then on. §12.5 reports a between-invocation `sd` (4.01%) but **no
rate-vs-invocation-index analysis and no warm-up-handling statement.** §12.5.7's "what remains open" list
does not mention it. Given that §12.4's own four matched draws were 20.203 / 24.762 / 24.763 / 24.755 —
first low, last three agreeing to four significant figures — this is the live hypothesis, and it is
unaddressed. It also bears directly on finding 9: a single `d5g` invocation at `D=8192` is exactly the
draw a cold-start effect would corrupt.

---

## FINDING 14 — FLAG. The harness prints a data-quality rule and then does not enforce it; the write-up quotes rows that fail it.

Every run header prints: "`cv% > 5% => the machine was not quiet, do not use the row.`" No code path
enforces it. Rows that breach it and are used anyway:

| row | `cv%` | used where |
|---|---|---|
| `control4,fp32_128MB` = 38.3910 | **13.40** | folded into control 4's mean unconditionally (`gemv_donor_bench.c:1101`) |
| `dsweep,D8192,nt=6` = 10.6884 | **9.13** | quoted in §12.5.3's table |
| `organs,llama70b_gateup,469762048,nt=6` = 11.0472 | **7.55** | quoted as 11.05 in §12.5.5 |
| `organs,llama70b_qo,268435456,nt=6` = 11.0011 | 8.72 | in the `q/o` series |
| `organs,llama70b_kv,16777216,nt=6` = 58.5219 | **17.03** | not quoted |

Control 4's verdict is unaffected — dropping the 13.40% row gives 38.826, *closer* to banked than 38.717.
**Real non-compliance, immaterial to that verdict.** The dsweep and gate/up cases are quoted without the
breach being noted.

---

## FINDING 15 — FLAG. ACHIEVED vs REQUESTED: every row's thread count and shape are printed from the arguments, not read back.

The pattern the sister probe was just caught on is present here, in a milder form.

- `set_threads()` (`:155`) calls `omp_set_num_threads(nt)` and **never reads `omp_get_num_threads()` inside
  a parallel region**. Every `D5CSV` row's `threads` column is the requested value.
- `mlp_point` (`:956`) prints `D, HID, L, nt` **from its arguments**, not from the constructed `MlpStack`
  (which holds `S->D`, `S->L`, `S->Mpg`, `S->TUP`). Same for `lut_one_point` (`:990`) and
  `fp32_streamed_point` (`:1146`) printing `M, K`.
- The header prints `omp_get_max_threads()` = 12 — the machine maximum, not the achieved count for any row.
- **Header field that can lie:** `# ... gather=%s` prints from the parsed `iid` flag, but `d5` mode passes
  a hardcoded `1` at all three `lut_one_point` call sites (`:1073-1076`, `:1185`, `:1216`). Invoking
  `d5 --seq` would print `gather=seq` for a run that gathered `iid`. Requested printed as achieved, in the
  provenance header itself.

**Partial credit where due:** `working_set_bytes` **is** achieved (`S.codes_layer*L`, or the computed
`pool`), and §12.4 documents a live `omp_get_place_proc_ids()` / `SetThreadAffinityMask` round-trip that
verified 6 threads on 6 distinct physical cores. That probe was a **separate binary in a separate session**,
not this run, and its result is asserted for this run rather than recorded by it.

---

## FINDING 16 — PASS. `d5g` is additive and gated. One caveat about the commit it arrived in.

Verified line by line:

- `mode_d5g` (`:1039-1050`) is reachable only via `argv[1]=="d5g"` (`:1300`). No existing dispatch changed.
- `mlp_time`'s signature gained three **optional** out-params. Every pre-existing call site passes `NULL`:
  `mlpint_row` (`:804-805`), `mlp_point` (`:951`). `*us_tok` and `*cv` are computed byte-identically to
  before (`best/ntok*1e6`; `100.0*sd/m`). **No numeric path moved.**
- The new per-rep `samp` array, its O(reps²) sort and the median are all **outside** the timed window.
- `getenv("GEMV_D5_DEBUG_REPS")` is read once before the loop, never inside it.
- `--D/--HID/--L/--nt` are parsed for all modes but consumed only by `d5g`; previously-unknown-arg
  behaviour is otherwise unchanged.
- `mode_d5g` touches no control's pass/fail computation.

**The Builder's claim about `d5g` is accurate.** The caveat is about scope, not `d5g`: the *same commit*
(`6b7e695`) also converted control 1 from a computed numeric gate into an environment-variable read
(finding 8). That is a change to existing pass/fail logic, it is mandated by Amendment 1 and documented at
`:1000-1004` — but "additive, with existing call sites updated for the new returns" describes `d5g` and
does not describe the commit.

---

## Explicit statement on Claim 2, as requested

> **"The donor-width ternary path is compute-bound, therefore 5-trit packing does not pay" is NOT safe to
> act on. It needs the causal test — and the causal test alone is no longer sufficient.**

Three separate reasons, in descending order of severity:

1. **It is contradicted, not merely unsupported.** The identical kernel spans 8.92-29.23 GB/s across shapes
   (finding 2). Bytes-per-instruction is a compile-time constant, so a compute-bound `matvec_lut_full`
   would be shape-invariant. And at a real donor organ, fully streamed, it reaches **76% of the fp32 DRAM
   ceiling** — a figure §12.5.4's "29-33% at every `D`" silently excludes.
2. **The instrument that produced the evidence has a byte-accounting defect that biases exactly this
   inference** (finding 3), and the defect's magnitude grows with `EB` — i.e. it grows in lockstep with the
   `ratio_fp32_over_ternary` curve that is read as the primary signature.
3. **The decision the claim licenses was never priced.** No branch of the pre-registered dichotomy costs an
   actual 5-trit kernel, which cannot use the `pshufb` LUT scheme at all (finding 5).

**What I would need to lift the BLOCK.** The Builder's proposed ablation — strip the LUT gather/accumulate,
stream identical bytes — is the **right discriminator and should be built**; my objection is not to it but
to concluding without it. It now needs one addition to be conclusive: **the ablation must be run at two
shapes on opposite sides of the break** (`k/v` 1024×8192 at 76% of ceiling, and `gate/up` 28672×8192 at
29%), not at one, because the thing being explained is a *shape-dependent* rate. If both shapes' rates jump
when compute is stripped, compute-bound survives and I withdraw. If the fast shape holds and the slow shape
jumps — or if neither jumps — the binding constraint is in the memory hierarchy and the packing question
reopens in the *favourable* direction.

**Cheaper still, and I would run it first** (~10 minutes, no new kernel): re-run the existing kernel at a
sequence of shapes holding `T` fixed at 4096 and varying `Mpad` across 1024 / 2048 / 4096 / 8192 / 28672 at
a fixed streamed pool. If the rate tracks `Mpad`/`EB` rather than staying flat, finding 2 is confirmed
independently and the compute-bound label is dead without writing an ablation kernel at all.

**On the direction of the flattering error, since I was told to weigh it.** This result is convenient — it
closes an expensive question cheaply and retires a lever. The specific shape of the convenience is that
**every defect I found pushes the same way**: the charged-byte accounting under-counts the ternary arm and
not the fp32 arm; the "29-33% of ceiling" figure excludes the one donor-shape row at 76%; the two
"independent" signatures are one measurement; the anomaly that contradicts the model is flagged and then
worked around. None of that is fabrication and I do not think any of it was deliberate — but a negative
result whose every error leans toward "stop looking" is exactly the case where the rule bites hardest.

---

## Checks I could not run, and why

I chose to run **no timing and no compute at all** — cost to the machine: zero. Everything above comes from
the committed log, the committed source, and arithmetic. That was a judgement that the log already contains
the refutation; it leaves these open:

1. **Direct measurement of actual DRAM traffic vs charged bytes** (finding 3). Needs hardware performance
   counters (`amd_uprof` / `perf` `l3_fills`, `dram_channel_data_controller`) or an equivalent proxy.
   Nothing in this harness can see it. **This is the single check that would settle findings 2-3 outright.**
2. **The `T`-fixed / `Mpad`-varied sweep** proposed above (~10 min). I did not run it because it needs a new
   invocation path and I was asked not to implement.
3. **The compute-ablation causal test.** Not built; the Builder's 30-60 min estimate looks right, plus a
   second shape.
4. **Reproducing `22.07` and `10.38`.** No artefact exists to check against (findings 8-9); reproducing
   them means re-running 14 `d5c1` invocations plus the `d5g` points, which is not a short check.
5. **Whether invocation order predicts rate at donor shapes** (A1.4, finding 13). Needs ≥4 `d5g`
   invocations at `D=8192`, order recorded.
6. **Live read-back of achieved thread count/affinity during the sweep rows** (finding 15). Needs a harness
   change.
7. **Whether the `q/o` 8.92 outlier reproduces.** Six other rows put `q/o` at ~11.0; I did not re-draw it.

---

*Controller, 2026-08-27. No commit, no push. Nothing in §12.5 may enter `ADAPTER_MEMO_01` or any decision
document while this BLOCK stands. Claims 1 and 3 are releasable on the Principal's judgement once findings
8, 9 and 12 are addressed; Claim 2 is not.*
