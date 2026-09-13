# BRIEF E49 — THE ATTENTION KERNEL WAS NEVER SWITCHED ON

**Registered before any E49 cell exists.** No speed claim in this brief may be quoted until
`G-E49a` (parity) passes; the Phase 60 law is that a kernel change is not a result until the
system is shown to compute the same thing.

## 1. The finding

`benchmarks/donor_adaptation/engine/donor_engine.c:202`:

    static int g_attn=ATTN_SERIAL;

The default attention kernel is the **scalar serial dot product**. `--sweep6` sets every one of
its ten arms to `ATTN_AVX4` (line ~1336). And **no runner from E26 onward passes `--attn`**:

* `bench()` — `e28_kernel_transfer.py:132` — builds `[engine, --weights, w, --threads, n] +
  flags + [--bench, N]`, with no `--attn`.
* `one_rep()` — `e44_interval.py:171` — the same.
* `e40_levers_exhausted.py:213`, the origin of the 112.7 tok/s headline, calls
  `bench(..., ["--carve-k", str(k)])`.

Only the E3/E4/E5 family ever set the kernel. **Therefore E36, E37, E39, E40, E43, E44, E45, E46,
E47 and E48's own pure-`none` control all ran scalar attention.**

Smoked on the target arm (`e40_r128.bin`, `--carve-k 3`, 6 threads, window 1280 = mean position
640, two reps each):

| kernel | wall | tok/s |
|---|---|---|
| `serial` — the default | 29.189 / 29.394 s | 43.85 / 43.55 |
| `avx4` | **18.801 / 18.345 s** | **68.08 / 69.77** |

**+56% at realistic context.** Pure `avx4` is also *faster* than `--sweep6` (19.64–20.50 s), which
runs `avx4` **plus** six doubled arms — so the whole effect is the kernel and there is no residual
neighbour-warming term. These are smoke numbers, two reps, and they are not the measurement.

## 2. How it was found, which is the part worth keeping

Not by reading the source. The chain was:

1. E48 addendum A.4 registered a **drift line** comparing E48's slope to E46's, and explicitly
   **demoted it to "not a gate"** so that E48 could not overrule E46.
2. That demoted line showed a **2.5× disagreement** on a quantity whose intercepts agreed to 2.6%.
3. That forced B.5's adjudication, which fired **THE SWEEP INSTRUMENT IS BIASED**.
4. The branch was right. **Its canned reason was wrong** — B.5 wrote "a regime the engine never
   runs in", and it is a regime the engine *can* run in and should.

**A closure gate could not have caught this, and mine didn't.** `G-E48b2` compared the whole-token
slope against the attention-organ slope and **fired at 6.4%** — both measured under the same
kernel, so both carried the same defect. **A closure test between two quantities that share a bias
is blind to that bias.** That is a new law and it is going in the record next to
`feedback_gate_vs_measured_dispersion`.

And it hid for twenty-two experiments for the **same reason E46's context cliff hid**: at
`NTOK = 40` attention is a sliver of the token, so the kernel is nearly invisible there. One blind
spot, two symptoms.

## 3. What reconciles, and what does not change

| | kernel | `b` (ms/pos) | `C50` |
|---|---|---|---|
| E46 phase B | `serial` | 0.02031 | 575 |
| E48 under `--sweep6` | `avx4` | 0.00818 | 1454 |

**Neither measurement was wrong.** They measured different kernels. `C50 = 575` is the correct
`serial` number and E48's B.5 control reproduced it four times over (515 / 540 / 544 / 575).

Consequently **E48 addendum B.1 is re-scoped, not withdrawn**: its shares are a valid
decomposition *of the `avx4` attention organ*, validated within that kernel by `G-E48b1`'s 3× test
(6/6 at the gated windows). And the softmax rising to 33.1% there is exactly what the kernel
change predicts — `avx4` vectorises the Q·K and A·V dot products and does **nothing** for the
scalar `expf` softmax. B.2 (S15 void) and B.3 (`G-E48d` void) stand unchanged.

## 4. The gates

### `G-E49a` — PARITY, and nothing else counts until it passes

`avx4` is **not** bit-identical to `serial`: E4 published `NATS_TOTAL` **166667.2449386003**
(`serial`) against **166667.1361128952** (`avx4`), a difference in summation order. So bit-identity
is the wrong test and **end-to-end parity is the right one** — this is Phase 60's law, that
kernel-level agreement does not compose to system correctness.

* Run `--bpb` on **S15** (`e37_carved_nf.bin`, real weights, real corpus) under both kernels.
* **`|ΔBPB| < 1e-4`**, i.e. fifty times inside `σ_seed = 0.005`.
* **PLANTED CONTROL, and the null does not count without it.** The same harness, unchanged, must
  be shown to **FIRE** on a difference that is really there. Control: the same comparison run
  against a deliberately different artifact (`e37_dense_nf.bin`), which must return a `ΔBPB` orders
  of magnitude larger. An instrument that cannot see a real difference has not shown that this one
  is absent.

If `G-E49a` fails, **every speed number in E49 is void** and `serial` stays the kernel of record.

### `G-E49b` — the speed, measured rather than smoked

Both kernels, **pure** (no `--sweep6`, no `--profile`), windows **{40, 160, 640, 1280, 2560}**,
**5 reps**, round robin with reps outermost. Refit `ms/tok = a + b·pos` per kernel and recompute
`C50` per kernel. Reported with dispersion; every absolute carries the standing ±5%, the ratio does
not.

### `G-E49c` — does the published headline move? (two-sided, and registered as such)

At **`NTOK = 40`** — the context at which *every* published number in this programme was taken —
the two kernels must differ by **less than ±5%**, the band the ledger already carries, **or the
112.7 tok/s headline is restated in E49 rather than defended.** This gate can go either way and is
written to be able to.

### `G-E49d` — the ladder, so that "`avx4` is the best available" is measured

`serial`, `serial2`, `serial3`, `ilp4`, `avx1`, `avx4` at window 1280, 3 reps. Cheap, and it stops
`avx4` becoming the answer merely because it is the one I happened to smoke.

## 5. My prediction, registered

| quantity | prediction |
|---|---|
| `G-E49a` parity | **passes**, `ΔBPB` below 1e-5 — it is a summation-order change in a 128-element dot product |
| `G-E49a` planted control | **fires**, `ΔBPB` above 0.1 against the dense artifact |
| `G-E49b` `b(avx4)` | 0.008–0.010 ms/pos |
| `G-E49b` `C50(avx4)` | **1100–1500** |
| `G-E49c` headline at `NTOK = 40` | moves **less than 5%** — the headline survives, scoped as it already is |
| `G-E49d` | `avx4` fastest; `avx1` and `ilp4` between it and `serial` |

My last four registered predictions scored half, one-of-three, zero-of-three and two-of-six. This
one is recorded on the same terms.

## 6. What E49 does not claim

It does not claim a new headline, it does not revise `C50` until `G-E49b` measures it, and it does
not touch quality. It also does **not** change the engine's default — changing `g_attn`'s
initialiser is a separate, gated change, and E49's job is to establish first whether it deserves to
be made.

---

# ADDENDUM A — `G-E49a` RETURNED VOID, AND THE CONTROL I CHOSE COULD NEVER HAVE FIRED

**Written against `e49_run.log` (run `3ee8735`), before the re-run.** Only the *identity of the
planted control artifact* changes. The bar, the `ΔBPB`→relative mapping, the VOID logic and §5's
prediction are untouched.

## A.1 What the first run returned

| | `NATS_TOTAL` | |
|---|---|---|
| S15 `serial` | 124963.9729339122 | |
| S15 `avx4` | 124963.9517608703 | pair: **1.694e-07** relative, i.e. `|ΔBPB| ≤ 8.5e-07` |
| control `e37_dense_nf.bin` `serial` | 124963.9628600173 | control: **8.061e-08** apart, bar 1e-03 |

**`G-E49a`: VOID.** The runner refused to run the speed phase, as §4 requires. The pair reading
looks excellent and **it does not count**: an instrument not shown capable of firing says nothing
when it is silent.

## A.2 The control could not have fired, and the record already said so

`e37_dense_nf.bin` and `e37_carved_nf.bin` are **equivalent by construction when the carve is
fully on**, and E37 measured exactly that — `results/e37_sparsity_cost.json`:

    G_E37A/bpb_carved_kE   3.4757066520304316
    G_E37A/bpb_dense       3.47570637184527
    G_E37A/bpb_diff        2.801851617384443e-07     tol 1e-4

**That was E37's own parity gate and it passed.** I chose, as my proof that the harness can see a
difference, the one pair in the programme that is *proven* to have none. This was knowable before
the run from a results file already in the repo, and it cost 28 minutes of engine time.

A second thing I read wrong on the way: 10.19 nats/token looked like near-uniform garbage and I
took it for a corpus mismatch. It is **correct** — these artifacts read **3.476 BPB against a
chance of 4.070** (E37 `S15_DENSE_NF/bpb`, `protocol/chance_bpb`). They are genuinely near-chance
models. The engine was never misbehaving.

## A.3 The replacement control, from E37's own table

One artifact, one flag. `e37_carved_nf.bin` has `carve_E = 256`, so `--carve-k 256` is the full
carve and `--carve-k 3` is the target operating point. E37 measured both:

    quality_arms/K256/bpb   3.4757066520304316
    quality_arms/K3/bpb     4.029398350226611     -> 0.5537 BPB apart

**0.5537 BPB is 5,537× the gate's own 1e-4 bar.** This control is *provably capable* of firing,
from a measurement that already exists.

The parity **pair** is therefore also pinned to `--carve-k 256`, so both kernels are compared at
one fixed operating point rather than at the artifact's default.

Smoked on a 2-sequence prefix (`D:/_ktmp/e49_ids_tq_2seq.bin`, 1024 tokens) before registering:

| run | `NATS_TOTAL` | |
|---|---|---|
| `--carve-k 256` `serial` | 9215.2183202812 | |
| `--carve-k 3` `serial` | 11199.3301610758 | **control 21.5% apart → FIRES** |
| `--carve-k 256` `avx4` | 9215.2209920680 | pair **2.899e-07**, inside the 2e-5 bar |

## A.4 Why this is a correction and not a gate re-rolled until it passed

This programme's two standing rules here are E40 addendum A (**a gate that fires is doing its job
and is not re-run to a pass**) and the E36 run-2 rule (**run 1 is the registered measurement; run
2 may not promote it**). Neither is being bent, and the distinction matters:

* `G-E49a` did not *fire against the treatment*. It returned **VOID**, which means **no
  measurement was made**. There is no verdict to re-roll and nothing to promote.
* The control's incapacity is **provable from the prior record, independently of the numbers it
  returned**. I could have established it before running and did not. It is not being replaced
  because its answer was inconvenient; it is being replaced because it was never an instrument.
* **Run 1's pair reading of 1.694e-07 is NOT carried forward.** It was taken under a void gate.
  The re-run is the measurement of record, and if the re-run's pair disagrees with 1.694e-07, the
  re-run wins.

Corpus for the re-run: the full 24×512 slice (`ids_qwen25-15b_tq.bin`), as E1 and E37 use. The
truncated prefix in A.3 is a smoke and is not the measurement.

---

# ADDENDUM B — E49 CLOSES: PARITY HOLDS, `C50` GOES 608 → 1443, AND THE HEADLINE IS RESTATED

**Results: `results/e49_attn_kernel.json`, log `e49_run2.log`, run at `917ea94`.**

## B.1 `G-E49a` — PARITY HOLDS

| | `NATS_TOTAL` | |
|---|---|---|
| S15 `--carve-k 256` `serial` | 124963.9729339122 | n = 12,264 |
| S15 `--carve-k 256` `avx4` | 124963.9517608703 | pair **1.694e-07** relative → `|ΔBPB| ≤ 8.47e-07` |
| CONTROL `--carve-k 3` `serial` | 144871.1519082308 | **1.593e-01** apart, bar 1e-03 → **FIRES** |

The bar is 2e-05 relative (what `|ΔBPB| < 1e-4` implies); the pair is **118× inside it**. The two
kernels compute the same thing end to end. Phase 60's law is satisfied without claiming
bit-identity, which `avx4` does not have and was never going to have.

The pair reproduced addendum A.1's value **to the last digit** — the engine is deterministic — but
it counts now and did not then, because only now has the harness been shown to fire.

## B.2 `G-E49b` — the speed, and this is the result

`e40_r128.bin`, `--carve-k 3`, 6 threads, pure (no `--sweep6`, no `--profile`), 5 reps, kernels
interleaved at rep level.

| | `serial` (the default) | `avx4` | ratio |
|---|---|---|---|
| `a` — position-0 cost (ms) | 8.2419 | 8.2468 | **1.0006** |
| `b` — context slope (ms/pos) | 0.019334 | **0.008144** | **2.374** |
| **`C50`** | **608** | **1443** | **2.37** |
| tok/s at mean position 20 | 112.19 | 121.65 | 1.084 |
| tok/s at mean position 320 | 69.71 | 89.89 | 1.289 |
| tok/s at mean position 640 | 48.59 | 74.11 | 1.525 |
| tok/s at mean position 1280 | 30.27 | **53.79** | 1.777 |

Dispersion at the three large windows: 2.3–5.9%. At the two small ones it is 7.9–26.1%, which is
E43's intrinsic oscillation showing up where the per-cell time is under two seconds; the fit is
dominated by the large windows and the small ones are reported rather than leaned on.

**The intercepts agree to 0.06%.** The kernel moves the context term and nothing else — which is
what §1 read out of the source, now confirmed by measurement at a place the source could not
promise it.

**Three independent corroborations, none arranged:**

1. `avx4`'s `C50 = 1443` lands **0.8%** from E48's sweep-derived **1454**. That is the strongest
   possible confirmation that E48's `--sweep6` was measuring `avx4` all along, arrived at by a
   different route on a different day.
2. `serial`'s `C50 = 608` sits with E46's **575** and B.5's **515 / 540 / 544**.
3. `serial` at `NTOK = 40` reads **112.19 tok/s** against E40's published **112.73** — the
   headline reproduces to **0.5%**, which also settles that the headline was a `serial` number.

## B.3 `G-E49c` — THE HEADLINE IS RESTATED, and it was a two-sided gate

**+8.4%** at `NTOK = 40` (112.19 → 121.65 tok/s), outside the ±5% band the ledger carries. The
gate was written to be able to go either way and it went the way I did not predict.

**So the published figure becomes `121.65 tok/s` at mean context position 20 on `avx4`**, and
`112.7` is retained as the `serial` number it always was. Neither is quotable without its context
position and now also without its kernel.

## B.4 `G-E49d` — the ladder

| kernel | tok/s at n=1280 | vs `serial` |
|---|---|---|
| **`avx4`** | **73.55** | **1.516×** |
| `avx1` | 72.56 | 1.495× |
| `ilp4` | 63.08 | 1.300× |
| `serial` (default) | 48.53 | 1.000× |
| `serial2` | 34.61 | 0.713× |
| `serial3` | 27.15 | 0.559× |

`avx4` is fastest, and `avx1` is within 1.4% of it — so most of the win is *any* vectorisation,
not the four-accumulator version specifically. Two of the six arms are **slower than the default**,
which is why the ladder was worth running: "switch the kernel on" is not uniformly good advice.

That `avx1` gap of 1.4% is the **portability** reading, and it is good news under
`feedback_portability_no_hardfit`: the win belongs to the **x86-64-v3 class**, not to a
four-accumulator schedule tuned to this 3600X. A machine where the four-accumulator version
schedules badly still collects ~1.50× of the 1.52×.

## B.5 What this buys the goal, stated with its limits

**The 10B target arm holds above 50 tok/s out to ~1443 tokens of context instead of ~608, and
reads 121.65 tok/s at short context** — past the 100 tok/s "ottimo" mark, and still **53.79 tok/s
at mean position 1280**. The cost of obtaining this was one flag that has been in the engine since
E4.

What it does **not** change:

* **The weights are synthetic.** E45 bounds any value-dependence of speed well under 5% with no
  reproducible sign, but that is a bound and not a measurement at 10B.
* **`--carve-k 3` is 1.17% of the FFN**, and quality at that operating point is near chance
  (E37: 4.029 BPB against a chance of 4.070). **Speed is real; quality at that carve is not
  there.** This is the standing gap and E49 does not touch it.
* Every absolute above carries the standing ±5%; the ratios do not.

## B.6 My prediction (§5), scored

| quantity | predicted | measured | verdict |
|---|---|---|---|
| parity passes, `ΔBPB` < 1e-5 | passes | 8.47e-07 | **right** |
| control fires | fires | 1.59e-01 | right, but **the control artifact was replaced** by addendum A, so the clause naming the dense artifact is void |
| `b(avx4)` | 0.008–0.010 | **0.008144** | **right** |
| `C50(avx4)` | 1100–1500 | **1443** | **right** |
| headline moves < 5% | survives | **+8.4%, restated** | **WRONG** |
| `avx4` fastest, `avx1`/`ilp4` between | as stated | exactly that | **right** |

**Four right, one wrong, one void.** The best scorecard of the session, and the one I got wrong is
the one where I assumed the context-20 blind spot would protect the old number.

## B.7 What is now owed — and the fix for the *class* of defect

1. **The engine's default should change to `avx4`, and E49 has established that it deserves to.**
   Parity holds, it beats `serial` at all five windows of `G-E49b` and wins the `G-E49d` ladder at
   1280, and leaving the fast kernel behind a flag is how this went unnoticed for twenty-two
   experiments. That is a one-line change to
   `donor_engine.c:202` and it gets its own parity gate rather than riding on this one.
2. **The real fix is that the engine must say which kernel it ran.** The `BENCH` line prints
   threads and quantisation and not the attention arm. Had it printed the arm, this would have
   been visible in every log since E26. **A configuration that does not appear in the output is a
   configuration nobody can audit** — that is the generalisable lesson, and it is worth more than
   the 2.37× it cost to learn.
3. **E48's B.1 shares stand, re-scoped to `avx4`** (see §3), and their RANK partner is still
   missing, so the softmax's 33.1% remains a score at one shape and not yet a ranked lever.
4. **Everything measured between E26 and E48 is a `serial` number.** Nothing is wrong, but the
   ledger's speed entries now carry a kernel as well as a context. The ones that matter —
   E40's 112.7 and E46's `C50` — are reproduced above and restated rather than discarded.

---

# ADDENDUM C — `G-E49c` IS WITHDRAWN AS UNRESOLVABLE. THE HEADLINE IS NOT RESTATED, AND I MIS-SPECIFIED THE GATE FOR THE THIRD TIME

**Written from run 2's own cells. No cell is re-run, no arm is added, and nothing in `G-E49a`,
`G-E49b` or `G-E49d` is touched.** This addendum only removes a claim.

## C.1 What the check is

`G-E49c` compared the two kernels' **medians** at `NTOK = 40` and read `+8.4%` against a `±5%`
band. `feedback_gate_vs_measured_dispersion` says a gate whose tolerance is tighter than the
dispersion measured on the axis it watches is mis-specified. The dispersions at that window,
printed by the runner itself in the same table, are **21.0% (serial)** and **7.9% (avx4)**.

So I asked whether the two five-rep sets are even separable there.

| n | `serial` [min..max] med | `avx4` [min..max] med | Δ median | separable? |
|---|---|---|---|---|
| **40** | [98.77 .. **121.38**] 112.19 | [117.26 .. 126.79] **121.65** | +8.4% | **NO — the ranges overlap** |
| **160** | [85.91 .. 109.36] 104.60 | [100.10 .. 119.29] 112.89 | +7.9% | **NO — the ranges overlap** |
| 640 | [66.92 .. 69.96] 69.71 | [88.97 .. 92.44] 89.89 | +28.9% | yes, clean |
| 1280 | [47.35 .. 49.11] 48.59 | [70.71 .. 74.92] 74.11 | +52.5% | yes, clean |
| 2560 | [29.73 .. 30.57] 30.27 | [53.13 .. 54.35] 53.79 | +77.7% | yes, clean |

**At `NTOK = 40`, `serial`'s best repetition (121.38) is `avx4`'s median (121.65).** The two arms
are not distinguishable at the window `G-E49c` was registered on.

## C.2 What the same run says the effect actually is there

`G-E49b`'s fit uses all five windows and is dominated by the three whose dispersion is 2.3–5.9%.
Its estimate of the kernel's effect, window by window:

| n | mean pos | Δ (ms) | as % of the `avx4` token | cells said |
|---|---|---|---|---|
| **40** | 20 | **0.224** | **+2.7%** | +8.4% (overlapping) |
| 160 | 80 | 0.895 | +10.1% | +7.9% (overlapping) |
| 640 | 320 | 3.581 | +33.0% | +28.9% |
| 1280 | 640 | 7.162 | +53.2% | +52.5% |
| 2560 | 1280 | 14.323 | +76.7% | +77.7% |

Where the cells are separable the two columns agree to a few points. Where they are not, they
disagree — which is what "not resolvable" means, and it is the first two rows that carry the
disagreement.

**So the kernel's effect at mean context position 20 is ~2.7%, inside the ±5% the ledger carries.**

## C.3 The verdict, and how it is scored

**`G-E49c` is VOID — the gate could not answer the question it asked.** It is not recorded as a
pass, and my §5 prediction is **not** promoted to "right" on the strength of a re-analysis I ran
after seeing the answer. A void gate scores nothing in either direction. E49's scorecard is
therefore **4 right, 0 wrong, 2 void** (this and the control clause superseded by addendum A),
not the 5-of-6 addendum B claimed.

**The headline is NOT restated.** What replaces addendum B.3:

* At mean context position 20 the target arm reads **~112–122 tok/s on either kernel**, and the
  two are **not separable on this instrument**. E40's published **112.7** stands as the number at
  that context, and `avx4` is not shown to move it.
* **This does not touch `G-E49b`.** `C50` 608 → 1443, `b(serial)/b(avx4) = 2.374` and the
  intercepts agreeing to 0.06% all rest on the three large windows, whose dispersion is 2.3–5.9%
  and whose separation is total. **The result of E49 is intact; only its claim about the headline
  is withdrawn.**
* The honest one-line statement of the win is therefore about **context, not about the headline**:
  the kernel buys nothing measurable at context 20 and buys **+29% at 320, +53% at 640 and +78%
  at 1280**, which is exactly what a change to the context term and not the intercept must look
  like.

One consequence for the goal, stated plainly because I overstated it: **passing 100 tok/s at
short context was already true before E49** (E40's 112.7). It is not something the kernel bought.
What the kernel bought is the context at which 50 tok/s still holds.

## C.4 Third time, same law

`feedback_gate_vs_measured_dispersion` has now been broken three times by me:

1. **`G-E43A`** — a ±5% tolerance on an axis dispersing 9–22%: too tight, could never fire.
2. **`G-E45c`** — "inside my own dispersion" as a PASS condition: passed by being noisy.
3. **`G-E49c`** — a ±5% tolerance read off a cell dispersing 21.0%: **fired when it should have
   been unresolvable**, which is the first of the three to produce a false POSITIVE.

The first two produced nulls that went nowhere. This one produced a **published claim** that
reached the index, the ledger row, two memory files and a verbal report before I checked it. That
is the difference worth recording: a mis-specified gate that fails costs a run; one that fires
costs a retraction.

**The procedural fix, adopted here and owed to every future speed gate:** a gate on a rate may
only be registered against a window whose **dispersion has already been measured**, and the
runner must **refuse to judge** any cell whose observed spread exceeds the gate's own tolerance —
printing the numbers and `UNRESOLVABLE` instead of a verdict. `G-E49c`'s own runner printed the
21.0% two lines above the verdict it contradicted, and nothing connected them. That connection
belongs in code, not in my reading of a table.
