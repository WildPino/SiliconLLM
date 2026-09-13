# BRIEF E48 — WHERE, INSIDE THE ENGINE, DOES THE CONTEXT SLOPE LIVE?

**Status: PART 1 IS POST-HOC DESK ANALYSIS OF DATA ALREADY IN THE RECORD. IT CARRIES NO GATE.**
E14 §6 forbids promoting a post-hoc metric to a gate, and this brief does not. Part 1 produces a
decomposition and a mechanism; part 2 registers the run that can refute it, with its gates and my
prediction written before any cell exists.

## 1. Why this question, and why it is answerable today

E46 measured that per-token cost is **linear in context position** and that the 10B target is
**context-limited at `C50 = 575` tokens**. E47 measured that the marginal cost is **MIXED** — both
query heads and KV heads move the slope. Neither says *which part of the engine* the slope is in,
and without that there is no lever.

Two things make it answerable without new data:

1. **A structural fact, read out of the source** (`donor_engine.c`). Every `pos`-dependent loop in
   the entire forward pass is inside the attention block, lines 1099–1176: the Q·K loops
   (1099–1135), the softmax passes (1148–1154), and the A·V loops (1164–1176). Everything else
   loops over *dimensions* — `T`, `H`, `nr`, `M`, `SL` — never over position. The KV cache write is
   a `memcpy` of `KVO*4` bytes (1071–1072), constant in `pos`. **So 100% of the context slope is
   the attention organ, by construction and not by measurement.** This is E46 §5's owed reading,
   done.

2. **E5 run 6 already measured the attention organ at two context lengths.** `--bench 300` (mean
   position 150.5) and `--bench 800` (mean position 400.5), five runs each, on `T10`
   (`L=48, NH=32, NKV=8, HD=128`, ~10.6B). It decomposed the organ into `X` (Q·K), `S` (softmax),
   `Y` (A·V) and `P` (residual), all bit-identical (G2 PASS, `166667.1361128952`) and all
   validated by a 3× out-of-sample test that E5 could have failed nine times and did not (worst
   error −1.15% against a 3% tolerance fixed before run 1).

E5 read those two cells as *levels*. **E46's linearity result licenses reading them as a slope** —
and that composition is the whole of part 1. Each term is also linear in `pos` by construction
(`X`, `S` and `Y` are `O(pos)` loops; `P`'s thread-imbalance component is a multiple of them).

## 2. The decomposition (post-hoc, no gate)

Source: `benchmarks/donor_adaptation/results/e5/analysis6.txt`, `T10` @300 and @800, medians over
the runs that passed G0. `P′ = P − X`, because E5's `P` contains `X` by construction (`R = S+Y+P`
with `X` reported separately).

| term | @300 (ms) | @800 (ms) | slope (ms/pos) | **share of the organ's slope** |
|---|---|---|---|---|
| `X` — the Q·K loop | 0.713 | 2.253 | 6.1600e-03 | **16.9%** |
| `S` — the softmax pass | 1.048 | 2.953 | 7.6200e-03 | **20.9%** |
| `Y` — the A·V loop | 0.781 | 2.406 | 6.5000e-03 | **17.9%** |
| `P′` — the non-loop residual | 1.530 | 4.064 | 1.0136e-02 | **27.9%** |
| the organ outside `R` | 0.728 | 2.290 | 6.2480e-03 | **17.2%** |
| **organ (`none`)** | 4.834 | 13.931 | **3.6388e-02** | 100% |
| sum of the five parts | | | 3.6664e-02 | 100.8% |

**The parts close on the whole to 0.8%**, which is inside E5's own run-to-run band on every term.

### 2.1 There is no single fix

**No part of the engine owns more than 28% of the context slope.** Deleting the softmax entirely
buys 20.9% of it; deleting both data loops buys 34.8%. Anyone promising to fix context cost with
one kernel is promising at most a third.

### 2.2 Three consistency checks that could have failed and did not

None of these was arranged; all three are checks on the decomposition, not inputs to it.

* **`Y/X = 1.055`.** The Q·K loop and the A·V loop read the same number of bytes (`HD` floats per
  head per position) and do the same number of FMAs. They must cost the same. They do, to 5.5%.
* **The `expf` count.** `L·NH = 1,536` per position → 615,168 per token at mean position 400.5.
  E5 §14.8 reports **615,168**, arrived at independently from the disassembly.
* **Charged versus moved bytes, which is the law that killed the "39.7% anomaly".** `X` reads
  `L·NH·HD·4 = 786,432` **charged** bytes per position, which at 6.16e-03 ms is **127.7 GB/s** —
  **3.52× above `BW-CEIL = 36.30 GB/s` (E30)**. That is impossible from DRAM, so the loop must be
  getting ≥3.52× reuse. **GQA supplies exactly 4.00×** (`NH/NKV = 32/8`): each KV head's slice is
  re-read once per query head that shares it. Crediting it, `X` **moves** 196,608 B/pos =
  **31.9 GB/s = 88% of `BW-CEIL`**.

### 2.3 What that buys: which levers are bandwidth and which are not

**`X` and `Y` are bandwidth-saturated and `S` and `P′` are not.** At 88% of the measured ceiling,
the two data loops cannot be made faster by better code — only by **moving fewer bytes**: fewer KV
heads, or a narrower KV dtype (the cache is **fp32**, `xmalloc(L*maxseq*KVO*4)`, line 965–966 — no
quantization anywhere in it). Together they are **34.8%** of the slope.

The other **65.2%** — softmax, the non-loop residual, and the organ outside `R` — is not bandwidth
and will not move for any KV compression. This is the answer to E46 §5, and it is the reason E47
came back MIXED rather than KV-BOUND.

### 2.4 The two named, sized, attackable levers

* **`S`, 20.9% of the slope, and it is calling the wrong function.** E5 §14.8: `expf` compiles on
  this toolchain to `vcvtss2sd → callq exp → vcvtsd2ss` — **double-precision, scalar, with a
  `vzeroupper` in front of each call**, 615,168 of them per token at @800. A vectorised
  single-precision exponential is standard work with a known shape. E5 declined to run it because,
  unlike every arm in E5, **it is not value-preserving** and therefore needs its own parity gate
  rather than bit-identity. That is a cost, not an objection.
* **`P′`, 27.9% of the slope — the largest single piece — and nobody has named its parts.** E5
  §14.6 priced the one named candidate (the OpenMP fork/join: **1.4% of `P`**) and then wrote:
  *"E5 establishes its size and refuses to name its parts. Decomposing `P` is a new experiment with
  its own arms, and it is owed, not answered here."* It is still owed. From the source, the
  candidates are the 32-heads-over-6-threads imbalance, the per-head address arithmetic, the
  `out[]` initialisation and the `mx` reduction (lines 1086–1180).

### 2.5 A candidate for E47's superlinearity — offered as a candidate, not a finding

E47 left `q = 2.8–4.0` unexplained against the 2.00 that doubling query FLOPs predicts. This
decomposition narrows where it can live but does **not** close it:

* `S` scales with `NH` and not with `NKV` (one `expf` per head per position) → exactly ×2. Not it.
* `X` and `Y` charge `NH·HD` but move `NKV·HD`. Doubling `NH` doubles *charged* and leaves *moved*
  unchanged, so they should rise by **less** than 2×, not more. Not it, and it pushes the wrong way.
* `P′` is the only term whose per-head overheads could rise faster than `NH`, and it is also the
  term nobody has decomposed.

**So the superlinearity, if it is real, is most likely inside the one term that has never been
opened.** That is a reason to open it, and it is not evidence about what is in it.

### 2.6 Four limits on part 1, stated here and not in a footnote

1. **It is post-hoc.** The cells were run for another question. No gate in this brief is applied to
   part 1, and part 1 may not be cited as a gated result.
2. **Two points per term is a slope with zero degrees of freedom.** Linearity is licensed by E46
   and by the loop structure, not by these data.
3. **The shape is `T10`, not the target.** `T10` is `L=48, NH=32, NKV=8, HD=128` (GQA 4); the A10B
   R128 headline arm is `L=16, NH=32, NKV=2, HD=128` (GQA 16). The shares are `T10`'s. The GQA
   factor in §2.2 is 4.00× at `T10` and would be 16× at R128 — the *argument* travels, the
   *number* does not.
4. **Absolute ms/pos carry E5's own warning.** E5 §14.7 refused to publish absolute tok/s because
   run 6 drifted **+15.5%** on the organ against E4. Slopes here are taken **within run 6**, where
   that drift is common to both cells and largely divides out of a difference — but the ±5% law
   still applies to every absolute in the table. **The shares are the result; the milliseconds are
   the arithmetic.**

---

## 3. PART 2 — the confirming run, registered before it exists

Part 1 is a composition of two experiments neither of which was designed for it. Part 2 measures
the same decomposition **directly, at more than two context points, on the target shape**, with the
harness E46 already validated.

### 3.1 Method

`e48_slope_decomposition.py`. For each arm in `{none, qk1, qk2, sm1, sm2, av1, av2}`, sweep the
context windows `{40, 160, 640}` with `1280` held out, `REPS = 5`, `--threads 6`, arms interleaved
at **rep level** (E45 run 4's schedule) so that any machine drift hits every arm equally.

The sub-organ cost is the difference between the 2× and the 1× wrapper, never against `none`:
`X = dt(qk2) − dt(qk1)`, `S = dt(sm2) − dt(sm1)`, `Y = dt(av2) − dt(av1)`. E5 §13.4 establishes
these wrappers are bit-identical to `none` and that the 1× form exists precisely so the wrapper's
own code shape is not charged to the loop.

Two shapes: **`E47-BASE`** (cheap, already built, GQA 8) and the **A10B R128 `--carve-k 3`**
headline arm (GQA 16, the shape `C50 = 575` was measured on).

### 3.2 Gates

* **`G-E48a` — the planted control. The instrument must fire before its nulls count.** Each of
  `X`, `S`, `Y` must be **positive at every window** and must **rise with context** by more than
  1.05× from the 40-window to the 640-window. A sub-organ probe that cannot see context in a loop
  that is `O(pos)` is not measuring the loop. If `G-E48a` does not fire, E48 part 2 reports no
  decomposition.
* **`G-E48b` — CLOSURE, and this is the gate that matters.** Fit each of `X`, `S`, `Y` and the
  residual against mean position. Their slopes must sum to the **independently measured** total
  slope `b` for that shape — for A10B R128 k3, E46's `b = 2.0307e-05 s/tok/pos`, obtained by a
  completely different route (a whole-token fit, no wrappers, on another day). **Tolerance ±15%.**
  The parts must add up to a whole that was measured without them. If they do not, the
  decomposition is wrong and part 1 goes with it.
* **`G-E48c` — the shares, which is the deliverable.** Report each term's share of the total slope
  with its dispersion across reps. **No threshold and no verdict**: this is a SCORE, and per E14 §3
  its RANK partner is `G-E48d`.
* **`G-E48d` — the RANK partner.** The *ordering* of the four terms by share must be identical
  between the two shapes, or the ordering is a property of the shape and not of the engine, and no
  lever may be prioritised from it.

The ±15% on `G-E48b` is set against **measured** dispersion, per the law this programme broke twice
in one day: E43 measured 9–22% intrinsic per-cell oscillation on this engine, and `G-E48b` is a sum
of four such quantities compared against a fifth. A tolerance tighter than that could never fire;
one looser than ~25% would be passed by noise. It is **not** set to my expected answer.

### 3.3 My prediction, registered

Written before any cell of part 2 exists, so it can be scored:

| quantity | prediction |
|---|---|
| `G-E48a` | **fires** on all three terms, both shapes |
| `G-E48b` closure at A10B R128 k3 | **passes**, within 10% |
| largest share at R128 | **`P′`**, 25–35% |
| `S` share at R128 | 15–25% |
| `X + Y` share at R128 | **lower than `T10`'s 34.8%** — GQA 16 instead of 4 means four times the reuse on the same moved bytes |
| `G-E48d` ordering | **holds** |

My last three registered predictions scored half-right, one-of-three and zero-of-three. This one is
recorded on the same terms.

### 3.4 What part 2 cannot do

It cannot open `P′`. It measures `P′` as a residual, exactly as E5 did, and inherits E5's refusal to
name its parts. **Decomposing `P′` is E49 and it is owed.** Part 2's only claim on `P′` is a number
and a rank.

---

# ADDENDUM A — THE INSTRUMENT IS BETTER THAN THE ONE I REGISTERED, AND MY CLOSURE GATE WAS VACUOUS

**Written after smoking the arms and before any E48 cell exists.** Part 2 §3.1 and §3.2 are
replaced by this section. §3.3's prediction is **not** revised — it is scored as written.

## A.1 The method changes: `--sweep6`, one process, ten arms

§3.1 registered seven separate processes per window. The engine already has a better instrument
and E5 validated it: **`--sweep6 --profile`** runs ten arms — `none, sm1/2/3, av1/2/3, qk1/2/3` —
**inside one process**, rotating by position on a palindrome of period 20, and prints per-arm
per-organ milliseconds per token. Three things follow:

* **Every arm gets the same mean context position** (`a` and `19−a` in each period), so no arm is
  handed a longer or shorter context than another. E5 §14.2 measured the residual at **0.1% of the
  organ** with `--sweepd`.
* **Process-to-process variance is gone**, which was the whole difficulty: the sub-terms are
  differences of two totals, and at window 40 `X` is ~3% of a total that E43 showed oscillates
  9–22%.
* It is roughly **seven times cheaper**, and it prices the **attention organ** directly (`T_ATTN`)
  rather than the whole token, so the term I want is not buried under the FFN.

The engine refuses a `--bench` not divisible by the schedule period, so every window is a multiple
of 20.

## A.2 The shapes change: `E47-BASE` is the badly-conditioned one, and E47 said so

§3.1 named `E47-BASE` as the cheap shape. Smoking it at window 160 shows why that was wrong:
attention is **1.5%** of its token there, and the measured increments come back **non-monotone** —
`av3 (0.262) < av2 (0.388)`, which is impossible for a loop run three times against twice. That is
exactly the conditioning failure **E47 addendum C.3 already recorded**, and I walked into it again
one experiment later.

The two shapes are therefore the two whose total slope E46 measured independently:

| shape | `a` (s/tok) | `b` (s/tok/pos) | attention share at pos 640 |
|---|---|---|---|
| **A10B R128 `--carve-k 3`** | 0.008327 | 2.0307e-05 | **61%** |
| **S15** (`e37_carved_nf.bin`, the real 1.5B) | 0.037310 | 1.6276e-05 | 22% |

Windows: **{160, 640, 1280, 2560}**, all divisible by 20, weighted to where attention is a large
share of the token. `REPS = 5`, windows round-robin with reps outermost.

## A.3 `G-E48b` as registered is VACUOUS, and it is replaced

§3.2 asked that "`X`, `S`, `Y` and the residual" sum to the measured total. Under this instrument
the residual is **defined** as `Rem = organ(none) − X − S − Y`, so the sum is an identity and the
gate could not fail. **A gate that cannot fail is not a gate** — this programme has now written one
too tight to ever fire (`G-E43A`), one passed by being noisy (`G-E45c`, `G-E45j`), and here one
that is true by construction. Same law, third face:
`feedback_gate_vs_measured_dispersion`.

`G-E48b` is replaced by two things that genuinely can fail:

* **`G-E48b1` — the 3× test.** Each wrapper family runs its loop once, twice and three times. The
  second increment must equal the first: `qk3−qk2 = qk2−qk1`, and likewise for `sm` and `av`.
  **Three predictions per window per shape, twenty-four in all, each able to miss.** Tolerance
  **±15%**, at the two largest windows only (at window 160 the term is too small to resolve and
  the test is reported, not gated). This is E5's own G1, which passed nine times out of nine at a
  3% tolerance — so it is a **known-positive**, and the planted-control law is satisfied: the
  instrument has fired before.
* **`G-E48b2` — nothing else may have a context slope.** §1 claims from the source that the
  attention block holds every `pos`-dependent loop in the engine. Fit the `none` arm's
  **whole-token** cost and its **attention-organ** cost separately against mean position; the two
  slopes must agree within **±15%**. If `qkv_proj`, `ffn`, `head` or `o_proj` carries a slope, the
  structural claim in §1 is false and part 1 goes with it.

## A.4 E46's `b` becomes a drift line, not a gate

The smoke run reads **89.13 tok/s** at window 640 where E46's fit for the same arm predicts
**67.5** at the same mean position — **32% apart**, and that is with the sweep's doubled arms
making it *slower*, not faster. E43's 9–22% intrinsic oscillation does not cover 32%; E5 §14.7 met
the same thing (**+15.5%** organ drift between sweeps) and responded by publishing **ratios only**.

So: **every slope E48 reports is measured inside one sweep**, and the comparison to E46's
`b = 2.0307e-05` is recorded as a **drift line under `G-E48b2`, never as a gate and never as a
correction to E46.** A share travels between sweeps; a millisecond does not.

This also means E48 **cannot** revise `C50 = 575`, and does not try to.

## A.5 What is unchanged

`G-E48a` (planted control), `G-E48c` (shares as a SCORE), `G-E48d` (the RANK partner across two
shapes) and **§3.3's registered prediction** all stand exactly as pushed in `275538f`.

---

# ADDENDUM B — WHAT E48 MEASURED, WHAT IS VOID, AND A 2.5× DISAGREEMENT WITH E46 ON C50

**Written against `results/e48_slope_decomposition.json` (run `c89928f`), before the adjudicating
run in B.5 exists.**

## B.1 R128, the target shape: every gate fires

| term | slope (ms/pos) | intercept (ms) | **share of the organ's slope** |
|---|---|---|---|
| `X` — Q·K loop | 1.7295e-03 | −0.068 | **22.5%** |
| `S` — softmax | 2.5478e-03 | −0.018 | **33.1%** |
| `Y` — A·V loop | 1.7179e-03 | −0.031 | **22.3%** |
| `Rem` — non-loop residual | 1.7459e-03 | +0.205 | **22.7%** |
| attention organ | 7.6876e-03 | +0.080 | 100% |
| whole token | 8.1821e-03 | +8.107 | |

* **`G-E48a` FIRES** — all three priced terms positive at every window, rising 16.7–22.5× across
  the range against a bar of 1.05.
* **`G-E48b1` PASSES** — the 3× test, 6/6 at the two gated windows, worst 9%. At the ungated
  window 160 `X` misses at 1.37×, exactly as addendum A.3 anticipated when it carved that window
  out **in advance**.
* **`G-E48b2` FIRES** — whole-token slope 8.1821e-03 against organ slope 7.6876e-03, **6.4%
  apart**. Nothing but attention carries a context slope, which is what §1 read out of the source.
* Dispersion at the largest window across five reps: `X` 0.07, `S` 0.06, `Y` 0.06, organ 0.032.

**At the target shape the softmax is the single largest piece of the context slope, at 33.1%** —
and E5 §14.8 already established that this engine's `expf` compiles to a scalar **double-precision**
`exp` call with a `vzeroupper` in front of each one. That is the most concrete lever this
programme has found on `C50`.

## B.2 S15 is VOID by its own instrument gates

`G-E48b1` **FAILS** (5 misses of 12, including 0.60× and 0.66× on `X` at gated windows) and
`G-E48b2` **FAILS** (whole token 24.8% steeper than the organ). The cells are rejected. Causes,
both visible in the log and both predicted in advance by addendum A.2:

* **Conditioning.** Attention is 10.08 ms of a 49–58 ms token — the FFN alone is 33.34 ms — so the
  priced terms are ~1% differences of a large number. Rep 3 at n=1280 reads `X = 1.958` where rep
  1 reads `0.488`, a 4× spread.
* **Load.** S15 ran at 62–75% occupancy against R128's 34–63%, and the same window drifts 44%
  between reps (n=160 reads 40.6 ms at rep 1 and 58.4 ms at rep 4).

**No S15 share is reported.** This is the instrument refusing data, not a result.

## B.3 `G-E48d` is VOID, not FAILED — and that is a defect in my runner

The runner printed `G-E48d ... FAILS` and it had no business doing so: **one of its two inputs had
already been rejected by that shape's own instrument gates.** A rank comparison against a void arm
is not a failed rank, it is no measurement. `G-E48d` should have been made conditional on
`G-E48b1` and `G-E48b2` firing on *both* shapes, and it was not.

**Recorded as: `G-E48d` VOID.** This is the fourth time the law in
`feedback_runner_verdict_outlives_spec` has earned its keep — the script prints numbers, the brief
decides — and the first time it has caught my own gate wiring rather than a stale rule.

Consequence: **`G-E48c`'s shares have no RANK partner**, so per E14 §3 the R128 shares in B.1 are a
SCORE with no rank, and **no lever may be prioritised from them yet**. The softmax's 33.1% is a
measured number at one shape; it is not yet "the biggest lever".

## B.4 Part 1 and part 2 disagree about the ordering, which is a result in itself

Put on the same footing (part 1's `P′` and "organ outside `R`" both belong in `Rem`):

| | `X` | `S` | `Y` | `Rem` | ordering |
|---|---|---|---|---|---|
| `T10`, part 1 desk, GQA 4 | 16.9% | 20.9% | 17.9% | 45.1% | `Rem > S > Y > X` |
| **R128, part 2 measured, GQA 16** | **22.5%** | **33.1%** | **22.3%** | **22.7%** | **`S > Rem > X > Y`** |

The largest term is not the same one. Part 1 was explicit that it is post-hoc, at a different
shape, and carries no gate — §2.6 item 3 said "the *argument* travels, the *number* does not", and
that has now been demonstrated rather than asserted. **Part 1's 27.9% for `P′` does not survive to
the target shape.** What does survive is the structural claim (§1), which `G-E48b2` confirmed
independently at 6.4%.

## B.5 THE 2.5× DISAGREEMENT WITH E46, AND THE RUN THAT ADJUDICATES IT

This is the most important thing in E48 and it is not about the decomposition.

| | `a` (ms) | `b` (ms/pos) | `C50` |
|---|---|---|---|
| E46, phase B | 8.327 | **0.02031** | **575** |
| E48, same arm | 8.107 | **0.00818** | **1454** |

Same engine, same weights file, same `--carve-k 3`, same machine. **The intercepts agree to 2.6%
and the slopes differ by 2.5×.** So this is not overall machine speed and not a drift in level: it
is specifically and only the **context-dependent term** — the one thing `C50` is made of.

Addendum A.4 already forbade E48 from revising `C50`, and that clause was written before these
data. It holds. But E48 may not leave `C50` standing unqualified either: **`C50` is currently a
number with two incompatible measurements, 575 and 1454.**

Candidate causes, none of them yet evidence:

1. **The fit range.** E46 fitted `{40, 160, 640}`; E48 fitted `{160, 640, 1280, 2560}`. If
   per-token cost is not one straight line over the whole range, both fits can be locally right
   and `C50` sits inside the bend. E46's out-of-sample check reached only 1280.
2. **The instrument.** Under `--sweep6` the `none` arm's tokens are surrounded by arms that walk
   the same K and V, which could leave the cache warmer than a pure `none` run ever finds it. E5
   measured neighbour effects at **0.1% of the organ** with `--sweepd` — but at mean position 4.5,
   not 1280.
3. **An L3 crossing.** For R128 the KV cache is `2·L·pos·NKV·HD·4 = 32,768·pos` bytes, so it
   crosses the measured 32 MB L3 at **position ≈ 1024** — inside E48's range and outside E46's.
   Note this predicts E48's slope should be *steeper*, not shallower, so it argues against itself.

**Registered now, before the run: `e48_none_control.py`.** Pure `none` — no `--sweep6`, no
`--profile`, nothing but `--carve-k 3 --bench N` — at windows **{40, 160, 640, 1280, 2560}**, which
contains E46's three fit windows and E48's four, **5 reps**, round-robin. One dataset, then two
fits of it.

**`G-E48e`, the adjudication, written as a four-way decision function so that the runner cannot
choose:**

Let `b_short` = the fit on `{40, 160, 640}` (E46's windows) and `b_long` = the fit on
`{160, 640, 1280, 2560}` (E48's windows), both from the same pure-`none` cells. Tolerance **±25%**,
set against E43's measured 9–22% intrinsic oscillation on this engine compounded across a fit, and
fixed before the numbers exist.

| condition | verdict |
|---|---|
| `b_short` ≈ E46 **and** `b_long` ≈ E48 | **THE LAW IS NOT LINEAR** — both prior fits are locally right, `C50` sits in the bend, and E46's linearity claim is scoped to short context |
| `b_short` ≈ `b_long` ≈ E46's `b` | **THE SWEEP INSTRUMENT IS BIASED** — E48's shares are measured in a regime the engine never runs in, and B.1 is withdrawn |
| `b_short` ≈ `b_long` ≈ E48's `b` | **E46's PHASE B IS THE OUTLIER** — `C50 = 575` is not reproduced and must be re-measured |
| anything else | **INCONCLUSIVE** — no verdict, and `C50` carries both numbers |

**My prediction, registered:** the first row — not linear, `b_short` high and `b_long` low. I say
so because E46's out-of-sample check at 1280 passed at 2.48% on a quantity `b` dominates, which is
hard to reconcile with E46 simply being wrong, and because `a` agrees to 2.6% between the two runs,
which is hard to reconcile with a machine-state explanation. I have been wrong on my last three
registered predictions and this one is recorded on the same terms.

**Whatever it returns, E48 does not rewrite `C50` on its own.** If the law is not linear, `C50`
needs a fit that is not a straight line, and that is E49.

## B.6 My registered prediction (§3.3), scored

| quantity | predicted | measured | verdict |
|---|---|---|---|
| `G-E48a` fires on all three, both shapes | fires | fires at R128; S15 void | **partly** |
| `G-E48b` closure within 10% | passes | `G-E48b2` 6.4% at R128 | **right** |
| largest share at R128 | **`P′`**, 25–35% | **`S`** at 33.1%; `Rem` 22.7% | **WRONG** |
| `S` share at R128 | 15–25% | **33.1%** | **WRONG** |
| `X + Y` share at R128 vs `T10`'s 34.8% | **lower** | **44.8%**, higher | **WRONG** |
| `G-E48d` ordering holds | holds | **VOID** | not scorable |

**Two right, three wrong, one void.** The reasoning behind the `X + Y` prediction was explicitly
that GQA 16 gives four times the reuse of GQA 4 and should make the data loops cheaper in share.
It went the other way. That reasoning is the same one §2.3 used to call `X` and `Y`
bandwidth-saturated, so **§2.3 is now under suspicion at the target shape** and B.5's run is the
first thing that can speak to it.
