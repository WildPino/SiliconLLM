# E46 — what does a token cost at a REAL context? Every rate in this programme was measured at context 20

**PRE-REGISTERED. Pushed before any measurement.**

## 0. The one sentence

**Every tok/s this programme has ever published was measured over 40 decoded tokens, at a mean
context position of 20**, and E45 run 3 found the per-token cost rises **linearly with context**
at a rate that would put the 10B headline below its own target at any realistic context length.
E46 measures that, at the target shape.

## 1. Where this comes from

`--bench N` decodes at positions 1…N, so the timing window and the context length are the same
knob. `NTOK = 40` in `e40_levers_exhausted.py` — the file the 10B headline comes from — means
the headline is a **mean-position-20** number. So are E36's, E37's, E39's, E43's and E45's.

E45 run 3 (`0897e70`) measured four windows on real 1.5B weights and, as a **post-hoc fit**
(E14 §6, not a gate), found per-token cost linear in mean position:

| arm | fit | at position 0 | residuals |
|---|---|---|---|
| K256 | `s/tok = 0.039967 + 1.454e-05 · pos` | 25.02 tok/s | −3.9%, +2.3%, +1.9%, −0.4% |
| K16 | `s/tok = 0.011840 + 1.427e-05 · pos` | 84.46 tok/s | 2 points |

The two slopes agree to **1.9%** across arms that differ **3.3× in FFN work** — which is what
must happen if the marginal cost is **attention**, since the carve does not touch attention.
That agreement is the only control the fit has, and it is why the fit is worth testing rather
than discarding.

## 2. What that arithmetic implies, written down so it can be wrong

The S15 donor has `L=28, NKV=2, HD=128`. A10B has `L=16` with the same `NKV` and `HD`, so if the
marginal cost is KV traffic it should scale by `16/28 = 0.571` to **≈8.3e-06 s per token per
position**. With the R128 headline rate as the position-0 cost (`1/113 = 0.00885 s`):

| context | attention term | total s/tok | implied tok/s |
|---|---|---|---|
| 40 (what is published) | 0.00017 | 0.00902 | **111** |
| 512 | 0.00212 | 0.01097 | **91** |
| 2048 | 0.00850 | 0.01735 | **58** |
| 4096 | 0.01700 | 0.02585 | **39** |

**This is desk arithmetic from a two-arm fit at a different scale, and it is exactly the kind of
number this programme refuses to quote.** It is written here to be *measured*, and it is the
registered prediction of §7.

## 3. The design

Three phases, cheapest first, each able to stop the next.

**Phase A — reproduce the linear law on the 1.5B artifact, out of sample.** `e37_carved_nf.bin`
at `K256`, windows `{40, 160, 640}`, ≥5 reps, round-robin. Fit on those three, then **predict**
`dt` at `NTOK = 1280` and measure it.

**Phase B — measure the slope at the TARGET shape.** `D:/_ktmp/e40/e40_r128.bin` (9,999,220,736
parameters, synthetic weights — this is a **speed** measurement and E45's bridge question is
separate and open), FFN fully on, windows `{40, 160, 640, 1280}`, ≥5 reps, round-robin.

**Phase C — the table.** Sustained tok/s at contexts 512, 2048 and 4096, from Phase B's fit,
each with the interval its own residuals allow.

## 4. The gates, registered now

**`G-E46a` — PLANTED CONTROL, and nothing counts until it fires.** The instrument must resolve a
context effect that is really there. Known positive: run 3's own K256 cells, where `dt/n` rose
from 0.0387 at `n=40` to 0.0583 at `n=2560`, a 51% rise. Phase A must see the same law on
`{40, 160, 640}` — specifically `dt(640)/640 > dt(40)/40 · 1.05`. If the rise is not there, run
3 was contention and E46 stops.

**`G-E46b` — THE OUT-OF-SAMPLE TEST, and this is what makes the fit a claim rather than a
curve.** Fit `a, b` on `{40, 160, 640}`, predict `dt(1280)`, measure it. The linear law
**survives** iff the measured median lands within **10%** of the prediction. If it does not,
the law is wrong, §2's table is void, and E46 reports that instead.

**`G-E46c` — THE SHAPE TEST.** The slope `b` at the A10B shape must land within **±30%** of
`b(S15) · 16/28`. Fires: the marginal cost is KV traffic and scales with layer count, so the
law transports. Fails: the marginal cost is something else — still measured, but §2's
extrapolation to other shapes is withdrawn. **Both outcomes are results.** The band is wide on
purpose: it is a mechanism test, not a precision test, and a tighter band would be narrower than
the dispersion of the axis it watches (E43's lesson).

**`G-E46d` — THE DECISION.** From Phase B's fit, let `C50` be the largest context at which the
median sustained rate is ≥ 50 tok/s.
* **TARGET HOLDS AT REALISTIC CONTEXT** iff `C50 ≥ 2048`.
* **TARGET IS CONTEXT-LIMITED** iff `C50 < 2048`. Then the goal's "50 tok/s" is met only up to a
  measured context, that context is E46's product, and it must be quoted beside every future
  rate.
* Every rate in the table carries dispersion or is not reported (`G-E44b`'s rule, which E45's
  runs did not repeal).

## 5. What E46 may and may not conclude

**MAY:** state the cost of a token as a function of context, at both shapes, and say at which
context the 50 tok/s target stops being met.

**MAY NOT:** revise any published rate retroactively — those were measured, they are simply
measured **at context 20**, and the correct repair is to quote the context beside them, not to
delete them. **MAY NOT** conclude anything about quality; Phase B is on synthetic weights and
E45's bridge is open. **MAY NOT** claim attention is or is not bandwidth-bound: the 9× figure in
E45 addendum E is arithmetic against E30's ceiling, and turning it into a claim needs the KV
dtype and layout read out of `donor_engine.c`, which is a separate and free job.

## 6. Cost

CPU only, no GPU, no training, no user action. Phase A ~6 min, Phase B ~8 min plus a cold 5.49 GB
read from the USB disk. Both artifacts already exist.

## 7. The predictions, written before the run

1. **`G-E46b` survives** — the linear law holds out of sample within 10%.
2. **`G-E46c` fires** — the slope scales with layer count, landing near 8.3e-06 s/token/position
   at A10B.
3. **`G-E46d` returns TARGET IS CONTEXT-LIMITED**, with `C50` somewhere between **1500 and
   3000**. I am recording that because it is the outcome that costs this programme its headline,
   and because if the measurement comes back saying 50 tok/s holds at 4096 I want it on the
   record that I did not expect it.

---

# ADDENDUM A — THE HEADLINE ARM IS `k = 3`, NOT "THE FFN FULLY ON"

**Written and pushed before any E46 measurement.** §2 and §3 described phase B as the R128
artifact "FFN fully on". That is wrong, and the error is carried in the project's own index
line, so it is worth fixing where it can be seen.

Read out of `results/e40_levers_exhausted.json`:

| arm | mean tok/s | median tok/s |
|---|---|---|
| `R128_k1` | — | 139.64 |
| `R128_k2` | 128.074 | 130.78 |
| **`R128_k3`** | **112.732** | **116.59** |
| `R128_k4` | — | 105.72 |
| `NKV2_k1` | 112.002 | 112.73 |

**The 112.7 that this programme quotes is `R128` at `--carve-k 3`** — 3 groups of `E = 256`,
**1.17% of the FFN**, which is exactly the activation budget E36 required and E37 priced. It is
not the FFN fully on. (`112.73` also appears as `NKV2_k1`'s median, which is a different arm
with the same leading digits; the R128 figure is the mean `112.732`.)

**Consequence for phase B:** it must run **the headline's own flags**, `--carve-k 3` on
`donor_engine_e26.exe`, or it prices a different arm and `G-E46d` decides about a rate nobody
published. The runner is written that way and records the flags in its output.

Nothing else changes: §2's desk table used `1/113` as the position-0 cost, and `1/112.7` is
that same arm, so the registered prediction of §7 stands as written.

---

# ADDENDUM B — WHAT `G-E46c`'s FAILURE POINTS AT, AND WHY IT CANNOT BE CONCLUDED FROM TWO SHAPES

**Written after the measurement (`868f379`).** `G-E46c` failed, which the brief said would be a
result. This is what it is a result *about*, kept strictly separate from what it licenses.

## B.1 The two candidate mechanisms, and what each predicts

Per position of context, attention does two different things:

* **reads KV** — proportional to `NKV · HD · L`;
* **arithmetic over the query heads** — `QK` and `AV` are each `NH · HD` MACs per position per
  layer, so proportional to `NH · HD · L`.

| | S15 | A10B | ratio A10B/S15 |
|---|---|---|---|
| `NKV · HD · L` (KV bytes) | 7,168 | 4,096 | **0.571** |
| `NH · HD · L` (query FLOPs) | 43,008 | 65,536 | **1.524** |
| **measured `b`** | 1.6276e-05 | 2.0307e-05 | **1.248** |

The KV-bytes prediction is **off by a factor of 2.2 and in the wrong direction**. The
query-FLOPs prediction is **22% high and in the right direction**. That is why `G-E46c` failed:
A10B has **fewer** KV bytes per position than S15 and a **steeper** slope.

**This matters beyond bookkeeping.** GQA — `NKV = 2` against `NH = 32` — is a *memory*
optimisation, and if the marginal cost is query-head arithmetic then GQA buys nothing on this
axis, and neither would KV quantisation or any other KV-compression idea. The lever would be
`NH · HD · L`, or the kernel itself.

## B.2 The two-term fit, and why it is NOT evidence

Fitting `b = α · (NH·HD·L) + β · (NKV·HD·L)` to the two measured slopes gives
`α = 2.687e-10`, `β = 6.585e-10`, both positive. **Two equations, two unknowns: the fit has
zero degrees of freedom.** It cannot fail, so it is not a test, and per E14 §6 it does not
become one by being plausible. It is written here only so the numbers are on the record.

## B.3 E47, registered now

**Question:** does the marginal cost per unit of context scale with `NH · HD · L`, with
`NKV · HD · L`, or with a mixture?

**Design.** Build **three small synthetic artifacts** (a few hundred MB each, `synth_export.py`,
minutes not hours) that hold `HD`, `L`, `D` and the FFN fixed and move only the head counts:

| arm | `NH` | `NKV` | what it separates |
|---|---|---|---|
| `H-BASE` | 16 | 2 | the reference |
| `H-QUERY` | **32** | 2 | doubles query FLOPs, KV bytes unchanged |
| `H-KV` | 16 | **8** | quadruples KV bytes, query FLOPs unchanged |

Then measure `b` on each with E46's own method (fit on `{40, 160, 640}`, out-of-sample check at
1280) and read the two slope ratios.

**`G-E47a` — the control that must fire first:** `G-E46b` must survive on **every** arm. If the
linear law does not hold at these shapes, no slope from them means anything.

**`G-E47b` — the decision, registered so every outcome is a result.** Let
`q = b(H-QUERY)/b(H-BASE)` and `v = b(H-KV)/b(H-BASE)`.
* **QUERY-BOUND** iff `q ≥ 1.5` and `v ≤ 1.5`.
* **KV-BOUND** iff `v ≥ 2.5` and `q ≤ 1.3`.
* **MIXED** iff both rise materially (`q ≥ 1.3` and `v ≥ 1.5`) — then `α` and `β` are both
  real and B.2's two-term model is tested rather than fitted.
* **NEITHER** otherwise, which would mean the marginal cost is something neither term captures
  and B.1 is withdrawn.

**Why it is worth the hour.** If the answer is QUERY-BOUND, then every KV-side idea in
`SCALEUP_ARCHITECTURE` is aimed at the wrong term at long context, and the context ceiling
`C50 = 575` moves only by changing `NH · HD · L` or the attention kernel. If it is KV-BOUND,
then A10B's steeper slope has some other cause and E46's fit needs re-examining before anything
is designed on it.

**Cost:** CPU only, no user action, well under an hour including the builds.

**Registered prediction:** **QUERY-BOUND**, with `q` near 1.7 and `v` near 1.1. I am recording
it because B.1 already points there and I want the prediction dated before the artifacts exist.

## B.4 What this does NOT change

`C50 = 575` stands as measured — it does not depend on *why*. `G-E46d`'s verdict, TARGET IS
CONTEXT-LIMITED, stands. And §5's limits are unchanged: no published rate is revised, phase B
was synthetic, and nothing here speaks to quality.

---

# ADDENDUM C — E47 ANSWERED: **MIXED**, AND B.1's GQA SENTENCE IS WITHDRAWN

**Written after E47 (`dd5f1d0`).** `G-E47a` fired on all three arms; `G-E47b` returned **MIXED**.

| arm | `NH` | `NKV` | query units | KV units | fitted `b` | `G-E46b` out of sample |
|---|---|---|---|---|---|---|
| BASE | 16 | 2 | 12,288 | 1,536 | 2.5707e-06 | 2.94% |
| QUERY | 32 | 2 | 24,576 | 1,536 | 1.0188e-05 | 1.58% |
| KV | 16 | 8 | 12,288 | 6,144 | 7.2377e-06 | 1.12% |

`q = 3.963` (query ×2), `v = 2.815` (KV ×4).

## C.1 Both single-term models are refuted

* **Query-only** predicts `v = 1.00`. Measured **2.82**.
* **KV-only** predicts `q = 1.00`. Measured **3.96**.

Neither term alone survives a third shape. B.2's two-term model, which had **zero degrees of
freedom** when it was written, now has data it could have failed against and did not.

## C.2 **B.1's sentence about GQA is withdrawn**

B.1 said that if the marginal cost were query FLOPs then *"GQA and every KV-compression idea in
`SCALEUP_ARCHITECTURE` is aimed at the wrong term at long context"*. **That is wrong.**
Quadrupling KV heads raised the slope **2.2–2.8×**, so reducing them — which is what GQA does —
**buys context**. GQA is not the whole story and it is not nothing.

## C.3 The verdict is robust; the numbers are not

Refitting each arm on **four** windows instead of the registered three moves `b(BASE)` from
2.5707e-06 to **3.4640e-06 (+35%)**, and the ratios to `q = 2.769`, `v = 2.201`. **MIXED holds
under both fits** (`q ≥ 1.3` and `v ≥ 1.5` either way), but the magnitudes move by a third.

The cause is **conditioning**: the context term is only **6.0%** of BASE's per-token cost at
position 320 and **11.3%** at 640, so BASE has the worst-determined slope of the three — and it
is the **denominator of both ratios**. Any future arithmetic on `q` and `v` needs a BASE arm
where context is a larger share of the cost (fewer FFN weights per token, or longer windows).

## C.4 Named, not concluded

Even the conservative `q = 2.77` exceeds the **2.00** that doubling query FLOPs predicts.
Something scales **worse than linearly in `NH`**. That is a puzzle with a number on it.

## C.5 My prediction, and what this buys the goal

**The registered prediction (QUERY-BOUND, `q ≈ 1.7`, `v ≈ 1.1`) is wrong on all three counts.**

For the goal: **`C50 = 575` moves on both head axes**, and reducing `NH` moves it *more than
proportionally*. `C50` itself does not depend on any of this — it was measured, not derived.
