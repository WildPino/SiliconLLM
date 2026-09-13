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
