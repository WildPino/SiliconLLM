# E36 — ten billion parameters at fifty tokens a second

**Read this first, because it was registered before any number existed (brief prediction 5).
This probe measures SPEED ONLY.** The weights in the artifact are noise. Nothing here says a
model of this shape can be trained, can be adapted from a donor, or is any good. The goal's
sentence is *"far funzionare un LLM già addestrato"* and **E36 does not satisfy it** — it
settles the half of it that says *"un modello grande (es 10B) a 50 token/s"*.

---

**Verdict: `TEN-B-NEAR-FIFTY`.** The verdict cell — a **9,999,220,736-parameter** artifact,
on the engine, with **540 of its 46,080 FFN neurons per layer active per token (1.17%)** —
reads **49.96 tok/s** against a 50.0 bar. Four hundredths short, on the registered run.

**The number that survives both runs, and is worth more than the verdict**: the crossing.
Measured twice under **opposite arm orders**, 50 tok/s costs **k\* = 2.867** and **k\* = 2.861**
carve groups — **516 and 515 of 46,080 neurons, 1.12% activation, agreeing to one neuron in
forty-six thousand.**

**The file weighs 10.8× what a token costs.** That ratio is the whole of what E18–E35 were
building toward, and it is now measured rather than argued.

**Brief**: `briefs/BRIEF_E36_TEN_BILLION_AT_FIFTY.md`, pushed at `1738aa2` before the runner
existed. **Runner**: `engine/e36_ten_billion.py`. **Results**: `engine/results/e36_ten_billion.json`
(run 1, registered) and `engine/results/e36_ten_billion_order_reversed.json` (run 2, the order
control). 92 s and 91 s of timing, six arms, five interleaved reps each, reps outermost.

---

## 1. The artifact

`A10B`: `D = 4096, F = 46080, L = 16, NH = 32, NKV = 8, HD = 128, V = 32768`, untied head,
`--carve 256` so a group is **180 neurons**.

| | |
|---|---|
| parameters on disk | **9,999,220,736** |
| bytes on disk | 5,485,658,936 (5.11 GB) |
| charged per token at `k=3` | **928,251,904** (0.9283 G) |
| ratio | **10.77×** |

Every 10 B-shaped file this programme had benched before today (`T10`, 10.74 B) was
**dense-active**: 10.6 G charged a token, 11× outside the envelope. `A10B` is the first artifact
in the branch whose *size* and whose *cost* are different numbers.

## 2. The gates

| gate | what it demanded | reading | |
|---|---|---|---|
| **`G-E36A`** the planted control | `T10-L16` within ±10% of E35's `56.16`, same box, same artifact | run 1 **58.08 (+3.4%)**, run 2 **54.85 (−2.3%)** | **FIRES** |
| **`G-E36B`** it must actually BE ten billion | exporter count == count recomputed from the **file's own header** == ≥ 9.9 G, **and** bytes on disk == E1's independently written v4 layout | 9,999,220,736 == 9,999,220,736 ≥ 9.9 G; 5,485,658,936 == 5,485,658,936 | **FIRES** |
| **`G-E36C`** charged accounting | at every `k`, the exporter's `active_weights_per_token` == the brief's closed form `822,083,584 + 35,389,440·k`, **zero tolerance** | exact at `k ∈ {1,2,3,4,6}` and on the file's JSON | **FIRES** |

`G-E36B` is the gate that mattered: it is what stops this probe from quietly demonstrating
something smaller than the goal. It is checked three independent ways because "10 B" *is* the
claim — and the third way (byte count against `e1_bpb_through_engine.layout_bytes_v4`, written
long before this shape existed) is the one that proves the tensors are **present**, not merely
declared in a header.

## 3. The measurement

**Run 1 — registered, forward arm order.**

| arm | neurons active | charged | **measured tok/s** | model | dev | spread |
|---|---|---|---|---|---|---|
| `T10-L16` (control) | — | 0.8331 G | 58.08 | — | — | 17.9% |
| `A10B-K1` | 180 (0.39%) | 0.8575 G | **54.86** | 55.11 | −0.4% | 5.2% |
| `A10B-K2` | 360 (0.78%) | 0.8929 G | **51.50** | 52.92 | −2.7% | 9.3% |
| **`A10B-K3`** | **540 (1.17%)** | **0.9283 G** | **49.96** | 50.91 | −1.9% | 7.2% |
| `A10B-K4` | 720 (1.56%) | 0.9636 G | **47.74** | 49.04 | −2.6% | 10.1% |
| `A10B-K6` | 1080 (2.34%) | 1.0344 G | **43.78** | 45.68 | −4.2% | 12.4% |

**Run 2 — the order control, reversed arm order.**

| arm | **measured tok/s** | dev vs model |
|---|---|---|
| `T10-L16` (control) | 54.85 | — |
| `A10B-K1` | **56.67** | +2.8% |
| `A10B-K2` | **52.65** | −0.5% |
| **`A10B-K3`** | **51.50** | +1.2% |
| `A10B-K4` | **46.21** | −5.8% |
| `A10B-K6` | **44.93** | −1.6% |

**RANK partner (E14 §3, which forbids a SCORE without one):** both runs are **strictly monotone
in `k`**, 5 of 5, under opposite arm orders. The ordering is the robust part of this probe; the
absolute levels move by a few percent between runs, exactly as the programme's ±5% law on
absolute tok/s says they must.

## 4. Why there are two runs, and why run 2 did not change the verdict

Run 1's arms ran in a **fixed order** inside every rep, and the box drifted within reps: rep 1
opened with the fastest control reading of the entire run (62.33) and closed with the slowest
`k=6` (40.60). A fixed order under a within-rep drift charges that drift to the **high-`k` arms**
— which is precisely the slope the verdict stands on. That is a defect in my instrument, not in
the engine, and it needed a control.

**The rule was written into the runner and pushed at `9819226` before run 2 existed**, because a
second run taken after a verdict misses its bar by 0.04 is the textbook way to fish:

> Run 1 is the registered measurement and its verdict stands. Run 2's only question is whether
> the k-slope depends on arm order. **If the slopes agree inside the reps' own dispersion the
> verdict is UNCHANGED — run 2 may not promote 49.96 over the bar no matter what number it
> prints.** If they disagree, both runs are reported and the cell is declared unresolvable.

**Run 2 printed 51.50 on the verdict cell.** Applying the rule:

| | run 1 (forward) | run 2 (reversed) | |
|---|---|---|---|
| slope, ms per carve group | **0.8995** | **0.9553** | **+6.2%** |
| worst per-arm spread | 12.4% | 19.3% | |
| `k*` | **2.867** | **2.861** | −0.2% |

**+6.2% is far inside the reps' own dispersion, so the slopes agree and the verdict is
unchanged: `TEN-B-NEAR-FIFTY`, 49.96 tok/s.** 51.50 is recorded and is not promoted.

And the control answered against my own suspicion: reversing the order made the slope **steeper,
not shallower**. The fixed order was not manufacturing the slope — the slope is the engine's.

## 5. Predictions — scored as registered. 3 HIT / 1 MISS / 2 held

| # | registered | outcome | |
|---|---|---|---|
| 1 | all three gates fire | +3.4% / exact / exact | **HIT** |
| 2 | `A10B-K3` in `[49, 54]` | **49.96** (run 2: 51.50) | **HIT** |
| 3 | it comes in **FASTER** than the charged model, because `GSZ=180` puts the `down` runs at 11,520 B | **−2.4%** and **−0.8%**: SLOWER in both runs | **MISS** — §6 |
| 4 | `k=6` misses 50, `k=1` clears it, so the crossing is bracketed by measured arms | 43.78 and 54.86 (run 2: 44.93 and 56.67) | **HIT** |
| 5 | a pass is the SPEED half only; say so in the first paragraph | said, first paragraph | held |
| 6 | does not rehabilitate any MoE quality result | nothing here touches quality | held |

## 6. Prediction 3 fails, and the failure is the most useful thing in this probe

I predicted the measurement would beat the charged model because E31's curve says an 11,520-byte
`down` run reads at ≈0.85 of dense while the 3,584-byte runs E35's numerator was fitted on read
at ≈0.685. It came in slower instead, in both runs. **The reason is visible once the cost is
split rather than averaged.**

Fit `time(k) = a + b·k` over the five arms — `a` is a token before any carve group is read
(attention + head + router, 822,083,584 charged), `b` is one group of 180 neurons (35,389,440
charged):

| | run 1 | run 2 | E35's envelope rate |
|---|---|---|---|
| `a`, ms | 17.413 | 16.933 | |
| **base rate** (attention + head + router) | **47.214 G-w/s** | **48.548 G-w/s** | **47.254** |
| `b`, ms per group | 0.8995 | 0.9553 | |
| **marginal group rate** | **39.345 G-w/s** | **37.046 G-w/s** | |
| **gather penalty at `GSZ=180`** | **0.833** | **0.763** | |
| R² | 0.995 | 0.920 | |

**The base reads at the full envelope rate — 47.2 and 48.5 against E35's 47.254. It is the
marginal carve group that is expensive, at ≈0.80 of it.** So the shortfall is neither a
per-layer fixed cost (the brief's own stated alternative) nor a failure of E35's envelope: it is
the gather, still, at a granularity four times coarser than any previous arm, and E31's ≈0.85
was **optimistic**.

**The envelope is therefore not one number.** It is two:

```
time per token  =  17.2 ms  +  0.93 ms per carve group of 180 neurons
                   ~~~~~~~     ~~~~~~~
                   attention   FFN, and FFN weights cost 1.25x what
                   + head      attention weights cost, per weight
                   + router
```

Any budget from here must price an FFN weight at **≈0.80 of the attention rate**, and every
table in E30/E34/E35 that priced them equally is optimistic by that factor on its FFN half.
This is the third time this programme has cut the same claim (E26 `2.01×` → E31 `1.25×` →
E33 `1.12×`), and it is the first time it has been measured **per group** rather than inferred
from a curve fitted at a different granularity.

## 7. What this says about the goal, in the goal's own words

**"un modello grande (es 10B)"** — 9,999,220,736 parameters, verified three ways from the file
itself. Yes.

**"a 50 token/s (risultato buono)"** — **49.96** on the registered run, 51.50 on the order
control, and a crossing measured twice at **1.12% activation**. The honest sentence is: *this box
runs a ten-billion-parameter artifact at the goal's rate to within the resolution of the
instrument, and the registered reading is four hundredths under the line.* It is not a clean
clear, and this document does not call it one.

**"risultato ottimo è 100 token/s"** — **out of reach at this shape, and now by a measured
margin rather than an argument.** The base term alone is 17.2 ms, so `L=16` at 4096 wide caps at
**≈58 tok/s with a zero-cost FFN**. 100 tok/s needs the whole token in 10 ms, which at the
measured base rate buys `(0.478 G − 0.134 G) / 41.94 M` ≈ **8 layers, with no FFN at all.**
100 tok/s at 4096 wide is not an FFN problem; it is an attention problem, which is what E34
said and this now prices.

## 8. What E36 cannot claim

- **Nothing about quality, at any activation rate.** Synthetic noise, no BPB, no donor, no
  training. 1.17% activation is a far more aggressive MoE than anything validated here for
  quality; Probe-4's own retro-audit demoted its MoE result and E27 measured in-budget
  configurations broken at 0.998 G. **None of that is rehabilitated by a speed number.**
- **Nothing about the router.** `k` is fixed and oracle-free. E23 priced a real router end to
  end and that cost is **additive** to everything here.
- **Nothing about long context.** 40-token benches, weight traffic only. KV grows with context
  and is still the largest un-priced term in the envelope (E30 §8, E35 §7).
- **Nothing about other boxes.** One Zen 2, six threads, 36.30 GB/s.
- **The absolute rates carry ±5%**, as every absolute in this branch does; the ratios and the
  rank ordering do not. The verdict cell sits inside that ±5% of the bar, which is why §4's rule
  mattered more than either reading.

## 9. Owed after E36

1. **The marginal group cost is now the dominant lever** (§6): FFN weights cost 1.25× attention
   weights per weight, and nothing has measured how that varies with `GSZ` on a *carved* file —
   E31 measured it on a synthetic gather. A `GSZ` sweep on `A10B` (`E ∈ {64, 128, 256, 512}` at
   matched charged weight) is the cheapest remaining numerator.
2. **The attention base is now the ceiling** (§7): 17.2 ms of a 20 ms budget. E34 §7's
   attention-shape sweep (width, rank, KV heads) is no longer one item among many — it is the
   only axis with room left on the speed side.
3. **KV traffic at long context** — unchanged, unmeasured, and it eats the same 17.2 ms base.
4. **Time `--lut --lut-group 32`** (E32 §8) — still never timed, still the cheapest numerator.
5. **E35's JSON key `active_budget_at_50_G` is mislabelled** — it holds E34's dense numerator ÷ 50
   (0.9348 G), not the interpolated envelope E35 §4 publishes (0.9451 G). The E35 *document* is
   correct; the key is not. Anything reading that key gets a model nobody registered. Caught here
   because this runner read it first.
