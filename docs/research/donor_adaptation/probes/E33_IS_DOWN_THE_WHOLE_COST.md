# E33 — is `down` the whole carve locality cost, and does a coarser carve pay the predicted `1.25×`?

**Verdict: `LOCALITY-PARTIAL`.** The registered cell reads **1.133 raw / 1.116 byte-normalised**
at S15 against a desk model of **1.253**. The direction I claimed survives; **the size does not —
less than half of the predicted gain shows up.** And at **T10, the goal's own shape, the lever is
not resolvable from 1.0**: 1.043 normalised, below the `1.06` line in every jackknife.

**This is the second time in two days that E31 §6.1's arithmetic has been cut down, and this time
by a measurement rather than by re-reading the code.** The sequence is: `2.01×` claimed → `1.25×`
after reading the engine → **`1.12×` measured at S15, `1.04×` (unresolvable) at the goal's shape.**

**Brief**: `briefs/BRIEF_E33_IS_DOWN_THE_WHOLE_COST.md`, pushed at `2fa83e9` before the runner
existed.
**Addendum**: `briefs/BRIEF_E33_ADDENDUM_TWO_GATES_CANNOT_FIRE.md`, pushed at `c6090f4` **before
any artifact was exported** — two of my own gates were unsatisfiable and are corrected there,
with the confound's sign, not here.
**Runner**: `engine/e33_down_locality.py` (`f5e773b`). **Result**:
`engine/results/e33_down_locality.json`, 144 s, six arms, five interleaved reps.
**Run 1 is VOID and kept on the repo** (`results/e33_down_locality_void_run1.json`) — see §6.

---

## 1. The gates

| gate | what it demanded | reading | |
|---|---|---|---|
| **`G-E33C` as written** | every carved arm charges an **identical** `active_weights_per_token` | S15 `−1.201%` / `−1.501%`, T10 `−1.094%` | **FAILS — as the addendum predicted before the export.** The router is charged (`synth_export.py:66`) and shrinks with `E`; exact equalisation needs `k = 4.1429` |
| **`G-E33C′`** (decides) | every carved arm within **2.0%** of `E256` | worst **1.501%** | **FIRES** |
| **`G-E33A` absolute** (informational) | `S15-E256` within ±10% of E26's `52.557` | **53.388**, `+1.6%` | PASS — informational only; E26 disowned its absolutes |
| **`G-E33A′` ratio** (decides) | `S15-E256 ÷ S15-DENSE` within ±10% of E26's `ratio_vs_dense = 1.9731` | **1.9247**, `−2.5%` | **FIRES** |
| **`G-E33B`** (decides) | `S15-DENSE` within ±10% of E28's `29.30` | **27.74**, `−5.3%` | **FIRES** |

Every anchor was read out of the source probe's JSON at runtime, never from a constant typed from
memory. Each of the six exports passed `GATE V3` — its byte count recomputed independently by
E1's layout function and matched exactly.

**Contention**: pre-run witness **10.5% mean / 25% peak** against a 25% bar on the mean; mid-run
witnesses after each rep **5.0 / 2.0 / 18.0 / 10.0 / 13.0%**, all under the bar.

## 2. The measurement

Five interleaved reps, reps outermost. `k/E` held at `1/4`, so the arms differ in **contiguity**,
not in size (within the 1.5% the addendum registered).

| arm | `GSZ` | `gate`/`up` run | **`down` run** | mean tok/s | median | spread |
|---|---|---|---|---|---|---|
| `S15-DENSE` | — | contiguous | contiguous | 27.738 | 28.230 | 11.4% |
| `S15-E256` | 35 | 26,880 B | **2,240 B** | 53.388 | 53.120 | 6.8% |
| `S15-E64` | 140 | 107,520 B | **8,960 B** | 56.980 | 55.900 | 8.8% |
| `S15-E16` | 560 | 430,080 B | **35,840 B** | 60.510 | 60.580 | 16.9% |
| `T10-E256` | 56 | 114,688 B | **3,584 B** | 9.564 | 9.870 | 12.1% |
| `T10-E16` | 896 | 1,835,008 B | **57,344 B** | 10.088 | 10.130 | 12.8% |

### 2.1 The verdict cell, raw and byte-normalised

| cell | raw | × charge factor | normalised | band | desk model |
|---|---|---|---|---|---|
| **`S15-E16 ÷ S15-E256`** | **1.1334** | ×0.98499 | **1.1164** | `LOCALITY-PARTIAL` | 1.253 |
| `T10-E16 ÷ T10-E256` | 1.0548 | ×0.98906 | **1.0432** | below the `1.06` line | 1.145 |

Both readings of the registered cell land in the same band, so the addendum's tie-break
(normalised wins) was not needed.

**Fraction of the predicted gain that materialised**: `(1.1164 − 1) ÷ (1.253 − 1)` = **46%** at
S15, and `(1.0432 − 1) ÷ (1.145 − 1)` = **30%** at T10.

### 2.2 Robustness — the S15 verdict survives a jackknife, the T10 one does not

Leaving out one rep at a time (five ways), normalised:

| | drop 1 | drop 2 | drop 3 | drop 4 | drop 5 | all | median-based |
|---|---|---|---|---|---|---|---|
| **S15** | 1.089 | 1.115 | 1.125 | 1.108 | **1.145** | **1.116** | 1.123 |
| **T10** | 1.040 | 1.049 | 1.029 | 1.053 | 1.045 | **1.043** | — |

**S15 is `LOCALITY-PARTIAL` under every leave-one-out and under the median**, so the band is a
property of the measurement and not of a lucky rep. **T10 is below `1.06` under every
leave-one-out**, and its *raw* reading flips band across jackknives (1.040–1.065) — which is the
definition of not resolvable. **At the goal's shape the exporter knob is worth nothing I can
measure.**

## 3. Predictions — scored as registered. 2 HIT / 2 MISS / 1 conditional, and the misses are mine

| # | registered | outcome | |
|---|---|---|---|
| 1 | `G-E33A`, `G-E33B`, `G-E33C` all fire | `A`/`A′`/`B` fire; **`C` fails as written**, exactly as the addendum said it must | **MISS as written, called in advance** |
| 2 | **`LOCALITY-CONFIRMED`, ratio `1.18–1.32`** | **1.116 normalised / 1.133 raw — `LOCALITY-PARTIAL`** | **MISS** |
| 3 | T10's ratio is SMALLER than S15's | 1.043 vs 1.116 | **HIT** |
| 4 | `E64` sits between, **nearer to `E16`** (desk: 77% of the way) | **50.4%** of the way — between, but squarely in the middle | **MISS** (the ordering is monotone, the position is not what I said) |
| 5 | absolute rates stay far from the goal; no reading here changes E30's ceiling or E31's budget | holds | **HIT** |
| 6 | if `LOCALITY-ABSENT`, a SECOND top-of-document correction to E31 and §6's disposition withdrawn | not triggered at S15; **effectively triggered at T10** — see §4 | conditional |

## 4. What this does to E31 §6.1 — the disposition, per the brief's own §6

The brief bound me in advance: **`LOCALITY-PARTIAL` ⇒ "the knob is still worth setting, but the
desk model over-credits `down` and E31 §6.1's arithmetic should not be used to price anything
else."** That is the disposition, and it is now in force.

What survives:

* **The mechanism.** `gate`/`up` are group-major and already contiguous; the fine granularity is
  `down` at `PT_BLK = 64`. Nothing here contradicts that, and the monotone ordering
  (`E256 < E64 < E16` at both shapes) is what the mechanism predicts.
* **The direction.** A coarser carve is faster, measurably, at S15.

What does not:

* **The size, for the second time.** `2.01×` → `1.25×` → **`1.12×`**. Each correction came from
  looking harder at something I had asserted.
* **Its usefulness at the goal's shape.** `1.043` at T10, not resolvable. The knob was advertised
  as an exporter-only change worth taking; at T10 there is nothing measurable to take.

**Why the desk model over-credits.** It priced the whole FFN read as if it moved at the rate
E31 measured for the `down` run's granularity, when `gate` and `up` — two of the three matrices,
and two thirds of the FFN bytes — were already in the flat part of the curve at `E=256` and had
nothing to gain. The desk arithmetic applied a `down`-sized penalty to a `gate`+`up`-sized
quantity. That is the same class of error as the `2.01×` claim it replaced: **a measured curve
applied to the wrong denominator.**

## 5. What E33 cannot claim

- **Nothing about quality.** Synthetic noise weights; no BPB, none computed.
- **Nothing about the goal.** Registered as prediction 5 before the run. `S15-E16` at 60.5 tok/s
  is a 1.5 B-shaped file with 56% of its weights deleted, and `T10-E16` at 10.1 tok/s is still
  **5.0× short** of 50.
- **Nothing about `PT_BLK`.** A compiled kernel constant; E31's banner named it as a second knob
  (`PT_BLK = 512` → `1.17×` at S15 on the same desk arithmetic **that has now over-credited twice
  in a row**, so that figure should be read as an upper bound, if at all).
- **Nothing about `--lutblk`.** `donor_engine.c:1437` refuses the LUT path on `quant==4`, so every
  arm here ran packed — which E32 has since made the operative path anyway.
- **Nothing about a real router.** `k` is fixed and oracle-free; E23's router cost is additive.

## 6. Run 1 was VOID, and the contention was mine

Run 1 passed its pre-run witness at **14.8%** and then read **35.5% mean** on the witness taken
after rep 2. **I caused it**: I was running the E32 documentation patch scripts on the same box
while the timings were in flight. It was not cosmetic — the verdict is not robust to that rep:

| | all three reps | reps 1+3 only |
|---|---|---|
| raw | 1.081 → `LOCALITY-PARTIAL` | 1.046 → `LOCALITY-ABSENT` |
| normalised | 1.065 → `LOCALITY-PARTIAL` | 1.030 → `LOCALITY-ABSENT` |

**A band that depends on a contended rep is not a band.** The run is kept at
`results/e33_down_locality_void_run1.json` and `e33_run_void1.log`; run 2 went to five reps with
the operator doing nothing else, and its band survives every jackknife. **The standing law
already said a contended timing is not a timing; what this adds is that the operator's own
documentation work is contention.**

## 7. Owed after E33

1. **E26 §9's exporter item closes with a measured number**, and the number is small: `1.12×` at
   S15, nothing resolvable at T10. It does not close as the "largest lever in the programme",
   which is where E31 §6.1 first put it.
2. **`PT_BLK` remains unmeasured** and is now the only untested part of the mechanism. Given two
   consecutive over-credits from the same desk model, it deserves a measurement before it
   deserves a paragraph.
3. **The T10 non-result is itself a question**: why does the shape with the *finer* `down` run
   (3,584 B vs 2,240 B — further up E31's curve, so more to gain) show *less*? Either the
   carve machinery's fixed cost dominates at T10's arithmetic intensity, or E31's curve does not
   transfer across shapes. Neither is measured.
4. **E29 §8, E26 §9 items 1–3, E22 §8, E20 §8** remain where they were.
