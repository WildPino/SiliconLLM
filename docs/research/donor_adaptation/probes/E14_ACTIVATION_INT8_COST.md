# E14 — what does int8 activation quantization cost in BPB?

Pre-registered `a6a6607`; §10 amendment (verdict cell moved to 1.5 B) `47184ca`, pushed after
reading the A0 baseline and **before any treatment arm ran**. Runner
`engine/e14_activation_cost.py`. Engine: the E13 build (`--lutblk`), `--threads 6`.

Slice: 12,288 tokens, 12,287 predicted, **51,941 scored bytes, 4.227313 bytes/token** — identical
for the 0.5 B and 1.5 B ids files, so **one chance line serves both cells: 4.071878 BPB**.

---

## 0. VERDICT — `CHEAP-BUT-NOT-NEUTRAL`

**The registered gate returns `ACTIVATION-CHEAP` at the 1.5 B verdict cell. The registered gate
cannot see what happened.**

| | A1 whole-vector | A2 group-32 |
|---|---|---|
| `dBPB` from fp32 activations | **-0.016961** (better) | **-0.005991** (better) |
| section 4's band, as registered | `ACTIVATION-CHEAP` | `ACTIVATION-CHEAP` |
| `abs(dBPB)`, regime-independent | `MARGINAL` | `CHEAP` |
| **G-N3 greedy agreement vs A0** | **45.6%** | **64.4%** |

Int8 activations **lower** bits-per-byte by 0.017 at a cell sitting 0.625503 **below** the chance
line -- where BPB means what it usually means -- while **changing the top-1 token on 54% of
positions**. BPB scores; greedy ranks. Noise that softens an over-confident logit vector improves
the score without improving the ranking, and can reorder the argmax freely while doing it.

**So `ACTIVATION-CHEAP` is true of the band and false of the model, and E14's registered
instrument set contains no gate that could have told the difference.** That gap is the result.

**Three gates, three outcomes:**

- **G-N0 PASSES** at both cells, `A3 - A1 = +0.000e+00`, and G-N3 finds A1 and A3 **token-for-token
  identical** across 160 generated tokens. **E13's bit-identity is now confirmed three independent
  ways** -- sha256 over 311 MB of logits, BPB at two scales, and greedy trajectories.
- **G-N1 DOES NOT FIRE at either cell**, `A2 - A1 = +0.300872` at 0.5 B and `+0.010970` at 1.5 B.
  The harness is not at fault: in `abs(dBPB)` the arms separate 24x and 2.83x with the more
  accurate arm nearer A0, and G-N3 ranks them the same way. **What fails is the direction I wrote
  into the gate**, and section 10.3's claim that G-N1 was regime-independent is falsified by its
  own run, two hours after I wrote it.
- **G-N2 returns `ACTIVATION-CHEAP`, and the label does not survive G-N3.**

**Nothing here licenses spending E13's 1.358x lever**, and section 3.5 adds a second reason: E13
measured that lever on `--lutblk` alone, which is arm A3 = A1 -- the arm with the *larger* quality
move. The quality-cheap arm, A2, **has no speed number at all.**

**The generalisation, which is the fifth pre-registered rule in this programme aimed at the wrong
number and the second one in this experiment:** G-N1 assumed a *direction* that holds only while
the model predicts; G-N2 assumed a *sign* that holds only while quantization damages. **A gate
written as an inequality between two damaged models inherits the chance line, and a gate written
as a one-sided band inherits the assumption that the treatment hurts.** Compare distances from the
reference, and pair every scoring metric with a ranking one.

---

## 1. A protocol defect, found while preparing E15 and recorded before the verdict

**E14 scored one 12,288-token sequence, not 24 documents of 512.** `donor_engine.c:1249` reads
`int SL = seqlen>0 ? seqlen : (int)n;` -- with no `--seqlen`, the whole ids file is one sequence.
E1 passes `--seqlen 512` (`e1_bpb_through_engine.py:310`); **E14's harness does not.** So every arm
here ran with RoPE positions up to 12,287 and attention reaching across 23 unrelated document
boundaries.

That is not a discrepancy with E1, it is a different measurement, and it explains a gap I would
otherwise have had to explain away:

| | 0.5 B TQH, the identical file | protocol |
|---|---|---|
| **E1** `e1_05b.log:30` | **4.531234** | 24 x 512, `--seqlen 512`, 51,870 bytes / 12,264 pred |
| **E14** A0 | **4.629292** | 1 x 12,288, no `--seqlen`, 51,941 bytes / 12,287 pred |

Both are internally consistent -- each divides its own charged bytes by its own predictions -- and
they answer different questions. **The denominator is where the unit hides, and here it moved
twice at once**: the byte convention *and* the context length.

**What survives unchanged.** G-N0 and G-N1 compare arms to each other at an identical protocol,
identical positions, identical bytes. They are untouched, including G-N1's falsification. And the
finding that the 0.5 B export sits above the chance line survives independently: **E1 measures the
same artifact at +0.461 above the same slice's line at 512 context**, so the conclusion does not
rest on E14's protocol at all.

**What does not survive cleanly.** G-N2's bands (`0.010` / `0.020`) were drawn from 512-context
regimes -- 2 sigma_seed, and the floor of what Phase 61 rejected. Reading a 12,288-context
`dBPB` against them is a provenance mismatch of exactly the kind E8 section 3 was written about.
**A `--seqlen 512` re-run of both cells is owed** (section 4), and until it lands the verdict below
is a verdict at 12 k context and is labelled that way everywhere it appears.

---

## 2. The 0.5 B cell — instrument only, and it earned that demotion

| arm | activations | BPB | Δ from A0 | vs chance |
|---|---|---|---|---|
| **A0** | fp32 (the shipped path) | 4.629292115 | — | **+0.557414 above** |
| **A1** | int8, one scale per vector | 4.315413082 | **−0.313879** | +0.243535 above |
| **A2** | int8, one scale per 32 ch | 4.616285418 | **−0.013007** | +0.544408 above |
| **A3** | int8, E13 blocked layout | 4.315413082 | **−0.313879** | +0.243535 above |

**Quantizing the activations "improved" BPB by 0.314.** It is not an improvement. A0 is +0.557 past
chance — confidently wrong — and adding noise drags it toward chance. Read against §4's bands
mechanically, A1 would return `ACTIVATION-CHEAP` and license spending E13's 1.358× lever. **That is
the false license §10.2 was written to refuse, and it is now demonstrated rather than argued.**

### 2.1 G-N0 — PASS, and it corroborates E13 independently

`A3 − A1 = +0.000e+00`, identical to the last digit (12.644809 nats/token both). **E13 proved the
blocked layout bit-identical with a sha256 over 311 MB of logits; E14 confirms it through a
different mode over 12,287 scored tokens.** Two independent gates, same answer: `--lutblk` is a
permutation of a permutation and changes no arithmetic.

### 2.2 G-N1 — DOES NOT FIRE, and the fault is mine

The gate required `BPB(A2) < BPB(A1)`. Measured: `A2 − A1 = +0.300872`. **It does not fire.**

**The harness is not at fault — it resolves the input-error difference better than the gate asked
it to.** E11 measured the input-side error at 1.40e-01 whole-vector against 3.10e-02 at G=32, a
4.5× difference. In `|Δ from A0|` the arms separate by **24×** (0.013007 vs 0.313879), with the
more accurate arm landing nearer A0, exactly as it should.

**What fails is the direction I wrote into the gate.** `BPB(A2) < BPB(A1)` presumes that deviating
from fp32 activations is harmful. Above the chance line it is "helpful", so the ordering inverts.
The regime-independent form is

> `|BPB(A2) − BPB(A0)| < |BPB(A1) − BPB(A0)|`

and **that fires decisively, 0.013007 < 0.313879.**

**§10.3 is falsified by its own run.** It claimed G-N1 was "a property of the harness and the arms,
not of where the model sits", and listed it as readable at both cells. That was wrong, and it was
written two hours before the run that refuted it. **Fifth pre-registered rule in this programme
aimed at the wrong number** — E10's G-K3 was the third, E11's G-L2 the fourth. (E12's arm `Z` is a
different failure: a control that had been retired, not a band pointed at the wrong quantity.)

**The pattern across the last three is the same and worth naming: every one of them assumed a
direction that only holds while the model still predicts.** E12's `Z > F`, E14's `A2 < A1`. **A gate
written as an inequality between two damaged models inherits the chance line whether or not its
author noticed.** The repaired form — compare *distances from the reference*, not the raw ordering
— is regime-independent, and is what the remaining gates should have been written as.

## 3. The 1.5 B cell — the verdict

`A0 - chance = -0.625503`. **G-N2 is readable here**, which is what section 10's post-registration
move to this cell was for.

| 1.5 B arm | activations | BPB | dBPB from A0 | G-N3 greedy vs A0 |
|---|---|---|---|---|
| **A0** | fp32 (the shipped path) | 3.446375376 | — | — |
| **A1** | int8, one scale per vector | 3.429414114 | **-0.016961262** | **45.6%** (73/160) |
| **A2** | int8, one scale per 32 ch | 3.440384359 | **-0.005991017** | **64.4%** (103/160) |
| **A3** | int8, E13 blocked layout | 3.429414114 | -0.016961262 | 45.6%, identical to A1 |

**The registered verdict is `ACTIVATION-CHEAP` for both arms.** Read section 4's bands
mechanically -- `dBPB <= 0.010` -- and that is what comes out, and it is what the runner printed.

**It is the wrong description of what happened, and G-N3 is why.**

### 3.1 dBPB is NEGATIVE at a cell that predicts

At 0.5 B the negative delta had an explanation: A0 sat +0.557 above the chance line, so noise
dragged it toward chance. **That explanation is unavailable here.** A0 is 0.625503 *below* the
line; BPB means what it usually means; and quantizing the activations still improved it, by
0.016961 for A1 (3.4 sigma_seed) and 0.005991 for A2 (1.2 sigma_seed).

**Section 4's bands are one-sided.** `<= 0.010 CHEAP`, `0.010-0.020 MARGINAL`, `>= 0.020 COSTLY`
were drawn for a *cost*, and a negative delta falls off the bottom into `CHEAP` by construction --
it cannot be anything else. Under the regime-independent form the picture changes:

| arm | `dBPB` (as registered) | `abs(dBPB)` (repaired) |
|---|---|---|
| A1 | -0.016961 -> `CHEAP` | 0.016961 -> **`MARGINAL`** |
| A2 | -0.005991 -> `CHEAP` | 0.005991 -> `CHEAP` |

**This is the same defect as G-N1, in the verdict gate.** G-N1 assumed a *direction* that holds
only while the model predicts. G-N2 assumes a *sign* that holds only while quantization damages.
Two of E14's three gates were written as though I already knew which way the number would move.

### 3.2 G-N3 says the model is not the same model

Section 6 fixed G-N3 as descriptive -- "it is NOT a gate, and no verdict is read off it" -- and
that stands; it is also the only instrument in the registered set that is rank-based, and it was
pre-registered and then never implemented, because `e14_activation_cost.py` runs `--bpb` only. It
was written and run afterwards (`e14_gn3_greedy.py`), against A0, exactly as section 6 specifies.

**A1 reproduces A0's greedy trajectory on 45.6% of tokens and diverges at token 0. A2 reproduces
64.4% and diverges at token 1.** So int8 activations change the top-1 token on **54% of positions**
while *improving* bits-per-byte by 0.017.

Those two facts are not in contradiction, and the resolution is the point of this probe:
**BPB is a scoring metric and greedy is a ranking one.** Quantization noise acts like a mild
softening of the logits, which lowers the negative log-likelihood of a slightly over-confident
model without making its argmax better -- and can reorder the argmax freely while doing so. A model
that were genuinely "as good" would not rewrite the top-1 on more than half its tokens.

**`ACTIVATION-CHEAP` is what the band returns and it is not a license.** The registered instrument
set has no gate that can tell "costs nothing" from "moved the metric", and section 6 correctly
forbade building one out of G-N3 after the fact. That gap is the finding, not the number.

### 3.3 The ordering that does hold

Both the repaired G-N1 and G-N3 rank the arms the same way, and in the direction E11's input-side
error predicts:

| arm | E11 input rel L2 | `abs(dBPB)` | greedy kept |
|---|---|---|---|
| A2, one scale per 32 ch | **3.10e-02** | **0.005991** | **64.4%** |
| A1, one scale per vector | **1.40e-01**, 4.5x worse | 0.016961, 2.83x worse | 45.6% |

**The harness resolves the difference E11 measured, on two independent instruments.** What failed
was never the apparatus.

### 3.4 G-N0 passes again, and E13 is now confirmed three ways

`A3 - A1 = +0.000e+00` at this cell too, and G-N3 finds the two arms **token-for-token identical**
across all 160 generated tokens. E13 proved the blocked layout bit-identical with a sha256 over
311 MB of logits; E14 confirms it through BPB at two scales and through greedy trajectories.
`--lutblk` is a permutation of a permutation and changes no arithmetic.

### 3.5 The composition with E13 is crossed

E13 measured **1.358x** with `--lutblk` alone -- whole-vector scaling, which G-N0 and G-N3 both
show is arm **A3 = A1**. That is the arm with the larger quality move. The arm that is quality-
cheap, A2, has **no speed number at all**.

| arm | quality (here, 1.5 B) | speed (E13) |
|---|---|---|
| A1 / A3, whole-vector | `abs(dBPB)` 0.016961, greedy 45.6% | **1.358x measured** |
| A2, group-32 | `abs(dBPB)` 0.005991, greedy 64.4% | **never measured** |

`--lutblk` and `--lut-group` are orthogonal flags in `donor_engine.c` (`--lutblk` sets
`g_lut=1; g_lutblk=1`; `g_group` is separate), so `--lutblk --lut-group 32` is reachable. But
Phase 61's law applies exactly here: a kernel that gains 1.358x applying one scale per vector does
not thereby gain it applying one per 32 channels, and the extra scales land in the inner loop.
**Nothing licenses carrying 1.358x across to A2.**


## 4. What this cannot claim

- **It prices the activations, not the LUT path.** A cheap verdict does not make `--lutblk`
  shippable: Phase 60's law still applies and adoption needs the end-to-end parity gate the packed
  default already carries.
- **It says nothing about speed.** No timing was taken and none may be quoted from this run.
- **It does not move the goal.** The lever is 1.358×; Coder-7B would reach 9.21 tok/s and remain
  **5.4× short of 50**. The gap is a property of the model (§19.3, E10, E13).
- **The 1.5 B number is a 1.5 B number.** E12 is the standing warning about quality numbers taken
  at one shape, and this one inherits it — a Coder-7B repeat is owed.
- **Every number here is at 12,288 context** (section 1). It cannot be quoted alongside E1's, T2's
  or E12's, all of which are at 512.

## 5. Owed

1. **The `--seqlen 512` re-run, both cells.** Four arms x two cells, roughly three hours. It puts
   E14 in the same column as every other quality number in this programme and lets G-N2 be read
   against bands that were drawn at 512. `e14_activation_cost.py` needs the flag threaded through;
   it is one line, and it is deliberately NOT being added while the current run is in flight.
   **Until it lands, the verdict here is a 12 k-context verdict.**
2. **Restate G-N1 in its repaired form wherever it is carried forward**:
   `|BPB(A2) - BPB(A0)| < |BPB(A1) - BPB(A0)|`. The raw ordering is regime-dependent; the
   distances are not.
3. **A speed number for A2.** `--lutblk --lut-group 32` is reachable (the flags are orthogonal in
   `donor_engine.c`) and unmeasured. Until it exists, the quality-cheap arm and the fast arm are
   different arms and the 1.358x cannot be carried across (section 3.5). Needs an idle machine and
   at least three repetitions, per the standing rule that a contended timing is not a timing.
4. **A ranking gate, properly banded, for any future quality probe.** G-N3 could not be promoted
   here and should not have been -- brief section 6 forbade it before the run, correctly, because
   no band for it could be justified from anything measured. The band is still missing. Deriving
   one (from what greedy agreement two *known-equivalent* models produce on this harness) is the
   prerequisite for any verdict that claims a conversion is harmless.
5. **A Coder-7B repeat**, per section 4 — and note that E15 now measures whether that donor's
   shipped artifact predicts at all, which has to be answered before pricing a lever on it.

---

## 6. Appended 2026-09-08 after E18 — §5 item 3 is still owed, and here is what did NOT pay it

E18 measured a floor for greedy agreement (`probes/E18_THE_RANKING_LADDER.md` part A, ledger §31.1):
the best possible **constant** token predictor scores `11/160` at 0.5 B and `12/160` at 1.5 B, both
`'\n'`. That is a lower bound on *degenerate* output, and it is the only band this programme has.

**It does not band §3's `45.6%`.** The int8-activation arm is not a degenerate model — it reads
`0.625503` BELOW the chance line, in the regime where BPB means what it usually means — so knowing
what a constant emitter scores says nothing about what a *genuinely different but equally good*
model scores against a reference continuation. **§5 item 3 — the intermediate ranking band — remains
owed and unsupplied.** Until it exists, `45.6%` is a measured number without an interpretation, and
the repaired `G-N1` (§4, distances from the reference) is still the only gate here that decides.

E18 does add one thing §5 could not assume: the two instruments **can** be made to agree exactly
when both are pointed at the same artifact. `G-L2` reproduced the engine's `12/160` and `10/160`
from PyTorch with zero error, so a future band for §5 item 3 can be built in either harness.
