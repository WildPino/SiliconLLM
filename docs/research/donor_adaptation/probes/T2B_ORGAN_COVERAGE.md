# T2b — Does T2's winning rule survive outside the FFN?

**Outcome: `UNIFORM` — the pre-registered label, computed mechanically from §4's thresholds.**
**But it passed by 1.0% of its own bar, and the bar sits INSIDE the ratio's 95% interval. Read §5.**
**Date: 2026-09-04. Donor: Qwen2.5-1.5B rev `8faed761…`. CPU only, no gradients, 1646 s = 27.4 min.**
**Brief: `briefs/BRIEF_T2B_ORGAN_COVERAGE.md` @ `63b1dc6` (pushed before the run).**
**Code: `benchmarks/donor_adaptation/ternary/t2b_organs.py`. Data: `density/results/t2b_organs.json`,**
**per-sequence BPB in `density/results/t2b_arms/*.npy`. Derived quantities (§4 additivity,**
**§5 ratio interval, §6 per-weight rates) recomputed from those arrays by `t2b_derived.py`.**

> **Not audited.** Written by the figure that ran it.

---

## 1. The answer in three lines

**The rule transfers.** Converting `q,k,v,o` and `lm_head` on top of the FFN costs `1.584×` what the
FFN alone costs, against a pre-registered bar of `1.60×`, so T2's `+1.709` may be quoted for the
whole runnable model rather than for 84 tensors of it.

**The smoke's HEAD-BOUND signal did not survive.** `Δ(H-only) = +0.339` against a trigger of
`+0.855`. R1 §4.1's head ternarization is **not** withdrawn and no re-pricing of the speed ledger
is owed.

**Attention is the precision-hungry organ per weight — by 5× — and it does not matter**, because
it is 10.0% of the weights. §6.

## 2. Gates, before any result is read

| gate | required | measured | |
|---|---|---|---|
| eval slice | `ids_sha256 = a1a48dc9…`, 24×512, seed 1234 | identical, 51,870 scored bytes | ✅ |
| baseline | `0.767594958` | **`0.767594958`** | ✅ |
| **arm I** (identity through the FAH code path, incl. the `lm_head` untie) | Δ exactly 0 over 197 tensors | **`+0.000000000`** | ✅ |
| **arm F** (re-derive T2's winner through *this* script) | within `0.01` of `+1.709372` | **`+1.709372490`** — 9 decimals | ✅ |
| calibration | `calib` 32×512 seed 42424, `T = 16,384`, disjoint corpus half | asserted in-run, sha256s recorded | ✅ |

Arm F is the load-bearing control. T2's rule lives in `t2_rules.py`; T2b **imports** it
(`T2.r3_actsearch`) rather than re-deriving it, so F re-running T2's number to nine decimals
verifies the import, the organ selection, the calibration capture and the BPB path in one shot.

Arm I matters more here than it did in T2 because this run **unties the embedding** to convert the
head. An untie that silently changed the output distribution would have shown up as a nonzero I;
it did not.

## 3. Arms

Rule = **R3** throughout (T2's pre-registered winner under `min(R1..R4)`; R5 scored better but is
post-hoc and gates nothing — swapping the rule after seeing its number is the drift this programme
keeps guarding against).

| arm | organs | tensors | BPB | Δ vs base | paired SE | σ_seed | ci95 | zero frac |
|---|---|---|---|---|---|---|---|---|
| base | — | 0 | `0.767594958` | — | — | — | — | — |
| **I** | identity | 197 | `0.767594958` | `+0.000000` | `0.000000` | 0 | — | 0.0000 |
| **F** | gate, up, down | 84 | `2.476967449` | `+1.709372` | `0.068968` | 342 | `[+1.584, +1.850]` | 0.4642 |
| **A** | q, k, v, o | 112 | `1.903568524` | `+1.135974` | `0.036106` | 227 | `[+1.072, +1.213]` | 0.4769 |
| **H** | lm_head | 1 | `1.106583608` | `+0.338989` | `0.012674` | 68 | `[+0.315, +0.364]` | 0.4478 |
| **FA** | F + A | 196 | `3.484251267` | `+2.716656` | `0.129001` | 543 | `[+2.478, +2.980]` | 0.4714 |
| **FAH** | F + A + H | 197 | `3.475705979` | `+2.708111` | `0.124308` | 542 | `[+2.474, +2.960]` | 0.4713 |

SEs are paired sequence bootstraps (2000 resamples, seed 7, byte-weighted) against `base`, from the
per-sequence arrays on disk. σ_seed = 0.005.

## 4. The increments, paired BETWEEN arms

T2 §4 established that a Δ-against-baseline cannot compare two arms — the arms are correlated
across sequences, so the difference of two Δs has a smaller SE than either Δ. These are computed
arm-against-arm on the same 24 sequences.

| contrast | Δ | SE | ci95 | significant |
|---|---|---|---|---|
| `FA − F` (attention, on top of a ternary FFN) | `+1.007284` | `0.079815` | `[+0.851, +1.167]` | yes |
| **`FAH − FA`** (the head, on top of ternary FFN+attention) | **`−0.008545`** | `0.019903` | `[−0.049, +0.030]` | **no** |
| `FAH − F` (everything outside the FFN) | `+0.998739` | `0.071760` | `[+0.856, +1.139]` | yes |

**`FAH − FA` is zero.** Adding the output head to an already-ternary FFN+attention model costs
nothing measurable — the point estimate is very slightly negative and the interval straddles zero
at ±4 σ_seed. Yet the same head measured **alone** costs `+0.339 ± 0.013`, which is 68 σ_seed and
about as unambiguous as this programme gets.

Both numbers are correct. What they say together is that **organ damage does not add.** A plausible
mechanism — and it is stated here as a hypothesis, not a result — is that once the FFN and
attention have moved the logits by `+2.7` BPB, the head's own `+0.34` of error is largely already
contained in that displacement. **This probe does not test that**, and the correct use of the pair
is operational, not mechanistic: *if you are going to ternarize the FFN and attention anyway, the
head is free; if you are not, it is not.*

The same effect, measured on the other organ:

| | Δ |
|---|---|
| attention alone (`A − base`) | `+1.135974` |
| attention on top of a ternary FFN (`FA − F`) | `+1.007284` |
| difference | **`−0.128690`** ± `0.076535`, ci95 `[−0.284, +0.018]` — not significant |

Directionally the same, but this one does not clear its own noise, so only the head's version is
load-bearing.

**Additivity, stated once:** the three single-organ arms sum to `+3.184335`; FAH measures
`+2.708111`. The defect is `−0.476224` ± `0.071950`, ci95 `[−0.620, −0.334]` — **significantly
sub-additive.** An additive model would have put the ratio at `3.184/1.709 = 1.863`, i.e. above the
1.60 bar and in **DIFFUSE** territory. The label this run produced is a direct consequence of
sub-additivity.

## 5. The pre-registered decision — and how thin it is

From `BRIEF_T2B_ORGAN_COVERAGE.md` §4, verbatim, with the measured values substituted:

| outcome | condition | measured | fires? |
|---|---|---|---|
| **UNIFORM** | `Δ(FAH) ≤ 1.60 × Δ(F)` | `2.708111 ≤ 1.60 × 1.709372 = 2.734996` | **YES** |
| ATTENTION-BOUND | `Δ(A) > Δ(F)` | `1.135974 > 1.709372` | no |
| HEAD-BOUND | `Δ(H) > 0.5 × Δ(F)` | `0.338989 > 0.854686` | no |
| DIFFUSE | none of the above and `Δ(FAH) > 1.60 × Δ(F)` | — | no |
| VOID | arm I ≠ 0, or arm F off by >0.01 | both gates passed | no |

**Label: `UNIFORM`.** Its meaning, from the brief: *"the rule transfers; the runnable model costs
what the FFN measurement said, and T2's number may be quoted for the whole model."*

**The margin is 1.0%.** `ratio = 1.584272` against `1.60`. Bootstrapping the ratio itself on the
same 24 sequences (2000 resamples, seed 7) gives:

| | value |
|---|---|
| point | `1.584272` |
| bootstrap SE | `0.034753` |
| ci95 | **`[1.5142, 1.6489]`** |
| fraction of resamples above the 1.60 bar | **`0.326`** |

**The bar is inside the interval.** A third of the resampled worlds land in DIFFUSE. The
pre-registered label is UNIFORM and it stands — the rule was fixed before the run and is not
being renegotiated after seeing the number — but **this is a near-miss reported as a near-miss, not
a comfortable pass.** Two consequences, both operational:

1. Quoting T2's `+1.709` "for the whole model" is licensed by the label, but the honest quote for
   the runnable model is the number actually measured: **`+2.708` at rule R3**, and `1.584×`, not
   `1.335×` (the uniform-per-weight expectation), is the multiplier from FFN to whole model.
2. The ratio is not resolved against its bar at 24 sequences × 512 tokens. Anything that turns on
   *which side of 1.60 this is* needs a larger eval slice, and this probe does not provide one.

**No re-pricing of `SPEED_LEDGER.md` §10 is owed.** The brief made that report conditional on
ATTENTION-BOUND or HEAD-BOUND; neither fired. Separately, §10's rate has already been withdrawn by
`SPEED_LEDGER.md` §11 for an unrelated reason (it was taken on a contended machine and was off by
35%) — that withdrawal stands and is not affected either way by this run.

## 6. Per weight, attention is 5× the FFN — and it does not matter

Counted from the config on disk (`D=1536 F=8960 L=28 V=151936 heads=12/2 hd=128`), the same counts
the brief used to derive its bar:

| organ set | weights | share of non-embedding | Δ | **BPB per 100 M weights** | vs FFN |
|---|---|---|---|---|---|
| FFN `gate,up,down` | 1156.1 M | 74.9% | `+1.709372` | **`0.1479`** | 1.00× |
| attention `q,k,v,o` | 154.1 M | 10.0% | `+1.135974` | **`0.7370`** | **4.98×** |
| head `lm_head` | 233.4 M | 15.1% | `+0.338989` | **`0.1453`** | 0.98× |

This is the P61 result appearing again in a transformer: **`PHASE61_PROJQUANT` measured the SSM
projections as precision-hungry** (+0.018–0.022 where the FFN tolerated far more) and left them
fp32. The attention projections are the closest analogue a transformer has, and per weight they are
five times more expensive to ternarize than the FFN is.

It does not trigger ATTENTION-BOUND because the brief's condition is stated in **absolute** Δ, not
per weight, and attention is only a tenth of the weights. That was the right way to write it — the
decision the condition gates is *"do these organs come out of the ternary set"*, and taking out
10.0% of the weights to recover `1.136` of `2.708` is a different trade from taking out 74.9% to
recover `1.709`. But the per-weight column is the number a **healing** brief should start from:
**if any organ deserves per-organ treatment, it is attention, and it is the cheapest organ to give
it to.**

The head's per-weight cost is indistinguishable from the FFN's (0.98×), which is the quiet good
news of this run: the organ the speed ledger most wants ternary is not special.

## 7. What this does not say

- **Not measured through the runtime.** These are PyTorch numbers about a model `donor_engine.c`
  executes. `--bpb` exists and has never been run at scale; until it does, the runtime's agreement
  with these numbers is assumed, not shown.
- **Not the calibration budget.** R3 is calibration-driven, `D4b` is still unrun, and every
  R3-family number in this programme is a **floor**, this one included.
- **Not R5.** R5 scored `+1.260` on the FFN and is post-hoc. Whether the organ picture holds under
  R5 is unmeasured; nothing here may be composed with it.
- **Not activation quantization.** The `--lut` path's `rel l2 1.40e-01` (`3.10e-02` at G=32) is a
  separate cost measured in `donor_engine.c` and is **not** composed with these weight-side numbers.
- **Not rotation.** `BRIEF_T3_ROTATION.md` is pre-registered and unrun; if it wins, this whole table
  is re-measured, because T3 changes the basis every one of these organs is quantized in.

## 8. What changes because of this

| | before T2b | after T2b |
|---|---|---|
| the runnable model's BPB at R3 | unknown; `+1.709` was a lower bound on 74.9% of the weights | **`+2.708`**, measured over 197 of 197 converted tensors |
| head ternarization (R1 §4.1, 23.5 → 38.0 tok/s) | at risk — the smoke suggested HEAD-BOUND | **not withdrawn**; the head is free once FFN+attention are ternary |
| where per-organ treatment should go | unknown | **attention**: 5× the FFN's damage per weight, at 10% of the weights |
| FFN→whole-model multiplier | assumed ≈ uniform (1.335×) | **1.584×**, ci95 `[1.514, 1.649]` |

---

## 9. Appended 2026-09-08 after E18 — these arms have now been read in GENERATION

T2b scored its arms in BPB only. E18 ran the same `ARM_ORGANS` table — imported from this probe's
own runner, not re-derived — through greedy generation against the donor's own continuations
(`probes/E18_THE_RANKING_LADDER.md`, ledger §31). Five of the six had never been generated with.

| arm | BPB (this probe) | vs chance `4.069819` | greedy agreement |
|---|---|---|---|
| `base` / `I` | `0.767595` | `−3.302224` | **160/160** |
| `H` | `1.106584` | `−2.963235` | **9/160** |
| `A` | `1.903569` | `−2.166250` | **4/160** |
| `F` | `2.476967` | `−1.592852` | **5/160** |
| `FA` | `3.484251` | `−0.585568` | `12/160` |
| `FAH` | `3.475706` | `−0.594113` | `10/160` |

The floor for agreement by frequency coincidence is `12/160` (E18 part A: the best possible constant
predictor, `'\n'`). **Every converted arm in §4's table is at or below it.**

Nothing here is withdrawn — the increments in §4 are exact, `I − base = +0.000e+00` still holds, and
E18's `G-L1` re-confirmed it in generation as token-identical. What changes is what §5's decision
was choosing *between*. **§6's "per weight, attention is 5× the FFN — and it does not matter" is
correct and now has a second reason: at the ranking instrument, none of these arms differ, because
none of them rank.** §5's organ policy was a choice among points that are, to a reader of output
tokens, the same point.

**And §4's ordering does not survive the second metric**: the best-BPB converted arm here (`H`,
`1.106584`) ranks **worse** (`9/160`) than the worst (`FA`, `3.484251` → `12/160`);
`r(BPB, agreement) = +0.4989` across the five, the wrong sign.
