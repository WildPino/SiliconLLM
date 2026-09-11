# E29 — does the residual RANK layers, or is E27's finding just "don't touch the ends"?

**Brief**: `briefs/BRIEF_E29_DOES_THE_RESIDUAL_RANK.md`, pushed at `d804b88` **before this runner
existed**.
**Code**: `ternary/e29_depth_rule.py`, committed at `c7df56a`.
**Result**: `engine/results/e29_depth_rule.json` (`..._smoke.json` for the smoke).
**Cost**: 7,049 s, CPU only, ten arms, `VOID: none`. **No timing is taken and none is reported** —
every `L21` arm here has the identical charged cost of `1.2160 G` active weights by construction, so
a busy box cannot touch this measurement.

## VERDICT — `RESIDUAL-RANKS`, and E27's headline survives at **half the size it was quoted at**

E29 exists because E27's one positive result rested on a comparison I had rigged without noticing.
`L21-MINRES` beat `L21-LAST` by **+56** teacher-forced tokens — but `LAST` deletes layer 27, which
carries the **second-highest residual of all 28** (`0.7618`, against a 12–18 band of `0.284–0.364`).
Beating a rule that throws away the most active block in the back half establishes almost nothing.

The fair control is a random draw of seven layers from the same interior band. Against **that**:

| | teacher-forced (of 160) | vs `MINRES` |
|---|---|---|
| `L21-MAXRES-IN` — the anti-rule | **78** | −35 |
| `L21-RANDMID` — **the null**, 3 seeds: 90, 87, 90 | **89.0** mean, spread **3** | **−24** |
| `L21-MINRES` — E27's rule | **113** | — |
| *(`L21-LAST` — E27's rigged control)* | *57* | *−56* |

**The ordering the brief registered — `MAXRES-IN < RANDMID < MINRES` — holds, and the margin over
the null is `+24` against a registered kill-bar of `≤ 8`.** The residual carries real information
about which layers are disposable: choosing by it is worth 24 tokens over choosing at random, and
choosing *against* it costs 11 more on top of that.

**But the number that belongs in the record is `+24`, not `+56`.** `+56` is the distance to a bad
rule and more than half of it was the bad rule's own fault. E27 §3 and
`decisions/T4_HEALING_PROPOSAL.md` §10 are corrected accordingly; the recommendation itself stands.

## 1. The arms, all ten

Every `L21` arm drops exactly 7 layers from the interior band `[3, 24]`, so depth, parameter count
and charged cost are identical across all five of them — `1.2160 G` active, confirmed per arm in the
result file. Only the *selection* moves.

| arm | dropped | BPB | free | **tf** | mean rank | band |
|---|---|---|---|---|---|---|
| `base` | — | 0.767595 | 160/160 | **160/160** | 1.0 | CHEAPER |
| **`L21-MINRES`** | `12–18` | **0.993446** | 14/160 | **113/160** | 2.4 | **COMPARABLE** |
| `L21-NOADJ` | `9,11,13,15,17,20,25` | 1.260462 | 12/160 | **106/160** | 6.5 | WORSE |
| `L21-RANDMID-s1` | `5,6,7,11,17,18,21` | 1.349121 | 6/160 | 90/160 | 17.2 | WORSE |
| `L21-RANDMID-s3` | `5,7,10,14,18,20,21` | 1.344427 | 11/160 | 90/160 | 36.0 | WORSE |
| `L21-RANDMID-s2` | `4,5,8,11,12,14,23` | 1.445469 | 4/160 | 87/160 | 24.9 | WORSE |
| `L21-MAXRES-IN` | `3,4,5,6,7,21,22` | 1.548773 | 4/160 | 78/160 | 39.3 | WORSE |
| `L21-LAST` | `21–27` | 2.500083 | 1/160 | 57/160 | 3805.8 | WORSE |
| `L18-MINRES` | 10 layers | 1.280834 | 11/160 | 94/160 | 4.7 | WORSE |
| `L17-MINRES` | 11 layers | 1.327885 | 6/160 | 79/160 | 9.7 | WORSE |

Bands are E27's, unchanged: free floor 12, `AT-FLOOR ≤ 14`; teacher-forced **> 119 CHEAPER**,
**107–119 COMPARABLE**, **< 107 WORSE**. Chance BPB 4.069819.

## 2. Replication — the run reproduced E27 before it extended it, at exactly zero

`Depth`, `measure_residuals` and `keep_set` are **imported** from `e27_floor.py`, not
reimplemented, and the anchor literals are read out of `e27_floor.json` before the model loads.

| gate | demanded | read |
|---|---|---|
| `G-E29a` | `base` = `0.7675949641196624`, 160, 160 | **FIRES** |
| `G-E29b` | `L21-MINRES` = `0.993446`, free 14, tf **113** | **FIRES**, `abs_diff 0.000e+00` |
| `G-E29c` | `L21-LAST` = `2.500083`, free 1, tf **57** | **FIRES**, `abs_diff 0.000e+00` |
| `G-E29d` | the residual profile reproduces E27 to `1e-6` | **FIRES**, worst abs diff `0.000e+00` |
| `G-E29e` | frozen slice sha `a1a48dc9…`, 51,870 bytes | **FIRES** |

`MAXRES-IN` is the planted control and it **fires**: at 78 it is 11 tokens below a random draw from
the same band, so the residual is informative in both directions and not merely correlated with
"middle".

## 3. What actually predicts the damage — and the one place it does not

Summing the residuals of the dropped layers orders the five `L21` interior arms almost perfectly:

| arm | Σ residual dropped | max residual dropped | tf |
|---|---|---|---|
| `L21-MINRES` | **2.222** | 0.364 | **113** |
| `L21-NOADJ` | 2.486 | 0.398 | 106 |
| `L21-RANDMID-s2` | 2.835 | 0.508 | 87 |
| `L21-RANDMID-s3` | 2.867 | 0.503 | 90 |
| `L21-RANDMID-s1` | 2.988 | 0.503 | 90 |
| `L21-LAST` | 3.363 | **0.762** | **57** |
| `L21-MAXRES-IN` | **3.432** | 0.544 | 78 |

**17 of 20 pairs are concordant** (more residual dropped → fewer tokens). Two of the three
exceptions are `RANDMID` seeds differing by 3 tokens, which is the seed spread itself. **The third
is the interesting one: `LAST` drops LESS residual mass than `MAXRES-IN` and is 21 tokens worse.**

So both readings of E27 were partly right, and E29 separates them:

* **the residual ranks** — `MINRES` 113 > `RANDMID` 89 > `MAXRES-IN` 78, at identical cost; and
* **the ends matter beyond residual mass** — `LAST` is punished for *where* it cuts, not for how
  much activity it removes. Its mean rank of the correct token is **3805.8**, against 39.3 for the
  anti-rule; it is not a degraded model, it is a broken one.

*(§3 is post-hoc. Per E14 §6 it is NOT promoted to a gate and no later probe may cite it as one.
The residual-mass ordering was not registered in the brief; only the three-point ordering was.)*

## 4. Contiguity is not load-bearing — with a confound named

`NOADJ` (no two adjacent, otherwise ascending residual) reads **106** against `MINRES`'s contiguous
**113**: within the ±10 the brief predicted, and on the *worse* side. But the non-adjacency
constraint **forces higher-residual layers into the set** (Σ 2.486 against 2.222), and §3's ordering
accounts for roughly that much on its own. **This arm cannot separate "contiguity helps" from "the
constraint made it pick worse layers", and it is reported as inconclusive on contiguity** rather
than as a result. What it does establish is the weaker, useful thing: a scattered set is not
catastrophic, so nothing about the finding depends on removing a contiguous block.

## 5. The cliff between 21 and 14 is not a straight line

E27 had 113 at 21 layers and 54 at 14, with nothing in between. Filling two points in:

| layers kept | 28 | **21** | **18** | **17** | 14 |
|---|---|---|---|---|---|
| tf of 160 | 160 | **113** | **94** | **79** | 54 |
| dropped | 0 | 7 | 10 | 11 | 14 |

From 21 to 18 costs 19 tokens over three layers; **from 18 to 17 costs 15 over one**. The knee sits
right there, and it is where `MINRES` is forced to start taking layers from outside the 12–18 trough
— `L18` already reaches out to layer 25, and `L17` adds layer 9.

## 6. The predictions, scored as written

| # | registered | read | |
|---|---|---|---|
| 1 | all five gates fire | all five, at `0.000e+00` | **HIT** |
| 2 | `MAXRES-IN` in `45–75` | **78** | **MISS** by 3 — the anti-rule is less destructive than I expected |
| 3 | `RANDMID` in `85–105` — *"the one I am least confident in"* | **89.0** (90, 87, 90) | **HIT** |
| 4 | `NOADJ` within ±10 of `MINRES` | **106**, −7 | **HIT** |
| 5 | `L18-MINRES` in `95–115`; `L17-MINRES` in `80–105` | **94**; **79** | **MISS** by 1 each |

Three hits, three misses, all three misses narrow and all three in the same direction: **I
consistently expected the damage to be slightly less than it was.** Prediction 3 was the load-bearing
one and it landed; prediction 2's miss is the honest surprise, because it means the gap between the
anti-rule and the null (11 tokens) is smaller than the gap between the null and the rule (24).

## 7. What E29 does NOT establish

* **Nothing about speed, and nothing about the goal.** No timing was taken. Every `L21` arm is
  `7.75 tok/s` at `T10` with a dense FFN per E27 §4 — corrected to ~7.4–7.5 by E26 part B's carve
  finding where a carve is involved, which here it is not. The budget does not move.
* **Nothing about a 10 B's residual profile.** These residuals are this donor's, on one calibration
  slice, at one width. Whether a larger model has the same first-block spike (`19.05` at layer 0),
  the same middle trough and the same tail rise (`0.762` at 27) is untested and is the obvious next
  question if depth is ever used at scale.
* **Nothing about healing.** Every arm is surgery on frozen weights, no retraining anywhere.
* **Three seeds is three seeds** — though the spread came out at 3 tokens, which is small against
  the 24-token effect, so the comparison is resolved rather than inconclusive.
* **Nothing about `NOADJ`'s question.** See §4: confounded by construction.

## 8. What this owes forward

1. **Restate the claim everywhere it appears.** "+56 over `LAST`" is retired in favour of **"+24
   over a random draw from the same band, seed spread 3"**. Done in E27 §3 and in the T4 proposal;
   anything written later that quotes 56 as the size of the effect is quoting a rigged control.
2. **A contiguity arm that is not confounded** — equal residual mass, scattered versus contiguous —
   if anyone ever needs the answer. E29 does not have it.
3. **The residual profile at another scale.** The rule is only usable at 10 B if the profile's
   shape transfers, and that is one calibration pass on any larger donor.
