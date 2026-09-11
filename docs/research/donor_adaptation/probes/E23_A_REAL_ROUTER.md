# E23 — a real router, because every carve number here is a ceiling

**Verdict: `ROUTER-HOLDS` on the carve alone, `ROUTER-COSTS` on the composed configuration —
and the composed one is the one the T4 proposal was about.**

Pre-registered at `briefs/BRIEF_E23_A_REAL_ROUTER.md`, commit `25bde22`, **pushed before the
runner existed**. Runner `benchmarks/donor_adaptation/ternary/e23_router.py` (`7d33921`).
Results `benchmarks/donor_adaptation/engine/results/e23_router.json`. 7 arms, 2694 s, CPU only.
**Nothing exported, no timing taken, `6.79 tok/s` untouched.**

---

## 0. The one-paragraph version

Every carve number this programme has published — D0, D0c, E19, E22 — used an **oracle** router
that reads the true squared activation mass and then keeps the top `k`. E23 replaced it with a
per-layer ridge regression from the block input to `sqrt(group mass)`: closed form, no gradients,
no tuning, `11.0 M` parameters charged to the budget. **On the FFN carve alone it keeps 91% of
the oracle's routing value** — better than I registered, and the registered alternative fired.
**On the composed configuration it keeps 71%**, and the composed configuration is what
`decisions/T4_HEALING_PROPOSAL.md` asked for GPU weeks to heal. Composition, which *gained* 9
teacher-forced tokens under the oracle, **loses 8 under a real router.** The T4 target must be
re-derived — and §7 shows the re-derivation is not a retreat, because `k = 133` was never the
depth the budget actually permits once the router is paid for.

## 1. Arms

Qwen2.5-1.5B, frozen eval slice 24×512 (ids sha `a1a48dc9…`, 51,870 scored bytes), five frozen
E6 prompts × 32 new tokens = 160 positions, D0c partition `E = 256`, `k = 133`.
Chance = `4.069819` BPB. Free-running bands (E17/E18): floor 12, `AT-FLOOR ≤ 14`, `RANKS ≥ 80`.
Teacher-forced band (E20 part B): `> 119` CHEAPER, `107–119` COMPARABLE, `< 107` WORSE.

| arm | router | BPB | Δ base | free | **tf** | mean rank | rank≤5 | act | band-free | band-tf |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` | — | 0.767595 | — | 160 | 160 | 1.00 | 160 | 1.0000 | RANKS | CHEAPER |
| `V52-ORACLE` | true mass | 0.909441 | +0.141846 | 12 | 117 | 2.49 | 151 | 0.5195 | AT-FLOOR | COMPARABLE |
| `V52-STATIC` | top-`k` by mean calib mass | 1.191953 | +0.424358 | 9 | **99** | 6.26 | 138 | 0.5195 | AT-FLOOR | WORSE |
| `V52-RANDOM` | fixed random `k`-of-`E` | 2.156555 | +1.388960 | 4 | 42 | 232.56 | 75 | 0.5195 | AT-FLOOR | WORSE |
| **`V52-LINEAR`** | **ridge** | 1.004558 | +0.236963 | 7 | **110** | 4.17 | 143 | 0.5195 | AT-FLOOR | **COMPARABLE** |
| `QO512+V52-ORACLE` | true mass | 1.005039 | +0.237444 | 15 | 126 | 2.03 | 148 | 0.5195 | PARTIAL | CHEAPER |
| **`QO512+V52-LINEAR`** | **ridge** | 1.201477 | +0.433882 | 9 | **102** | 5.15 | 140 | 0.5195 | AT-FLOOR | **WORSE** |

## 2. Gates

- **`G-T0`** — `base` 160/160 free and teacher-forced. **FIRES.**
- **`G-T1`** — `V52-ORACLE` reproduces E22 to `< 1e-9`: `0.909441`, free 12, tf 117, act
  `0.51953125`. **FIRES.** `QO512+V52-ORACLE` likewise reproduces E22's `1.005039`, 15, 126.
  The harness has not drifted and both published ceilings are re-read exactly.
- **`G-T2`** — the planted negatives must be separable from the verdict arm. Linear 110, static
  99, random 42. **FIRES**, by 11 and 68 tokens.
- **`G-T4`** — achieved activation `0.51953125` on every routed arm, `ACT_TOL = 0.002`.
  **No failures.** All seven arms are compared at the same cost.
- **`VOID`: none.**

**A disclosure about `G-T2`'s wording, made before the result was known** (commit `7d33921`):
the brief says the negatives must not score *"within 2 teacher-forced tokens"* of the verdict
arm, which is ambiguous about sign. The code took the strict reading — LINEAR must **beat** both
by more than 2 — and that is the reading scored above. Under either reading `G-T2` fires here,
so the ambiguity did not decide anything; it is recorded because resolving a registered wording
after seeing the data is how E20 run 1 went wrong.

## 3. The verdict, in the brief's own bands

`retention = (tf(LINEAR) − tf(RANDOM)) / (tf(ORACLE) − tf(RANDOM))`

| configuration | tf linear | tf oracle | tf random | **retention** | BPB cost of a real router | band |
|---|---|---|---|---|---|---|
| `V52` (carve only) | 110 | 117 | 42 | **0.9067** | `+0.095117` | **`ROUTER-HOLDS`** |
| `QO512+V52` (composed) | 102 | 126 | 42 | **0.7143** | `+0.196438` | **`ROUTER-COSTS`** |

`ROUTER-HOLDS` requires retention ≥ 0.80 **and** tf ≥ 107. `V52-LINEAR` clears both (0.9067, 110).
`QO512+V52-LINEAR` clears neither (0.7143, 102).

**`G-T3` is the verdict gate and it was registered over both LINEAR arms. They disagree, so the
verdict is the weaker of the two for the configuration that was being proposed for GPU time:
`ROUTER-COSTS`.** The brief fixed what that means before the run: *"the carve survives but the
healing target must be re-derived at a shallower depth."* §7 does that re-derivation.

## 4. The finding that is actually new: composition reverses sign

This is the sharpest thing in the run and it was pre-registered as prediction 4.

| | `V52` | `QO512+V52` | composition |
|---|---|---|---|
| **oracle** router | 117 | 126 | **+9** — adding low-rank attention *helped* |
| **real** (ridge) router | 110 | 102 | **−8** — adding low-rank attention *hurt* |

Under the oracle, E22 found the composed arm RANK-SUB-ADDITIVE: `126 ≥ min(144, 117)`, better
than either part predicted. **That behaviour is an artefact of the oracle.** Give the router only
what a router can see and the same composition goes the other way, a swing of 17 teacher-forced
tokens. Mechanism, stated as a reading and not as a proof: the oracle's group scores are computed
from the *true* post-gate activations, so they are unaffected by upstream damage; the ridge
router reads the **block input**, which the rank-512 attention cut has already perturbed. The
carve and the rank cut are independent under an oracle and **coupled through the router's input**
under a real one. Mean rank confirms the coupling is where the damage lands: `2.03 → 5.15` on the
composed arm versus `2.49 → 4.17` on the carve alone.

**E22 §6's additivity reading therefore needs restating, and this is the second time an E2x
composition claim has moved** (E21 §4a was weakened by E22, and is weakened again here in the
sense that *both* of its measured composition directions were oracle-conditioned). Recorded in
`E22_DOES_CHEAP_COMPOSE.md` §9 and `E21_CAN_RANK_BUY_IT.md` §10.

## 5. Predictions

| # | registered | measured | |
|---|---|---|---|
| 1 | `G-T0`, `G-T1` fire; oracle reproduces E22 to the token | both fire, both oracles exact | **HELD** |
| 2 | STATIC and RANDOM both `AT-FLOOR` free and **below 40** tf | both AT-FLOOR, but **RANDOM 42** and **STATIC 99** | **MISSED on both** |
| 3 | `V52-LINEAR` reads `ROUTER-COSTS`, retention 0.40–0.80 | **0.9067 → `ROUTER-HOLDS`** | **MISSED** |
| 4 | `QO512+V52-LINEAR` falls below `V52-LINEAR` by more than the fp32 pair did | −8 vs oracle's +9 | **HELD** |
| 5 | `G-T4` exact on every arm — top-`k` is a hard count | no failures | **HELD** |

**3 held, 2 missed.** Prediction 3 missed on the arm it named and *held* on the composed arm
(0.7143, inside the registered band) — so the registered intuition was right about where the
router is weak and wrong about how weak, which is the honest way to report it rather than
claiming the band "basically held".

**Prediction 2 missed on both arms.** `RANDOM` read `42` against a registered `below 40` — a small miss, and it is still a miss; the number is recorded rather than rounded into the band. **`STATIC` is the substantive one, and it falsifies a transfer I asserted.** The brief
argued from Probe-4 that *"a static choice of experts should be worthless"* because working sets
are approximately i.i.d. across tokens. **`V52-STATIC` reads 99/160** — retention 0.76, seven
tokens short of `ROUTER-HOLDS` on its own, from a router that does not look at the token at all.
Probe-4's i.i.d. finding is about **per-neuron** working sets; **group mass over D0c's 256
co-activation groups is not i.i.d. across tokens on this donor.** The whole token-dependent
routing decision is worth `110 − 99 = 11` teacher-forced tokens, and the difference between
routing well and routing at random is 68. **Most of the carve's survival is in which groups are
chosen on average, not in choosing per token.** That is a cheaper engine than a router, and it is
recorded as a question for E24 rather than claimed as a design.

**THE REGISTERED ALTERNATIVE FIRED, on `V52-LINEAR` only.** The brief wrote, before the data
existed: *"If `V52-LINEAR` reads `ROUTER-HOLDS` … then the carve is not an oracle artefact."*
It does, at retention 0.9067, from a closed-form ridge with no gradients and no tuning. **The
carve as such is not an oracle artefact.** The alternative said more than that — that
`QO512+V52` would be *"constructible without an oracle"* — and **that half is refused by the
composed arm.** Both halves are reported; taking only the half that fired would be the error
this programme's own no-anchoring rule exists to prevent.

## 6. Router diagnostics

Fit on the frozen calibration slice (32×512, seed 42424, 16,384 tokens), read on held-out.
Ridge `λ = 0.01 · mean(diag(XᵀX))`, per layer, ranging `16.2` (L0) to `667.0` (L1).
Pre-run diagnostic, top-133 overlap against the oracle's choice (chance = 0.5195):

| layer | overlap | mass kept, ridge | mass kept, oracle |
|---|---|---|---|
| 0 | 0.6042 | 0.7918 | 0.8901 |
| 13 | 0.6330 | 0.6759 | 0.7899 |
| 27 | 0.7404 | 0.9698 | 0.9869 |

**Overlap of 0.60–0.74 against a 0.52 chance line is a weak router, and it still retains 0.91 of
the routing value.** The two numbers only look inconsistent: keeping 79% of the mass while
agreeing on 60% of the *identities* means the groups it misses are the small ones. That is the
mechanism behind `ROUTER-HOLDS`, and it is also why `V52-STATIC` does as well as it does.

Three unregistered variants were tried on the diagnostic **before** the run: an intercept, a
`log1p` target, and `|x|` features. Two moved overlap by less than 0.02. **The third did not:
`log1p` on L27 read `0.8197` against the registered form's `0.7404`.** The run proceeded with the
registered ridge anyway, because switching a pre-registered construction on the strength of a
three-layer diagnostic is exactly the move pre-registration exists to prevent — but it is
recorded here as a real degree of freedom that was declined, and as a concrete lead for E24
rather than a footnote. Had `log1p` been registered, the composed arm might have read
differently, and I cannot say by how much because I did not run it.

## 7. The re-derivation the verdict requires — and `k = 133` was never the right depth

`ROUTER-COSTS` obliges a shallower carve. **Doing the arithmetic first shows the obligation is
already satisfied by the budget, not a concession to the result.**

The router is charged: `1536 × 256 × 28 = 11,010,048` = `11.0 M`, active every token.

| component | active weights/token |
|---|---|
| `q/o` at rank 512 (`2·D·r` per projection) | 88.1 M |
| `k/v`, untouched | 22.0 M |
| head, untouched | 233.4 M |
| **router** | **11.0 M** |
| fixed subtotal | **354.5 M** |
| FFN, full | 1156.1 M |

E18 §31's 50 tok/s budget is `0.982–1.060 G`. That leaves **`627.5–705.5 M` for the FFN**, i.e.
an activation fraction of **54.28%–61.03%**, i.e. **`k = 139 … 156` of 256.**

| `k` | activation | total | |
|---|---|---|---|
| **133** (E19/E22/E23) | 0.5195 | **0.9551 G** | **under budget** |
| 139 | 0.5430 | 0.9822 G | in budget |
| 148 | 0.5781 | 1.0228 G | in budget |
| 156 | 0.6094 | 1.0590 G | in budget |

**`k = 133` sits 2.8% *below* the budget floor.** It was inherited from E19, where it was chosen
before the router was charged to anything. So the re-derivation `ROUTER-COSTS` demands is not a
retreat to a weaker configuration — it is a move to the depth the budget always permitted, and
the depth that costs the least quality. **E24 is that sweep: the ridge router at `k ∈ {139, 148,
156}`, plus `V52-STATIC` carried forward at the same depths**, since §5 shows the static set is
most of the story and is free.

**Nothing here says the shallower carve will clear `107`.** `k = 133 → 102` is 5 short, the three
depths above add 2.4–9.9% of the FFN back, and whether that buys 5 teacher-forced tokens is
exactly what E24 measures. It is registered as an open question, not as a prediction dressed as
a plan.

## 8. What this changes elsewhere

- **`decisions/T4_HEALING_PROPOSAL.md` §2's precondition is discharged, and its §1 target is
  not.** The proposal said *"I will not ask you to launch anything until E23 reports"* and
  named a withdrawal condition: *"E23 says the real router costs most of the carve."* It does
  not — retention 0.71 is not "most". But the composed target reads `WORSE`, so **§1's
  `QO512+V52` at `k = 133` is no longer the H1/H2 target** and E24 must fix the depth first.
- **H0 is unaffected, deliberately.** H0 trains the ternary low-rank attention factors with **no
  FFN carve and no router** — a choice made in this session's H0 design precisely so that E23
  could not invalidate it, and the choice held. H0's target (`QO512-TB`, tf 28, gate ≥ 48)
  contains nothing E23 measured.
- **E19 and E22's carve numbers stay as published and are now bounded rather than open.** They
  were always labelled oracle ceilings; E23 supplies the first floor under them: `0.91` of the
  ceiling on the carve alone, `0.71` composed.
- **D0c §132–135's self-criticism is vindicated and quantified.** *"The oracle router flatters
  the carve"* — by 7 teacher-forced tokens alone and 24 composed.
- **Probe-4's hot-pool falsification does not transfer to group mass on this donor** (§5).
  `project_probe4_moe.md` already carries a retro-audit downgrade; this is a second, independent
  boundary on its scope.

## 9. What E23 cannot claim

- **No speed claim and no timing. `6.79 tok/s` stays exact.** Nothing was exported; the router's
  `11.0 M` is priced by arithmetic, not measured, and `engine.c` implements neither the router
  nor a factored matvec.
- **Linear is a floor on routers, not a ceiling.** A two-layer router, one reading the previous
  layer's activity, or one trained jointly during healing could all beat it. `ROUTER-HOLDS` on
  the carve is a positive result about *this* router; `ROUTER-COSTS` composed is a null about
  this router only.
- **One donor, one depth, one partition.** `k = 133`, `E = 256`, Qwen2.5-1.5B. E16's
  non-monotonicity in scale applies, and §7's re-derivation is arithmetic, not measurement.
- **No healing.** E23 fits a regression; it does not train the model. Everything about whether
  any of this can be *learned* is H0's question and is untouched here.
- **The static result is a hint, not a design.** `V52-STATIC` at 99 was measured as a planted
  negative, under E14 §6's rule that a post-hoc metric may not be promoted to a gate. It enters
  E24 as a registered arm, not as a conclusion.


---

## 10 — E25 discharges this probe's owed item, and prices the cut

**`engine.c` now has a factored matvec** (`probes/E25_WHAT_THE_RANK_COSTS.md`, verdict
`RANK-PAYS-WHAT-IT-WEIGHS`). §9 carried the item forward, and §7's budget re-derivation — fixed `354.5 M`, FFN allowance `627.5–705.5 M`, `k = 139…156` — charges `q/o` at `2·D·r = 88.1 M`. **E25 measures that charge and finds it correct.**

- **Correctness first.** `G-E25P`: worst relative l2 `6.445e-04` against a `2e-3` bar, top-1
  `1.0000` on 10/10, on E22's `QO512-TB` against a PyTorch reference running
  `h0_qat.TernaryLowRank`. `G-E25a`: the patched engine is **bit-identical** to the unpatched one
  on `qwen25-15b_tq.bin`.
- **The planted control.** At `r = D/2` a square projection is byte-neutral to the last weight,
  so what it loses IS the factored path's own cost: `−0.43%` at the 10 B shape and `−2.46%` at
  the 1.5 B shape, dispersions 7.3% and 6.4%. **Neither is resolvable from zero** — the second
  matvec call costs less than this box can measure.
- **The price.** At `T10` (10.60 G active, the goal's dimensions) rank-512 `q/o` reads
  **5.36 vs 4.70 tok/s = `+14.12%`**, against a byte prediction of `+12.86%`. Four `T10` arms
  whose rates differ by 15% deliver charged throughput inside a **1.54% band** — **the engine
  converts active weights into time at a rate that does not care how they are arranged**, so
  `2·D·r` is the right charge and this probe's arithmetic stands.

**What it does not touch**: every quality number in this probe. E25's timing weights are noise,
and `T10-R512` is `r/D = 1/8` where E21 validated only `r/D ≈ 1/3` at `D = 1536`.
