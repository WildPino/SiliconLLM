# E22 — does cheap compose? `FITS-BUT-NOT-IN-THE-FORMAT`

**Brief**: `briefs/BRIEF_E22_DOES_CHEAP_COMPOSE.md` @ `3f8465a`, pushed before the runner existed.
**Runner**: `benchmarks/donor_adaptation/ternary/e22_compose.py`. **Results**:
`engine/results/e22_compose.json`. Nine arms, 3793 s, `VOID: none`.
**No timing taken; nothing exported; `6.79 tok/s` exact.**

---

## 0. The finding, in two halves that point opposite ways

**Half one. The first configuration this programme has derived inside the 50 tok/s budget is also
the best-ranking modified donor it has ever measured.** `QO512+V52` — E21's activation-weighted
rank-512 `q/o` plus E19's 52% FFN carve, `0.9441 G` active against a budget of `0.982–1.060 G` —
reads **126/160 teacher-forced**, above every ternary head E20 measured, at `+0.237444` BPB.
**The registered alternative fires.**

**Half two. That artefact is fp32, and the budget that admits it was priced in ternary.** Put the
same configuration into the format it would have to ship in and it collapses: ternarizing both
factors of the rank-512 decomposition reads **28/160** teacher-forced at best construction, and
the fully assembled runnable model (`STACK`) reads **4/160**. **The budget-feasible object and the
well-ranking object are not the same object.**

---

## 1. Gates — three replications, all exact

| gate | demands | result |
|---|---|---|
| `G-S0` | `base` at 160/160 free **and** 160/160 teacher-forced, BPB `0.767595` | **FIRES** |
| `G-S1` | `QO512` reproduces **E21** to `< 1e-9`: BPB `0.8202837636996289`, free `26`, tf `144` | **FIRES**, all three |
| `G-S2` | `V52` reproduces **E19** to `< 1e-9`: BPB `0.909440994415161`, free `12`, activation `0.51953125` | **FIRES**, all three |
| `G-S3` | the verdict, both metrics | read below |
| `G-S4` | teacher-forced fires on the known-positive first | `base` 160/160 |

`VOID: none`. **Two independent prior runs, from two different runners written weeks apart,
reproduced exactly before any composed arm was read.** That is possible only because the runner
imports `e21_rank.lowrank`, `e19_carve_rank.install` and `t2_rules.r3_actsearch` rather than
reimplementing them. Free identity checks: `A·B` reproduces E21's low-rank to `1.81e-07` (`G-S5`),
and `act_rms` equals `sqrt(diag(H)/n)` to `1.91e-06`.

## 2. The table

Free-running bands unchanged since E17/E18 (floor `12`, `AT-FLOOR ≤ 14`, `RANKS ≥ 80`);
teacher-forced bands from E20 part B (`> 119` CHEAPER, `107–119` COMPARABLE, `< 107` WORSE).

| arm | BPB | Δ base | free | **tf** | mean rank | rank ≤ 5 | act | band-free | band-tf |
|---|---|---|---|---|---|---|---|---|---|
| `base` | `0.767595` | — | 160 | 160 | `1.00` | 160 | 1.0000 | — | — |
| `QO512` | `0.820284` | `+0.052689` | 26 | 144 | `1.16` | 160 | 1.0000 | PARTIAL | CHEAPER |
| `V52` | `0.909441` | `+0.141846` | 12 | **117** | `2.49` | 151 | 0.5195 | AT-FLOOR | COMPARABLE |
| **`QO512+V52`** | **`1.005039`** | **`+0.237444`** | **15** | **126** | **`2.03`** | 148 | 0.5195 | PARTIAL | **CHEAPER** |
| `QO512-T` | `5.209149` | `+4.441554` | 0 | 1 | `35801` | — | 1.0000 | AT-FLOOR | WORSE |
| `QO512-TB` | `2.812226` | `+2.044631` | 1 | 28 | `1476` | 68 | 1.0000 | AT-FLOOR | WORSE |
| `QO512-T+V52` | `5.415063` | `+4.647468` | 1 | 1 | `42617` | — | 0.5195 | AT-FLOOR | WORSE |
| `QO512-TB+V52` | `3.100314` | `+2.332719` | 0 | 12 | `9293` | — | 0.5195 | AT-FLOOR | WORSE |
| `STACK` | `3.947669` | `+3.180074` | 5 | 4 | `24829` | 13 | 0.5195 | AT-FLOOR | WORSE |

`-T` is the factored form **exactly as brief §3 registered it**; `-TB` is the same with A's column
norms folded into B's rows — see §5.

## 3. Half one: the budget configuration, read carefully

`QO512+V52` is the §2 arithmetic made real: **`0.9441 G` active ternary weights per token**
(FFN `600.6 M` at 51.953125%, `q+o` `88.1 M` at rank 512, `k+v` `22.0 M`, head `233.4 M`) against
E18's `0.982–1.060 G` for 50 tok/s — **3.9% under the low end of the budget**.

E20 §4's one-lucky-prompt trap is checked first: teacher-forced per prompt
**`[27, 24, 22, 28, 25]`**, uniform; free-running `[3, 0, 1, 7, 4]`. Median rank of the donor's
token is `1.0`, mean `2.03`, top-5 at 148 of 160. Its text is partially correct — Paris, a working
recursive Fibonacci, the right multiple-choice answer — and partially wrong (it puts water's
boiling point at 108.4 °C).

**Two things it is not.** Free-running `15/160` is only three tokens above the floor: **this model
cannot generate**, and E20 part B's lesson applies to it exactly — per-step fidelity of 79% does
not survive 32 autoregressive steps. And the carve's router is an **oracle**: it reads the true
activation mass, so `V52` and everything containing it is a **ceiling**, not a runnable number.

## 4. Additivity — three new data, and E21 was the outlier

`excess = BPB(A+B) − [BPB(A) + BPB(B) − BPB(base)]`, band `±0.020` (4·σ_seed), registered in §6.

| composition | additive prediction | measured | excess | score band | tf vs min(parts) | rank band |
|---|---|---|---|---|---|---|
| `QO512+V52` | `0.962130` | `1.005039` | **`+0.042909`** | SUPER-ADDITIVE | 126 vs 117 | RANK-**SUB**-ADDITIVE |
| `QO512-T+V52` | `5.350995` | `5.415063` | `+0.064067` | SUPER-ADDITIVE | 1 vs 1 | (both at floor) |
| `QO512-TB+V52` | `2.954072` | `3.100314` | `+0.146241` | SUPER-ADDITIVE | 12 vs 28 | RANK-SUPER-ADDITIVE |
| *E21 `BOTH-ACT-256`* | *`2.108670`* | *`1.766172`* | *`−0.342498`* | *SUB-ADDITIVE* | *48 vs 68* | *RANK-SUPER-ADDITIVE* |

**All three of E22's compositions are super-additive in BPB; E21's single datum was sub-additive,
and by seven times the margin.** The difference is what is being composed: E21 put two
activation-weighted low-rank cuts on **the same objective** (they share a basis and their errors
overlap), while E22 composes a low-rank cut with a **sparsity** cut — different mechanisms, whose
errors do not overlap and instead interact.

**So neither E21's rule nor E22's is a law.** The registered prediction (sub-additive, from E21's
one datum) missed. **Four compositions is not a law either** — E16 §9's qualifier applies to this
axis now: *one composition does not predict another, and a stack must be measured.* What the four
do establish is that **the score and rank halves can disagree in either direction**, which is why
§6 registered both.

The one clean rank reading is the verdict arm's: **126 ≥ min(144, 117)**, so the composition
ranks **better than its worse half**. That was registered as the outcome that would *not* happen.

## 5. Half two: the format destroys it, and the defect the smoke caught

**Brief §3's registered construction is degenerate, and the smoke said so before the run.**
`A = W H^½ Bᵣ` puts the entire singular-value range into A's **columns** — measured spread `5.2e2`
on layer 0's `q_proj` — while `r3_actsearch` carries only a per-**row** scale. Result:
**77.6% of A ternarizes to zero** and the product lands at relative weight error `1.5570`, *worse
than replacing W with zeros*. That is a property of where the scale was placed, not of the axis.

Folding A's column norms into B's rows is **exactly identity-preserving** (`2.3e-16`) and costs no
storage, because B's ternary format already carries a per-row scale. Zero fraction falls
`0.776 → 0.365`, relative error `1.5570 → 0.9874`. **Both arms were run and both are in §2's
table**, because repairing a registered construction quietly is how E20 run 1 went wrong.

**And the fair version still loses, decisively:**

| attention, ternary, same donor | BPB | organs converted |
|---|---|---|
| **T2b arm `A`** — dense attention, `R3` | **`1.903569`** | `q`, `k`, `v`, `o` |
| `QO512-TB` — rank-512 `q/o`, both factors `R3` | `2.812226` | `q`, `o` only (`k/v` left fp32) |

**`+0.908657` BPB worse, on a strictly easier organ set.** The mechanism is elementary and the
layer-0 microbenchmark shows it directly: dense ternary R3 on that `q_proj` has relative error
`0.8084`; the *fp32* rank-512 factorisation has `0.3515`; the two ternary factors multiply their
errors into `0.9874`. **Rank and precision do not compose on this organ — the product of two
ternary factors is worse than one ternary dense matrix, and it is worse than either ingredient.**

`STACK`, the whole thing that would actually ship (`QO512-TB` + `V52` + ternary FFN + ternary
head; `k/v` untouched, 22.0 M, 1.4% of the model, recorded as the deviation): BPB `3.947669` —
`0.122` below the chance line — free `5/160`, teacher-forced `4/160`. **Prediction 5 held.**

## 6. What this does and does not change for the goal

**The budget claim and the quality claim are about different artefacts, and that is the whole
result.** §2 charges every weight as ternary at `0.500000` B/weight to reach `0.9441 G`. The arm
that reads 126/160 is fp32. The arm that is actually ternary reads 12/160. **E22 has not produced
a model that is both inside the budget and working; it has produced a proof that the two
requirements are met by two different objects and that the gap between them is the format.**

**This is E18's conclusion arriving from a new direction and getting sharper.** E18: no rung of
the conversion ladder is both fast and good, so the model must be *trained into* the format. E22
adds the target: **the structure to train into is low-rank attention plus a carved FFN**, because
that structure is budget-feasible (`0.9441 G`) and, at fp32, is the best-ranking donor derivative
this programme has measured (126/160). What cannot be done is to *convert* into it.

**`6.79 tok/s` unchanged.** Nothing was exported, no timing was taken, `engine.c` still has no
factored matvec, `QWENDON1` has no kind for one, and the carve still needs a real router — E22
used an oracle.

**The 7 B is untouched by any of this.** §2 said so before the run: the same two cuts leave the
Coder-7B at `4.090 G`, 4.0× the budget, and once `q/o` is cut the budget permits **negative** FFN.
The 1.5 B fits because its head is `233 M`; the 7 B does not because its head is `545 M`. **The
head remains the named binding constraint** (E21 §7).

## 7. Predictions — three held, two missed

| # | brief §7 | outcome |
|---|---|---|
| 1 | `G-S0`/`G-S1`/`G-S2` fire, both anchors exact | **held** — three replications, all exact |
| 2 | `QO512+V52` SUB-ADDITIVE in BPB | **missed** — `+0.042909`, SUPER-ADDITIVE |
| 3 | `QO512+V52` RANK-SUPER-ADDITIVE | **missed** — 126 ≥ 117, RANK-SUB-ADDITIVE |
| 4 | `V52` teacher-forced ≥ 107 | **held** — `117`, COMPARABLE |
| 5 | `STACK` AT-FLOOR free and tf < 107 | **held** — `5` free, `4` tf |

**Prediction 4 was registered as a conflict between two published laws and BPB ordering won.**
`V52` costs `+0.141846`, less than E20's best ternary head at `+0.170414`, and it reads `117`
against that head's `117` — the ordering held to the token. **E21 §4a's "BPB does not order
interventions across axes" is therefore too general as written**: it was derived from one pair
(`H-ACT-256` vs `R0H`, 68 vs 107 at equal BPB) and this pair contradicts it. The narrower
statement that survives both: *BPB ordering can fail across axes, so it must be checked, not
assumed* — which is weaker than what E21 §4a claims and the claim is corrected there.

**Both misses are in the favourable direction.** The composition is better in ranking than
registered, and that is worth more than being right would have been: the arm that carries the
verdict was predicted to fail and did not.

## 8. Owed

1. **Healing / QAT on `QO512+V52`.** First owed item since E18 §9, and now fully specified for
   the first time: a **1.5 B, budget-feasible at `0.9441 G`, right 79% of the time per step**,
   whose defect is free-running drift (15/160) and whose format conversion is the thing that must
   be *learned* rather than applied. **Needs GPU; the user launches it, and has offered T4 weeks.**
   §5 says what the training must absorb: the factored form must be ternary-aware *during*
   training, because post-hoc it reads 28/160.
2. **A real router for the carve.** Every `V52`-containing number here is an oracle ceiling. D0c
   builds routers; none has been read in ranking.
3. **`k/v` in the factored arms** — left fp32 throughout §5, 22.0 M, and the comparison against
   T2b arm `A` is favourable to E22 because of it. A like-for-like repeat would only widen the gap.
4. **A factored matvec in `engine.c` and a kind in `QWENDON1`** — unchanged from E21 §8; without
   it no rank result becomes tok/s.
5. **The head** — unchanged, and now the only organ neither cut touches. `545 M` on a 7 B.
6. **The rank fraction at scale**, `R5` in the exporter, and the E14/E19 items — unchanged.

## 9. Appended 2026-09-11 after E23 — §6's additivity reading was ORACLE-CONDITIONED

`probes/E23_A_REAL_ROUTER.md` replaced the oracle router with a closed-form ridge regression and
re-read both composed arms. **The composition reverses sign.**

| | `V52` | `QO512+V52` | composition |
|---|---|---|---|
| **oracle** (this probe) | 117 | 126 | **+9** |
| **ridge** (E23) | 110 | 102 | **−8** |

§5 read `QO512+V52` as RANK-SUB-ADDITIVE — `126 ≥ min(144, 117)`, better than either part
predicted — and §6 built on it. **That reading holds only under an oracle.** The oracle scores
expert groups from the *true* post-gate activations, so damage done upstream by the rank-512
attention cut cannot reach its decision; a real router reads the **block input**, which that same
cut has already perturbed. **Under an oracle the two cuts are independent; under a real router
they are coupled through the router's input.** Mean rank locates the coupling: `2.03 → 5.15` on
the composed arm against `2.49 → 4.17` on the carve alone.

**What is withdrawn and what is not.** The measurements in §§4–5 stand exactly as published —
they were labelled oracle numbers throughout and E23's `G-T1` reproduces both of them to the
digit. What is withdrawn is any reading of them as a statement about *composition in a buildable
model*: `+0.042909`, `+0.064067`, `+0.146241` are oracle-conditioned super-additivity, and the
sign of the ranking half is not robust to making the router real.

**§7's prediction 4 is unaffected** — it was about BPB-versus-ranking disagreement within this
probe's own arms, not about routers.

**And §5's own warning is reinforced rather than replaced.** It said four compositions do not
make a law and a stack must be measured on both metrics. There are now five, they disagree, and
E23 adds the reason they disagree: **the measuring apparatus was part of the result.**

**Owed item 1 (a real router) is discharged.** The answer is split: the carve alone survives at
retention `0.9067` (`ROUTER-HOLDS`), the composed configuration at `0.7143` (`ROUTER-COSTS`), and
`QO512+V52` at `k = 133` is therefore **no longer the T4 healing target**. E23 §7 re-derives the
depth the budget permits once the router is charged — `k = 139 … 156`, where `k = 133` sits 2.8%
*under* the floor — and that sweep is E24.


---

## 10 — E25 discharges this probe's owed item, and prices the cut

**`engine.c` now has a factored matvec** (`probes/E25_WHAT_THE_RANK_COSTS.md`, verdict
`RANK-PAYS-WHAT-IT-WEIGHS`). §8 carried the item forward. **The `0.9441 G` this probe derived is now a charge the engine honours**: the rank term is paid at `2·D·r` and nothing is lost to the extra call.

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
