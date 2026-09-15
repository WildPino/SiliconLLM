# H1 — the carve TRAINED rather than applied: `G-H1` PASSES at both checkpoints, band `TRAINING-HELPS`

**Brief:** `briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md` + addenda A–N
**Trainer:** `s1/h1_qat.py` (T4, fp16) · **Gate:** `s1/h1_eval.py` (CPU fp32, frozen slice)
**Results:** `s1/results/h1/h1_eval_h1_s2_cum390.json`, `h1_eval_h1_s1_mid195.json`,
`h1_trained_s{1,2}.json`, `h1-qat-run-session-2.log`
**Donor:** Qwen2.5-1.5B rev `8faed761` · 8 of 28 layers (3,6,9,12,15,18,21,24) · `k=16` of `E=256`
**Slice:** 24×512, 51,870 scored bytes, `ids_sha256 a1a48dc9…`, `B/token 4.22945205479452`
**Cost:** 2 × 2.8 GPU-h on T4 (the registered budget, spent in full) + 2 × ~1,200 s CPU fp32

## 0. The verdict

> **`G-H1` PASSES at both checkpoints.** Trained BPB **0.962593** < `applied-8L` **1.096636**,
> delta **−0.134043**. Ordinal, no tolerance. **Band: `TRAINING-HELPS`.**
>
> **Applying the carve and training with it are not the same thing, and the difference is
> measured.** At ~390 optimizer steps the trained carve is **0.014742** short of
> `CARVE-IS-TRAINABLE`.

Per the reading rule registered in **addendum J.5 / K.3 before either number existed**:
`TRAINING-HELPS` is an **achievement** band and is read exactly as registered, *a fortiori*.
The rule's restriction bites only on `CARVE-NOT-TRAINABLE`, which is not the outcome.

## 1. Every control fired, on the gate instrument too, at both checkpoints

| control | s1 (~195) | s2 (~390) | |
|---|---|---|---|
| `intact` vs anchor 0.767595 | **0.767595** | **0.767595** | FIRES — the fp32 instrument reproduces the dense donor |
| `h0-run3` vs anchor 0.810022 | **0.810022** | **0.810022** | reproduced, 56 organs over 28 layers |
| `G-H1a` k=E, HARD gate, any router | max\|d\| **0.0** | max\|d\| **0.0** | FIRES |
| `G-H1a` k=E, SOFT gate, router=0 | max\|d\| **0.0** | max\|d\| **0.0** | FIRES |
| `carve actually masks at k=16` | max\|d\| **20.0** | max\|d\| **13.96** | FIRES — discrimination |
| slice hash `a1a48dc9…` | matched | matched | the frozen instrument |

`G-H1a` at exactly 0.0 paired with `carve_is_live` at 14–20 is the pair that earns the nulls:
the mask is provably inert where it should be and provably active where it should be. On the
T4 side all five planted controls fired in both sessions as well (`h1_trained_s*.json`).

## 2. The two-point curve, and the honest limits on reading it

| | steps | `trained-8L` | Δ from `applied-8L` | bought by that session |
|---|---|---|---|---|
| `applied-8L` | 0 | 1.096636 | — | — |
| **s1** | ~195 | **0.983337** | −0.113299 | −0.113299 |
| **s2** | ~390 | **0.962593** | −0.134043 | **−0.020744** |

The second session bought **18.3%** of the first. **No curve is fitted to this, and the
increments are not comparable**, for a reason recorded in the artefacts: both sessions carry
`adam_state_restarted: true`. Session 1 starts from the applied carve with a fresh optimizer;
session 2 starts from session 1's weights **with the optimizer state thrown away**. Part of the
shortfall is restart transient and part may be genuine flattening, and **two points cannot
separate them.** `feedback_rank_partner_survives_refusal` applies directly: *a fit on two
adjacent points has no exponent, it has an artefact.*

**Session 2's −0.020744 is the only "steady-state" increment that exists** — the only one
measured after a restart, which is the shape every further session will have.

## 3. The decomposition — and the router is the term still accelerating

`h1_eval.py` separates the two things §6 of the brief said it could not, by holding the trained
experts fixed and swapping only the router (`arm-E` uses E37's fitted router, `arm-ER` the
trained one):

| term | s1 (~195) | s2 (~390) | change |
|---|---|---|---|
| experts trained | −0.115183 | −0.128172 | **−0.012989** |
| **router trained** (vs E37's router) | **+0.001884** | **−0.005871** | **−0.007755** |
| soft vs hard gate (engine discards) | −0.003950 | −0.005298 | — |
| `arm-E` (trained experts, E37 router) | 0.981453 | 0.968464 | |

**The trained router starts HARMFUL and becomes helpful.** At ~195 steps E37's fitted router
beat the jointly trained one; by ~390 the sign has flipped. In session 2 the experts added
−0.012989 and the router −0.007755 — so the router is **37% of the second session's movement
against −1.7% of the first's.**

**The experts are flattening. The router is not.** That is the single most decision-relevant
thing in this probe, and it is the term that carries the MoE question.

**§6's "the router's contribution is not separable from the experts'" is falsified in
practice** — `arm-E` prices it directly, additively, residual 0.0.

## 4. `G-H1e` FIRES at both checkpoints — and the brief predicted it would FAIL

Addendum E.5, written before the hours were spent:

> *"Registered prediction, before the T4 hours are spent: on this evidence I expect `G-H1e` to
> FAIL… so that a pass counts for something: it would mean the joint regime does what the
> frozen-expert proxy says it cannot, which is a result about MoE on this branch and not a
> hyper-parameter."*

Measured on CPU fp32, **both arms HARD-gated** (addendum F's repair), against STATIC = top-k
chosen once by global activation mass over the calibration stream:

| | router nats/token | STATIC nats/token | margin |
|---|---|---|---|
| s1 (~195) | 2.882782 | 3.082401 | **0.199619** |
| s2 (~390) | 2.821969 | 3.077081 | **0.255112** |

**FIRES at both, and the margin GROWS with training.** The registered prediction is
**FALSIFIED, in the favourable direction**, and by the brief's own words that makes it a result
about joint training: **a per-token learned router beats a fixed selection end-to-end on the
donor branch.**

**Two comparators, and they must not be conflated.** `G-H1e` asks *per-token routing vs a FIXED
selection* — an easy bar, cleared already at 195 steps. §3's router term asks *trained router vs
E37's FITTED router* — a hard bar, not cleared until somewhere between 195 and 390 steps. Both
are true and they say different things: beating static is early, beating a good fitted router
takes training.

The frozen-expert smoke (addendum E, 2 of 8 layers, +2.27% against STATIC) was a **structurally
blind proxy** — it could not see the mechanism that matters, which is that the router changes
*which experts receive gradient*. E.5 said exactly that and still predicted failure; the
prediction was wrong and the reasoning attached to it was right.

**This is the first evidence on the donor branch for a jointly-trained router** — what
`project_moe_status` records as never having been tried here. It is modest (−0.005871 BPB, 4.4%
of the gate's movement), one donor, 8 layers, one setting. **It is not a MoE result at 10 B.**

## 5. Predictions scored

| # | registered | outcome |
|---|---|---|
| 1 | `applied-8L` lands 3.10–3.55 | **MISSED** — 1.096636. Scored in addendum C, where the band table had to be repaired because of it. |
| 2 | `G-H1` fires: trained beats applied | **TAKEN** at both checkpoints |
| 3 | lands in `TRAINING-HELPS`, not `CARVE-IS-TRAINABLE` | **TAKEN post-repair, unscoreable pre-repair** — see below |
| 4 | free-running stays `AT-FLOOR`/`PARTIAL` (≤40/160) | **NOT MEASURED** — `h1_eval.py` has no generation arm. **OWED.** |
| 5 | router beats `k/E` recall; contribution **not separable** | second clause **FALSIFIED** — §3 separates it |
| 6 | weakest: that 8 of 28 trained layers move whole-model BPB at all | **TAKEN** — −0.134043 |
| E.5 | `G-H1e` FAILS | **FALSIFIED**, favourable direction, at both checkpoints |

**Caveat on prediction 3, stated rather than banked.** It was written against §5.2's *original*
band table, in which `TRAINING-HELPS` was **empty** (addendum C.3: `3.475707 ≤ BPB < 1.096636`).
A prediction naming an empty band cannot be scored as written. It is scored against the
**repaired** table — registered before the run — but the honest reading is **taken post-repair,
unscoreable pre-repair**, not a clean hit.

**Two of seven registered predictions were falsified in the FAVOURABLE direction** (5's second
clause, E.5). That is worth naming: this brief's errors ran pessimistic, which is the
comfortable direction to be wrong in and therefore the one to watch.

## 6. What this does NOT say

* **Nothing about 10 B.** 1.5 B, 8 of 28 layers, exactly as §9 registered.
* **Nothing about speed.** No timing taken, nothing exported. **`G-E63d` is still `VOID` and
  OWED.**
* **Nothing about the head or `STACK`** — untouched; E43 measured the head as the largest single
  term of the floor.
* **It does not separate ternarisation from carving**, and this is the sharpest limit.
  `applied-8L` and both trained arms sit on a **ternary** FFN. §62.12 prices the ternary format
  at **82%** of the dense→chance damage and the carve at **17%** — so **the axis H1 just proved
  trainable is the smaller of the two.** The `k=E` trained arm that would separate them was
  registered in §9 as not fitting the budget, and still does not.
* **It is not a completed training curve** — two points, one restart, no exponent.

## 7. What is owed, and the T4 ask that follows from the curve

1. **ONE further T4 session (2.8 h), not four.** Addendum K sized four sessions against H0's
   1000-step mark *before* the curve existed. With the curve in hand that is buying hours
   against an unread trend. One session is decisive either way:
   * it is the **second post-restart increment**, directly comparable to session 2's
     −0.020744, so two comparable increments finally say whether the curve is decaying;
   * the gap to `CARVE-IS-TRAINABLE` is **0.014742**, *less than one session-2 increment*, so a
     non-decaying curve **crosses the band**;
   * and the router — the term still accelerating, and the MoE question — gets a third point.
2. **Free-running (prediction 4) is unmeasured.** `h1_eval.py` has no generation arm; it is the
   axis that has never recovered in any arm at any scale, and it is owed.
3. **`steps_completed`, `stop_reason`, `seconds_per_step` in the trainer's JSON** (addendum
   K.5). Session 1's step count is unrecoverable because its log is 0 bytes.
4. **The ternary/carve separation** — the `k=E` trained arm — remains unaffordable and remains
   the reason H1's result, while real, prices the smaller of the two damage axes.

## 8. Session 3 is provenance-complete and READY, but has not run

Addendum M found that the two 2.8-hour sessions were capped by our own default, not Kaggle's
12-hour limit, and registered one **11-hour** continuation from S2. Addendum N then audited the
forgotten untracked packer before use: its first draft would have hashed `RUN.md` before changing
it, could reuse a same-size wrong checkpoint, and misstated the avoided Adam restarts.

The repaired packer was committed at `1a914ba`, its read-only full-input check passed, and the
bundle was rebuilt once. Independent re-hashing confirms all nine manifest entries. The resume
checkpoint is 1,334,711,974 bytes, sha256
`4030d2af63aed924d7e17559bc3dc84ba8db519056c389cdb068b38b58a7e3b3`; the final manifest is
sha256 `bae2701ac8091b79a0f690298d0aa3ddfd3bad255f714dbe5aa05b678bbea0ee`.
Machine-readable audit: `s1/results/h1/h1_s3_bundle_audit.json`.

**Status: `READY_NOT_RUN`.** The command is frozen in `_h1_bundle/RUN.md`: seed 3141,
`--max-hours 11.0`, resume S2, expected ~765 steps at the measured 51.7 s/step. The first
periodic save at step 250 is itself a registered planted check because neither earlier session
reached it. The scientific predictions and limits remain addendum M's; bundle construction is
not a result.

This session remains useful for optimizer continuity and the trained-router trajectory, but it
is **not the one-byte deliverable**. E64/E66 make the subsequent branch explicit: do not launch
the old `H2T` proposal as written to heal ternary-format damage that the one-byte rung avoids.
The next new healing apparatus must keep one-byte weights and train the carve/routing damage.

---

## 9. Canonical S3 terminal result — COMPLETE / PASS / `CARVE-IS-TRAINABLE`

The canonical records are `s1/results/h1/h1_eval_h1_s3.json` and
`s1/results/h1/h1_eval_h1_s3_adjudication.json`. The write-once guard returned **PASS**; the
unchanged CPU-fp32 evaluator returned **0**. Its final HARD-gate reading is
**0.9234896239116439**, below `applied-8L` **1.0966361325809948** by
**−0.17314650866935088**, crossing the preregistered `CARVE-IS-TRAINABLE` band
`0.810022 < BPB < 0.947851`.

This is a longer joint-training result, not a replacement of S2: S2's canonical
**0.962592908257438** stands, while S3 improves it to **0.9234896239116439**. The final
decomposition is experts **−0.14794307283419295**, trained-router **−0.02520343583515794** and
gate-form diagnostic **−0.0013456311137338695**. Experts account for most recovery; the trained
router remains material, and `G-H1e` fires (**2.707332441530957** versus STATIC
**3.053102243577019** nats/token).

Operationally, S3 completed 765 time-capped updates in 39,619.556 s at 51.79027 s/step, with
zero non-finite microbatches and all frozen controls passing. Both registered terminal predictions
that name hard observables are met: at least 700 steps with `time-cap`, and HARD BPB below
0.947851. Relative to canonical S2 (HARD 0.9625929082574385; experts
−0.12817234545341805; router −0.005870878870138263), S3 adds −0.0391032843457946: experts add
−0.0197707273807749 and router −0.0193325569650197. Thus the router supplies **49.4397%** of the
S2→S3 recovery, exceeding the registered 37%: prediction M.5.3 **PASS**. Its `G-H1e` margin is
0.345769802046062, above S2's 0.2551122115529377: M.5.4 **PASS**. M.5.5 is **not independently
adjudicated / unavailable**: the final-pair history records a step-250 diagnostic, not a
canonical preserved/loadable checkpoint; no checkpoint is selected or inferred. The SOFT
diagnostic is 0.92214399279791 (SOFT minus HARD −0.0013456311137338695); it is not the gate.

The Kaggle CLI character-map error occurred only after the final NPZ and JSON had both downloaded.
It neither changes this result nor authorizes a re-download. Scope remains ternary, eight layers,
and H1 only: there is no rank, rate, one-byte, 10B, scale, export, or post-hoc-carve claim.
