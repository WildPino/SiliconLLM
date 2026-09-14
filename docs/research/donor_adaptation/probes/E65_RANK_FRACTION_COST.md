# E65 — `R128`'s rank fraction costs HALF the distance to chance, post-hoc

**Brief:** `briefs/BRIEF_E65_WHAT_DOES_R128S_RANK_FRACTION_COST.md` + addendum A
**Runner:** `ternary/e27_floor.py` (E27's, one arm added, `E27_OUT` override)
**Result:** `engine/results/e65_rank_fraction.json` (run 2; run 1 was malformed, addendum A)
**Donor:** Qwen2.5-1.5B, `D = 1536` · frozen slice 24×512, 51,870 scored bytes, `ids_sha256` OK
**Anchors:** dense **0.767595**, chance **4.069819**, gap **3.302224** · **wall 1,364 s**
**QUALITY ONLY. No rate measured. `G-E63d` is `VOID` and OWED and E65 did not touch it.**

## 0. The verdict

> **At `r/D = 1/32` — the fraction `A10B-R128` uses to reach ~113–130 tok/s — an
> activation-weighted low-rank projection of a real donor reads BPB **2.473430**: **+1.705835
> over dense, 51.66% of the entire distance from dense to chance.**
>
> **Over half the model's usable signal, on the axis that buys every tok/s above 50.**

## 1. Controls — all three fire, on recomputation

Run 1's controls were tautological (addendum A). Run 2 recomputes every arm:

| arm | recomputed BPB | anchor | gate |
|---|---|---|---|
| `base` | 0.767594964120 | 0.7675949641196624 | **`G_F0` FIRES** |
| `QO-512` | 0.820283763700 | E21's `QO-ACT-512` 0.8202837636996289 | **`G_F2` FIRES** (\|d\| ≤ 1e-9) |
| `QO-192` | 1.856378431038 | E27's published 1.8563784310382423 | **exact** |

`VOID: none`. The instrument reproduces two independent prior experiments to 1e-9 before the
new cell is read.

## 2. The ladder

| `r/D` | rank | BPB | Δ vs dense | share of dense→chance gap | added by this halving | tf | source |
|---|---|---|---|---|---|---|---|
| dense | — | 0.767595 | — | — | — | 160/160 | control |
| 1/3 | 512 | 0.820284 | +0.052689 | 1.60% | — | 144 | E21 / replicated here |
| 1/6 | 256 | 1.545880 | +0.778285 | 23.57% | **+21.97 pts** | 93 | E21 |
| 1/8 | 192 | 1.856378 | +1.088783 | 32.97% | (×0.75, not a halving) +9.40 | 56 | E27 / replicated here |
| 1/16 | 96 | 2.097275 | +1.329680 | 40.27% | **+7.29 pts** | 42 | E27 |
| **1/32** | **48** | **2.473430** | **+1.705835** | **51.66%** | **+11.39 pts** | **33** | **E65, new** |

**`G-E65c` — the ladder RANKS on both metrics.** BPB is monotone in `r/D`; teacher-forced top-1
is monotone (144 → 93 → 56 → 42 → 33). The SCORE and its RANK partner agree, which is not
guaranteed — E62's finding was the opposite.

**Free-running is NOT usable below `1/8`**: 26 → 4 → 1 → 4. It is pinned at the floor and its
non-monotonicity at the bottom is a saturated counter, not a signal
(`feedback_gate_is_not_a_progress_meter`). It is reported and not read.

## 3. The damage per halving is NON-MONOTONE, and that forbids an exponent

**+21.97, then +7.29, then +11.39.** The curve decelerated and then **re-accelerated**.

This matters more than the headline number, because it is the thing that decides whether the
`D = 1536` ladder may be carried to `D = 4096` at all:

> **§3 — the rank-damage curve has no exponent. Its second difference changes sign inside the
> measured range, so no smooth law fits it, and no extrapolation across width is licensed by
> this data.**

That is `feedback_rank_partner_survives_refusal` in its E62 form, one level down: *a fit on two
adjacent points has no exponent, it has an artefact* — here even five points do not supply one.
E21 §8 and E16's non-monotonicity said "do not extrapolate `r/D` across width" as a caution;
**E65 supplies the direct evidence.**

**This cuts both ways and is stated as such.** `D = 4096` has more redundancy per direction and
could tolerate `1/32` better than `D = 1536` does. E65 cannot say it will not. What E65 kills is
the idea that a curve fitted at one width predicts the other.

## 4. What this does to the 10 B speed result

| shape | rank | `r/D` | rate | post-hoc same-fraction analogue at `D=1536` |
|---|---|---|---|---|
| `A10B-K3` | **none** | — | ~49 tok/s (composite, `G-E63d` OWED) | **0%** — no rank |
| `A10B-R512` | 512 | 1/8 | ~93 floor | 32.97% of the gap |
| `A10B-NKV2` | 512 | 1/8 | ~127 floor | 32.97% of the gap |
| **`A10B-R128`** | 128 | **1/32** | **~113–130 measured, ~163 floor** | **51.66% of the gap** |

**The only rank-free shape on the ladder is the one that sits AT the target rather than above
it.** Everything above 50 tok/s is bought on an axis whose post-hoc cost, at the fraction
actually used, is over half the distance to chance.

**E40 is not retracted and was never wrong.** Its rates are correct for what it measured, and
its §6 says plainly *"Nothing about quality. Synthetic weights."* E65 prices the quantity E40
declined to price. What changes is the **reading** of the speed ladder, not its numbers:
`~113–130 tok/s` is a statement about a shape, and this programme now knows what that shape's
rank fraction costs when it is filled with real weights by projection.

## 5. Why this is a FLOOR and not a verdict, and the number that matters next

**H1, closed today, is the reason.** Applying the carve reads `1.096636`; training with it reads
`0.962593`. H0 showed the same for rank at the gentle fraction: post-hoc `0.820284` → trained
`0.810022`. **Post-hoc conversion and training are different objects, and this programme has
measured that twice.**

**Nobody has ever trained at an aggressive rank fraction.** So E65 does not close `R128`. It
sizes the hole:

| | damage to be recovered |
|---|---|
| the applied carve H1 was working against | **0.329041** |
| **`R128`'s rank fraction, post-hoc** | **1.705835** |
| ratio | **5.18×** |

**The hole on the rank axis at `R128`'s fraction is 5.18× the one H1 just showed training can
partially fill.** H1 recovered 40.7% of its hole in ~390 steps on 8 of 28 layers. **No transfer
between axes is claimed** — different organ, different mechanism, and §3 has just shown this
curve does not even admit a fit within its own axis. The 5.18× is a **scale statement**, and it
is the quantity any future "train the rank" proposal must be argued against.

## 6. Predictions scored

| # | registered | outcome |
|---|---|---|
| 1 | `QO-48` lands in **2.10–2.60** | **TAKEN on the number** (2.473430) — **but the stated REASONING was WRONG**: I argued "the ladder decelerates, so 1/32 adds less again"; it added **more** (+11.39 vs +7.29). Right answer, wrong mechanism. |
| 2 | stays below chance 4.069819 | **TAKEN** (2.473430) |
| 3 | `QO-48` tf below `QO-96`'s | **TAKEN** (33 < 42) |
| 4 | ladder monotone in `r/D` on BPB | **TAKEN** |
| 5 | `G-E65a` fires on all three | **TAKEN after the repair** — in run 1 it was UNEVALUATED (addendum A), not failed |

**Five of five taken, and that is a reason for suspicion rather than satisfaction.** Prediction
1's band was 0.5 wide on a ladder spanning 1.65 — not heroic — and its mechanism was falsified
even though its number landed. Predictions 2–4 were close to safe. **The only prediction that
did real work was the one whose reasoning turned out to be wrong.**

## 7. What E65 may NOT conclude

1. **Nothing about 10 B directly.** `D = 1536`, not 4096, and **§3 has now shown the curve
   admits no exponent**, so the `r/D` analogue is an analogue and not a projection. Every
   sentence above that uses it says so.
2. **Nothing about a TRAINED low-rank at this fraction** — the actual open question, and §5 is
   the size of it.
3. **No rate.** `G-E63d` stays `VOID` and OWED.
4. **No retraction of E40**, and no claim that `R128` is dead.
5. **Nothing about `k/v`** — `KV-96` is E27's and was not re-run here; the rank axis on `k/v`
   remains E27's business.
