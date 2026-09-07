# E12 — the exporter's default rule is at chance at every scale tested. `CHANCE-LINE`

**Verdict: `CHANCE-LINE`.** The pre-registered question — does the ternarization cost shrink with
donor scale? — **is not answered**, and the reason is neither the instrument nor the donors. **Every
ternarized arm, at every cell, scores worse than a uniform distribution over the vocabulary.** Above
that line BPB no longer orders by damage, so the pre-registered ratio has nothing to measure.

Pre-registered and pushed as `briefs/BRIEF_E12_TERNARY_COST_VS_SCALE.md` (`11a9d83`); runner
`ternary/e12_scale.py` (`87ba957`); §9 amendment and the Z-dispersion diagnostic `ae403c2`,
`ternary/e12_zvar.py`. Sweep 2161 s, 24×512 heldout slice, all layers, 6 threads.

**This supersedes the first write-up of E12, which returned `CONTROL-FAILED`. That verdict is
withdrawn — it rested on a control this programme had already retired. See §4.**

---

## 1. What was measured

The **chance line** is `log2(V) / (bytes per token)` = `log2(151936) / 4.229452` = **4.069819 BPB**:
what a model scores by assigning every token equal probability. Using the padded config vocab is the
conservative choice — on the 151,665 tokens the tokenizer can actually emit it is 4.069210, and
nothing below turns on the difference.

| donor | BPB fp32 | BPB ternary (FA) | **dBPB** | FA vs chance line |
|---|---|---|---|---|
| 0.5 B | 0.871795121 | 4.587452739 | **3.715657618** | **+0.518 above** |
| 1.5 B | 0.767594958 | 5.505834291 | **4.738239332** | **+1.436 above** |
| 3 B | 0.724449797 | 5.734699003 | **5.010249206** | **+1.665 above** |

Every donor's fp32 baseline is far **below** the line (0.72–0.87), so the donors and the eval slice
are fine. **It is the conversion that puts them past it — at all three scales.**

Each donor is measured against **its own** fp32 baseline; `t1_ternarize.BASELINE_STANDING` is a
1.5 B constant and was deliberately not reused. The rule is **imported** from
`t1_ternarize.ternarize`, the same oracle `qwen_export.py` calls — and `qwen_export.py`'s `--rule`
defaults to **`R0`**, so this is the conversion the exporter actually ships.

**Mechanically `r = dBPB(3B)/dBPB(0.5B) = 1.3484`, which the brief's §4 would read `COST-GROWS`.
It is not reported as a result** — §2 says why.

## 2. Why the ratio cannot be read

A model above the chance line is not merely damaged: it assigns the truth **less** probability than
knowing nothing would. Differences between two such models measure how confidently wrong each one
is, and there is no reason for that to order by amount of damage.

Two orderings in this data invert, and **both inversions live entirely above the line**:

| | 0.5 B | 1.5 B | 3 B |
|---|---|---|---|
| `F` (FFN only) | 4.546298 | 4.076694 | **5.928395** |
| `FA` (FFN **and** attention) | 4.587453 | 5.505834 | **5.734699** |
| `F < FA`? — more organs, more damage | yes | yes | **no** |

At 3 B, converting **more** organs did **less** damage by 0.194 BPB. `F` and `FA` are both
deterministic, so this is not noise in the arms — it is the estimand.

**The ratio `r` is a ratio of distances measured in a regime where distance has stopped meaning
damage.** That is the finding, and it is a stronger statement than either pre-registered outcome:
`COST-SHRINKS` and `COST-GROWS` both presuppose that dBPB measures cost, and at these settings it
does not.

## 3. What E12 does establish

1. **The exporter's default rule (`R0`) destroys every donor tested, and worse as scale grows.**
   0.5 B lands +0.518 past chance, 1.5 B +1.436, 3 B +1.665. The direction is consistent even though
   the magnitudes are not interpretable as costs. **No donor at any tested scale survives `R0`.**
2. **The instrument is sound.** `I` (identity substitution through the same code path) returns
   `I − base = +0.000e+00` **exactly at all three cells**, over 168 / 196 / 252 substituted tensors.
   Substitution counts are structurally correct (7 per layer for FA, 3 for F, at 24 / 28 / 36
   layers). Tokenizers are verified identical (**1 distinct fingerprint**) — necessary because the
   eval slice is disk-cached and **not** keyed by model. The fp32 baselines fall monotonically with
   size. **The BPB numbers are trustworthy as measurements.**
3. **T2's headline re-reads, and one of its conclusions does not survive.** Placing T2's own §3 table
   against the chance line:

   | T2 arm | BPB | vs chance line |
   |---|---|---|
   | **R0 — what the exporter ships** | 4.076694 | **+0.007 ABOVE** |
   | Z (random signs) | 4.140276 | **+0.070 ABOVE** |
   | R4 (GPTQ on R0's grid) | 4.299819 | **+0.230 ABOVE** |
   | R1 (TWN) | 3.851979 | −0.218 below |
   | R2 (scale search) | 3.390467 | −0.679 below |
   | **R3 (activation-weighted)** | 2.476967 | **−1.593 below** |
   | **R5 (R3 + GPTQ)** | 2.027495 | **−2.042 below** |

   **T2 §4(a) concluded that "BitLinear158's choice of which sign carries no measurable information
   about the donor".** That reading does not hold. `R0 − Z = −0.064 ± 0.126` is a contrast between
   **two points both pinned at chance**, and no contrast between two such points can resolve
   anything. The signs are `sign(w)` under *every* rule in that table — a positive per-row scale
   cannot change them — and R3/R5 carry the identical signs to **1.6–2.0 BPB below the line**. The
   defensible statement is narrower: **at R0's scale the converted model is at chance, so nothing
   measured against it resolves.** T2's `RULE-HELPS` verdict is untouched; what changes is the
   mechanism §4(a) claimed.

   **This also says what T2's 62% actually bought**: not a cheaper conversion of the same kind, but
   the difference between a model at chance and a model that predicts.

## 4. The control that was retired, and the one that passed

**The first write-up of E12 returned `CONTROL-FAILED`, and that was my error.** Arm `Z`
(`random_sign`, same organs as `F`) was adopted as E12's planted control, and read as failing at
3 B because `Z − F = −1.490`. Three things were available before the run and I did not carry them in:

1. **T2 had already retired arm `Z` as a control**, in terms: *"it is why arm Z was the wrong
   control: the brief assumed Z would be far worse than the treatment, and it is not worse at all"*
   (`T2_TERNARIZATION_RULE.md` §4(a)).
2. **The same amendment, dated 2026-09-04, is in the docstring of `t1_ternarize.ternarize` — the
   function E12 imports** — recording that `Z` tested a scientific claim rather than an instrument
   property, and that **arm `I` is the control that replaced it**. `t1_ternarize.ARMS` labels its own
   `Z` row `"PLANTED CONTROL (mis-specified, see report)"`.
3. **E12's `Z` does not reproduce T2's `Z` at the shared cell.** Same slice (`ids_sha256
   a1a48dc9…`), same revision, same organs, same rule:

   | source | `F` / `R0` | `Z` |
   |---|---|---|
   | T2 (`t2_rules.py`, seeds `1000 + stats["n"]`) | **+3.309099** | **+3.372681** |
   | E12 (`t1_ternarize.apply_arm`, seeds `1000 + rng`) | **+3.309099** | **+4.001257** |

   **`F` reproduces to six decimals**, so harness, slice and rule are identical. An **off-by-one in
   which per-tensor seed lands on which tensor** moves `Z` by **0.628 BPB = 126 σ_seed**.

**Arm `I` — the control this programme actually sanctions — passed exactly, at every cell.**

### 4.1 The dispersion, measured

Pre-registered in `BRIEF_E12` §9 with the band fixed before the run (`ae403c2`), predicting
`Z-UNSTABLE`. `ternary/e12_zvar.py`, K = 5 draws of arm `Z` per cell, `seed_base` ∈ {1000 … 5000},
everything else held. **G-Z0 replication gate**: `seed_base = 1000` must reproduce
`e12_scale.json`'s `dBPB_Z` to the last digit.

**`ternary/e12_zvar.py`, 2351 s.** `dBPB` of arm `Z`, five draws per cell:

| cell | `Z` draws (dBPB), 5 seed bases | spread | mean | `F` | draws with **`Z` < `F`** (control fails) |
|---|---|---|---|---|---|
| 0.5 B | 3.601252 … 4.465219 | 0.863967 | 3.891365 | 3.674503 | **2 / 5** |
| 1.5 B | 3.463229 … 4.042358 | 0.579129 | 3.793314 | 3.309099 | 0 / 5 |
| **3 B** | 3.714379 … 5.410317 | **1.695938** | 4.524840 | 5.203945 | **4 / 5** |

**G-Z0 passed at all three cells** — `seed_base = 1000` reproduced `e12_scale.json`'s `dBPB_Z` to the
last digit (3.714379009 at 3 B), so the `seed_base` parameter left the default path untouched.

**G-Z1: `spread(3B) = 1.695938 ≥ 1.490` → `Z-UNSTABLE`. The §9.3 prediction landed** — the third
band in this programme to contain its own result, after E9 and E13, and derived the same way.

Three readings, in order of what they license:

1. **The 3 B spread (1.696) is larger than the entire `Z − F` gap (1.490) that `CONTROL-FAILED` was
   read off.** A single draw of arm `Z` could not have decided that cell in either direction.
2. **At 0.5 B the control's outcome flips with the seed** — 2 of 5 draws put `Z` below `F`. The cell
   the first write-up recorded as "fires, +0.019" fires on three seeds and fails on two.
3. **At 1.5 B it fires 5/5**, which is why T2's shape looked healthy. T2's own `Z` (`+3.372681`)
   sits 0.091 below the five-draw minimum — consistent with being a sixth draw, so §4's off-by-one
   explanation holds and no further implementation difference needs to be posited.

**What this does NOT say.** At 3 B, 4 of 5 draws beat `F`, and the draw mean (4.524840) is 0.679
below `F`. **That is not noise**: under this comparator the real rule really does tend to score worse
than random signs at 3 B. **It still licenses no claim about damage**, because `F` and every `Z` draw
at that cell sit above the chance line — `F` at 5.928 BPB, the `Z` draws at 4.439–6.135, against
4.070. **The dispersion disqualifies reading any single draw as a gate; the chance line disqualifies
the comparison itself.** §2 is the reason the verdict changed; §4.1 is only the reason the original
gate could not have worked either way.


## 5. What this cannot claim

- **It does not price ternarization.** It prices `R0`, one rule, which T2 had already shown to be
  the worst of five and which lands at chance. **The obvious experiment — R3/R5 across scale — has
  never been run**, and is §6 item 1.
- **It does not extrapolate to 10 B.** 3 B is 3.3× below that.
- **It does not identify why 3 B inverts.** The chance line explains why the inversion is *possible*
  and why it is *unreadable*; it does not explain the sign. The candidates — degenerate per-tensor
  scales, activation explosion through the ternarized FFN — are **hypotheses, and this probe
  measured neither**.
- **It does not price a healed model.** Every arm is a raw substitution: no healing, no rotation
  (T3), no organ-selective policy (T2b).
- **It says nothing about speed**, and no timing was taken.
- **The chance line is a reference, not a floor.** A model can be arbitrarily worse than uniform. It
  marks where BPB stops carrying information about damage, not a bound.

## 6. Owed, in priority order

1. **The same sweep with R3 (and R5), across scale.** This is now the experiment E12 should have
   been. R3 is the exporter's own `--rule R3` and needs only calibration activations; it lands
   1.59 BPB below the chance line at 1.5 B, which is the only regime where a cost ratio would mean
   anything. **Until this runs, this programme has no measurement of how ternarization cost scales —
   only of how `R0` fails.**
2. **Re-read T2b's organ policy** against the `F` > `FA` inversion, and against the chance line: if
   T2b's arms sit above it, its organ ranking is subject to the same objection.
3. **T3's rotation across scale** (`7cdeca8`), for the same reason as (1).
4. **A diagnostic for the 3 B inversion** — per-tensor scale distributions and activation magnitudes
   through the ternarized 3 B FFN against 1.5 B — but **after** (1), since it may be an artefact of
   `R0` specifically.

## 7. Against myself

- **I re-adopted a control that the module I imported told me was mis-specified.** The docstring, the
  `ARMS` table and T2 §4(a) all said so. **The planted-control law says an instrument must fire on a
  known-positive before its nulls count; it does not say any arm labelled "planted" is a control.**
  Checking that a control is still sanctioned is now part of reading it.
- **I published `CONTROL-FAILED` on one draw of a stochastic arm** without asking what its
  draw-to-draw spread was — the same class of error as quoting a contended timing. **A stochastic
  comparator needs its own dispersion before any single reading of it decides anything.**
- **The smoke's margins did not survive.** At 0.5 B the smoke showed `Z − F = +0.147`; the full sweep
  showed **+0.019**. **A control that passes on a 2-sequence, 4-layer smoke is not thereby a
  control.**
- **The chance line cost nothing to compute and was available from the start.** It is `log2(V)` over
  a number already printed in every result file. **Both anomalies that sent me chasing an instrument
  bug are explained by a constant I never took.** A converted model's BPB should be read against it
  before anything else is concluded.
