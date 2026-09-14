# BRIEF E65 — what does `R128`'s rank fraction cost, on weights that are not noise?

**Pre-registered 2026-09-14, pushed before the runner is touched and before one BPB is read.**
**QUALITY ONLY. No rate is measured. `G-E63d` is `VOID` and OWED and E65 does not touch it.**

## 0. The question, in one sentence

**Every tok/s above 50 at ten billion is bought with RANK, and rank's quality has only ever been
measured post-hoc at fractions gentler than the one the fast shape uses.** E65 measures the
fraction `R128` actually uses.

## 1. Why this is the next thing, stated from the record

The speed ladder that produced this programme's 10 B headline (E39, E40):

| shape | `D` | rank | `r/D` | rate |
|---|---|---|---|---|
| `A10B-K3` | 4096 | **none** (dense q/o) | — | ~49 tok/s (composite; `G-E63d` OWED) |
| `A10B-R512` | 4096 | 512 | **1/8** | ~93 floor |
| `A10B-NKV2` | 4096 | 512 | **1/8** | ~127 floor |
| **`A10B-R128`** | 4096 | **128** | **1/32** | **~113–130 measured, ~163 floor** |

**The only rank-free shape is the one that sits AT the target, not above it.** Everything above
50 tok/s is rank. E40 §6 says so plainly about its own scope: *"Nothing about quality. Synthetic
weights, by construction and for the third probe running."*

And rank's quality **has** been measured on the real donor — Qwen2.5-1.5B, `D = 1536`, dense
`0.767595`, chance `4.069819`, gap `3.302224` — twice, in E21 and E27, read here from their own
result files (`engine/results/e21_rank.json`, `engine/results/e27_floor.json`):

| `r/D` | rank | BPB | Δ vs dense | share of the dense→chance gap |
|---|---|---|---|---|
| 1/3 | 512 | 0.8202837636996289 | +0.052689 | **1.60%** |
| 1/6 | 256 | 1.5458802267622023 | +0.778285 | **23.57%** |
| **1/8** | 192 | 1.8563784310382423 | +1.088783 | **32.97%** |
| 1/16 | 96 | 2.0972750188750418 | +1.329680 | **40.27%** |
| **1/32** | **48** | **NEVER MEASURED** | | ← **`R128`'s fraction** |

**At the fraction `R512` and `NKV2` use, post-hoc rank already costs a third of the distance to
chance.** `R128` goes four times further and sits off the bottom of the ladder.

**The ladder DECELERATES** — 1.60 → 23.57 → 32.97 → 40.27 — so this is saturation towards
chance, not an explosion. Any claim that halving the fraction multiplies the damage is false
after the first step, and §5 registers a prediction that respects that.

## 2. What is actually being measured, and what it is NOT

E65 measures the **POST-HOC** cost: an activation-weighted low-rank projection of a trained dense
donor, exactly as E21/E27 constructed it.

**H1, closed today, is the reason this is a floor and not a verdict.** H1 showed applying the
carve (`1.096636`) and training with it (`0.962593`) are different objects, and H0 showed the
same for rank at `r/D = 1/3` (post-hoc `0.820284` → trained `0.810022`). **Nobody has ever
trained at an aggressive rank fraction.** So a bad number here does **not** close `R128`; it
**sizes the gap that training would have to close**, which is the quantity the next T4 ask would
be argued from.

## 3. Apparatus

`ternary/e27_floor.py`, unchanged except **one arm added**:

```
("QO-48", 48, None, 0, None)        # r/D = 1/32 at D = 1536, R128's fraction
```

Run with `E27_ONLY="QO-512,QO-192,QO-48"` (`base` is always included). Same donor, same frozen
slice, same activation-weighted construction, same code path — the arms differ only in `r`.

## 4. Gates

| gate | requires | kind |
|---|---|---|
| `G-E65a` | **three planted positives**: `base` = `0.7675949641196624`, `QO-512` reproduces E21's `0.8202837636996289`, `QO-192` reproduces E27's `1.8563784310382423`, each to \|d\| ≤ **1e-9** | the instrument must fire on known positives before the new cell counts |
| `G-E65b` | **RANK partner**: teacher-forced top-1 reported for every arm, not BPB alone | E14 §3; E20 made tf a standing second metric |
| `G-E65c` | **ORDINAL**: is the ladder monotone in `r/D` on BOTH metrics? | a SCORE that does not rank is E62's outcome and a real result |

**Why 1e-9 is the right tolerance here, and why §63.1 does NOT apply.** E64's law — *a selection
path amplifies build noise from ~1e-07 to ~1e-03* — is scoped to **selection**. A low-rank
projection is a **continuous** path with no top-`k` anywhere, and this runs in the same process
on the same build via the same imported functions, not across binaries. E27 already replicates
E21 at `REPL_TOL = 1e-9` on exactly this arm. Applying E64's loose tolerance here would be
over-applying my own new law.

## 5. Predictions — registered, falsifiable

1. **`QO-48` lands between 2.10 and 2.60 BPB.** The ladder is decelerating (the last step,
   1/8 → 1/16, added only 7.3 points of gap); 1/32 should add less again, i.e. ~43–50% of the
   gap = 2.19–2.42, and the band is widened to 2.10–2.60 to be falsifiable rather than lucky.
2. **It stays BELOW chance (4.069819).** Unlike E38's oracle, which went above it.
3. **`QO-48` teacher-forced lands below `QO-96`'s.**
4. **The ladder stays monotone in `r/D` on BPB.** If it does not, that is E62's `does-not-rank`
   outcome again on a second axis, and it is a real result.
5. `G-E65a` fires on all three. If it does not, **no cell in E65 may be read** — E64's discipline,
   adopted verbatim.

## 6. What E65 may NOT conclude

1. **Nothing about 10 B directly.** `D = 1536`, not 4096. **`r/D` is not established to
   transfer** — E21 §8 validated it at `D = 1536` only, and E16's non-monotonicity forbids the
   extrapolation. The number is a **same-fraction analogue**, and the probe must say so in every
   sentence that uses it.
2. **Nothing about a TRAINED low-rank at this fraction**, which is the actual open question.
3. **No rate.** Not one tok/s. `G-E63d` stays `VOID` and OWED.
4. **No retraction of E40.** E40's rates are correct for what they measured; E65 prices a
   quantity E40 explicitly declined to measure.
5. **No claim that `R128` is dead.** See §2 — post-hoc is a floor, not a verdict.

---

## 7. Addendum A — run 1's planted control was TAUTOLOGICAL, and the runner mutated a published record

**Written after run 1 completed and BEFORE its cell was read into anything.** Run 1's number is
not quoted here and does not enter the probe; it is superseded by run 2, whose controls actually
execute.

### A.1 What happened

`e27_floor.py` resumes: any arm already present in the output file is **skipped and echoed**.
Run 1 was launched with the output file being E27's own published
`engine/results/e27_floor.json`, so the three registered known-positives printed

```
  base      CACHED  bpb 0.767595
  QO-512    CACHED  bpb 0.820284
  QO-192    CACHED  bpb 1.856378
```

**They were read from the very file they were supposed to be validating against.** `G-E65a` as
executed compared a file to itself. Its `G_F2` clause — the real replication test, which
compares a **recomputed** BPB to E21's anchor at `REPL_TOL = 1e-9` — is inside the compute
branch and therefore **never ran**.

**This is not a FAILED gate, it is an UNEVALUATED one**, which is the E4 precedent: a gate that
cannot be answered as written is MALFORMED, not failed, and is re-specified — here simply
*executed properly* — rather than reinterpreted. §5's prediction 5 stands unchanged: if
`G-E65a` does not fire, no cell may be read.

### A.2 The second half, which is worse

Run 1 **wrote its output back over `engine/results/e27_floor.json`**, inserting a `QO-48` arm
into E27's published record. **A new experiment silently mutated an old experiment's result
file.** The file is tracked, so it was restored with `git checkout --`, and no committed record
was ever wrong.

> **The defect, stated generally: a runner whose resume cache lives INSIDE its published result
> file will (a) make every replication control tautological and (b) overwrite the record it is
> replicating.** Those are the same line of code doing both.

It is the family of `feedback_instrument_must_not_measure_itself` — *the control must
DISCRIMINATE, not merely refuse* — in its sharpest form yet: here the control could not even
refuse, because it was handed its own answer.

### A.3 The fix, and it is in the apparatus not the brief

`e27_floor.py` gains one override:

```python
OUT = os.environ.get("E27_OUT") or os.path.join(
    ENGDIR, "results", "e27_floor%s.json" % ("_smoke" if SMOKE else ""))
```

Run 2 writes `engine/results/e65_rank_fraction.json`, which does not exist, so **all four arms —
`base`, `QO-512`, `QO-192`, `QO-48` — are computed from scratch** and `G_F2` executes against
E21's anchor. E27's file is untouched.

**Owed, beyond E65:** every runner in this programme that resumes from its own output has this
hazard. The general repair is that a resume cache belongs in a **scratch** file keyed to the
run, never in the published record.

### A.4 What does not change

§5's predictions are **unchanged and unread** — prediction 1's band `2.10–2.60` was registered
before run 1 produced anything and is scored against run 2. §6's prohibitions stand.
**`G-E63d` is still `VOID` and OWED.**
