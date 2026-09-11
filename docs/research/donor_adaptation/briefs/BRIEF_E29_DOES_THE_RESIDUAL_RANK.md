# BRIEF E29 — does the residual RANK layers, or is E27's finding just "don't touch the ends"?

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 0. This is a control on E27's best result, and it is designed to be able to kill it

E27's headline — the one positive result the donor branch has produced in three probes — is:

> `L21-MINRES` reads **113/160** teacher-forced against `L21-LAST`'s **57/160** at identical
> depth, identical parameter count and identical charged cost. +56 tokens purely from choosing
> WHICH seven layers to drop.

I wrote that up, put it in the INDEX, and proposed `L21-MINRES` to the T4 proposal as a healing
candidate (`T4_HEALING_PROPOSAL.md` §10). **And the comparison it rests on is not a fair one.**

`LAST` drops layers 21–27. Layer **27 has the second-highest residual of all 28** (0.7618, against
a 12–18 band of 0.28–0.36). So `LAST` is not a neutral alternative rule — it is a rule that
deletes the single most active block in the back half. **Beating it establishes almost nothing.**

The question E27 should have asked, and did not:

> **Does the residual RANK layers — or is the entire finding "the ends matter and the middle does
> not", in which case ANY seven middle layers would do just as well?**

If the second reading is right, then `MINRES` carries **no information**, the "rule space" E27 §7
said was unexplored does not exist, and the T4 proposal is carrying a recommendation built on a
measurement artifact. E29 is the probe that decides it.

---

## 1. The three-point test

All three conditions drop **exactly 7 layers** from the **same interior band**, so they have
identical depth, identical parameter count and identical charged cost. Only the *selection* moves.

**The interior band** is `[3, 24]` inclusive — the 22 layers that remain after protecting the
first 3 and the last 3. It is defined here, before the run, and it is chosen so that **E27's
`MINRES` set is unchanged by the restriction**: the seven smallest residuals globally are
`{15, 16, 14, 13, 12, 17, 18}`, all of which already lie inside `[3, 24]`. So `MINRES` is the
same arm E27 ran, and the comparison is clean rather than re-tuned.

| condition | the seven dropped | what it tests |
|---|---|---|
| **`L21-MINRES`** | the 7 **smallest** residuals = `{12,13,14,15,16,17,18}` | E27's rule, unchanged |
| **`L21-RANDMID`** ×3 seeds | 7 drawn uniformly at random from `[3,24]` | **the null: is the middle just fungible?** |
| **`L21-MAXRES-IN`** | the 7 **largest** residuals inside `[3,24]` = `{3,4,5,6,7,21,22}` | **the anti-rule: it must FAIL** |

**The ordering `MAXRES-IN < RANDMID < MINRES` is what "the residual ranks layers" predicts.** Any
other ordering falsifies it, and the one that matters is `RANDMID ≈ MINRES`.

`MAXRES-IN` is deliberately restricted to the interior rather than taken globally, because the
global 7 largest would include layer 0 (residual **19.05**, the block that builds the
representation) and layer 27, and demolishing the model with those proves nothing anyone doubted.
Restricting it makes it a **fair** anti-rule and therefore a real planted control: if the residual
carries information, `MAXRES-IN` must be clearly worse than a random draw from the same band.

---

## 2. The other two arms, both owed by E27 §7

| arm | what it is | why |
|---|---|---|
| **`L21-NOADJ`** | greedily take layers in ascending residual order, but **never two adjacent** | E27 §7 item 3. `MINRES` dropped a **contiguous** block (12–18). Is contiguity part of why it worked, or in spite of it? This forces a scattered set at equal count. |
| **`L17-MINRES`, `L18-MINRES`** | the same rule at 11 and 10 dropped | E27 §7 item 2 — the cliff. `L21` reads 113 and `L14` reads 54, and nothing between them was measured. |

---

## 3. Replication gates — the run must reproduce E27 before it may extend it

| gate | demands |
|---|---|
| **`G-E29a`** | `base` reproduces E27's `0.7675949641196624`, free 160, tf 160, within `1e-9` |
| **`G-E29b`** | `L21-MINRES` reproduces E27's **`0.993446`**, free **14**, tf **113** within `1e-9` |
| **`G-E29c`** | `L21-LAST` reproduces E27's **`2.500083`**, free 1, tf **57** within `1e-9` |
| **`G-E29d`** | the measured residual profile reproduces E27's to `1e-6` — layer 0 `19.0500`, min at 15 `0.2836`, layer 27 `0.7618` |
| **`G-E29e`** | the eval slice is the frozen one, sha `a1a48dc9…`, 51,870 scored bytes, 0 rejected |

`G-E29b` and `G-E29c` are the point: E29 reuses E27's own `Depth`, `measure_residuals` and
`keep_set` by **import**, not reimplementation, so if the anchors do not come back bit-identical
then something moved underneath and no new arm is readable.

**Quality is deterministic and may be measured under load** (the standing rule). E29 takes **no
timing** and reports none, so a busy box does not touch it.

---

## 4. Bands

E27's, unchanged: free floor 12, `AT-FLOOR ≤ 14`; teacher-forced **> 119 CHEAPER**, **107–119
COMPARABLE**, **< 107 WORSE**. Chance BPB 4.069819. `base` is 160/160.

---

## 5. Predictions — fixed here, before the run

1. **`G-E29a`–`G-E29e` all fire.** If `G-E29b` misses, E27's number is not reproducible and that
   is the result.
2. **`MAXRES-IN` lands WORSE, in `45–75` teacher-forced.** It removes seven of the most active
   interior blocks.
3. **`RANDMID` (mean of 3 seeds) lands in `85–105`** — better than `LAST`'s 57 because it spares
   the ends, worse than `MINRES`'s 113 because the residual carries *some* signal. **This is the
   prediction I am least confident in, and it is the one the probe exists for.**
4. **`NOADJ` lands within ±10 of `MINRES`.** I expect contiguity to be close to irrelevant and
   the residual ordering to be what matters — but I have no measurement either way, and if
   `NOADJ` beats `MINRES` clearly then the contiguous block was a handicap, not a help.
5. **The cliff is steep and closer to `L14` than to `L21`**: `L18-MINRES` in `95–115`,
   `L17-MINRES` in `80–105`. E27 measured 113 at 21 and 54 at 14.

**REGISTERED ALTERNATIVE, and it is the one that matters.** If `RANDMID`'s mean is **within 8
teacher-forced tokens of `L21-MINRES`**, then **the residual does not rank layers**, E27 §3's
headline is an artifact of comparing against a bad rule, and three things follow that I commit to
here rather than deciding afterwards:

- E27's probe document gets a correction at the top, not a footnote.
- `T4_HEALING_PROPOSAL.md` §10's `L21-MINRES` recommendation is **withdrawn** and restated as
  "drop seven middle layers, any seven".
- The claim that survives is the weaker and still useful one: **a quarter of a trained
  transformer's layers can go for almost nothing provided the first and last few are spared** —
  which is a real result, just not the one I wrote.

---

## 6. What E29 will NOT be able to claim

- **Nothing about speed.** No timing is taken. The charged cost of every `L21` arm is identical
  by construction (`1.2160 G` at the donor, E27's table), which is exactly what makes the
  comparison clean.
- **Nothing about the goal.** Every arm here is `7.75 tok/s` at `T10` with a dense FFN, per E27
  §4. E29 decides whether a *quality* finding is real; it does not move the budget.
- **Nothing about a 10 B's profile.** The residuals are this donor's, on one calibration slice.
  Whether a larger model has the same first-block spike and middle trough is untested.
- **Nothing about healing.** Every arm is surgery on frozen weights.
- **Three seeds is three seeds.** `RANDMID`'s spread across seeds is reported and, if it is wide,
  the comparison is reported as inconclusive rather than resolved in either direction.
