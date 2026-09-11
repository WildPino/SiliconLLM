# E19 — the carve does not rank, and FFN-only carving cannot reach the target anyway

**Brief**: `briefs/BRIEF_E19_DOES_THE_CARVE_RANK.md`, **pushed before part B ran** (`3173edc`).
**Runners**: `engine/e19_carve_budget.py` (A), `density/e19_carve_rank.py` (B, `bdd87a4`).
**Results**: `results/e19_carve_budget.json`, `results/e19_carve_rank.json`. Run: 2280 s, `VOID: none`.

---

## 0. Verdict

**`OUTCOME_LABEL: CARVE-DOES-NOT-RANK`**

Two independent findings, either of which alone closes the carve route for this donor.

**(A) FFN-only carving cannot reach 50 tok/s at the target size, at any depth — including deleting
the FFN outright.** On Qwen2.5-Coder-7B, `attn + head` alone come to **`1.367 G`** active weights
while E18's 50 tok/s budget is **`0.982–1.060 G`**. Every carve this programme has ever built
(D0, D0c, Probe-4) is FFN-only.

**(B) No carve ranks.** Seven arms, oracle-routed. `base` and `FULL` hold `160/160`; **every carved
arm is `AT-FLOOR`**, including the one at exactly the depth this donor's own budget requires.

| arm | E | k | activation | BPB | vs chance `4.069819` | greedy | label |
|---|---|---|---|---|---|---|---|
| `base` | — | — | 100% | `0.767595` | `−3.302224` | **160/160** | **`G-C0` FIRES** |
| `FULL` | 256 | 256 | 100% | `0.767595` | `−3.302224` | **160/160** | **`G-C1` FIRES** |
| **`V52`** | 256 | 133 | **51.95%** | **`0.909441`** | `−3.160378` | **12/160** | **`AT-FLOOR`** |
| `S1` | 256 | 64 | 25% | `1.383868` | `−2.685951` | 7/160 | `AT-FLOOR` |
| `A0` | 32 | 8 | 25% | `1.858218` | `−2.211601` | 5/160 | `AT-FLOOR` |
| `N0` | 32 | 8 (null) | 25% | `2.578731` | `−1.491088` | 6/160 | `AT-FLOOR` |
| `D10` | 256 | 26 | 10.16% | `2.806167` | `−1.263652` | 3/160 | `AT-FLOOR` |

Floor `12/160` (E18 part A), margin `2` (E17), so `AT-FLOOR ≤ 14`. **All five carved arms inside.**

## 1. Part A — the depth the budget requires, and why FFN-only cannot get there

Derivation over measured quantities, importing `qwen_shapes` and the bytes-per-weight verification
from `e18_ladder_bandwidth.py` rather than restating them. **No timing taken.** Every weight is
charged as **ternary** (`0.500000` B, recomputed from the artifact) — the friendliest possible
assumption, since it grants full ternarization for free.

| model | active/token | FFN | **attn + head** | tok/s with the **FFN at ZERO** |
|---|---|---|---|---|
| Qwen2.5-Coder-7B | `7.070 G` | `5.703 G` (80.7%) | **`1.367 G`** (19.3%) | **35.9 – 38.8** |
| Qwen2.5-1.5B (D0/D0c's donor) | `1.544 G` | `1.156 G` (74.9%) | `0.388 G` (25.1%) | 126.7 – 136.8 |

**On the 7 B, `attn + head` alone exceed the entire 50 tok/s budget.** Deleting the whole FFN still
leaves 35.9–38.8 tok/s. Projected to the goal's size on each donor's **measured** attn+head share —
a projection, labelled as one, with no config invented — a 10 B carries a **`1.93–2.51 G` uncarvable
floor = `1.82–2.37×` the entire budget**, i.e. **19.6–27.4 tok/s with its FFN carved to zero.**

This was computed and **published in the brief before part B ran**, and it fixed part B's arm list:
at 1.5 B, 50 tok/s needs FFN activation `0.5144–0.5817`, so the verdict cell is `V52` at **51.95%**,
and the 25% cells D0/D0c already scored are *more* aggressive than that donor's own budget requires.

## 2. The controls, at zero error

`G-C0`: `base` reproduces `results/e6/ref.json` at **`160/160`**.

`G-C1`: `FULL` — the carve path with `k = E`, so the hook runs, loads the partition, routes, and
removes nothing — is **token-identical to `base`**, with a BPB difference of **`0.000e+00`**. The
carve machinery is lossless when asked to remove nothing.

**`G-C2`, the one that licenses the rest.** The oracle hook had to be restated here because
`d0c_granularity.py` loads a model at module level and cannot be imported, so it is gated
**behaviourally** instead of textually:

| arm | D0c published | E19 measured | \|diff\| |
|---|---|---|---|
| `base` | `0.7675949641196624` | `0.7675949641196624` | **`0.000e+00`** |
| `S1` | `1.3838675903770394` | `1.3838675903770394` | **`0.000e+00`** |
| `A0` | `1.8582178498441495` | `1.8582178498441495` | **`0.000e+00`** |
| `N0` | `2.5787313453805165` | `2.5787313453805165` | **`0.000e+00`** |

Four BPBs reproduced to **sixteen decimals**. The hook is D0c's hook; the partitions are literally
D0c's, loaded from the caches its own run wrote, so `CLUSTER_SEED` and the B3 seeding repair are
inherited by construction rather than by re-execution. `G-C4`: achieved activation equals nominal
**exactly** on all six carved arms.

## 3. The finding — a well-ordered degradation whose best point is already dead

`V52` is the cell that matters, and it is worse news than E18's `H` was.

**Keeping 52% of the FFN — the depth this donor's own 50 tok/s budget requires, chosen by an oracle
that reads the true activation mass — costs `+0.141846` BPB.** That is a change which, by the metric
this programme has used to choose every rule, fold, organ and granularity, is very nearly free: it
lands at `0.909441`, **`3.160` below the chance line**, closer to the untouched donor than to
anything this programme ships. **It agrees with its own donor on 12 of 160 greedy tokens — exactly
the constant-`'\n'` floor** — and diverges at token 0 of prompt 0.

Every deeper cut is further below the floor: `S1` 7, `A0` 5, `D10` 3.

**Unlike E18, the ordering here is not scrambled.** Across the five carved arms,
`r(BPB, agreement) = **−0.8562**` — the *expected* sign, over a `1.896726` BPB span. The carve
degrades gracefully and monotonically. **That is what makes the result sharper, not weaker: the
degradation is well-behaved and its very first usable point is already at zero information.** There
is no depth at which the carve is both fast enough to matter and able to choose a token; the whole
ordered range `3..12` sits at or under a model that emits `'\n'` forever.

## 4. Co-activation buys BPB and buys nothing in ranking

D0's headline and D0c's decision both rest on the co-activation partition beating a matched random
one. At 25% activation that gap is real in BPB and large: `A0 − N0 = **−0.720513**`.

**In ranking it is absent.** `A0` scores `5/160`, `N0` scores `6/160`.

The honest reading is *not* that the null partition ranks better — `5` vs `6` is inside the floor
and is noise, by E18's own law that ordering within the floor carries no information. The reading is
that **`0.72` BPB of partition quality produces no measurable ranking difference at all**, because
both arms are already degenerate. D0c §3.2's `G32` gap, and the granularity decision built on it,
are statements about score in a regime where score does not correspond to competence — the same
structure E18 found in T2b, now confirmed on the partition axis.

## 5. Predictions, scored — three called directions, three misses (again)

| # | prediction | outcome |
|---|---|---|
| 1 | `G-C0` fires; `G-C1` token-identical | **HELD** |
| 2 | `G-C2` replicates all four BPBs to `< 1e-6` | **HELD — exactly `0.000e+00` on all four** |
| 3 | **`V52` → `RANKS`** | **WRONG.** `AT-FLOOR` at `12/160` |
| 4 | **`S1`/`A0` → `PARTIAL`**, `S1` above `A0` | **label WRONG** (both `AT-FLOOR`); the ordering held, `7` vs `5` |
| 5 | `D10` → `AT-FLOOR` | **HELD** at `3/160` |
| 6 | **`N0` scores below `A0`** | **WRONG** (`6` vs `5`), though the difference is floor noise |

Prediction 3 was argued from a real mechanism — a carve leaves every surviving weight **bit-exact**
and computes a *subset* of the true function, unlike ternarization which perturbs every weight. The
mechanism is true and the prediction was still wrong by 68 tokens. **A different kind of damage is
not a smaller kind of damage.**

This is the fourth consecutive experiment (E16, E17, E18, E19) whose called directions missed and
whose verdict survived only because the brief had registered the alternative outcome in advance —
here brief §6's *"if instead `V52`, `S1`, `A0` and `D10` all come back `AT-FLOOR` … the donor's
argmax does not survive ANY structural modification — not precision, not sparsity."* **That is what
happened, and it is the reading.**

## 6. What E19 cannot claim

- **The oracle router is not a model.** It reads the true activation mass to choose experts, so
  every number here is a **ceiling**; any trainable router is worse. This makes the null result
  strong and would have made a positive result weak — a `RANKS` here would have been permission to
  try, never a working system.
- **No healing.** Same scope limit as E18, unchanged and now doubly load-bearing.
- **One donor (1.5 B), one partitioner, one corpus, 160 positions, 5 prompts.** The labels are D0's,
  fit at `p = 10%`; a different partitioner might carve better, though `N0` shows partition quality
  is not what is binding.
- **`r = −0.8562` is not a licence to use BPB.** All five points are at or below the floor; the
  correlation orders *degenerate* outputs, which is not competence.
- **Part A is arithmetic.** No timing taken; **`6.79 tok/s` stays exact and §19.3 is unchanged.**
- **Nothing here measures the engine.** The carve has no engine implementation, and Phase 61's law
  forbids carrying a PyTorch result into an engine rate.

## 7. Where this leaves the goal

E18 closed conversion on both axes. E19 closes the structural escape it left open.

**Quality**: the donor's argmax survives neither precision change (E17, E18) nor structural sparsity
(E19), at any depth, with an oracle router, on a donor that is otherwise intact. The failure is not
a property of ternarization — it is a property of **post-hoc modification of a pretrained dense
model as such**.

**Speed**: even granting a carve that worked, FFN-only carving cannot reach 50 tok/s at 7 B or
above, because attention and the head alone exceed the budget. A carve that reached the target would
have to cut **attention and the output head too** — which nothing in this programme has attempted,
and which E17 already showed is where ranking lives.

The remaining branch is unchanged and is now the only one: **train into the format**, with ~10% of
the model active per token — `SCALEUP_ARCHITECTURE`'s premise and Phase 64's actual programme.
Within the donor route, the only untested step is **healing**, and E19 raises its bar: healing would
now have to repair a model that is degenerate under *either* kind of modification.

## 8. Owed

1. **Healing** — E18 §9 item 1, unchanged, still the only untested branch. E19 sharpens the target:
   heal one carve depth (`V52`, the cheapest at `+0.141846` BPB) and re-measure ranking. If `160/160`
   does not return there, it will not return anywhere on the donor route.
2. **The intermediate ranking band** — E14 §5 item 3, owed since E14, **not** supplied by E19
   either: every arm here is degenerate, so nothing brackets a genuinely-different-but-good model.
3. **Carving attention and the head** — named by part A as what the target would actually require,
   never attempted, and prima facie hostile to E17's finding that the head is where ranking lives.
4. A re-read of **D0 §III and D0c §5**: their decisions rest on a BPB gap (`0.720513`) that §4 shows
   carries no ranking signal. Their numbers stand; the decisions built on them need restating.

---

## 9. Appended 2026-09-11 after E20 — the carve's floor readings are compound too

§8 item 2 recorded that every arm here was degenerate and so this probe could not supply the
intermediate band E14 §5 item 3 asked for. E20 supplied four such points and then showed what they
are worth (`probes/E20_RULE_OR_FORMAT.md`, ledger §33): **160 greedy positions are five trials of
thirty-two**, and two of the four intermediate arms are one prompt at `32/32` plus floor noise.

**More directly relevant to this probe**: E20 part B teacher-forces the donor's own context and
finds ternary heads keep the donor's token first at **67-74%** of positions against `6-26%`
free-running, with an exact identity (`G-B1`, `50/50`) tying the two harnesses. **Every agreement
number in §5 — `V52` 12, `S1` 7, `A0` 5, `N0` 6, `D10` 3 — is therefore a compound** of per-step
argmax fidelity and autoregressive drift, and none of them has been decomposed. **`V52`'s
`12/160` at `+0.141846` BPB is the cell most likely to be understating a carve**: its BPB cost is
smaller than any head arm E20 measured, and no teacher-forced number exists for it.

**Nothing here is withdrawn.** `CARVE-DOES-NOT-RANK` is a statement about free-running generation,
which is what a runnable model does, and part A's arithmetic — FFN-only carving cannot reach
50 tok/s at any depth — is untouched by any of this. **Owed, new: teacher-forced top-1 for the
carve arms**, which is cheap and would say whether a carve damages a step or only compounds.

**§7's `r(BPB, agreement) = −0.8562`, the expected sign, is now the minority reading**: E18 gave
`+0.4989` and E20 `+0.6238`. Two of three occurrences have the wrong sign.

---

## 10. Appended 2026-09-11 after E21 — the "cut attention too" owed item is answered, half yes

§7 closed FFN-only carving and left one instruction: **a carve that reaches the budget would have
to cut attention and the output head too, and that has never been attempted.** E21 attempted it,
by rank rather than by carving.

**Attention: yes.** Activation-weighted rank-512 on all 28 layers' `q_proj`+`o_proj`
(`QO-ACT-512`) costs **`+0.052689` BPB** — against `V52`'s `+0.141846` — and keeps the donor's
token first at **144/160** fixed-context positions. It is the cheapest structural cut this
programme has measured.

**The head: no.** The same construction at the same rank on `lm_head` costs `+0.278096` BPB and
101/160. The head is `545 M` on the Coder-7B = **51–56% of the entire 50 tok/s budget**, and no
lever measured here shrinks it.

**So §7's floor moves and does not open.** At the validated rank fraction (`r/D ≈ 1/3`, not the
brief's `r = 256` — `BOTH-ACT-256` shows this donor cannot take that), `attn+head` falls from
**`1.367 G` to `1.127 G`**, still **1.06–1.15×** the whole budget with the FFN at zero. E19's
arithmetic stands; the binding term is now named, and it is the head.

**And E19's own §8 item on composition acquires a warning.** `BOTH-ACT-256` reads **48/160**
teacher-forced where its two halves read 68 and 93 alone -- **while its BPB beats the additive
prediction by `0.342498`** (E21 §4a: the two metrics disagree in opposite directions). The obvious next
move — stack `QO-ACT-512` on `V52` to land the 1.5 B at ≈`0.98 G` — must therefore be **measured,
not assumed additive** (`probes/E21_CAN_RANK_BUY_IT.md` §8 item 1).

---

## 11. Appended 2026-09-11 after E22 — `V52`'s teacher-forced number, owed since §8

E19 and E18 both owed it. E22 measured it as a by-product of its replication anchor, under the
same oracle router, on the same five prompts:

| | BPB | free-running | **teacher-forced** | mean rank | rank ≤ 5 |
|---|---|---|---|---|---|
| `V52` | `0.909441` | 12/160 | **117/160** | `2.49` | 151/160 |

**`G-S2` reproduced §4's numbers exactly** — BPB to `< 1e-9`, free-running `12`, achieved
activation `0.51953125` — so this teacher-forced figure attaches to the identical arm and not to
a re-run that drifted.

**Two readings.**

**(1) §4's `AT-FLOOR` was a drift verdict, not a competence verdict.** With the donor's context
held fixed, the 52% carve puts the donor's token first **73% of the time** and in the top five at
151 of 160 positions. E20 part B's law applies here exactly as it did to the ternary heads:
*an autoregressive metric is not a measure of the component.* E19's verdict `CARVE-DOES-NOT-RANK`
stands as written — free-running is what a runnable model does — but the mechanism behind the 12
is now known to be the same as everywhere else in this programme.

**(2) It lands inside E20's ternary band, and that was a registered contest.** `117` sits in
`107`–`119`, which is `COMPARABLE`, and it equals E20's best ternary head (`GPTQH`, also `117`) to
the token, at a lower BPB cost (`+0.141846` vs `+0.170414`). E22's brief registered this as a
conflict between BPB ordering and E21 §4a's "BPB does not order across axes"; **BPB ordering won,
and E21 §4a is weakened as a result** (E22 §7).

**And composed, it gets better, not worse.** `QO512+V52` — this carve plus E21's rank-512 `q/o` —
reads **126/160**, above both its parts' worse half and above every ternary head, at `0.9441 G`
active, inside the 50 tok/s budget. That composition is the subject of `E22_DOES_CHEAP_COMPOSE.md`,
and §10's warning that composition can be worse than either part did **not** reproduce here.
