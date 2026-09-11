# E18 — the floor under every agreement number, and the ranking ladder

**Brief**: `briefs/BRIEF_E18_THE_RANKING_LADDER.md`, part B pre-registered and pushed before any arm
ran (`2ac74f5`). **Runners**: `e18_agreement_floor.py` (A), `ternary/e18_ranking_ladder.py` (B),
`e18_ladder_bandwidth.py` (C). **Results**: `results/e18_agreement_floor.json`,
`results/e18_ranking_ladder.json`, `results/e18_ladder_bandwidth.json`.

---

## 0. Verdict

**`OUTCOME_LABEL: CLIFF-NOT-SLOPE`**

**Ternarizing ANY single organ of this pretrained donor destroys its argmax completely.** Not
degrades — destroys, to at or below the score of a model that emits `'\n'` and nothing else. There
is no partial-conversion operating point: the ladder from "nothing converted" to "everything
converted" is a cliff at the first step, and the whole rest of it is flat along the floor.

| arm | organs ternarized | BPB | vs chance `4.069819` | greedy | label |
|---|---|---|---|---|---|
| `base` | — | `0.767595` | `−3.302224` | **160/160** | **`G-L0` FIRES** |
| `I` | identity through the same path | `0.767595` | `−3.302224` | **160/160** | **`G-L1` FIRES** |
| **`H`** | `lm_head` only | **`1.106584`** | `−2.963235` | **9/160** | **`AT-FLOOR`** |
| **`A`** | `q,k,v,o` only | `1.903569` | `−2.166250` | **4/160** | **`AT-FLOOR`** |
| **`F`** | `gate,up,down` only | `2.476967` | `−1.592852` | **5/160** | **`AT-FLOOR`** |
| `FA` | `F+A` | `3.484251` | `−0.585568` | `12/160` | `AT-FLOOR` |
| `FAH` | `FA+H` | `3.475706` | `−0.594113` | `10/160` | `AT-FLOOR` |

Floor `12/160` (part A), uninterpretable margin `2` (E17), so `AT-FLOOR` is `≤ 14`. **Every
converted arm is inside it.**

## 1. Part A — the floor is not zero

Every greedy-agreement number in E6, E14, E16 and E17 was quoted against an implicit floor of zero.
Zero is the floor for a *uniform* guesser (`160/151936 ≈ 0.001` expected matches). It is not the
floor for a degenerate model that emits high-frequency tokens, and these arms emit exactly those.

Computed from `results/e6/ref.json` alone — no engine, no weights — so it cannot be contaminated by
the arms it judges. The floor is the score of the best possible **constant** predictor: the
strongest zero-information model.

| reference | positions | distinct tokens | **best constant predictor** | token |
|---|---|---|---|---|
| Qwen2.5-0.5B | 160 | 91 | **11/160 = 6.88%** | `'\n'` |
| Qwen2.5-1.5B | 160 | 82 | **12/160 = 7.50%** | `'\n'` |

Against which every ternary arm the programme has ever built:

| arm | scored | floor | excess |
|---|---|---|---|
| fp32, all three scales | `160/160` | 11–12 | **+149** |
| 1.5 B `TQ` (E17 `H1`) | `12/160` | 12 | **0** |
| 1.5 B `TQH` (E6 `A3`) | `10/160` | 12 | **−2**, and it emits `'\n'` **77 times of 160** |
| 0.5 B `TQ` / `TQH` | `3/160` | 11 | **−8** |
| 7 B `B1`/`B2`/`B3` (E16) | `0/160` | — | below |

**E17's "the best is 12/160" means "the best is exactly the floor".** A correction note has been
appended to E17 §10; none of its numbers change, but its §5 finding is *stronger* than it was
stated. Recorded as owed-and-now-paid: E17 §7 named this the cheapest open item in the programme.

## 2. Part B — the controls, which are why the nulls count

`G-L0`: `base` reproduces `results/e6/ref.json` at **160/160**. The PyTorch harness fires on the
known-positive before any converted rung is read.

`G-L1`: `I` — the identity arm, which walks the same substitution code path and substitutes nothing
— is **token-identical** to `base`. T2b gated this in BPB at `+0.000e+00`; it had never been gated
in generation, and now is.

**`G-L2`, the one that matters: the PyTorch harness reproduces the engine EXACTLY.**

| arm | E18 (PyTorch) | engine | Δ |
|---|---|---|---|
| `FA` (= E1 `TQ`) | **12/160** | 12/160 (E17 `H1`) | **0** |
| `FAH` (= `TQH`) | **10/160** | 10/160 (E6 `A3`) | **0** |

Not "within the margin" — on the nose, both. Two independent known values reproduced across two
instruments before a single empty rung was read. That is the entire licence for §0's table, and it
is why `H`, `A` and `F` can be trusted despite never having existed as engine artifacts.

## 3. The cliff

Five of the seven rungs had never been generated with. All five are at the floor.

**The sharpest single reading in this programme is `H`.** It ternarizes **one tensor** — the output
projection — and leaves the entire body in fp32. It costs **`+0.338989` BPB** against the untouched
donor: by the metric this programme has used to choose every rule, fold, organ set and scale, that
is a nearly-free change, and it lands at `1.106584`, **`2.963` below the chance line**, a better
score than most arms the programme has ever shipped. **It agrees with its own donor on 9 of 160
greedy tokens — below the constant-`'\n'` floor of 12.**

`A` ternarizes attention only and scores `4/160`. `F` ternarizes the FFN only — 88% of the body's
parameters at this width — and scores `5/160`.

There is no rung between "exact" and "broken". The brief's §6 prediction 6 registered both readings
in advance and named this one as the stronger: *"If instead `H`, `A` and `F` all come back at the
floor, the collapse is at the very first rung and the readable conclusion is far stronger — that
ternarizing any single organ of this donor destroys ranking."* That is what happened.

## 4. BPB and ranking are not merely decorrelated — they run backwards

Across the five converted arms, spanning **`2.377667` BPB** (from `−2.963` to `−0.586` against
chance):

**`r(BPB, agreement) = +0.4989`** — a *positive* correlation, meaning the worse-scoring arms rank
slightly *better*. The best-scoring converted arm, `H` at `1.106584`, ranks **worse** (`9/160`) than
the worst-scoring one, `FA` at `3.484251` (`12/160`).

The honest reading of the sign: **n = 5, and all five are at the floor**, so the ordering inside the
floor is noise and the `+0.4989` is not a mechanism — it is what "no signal" looks like when you
compute a correlation on it anyway. It is reported because the *absence* of the expected negative
correlation, across 2.4 BPB, is the finding.

E17 §5 said BPB has resolution inside the ternary regime while ranking has none. E18 extends that
to a regime E17 never sampled: **`H` at `−2.963` is not "inside the ternary regime" by any BPB
reading — it is close to the fp32 arms — and it still cannot rank.** The decoupling is not a
property of bad models. It is a property of *converted* ones.

## 5. Part C — the speed axis of the same ladder

Derivation over measured quantities, every input cited, no timing taken. It validates itself twice:
**bytes-per-weight computes to exactly `0.500000` (`4.0000` bits)** from `qwen25-15b_tqh.bin` rather
than being assumed, and the derived `FAH` weight-path rate of **7.21 tok/s** sits just *above* the
engine's measured **6.79** — the correct side, since the weight path excludes attention math, norms,
softmax and glue.

**The budget the goal implies**, at the packed kernel's measured ceiling:

| target | active ternary weights / token | share of a 10 B model |
|---|---|---|
| **50 tok/s** | **0.982 – 1.060 G** | **9.8 – 10.6%** |
| 100 tok/s | 0.491 – 0.530 G | 4.9 – 5.3% |

Reported as a span because **ledger §23.3 pairs "~25.5 GB/s" with "~53 G-w/s" in one row and
`25.5 / 0.5 = 51.0`** — its own two companion numbers disagree by 4%. Recorded rather than resolved
by picking one.

**And the ladder's two axes point in opposite directions.** An fp32 weight costs **5.87×** a ternary
one on this machine (`34.75 GB/s ÷ 4 B` against `25.5 GB/s ÷ 0.5 B`), so every rung that keeps an
organ exact is slower:

| arm | ternary G-w | fp32 G-w | tok/s (weight path) |
|---|---|---|---|
| `base` | 0.000 | 7.070 | **1.23** |
| `H` | 0.545 | 6.525 | 1.31 |
| `A` | 0.822 | 6.248 | 1.36 |
| `F` | 5.703 | 1.367 | 3.71 |
| `FA` | 6.525 | 0.545 | 5.24 |
| **`FAH`** | 7.070 | 0.000 | **7.21** |

**The fastest rung is the one that converts everything, and it is 6.9× short of 50 tok/s.** Quality
improves down this table and speed improves up it. There is no point on the ladder that is both.

## 6. Predictions, scored — three called directions, three misses

| # | prediction | outcome |
|---|---|---|
| 1 | `G-L0` and `G-L1` fire | **HELD** |
| 2 | `G-L2` replicates `FA` at 12, `FAH` at 10 | **HELD, exactly — 0 error on both** |
| 3 | **`H` → `RANKS`** | **WRONG.** `AT-FLOOR` at `9/160` |
| 4 | **`A` → `RANKS`**, below `H` | **WRONG** on the label; `AT-FLOOR` at `4/160`. Below `H`: held |
| 5 | **`F` → `ABOVE-FLOOR-DOES-NOT-RANK`** | **WRONG.** `AT-FLOOR` at `5/160` |
| 6 | the ladder is not linear in BPB; collapse between `F` and `FA` | **the primary reading was WRONG** — the collapse is at the *first* rung — but the brief registered this exact alternative in §6 and named it the stronger conclusion, **before the run** |

**Every substantive directional call in this brief was wrong.** Prediction 3 was flagged in advance
as "the least confident call in this brief" and it was still wrong by 71 tokens. What preserved the
interpretation was not the predictions but **having written the alternative outcome and its reading
into the brief before the data existed** — the pre-registration did the work the forecasting did not.

**Law: registering the alternative outcome is worth more than getting the direction right, because
the alternative is what protects the reading when the direction fails.** This programme has now
missed called directions in E16 (`G-R3`), E17 (`H2 > A2`) and E18 (three of three); in all three the
verdict survived because the brief had said in advance what each outcome would mean.

## 7. What E18 cannot claim

- **Nothing here tests healing, fine-tuning or QAT.** E18 measures *post-hoc* conversion of a frozen
  pretrained donor. That a converted donor cannot rank says nothing about whether a donor *trained*
  or *healed* into the format can, and the literature's ternary results are all trained-in. **This
  is the single largest scope limit and it names the only branch left standing.**
- **The intermediate ranking band is still owed.** Part A bounds a *degenerate* model from below; it
  says nothing about what a genuinely different but equally good model produces. **E14's `45.6%`
  remains unbanded** and E18 does not supply it.
- **`r = +0.4989` is not a mechanism**, and no sign is claimed from it: n = 5, all at the floor.
- **`RANKS ≥ 80/160` is a registered convention**, not a discovered threshold.
- **One donor, one scale (1.5 B), one rule (`R3`), one corpus, 160 positions, 5 prompts.** The cliff
  is demonstrated at 1.5 B; that `FA`/`FAH` reproduce the engine at 1.5 B and E16 saw `0/160` at 7 B
  is consistent with it holding at scale, but E18 did not measure the cliff at 7 B.
- **Part B is PyTorch, not the engine** (Phase 60's law). `G-L2` bridges the two exactly at the two
  rungs where both exist; the empty rungs are model results, not runtime results, and `H`/`A`/`F`
  could not be built as engine artifacts without changing `QWENDON1`, which carries one global
  `quant` field.
- **Part C is arithmetic, not a measurement.** No timing was taken and none is quotable.
  **6.79 tok/s exact stands; §19.3 unchanged.**

## 8. Where this leaves the goal

The two axes fail independently, and both are now measured rather than argued.

**Quality**: there is no partial-conversion operating point. Ternarizing one organ of a frozen donor
— even the one that costs `0.34` BPB — takes it from `160/160` to below a constant-`'\n'` emitter.
Combined with E15/E16/E17, the elimination is complete across rule, fold, head, organ coverage and
scale: **no post-hoc conversion of a pretrained donor has ever produced a model that can choose a
token, and E18 shows the failure begins at the first tensor converted.**

**Speed**: 50 tok/s permits **0.98–1.06 G active ternary weights per token — 9.8–10.6% of a 10 B
model.** The 7 B donor activates 7.07 G. Even the all-ternary rung, the fastest possible point on
the conversion ladder, is **6.9× short**, and every rung that would improve quality is slower still.

So the target is not reachable by converting a dense donor, at any conversion quality, for two
independent reasons. **It requires a model that is trained into the format and activates ~10% of
itself per token** — which is exactly `SCALEUP_ARCHITECTURE`'s premise and Phase 64's actual
programme. E16 called the constraint the format rather than the rule; E17 removed the last cheap
alternative; **E18 gives the constraint a number on both axes.**

## 9. Owed

1. **The same ladder with healing** — convert one organ, then fine-tune briefly, and re-measure
   ranking. §7's largest scope limit, and now the only untested branch of the conversion route.
   If one organ plus healing recovers `160/160`, the route reopens; if it does not, it closes.
2. **The intermediate ranking band** — E14 §5 item 3, owed since E14, not supplied by E17 or E18.
3. **The cliff at 7 B** — E18 measured it at 1.5 B. E16's `0/160` is consistent but is a different
   family and a different coverage.
4. **`R1`/`R2` for completeness**, deliberately deprioritized in the brief §7 with the reason
   recorded: both are worse than `R3` at equal coverage, and `R3` at *any* coverage is now at the
   floor, so they cannot open a route.
5. Everything E16 §9 still owes: the 3 B cell of the R3 sweep, a clean scale axis, a fold sweep.

---

## 10. Appended 2026-09-11 after E20 — the floor is doing more work than part A could show

Part A's floor (`12/160`, best constant-token predictor, `'\n'`) has now gated three experiments.
E20 used it unchanged and produced the first arms above it (`probes/E20_RULE_OR_FORMAT.md`,
ledger §33): `17`, `41`, `15`, `42` of 160.

**Part B's `H` arm is replicated exactly.** E20's `R3H` — `t2b_organs.apply_arm(model, "H", ...)`,
this ladder's own construction — reads **`9/160`**, identical to this probe's rung, and BPB
`1.1065835970951252` against T2b's `1.1065836079824596`.

**And the cliff is partly an artefact of the metric, though not of this probe's conclusion.**
E20 part B teacher-forces the donor's context and finds every ternary head keeps the donor's token
first at **67-74%** of positions. So the ladder's rungs are not "empty": `H` at `9/160` is a head
that is right 110 times out of 160 per step and cannot survive its first mistake. **`CLIFF-NOT-
SLOPE` stands as a statement about free-running behaviour** — which is what a runnable model does
— but the cliff is in the *compounding*, not in the per-step damage, and part A's floor is what
makes the compounded numbers look flat.

**§7's `r(BPB, agreement) = +0.4989` has a third data point.** E19 read `−0.8562`, E20 reads
**`+0.6238`**. Inside the ternary regime the sign of this correlation is not stable, and E20 part B
says why: BPB tracks the logit geometry, which the data-aware rules preserve well, while argmax
tracks the top-1/top-2 boundary, which none of them protects.
