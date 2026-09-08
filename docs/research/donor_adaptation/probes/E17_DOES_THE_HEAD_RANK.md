# E17 — does the ternary HEAD rank?

**Brief**: `briefs/BRIEF_E17_DOES_THE_HEAD_RANK.md`, pre-registered and pushed before any arm ran
(`cf33251`). **Runner**: `benchmarks/donor_adaptation/engine/e17_head_rank.py`.
**Results**: `results/e17_head_rank.json`, `results/e17_head_vs_twin.json`.

---

## 0. Verdict

**`OUTCOME_LABEL: HEAD-IS-NOT-THE-MECHANISM`**

| gate | reading | registered label |
|---|---|---|
| `G-H0a` 0.5 B fp32, known-positive | **160/160** | **FIRES** |
| `G-H0b` 1.5 B `TQH` = E6 `A3` = E16 `C0` | **10/160**, `first_div (0,0)` | **REPLICATED** |
| `G-H0c` 1.5 B fp32 — a new cell | **160/160** | **FIRES** |
| `G-H0d` E13 build vs E6 build on `H0b` | token-identical | **COMPARABLE** |
| **`G-H1`** 1.5 B, head fp32 | **12/160 = 7.50%** | **`HEAD-IS-NOT-THE-MECHANISM`** |
| **`G-H2`** 0.5 B, head fp32 | **3/160 = 1.88%** | **`HEAD-IS-NOT-THE-MECHANISM`** |

Bars, derived in brief §5 from the measured known-negative ceiling (`10/160`) and the known-positive
band (`160/160`): mechanism at **≥ 85/160**, not-mechanism at **≤ 20/160**. Both arms are far below
the lower bar. **Stage 2 — the 7 B head-fp32 export — does not run**, as the brief registered.

Ternarizing the output projection is **not** what stops the converted donor from ranking.

## 1. The gates that had to fire before anything new was read

Three of the four are replications, and they are the reason the null counts.

`G-H0a` reproduces E6's `A1` at **160/160**. `G-H0b` reproduces E6's `A3` at **10/160 diverging at
token 0**, from the same file, the same binary and the same prompts. The instrument fires on a
known-positive and reproduces a known-negative in the configuration that produced E17's nulls —
the planted-control law, satisfied before the head axis was looked at.

`G-H0c` is new: **the 1.5 B fp32 arm reads 160/160**. That is a third independent known-positive
(after E6's 0.5 B and E7's 7 B), at a third scale, and it also settles E1 §4.4's open question in
the generate path — the engine now loads a **6,174,857,268-byte** file and reproduces PyTorch on it
exactly.

`G-H0d` was registered because E17 compares numbers across two engine builds: E6 used
`engine/donor_engine.exe` (315,904 B, 15:25) and E16 used `D:\_ktmp\e13\donor_engine.exe`
(317,952 B, 18:01). Re-running `H0b` under the E13 build gives **token-identical ids**. The two are
the same instrument, so E6's and E16's agreement numbers may be placed in one table. **This was a
gate and not an assumption**, and it is the only reason §4 below is allowed to exist.

**The single-axis gate passed before any agreement was read.** The `TQ`/`TQH` sidecars at both
scales differ on `head_ternary` (False→True), `rule_applied_to` (7→8 entries, the added entry being
exactly `lm_head`), `tied` (1→0), and on `bytes`/`sha256`/`mean_ternary_zero_fraction` — and on
nothing else. Same rule `R3`, same `calib_seqs: 32`, same pinned revision, same
`calib_matches_t2_operating_point: true`.

## 2. The head axis

| donor | head ternary | head fp32 | Δ |
|---|---|---|---|
| **1.5 B** | `10/160` | **`12/160`** | **+2 tokens** |
| **0.5 B** | `3/160` | **`3/160`** | **0 tokens** |

Removing the ternary head buys **two tokens out of 160** at 1.5 B and **nothing at all** at 0.5 B.
Both arms still diverge from the donor at **token 0**.

This was the cheapest remaining mechanism for E16's central finding, and it is dead.

## 3. `G-H3` — the pairing, and the thing worth keeping

E14's law requires every scoring metric to be paired with a ranking one. Done here for the head:

| donor | `ΔBPB` (head-ternary − head-fp32), E1 §4.3 | `Δagreement` (same direction) |
|---|---|---|
| 1.5 B | **`−0.008546`** — the head *helps* the score | **`−1.25` points** — and *hurts* the rank |
| 0.5 B | **`+0.022070`** — the head hurts the score | **`0.00` points** — and does nothing to the rank |

The signs disagree at 1.5 B and the magnitudes are meaningless at both. But the pairing is not the
interesting part; **this is**:

| pair | tokens identical to its own head-ternary twin | agreement with PyTorch |
|---|---|---|
| 1.5 B head-fp32 vs `TQH` | **81/160** — 49% of the output changes | `12/160` vs `10/160` |
| 0.5 B head-fp32 vs `A2` | **28/160** — **82.5%** of the output changes | `3/160` vs `3/160` |

**The head is not inert. It rewrites half the output at 1.5 B and five sixths of it at 0.5 B, and
changes the number of correct tokens by two and by zero.** The two arms are not similar models;
they are two differently-wrong models. A component can dominate what comes out and be irrelevant to
whether any of it is right.

**Law: "this component changes the output" and "this component changes the answer" are different
measurements, and a null on the second does not license calling the component inert.** E17 nearly
made that error: `G-H2`'s `3/160` against `A2`'s `3/160` reads as "the head does nothing", and the
twin comparison shows 82.5% of the tokens moved underneath it.

## 4. The population — every arm this programme has ever generated with

`G-H0d` licenses one table. Chance is `4.069819` at `V = 151936` and `4.070106` at `V = 152064`.

| arm | scale | rule / fold / head | BPB | vs chance | greedy vs its own fp32 |
|---|---|---|---|---|---|
| fp32 | 0.5 B | — | `0.871810` | `−3.198` | **160/160** |
| fp32 | 1.5 B | — | `0.767595` (E12 §1, PyTorch) | `−3.302` | **160/160** |
| fp32 | 7 B | — | `0.674027` | `−3.396` | **160/160** |
| `TQ` | 0.5 B | R3 / none / fp32 | `4.509164` | `+0.439345` | `3/160` |
| `TQH` | 0.5 B | R3 / none / ternary | `4.531234` | `+0.461415` | `3/160` |
| **`TQ`** | 1.5 B | R3 / none / fp32 | `3.484253` | `−0.585566` | **`12/160`** |
| `TQH` | 1.5 B | R3 / none / ternary | `3.475707` | `−0.594112` | `10/160` |
| `B1` | 7 B | R0 / layers / ternary | `5.299200` | `+1.229094` | `0/160` |
| `B2` | 7 B | R3 / layers / ternary | `4.017233` | `−0.052874` | `0/160` |
| `B3` | 7 B | R3 / none / ternary | `4.168325` | `+0.098219` | `0/160` |

**Three known-positives at exactly 160/160, across three scales and two model families. Seven
ternary arms across three scales, two rules, two folds and both head settings: the best is 12/160,
and every one of them diverges at token 0.** Not one ternary artifact this programme has built
ranks, and the head was the last cheap axis that had never been tried.

## 5. What this does to BPB as an instrument

Across those seven ternary arms, **BPB spans `1.823493`** — from `1.229094` above the chance line to
`0.594112` below it — while **greedy agreement spans 0 to 12 out of 160**.

The metric that has chosen every rule, fold, organ set and scale in this programme moves by 1.8
bytes-per-byte across a population whose ranking behaviour does not move at all. Both facts are
measured on the same artifacts.

The honest scope of that: **BPB is not broken.** Over the full range including fp32 it tracks
perfectly — `0.67`–`0.87` goes with 160/160, and `3.48`–`5.30` goes with 0–12/160. The problem is
narrower and worse: **inside the ternary regime, which is the entire operating range this programme
ships in, BPB has resolution and ranking has none, because every ternary arm is already on the
floor.** E16's `RULE-FIXES-IT` was read off `0.052874` of BPB movement in a regime where 1.8 BPB of
movement buys nothing. The chance line is what separates the two groups: every fp32 arm sits ~3.2–3.4
below it, every ternary arm within ±1.23 of it.

## 6. Predictions, scored

| # | prediction | outcome |
|---|---|---|
| 1 | `G-H0a` fires 160/160, `G-H0b` replicates 10/160 | **BOTH HELD** |
| 2 | `G-H0d` passes | **HELD** — token-identical |
| 3 | `G-H1` → `HEAD-IS-NOT-THE-MECHANISM`, **2–15%** | **HELD**, `7.50%` — inside the called band |
| 3′ | the *reasoning*: `TQ` retains ≈18% of the donor's information over uniform, so it should not reproduce the argmax | **HELD as a direction.** Registered as weaker than a two-point trend, and it is not promoted now that it landed |
| 4 | `G-H2` → `HEAD-IS-NOT-THE-MECHANISM` under 12.5% | **HELD**, `1.88%` |
| 4′ | **`H2 > A2`** at 0.5 B, since the head costs `+0.022` BPB there | **WRONG — exactly equal, `3/160` vs `3/160`.** A called direction, missed. §3's twin comparison shows why the equality is a coincidence of counting and not a sign of similarity: 82.5% of the tokens differ |
| 5 | `\|Δagreement\| < 10` points at both scales | **HELD**, `1.25` and `0.00` |
| 6 | stage 2 will not run | **HELD** |

`G-H0c` had no direction called beyond "fires"; it fired.

## 7. What E17 cannot claim

- **It still does not supply the ranking band E14 and E16 owe.** The known-positive at 160/160 is a
  *numerical-equivalence* reading — the same weights, engine against PyTorch — now with three
  readings instead of two, but it bounds what identity produces, not what a harmless difference
  produces. **E14's `45.6%` remains unbanded**, and that item stays open.
- **There is no measured floor for agreement by frequency coincidence.** These arms emit ` the` and
  `\n` repeatedly, and PyTorch also emits those tokens sometimes. `12/160` therefore **cannot be
  claimed to be above zero information** — a model emitting only the most frequent token would score
  something. That floor is unmeasured, and every "the best is 12/160" statement here should be read
  as an upper bound on a quantity whose lower bound is unknown. This is the true residual of the
  ranking-band debt and it is now the cheapest open item in the programme.
- **A null on the head does not identify the body.** Ruling out one tensor is not evidence for
  another. E17 says the head is not *sufficient* to explain the failure. It does not locate it.
- **No speed number moves.** Nothing was timed and nothing is quoted; **6.79 tok/s exact stands**,
  and §19.3 is unchanged. (Decode rates were recorded by the engine and are discarded: the machine's
  idleness was not controlled and a contended timing is not a timing.)
- **The 7 B rows enter as members of a population, never as a scale term** — `Coder-7B` is a
  different family from `Qwen2.5-0.5B/1.5B`, exactly as E16 §0 refused `+0.692619` as a pure scale
  term.
- **160 positions, 5 prompts, one decoding rule.** A single arm's reading could not carry this; the
  population is what carries it.

## 8. Where this leaves the goal

Nothing here is a speed result and nothing here moves `6.79 tok/s` on the 7.072 B active donor,
still 7.4× short of 50. What E17 moves is the map of what is left to try.

By elimination, each step measured rather than argued:

- the **rule** is most of the BPB damage and does not restore ranking (E16, `G-R2 = −1.281967`);
- the **fold** is load-bearing in BPB and does not restore ranking (E16, `G-R4 = −0.151093`);
- the **head** is not the mechanism at either scale (E17);
- **scale does not rescue it** — 0.5 B, 1.5 B and 7 B all fail, and the 1.5 B arm that E16's brief
  called working because it sits `0.594` below the chance line is E6's *planted control*, which has
  emitted `" the\n\n the\n the the\n the"` since 2026-09-05.

The axes never tried are `R1`/`R2`, T3's rotation, healing/QAT — and the one the blueprint names.
**E16 concluded the binding constraint is the format rather than the rule; E17 removes the last
cheap alternative to that reading and adds the sharper form of it: no post-hoc conversion of these
donors has ever produced a model that can choose a token, at any scale, under any setting tried.**
*Training into the format rather than converting into it* — `SCALEUP_ARCHITECTURE`'s premise — is
now supported by an eliminated alternative rather than by a preference.

## 9. Owed

1. **The frequency-coincidence floor** (§7, new and now first): generate against a trivial baseline —
   the unigram-most-frequent token, and a shuffled-logit control — on the same 5 prompts, so
   `12/160` can be compared to *something*. Cheap, and it is load-bearing for every agreement number
   in §4.
2. **The ranking band for intermediate values** — E14 §5 item 3, still owed, still needed for
   E14's `45.6%`, and not supplied by E17.
3. **`R1` and `R2` at 1.5 B**, where a screen is cheap and both artifacts would be new axes; only
   `R0` and `R3` have ever been generated with.
4. **A `TQ`-style head-fp32 arm at 7 B** is *not* owed — the brief made it conditional on `G-H1`
   firing, and it did not.
5. Everything E16 §9 still owes: the 3 B cell of the R3 sweep, a clean scale axis (Qwen2.5-7B), a
   fold sweep at 3 B.

---

## 10. Appended after E18 part A — "best is 12/160" means "at the floor"

E17 §7 recorded that no floor for agreement by frequency coincidence had been measured, and that
`12/160` therefore could not be claimed to be above zero information. **`e18_agreement_floor.py`
measured it**, from the reference continuations alone — no engine, no weights, so it cannot be
contaminated by the arms it judges:

| reference | positions | distinct tokens | best CONSTANT predictor | token |
|---|---|---|---|---|
| Qwen2.5-0.5B | 160 | 91 | **11/160 = 6.88%** | `'\n'` |
| Qwen2.5-1.5B | 160 | 82 | **12/160 = 7.50%** | `'\n'` |

Against which §4's ternary population reads:

| arm | scored | floor | excess |
|---|---|---|---|
| 1.5 B `TQ` (this probe's `H1`) | `12/160` | 12 | **0** |
| 1.5 B `TQH` (`H0b`) | `10/160` | 12 | **−2**, and it emits `'\n'` **77 times of 160** |
| 0.5 B `TQ` (`H2`) / `TQH` (`A2`) | `3/160` | 11 | **−8** |
| 7 B `B1`/`B2`/`B3` | `0/160` | — | below |
| fp32, all three scales | `160/160` | 11–12 | **+149** |

**Every sentence in this probe that reads "the best is `12/160`" must be read as "the best is
exactly the floor".** §4's ternary column does not describe weak rankers; it describes models that
carry **no ranking information at all**, and the `0`–`12/160` spread §5 contrasted against
`1.823493` of BPB movement is dispersion around a degenerate baseline rather than a range of
partial competence.

Nothing in §§0–9 is withdrawn — every number stands, `G-H1` and `G-H2` keep their labels, and §7
had already refused the claim this note now settles. What changes is the *reading*: §5's finding is
stronger than it was stated. BPB does not merely have poor resolution inside the ternary regime —
**it spans 1.8 across a set of models that are, to the ranking instrument, uniformly
indistinguishable from a constant `'\n'` emitter.**

The **intermediate** ranking band is still owed and still unsupplied: this floor bounds a
*degenerate* model from below and says nothing about what a genuinely different but equally good
model produces, so **E14's `45.6%` remains unbanded.**
