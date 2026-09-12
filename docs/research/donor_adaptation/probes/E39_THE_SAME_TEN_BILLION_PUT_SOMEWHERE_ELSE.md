# E39 — the same ten billion, put somewhere else

**Read this first: the verdict cell sits 1.5 tok/s from a band edge and the two runs straddle it.**
Run 1 (registered) printed **78.46** → `RANK-BUYS-SPEED`. Run 2 (the order control) printed
**80.55** → `TEN-B-NEAR-HUNDRED`. E36's run-2 rule, adopted by this brief §4 as standing
procedure before either run existed, says run 1's verdict stands and run 2 may not promote it.
**The band is therefore `RANK-BUYS-SPEED`, and the band is the least interesting thing here** —
§3 and §4 are what this probe actually bought.

**SPEED ONLY and the weights are NOISE** (brief §0). This prices a *shape*. It says nothing about
whether a model of this shape is any good, and §7 is explicit about that.

**My predictions scored 1 HIT / 4 MISS.** I priced factored attention as if moving fewer weights
cost proportionally less time. It does not, and §4 measures by how much.

---

**Verdict: `RANK-BUYS-SPEED`.** A file holding **exactly 9,999,220,736 parameters** — E36's
integer, to the parameter — with `q_proj`/`o_proj` written as rank-512 factors and the weight
that leaves them put into the FFN, reads **78.46 tok/s** at the goal's carve setting, against the
matched-parameter `A10B` re-timed in the same session at **46.34**. Within-session ratio
**1.693×** (run 2: **1.648×**).

**But the number that survived both runs is the FLOOR.** The fitted attention+head+router term is
**10.059 ms** (run 1) and **9.996 ms** (run 2) — **agreeing to 0.62%** where the slope disagrees
by 10.4%. That floor is **99.4 / 100.0 tok/s**.

> **E34 measured T10's floor at 20.03 tok/s with the FFN deleted and called 50 tok/s unreachable
> at that shape. This shape's floor is at the EXCELLENT target.** The same ten billion parameters,
> moved, turned a 2.3×-over-budget floor into one that sits on 100 tok/s.

**Brief**: `briefs/BRIEF_E39_THE_SAME_TEN_BILLION_PUT_SOMEWHERE_ELSE.md`, pre-registered and
pushed before the runner existed *and* before the exporter could build the object.
**Runner**: `engine/e39_same_ten_billion.py`. **Results**:
`engine/results/e39_same_ten_billion.json` (run 1, registered) and
`engine/results/e39_same_ten_billion_order_reversed.json` (run 2, the order control).
Build 538 s, each timing run 134 s.

---

## 1. The gates

| gate | what it requires | run 1 | run 2 | fires |
|---|---|---|---|---|
| `G-E39A` | parameter count recomputed **from each file's own header** == 9,999,220,736 exactly | `A10B` 9999220736 == `R512` 9999220736 | same | **yes** |
| `G-E39B` | charged == `419,430,400 + 36,962,304·k` at every `k`, zero tolerance | agrees at all of `k ∈ {1,2,3,4,6}` | same | **yes** |
| `G-E39C` | **planted control**: rank 4096 moves *more* weight than dense `q/o`, so it must be SLOWER | **30.80 < 46.47** | **31.66 < 48.74** | **yes** |

`G-E39C` is the one that licenses everything below it. `R4096` is charged **1.4698 G** against
`R0`'s **0.9330 G** — 58% more weight through the same factored code path — and it costs 34%
more time in both runs. **A rank that costs more did cost more, so the stopwatch is on the
factored path and not on something else.**

All four artifacts also passed E1's independently-written v4 layout predictor at zero tolerance
(`R512` 5,486,052,536 B, `R0` 5,687,247,672 B, `R4096` 5,956,732,088 B). That predictor **fired
for real earlier the same day**: the first combined carve+rank export was refused because
`layout_bytes_v4` had no rank term. The term was then derived from the format, not copied from
the writer.

## 2. Applying E36's run-2 rule, before reading anything else

The rule, quoted from E36 §4 and pushed at `9819226` long before E39:

> Run 1 is the registered measurement and its verdict stands. Run 2's only question is whether
> the k-slope depends on arm order. **If the slopes agree inside the reps' own dispersion the
> verdict is UNCHANGED — run 2 may not promote it over the bar no matter what number it prints.**
> If they disagree, both runs are reported and the cell is declared unresolvable.

| | run 1 (forward) | run 2 (reversed) | |
|---|---|---|---|
| verdict cell, `R512` at `k=3` | **78.456** | **80.548** | +2.7% |
| band | `RANK-BUYS-SPEED` | `TEN-B-NEAR-HUNDRED` | **straddles the edge at 80** |
| slope, ms per carve group | **0.9134** | **0.8184** | **−10.4%** |
| base term, ms | **10.059** | **9.996** | **−0.62%** |
| worst per-arm spread | **19.3%** | 8.6% | |
| `k*` for 100 tok/s, groups | −0.064 | +0.005 | **13 neurons of 48,128 apart** |
| `A10B` control vs E36's 49.96 | **−7.2%** | −2.1% | |

**Comparator, the same one E36 used**: the slope difference against the worst per-arm spread
across the two runs. **10.4% < 19.3% → the slopes agree inside dispersion → the verdict is
UNCHANGED.** `RANK-BUYS-SPEED` is assigned as registered.

**And the awkward part, stated rather than buried.** Run 1 — the run whose verdict governs — is
also the run that **failed its own prediction 5**: its `A10B` control read 7.2% below E36, outside
the ±5% the brief demanded. Run 2, which printed the better cell, is the comparable session. The
rule was written precisely so that a second run cannot be used to fish a better band, and it
costs me here; I am applying it in the direction that goes against me, which is the only
direction in which such a rule means anything. **The consequence: the BAND is not a result of
this probe.** What is a result is the floor (§3), the per-weight cost (§4), and the activation
budget (§5) — each of which the two runs agree on.

**The runner printed `1.61x A10B's 49.96` and that figure may not be used.** Prediction 5's own
text says a drifted control means the probe reports the drift *instead of* a cross-session ratio.
The legitimate comparison is within-session: **1.693× (run 1), 1.648× (run 2)**.

## 3. The floor, which is the number that survived

```
run 1   time = 10.059 ms + 0.9134 ms per carve group
run 2   time =  9.996 ms + 0.8184 ms per carve group
```

The base term is attention + head + router with the FFN contributing nothing. **The two runs
agree on it to 0.62%** while disagreeing on the slope by 10.4% — which is what you expect from a
within-rep drift that a reversed arm order redistributes: the drift lands on the slope, not on
the intercept.

| shape | attention+head floor | as tok/s | source |
|---|---|---|---|
| T10 (`D=4096, L=48`, dense `q/o`) | 46.5 ms | **20.03** measured with `--carve-k 1` | E34 |
| `A10B` (`D=4096, L=16`, dense `q/o`) | 17.2 ms | 58.1 | E36's fit |
| **`A10B-R512`** (`D=4096, L=16`, rank-512 `q/o`) | **10.0 ms** | **99.7** | this probe, both runs |

**This is the first shape in the programme whose floor is at 100 tok/s**, and it holds
9,999,220,736 parameters while doing it. E34 §7 and E36 §9 item 2 both named the attention shape
as the only lever left and neither pulled it; this pulls it, and the lever is as large as they
implied.

`k*` for 100 tok/s comes out **−0.064 groups** (run 1) and **+0.005 groups** (run 2) — i.e.
**zero**. 100 tok/s at this shape costs the entire FFN. My registered prediction 3 was `k* ≈ 1.0`
and it is a MISS by one group, in the optimistic direction.

## 4. What the factored path costs per weight — the controlled measurement

Prediction 4 registered that if the base came in below E36's 47.2–48.5 G-w/s envelope, the
factored matvec carries a per-weight cost E25's rank-invariance never saw. **It did**, and
`R0` exists so the cost can be measured without a fit. `R0` and `R512` are the **same file
shape, same `F = 48128`, same carve, same everything** — differing only in whether `q/o` are
dense or rank-512 factors.

| | charged G-weights/s at `k=3` | |
|---|---|---|
| | run 1 | run 2 |
| `R0`, dense `q/o` | 43.359 | 45.469 |
| `R512`, rank-512 `q/o` | 41.607 | 42.716 |
| **factored penalty** | **−4.0%** | **−6.1%** |

**The factored path delivers 4–6% fewer charged weights per second than the dense one.** The
mechanism is visible in the exporter's own log: rank 512 on `q/o` takes matvec calls per token
from **113 to 145** — two extra calls per layer, sixteen layers — so the same charged weight
arrives in more, smaller calls. E25 measured the rank axis inside a **1.54%** band and concluded
charged-throughput invariance; **at 10 B with 32 extra calls per token the effect is 3–4× that
band and E25's conclusion does not extend here.**

Against the comparable session (run 2, control within 2.1%), the two-term model reads:

| | E36 (`A10B`) | E39 run 2 (`A10B-R512`) | |
|---|---|---|---|
| base rate | 47.2 – 48.5 G-w/s | **41.96** | **−11 to −13%** |
| marginal carve group | 37 – 39 G-w/s | **45.17** | **+16 to +22%** |
| gather penalty (group / base) | **0.80** | **1.076** | |

**The gather penalty of 0.80 is gone, and it is gone for two reasons that pull in opposite
directions** — the base got 12% slower per weight (the factored path, §4) and the group got ~19%
faster per weight (group 188 against E36's 180, more contiguous work per gathered group). Quoting
the penalty alone would hide both. Run 1 reads 0.970 on the same quantity; the two runs disagree
by 11%, the same disagreement as the slope, so **I record ~1.0 ± 0.1 and not a third decimal.**

## 5. What the shape buys the goal — the constructive half

At 50 tok/s the FFN activation this box can afford:

| | active FFN neurons per layer at 50 tok/s | as a fraction of `F` |
|---|---|---|
| `A10B` (E36) | 540 of 46,080 | **1.17%** |
| **`A10B-R512`** | **2,046 (run 1) / 2,298 (run 2) of 48,128** | **4.25% / 4.78%** |

**Same ten billion parameters, and the 50 tok/s budget buys about four times as many active FFN
neurons.** That is the first movement on this axis since E35 fixed the envelope, and it matters
for the *other* half of the programme: E37 measured 1.17% reading `4.029398` BPB, below the
chance line, and E38 measured the value of selection peaking at 25% activation. **4.5% is still
far below that peak — this does not make the model good — but it is 4× closer, and it was bought
with zero parameters.**

Measured arms, for the record (mean of 5 reps):

| `k` | activation | run 1 tok/s | run 2 tok/s | charged |
|---|---|---|---|---|
| 1 | 0.39% | 89.95 | **92.00** | 0.4564 G |
| 2 | 0.78% | 85.40 | 86.32 | 0.4934 G |
| **3** | **1.17%** | **78.46** | **80.55** | 0.5303 G |
| 4 | 1.56% | 72.57 | 75.31 | 0.5673 G |
| 6 | 2.34% | 64.29 | 67.00 | 0.6412 G |

## 6. Predictions, scored as registered

| # | registered | outcome |
|---|---|---|
| 1 | all three gates fire | **HIT** — and `G-E39C` fires in both runs by 34% |
| 2 | `TEN-B-NEAR-HUNDRED`, specifically **84.8 tok/s** | **MISS.** Registered run 78.46 → `RANK-BUYS-SPEED`; the number is 7.5% optimistic. Run 2's 80.55 is in-band and **may not be promoted** (§2). |
| 3 | `k* ≈ 1.0` for 100 tok/s | **MISS.** Measured ≈ **0.0** in both runs. 100 tok/s costs the whole FFN, not one group. |
| 4 | E36's two-term model transfers: base at 47–48 G-w/s, group at ~0.80 | **MISS on both halves, in the direction the brief flagged.** Base **41.96**, penalty **~1.0**. The brief said a slow base would mean a factored per-weight cost E25 did not see — §4 measures it at **4–6%** against a within-run control. |
| 5 | `A10B` re-timed lands within 5% of 49.96 | **MISS in the registered run (−7.2%)**, HIT in run 2 (−2.1%). Scored on run 1. The cross-session ratio is withheld as the brief required. |

**1 HIT / 4 MISS, and they share one error.** Every miss is me pricing the factored attention as
though moving 49% fewer weights bought 49% less time. It bought 41.5% less (17.2 → 10.06 ms),
because the weights that remain move ~12% slower per weight. **The desk model that has priced
every shape in this programme since E35 does not carry across a change of matvec kind**, which is
Phase 61's law (`microbench does not compose`) arriving on the speed side.

## 7. What this does NOT say

- **Nothing about quality. The weights are noise.** A rank-512 ternary `q/o` is not free: E22
  measured it collapsing to `28/160` post-hoc. **H0 then trained that same organ back to
  `111/160`**, which is why this shape was worth pricing at all — but H0 was 1.5 B, `q/o` only,
  and its free-running generation did **not** recover (8/160). **E39 prices a shape H0 showed can
  be trained; it does not show a trained one exists at 10 B.**
- **The band is not a result** (§2). Two runs straddle the edge; the floor, the per-weight cost
  and the activation budget are what both runs agree on.
- **Nothing about the extra FFN capacity.** `F` goes 46,080 → 48,128 and at `k=3` only 564 of
  those are ever read. Whether the extra 2,048 per layer are useful is a training question.
- **Nothing about 12 k context.** Short context, as every speed probe since E30. E32 §8's
  context-ladder hypothesis is still owed.
- **One box, one thread count, one engine build** (`donor_engine_e26.exe`, `--threads 6`).

## 8. Owed

1. **The floor is now the whole cost, so attack the floor.** At `k=0` this shape reads 99.7 tok/s
   and its `419,430,400` base breaks down as **268,435,456 `q/k/v/o` (64.0%)**, 134,217,728 head
   (32.0%), 16,777,216 router (4.0%). Inside the attention half the split is now **exactly even**:
   `q/o` factored cost 8,388,608 a layer and `k/v` dense cost 8,388,608 a layer. **`k/v` are the
   larger remaining lever precisely because `q/o` were cut** — factoring them, or cutting `NKV`
   below 8, is the next cut and it is cheap to measure. The head is the second (32% and untied).
2. **The 4–6% factored penalty against call count.** §4 attributes it to 113 → 145 matvec calls.
   A rank sweep at fixed charged weight would separate call overhead from the factored kernel
   itself; E25's protocol already exists.
3. **Group size as the other half of §4's penalty story.** 180 → 188 moved the marginal group
   rate 16–22%. E37 §9 item 4 (`GSZ` sweep on A10B) now has a second reason to run.
4. **A session whose control is inside 5%.** Run 1 drifted 7.2%. The cell would be resolvable on
   a quiet box, and it is the cheapest owed item here (134 s).
5. **E34 §7 and E36 §9 item 2 are PARTLY PAID.** Both asked for the attention shape to be moved;
   this moves one axis (rank on `q/o`). `L`, `NH`, `HD` and `NKV` are untouched.
