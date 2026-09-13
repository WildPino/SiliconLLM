# E58 — the 1.75× went nowhere. It was never there.

**Verdict: `NO-HEADROOM`.** Scores `briefs/BRIEF_E58_WHERE_DID_THE_1_75x_GO.md`, pre-registered
and pushed (`a9f5697`) before the runner existed. Runner `benchmarks/donor_adaptation/engine/
e58_organs.py`, results `results/e58_organs_run1.json` (registered) and `results/e58_organs.json`
(run 2, with the 1.5 B extension). Logs `D:/_ktmp/e58_run1.log`, `D:/_ktmp/e58_run2.log`.

**E57 addendum D.3 published a ×1.75 of engine headroom on the ternary path. It does not exist.**
It is the ratio of a packed-kernel rate to a bound `SPEED_LEDGER` §23.3 had **already withdrawn
for the packed path**, six days before I wrote it. Against the bound that governs this kernel the
engine is at **85–88% of the best cell E10 ever measured**, and the honest remainder is
**×1.11–1.17 at 0.5 B and ×1.07–1.14 at 1.5 B**.

---

## 1. The instrument closed, so the attribution may be read

`G-E58a` is the planted control: a decomposition that does not sum to the thing it decomposes is
not a decomposition.

| arm | organs summed | wall | Δ | bar | `G-E58a` |
|---|---|---|---|---|---|
| `05b_tqh` (registered, run 1) | 11.445 ms | 11.418 ms | **+0.24%** | ±8% | **CLOSES** |
| `05b_tqh` (run 2) | 11.834 ms | 11.813 ms | +0.18% | ±8% | CLOSES |
| `15b_tqh` (extension) | 34.009 ms | 33.889 ms | +0.35% | ±8% | CLOSES |

It closes forty times tighter than its own bar, on both scales, in two sessions. The `--bench 32`
smoke that preceded the run read **+5.7%** — the timer overhead is a fixed per-call cost and it
amortises. **A gate registered on the smoke's dispersion would have been set five times too loose;
the one registered on E57's 2.62% interquartile width was right.**

Conditions: `donor_engine_e53.exe`, `--threads 6 --bench 160 --profile`, k = 9 reps plus a
discarded warm-up, idle box (foreign occupancy median **4.81 / 4.65 / 4.86%**, clock 104.1–104.7%
of nominal). Rate dispersion **IQR 1.75 / 1.55 / 0.88%**, bootstrap 95% intervals
`[85.58, 88.36]`, `[83.53, 85.53]`, `[29.31, 29.79]`. Dispersion is reported as data: `G-E55a2` is
malformed (E57 A.5) and the programme still has no working interval gate.

**The byte table is derived from the shape the engine prints about itself and checked against a
file, not against the table it replaces.** Both artifacts' shape arithmetic reproduces E1's stored
`n_codes` **exactly** — 493,961,216 and 1,543,569,408 — which is self-test C-1 and C-3.

## 2. `G-E58b` — the organ table, on trained weights, at two scales

`MB/tok` is the ternary code bytes streamed per decoded token at 0.5 B/weight. `G-w/s` is the
column that compares to E10; `GB/s` is the column E57 D.3 used.

**`qwen25-05b_tqh`, registered arm, run 1 — 87.58 tok/s, 11.418 ms/token**

| organ | MB/tok | ms | IQR | GB/s | **G-w/s** | × to 25.5 GB/s | × to 37 GB/s |
|---|---|---|---|---|---|---|---|
| `qkv_proj` | 12.4 | 0.720 | 1.8% | 17.2 | **34.41** | 1.48 | 2.15 |
| `o_proj` | 9.6 | 0.466 | 1.1% | 20.7 | 41.35 | 1.23 | 1.79 |
| `ffn` | 156.9 | 7.088 | 1.6% | 22.1 | 44.27 | 1.15 | 1.67 |
| `head` | 68.1 | 2.717 | 1.3% | **25.1** | **50.10** | 1.02 | 1.48 |
| **total** | **247.0** | **10.991** | | **22.5** | **44.94** | | |
| *of which* `gate+up` | | 4.335 | | | 48.26 | | |
| *of which* `down` | | 2.102 | | | 49.76 | | |

**`qwen25-15b_tqh`, UNREGISTERED EXTENSION — 29.51 tok/s, 33.889 ms/token**

| organ | MB/tok | ms | IQR | GB/s | **G-w/s** |
|---|---|---|---|---|---|
| `qkv_proj` | 44.0 | 2.109 | 0.8% | 20.9 | **41.76** |
| `o_proj` | 33.0 | 1.437 | 0.6% | 23.0 | 45.97 |
| `ffn` | 578.0 | 25.002 | 1.1% | 23.1 | 46.24 |
| `head` | 116.7 | 4.673 | 1.9% | **25.0** | **49.94** |
| **total** | **771.8** | **33.221** | | **23.2** | **46.46** |
| *of which* `gate+up` | | 15.848 | | | 48.63 |
| *of which* `down` | | 7.787 | | | 49.49 |

**The laggard is `qkv_proj` at both scales** — the registered prediction, and it holds twice. It
is also the organ it is least worth fixing: **6.3%** of the 0.5 B token and **6.2%** of the 1.5 B
one, and bringing it alone to the head's rate buys **2.0%** at 0.5 B and **1.0%** at 1.5 B.
The laggard is named, and naming it is the end of that idea.

## 3. `G-E58c` — `Rem` is 2–4%, not 25–38%, and that is the finding

| | `rope` | `attention` | `norm+glue` | **`Rem` = wall − Σ weight organs** |
|---|---|---|---|---|
| `05b_tqh` | 0.016 ms | 0.373 ms | 0.069 ms | **0.427 ms — 3.7% of the token** |
| `15b_tqh` | 0.029 ms | 0.566 ms | 0.136 ms | **0.668 ms — 2.0% of the token** |

The brief registered the arithmetic that made this decisive: *"at 84.88 tok/s the token is 11.78 ms
and the weights at the 37.0 GB/s floor would be 6.73 ms, so if `Rem` is near 5.05 ms the 1.75× is
not in the kernels at all."* **`Rem` came back eleven times smaller than the gap it would have had
to fill.** The token is 96.3% weight-organ time at 0.5 B and 98.0% at 1.5 B. There is no glue, no
thread-wake, no attention cost and no sampling cost hiding a 1.75×. **Whatever the gap is, it is
inside the weight organs — which means it is the kernel or it is nothing.**

## 4. It is nothing. The bound was wrong, and the record said so on 2026-09-07.

`SPEED_LEDGER` §23, written from E10 (`probes/E10_PACKED_KERNEL_BINDER.md`, verdict `CORE-BOUND`):

> §19 priced the remaining engine budget by assuming the weight organs are **bandwidth-bound**…
> **E10 measures it, and it is false for the packed kernel** — the kernel stops at ~25.5 GB/s with
> the data *in L3*, so feeding it faster changes nothing.
>
> | §19.4 row | status |
> | 37.0 GB/s → 1094 M active weights at 50 tok/s | **WITHDRAWN for the packed path** |

E10's bench holds row length fixed at 3584 and moves the footprint 512×, from 4 MB to 2 GB. The
packed arm moves **less than 8%** across that range — 49.11 to 52.95 G-w/s, grand median 25.49 GB/s
— while the fp32 arm moves **4.4×**. The packed kernel is not reading memory too slowly; it is
core-bound at ~14% of this machine's FMA issue capability.

**So the right comparison is E58's `G-w/s` column against E10's 49.11–52.95:**

| | measured | E10's best cell | at E10's best |
|---|---|---|---|
| `05b_tqh` weight path | 44.94 G-w/s | 52.95 | **85%** |
| `15b_tqh` weight path | 46.46 G-w/s | 52.95 | **88%** |
| `05b_tqh` `head` alone | 50.10 G-w/s | 52.95 | **95%** |
| `15b_tqh` `head` alone | 49.94 G-w/s | 52.95 | 94% |
| `15b_tqh` `down` | 49.49 G-w/s | 52.95 | 93% |

**Three organs are inside E10's own measured kernel band.** The engine is not leaving a 1.75× on
the floor; it is running the kernel it has at 85–88% of the fastest that kernel has ever been made
to go on synthetic data with a 4× longer row.

**The honest ceiling, recomputed:**

| arm | measured | if every organ hit the fastest organ | if every organ hit E10's best cell |
|---|---|---|---|
| `05b_tqh` | 87.58 tok/s | 97.22 (**×1.11**) | 102.50 (**×1.17**) |
| `15b_tqh` | 29.51 tok/s | 31.67 (**×1.07**) | 33.54 (**×1.14**) |

Not ×1.75. Not ×1.59. **And E10 §23.3 had already put the weight path's own remainder at ~1.03×**;
E58 finds ×1.11 because E10's fixed 3584-row bench could not see that the two small attention
projections fall off the ceiling at short rows.

## 5. WITHDRAWN — E57 addendum D.3's transfer number, and it is the one that mattered

D.3 concluded:

> A healed ternary 1.5 B at `15b_tqh`'s byte count, running at the streamed floor, would read
> **47.71 tok/s** — 95% of the good bar, from a 1.5 B, with the 1.59× coming from work that has
> nothing to do with training. **That is the first time the two halves of the goal have had a
> common denominator.**

**Withdrawn.** The streamed floor does not govern this path. The replacement is not a projection at
all — the 1.5 B ternary artifact has now been *measured* with its organs open:

| | E57 D.3 said | E58 measures / derives |
|---|---|---|
| `15b_tqh` today | 30.03 tok/s | **29.51 tok/s** (confirms) |
| a healed 1.5 B at these bytes | **47.71 tok/s** | **33.54 tok/s**, the kernel's own ceiling |
| share of the 50 tok/s bar | 95% | **67%** |

The sentence D.3 was proud of is false. The two halves of the goal do **not** have a common
denominator at 1.5 B: healing the quality of a 1.5 B ternary artifact, even perfectly, even with
the kernel run at the best rate it has ever shown, lands at **two thirds of the good bar**.

## 6. What is left is the budget E18 already wrote, reproduced by a different route

Turn the measurement around and ask what 50 tok/s costs at the 1.5 B shape. Token budget 20.0 ms,
minus the **measured** `Rem` of 0.668 ms, leaves 19.33 ms of weight path; at E10's best cell that
buys **1.024 G active ternary weights per token**.

`E18_THE_RANKING_LADDER.md` §31 derived, from the artifact rather than from the engine, that
50 tok/s permits **0.982–1.060 G**. **E58's number sits inside E18's band**, from an independent
direction: E18 priced the format, E58 times the organs. Neither used the withdrawn floor. E57 D.3
was the outlier, and it was mine.

`15b_tqh` streams **1.544 G**. So:

* **50 tok/s at the 1.5 B shape requires cutting 33.7% of the streamed weights** — not a faster
  kernel, which is worth 14%.
* The FFN is **74.9%** of those weights (1.156 G of 1.544 G) and **73.8% of the token's time**.
  Cutting 33.7% overall means cutting **45.0% of the FFN** and nothing else.
* **That is the same order as E19's measured 52% FFN carve**, and it is why E22's `QO512+V52` came
  in at 0.9441 G — under budget. E18 §11's conclusion is unchanged and now has a timed organ table
  under it: *"the budget-feasible object and the working object are two different objects."*

**The FFN's share rises with scale** — 62.1% of the token at 0.5 B, **73.8% at 1.5 B**. The
theoretical ceiling of any FFN-sparsity or MoE work rises with it: 1/(1−0.738) = **×3.81 at
1.5 B**, against ×2.64 at 0.5 B. This is arithmetic on measured shares, not a claim that any such
work succeeds; the MoE note's ×1.52 ceiling was computed at the donor's 0.5 B-like shape and is
**scale-dependent, not a constant.**

## 7. Predictions, scored — 4 of 6, and both misses are load-bearing

| # | registered prediction | outcome |
|---|---|---|
| 1 | `G-E58a` closes inside ±8% | **HELD** — +0.24%, and on all three arms |
| 2 | laggard is `qkv` | **HELD**, at both scales |
| 3 | `head` still fastest, ≥ 25 GB/s | **HELD** — 25.1 GB/s. Margin 0.4%; marginal, and it held |
| 4 | `ffn` 20–30 GB/s | **HELD** — 22.1 |
| 5 | **`Rem` 3.0–4.5 ms, 25–38% of the token** | **WRONG** — 0.427 ms, 3.7%. Off by 7–11× |
| 6 | **ceiling ×1.25–1.45** | **WRONG** — ×1.11 to ×1.17, below the range |

I predicted the 1.75× was *split* between a slow organ and glue. It is in **neither**, because it
was not a quantity. Both misses point the same way — **less headroom than I thought** — and the
direction is the one that costs the programme, not the one that flatters it.

## 8. Cause, and the rule this adds

The mechanism is not "I forgot E10". E57 D.4 says exactly what I did:

> Bandwidth bounds taken from `SPEED_LEDGER` §1 (DRAM 40–44, proj floor 37.0), **not** from the
> 34.75 GB/s figure quoted in older sections, which §1's correction supersedes.

I searched, I found two candidate bounds, I picked the one §1 blessed, and I wrote down which one I
picked and why. **What I never asked is whether the bound I picked applies to the kernel I was
measuring.** §23.3 does not contradict §1 — it says §1's floor is the *fp32 stream's* floor and is
**withdrawn for the packed path**, which is a scope restriction living four sections later in the
same file. My search was over `briefs/`, and the withdrawal is in the ledger.

This is the third instance this week of the same shape (E57 B.3, C.7, now this), and the previous
two rules do not catch it: the artifact *was* searched and the sources *were* opened. The new
clause is about the bound, not the search:

> **A bound is a number with a scope.** Before a measured rate is divided by a bound, the bound's
> own source must be opened and its **scope** quoted beside it — which kernel, which precision,
> which access pattern it was measured on. A ratio whose denominator's scope is unstated is not a
> ratio, and "the ledger's current headline figure" is not a scope.

Registered by extending `feedback_charged_vs_moved_bytes.md` rather than as a new memory: that
file already said *"a ceiling is a denominator, and a denominator is where the wrong unit hides"*
and **it was not enough, because the unit was right and the scope was wrong.** The second question
belongs next to the first. The runner enforces the visible half: it prints **both** bounds on every
row and names which section withdrew which.

## 9. `G-E58d` — nothing is promoted

No flag became a default, no rate is published as an engine result, `donor_engine.c` is untouched.
E58 is a map. What it suggests — that the packed kernel itself, not its feed, is the wall, and that
E10 §23.4's two named candidates (`vpshufb` producing 16 bytes where `vpmovsxbd` consumes 8; the
never-measured `--lut` path at donor scale) are the only kernel-side leads left — is registered as
a separate experiment with its own control, not concluded here.

## 10. Run discipline

Run 1 is the registered measurement and its numbers are the ones quoted; that was written down
(`D:/_ktmp/e58_run2_registration.txt`) **before** run 2 launched, per the E36 run-2 rule. Run 2
exists for the 1.5 B extension and re-ran the registered arm only because the runner changed
(shape derived from the engine's footer, parser whitelisted, stderr merged). Run 2's registered
arm reproduces run 1's structure exactly — same laggard, same fastest organ, `Rem` 0.423 vs
0.427 ms — at a rate 3.3% lower, which is within the between-session drift E53 measured and
attributed to time rather than load. **Run 2 does not promote run 1 and is not permitted to.**

## 11. Checked, not assumed

Every source below was **opened and read**, and what it *said* is recorded, per E57 C.7's amended
rule:

* **`SPEED_LEDGER` §23.1–23.4** — read in full. Gives E10's six-footprint table, the `CORE-BOUND`
  verdict, `G-K1 = 1.022`, the planted control `G-K0` at 4.40 reproducing probe-3's 16 MB cliff
  unprompted, and the explicit **WITHDRAWN** row for 37.0/42 GB/s on the packed path. Also gives
  §23.2's engine readings of 51.66/52.22 G-w/s for `gate+up`/`down` **on the synthetic shape**;
  E58 reads 46.40–48.63 and 47.91–49.49 on trained artifacts, 6–10% lower, consistent with a
  shorter row (D = 896/1536 against the bench's fixed 3584). The disagreement is reported, not
  smoothed.
* **`probes/E18_THE_RANKING_LADDER.md` §31 and §11** — read. §31 gives the 0.982–1.060 G budget
  at 50 tok/s *and* flags that §23.3's own two companion numbers disagree by 4%
  (`25.5 / 0.5 = 51.0` against "~53 G-w/s"), recorded rather than resolved. §11 gives E22's
  `QO512+V52` at 0.9441 G, 126/160 teacher-forced at fp32, and the finding that ternarising it
  collapses it to 28/160 — the "budget-feasible object and the working object are two different
  objects" line quoted in §6 above.
* **`BRIEF_E57_…` D.3 and D.4** — reread to quote the withdrawn claim exactly rather than
  paraphrase it.
* **`results/e57_wall.json`** — the moved-byte and `n_codes` figures E58's shape arithmetic is
  checked against (1,543,569,408 for `15b_tqh`, reproduced exactly).
* **The engine's own footer**, `D=… F=… L=… heads=…/… hd=… V=…`, read from **stderr** — which is
  the defect note below.

**A defect found and fixed mid-run:** the engine prints its profile table to **stdout** and the
shape footer identifying *which model it just timed* to **stderr**. The first version of the runner
captured both but parsed only stdout, and would have timed an artifact without ever seeing which
one it was. Same class as `feedback_config_must_appear_in_output`, one stream over. The runner now
parses both streams and refuses to proceed if no shape footer is found.
