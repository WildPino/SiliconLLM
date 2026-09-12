# E40 — after every attention lever is pulled, how much FFN can 50 tok/s buy at ten billion?

**Read this first: the band edge I chose runs straight through the measurement, and I am not
going to pretend otherwise.** The cell came out **5.99%** (run 1, registered), **6.18%** (run 2),
and **6.49% / 6.00%** on the same runs' medians. The band boundary is at **6.00%**. Every one of
those four numbers is within half a point of the bar and they fall on both sides of it.
**`ATTENTION-LEVERS-EXHAUSTED` is assigned as registered, and the band carries no information
here. The number is `~6%`, and §4 is why that number settles the question anyway.**

**SPEED ONLY and the weights are NOISE** (brief §0). Prices shapes, says nothing about quality.

**Predictions: 0 clean HIT, 2 split, 3 MISS.** The desk model got every *ordering* right and
every *magnitude* wrong, in one consistent direction — §5.

---

**Verdict: `ATTENTION-LEVERS-EXHAUSTED`.** With every attention lever this engine really has
pulled — `k/v` heads 8 → 2, `q/o` at rank 128 — a file holding **exactly 9,999,220,736
parameters** buys **~6% FFN activation at 50 tok/s**. E38 measured a selector at 6.25% activation
reading **3.597 BPB** against a dense **0.767595** and a chance line of **4.069819**: **86% of the
way to chance.** There is no attention shape in this programme's reach that makes post-hoc
selection affordable at ten billion on this box.

**But the milestone is on the other side of the same object.** `R128` at `k = 3` reads
**112.73 tok/s** (run 1) and **106.44** (run 2), and crosses 100 tok/s at **1.65% / 1.74%**
activation:

> **First artifact in this programme that is simultaneously a genuine ten billion parameters,
> above the EXCELLENT target, and still computing something in its FFN.** E39's `R512` reached
> 100 tok/s only at `k* ≈ 0` — with the FFN switched off. This one does not.

**Brief**: `briefs/BRIEF_E40_HOW_MUCH_FFN_CAN_FIFTY_TOK_S_BUY.md` + **addendum A**, both
pre-registered and pushed before the objects they govern existed. **Runner**:
`engine/e40_levers_exhausted.py`. **Results**: `engine/results/e40_levers_exhausted.json`
(run 1, registered) and `..._order_reversed.json` (run 2). Builds 769 s, timings 207 s and 233 s.

---

## 1. The gates, including the one that went VOID and cost me an arm

| gate | requires | outcome |
|---|---|---|
| `G-E40A` | every arm's parameter count, recomputed **from its own exported header**, == 9,999,220,736 exactly | **VOID on the first build**, then fires after the arm was replaced — see below |
| `G-E40B` | charged == a closed form written independently in the runner, zero tolerance, every arm, every `k` | **fires** |
| `G-E40C` | **planted control**: `NKV2` charges strictly less than `R512` at every `k`, so it must be FASTER at every `k` | **fires in both runs, at all five `k`** |

**`G-E40A` disqualified an arm I designed, which is the entire point of a planted gate.**

```
G-E40A  ALL 10133438464 == NKV2 9999220736 == R512 9999220736 == 9999220736 ?  -> VOID
```

The withdrawn `ALL` arm was over by **134,217,728 = V·D exactly**, one whole head. The cause is a
rule older than this probe, at `synth_export.py:207`: a tied model runs its head as the **fp32
embedding** — 544 MB/token at S05, 13.8 ms, **52% of the token** — so `--head ternary`, the
default and the only head any speed number in this programme was measured on, forces `tied = 0`.

**My brief was wrong about tying twice, in opposite directions.** Tying saves *parameters*, never
*charged weights* (caught before the brief was written; its §2 table is already corrected).
And at this engine's runnable head, tying is not a lever at all — **it is a 5× slowdown on the
head**, which is why the exporter refuses it. A lever that costs speed had no business in an arm
called `ALL`.

Addendum A withdrew it and substituted **`R128`** (`NKV = 2`, untied, rank 128, `F = 49152`),
which is a **stronger** arm than the one it replaced — base charged 0.218 G against 0.252 G — so
the substitution made the registered prediction *harder* to hit in the direction I had predicted.
It was pushed before `R128` existed and before any timing. **Prediction 1 is scored a MISS and
the gate is not re-run to a pass.**

All three surviving artifacts matched E1's independently written v4 layout at zero tolerance
(`R512` 5,486,052,536 B, `NKV2` 5,485,921,464 B, `R128` 5,485,888,696 B).

## 2. Applying E36's run-2 rule

| | run 1 (forward, registered) | run 2 (reversed) |
|---|---|---|
| **cell: `R128` activation at 50 tok/s** | **5.99%** → `ATTENTION-LEVERS-EXHAUSTED` | **6.18%** → `LEVERS-NEARLY-EXHAUSTED` |
| `R128` slope, ms/group | 0.9001 | 0.8798 (**−2.3%**) |
| `R128` base, ms | 6.198 | 6.073 (**−2.02%**) |
| worst per-arm spread | 48.9% | 34.9% |
| `R512` control vs E39's 78.46 | **−1.8%** | −2.6% |

`R128`'s slopes agree to **2.3%** against worst spreads of 48.9% and 34.9% — comfortably inside
dispersion — so by E36's rule **the verdict is UNCHANGED and run 2 may not promote it.**
`ATTENTION-LEVERS-EXHAUSTED` stands as registered.

**Both sessions are comparable this time** (−1.8% and −2.6% on the shared `R512` control, inside
the ±5% the brief demanded), which is the one protocol thing that went better than E39.

**Dispersion, reported because it is bad.** Two cells carry single-rep outliers — `R128_k4` run 1
reads `61.6` among `[108.7, 105.7, 98.7, 106.3]`, and `R128_k3` run 2 reads `84.2` among
`[121.4, 118.2, 105.2, 103.0]`. Both are first-rep readings and look like page-in, not engine
behaviour. **Sensitivity, NOT promoted** (E14 §6 forbids swapping to a post-hoc statistic):

| statistic | run 1 cell | run 2 cell |
|---|---|---|
| **mean (registered)** | **5.99%** | **6.18%** |
| median | 6.49% | 6.00% |

The median of run 1 lands at 6.49%, within a hundredth of my registered 6.50%. **I am not taking
that.** The registered statistic is the mean, the verdict stands on the mean, and the honest
summary of all four numbers is **`6.0 ± 0.3%`**.

## 3. The ladder — what each lever is actually worth

| arm | `NKV` | rank `q/o` | base charged | **floor tok/s** (run 1 / run 2) | **50 tok/s buys** | **100 tok/s buys** |
|---|---|---|---|---|---|---|
| `R512` (E39's object) | 8 | 512 | 0.419 G | 95.4 / 90.2 | 4.03% / 4.45% | ~0% |
| `NKV2` | **2** | 512 | 0.319 G | 129.3 / 124.8 | 4.67% / 4.24% | 0.86% / 0.70% |
| **`R128`** | **2** | **128** | **0.218 G** | **161.3 / 164.7** | **5.99% / 6.18%** | **1.65% / 1.74%** |

Measured arms for `R128` (mean of 5 reps):

| `k` | activation | run 1 | run 2 | charged |
|---|---|---|---|---|
| 1 | 0.39% | **142.29** | **150.84** | 0.2559 G |
| 2 | 0.78% | 128.07 | 131.59 | 0.2936 G |
| **3** | **1.17%** | **112.73** | **106.44** | 0.3314 G |
| 4 | 1.56% | 96.21 | 101.64 | 0.3691 G |
| 6 | 2.34% | 88.58 | 91.11 | 0.4446 G |

**Halving `k/v` heads is the cheapest large lever in the programme**: `NKV` 8 → 2 costs nothing
in parameters (they go back into `F`) and moves the floor from ~93 to ~127 tok/s. Cutting the
`q/o` rank 512 → 128 moves it again to ~163. **Together they take the floor from E36's 58 tok/s
to 163 — a 2.8× on the term E34 called the wall.**

## 4. Why ~6% settles the question, whichever side of the bar it falls

The two halves of this programme, in their own measured numbers:

| activation | what the box charges for it | what the best attainable selector reads there |
|---|---|---|
| 25% | **~16 tok/s** on `R128`'s own fit (extrapolated well outside the measured range) | **1.5689 BPB** — E38's peak value of selection |
| **~6%** | **50 tok/s — measured here, every real lever pulled** | **3.597 BPB** — E38, `k=16`, 86% of the way to chance |
| 1.17% | **~110 tok/s** — measured here | **4.398 BPB** — E38, above the chance line |

Dense fp32 is `0.767595`; chance is `4.069819`. **The rate the box can afford and the rate
selection is worth anything at do not overlap, and the gap is not one lever wide — it is 4×.**
`D`, `L` and `HD` are untouched (§6), but no plausible move on those three multiplies the
affordable fraction by four without taking the parameter count apart.

**So post-hoc conversion is finished as a route to this goal, and it is finished for a reason
that is now measured from both sides rather than argued.** E37 showed the conversion fails at the
rate speed demands; E38 showed no router can rescue it and located where selection *would* be
worth something; E40 shows the box cannot be made to afford that place. **The only remaining
route is training into the format**, which is H0's axis and the one that has actually moved
(`28 → 111`/160 teacher-forced at 1.5 B). That is what the standing goal's T4 clause should buy,
and `docs/COMMUNICATION.md` now says so with a costed ask.

## 5. Predictions, scored as registered

| # | registered | outcome |
|---|---|---|
| 1 | all three gates fire | **MISS.** `G-E40A` went VOID on the first build and took an arm with it. Not re-run to a pass. |
| 2 | `ATTENTION-LEVERS-EXHAUSTED`, specifically **6.50%** | **SPLIT: band HIT by 0.01 of a point, number MISS by 7.8%** (5.99% measured). §2 says plainly that a band won by a hundredth is not a result. |
| 3 | floors: `NKV2` **131.8**, replacement arm **192.7** | **SPLIT.** `NKV2` measured **129.3** (−1.9%, HIT). `R128` measured **161.3** (−16%, MISS). |
| 4 | the factored penalty GROWS at low rank; `R128` base rate **39–41 G-w/s** | **MISS on the number, HIT on the mechanism.** Measured **35.2 / 35.9** — below the registered band. Rank 128 costs ~12% per charged weight against rank 512 at the *same* 145 matvec calls. |
| 5 | `R128` clears 100 tok/s only under 1% activation, `k*₁₀₀ ∈ (0, 2]` groups | **MISS, and in the good direction.** Measured **4.2 groups, 1.65%** — the excellent target arrives with **four times more FFN alive** than I predicted. |

**The desk model's error is now characterised, not just noted.** It assumes a per-weight rate
independent of matvec shape. Measured base rates: rank 512 at `NKV` 8 → **38.9** (mean of runs),
rank 512 at `NKV` 2 → **40.5**, rank 128 → **35.6**. **The rate falls as the factored inner
dimension falls**, so every floor the model predicts is optimistic by roughly the amount the rank
was cut. E39 §8 item 2 asked whether the factored penalty was call overhead or kernel shape;
**`R128` holds the call count fixed at 145 and still loses 12%, so it is shape, not call count.**

## 6. What this does NOT say

- **Nothing about quality.** Synthetic weights, by construction and for the third probe running.
- **Nothing about `D`, `L`, `HD`.** Three shape levers untouched. This closes the levers *pulled*,
  not every conceivable shape — §4 argues the gap is too wide for them, and argues it, not
  measures it.
- **Nothing about whether ~6% can be TRAINED into.** That is the question handed to the T4 clause
  and it cannot be answered on a CPU with noise for weights.
- **The dispersion is poor** (§2): two first-rep outliers near 50% and 35%. The `R128` fit is
  nonetheless stable across arm orders to ~2%.
- **One box, one thread count, one engine build** (`donor_engine_e26.exe`, `--threads 6`).

## 7. Owed

1. **The head is now the single largest item in the floor.** At `R128` the base is 0.218 G, of
   which the untied head is **134,217,728 — 61.5%**. Attention is down to **67,108,864 (30.8%)**
   and the router is 16,777,216 (7.7%). **The head is now bigger than all the attention put
   together**, and it has never been probed on this axis. `V = 32768` is a choice, not a law.
2. **`D` and `L`, the two untouched shape levers**, now that `NKV` and rank are spent.
3. **A quiet box.** Two first-rep outliers cost this probe its dispersion; E39 §8 item 4 is the
   same request and still open.
4. **Group size**: `R128`'s groups are 192 neurons against `R512`'s 188 and E36's 180, and the
   marginal group rate keeps moving (40.1 / 47.2 / 41.9 / 42.9 across arms and runs). The `GSZ`
   sweep owed since E37 §9 item 4 would separate this from the rank effect.
