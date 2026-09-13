# E57 — WHAT DOES A **TRAINED** LLM ACTUALLY COST ON THIS ENGINE, AND AT WHAT QUALITY?

**Pre-registered 2026-09-13, pushed before the measured runs.** The six single-cell probes in §2
were taken first and are marked as such; everything gated below is unmeasured.

---

## 1. The gap this closes, stated plainly

The goal is *"far funzionare un LLM **già addestrato** sulla nostra architettura"* at 50 tok/s
(good) / 100 tok/s (excellent). Fifty-six experiments in, **every tok/s in this programme is
measured on `e40_r128.bin` — synthetic weights.** E55 produced a careful demonstration table and
had to stamp `WEIGHTS SYNTHETIC` on it.

Meanwhile a **real trained model has been running on this engine since E6**, with token-for-token
greedy parity against HuggingFace (`G-E53c2`: A1 160/160), and it has been used only as a
*correctness control*. **Its speed has never been measured, never tabulated, and never compared
to the synthetic number the programme quotes.**

That is not a small omission. It is the difference between "the architecture can be fast" and
"an already-trained LLM is fast on it", and only the second is the goal.

## 2. Six probes, `ALREADY MEASURED`, one cell each at `--bench 160`

| weights | quant | head | GB | tok/s | `ffn~` ms/tok |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | fp32 | fp32 | 1.98 | 18.98 | 33.81 |
| Qwen2.5-0.5B | packed | fp32 | 0.72 | 42.83 | 7.64 |
| Qwen2.5-0.5B | packed | **ternary** | 0.79 | **81.28** | 7.70 |
| Qwen2.5-1.5B | fp32 | fp32 | 6.17 | 6.08 | 121.82 |
| Qwen2.5-1.5B | packed | fp32 | 1.59 | 18.32 | 25.85 |
| Qwen2.5-1.5B | packed | **ternary** | 1.71 | **29.14** | 25.80 |
| *`e40_r128` `--carve-k 3`* | *ternary* | — | *5.11* | *119.57 (E55)* | — |

**Two things fall out of these numbers before any gate runs.**

**(a) The head is 47% of the token at 0.5B and 37% at 1.5B.** Ternarising it is worth **1.90×**
and **1.59×** while `ffn~` does not move (7.64 → 7.70, 25.85 → 25.80). Qwen2.5 ties its embedding
to its output head, so an fp32 head is `151936 × 1536 × 4 B = 933 MB` streamed **per token** — at
this box's measured ~30 GB/s (Probe-3) that is ~31 ms of a 54.3 ms token, and the arithmetic
agrees with the measurement.

**(b) The synthetic 10B shape is 6.5× faster than a real model 6.7× smaller.** That is not an
error and not cheating — it is `--carve-k 3` activating **1.17% of the FFN**. But it means the
programme's headline rate is produced *entirely by sparsity that no trained model has yet been
shown to survive* (E37's verdict on post-hoc conversion was **NO**).

## 3. The thing that makes this experiment necessary

Cross-referencing §2 against E6's stored quality scores:

| arm | tok/s | HF greedy match |
|---|---|---|
| 0.5B fp32 | 18.98 | **160/160** — E6 A1, the known-positive |
| 0.5B packed, fp32 head (`tq`) | **42.83** | **NEVER MEASURED** |
| 0.5B packed + ternary head (`tqh`) | 81.28 | **3/160** — E6 A2, a *planted control* |
| 1.5B fp32 | 6.08 | NEVER MEASURED |
| 1.5B packed, fp32 head (`tq`) | **18.32** | **NEVER MEASURED** |
| 1.5B packed + ternary head (`tqh`) | 29.14 | **10/160** — E6 A3, a *planted control* |

**The two fastest real arms are E6's deliberately-degraded controls.** They are fast and they
emit almost entirely different tokens from HuggingFace. The only arm with proven parity is the
*slowest* one at 0.5B.

**And the `tq` arms — ternary body, fp32 head — have never been scored.** They sit at 42.83 and
18.32 tok/s, between a proven-correct 18.98 and a known-broken 81.28. *Whether a trained LLM runs
correctly on this engine above 20 tok/s is an open question, and it has been open since E6
without anyone noticing.*

## 4. Design

**Quality** reuses E51's `score_against_ref` unchanged, against E6's stored HuggingFace
references — 5 prompts × 32 new tokens, greedy. The references are keyed by *model name*, not by
quantisation, so `Qwen/Qwen2.5-0.5B`'s reference scores every 0.5B arm. **No new scoring code is
written**, so E57 cannot drift away from E6/E50/E51/E53.

**Speed** reuses E55's method: discarded warm-up cell, rotated arm order, quantile intervals,
per-cell occupancy + timestamp + clock, `k = 15` per arm at `--bench 160`.

## 5. Gates

### `G-E57a` — the scorer must fire before any null counts
> A1 (`05b_f32`) must score **160/160**, A2 (`05b_tqh`) **3/160** and A3 (`15b_tqh`) **10/160**,
> reproducing E6 exactly. Any deviation stops the experiment.

Three planted controls: a known-positive and two known-negatives at different degradation levels,
all with numbers fixed by an experiment run seven days ago. This is the strongest control set in
the programme and it costs nothing because it already exists.

### `G-E57b` — the unscored arms
> For `05b_tq`, `15b_tq` and `15b_f32`: matched tokens out of 160, **and the position of the
> first divergence per prompt**. Greedy divergence compounds, so a match *count* alone hides
> whether an arm failed at token 1 or token 31.
>
> **`FAITHFUL`** only at **160/160**, the same bar `G-E53c2` had to clear. Anything else is
> reported as its count and is **not** called faithful.

### `G-E57c` — the speed table
Median, p25, p75 and interquartile width per arm, judged by **`G-E55a2`** — the median of
`|half − full| / full` over **200** random splits against a 20% bar, registered in E55 A.2 as a
variance reduction after `G-E55a`'s single split was shown to be a coin flip (P(PASS) 38–60%).
These are fresh cells, which is the only condition under which E55 A.2 permits it.

### `G-E57d` — the joint claim, which is the whole point
> An arm **DEMONSTRATES THE TARGET** only if it is `FAITHFUL` **and** its p25 is ≥ 50 tok/s.
> **DEMONSTRATES EXCELLENT** at ≥ 100 tok/s.
>
> Speed from a non-faithful arm and quality from a slow one may **not** be combined into a claim.

That last sentence is the entire reason this brief exists.

## 6. Predictions

| | prediction |
|---|---|
| `G-E57a` | all three reproduce exactly — they are stored constants |
| `05b_tq` quality | **160/160 FAITHFUL.** The body is ternary in both A1-adjacent arms and E6 showed ternarising the *body* is survivable; it is the *head* that breaks A2. |
| `15b_tq` quality | **160/160 FAITHFUL**, same reason |
| `15b_f32` quality | 160/160 |
| `G-E57d` at `05b_tq` | **DEMONSTRATES THE TARGET** — 42.83 is below 50 on one cell, but that cell was unreplicated; I expect p25 in **44–50** and therefore a **narrow miss** |
| `G-E57d` at `15b_tq` | below target — p25 near 18 |
| best faithful arm overall | **`05b_tq`, 42–48 tok/s** |

**I am predicting that nothing demonstrates the target.** If `05b_tq` is faithful at ~43 tok/s,
the honest headline becomes *"a trained LLM runs correctly on this engine at 43 tok/s at 0.5B,
14% short of the good bar, and the 50 tok/s claim rests on sparsity no trained model has yet
survived."* That is a worse headline than the programme has been carrying and it is the one the
evidence would support.

## 7. What this does not do

It does not test a 10B trained model — none exists locally, and acquiring one is a separate
decision with a real download cost. It does not test the carve on trained weights (that is H1).
It does not change the engine. **It measures what is already on this disk and has never been
read.**
