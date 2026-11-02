# Stage 1 (b): Ling-mini's routed experts in ternary, retrained block-wise. PROTOCOL

2026-11-02. Follows stage 1 (a) (`S1C_SHORT_TRAINING_RESULT_20261102.md`): stage 2 carries decode-only k = 4 with
k₂ = 16. Same panel, reference outputs, labels and chat gate as part 1 (`S1A_FEWER_EXPERTS_PROTOCOL_20261030.md`).
The test is the one the stage 0 map (`WALL_MAP_RESULT_20261029.md`, §5) named for the routed experts.

## Question

Do ternary routed experts, trained briefly, fit the quality budget? The stage's criterion is per layer: one layer
must cost at most 1/19 of the 5-point budget, which is 0.26 points. The panel cannot see 0.26 points (its CI95 is
about ±3), so the cell reads the criterion over all 19 layers: **with every MoE layer ternary, Δ against the whole
budget.**

## Thesis line (rule 4)

The k = 4 ceiling of stage 1 (a), 25.1 tok/s, assumes ternary experts. Desk model, Ling-flash at context 2048, decode
at k = 4, every other organ ternary (`WALL_MAP_RESULT_20261029.md`, §1):

| Routed experts as | Expert MB per token | Total MB | Ceiling, tok/s |
| --- | ---: | ---: | ---: |
| ternary, 0.25 B/weight | 390 | 1,141 | 25.1 |
| Q2_K, 0.33 B/weight | 512 | 1,263 | 22.9 |
| Q4_K, 0.56 B/weight | 942 | 1,693 | 17.5 |

Without ternary experts, no measured lever reaches 24.7. The cell runs at Ling-mini's scale. Ling-flash has the same
expert shape class (256 experts, top-8) at 4× the width.

## Literature (read in the papers)

| Source | Setting | What it measured |
| --- | --- | --- |
| CAT-Q, arXiv 2606.26650, Table 1 | post-training ternary (group 128) of attention and MLP, routers kept; 512 × 2,048 C4 tokens; 8-60 h on 8 A100 | Ring-flash-2.0 (100B-A6.1B) 74.88 → 64.68; Qwen3-30B-A3B 72.64 → 62.55 (five commonsense tasks) |
| QMoE (stage 0 map, §4) | Switch, experts only, no training | ternary +6.7% relative C4 loss |
| Super Weight, arXiv 2411.07191, Table 1 | Llama-7B: one weight of an early `down_proj` writes the massive activation | removing it: zero-shot 70.11 → 35.14, Wiki-2 PPL 5.67 → 1211; removing the next 7,000 outliers: 69.22 |
| TWN, arXiv 1605.04711 | least-squares ternary scale for a threshold Δ | α = mean of \|w\| above Δ; Δ ≈ 0.75·E\|w\| (normal), 2/3 (uniform) |
| MoTE, arXiv 2506.14435 (abstract only) | ternary routed experts beside a full-precision shared expert | "comparable to" the full-precision MoE, after up-cycling and training |

Known ways to fail:
- **Post-training ternary loses about 10 points** even at 100B scale with tens of A100-hours (CAT-Q).
- **One weight can carry the model** (Super Weight): ternary rounding shrinks it.
- **MoE models are more sensitive than dense ones** of similar size (CAT-Q, §3).

## Found before registration (smoke tests, 40k tokens, layers 1-3)

- **Ling-mini has a super weight.**
  - Where: layer 1, expert 13, `down_proj[1149, 256]` = 2.52.
  - What it does: with an intermediate activation of −564, it writes −5,985 into dim 1149 on one token in 512.
  - What ternary does to it: the group's TWN scale is 0.21, so it becomes 0.21. The student stream then loses the
    massive activation, and the error carried into layer 2 has mean square 43.7 (stage 1 (a)'s k = 2: 0.065).
- **The loss did not see it.** The loss of stage 1 (a) divides each token by its mean square, so that token weighs
  almost nothing. The exact normalized loss below did not see it either: after RMSNorm, −500 and −6,000 both read as
  "nearly all in dim 1149".
- **The fix: outlier groups keep their maximum.**
  - Where a group's max |w| exceeds 8× its TWN scale, the scale is that max, and it is not trained. The outlier
    is then exact, and the format stays ternary.
  - It touches 5,588 groups of about 60 M (0.009%).
  - With it, layer 1's expert 13 writes −4,764 (ternary all over), and the error carried into layer 2 drops to 2.23.

## Design

**Format.** w = s·q, q ∈ {−1, 0, +1}, one scale s per 256 consecutive weights along the input dim. This is exactly
llama.cpp's TQ2_0 (2.06 bits per weight): its block scale is the block's max |w|, which is s. The router, the shared
expert, attention, the dense layer 0 and the head stay as in P0 (Q4_K_M).

**Training (`s1t_train.py --ternary 256`), sequential and block-wise, as in stage 1 (a) part 2:**
- the teacher stream is the original model; the student stream carries the layers already made ternary;
- the target is the teacher's layer output, so each layer also absorbs the error carried in;
- the latent weights start at the original ones, and s at the TWN scale (Δ = 0.7·mean|w|);
- the gradient passes through rounding inside [−1, 1] (STE), and s gets the LSQ gradient;
- the loss is measured where later layers read it: the student's residual and the teacher's are each divided by their
  own RMS, and the difference is weighted per dim by g² averaged over every later RMSNorm;
- Adafactor (relative lr), batch 4,096, 1 epoch over 1.5 M tokens (75 k held out); the corpus is the same as in part 2.

**lr.** Adafactor relative lr in {1e-3, 3e-3, 1e-2}, picked on layer 1 by the loss on held-out corpus tokens.

**Export (`s1t_export.py patch --ternary`).** s·q goes into the bf16 GGUF, and `llama-quantize` makes Q4_K_M as P0 was
made, with `--tensor-type blk.L.ffn_{gate,up,down}_exps=tq2_0` for the ternary layers. The output is read back, and
every TQ2_0 tensor must dequantize to the trained s·q bit for bit.

**Arms (`s1b_run.py`, every panel at k = 8):**
- **tern:** all 19 MoE layers, trained, all 7 panels. This arm is judged.
- **tern_ph:** all 19 layers post-hoc (initial scales, outlier rule included, no training), documents only.
- **tern_l10:** layer 10 alone, trained on the original model's inputs (`--alone`), documents only. This is the
  stage 0 map's literal test.

| Step | What | Cost |
| --- | --- | --- |
| lr | three lrs on layer 1 | ~0.7 GPU-h |
| train | layers 1-19; layer 10 alone | ~5.5 GPU-h |
| export | tern, tern_ph, tern_l10 | ~1 h CPU |
| panels | tern 7 panels; tern_ph and tern_l10 documents | ~4 h CPU |

## Controls before any run

| Control | Result |
| --- | --- |
| the export with nothing trained reproduces P0 | PASS (stage 1 (a), `results/s1t_export_control.json`) |
| ternary export of the smoke weights: 9 TQ2_0 tensors dequantize to s·q bit for bit | PASS: 9/9 exact (`results/s1b_export_smoke.json`) |
| the runner on the smoke export: CONFIG and MOE confirm mode none | PASS on nq (generate) and arc (score), 3 items each |
| verdict selftest, 5 planted cases | PASS |

## Predictions (registered before the lr pick)

| Arm | Predicted |
| --- | --- |
| tern | Δ −8 to −30, OVER BUDGET more likely than not; chat fails the gate |
| tern_ph | document top-1 under 0.5 |
| tern_l10 | dBPB +0.003 to +0.03; below tern's dBPB / 19 is not predicted either way |

## Decision rule (`s1b_verdict.py`, separate from the runner)

- **Labels and chat gate:** as in part 1, against part 1's ref, on the same bootstrap draws.
- **Outcome:**
  - **ternary experts survive:** tern FITS, i.e. WITHIN SHARE or WITHIN BUDGET, with chat OK;
  - **ternary experts closed:** tern is OVER BUDGET (CI95 upper end below −5);
  - **undecided:** otherwise.
- **Reported, not judged:** the document metrics of the three arms, and tern_l10's dBPB against tern's / 19.

## What a result changes (rule 5)

- **Survive:** stage 1 (c) measures the ternary kernel, and stage 2 carries ternary experts with this recipe.
- **Closed:**
  - stage 1 (c) measures the kernel at the next width still in play (2 bits);
  - the stage 1 delivery states that no measured lever reaches 24.7 (the table above), and the Consulente is consulted.
- **Undecided:** stage 1 (c) stays at ternary, since the kernel question is independent. The delivery states what a
  longer training would have to buy.

The prediction is "closed". It is registered with the decision it changes: the bit width of (c) and the delivery.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| one layer, against 1/19 of the budget | 19 layers against the budget; one layer reported on documents | 0.26 points is under the panel's resolution |
| "trained alone on dumped inputs" | sequential (tern), alone (tern_l10) | the sequential student absorbs the carried error; the literal test is kept as an arm |
| pure ternary | 0.009% of groups take their max as scale | a super weight; the format stays TQ2_0 |
| attention, shared expert, head ternary | not here | (b) is the routed experts; the others are separate tests of the map |
| the packed 1.6-bit layout of (c) | TQ2_0 (2.06 bits) | same values; the layout changes speed, not quality |
| end-to-end training on the logits | not here | does not fit 12 GB; block-wise needs one layer |
| more tokens or epochs | 1.5 M, 1 epoch | RAM (34 GB at 1.5 M) and comparability with (a) part 2 |
| an independent reviewer | not needed by rule 6 | ~6 GPU-h, no new primary metric |
