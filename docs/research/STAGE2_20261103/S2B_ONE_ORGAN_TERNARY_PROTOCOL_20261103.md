# Stage 2, part 1, test 2: one organ of Ling-mini ternary with a scale per row, trained alone. PROTOCOL

2026-11-03. The second of stage 2's cheap tests. Stage 1 (b) trained the routed experts and found that layer 10 alone
costs almost nothing (`../STAGE1_20261030/S1D_TERNARY_EXPERTS_RESULT_20261103.md`: top-1 0.938, KL 0.034). The other
organs had no measurement at ternary. The desk ceiling needs all of them (`../STAGE1_20261030/S1_DELIVERY_20261103.md`).

## Question

For each organ of Ling-mini, how much does it cost when it is ternary in IQ2_BN's layout (codes {−1, 0, 1}, one scale
per row) and trained alone on the original model's inputs, as stage 1 (b)'s layer 10 was? The organs are:
- attention;
- the shared expert;
- the dense FFN;
- the routed experts with one scale per row instead of one per 256 weights;
- the head: test 1 read HEAD NEEDS TRAINING (`S2A_HEAD_TERNARY_RESULT_20261103.md`).

## Thesis line (rule 4)

Ling-flash shares the organs of Ling-mini, at twice the width. Every organ ternary is the 25.6 tok/s line of the desk
ceiling. The best case with any one organ left as stored is 18.1-24.6. One layer within its share does not show that
the stacked organ is: stage 1 (b) showed that stacking costs more than the sum. A layer over its share does show that
this training does not keep the organ ternary. That is the stage's case for talking with the Consulente before part 2.

## Literature (read in the papers)

| Source | What it gives here | Known ways to fail |
| --- | --- | --- |
| EfficientQAT, arXiv 2407.11062, §3.2, Tables 5, 12 | Block-AP: each block trained alone on its reconstruction, every weight and the step sizes trained; at 2 bits better than training the rounding or the clipping only (Table 5) | overfitting with few samples: a train/validation gap of 1.07 that falls to 0.06 with more samples (Figure 3); coarser groups cost: Llama-2-7B at 2 bits, group 32 → 256: perplexity 7.59 → 8.18, accuracy 60.28 → 58.67 (Table 12); the head stays FP16 (§E) |
| BitDistill, arXiv 2510.13998, §3.1 and §3.3 | ternary weights with a per-tensor absmean scale; SubLN, a norm inserted before the attention's and the FFN's output projections | without SubLN, the fine-tuning to 1.58 bits is unstable (Figure 3a); the gap to the full-precision model grows with size unless continued pre-training comes first |
| Stage 1 (b) (this repo) | the quantizer, the loss in the space the later norms read, the outlier rule, the one-layer setting | the error from stacking, not one layer; the error carried in jumps at layer 6 (test 3) |
| Test 1 (this stage) | the head: an untrained per-row fit gives top-1 0.698 (head_tern_im) | an error proxy that ignores the inputs' correlations |

Coarser groups in Table 12 go only to 256. The rows here are 512 to 5,120 weights long. One scale per row is coarser
than anything measured there.

## Found before registration (dry runs, 30 k training tokens, 9.5 k held out)

Each dry run is the trainer at a 50th of the data, with one lr. The table gives the held-out loss over the loss with
the organ zeroed: 0 is the original, 1 is the organ removed.

| Organ | Quantizer | Untrained | After the dry run's steps |
| --- | --- | ---: | ---: |
| exps | (b)'s: group 256, TWN, outlier rule | **0.353** | 0.301 |
| exps | row, Lloyd, no rule | 0.337 | 0.291 |
| shexp | group 256, Lloyd | 0.690 | 0.640 |
| shexp | row, TWN, no rule | 0.785 | 0.737 |
| shexp | row, Lloyd, no rule | 0.708 | 0.657 |
| shexp | row, Lloyd, rule (139 rows) | 0.678 | 0.636 |
| attn | group 256, Lloyd | 0.598 | 0.434 |
| attn | row, TWN, no rule | 0.597 | 0.436 |
| attn | row, Lloyd, no rule | 0.611 | 0.443 |
| dense | row, Lloyd, no rule | 0.282 | 0.254 |
| dense | row, Lloyd, rule (20 rows) | 0.288 | 0.260 |

- **The measure reproduces stage 1 (b).** (b)'s quantizer on layer 10's experts gives 0.353, against the 0.352 that (b)
  logged untrained.
- **One scale per row costs little.** Against group 256 it adds 0.018 on the shared expert and 0.013 on attention. On
  the experts it is lower, with Lloyd's start.
- **The shared expert and attention lose far more than the routed experts:** 0.69 and 0.60 against 0.35, untrained.
- **The outlier rule changes little.** Its effect runs −0.03 to +0.01, and it is off. With one scale per row, a row
  that keeps its max loses its other weights.
- **The head:** untrained, held-out top-1 0.690, which head_tern_im's 0.698 on the documents matches; 0.796 after 29
  steps at 3e-3. It fits on the 12 GB GPU at batch 1,024.
- **The export, on two dry-run organs** (the head and the shared expert): every replaced tensor TQ2_0 and bit-exact,
  the layout equal to the HF checkpoint's, none of the other 276 and 275 tensors changed, the head's gain exact. About
  44 min each.

The dry runs' records are in `benchmarks/stage2/results/s2o_dry/`.

## Amendment 1 (2026-11-03, before any panel of org_attn)

- **What C2 caught:** org_attn's first export changed four tensors outside the organ: attn_qkv of blocks 11, 13, 14
  and 16. Every other check passed.
- **Cause, in llama.cpp `5b335f4`'s `src/llama-quant.cpp`:**
  - a manual `--tensor-type` skips `llama_tensor_get_type_impl`, which advances the per-layer counters (here
    `i_attention_wv`);
  - Q4_K_M's `use_more_bits` then picks the later layers of the same class one step off.
- **Fix in `s2o_export.py`:** every other quantized tensor is pinned to P0's type by exact name, with
  `--tensor-type-file`. No tensor goes through the mixture, and C2 is unchanged.
- **What stands and what is VOID:**
  - The chain was stopped before any org_attn panel was written.
  - The first export is VOID and kept: `s2o_export_org_attn_void1.json`, with the file in
    `E:/SiliconLLM_Artifacts/stage2/s2o/void1/`.
  - org_dense's export passed C2 with no other tensor changed, so it stands.
  - The trainings stand: they do not depend on the export.

## Design

| Arm | Organ, layer | Trained tensors | Panels |
| --- | --- | --- | --- |
| org_attn | attention, layer 10 | the fused QKV (3,072 × 2,048) and the output projection (2,048 × 2,048) | documents |
| org_shexp | shared expert, layer 10 | gate, up (512 × 2,048), down (2,048 × 512) | documents |
| org_exps | routed experts, layer 10 | the 256 experts' gate, up, down, one scale per row | documents |
| org_dense | dense FFN, layer 0 (the only one) | gate, up (5,120 × 2,048), down (2,048 × 5,120) | documents, chat |
| org_head | head (test 1: NEEDS TRAINING) | lm_head (157,184 × 2,048) | documents, chat |
| org_attn_shuf | planted | org_attn with its output projection's rows permuted (seed 20261103) | documents |

Training (`benchmarks/stage2/s2o_train.py`), one organ at a time on the RTX 3060:
- **Every other weight** is the original bf16 one. The routing is k = 8. The organ's inputs come from the original model.
- **Quantizer:** stage 1 (b)'s, with the group = the whole row.
  - Latent fp32 weights train through a straight-through estimator. Each row's scale gets LSQ's gradient.
  - The scale starts at the fixed point of s = mean(|w| where |w| > s / 2), from TWN's: the least-squares scale for
    the rounding the quantizer applies.
  - (b)'s outlier rule is off.
  - Both choices come from the dry runs above.
- **Loss:**
  - for the layer organs, the layer's output against the teacher's, in the space the later norms read (stage 1 (b));
  - for the head, KL(teacher ‖ student) on the full vocabulary, without a column scale (c = 1): in test 1,
    head_tern_col's top-1 was not above head_tern_im's.
- **Data:** stage 1's corpus, 1.5 M tokens for training and 75 k held out, one epoch, Adafactor.
  - Batch 4,096 tokens: whole sequences for attention, 1,024 tokens for the head.
- **lr:** 1e-3, 3e-3 and 1e-2 (stage 1 (b)'s grid) each train from the same start. The pick is the least held-out loss, chosen by the
  trainer before any export or panel.
- **Logged, held out:** the loss with the original weights (the floor), post-hoc (no training), trained, and with the
  organ zeroed (the loss's scale). For the head, also top-1 and KL.

Export (`s2o_export.py`):
- The trained rows go into the bf16 GGUF, which is quantized as P0 was (Q4_K_M, fit24 imatrix), with the organ's
  tensors forced to TQ2_0.
- One scale per row means every 256-weight block has the row's scale as its maximum, so TQ2_0 holds the rows exactly,
  as IQ2_BN does.

Panels: `s2o_run.py`, stage 1's tools and inputs, as in test 1.

## Controls before any label (`s2o_verdict.py`, selftest PASS, 11 cases)

| Control | Rule |
| --- | --- |
| C1 planted | org_attn_shuf has top-1 < 0.90 and KL > 0.10: the rule fires on a known one-layer damage |
| C2 exports | each replaced tensor: layout equal to the HF checkpoint's before patching, TQ2_0 and bit-exact after; every other tensor equal to P0's; the head's gain equals P0's over c (max relative error < 1e-6, read from the files) |
| noise floor | replay: top-1 1.0, KL 4e-10 |

A failed C1 or C2 makes the run VOID.

## Decision rule (registered)

One organ's share is 1 point of 5. On the documents the line is stage 1's top6 anchor: top-1 ≥ 0.90 and KL ≤ 0.10.
- **One layer of a stacked organ** (org_attn, org_shexp, org_exps): ONE LAYER WITHIN THE SHARE at the line; ONE LAYER
  OVER THE SHARE otherwise.
- **A whole organ** (org_dense, org_head): FITS at the line, with chat passes ≥ ref's − 4 and degenerate rollouts
  ≤ ref's + 3; DOES NOT FIT otherwise.
- **Stage flag:**
  - any OVER or DOES NOT FIT gives "AN ORGAN DOES NOT STAY TERNARY WITH THIS TRAINING", and the Consulente is heard
    before part 2;
  - otherwise "EVERY ORGAN TESTED WITHIN ITS SHARE, ONE LAYER AT A TIME".
- **Reported, not judged:**
  - org_exps against tern_l10 (group 256): the price of one scale per row;
  - the trainers' held-out losses.

## Predictions (registered after the dry runs, before any full training)

| | Predicted |
| --- | --- |
| org_attn | top-1 0.90-0.94, KL 0.04-0.10; WITHIN 55% |
| org_shexp | top-1 0.90-0.95, KL 0.03-0.09; WITHIN 60% |
| org_exps | top-1 0.93-0.945, KL 0.03-0.045, within 0.01 of tern_l10's top-1; WITHIN 90% |
| org_dense | top-1 0.90-0.96, KL 0.02-0.08, chat 24-29; FITS 60% |
| org_head | top-1 0.85-0.93, KL 0.08-0.25; DOES NOT FIT 55% |
| org_attn_shuf | top-1 under 0.80 |
| stage flag | AN ORGAN DOES NOT STAY TERNARY 70% |

## What a result changes (rule 5)

- **The flag:** the Consulente is heard before part 2, with the organ that does not stay and what training would have
  to buy for it.
- **Otherwise:** part 2's design sees every organ ternary, and the stacking is its question.
- **org_exps against tern_l10:** whether IQ2_BN's single scale costs the experts anything. TQ2_0's per-256 scale has
  the same kernel speed class (stage 1 (c)).

## Parts specific to Ling

- **Specific to Ling:**
  - the fused QKV with q/k RMSNorm and RoPE on half of each head;
  - one dense layer, layer 0;
  - a shared expert of 512 beside 256 routed experts of 512;
  - the sigmoid router with a selection bias;
  - layer 10 as the middle of 19 MoE layers;
  - the vocabulary.
- **Not specific:**
  - the per-row ternary training and its loss need only RMSNorm before each block;
  - the head's KL needs an untied head.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| IQ2_BN's kernel | the panel runs TQ2_0 with the same values | the panel's llama.cpp has no IQ2_BN |
| every layer of each organ | one layer (layer 10, layer 0 for the dense FFN) | the stage asks for one layer each; the stacking is part 2's |
| Ling-flash | Ling-mini | conversions of flash are outside the stage |
| SubLN and attention-relation distillation (BitDistill) | not used | SubLN changes the graph the engine runs; the layer's output already carries the attention's error |
| the five-panel points | documents (chat for whole organs) | as test 1 |

## Cost

- **GPU, RTX 3060:** about 6 h, from the dry runs' rates.
  - The layer-10 organs take about 1 h each: 40 min of prefix and 3 lr.
  - The dense FFN takes about 20 min.
  - The head takes about 2 h: 75 min of prefix (20 layers) and 3 lr of about 12 min.
- **CPU:** about 9 h.
  - Six exports, about 44 min each (the dry runs').
  - Six document panels, about 37 min each.
  - Two chat panels.
- **Total:** under the 10 GPU-hours that call for an independent reviewer. No new metric.
