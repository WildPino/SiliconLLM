# Stage 1 (a), part 2: decode-only k = 2 with k₂ = 16, untrained and after a short block-wise training. PROTOCOL

2026-11-01. Follows part 1b (`S1B_MASS_DECOUPLED_RESULT_20261101.md`): decode-only k = 4 with k₂ = 16 fits the share
untrained, so the registered rule sets training_k = 2, at decode, from k₂ = 16. Same panel, reference outputs, labels and
chat gate as part 1 (`S1A_FEWER_EXPERTS_PROTOCOL_20261030.md`).

## Question

Can decode-only k = 2 fit the budget, and the 2.5-point share, untrained or after a short training of a few GPU hours
on the local RTX 3060? On the desk model k = 2 at decode reads 946 MB per token, for a ceiling of 29.6 tok/s. k = 4
reaches 25.1, a 1.6% margin over the 24.7 threshold; k = 2 would leave 20%.

## Literature (read in the papers' tables)

| Source | Setting | What it measured |
| --- | --- | --- |
| arXiv 2609.04575, Tables 5-6 (part 1b) | training-free, k₁ = k/2 | the mass of the top 2k recovers most of the loss; part 1b confirmed it on Ling-mini at k = 4 |
| ZEDA, arXiv 2605.18643, Table 1 | Qwen3-30B-A3B, k halved, full-parameter self-distillation, 60k prompts | fixed k/2 everywhere: 72.3 against 74.9 (−2.6); ~80 H200-hours (Table 2) |
| ZEDA, Table 6 | the same, with and without renormalizing the kept weights | without 73.3, with 71.6 |
| arXiv 2609.06076, Tables 5 and 8 | pruning/merging, 1,024-3,000 samples × 512 tokens | full FT recovers 37% of the gap; router-only 2.5%; FT beats KD |
| MoE-Pruner, arXiv 2410.12013, Table 4 | Mixtral-8x7B, 50% pruned, expert-wise MSE distillation | 67.23 → 68.40, against 69.16 |

Known ways to fail:
- **Router-only training recovers almost nothing** (2609.06076).
- **A fixed reduced k keeps a residual loss** even after large training (ZEDA, −2.6 at k/2).
- **No paper reaches k/4** (2 of 8), and none trains block-wise against layer outputs at a reduced k: the block-wise
  objective comes from MoE-Pruner's expert-wise distillation, applied to the whole block.

## Step 1: top2dm16 untrained

Arm **top2dm16** on P0: decode at k = 2 with k₂ = 16, the prompt at k = 8. Its score panels are ref's outputs by
construction (prompt positions only), copied as in part 1. About 45 minutes of CPU. Step 2 runs only if top2dm16 does
not fit the share.

## Step 2: training

**Model (`s1t_model.py`).** Ling-mini in plain torch, one layer at a time, with the hook's routing forms. Parity with
llama.cpp on part 1b's 2 dry-run documents (`results/s1t_parity.json`):
- **top-1 agreement at k = 8:** 0.974, bf16 against Q4_K_M;
- **target hit:** 0.8797 against 0.8795;
- **the forms' agreement with k = 8, torch against the hook:** top4 0.922 vs 0.924, top4m8 0.955 vs 0.953,
  top4m16 0.933 vs 0.931.

**Training (`s1t_train.py`), sequential and block-wise.** The teacher stream is the model at k = 8. The student stream
carries the layers already trained, run at the form. For each MoE layer:
- **the target** of the routed experts is the teacher's layer output minus the student's attention and shared-expert
  output, so the layer also absorbs the error carried in from earlier layers;
- **the trained weights** are the router weight and the 256 experts: fp32 masters, bf16 compute, Adafactor, batch
  4,096 tokens, 1 epoch. The selection bias is not trained;
- **the loss is measured where later layers read it:** each dimension is weighted by g² averaged over every later
  RMSNorm, and each token is divided by the mean square of its teacher residual;
- **a second term** (weight 1) is the same loss at k = 8 with the trained weights: decode-only runs the prompt at k = 8
  on the same weights.

**Export (`s1t_export.py`).** The trained tensors overwrite their bytes in a copy of the bf16 GGUF; `llama-quantize`
makes Q4_K_M as P0 was made, with P0's imatrix.

**Arm trained.** Generation panels at decode-only k = 2 with k₂ = 16. Score panels run at k = 8 on the trained model,
because training changed the weights the prompt runs on. That is how the k = 8 path is checked.

| Step | What | Cost |
| --- | --- | --- |
| lr | Adafactor relative lr in {1e-3, 3e-3, 1e-2}, layer 1, the training loss on held-out corpus tokens (no panel item) | ~0.3 GPU-h |
| train | 1.5 M tokens, 75 k held out, layers 1-19 | ~2 GPU-h |
| export | patch + quantize | ~15 min |
| panels | the trained arm, 7 panels | ~2.5 h of CPU |

**Data:** `s3/corpus_v1`, the first 1.5 M tokens in its balanced order (wiki, edu and math in thirds), decontaminated
against every panel by 13-token overlap (C50-S3 protocol).

## Controls before any run

| Control | Result |
| --- | --- |
| bf16 GGUF = HF checkpoint, byte for byte, for every tensor training touches (95) | PASS (`results/s1t_export_check.json`) |
| the export with nothing trained reproduces P0 | PASS: sha256 `44de7315…` (`results/s1t_export_control.json`) |
| torch forms against the hook | PASS (above) |
| verdict selftest, 5 planted cases | PASS |

## Found in the smoke tests (40k tokens, layers 1-3), before registration

- **A plain MSE is the wrong objective.**
  - Layer 1's routed output has a mean square of 50.5, against 0.03-0.08 at layers 2-3: it writes the massive
    activation (dim 1149).
  - 78% of layer 1's carried error sits in dim 1149, whose norm gains are low (0.031, median 0.085).
  - The weighted loss above follows from that.
- **k = 2 is about twice as hard as k = 4.** On held-out tokens, layer 1 at k₂ = 16 has an untrained error of 0.347
  of the target's energy (k = 4: 0.164), and training brings it to 0.282 (k = 4: 0.085).
- **The k = 8 term costs that path little:** its error goes from 0.00004 to 0.020 at layer 1.
- **At 40k tokens the training overfits.**

## Predictions (registered before step 1)

| Arm | Predicted Δ points (ref 55.19) |
| --- | --- |
| top2dm16 | −6 to −20, not WITHIN SHARE; its chat may fail the gate |
| trained | recovers 20-50% of top2dm16's loss on generation; its k = 8 score panels lose 0-2 points; Δ −4 to −14 |

The most likely outcome is "k = 2 does not fit" or "fits the budget only". The decision this changes is the k that
stage 2 carries; see below.

## Decision rule (`s1t_verdict.py`, separate from the runner)

- **Labels and chat gate:** as in part 1, against part 1's ref, on the same bootstrap draws.
- **TRAINING HELPS** iff the CI95 of Δ(trained) − Δ(top2dm16) lies above 0.
- **Outcome:**
  - **k = 2 fits the share, untrained:** top2dm16 FITS THE SHARE, and step 2 does not run;
  - **k = 2 fits the share, trained:** the trained arm FITS THE SHARE;
  - **k = 2 fits the budget only:** an arm FITS, but none fits the share;
  - **k = 2 does not fit:** otherwise.

## What a result changes (rule 5)

- **k = 2 fits the share:** stage 2 carries decode-only k = 2 (ceiling 29.6 tok/s), and, if trained, the recipe, whose
  transfer to flash it checks.
- **Otherwise:** stage 2 carries decode-only k = 4 with k₂ = 16 (ceiling 25.1, 1.6% margin). The stage 1 delivery
  records that k = 2 does not fit with a few GPU hours of this training at this scale.
- **Lever (a) does not fail either way:** four experts fit at decode.

## Thesis line (rule 4)

This cell changes no bytes beyond k itself. It decides between a 1.6% and a 20% margin on the desk ceiling. It runs
at Ling-mini's scale, and flash is bigger:
- **a flash layer** has 3.22 B routed parameters, against 0.81 B; that is 12.9 GB in fp32, too big for a 12 GB GPU
  whole, so it would train in chunks of experts;
- **its cost** is ~4× per token per layer, over 31 MoE layers: ~6.5× the GPU time at the same tokens, about 13 GPU-h on
  this GPU.

## What this protocol takes from, or changes in, the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| "router and/or experts" | both, jointly | router-only recovers 2.5% of the gap (2609.06076) |
| end-to-end training on the final logits | not here | the bf16 model (32 GB) and its gradients do not fit 12 GB; block-wise needs one layer (~11 GB peak) |
| training at decode only | approximated | the student stream runs every corpus position at k = 2 (stricter than decode-only); the k = 8 term keeps the prompt path; the k = 8 term's inputs come from the k = 2 stream, not from a k = 8 one |
| a chat-format corpus | not here | the decontaminated corpus is plain text; the chat gate guards chat |
| the imatrix recomputed on the trained model | not here | P0's imatrix is reused, so the export control reproduces P0 bit for bit |
| the selection bias trained | not here | top-k is not differentiable; the bias balances load |
| a dynamic k (ZEDA's zero experts), or k = 3 | not here | k = 3 is not in the stage's series (6, 4, 2); ZEDA's dynamic k needs the group loss and far more data |
| several lrs, epochs or k₂ on the panel | not here | lr is picked on held-out corpus tokens only; k₂ = 16 is part 1b's winner, not tuned at k = 2 |
| an independent reviewer | not needed by rule 6 | ~2.3 GPU-h, no new primary metric |
