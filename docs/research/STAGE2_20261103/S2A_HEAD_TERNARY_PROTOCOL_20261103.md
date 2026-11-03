# Stage 2, part 1, test 1: Ling-mini's head ternary with a scale per row, untrained. PROTOCOL

2026-11-03. The first of stage 2's cheap tests. Stage 1's delivery (`../STAGE1_20261030/S1_DELIVERY_20261103.md`)
found that the success line needs every weight-bearing organ at about 2 bits. The head is the organ with no published
measurement at that width.

## Question

Untrained, on Ling-mini with the rest of the model as stored (P0), how much does the head cost when it is ternary in
IQ2_BN's layout (codes {−1, 0, 1}, one scale per row)? Does it fit its 1-point share of the 5-point budget?

## Thesis line (rule 4)

Ling-flash's head is 157,184 × 4,096 (644 M weights). Desk ceiling at k = 4, every other organ ternary at its measured
IQ2_BN rate, full q8_0 KV (`benchmarks/stage1/results/s1_ceiling.json`):
- **25.6 tok/s with the head ternary** (4.14 ms per token at 155.63 G weights/s);
- **19.4 tok/s with the head as stored.**

The test runs on Ling-mini's head, which has the same vocabulary and half the width (2,048). A FIT makes the untrained
head a candidate at the thesis scale; flash's own head is checked when flash is converted.

## Literature (read in the papers and the code)

| Source | What it says about the head |
| --- | --- |
| ParetoQ, arXiv 2502.02631, §5.1 and A.3 | "all weights except for the embedding and output layers are quantized", "following the common practice" (GPTQ, LLM-QAT); in its CPU timings the output layer is 8-bit, channel-wise |
| arXiv 2608.28003, §6 | lm_head at INT8 (W8A8) on Gemma-3-1B: with 5+5 FFN layers, 98.90% top-1 agreement, +0.85% perplexity. Through TensorRT-LLM's standard path, generation "reproducibly broke" (incoherent, repetitive), or the exporter silently dropped the head's quantization |
| BitNet b1.58 2B4T, arXiv 2504.12285, §2 and Table 1 | replaces the linear layers with BitLinear; reports memory as "non-embedding" (0.4 GB) and does not state the head's precision |
| llama.cpp `5b335f4`, `src/llama-quant.cpp` | the output tensor stays Q6_K under every k-quant (Q5_K under the IQ1/IQ2 mixes, Q8_0 if the row is not a multiple of the block) |
| AWQ, arXiv 2306.00978, §3.2 | a per-input scale s = s_X^α, α on a grid in [0, 1] minimizing the output error, folded into the previous operator. Known failure: a large s raises the error of the other channels (Table 2). Measured at 3-4 bits |
| OneBit, arXiv 2402.11295, §3.2 | a 1-bit matrix between a row and a column value vector (FP16), initialized from a rank-1 fit of \|W\|, then trained by distillation; its limitation: the loss needs knowledge distillation to recover |

No measurement of a ternary or 2-bit head was found. Known ways to fail, and what covers each here:
- generation breaks while the scores look fine: the chat panel, with degenerate rollouts;
- the head's conversion is silently dropped: a planted control and a bit-exact check of the file;
- a few large inputs dominate the error: the weighted fit and the column scale below.

## Found before registration (`benchmarks/stage2/results/`)

- **The head's inputs are dominated by a few dimensions** (`s2h_imatrix.json`: llama-imatrix on P0, `--process-output`,
  the fit24 sample, 302 chunks of 512).
  - Mean square per input: 5.50 on average; 1,390 at dimension 1212.
  - The 8 largest inputs carry 33% of the total.
- **Proxy of the weighted error** (Σ h (w − t)² over Σ h w², h = the imatrix, 16,384 sampled rows):
  - unweighted fit: 0.366;
  - weighted fit: 0.169.
  - Column scale c = rms^α, by α:

    | α | −1 | −0.5 | −0.35 | −0.25 | 0 | 0.5 | 1 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | weighted error | 0.210 | 0.112 | **0.100** | 0.105 | 0.169 | 0.281 | 0.343 |

  - AWQ's direction (α > 0) makes it worse at ternary. The best α is negative: the inputs with large activations get
    a finer level.
- **The builds** (`s2h_build_*.json`) all pass C2 below: head exact in TQ2_0, gain as built, the other 276 tensors equal
  to P0's. α = −0.35 on the full grid (−1 to 1, step 0.05).

  | Arm | α | nonzero share | relative error | weighted error | q4 sha256 |
  | --- | ---: | ---: | ---: | ---: | --- |
  | head_tern_im | 0 | 0.179 | 0.566 | 0.169 | `ef30374e…` |
  | head_tern_col | −0.35 | 0.525 | 0.212 | 0.0999 | `67490b92…` |
  | head_shuf | −0.35 (rows permuted) | 0.525 | 0.212 | 0.0999 | `f2a0c99d…` |

  - The errors are means over rows.
  - Without the column scale the fit keeps 18% of the weights: the large inputs force a high threshold.

## Design

Every arm keeps P0 as stored except the head, and the routing at k = 8 (stage 1 (b)'s mode none).

| Arm | Head | Panels |
| --- | --- | --- |
| **head_tern_col** (primary) | per row, the weighted least-squares ternary fit (support = top-k by \|w\|, k and the scale from the fit) of W diag(c); the final norm's gain becomes g / c, an exact change of the float model; α from the grid (−1 to 1, step 0.05) on the proxy | documents, chat |
| head_tern_im | the same fit with c = 1 | documents |
| head_shuf | head_tern_col's rows in a random order (seed 20261103): the planted control | documents |

- The fit and the builds: `benchmarks/stage2/s2h_head.py`. Each build is quantized from the bf16 GGUF as P0 was
  (Q4_K_M, fit24 imatrix), with the head as TQ2_0. TQ2_0 holds a bf16 scale and ternary codes exactly, as IQ2_BN does.
  The panel's llama.cpp has no IQ2_BN.
- The panels are stage 1's (`benchmarks/stage2/s2h_run.py`, the same tools, inputs and checks).
  - Documents: 72 documents against ref, giving top-1, KL on ref's top-32 and paired dBPB.
  - Chat: 40 items, C50-A4's check and its degenerate-rollout flags.

## Controls before any label (`s2h_verdict.py`, selftest PASS, 8 cases)

| Control | Rule |
| --- | --- |
| C1 planted | head_shuf's document top-1 < 0.10: the panel sees the head |
| C2 builds | each file's head is TQ2_0 and dequantizes to the built rows bit for bit; the final norm's gain is the built one; every other tensor equals P0's byte for byte |
| noise floor | replay against ref on the documents: top-1 1.0, KL 4e-10 (stage 1's record) |

A failed C1 or C2 makes the run VOID. A temperature control would need the full logits, which the panel does not keep.
Top-1, which no temperature moves, is reported beside KL.

## Decision rule (registered)

The head's share is 1 point of 5 (one of five organs). On the documents it reads through stage 1's top6 arm, which
cost −0.96 points with top-1 0.898 and KL 0.094.
- **HEAD FITS ITS SHARE UNTRAINED:** head_tern_col has document top-1 ≥ 0.90 and KL ≤ 0.10; its chat passes ≥ ref's −
  4 (24 of 40); its degenerate chat rollouts ≤ ref's + 3.
- **HEAD NEEDS TRAINING:** otherwise.

## Predictions

Registered after the builds, before any panel:

| | Predicted |
| --- | --- |
| outcome | HEAD NEEDS TRAINING slightly more likely than FITS (55 / 45) |
| head_tern_col | document top-1 0.86-0.94, KL 0.06-0.25; chat 22-30 passes, at most 3 degenerate |
| head_tern_im | worse than head_tern_col on top-1 and on KL |
| head_shuf | document top-1 under 0.01 |

## What a result changes (rule 5)

- **FITS:** the head goes ternary untrained, with its column scale in the final norm. Test 2 trains attention, the
  shared expert and the dense FFN only.
- **NEEDS TRAINING:** the head joins test 2's one-organ trainings: the ternary head trained alone on saved final hidden
  states, against the original's logits.
- In either case the head is not closed: an untrained miss does not show that the head cannot be ternary.

## Parts specific to Ling

- **Specific to Ling:**
  - the vocabulary (157,184, shared by mini and flash);
  - the width (2,048 here, 4,096 in flash);
  - the input dimensions that dominate (1212 first);
  - P0's head at Q6_K as the reference.
- **Not specific:**
  - the per-row fit and the column scale folded into the final norm's gain need an RMSNorm with a gain right before an
    untied head;
  - a head tied to the embedding would carry the change into the embedding as well.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| IQ2_BN's kernel | the panel runs TQ2_0 with the same values | the panel's llama.cpp has no IQ2_BN; the speed is stage 1 (c)'s |
| the five-panel points | documents and chat only | the stage plan asks for top-1, KL and BPB first; points come with the whole model |
| Ling-flash | Ling-mini only | conversions of flash are outside the stage |
| AWQ's search on cached inputs | a diagonal proxy from the imatrix | the imatrix holds the inputs' mean squares, not their correlations; the panel judges the choice |
| training | none | that is test 2, if this one asks for it |

## Cost

CPU only, about 4 h:
- the imatrix: 27 min, done;
- three builds: about 20 min each;
- three document panels: about 40 min each;
- one chat panel: about 3 min.

The stage plan estimated 1 h. The difference is the full document panel per arm.
