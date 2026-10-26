# C50-T2 trained stitch cell: can a map trained on F0's experts' output make them read v0's state? (protocol)

26 October 2026. Registered before the cell runs: no v0 state has been read through a trained map on the main
documents. It is the second cell of option X.
- **The choice:** "1", a stitch trained locally on F0's experts' output, after the linear stitching FAIL
  ([result](C50_T2_STITCH_RESULT_20261026.md); §12).
- **Option X:** F0's own experts, frozen, are read by v0 through a learned map 2048 → 4096, inside a gated additive
  branch ([design](C50_T2_DESIGN_20261025.md)).

## Why this cell

**What the linear cell left open:**
- A ridge map fitted on the states recovered about half of F0's routing and little of its experts' computation.
- The fit target was the wrong one: R² on the state weighs every direction alike, while F0's router and experts
  read only some.

**This cell trains the map on what the experts output.** It answers directly whether any map in two simple
families (affine, and affine plus a small MLP) lets F0's experts read v0, before the gate and P_out are built.
- It runs on the local 12 GB GPU: one F0 layer's experts in bf16 are 6.4 GB.
- It uses the dumps already made, so no T4 time is spent.

## What is measured

**Same inputs as the linear cell:**
- the dumps `C:/SiliconLLM_stage/t2/main/v0` and `…/f0r`;
- the same 128 documents (ids sha256 `14a28e93…`), the same split by document: fit 64, select 32, test 32;
- positions 32..287: 16,384 / 8,192 / 8,192 rows.

**Pairs:** each v0 MoE layer l (1-19) at its depth-matched F0 layer j = round(l · 31/19).
- The linear cell's select-R² pairing put 17 layers on F0 layer 2, the easiest target. Depth-matched pairs are what
  the branch would use, and the post-hoc read them for the ridge.
- No pairing search: one training per arm per layer.

**F0's block on the GPU** (`c50_t2_tstitch.py`, class `Block`):
- `c50_t2_moe`'s routing, in torch;
- the experts dequantized by the pinned gguf-py, stored and multiplied in bf16, accumulated in float32.

**Arms, per layer.** Each starts from the ridge map of the linear cell: fit rows, λ on the select R².

| Arm | Map | Trained |
| --- | --- | --- |
| ridge | affine, the ridge itself | no (it reproduces the linear post-hoc) |
| linear | affine | yes |
| mlp | affine plus a residual MLP (2048 → 2048 → 4096, GELU, inputs standardised, output layer zero-initialised) | yes |

**Training:**
- **Loss:** (1 − cosine of F0's routed output on the mapped state against F0's routed output on F0's own state)
  + 0.1 · (router-logit MSE / logit variance).
- **Gradients:** the top-8 selection is the block's own and discrete; gradients flow through the selected weights,
  the experts and the logit term.
- **Optimiser:** AdamW, lr 3·10⁻⁴; weight decay 0 on the affine part and 10⁻² on the MLP; batch 512, 16 epochs.
- **Epoch selection:** the select cosine picks the epoch, and epoch 0 (the ridge start) is a candidate, so training
  never loses to the ridge on select.
- **Measured map of a layer:** the trained arm (linear or mlp) of higher select cosine.

**Measures on the test rows:**
- top-8 overlap with F0's own selection;
- mean cosine of the routed output;
- R² of the mapped state.

**Diagnostics, reported and not gated:**
- the same measures on 4,096 fit rows (overfitting);
- the linear arm trained on the first 32 fit documents only (is the cell data-limited?);
- the select cosine by epoch (was the budget enough?).

## Calibration on F0's own states, before this push (no v0 state involved)

The planted positive and the learning rate were fixed on the main documents, from F0's states alone. Records are in
`results/c50_t2_tstitch_cal/`, scripts `c50_t2_tstitch_cal_pca.py` and `c50_t2_tstitch_cal_positive.py`. The early
runs used 12 epochs and a float32 cosine.

**The linear ceiling at 2048 dimensions.** F0's own state reconstructed from its top-k principal components (fit
rows), read by F0's block on the test rows, as overlap / cosine:

| F0 layer | k = 1024 | 2048 | 3072 | 3584 | noise-0.5 bar |
| ---: | --- | --- | --- | --- | --- |
| 2 | 0.818 / 0.908 | 0.901 / 0.979 | 0.947 / 0.995 | 0.967 / 0.998 | 0.734 / 0.948 |
| 16 | 0.598 / 0.488 | 0.688 / 0.698 | 0.776 / 0.850 | 0.831 / 0.917 | 0.643 / 0.685 |
| 31 | 0.737 / 0.750 | 0.803 / 0.852 | 0.859 / 0.927 | 0.893 / 0.962 | 0.724 / 0.811 |

- **An affine map from a 2048-dim state lands in a 2048-dim affine subspace.** In the middle layers the best such
  subspace for R² clears the bar only just: at F0 layer 16, cosine 0.698 against 0.685.
- **So in the middle layers the linear arm can stitch only if v0's state carries F0's top components almost
  exactly.** The mlp arm is not bound by this.
- **Registered in advance:** a FAIL of the linear arm in the middle layers is expected from this alone, and says
  little beyond it.

**The planted positive, chosen on calibration:**

| Positive tried | lr | F0 layer 2 / 16 / 31: overlap / cosine | Bars (noise 0.5) |
| --- | --- | --- | --- |
| PCA-2048 input, from the constant mean | 10⁻⁴ | 0.686 / 0.817; 0.347 / 0.178; 0.683 / 0.667 | 0.734 / 0.948; 0.643 / 0.685; 0.724 / 0.811 |
| Rotated full state (4096), from the constant mean | 10⁻³ | 0.842 / 0.943; 0.652 / 0.507; 0.774 / 0.772 | the same |
| The same, layer 16 only | 3·10⁻³; 10⁻² | 0.567 / 0.460; diverged | |
| PCA-3072 input, from its PCA-1024 reconstruction, layer 16 | 10⁻³; 3·10⁻⁴ | 0.632 / 0.665; 0.695 / 0.691 | |
| **PCA-3584 input, from its PCA-1024 reconstruction** | **3·10⁻⁴** | **0.901 / 0.972; 0.719 / 0.722; 0.819 / 0.847** | |
| The same, layer 16 | 10⁻⁴ | 0.720 / 0.664 | |

- **From the constant mean, no lr reaches the bar in 12 epochs.** A map built from nothing is a harder task than
  the arms', which start from the ridge.
- **The registered positive is the last row at lr 3·10⁻⁴:**
  - its input is F0's own state on 3,584 components, so a map that stitches exists in the affine family;
  - its start, the 1024-component reconstruction, reads about as badly as the ridge maps do in the middle layers
    (0.598 / 0.488 at layer 16);
  - it stitches at layers 2, 16 and 31, with a cosine margin of 0.02-0.04.
- **The j = 16 file of that row was overwritten by the j = 2 / 31 run.** Its numbers come from the run's console
  output, quoted above.
- **lr 3·10⁻⁴ is the arms' lr too.** It beat 10⁻⁴ and 10⁻³ on the positive at layer 16.
- **The run at lr 10⁻² diverged, and the float32 cosine of its huge outputs came out far above 1.** The epoch
  selection would have chosen it. Two fixes followed:
  - the cosine is now computed in float64;
  - an epoch with a non-finite loss or a cosine outside [−1, 1] ends that training, and the best earlier epoch is
    kept (recorded as `diverged_at_epoch`).
- **The select cosine was still rising at the last epoch in every calibration.** So the budget is 16 epochs, not 12.
  The cell records the best epoch per arm.

**Tools on the dev set** (4 documents of shards 470-471, a fresh F0 dump of them, not evidence):
- the bf16 block against llama.cpp: top-8 overlap 1.000, median cosine ≥ 0.99982 at F0 layers 2, 16 and 31;
- the whole cell, smoke (`--smoke`, layers 1, 10 and 19): 230-340 s per run, 20-30 s per training.

## Run

| Step | Command (from the repo root) |
| --- | --- |
| Measure | `python -I benchmarks/chatbot50/c50_t2_tstitch.py --inputs C:/SiliconLLM_stage/t2/inputs --v0 C:/SiliconLLM_stage/t2/main/v0 --f0 C:/SiliconLLM_stage/t2/main/f0r --f0-gguf C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf --posthoc benchmarks/chatbot50/results/c50_t2_stitch_posthoc.json --json benchmarks/chatbot50/results/c50_t2_tstitch.json` |
| Verdict | `python -I benchmarks/chatbot50/c50_t2_tstitch_verdict.py benchmarks/chatbot50/results/c50_t2_tstitch.json --json benchmarks/chatbot50/results/c50_t2_tstitch_verdict.json` |

- `C:/SiliconLLM_stage/t2/inputs` is a copy of `t2/stitch/inputs` (sha256 of all three files equal).
- The measure checks the ids against the manifest.

## Rule (applied by `c50_t2_tstitch_verdict.py`, from the record only)

**Layer l stitches** iff the measured map reads, on the test rows, at least as well as F0's own state at noise R² 0.5,
on both measures. This is the linear cell's rule:

    overlap(map) ≥ overlap(noise 0.5)   and   cosine(map) ≥ cosine(noise 0.5)

| Verdict | Condition | Next |
| --- | --- | --- |
| **PASS** | ≥ 10 of 19 layers stitch | Option X goes to its training protocol: gate, P_out and v0's own CE, with the stitch maps of this cell as P_in |
| **PARTIAL** | 1-9 layers | X goes on those layers only, and the bytes estimate is redone for that L |
| **FAIL, DATA-LIMITED** | no layer stitches, and doubling the fit documents raised the linear arm's test cosine by ≥ 0.05 (median over layers) | The choice returns, with a rerun on more documents (a new dump) as one option |
| **FAIL** | no layer stitches, and the cell is not data-limited | Neither family of map lets F0's experts read v0 at this depth. Option X closes; the choice returns: N or A′ |

**STOP (no verdict)** if a check or a planted control fails, on any layer:
- **Tool:** the bf16 block on F0's own test states against llama.cpp's `ffn_moe_topk` and `ffn_moe_out`. It needs
  a top-8 overlap ≥ 0.97 and a median cosine ≥ 0.99.
- **Ridge:** the ridge arm reproduces the linear post-hoc record. It needs the same λ, and overlap and cosine within
  0.02.
- **Planted positive:** the PCA-3584 map, trained from its PCA-1024 start, must stitch.
- **Planted negative:** maps trained on shuffled fit pairs (linear and mlp, chosen on select like the measured map)
  must not stitch.
- **Self-test:** the verdict script's self-test (11 synthetic records) passes before the run.
- **A crash is a VOID,** reviewed before any rerun.

## Registered predictions (scored, not gated)

| Prediction | Value |
| --- | --- |
| Verdict | **FAIL**, not data-limited |
| Arm chosen | mlp on at least 12 of 19 layers |
| Test cosine gain over the ridge, median over layers | 0.05-0.20 |
| Middle layers (l = 4..16) | test cosine ≤ 0.50 on every one |
| Best stitching candidates | l = 1 (F0 layer 2) and l = 19 (F0 layer 31); if any layer stitches, it is one of them |
| Overfitting | fit-minus-test cosine ≥ 0.10 on most layers |
| Data gain (half → full), median | 0.02-0.05 |
| Planted controls | the positive stitches on all 19; no shuffled map does, with a shuffled cosine ≤ 0.55 everywhere |
| Wall | ≤ 6 h |

## What this cannot say

- **Two families of map, one budget, one seed, one draw of documents.** A larger map, more data or more epochs could
  do better. The diagnostics say which of these is binding: data gain, best epoch, fit-minus-test.
- **A PASS says F0's experts can be addressed from v0's state at one layer.** It does not say the branch adds
  knowledge. That is the training protocol's job, with its random-donor control and fact panels on plain draws.
- **The bar is strict and self-calibrated, as in the linear cell.** The calibration above shows how close even
  F0's own 2048-component state comes to it in the middle layers.
