# C50-T2 stitching cell: can F0's experts read v0's state? (protocol)

26 October 2026. Registered before the cell's dumps are made. It is the first cell of step 2 of decision C, option X
([design draft](C50_T2_DESIGN_20261025.md)).
- **The choice:** option X, and the gate may learn to switch off where T0 is sure ("X, yes", 25 October; §12).
- **Option X:** F0's own experts, frozen, are read by v0 through a learned map 2048 → 4096, inside a gated additive
  branch. The knowledge stays in the 100B's weights, and only the experts run, not F0's trunk.

## Why this cell sits where it does

**The thesis:** capacity added to v0 at nearly fixed active bytes, carried by the 100B's own weights.
- **Its first risk:** v0's state may not carry what F0's router and experts need. The literature read
  (Chen et al. 2025) finds affine maps between related models workable, but bottlenecked by the smaller model.
- **Untested setting:** none of the work read runs only the donor's experts.
- **The cheapest test:** a linear map, no training, CPU and the local GPU, a few hours. If no layer stitches,
  option X closes at linear stitching before any T4 time is spent.

## What is measured

**Documents:**
- `c50_t2_stitch_inputs.py` draws 128 documents of S3 corpus v1 from teacher shards 472-493: the validation slice,
  which no arm trained on, and outside the two shards its curves used. Each has at least 288 tokens and is cut to its
  first 288.
- **Split by document:** fit 64, select 32, test 32. Positions 32..287 of each document are used: 16,384 / 8,192 /
  8,192 rows.
- **Ids sha256** `14a28e93…` (main). Four documents of shards 470-471 form a dev set for checking the tools only.

**States:**
- `c50_llama_dump.cpp`, a new tool on the pinned llama.cpp (5b335f4), built by `c50_build_scorer.sh`
  (`c50_llama_dump_avx2.exe` sha256 `b961d7b2…`). It uses the scheduler's eval callback, so it
  changes neither the graph nor the numerics. It runs with S2's settings: KV F16, flash attention off, 6 threads, one
  document per batch.
- **What it dumps:**
  - every MoE layer's input `ffn_norm` (the normed state F0's router and experts read);
  - `ffn_moe_topk` and `ffn_moe_out` (the routed experts' sum, without the shared expert).
- **The models:**
  - v0 = `t1/t1_a1_q4_k_m.gguf` (`e490f803…`), layers 1-19;
  - F0 = Ling-flash-2.0 Q4_K_M (`5348c73f…`), layers 1-31, read from a staged SSD copy whose sha256 is checked.

**F0's block in numpy** (`c50_t2_moe.py`) reproduces the pinned `build_moe_ffn` for bailingmoe2:
- sigmoid gate on the F32 router;
- the bias for selection only, 4 of 8 groups by top-2 sum, then top-8;
- the selected sigmoid weights normalised and scaled 2.5;
- SwiGLU experts dequantized by the pinned gguf-py.

**Measure** (`c50_t2_stitch.py`):
1. **Pairing.** For each v0 layer l and F0 layer j, a centred ridge map with λ on a grid of 5 (relative to the mean
   squared singular value), fitted on fit rows. The R² on select rows picks j and λ for each l.
2. **Test, for each l at its pair:**
   - R²;
   - **top-8 overlap:** the share of F0's experts selected on the mapped state that F0 also selects on its own state;
   - **routed-output cosine:** F0's block on the mapped state against F0's block on its own state.
3. **Calibration, on each chosen j, 2,048 test rows:** F0's own state plus Gaussian noise with per-dimension variance
   (1 − r) · var, so that its R² is r = 0.3, 0.5, 0.7, 0.9. It shows what an R² buys in F0's own reading.

## Before this push: the tools on the dev set (not evidence of the cell)

- **The numpy block against llama.cpp on v0's dev states** (4 documents, 1,152 positions, layers 1, 10, 19):
  top-8 overlap 1.0000 on all three, median cosine 0.99985-0.99987, minimum 0.9993. F0 uses the same code, and the
  same check runs on F0 in the cell itself, as a STOP condition.
- **The whole pipeline, smoke** (`--smoke`, the dev documents split 2/1/1, with v0 standing in for F0):
  - it runs end to end in 411 s;
  - the pairing picks j = l on all 19 layers, as it should when the two sides are the same model;
  - the shuffled maps read overlaps of 0.03-0.14.
  - **The verdict script answers STOP** on it, because with 512 fit rows in 2,048 dimensions even the self map
    reaches only R² 0.14-0.31 and does not stitch. That is the planted positive working: too little data blocks the
    verdict.
- **Two infrastructure incidents, no number lost:**
  - the first F0 dev dump hung for 23 minutes on a read of drive E: (the USB disk stopped serving reads) and ended
    without output. F0 now runs from a staged SSD copy (`C:/SiliconLLM_stage/f0/`, sha256 `5348c73f…`, checked);
  - the second was stopped by the session host for low memory, since F0's 62 GB fill the 80 GB of RAM. F0's dump
    therefore runs from a terminal of its own.

## Run

| Step | Who | Command (from the repo root) |
| --- | --- | --- |
| v0 dump | in session | `benchmarks/chatbot50/bin/c50_llama_dump_avx2.exe --model t1/t1_a1_q4_k_m.gguf --ids t2/stitch/inputs/main.ids --out C:/SiliconLLM_stage/t2/main/v0 --names ffn_norm,ffn_moe_topk,ffn_moe_out --layers 1-19 --threads 6 --ubatch 512 --ctx 1024` |
| F0 dump | own terminal | the same with `--model C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf --out C:/SiliconLLM_stage/t2/main/f0 --layers 1-31` |
| Measure | in session | `c50_t2_stitch.py --inputs t2/stitch/inputs --v0 …/main/v0 --f0 …/main/f0 --v0-gguf … --f0-gguf … --json results/c50_t2_stitch.json` |
| Verdict | in session | `c50_t2_stitch_verdict.py results/c50_t2_stitch.json --json results/c50_t2_stitch_verdict.json` |

Paths under `t1/` and `t2/` are under `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/`. The dumps (about 50 GB)
stay out of git; the records go in.

## Rule (applied by `c50_t2_stitch_verdict.py`, from the record only)

**Layer l stitches** iff F0's experts read the mapped state at least as well as F0's own state at noise R² 0.5, on
both measures:

    overlap(mapped) ≥ overlap(noise 0.5)   and   cosine(mapped) ≥ cosine(noise 0.5)

**Why this bar:**
- **It calibrates itself on F0's own layer.** It needs no number guessed in advance, and it asks for a state that
  F0's experts treat as at most "half noise".
- **It is a strict bar.** In the smoke run below, v0's own block at noise R² 0.5 kept an overlap of 0.73-0.78 and a
  cosine of 0.81-0.92: isotropic noise of that size barely moves the reading. A regression error of the same R²,
  concentrated where v0 does not encode F0, can move it much more.
- **What follows from it:** a FAIL closes linear stitching, not option X. A trained stitch can still do better.

| Verdict | Condition | Next |
| --- | --- | --- |
| **PASS** | ≥ 10 of v0's 19 MoE layers stitch | Option X proceeds to its training protocol (stitch maps, gate, P_out), on the stitching layers |
| **PARTIAL** | 1-9 layers stitch | X proceeds on those layers only; the bytes estimate is redone for that L |
| **FAIL** | no layer stitches | Linear stitching does not support X. The choice returns with the numbers: N, A′, or X with a stitch trained end to end |

**STOP (no verdict)** if a check or a planted control fails:
- **The tool:** on every chosen j, F0's numpy block on F0's own state against llama.cpp's own `ffn_moe_topk` and
  `ffn_moe_out`. It needs a top-8 overlap ≥ 0.97 and a median cosine ≥ 0.99. The gap is llama.cpp's 8-bit
  activations.
- **Planted positive:** the same ridge pipeline from F0 layer j onto itself must stitch, on every chosen j.
- **Planted negative:** maps fitted on shuffled pairs (fit rows of F0 permuted across positions) must not stitch, on
  any layer.
- The verdict script's self-test, with eight synthetic records (PASS, PARTIAL, FAIL, each STOP), passes before the run.

## Registered predictions (scored, not gated)

| Prediction | Value |
| --- | --- |
| Verdict | **PARTIAL** (written after the smoke run showed how strict the bar is, which lowered it from PASS) |
| Chosen j | within ±4 of l · 31/19 on at least 14 of 19 layers |
| Test R², median over layers | 0.50-0.80 |
| Mapped top-8 overlap, median over layers | 0.40-0.65 |
| Stitching layers | most of them at l ≤ 13 |
| Shuffled maps | top-8 overlap ≤ 0.15 on every layer |
| Wall | v0's dump ≤ 30 min; F0's ≤ 1.5 h; the measurement ≤ 1.5 h |

## What this cannot say

- **A linear map, no training.** A PASS says F0's experts can be addressed from v0's state. It does not say the
  branch adds knowledge: that is the training protocol's job, with its random-donor control and fact panels on plain
  draws.
- **The CE through the stitch** (Chen et al.'s measure) needs F0's suffix to run on the mapped state, which means
  writing into llama.cpp's graph. It is left for the training cell, where the branch's effect on v0's own
  cross-entropy is the measure that matters.
- **One draw of documents, one seed.** Rows within a document are correlated, so the split is by document.
