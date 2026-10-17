# C50-A5 protocol, part 1: approximate head, offline selection on FIT

17 October 2026. Cell A5 of the [CHATBOT50 roadmap](ROADMAP.md) §5. Pushed before the offline
evaluation runs (programme rule 5). The capture of hidden states was started before this
push; it is data collection by a fixed rule, and no statistic of it had been computed.

## Question

Which approximate scan of the 157,184 × 2048 Q6_K head is the cheapest in bytes per
decode step while still returning the exact head's top-1 on at least 99.9% of FIT
positions, once its candidates are reranked exactly? The exact head streams 264.1 MB
per token, 30% of P0's 867 MB at context 2048 (A2 result). The roadmap ledger
budgets the approximate head at ~25 MB.

## Capture (fixed rule)

- **Engine:** `c50_engine.c` with the new `--dump-hidden` flag, binary
  `c50_engine_a5.exe`, sha256 `e524dd87aaf530ee…`.
  - The flag writes the final-norm output (the vector the head multiplies), 2048
    float32 per scored position. Forward numerics are unchanged:
    - the self-test passes;
    - on the 159-position smoke document its records are bit-identical to the
      A3 gate binary's.
- **Positions:** the first 12 FIT items of each category in role order
  (`c50_make_ids.py --per-category 12`). That is 36 documents and 79,709
  positions, ID file sha256 `c0897197ed010491…`.
  - These include the imatrix documents. That does not matter here: the head
    weights are fixed, and the question is about the geometry of the head and of
    the states.
- **Ground truth:** the engine's own top-1 at each position, i.e. the deployed
  exact Q6_K head with its Q8_K activation.
  - With an exact rerank of the candidate set by the same kernel, the approximate
    head returns that top-1 iff it is in the set. **Top-1 recall** = the fraction
    of positions whose engine top-1 is a candidate.

## Candidates (`c50_a5_offline.py`)

| Family | Configurations | Bytes per token |
| --- | --- | --- |
| low-rank preview: W ≈ (W V_r) V_r^T, V_r = top-r eigenvectors of WᵀW; candidates = top-K preview logits | r ∈ {128, 256, 512} × K ∈ {8 … 1024}; preview in float32 and with simulated 4-bit storage (groups of 32, symmetric, float16 scale) | 157,184·r·4.5 bit + 2048·r·4 B + K·1,680 B |
| IVF: k-means of the head rows (L2, seed 20261016, 12 iterations); probe the P clusters with the largest centroid·h; candidates = all rows of those clusters | C ∈ {256, 1024, 4096} × P ∈ {1 … 128} | C·2048·8.5 bit (Q8_0 centroids) + mean candidates·1,680 B |

The selection prices the 4-bit simulated preview. The float32 preview is reported
only to separate the rank error from the storage error.

**Controls (each failure makes the evaluation VOID):**
- **Oracle:** the head dequantized by `gguf-py` (the pin's own reference, independent
  of the engine's transcription) × the captured state, on every 10th position.
  - Its argmax must agree with the engine top-1 on ≥ 99% of those positions, and
    its log-sum-exp must match the record within 1e-2 nats (median).
  - Exact agreement is not expected, because the engine quantizes the activation
    to Q8_K.
- **Rank code:** through the same rank routine, the full head must reproduce the
  oracle agreement up to exact ties.

## Selection rule (fixed by the roadmap; nothing chosen after reading)

- **Selected:** the configuration with the fewest bytes per token among those
  whose FIT top-1 recall point estimate is ≥ 0.999. Ties go to the smaller
  candidate count.
- **If none reaches 0.999:** A5 offline FAILS. The roadmap's own fallback applies:
  a larger probe set, or a cheaper donor (ROADMAP §9, risk register).
- **Then, once:** the selected configuration is confirmed on DEV positions. The
  gate is teacher-forced top-1 agreement with the exact head ≥ 99.9% on DEV.
  - The DEV capture and its protocol (part 2) come after this selection. DEV is
    read once (programme rule 4).
- **Implementation:** the C implementation, joined to the decode and its rate,
  follows DEV, in part 3.
- **Exact head for likelihood:** likelihood scoring (BPB, PIQA) always uses the
  exact head of the same weights. The approximate head is a decode-time device.

## Registered prediction (ROADMAP §5; scored, not a gate)

- IVF with exact rerank needs < 15 MB/token for 99.9% recall; a rank-256 preview alone
  may not reach it.

## What this part cannot say

- **No rate.** Bytes per token here are counted, not streamed. The rate of a C
  implementation is part 3's measurement.
- **No sampling claim.** Top-1 recall is a greedy criterion. Sampling needs the
  candidate set to contain the tokens a repetition penalty touches; the runtime
  adds those, and part 3 tests that.
- FIT positions only. DEV is part 2.
