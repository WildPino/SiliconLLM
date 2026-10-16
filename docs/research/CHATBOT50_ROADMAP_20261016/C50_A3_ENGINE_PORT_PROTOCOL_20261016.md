# C50-A3 protocol: the engine port, P0 artifact, system fidelity against llama.cpp

16 October 2026. Cell A3 of the [CHATBOT50 roadmap](ROADMAP.md) §5. Pushed before the
engine has produced any output on the real model (programme rule 5).

## Question

Does the project's own C decode path (`benchmarks/chatbot50/c50_engine.c`) run the
complete Ling-mini-2.0 Q4_K_M artifact and agree with pinned llama.cpp on the same
GGUF and the same source-tokenizer IDs within the tolerance A2b derives from
llama.cpp's disagreement with itself?

## Instrument

- **Engine:** `c50_engine.c`, built with clang 21.1.8 (llvm-mingw 20251216),
  `-O2 -mavx2 -mfma -mf16c`, no `-ffast-math`.
  - Threads: six workers pinned one per physical core, with the A1 spin pool.
  - Decode: batch 1, one token per forward, KV cache F16.
  - Precision: attention, norms, router and RoPE in F32.
  - Matrix products: the activation is quantized as llama.cpp's CPU backend does
    (Q8_K for k-quants, Q8_0 for Q8_0), followed by AVX2 dots. Q4_K and Q6_K
    reuse the STRAT-01 transcriptions; Q5_K, Q8_0 and the dequantizers are new
    transcriptions of llama.cpp `5b335f4` (MIT).
- **Semantics:** the pinned source model code (`modeling_bailing_moe_v2.py` @
  `a810f64`) and llama.cpp `src/models/bailingmoe2.cpp` agree on every step. The
  engine header states them:
  - QK-norm before NEOX RoPE on 64 of 128 dims, theta 600000;
  - GQA 16/4;
  - sigmoid router in F32 with the expert bias used for selection only;
  - group score = sum of the top-2 biased scores, keeping 4 of 8 groups, then top-8;
  - weights = unbiased scores / their sum × 2.5;
  - one shared expert; dense layer 0;
  - RMSNorm eps 1e-6.
- **Unit test before any use:** `c50_engine.exe --selftest`.
  - For Q4_K, Q5_K, Q6_K and Q8_0, the AVX2 dot is compared with the generic
    transcription. Both are compared with an independent oracle: dequantized
    weights × dequantized activation, in double. This oracle also validates the
    embedding dequantizers.
  - Bound 1e-5 of the A1 addendum-1 normaliser.
  - Planted control: ≥ 190/200 detections.
  - Result: PASS (`benchmarks/chatbot50/results/c50_a3_selftest_20261016.txt`;
    the first version, with an undersized normaliser, is kept there).
- **Output:** the same per-position records as `c50_llama_score.cpp`, so
  `c50_compare.py` compares the two runtimes directly.

## Positions and reference

- **Positions:** the A2b positions — 12 FIT documents, 26,363 positions, input
  SHA-256 `cf76819b…`. Same scoring convention: every position p ≥ 1 is predicted
  from `ids[0..p-1]`, context is cleared per document, no special token is added.
- **Reference arm:** A2b **R3**: llama.cpp AVX2, 6 threads, micro-batch 1. This is
  the single-token decode path, the same computation shape as the engine. The
  comparison with R0 (micro-batch 512) is reported beside it without a gate.

## Gate (pre-registered; the thresholds come from A2b and the roadmap)

With `floor_top1` and `floor_KL` taken from `c50_a2b_floor.py`:

| Statistic (engine vs R3) | Pass |
| --- | --- |
| top-1 disagreement (point estimate) | ≤ max(2 × floor_top1, 0.002) |
| mean KL(R3 ‖ engine), nats (point estimate) | ≤ max(2 × floor_KL, 1e-4) |
| dBPB, document-bootstrap 95% CI | both ends inside [−0.002, +0.002] |

All three must pass. If the A2b floor is VOID, A3 is not adjudicated until the
floor is repaired.

**Failure.** Rule 3 applies: at most three diagnostic cells or three working days,
then a written design review. Diagnostics may compare intermediate tensors
(`llama-eval-callback` on the same positions); they never become a gate.

**Development runs.** Short smokes of the engine against llama.cpp on the A2b
smoke document (159 positions) are allowed while building. They are kept in
`benchmarks/chatbot50/results/` and are not evidence. The gate run is the first
run on all 26,363 positions with the binary whose SHA-256 the run records.

## Recorded beside the gate (no gate)

- The engine's wall time per position on this scoring run. The box is not clean
  and scoring dumps 157,184 log-probabilities per position, so this is not a rate.
- The organ list and tensor types of the artifact (`CONFIG` line).

## What this cell cannot say

- A pass says the engine computes what llama.cpp computes on this artifact, to
  within llama.cpp's own sensitivity. It says nothing about quality (A4) or
  speed (A7).
- FIT positions only: A3 selects no threshold. DEV is read once in A6, and
  RESERVED in A7.

## Addendum 1 (16 October 2026, before the gate run)

1. **Runner and verdict are separate.**
   - `c50_a3_run.py` re-runs the engine self-test with the exact binary it then
     uses; a failure makes the run VOID.
   - It checks the A2b position file hash, asserts the `CONFIG` line
     (26,363 positions, dump on), and calls `c50_compare.py` against R3 and R0.
   - `c50_a3_verdict.py` applies the table above with the floor file from
     `c50_a2b_floor.py`.
2. **The engine gained a `--bench` mode** (a decode-rate bench for A7). Forward
   numerics are unchanged, so the self-test recorded in
   `c50_a3_selftest_20261016.txt` belongs to an earlier binary. The gate run records
   its own self-test output and binary hash.
3. **A development smoke on the Q8_0 artifact is kept**
   (`c50_a3_smoke_20261016.txt`; 159 positions, contended box, not evidence). The
   engine against llama.cpp micro-batch 1 gave a top-1 disagreement of 2/159 and
   a mean KL of 3.9e-3 nats. On the same positions llama.cpp differs from itself
   (micro-batch 512 vs 1) by 6/159 and 4.3e-3.
