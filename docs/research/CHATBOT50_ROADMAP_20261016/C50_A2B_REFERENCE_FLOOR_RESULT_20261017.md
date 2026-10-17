# C50-A2b result: llama.cpp's disagreement with itself on the P0 artifact

17 October 2026. Cell A2b of the [CHATBOT50 roadmap](ROADMAP.md) §5, run under the
[A2/A2b protocol](C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md) and its addenda 1-3.
This covers the floor half of A2b. The external rate (`c50_a2b_rate.py`) waits for the
owner's clean-box session together with A1 (`docs/COMMUNICATION.md`, item 1).

## Inputs

- **Artifact:** `ling-mini-2.0-a810f64-q4_k_m.gguf`, sha256 `44de7315…`
  ([A2 result](C50_A2_ACQUISITION_RESULT_20261016.md)).
- **Positions:** 12 FIT documents, 4 per category, giving 26,363 positions and
  98,271 scored bytes; ID file sha256 `cf76819b…`.
- **Scorer:** `c50_llama_score_{avx2,generic}.exe`, llama.cpp `5b335f4`; context
  8192, F16 KV, flash attention off.
- **Records:**
  - Run: `benchmarks/chatbot50/results/c50_a2b_floor_run.json`. The pairs come
    from `--compare-only` (addendum 3). The first attempt's log and report are
    kept beside it.
  - Floor: `benchmarks/chatbot50/results/c50_a2b_floor.json`.

## Arms (all COMPLETE; each CONFIG line confirmed)

| Arm | Build | Threads | Micro-batch | Dump | Wall (s) |
| --- | --- | ---: | ---: | --- | ---: |
| R0 | AVX2 | 6 | 512 | yes | 820 |
| R3 | AVX2 | 6 | 1 | yes | 2,297 |
| R1 | AVX2 | 1 | 512 | yes | 1,659 |
| R2 | generic | 6 | 512 | yes | 3,434 |
| R0b | AVX2 | 6 | 512 | no | 394 |

Wall times include writing 16.6 GB of log-softmax rows per dumped arm, on a box
that was not clean. They are not rates.

## Pairs (document bootstrap 95% CI, 10,000 draws, seed 20261016)

| Pair (what changes) | top-1 disagreement | mean KL (nats) | p99 / max KL | dBPB | target log-probs bit-identical |
| --- | --- | --- | --- | --- | --- |
| R0-R1 (6 → 1 thread) | 0 | 0 | 0 / 0 | 0 | **yes** |
| R0-R2 (AVX2 → generic build) | **4.10%** (3.11-5.18) | **0.0149** (0.0125-0.0176) | 0.188 / 3.85 | −0.0022 (−0.0048, +0.0003) | no |
| R0-R3 (512 → 1-token micro-batch) | **3.84%** (2.90-4.85) | **0.0139** (0.0116-0.0165) | 0.174 / 3.68 | −0.0014 (−0.0038, +0.0009) | no |
| R0-R0b (repeat, negative control) | 0 | — | — | 0 | **yes** |

R0 scores 0.7865 BPB on these positions.

## Floor (definition registered before the runs)

- **Negative control:** R0 against R0b is bit-identical, so the floor is not VOID.
- `floor_top1` = 0.0410 (R0-R2); `floor_KL` = 0.0149 nats (R0-R2).
- **A3 tolerances by the registered formula:**
  - top-1 disagreement ≤ 0.0820;
  - mean KL ≤ 0.0299 nats.

## Observations

1. **The thread count does not change llama.cpp's numerics.** The build and the
   batch path do. One thread against six is bit-identical on every position.
   The generic build and the single-token path each move about 4% of top-1
   choices.
2. **The floor is about ten times the 159-position smoke** (KL ~4e-3 there). The
   smoke covered the first 159 positions of one document, and the per-position KL
   grows along a document (addendum 2 observation 4). So this full-document
   measurement replaces it; the smoke is not evidence.
3. **The disagreement is heavy-tailed.** p99 KL is ~0.18 nats and the maximum
   ~3.8 nats. That shape fits discrete top-8 routing flips amplifying
   rounding-level differences. This remains a hypothesis: routing was not
   instrumented.
4. **Both changing pairs lower BPB slightly** (R2 and R3 against R0), but every
   dBPB CI includes zero.

## The A3 dBPB criterion cannot pass llama.cpp against itself

The A3 protocol registered "dBPB 95% CI inside [−0.002, +0.002]". Both non-trivial
floor pairs fail it: R0-R2 reaches −0.0048 and R0-R3 −0.0038. With 12 documents the
document-bootstrap CI has a half-width of ~0.0025 even for a pair that is unbiased
by construction. A port as faithful as llama.cpp's own alternative paths would
therefore fail, which contradicts programme rule 2 (the tolerance is derived from
the reference's own disagreement).

[A3 addendum 2](C50_A3_ENGINE_PORT_PROTOCOL_20261016.md#addendum-2-17-october-2026-after-the-a2b-floor-before-any-engine-output-on-p0)
puts the dBPB criterion in the same form as the other two: |dBPB| point estimate
≤ max(2 × largest floor |dBPB|, 0.002) = **0.00444**, with the CI reported. It was
pushed before any engine output on the P0 artifact.

**Calibration check.** Fed to the revised verdict as if they were the engine,
both non-trivial floor pairs PASS all three criteria.

## Registered predictions, scored

| Prediction (ROADMAP §5) | Measured | Score |
| --- | --- | --- |
| The floor is above zero: top-1 disagreement of order 0.1-1% for at least one pair | 4.10% and 3.84% (build and batch path); 0 for threads | **Wrong in size.** The sign is right, but the disagreement is ~4× above the top of the range. The thread part predicted by "build/thread" is exactly zero. |
| llama.cpp decodes Ling-mini Q4_K_M at ≤ 30 tok/s at depth 2048 | not run yet (owner's clean-box session) | pending |

## What this result cannot say

- **This is llama.cpp's sensitivity, not quantization damage.** A KL of
  0.014-0.015 nats between two llama.cpp paths on one artifact is not a measure
  of Q4_K_M against BF16; A4 measures that.
- **The resulting tolerance is wide.** It catches errors larger than twice
  llama.cpp's own build and path disagreement, such as a wrong RoPE, router or
  scale. It cannot see a bias smaller than ~0.004 BPB.
- FIT positions only.
