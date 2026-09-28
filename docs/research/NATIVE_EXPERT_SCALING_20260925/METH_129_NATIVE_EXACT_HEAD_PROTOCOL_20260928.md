# METH-129: native int8 head shortlist plus exact selected-row score

## Uncertainty, prior and decision

METH-128 found that the stored per-row R8 tied head's top-16 includes
the original BF16 top-1 at 4,494/4,494 viewed E1280 positions. Exact
BF16 candidate scoring with lowest-ID tie breaking recovered every
choice. Whether that holds on native C full-model states, and whether a
one-byte scan plus 64 exact rows is faster than the full FP32 head on
Ryzen 5 3600X, are unknown. METH-127 provides a full C Qwen+E1280
reference with exact dense/composed parity and 16.818 greedy tok/s.
This experiment decides whether to retain the two-pass head as a native
cost/argmax primitive. It does not establish full likelihood or semantic
quality, compact body speed or the final `engine.c` target.

## Bound assembly and gates

- Native core: METH-127 FP32 Qwen2.5-0.5B-Instruct file SHA-256
  `6b2be143303510f1f15785783542649026b719488407f48b350de67f429e206029`.
  Centered E1280 bank: METH-126 SHA-256
  `1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`.
  Export only the METH-59 head code and scale tensors from stored R8
  artifact SHA-256
  `5b6ace4a027d127a6eb49ca99197edd4a1b810ebe681d8e3b6ab7edc088ca9bd`
  into a versioned native sidecar; verify byte readback, dimensions,
  and the source metadata. Keep the original tied FP32 matrix for exact
  selected-row scores. The extra resident bytes are explicit.
- Native approximate scan computes one score for all 151,936 rows using
  int8 codes and per-row FP32 scales. Select K=64 with a deterministic
  score-descending/ID-ascending rule; recompute those rows using the
  same FP32 accumulation path as the full native head. Greedy generation
  uses the best exact shortlisted candidate. Full-vocabulary logits
  remain approximate outside the shortlist: do **not** count `--bpb` or
  task log-probability as exact full-head parity.
- On the first 64 METH-121 prompt tokens, int32 SHA-256
  `990924bdcf23204fc28ac091e960834dc775efc09cd6ce9a38f73f307ee13db7`,
  require 64/64 native chosen IDs equal to the full FP32 head. Then
  generate 64 greedy tokens after the eight-token prefix (SHA-256
  `b9ffb0827e01fec4ef13027d46e04a61421d2ef62c571cc3827aa049bd3013ed`)
  and require the entire stream equal to METH-127 full-head output,
  with no EOS before the 64th token. A one-byte corrupted sidecar magic
  must fail load, not silently fall back.
- Time the same 64-token greedy run at one and six threads, sequentially
  for full and two-pass heads. Report accepted-token rate, per-organ
  timing, peak RSS, sidecar bytes and selected-row traffic. Require a
  measured six-thread speed increase of at least 5% for continuation;
  otherwise reject this native path even if parity passes. This speed
  gate is relative to the FP32 reference, not the final >=50 tok/s gate.

## Budget and interpretation

Local CPU/RTX 3060 only; no T4. Stop on a binding/hash failure, native
chosen-token mismatch, nonfinite score, negative-control failure,
>20 minutes, >16 GiB process RSS or >1 GiB new disk beyond the existing
core/bank. METH-128's BF16-state result is a viewed development screen;
METH-129's FP32-native parity cannot be substituted for independent
BF16 donor quality. A passing path next needs integration with a compact
quality-valid body and fresh document/generation/task/grounding gates.
