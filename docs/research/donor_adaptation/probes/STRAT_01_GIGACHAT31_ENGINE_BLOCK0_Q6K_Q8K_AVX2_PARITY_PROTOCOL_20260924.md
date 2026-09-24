# STRAT-01 block-0 Q6_K×Q8_K AVX2 parity protocol

**Status:** frozen before apparatus implementation or scientific execution.

## Question and non-duplication boundary

The canonical
[attention-output residual result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT_RESULT_20260924.md)
shows that the repaired production block-0 `l_out-0` residual is sufficient
under the closed layer-2 amplifier. Existing immutable payloads then close the
terminal composition without another run: `ffn_inp-0` is byte-identical to
reference, repaired no-FMA SSE2 SwiGLU is byte-identical to reference, and the
current Q6 output plus that exact residual input reconstructs production
`l_out-0` byte-for-byte. The first non-exact value is therefore the current
Q6_K×Q8_K down-projection output.

This is not a repeat of Rung 1, the 23 September Q6 cross-input cell, or the
post-F16 operator-chain cell. Rung 1 explicitly permitted Q6 output hashes to
differ because reduction order can round differently. The later cells proved
the current scalar/generic Q6 path passed their then-available layer-1 gates;
they did not compare it with pinned llama.cpp's active x86 AVX2/FMA reduction.
The stronger layer-2 amplifier now makes the already measured direct residual
causally relevant. The Q4_K AVX2 parity work is a different codec and kernel.

Question: on the exact same quantized activation bytes and Q6_K rows, does a
literal port of pinned llama.cpp's active AVX2/FMA Q6 dot product reproduce
the immutable reference `ffn_out-0` byte-for-byte, while the existing generic
path reproduces its known candidate hash?

## Immutable evidence and source binding

Accepted GGUF: `6474702976` bytes, SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
Matrix: `blk.0.ffn_down.weight`, Q6_K `[8960,1536]`, tensor offset
`286755840`, GGUF file offset `292858752`, span `11289600` bytes.

| payload | bytes | SHA-256 |
|---|---:|---|
| exact reference `ffn_swiglu-0` | `286720` | `de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef` |
| exact reference/current `ffn_inp-0` | `49152` | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` |
| pinned llama.cpp `ffn_out-0` | `49152` | `f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4` |
| current generic-Q6 `ffn_out-0` | `49152` | `f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce` |
| pinned reference `l_out-0` | `49152` | `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa` |
| current production `l_out-0` | `49152` | `a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11` |

Bind llama.cpp revision
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2`. Its
`ggml/src/ggml-cpu/arch/x86/quants.c` SHA-256 is
`99a98747c1ac84ec40e2d1a31227b947aeb0a05bf1d8e79ec630a6d783b89f86`;
the generic `ggml/src/ggml-cpu/quants.c` SHA-256 is
`459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad`.
The current scalar helper source `strat01_gguf_rung2b.h` is bound at SHA-256
`d04cd205dcc820417116f635071bf93ddb04a7752f41fad861214bea5fa35655`.

## Frozen apparatus and arms

Build a small pinned-reference oracle from the clean llama.cpp checkout and a
C-side diagnostic in `engine.c`. Both must consume the same immutable
`ffn_swiglu-0`, quantize each 8,960-value row to Q8_K, and read only the
admitted Q6_K matrix. Emit and hash the complete Q8_K activation population
and all `[8,1536]` outputs.

The C diagnostic must provide these disjoint arms:

1. `current_generic_replay`: existing scalar/generic Q6 helper; it must
   byte-reproduce `f52e0e5...97d6ce`.
2. `pinned_avx2_candidate`: literal pinned x86 AVX2/FMA integer accumulation,
   block-scale accumulation, and `hsum_float_8` order.
3. `pinned_oracle`: the clean pinned llama.cpp Q8_K quantizer and active
   `ggml_vec_dot_q6_K_q8_K` implementation.
4. `control_generic_not_avx2`: the generic result is the planted numerical
   negative and must remain distinct from the immutable reference hash.
5. `control_mutated_q6_row`: a one-byte Q6 payload mutation in a copied row
   must change the candidate output and never modify the accepted artifact.

The current and pinned-oracle Q8_K byte populations must be equal before any
reduction verdict. Candidate and oracle Q6 outputs must be byte-equal. Add each
output to exact `ffn_inp-0` with production float32 order. For the candidate,
propagate the resulting `l_out-0` through the same already closed layer-1 and
layer-2 amplifier used by the predecessor; no donor or reference model graph
may run.

## Gates and decisions

Require exact artifact, matrix span, source revision/hash, payload, compiler,
arm-label, one-byte-mutation, helper-count, and schedule-twin controls. Compile
the candidate with Clang C11 `-O3 -mavx2 -mfma` and no fast-math; preserve the
pinned FMA and horizontal-reduction order explicitly.

- if Q8_K bytes, candidate output, reference terminal sum, and complete
  downstream output are all byte-exact to the pinned oracle/reference, while
  generic replay and the mutation control behave as frozen:
  `BLOCK0_Q6_AVX2_EXACT_REPAIR`;
- if current Q8_K bytes differ from the pinned oracle:
  `BLOCK0_Q6_Q8K_QUANTIZER_MISMATCH`;
- if Q8_K bytes are exact but the literal AVX2 candidate differs from the
  pinned oracle/reference:
  `BLOCK0_Q6_AVX2_REDUCTION_INSUFFICIENT`;
- otherwise: `VOID_BLOCK0_Q6K_Q8K_AVX2_PARITY`.

Qualify model-free first. Apparatus-only must open neither the GGUF nor any
scientific payload and must report zero Q8, Q6, diagnostic, and graph counts.
Commit exact sources before permitting one zero-graph scientific invocation.

## Stop rule and non-claims

If exact AVX2 parity is obtained, integrate the same primitive into the normal
block-0 Q6 path under a separately frozen production-integration gate; do not
rerun this parity cell. If the quantizer differs, localize only Q8_K packing or
scale semantics. If AVX2 still differs with exact Q8_K bytes, inspect only the
pinned integer-lane/FMA/horizontal-sum semantics.

This cell does not rerun SwiGLU, attention, either model graph, quality,
generation, RAM, or timing, and it makes no full-model or tok/s claim.
