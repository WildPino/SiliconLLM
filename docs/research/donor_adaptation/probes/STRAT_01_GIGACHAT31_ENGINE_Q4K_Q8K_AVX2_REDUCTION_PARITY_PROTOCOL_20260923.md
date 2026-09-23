# STRAT-01 Q4_K×Q8_K AVX2 reduction-parity repair protocol

**Frozen:** 2026-09-23, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-Q4K-Q8K-AVX2-REDUCTION-PARITY`

**Purpose:** replace only the generic scalar multi-block reduction order in
the standalone Q4_K×Q8_K dot with the active pinned x86 AVX2/FMA reduction
order, and require byte-exact block-0 Q/KV projection outputs.

## Prior evidence and changed coordinate

The 21 September
[Q4_K×Q8_K repair](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REPAIR_RESULT_20260921.md)
correctly closed activation-format and packed-dot semantics under its frozen
numerical gates. Its project outputs were not byte-identical to the pinned
active x86 kernel: Q and KV NRMSE were `6.4478969411e-8` and
`7.3224546015e-8`. The result explicitly allowed that difference.

The later
[SwiGLU cross-input result](STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md)
changes the engineering requirement: residuals descended from the production
attention path are small enough to pass every block-0 local gate, yet either
gate or up operand independently causes the first layer-1 failure. Existing
cross-input, double-RMSNorm, combined, and production records also establish:

- exact reference `ffn_norm-0` passes the unchanged gate/up Q4_K path;
- the production `ffn_norm-0` has a new SHA-256
  `4b17c45fcfc6573f9a5e1461e4f2c232a9536d6c8cf689884a6c8050ee0b2632`,
  distinct from the old C cross-input payload, and deterministically produces
  the measured C gate/up tensors;
- `attn_norm-0` is byte-exact in current production, while the first remaining
  nonzero tensor errors are Q and KV at approximately `6–7e-8`.

Therefore another gate/up cross-input run would duplicate existing evidence.
The changed coordinate here is strictly:

> generic scalar Q4_K×Q8_K multi-block accumulation and final reduction →
> pinned x86 AVX2/FMA accumulation and horizontal reduction order.

Q8_K quantization, packed Q4 decoding, scales/minima, matrix layout, inputs,
weights, and every non-Q4 operator remain fixed.

## Immutable evidence

- Accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Exact reference `attn_norm-0`: SHA-256
  `c8c7bd47772b1f153f28183892795b9bc91be978f5473bb322f869b2c10c1efd`,
  shape `[1536,8]`.
- Exact reference Q target: SHA-256
  `4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b`,
  shape `[6144,8]`.
- Exact reference KV target: SHA-256
  `6a364dd45c12142fb45ab90287caa14874ed40160089e716e23bd58d3785653c`,
  shape `[576,8]`.
- Accepted generic project Q/KV hashes:
  `6255f40d5a717af2a742c484b0ab75ff549b37acea70f73556698c794c8b3366`
  and `5c3fdab029c1660bae7c4f4d5d256991def1ccce39468ccdb5d9b1346d5be9ce`.
- Current standalone operator SHA-256:
  `9af4251239aa753983fc52a8b1af086ba942486aaeda34351b030d25004d2742`.
- Pinned llama.cpp revision:
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`; clean x86 `quants.c`
  SHA-256:
  `99a98747c1ac84ec40e2d1a31227b947aeb0a05bf1d8e79ec630a6d783b89f86`.

The admitted tensors remain `blk.0.attn_q.weight` Q4_K `[1536,6144]` and
`blk.0.attn_kv_a_mqa.weight` Q4_K `[1536,576]` with the already frozen
offsets and spans. No producer graph may run.

## Implementation contract

1. Transcribe the pinned `__AVX2__` branch of
   `ggml_vec_dot_q4_K_q8_K` into the standalone C header without linking or
   depending on GGML.
2. Preserve the current Q8_K block bytes exactly.
3. Preserve Q4 unpacking and scale/minimum semantics exactly; only the integer
   lane accumulation, FMA order, minimum accumulation, and terminal horizontal
   reductions may change.
4. Require AVX2 and FMA at compile time; fail closed on unsupported builds.
5. Keep the old generic helper only as an explicitly named diagnostic control,
   not as a production fallback.
6. Route every production Q4_K×Q8_K matrix use through the repaired helper.
7. Compile with Clang C11, `-O3 -mavx2 -mfma`, no fast-math. The explicit FMA
   intrinsics define the contraction sites for this cell.

## Gates and controls

All must pass:

- byte-exact Q output SHA-256 equals the immutable reference Q hash;
- byte-exact KV output SHA-256 equals the immutable reference KV hash;
- Q8_K payloads remain byte-exact;
- active project versus pinned active oracle is bit-exact on deterministic
  one- and multi-block populations with `n = 256, 512, 1536, 6144`;
- the old generic reduction differs on at least one registered multi-block
  fixture and reproduces the two accepted generic output hashes on the frozen
  projection;
- packed-scale, minimum, high-nibble, Q8-scale, row/token transpose, wrong
  descriptor, short-read, mutation, and non-finite controls reject;
- all existing Q4 operator, Rung-1/2A/2B/2C, downstream diagnostic, and 73,024
  legacy checks remain green;
- source inventory and pinned-oracle revision/hash checks pass;
- donor and reference graph executions are both zero.

## Decision and stop rule

- `PASS_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY` only if both full projection
  hashes and every control are exact.
- `FAIL_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY` for a valid non-exact result.
- `VOID_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY` for any identity, source,
  apparatus, or control failure.

Exactly one non-VOID invocation is allowed. A PASS authorizes a separately
frozen production propagation confirmation; it does not rewrite prior results
or establish Rung 2C, later-layer, quality, generation, RAM, or rate parity.
A FAIL stops this repair route and requires a new arithmetic hypothesis rather
than gate relaxation.
