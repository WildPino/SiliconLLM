# STRAT-01 Q4_K×Q8_K reference-generic compile-parity result

**Verdict:** `PASS_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY`

The frozen hypothesis is confirmed. The remaining block-0 Q/KV discrepancy
was caused by compiling the generic Q4_K×Q8_K operation structure inside the
AVX2/FMA `engine.c` translation unit. Isolating the same operation order behind
a non-inlined Clang function with AVX, AVX2, and FMA disabled reproduces both
immutable reference projections byte-for-byte.

This is a measured operator/projection parity PASS. It supersedes the
[apparatus-only result](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_APPARATUS_RESULT_20260923.md)
for this cell's numerical question, but does not supersede the earlier
[active-AVX2 FAIL](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY_RESULT_20260923.md):
the active-x86 and historical AVX-translation-unit implementations remain
distinct controls and reproduce their frozen outputs exactly.

## Immutable execution identity

- implementation commit:
  `9709bca572ed71bb6a6d18ddb205632efec3ff5a`;
- accepted GigaChat 3.1 Q4_K_M GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- exact `attn_norm-0` input SHA-256:
  `c8c7bd47772b1f153f28183892795b9bc91be978f5473bb322f869b2c10c1efd`;
- preserved reference `compile_commands.json` SHA-256:
  `de27906dc35179b5efd47ffab98adb33476425ef215f9b0513ae9229454ecb80`;
- preserved reference executable SHA-256:
  `abdadf3fdac1bed75e24b1353d1fbff85159c1f6e4c3e8d81e9efe5e1e246627`;
- candidate source `strat01_q4k_q8k.h` SHA-256:
  `49b5f39fd1a1313e15038208420cf7484765b01bc3f83952382577c6be68bc5f`.

The preserved compile inventory proves that the reference producer selected
baseline `ggml-cpu/quants.c` with `GGML_CPU_GENERIC`, without AVX/FMA flags.
The candidate's surrounding engine remains Clang C11 `-O3 -mavx2 -mfma`;
only the named helper has `no-avx,no-avx2,no-fma` target controls.

## Measured outputs

| Arm | Q SHA-256 | KV SHA-256 | Frozen identity |
|---|---|---|---|
| reference-generic candidate | `4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b` | `6a364dd45c12142fb45ab90287caa14874ed40160089e716e23bd58d3785653c` | exact reference |
| historical AVX-TU generic | `6255f40d5a717af2a742c484b0ab75ff549b37acea70f73556698c794c8b3366` | `5c3fdab029c1660bae7c4f4d5d256991def1ccce39468ccdb5d9b1346d5be9ce` | exact historical replay |
| active x86 AVX2/FMA | `b1d355be461cf30f42cc0acc88c4cf08963678a252be74761031a41e44978cd3` | `a2c177598c5ea5e7a9b5cb5707a6a738b711edd548ce6d9430eb3f66be2222f2` | exact closed-FAIL replay |

All six digest comparisons pass. The three identities are pairwise distinct,
so the result cannot be explained by a dead diagnostic switch or by reusing
one output for multiple arms.

## Controls and scope

All 18 adjudication controls are true:

- candidate versus independently built baseline-generic oracle is bit-exact
  at `n = 256, 512, 1536, 6144`;
- both diagnostic controls fire on registered multi-block fixtures;
- source, model, input, reference build, and prior-output identities pass;
- 127 STRAT-01 Python tests, all 16 registered C self-tests, and the 73,024-
  check legacy kernel suite pass;
- donor and reference graph executions are zero.

The accepted cell performed three controlled model reads, one per arm, within
one preregistered adjudication. It did not execute a donor or reference graph.
The approximately 30-second projection command durations are operational
diagnostics, not emitted-token throughput, and must not enter
`SPEED_LEDGER.md`.

## Provenance

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q4k_q8k_reference_generic_parity_20260923/`

`adjudication.json` SHA-256:
`f0371d9c98e762e6434907ea5c386a7b622adac30dae9939a101fdc22df995f1`

Frozen protocol:
[STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_PROTOCOL_20260923.md](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_PROTOCOL_20260923.md)

Runner:
`benchmarks/donor_adaptation/engine/run_strat01_q4k_q8k_reference_generic_parity.py`

## Decision and next gate

This cell is closed and must not be rerun, retuned, or replaced by the faster
active-AVX2 arm when exact parity is required. The candidate becomes the
production Q4_K×Q8_K path for the current fidelity branch.

The result proves exact block-0 Q and KV projection parity on the immutable
input. It does **not** prove downstream block-0/layer-1 propagation, full-model
logits, generation, C-tokenizer parity, quality, RAM, or accepted-token rate.
The only authorized successor is a separately frozen changed-coordinate
propagation confirmation that reuses immutable reference checkpoints and
tests whether this exact Q4 arithmetic closes the previously measured block-0
terminal and Rung-2C residuals before any later-layer expansion.
