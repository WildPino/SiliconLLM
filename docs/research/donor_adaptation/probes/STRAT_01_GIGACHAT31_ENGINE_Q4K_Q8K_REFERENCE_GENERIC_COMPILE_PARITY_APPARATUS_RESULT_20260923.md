# STRAT-01 Q4_K×Q8_K reference-generic compile-parity apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The apparatus required by the frozen
[reference-generic compile-parity protocol](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_PROTOCOL_20260923.md)
is qualified. No donor or reference graph ran and the accepted GGUF was not
read. This result authorizes exactly one full-projection adjudication after the
implementation and this apparatus record are committed; it is not a projection
parity result.

## Changed coordinate

The production Q4_K×Q8_K helper now preserves the pinned generic operation
order but is compiled behind a non-inlined Clang function boundary with AVX,
AVX2, and FMA disabled. The surrounding `engine.c` translation unit remains
Clang C11 `-O3 -mavx2 -mfma`. Two named diagnostic controls retain the prior
AVX-translation-unit generic implementation and the closed active-x86 AVX2
implementation.

The candidate is a compile-lowering hypothesis only. Q8_K bytes, packed Q4_K
semantics, matrix layout, inputs, weights, and every non-Q4 operator remain
unchanged.

## Qualification evidence

- An independently built oracle selected exactly pinned
  `ggml-cpu/quants.c` with `GGML_CPU_GENERIC`, without AVX/FMA flags. Its source
  SHA-256 is
  `459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad`.
- The candidate is bit-exact to that oracle at
  `n = 256, 512, 1536, 6144` on deterministic fixtures with non-binary Q4 and
  Q8 scales.
- Both causal controls fire: the historical AVX-translation-unit generic path
  differs at `n = 512, 6144`; active AVX2 differs at
  `n = 512, 1536, 6144`.
- The preserved reference build identity passes:
  `compile_commands.json`
  `de27906dc35179b5efd47ffab98adb33476425ef215f9b0513ae9229454ecb80`
  and executable
  `abdadf3fdac1bed75e24b1353d1fbff85159c1f6e4c3e8d81e9efe5e1e246627`.
- All **127** registered STRAT-01 Python tests pass.
- All **16** registered C self-tests pass, including the legacy kernel suite:
  **73,024 checks**, worst scalar-reference error `0`.
- Candidate, historical-generic, and active-AVX2 engine executables compile as
  distinct artifacts.
- `model_reads = 0`, `donor_graph_executions = 0`, and
  `reference_graph_executions = 0`.

## Reproducibility

Runner:
`benchmarks/donor_adaptation/engine/run_strat01_q4k_q8k_reference_generic_parity.py`

Raw apparatus directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q4k_q8k_reference_generic_parity_apparatus_20260923/`

`adjudication.json` SHA-256:
`d9114eed42059c7077c62153ee8a759258637ff3472994415a6d348f0b1c904a`

Observed pre-implementation commit:
`09977e47a9c659c6f9da6dcd95fa9b43b331932d`

Critical candidate source SHA-256 values recorded by the adjudicator include:

- `strat01_q4k_q8k.h`:
  `49b5f39fd1a1313e15038208420cf7484765b01bc3f83952382577c6be68bc5f`;
- runner:
  `3448125bfec4977068fcd68a6f61a2d4fa063e5f70a54bd16b379095f7633751`;
- model-free parity test:
  `fc98de030d3396498db5cc2f40937796476dc5fef9e0b1b0f7c6936eb3b787d0`.

## Claims and stop rule

This apparatus establishes only that the intended compiler-arithmetic
coordinate is implemented, independently identified, causally controlled, and
regression-clean. It makes no claim about full Q/KV projection parity,
downstream propagation, quality, generation, RAM, or rate.

The next permitted action is the protocol's single non-VOID full-projection
adjudication. PASS requires exact candidate Q and KV hashes plus exact replay of
both historical controls. FAIL closes compile-lowering parity. Neither outcome
permits retuning this frozen cell.
