# STRAT-01 block-0 Q6_K×Q8_K reference-generic compile parity apparatus

**Status: `APPARATUS_READY_NO_SCIENTIFIC_EXECUTION`, superseded for execution
authorization by
[repair 1](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_APPARATUS_REPAIR1_RESULT_20260924.md).**

The frozen reference-generic compile coordinate is executable and its controls
are live. This qualification opened no accepted GGUF, no scientific payload,
and no donor or reference graph. It does not decide scientific parity.

## Bound apparatus

- observed repository HEAD before the apparatus sources are committed:
  `9abb62b533a244e8a2a26edbff7b27be8a59cce0`;
- apparatus adjudication SHA-256:
  `b265d709a141ceedb516c3e50694ad361e608201ef995c6ef57f8f263cdf7725`;
- pinned baseline `ggml-cpu/quants.c` SHA-256:
  `459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad`;
- engine source SHA-256:
  `9dad204f987b7267dd27b3035047dad2c550309d72beec7deb71f4de18109618`;
- Q6 primitive source SHA-256:
  `4de37b009a9d3173178c60a7db309d73179bb1909fba049e36c6240f23d7544d`;
- full diagnostic source SHA-256:
  `e90f34b2be1eabeca976903fe8051793c962b0b859be96d564ab79fa2383c4a2`.

The raw, unversioned record is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_reference_generic_parity_apparatus_20260924/`.

## Qualification

- the complete engine diagnostic compiles with Clang `-O3 -mavx2 -mfma`;
- its reference-generic command self-test passes all five checks;
- the candidate is bit-exact to the independently built pinned
  `GGML_CPU_GENERIC` oracle at vector lengths 256, 512, 1536, and 8960;
- a deterministic six-block fixture makes reference-generic candidate,
  AVX-translation-unit generic, and active AVX2 pairwise distinct;
- short reads and invalid arm labels are rejected;
- the candidate is non-inlined and explicitly restricted to
  `no-avx,no-avx2,no-fma`;
- the engine command emits the candidate, historical AVX-TU generic, closed
  active-AVX2 control, candidate residual sum, and candidate downstream arm.

All model-read, scientific-payload-read, Q8-population, Q6-matrix-read,
diagnostic, donor-graph, and reference-graph counters are zero.

## Authorization boundary

This record does not authorize execution because a trailing-whitespace repair
changed the diagnostic source after qualification. No semantic behavior was
intended to change, but source identity is an exact gate. Repair 1 requalified
the final bytes and is the sole execution authority.

No quality, generation, RAM, throughput, or full-model claim follows from
this apparatus result.
