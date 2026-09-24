# STRAT-01 block-0 Q6_K×Q8_K reference-generic compile parity apparatus — repair 1

**Status: `APPARATUS_READY_NO_SCIENTIFIC_EXECUTION`.**

Repair 1 repeats the model-free qualification on the exact final source bytes
after removing one trailing space from the diagnostic header. The initial
apparatus remains valid for its own source hash but is superseded as execution
authority. This run opened no accepted GGUF, no scientific payload, and no
donor or reference graph.

## Bound apparatus

- observed repository HEAD before these apparatus sources are committed:
  `9abb62b533a244e8a2a26edbff7b27be8a59cce0`;
- repair-1 adjudication SHA-256:
  `374f002772e940ad3b8db1d0ef87fd81f09a0809853aa42db8c6c259ad44739b`;
- pinned baseline `ggml-cpu/quants.c` SHA-256:
  `459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad`;
- engine source SHA-256:
  `9dad204f987b7267dd27b3035047dad2c550309d72beec7deb71f4de18109618`;
- Q6 primitive source SHA-256:
  `4de37b009a9d3173178c60a7db309d73179bb1909fba049e36c6240f23d7544d`;
- full diagnostic source SHA-256:
  `09f448e2b103920ea6517194635ed8bebd3e454e9a69fc7a260fdc4443b19c2c`;
- apparatus runner SHA-256:
  `8049afef1f60eab4fc2b3fe848ff3ede840d6e4df3543a92a6c84f6c4e39474d`.

The raw, unversioned record is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_reference_generic_parity_apparatus_repair1_20260924/`.

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

Commit the exact hash-bound sources before execution. Then a separate
scientific runner may perform one accepted-artifact read and one independent
baseline-generic oracle read, both with zero model graphs. It must require
byte-exact Q8 populations, candidate/oracle output, immutable `ffn_out-0`,
reference `l_out-0`, and the complete downstream target. Any identity,
control, accounting, or source mismatch is VOID.

No quality, generation, RAM, throughput, or full-model claim follows.
