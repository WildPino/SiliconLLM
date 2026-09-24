# STRAT-01 block-0 Q6_K×Q8_K reference-generic compile parity apparatus — repair 2

**Status: `APPARATUS_READY_NO_SCIENTIFIC_EXECUTION`.**

Repair 2 adds the missing independent full-matrix oracle surface and proves
its Q8 population and output byte-exact against the C reference-generic
candidate on a synthetic matrix. The initial apparatus and repair 1 remain
preserved but are superseded as execution authorities. This run opened no
accepted GGUF, no scientific payload, and no donor or reference graph.

## Bound apparatus

- observed repository HEAD before these repair-2 sources are committed:
  `98d39cb29a4ec8a5dc43516fbad8547f1ab36f0e`;
- repair-2 adjudication SHA-256:
  `60675f36971f9fd67982ecfcdd1eac8aa5ffa9a3f0c174cd63efc8f7b867e838`;
- pinned baseline `ggml-cpu/quants.c` SHA-256:
  `459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad`;
- engine source SHA-256:
  `9dad204f987b7267dd27b3035047dad2c550309d72beec7deb71f4de18109618`;
- Q6 primitive source SHA-256:
  `4de37b009a9d3173178c60a7db309d73179bb1909fba049e36c6240f23d7544d`;
- full diagnostic source SHA-256:
  `09f448e2b103920ea6517194635ed8bebd3e454e9a69fc7a260fdc4443b19c2c`;
- reference-generic probe SHA-256:
  `0eed2952aa8cf020541f2bc592dc0c71aeb7051ef76ef86d3406a875dce3b345`;
- independent oracle SHA-256:
  `9f8f5753e1821d9c2857251633347ec99f2109c74845cb72c9157a6719f409de`;
- model-free test SHA-256:
  `9580587e9064435811344c78c3d8c8a9f553d1880924988368c7f2b6d90c873c`;
- apparatus runner SHA-256:
  `f9123ab2307c31bb6ed5c8a60ea6ea8679664d9019e4b25e3ec36597811966cb`.

The raw, unversioned record is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_reference_generic_parity_apparatus_repair2_20260924/`.

## Qualification

- the complete engine diagnostic compiles and its five-check self-test passes;
- the candidate is bit-exact to the pinned `GGML_CPU_GENERIC` oracle at
  lengths 256, 512, 1536, and 8960;
- candidate and oracle independently quantize a synthetic batch, traverse a
  five-row Q6_K matrix, and emit byte-identical Q8 populations and outputs;
- a deterministic six-block fixture makes candidate, AVX-TU generic, and
  active AVX2 pairwise distinct;
- short reads and invalid arm labels are rejected;
- the candidate remains non-inlined under `no-avx,no-avx2,no-fma`;
- every scientific/model/payload/matrix/diagnostic/graph counter is zero.

## Authorization boundary

Commit the exact hash-bound sources before execution. Then one scientific
invocation may run the accepted matrix/input through both the C candidate and
the independently compiled baseline-generic oracle. Required exact gates are
Q8 population, candidate/oracle full output, immutable `ffn_out-0`, reference
`l_out-0`, and complete downstream output. Historical AVX-TU generic, closed
active AVX2, one-byte mutation, source, compiler, descriptor, size, twin, and
zero-graph controls must all fire. Any failure outside the frozen scientific
decision branches is VOID.

No quality, generation, RAM, throughput, or full-model claim follows.
