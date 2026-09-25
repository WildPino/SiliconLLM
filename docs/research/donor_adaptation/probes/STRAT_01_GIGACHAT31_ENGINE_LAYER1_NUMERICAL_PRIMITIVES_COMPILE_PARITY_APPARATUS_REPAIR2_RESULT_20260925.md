# STRAT-01 layer-1 numerical-primitives compile-parity apparatus repair 2

**Date:** 2026-09-25

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

## Result

Following [VOID 1](STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY_VOID1_20260925.md)
and [addendum A](STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY_ADDENDUM_A_20260925.md),
the repaired apparatus qualifies the same frozen estimand. The standalone
production replay now carries `#pragma STDC FP_CONTRACT OFF`, matching the
Rung-2C source scope, and the three one-value mutations use finite `+1.0f`
changes instead of changes of one ULP. Candidate, oracle and decision gates
are otherwise unchanged.

The qualifying record is:

- raw directory:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_numerical_primitives_compile_parity_apparatus_repair2_20260925/`;
- adjudication SHA-256:
  `57348fa376ac340f2d36a122e03dbd4189b7ee21c62a2acb4078d6bd636148cd`;
- observed pre-repair commit:
  `2e255eba96288e0a6562f559fc08688333562ea9`;
- five registered Python tests, both compilation steps, link and C/C++
  self-test passed;
- all eleven source, repair, compiler-path, descriptor and ownership controls
  passed;
- diagnostic invocations: 0; model reads: 0; producer invocations: 0; donor
  graph executions: 0; reference graph executions: 0;
- total duration: 0.885 s;
- compiled binary SHA-256:
  `562ff63191bc78df7f08aeaf8762f6a465ed067bdc905625f8fe2911fc5cccb8`.

Recorded source identities include runner
`97536ab0130272626669d8c2f5a73b2d38eaf44423f255b2b32db2adee84b270`,
tests `56790a519f7585a16bc6aec1774186f2ff40616859eab4e6f7d38f306d79a5ed`,
probe `2e6a5cf9d93a79b57eaaf24cc6acdc8a139cf490a56395604f037918ccd266ec`,
addendum `fee73d944d2484d65aab0a45361676b64ae523cf68eaa51597a27a07395c9252`,
independent oracle
`15d9c3c2d397e09079a0a55d7c78f0b466214c0c5612dab84ef63bfec2653f61`,
and unchanged candidate header
`362abe0f3fddb6e49b55ce289cbb4212f3b5fa001b991b4a84db894e573d0ccd`.

## Authorization and non-claims

Commit these exact sources before execution. Then run the one `repair1`
scientific invocation authorized by addendum A in its separate raw directory.
Do not run a producer, donor graph or reference graph, and do not repeat either
earlier apparatus or VOID 1.

This apparatus establishes no primitive parity, production integration,
quality, generation, memory or rate result by itself.
