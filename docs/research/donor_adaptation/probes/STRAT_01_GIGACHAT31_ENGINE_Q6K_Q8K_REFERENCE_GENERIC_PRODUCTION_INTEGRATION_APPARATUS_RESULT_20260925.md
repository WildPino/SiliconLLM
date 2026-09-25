# STRAT-01 Q6_K×Q8_K reference-generic production-integration apparatus result

**Date:** 2026-09-25

**Cell:** `STRAT-01-ENGINE-Q6K-Q8K-REFERENCE-GENERIC-PRODUCTION-INTEGRATION`

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

## Result

The model-free apparatus qualifies the production-only change frozen in the
[protocol](STRAT_01_GIGACHAT31_ENGINE_Q6K_Q8K_REFERENCE_GENERIC_PRODUCTION_INTEGRATION_PROTOCOL_20260924.md).
The normal shared Q6_K×Q8_K dispatch delegates directly to the already proven
non-inlined `no-avx,no-avx2,no-fma` reference-generic helper. Rung-2B and
Rung-2C declare that implementation in `CONFIG`; Q4, F16, SwiGLU, residual
order, routing, layouts, and the accepted artifact are unchanged.

The qualifying record is:

- raw directory:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_reference_generic_production_integration_apparatus_repair1_20260925/`;
- adjudication SHA-256:
  `8e6d642efc9efef3483fb2971e1f57535b171c693b6331dbcfee549f27005740`;
- observed pre-commit HEAD:
  `149dff78fdaeb306c93c5776eb57bc09dca06f61`;
- 300 registered Python tests passed in 164.497 s;
- 22 C self-test commands passed;
- exact candidate/oracle parity passed at lengths 256, 512, 1280, 1536, and
  8960, including the routed-expert width newly required by this cell;
- every source/ownership/configuration/unchanged-coordinate control passed;
- production invocations: 0; donor graph executions: 0; reference graph
  executions: 0;
- total apparatus duration: 178.857 s;
- compiled binary SHA-256:
  `1e717e42a39872ce019d58c031d6ba5e3d9c32cff5a3ec5ca5411f87d17e6700`.

The source identities recorded by the apparatus include production runner
`1ca23418f928954035d2a28e699ccbaf07dc9d39376ab3abd84867b9924405d6`,
tests `b5021ba7082e918b6c4ede50bc54bb69b13ebf244454c2f92b4c0102c98dfb04`,
protocol `32b42ab95a9e5a7c72f38e1a59bfa6dbd44d374b18827e123e0057f4b6adc81d`,
Rung-2B header
`ff6ee610640be7baad873808beac4a9a653c4f8a9fa7a77f0bc187aa733fdfaf`,
and Rung-2C header
`2ef202bcb475b32e4079718637a094b36789e3eda4c4fc4ba339df250313c01b`.

## Preserved interrupted attempt

The initial apparatus directory without the `repair1` suffix is intentionally
retained as an interrupted, non-evidentiary attempt. It wrote no adjudication,
opened no model, and executed no donor or reference graph. Its full registered
test run found one stale historical assertion that required the old `CONFIG`
line to end immediately after `kb=q5_0xq8_0`. The repair only changed that
test to accept the newly appended, explicitly required Q6 implementation
marker; it did not change an operator, gate, payload, or scientific decision.
The repaired apparatus then passed the entire suite.

## Authorization and non-claims

Commit the exact qualified sources before execution. Then, and only then, run
one standard two-schedule Rung-2C producer against the accepted GGUF and the
immutable reference manifests. Do not run a reference graph and do not repeat
compile parity, cross-input diagnostics, or the apparatus.

This result authorizes that single scientific invocation. It establishes no
model fidelity, generation quality, throughput, RAM, or later-layer claim by
itself, and it does not change `SPEED_LEDGER.md`.
