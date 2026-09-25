# STRAT-01 layer-1 numerical-primitives production-integration result

**Date:** 2026-09-25

**Cell:** `STRAT-01-ENGINE-LAYER1-NUMERICAL-PRIMITIVES-PRODUCTION-INTEGRATION`

**Status:** `FAIL_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION`

## Result

The sole authorized standard Rung-2C producer is valid and narrows the first
remaining residual to normalized selected-router weights. Both schedules are
byte-exact through:

- repaired block 0 and complete layer-1 attention;
- `ffn_norm-1`;
- F32 router logits, sigmoid probabilities, selection bias and ordered top-4;
- selected unbiased router weights;
- routed expert up/gate/SwiGLU/down outputs;
- shared up/gate/SwiGLU/down output.

The first failing checkpoint in both schedules is
`ffn_moe_weights_norm-1`: 11/32 values differ, with maximum absolute error
`5.960464477539063e-8`, NRMSE `5.617113951337541e-8`, and normalized maximum
`7.559676082999196e-8`. The mismatch then propagates only through routed
weighting/reduction, `ffn_out-1`, and terminal `l_out-1`.

The raw directory is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_numerical_primitives_production_integration_20260925/`.
Its adjudication SHA-256 is
`3cec456bc7b1d2798976cb9544cd91d36d45451b94911efa1f0ec1ac97437816`.
Execution commit was `25547e97fd2b70c3125675c898ff593c16c559d3`;
the binary SHA-256 was
`0447b4b35427e15911d369757797832d84eed4c450923f8e1f9706e555f4dcb9`.

## Passed gates

- 54/64 schedule-checkpoint comparisons are exact;
- all 64 candidate/reference schedule-continuity comparisons pass;
- all six cache comparisons and all integer routing payloads pass;
- all causal negative controls and all 13 production source controls pass;
- repaired block-0 `l_out-0` has the exact closed SHA-256
  `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa`
  in both schedules;
- the three installed primitive hashes exactly equal their local results in
  both schedules: router `8188ff0d...a0bbf5`, routed SwiGLU
  `5097dc85...e29e8`, shared SwiGLU `fa0d1b4f...64c6003`.

One producer invocation completed exactly two donor schedules. No reference
graph ran. There were no errors; total duration was 195.278 s.

## Interpretation and next boundary

Close the router projection, sigmoid, bias, top-4, gather, both SwiGLU
populations, every up/gate/down projection, Q6 and shared path. Do not rerun
this producer or any earlier cell.

Post-result source inspection shows that pinned `ggml_sum_rows` calls
`ggml_vec_sum_f32`, whose `ggml_float` accumulator is double before one F32
conversion; production sums the four selected F32 weights directly in F32.
An unregistered exploratory calculation on the preserved exact operands made
the double-sum candidate byte-exact 32/32, while float sequential and pairwise
sums remained at the observed 11/32 mismatch. This is hypothesis-generation,
not the decision record.

The only next action is the preregistered zero-graph
[selected-weight normalization compile-parity protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTER_WEIGHT_NORMALIZATION_COMPILE_PARITY_PROTOCOL_20260925.md).
No quality, generation, memory or rate claim follows.
