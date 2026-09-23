# STRAT-01 exact F16 reduction propagation result

**Verdict:** `FAIL_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION`

The frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION_PROTOCOL_20260923.md)
closed validly. The exact historical `GGML_CPU_GENERIC` F16 reduction reached
both accepted C schedules and materially reduced every reported downstream
NRMSE, but it did not close the complete layer-1 gate. The first remaining
failures are the routed and shared Q6 down outputs.

This is a scientific FAIL, not an apparatus failure: `errors = []`, every
identity and causal control passed, and no reference graph ran.

## Result

Both schedules produced identical checkpoint metrics and exact route IDs.
Selected old-versus-new NRMSE values are:

| checkpoint | previous scalar F16 reduction | exact generic-F64 reduction | gate | result |
|---|---:|---:|---:|---|
| `kqv_out-1` | `3.53432e-4` | `5.03513e-5` | `0.002` | PASS |
| `ffn_inp-1` | `2.56193e-4` | `1.15479e-4` | `0.001` | PASS |
| `ffn_norm-1` | `3.53176e-4` | `1.39334e-4` | `0.002` | PASS |
| `ffn_moe_up-1` | `1.58460e-3` | `9.91159e-4` | `0.002` | PASS |
| `ffn_moe_gate-1` | `1.48423e-3` | `8.99088e-4` | `0.002` | PASS |
| `ffn_moe_swiglu-1` | `1.49066e-3` | `1.11190e-3` | `0.002` | PASS |
| `ffn_moe_down-1` | `3.85395e-3` | `3.76095e-3` | `0.002` | **FAIL** |
| `ffn_swiglu-1` | `1.34965e-3` | `8.78561e-4` | `0.002` | PASS |
| `ffn_shexp-1` | `2.88634e-3` | `2.37204e-3` | `0.002` | **FAIL** |
| `ffn_out-1` | `2.41334e-3` | `2.36174e-3` | `0.002` | **FAIL** |
| `l_out-1` | `1.80772e-3` | `1.76698e-3` | `0.001` | **FAIL** |

The intervention therefore removes most of the attention residual and
substantially improves both up/gate/SwiGLU branches. It is causally useful but
insufficient. No tolerance was relaxed.

## Localization

The remaining discrepancy is sharply concentrated at token 6. Its NRMSE is
`9.30982e-5` at `kqv_out-1`, grows to `0.00305498` at routed SwiGLU and
`0.0103737` at routed Q6 down, and reaches `0.00522802` at `l_out-1`.
The shared branch follows the same pattern: token-6 SwiGLU is `0.00247047`
and shared Q6 down is `0.00749379`. Other tokens remain inside their gates.

The already closed layer-1 Q6 cross-input result still rules out a standalone
Q6 defect: exact reference routed/shared SwiGLU inputs pass the same actual
matrices. The new evidence instead shows that a much smaller upstream input
residual remains sufficient after nonlinear and Q6 amplification.

## Validity and controls

- Stage A reproduced the pinned QK, pinned value, both scalar controls, and
  the complete conversion stream byte-for-byte.
- QK and value mutations rejected the tight `2e-6` / `1e-5` gate.
- Production helper counts were exact: 4,608 QK calls and 524,288 value calls
  in mode `pinned-generic-f64`.
- The attention hash changed in both schedules, proving intervention reach.
- Top-4 expert IDs matched exactly for all eight tokens; route-weight NRMSE
  was `2.76914e-5`.
- All continuity comparisons were exact, all six cache checks passed, and
  all negative controls passed.
- One accepted-artifact producer invocation executed two C schedules;
  reference producer and graph counts were zero.
- 141 STRAT-01 Python tests and 17 C self-tests passed.

## Canonical evidence

Scientific source commit:
`41001c884a8900f3b4261445664cccc5017ee99e`

Raw result directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_f16_vector_propagation_20260923/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `512e3ea7754c5e8fa067dab5692a8a192879be26895547cbedbb6023a896edcf` |
| scientific binary | `3df61759e385c23ebcb058edac2909132131dbd0140f64d6923e582387ec5369` |
| current `l_out-0` | `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4` |
| current `kqv_out-1` | `482e240b5723dfdccda2c8624674caff2848104c8a6bb6a66444bcf936198827` |
| current `l_out-1` | `06c08094299c0fa9901f589f8a7504e18d71a76f0d15b64d7444087031323157` |

## Consequence and stop rule

Close this exact reduction-order cell. Do not rerun Stage A or either graph,
and do not reopen F16 conversion, reduction order, Q4, SwiGLU, Q6, routing,
or cache continuity.

The next non-duplicate question is whether the now tiny accepted
`l_out-0` residual is itself sufficient under the new reduction semantics.
That requires an offline reference/current `l_out-0` cross-input through the
complete current layer 1, with zero graph executions. It must byte-replay the
new current arm and must not reuse the old pre-F16 C-input hash as evidence.
No tokenizer, logits, generation, quality, RAM, or rate claim follows.
