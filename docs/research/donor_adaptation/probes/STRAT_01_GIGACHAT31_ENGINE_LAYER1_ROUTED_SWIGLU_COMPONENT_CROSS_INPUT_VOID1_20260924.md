# STRAT-01 layer-1 routed-SwiGLU component cross-input — VOID 1

**Status:** `VOID_LAYER1_ROUTED_SWIGLU_COMPONENT`

The first scientific invocation was correctly refused before any Q6 arm by a
contradictory apparatus precondition. The original protocol text required
both scalar recomputations to replay their corresponding captures. But
`scalar(reference gate, reference up)` is itself the expression-semantics
estimand and cannot simultaneously be required to equal the reference
capture.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_swiglu_component_cross_input_20260924/`

| record | value |
|---|---|
| `adjudication.json` SHA-256 | `016babb6891e699a3f0bbbce7259bc17f923ac3774eac875eb788c46b5d1eab0` |
| binary SHA-256 | `6e99fad2e9ed12722051d60d4101b3c1dca78aebdfdac43050886f81f787a1c5` |
| producer commit | `4f79078d82b44fc522df019df50cdcb1ccfe6f5e` |
| diagnostic invocations | `1` |
| donor/reference graphs | `0 / 0` |
| diagnostic error | `layer-1 routed-SwiGLU scalar capture replay mismatch` |

The runner's top-level `q6_arms = 9` field was prospective and is not an
execution counter. Source order proves the refusal occurred before the Q6
loop; repair 1 replaces this field with an exact completed-arm counter in both
success and failure records.

No output arm, causal verdict, candidate repair, quality, RAM, or rate claim
is admissible from this run. Repair 1 changes only the contradictory replay
assertion and counter provenance; the scientific matrix remains frozen.
