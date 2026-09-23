# STRAT-01 post-F16 block-0 FFN operator-chain apparatus VOID 1

**Status:** `VOID_APPARATUS_SELFTEST_FIXTURE_BOUNDS`

The first apparatus-only qualification did not execute a diagnostic, donor
graph, reference graph, or model. It compiled successfully and completed all
158 Python tests, but the new C self-test exited with Windows access violation
`0xC0000005`.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_block0_ffn_operator_cross_input_apparatus_20260923/`

`adjudication.json` SHA-256:
`423e16ea6ea3edb140d611df63821498f5c2d385cc15f0a9b8c1e796156d72e5`.

The defect was confined to the model-free fixture: it called the inherited
12,288-element terminal-sum helper on three four-element local arrays. The
scientific implementation, identities, arm construction, gates, helper-count
arithmetic, and frozen protocol were not reached or changed.

The repair removes that invalid fixture call and tests the already qualified
sum helper indirectly through the frozen arm/replay contracts. This VOID
contributes no scientific value and does not consume the one authorized
diagnostic invocation.
