# STRAT-01 layer-1 Q6 down cross-input protocol

**Frozen:** 2026-09-23, before implementation or execution

**Cell:** `STRAT-01-ENGINE-LAYER1-Q6-CROSS-INPUT`

**Purpose:** determine whether the first remaining Rung-2C failures originate
in Q6_K×Q8_K semantics on the actual layer-1 routed/shared down tensors or in
amplification of the admitted upstream SwiGLU residuals.

## No-duplication checklist

The closest result is
[`FAIL_ENGINE_REFERENCE_GENERIC_PROPAGATION`](STRAT_01_GIGACHAT31_ENGINE_REFERENCE_GENERIC_PROPAGATION_RESULT_20260923.md).
It moves the first failures to routed `ffn_moe_down-1` and shared
`ffn_shexp-1`; both preceding SwiGLU tensors pass their `0.002` NRMSE gate.

The earlier [block-0 down cross-input](STRAT_01_GIGACHAT31_ENGINE_FFN_DOWN_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md)
proved that the current Q6 helper passes on exact block-0 SwiGLU input. It does
not settle this coordinate: layer 1 uses width 1280 instead of 8960, different
Q6 tensors, four token-dependent selected experts, and a separate shared
expert. Conversely, local SwiGLU PASS does not prove its residual is harmless
after matrix amplification.

Both production schedules have identical metrics at this boundary, all 64
continuity checks pass, and candidate/reference top-4 IDs are exact. This cell
therefore uses only immutable `prefill8` payloads. Repeating `cached7p1` would
duplicate the same direct estimand.

## Changed coordinate and hypothesis

Production code does not change. The diagnostic crosses input origin through
the unchanged layer-1 Q6 paths:

1. captured reference routed SwiGLU → current selected-expert Q6 down;
2. captured current routed SwiGLU → current selected-expert Q6 down;
3. captured reference shared SwiGLU → current shared-expert Q6 down;
4. captured current shared SwiGLU → current shared-expert Q6 down.

The selected-expert arms use the exact frozen `token,slot` top-4 IDs to view
the corresponding expert slices of `blk.1.ffn_down_exps.weight`. The shared
arms use `blk.1.ffn_down_shexp.weight`. No router, up/gate, SwiGLU, attention,
or graph computation is permitted.

Primary hypothesis: exact reference SwiGLU inputs make both current Q6 paths
pass, while current-input arms exactly replay the measured failing outputs.
That would attribute sufficiency to upstream SwiGLU residual amplification,
not to Q6 semantics.

## Immutable identities

- Accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Propagation adjudication SHA-256:
  `79a19806fcf80df3e4d8a84335eaaecfc7beb9aef543f021dd1da01f6558b9c3`.
- Candidate `prefill8` manifest and reference `prefill8` manifest are bound by
  the runner before any diagnostic output is accepted.
- Exact top-4 IDs, common to candidate/reference, SHA-256:
  `557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95`.

| payload | bytes | SHA-256 |
|---|---:|---|
| current routed SwiGLU | 163,840 | `ba62cdfd1689cd299dd476e3ac8a31d1b5e374a4028596c8472ef2b7345cf8fa` |
| reference routed SwiGLU | 163,840 | `5097dc8599d0635477ad83f63f7e2de21b659775abc345dc933ce1e173be29e8` |
| current routed down | 196,608 | `2a2d153787683a520a5c6e3632406a33de2b7b22d0b47ad33521eb0d57f6ffe8` |
| reference routed down | 196,608 | `d19ce37968e1c2c6b04dffbc56ebba7ed25d704db2447e45031786b4710bb0af` |
| current shared SwiGLU | 40,960 | `30ec7ceafc08c3d30b73c21919a7d31619f69b9cd8045421e00c2e38af81a6b8` |
| reference shared SwiGLU | 40,960 | `fa0d1b4fe5da3f561ac30b8b8cb95520f359232528ea66c1d60e9244642c6003` |
| current shared down | 49,152 | `dd05bd9e815a4a2e7e2957004b157b8ae48f12a60019409963f17e39583051dd` |
| reference shared down | 49,152 | `7bc8b2104cacb7742e57df0c34a1af64fadbff8cc61006ed1aedc034a030ddae` |

The runner must also verify the tensor descriptors:

- `blk.1.ffn_down_exps.weight`: Q6_K, logical shape `[1280,1536,64]`;
- `blk.1.ffn_down_shexp.weight`: Q6_K, logical shape `[1280,1536]`.

## Gates and controls

For each routed and shared direct output, compare to its immutable reference
target with the unchanged general gate:

- NRMSE `<= 0.002`;
- normalized maximum `<= 0.01`.

The routed aggregate is also reported per token and slot, including expert ID,
but the frozen decision uses the complete `[8,4,1536]` payload. The shared
aggregate is reported per token over `[8,1536]`.

Mandatory controls:

- current-input routed/shared outputs are byte-exact to the two captured
  candidate outputs;
- captured reference down payloads pass trivially against themselves;
- top-4 IDs are exact, in range `[0,63]`, and each selected expert slice is
  descriptor-bounded;
- Q8_K inputs generated from the same source payload are deterministic;
- one-byte mutations of every admitted payload are refused;
- negated reference SwiGLU inputs reject their corresponding output gates;
- changing one routed expert ID changes output and rejects the reference gate;
- source inventory, compiler, model, descriptors, input hashes, output
  containment, and non-finite checks pass;
- donor and reference graph executions are both zero.

## Decision and stop rule

- `LAYER1_SWIGLU_RESIDUAL_SUFFICIENT` if both exact-reference-input Q6 arms
  pass and both current-input arms replay their captured failures exactly.
- `LAYER1_Q6_OPERATOR_RESIDUALS` if both exact-reference-input arms fail valid
  gates while current replay and all controls pass.
- `ROUTED_Q6_OPERATOR_RESIDUAL` if only the routed exact-input arm fails.
- `SHARED_Q6_OPERATOR_RESIDUAL` if only the shared exact-input arm fails.
- `VOID_LAYER1_Q6_CROSS_INPUT` for any apparatus, identity, replay,
  descriptor, mutation, causal-control, or accounting failure.

Every non-VOID outcome closes this exact cross-input cell. It authorizes only
a separately frozen repair/attribution at the branch named by the result. It
does not authorize a graph rerun, later-layer expansion, quality, RAM, or rate
claim. No outcome updates `SPEED_LEDGER.md`.

Planned raw directories:

- apparatus:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_q6_cross_input_apparatus_20260923/`;
- scientific:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_q6_cross_input_20260923/`.
