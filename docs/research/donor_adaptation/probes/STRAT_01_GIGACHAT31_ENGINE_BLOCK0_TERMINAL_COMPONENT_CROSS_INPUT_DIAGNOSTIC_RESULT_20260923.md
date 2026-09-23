# STRAT-01 block-0 terminal-component cross-input result

**Verdict:** `FFN_RESIDUAL_SUFFICIENT`

The single invocation authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md)
completed at implementation commit
`1a1fb47108fd8d167a76c9d908d3a7b9bee53a16`. It executed zero donor graphs
and zero reference graphs. The raw record is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_block0_terminal_component_cross_input_20260923/`

The canonical adjudication and run-manifest SHA-256 values are
`edbb82a814a92be9f822d37f6683375dfa437b5b0774cbae13b86632d2a9a6f5`
and `4f46f0781e823b9d2fe2f77b01e882ce6f420bf1c9b3f6ce0434f79d85ce8514`.
The run completed in 42.626 seconds and reports no error.

## Frozen-arm outcome

Each arm was constructed from the immutable captured addends using one F32
addition per element, then passed through the unchanged qualified layer-1
builder. The unchanged acceptance limits are NRMSE `<= 0.002` and normalized
maximum `<= 0.01` at every checkpoint.

| arm | composed `l_out-0` SHA-256 | first failure | `kqv_out-1` NRMSE | normalized max | gate |
|---|---|---|---:|---:|---|
| reference `ffn_inp` + reference `ffn_out` | `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa` | none | `0.0003855998766858119` | `0.0011237641335101218` | PASS |
| C `ffn_inp` + C `ffn_out` | `258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44` | `kqv_out-1` | `0.006136167591602251` | `0.0030491102772673647` | FAIL |
| C `ffn_inp` + reference `ffn_out` | `3103e6f7ff3c226d09b2c532c8796ae91e1b4bdaa98399780eb355cf66c87363` | none | `0.0009139590512046808` | `0.000991825271707923` | PASS |
| reference `ffn_inp` + C `ffn_out` | `b822e425fb6c88faa62cc59dec8011a7869ebb9d64c3731e54beb585f66cb541` | `kqv_out-1` | `0.0061366360889986695` | `0.0030491102772673647` | FAIL |

All ten pre-attention checkpoints pass in all four arms. In the
C-attention/reference-FFN arm, even the terminal `kqv_out-1` remains inside
gate. Conversely, substituting only the captured C FFN addend into the exact
reference attention-stream term reproduces essentially the full downstream
failure. The difference is not a marginal threshold crossing: its NRMSE is
more than three times the limit.

## Replay and controls

- The two homogeneous sums reconstruct their frozen `l_out-0` payloads with
  zero mismatched bytes.
- C/C byte-replays all eleven captured C layer-1 checkpoints and every frozen
  C/reference metric within `1e-12`.
- Reference/reference passes all eleven gates and reproduces the preceding
  layer-1-start result.
- One-byte mutations of all six admitted component/replay payloads are
  refused.
- Addend-origin relabeling is rejected.
- Negating hybrid token 7 rejects at `kqv_out-1`: NRMSE
  `0.8199025952931809`, normalized maximum `0.9857931633223015`.
- Swapping hybrid rows 0 and 7 rejects: NRMSE `1.0468744495737792`,
  normalized maximum `0.7569639836923044`.
- New C self-test: 9 checks PASS; shared layer-1-start self-test: 8 checks
  PASS; new Python tests: 5 PASS; legacy kernel checks: 73,024 PASS with zero
  worst error; compiler stderr is empty.

## Interpretation and next boundary

At the captured terminal-addend boundary, the accepted attention-stream term
is not sufficient to breach the layer-1 gates when paired with exact reference
`ffn_out-0`. The captured C `ffn_out-0` residual is sufficient when paired
with exact reference `ffn_inp-0`. This closes the terminal-component split and
forbids repeating it, changing its gates, repairing layer 1, or reopening
block-0 attention.

The next distinct diagnostic boundary is inside the production of
`ffn_out-0`. It should cross the immutable C/reference `ffn_swiglu-0` payloads
through the unchanged C Q6_K down-projection operator, then compose each
result with exact reference `ffn_inp-0` and propagate through the already
qualified layer-1 builder. That zero-producer cell can distinguish a
sufficient upstream SwiGLU-input residual from a sufficient Q6_K down-operator
residual under the downstream gate. It must be separately frozen before code
or execution.

## Non-claims

This intervention attributes sufficiency only to the captured terminal FFN
addend. It does not yet identify whether the responsible residual originates
in FFN normalization, up/gate Q4_K projections, SwiGLU, Q6_K down projection,
or their interaction. It is not a natural mixed forward pass and makes no
claim about Rung-2C repair, MoE, later layers, tokenizer, logits, generation,
quality, RAM, or rate. No `SPEED_LEDGER.md` update is due.
