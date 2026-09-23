# STRAT-01 reference-generic Q4 downstream-propagation result

**Verdict:** `FAIL_ENGINE_REFERENCE_GENERIC_PROPAGATION`

The frozen changed-coordinate cell is valid and materially improves the
production trajectory, but it does not close all Rung-2C gates. Exact
reference-generic Q4_K×Q8_K arithmetic closes the former block-0 terminal,
layer-1 attention, router, up/gate, and SwiGLU failures. The first remaining
failures are now the Q6_K down projections of both the routed experts and the
shared expert.

This is a numerical FAIL, not an apparatus VOID: `errors=[]`, the changed
coordinate reaches both graphs, and all identity, continuity, cache, source,
and causal controls pass.

## Immutable execution record

- implementation commit:
  `cfc125013eb792d089cd33957569e642a58067d3`;
- accepted GGUF SHA-256:
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- raw directory:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_reference_generic_propagation_20260923/`;
- `adjudication.json` SHA-256:
  `79a19806fcf80df3e4d8a84335eaaecfc7beb9aef543f021dd1da01f6558b9c3`;
- C binary SHA-256:
  `ac734e50c283def08d4e122716f6e875ca7a2e8448355873fde4e5bed9d4f315`;
- donor producer invocations/graphs: `1 / 2` (`prefill8`, `cached7p1`);
- reference producer invocations/graphs: `0 / 0`.

Both candidate schedules start from the same new block-0 terminal hash,
`29f7e1d9af96df1e419088c3e64fbcb8473e85a80f44ff050e463bcf19f0d8fd`.
It differs from the frozen old-C hash, proving that the intervention reached
the graph. The immutable reference start remains
`385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa`.

## Result surface

The two schedules are numerically identical at every displayed checkpoint.
Checkpoint passes improve from the original Rung-2C **34/64** to **52/64**.
All **64/64** continuity comparisons, **6/6** cache comparisons, and all ten
causal negative controls pass.

| checkpoint | old NRMSE | new NRMSE | gate | new result |
|---|---:|---:|---:|---|
| `l_out-0` | `5.6057546e-4` | `9.9690826e-8` | `0.002` | PASS |
| `kqv_out-1` | `0.00613617` | `0.000353432` | `0.002` | PASS |
| `ffn_inp-1` | `0.00266481` | `0.000256193` | `0.001` | PASS |
| `ffn_norm-1` | `0.00397785` | `0.000353176` | `0.002` | PASS |
| routed `ffn_moe_up-1` | `0.00750198` | `0.00158460` | `0.002` | PASS |
| routed `ffn_moe_gate-1` | `0.00702041` | `0.00148423` | `0.002` | PASS |
| routed `ffn_moe_swiglu-1` | `0.00725855` | `0.00149066` | `0.002` | PASS |
| routed `ffn_moe_down-1` | `0.0127456` | **`0.00385395`** | `0.002` | **FAIL** |
| shared `ffn_up-1` | `0.00702094` | `0.00152029` | `0.002` | PASS |
| shared `ffn_gate-1` | `0.00389859` | `0.000847520` | `0.002` | PASS |
| shared `ffn_swiglu-1` | `0.00662075` | `0.00134965` | `0.002` | PASS |
| shared `ffn_shexp-1` | `0.00950035` | **`0.00288634`** | `0.002` | **FAIL** |
| `ffn_out-1` | `0.00777338` | `0.00241334` | `0.002` | FAIL |
| `l_out-1` | `0.00594077` | `0.00180772` | `0.001` | FAIL |

The normalized-maximum gate passes at every listed boundary, including both
first failures (`0.00261083` routed and `0.00105641` shared). The verdict is
therefore driven by the preregistered NRMSE limits, not an isolated outlier.
Router top-4 IDs remain exact.

## Interpretation

The reference-generic Q4 helper is causally useful and remains the accepted
fidelity implementation. It reduces block-0 terminal NRMSE by more than three
orders of magnitude and moves the first Rung-2C failure from attention output
to both Q6 down branches. It is nevertheless insufficient for full layer-1
parity and this propagation cell is closed.

The earlier block-0 down-projection result established that the current Q6
operator passes on exact block-0 SwiGLU input. It does not decide this new
boundary: Rung 2C uses different Q6 tensors, width 1280, four selected routed
experts per token, and a shared expert. Nor does a passing local SwiGLU gate
prove that its residual cannot be amplified by the down matrices.

The next non-duplicate experiment is therefore a separately frozen,
zero-producer layer-1 Q6 cross-input diagnostic. It must feed immutable
reference and current routed/shared SwiGLU payloads through the unchanged
selected-expert and shared-expert Q6 paths, require current/current byte
replay, and compare against immutable reference down outputs. This distinguishes
Q6 semantics on the actual layer-1 tensors from amplification of the admitted
SwiGLU residuals without rerunning either graph.

## Non-claims and stop rule

Do not rerun this propagation cell, relax its gates, reopen Q4, alter layer-1
attention/router/up/gate/SwiGLU, or infer a generic Q6 defect. The result does
not establish later-layer parity, tokenizer/logits/generation, C-path quality,
HumanEval, RAM, or rate. The 207-second orchestration duration is not emitted-
token throughput and does not update `SPEED_LEDGER.md`.
