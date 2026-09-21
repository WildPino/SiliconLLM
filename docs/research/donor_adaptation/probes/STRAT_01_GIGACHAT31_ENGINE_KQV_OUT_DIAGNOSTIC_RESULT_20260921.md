# STRAT-01 GigaChat 3.1 `kqv_out-0` attribution result

**Verdict:** `PARTIAL_VB_Q8K_ATTRIBUTION`
**Date:** 21 September 2026
**Scope:** offline attribution of the first remaining block-0 mismatch; zero donor executions and no production-engine, quality, RAM, or speed claim

## Result

Replacing the dequantized-Q4_K/F32 V-B control with exact Q4_K×Q8_K
semantics removes most, but not all, of the immutable `kqv_out-0` error.
The project implementation and pinned GGML oracle agree tightly with each
other, proving that the remaining target mismatch is not an implementation
error in the new standalone Q8_K operator.

| Arm | NRMSE vs target | Normalized maximum | Frozen target gate | Result |
|---|---:|---:|---:|---|
| D32 control | `0.01199013510` | `0.00700039312` | `0.002` / `0.01` | FAIL (NRMSE) |
| project Q8_K | `0.001253480487` | `0.001516979052` | `2e-6` / `1e-5` | FAIL |
| pinned Q8_K | `0.001253480441` | `0.001516964641` | `2e-6` / `1e-5` | FAIL |

Project-versus-pinned Q8_K passes its independent tight gate at NRMSE
`1.244060665e-7` and normalized maximum `7.686301734e-8`; their Q8_K byte
streams are identical.  Relative to D32, exact Q8_K improves target NRMSE by
about `9.56x` and normalized maximum by about `4.61x`, and crosses the old
Rung-2A NRMSE gate.  It nevertheless misses the preregistered tight target
gate by a wide margin.  Therefore V-B activation quantization is a material
cause, but the frozen reconstruction cannot distinguish the residual between
causal-attention arithmetic and other V-B graph/layout semantics.

## Controls and provenance

- Computational commit: `66db65960728b2e5b1f9f99c385fe47c11ff4c66`.
- Exact GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Reconstructed latent SHA-256:
  `88990b6ba96f21a001fa591a0fe63365531f31672fbb43360467df3a2407465d`.
- Diagnostic helper SHA-256:
  `cbbd532cb0daea181813e48033e91f3cb5c4925f26cd34173707b8c96c737b58`.
- Identity/schedule equality, model-free tests, helper self-test and exact
  Q8_K bytes all pass.
- The one-byte Q8_K mutation, head transpose and packed-scale-layout error
  fire at target NRMSE `0.426116`, `0.982033` and `0.332406` respectively.
- `donor_executions=0`; the helper read only immutable traces and the exact
  V-B span.

Computational raw adjudication:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_kqv_out_diagnostic_20260921/adjudication.json`,
SHA-256 `859b59d36634ded8e482a87fa1674417cbbd1c70f035f9a03ebe1dc2e1df9e55`.

Canonical offline adjudication:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_kqv_out_offline_adjudication_20260921/adjudication.json`,
SHA-256 `e85e53a0dc7a6154ae9b26beccaa751019fb470d099cb2901225627b8256d9fa`.

## Classification-only apparatus repair

The completed computational run reported
`ATTENTION_RECONSTRUCTION_OR_OTHER_VB_SEMANTICS_REMAIN` because its runner
made the partial branch depend on first passing the tight target gate.  That
condition contradicted the frozen table, where partial means materially
improved but still out of gate.  Commit `630b450` repaired only this branch and
re-adjudicated the hash-bound products offline.  No helper, model, arithmetic,
input, metric, threshold, or control was rerun or changed.  The original raw
label remains preserved as apparatus history.

## Consequence and no-duplication boundary

Do not repeat this reconstruction/Q8_K cell and do not install V-B Q8_K in the
production engine from this result alone.  The next coordinate must capture
the pinned graph's existing internal `kqv-0` (before V-B) and `kqv_mla-0`
(after V-B, before final permutation/contiguity) callbacks.  Comparing the
reconstructed latent directly with `kqv-0`, and applying project/pinned V-B to
the true latent, can separate attention reconstruction from residual V-B
semantics.  Rung 2B, quality and speed remain unauthorized;
`SPEED_LEDGER.md` is unchanged.
