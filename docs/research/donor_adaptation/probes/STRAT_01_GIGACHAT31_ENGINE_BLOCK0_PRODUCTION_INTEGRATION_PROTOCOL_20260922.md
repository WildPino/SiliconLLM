# STRAT-01 GigaChat 3.1 block-0 production integration protocol

**Frozen:** 2026-09-22, before implementation or accepted-artifact execution

**Cell:** `STRAT-01-ENGINE-BLOCK0-PRODUCTION-INTEGRATION`

## Question and non-duplication

Does the standard `benchmarks/phase60/engine.c`
`--strat01-gguf-rung2b` path pass the complete frozen block-0 dense boundary
after integrating exactly the double-RMSNorm and K-B Q5_0×Q8_0 semantics that
closed the separate combined diagnostic?

The [combined result](STRAT_01_GIGACHAT31_ENGINE_COMBINED_RMS_Q5Q8_PROPAGATION_RESULT_20260922.md)
already closes all attention/cache boundaries and the dense `ffn_norm`,
`ffn_up`, and `ffn_gate` projections. It does not install those semantics in
the standard Rung-2B path and does not measure their production propagation
through `ffn_swiglu-0`, Q6_K `ffn_out-0`, or terminal residual `l_out-0`.
Those three downstream boundaries and production-path identity are the new
estimand. No diagnostic threshold or isolated operator is reopened.

## Immutable bindings

| object | identity |
|---|---|
| accepted GGUF | 6,474,702,976 bytes; SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| pinned llama.cpp | `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` |
| combined adjudication | `0a0aac6a58fe4bbf37d414d80d5681afdf357c7fda846596bea10685092c1b77` |
| accepted Rung-2B source run manifest | `5fa1ca6317c3afb466df937f4a9f2e622c38f7d3787d84d2874b2e8afb6ca0b7` |
| accepted Rung-2B reference root manifest | `dc856554fef8b538a9a699363f78989c9dbfc9c2e7263ad0509a683aa97254b5` |
| tokens | `[1,72,14,14129,14,2135,1512,2015]` |
| positions | `[0,1,2,3,4,5,6,7]` |
| schedules | `prefill8` and `cached7p1` |

The production path must reproduce these combined intermediate payload hashes
in both schedules before downstream adjudication:

| checkpoint | SHA-256 |
|---|---|
| `ffn_norm-0` | `4b17c45fcfc6573f9a5e1461e4f2c232a9536d6c8cf689884a6c8050ee0b2632` |
| `ffn_up-0` | `34a98ab5d44a7c1282901db0be2e152ea1a37f0cdf88366e65acd0597dbc390f` |
| `ffn_gate-0` | `bb52399fa69bc0f9295c0b2b2fc9588ec7689e440b7306f6df2e43f53c3710fa` |

Reference tensors are immutable and may not be regenerated. Timing observed
during this cell is inadmissible and must not enter `SPEED_LEDGER.md`.

## Production integration contract

The change is limited to shared non-diagnostic STRAT-01 primitives and their
standard Rung-2A/Rung-2B callers:

1. attention-input, compressed-KV, and final FFN RMSNorm use the already
   measured double-precision sum-of-squares accumulation with unchanged output
   arithmetic and epsilon;
2. `blk.0.attn_k_b.weight` uses the already measured pinned Q8_0 activation
   quantizer and Q5_0×Q8_0 dot semantics, with the frozen head stride and row
   layout;
3. Q/KV Q4_K×Q8_K, RoPE, F16 cache conversion, causal attention, V-B,
   attention output, Q4_K gate/up, SwiGLU, Q6_K down, and residual order remain
   unchanged;
4. the standard production `CONFIG` explicitly records double RMSNorm and
   Q5_0×Q8_0 K-B semantics;
5. prior diagnostic commands remain available for historical reproduction but
   are not called by the production command.

Factor shared Q8_0/Q5_0 primitives out of diagnostic-only ownership rather
than copying a second production implementation. Model-free tests must compare
the shared primitive with the frozen diagnostic outputs and reject omitted
activation quantization, wrong head stride, and corrupted Q5 high bits.

## Frozen gates

Use the original Rung-2B gates unchanged for every checkpoint in both arms:

- `ffn_norm-0`, `ffn_up-0`, `ffn_gate-0`, `ffn_swiglu-0`, and `ffn_out-0`:
  NRMSE `<= 0.002`, normalized maximum `<= 0.01`;
- terminal `l_out-0`: NRMSE `<= 0.001`, normalized maximum `<= 0.005`;
- production prefill/cached token-7 continuity for all six checkpoints: NRMSE
  `<= 2e-6`, normalized maximum `<= 1e-5`.

Validity additionally requires exact intermediate hashes above, complete and
finite payloads, exact source/config/artifact identity, all causal controls,
`errors=[]`, one standard production C invocation, and zero reference
invocations. The runner must report every checkpoint, not only failures.

## Decision rule

- **`PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION`** if production identity,
  intermediate hashes, all twelve arm/checkpoint comparisons, all continuity
  checks, and all controls pass.
- **`FAIL_ENGINE_BLOCK0_PRODUCTION_INTEGRATION`** if the run is valid and the
  three frozen intermediate hashes pass, but any downstream SwiGLU, down, or
  terminal gate fails. Report the first failed boundary; do not add another
  coordinate in this cell.
- **`VOID_ENGINE_BLOCK0_PRODUCTION_INTEGRATION`** for any identity, source,
  config, hash, completeness, finiteness, control, build, or execution-count
  failure.

Exactly one non-VOID production execution is allowed. Preserve every VOID in
a unique directory; only a narrow apparatus repair may proceed without
changing the estimand or rerunning a completed scientific cell.

## Non-claims and stop rule

This cell establishes at most one dense block on the standard C path. It does
not establish block-1 MoE/Rung 2C, later layers, tokenizer/logits, generation,
quality, RAM, or rate. Document and index the result before deciding whether
Rung 2C is authorized.
