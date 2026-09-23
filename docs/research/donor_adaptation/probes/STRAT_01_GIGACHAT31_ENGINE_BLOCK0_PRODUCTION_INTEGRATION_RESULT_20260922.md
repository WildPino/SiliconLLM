# STRAT-01 GigaChat 3.1 block-0 production integration result

**Result:** `PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION`

**Scope:** standard `--strat01-gguf-rung2b` production path for the accepted
GigaChat 3.1 artifact; exact installation of the already measured double
RMSNorm and Q5_0×Q8_0 K-B semantics; frozen `prefill8` and `cached7p1`
schedules; dense block 0 only. This is not Rung 2C and makes no later-layer,
tokenizer, logits, generation, quality, RAM, or rate claim.

## Canonical evidence

- Accepted artifact: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Production implementation commit:
  `2087e630ee9166ea20a89609e75805bfb3980bca`.
- Canonical run:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_block0_production_integration_20260922/`.
- Adjudication SHA-256:
  `58f661d9461f3dc17b2a52f830dcc9f36423621528941d475e23ee2aed8e0bf2`.
- Run-manifest SHA-256:
  `7b8de95a3766e4f76bde37f4038ece2986b9067ad9ac7a0e4cc11855a3062497`.
- Accepted Rung-2B source-run manifest:
  `5fa1ca6317c3afb466df937f4a9f2e622c38f7d3787d84d2874b2e8afb6ca0b7`.
- Accepted reference-root manifest:
  `dc856554fef8b538a9a699363f78989c9dbfc9c2e7263ad0509a683aa97254b5`.
- Required predecessor combined adjudication:
  `0a0aac6a58fe4bbf37d414d80d5681afdf357c7fda846596bea10685092c1b77`.
- Execution accounting: one standard production donor invocation, zero
  reference graph executions, `errors=[]`, and an empty failure list.

## Result

The standard production path reproduces the three frozen combined
intermediate hashes in both schedules:

| checkpoint | required and observed SHA-256 |
|---|---|
| `ffn_norm-0` | `4b17c45fcfc6573f9a5e1461e4f2c232a9536d6c8cf689884a6c8050ee0b2632` |
| `ffn_up-0` | `34a98ab5d44a7c1282901db0be2e152ea1a37f0cdf88366e65acd0597dbc390f` |
| `ffn_gate-0` | `bb52399fa69bc0f9295c0b2b2fc9588ec7689e440b7306f6df2e43f53c3710fa` |

All twelve candidate-versus-reference comparisons pass their preregistered
gates. Both schedules produce the same metrics:

| checkpoint | NRMSE | normalized maximum | NRMSE gate | maximum gate |
|---|---:|---:|---:|---:|
| `ffn_norm-0` | `2.3612375112384717e-5` | `3.108848345891282e-5` | `0.002` | `0.01` |
| `ffn_up-0` | `4.02014425233629e-4` | `4.2690695118983e-4` | `0.002` | `0.01` |
| `ffn_gate-0` | `3.4697064936483897e-4` | `2.951011547638659e-4` | `0.002` | `0.01` |
| `ffn_swiglu-0` | `3.0753678943417533e-4` | `1.064854879380218e-4` | `0.002` | `0.01` |
| `ffn_out-0` | `9.913516476391816e-4` | `1.0415762554367286e-3` | `0.002` | `0.01` |
| `l_out-0` | `5.605754630875132e-4` | `4.589515255807851e-4` | `0.001` | `0.005` |

All six production and all six pinned-reference prefill-versus-cached
continuity comparisons have exactly zero NRMSE and normalized maximum. The
four causal controls reject decisively: gate/up swap NRMSE `0.40326664`, SiLU
omission `1.05951249`, mutated Q6_K down input `1.0`, and final-residual
omission `0.82559748`.

The source controls also establish that the Q8_0 type, quantizer, and
Q5_0×Q8_0 dot primitive each have one shared implementation; all three
production RMSNorm sites use the pinned semantics; diagnostics delegate to the
shared helper; and the production configuration declares
`rms_accum=double;kb=q5_0xq8_0`.

The apparatus-only predecessor ran zero donor and reference graphs and closed
`APPARATUS_READY_NO_DONOR_EXECUTION`. Compilation, 34 Rung-2A/Rung-2B
self-test checks, all combined diagnostic self-tests, the 73,024-check legacy
kernel self-test, and five Python fail-closed tests passed before the sole
scientific execution.

## Interpretation and next gate

The exact semantics isolated by the earlier diagnostic chain now work in the
ordinary production Rung-2B path through terminal dense block-0 residual
`l_out-0`. The original `FAIL_ENGINE_RUNG2B` remains valid historical evidence
for the old implementation, but its diagnosed defect is repaired and the
production block-0 boundary is now closed positively. Do not repeat Rung 2A,
Rung 2B, any RMSNorm/K-B diagnostic, combined propagation, or this production
integration cell on the same artifact and estimand.

This PASS authorizes only the separately preregistered [Rung 2C
protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_PROTOCOL_20260923.md): one block-1
routed-expert plus shared-expert MoE layer, starting from the accepted block-0
terminal state. It does not authorize a full-model or speed claim. No
implementation or donor execution may widen that frozen coordinate. No
`SPEED_LEDGER.md` entry is due.
