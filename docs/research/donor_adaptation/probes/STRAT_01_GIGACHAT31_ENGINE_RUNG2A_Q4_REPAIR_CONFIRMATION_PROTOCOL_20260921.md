# STRAT-01 GigaChat 3.1 changed-coordinate block-0 attention confirmation

**State:** FROZEN PROTOCOL; NOT YET EXECUTED  
**Date frozen:** 21 September 2026  
**Scope:** confirm the integrated Q4_K×Q8_K repair at the first Rung-2A failure and localize the next downstream mismatch; no rewrite of old Rung 2A, no 2B, quality, RAM, or speed

## Question and changed coordinate

The old Rung-2A cell is closed `FAIL_ENGINE_RUNG2A`.  Its projection diagnostic
isolated missing Q8_K activation semantics, and the separate repair cell is
closed `PASS_ENGINE_Q4K_Q8K_REPAIR`.  The next question is deliberately
narrow:

> When the repaired operator is integrated into the complete C block-0 MLA
> path, do both arms cross the old first-failure Q/KV boundary, and where is
> the first remaining disagreement against the already captured pinned
> reference?

Relative to the old C capture, the only authorized semantic change is the
committed generic Q4_K stored-row path: F32 activation rows are quantized once
to Q8_K and reused for Q4_K×Q8_K dots.  This affects Q, KV and the generic
output projection.  The special-layout V-B path and Q5_K K-B path remain
unchanged and are candidates for the next localized failure, not silently
repaired coordinates.

## Frozen evidence and execution

- exact GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- old complete capture:
  `strat01_gigachat_engine_rung2a_repair2_20260921/`, run-manifest SHA-256
  `0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f`;
- pinned reference revision:
  `llama.cpp 5b335f413e4f73b0809c4fe39af894efbcc6a0d2`;
- fixed token IDs `[1,72,14,14129,14,2135,1512,2015]`, positions `0..7`,
  paired `prefill8` and `cached7p1` schedules, F16 compact K-only cache;
- repaired-operator result and accepted implementation commit:
  `PASS_ENGINE_Q4K_Q8K_REPAIR` at
  `7cb1edfef07836ae0896c5066115a30007d3500b`.

Execute exactly one newly built C block-0 path.  Do not execute the pinned
reference again: validate and consume its immutable captured payloads.  Record
`c_donor_executions=1`, `reference_donor_executions=0`.  Preserve build/test
logs, hashes, complete C tensor/cache payloads, all comparison metrics and the
ordered first-failure report.

## Controls and gates

Before model execution:

1. verify all frozen identities and committed critical sources;
2. pass the 20 repair/oracle/negative/legacy tests and the 73,024-check kernel
   self-test;
3. validate every pinned-reference manifest, tensor, shape, operation,
   ordinal, token decomposition and payload hash;
4. validate the old captured C Q/KV negative control: both arms must still
   fail their old `0.002` NRMSE gate before the changed coordinate.

Primary confirmation cells, in both `prefill8` and `cached7p1`:

- `attn_norm-0` must remain within NRMSE `2e-6` and normalized maximum `1e-5`;
- `q-0` and `kv_cmpr_pe-0` must each pass NRMSE `2e-6` and normalized maximum
  `1e-5`;
- C prefill/7+1 continuity at terminal `ffn_inp-0` must pass the same tight
  gate.

For every later tensor and all three cache checkpoints, preserve the old
Rung-2A metrics and thresholds (`0.002`/`0.01`, terminal
`0.001`/`0.005`) solely to identify the first remaining failure in execution
order.  A downstream failure does not negate successful confirmation of the
Q4 repair and does not authorize threshold tuning.  If all old full-attention
gates happen to pass, record that observation explicitly, but do not rewrite
the historical Rung-2A result.

## Adjudication

| Label | Frozen rule |
|---|---|
| `PASS_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION` | Apparatus/controls valid; both schedules pass `attn_norm-0`, Q and KV tight gates; C continuity passes; complete downstream localization is present. |
| `FAIL_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION` | Apparatus valid, but at least one required early or continuity cell fails. |
| `VOID_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION` | Identity, source, build/test, immutable reference, negative control, payload, schema, shape, finiteness, or completeness is invalid. |

Stop after one non-VOID result.  Preserve any VOID in its own directory; a
narrow apparatus repair may not change semantic code, evidence, metrics,
thresholds, labels, or ordering.

## Eight-item control card

1. **Nearest prior cell:** measured [Q4_K×Q8_K repair](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REPAIR_RESULT_20260921.md).
2. **Changed coordinate:** integrate that already accepted operator into the complete C block-0 schedule.
3. **Why prior is insufficient:** isolated projection CLI parity does not prove the full schedule dispatches the repaired path or localize the next mismatch.
4. **Identity/controls:** exact model, existing paired reference traces, old failing-C negative control, repair oracle suite, legacy tests and hashes.
5. **Separate gates:** early integration confirmation versus downstream localization; neither is quality or speed.
6. **Claim labels:** old Rung-2A FAIL and repair PASS remain measured; this cell is proposed until adjudicated.
7. **Void/stop:** preserve voids; one non-VOID execution; no tuning or reference rerun.
8. **Raw/canonical record:** planned raw directory `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_q4_repair_confirmation_20260921/`; future result beside this protocol.

## Consequence boundary

A pass identifies the next operator boundary from complete block-0 evidence.
It authorizes only a separately frozen repair/diagnostic for that boundary, or
2B only if every full-attention gate passes and a new protocol says so.  It
does not establish tokenizer, logits, generation, C-path quality, RAM, or
accepted-token throughput.  No `SPEED_LEDGER.md` update is due.
