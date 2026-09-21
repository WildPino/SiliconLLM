# STRAT-01 GigaChat 3.1 F16 converter audit protocol

**State:** EXECUTED; CLOSED AS `FAIL_F16_CONVERTER`
**Date frozen:** 21 September 2026
**Scope:** zero-donor byte-level audit of the project binary16 converter against pinned GGML; no dot, engine integration, quality, RAM, or speed claim

## Question

The F16 vec-dot diagnostic stopped at its conversion control.  Determine
exactly where the project IEEE-binary16 implementation differs from pinned
`ggml_fp32_to_fp16`, without evaluating or interpreting dot outputs.

## Immutable population

Audit every finite F32 value, in fixed order, from:

1. `Kcur-0` SHA-256
   `2860d9791b620d19788b8112e5366424be21669153eb1ec3255fd5a6c1b167f3`;
2. `Qcur-0` SHA-256
   `4aca21f044acf71404ef0a7a000ed7b1bfe894efa82c5c1e49f7a5764cc6314b`;
3. mapped padded `kq_soft_max-0`, derived only by the frozen
   `[head,query,slot]`→`[query,head,slot]` transpose from source SHA-256
   `8def8f8d6969dab39987e915084a74cb0c766c9d842db8b272886c248d60092e`.

Also audit a fixed model-free boundary suite: signed zero, normal/subnormal
ties, smallest/largest subnormal, smallest normal, values around max finite
and overflow, infinities, and quiet/signalling NaN bit patterns.  Report NaN
payload policy separately from finite-value equality.

## Outputs and controls

Emit project and pinned uint16 streams, exact SHA-256, total/mismatch counts,
counts by input segment and F32 exponent class, and the first 32 mismatches as
F32 bits/value plus project/pinned F16 bits.  Inputs and records must be
hash-bound.  No model/reference/C donor process may execute; no dot-product
metric may be used.

## Adjudication

- `PASS_F16_CONVERTER`: zero finite mismatches across immutable and boundary
  populations; NaN policy matches the frozen canonical quiet-NaN rule.
- `FAIL_F16_CONVERTER`: controls valid and one or more finite mismatches.
- `VOID_F16_CONVERTER_AUDIT`: identity, population, completeness, finiteness
  classification, output, or boundary-control failure.

One non-VOID audit and stop.  A FAIL authorizes a source-derived converter
repair plus exhaustive/model-free regression before resuming the vec-dot
diagnostic; it does not itself authorize attention integration.

## Apparatus repair A — conversion-stream order witness

The first audit at commit `c3d3d2b` is preserved but voided, adjudication
SHA-256 `b6caf4b26c19f6805f9eabb00a0ae6f878d1cf92df18b40f0998c9f99afc46a0`.
The helper emitted `K, (Q row, softmax row)*, boundary`, while the adjudicator
interpreted `K, all Q, all softmax, boundary`.  Its reported mismatch examples
therefore pair unrelated values (including normal F32 values with padded-zero
conversions) and are not converter evidence.

Repair A emits the frozen contiguous order `K, all Q, all softmax, boundary`
and a parallel uint32 stream containing the exact source F32 bits.  The
adjudicator must reconstruct the same source population and require that
witness stream to be byte-identical before comparing any F16 output.  No input,
converter, boundary population, label, or threshold changes.  Use a distinct
raw directory and zero donor executions.

The canonical result is
[STRAT_01_GIGACHAT31_ENGINE_F16_CONVERTER_AUDIT_RESULT_20260921.md](STRAT_01_GIGACHAT31_ENGINE_F16_CONVERTER_AUDIT_RESULT_20260921.md).
