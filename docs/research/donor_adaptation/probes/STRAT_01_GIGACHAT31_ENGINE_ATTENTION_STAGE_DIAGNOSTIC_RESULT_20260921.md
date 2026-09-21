# STRAT-01 GigaChat 3.1 attention-stage diagnostic result

**Verdict:** `MIXED_ATTENTION_STAGE_RESIDUAL`
**Date:** 21 September 2026
**Scope:** pinned block-0 QK/softmax/value attribution; one preserved reference trace, offline adjudication, no production-engine, quality, RAM, or speed claim

## Result

The softmax equation is not the remaining defect.  Applying the project
scaled causal softmax to captured `kq-0` passes captured `kq_soft_max-0` by a
wide margin.  Both dot-product stages surrounding it fail independently:

| Stage | NRMSE | Normalized maximum | Frozen gate | Result |
|---|---:|---:|---:|---|
| scalar QK dot vs captured `kq` | `2.125561370e-4` | `2.419999731e-4` | `2e-6` / `1e-5` | FAIL |
| project softmax on captured `kq` | `7.755602019e-8` | `1.788139343e-7` | `2e-6` / `1e-5` | PASS |
| scalar captured-softmax×V vs true `kqv` | `1.527725445e-4` | `2.678968664e-4` | `2e-6` / `1e-5` | FAIL |
| fully project-composed latent | `2.159923284e-4` | `5.265231413e-4` | `2e-6` / `1e-5` | FAIL |

This is a valid mixed result, not an ambiguous graph boundary: QK accumulation
and value reduction fail while the intervening softmax passes.  Pinned source
inspection gives one shared, unmeasured explanation.  For an F16 matrix,
GGML's CPU type traits set `vec_dot_type=F16`, convert the other operand from
F32 to F16, and call `ggml_vec_dot_f16`.  The current project reconstruction
instead multiplies F16 cache values by F32 queries/probabilities with scalar
accumulation.  That shared explanation requires a separately frozen offline
test; it is not promoted by this result alone.

## Controls and provenance

- Computational commit: `982701fa6b6020f5a0231b4773de51f3d008727f`.
- Exact GGUF SHA-256:
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Captured `kq-0` and `kq_soft_max-0` shapes are `[256,8,32]`; padded softmax
  slots 8–255 are exactly zero.
- Immutable callback SHA-256: `kq`
  `9c5d5e73b21b5b92482c0ba396a49a0dba1be3d5bb99be332911f56787e2f74c`;
  softmax `8def8f8d6969dab39987e915084a74cb0c766c9d842db8b272886c248d60092e`;
  true `kqv` `3922f34159f499dd26788acb7d9600d72fded004425392946bc09b8098fd88df`.
- Q mutation, legal-past probability swap and unrounded-F32-cache negatives
  all fire.  Identity, source, payload, shape, op, finiteness and self-test
  controls pass.
- Exactly one reference prefill was used, in the preserved VOID producer run;
  the canonical adjudication performed zero additional donor executions.

Canonical offline adjudication:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_attention_stage_offline_adjudication_20260921/adjudication.json`,
SHA-256 `9292a15db752cfe63cead091d7d8a4f7f3c4f7295c13c68bcb5981ebd87541d8`.

## Preserved VOID and apparatus repair

The first runner at commit `6f9b747` completed the reference producer but
stopped before reading numerical values because it preregistered the KQ slot
extent as 8 rather than the resolved context extent 256.  Its raw directory is
preserved as `strat01_gigachat_engine_attention_stage_diagnostic_20260921/`
with `VOID_ATTENTION_STAGE_DIAGNOSTIC`, adjudication SHA-256
`c54c7d47e1c60f76f70e08e21d2443caf93e49eb1b2cab4dcb1f931b276c089d`.
Apparatus repair A only bound the padded shape, selected occupied slots 0–7,
and required padded softmax zeros; it reused the captured trace offline.

## Consequence and no-duplication boundary

Do not repeat this capture or tune the softmax.  The next coordinate is a
zero-donor, source-derived F16-vector-dot diagnostic on the immutable KQ,
softmax and true-latent payloads.  It must test both F32→F16 operand conversion
and pinned `ggml_vec_dot_f16` reduction semantics independently for QK and
softmax×V.  No production attention repair, Rung 2B, quality or speed step is
authorized yet; `SPEED_LEDGER.md` is unchanged.
