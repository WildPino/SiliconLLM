# STRAT-01 GigaChat 3.1 attention and V-B production repair result

**Verdict:** `PASS_ENGINE_ATTENTION_VB_REPAIR`
**Date:** 21 September 2026
**Scope:** block-0 C-engine parity through attention residual input for prefill and cached schedules; no Rung 2B, quality, RAM, or speed claim

## Result

The committed C engine passes every available rung-2A tensor and cache gate
after integrating F16 query/probability conversion and Q4_K×Q8_K V-B.

| Control | Result |
|---|---:|
| tensor gates | 24/24 PASS |
| cache gates | 3/3 PASS |
| first remaining failure, both arms | none |
| token-7 prefill/cached continuity | exact (`NRMSE 0`) |
| accepted-artifact C executions | 1 |
| pinned-reference executions | 0 |

The terminal comparisons are identical in both schedules:

- `kqv_out-0`: NRMSE `5.636563e-4`, normalized maximum
  `1.114301e-3`, inside the general `2e-3` / `1e-2` gate;
- `ffn_inp-0`: NRMSE `5.954835e-4`, normalized maximum
  `3.654045e-4`, inside the terminal `1e-3` / `5e-3` gate.

The largest upstream NRMSE remains the already accepted absorbed-query/Qcur
Q4_K path (`3.151504e-4` / `3.073227e-4`).  No new failure appears at
attention, V-B, output projection, residual addition, cache sequencing, or
schedule continuity.

## Provenance

- Integrated engine commit: `4913dc1096261eeabf6c7eb2c61c3f3afe8cff56`.
- Canonical adjudication:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_attention_vb_repair_20260921/adjudication.json`,
  SHA-256 `e99f0b3e05286b02546a9d9c83f1a8745ba486d0c05087db313f23f9d87f0627`.
- Reused immutable reference-run manifest SHA-256:
  `0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f`.
- The accepted Q4_K×Q8_K and F16-dot adjudications are hash-bound in the raw
  record.

## Consequence

Rung 2A is now closed positively at the changed coordinate and must not be
rerun.  The next engine compatibility cell is dense block-0 SwiGLU (Rung 2B),
which must be separately frozen before implementation.  One routed/shared MoE
layer remains Rung 2C after 2B.  No full-model generation, task-quality, RAM,
or token-rate claim follows from this one-block pass.
