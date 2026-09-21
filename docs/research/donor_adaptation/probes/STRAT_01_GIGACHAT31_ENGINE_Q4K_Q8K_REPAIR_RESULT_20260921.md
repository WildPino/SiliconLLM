# STRAT-01 GigaChat 3.1 engine Q4_K×Q8_K repair result

**Verdict:** `PASS_ENGINE_Q4K_Q8K_REPAIR`
**Date:** 21 September 2026
**Scope:** standalone C operator fidelity for the immutable block-0 Q and KV projections only; no full-attention, quality, RAM, or speed claim

## Result

The project C engine now quantizes each F32 activation row to the exact Q8_K
intermediate and evaluates stored Q4_K rows with Q4_K×Q8_K dot semantics.  The
implementation is self-contained in `benchmarks/phase60/strat01_q4k_q8k.h`;
the final engine neither links nor depends on GGML.

Against the immutable pinned-reference targets:

| Projection | NRMSE | Normalized maximum | Frozen gate | Result |
|---|---:|---:|---:|---|
| `blk.0.attn_q.weight` | `6.4478969411e-08` | `9.3328801287e-08` | `2e-6` / `1e-5` | PASS |
| `blk.0.attn_kv_a_mqa.weight` | `7.3224546015e-08` | `4.8560459180e-08` | `2e-6` / `1e-5` | PASS |

The scalar result need not be bit-identical to the pinned active AVX2 kernel;
both residuals are about two orders of magnitude below the tighter frozen
gate.  This closes the implementation gap diagnosed by
`ATTRIBUTED_Q8K_ACTIVATION_QUANTIZATION`.

## Controls and provenance

- Accepted commit: `7cb1edfef07836ae0896c5066115a30007d3500b`.
- Exact GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Frozen input SHA-256:
  `c8c7bd47772b1f153f28183892795b9bc91be978f5473bb322f869b2c10c1efd`.
- Built engine SHA-256:
  `9625490ee7bbedd018d854ba6de834fcbd1696c09fef0105adb18b574e7b84bc`.
- Twenty model-free and legacy tests pass.  Across 21 synthetic oracle cells,
  Q8_K bytes are exact and dot NRMSE/normalized maximum are both zero.
- The planted wrong packed-scale interpretation, token/row transpose, and
  one-byte Q8_K scale mutation fire with absolute dot deltas `35.8073578`,
  `217.592407`, and `84.09021`, respectively.
- Rung-1 and Rung-2A model-free tests pass; the historical kernel self-test
  remains 73,024/73,024 with zero worst integer error.
- `donor_executions=0`; the accepted projection command consumed only the
  exact GGUF spans and previously captured input.

Accepted raw adjudication:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q4k_q8k_repair1_20260921/adjudication.json`,
SHA-256 `53931d8e141c9e6faff2fce04d1665ff71fd8976e6a3c763721ddcd5435fdc39`.

## Preserved VOID

The first launch at commit `3d1861a` is preserved in
`strat01_gigachat_engine_q4k_q8k_repair_20260921/` as a pre-execution
`VOID_ENGINE_Q4K_Q8K_REPAIR`.  It recorded no commands and performed no model
read: the runner applied the frozen successful-manifest hash to the older
Rung-2A directory.  Apparatus repair A changed only the source directory to
the already frozen `rung2a_repair2` payload set.  The VOID adjudication SHA-256
is `9e0e9b08bf8235b6eb72180db461b9fd75d4b9eda5d427c7a911afe072aaea90`.

## Consequence and no-duplication boundary

Do not repeat the projection repair cell on this artifact and captured input.
The old `FAIL_ENGINE_RUNG2A` remains valid and is not rewritten.  This pass
authorizes one separately frozen changed-coordinate block-0 attention
confirmation using the repaired operator and the already captured pinned
reference.  It does not authorize Rung 2B, prove downstream Q5_K/Q8_K or
special-layout V-B semantics, establish model quality, or move any speed
number.  `SPEED_LEDGER.md` is intentionally unchanged.
