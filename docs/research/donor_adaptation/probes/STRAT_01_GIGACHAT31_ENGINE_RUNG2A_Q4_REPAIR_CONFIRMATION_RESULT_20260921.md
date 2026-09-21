# STRAT-01 changed-coordinate block-0 attention confirmation result

**Verdict:** `PASS_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION`
**Date:** 21 September 2026
**Scope:** integration of the accepted Q4_K×Q8_K operator at the old first-failure boundary and localization of the next block-0 mismatch

## Result

The repaired C path crosses the old Rung-2A Q/KV boundary in both the
single-prefill and 7+1 schedules.  All six frozen primary cells pass the tight
NRMSE `2e-6` / normalized-maximum `1e-5` gate:

| Tensor | NRMSE, both arms | Normalized maximum, both arms | Result |
|---|---:|---:|---|
| `attn_norm-0` | `7.9650574584e-7` | `1.0964520347e-6` | PASS |
| `q-0` | `7.9035004140e-7` | `8.3995921158e-7` | PASS |
| `kv_cmpr_pe-0` | `7.9013215722e-7` | `9.7120918360e-7` | PASS |

The repaired C implementation's prefill/7+1 terminal continuity is exact
(NRMSE and normalized maximum both zero).  The old captured C outputs remain
valid planted negatives: Q NRMSE `0.0029148289` and KV `0.0047220343` fail the
old `0.002` gate in both arms.

## Downstream localization

Using every old Rung-2A comparison and threshold, the repaired path now passes
through `Vcur-0` and all three compact-cache checkpoints.  The only remaining
tensor failures in either arm are:

| Tensor | NRMSE | Normalized maximum | Old gate |
|---|---:|---:|---|
| `kqv_out-0` | `0.01199015083` | `0.007000378713` | FAIL (`0.002` / `0.01`) |
| `ffn_inp-0` | `0.005749956285` | `0.003861914276` | FAIL (`0.001` / `0.005`) |

The first remaining failure is therefore `kqv_out-0` in both schedules.  This
is the composite boundary after causal attention and the special-layout
Q4_K V-B expansion.  The evidence does **not** yet distinguish attention
arithmetic from the V-B operator; it would be premature to patch V-B without
a separate diagnostic.  `ffn_inp-0` is downstream and cannot be interpreted
independently yet.

## Controls and provenance

- Accepted commit: `76fbe41af944983ee027c00c6a1674e387e54fcb`.
- Engine binary SHA-256:
  `12f4ebec489024489910aa37f7b8442acd03c148f81a56ed53ca3256243a5827`.
- One C donor/block-0 execution; zero reference reruns.  The immutable pinned
  reference and old C negative capture were both fully revalidated.
- The 21-cell Q8_K/oracle population remains byte-exact/zero-error; all repair,
  negative and legacy tests pass; the 73,024-check kernel self-test passes.
- Exact C and reference manifests, complete tensor/cache payloads, metrics,
  hashes and command logs are preserved in the raw directory.

Raw adjudication:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_q4_repair_confirmation_20260921/adjudication.json`,
SHA-256 `94becead7489f59e725b4d504552de3de98b8059c8c096458c1627f733991475`.

## Consequence and no-duplication boundary

Do not repeat the old Rung-2A cell, the isolated Q4_K×Q8_K repair, or this
integration confirmation.  The historical `FAIL_ENGINE_RUNG2A` remains valid;
this result records a changed-coordinate pass and a new first-failure
boundary.  The next permissible step is a frozen zero-donor diagnostic that
separates causal-attention reconstruction from special-layout V-B Q4_K
activation semantics using existing traces and exact weight spans.  No 2B,
quality, RAM, or speed claim follows, and `SPEED_LEDGER.md` remains unchanged.
