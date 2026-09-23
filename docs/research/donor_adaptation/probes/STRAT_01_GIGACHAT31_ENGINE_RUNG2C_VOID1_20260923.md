# STRAT-01 GigaChat 3.1 engine Rung 2C — reference serialization VOID 1

**State:** `VOID_ENGINE_RUNG2C`; apparatus defect, no C donor execution and no
numerical verdict.

The first frozen Rung 2C scientific invocation reached and completed both
pinned-reference graph schedules, then refused while composing the cached
`ffn_moe_topk-1` payload. The captured callback tensor was GGML I32 and its
integer bytes were present, but `ggml_type_name()` supplied the lowercase
runtime spelling `i32` while the stitching and payload-writing branches tested
for uppercase `I32`. The event consequently entered the float branch, whose
empty float vector correctly triggered a refusal.

## Immutable raw record

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_20260923/`.

| item | value |
|---|---|
| status | `VOID_ENGINE_RUNG2C` |
| source HEAD | `de2e9886d112818049ef9b69fd2cea4c1cf098ae` |
| run interval | `2026-09-23T11:49:18.873978+02:00` to `2026-09-23T11:50:56.125587+02:00` |
| pinned-reference producer invocations | `1` |
| pinned-reference graph schedules actually completed | `2` (`prefill8`, `cached7p1`) |
| C producer invocations / schedules | `0 / 0` |
| adjudication SHA-256 | `4a88562556544fe4128a6d509a676721be04c5f746dcc14093b39b4478839718` |
| run-manifest SHA-256 | `6b4887032cf9ca7174618eea9e596e1c5aaebccae6dd77cf621aff7e93b7bbde` |

The immutable JSON records `reference_graph_executions: 0`. That value is an
apparatus accounting defect, not the forensic count: the producer constructs
both traces before serializing either trace, and the refusal occurred inside
cached-trace stitching after both `run_arm` calls had returned. Both arm
directories are also present. The repair adds one durable completion marker per
arm and makes the runner count those markers before checking the producer exit
code, so future VOID records retain completed work accurately.

## Repair boundary

The repair is restricted to apparatus semantics:

- canonicalize callback types from the GGML enum to `I32`, `F32`, or `F16`;
- regress the canonical type mapping and cached I32 stitching;
- emit and count unique `prefill8`/`cached7p1` completion markers for both the
  reference and C producers, including failed invocations.

It does not alter weights, token IDs, positions, schedules, numerical kernels,
tolerances, checkpoints, negative controls, or the frozen protocol. A new
apparatus-only directory must pass before a changed-coordinate scientific
rerun. The failed raw directory must not be overwritten or adjudicated offline.

The required apparatus rerun has now passed as
`strat01_gigachat_engine_rung2c_apparatus_repair2_20260923` at source HEAD
`4c5d1af6274d1067076485398d58ce67dfc1ab96`, with zero producer invocations
and zero graph schedules. Its adjudication SHA-256 is
`280ab3f69960fdb8abc81697b5143c9851c0c8c9fc261e4e0412e587c3d5371e` and
its run-manifest SHA-256 is
`97e5ce98efc61a6b40eacd893d29e7f2d03214517e323f28c7aa1345db881ded`.
The one changed-coordinate scientific rerun subsequently completed and is
closed as [Rung 2C FAIL](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_RESULT_20260923.md).
No further Rung 2C producer invocation is authorized.

## Non-claims

This VOID establishes no C/reference parity, model quality, RAM, rate, or
later-layer result. It does not update `SPEED_LEDGER.md`. The C producer was
correctly gated off.
