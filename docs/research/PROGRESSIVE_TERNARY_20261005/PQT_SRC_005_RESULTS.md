# PQT-SRC-005: qualified synthetic Switch execution on both T4s

6 October 2026. All preregistered synthetic deployment gates pass. This admits
the tested no-cache F32 SDPA MATH and ephemeral-expert execution path on both
T4 devices. No pretrained model function, natural context, optimizer update,
ternary preservation or native CPU timing was performed.

## Execution and independent verification

The private kernel137289254 v1 was dispatched once as
sirwildpino/pqt-src-005-switch-t4-20261006-001. It was observed RUNNING07:31:44 UTC
and COMPLETE07:34:12 UTC. Terminal outputs were fetched once, completing07:43:20
UTC. The status monitor stopped on terminal detection; the coordinator waited
without independent calculations while the GPU job was live. No failed remote
attempt, restart, second dispatch or second retrieval occurred.

All seven actual scientific files match Git169ad6661a3312b431e42554089ba51c5b386dae.
The CPU-qualified attention/lazy-expert source remains unchanged. The returned
bootstrap matches the frozen source after only CRLF-to-LF normalization. The
returned source manifest differs only by its omitted final LF; both raw forms
are retained. Pinned runtime/distribution and official Switch source checks pass.
The consumed6,539,661-byte fixture is identical to SRC004 synthetic003, including
checkpoint, input masks/tokens and CPU eager reference arrays.

Each GPU executes32 cases spanning capacity1/64, batch1/3, encoder7/13,
decoder5/9 and full/right-padded masks:64 cases,192 complete model forwards,
2,304 instrumented SDPA calls. Maximum per-array relative RMS against CPU eager
is8.196127306114756e-7, below1e-5. Route masks and final argmax agree exactly.
Resident, ephemeral and repeated ephemeral numerical payloads are byte-identical.
All192 numerical trace archives are preserved, including repeats.

The independent server stdlib audit passes. A subsequent local stdlib
reaggregation agrees for all64 cases and3,144 CPU-reference array comparisons;
9,624 trace ZIP entries are retained. Neither audit repeats a model forward.
Both devices pass all23 malformed/unsupported-call controls, including partial
pair load and FFN failure cleanup. Core bytes and four tied Parameters remain
intact; at most one1024-element expert pair is resident, and all experts return
to meta between calls and after failures.

On each device the capacity1 independent F64 router/FFN equation reproduces
the preserved witness; GPU relative RMS is8.062178924162598e-8 and eight dropped
tokens have exactly zero output. Each GPU/capacity1 sequence records917 source
reads/1,777,024 raw bytes; capacity64 records925/1,793,408. These are checked
synthetic tensor reads, not complete original-checkpoint I/O measurements.

## Costs and complete retention

| Scope | Elapsed seconds | Process peak RSS bytes |
| --- | ---: | ---: |
| Install complete child | 18.419554 | Not separately instrumented |
| Qualification complete child, including imports/report/exit | 66.462771 | 2,481,627,136 |
| Qualification nested timer before manifest | 18.136405 | Same cumulative process peak |
| Qualification nested timer before report | 18.206881 | Same cumulative process peak |
| Audit complete child | 1.418897 | 45,772,800 |
| Audit nested timer before report | 1.325479 | Same cumulative process peak |
| Complete server controller through final cost update | 86.316574 | Not separately instrumented |
| Local source/raw retention and numerical reaggregation before report | 1.844000 | 54,034,432 |

The complete qualification child costs66.46s; its18.21s nested timer excludes
imports and process exit and must not substitute for the complete cost. Each
T4 reports33,856,512 peak allocated bytes and35,651,584 peak reserved bytes,
both below4 GiB. Physical device capacity15,636,037,632 bytes is not peak usage.
GPU figures are allocator statistics, not physical total VRAM measurements.
Process RSS remains below4 GiB. Generated worker bytes before report15,664,524;
the whole411-file retrieved namespace, including reports/logs/source, totals
17,386,574 bytes, below64 MiB. Queue/status/retrieval/authoring/Git durations
are outside server timers; total retrieval duration was not instrumented.

[Retention](PQT_SRC_005_REMOTE_RETENTION_001.json) preserves every fetched byte
in one ZIP_STORED archive17,455,244 bytes, SHA256
e43ebfc4ac7ca12b84646d62fd5b4f4e85e40189960774de90046b0b989f0607.
Intentionally nonfinite/invalid negative-control fixtures are kept as raw
evidence, independently of the finite positive numerical comparisons.
[Git proof](PQT_SRC_005_REMOTE_GIT_VERIFICATION.json) verifies eight files
totaling17,611,549 bytes at6ed2f74ef4337845645ba0d26d0d0b55b5c54b24 and every
one of the411 retained raw entries against its original retrieval.
Preparation separately retains211 raw control entries with16 parser/transport
and nine auditor malformed stops. All source and input identities remain bound
by the [protocol](PQT_SRC_005_PROTOCOL.md) and [binding](PQT_SRC_005_BINDING.json).

## Decision and resumption

Synthetic deployed execution is qualified. It is not evidence that original
float contexts were captured, that any ternary method preserves pretrained
capabilities, or that this path provides useful native performance. Ten previous
audited quality screens still have no ternary promotion. The full goal is active.

Do not rerun the completed fixture or refetch terminal outputs. No live remote
handle remains. Next qualify the original-checkpoint source provider and freeze
fresh masked-span inputs, splits, tokenizer, routing/coverage/discard rules and
complete-source I/O/scratch/memory/time bounds before original function capture.
See [next direction](PQT_SRC_006_DIRECTION.md). Native CPU timing continues to
await the explicit owner availability window; silence does not authorize it.
