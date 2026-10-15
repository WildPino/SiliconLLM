# Actual first-attention rank-256 recipe: rejected

15 October 2026. **Independent numerical rejection confirmed; producer cleanup
failure retained.** The prescribed actual-weight Q/O rank-256 recipe fails all
four local gates on both FIT and DEV. Its full-rank F32 control passes. Stop
this recipe; no automatic rank grid, extra updates or full-model acquisition.

This is the first source attention block of pinned GPT-OSS-120B, not a complete
model or a CPU speed test. The [frozen protocol](FIRST_ATTENTION_RANK_PROTOCOL_20261015.md)
uses genuine embedding-derived inputs, 24 FIT cases from eight sources and 12
DEV cases from four disjoint sources. Each prefix has at most 256 tokens; the
common 62-token prefix is excluded from the gates. K/V, norm, biases, sinks,
YaRN and sliding-window semantics stay source-fixed. Only four Q/O factors,
3,571,712 F32 coefficients, are fitted by the predeclared initializer and 512
fixed Adam updates. DEV does not choose the rank, checkpoint or schedule.

| Path / split | Branch relative L2 | Sequence L2 p95 | Mean sink-inclusive TV | TV p95 |
| --- | ---: | ---: | ---: | ---: |
| Gate | <=0.01 | <=0.05 | <=0.01 | <=0.05 |
| Full F32 / FIT | 0.006635 | 0.007126 | 0.005121 | 0.015264 |
| Full F32 / DEV | 0.006589 | 0.007460 | 0.005102 | 0.015665 |
| Rank256 / FIT | 0.211763 | 0.302660 | 0.081168 | 0.245742 |
| Rank256 / DEV | 0.292680 | 0.356005 | 0.122956 | 0.408466 |

These are independent CPU F64 metric reductions over the saved traces.
Bias-removed branch error is also large (0.215199 FIT / 0.297356 DEV), and
including the common prefix does not rescue the candidate. The gap already
occurs on FIT, so this result does not support proceeding directly to a larger
donor, internal-layer calibration or physical factor quantization. It rejects
this finite initialization/training recipe, not every possible rank-256 optimum,
all attention compression or the pretrained family on every execution scheme.

## Complete numerical evidence with a failed cleanup check

The numerical worker completed all 512 updates and all 108 source/F32/candidate
traces. Its recorded pre-posthash wall was 78.125 seconds: source import 0.281,
teacher/precision control 11.422, initialization 5.219, optimization 52.078 and
export/evaluation 5.922 seconds. Peak GPU allocated/reserved bytes were
1,710,806,528 / 1,927,282,688. Frozen source hashes, source identity/version
receipts and post-binding checks matched.

After garbage collection and `empty_cache`, 67,108,864 GPU bytes were still
allocated and reserved. The producer therefore exited 1 with
`FAILED_GPU_RELEASE`; that receipt is unchanged. A workspace is a possible
cause, not an established diagnosis. The worker and watchdog subsequently
exited. No cleanup retry or numerical GPU rerun was used to turn it into PASS.

The separately frozen CPU auditor supports only a numerical **rejection** from
this completed failed-cleanup run. It requires complete sealed artifacts, dead
owned PID/create-time identity, unchanged source weights, passing precision
control and failing candidate gates. It independently checked all file/input
seals, all legal-key-plus-sink probability rows, all 512 ledger entries and final
factor/optimizer/gradient records, and reconstructed the source parameter hash.

It also rebuilt both F32 source and final candidate on all 36 cases directly
from the acquired source rows/factors: 72 graph comparisons. Maximum branch
relative discrepancy was 7.745e-7 and maximum probability absolute discrepancy
5.574e-6, below the previously fixed 3e-4 / 5e-4 tolerances. Audit exit 0,
17.875 seconds, peak working set 931,737,600 bytes, six CPU threads and empty
CUDA visibility. Its status is `PASS_NUMERICAL_REJECTION_PRODUCER_FAILED`;
`admitted=false`, and producer failure remains explicit.

The audit does not reproduce intermediate Adam states or establish original
BF16 GPU probability execution independently. Official pre-O/pre-residual
parity and source identity/version checks remain producer execution witnesses.
No whole-model/chat quality, physical Q4 factor kernel or useful token rate is
admitted. The earlier [protected-core cost rejection](PROTECTED_CORE_120B_CPU_RESULT_20261015.md)
also remains in force. A next path needs a structurally different, quantitatively
credible intervention rather than another unchanged training or timing grid.

Receipts: [closure](first_attention_rank_closure_20261015.json),
[producer result](first_attention_rank_result_20261015.json),
[terminal observation](first_attention_rank_terminal_20261015.json),
[CPU audit and all replay differences](first_attention_cpu_audit_20261015.json),
[binding](first_attention_rank_binding_20261015.json),
[resource probe](first_attention_rank_probe_20261015.json).
