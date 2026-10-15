# One-group source attention oracle: rejected

15 October 2026. **ONE_GROUP_ORACLE_LOWER_BOUND_FAIL.** Even target-aware
selection of the best source group and any signed real scalar leaves about
50% branch error. The all-group numerical control passes. Do not train an
ID/mass predictor for this representation or enlarge its active-group budget
after observing these results.

| Split / representation | Pooled branch relative L2 | Sequence L2 p95 |
| --- | ---: | ---: |
| Gate | <=0.01 | <=0.05 |
| FIT / all groups control | 0.006635 | 0.007126 |
| DEV / all groups control | 0.006589 | 0.007460 |
| FIT / best one group, any real mass | 0.518095 | 0.551015 |
| DEV / best one group, any real mass | 0.499741 | 0.537195 |
| FIT / clipped [0,2] diagnostic | 0.518472 | 0.551015 |
| DEV / clipped [0,2] diagnostic | 0.500276 | 0.537520 |

The [frozen protocol](FIRST_ATTENTION_SOURCE_GROUP_ORACLE_PROTOCOL_20261015.md)
partitions the64 source query heads into16 contiguous groups of4. It preserves
the source O columns, uses authentic embedding-derived source V and sealed
F32 source probabilities, and adds O bias once. On every token it computes all
groups and the analytic optimal real scalar for each, then selects the smallest
direct F64 residual. Direct residuals and the quadratic identity agree within
the frozen tolerance. This is an optimistic representational diagnostic, not
a deployable router: it sees the true target on DEV too.

All36 cases were processed,24FIT/12DEV with4,498 variable queries after the
shared62-token prefix. Maximum discrepancy between reconstructed all-group
output and sealed full-F32 source was5.984e-7 relative L2, below3e-4. The
all-group branch passes the same1%/5% gates against original BF16 output.
Thus the one-group failure is not explained by the precision control. No TV
claim is made for the60 absent query heads. The earlier rank protocol and its
completed rejection remain unchanged.

Allowing arbitrary real masses removes the proposed[0,2] restriction from the
primary lower bound. On variable queries none of the chosen masses is negative;
120 exceed2 in magnitude and the maximum is3.705631. The clipped diagnostic's
similarly large error confirms that the failure is not rescued by removing
the mass cap. No seed, optimizer, new labels, GPU call or download was used.

Oracle freeze `f9f5637`, worker exit0,17.687 seconds, peak working set
1,166,295,040 bytes and312,118,883 output bytes, within600s/3GiB/512MiB. All36
F64 case records and their output hashes are retained, together with before/
after source/runtime/trace seals. Input custody explicitly retains the earlier
producer's `FAILED_GPU_RELEASE`; this result does not reclassify that run.

A separately frozen [metric readback](../../../benchmarks/native_expert_scaling/first_attention_source_group_readback.py)
at `5ab9f13` exited0 in2.688 seconds. It verified all declared input/output
hashes, recomputed branch metrics from the36 saved case records and actual
BF16 targets, checked chosen IDs against the16 saved optimal residuals and
selected residuals against actual branch error, and independently derived the
same gates. It does not reconstruct all16 group contributions or optimal masses
from source weights; those calculations and their direct/quadratic checks
remain producer witnesses. Earlier independent full-F32 source graph replay
is recorded in the [rank result](FIRST_ATTENTION_RANK_RESULT_20261015.md).

## Decision and remaining scope

Close this one-of16 contiguous source-group representation, even under an
optimistic per-token teacher-chosen signed mass. A better selector cannot repair
the fixed class's minimum error. This is not a lower bound on arbitrary new
attention banks, transformed source functions, trained models or every possible
native MoE family.

Together with the uniform rank256 rejection and the protected-core cost screen,
this parks the two tested cheap GPT-OSS core transformations on this host.
The [complete cost rationale](CONDITIONAL_HEAD_DECISION_20261015.md) also leaves
the expert/head/frontend path unresolved; a core oracle PASS would not have
established20ms. No wider active-group grid, router fit, unchanged Q/O training
or full120B acquisition follows. A new pipeline proposal must identify a
distinct mechanism and screen its complete active cost and conversion budget.
The overall same-artifact chatbot/useful50 goal remains incomplete and active.

Receipts: [closure](first_attention_source_group_closure_20261015.json),
[result](first_attention_source_group_result_20261015.json),
[binding](first_attention_source_group_binding_20261015.json),
[terminal](first_attention_source_group_terminal_20261015.json),
[readback](first_attention_source_group_readback_20261015.json).
