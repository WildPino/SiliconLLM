# METH-302 result: full active matvec and MLA dominate the unchanged LUT

**Decision: next mechanism must reduce active matvec/lookup complexity;
table-only tuning is insufficient at unchanged measured other costs.**
Projection matvecs account for80.139% of the operator body. The original301
cost failure remains; this diagnostic neither regrades it nor tests quality.

## Frozen instrumentation and exact controls

[Protocol](METH_302_VECTOR4_COST_PROFILE_PROTOCOL_20261003.md),
[controller](../../../benchmarks/native_expert_scaling/meth302_vector4_profile.py)
and [C instrumentation](../../../benchmarks/native_expert_scaling/meth302_vector4_profile_cpu.c)
were committed at **`5953709` BEFORE observation**. The new actual phase60
profile selector includes immutable301 C with its main renamed. Every
router, input generator, table, gather, scale, layout and numeric self-test
is reused without modification. Only timestamps and organ accounting are
added. Code execution order and fixtures remain the same.

All30 full-output/route hashes are EXACT301. Scalar16,968 rows/FP64 18,568
rows/negative controls pass unchanged, same2,951,464,192B allocations.
All309 organ classifications match130MLA/75routed/75shared/3dense/1head/
25routers. The largest unattributed whole-minus-components duration is
**68.500006microseconds**, below the prospective1ms closure limit.
Three whole-body medians89.41995/88.91000/86.90255ms, max/min1.02896808,
pass repeatability. Instrumented timings are a distinct run, not replacement
medians for301. Local command exits0, total4.594s, child3.319s, peak RSS
2,958,184,448B. No larger n, quality observations, inference, fit or GPU/T4.

Raw [result](meth302_vector4_profile_result.json),35,301B SHA
`c6496836fdb2f01e89f74aaf4667f6df6a9ff6917f34b645118e9e98be69704b`.
Profile C SHA `fdbb9d0fa7052aa4c516fd644fc78628de79cc0d93953892ef3bb9e2394ac9a1`;
controller SHA `9e29c14a65423425161976dbad74ad603f1b90c644ec37591e091a9c239d9fd1`;
binary SHA `92294d4ce557e73ff0ce84e68b30b00097ce0c6e2850770fb7ce12e8c7a77c5d`.
Exact flags, engine/compiler/catalogue and inherited301 hashes are in raw.
Independent Python recount reproduces all24 measured organ/aggregate medians,
all30 prior hashes,80.139% fraction and all24 table+router>=14ms observations.

## Actual component attribution

The columns are separate medians over24 measured inputs. Combined medians
are computed from paired sums for each input, not addition of the two medians.

| Organ | Table median ms | Matvec/scale median ms | Paired combined median ms |
| --- | ---: | ---: | ---: |
| MLA | 9.895 | 31.487 | **41.439** |
| Routed | 3.648 | 22.854 | **26.541** |
| Shared | 2.942 | 8.841 | 11.702 |
| Dense first FFN | .169 | 1.721 | 1.934 |
| Full head | .036 | 5.704 | 5.737 |
| Flat F32 router, n64 | 0 | .915 | .915 |

Per-observation aggregates have median **16.650ms tables**, **.915ms router**,
**71.137ms encoded matvecs**, **88.910ms whole body**. Projected lookup work
has median80.139% share; the prospectively fixed70% priority threshold passes.
All24 table+router observations already exceed14ms. With other observed costs
held fixed, an impossible free-table diagnostic still has median72.037ms.
Thus this profile needs BOTH materially cheaper lookup/active work and cheaper
construction. These deductions assume unchanged other costs; removing tables
can change cache behavior and requires another measured implementation.

The synthetic flat n64 router is comparatively small here. This is not a
claim for larger n;300/301's explicit n640 router growth remains relevant.
The full-source MLA is the largest component, so an experts-only rewrite
cannot establish a complete fast pretrained transfer.

## Method consequence and next distinct variable

Do not train unchanged pair/vector4 full-source coefficient LUTs; no n640
allocation or layout tweak is promoted by this diagnostic. Reusing full
source feature inventory requires too much work in this measured backend.
Rank192 and direct channel omission also fail their real input screens.

The [next compact-core proposal](GIGACHAT_JOINT_COMPACT_CORE_PROPOSAL_20261003.md)
changes the FUNCTION learned by the active core: source-derived narrower
attention plus smaller nonlinear routed/shared FFNs, distilled against the
complete teacher organs and then composed whole quality. It is explicitly
UNMEASURED, with no student weights or fidelity/rate claim. First freeze a
complete native cost screen for its row-I8 geometry and full-head treatment;
only then consider collecting additional whole-function teacher states.
This avoids another table-only or expert-only route with an unchanged costly
MLA. All source quality/useful-n/same-artifact accepted50 requirements remain.
