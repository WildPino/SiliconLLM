# Preserve all3072 trained channels:state96 native qualification and actual cost

13 October2026. **COMPLETE / INDEPENDENT_STORED_PASS.** Own-norm local quality
PASS at ALL3 tested sites;fixed-source-denominator controls FAIL at0/12.
Same-artifact component timing COMPLETE/arithmetic audit PASS. Goal active/incomplete.

## Fixed transformation and scope

Reuse exact FIT dual96 source state maps,no channel bottleneck or new fit.
All3072 trained channels/48 head decays,source x/B/C/delta/gate/fullskip remain.
Actual C float32 state/B/C projections,scalar expf cached per head,F64 gate/full
postgate RMS->BF16->original AVX F32 dotf output/BF16. Original engine.c unchanged.
All48 histories at sites0/12/23,144 NEW native scans. State3072x96,
1,179,648 F32bytes (1.125MiB);source3072x256 was3MiB. Compared with384x96 this is
8x more state coordinates,not8x more total measured operator time.

## Source-relative held-out output errors

Equal case weight,24 DEV histories/site,12 domains;errors are relative RMS of
source mamba output,not chatbot accuracy. Reconstructed own denominator:

| Site | Mean | Worst case | Centered mean | Worst domain mean |
|---|---:|---:|---:|---:|
| 0 | .076342029 | .093382972 | .080452575 | .085993724 |
| 12 | .087177247 | .107002135 | .089714608 | .102571024 |
| 23 | .024903750 | .064048301 | .028022312 | .038258774 |

ALL four original local gates PASS per site:mean/centered<=.1,
worstcase/worstdomain<=.2. Fixed source-denominator diagnostic:

| Site | Mean | Worst case | Centered mean | Worst domain mean | Failed original flags |
|---|---:|---:|---:|---:|---|
| 0 | .102165523 | .122689762 | .107949891 | .111535142 | mean,centered |
| 12 | .157694560 | .203979742 | .161060292 | .194436941 | mean,worst,centered |
| 23 | .026243898 | .064235739 | .029521395 | .039213718 | none |

Own normalization compensates part of the discrepancy;do not say both modes
pass everywhere or that state-only preserves the raw source amplitude. Before
rounding,own versus source-denominator output for the SAME reconstructed gated
vector differs by scalar r_source/r_reconstructed. This identity does not show
that all errors are radial or that future causal composition preserves quality.
It explains why the diagnostic denominator is not necessarily a better predictor.

At12 the prior fixed whitened384 candidate mean error was.775702;full trained
channels/state96 now.087177. Different approximations/control paths,not a paired
all-layer model trial. Site23 previous384x96 mean.068813 versusfull.024904.
This changes the useful geometry choice;neither comparison proves a width floor.

## Full independent qualification

All12,987,072 B/C projection values,207,793,152 scan values,
42,467,328 complete final-state values,415,586,304 normalized BF16 values,
277,057,536 raw F32 scores and277,057,536 BF16 outputs checked. EVERY direct
CPU3072x96 state timestep computed. ALL raw F32 scores bitexact separate scalar
eight-FMA reference. Normalized BF16 vectors independently reproduced EXACTLY0.
Maximum projection1.430e-7,scan1.321e-6/finalstate1.497e-6,
promoted rawscore2.445e-7/BF16output6.008e-5 within unchanged1e-4.
Source/export bits,metrics/domains/labels/old scalar delta flags/gates and
before-after declared-input/output hashes qualified. Older state96 comparisons
are scalar observations,not unavailable old-vector precision equivalence.

## Same operator: actual1-core batch1 costs

Two complete passes of48 histories/site,288 rows/135,282 supplied positions,
three shortest-FIT warmups/site. Native state allocation/free/final-state copy
and output writes charged;field loading,allocation/checks/hashes outside call.
Every timed normalized/rawscore/output/final-state bit matches qualified data.

| Site,pass1 | B/C projection us | Scan us | Gate/RMS us | Normalize+output us | Total supplied-operand us |
|---|---:|---:|---:|---:|---:|
| 0 | 6.8525 | 58.5504 | 54.7662 | 1176.4309 | 1297.5206 |
| 12 | 6.5638 | 59.1115 | 54.6946 | 1180.0532 | 1301.3785 |
| 23 | 7.0414 | 58.8756 | 54.9247 | 1175.1761 | 1296.9593 |

Pass0 totals1298.8484/1304.8201/1300.0724us. Call/instrumentation remainder
~.92-.97us. Compared to earlier site23 384x96~1248.9648us/full3072x96~1296.9593us
there is~3.84% observed total difference despite8x state size and better local
source quality. Different runs/workloads/normalization vectors,not a simultaneous
matched causal speedup attribution. Scan now~4.54% of operator time;source
normalize+output~90.6%. Cached48 scalar head decays are part of this algorithm;
do not apply its time to an arbitrary per-state-pole/unfrozen-A kernel.

Separate source in_proj GEMV on128 declared arbitrary vectors,two passes/site:
pass1 averages2515.5656/2509.4234/2526.7875us at0/12/23. Full source F32 matrix
54,919,168B/site. Final deterministic858,112 outputs/site independently checked
against NumPy F64,relative<=5.42e-8;repeat checksums identical. These kernel and
operator times are separate workloads,NOT measured joined source-layer time.
QPC10MHz/perf_counter1e-7,process CPU snapshots retained,clean-system isolation
and actual DRAM bytes NOT admitted. No end-to-end useful token/s/n/model claim.

All288 timing case sets/positions/stage sums/remainders independently recomputed
with math.fsum,discrepancy0. Timed bit equality is a recorded bound check,not
independently replayed by that arithmetic audit. Source generator/conv/attention/
FFN/LMhead and compact hidden causal generation remain outside component time.

## Costs,custody and limitations

Preparation59.5123s/terminal59.5557,OS359,628,800B;
qualification322.89s/exactterminal323.1776,OS443,813,888B;
audit467.9654s/terminal468.0049,OS770,920,448B;
timing exactterminal228.9200s,OS278,949,888B. Immutable E run3,546,915,264B,
family withthree parameter packets/compile3,787,944,462B,under6GiB cap.
Source/protocol17ab638,bindingSHA
`288b9ff3f565ebf8f11a169f54dc0f604d2857c2717e299029936fbaddd144e2`,
launcheaec4b5,auditorb072803/launch4570a6a,timing58a773c/launcha020316.
Prepare23038/qualification94414/audit15453/timing57125 CLOSED0.

[Runtime alias witness](numerical_runtime_alias_witness_20261013.json) records
an enumeration gap:linked package directories resolve outside the runtime
parent-prefix filter,so NumPy/psutil files were not directly in this before-after
loop. Native/scalar DLLs/compiler/source/data were directly bound;current7
numerical module byte hashes match the older explicit12 October witness.
This supplement is NOT a retroactive before-run observation/proof against
unobserved transient changes. New gate preflight explicitly binds resolved
NumPy ABI/BLAS/psutil modules. Numeric/vector/bit evidence remains as observed;
do not overstate entire process-runtime custody from the earlier filter.

- [C](../../../benchmarks/native_expert_scaling/full_channel_native.c), [runner](../../../benchmarks/native_expert_scaling/full_channel_native.py), [protocol](FULL_CHANNEL_NATIVE_PROTOCOL_20261013.md), [binding](full_channel_native_binding_20261013.json), [launch](full_channel_native_launch_20261013.json), [storage](full_channel_native_storage_20261013.json)
- [Result](full_channel_native_result_20261013.json), [terminal](full_channel_native_result_20261013.terminal.json), [audit](../../../benchmarks/native_expert_scaling/full_channel_native_audit.py), [adjudication](full_channel_native_adjudication_20261013.json), [audit terminal](full_channel_native_adjudication_20261013.terminal.json)
- [Timing](full_channel_native_timing_result_20261013.json), [timing terminal](full_channel_native_timing_result_20261013.terminal.json), [timing protocol](FULL_CHANNEL_NATIVE_TIMING_PROTOCOL_20261013.md), [arithmetic adjudication](full_channel_native_timing_arithmetic_adjudication_20261013.json), [closure](full_channel_and_gate_operator_session_closure_20261013.json)

## Pipeline decision

Keep full trained channels/state96 as a candidate shared recurrence for this
family;do not spend long fits shrinking a state whose marginal cost is small.
Source D2048 input/output functions remain expensive and supplied. Next actual
transfer work needs cheap source-informed generator/readout and causal hidden
encoding,not a generic retained donor runtime. [Gate information preflight](GATE_INPUT_OPERATOR_RESULT_20261013.md)
supplies necessary geometry for an OFFLINE data-recovery test,not hidden states.
Source families/scales,useful distinct bank capacity/CPU IDs+masses/actual expert
DRAM/conversionO(n)/donor-relative+own-history quality/same useful artifact50
remain unfulfilled.
