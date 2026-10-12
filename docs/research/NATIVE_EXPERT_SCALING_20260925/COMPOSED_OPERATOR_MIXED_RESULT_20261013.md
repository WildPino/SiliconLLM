# Mixed native384x96 operator: quality/precision qualified, active cost isolated

13 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / SITE23 LOCAL PASS.**
Separate same-artifact component timing COMPLETE/arithmetic audit PASS.
Goal ACTIVE/INCOMPLETE;no owned run live. No whole chatbot or token-rate admission.

## Fixed correction and scientific result

[V1 native result](COMPOSED_OPERATOR_NATIVE_V1_RESULT_20261013.md) completed its
entire audit,retaining one false1e-4 normalized-input precision control. All
F32 scores bitexact scalar reference did not repair a changed input boundary.
No V1 timing was launched. New fixed path keeps F32 state/input projections,
F64 physical decode/source gate/combined RMS,then BF16 normalized inputs and
original engine AVX F32 dotf output projection/BF16 output. It changes one
controlled-branch output contraction expression in a separate immutable source.
No weights/ranks/refit/DEV alternative or loosened threshold.

All48 actual new native histories,state384x96/source fixed site23 adaptive384
and dual96 maps. Fullsource skip/gate/generator/conv/output widths retained.

| DEV equal-case source output error | Reconstructed denominator | Source denominator oracle |
|---|---:|---:|
| Mean relative RMS | .068812731534 | .070422986388 |
| Worst case | .096949337896 | .097636669780 |
| Centered mean | .073770340312 | .075506985032 |
| Worst domain mean | .083198082373 | .085286949941 |

All original local gates PASS,both modes:mean/centered<=.1,
worstcase/worstdomain<=.2. These are source-operator output errors,not chatbot
accuracy. Source inputs still supplied;sites0/12 and all-layer/autonomous
conversation quality not admitted by this artifact.

Full independent audit:12,987,072 projected values,8,658,048 scan values,
1,769,472 final-state values,138,528,768 normalized values,92,352,512 raw F32
and92,352,512 BF16 output values. Every explicit CPU384x96 state timestep and
final state checked. ALL raw F32 score words bitexact separate scalar eight-FMA
reference. Normalized BF16 input vectors match independent smooth reference
EXACTLY0. Promoted-matrix rawscore relative2.39731e-7/BF16output4.31905e-5,
within unchanged1e-4. Scan5.22288e-7/finalstate1.39064e-6,projection1.16551e-7.
All export coefficient bits/source/maps/metrics/domains/gates/hash controls PASS.

This qualifies differentiated precision at an explicit numerical boundary.
It does not prove F64 is required for every individual decode/gate/norm step;
minimal precision allocation remains untested. No old failed flag is overwritten.

## Same artifact: batch1 supplied-operand timing

Native one thread,affinity0,Ryzen53600X,two full48-history passes:
22,547 supplied positions each (45,094 total),three shortest-FIT history warmups.
All timed normalized-input/rawscore/BF16output/final-state bits match the
qualified artifact. Source field loading,array allocation,checks/hashes outside
native call;state allocation/free and output writes inside. QPC10MHz,
perf_counter resolution1e-7;stage instrumentation/call remainder included.

| Microseconds per supplied position | Pass0 | Pass1 |
|---|---:|---:|
| Input/B/C projections | 12.9124 | 12.8022 |
| State384x96 scan | **10.2458** | **10.1675** |
| F64 physical decode/fullskip | 14.8165 | 14.8569 |
| F64 gate/combined RMS | 53.3320 | 53.3685 |
| BF16 normalize + AVX out projection | **1156.0111** | **1157.0695** |
| Instrumentation/call remainder | .6390 | .7002 |
| Whole supplied-operand call | **1247.9568** | **1248.9648** |

Normalize+output accounts for92.63%/92.64% of measured component time;
scan~.82%. This is not a separately measured out-projection-only time:BF16
normalization and conversion share that stage. State compression now has a
small marginal speed benefit in THIS supplied-operand implementation.

Separate retained source in_proj6704x2048 GEMV on128 arbitrary deterministic
vectors:2540.8688/2531.6742us per vector,two passes after one warmup. Full
13,729,792 source coefficients,54,919,168B F32. Final858,112 output values
checked with independent NumPy F64 contraction,relative5.34792e-8;both repeated
fixed passes have identical scalar checksum. This is a kernel cost test,not
source-history generation/conv/activation fidelity. Generator and operator are
separate workloads,NOT a measured sum for one source layer.

All96 timing rows/case-ID sets/positions/stage sums/remainders independently
recomputed with math.fsum;maximum arithmetic discrepancy0. Native timed-bit
checks are bound recorded checks,not independently replayed by the arithmetic
adjudication. Process CPU snapshots show other System/desktop activity;
clean-system isolation false. Actual DRAM bytes/PMU not measured. Do not infer
ideal cache residency/bandwidth or end-to-end useful tokens/s from these numbers.

## Resource/custody record

Mixed preparation13.6562s/terminal13.6951,OSpeak268,959,744B.
Qualification107.38s (exact raw receipt),OSpeak493,424,640B;
audit131.6703s/terminal131.6843,OSpeak850,673,664B.
New run arrays924,991,968B/family withcompile+parameters1,005,632,978B on E.
Timing88.4473s/terminal88.4729,OSpeak315,813,888B,no new large arrays.
All900/1200s/6GiB/2GiB per-family caps unchanged. Native48 histories;
source/model/GPU/T4/RESERVED calls0,no optimizer/basis fit.

Source/protocol/auditor7ada11a,binding SHA256
`3d3cde52f4490c53dc2bb7c80b1970e5d3a0f5aa7983883d2b9d63b54aecf0b5`,
launch8a048c3,audit launchf96e396. Timing source/protocol6691c6e,
normalized-bit check3a23ab9,launch7b9c1cf.
Preparation94820/native84354/audit50454/timing71390 CLOSED0.
Original engine.c and foreign work preserved;[closure](composed_operator_native_session_closure_20261013.json)
retains V1 numerical failure and all seven completed process receipts.

- [Native C](../../../benchmarks/native_expert_scaling/composed_operator_mixed.c), [runner](../../../benchmarks/native_expert_scaling/composed_operator_mixed.py), [protocol](COMPOSED_OPERATOR_MIXED_PROTOCOL_20261013.md), [binding](composed_operator_mixed_binding_20261013.json), [launch](composed_operator_mixed_launch_20261013.json), [storage](composed_operator_mixed_storage_20261013.json)
- [Result](composed_operator_mixed_result_20261013.json), [terminal](composed_operator_mixed_result_20261013.terminal.json), [audit](../../../benchmarks/native_expert_scaling/composed_operator_mixed_audit.py), [adjudication](composed_operator_mixed_adjudication_20261013.json), [audit terminal](composed_operator_mixed_adjudication_20261013.terminal.json)
- [Timing source](../../../benchmarks/native_expert_scaling/composed_operator_mixed_timing.py), [timing protocol](COMPOSED_OPERATOR_MIXED_TIMING_PROTOCOL_20261013.md), [timing launch](composed_operator_mixed_timing_launch_20261013.json), [timing result](composed_operator_mixed_timing_result_20261013.json), [timing terminal](composed_operator_mixed_timing_result_20261013.terminal.json), [arithmetic adjudication](composed_operator_mixed_timing_arithmetic_adjudication_20261013.json)

## Decision toward the real pipeline

The primary remaining active cost is source input/output functions,not this
small recurrence. Do not launch indefinite refinement of failed384 channel
maps merely to make an already10us scan smaller. Consider preserving more
trained recurrent channels when cheap target input/output geometry permits it,
and measure that cost instead of assuming width alone is prohibitive.
Source state-only dual96 already has much smaller local errors at0/12/23 than
channel384 at0/12;composition with a cheap CAUSAL generator/readout is still
missing. [Selected next](SOURCE_FUNCTION_COST_AND_STATE_WIDTH_NEXT_20261013.md)
separates the wider-state cost control from acquisition of generator inputs.

Goal remains pretrained useful CHATBOT->compact reusable SSM/SWA/selectively
consulted ternary functions->engine.c;sealed donor-relative/autonomous quality,
useful n/CPU IDs+masses/physical expert DRAM,multiple families/scales and>=50
useful batch1 tokens/s on the SAME complete artifact. None of these full claims
is substituted by this successful component.
