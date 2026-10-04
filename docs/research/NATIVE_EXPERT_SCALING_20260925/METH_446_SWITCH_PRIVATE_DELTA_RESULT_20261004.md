# M446: exact native neuron permutation; Frobenius private factors inadequate

Source/math/protocol frozen63a5f04 before first import/forward/matching/eigensolve;
resumption docs d3c85e1. Terminal exit0, zero optimizer updates.
[Protocol](METH_446_SWITCH_PRIVATE_DELTA_PROTOCOL_20261004.md),
[raw](meth446_switch_private_delta_result.json) SHA256
b6c525905cb2df1b3adb88c0131a09d34ac384328ff4208e4cd95609ed9773ee.
Tools: [math](../../../benchmarks/native_expert_scaling/meth446_switch_private_delta_math.py),
[controller](../../../benchmarks/native_expert_scaling/meth446_switch_private_delta_pilot.py).

## What is now qualified

ALL9 apparatus gates PASS. ALL1344original source128 captured RMS/WI/ReLU/WO
states byte-exact; four fixed functions0/1/64/127 evaluated at every input.
Each of the three matched neuron permutations preserves the entire native
function byte-exact at ALL1344 inputs:3*1344=4032 matched function
comparisons, with both raw-neuron reorder and complete down output checked.
This is a coupled WI-row/WO-column permutation with unchanged original scales,
not an assumed positive-rescaling invariance. Source bank11/core unchanged.

Unit-WI assignment changes3070/3071/3070 of3072 hidden positions for targets
1/64/127 and lowers its specified cost. It optimizes that cosine-row matching
metric only, not the function metric or joint WI/WO delta spectrum. Correct
native permutation does not guarantee compressible differences.

Dequantized F64 full-function shadow is qualified before utility: max native
relative discrepancy.000732397 across all4*1344positions, below fixed1e-3.
Candidate factors are storedF32 but evaluated in this F64 shadow. No native
factor operator, head, posterior, full rollout or original task-quality test.

## All45 fixed candidates fail feasibility

Ranks16/32/64/96/128, two private-delta gauges and a full-matrix factor control.
Each candidate passes only nominal bank-storage saving; matrix95%energy,
function median<=.05 and p95<=.10 all FAIL. No common recipe qualifies all3targets.

| Representation at rank128 |Target1 median/p95|Target64 median/p95|Target127 median/p95|
| --- | ---: | ---: | ---: |
|Full matrices|.78771 /5.55324|.67037 /.90864|.84223 /1.00000|
|Delta, identity coordinates|.85788 /9.59722|.53425 /.92511|.66837 /1.12658|
|Delta, matched neurons|.89706 /4.57945|.53316 /.85741|.67060 /.98592|

These are relative down-vector errors on336FORCED validation inputs per expert,
not accuracy percentages or source posterior losses. ReLU and private corrections
remain correctly coupled. Source-unavailable foreign core/input/readout is absent.
Full base removal and cyclic private-ID controls retained for every delta rank.

## Algebraic density constraint from the complete spectra

The minimum matrix Frobenius rank retaining95%energy is measured from the complete
qualified spectra, without additional fits or native forwards:

| Matrices |Target1 WI/WO|Target64 WI/WO|Target127 WI/WO|
| --- | ---: | ---: | ---: |
|Full|584/538|594/619|596/613|
|Identity deltas|613/545|619/631|619/619|
|Matched deltas|618/545|624/631|624/618|

To retain95%of BOTH matched delta matrices for ALL3targets at one equal rank
requires at least631. With D768/M3072, F32 private factors then contain
2*631*(768+3072)*4=19,384,320B per expert, before shared base/index/header.
Original source I8 expert including row scales is4,733,952B. Thus increasing
rank to satisfy THIS Frobenius metric loses the storage advantage by a factor
about4.095, and top1 arithmetic adds the full base. This is a metric-specific
constraint, not a lower bound on input/curvature/KL-aware or nonlinear compression.

**Close reference0/unit-WI matching/Frobenius low-rank private-factor recipe and
the same fixed full-matrix controls. Do not increase rank or start training it.**
No global exclusion of other references, joint alignment, nonuniform metrics,
precision/dictionaries or learned representations follows.

## Exposure and interpretation limit

Natural captured development/validation counts for0/1/64/127 are0/0,0/1,4/0,9/3.
Forced-input coverage is broad but differs from actual conditional routing domains.
Consequently this is not proof of poor task quality on naturally routed experts,
and the existing data cannot identify a supervised private fit for these IDs.
Any activation/domain-aware recipe needs a NEW justified exposure/data protocol,
not fitting validation or relabeling forced responses as natural training.

The shadow/native numerical error is orders below the candidate vector error;
native quantization mismatch is not its scale. Matching is exact functionally
but not sufficient as a compression method. A small collective output basis and
small private Frobenius matrix factors have now BOTH failed their specified tests.

## Cost, retention and next action

116.235s total/admission13.297s/numerical102.938s,2,582,118,400B peak,
10,810,105,900B freshly hashed,3NPZ archives279,931,248B. All45F64 validation
candidate responses,30private-permutation responses, common-only control,
F32 factor parameters/permutations and original inputs/native references retained.
CPU0/Torch1/BLAS1/SciPy1.18.1, runtime library SHA hashes recorded. No gradient
fit/GPU/T4/network/source download/C edit/native benchmark or actual DRAM counter.

Next prioritize a NEW calibration-free packed-I4 native-function/head quality
probe on the original source, without imposing rank truncation or changing source routing/core.
Lower precision changes scalar encoding rather than assuming low-dimensional
matrix geometry. It can reduce stored/read bytes, but has no quality/rate promise.
Qualify nibble packing/integer bounds/operator before local source-relative
posterior gates; only a promising result licenses full-native export and a cost
screen charging LUT construction, unpacking, routing and real memory traffic.
This is different from398/400 additive I8-book LUT and410 static vector-pair/Q4-head
formats. No NEW447 source/protocol/output exists yet. Goal ACTIVE/INCOMPLETE:
usefulRAM-n/LUT/routing/physicalDRAM/whole quality/SAMEartifact50/families/~100B OPEN.
