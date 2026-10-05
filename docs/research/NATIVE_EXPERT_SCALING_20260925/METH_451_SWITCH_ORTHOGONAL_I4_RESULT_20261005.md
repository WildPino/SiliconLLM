# METH-451: orthogonal arithmetic qualifies, fixed compact recipe still fails

**ALL11 apparatus PASS; ALL3 uncompressed-control gates PASS;4of5 compact gates
PASS. Argmax4of336 exceeds predeclared3. Goal ACTIVE / INCOMPLETE.**

Math/controller/protocol frozen ef63b0e BEFORE first import; exact resumption
72e181b. ONE session3229, terminal exit0; no scientific file changed or rerun.
[Protocol](METH_451_SWITCH_ORTHOGONAL_I4_PROTOCOL_20261005.md),
[raw](meth451_switch_orthogonal_i4_result.json) SHA256
26e6ee00de51ea1cfc4cf6617fc69cef36950e1298ae19f153357545dd19f46f.
All source/runtime/helpers/parents/raw archives freshly bound in the record.

    .venv/Scripts/python.exe benchmarks/native_expert_scaling/meth451_switch_orthogonal_i4_pilot.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth451_switch_orthogonal_i4_result.json

## Exact components and staged arithmetic

ALL336 original native FFNs/state/A16/p/full-head rows replay BYTE-EXACT before
real coefficient transformation. Original source128 payload/engine/binaries kept.
ALL256 real WI/WO matrices: signed I32 block-Walsh coefficient transform/inverse
exact, independent selected row/parity sums exact, full I4 inverse packs and
saved bank/signs exact. Tiny full explicit Walsh matrices/orthogonality/integer
products/dyadic activations/private ReLU/unchanged448 pack/primal all qualified.
Selected actual coefficient parity checks are not exhaustive independent full-
matrix transform products; full actual integer inverse witnesses are exhaustive.

Fixed Q_D=diag(H256,H256,H256)S_WI and Q_M=diag(H1024,H1024,H1024)S_WO.
Private ReLU retained in original neuron coordinates before Q_M. Exact-real
linear-map invariance implemented without intermediate I8 coefficient rounding.
Native new activation F64 FWHT then F32 cast/A16 is a separate operator.
ALL2016 actual transforms independently equal explicit F64 Walsh products here:
max relative error0.0,1008 WI/1008 WO. All672 higher-precision transformed-source
projection I64/reference sums/F32 outputs and1344 I4 I32/I64 block projections
exact. Original p/residual/finalnorm/I8 tied head fixed as LOCAL controls.

## Uncompressed control qualifies the new activation arithmetic

| Fixed control gate | Observation | Result |
| --- | ---: | --- |
| Mean original posterior KL<=1e-6 |1.2097144759e-8|PASS|
| Every book KL<=1e-5 |maximum2.0150986281e-8|PASS|
| Zero changed argmax |0of336|PASS|

Maximum per-position KL2.4213394578e-7. This control uses transformed source I32
coefficients/F64 source scales, not the compact bank. It supports local activation
arithmetic eligibility at these336 prefixes, not all-bank equality or deployed cost.

## Compact predictions: four of five unchanged gates pass

| Arm | Mean source posterior KL |Changed argmax |
| --- | ---: | ---: |
| Rotated source uncompressed |1.2097144759e-8|0|
| Rotated block64-I4 correct ID |.0006526655533|4|
| SAME compact bank ID+1 |.1803643085936|39|
| Function removed |.0327040931526|21|

Correct-ID median KL.0000566485667,p95.0037428905968,max.0105381960216.

| Book |Mean correct-ID KL |Changes |
| --- | ---: | ---: |
|18|.0008723654173|1|
|19|.0005575731038|0|
|20|.0007194344068|0|
|21|.0007791042346|1|
|22|.0006397817852|2|
|23|.0003477343719|0|

Mean<=.01/every book<=.05/identity mean-KL harm>=.01/storage<=.60 PASS;
argmax4of336 (.01190476)>.01 FAIL. Identity mean-KL increase.1797116430404,
not a count of individually useful functions. Removed full heads byte-exact443.

Original bank605,945,856B; compact coefficients/block scales/signs339,742,464B,
ratio.5606812236372486. Actual bank.npz339,743,982B, SHA256
81792b8120d7a32bc313edda11455e590ef873107579a56ac9407d71f3532e15.
All128 encoded fingerprints distinct; source representation identities retained.
WI relative Frobenius errors .107239648.. .107842940;
WO .106537504.. .108082476. Narrow matrix-error distribution does not certify
all conditional functions, source decision margins or whole model quality.

Changed indices30/217/257/276.30/217 overlap447 row;30/217/257 overlap448 block.
276 is new relative to both.448 had9 flips;451 has4 with slightly higher meanKL.
No monotonic relationship between average KL/matrix error and argmax count.
These are descriptive comparisons on consumed data, not basis selection evidence.

## Decision: diagnose the two operators, not another seed

THIS fixed signed-basis/block64-I4 recipe CLOSED before native C timing/export.
No blocksize/precision/sign sweep, gate relaxation or interpretation as whole
quality. Exact orthogonal coefficient/activation witnesses are reusable components;
this particular compact representation remains ineligible.

Next proposed NEW452 [operator attribution](SWITCH_OPERATOR_ERROR_ATTRIBUTION_NEXT_20261005.md):
exact2x2 WI/WO intervention at SAME original IDs/input/p, replay BOTH451 diagonals
byte-exact first, then two hybrid diagnostics and readout-margin main/interaction
terms. Determine WI, WO or joint-error constraints before selecting a conditional
metric/calibration/representation change. Hybrid higher precision is a diagnostic,
not a compact model or satisfaction of the original storage/quality/rate goal.

## Resources and retained outputs

296.062s total,15.203s admission/280.859s numeric+reporting; peak2,625,191,936B,
12,718,121,756B hashed,11 outputs606,546,914B, zero optimizer updates/GPU/C changes.
All inside prospectively frozen300/600/900s,4GiB/640MiB. Complete bank/signs,
original prefixes/logits, four arms' full logits/states/IDs retained with sizes/SHA.
Original/removal archives identical447/448/443 where specified. All sessions
terminal, no model jobs; engine/original374/389/Torch CPU runtime preserved.

The full objective still requires useful RAM-scaled n, routing/LUT/actual DRAM,
new all-layer source-relative held-out/generation/tasks AND>=50 acceptedIDs/s on
SAME artifact, other families and actual~100B as resources allow. No rate inherited.
