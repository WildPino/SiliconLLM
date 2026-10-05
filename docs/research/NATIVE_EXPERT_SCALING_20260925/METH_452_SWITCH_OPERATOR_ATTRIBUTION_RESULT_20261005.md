# METH-452: WO quantization suffices for all four observed compact pair crossings

**ALL9 apparatus PASS. Diagnostic completed; no candidate-model promotion.
Goal ACTIVE / INCOMPLETE.**

Source/math/protocol frozen ae28ebd, operational resumption f252a01 BEFORE first
import. ONE session69249, exit0. [Protocol](METH_452_SWITCH_OPERATOR_ATTRIBUTION_PROTOCOL_20261005.md),
[raw](meth452_switch_operator_attribution_result.json) SHA256
857128c717284ab7796384e1ad6d3d214d64e46da43158efd127a0839080e3c7.
451 first compact-gate failure retained d39fdf6 before this diagnostic; no451 edits.

    .venv/Scripts/python.exe benchmarks/native_expert_scaling/meth452_switch_operator_attribution.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth452_switch_operator_attribution_result.json

## Scope and numerical qualification

SAME336 consumed source128 teacher-prefix positions/IDs/input/p, retained451
basis/codecs/native original norm/head.0=451 rotated source-I32/F64 scale;
1=451 block64-I4. BOTH z00(source) and z11(compact) all672 complete logits/
states/IDs BYTE-EXACT451 before hybrid metrics.21,590,016 vocabulary rows replayed.
212 actual source-I32 matrix fingerprints match451, covering106 naturally routed
IDs, not all128 functions' new exposure or counts of useful experts.

All2016 projection primals exact:1008 source I64/reference,1008 compact I32/I64/
ascending-F64. All1008 actual activation basis transforms match independent
explicit F64 Walsh products here (relative error0.0):336WI/672WO. Exact tiny
baseline/main/joint/tie qualifiers, independent complete-vocabulary KL and all
336 full factorial/pair identities passed, max observed residual0.0.

## Separate local information loss and global native argmax

| Precision at WI/WO |Mean original posterior KL |Changed native argmax |
| --- | ---: | ---: |
|0/0 source diagonal (451)|1.2097144759e-8|0|
|1/0 only WI compact |.0004016973340|3|
|0/1 only WO compact |.0002989784082|7|
|1/1 compact diagonal (451)|.0006526655533|4|

WI-only median/p95/max KL .0000399709184/.0020180036984/.0090806925187;
WO-only .0000219126487/.0014460265805/.0140940258122.
Book maxima WI-only .0006370169601, WO-only .0006282278498.
WI-only global changes30/108/217; WO-only30/86/175/217/257/276/332.
No precision/counts chosen from these consumed metrics. Lower mean KL again
coexists with more native argmax changes; neither metric implies semantic harm.
The higher-precision hybrids are diagnostics and exceed the prior60% storage cap.
Their predictive counts do NOT constitute an eligible compact candidate.

## Exact factorial readout-plane attribution

For C(v)=v_b-v_a, original winnera and retained compact competitorb:

    z11_a-z11_b=m-baseline-WI-WO-interaction
    baseline=C(z00-z0); WI=C(z10-z00); WO=C(z01-z00)
    interaction=C(z11-z10-z01+z00).

This is an exact discrete identity, not a linearized ReLU/RMS/head model.
Selected-pair loss uses negative gap or zero gap withb<a; global winners recorded
separately. All observed451 changed pairs have the following terms:

| Index |Original pair gap |WI contrast |WO contrast |Interaction |WI-only pair gap |WO-only pair gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|30|.0074970722|.0706222057|.0227284431|.0043776035|-.0630958080|-.0152020454|
|217|.0232577324|.0949954987|.0610418320|-.0305376053|-.0714259148|-.0374722481|
|257|.0069921017|-.0467748642|.1110846996|-.0014102459|.0535349846|-.1043245792|
|276|.1027632952|.0512014627|.1067544222|-.0004616976|.0509828329|-.0045701265|

Raw retains baseline and every exact term/identity for all336. Baselines at these
indices are -.0000293255/-.0003118515/+.0002319813/+.0005789995.
WO-only crosses ALL4 compact-selected pairs; WI-only crosses2(30/217). Both
hybrid pairs cross2, WO-only crosses2, WI-only-alone0, only-joint0.
No selected pair requires the explicit interaction term to cross. This does NOT
mean interaction is zero or irrelevant: at217 its negative value reduces the
combined adverse contrast. WI can oppose WO (257) and affects other positions108.
No claim that WO dominates all norms, all KL or whole-model errors follows.

276's SOURCE-to-compact competitor gap is.1027633; that competitor is not assumed
the source runner-up. The '<.1 original top2' descriptive property of450's earlier
447/448 flips cannot be exported as a certificate for this new encoding.

WI compact changes the source-control ReLU sign mask by mean25.3244/max76 of3072
neurons (actual source0 versuscompact1 WI). Counts do not separate amplitude from
mask effects and do not prove mask changes alone cause the crossings.

## Decision: prioritize WO fidelity with an explicit physical path

Preserve451's FAIL and full goal. Restoring WI to the uncompressed reference
alone leaves these WO-only crossings in this local diagnostic. Before calibration/learning, test a
NEW physical possibility: retain ORIGINAL-coordinate native I8 WO and skip ONLY
zero A16 activation codes exactly, with transposed column-contiguous coefficients.
[Prospective applicability/primal plan](SWITCH_SPARSE_NATIVE_WO_NEXT_20261005.md)
explains the changed operator/layout/storage/active-cost variable. This is NOT a
promotion of the expensive rotated-source hybrid or a relaxed451 storage gate.
If zeros do not buy sufficient complete cost, follow the separate conditional-
information-aware compact WO route with justified development exposure.
No453 source/protocol/weights/zero-rate measurement or cost outcome yet.

## Resources/retention/limits

183.390s total,7.906s admission/175.484s numeric; peak2,308,653,056B;
8,538,237,148 streamed file bytes hashed,4 outputs102,883,788B.
Zero updates/new encodings/GPU/C changes.1344 complete native head forwards,
672 diagonal replays plus672 new hybrid heads, all charged. Source-C/I8 payload/
engine/original374/389 unchanged. All sessions terminal, no model jobs.

Full conditional usefulness/n increments, CPU routing/LUT/actual DRAM, new
all-layer source-relative held-out/generation/tasks AND>=50 acceptedIDs/s on SAME
artifact, other families and actual~100B remain required. Local teacher-prefix
causality is not fresh complete donor equivalence or generation evidence.
