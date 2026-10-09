# Direct tensor export and original AQ63 learner: COMPLETE/PASS

9 October 2026. Goal ACTIVE/INCOMPLETE. [Prospective protocol](ORIGINAL_TENSOR_LEARNER_PROTOCOL_20261009.md)
was frozen before observations in commit `1d7988c67d9e0f379f4a8307a8a39d15d48a74b1`.
Binding SHA `a0b835d2c301bc4f69baa8bd31f3e5890ffb2ed4f33e0dfe8d800be8a632c49f`;
[raw result](original_tensor_learner_result_20261009.json) SHA
`e3732b340c045681c6b6994d162d89b5420a7d3a8065a8d36ec96c7ee284e9ce`.
[Held terminal](original_tensor_learner_result_20261009.terminal.json),
[worker log](original_tensor_learner_result_20261009.worker.log).

## Implemented change

`original_tensor_learner.py` provides a complete trainable original D256/L6,
five Mamba1 SSM blocks/one SWA block, dReLU/flat router/full softmax/stable lower-ID
ties/normalized top8 mass, original AQ63 and scale application. The retained
phase59 learner omitted activation quantization. The new forward uses integer
sums before separate F32 activation/weight scale products and puts mass BEFORE
hidden AQ63, as the actual engine does. STE and its limitations are specified
in the protocol. Torch reductions/transcendentals/scan remain numerical
approximations to original AVX/FMA/libm/fast exponential arithmetic.

The direct producer accepts trainable master tensors and arbitrary supported
runtime E/V. It streams one field into E4BPv001, retaining only packed ternary
codes/scales and F32 organs in the native file. This removes the legacy-parser
dependency for future donor-derived tensors. Arbitrary E/V support is implemented;
the real-master observation here uses E32/V1024. Prior large-n/full-V loader
verification remains separate evidence.

All92 real checkpoint tensors/22,516,672 master coefficients were adopted, with
complete key/shape coverage. No old optimizer or training history was adopted.
Initial direct export24,411,136B is bit-identical to the immutable previously
verified packed E32 file, including all organs/codes/scales.

## Actual evidence

| Observation | Result |
|---|---|
| Four fixed bank inputs/all6 sites/all256 outputs | All6,144 coordinates compared; worst row normalized RMS7.533773e-7 <=1e-4 |
| Zero input | Exact zero in native and learner |
| Temporary all-zero router tie control | IDs0..7/mass1/8 exact |
| Actual bank routes24 calls | All192 IDs exact; max mass difference4.172326e-7 <=1e-6 |
| Before-update complete16-ID history/all16,384 logits | All16 rows PASS; worst normalized RMS4.443173e-7;16/16 greedy agreement |
| Before-update96 actual site routes | All768 IDs exact; max mass difference4.172326e-7 |
| One NEW whole-model CPU Adam update | Completed; all92 gradients/parameters/moments finite; nonzero aggregate organ/bank/router gradients |
| After-update native and learner/all16,384 logits | All16 rows PASS; worst normalized RMS6.271990e-7;16/16 greedy agreement |
| After-update96 actual site routes | All768 IDs exact; max mass difference3.278255e-7 |
| New actual-master native integer witnesses | All6,144 coordinates exactly equal independent CPU int64 dots |

The brief whole history reuses the old immutable native input/output packet as
reference for a NEW learner forward. No previous legacy native observation was
replayed. Changed-model packed native forward and four bank-native inputs are
new observations required to verify the new learner/export path.

The one update uses original16-ID small-vocabulary history/shifted labels,
next-token CE1.755495787, preclip global norm3.118798018, Adam lr1e-5/betas(.9,.999)/
eps1e-8/wd0/foreachFalse/clip1, CPU F32/seed1709/nonreentrant block checkpointing.
It proves an actual trainable/exportable state transition, not model improvement
or fresh-quality evaluation. This original validation fragment is an apparatus
control, not an untouched quality holdout after this update.

Actual state:
`results/native_expert_scaling/original_tensor_learner_20261009/candidate.pt`,
270,309,072B/SHA `5de1d580d45d3b7aca97155abb4e8ba2c7e978ed3597cce35915f59d1cf3fae8`.
Contains model/Adam1/RNG/input/labels/gradient metadata. Changed packed artifact
SHA `26378408e2433e716ec823bb43751cffd485bf3e709335932e3635cfafe4d167`.
The candidate is an enabling control, not a chatbot candidate for the goal.

## Resources and provenance

Worker PID30980/create time retained, exit0, session31156 CLOSED. Held family
36.281s <=240s including launch/pre-post checks; binder5.620s is additional.
Result elapsed33.953s was sampled before final serialization; last complete
worker event34.125s. Held worker peak949,977,088B, launcher27,402,240B, largest
directly held child30,658,560B: conservative sum1,008,037,888B <=3GiB. Namespace
319,772,161B <=1GiB. Nested compiler/linker peaks were not separately held.
No fault/retry/source query/GPU execution/RESERVED/T4. All bound input pre/post
checks and resource gates pass. Original source and three foreign tracked
modifications retain their exact previous disk hashes.

Previous packed-stage launcher bytes are retained in
[frozen prior launcher](original_tensor_previous_launcher_frozen_20261009.py.txt),
SHA `651a637ee24ca1a3f344a926009b7b095eb6855ab56d04bf8fce6bbc3a3d6cc8`.
Old bindings continue to refer to their historical launcher bytes.

## Decision and limits

Direct new-learner tensor export and an actual original-operator learning step
now pass the declared finite numerical scope. No additional E32 dose or loader
sweep is selected. [Next](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md) constructs
source-informed functions/core/readout and measures complete longest Falcon FIT
history with explicit touched-union residency.

This is E32/V1024/16 IDs/CPU evidence. It does not establish long-history native
parity, useful Falcon preservation, n1152 GPU residency, structured routing,
physical DRAM scaling, multi-family transfer or accepted>=50. The complete
original-shaped donor learner and full-history pilot remain missing.
