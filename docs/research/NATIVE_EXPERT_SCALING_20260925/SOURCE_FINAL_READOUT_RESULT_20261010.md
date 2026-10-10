# Source final readout: complete qualification and retained alignment failure

10 October2026. Goal ACTIVE/INCOMPLETE. Scientific decision
**SOURCE_STATE_LABEL_ALIGNMENT_FAIL**, independently confirmed. No compact
codec fit, source history rerun, learner update or native inference.

## What was tested

The ideal projected targets failed with actual51 heads (KL13.567/disagreement
99.935%). Before fitting a different paired codec, this control tests the actual
source final operators on retained full-prefill h24 against cached-generation
teacher labels. [Frozen protocol](SOURCE_FINAL_READOUT_PROTOCOL_20261010.md).
It is a source-operator alignment prerequisite, not a converted chatbot.

Pinned Falcon-H1-1.5B source revision80ebc50d7799a440b96c93bb6686a3924a09b0cb,
D2048/V65537, ALL48 FIT/DEV histories/22547 retained positions/8808 labels.
24 FIT cases/4422 labels and24 DEV cases/4386 labels,12 domains in each split.
The original full-prefill states and cached labels remain separate observations.

Source BF16 h is normalized in F32 with eps1e-5, cast toBF16 before multiplying
BF16 gamma. Save these post-norm features. BF16 GPU source head is called one
label at a time and multiplied by.01953125 inBF16; F64 CPU head uses the SAME
savedBF16 normalized features and exactly decodedBF16 source weights. No F64
normalization is claimed for the second score stream. Source history forwards,
generations/model instances/optimizer updates/native calls/RESERVED queries0;
new BF16 source-head calls8808 and F64 head contractions8808.

## Frozen custody and commands

Code/protocol freeze `c30af06e9fddf4c701ec173053dc2335272a7dc8`.
[Binding](source_final_readout_binding_20261010.json) SHA256
`b656c6000cedd0a4646e021427b1632731e6cbfdbd48ec5fdb0edf30a9f40dee`;
135 inputs/6,174,991,446B. Source safetensors SHA256
`1fb788513f2c58e3abb91af8730afe19e247bce7a7f01a46e452a084e98b2abd`.
Principal Python/Torch CPU-CUDA/cuBLAS/NumPy/psutil runtime identities bound;
complete CUDA driver/runtime closure is not established.

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/source_final_readout_launch.py --binding docs/research/NATIVE_EXPERT_SCALING_20260925/source_final_readout_binding_20261010.json --binding-sha b656c6000cedd0a4646e021427b1632731e6cbfdbd48ec5fdb0edf30a9f40dee --freeze c30af06e9fddf4c701ec173053dc2335272a7dc8 --directory results/native_expert_scaling/source_final_readout_20261010 --out docs/research/NATIVE_EXPERT_SCALING_20260925/source_final_readout_result_20261010.json
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/source_final_readout_audit_launch.py --result docs/research/NATIVE_EXPERT_SCALING_20260925/source_final_readout_result_20261010.json --binding docs/research/NATIVE_EXPERT_SCALING_20260925/source_final_readout_binding_20261010.json --out docs/research/NATIVE_EXPERT_SCALING_20260925/source_final_readout_stored_adjudication_20261010.json --log docs/research/NATIVE_EXPERT_SCALING_20260925/source_final_readout_stored_adjudication_20261010.worker.log --receipt docs/research/NATIVE_EXPERT_SCALING_20260925/source_final_readout_stored_adjudication_20261010.receipt.json
```

Source session71059 CLOSED/exit0; launcher28460 created11:17:10.161573+02:00,
worker12572 created11:17:16.631149+02:00, both gone at11:29:07 observation.
Audit session99281 CLOSED/exit0; launcher1836 created11:29:35.987126+02:00,
worker27900 created11:29:36.313937+02:00, both gone at11:32:41 observation.
No owned overlap. Foreign work/publisher preserved. User temporarily requested
Fail-Fast, then explicitly revoked it; full frozen stored audit was completed.
No result or consumed protocol was edited.

## Measured quality

| Readout | Split | Case mean KL | Case mean disagreement | Label disagreement | Wrong/labels |
|---|---|---:|---:|---:|---:|
| BF16 | FIT | .000475258798 | .7201535% | .8819539% |39/4422|
| BF16 | DEV | .000562327626 | .8851989% | .7523940% |33/4386|
| F64 | FIT | .000466150598 | .7501738% | .9045681% |40/4422|
| F64 | DEV | .000553281857 | .9542266% | .7979936% |35/4386|

All four meanKL<=.01/mean disagreement<=1% gates pass. Every caseKL<=.05
and all domain gates pass. The sole failed criterion is every-case disagreement
<=5% in DEV, for BOTH readouts: `broad_dev_everyday_conversations_036` has
3/53=5.6603774%. Its BF16 caseKL.000724995164/F64.000693541508.
FIT every-case disagreement passes. These are the originally frozen criteria;
the failed prerequisite is not relaxed or converted into a success.

The low KL and localized ID changes motivate a margin/tie diagnosis. They do
not prove that the changes are harmless in autoregressive generation, nor
identify cached/full-prefill drift as the sole cause. Even an exact source head
on saved full-prefill states need not reproduce cached-generation IDs exactly.
Changing to F64 head accumulation did not remove this case's count of3 errors.

## Independent numerical adjudication

All input/output hashes, per-label KL/entropy/uniform baseline/argmax counts,
BF16-to-F64 rounding losses, aggregates and scientific decision independently
verified. Worst per-label metric difference7.05643e-13<=1e-10; aggregate
7.10543e-15. Analytic F64 norm/gamma versus savedBF16 feature relativeRMS
worst.00241483703 (0.241484%)<=1%; not exact bitwise identity.576 scalar
F64 head witnesses by math.fsum have maximum absolute error0<=1e-8.
No new full-head contractions, source histories or optimizer work in the audit.

[Result](source_final_readout_result_20261010.json),2,676,573B, SHA256
`47180815b05c2c3bccbda69b077e10e3440331ab9443bdb696630ae121466066`.
[Terminal](source_final_readout_result_20261010.terminal.json) and worker log retained.
[Stored adjudication](source_final_readout_stored_adjudication_20261010.json),30,274B,
SHA256`4c6ebc7922c91b94e2de468031dd9712c8e77a263d59ac3777cd092a597bd6bb`.
Audit receipt/log retained. Scientific FAIL and numerical audit PASS are distinct.

## Measured cost and limits

Held source family604.844s; worker572.610s. Forecast3–8min plusIO was optimistic;
actual held time about10.08min includes pre/post hashes and output sealing.
Worker OS peak2,388,025,344B +launcher34,652,160B =2,422,677,504B<=6GiB.
GPU allocated308,289,536B/reserved329,252,864B<=2/3GiB caps.240 files
5,810,321,381B in the namespace, below8GiB. Numeric payload5,808,576,528B:
BF16 scores1,154,499,792B/F64 scores4,617,999,168B/features36,077,568B.
Held audit177.391s (worker176.360s); OS253,870,080+28,000,256=281,870,336B
<=6GiB/900s. Binder preparation was outside held costs; exact total binder
duration was not captured, and is not fabricated. All original deadlines met.

## Decision and next action

Diagnose the stored score margins in the single failing case before compression:
[bounded protocol](SOURCE_READOUT_MARGIN_PROTOCOL_20261010.md). No new history
capture or paired codec fit yet. Retained full-prefill/source readout is very
close in distribution but fails the strict fixed ID gate. Diagnose exactBF16
ties, positive gaps, constant-shift invariant perturbation and off-by-one
controls. This can identify the next correction without attributing an
unobserved source computation or changing already observed criteria.

Whole compact chatbot, own-history quality, same-artifact>=50 accepted IDs/s,
useful large-n/structured CPU ID-mass/physical DRAM and family/scale remain open.
