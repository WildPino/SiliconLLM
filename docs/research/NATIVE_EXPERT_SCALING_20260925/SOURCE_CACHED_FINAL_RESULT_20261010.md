# Cached final-state custody: exact paired readout, measured prefill drift

10 October2026. Goal ACTIVE/INCOMPLETE. **PAIRED_CACHED_STATE_PASS** for the
single identified conversation; independent stored adjudication COMPLETE.
Original ALL48 full-prefill/cached-label alignment FAIL remains unchanged.
No compressed chatbot, native speed or large-n admission.

## Question and actual observation

Source full-prefill readout differed from cached-generation labels on3/53 IDs
in broad_dev_everyday_conversations_036, with positive teacher gaps in both
BF16/F64 head controls. We lacked the actual cached pre-final-norm states.
[Original protocol](SOURCE_CACHED_FINAL_PROTOCOL_20261010.md) freezes the new
observable and exact gates; [repair1](SOURCE_CACHED_FINAL_REPAIR1_PROTOCOL_20261010.md)
preserves the first startup fault and moves one check after mixer initialization.

One new original-source generation,148 prompt IDs/53 labels; each label's
raw h24 and postnorm feature captured at the final norm of its SAME invocation.
Cached contexts/positions exactly verified by worker: first full prompt then
52 one-token steps, positions147..199, cache lengths0/148..199. Source
revision80ebc50d7799a440b96c93bb6686a3924a09b0cb, BF16/eager/qualified SSD
tiling, original cache/EOS11-or228/max_new_tokens256/no-sampling settings.

All four exact gates pass: token IDs match retained original53 IDs; all cached
BF16 logits match original packet bytes; source norm reconstructed from captured
state matches observed normalized feature bytes; source BF16 head/multiplier
on observed normalized features matches same-invocation logits bytes. Every411
named source parameter's actual loaded BF16 bytes matches its safetensor field,
and identity/version/value manifests before/after match. Nonparameter buffers
are not covered by the parameter-value claim. No original runtime file modified.

Actual calls:1 generation/model instance,53 source base forwards/53 source heads,
1 new batched final norm reconstruction and53 additional head reconstructions.
1272 source cache-state updates. Optimizer/native/RESERVED queries0. The stored
audit performs0 new history calls and0 full-head contractions.

## Independent adjudication and new information

Full input/output hashes, all exact replay/norm/head gates, before/after
parameter manifests against411 source-field hashes, output argmax IDs and
resource/call counts verified. Analytic F64 norm/gamma versus observed BF16
feature relativeRMS.00231661763 (0.231662%)<=1%;24 scalar math.fsum head
witnesses max absolute discrepancy.0940732923<=1.0, allowing actual BF16
head rounding. Exact same-call BF16 reconstruction is the stronger pairing
observation; the scalar test is not an exact F64 output equivalence claim.

All90 fixed cache coordinates verify independent BF16 nearest-even conversion
of the supplied F32 values. All90 have nonzero discarded F32 bits. This
confirms that this source runtime actually rounds cached SSM states toBF16.
It does not attribute the whole-model drift uniquely to that rounding.

Cached raw h24 differs from retained full-prefill in93,843 of108,544 coordinates:
relativeRMS **1.1708068%**. Postnorm feature relativeRMS **.5549610%**.
All53 positions have identical consumed token contexts. The first label's raw
state already differs, before iterative one-token cached updates. Thus later
cache rounding alone is insufficient to explain the entire discrepancy;
different prefill shapes/reduction order/other finite precision remain relevant.
No new full-prefill forward was made here; existing qualified bytes were reused.

An additional independent PowerShell JSON comparison verified all53 base-entry
frames against original prompt/output IDs, expected cache lengths and positions;
[frame witness](source_cached_final_frames_witness_repair1_20261010.json) records
exact frame/record hashes. No new history/head call in this check.

| Label | Cached correct ID | Prefill ID | Positive cached gap | Prefill-minus-cached pairwise perturbation | Raw h24 RMS | Postnorm RMS |
|---|---:|---:|---:|---:|---:|---:|
|21|2632|3942|.125|.171875|.9737402%|.5201679%|
|23|4496|15155|.015625|.0625|1.3028376%|.6004022%|
|33|9330|228|.140625|.1796875|1.1165529%|.5138933%|

These pairwise perturbations match the prior saved margin diagnosis. ID228
is an EOS token under the actual source policy, so small distributional errors
must not be declared harmless. This teacher-forced comparison does not measure
a free generation following the earlier changed tokens, or prove its stop time.

The exact paired capture fixes an observability/custody defect for this control:
supervision must pair each state with the logits actually computed from it.
Full-prefill h24 and cached logits are not interchangeable exact targets in
this pinned numerical implementation. This does not explain actual51's0/16
tasks or the ideal h24P/actual-head KL13.567 collapse by itself.

## Immutable first faults and execution

Original freeze129a9a8/bind4914ab1f launch rejected overlapping process20380;
no worker. Rejection retained; process disappeared naturally before actual launch.
Original session42559 then exit1/71.625s, launcher25876/worker28136 gone: code
checked is_fast_path_available before FalconH1Mixer.__init__ defines it.
No model/source forward/generation/head/state capture occurred. Log/terminal
retained; no namespace/result existed. This is a mechanism fault, not scientific
FAIL. OS499,113,984+29,802,496B below8GiB. Original code/binding untouched.

Repair1 freeze`a6a0800a6df475fcdef65d135aa4b10eb16a0720`, sealed binding
`d787a906d0c52cfe5ec28e9792a9477ec7fe509e5146c82bd8fdac45d88307d1`:
60 inputs/4,949,692,079B. Initial repair1 binder output is retained separately;
sealed augmentation adds original fault/code/protocol/parent-binding identities
to satisfy repair protocol. Scientific code differs only by moving the flag
assertion after model construction. Scientific gates/caps unchanged.

Repair capture session36559 CLOSED/exit0, launcher25696/worker27836,
worker created11:57:25.627147+02:00, gone by11:59:22 observation.
Audit session61396 CLOSED/exit0, launcher2512/worker26904,
worker created11:59:37.148321+02:00, gone by12:00:03 observation.
No owned overlap; foreign work preserved, no T4.

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/source_cached_final_capture_repair1.py --binding docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_final_binding_repair1_sealed_20261010.json --binding-sha d787a906d0c52cfe5ec28e9792a9477ec7fe509e5146c82bd8fdac45d88307d1 --freeze a6a0800a6df475fcdef65d135aa4b10eb16a0720 --directory results/native_expert_scaling/source_cached_final_repair1_20261010 --out docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_final_result_repair1_20261010.json
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/source_cached_final_capture_repair1.py --audit --binding docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_final_binding_repair1_sealed_20261010.json --binding-sha d787a906d0c52cfe5ec28e9792a9477ec7fe509e5146c82bd8fdac45d88307d1 --freeze a6a0800a6df475fcdef65d135aa4b10eb16a0720 --source-result docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_final_result_repair1_20261010.json --out docs/research/NATIVE_EXPERT_SCALING_20260925/source_cached_final_stored_adjudication_repair1_20261010.json
```

## Measured cost

Successful held capture46.125s/worker28.656s (source loaded/sealed10.312s,
53 cached observations complete18.953s). Actual source generation phase includes
instrumentation8.641s; not CPU engine accepted throughput. OS3,726,835,712+
30,388,224=3,757,223,936B<=8GiB; GPU allocated4,024,430,592/reserved4,068,474,880B
<=10/11GiB.10 namespace files14,746,727B +result3,573B<128MiB; five BF16
numeric payloads14,545,108B (raw/observednorm/reconstructednorm plus two logits).
Original forecast14,328,020B omitted the217,088B reconstructed norm stream;
actual complete output remains far below cap. First startup/import cost71.625s
reported separately. Binder preparation outside held cost, about7.7s tool wall
for repair binder/AST and separately recorded augmentation, not a scientific timer.

Held stored audit14.313s/worker4.438s. OS68,251,648+30,502,912=98,754,560B
<=512MiB/120s. All gates/hashes/resources PASS; no new head contraction/history.

[Capture result](source_cached_final_result_repair1_20261010.json) SHA256
`0f7c069263ee87c1fcbcbe3c05eac057a6b87d407dedd1cdb9c561a32db577f0`.
[Stored audit](source_cached_final_stored_adjudication_repair1_20261010.json),2,589B,
SHA256`2e6eed570af9da23a5eba9e8616c38df587bf37a1f86187dfa263d87bfdacb38`.
Both terminal/log records retained. Actual measurements override forecasts.

## Next step toward the converter

Reuse the now-qualified paired capture instrument to obtain consistent FIT
features before fitting a paired output codec. Keep this diagnosed DEV case
out of calibration. Do not recapture completed53 states or continue unchanged
history training. [RMS-compatible codec algebra](RMS_COMPATIBLE_OUTPUT_CODEC_20261010.md)
gives two explicit original-engine representations; neither has been fitted,
exported or shown useful. [Next custody/codec work](PAIRED_OUTPUT_CODEC_CUSTODY_NEXT_20261010.md)
separates necessary calibration observations from compression fitting and
declares the still missing whole quality/runtime evidence.
