# Common C-input core and norm diagnostic: complete and close

9 October2026. Actual NEW72 module/case evaluations, ALL3132 core outputs and
6264 norm vectors. **COMMON_INPUT_CORE_AND_NORM_CLOSE**, zero material rows at
fixed1e-4 relative RMS. Old whole C parity FAIL19/32 remains unchanged.

[Implementation](../../../benchmarks/native_expert_scaling/chatbot_hybrid_common_core.py),
[protocol](CHATBOT_HYBRID_COMMON_CORE_PROTOCOL_20261009.md),
[binding](chatbot_hybrid_common_core_binding_20261009.json),
[result](chatbot_hybrid_common_core_result_20261009.json),
[terminal](chatbot_hybrid_common_core_result_20261009.terminal.json).
Freeze4e71039ce0732ec22a3747f0bc6fc3371cd75cce;
binding SHA558200ba7fe3de548cd3c15dc730827250f92e1ef640433bf29620d764fd799b.

## Controlled observation and result

Read actual packed learned F32 core/norm fields and exact C-trace operands.
Each of10 SSM and2 SWA modules starts zero once per case, then processes its
whole common-input sequence; no per-token reset. Source-compatible F32 SSD
chunk16/eager fallback, actual packed RoPE frequencies, TF32 off. No teacher,
embedding/head/full target, bank, training or old whole-prefix evaluation replay.

Norm1 uses stored C block input. Norm2 uses F32(saved input+saved C core output),
independent of the NEW GPU core output. Thus core and normalization discrepancies
are separate observables. All12 sites/all6 cases/all261 inputs per site retained.

| Common operand comparison | Vectors | Maximum relative RMS | Rows>1e-4 |
|---|---:|---:|---:|
| SSM/SWA outputs |3132|2.8729021374416037e-6 =0.0002872902%|0|
| Input/FF normalization |6264|7.862290452927573e-7 =0.0000786229%|0|

Maximum core error is layer3; maximum norm error is input norm layer1. All
sequences finish; all outputs finite. F64 energy/error reductions, no tolerances
changed. This supports the implemented core algebra on these common short C
operands, **not equality of the original whole GPU/C trajectories** or long drift.

[Common banks](CHATBOT_HYBRID_NATIVE_RESULT_20261009.md) found one0.831066%
same-input backend anomaly/3132; later original code logits pass. Combined
evidence leaves discontinuous activation rounding under small upstream
perturbations as a plausible mechanism, not a proof explaining all19 whole
failures. No packed/LUT/core structural defect has been localized. Do not infer
source preservation, accepted50 or new numerical admission from component closeness.

## Cost and continuity

Family54.735s, worker52.500s, actual exit0. Worker OS peak1,238,282,240B through
exit; GPU allocated157,537,280B/reserved278,921,216B. All fixed600s/4GiB OS/
2GiB allocated/3GiB reserved/64MiB output/4MiB log gates PASS. No first fault.
289 outputs20,150,505B include ALL19,243,008B raw vectors,72 case receipts and
frame metrics. Exact commands/PIDs/output hashes in terminal receipt; all data
remain off-repo at bound paths. Frozen target/launcher bytes identify this run;
later source-capture launcher schema changes do not replace that identity.

**Decision:** stop this common-core diagnostic as promised. Reuse all outputs.
Independent broader original-source supervision acquisition is now the next
concrete quality-recovery prerequisite; it does not waive native parity failure
or start long training. Then adopt saved transport and implement a finite whole
robust-QAT pilot under [balanced recovery NEXT](CHATBOT_HYBRID_BALANCED_RECOVERY_NEXT_20261009.md).
Keep original LUT/ternary/SSM forward and all full quality/rate/n/family gates.
