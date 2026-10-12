# Native composed operator V1: full audit and a retained precision failure

13 October2026. **COMPLETE / INDEPENDENT_STORED_NUMERIC_FAILURE.** No timing run.
All48 histories/two precisions/two norm modes collected and completely audited,
even though one normalized-input control failed. Goal ACTIVE/INCOMPLETE.

Actual CPU384x96 recurrence/source x/B/C projections. Original engine.c
dotf/hsum256 bodies retained verbatim in separate DLL,AVX2/FMA/native one thread.
Scalar head expf decay,fullskip/source gate/postgate RMS/BF16 input/output;
controlled F64 decode versus AVX_F32 decode,gating and normalization.

DEV reconstructed-denominator local errors:controlledmean.0688127314/
worst.0969493318;AVX_F32mean.0688128391/worst.0969495982. All original local
quality gates PASS,both precisions/both modes. This is supplied-source-operand
operator quality,not a chatbot. All prior1e-3 precision bridge flags PASS,
maximum.0002908011;historical earlier bridge failures are not superseded.

Full independent audit:12,987,072 projection values,8,658,048 scan values,
1,769,472 final state values,277,057,536 normalized values,184,705,024 rawscore
and184,705,024 BF16 output values. ALL92,352,512 AVX F32 rawscore words bitexact
separate scalar eight-FMA reference. Controlled normalized/raw/output vector
differences exactly0. Maximum scan5.22288e-7/finalstate1.39064e-6,
projection1.16551e-7,all source/export/hash/metrics controls qualify.

One false normalized-input flag at broad_dev_smol_summarize_034:
AVX_F32 relative1.264113100747e-4>frozen1e-4. Flag was retained and remaining
audit completed;terminal exit0 means successful complete adjudication,not
numerical PASS. No threshold change or failed-case exclusion. This locates a
discrepancy before final output contraction;it does not isolate which individual
decode/gate/RMS rounding dominates. Full scalar score agreement alone does
not repair a changed normalized input.

Selected correction:[mixed F64 decode/gate/RMS,BF16,AVX F32 output](COMPOSED_OPERATOR_MIXED_PROTOCOL_20261013.md).
Reuse exact admitted F64 normalized-input computation and independently qualified
F32 output primitive at its BF16 boundary. New artifact/all48 qualification
required before timing. Same weights/ranks,no learned DEV candidate selection.
V1 C/Python/protocol/binaries/results remain immutable.

Producer225.490s (see exact raw receipt),OSpeak549,408,768B;
audit139.0515s,OSpeak894,939,136B. New run arrays1,756,253,088B,
family with compile/parameters1,836,894,098B on E. Preparation17.8693s.
All900/1200s/6GiB/2GiB caps unchanged. Prepare92887/producer49207/audit76951
CLOSED0. Source/protocold1a471a,receipt extensiona9f35de,launchfcccf7a;
auditorbf8e117,launchae7c4c2. Original engine/foreign work preserved.

- [Binding](composed_operator_native_binding_20261013.json), [preparation terminal](composed_operator_native_binding_20261013.preparation_terminal.json), [launch](composed_operator_native_launch_20261013.json), [storage](composed_operator_native_storage_20261013.json)
- [Result](composed_operator_native_result_20261013.json), [terminal](composed_operator_native_result_20261013.terminal.json), [audit launch](composed_operator_native_audit_launch_20261013.json), [adjudication](composed_operator_native_adjudication_20261013.json), [audit terminal](composed_operator_native_adjudication_20261013.terminal.json)

Native source generator/conv/gates/fullskip/output widths still retained;
no source forward/refit/GPU/T4/RESERVED/whole model,n/routing/DRAM/same50 claim.
