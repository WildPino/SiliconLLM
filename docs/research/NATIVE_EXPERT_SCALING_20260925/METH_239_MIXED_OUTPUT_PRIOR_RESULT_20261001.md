# METH-239: mixed output codec conserves source-prior derivatives

Frozen at `2b830aa`; session35853 completes,exit0. All original-source,
native codec/input/table and full-input route/count/hash controls pass.
Same output-only FP64 projection,now initialized/encoded using physical
fixed32 BF16 escapes and row-Q8 remainder. No affine branch or extra cycles.

All16 QR condition/growth,feature/autograd derivative,unrounded gradient,
source-center,stored-gradient,function/distinctness/readback gates pass.
Maximum stored derivative Frobenius error **.006468215 (0.64682%)** versus
unchanged1% limit. Maximum condition11.7816,coefficient change.01102034.
Stored mixed rounding is approximate,not called an exact derivative.

Pooled65536 fit-input original-source SSE/energy=.0001716408 (0.017164%)
versus unchanged mixed source-down baseline.0002172962 (0.021730%),both
within1%. No captured output-label fitting,validation inputs/labels or new
prediction/generation/task source is scored. All16 actual representations
are distinct; this does not establish useful tenfold expert-count gain.

**Decision: mixed source-prior prerequisite passes; freeze actual E16/E160
conditional fitting next.** METH-238's8.727ms source fixture cannot substitute
for actual trained-bank native routing/DRAM/LUT or a complete model/rate.
Full donor-relative independent LLM quality,>=50 accepted token/s,large-RAM
n scaling and real second-family/approximately10B remain open.

[Raw result](meth239_mixed_output_prior_result.json),SHA256
`e6e72a440dc18e01411ab85ce79ec3f4e1bfe7e8aa9b1c80a3de43342a437a30`.
Local ignored snapshot
`results/native_expert_scaling/meth239_layer12_mixed_output_priors.safetensors`,
85,546,068bytes,SHA256
`1ffd777ad4115a73016acfdc99abd8a52cbd34836ab962bf0f8a2749f4b71dc4`.
Runtime25.437s after imports,RSS1.719GB,GPU peak822.556MB,local3060,no T4.
All16 audit,source/codec/key/tensor hashes and scoped derivative tests retained.

Next: [METH-240](METH_240_MIXED_CONDITIONAL_PAIR_PROTOCOL_20261001.md),fixed
actual source-output fitting over all4864 features with mixed readouts,
source derivative child priors,E16/E160/rotated and parent/child-prior controls.
