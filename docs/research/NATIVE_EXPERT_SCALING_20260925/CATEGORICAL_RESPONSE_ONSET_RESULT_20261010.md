# First response before any own-answer feedback: stored result

10 October2026. **FIRST_RESPONSE_CONDITIONAL_GAP**. Complete56-input before/after
hash checks,all16 full-V first score normalizations/independent logaddexp+math.fsum
KLs and48 phase metrics PASS. [Protocol](CATEGORICAL_RESPONSE_ONSET_PROTOCOL_20261010.md),
[result](categorical_response_onset_result_20261010.json). No model/history/core/
native/head FG/optimizer/GPU/T4/RESERVED call.

All16 task prompts use exact original source IDs; every new response begins
with ID1563. First-ID disagreement16/16,including14/14 cases the donor solved.
Mean source-relative first-step KL9.10091974726. Native failure is
already observable before any generated ID is fed back into the candidate.
This does not prove all internal states identical or rule out another readout.

| Native forced cohort | First-response caseKL | Continuation caseKL | First-response loss mass |
|---|---:|---:|---:|
| FIT24 | 4.57825472483 | 2.89729295054 | 1.175352% |
| DEV24 | 4.65732440147 | 6.26922903909 | 1.081925% |

In the case-balanced whole-response loss each first output carries1/(24m).
Its total mass is E_case[1/m], so98.82465% of FIT weight is continuation under
teacher-provided answer prefixes. This is a measured objective imbalance,
not proof that reweighting alone will repair generalization: DEV continuation
is also poor, and the32-update convex fit has an unclosed certificate.

Freeze68a863005eb5cdbcd7bd0c50cd3942941ad34d30;
binding408337284162f98d085d1ba76588de09f3c220da734e5d63c2bbf7674ba80ed7.
56 inputs175567157B, producer0.812s/OSpeak53325824B;
direct command exit0/observed1.278s including startup,60s worker guard/256MiBOS/
1MiB output. Scalar audit included in the same producer; no independent full
program replay claimed. Parent complete native audit adopted for all48 KL arrays,
no repeated6GB broad-score hash. All16 source/native full score files sealed.

Next use algebraic first-state controllability from existing24 good per-label
codes and native whitened features. A later onset-balanced objective would give
half the FIT mass to first responses,half to continuation, with all original
quality gates unchanged; not launched and not assumed sufficient. If linear
control is absent/ill-conditioned under native uncertainty, prioritize causal
code/SSM/SWA/ternary function construction rather than more unchanged head doses.
