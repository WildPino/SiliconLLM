# One output-weighted paired codec: numeric PASS, FIT quality FAIL

10 October 2026. COMPLETE including independent full stored audit.
Decision **PAIRED_CODEC_FIT_QUALITY_FAIL**. Goal INCOMPLETE.

## Changed variable and record

[Coherent FIT states](SOURCE_CACHED_FIT_RESULT_20261010.md) supplied actual
postnorm features paired with cached-generation logits. The new rank255
encoder B and decoder K minimize one declared mixture-Fisher quadratic on
these24 cases, with a constant-norm carrier in original256-dimensional RMSNorm.
This corrects representation/norm/head pairing; it is not a trained causal core.
Old raw-projection coupling failures remain immutable, not reinterpreted as
universal compact-state limits. No source forward, optimizer or DEV observation.

[Tool](../../../benchmarks/native_expert_scaling/paired_output_codec.py),
[protocol](PAIRED_OUTPUT_CODEC_PROTOCOL_20261010.md), freeze
`b82c4b722c3ce1d3883bf48ba352fef7ed03f987`.
[Binding](paired_output_codec_binding_20261010.json) SHA256
`027117b4676a934adb4269f0129ad4f07ff6d4dda69035d4d5ed55a04041c9a7`,
301 inputs/6,797,817,706B. Native DLL SHA256
`53416b0f93fc928d8d358cefa89b291d5b90857511c81f896ac4c20d32d428bb`;
four original hsum256/dotf/matvec/rmsnorm bodies unchanged, compiler/body/DLL
extents sealed. [Compile preparation](paired_output_codec_native_preparation_20261010.json)
2.875s/no readout call; [ABI preflight](paired_output_codec_native_abi_preflight_20261010.json)
load/dimensions256/65537/AST PASS. Binder about6.904s tool wall, separately.

Fit session23652/launcher26956 created12:57:12/worker23292 created
12:57:19.0497663+02:00, exit0/222.297s, CLOSED. Audit session40785/launcher25468
created13:01:02/worker19648 created13:01:08.9967425+02:00, exit0/140.047s,
CLOSED. All four PIDs absent at13:04:19 observation. No first fault in this
family; all cases and full audit completed despite scientific quality failure.

## Quality on the fixed FIT corpus

24 cases/12 domains/4422 labels, equal case weight in calibration and case
metrics. Source actual cached BF16 logits are the reference. The SAME pair is
evaluated in real F64 and by original F32 norm/matvec on injected final states.

| Metric | Real F64 | Original F32 |
|---|---:|---:|
| Case-average full-V KL | 0.8041212050514369 | 0.8041212044322580 |
| Label-average KL | 0.9509256593649659 | 0.9509256545071849 |
| Case-average ID disagreement | 22.16840061% | 22.16840061% |
| Wrong labels /4422 | 1102 | 1102 |
| Highest case KL | 1.938693223553071 | 1.9386931736715325 |
| Highest case disagreement | 47.265625% | Same |

Case-average uniform KL on THESE cases10.6993726481; teacher entropy.3909974996.
Real predictor entropy.8504411688. Lower loss than uniform does not admit useful
chatbot quality. Every frozen quality flag FAILS for BOTH views:meanKL<=.01,
meanDis<=1%, every-case KL<=.05/dis<=5%, and all-domain gates. Two domain
aggregates pass their domain screen;10 fail, not a claim that every domain fails.

| FIT domain | Case-average KL | ID disagreement |
|---|---:|---:|
| apigen-80k | .0308853 | 0% |
| explore-instruct-rewriting | .0237423 | 0% |
| metamathqa-50k | .339159 | 9.26285% |
| numina-cot-100k | .546665 | 12.5% |
| openhermes-100k | .589890 | 13.8672% |
| self-oss-instruct | .753924 | 17.5781% |
| smol-summarize | .767708 | 29.8182% |
| everyday-conversations | 1.04146 | 31.6810% |
| smol-constraints | 1.18714 | 32.9497% |
| smol-rewrite | 1.37216 | 41.2153% |
| systemchats-30k | 1.46593 | 36.7188% |
| smol-magpie-ultra | 1.53079 | 40.4297% |

These are FIT injected-readout measurements. They cannot be compared as a
matched recovery ratio to the old actual51 DEV/native or DEV h24P interventions,
which use different cases/heads/execution and a real causal path.

## Algebra, conditioning and precision

Uncentered covariance rank2047 after the predeclared relative1e-12 support cut.
Raw smallest eigenvalue2.3848896591e-6, largest11,399,484.8777; retained
condition1.25419923483e11, discarded positive trace fraction2.05437695669e-13.
One small direction excluded, not a claim of intrinsic source rank2047.
Whitening relative3.59646760846e-12; encoder-root identity1.01336316120e-14.

Gain4.328941201436105/rho1; maximum FIT ||phi||34.63152961149, information
coordinate radius<=8 within declared16 carrier domain. Gain is now frozen,
with no future-input guarantee. Actual native final norm/head/state are F32.

Direct quadratic J=.35160991716072604 agrees with discarded metric trace plus
eigenvalue tail .35160991716072854, relative to total17.159117509079024 error
1.45578687487e-16. About97.95% of this quadratic metric energy is retained.
All categorical fidelity gates still fail: the surrogate is not exact KL,
average context Fisher or a claimed KL bound. This is the optimum for the
chosen support-constrained linear quadratic problem, not optimal categorical
coordinates/head, a best nonlinear codec or a D256 theorem.

ALL numeric flags PASS. Maximum native norm versus F64 analytic norm of ACTUAL
F32 state relative5.88230140005e-8 (gate1e-5); all-V native-real score maximum
absolute4.89859137005e-5 (gate1e-4). Case-average KL(real||native)
5.85513547958e-13 (gate1e-6), zero argmax differences across all4422 labels.
Native arithmetic is faithful for this injected artifact within frozen gates.

## Complete independent stored audit

[Audit](paired_output_codec_stored_adjudication_20261010.json) checks every input/
output hash, sealed case metadata/native bodies, all eigensystem residuals and
orthogonality, all decoder rows against source M/T, all qbar coordinates through
independent logaddexp normalization,32 scalar covariance and32 output-metric
math.fsum witnesses, every phi/state, BOTH full-V score streams via CPU F64
linear reference, independent per-label metrics/aggregates/flags/counts/caps.

Maximum eigensystem/identity relative3.97e-15; decoder reconstruction absolute
1.38555833473e-13; qbar delta9.71445146547e-17; C scalar delta0; G scalar
delta5.25160427298e-20. Full real-score reference max1.27897692437e-13;
native scores versus F64 head on ACTUAL native features4.93489109843e-5;
metric delta1.77191594730e-12<=1e-10. All audit checks PASS and reproduce FAIL.

One paired linear fit, two eigensystems,17 output-metric blocks/17 decoder
factorization blocks. Fit executes4422 compact real heads and4422 original
final norm/head rows in24 DLL invocations. Full engine calls/source model/
generations/history/LM-head/optimizer/DEV/RESERVED counts0. Audit reconstructs
8844 stored linear head rows and verifies one decoder factorization; no native
binary call, eigensolver replay, new fit or donor history. These arithmetic
verification costs are counted, not described as zero work.

## Resources and reproducibility

| Resource | Fit/control | Full audit |
|---|---:|---:|
| Held seconds including seals | 222.297 | 140.047 |
| Stored worker prewrite seconds | 203.031 | 126.859 |
| Worker OS peak through exit, B | 2,203,852,800 | 1,668,169,728 |
| Launcher OS peak, B | 34,271,232 | 32,452,608 |
| Combined OS, B | 2,238,124,032 | 1,700,622,336 |
| GPU allocated peak, B | 209,895,424 | No GPU |
| GPU reserved peak, B | 358,612,992 | No GPU |

All caps PASS:900s/6GiB OS/4-5GiB GPU/5GiB output and900s/3GiB audit.
162 namespace files3,945,882,081B; result1,613,572B; total3,947,495,653B.
Score payload3,477,655,368B matches frozen4422x65537x(8+4), matrices extra.
Injected final-head timings do not admit end-to-end CPU50token/s/DRAM/useful n.

[Result](paired_output_codec_result_20261010.json) SHA256
`bfe8173b00c3dbaeb4d9b5656e734e249baf7cf0d4e036163941bb4b7e7879ad`.
[Audit](paired_output_codec_stored_adjudication_20261010.json),6,121B, SHA256
`516c44f13b9eb55fd50ecb447000d8f021fddcf509b9d67591965d606ae73ae7`.
Terminal/log JSONs beside both outputs retain exact command/creation times/caps.

Exact completed commands from repository root:

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/paired_output_codec.py --binding docs/research/NATIVE_EXPERT_SCALING_20260925/paired_output_codec_binding_20261010.json --binding-sha 027117b4676a934adb4269f0129ad4f07ff6d4dda69035d4d5ed55a04041c9a7 --freeze b82c4b722c3ce1d3883bf48ba352fef7ed03f987 --directory results/native_expert_scaling/paired_output_codec_20261010 --out docs/research/NATIVE_EXPERT_SCALING_20260925/paired_output_codec_result_20261010.json
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/paired_output_codec.py --audit --binding docs/research/NATIVE_EXPERT_SCALING_20260925/paired_output_codec_binding_20261010.json --binding-sha 027117b4676a934adb4269f0129ad4f07ff6d4dda69035d4d5ed55a04041c9a7 --freeze b82c4b722c3ce1d3883bf48ba352fef7ed03f987 --source-result docs/research/NATIVE_EXPERT_SCALING_20260925/paired_output_codec_result_20261010.json --out docs/research/NATIVE_EXPERT_SCALING_20260925/paired_output_codec_stored_adjudication_20261010.json
```

Immutable namespaces must not be replayed or consumed scripts/results edited.

## Next converter decision

[Head image versus encoder diagnosis](PAIRED_CODEC_HEAD_IMAGE_NEXT_20261010.md)
derives exact forward-KL/moment-matching and ball-constrained certificates.
Price/freeze one fresh-head probe using this new K/a and stored initial
coordinates; no old-head iterations or gain grid. A feasible better coordinate
can establish encoder room, while a certified whole-cohort lower can reject
this head/domain. Finite unresolved optimization is not a capacity theorem.
Matching DEV final-state custody/held-out evaluation remains missing and must
reuse the completed53-label case without a DEV representation refit.

Original SSM/SWA/LUT/ternary causal core/functions, useful own-history chatbot,
same-artifact50token/s, RAM-driven n/CPU IDs and normalized mass/physical DRAM,
and another family/scale remain unproved. Do not restart unchanged joint
training merely because this new output arithmetic is valid.
