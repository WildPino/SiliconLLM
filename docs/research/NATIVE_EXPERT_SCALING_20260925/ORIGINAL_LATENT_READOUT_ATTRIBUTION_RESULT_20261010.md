# Retained projected target versus actual51 decoders

10 October2026. COMPLETE; independent metric audit PASS. Scientific branch
TARGET_DECODER_COUPLING. Goal INCOMPLETE; no chatbot or speed admission.

[Frozen protocol](ORIGINAL_LATENT_READOUT_ATTRIBUTION_PROTOCOL_20261010.md),
[binding](original_latent_readout_attribution_binding_20261010.json),
[raw result](original_latent_readout_attribution_result_20261010.json),
[held terminal](original_latent_readout_attribution_result_20261010.terminal.json),
[independent audit](original_latent_readout_attribution_stored_adjudication_20261010.json),
[audit receipt](original_latent_readout_attribution_stored_adjudication_20261010.receipt.json).

## Actual intervention

Freeze67d2b03a09349cf0413346581161ff488817b6f5;74 consumed inputs/binding
SHA2e95dc67b836a96f4c17037a676febcb1db26980d86301114bbeb5022a0afb28.
Use each actual51 F32 packed head/gamma, source final residual h24P already
captured, and the same24 DEV cases/4386 full-V labels per head. F64 formula:

    z = retained h24P at fixed label positions
    a = z / sqrt(mean(z*z)+1e-5) * actual51 final_norm
    logits = a @ actual51 head.T

Exactly8772 new label-head contractions. All65537 logits per label retained
losslessly,4,599,124,512B. No decoder fitting, grid sweep, source/student history,
native execution, optimizer update, GPU, RESERVED or new labels.

## Results, same case weighting as the actual native endpoint

| Head | Native case KL | Injected case KL | Injected/native | Native disagreement | Injected disagreement |
|---|---:|---:|---:|---:|---:|
| A51 | 6.519307269 | 13.567111869 | 2.081066486 | 89.789302% | 99.934896% |
| B51 | 6.516114953 | 13.567607163 | 2.082162034 | 89.681414% | 99.934896% |

Label-weighted injected KL13.644303965/13.644764164; disagreement99.908801% for
both. Case donor entropy0.516811926/label0.522161448; uniform distribution KL
10.573558221/10.568208699, respectively. Both injected means are worse than
uniform. This is full distribution loss, not accuracy inferred from entropy.
All12 injected domains have case KL>11.83 and disagreement>=99.41%; all4 frozen
absolute/domain screens FAIL. A domain case KL range11.83657156..17.57968772;
B11.83677397..17.57819670. Full case/label/domain values remain in the raw result.

Both retain>=80% native KL and fail absolute/domain screens; both fail the
>=50% KL and>=20% disagreement improvement branch. The prospectively fixed
decision is TARGET_DECODER_COUPLING. No criteria changed after observation.

## Numerical, input and resource qualification

Worker stable sum-exp versus independent logaddexp maximum absolute delta
7.698730542e-12. Stored-only auditor recomputed all8772 metrics from saved F64
scores: maximum absolute KL/entropy/uniform delta7.840839089e-12 <=1e-10;
aggregate delta<=9.238e-14. Argmax disagreements, cases/labels/domains, both
decision flags, resource assertions and complete input/output hashes qualify.

Result SHA86c0928326ea6ddf9b1d1799756a20dfe0286f916be4b4be439b0d49571bb797;
audit SHA1d3c8f48b26d5256a3b6038ab16a1caf7da96bd328b29de641be9b1419266458.
Held contraction150.531s (worker141.031s); OS worker246,108,160B plus
launcher29,294,592B=275,402,752B, within900s/4GiB/5GiB output/log8MiB caps.
Independent held audit52.922s; worker209,055,744B plus launcher27,541,504B=
236,597,248B, within600s/4GiB/log8MiB/output2MiB caps. Binding preparation,
commit and separate stored audit are outside contraction-family time.

Session37053/launcher20132/worker4068 and audit34538/launcher21656/worker21452
are TERMINAL0/gone. First launch rejected before worker creation due foreign
uploader3116/18976; [rejection](original_latent_readout_attribution_prelaunch_rejection_20261010.json)
preserved. Uploader exited naturally; identical frozen launch succeeded. This
was not a failed candidate contraction or model replay. Principal Python,
NumPy/BLAS/psutil and peak-memory helper dependencies were bound prospectively;
no retroactive full-runtime closure of older experiments.

## What this changes toward the conversion pipeline

Even supplying the exact retained projected donor target does not make either
current learned decoder sufficient. The fixed-coordinate auxiliary target is
not currently an adequate readable endpoint. Prioritize a *paired* target,
normalization and decoder before another history-recovery dose. This selects
the declared coupling branch; it does not isolate discarded directions versus
normalization versus coordinate/head adaptation, which can cancel as shown in
[the algebra](TARGET_READOUT_GEOMETRY_20261010.md).

The actual student still produced a lower KL through its own representation;
that representation remains behaviorally useless at0/16 tasks. Neither result
proves a global D256 ceiling, optimal compact decoder, causal-history capacity
limit or inability of redundancy to approximate missing nonlinear functions.
Do not repair the reported result by fitting on these DEV cases.

Next investigate a source-paired compact output codec using FIT-only geometry,
with explicit RMS compatibility, then unchanged DEV full-V checks before
another original history learner. This is a new uncertainty about source final
readout, not a replay of the closed3072->384 recurrent channel studies.
[Next investigation](PAIRED_OUTPUT_CODEC_NEXT_20261010.md) is prospective.
Useful own-history chatbot AND same-artifact50, useful large n/structured CPU
IDs-mass, actual DRAM and family/scale generality remain missing. First training,
source capture, numerical and audit deadline failures remain recorded.
