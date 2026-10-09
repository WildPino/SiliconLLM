# Fixed-channel transfer and closed-form output repair: complete, representation FAIL

9 October2026. Goal ACTIVE/INCOMPLETE. Original engine/LUT/ternary endpoint and
actual Adam25 unchanged. Two new separately frozen stored-only families, both
exit0/resource PASS/numerically valid, both local engineering-budget FAIL.
No source-model/optimizer/native/RESERVED/T4/new replies or label calls.

## New uncertainty and controlled variables

[Rank96 state transport](ORIGINAL_FALCON_RECURRENT_PROJECTION_RESULT_20261009.md)
keeps all3072 source x/gate channels. It cannot justify eliminating channels to
fit an original-core width. These assays reuse all48 internal full histories/
sites0/12/23,24 FIT+24 DEV/12 domains,22547 positions. Actual source state256,
gates/generators/hidden inputs remain. No combined state96/P256/depth conversion.

Exactly384 retained channels:8 per each48 heads versus6 complete64-channel heads.
FIT-only equal-case greedy output residual reduction accounts for correlations/
cancellation of actual gated source output contributions. It is a fixed-unit
subset heuristic, not globally optimal selection or a capacity lower bound.
Full normalized BF16 source features/references and pinned BF16 out_proj used.
All384 constrained channel steps and6 whole-head steps replay exactly from durable
F64 sufficient-statistic rows/groups; original selected IDs/calibration preserved.

First assay restricts source out_proj columns. Modes:actual full-source denominator
(optimistic diagnostic);subset384 RMS+FIT nonnegative scalar;FIT constant denominator
+FIT nonnegative scalar. Alphas applied F32 AFTER BF16 out_proj. No norm extension
implemented in original engine, no exact folded-weight implementation claimed.

Second assay freezes SAME channel IDs/constant denominator, changes ONLY the
output estimator:18 FIT-only linear ridge fits (3sites x2layouts x3modes),
G=equalcase X^T X/T,C=equalcase X^T Y/T,lambda=1e-6 tr(G)/384,
Beta=(G+lambda I)^(-1)C via F64 CPU eig. Moments F64 CUDA,TF32off. Save actual
G/C/eigens/F32 coefficients; prediction is X(F32)*Beta(F32). Previous alphas
not applied. No channel reselection/grid/Adam or training-state continuation.
This tests recoverable omitted correlations with a fixed linear estimator;
it is not an arbitrary free-feature oracle or globally optimum representation.

## Numerical and predeclared engineering criteria

ALL144 baseline gate/full-source RMSNorm/BF16 out_proj errors exactly0 in EACH
family; reconstruction PASS<=1e-4. First greedy moment objective versus direct
F32 prediction discrepancy/target energy<=1.583431e-6 (criterion1e-4).
Second18 fitted normal equations:F64 residual<=1.031188e-14 (criterion1e-8),
exported F32 residual<=4.264129e-8 (criterion1e-4). Actual FIT loss quadratic
versus direct F32 outputs discrepancy/target energy<=2.493311e-10 (criterion1e-4).
Largest regularized condition1.829174e8; coefficients finite. These numerical
checks qualify the reported experiment, not whole C parity or future precision.

Each deployable mode needed at EVERY assayed site:DEV output mean<=10%,case
worst<=20%,domain mean worst<=20%,centered mean<=10%. All four layout/mode
combinations FAIL in BOTH assays. No provisional layout selected. Full-source
denominator diagnostic cannot admit a deployable layout even if it passes.
Thresholds unchanged; every case/domain/label/centered result retained in raw JSON.

## Observed DEV mean output errors

Percent relative RMS of complete source mamba output;equal-case24 DEV mean per
site. Errors are not token disagreement, accuracy or retained chatbot quality.

| Layout/site | Old columns/full denominator | Refit/full denominator | Refit/subset RMS | Refit/constant RMS |
|---|---:|---:|---:|---:|
| 48x8 /0 | 80.4725% | 76.7103% | 78.0140% | 77.8773% |
| 48x8 /12 | 81.7383% | 80.2699% | 83.8087% | 82.5367% |
| 48x8 /23 | 53.9158% | 47.4408% | 48.8525% | 49.5649% |
| 6x64 /0 | 88.0420% | 83.4456% | 83.8019% | 83.4031% |
| 6x64 /12 | 89.3261% | 89.1452% | 91.5440% | 91.4670% |
| 6x64 /23 | 75.1710% | 62.7874% | 64.3354% | 63.9840% |

Refit/full-denominator worst48x8:85.7975%/87.9612%/62.8134%;6x64:
90.7541%/94.8418%/76.5545%. Refit/subset RMS centered means48x8:
82.3984%/85.6517%/54.5977%;label-position means81.1969%/83.9680%/47.6818%.
Mean removal and supervised-position restriction do not erase the error.
Refit/full-denominator FIT means48x8:64.2264%/64.6811%/42.5280%;6x64:
70.1694%/74.5203%/57.3236%. Refit improves many means but remains far from budget.
This finite dataset/selection/ridge result is not a general impossibility theorem.

First calibrated-subset means48x8:80.5929%/85.0522%/55.0054%;6x64:
87.9610%/92.6193%/76.7441%. Implied full-denominator RMS error48x8:
18.3891%/39.8247%/34.0542%;6x64:30.2872%/68.9133%/47.4448%.
Normalization adds another measured omission, but the large full-denominator
errors already occur with its actual value available. Do not attribute all loss
to denominator approximation or combine independent state/channel errors as measured.

## Actual resources, receipts and immutable outputs

First [protocol](ORIGINAL_FALCON_CHANNEL_TRANSPORT_PROTOCOL_20261009.md),
[455-input binding](original_falcon_channel_transport_binding_20261009.json) SHA
`635c73794738f9b381ffc00aeeccea9ed6959b831b5ae27bd6a8aa60c55fd59b`, freeze
`6f2a8438b14a37065e83c918c164ea0f8b562473`.
[Raw result](original_falcon_channel_transport_result_20261009.json) SHA
`f936551337e861ec24d21d90c86e814c391eb4013faa2506372cfaaaed2f3692`,
[terminal](original_falcon_channel_transport_result_20261009.terminal.json),
[log](original_falcon_channel_transport_result_20261009.worker.log).
Launcher22740/worker16816/creation1791560202.8060725,exit0/session55850 CLOSED.
Held family56.750s/worker44.281s. Held worker OS1,648,082,944B+launcher29,708,288B
=1,677,791,232B conservative;GPU allocated281,280,512B/reserved402,653,184B,
allocator only. Namespace150 files28,908,968B. All original1800s/reserve90/OS6GiB/
GPU4/5GiB/output32MiB gates PASS. No first fault/retry. 144 complete comparisons.

Refit [protocol](ORIGINAL_FALCON_CHANNEL_READOUT_REFIT_PROTOCOL_20261009.md),
[463-input binding](original_falcon_channel_readout_refit_binding_20261009.json) SHA
`aca40ea03a72f085f464f2af4e4100e13f55c0d8f66844c0d8959f7e07bd3087`, freeze
`f219eeacc7f234a883823b7ae9e40d1e38447fda`.
[Raw refit](original_falcon_channel_readout_refit_result_20261009.json) SHA
`10bb2ccd9364782bdb57115c4422bc899987c8e19cc5a58eda660a263e8e1eb2`,
[terminal](original_falcon_channel_readout_refit_result_20261009.terminal.json),
[log](original_falcon_channel_readout_refit_result_20261009.worker.log).
Launcher2168/worker28400/creation1791560755.9214897,exit0/session68105 CLOSED.
Held family74.750s/worker62.031s. Held worker OS1,510,854,656B+launcher28,790,784B
=1,539,645,440B conservative;GPU allocated238,544,896B/reserved331,350,016B,
allocator only. Namespace150 files191,450,515B;18 fitted coefficient sets/144
complete comparisons. All900s/reserve90/OS6GiB/GPU4/5GiB/output256MiB gates PASS.
Larger predeclared output allowance stores actual G/C/coefficients/eigens;active
channel budget unchanged. No first fault/retry. No original source/candidate changes.

Combined held families131.500s;binding/preparation and commits are additional.
Combined300 files220,359,483B excludes DOC receipts. All input hashes checked
before/after;all namespace hashes captured in held terminals. Principal runtime
binaries bound,not full DLL tree. Earlier source-capture runtime gaps stay.
Raw schema reused by refit with explicit experiment=CHANNEL_READOUT_REFIT_V1;
optimizer_updates0 does NOT mean no fit:18 closed-form fits explicitly counted.

Final stored-only PowerShell adjudication recomputes all case/split/site/layout/
mode full/centered/label means/worst:maximum absolute delta4.440892e-16. Both
144-record inventories/12 balanced domains and all300 output-file extents/raw
hashes match held terminals;raw-result hashes match too. Protected engine and
three foreign-file hashes unchanged,no owned Python/compiler/native process live.
Direct commands exited0;no new predictions/fits and no separate held adjudication
family resource record.295 relative documentation links checked,no broken links.

## Decision: change the encoded functions before further dose

Reject the tested fixed-column384-channel layouts as a warm initializer under
this declared local budget. Fitted linear readout is an improvement, not a rescue.
Raw channel pruning differs from representing mixed functions in384 dimensions;
other bases/selections/nonlinear conditional decoders remain untested.

Next finite stored-only investigation:per-head linear mixing of UNGATED recurrent
channels. For each source head,scalar A/delta/shared B/C and same Dskip imply
exact real-arithmetic commutation with a fixed R_h(64x8):

```text
Z_h = R_h^T S_h
Z_h,t = decay_h,t Z_h,t-1 + delta_h,t (R_h^T x_h,t) B_t^T
y'_h,t = R_h^T y_h,t
```

Fit head bases on FIT only;evaluate reconstruction after the actual source gate
and output on DEV/full/centered/tails. This tests mixed history observability;
current gate multiplication does not commute with R_h, so full-gate reconstruction
is an explicit diagnostic,not a cheap original-engine decoder. Before implementing
a warm target,measure whether a conditional ternary bank can represent the
input-dependent readout and mixed nonlinear generators at the fixed active budget.
Likewise R_h^T SiLU(conv(...)) cannot be folded exactly before SiLU into one linear
projection. The proposed992-channel construction with copied x columns is no
longer selected. Keep dual96 as a separately measured state-coordinate map.

P256/depth/parallel attention/FFN/routing losses remain. Original-engine useful
chatbot/same-artifact>=50,useful RAM-driven n/CPU IDs AND mass/physical DRAM and
family/scale generality still missing. No blind Adam25 extension or T4 allocation;
long adaptation should train a justified representation,not this failed copy map.
[Operational next](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md) updated accordingly.

## Reproduction

Python312 -I -S -B -X utf8,repo root. Code [channel assay](../../../benchmarks/native_expert_scaling/original_falcon_channel_transport.py),
[readout repair](../../../benchmarks/native_expert_scaling/original_falcon_channel_readout_refit.py).
Completed namespaces immutable;bindings carry exact raw inputs. Held command
arrays/full executable paths are in terminals. Actual invocations:

```text
original_falcon_channel_transport.py --launch --binding docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_channel_transport_binding_20261009.json --binding-sha 635c73794738f9b381ffc00aeeccea9ed6959b831b5ae27bd6a8aa60c55fd59b --freeze 6f2a8438b14a37065e83c918c164ea0f8b562473 --directory results/native_expert_scaling/original_falcon_channel_transport_20261009 --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_channel_transport_result_20261009.json
original_falcon_channel_readout_refit.py --launch --binding docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_channel_readout_refit_binding_20261009.json --binding-sha aca40ea03a72f085f464f2af4e4100e13f55c0d8f66844c0d8959f7e07bd3087 --freeze f219eeacc7f234a883823b7ae9e40d1e38447fda --directory results/native_expert_scaling/original_falcon_channel_readout_refit_20261009 --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_channel_readout_refit_result_20261009.json
```

No replay of completed predictions for reconfirmation;new variables/new namespaces
need their own frozen protocol/binding/criteria. Original failures remain in reports.
