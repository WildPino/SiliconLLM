# Mixed384 history: recurrence projection valid, plain decoder FAIL

9 October2026. Goal ACTIVE/INCOMPLETE. Original engine/LUT/ternary endpoint and
actual Adam25 unchanged. One new frozen stored-only family,exit0/resource PASS;
144 algebraic head bases and144 actual projected-history case-sites COMPLETE.
No source-model/optimizer/native/RESERVED/T4 or new reply/label calls.

## Controlled change and available artifact

Previous [fixed-channel selection/output repair](ORIGINAL_FALCON_CHANNEL_TRANSPORT_RESULT_20261009.md)
failed. This changes384 selected channels into384 UNGATED mixtures,8 per each48
source heads,without increasing the carried channel count. Uses ALL48 retained
forced histories/sites0/12/23/every position;24 FIT+24 DEV/12 balanced domains.
State256 and actual source x/B/C/delta/gate/full input generators remain.
No joint state96/P256/depth/FFN representation or compact native implementation.

For each head,scalar A/delta/shared B/C and same Dskip commute with fixed R(64x8):

```text
Z = R^T S
Z_t = exp(A_h delta_h,t) Z_t-1 + delta_h,t (R^T x_h,t) B_t^T
y'_h,t = R^T y_h,t
```

Projected dynamics are exact in real arithmetic;this does not preserve discarded
directions or nonlinear gated output. R comes from top8 eigenvectors of the
equal-FIT-case mean of separately trace-normalized UNGATED y_h^T y_h Grams.
Centered Grams recorded separately. Source SiLU/conv nonlinear feature generators
and gates are not cheaply replaced by that coordinate change.

Source chunk128/F32 SSD run with actual mixed x(F32)/8 channels per head/state256,
zero initial state;source A_log/D read directly from pinned safetensors. No model
load/forward. Source reshape/segment/pad helpers retained,chunk temporary bounded.
R savedF64;actual scan usesF32 R. Reconstruct y_hat=R y' then apply full3072 source
gate/norm/BF16 out_proj. Two diagnostics:original full-source denominator and
recomputed denominator of projected/gated y_hat. Both retain full source gate;
neither is a deployable original-engine decoder. No readout refit in this family.

ALL144 actual projected recurrence outputs are retained as F32(T,48,8),
103,896,576B/22547 positions per3 sites,with exact shapes/extents/hashes. These
are reusable source-local mixed histories,not chatbot checkpoints. Later decoder
fitting can consume them without replaying this projected recurrence.

## Qualified numerical implementation

ALL144 projected scan versus R^T captured source y relative RMS<=3.453259e-7;
recurrent-only (identical Dskip*x removed)<=2.104842e-7. Both PASS predeclared1e-4.
Independent sequential F64 recurrence/full1507 FIT history/eight fixed head/coordinate
pairs atsites0/12/23 errors7.239890e-8/1.122476e-7/2.656141e-7;PASS1e-4.
Every original gate/fullnorm/BF16 output reconstruction error EXACT0,144/144.
144 head-basis orthogonality max3.330670e-15<=1e-10.
This validates this projected operator,not the earlier whole C/GPU numerical gate.
Original all-history numeric FAIL stays.

## Observed decoder/encoder errors

Relative RMS of full source mamba output,24 equally weighted DEV cases per site.
Full source gate always available. Percent errors are not accuracy/token agreement
or retained useful chatbot quality. Reference is source output before the common
nonzero decoder SSM multiplier,which cancels from relative RMS.

| Site | Original denominator mean /worst | Recomputed denominator mean /worst | Ungated y reconstruction mean /worst |
|---|---:|---:|---:|
| 0 | 56.9276% /63.0918% | 58.7470% /66.2097% | 33.7061% /37.0810% |
| 12 | 84.6118% /91.2898% | 84.8222% /91.8479% | 36.5870% /42.6530% |
| 23 | 69.7354% /74.6826% | 67.3446% /73.4893% | 41.9743% /46.0042% |

Original-denominator centered means59.5431%/84.2142%/76.3043%;label-position
means57.3006%/85.7012%/68.1105%. Recomputed-denominator centered means61.5133%/
84.3154%/74.7208%;label means59.2460%/85.9903%/65.0736%.
FIT original-denominator mean56.6271%/80.0929%/68.0076%;recomputed58.2678%/
80.1986%/65.4939%. All case/domain/label/centered results in raw JSON.

FIT mean per-head normalized UNGATED Gram retention82.3465%/59.2495%/68.6370%;
lowest head62.4507%/39.7729%/45.1409%. Centered mean69.8664%/47.9224%/62.4192%;
lowest49.4879%/33.0975%/41.3456%. These are separately normalized/equal-head FIT
energies;do not identify them with DEV output retention or total case y energy.

Both diagnostics FAIL every-site engineering budget(mean<=10%,caseworst<=20%,
domainmeanworst<=20%,centeredmean<=10%). No encoder/decoder candidate admitted.
Compared with previous repaired48x8 columns/full denominator76.71%/80.27%/47.44%,
raw PCA mixing improves site0 and worsens sites12/23. This is not uniformly better,
and changing basis alone has not solved transfer. Numerical error~1e-7 is far
below the observed output loss;there is no numerical implementation explanation
of these large errors within the qualified assay.

## Actual costs and reproducibility

[Protocol](ORIGINAL_FALCON_MIXED_HISTORY_PROTOCOL_20261009.md),
[1,037-input binding](original_falcon_mixed_history_binding_20261009.json) SHA
`d7c236318cb6d9a57dad5f8886d1cfc1146de90d7e12738e5e34fdc9b07a5015`,
freeze`aa0ab15b5ffc16ccac6a43baa90d70c5912342a0`.
[Code](../../../benchmarks/native_expert_scaling/original_falcon_mixed_history.py),
[raw result](original_falcon_mixed_history_result_20261009.json) SHA
`efd7fb64821a6c46494aaa70a45eb95309f460143a83a2df42961d3a97a49557`,
[held terminal](original_falcon_mixed_history_result_20261009.terminal.json),
[log](original_falcon_mixed_history_result_20261009.worker.log).
Launcher10836/worker27432/creation1791561921.870343,exit0/session36520 CLOSED.
Held family70.531s;worker56.407s. Held worker OS1,530,552,320B+launcher29,388,800B
=1,559,941,120B conservative. GPU allocated1,151,896,576B/reserved1,293,942,784B,
allocator only. Actual namespace294 files114,176,791B includes3 bases/3 metadata/
144 F32 latents/144 metrics;output128MiB cap PASS. Pre-observation allowance
increased from NEXT planning96MiB to retain103.9MB F32 latents plus~10MB statistics.
Other caps1800s/reserve90/OS6GiB/GPU4/5GiB unchanged/allPASS. No firstfault/retry.
Binder/preparation/commits are additional to held-family seconds.

Pre/post all input bytes matched. Stored-only PowerShell adjudication confirms
all294 output hashes/extents/result-terminal hash,144 balanced records/3 F64
witnesses/103,896,576 latent bytes;aggregate maxdelta2.220447e-16. Direct command
exit0;no new predictions/refits and no separate held adjudication resource record.
Principal runtime binaries bound,not full DLL tree. Original source-capture parent
exit/resource/final identity gaps remain unknown and are not qualified here.

```text
original_falcon_mixed_history.py --launch --binding docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_mixed_history_binding_20261009.json --binding-sha d7c236318cb6d9a57dad5f8886d1cfc1146de90d7e12738e5e34fdc9b07a5015 --freeze aa0ab15b5ffc16ccac6a43baa90d70c5912342a0 --directory results/native_expert_scaling/original_falcon_mixed_history_20261009 --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_mixed_history_result_20261009.json
```

Python312 -I -S -B -X utf8,repo root;full held command/executable in terminal.
Completed namespaces immutable;reuse actual bases/latent fields for subsequent
new-variable work. No replay of these predictions for reconfirmation.

## Decision: read/write-weighted geometry before conditional decoder training

Do not promote raw-y PCA rank8 to a warm core. Source output reads directions
through a time-varying matrix O_h diag(SiLU(g_h,t))*norm_weight_h/denominator_t;
small unweighted y energy does not imply small readout error. The current PCA
objective lacks that metric. Preserving head dynamics is a validated reusable
operation,but the retained function space is still poorly chosen for source output.

Next choose within-head dual read/write bases using FIT y covariance AND that
output-reading metric. The optimal rank-r Cartesian pair-kernel construction is
the same algebra independently verified for state96;it applies to channel64 now.
Actual same-time gating/output and causal dynamics must still be evaluated rather
than inferred from singular tails. Keep an explicit fixed8/head control;compare
a FIT-only rank allocation across48 heads under sum r_h=384/minimum1/head. That
tests equal-rank allocation as well as the readout metric,without more active
channels. No new source/labels/Adam25 dose/T4. New protocol/code/binding still missing.

This is not an information-capacity ceiling for384 learned/conditional functions.
Mixed nonlinear generators and compact input-dependent decode remain unimplemented;
R/W mixing cannot simply be folded before SiLU/conv or through the source gate.
If a read/write-weighted encoder earns further work,price actual conditional
ternary functions/routing/residency and native active work before longer adaptation.
P256/depth/parallel attention/FFN and normalized routing losses remain distinct.
Useful chatbot/same-artifact>=50,useful RAM-driven n/CPU IDs AND mass/physical DRAM,
additional families/scales remain required. [Exact next](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md).
