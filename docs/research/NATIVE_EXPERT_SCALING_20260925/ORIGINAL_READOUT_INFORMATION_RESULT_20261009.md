# Original readout: feasible distributions versus actual history features

9 October 2026. Goal ACTIVE/INCOMPLETE. Mathematical probe COMPLETE; neither
optimizer reaches the all72-label1e-5 gap gate. Useful chatbot/native parity/rate
remain unqualified. Prior original native numerical FAIL is unchanged.

## Reproducible record

Code/prospective protocol/binding freeze `40a48e7e8b2a0ead4626a54b9be26279a9e72a26`:
[protocol](ORIGINAL_READOUT_INFORMATION_PROTOCOL_20261009.md),
[worker/binder/held launcher](../../../benchmarks/native_expert_scaling/original_readout_information.py),
[45-input binding](original_readout_information_binding_20261009.json), SHA
`9d59cb866cdfd5a8447b11c2d561f44b15c44456f387181fbacea6d35ae2e4f5`.
[Raw result](original_readout_information_result_20261009.json), SHA
`26f590955aada522ee22d7ab1be2e3d1fdde21700d85c2c43908bfa31c5f6b8b`;
[held terminal](original_readout_information_result_20261009.terminal.json),
[worker log](original_readout_information_result_20261009.worker.log).

Exact isolated Python is `C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe`,
flags `-I -S -B -X utf8`. From repository root:

```text
python.exe -I -S -B -X utf8 benchmarks/native_expert_scaling/original_readout_information.py --bind --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_readout_information_binding_20261009.json
python.exe -I -S -B -X utf8 benchmarks/native_expert_scaling/original_readout_information.py --launch --binding docs/research/NATIVE_EXPERT_SCALING_20260925/original_readout_information_binding_20261009.json --binding-sha 9d59cb866cdfd5a8447b11c2d561f44b15c44456f387181fbacea6d35ae2e4f5 --freeze 40a48e7e8b2a0ead4626a54b9be26279a9e72a26 --directory results/native_expert_scaling/original_readout_information_20261009 --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_readout_information_result_20261009.json
```

Commands name already completed immutable namespaces; do not replay them there.
Binder shell4.120s includes syntax parsing; preparation is separate from the family.
No donor/model/native forward, autograd/Adam update or RESERVED/T4 use.

## What was measured

Actual source-informed original-shaped Adam1 packed head/final RMS gamma;
D256/V65537. Each of24 FIT cases contributes first/middle/last label, selected
by metadata:72 distributions/12 domains, full vocabulary, saved BF16 source
logits decoded exactly. For each label optimize arbitrary u in ||u||<=16 under
fixed A=head*diag(gamma). This is the closure of original RMSNorm's feature set;
it is optimistic because history functions are unrestricted and boundary16 is
only approached by finite original pre-normalization features.

F64 projected descent/full objective/gradient,512 updates/control. Tangent lower
bounds and best feasible upper bounds bracket the mathematical optimum; computed
in F64, not interval-certified. The uniform predictor has mean KL10.6265553981;
teacher entropy mean0.4638147495 nat on THESE72 labels.

| Fixed readout | Mean lower | Mean feasible upper | Mean gap | Max gap | Labels gap<=1e-5 | Optimistic argmax disagreement |
|---|---:|---:|---:|---:|---:|---:|
| Current A | 0.6598627634 | 0.6617438241 | 0.0018810608 | 0.0287002709 | 40/72 | 1/72 |
| sqrt8*A | 0.0425939947 | 0.0594680835 | 0.0168740888 | 0.4076895065 | 12/72 | 0/72 |

Worst feasible per-label KL current4.5053573521/scaled1.1119959539. Not all labels
are well represented; neither arm's optimum is fully resolved. In particular,
scaled0.05947 is a feasible mean upper, not a proved minimum. Nonetheless the
brackets prove, within the stated numerical scope, a substantial improvement
between the two fixed images:0.05947 <.75*0.65986. Current feasible mean<=1 is
also true; current lower mean>1 is false. These are all three frozen flags.

The three already available native rows allow a matched comparison; other23
cases have no native outputs at this actual state, and were not replayed:

| Longest FIT label index/position | Actual native KL | Uniform KL | Current feasible KL | Scaled feasible KL |
|---|---:|---:|---:|---:|
| 0/1251 | 8.787651809 | 9.510230494 | 0.113600519 | 0.057466021 |
| 128/1379 | 10.352866651 | 10.362519201 | 0.011812155 | 0.011812155 |
| 255/1506 | 10.322008777 | 8.213534934 | 0.874715632 | 0.407689506 |

Thus the current fixed readout does not impose anything close to the actual
observed loss on these three distributions. The inability of the actual history
functions to supply good features is the primary next uncertainty. This does
NOT prove such functions are realizable, learnable or generalizable with this
core/experts. The head is trainable; no universal D256 capacity bound follows.

## Actual-direction scale control: greater amplitude currently hurts

Adaptive, explicitly post-hoc to the result above; separate code/protocol freeze
`419733b` before execution:
[protocol](ORIGINAL_READOUT_STORED_SCALE_PROTOCOL_20261009.md),
[stored-only code](../../../benchmarks/native_expert_scaling/original_readout_stored_scale.py),
[result](original_readout_stored_scale_result_20261009.json), SHA
`2443ff92e2f42e11d21acd1a528bef698d9acc3ab0c8dc8909f7b09730e159a4`.

```text
python.exe -I -S -B -X utf8 benchmarks/native_expert_scaling/original_readout_stored_scale.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_readout_stored_scale_result_20261009.json
```

Use ALL256 cached labels of the actual1507-history trajectory; F64 scaling of
the byte-matched saved native F32 logits, no model/native/teacher/GPU calls.

| Predictor on the SAME256 labels | Full distribution mean KL |
|---|---:|
| Uniform | 9.8809092324 |
| Actual native Adam1 | 10.5807693345 |
| Actual fixed-direction logits times sqrt8 | 22.9471990109 |

All256 native argmax predictions disagree with source. Positive scaling cannot
change those IDs. This is not a re-exported F32 gamma variant; its exact native
rounding is unmeasured. It does show that amplifying current directions is a
poor proposed fix on this trajectory. The original11.60 ->10.58 update remains
an observed loss decrease; it is NOT useful knowledge preservation and has not
yet beaten even uniform on this full packet. Some individual rows beat uniform,
so the mean observation does not prove that every feature carries no information.

## Resources and preserved state

Readout worker PID17500/creation time1791548353.1002092/exit0; launcher26940.
Session14841 CLOSED. Family196.078s/worker190.640s <=600s. Held worker OS
1,124,139,008B plus launcher29,347,840B =1,153,486,848B <=4GiB. PyTorch GPU
allocated550,362,112B/reserved654,311,424B <=3/4GiB; allocator scope, not total
driver memory. RTX3060/Torch2.6.0+cu124/NumPy2.4.6/psutil7.2.2.

Namespace `results/native_expert_scaling/original_readout_information_20261009`
contains only `solutions.npz`,298,566B/SHA
`e431feafa05ce4b2a6495245c680ebd35a3fba4d7468e0e96349f8cf22158644`.
Per-label features/bounds/traces are retained;45 inputs hash-verified pre/post.
Stored scale audit reports1.360s; surrounding shell5.072s includes its scoped
freeze commit, not a held runtime family. No failures/retries/discarded solutions.
Original engine, foreign tracked files, actual source Adam1/RNG and packed
artifact remain byte-preserved. No new checkpoint/update or native parity/rate claim.

## Finite recovery inventory and decision

[Exact metadata/byte inventory](original_falcon_recovery_inventory_20261009.json),
SHA `b9e7de41671a639a4ef963eeccc3ed5d66dd38f1f3eb6e5ea0aa81028695ded8`,
[code](../../../benchmarks/native_expert_scaling/original_falcon_recovery_inventory.py)
frozen36595a3. Actual checkpoint8,614,643,514B and all source packets hash-checked
pre/post; no tensor/model/teacher/native/GPU calls. Result's10.609s timer is
captured before its second input hash pass; it is not full-family cost. A harmless
PowerShell formatting parse error occurred before this inventory, no work/state
was changed by it. Inventory session58554 CLOSED/exit0.

| Split | Cases/domains | Entire history tokens | Max history | Labels | Saved teacher bytes | Full native output bytes per state |
|---|---:|---:|---:|---:|---:|---:|
| FIT | 24/12 | 11,148 | 1,507 | 4,422 | 579,609,228 | 2,922,425,904 |
| DEV | 24/12 | 11,399 | 1,488 | 4,386 | 574,890,564 | 2,988,225,052 |

Proposed one24-case whole FIT pass starts actual Adam1, ends25, with durable13/25.
Complete native before48 can adopt the existing longest-FIT output, so47 new
before+48 new after cases. Full outputs/routes/two actual-size snapshots/final
packed file give29,179,775,232B subtotal, excluding small queries/witnesses/logs/
metrics. Free disk at inventory is in the raw record. Proposed caps3600s/reserve
300s/OS32GiB/GPU10/11GiB/output40GiB are a conservative future protocol candidate,
not executed or qualified resource observations.

Reference-history linear estimates: FIT forward/backward896.478s and95 native
history computations340.645s. These omit optimizer/gradient inspection/exports/
snapshots/checks/hashing/load/launch overhead and unknown routing unions; union
counts do NOT necessarily scale linearly with history. Prior snapshot I/O is
material. No claim that the recovery necessarily fits these forecasts.

**Selected:** retain current gamma/head/core/banks and actual Adam1; implement,
freeze and run one finite24-case coverage continuation, measuring native before/
after48. Do not apply sqrt8 to current predictions or spend a month on the
fresh-core proxy now. If broader native DEV quality does not recover, next work
must supply new internal history/state information or justify a geometry change;
do not infer a capacity ceiling from one finite dose. This mathematical probe is
complete; no extra oracle iterations/scales are selected. [Exact next](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md).
