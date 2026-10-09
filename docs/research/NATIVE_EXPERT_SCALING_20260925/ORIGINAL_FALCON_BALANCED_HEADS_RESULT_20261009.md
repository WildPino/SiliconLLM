# Balanced384 channel histories: valid operator, failed reconstruction

9 October2026. Goal ACTIVE/INCOMPLETE. Execution COMPLETE, local reconstruction
budget FAIL. Original `engine.c`, all previous workers/results and Adam25 unchanged.
No model forwards, source labels, optimizer, native, RESERVED or T4 calls.

[Frozen protocol](ORIGINAL_FALCON_BALANCED_HEADS_PROTOCOL_20261009.md),
[worker](../../../benchmarks/native_expert_scaling/original_falcon_balanced_heads.py),
[binding](original_falcon_balanced_heads_binding_20261009.json),
[raw measurements](original_falcon_balanced_heads_result_20261009.json),
[held terminal](original_falcon_balanced_heads_result_20261009.terminal.json),
[stored adjudication](original_falcon_balanced_heads_stored_adjudication_20261009.json).
Freeze `8bcd42e1763155c16c15553ebf026bf09b158470`;1042 bound inputs, binding SHA
`325e4cfe6b9795354bde92c93d6a30080a423c0e182c829c4b146c61051cd40c`.
Raw result SHA `37ccf69de18996e0b1aa514e40bba34677f7d33436d705dcc0501c09e8a535c7`.

## What was actually changed

ALL24 FIT and24 DEV retained full histories/sites0/12/23,22547 positions.
One equal-case unnormalized F64 write Gram GY and read Gram GM per48 heads/site.
GM uses actual full source SiLU gate, norm weight/denominator and out_proj. It
weights real-arithmetic reading before source BF16 feature rounding. No DEV fit.

For sqrt(GM)sqrt(GY)=U Sigma Q^T, V=sqrt(GY)Q_r Sigma^-1/2 and
W=sqrt(GM)U_r Sigma^-1/2. Source scalar head dynamics commute with W^T. Actual
F32 x_h W_h enters source SSD/state256, then yhat_h=V_h y'_h. Two layouts:
fixed8/head and adaptive384 with min1/max64/head. All48 timescales retained.
Adaptive allocation uses336 largest remaining squared singular values after one
per head; independent global marginal sorting and prefix verification agreed.
Actual packed(T,384) scans, without padded3072 recurrence. Full source nonlinear
generators and3072 gate/readout still used; this is not a native compact model.

The optimized objective is separable **independent Cartesian-time pair** error.
It does not optimize same-time correlations, cross-head cancellation, causal
decay, BF16 rounding or chatbot loss. Optimum of this proxy is not an optimum
over all384-channel recurrent models or conditional functions.

## Numerical qualification

All144 source output baselines exactly0; both layouts qualify on every case.
144 FIT head Gram/SVD fits,288 selected basis audits; no negative Gram eigenvalues
were clipped. Independent NumPy F64 weighted-kernel/tail audits:

| Check | Actual maximum | Frozen limit |
|---|---:|---:|
| W^T V entrywise error | 1.9361249e-14 | 1e-8 |
| Independent truncated weighted kernel relative error | 1.6330804e-12 | 1e-8 |
| Residual tail energy delta / total kernel energy | 5.3017531e-16 | 1e-10 |
| Fixed8 projected scan relative RMS | 4.0043124e-7 | 1e-4 |
| Adaptive384 projected scan relative RMS | 7.2501474e-7 | 1e-4 |
| Fixed8 recurrent-only relative RMS | 1.9618457e-7 | 1e-4 |
| Adaptive384 recurrent-only relative RMS | 4.3298796e-7 | 1e-4 |

Six independent sequential F64 witnesses over the full1507-position FIT history,
eight fixed heads and actual mixed F32 x: fixed/adaptive site0 7.3821739e-8/
7.3820336e-8;site12 1.0327495e-7/1.0290581e-7;site23 2.2592283e-7/2.2899534e-7.
All <=1e-4. This qualifies the projected dynamics, not discarded information.
Maximum V/W spectral norms1991.4198702/4.9572087; scaling is imbalanced in the
physical units of write/read operands. Any lower-precision export needs its own
error check; no balanced trit/quantized recurrent implementation was tested.

## Downstream measurements

Actual FULL source gate and BF16 source output rounding for both modes. RMS is
relative Frobenius error over one history; table reports equal-case DEV mean/worst.

| Site | Layout | Source denominator mean / worst | Reconstructed denominator mean / worst |
|---|---|---:|---:|
| 0 | fixed8 | .544858613206 / .612919699992 | .571220742069 / .650216810912 |
| 0 | adaptive384 | .544530645651 / .612094013311 | .568894622403 / .647148716923 |
| 12 | fixed8 | .797085006435 / .876104116939 | .841039290355 / .950173825623 |
| 12 | adaptive384 | .786817418144 / .869454193932 | .832677114872 / .953135263977 |
| 23 | fixed8 | .538433069079 / .660696041567 | .557973208905 / .691958521432 |
| 23 | adaptive384 | .537223442539 / .662377661857 | .557104875733 / .693272028945 |

| Site/layout | Source denominator centered / labels DEV mean | FIT full mean | DEV ungated mean / worst |
|---|---:|---:|---:|
| 0/fixed8 | .571515159650 / .554386006543 | .536124133913 | .396484613475 / .430729732124 |
| 0/adaptive384 | .570922461643 / .551894718252 | .535877065952 | .392926341681 / .426784738603 |
| 12/fixed8 | .796106896723 / .794731666480 | .746007257407 | .565121738893 / .875884412649 |
| 12/adaptive384 | .785741709962 / .781160954079 | .738822667530 | .577109420878 / .960344869218 |
| 23/fixed8 | .594232058237 / .514959556755 | .509495145176 | .610141206572 / .657674766294 |
| 23/adaptive384 | .592660443504 / .513595861750 | .507998937501 | .607359232430 / .653497479075 |

Reconstructed denominator centered DEV means fixed/adaptive:0 .599148892312/
.596586022474;12 .835645560631/.828006695736;23 .618449099789/.617328512272.
Corresponding label means:0 .580276827416/.576526720485;12 .830742944751/
.818714834955;23 .529728389857/.529244798278. Complete FIT/DEV/domain means,
worst cases and all144 case-site packets are retained in raw measurements.
Every layout/mode/site fails the frozen10%mean/20%case/20%domain/10%centered gates.
No provisional decoder selected; quality/speed/deployable admission false.

Compared with immutable raw-y PCA/source denominator, adaptive384 improves
56.9276% ->54.4531%,84.6118% ->78.6817%,69.7354% ->53.7223% at sites0/12/23.
Read weighting helps; unequal ranks add only .033/1.027/.121 percentage points
over fixed8 at these sites. Adaptive improves the declared separable pair proxy:

| Site | Fixed8 tail energy | Adaptive384 tail energy | Adaptive rank range / groups |
|---|---:|---:|---:|
| 0 | 12888.8814568743 | 12596.6882729294 | 2..13 / 11 |
| 12 | 8209.47344447938 | 7937.58046333115 | 4..18 / 14 |
| 23 | 217663.827808020 | 213979.550689851 | 4..15 / 9 |

Proxy tail energies use raw physical units and cannot be compared as chatbot
scores or relative same-time output error. Ungated reconstruction gets worse
than raw-y PCA in some sites while gated output improves: different metrics read
different directions. No overall information-capacity ceiling follows.

## Held execution and retained fields

Launcher15796/worker6740,creation_time1791563465.49793,exit0;session73310 CLOSED.
Held family94.78100000001723s;worker80.1720000000205s. Held worker OS1,520,574,464B
plus launcher29,765,632B = conservative1,550,340,096B. GPU allocator peaks
1,227,675,136 allocated/1,447,034,880 reservedB; not whole-device residency.
1800s/reserve90/OS6GiB/GPU4/5GiB/output256MiB/log4MiB all PASS. No fault/retry.
Binder/preparation/commits/stored adjudication are additional to held-family time.

438 namespace files248,797,448B:3 NPZ/3 basis JSON/144 metrics/288 actual packed
F32 histories,207,793,152B latent payload. Head offsets in basis packets define
the coordinate order. Full input/output bytes pre/post checked. Stored PowerShell
adjudication independently verifies438 hashes/extents,144 balanced records,
288 basis audits/six scalar witnesses/all latent bytes,case/domain aggregate
maxdelta2.220446049250313e-16. No new prediction/refit or separately held
adjudication timing/resource qualification. Source capture parent runtime gaps
remain unknown; principal binaries bound, not a full DLL tree.

CUDA-synchronized scan-only seconds over all48 histories per site, fixed/adaptive:
0 3.043/4.601;12 2.831/4.784;23 2.871/4.206. Variable-rank groups repeat shared
B/C kernel work. These timings exclude readout/IO/basis construction and are
diagnostic GPU timings, not a native latency or accepted token/s claim.

## Decision toward the engine pipeline

Close the tested **fixed linear channel projection + source gate** route at384
channels. No new rank grid, PCA replay or extra linear-map repair is selected.
Extra stored FFN experts cannot reconstruct a history component already discarded
unless the compact state/input representation supplies the missing information.
This is a causal information requirement, not evidence that all small learned
states fail. Redundancy in conditional functions can encode multiple local maps;
its adequacy and runtime selection remain to be demonstrated.

Next measure one complete original-kernel deployment envelope before choosing a
new learned state. Compare actual Adam25 DN512/DT16 with a separately exported,
algebraically function-preserving duplicated-core DN1024/DT48 cost fixture,
sameD256/N96/L6/n1152/k8/fullV65537/bank/LUT/AQ63/dReLU. This adds no knowledge
and earns no quality admission. Measure full batch1 native core/router/LUT/head
and finite-precision preservation, not another cache-only LUT score. Freeze exact
code/fixtures/histories/criteria/caps first. No wider native backend exists yet.

If a wider compact core is affordable, its functions must be **learned jointly**:
recurrent input/transition, gate/decode, compact residual mapping and conditional
ternary bank. The present W/V maps remain failed diagnostic controls, not warm
initialization certificates. Use retained source state96 evidence as a guide,
not a compositional guarantee; lost24->6 depth/attention/FFN remain open. A T4
allocation requires measured training residency/throughput and an explicit finite
recovery pilot with checkpoint/plateau rules. No allocation now.

Useful fresh chatbot AND >=50 accepted IDs/s on the same artifact,useful RAM-driven
n/CPU IDs and normalized mass/physical DRAM,additional donors/scales still missing.
[Current resumption](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md).
