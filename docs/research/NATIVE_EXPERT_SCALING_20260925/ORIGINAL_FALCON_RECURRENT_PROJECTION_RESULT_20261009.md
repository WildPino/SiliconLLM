# Rank96 recurrent transport: complete local measurements, compact chatbot still open

9 October2026. Goal ACTIVE/INCOMPLETE. All projection and audit processes terminal;
no T4 allocation. Original `benchmarks/phase60/engine.c`, LUT/ternary endpoint and
actual source-informed Adam25 unchanged. This implements a source-state analysis
for the converter; it is not a donor-sized runtime adopted as the target.

## Question, data and scope

Can a rank96 recurrent state preserve more source output when its read/write
bases differ, and what generator representation cost does this impose on the
original engine? Reuse [all48 adopted internal histories](ORIGINAL_FALCON_RECURRENT_CAPTURE_RESULT_20261009.md)
of useful pinned Falcon-H1-1.5B-Instruct, revision80ebc50d/weights1fb78851.
24 FIT/24 DEV,12 domains/two cases each split;22,547 full history positions,
FIT11,148/DEV11,399, longest1507. ALL24 sites supply B/C for FIT bases.
Sites0/12/23 supply full propagated comparisons on ALL48 histories/positions.
No source forward, optimizer update, native call or new teacher reply here.

**All source3072 x/gate channels,48 heads/delta functions, Dskip, post-gate full
RMSNorm and BF16 out_proj remain. Only state256->96 is changed.** These local
outputs precede the common decoder SSM multiplier; relative RMS is invariant
under that common nonzero scalar. Source nonlinear generators are not replaced
by original-core generators in this assay. D2048->P256, channel/head elimination,
attention,24->6 composition and whole-chatbot quality are not qualified.

Original vanished source-family exit/resources/final parameter check remain
unknown; all48 data were separately adopted with exact finite/byte witnesses.
The [capture result](ORIGINAL_FALCON_RECURRENT_CAPTURE_RESULT_20261009.md) explains
45 completed originals/one partial/new3 completions and resource-scope limits.
This successful stored projection does not fill the original runtime gaps.

## Actual FIT bases and algebra

Each case contributes its separately trace-normalized B/C Gram with equal weight:
G_B=mean(B^T B/tr(B^T B)), likewise G_C. FIT only, all histories, all24 sites.
Joint=(G_B+G_C)/2. Centered Grams are retained as a separate diagnostic.

| Arm | Construction | State input/readout |
|---|---|---|
| coordinate96 | Top96 joint diagonal coordinates | Select those B/C entries |
| dense96 (orthogonal) | Top96 joint eigenvectors R | B'=R^T B, C'=R^T C |
| dual96 | H=G_C^(1/2)G_B^(1/2)=U Sigma Q^T; V=G_B^(1/2)Q96 Sigma96^(-1/2), W=G_C^(1/2)U96 Sigma96^(-1/2) | B'=W^T B, C'=V^T C; z=W^T s |

Column convention above; implementation uses corresponding row products.
W^T V=I. With scalar head decay, the map commutes with the recurrent transition:
z_t=exp(a_h delta_t)z_(t-1)+delta_t x_t W^T B_t. Readout is C_t^T V z_t;
the omitted kernel is C_t^T(I-VW^T)B_u, with its actual causal decays accumulated.

G_C^(1/2) V W^T G_B^(1/2) equals the rank96 truncation of H. Its singular-value
tail is the optimum of the FIT-weighted Cartesian B/C pair-kernel Frobenius
objective among unrestricted rank96 matrices. This algebra does **not** optimize
the actual causal-decay/kernel weighting, nonlinear normalization, output projection
or the whole chatbot. Joint Gram energy alone does not measure that output error.

All24 bases/Grams/eigenvalues/singular values saved as F64 NPZ,2,694,300B each,
64,663,200B total. No basis refit after the first family fault. For sites0/12/23,
orthogonal joint retention84.74%/96.05%/98.76%, coordinate72.04%/92.47%/98.39%;
centered orthogonal80.89%/94.12%/94.87%, coordinate68.22%/90.52%/93.56%.
Dual pair-tail relative error8.1924%/7.4495%/2.8914% is a different objective.
Dual V/W spectral norms:5.260/3.615,13.565/16.276,32.182/45.446 respectively.
These nonorthogonal gains require numerical/quantization checks in a real target;
they are not automatically benign when generator precision changes.

## Numerical qualification of these local measurements

Independent reconstructed source scan/source gate/full RMSNorm/BF16 output:
ALL144 case-sites have relative RMS **exactly0** against captured source fields,
within predeclared1e-4 criteria. Original source SSD chunk128 reduction is retained.
Independent sequential F64 recurrence, eight fixed channels/full1507 FIT history:
relative RMS9.3744263e-8/1.0941608e-7/1.9843520e-7 at sites0/12/23, all PASS1e-4.
This checks a separate recurrence calculation, not only reuse of source contractions.

Separate [stored algebra audit](ORIGINAL_FALCON_RECURRENT_PROJECTION_AUDIT_PROTOCOL_20261009.md)
uses NumPy F64 on saved Grams, with no predictions/refit/GPU/source calls.
ALL24 identities pass: max W^T V error2.266590e-14, R^T R4.218847e-15;
independent truncated-kernel relative error<=1.309780e-12; actual residual versus
reported tail difference<=3.039236e-14. ALL144 saved aggregates recompute with
absolute delta0;24 bases/18 inherited records match the fault exactly,126 are new.
All12 domain means/worst and per-case better counts are in the audit raw result.

This is a LOCAL reconstruction/algebra PASS. The earlier original C/GPU
all-history numerical FAIL remains. No useful-chatbot, speed or routing admission.

## Observed DEV errors: mean / worst case

Relative RMS of the complete source mamba output after gate/full norm/out_proj;
each case has equal weight,24 DEV cases per site. Percentages below are error,
not retained quality, accuracy or token agreement.

| Site | coordinate96 | orthogonal96 | dual96 |
|---|---:|---:|---:|
| 0 | 20.1061% /27.1645% | 8.2659% /10.5020% | **7.6349% /9.3420%** |
| 12 | 25.7026% /31.6815% | 12.3503% /15.2862% | **8.7184% /10.7008%** |
| 23 | 4.4961% /8.2504% | 3.2013% /6.4695% | **2.4918% /6.4048%** |

FIT mean output errors: coordinate20.6118%/26.2129%/4.6941%,
orthogonal7.5661%/11.6855%/3.0816%, dual6.5008%/7.3538%/2.0367%.
Dual beats orthogonal on DEV output in20/24,24/24,22/24 cases; beats coordinates
in24/24,24/24,23/24. Improvement is not universal for every case or metric.

| Site | dual recurrent only (Dskip removed) | dual centered output | dual output at supervised positions |
|---|---:|---:|---:|
| 0 | 10.5659% /17.0847% | 8.0459% /9.8495% | 7.9284% /9.8840% |
| 12 | 9.5443% /12.0269% | 8.9722% /10.8176% | 8.4976% /10.5353% |
| 23 | 2.4956% /5.9779% | 2.8039% /7.3661% | 2.3804% /8.4475% |

Centering removes each history's own per-output-channel mean; recurrent-only
subtracts the identical source Dskip*x. These prevent a small aggregate error
being attributed solely to common mean or skip. At site23 the dual label-position
worst8.4475% is worse than orthogonal6.9925%, despite better mean. No newly invented
1% quality threshold is applied: protocol admitted numerical validity, not quality.

## Preserved first GPU-cap failure and bounded completion

First projection family freezes actual code/binding in commit
`a2d4af06826caeb85f8532294e5c15e3beb116fa`; [protocol](ORIGINAL_FALCON_RECURRENT_PROJECTION_PROTOCOL_20261009.md),
[binding](original_falcon_recurrent_projection_binding_20261009.json) SHA
`b47fb412f0100ce8ad51f00ac3167ae93a8543ebeb94192990f24c700329d59c`/3,048 inputs.
Launcher29972/worker9408/creation1791557756.240087,exit1/session54410 CLOSED.
GPU reserved6,731,857,920B exceeds frozen5GiB cap; allocated2,838,058,496B.
First fault at broad_fit_smol_magpie_ultra_022/site00, after24 bases/18 complete
saved case-sites (log prints17 before guard). [Launcher failure](original_falcon_recurrent_projection_result_20261009.launcher_failure.json),
[log](original_falcon_recurrent_projection_result_20261009.worker.log), original
first_fault.json and [worker byte archive](original_falcon_recurrent_projection_frozen_20261009.py.txt)
SHA07caece3d3825ee8c736d245f3fa91b58b25dedcce55dc56796374b04aa031ae retained.
No original successful result/terminal. Original resource gate remains FAIL.
Held family34.829s; worker fault26.390s; held worker OS1,419,501,568B/launcher30,986,240B.

New [completion protocol](ORIGINAL_FALCON_RECURRENT_PROJECTION_FINISH_PROTOCOL_20261009.md),
[binding](original_falcon_recurrent_projection_finish_binding_20261009.json) SHA
`82ec3a783b9d6f80d75947479d21c14a48fdaebad176515d92bd5461033197a1`/3,097 inputs,
freeze`a12b282b00911faf2c41490c3e7cc1594cd4d3e0`. Adopts24 actual bases/18 actual
comparisons and computes ONLY missing126. No completed prediction or FIT refit
replayed. One F64 witness lost in the fault's RAM is explicitly recomputed from
stored fields; the final three witnesses are all preserved.

Repair bounds Y_diag's outer source-chunk axis; inner S reduction/products/F32
unchanged. At1507 positions its temporary changes from2,415,919,104B to201,326,592B.
CUDA cache released after each new case. Neither numerical criteria nor GPU caps
increase. No change to original engine, source recurrence or model geometry.
New limits900s/reserve60/OS6GiB/GPU4 allocated/5 reserved/output4MiB, all PASS.

Launcher11864/worker25536/creation1791558444.0739255,exit0/session46714 CLOSED.
Held family60.860s; worker44.313s. Held worker OS1,425,440,768B +launcher32,141,312B
=1,457,582,080B conservative sum; GPU allocated1,168,875,008B/reserved1,642,070,016B
(allocator only). Actual combined held projection families **95.689s**; binder/
stored-adoption preparation is additional. Original namespace43 files64,728,700B,
completion126 files145,718B;169 files64,874,418B retained, excluding DOC receipts.

Successful [raw result](original_falcon_recurrent_projection_finish_result_20261009.json)
SHA`a00fa53213b20d0d23d87d921f1b4bfed8375cfa8a1fdab89547912c8160b0c5`,
[terminal](original_falcon_recurrent_projection_finish_result_20261009.terminal.json),
[log](original_falcon_recurrent_projection_finish_result_20261009.worker.log).
Audit freeze`19e1a3661b407fed9e7beb6c84abbccbe9434360`, [binding](original_falcon_recurrent_projection_audit_binding_20261009.json)
SHA69eeae325ccd99b583ab55017e2ebe79b6ee89fbb181937eb2d24bdbb1804157,
[raw audit](original_falcon_recurrent_projection_audit_result_20261009.json) SHA
`ae5c8e839fe9052185089043fdad8325e7123bf3257d827b386468acd57d29d8`.
Audit direct command exit0/output chunk678a43; process28704/creation1791559039.244167, self1.515s/
OS46,067,712B. No separately held audit launcher-family resource record collected.

## Original-core geometry accounting, not a speed extrapolation

D256/N96/L6/five SSM+one SWA/H128/k8/n1152/V65537. Exact matrix-shape products;
not native measured latency. Count source B/C nonlinear auxiliary channels before
claiming a warm map: a dense state map needs all256+256 source SiLU features;
linear projection before SiLU does not commute. Each raw delta function can use
paired SiLU auxiliaries via SiLU(t)-SiLU(-t)=t in real arithmetic.

| DN/DT | Core products | All counted products/token | Core F32 coefficients bytes | Recurrent F32 state bytes |
|---|---:|---:|---:|---:|
| 512/16 current | 2,801,664 | 26,067,200 | 12,273,664 | 983,040 |
| 512/48 proposed | 2,965,504 | 26,231,040 | 12,929,024 | 983,040 |
| 1024/16 proposed | 5,341,184 | 28,606,720 | 23,486,464 | 1,966,080 |
| 1024/48 proposed | 5,668,864 | 28,934,400 | 24,797,184 | 1,966,080 |

Each total includes full head16,777,472, selected-expert4,718,592 and flat-router
1,769,472 products. Core coefficients exclude head/embedding/FFN outside core.
Products exclude vector/conv/window attention/gated norm/packing/alignment/physical
DRAM. Untied state exponentials245,760->491,520. Source repeated head A/delta can
algebraically permit hoisting, not yet implemented/qualified/timed. Proposed wider
geometries are not implemented;11% more counted products is not11% more latency.
Original small-V/cache701.7/s cannot be inherited by this full-V chatbot.

## Decision and exact next

Use dual96 as the leading **source-local state initializer** for the next finite
assay; keep coordinates/orthogonal comparisons as retained controls. Do not select
a compact whole model from these metrics. The generator/observable-channel budget
is the next missing evidence, using the existing capture without source replay.

A proposed DN1024/DT48 allocation can carry512 B/C auxiliaries +96 paired delta
auxiliaries +384 active x channels (8 per each of48 heads)=992 channels. It keeps
all48 head timescales but discards56/64 x/gate channels per head. Previous four-head
256-x proposal discards44/48 timescales. Neither proposal is measured here.
Measure FIT-derived output-weighted channel/head selection, gating and full-norm
denominator approximation on DEV before choosing one real warm candidate.
Use full/centered/supervised-position errors and domain/worst tails, not energy
alone. Separate a full-source-denominator diagnostic from deployable retained-
channel normalization. Strong omission error redirects compression toward grouped
features or extra depth, instead of blindly adding training dose.

P256 observability,24->6 composition, parallel attention, SiLU->dReLU ternary
function conversion and normalized selected routing remain independent losses.
Longer T4 is permitted in principle but should train a justified real candidate,
with measured residency/throughput and checkpoint/plateau stops; none allocated.
Keep Adam25 and all old failures. Useful own-history chatbot+same-artifact>=50,
useful RAM-driven n/CPU IDs AND mass/physical DRAM and other families/scales remain.
The operational [next document](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md) is updated;
its historical capture-consumed bytes remain in [byte archive](original_engine_transfer_next_frozen_20261009.md.txt).

## Reproduction

Python312 -I -S -B -X utf8, repo root. Code [projection](../../../benchmarks/native_expert_scaling/original_falcon_recurrent_projection.py),
[completion binder](../../../benchmarks/native_expert_scaling/original_falcon_recurrent_projection_finish.py),
[audit](../../../benchmarks/native_expert_scaling/original_falcon_recurrent_projection_audit.py).
Existing namespaces/results immutable; bindings retain exact input bytes. Reproduce
only in newly named outputs with a freshly frozen binding; do not overwrite/replay
historical completed comparisons. Commands actually used:

```text
original_falcon_recurrent_projection.py --launch --binding docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_projection_binding_20261009.json --binding-sha b47fb412f0100ce8ad51f00ac3167ae93a8543ebeb94192990f24c700329d59c --freeze a2d4af06826caeb85f8532294e5c15e3beb116fa --directory results/native_expert_scaling/original_falcon_recurrent_projection_20261009 --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_projection_result_20261009.json
original_falcon_recurrent_projection_finish.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_projection_finish_binding_20261009.json
original_falcon_recurrent_projection.py --launch --binding docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_projection_finish_binding_20261009.json --binding-sha 82ec3a783b9d6f80d75947479d21c14a48fdaebad176515d92bd5461033197a1 --freeze a12b282b00911faf2c41490c3e7cc1594cd4d3e0 --directory results/native_expert_scaling/original_falcon_recurrent_projection_finish_20261009 --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_projection_finish_result_20261009.json
original_falcon_recurrent_projection_audit.py --run --binding docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_projection_audit_binding_20261009.json --binding-sha 69eeae325ccd99b583ab55017e2ebe79b6ee89fbb181937eb2d24bdbb1804157 --freeze 19e1a3661b407fed9e7beb6c84abbccbe9434360 --out docs/research/NATIVE_EXPERT_SCALING_20260925/original_falcon_recurrent_projection_audit_result_20261009.json
```

First command's historical worker is the byte archive, not current repaired code.
Executable full paths and held commands are recorded in respective terminal/failure
JSON. Audit changes no source/core/candidate/capture/prediction files.
