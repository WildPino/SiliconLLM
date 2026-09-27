# Native expert scaling: research control index

**Date:** 27 September 2026. **Branch:** `research/native-expert-scaling`.
**Status:** the jointly trained Instruct product-key E128 adapter passes
its independent donor-relative quality audit (METH-57), and its native
selected-expert component matches the PyTorch oracle (METH-58). A
stored 539.955 MB BF16-attention/R8-head+FFN core passes new document,
generation and full PIQA endpoints but **fails** blind semantic
conservation (METH-62: 41 versus 37 unsupported claims; one versus zero
missing details). It does not advance to full native integration.
Accepted-token rate, CPU LUT validity and quality at large counts of
distinct learned experts remain open. The older base-donor R8 attempt
failed generation (METH-27: 16/24 loops).
METH-29 also rejects a post-hoc balanced coarse router index on real
E128 inputs: 32/64 candidates recover only 62.48%/85.85% of exact
top-4 IDs. METH-30's learned gate on those frozen groups raises
64-candidate recall to 91.61% but also fails the 99.9% gate; a
jointly learned hierarchy or different index was needed. METH-35/36
provide a route-fidelity-passing reduced-rank index at E128.
METH-37 rejects its C64 route replacement on top-1 quality.
METH-40 measures the C96 sketch CPU component at hypothetical
E27,355/E273,547: six-thread 3.714/18.453 ms/token, leaving only
1.547 ms inside the whole-model 20 ms target at the larger point.
METH-41 independently passes the C96 E128 route-replacement
quality gate (99.268% top-1), but the parent still loops on
15/24 continuations and C96 has no full native rate.
METH-42 finds that a same-geometry Qwen0.5B-Instruct donor has
0/24 chat-format loops on a frozen diagnostic versus 7/24 for
base on identical inputs. Directly grafting the base-trained
adapter fails its ≥95% ranking gate at 84.277%; new Instruct
experts must start at donor parity and be trained.
METH-43's fresh E128 Instruct adapter starts at exact donor logits and
improves raw BPB by 0.012562 after 16 updates, but fails chat retention:
89.716% top-1 on the viewed METH-42 prompt set, below its 95% gate.
METH-44 prospectively corrects the chat objective to retain the entire
prompt distribution and reduces the update step. From a new donor start,
it passes a new, disjoint 24-prompt development gate at 95.981% top-1
and raw BPB −0.001843 after 16 updates. Its checkpoint is eligible only
for a preregistered longer continuation. METH-45 ran that continuation
and stopped at update 512 when chat top-1 fell to 93.985% below 95%,
despite raw BPB −0.028953. Its 12 external documents/chat prompts and
PIQA gate remained unopened. METH-46 shows meaningful conditional
routing at update 256 (permuting router rows worsens BPB +0.020330),
but severe load concentration (worst expert 24.56× mean selections).
METH-47's stronger KL and smaller step pass update-512 chat retention
(98.270%) and the preregistered external automatic gates: document BPB
−0.005033, prompt top-1 96.078%, 12/12 EOS without loops, and full PIQA
1290/1838 versus donor 1291/1838. Manual review finds unsupported
student claims in saved summaries; semantic conservation, large-E quality
and native rate remain unproven.
METH-48/49 export the **actual METH-47 trained factors** as int8 with
per-expert and per-factor-row scales. Both reproduce near-flat pooled
document BPB and 12/12 EOS, but fail the fixed ≥99% prompt top-1
gate at 95.839% and 95.744%. Neither is a quality-valid packed bank.
Their E27,355/E273,547 payload projections grow ~10× with expert count
(9.42→94.17 GB per-expert-scale; 11.79→117.86 GB row-scale), excluding
router/index, donor and workspace. These are arithmetic, not trained
large-E or CPU LUT results.
METH-50 shows that the METH-49 factor perturbation changes 6.58% of
full top-4 sets across layers, but forcing original IDs restores prompt
top-1 only from 95.744% to 96.270%. Factor arithmetic remains a
separate ranking failure; a better router cannot rescue that bank alone.
METH-51 now exports the METH-47 BF16-effective trained factors with
bit-identical logits over 2,091 prompt positions and 12/12 identical
greedy continuations. METH-52 measures their selected path in C:
0.321 ms/token at E27,355 on six CPU threads with an 18.82 GB
**replicated** pool; the E2,735→E27,355 ratio is 0.978×. This passes
the selected-component gate, but it is not a LUT, full-model rate, or
distinct learned large-E quality result.
METH-53 validates exact top-4 pair search for a synthetic product-key
router but rejects full-width FP32 keys: six-thread E27,392→E273,408
latency rises 4.800×, above its 4× gate. METH-54's separately
trainable rank-64 product keys pass their CPU cost gates: 0.4405
ms/token at E273,408 and a 1.392× same-thread 10× ratio. These are
synthetic router timings. No product keys have been trained with
distinct experts at large E. METH-55 jointly trains the same router
geometry with distinct E128 factors for 16 updates from exact donor
parity: fresh chat-prompt top-1 95.985%, raw ΔBPB −0.001552, at least
125 changed B slots/layer. It passes the early screen but has
21.76× worst maximum/mean route load. Longer retention and semantic
audit remain necessary before any larger-E quality claim.
METH-56 resumes that exact checkpoint with stronger donor KL through
update 512. On a new disjoint 24-prompt set it passes terminal gates:
96.824% top-1, raw ΔBPB −0.014485 and all 128 B slots changed/layer.
Agreement peaks at 98.163% at 256, then declines; maximum/mean
route load is still 14.77× at 512. The terminal checkpoint is
eligible for a frozen external semantic/task audit, not yet native
promotion or a trained large-E claim. METH-57 then passes that frozen
24-document, full PIQA and arm-blind semantic audit: external prompt
top-1 is 96.897%, pooled ΔBPB −0.005734, PIQA 1292/1838 versus donor
1291/1838; unsupported excerpt claims are 22 student versus 30 donor.
This is relative E128 quality eligibility for native parity work, not a
native, LUT or large-E learned quality result.
METH-58 exports that exact trained E128 bank and checks a standalone C
product-key/selected-factor component on 96 hidden-state inputs: all
routes, BF16 gate weights and residuals match the stored PyTorch oracle.
The dense Qwen core, full `engine.c` integration and accepted-token rate
are still unmeasured.
METH-59 exports the matching Instruct donor as a 496.122 MB R8 core;
the combined E128 student passes viewed document BPB but fails prompt
top-1 at 94.230% versus the registered 95% floor. METH-60 restores
individual organs: BF16 attention gives the best under-budget result,
94.885% at 548.320 ideal MB/token, still five positions short.
No compressed core is yet eligible for native model promotion.
METH-61 applies a fixed weight-only least-squares row-scale correction
to the remaining R8 head/FFN under BF16 attention. It lowers local
weight error but worsens prompt top-1 (3,834/4,125 when applied to
both); this rule is rejected. Direct quality on a new, unviewed
stored mixed artifact is the next decision, while the failed 95%
proxy gate remains recorded.
METH-31 measures the existing CPU LUT kernel in the Qwen rank-8
shape with synthetic packed factors: E1280→E12800 raises the
six-thread selected path only 1.136× to 0.505 ms/token. That
component passes its cost gate; route cost, factor quality and
end-to-end rate remain unverified.
METH-32 packs the **real trained** rank-8 factors in that LUT layout;
reused-document BPB passes its diagnostic screen, but next-token
top-1 agreement falls to 96.01% versus the ≥99% gate. The all-ternary
factor format is held before native promotion.
METH-33 isolates A and B: keeping either factor fp32 raises top-1
agreement only to 97.02% or 97.28%. Both mixed ternary variants
fail the same prospective ranking gate despite passing their byte
and reused-document BPB screens.
METH-34 stores both factors as 15-level LUT indices and lowers the
reused-document BPB penalty to +0.000069, but top-1 agreement is
98.324% and still fails the fixed 99% gate. Factor-code tuning is
paused while generation repair and bounded large-E routing remain open.
METH-35 finds a passing fp32 rank-64/64-candidate SVD router
shortlist on the reused E128 route inputs. METH-36 exports its
one-byte sketch and passes the same route gate on 24 distinct
documents (99.939% exact top-4 ID inclusion). METH-37 then
replaces the route: document loss and relative generation gates
pass, but top-1 is 98.356% versus the fixed 99% gate. METH-38/39
isolate both shortlist misses and fine-rescore arithmetic;
the rank-64/64 route is held. METH-40 times a synthetic large-E
sketch scan. METH-41 passes an independent C96 route-replacement
gate at E128; full CPU rate and large-E quality remain open.
NES-01 E128 improves BPB but fails greedy generation; NES-02
finds a dense-router CPU scaling limit; NES-03 int8 shortlist preserves
tested routes and cuts E1280 router cost. No pretrained-to-native conversion
has passed joint quality and rate. METH-06 passed source-ID calibration
coverage and produced a cost-qualified GigaChat IQ2 GGUF, but quality failed.
METH-07–09 isolate precision effects: Q2_K experts help modestly, while the
head/dense precision payment makes a cost-qualified map worse. Goal remains
open.
Fork point: donor pause checkpoint `90bf966`.

## Goal and current decision

Test the architecture thesis of `benchmarks/phase60/engine.c`: whether more
**independently learned experts** improve useful quality while a compact core,
fixed top-k routing, and affordable total work/DRAM traffic keep per-token cost
contained. The user clarified that expert count `E` should eventually scale
with available RAM, potentially by about 10× between 10B and 100B variants;
the CPU LUT/router and model quality must both remain viable as choice grows.
[NES-01](NES_01_E128_RESULT_20260925.md) isolates E32→E128
at small L6/TinyStories scale. Equal tokens do not equal equal exposure per
expert; fixed top-k does not fix dense-router cost. A result at this scale
cannot establish target-scale performance.

The end state is a reproducible method that transfers pretrained capability
to a conditional `engine.c` artifact, retains donor-relative quality and
runs at ≥50 accepted batch-1 tok/s on the **same** artifact and declared
hardware/context; 100 tok/s is a stretch goal. It must be tried across
families/scales, including order-10B and, where resources permit, order-100B.
Scaling experts is an enabling question, not a substitute for this transfer.
[METHOD.md](METHOD.md) tracks the provisional procedure and missing gates.

The historical [donor-adaptation line is paused](../donor_adaptation/PAUSE_20260925.md).
Reuse its evidence and instruments; a direct port alone is a baseline and does
not finish the architecture goal. Read [PRIOR_EVIDENCE.md](PRIOR_EVIDENCE.md)
for native anchors and the [cross-program index](../RESEARCH_INDEX.md) for
scoped donor verdicts. New experiment records live beside this index.

## Latest decisive evidence

[NES-01](NES_01_E128_RESULT_20260925.md) trained 128 distinct experts for
32.768M token positions. E128 BPB **0.842529** beats E32 **0.858854**, with
no dead experts and C pilot latency **1.198×** E32. Yet greedy repetition
fails the frozen gate: **10/16 E128** versus **3/16 E32** continuations loop
in fp32; LUT results are 9/16 versus 4/16. The failure persists despite
healthy aggregate routing on free generation. Median CPU router+selection
grows **14.4→52.1 µs/token** and selected-expert LUT path **456.4→588.0**;
the latter shows fixed top-k is insufficient to assume constant CPU cost.
This is a single-seed small-model result; do not escalate E256 or treat it
as pretrained transfer.

[NES-02](NES_02_CPU_EXPERT_COUNT_STRESS_20260925.md) tested E128 against a
throughput-only synthetic E1280 pool with fixed top-8. Median total CPU time
rose **1.637×**, router+selection **10.002×**, and selected-expert LUT time
**1.285×**. Parallel row scoring preserved bit-identical fp32 logits but
missed its predeclared speed gates; it remains opt-in. At the current tiny
geometry, 100B distinct weights would require E169,094 and an exhaustive
fp32 router would address about **1.039 GB/token**. At a favorable 40 GB/s,
that router payload alone needs **25.97 ms**, above the complete 20 ms budget
for 50 tok/s. This is arithmetic, not a 100B model measurement. It motivates
bounded candidate routing and a packed-only expert export.

[NES-03](NES_03_INT8_ROUTER_SHORTLIST_RESULT_20260925.md) uses an int8 router
sketch and exact fp32 rescore of 32 candidates. On frozen held-out inputs,
both E128 and synthetic E1280 C audits miss **zero** exact top-8 IDs across
8,192 positions × 6 layers; E128 BPB/top-1 and greedy output are unchanged.
The parallel E1280 router median is **116.8 µs/token** versus **536.1** in
NES-02; total **1,474.2** versus **1,880.4**. The frozen E128 cost gate passes
by only 0.12 µs/token and machine variation is comparable, so this is an
opt-in provisional router candidate. It does not repair E128 generation or
establish learned large-E quality.

[METH-04](METH_04_GIGACHAT_LOWBITS_PREFLIGHT_RESULT_20260925.md) now names
a concrete GigaChat BF16→mixed low-bit transformation: IQ2_XS for most MLA
and routed/shared experts, Q4_0 for block-incompatible MLA key tensors,
Q3_K head and Q4_K dense first FFN. The bound 414-tensor plan addresses
**534.025 MB/token**, 25.975 MB below the 560 MB design allotment. The
subsequent METH-05/06 imatrix and actual GGUF confirmed the types/bytes
but failed donor-relative quality. At 40 GB/s, payload alone prices to
13.351 ms/token; the margin remains narrow.

[METH-05](METH_05_SOURCE_ID_IMATRIX_RESULT_20260926.md) repaired the
GigaChat calibration input path to use source-tokenizer IDs. A local
106×512-token BF16 importance matrix contains all 254 required tensors,
but layer 19 expert 29 received 16 activations and layer 20 expert 51
received 39, below the frozen 64-count floor. The 20-minute run used about
21.25 GB resident at the observed peak. That matrix alone stopped
quantization. [METH-06](METH_06_CYRILLIC_ROUTE_RESULT_20260926.md) added
distinct Russian/Ukrainian source-ID calibration and merged 231 chunks;
minimum routed expert count became **94**, above the frozen 64 floor. The
actual 3.208 GB IQ2 GGUF matches all 414 planned types and addresses
**534.025 MB/token**, but donor-relative pilot BPB worsened by **+0.239465**.
All three categories exceed the frozen +0.20 gross-failure stop. This map
is rejected before the full quality and native speed gates.

[METH-07](METH_07_ORGAN_PRECISION_ABLATION_RESULT_20260926.md) restored
Q4_K precision to MLA or experts separately. Expert restoration rescued
**0.123819 BPB**, just over its frozen 0.120 priority floor; MLA rescued
**0.119356**, just under. Both overrun 540 MB/token. The budgeted
[METH-08](METH_08_Q2_REALLOCATION_RESULT_20260926.md) Q2_K
expert/head/dense swap addresses **533.140 MB/token** but worsens the
pilot loss to **+0.275153 BPB**. [METH-09](METH_09_Q2_EXPERT_ISOLATION_RESULT_20260926.md)
shows Q2_K experts alone improve the IQ2 map by only **0.023740 BPB**,
while the head/dense payment worsens it by **0.059428 BPB** conditional
on those experts. This measured frontier does not justify another
quantization-only reallocation of the same organs.

For pretrained transfer, the local GigaChat base Q4 passes fresh paired BPB,
PIQA and document rollout against BF16. The actual mixed-format GGUF header
prices **1,016 MB of active payload/token**; at 50 tok/s that requires
50.8 GB/s of compressed weight payload before other work. This is calculated,
not a measured DRAM or C rate. The official Granite H Tiny Q4 tensor header
prices **913 MB/token** by the same addressed-payload method; its full weights,
quality and C behavior have not been tested locally. Both direct Q4 paths
exceed the 560 MB/14 ms streaming design allotment at the 40 GB/s yardstick.
GigaChat's C fidelity port is partial and no
quality-plus-≥50 C artifact exists. Qwen2.5-1.5B H1 S3 shows that training a
carve helps but leaves +0.155895 BPB after about 11 h T4. METH-11's
25%-shared, top-16 residual, all-layer step-zero carve fails at +0.706052
BPB on its predeclared CPU pilot; its full-active wiring control is exact.
METH-12 fits a rank-256 shared correction on disjoint calibration data
and routes top-16 donor groups across all 28 layers: local residual SSE
improves 5.23% in aggregate, yet paired pilot BPB loses +0.943873.
[METH-13](METH_13_QWEN05B_JOINT_UPCYCLE_RESULT_20260926.md) tested a
different dense donor, Qwen2.5-0.5B, partitioned into E128 groups with
top-32 routing. Its fitted router recovered 0.7917 of oracle top-32
activation mass versus 0.4049 random, but the all-layer hard carve lost
**+1.360474 BPB**. The frozen CPU quality stop rejects this exact
geometry before GPU work; the independent-expert and LUT scaling
questions remain open.
[METH-14](METH_14_QWEN05B_ORACLE_ROUTE_RESULT_20260926.md) replaces only
the fitted top-32 route with a non-deployable current-activation oracle.
The oracle improves the sparse model by 0.584748 BPB but remains
**+0.775726 BPB** behind donor, above its frozen +0.40 stop. The fitted
route leaves recoverable loss under this oracle, which still leaves a
large donor gap; the remaining causes are not isolated.
[METH-15](METH_15_ZERO_RESIDUAL_EXPERT_SMOKE_RESULT_20260926.md) then
retained the entire Qwen0.5B donor as a frozen shared core and added
zero-initialized E128 rank-8 residual experts/top-4 across all 24 layers.
Its RTX 3060 smoke verified exact step-zero donor logits, gradients in
experts and routers, and a −0.008125 BPB internal-pilot change after 16
updates. [METH-16](METH_16_RESIDUAL_EXPERT_CONTINUATION_RESULT_20260926.md)
continued that exact optimizer/RNG state to 1,024 updates. On 33,374
heldout bytes, student/donor BPB is **0.824196/0.840579** (−0.016383),
the trained route beats a row-permuted null by **0.042131 BPB**, all
128 expert output slots per layer changed, and greedy repetition is
**11/16 versus donor 12/16**. The frozen development gate passes, but
both arms loop often and the heldout windows share a previously used
corpus. The donor core stays dense and no native C/50 tok/s result exists.
[METH-17](METH_17_FRESH_DOCUMENT_TRANSFER_RESULT_20260926.md) used
60 selected documents whose tested fragments were absent from the Qwen
calibration and prior heldout corpora. The same METH-16 checkpoint
scores **0.955405 versus donor 0.942190 BPB** (+0.013215); its
stratified one-sided CI95 upper is +0.015317. The upper bound passes
+0.02, but the mandatory point limit +0.01 fails. Code loses
**+0.029126 BPB, 24/24 documents worse**. Prose and technical changes
are small. The development gain did not generalize across these
documents, so the checkpoint is not ready for native export.
[METH-18](METH_18_RESIDUAL_SCALE_ROUTE_DIAGNOSTIC_RESULT_20260926.md)
kept those documents diagnostic-only: reducing expert output factors
to 0.50 moved the pooled delta to −0.002339 and code to +0.005109.
The full-amplitude trained route beat a row-permuted null by only
0.000652 BPB on code, compared with 0.047306 on prose.
[METH-19](METH_19_HALF_RESIDUAL_INDEPENDENT_RESULT_20260926.md)
fixed factor 0.50 before a different 56-document audit. It passes
the frozen document gate: **−0.000737 pooled BPB**, one-sided CI95
upper **+0.000133**, code **+0.005567**, and trained-route utility
**+0.007235 BPB**. The factor-1.00 control again loses +0.015654.
However **23/24 new code documents** still worsen. The bound
half-amplitude adapter has SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`;
it is not a compact core or native model.
[METH-20](METH_20_HALF_ADAPTER_GENERATION_RESULT_20260926.md)
loads that exact adapter and passes relative loop nonregression on
24 deterministic prompts, **16/24** versus donor **17/24**; both
absolute rates are poor, with adapter code **7/8**. [METH-21](METH_21_HALF_ADAPTER_PIQA_RESULT_20260926.md)
scores all **1,838** labelled PIQA items: adapter/donor accuracy
**69.967%/70.620%**, paired delta **−0.653 points**, lower CI95
**−1.360 points**, passing its frozen task-retention gate. These
support one task-specific relative claim, not useful open generation.
[METH-22](METH_22_QWEN05B_ACTIVE_LEDGER_RESULT_20260926.md)
counts **919.166 MB/token** for an ideal one-byte FFN/attention body
with the existing fp32 tied head and this fp32 adapter. At 40 GB/s,
the conditional streaming estimate is **22.979 ms/token** before
all other work. A one-byte head/body estimate is 510.762 MB/token,
but its quality and native rate were untested at that point.
[METH-23](METH_23_R8_COMPOSITION_RESULT_20260926.md) finds that R8
head/body quantization retains 95.2–95.4% exact top-1 IDs on a reused
diagnostic set. [METH-24](METH_24_R8_CORE_EXPORT_RESULT_20260926.md)
materializes a **496,122,224-byte** packed-only R8 core, SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`.
[METH-25](METH_25_FRESH_R8_ARTIFACT_RESULT_20260926.md) reconstructs
from those stored bytes and passes new 48-document/ranking gates:
adapter **−0.002282 pooled BPB** versus original BF16 donor, upper
CI95 **−0.000909**, route utility **+0.007488 BPB**, donor/adapter
top-1 agreement **95.915%/95.996%**. Code still loses +0.005305 BPB.
This is quality evidence for one 0.5B family, not a C int8 rate.
[METH-26](METH_26_LARGE_E_ROUTER_LEDGER_20260926.md) shows that in
this same rank-8/L24/D896 geometry, E273,547 gives ~100B added
parameters but an exhaustive one-byte router would address **5.882
GB/token** (147 ms at 40 GB/s before other work). Sublinear routing
is necessary before a RAM-scaled E claim.
[METH-27](METH_27_R8_FRESH_GENERATION_RESULT_20260926.md)
tests the same stored R8 core and adapter on 24 prebound prompts:
R8 donor and adapter both loop **16/24**, but the adapter raises
technical loops **2/8→4/8**, failing the relative gate; absolute
code loops remain **6/8**. [METH-28](METH_28_R8_PIQA_COMPOSITION_RESULT_20260926.md)
uses all 1,838 pinned PIQA items and reproduces BF16 donor 70.620%:
R8 donor scores 70.185% and R8+adapter **69.695%**, a −0.925-point
delta with one-sided paired lower CI95 −1.687 points. The frozen task
gate passes, but generation failure blocks this artifact's native
promotion. These are PyTorch BF16-reconstruction results, not C rates.
[METH-29](METH_29_BALANCED_ROUTER_INDEX_RESULT_20260926.md)
tests a bounded hierarchical search on the exact R8+E128 trajectory.
Balanced router-row groups and their means are built without input
data; at 32 or 64 candidates, exact top-4 ID inclusion is only
**62.484%** or **85.848%** over 147,456 input-layer cases, far below
the fixed 99.9% gate. This rejects the post-hoc centroid rule before
CPU work; it does not reject jointly trained or other sublinear routes.
[METH-30](METH_30_LEARNED_COARSE_ROUTER_RESULT_20260926.md)
fits a small nonlinear gate to the same frozen groups on separate
source windows. Its 64-candidate exact top-4 ID inclusion reaches
**93.562%** on internal validation and **91.609%** on the reused
METH-27/29 prompt set (147,456 input-layer cases); full-set match
is **71.523%** externally. Both the 32- and 64-candidate arms miss
the frozen gate. Its 64-candidate arithmetic already exceeds 128
exhaustive row-dot equivalents at E128, so no C lookup is promoted.
[METH-31](METH_31_RANK8_LUT_POOL_RESULT_20260926.md) isolates
the CPU selected-expert LUT path at Qwen L24/D896/rank8/top4
with synthetic ternary factors and random selected IDs. At E12,800,
the 5.505 GB pool yields **0.5046 ms/token** on six threads versus
**0.4440 ms/token** at E1,280; the 10× ratio 1.136× passes the
fixed 1.25× and 5 ms component gates. Single-thread execution
is faster in every arm. This is neither a valid packed Qwen model
nor an end-to-end CPU or route-quality result.
[METH-32](METH_32_TERNARY_FACTOR_RESULT_20260926.md) stores the
trained E128 rank-8 factors as two-trit LUT codes plus fp32 row
scales and unchanged router in a 77.18 MB artifact. Exact reload
passes. Reused METH-25 documents improve pooled BPB by 0.000596
versus intact fp32 factors, but only **5,899/6,144 top-1 IDs**
match and 86.332% of input-layer cases retain the same top-4 set.
The fixed 99% top-1 gate fails. The mixed quality read cannot
establish a usable packed model or native rate.
[METH-33](METH_33_FACTOR_PRECISION_ABLATION_RESULT_20260926.md)
tests A-ternary/B-fp32 and A-fp32/B-ternary separately, with the
same R8 core and original router. Their selected factor bytes are
4.132 and 3.441 MB/token, and both pass BPB screens on reused
METH-25 documents. Top-1 agreement is only **97.021%** and
**97.282%** respectively versus intact factors, so neither passes
the fixed 99% gate. No mixed candidate is promoted.
[METH-34](METH_34_I4_LUT_FACTOR_RESULT_20260926.md) exports both
factors as 15-level LUT indices with fp32 row scales. Selected
factor codes+scales address **3.788 MB/token**. On the reused
METH-25 documents, packed-minus-intact BPB is **+0.000069**, but
top-1 agreement is **6,041/6,144 = 98.324%**, below its fixed
99% gate. This weight-only diagnostic does not establish C LUT
activation fidelity, generation, task retention or native rate.
[METH-35](METH_35_LOW_RANK_ROUTER_RESULT_20260926.md) replaces
the failed fixed-group index with a frozen-router SVD sketch.
The fp32 rank-64/64-candidate arm includes **99.929%** of exact
top-4 IDs on the reused METH-27 inputs and passes its route gate;
five cheaper fixed arms fail. At E128, the passing arm uses
1.071× exhaustive float multiplies, so no CPU gain follows.
[METH-36](METH_36_INT8_SKETCH_RESULT_20260926.md) saves a
5.720 MB rank-64 int8 sketch and tests 24 different METH-17
documents. It includes **589,466/589,824 = 99.939%** of exact
top-4 IDs and matches **99.757%** of complete sets, passing all
fixed route gates. Replacing routes in the actual model and
measuring CPU traffic/time remain required. At E273,547, the
one-byte sketch alone would address 420.17 MB/token.
[METH-37](METH_37_ROUTE_REPLACEMENT_RESULT_20260926.md) applies
that stored shortlist to the full R8+E128 model on 24 METH-19
documents. Pooled BPB changes **−0.000140** versus exact route,
and relative repetition is 17/24→16/24, but top-1 agrees only
**6,043/6,144 = 98.356%** versus the fixed ≥99% gate.
[METH-38](METH_38_CANDIDATE_CAUSE_RESULT_20260926.md) finds
that 96 candidates raise reused-prompt top-1 to 99.349% with
only 9 of 589,824 exact IDs omitted. Even 128 candidates
change 21 top-1 outputs despite perfect route-set agreement;
its numeric apparatus gate fails.
[METH-39](METH_39_RESCORE_NUMERICS_RESULT_20260926.md) confirms
the source: a full `F.linear` score/gather oracle restores
6,144/6,144 top-1, while candidate-only batched rescore
still changes 35 at C128. This diagnostic did not itself
promote a bounded route.
[METH-40](METH_40_ROUTER_SCAN_CPU_RESULT_20260926.md) measures the
stored rank-64 sketch's AVX2 CPU scan with synthetic expansion to
E27,355 and E273,547. Six-thread median projection + scan/selection
+ merge is **3.714/18.453 ms/token**, a **4.968×** increase for
about 10× E. The larger arm passes the 20 ms router-component
ceiling by just **1.547 ms**. It excludes exact fine rescore,
selected experts, core and generation, and establishes no large-E
quality or full-model rate.
[METH-41](METH_41_C96_INDEPENDENT_ROUTE_RESULT_20260927.md) freezes
24 source-disjoint documents before an actual C96 route replacement.
Its pooled BPB penalty is **+0.000059**, bootstrap 95% upper
**+0.000198**, and top-1 agreement **6,099/6,144 = 99.268%**.
Relative repeated-8-gram failures are 16/24 exact versus 15/24
C96. All prospective route and donor-relative document gates
pass. The 15/24 absolute loops and METH-40's scan cost still
prevent full model or large-E promotion.
[METH-42](METH_42_INSTRUCT_DONOR_PILOT_RESULT_20260927.md) tests
the same L24/D896 family's Instruct donor on 24 frozen
chat-format summary prompts. Base/Instruct/grafted-Instruct
repeat on **7/0/0 of 24**; Instruct terminates 24/24 with EOS.
The direct base-trained adapter graft passes relative loss and
repetition but changes prompt top-1 to **84.277%** agreement,
below its prospective ≥95% gate. Instruct is a donor candidate,
while the graft is rejected and semantic/task quality remains
untested on independent prompts.
[METH-43](METH_43_INSTRUCT_ZERO_EXPERT_RESULT_20260927.md) fails its
chat gate at 89.716% top-1 despite exact step-zero donor parity and a
raw BPB gain. [METH-44](METH_44_FULL_CHAT_RETENTION_RESULT_20260927.md)
passes a new same-corpus 95% development screen at 95.981% after 16
updates. [METH-45](METH_45_INSTRUCT_CONTINUATION_RESULT_20260927.md)
stops at update 512: chat top-1 declines to 93.985% while raw BPB
continues to improve. [METH-46](METH_46_ROUTE_UTILITY_DIAGNOSTIC_RESULT_20260927.md)
confirms route-to-factor alignment matters (+0.020330 BPB if permuted
at update 256), but the most loaded raw expert reaches 24.56× mean.
[METH-47](METH_47_STRONG_KL_RETENTION_RESULT_20260927.md) passes the
new 512-update development gate and all preregistered automatic
external document, prompt, generation and PIQA gates. Its manual
review finds unsupported student-specific claims in some summaries;
it is retained as a candidate, not a quality-validated native model.
[METH-48](METH_48_INT8_FACTOR_BANK_RESULT_20260927.md) and
[METH-49](METH_49_INT8_ROW_FACTOR_BANK_RESULT_20260927.md) test actual
METH-47 factor storage. Both int8 variants fail ranking retention on
the viewed external set despite small reconstruction and BPB errors.
The row-scale variant's exact factor payload grows from 11.786 GB at
E27,355 to 117.857 GB at E273,547. The RAM proportionality desired
for 10B→100B is feasible as storage arithmetic, but the tested int8
formats do not preserve this checkpoint's outputs and the router remains
an exhaustive large-E cost blocker.
[METH-50](METH_50_ROUTE_AMPLIFICATION_RESULT_20260927.md) measures that
METH-49 changes 6.58% of full top-4 sets across layer-position cases;
forcing the original IDs still gives only 96.270% top-1 versus the
original factors. A bounded router must be paired with a factor bank
that preserves rankings directly.
[METH-51](METH_51_BF16_EFFECTIVE_FACTOR_RESULT_20260927.md) supplies
that exact factor anchor for the BF16 PyTorch forward: all 317.7M
tested logit elements and 12 greedy continuations match after reload.
[METH-52](METH_52_BF16_SELECTED_CPU_RESULT_20260927.md) runs the
trained BF16 factors through a C selected-path kernel. The E27,355
replicated pool costs 0.321 ms/token and 18.82 GB RSS on the local
six-core CPU; router, core and E>128 learned quality remain open.
Frozen H4+H2I composition also fails. See [METHOD.md](METHOD.md) for
links and scope. No step currently transfers donor knowledge into the native
SSM/SWA target and passes joint quality/rate.

## Running cell and experiment register

| ID | State | Record / raw evidence |
|---|---|---|
| NES-00 | APPARATUS PASS; invalid smokes retained | [asset/dispatch record](NES_00_ASSET_AND_DISPATCH_20260925.md), [RTX 3060 JSON](dispatch_probe_e128_fullstep_rtx3060_20260925.json) |
| NES-01 | JOINT FAIL: generation; other pilot gates pass | [result](NES_01_E128_RESULT_20260925.md), [frozen protocol](E128_EQUAL_TOKEN_PROTOCOL_20260925.md), [decoded fp32 samples](nes01_greedy_fp32_samples_20260925.json), [CPU timing](nes01_cpu_timing_20260925.json) |
| NES-02 | COST STRESS: dense router 10× cost; parallel speed gates fail | [result and shape ledger](NES_02_CPU_EXPERT_COUNT_STRESS_20260925.md), [serial measurements](nes02_capacity_stress_baseline_20260925.json), [parallel measurements](nes02_parallel_router_result_20260925.json); E1280 is synthetic, no quality inference |
| NES-03 | PROVISIONAL ROUTER COST PASS: E128 quality nonregression; E1280 synthetic | [result](NES_03_INT8_ROUTER_SHORTLIST_RESULT_20260925.md), [frozen protocol](NES_03_INT8_ROUTER_SHORTLIST_PROTOCOL_20260925.md), [probe](nes03_int8_router_probe_20260925.json), [raw result ledger](nes03_int8_router_result_20260925.json) |
| METH-00 | ARITHMETIC PREFLIGHT | [GigaChat active-organ traffic](METH_00_GIGACHAT_COST_PREFLIGHT_20260925.md); no decoder or quality measurement |
| METH-01 | HEADER-DERIVED PAYLOAD | [actual Q4_K_M organ ledger](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md), [raw JSON](meth01_gigachat_q4_active_ledger.json); no decoder or quality measurement |
| METH-02 | METADATA SCREEN | [Granite H Tiny](METH_02_GRANITE_H_TINY_METADATA_SCREEN_20260925.md): recurrent/sparse candidate; no local weights or quality/rate result |
| METH-03 | HEADER-DERIVED PAYLOAD | [official Granite Q4 organ ledger](METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md), [raw JSON](meth03_granite_q4_active_ledger.json); 8 MB verified Range prefix, no full local weights or quality/rate result |
| METH-04 | DONOR TRANSFORMATION PREFLIGHT; quantizer dry run matches map; no weights changed | [result](METH_04_GIGACHAT_LOWBITS_PREFLIGHT_RESULT_20260925.md), [frozen map/gate](METH_04_GIGACHAT_LOWBITS_PROTOCOL_20260925.md), [414 tensor overrides](meth04_gigachat_tensor_types.txt), [byte ledger](meth04_gigachat_lowbit_preflight.json), [quantizer log](meth04_gigachat_quantize_dryrun.log) |
| METH-05 | SOURCE-ID BF16 IMATRIX; STOP on rare expert exposure | [result](METH_05_SOURCE_ID_IMATRIX_RESULT_20260926.md), [frozen protocol](METH_05_SOURCE_ID_IMATRIX_PROTOCOL_20260926.md), [machine audit](meth05_bf16_106chunks_audit.json); no converted weights or quality result |
| METH-06 | CALIBRATION COVERAGE PASS; IQ2 QUALITY GROSS FAIL | [result](METH_06_CYRILLIC_ROUTE_RESULT_20260926.md), [frozen protocol](METH_06_CYRILLIC_ROUTE_PROTOCOL_20260926.md), [merged audit](meth06_merged_231chunks_audit.json), [GGUF verification](meth06_gigachat_bf16_imatrix_iq2_verify.json), [paired pilot](meth06_iq2_pilot9_adjudication.json); no full quality or native speed pass |
| METH-07 | OVER-BUDGET ORGAN DIAGNOSIS; experts prioritized | [result](METH_07_ORGAN_PRECISION_ABLATION_RESULT_20260926.md), [protocol](METH_07_ORGAN_PRECISION_ABLATION_PROTOCOL_20260926.md), [machine rescue decision](meth07_organ_rescue_summary.json) |
| METH-08 | ACTIVE COST PASS; Q2 REALLOCATION QUALITY GROSS FAIL | [result](METH_08_Q2_REALLOCATION_RESULT_20260926.md), [protocol](METH_08_Q2_REALLOCATION_PROTOCOL_20260926.md), [GGUF audit](meth08_q2_reallocation_verify.json), [paired pilot](meth08_q2_reallocation_pilot9_adjudication.json) |
| METH-09 | OVER-BUDGET Q2 EXPERT ISOLATION; small quality rescue | [result](METH_09_Q2_EXPERT_ISOLATION_RESULT_20260926.md), [protocol](METH_09_Q2_EXPERT_ISOLATION_PROTOCOL_20260926.md), [machine conditional effect](meth09_q2_isolation_summary.json) |
| METH-10 | LOCAL SOURCE/TRAINING ASSET READINESS; no new quality result | [Qwen2.5-1.5B inventory](METH_10_QWEN_LOCAL_READINESS_20260926.md), [H1 prior](../donor_adaptation/probes/H1_THE_CARVE_TRAINED.md), [H5 adverse composition](../donor_adaptation/probes/H5_CROSS_COMPOSITION_RESULT.md) |
| METH-11 | FULL-LAYER STEP-ZERO GEOMETRY REJECTED; +0.706052 BPB | [result and cost ledger](METH_11_SHARED_RESIDUAL_RESULT_20260926.md), [frozen protocol](METH_11_SHARED_RESIDUAL_PROTOCOL_20260926.md), [raw plan/scores](meth11_qwen_shared_residual_pilot.json), [executable](../../../benchmarks/donor_adaptation/s1/meth11_shared_residual.py); no training or native speed result |
| METH-12 | FITTED SHARED + TOP-16 FULL-LAYER QUALITY FAIL; +0.943873 BPB | [result and cost ledger](METH_12_FITTED_SHARED_RESULT_20260926.md), [frozen protocol](METH_12_FITTED_SHARED_PROTOCOL_20260926.md), [machine fit/pilot](meth12_fitted_shared_pilot.json), [executable](../../../benchmarks/donor_adaptation/s1/meth12_fitted_shared.py); no GPU or native speed result |
| METH-13 | E128/TOP-32 ROUTER PASS, FULL-LAYER QUALITY FAIL; +1.360474 BPB | [result](METH_13_QWEN05B_JOINT_UPCYCLE_RESULT_20260926.md), [frozen CPU protocol](METH_13_QWEN05B_JOINT_UPCYCLE_PROTOCOL_20260926.md), [machine route/pilot](meth13_qwen05b_preflight.json), [executable](../../../benchmarks/donor_adaptation/s1/meth13_qwen05b_preflight.py); no GPU or native speed result |
| METH-14 | LOCAL-MASS ORACLE IMPROVES ROUTE, QUALITY STILL FAILS; +0.775726 BPB | [result](METH_14_QWEN05B_ORACLE_ROUTE_RESULT_20260926.md), [frozen diagnostic](METH_14_QWEN05B_ORACLE_ROUTE_PROTOCOL_20260926.md), [machine comparison](meth14_qwen05b_oracle_route.json), [executable](../../../benchmarks/donor_adaptation/s1/meth14_qwen05b_oracle_route.py); non-deployable route, no native speed result |
| METH-15 | RTX 3060 EXACT-DONOR E128 RESIDUAL APPARATUS PASS; wrong-GPU run invalid | [result](METH_15_ZERO_RESIDUAL_EXPERT_SMOKE_RESULT_20260926.md), [protocol](METH_15_ZERO_RESIDUAL_EXPERT_SMOKE_PROTOCOL_20260926.md), [RTX 3060 machine record](meth15_zero_residual_expert_smoke_rtx3060.json), [wrong-GPU record](meth15_zero_residual_expert_smoke.json), [runner](../../../benchmarks/donor_adaptation/s1/meth15_zero_residual_expert_smoke.py) |
| METH-16 | TRAINED RESIDUAL E128 DEVELOPMENT GATE PASS; −0.016383 BPB | [result](METH_16_RESIDUAL_EXPERT_CONTINUATION_RESULT_20260926.md), [protocol](METH_16_RESIDUAL_EXPERT_CONTINUATION_PROTOCOL_20260926.md), [machine training/generation](meth16_residual_expert_continuation.json), [runner](../../../benchmarks/donor_adaptation/s1/meth16_residual_expert_continuation.py); no fresh documents or native C rate |
| METH-17 | FRESH-DOCUMENT POINT GATE FAIL; +0.013215 BPB, code +0.029126 | [result](METH_17_FRESH_DOCUMENT_TRANSFER_RESULT_20260926.md), [frozen protocol](METH_17_FRESH_DOCUMENT_TRANSFER_PROTOCOL_20260926.md), [selection manifest](meth17_fresh_document_manifest.json), [paired document result](meth17_fresh_document_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth17_fresh_transfer_audit.py); no native export |
| METH-18 | DIAGNOSTIC: RESIDUAL AMPLITUDE DRIVES CODE LOSS | [result](METH_18_RESIDUAL_SCALE_ROUTE_DIAGNOSTIC_RESULT_20260926.md), [frozen protocol](METH_18_RESIDUAL_SCALE_ROUTE_DIAGNOSTIC_PROTOCOL_20260926.md), [machine result](meth18_residual_scale_route_diagnostic.json), [runner](../../../benchmarks/donor_adaptation/s1/meth18_residual_scale_route_diagnostic.py); reused METH-17 documents, no promotion |
| METH-19 | INDEPENDENT HALF-AMPLITUDE DOCUMENT GATE PASS; −0.000737 BPB | [result](METH_19_HALF_RESIDUAL_INDEPENDENT_RESULT_20260926.md), [frozen protocol](METH_19_HALF_RESIDUAL_INDEPENDENT_PROTOCOL_20260926.md), [selection manifest](meth19_half_residual_independent_manifest.json), [paired result](meth19_half_residual_independent_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth19_half_residual_independent_audit.py), [adapter exporter](../../../benchmarks/donor_adaptation/s1/meth19_export_half_adapter.py); generation, tasks and C rate open |
| METH-20 | EXPORTED ADAPTER RELATIVE GENERATION PASS; ABSOLUTE LOOPS HIGH | [result](METH_20_HALF_ADAPTER_GENERATION_RESULT_20260926.md), [frozen protocol](METH_20_HALF_ADAPTER_GENERATION_PROTOCOL_20260926.md), [prompt manifest](meth20_half_adapter_generation_manifest.json), [paired generations](meth20_half_adapter_generation_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth20_half_adapter_generation.py) |
| METH-21 | FULL PIQA TASK-RETENTION PASS; −0.653 ACCURACY POINTS | [result](METH_21_HALF_ADAPTER_PIQA_RESULT_20260926.md), [frozen protocol](METH_21_HALF_ADAPTER_PIQA_PROTOCOL_20260926.md), [token manifest](meth21_half_adapter_piqa_manifest.json), [all paired outcomes](meth21_half_adapter_piqa_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth21_half_adapter_piqa.py) |
| METH-22 | EXACT ACTIVE-BYTE PREFLIGHT; FP32 TIED HEAD BINDS ONE-BYTE BODY | [result](METH_22_QWEN05B_ACTIVE_LEDGER_RESULT_20260926.md), [protocol](METH_22_QWEN05B_ACTIVE_LEDGER_PROTOCOL_20260926.md), [ledger](meth22_qwen05b_active_ledger.json), [tool](../../../benchmarks/donor_adaptation/s1/meth22_qwen05b_active_ledger.py); arithmetic, no native rate or low-bit quality |
| METH-23 | R8 HEAD/BODY COMPOSITION DIAGNOSTIC PASS ON REUSED TEXTS | [result](METH_23_R8_COMPOSITION_RESULT_20260926.md), [protocol](METH_23_INT8_CORE_COMPOSITION_PROTOCOL_20260926.md), [machine result](meth23_int8_core_composition_diagnostic.json), [runner](../../../benchmarks/donor_adaptation/s1/meth23_int8_core_composition_diagnostic.py) |
| METH-24 | PACKED-ONLY R8 CORE EXPORT AND EXACT RELOAD PASS | [result](METH_24_R8_CORE_EXPORT_RESULT_20260926.md), [protocol](METH_24_R8_CORE_EXPORT_PROTOCOL_20260926.md), [artifact ledger](meth24_r8_core_export.json), [exporter](../../../benchmarks/donor_adaptation/s1/meth24_export_r8_core.py) |
| METH-25 | NEW-DOCUMENT AND TOP-1 STORED-ARTIFACT QUALITY SCREEN PASS | [result](METH_25_FRESH_R8_ARTIFACT_RESULT_20260926.md), [protocol](METH_25_FRESH_R8_ARTIFACT_PROTOCOL_20260926.md), [manifest](meth25_fresh_r8_manifest.json), [machine result](meth25_fresh_r8_artifact_result.json), [selection](../../../benchmarks/donor_adaptation/s1/meth25_fresh_r8_manifest.py), [audit](../../../benchmarks/donor_adaptation/s1/meth25_fresh_r8_artifact_audit.py); no native rate |
| METH-26 | LARGE-E EXHAUSTIVE ROUTER INFEASIBLE BY BYTE ARITHMETIC | [geometry ledger](METH_26_LARGE_E_ROUTER_LEDGER_20260926.md); no large-E learned model or timing |
| METH-27 | STORED-R8 GENERATION RELATIVE AND ABSOLUTE GATES FAIL | [result](METH_27_R8_FRESH_GENERATION_RESULT_20260926.md), [protocol](METH_27_R8_FRESH_GENERATION_PROTOCOL_20260926.md), [prompt manifest](meth27_r8_fresh_generation_manifest.json), [all continuations](meth27_r8_fresh_generation_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth27_r8_fresh_generation.py) |
| METH-28 | STORED-R8 FULL PIQA TASK-RETENTION GATE PASS | [result](METH_28_R8_PIQA_COMPOSITION_RESULT_20260926.md), [protocol](METH_28_R8_PIQA_COMPOSITION_PROTOCOL_20260926.md), [all paired choices](meth28_r8_piqa_composition_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth28_r8_piqa_composition.py) |
| METH-29 | POST-HOC BALANCED ROUTER INDEX RECALL FAIL | [result](METH_29_BALANCED_ROUTER_INDEX_RESULT_20260926.md), [protocol](METH_29_BALANCED_ROUTER_INDEX_PROTOCOL_20260926.md), [route/group ledger](meth29_balanced_router_index_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth29_balanced_router_index.py); no CPU rate or large-E quality |
| METH-30 | LEARNED GATE ON FROZEN GROUPS RECALL FAIL | [result](METH_30_LEARNED_COARSE_ROUTER_RESULT_20260926.md), [protocol](METH_30_LEARNED_COARSE_ROUTER_PROTOCOL_20260926.md), [training manifest](meth30_learned_coarse_router_manifest.json), [route counts](meth30_learned_coarse_router_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth30_learned_coarse_router.py); no CPU rate or large-E quality |
| METH-31 | SYNTHETIC SELECTED LUT 10× POOL COST GATE PASS | [result](METH_31_RANK8_LUT_POOL_RESULT_20260926.md), [protocol](METH_31_RANK8_LUT_POOL_PROTOCOL_20260926.md), [raw log](meth31_rank8_lut_pool_raw.log), [machine summary](meth31_rank8_lut_pool_result.json), [engine mode](../../../benchmarks/phase60/engine.c), [summary parser](../../../benchmarks/native_expert_scaling/summarize_meth31_rank8_lut_pool.py); no packed-factor quality or router cost |
| METH-32 | REAL TRAINED TERNARY FACTOR TOP-1 GATE FAIL | [result](METH_32_TERNARY_FACTOR_RESULT_20260926.md), [protocol](METH_32_TERNARY_FACTOR_PROTOCOL_20260926.md), [export ledger](meth32_ternary_factor_export.json), [quality audit](meth32_ternary_factor_audit.json), [exporter](../../../benchmarks/donor_adaptation/s1/meth32_export_ternary_factors.py), [audit runner](../../../benchmarks/donor_adaptation/s1/meth32_ternary_factor_audit.py); no fresh promotion or native rate |
| METH-33 | A-ONLY/B-ONLY TERNARY TOP-1 GATE FAIL | [result](METH_33_FACTOR_PRECISION_ABLATION_RESULT_20260926.md), [protocol](METH_33_FACTOR_PRECISION_ABLATION_PROTOCOL_20260926.md), [paired audit](meth33_factor_precision_ablation.json), [runner](../../../benchmarks/donor_adaptation/s1/meth33_factor_precision_ablation.py); reused texts, no mixed export or native rate |
| METH-34 | 15-LEVEL LUT FACTOR TOP-1 GATE FAIL | [result](METH_34_I4_LUT_FACTOR_RESULT_20260926.md), [protocol](METH_34_I4_LUT_FACTOR_PROTOCOL_20260926.md), [export ledger](meth34_i4_lut_factor_export.json), [paired audit](meth34_i4_lut_factor_audit.json), [exporter](../../../benchmarks/donor_adaptation/s1/meth34_export_i4_factors.py), [audit runner](../../../benchmarks/donor_adaptation/s1/meth34_i4_factor_audit.py); reused texts, no activation LUT or native rate |
| METH-35 | FP32 LOW-RANK E128 ROUTE DIAGNOSTIC PASS AT R64/C64 | [result](METH_35_LOW_RANK_ROUTER_RESULT_20260926.md), [protocol](METH_35_LOW_RANK_ROUTER_PROTOCOL_20260926.md), [route audit](meth35_low_rank_router_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth35_low_rank_router.py); reused prompts, no CPU speed or route-replaced quality |
| METH-36 | STORED INT8 E128 ROUTE GATE PASS ON DISJOINT INPUTS | [result](METH_36_INT8_SKETCH_RESULT_20260926.md), [protocol](METH_36_INT8_SKETCH_PROTOCOL_20260926.md), [prompt manifest](meth36_route_prompt_manifest.json), [export ledger](meth36_int8_sketch_export.json), [route audit](meth36_int8_sketch_audit.json), [exporter](../../../benchmarks/donor_adaptation/s1/meth36_export_int8_sketch.py), [audit runner](../../../benchmarks/donor_adaptation/s1/meth36_int8_sketch_audit.py); route replacement and C CPU time open |
| METH-37 | STORED INT8 ROUTE-REPLACEMENT TOP-1 GATE FAIL | [result](METH_37_ROUTE_REPLACEMENT_RESULT_20260926.md), [protocol](METH_37_ROUTE_REPLACEMENT_PROTOCOL_20260926.md), [manifest](meth37_route_quality_manifest.json), [paired quality/generation](meth37_route_replacement_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth37_route_replacement.py); no C rate |
| METH-38 | C96 DIAGNOSTIC PASS; C128 NUMERIC APPARATUS FAIL | [result](METH_38_CANDIDATE_CAUSE_RESULT_20260926.md), [protocol](METH_38_CANDIDATE_CAUSE_PROTOCOL_20260926.md), [route/ranking counts](meth38_candidate_cause_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth38_candidate_cause.py); reused prompts |
| METH-39 | FULL-SCORE ORACLE PARITY; BOUNDED BMM NUMERIC GATE FAIL | [result](METH_39_RESCORE_NUMERICS_RESULT_20260926.md), [protocol](METH_39_RESCORE_NUMERICS_PROTOCOL_20260926.md), [arithmetic counts](meth39_rescore_numerics_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth39_rescore_numerics.py); reused prompts, no CPU rate |
| METH-40 | SIX-THREAD E273,547 SCAN UNDER 20 MS, ONLY 1.547 MS LEFT | [result](METH_40_ROUTER_SCAN_CPU_RESULT_20260926.md), [protocol](METH_40_ROUTER_SCAN_CPU_PROTOCOL_20260926.md), [summary](meth40_router_scan_summary.json), [seed ledger](meth40_router_seed_export.json), [C benchmark](../../../benchmarks/native_expert_scaling/meth40_rank64_router_scan.c), [exporter](../../../benchmarks/native_expert_scaling/meth40_export_router_seed.py), [verifier](../../../benchmarks/native_expert_scaling/summarize_meth40_router_scan.py); synthetic E expansion, no quality or full rate |
| METH-41 | INDEPENDENT E128 C96 ROUTE-REPLACEMENT GATE PASS; FULL MODEL HELD | [result](METH_41_C96_INDEPENDENT_ROUTE_RESULT_20260927.md), [protocol](METH_41_C96_INDEPENDENT_ROUTE_PROTOCOL_20260927.md), [manifest](meth41_fresh_c96_manifest.json), [raw paired outcomes](meth41_c96_independent_route_result.json), [selector](../../../benchmarks/donor_adaptation/s1/meth41_fresh_c96_manifest.py), [runner](../../../benchmarks/donor_adaptation/s1/meth41_c96_independent_route.py); parent generation and native rate open |
| METH-42 | INSTRUCT DONOR CHAT LOOP GATE PASS; DIRECT BASE ADAPTER GRAFT TOP-1 GATE FAIL | [result](METH_42_INSTRUCT_DONOR_PILOT_RESULT_20260927.md), [protocol](METH_42_INSTRUCT_DONOR_PILOT_PROTOCOL_20260927.md), [prompt manifest](meth42_instruct_prompt_manifest.json), [raw outcomes](meth42_instruct_donor_pilot_result.json), [builder](../../../benchmarks/donor_adaptation/s1/meth42_instruct_prompt_manifest.py), [runner](../../../benchmarks/donor_adaptation/s1/meth42_instruct_donor_pilot.py); new training and independent audit needed |
| METH-43 | EXACT INIT/GRADIENTS PASS; CHAT TOP-1 GATE FAIL | [result](METH_43_INSTRUCT_ZERO_EXPERT_RESULT_20260927.md), [protocol](METH_43_INSTRUCT_ZERO_EXPERT_PROTOCOL_20260927.md), [teacher data](meth43_instruct_teacher_chat_result.json), [smoke outcome](meth43_instruct_zero_expert_smoke_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth43_instruct_zero_expert_smoke.py) |
| METH-44 | FULL-CHAT DEVELOPMENT GATE PASS AT 16 UPDATES | [result](METH_44_FULL_CHAT_RETENTION_RESULT_20260927.md), [protocol](METH_44_FULL_CHAT_RETENTION_PROTOCOL_20260927.md), [new prompt manifest](meth44_instruct_fresh_chat_manifest.json), [smoke outcome](meth44_instruct_full_chat_smoke_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth44_instruct_full_chat_smoke.py) |
| METH-45 | LONG CONTINUATION STOP: CHAT TOP-1 BELOW 95% AT UPDATE 512 | [result](METH_45_INSTRUCT_CONTINUATION_RESULT_20260927.md), [protocol](METH_45_INSTRUCT_CONTINUATION_PROTOCOL_20260927.md), [sealed external manifest](meth45_fresh_external_manifest.json), [training result](meth45_instruct_continuation_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth45_instruct_continuation.py), [unused external evaluator](../../../benchmarks/donor_adaptation/s1/meth45_instruct_external_audit.py) |
| METH-46 | ROUTE-ALIGNMENT DIAGNOSTIC PASS; LOAD IMBALANCE MEASURED | [result](METH_46_ROUTE_UTILITY_DIAGNOSTIC_RESULT_20260927.md), [protocol](METH_46_ROUTE_UTILITY_DIAGNOSTIC_PROTOCOL_20260927.md), [raw counts](meth46_route_utility_diagnostic_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth46_route_utility_diagnostic.py); viewed raw/chat data only |
| METH-47 | STRONGER RETENTION: DEVELOPMENT AND AUTOMATIC EXTERNAL GATES PASS; MANUAL SEMANTIC REVIEW OPEN | [result](METH_47_STRONG_KL_RETENTION_RESULT_20260927.md), [protocol](METH_47_STRONG_KL_RETENTION_PROTOCOL_20260927.md), [new development manifest](meth47_retention_dev_manifest.json), [training result](meth47_strong_kl_continuation_result.json), [external paired result](meth47_frozen_external_audit_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth47_strong_kl_continuation.py), [evaluator](../../../benchmarks/donor_adaptation/s1/meth45_instruct_external_audit.py) |
| METH-48 | PER-EXPERT-SCALE INT8 FACTOR TOP-1 GATE FAIL: 95.839% | [result](METH_48_INT8_FACTOR_BANK_RESULT_20260927.md), [protocol](METH_48_INT8_FACTOR_BANK_PROTOCOL_20260927.md), [paired result](meth48_int8_factor_bank_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth48_int8_factor_bank.py); viewed inputs, no C LUT or large-E quality |
| METH-49 | ROW-SCALE INT8 FACTOR TOP-1 GATE FAIL: 95.744% | [result](METH_49_INT8_ROW_FACTOR_BANK_RESULT_20260927.md), [protocol](METH_49_INT8_ROW_FACTOR_BANK_PROTOCOL_20260927.md), [paired result](meth49_int8_row_factor_bank_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth49_int8_row_factor_bank.py); viewed inputs, no C LUT or large-E quality |
| METH-50 | FACTOR ERROR STILL FAILS TOP-1 WITH ORIGINAL ROUTES: 96.270% | [result](METH_50_ROUTE_AMPLIFICATION_RESULT_20260927.md), [protocol](METH_50_ROUTE_AMPLIFICATION_PROTOCOL_20260927.md), [route/load/ranking counts](meth50_route_amplification_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth50_route_amplification.py); viewed inputs, no CPU or large-E quality |
| METH-51 | EXACT BF16-EFFECTIVE E128 FACTOR EXPORT PASS | [result](METH_51_BF16_EFFECTIVE_FACTOR_RESULT_20260927.md), [protocol](METH_51_BF16_EFFECTIVE_FACTOR_PROTOCOL_20260927.md), [logit/generation readback](meth51_bf16_effective_factor_result.json), [exporter](../../../benchmarks/donor_adaptation/s1/meth51_bf16_effective_factors.py); no native model or larger-E learned quality |
| METH-52 | TRAINED BF16 SELECTED CPU 10× POOL COST GATE PASS | [result](METH_52_BF16_SELECTED_CPU_RESULT_20260927.md), [protocol](METH_52_BF16_SELECTED_CPU_PROTOCOL_20260927.md), [validated summary](meth52_bf16_selected_cpu_summary.json), [seed exporter](../../../benchmarks/native_expert_scaling/meth52_export_bf16_seed.py), [C kernel](../../../benchmarks/native_expert_scaling/meth52_bf16_selected_cpu.c), [summary parser](../../../benchmarks/native_expert_scaling/summarize_meth52_bf16_selected.py); E>128 replicated, no router/core/full rate |
| METH-53 | EXACT SYNTHETIC PRODUCT-KEY ROUTE; FULL-WIDTH CPU GATE FAIL | [result](METH_53_PRODUCT_KEY_ROUTER_RESULT_20260927.md), [protocol](METH_53_PRODUCT_KEY_ROUTER_PROTOCOL_20260927.md), [summary](meth53_product_key_summary.json); no learned route |
| METH-54 | SYNTHETIC R64 PRODUCT-KEY CPU 10× GATE PASS | [result](METH_54_R64_PRODUCT_KEY_RESULT_20260927.md), [protocol](METH_54_R64_PRODUCT_KEY_PROTOCOL_20260927.md), [summary](meth54_r64_product_key_summary.json); no trained expert quality |
| METH-55 | JOINT E128 PRODUCT-KEY 16-UPDATE SCREEN PASS | [result](METH_55_PRODUCT_KEY_E128_SMOKE_RESULT_20260927.md), [protocol](METH_55_PRODUCT_KEY_E128_SMOKE_PROTOCOL_20260927.md), [raw result](meth55_product_key_e128_smoke_result.json), [trainer](../../../benchmarks/donor_adaptation/s1/meth55_product_key_e128_smoke.py) |
| METH-56 | JOINT E128 UPDATE-512 RETENTION PASS | [result](METH_56_PRODUCT_KEY_RETENTION_RESULT_20260927.md), [protocol](METH_56_PRODUCT_KEY_RETENTION_PROTOCOL_20260927.md), [raw result](meth56_product_key_retention_result.json), [trainer](../../../benchmarks/donor_adaptation/s1/meth56_product_key_retention.py); no native or large-E proof |
| METH-57 | FROZEN EXTERNAL AUTOMATIC AND BLIND SEMANTIC GATES PASS | [result](METH_57_PRODUCT_KEY_EXTERNAL_RESULT_20260927.md), [protocol](METH_57_PRODUCT_KEY_EXTERNAL_PROTOCOL_20260927.md), [paired result](meth57_product_key_external_audit_result.json), [blind verdict](meth57_blind_semantic_verdict.json), [unblinded score](meth57_semantic_score.json), [evaluator](../../../benchmarks/donor_adaptation/s1/meth57_product_key_external_audit.py); E128 donor-relative only |
| METH-58 | TRAINED E128 PRODUCT-KEY NATIVE COMPONENT PARITY PASS | [result](METH_58_PRODUCT_KEY_NATIVE_COMPONENT_RESULT_20260927.md), [protocol](METH_58_PRODUCT_KEY_NATIVE_COMPONENT_PROTOCOL_20260927.md), [export ledger](meth58_product_key_export_result.json), [96-case C log](meth58_product_key_native_raw.log), [exporter](../../../benchmarks/native_expert_scaling/meth58_export_product_key.py), [C kernel](../../../benchmarks/native_expert_scaling/meth58_product_key_native.c); no core/full rate |
| METH-59 | STORED INSTRUCT R8 CORE BPB PASS; PROMPT TOP-1 FAIL 94.230% | [result](METH_59_INSTRUCT_R8_CORE_RESULT_20260927.md), [protocol](METH_59_INSTRUCT_R8_CORE_PROTOCOL_20260927.md), [export ledger](meth59_instruct_r8_core_export.json), [paired stopped audit](meth59_instruct_r8_composition_result.json), [exporter](../../../benchmarks/donor_adaptation/s1/meth59_export_instruct_r8_core.py), [evaluator](../../../benchmarks/donor_adaptation/s1/meth59_instruct_r8_composition.py) |
| METH-60 | SINGLE-ORGAN BF16 RESCUES FAIL JOINT TOP-1/TRAFFIC GATE | [result](METH_60_R8_CORE_ORGAN_ABLATION_RESULT_20260927.md), [protocol](METH_60_R8_CORE_ORGAN_ABLATION_PROTOCOL_20260927.md), [raw rows](meth60_r8_core_organ_ablation_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth60_r8_core_organ_ablation.py); viewed prompts only |
| METH-61 | DATA-FREE L2 ROW-SCALE CORRECTION WORSENS TOP-1 | [result](METH_61_R8_L2_SCALE_RESULT_20260927.md), [protocol](METH_61_R8_L2_SCALE_PROTOCOL_20260927.md), [raw rows](meth61_r8_l2_scale_result.json), [runner](../../../benchmarks/donor_adaptation/s1/meth61_r8_l2_scale.py); no packed candidate promoted |

NES-01 ran locally **2026-09-25 11:28–18:09 UTC** and exited 0 at step 4000.
Final checkpoint: `results/native_expert_scaling/nes01_e128_final.pt`, SHA-256
`58c84a8730b03c37e98481b8508f5cc84e15757de5513c3e6354f09b975cd462`.
The optimizer/RNG resume state remains for audit. No T4 work is running or
scheduled.

## Decisions and next exact action

- **Open:** cause/remedy of the E128 free-generation loops; bounded-candidate
  router quality and CPU cost at large E; trained quality at that scale; how to
  reduce a pretrained donor's active traffic while preserving quality;
  pretrained-to-native export/fidelity; generality across families and scales.
- **Closed within their scope:** H5's frozen H4+H2I assembly is adverse;
  STRAT-03's tested shared/router geometry fails; METH-11's 64-shared plus
  top-16-of-192 residual geometry loses +0.706052 BPB before quantization;
  METH-12's fitted rank-256 shared correction plus top-16/256 loses
  +0.943873 BPB despite a 5.23% local SSE reduction. METH-13's E128
  Qwen0.5B hard carve loses +1.360474 BPB despite useful route recall.
  METH-14's activation oracle recovers 0.584748 BPB but still loses
  +0.775726, so the frozen local-mass route-only remedy fails.
  METH-15/16 show that a changed additive residual-expert geometry can
  start exactly at donor and improve its internal BPB after joint training.
  METH-17 shows that this checkpoint's pooled fresh-document point
  delta fails, uniformly on the selected code documents.
  METH-18/19 establish a fixed half-amplitude adapter that passes a
  second, disjoint document gate but still harms 23/24 code documents.
  METH-20 passes relative loop nonregression but the adapter still
  loops on 16/24 continuations. METH-21 passes full PIQA relative
  retention; its −0.653-point delta is not an improvement claim.
  The old donor parity queue is not an
  automatic next step.
- **CPU design state:** NES-03 established a 32-candidate int8 path on its
  pilot, but still scans all E rows and keeps redundant reference weights.
  METH-26 maps the currently quality-screened Qwen geometry to E27,355
  and E273,547 for ~10B/~100B added parameters. Even one-byte exhaustive
  routing would read 0.588/5.882 GB per token. METH-29/30 test
  bounded post-hoc group selection and reject both centroid and
  learned-gate rules on actual E128 route recall, even at 64/128
  candidates. A jointly trained hierarchy or other index,
  packed-only expert bank, measured CPU router/LUT costs, and learned
  large-E quality evidence remain open. METH-31 now isolates one CPU
  LUT selected-path cost at E12,800 and passes that component's 10×
  pool gate, but uses synthetic codes and excludes the router.
  METH-32 produces actual trained-factor LUT codes and reloads them,
  but its all-ternary version fails the prospective top-1 gate on
  reused documents. METH-33 shows that replacing only A or only B
  with the same ternary codebook also fails that gate. METH-34's
  15-level codes improve top-1 agreement to 98.324% and keep
  reused-document BPB nearly flat, but still fail the 99% gate.
  No quality-valid LUT export or actual-factor C rate exists.
  METH-35/36 establish an alternative frozen-router shortlist:
  rank-64 SVD plus 64 exact candidates passes a reused route
  diagnostic, and its stored int8 sketch passes on disjoint
  METH-17 documents. METH-37 shows that its actual route
  replacement fails the fixed top-1 gate despite near-flat
  document BPB and passing relative repetition. METH-38/39
  show that 96 candidates nearly eliminate omitted routes on
  reused prompts but candidate-only fine-score numerics also
  change next-token outputs. The full-score oracle restores
  parity but forfeits bounded routing. METH-41 now passes the
  independent E128 C96 route-replacement gates on new code,
  prose and technical sources; C64 remains rejected.
  METH-40 measures that E273,547 sketch path at 18.453 ms/token
  with six CPU threads, leaving only 1.547 ms of the 20 ms
  whole-model target for exact rescore, experts, core and overhead.
  The one-byte codes alone address 420.17 MB/token. Its expanded
  rows are synthetic, so larger-E learned quality remains open.
- **Next quality design:** METH-27 disproves generation readiness for
  this exact R8 composition despite METH-25/28 loss and task passes.
  METH-42 shows that a same-geometry Instruct donor avoids loops on
  its chat diagnostic, but direct reuse of base-trained factors fails
  ranking. METH-43 fails chat retention; METH-44 initially passes but its
  METH-45 continuation fails at update 512. METH-46 shows the router
  itself matters; METH-47's stronger retention objective passes its
  terminal development and automatic external gates, but its saved
  summaries contain unsupported claims. METH-57's distinct product-key
  checkpoint now passes a larger source-grounded relative audit. Do not
  retune on viewed continuations or extend E256 on a BPB gain alone.
- **Next exact action: repair the compressed Instruct core.** METH-57
  passes its frozen automatic and blind semantic gates; METH-58's
  trained E128 C component matches 96/96 BF16 inputs. METH-59 rejects
  all-R8 core composition at 94.230% top-1 despite near-flat BPB.
  METH-60 finds BF16 attention improves this to 94.885% at 548.320
  ideal MB/token, five positions short of its gate. METH-61's fixed
  weight-only scale correction worsens top-1 despite lower weight L2.
  METH-62 exported that mixture and passed new source-disjoint document,
  generation and full PIQA gates, but its precommitted blind semantic
  comparison failed: 41 versus 37 unsupported claims and one versus
  zero missing details. The packed artifact is rejected for native
  integration. Its prompt top-1 is 93.401% on the new set, and the
  older METH-60 proxy failure remains visible. Develop a new
  precision/adaptation rule on separate material; METH-62 excerpts and
  responses are viewed and cannot serve as a promotion set. Then test
  actual quality before full `benchmarks/phase60/engine.c` integration
  and accepted-token timing on the same artifact.
  METH-47 remains a separate candidate whose saved summaries showed
  unsupported claims.
  METH-50 isolates downstream route churn from factor
  error: forcing original expert IDs still fails the 99% top-1 gate.
  METH-51 has now preserved the trained factors exactly under the BF16
  forward; METH-52 shows their selected C path is only 0.321 ms/token
  at E27,355 with a replicated pool. METH-53 rejects full-width
  product keys under its 10× scaling bound; METH-54 passes that CPU
  bound with rank-64 product keys. METH-55's joint E128 smoke passes
  its early donor-retention gates with distinct factors. METH-56 passes
  the longer E128 retention screen at update 512; METH-57 passes its
  external source-grounded semantic/task audit.
  A **distinct learned** large-E quality ladder remains necessary.
  METH-48/49 reject two int8 factor layouts;
  any one-byte LUT export needs a new quality-valid adaptation, not
  further scale tuning on these prompts.
  Promote only after joint semantic quality and usefulness pass; then
  implement the exact pair
  in `benchmarks/phase60/engine.c` with logit parity, accepted batch-1
  tok/s and split core/router/expert timing and traffic. The separate
  `donor_engine.c` Qwen implementation is a fidelity reference, not
  this joint-quality/rate result. In parallel, test a **jointly trained**
  hierarchy on distinct experts. For the stored SVD index,
  METH-37 failed C64 route-replaced top-1; METH-38/39
  identified candidate omissions plus rescore numerics.
  METH-41's frozen C96 candidate-only score rule passes the
  independent E128 route gate, but its 15/24 generation loops
  mean the parent artifact still needs repair before joint
  native promotion. METH-40 already measures projection/sketch
  scan on a 10× synthetic E ladder;
  a full router still needs exact rescore and sufficient CPU
  margin for the selected factor path and core. Independently test a
  bounded/sublinear large-E index against trained route targets.
  METH-29/30's fixed-group rules failed; METH-36's scan is
  lower dimensional but still linear in E.
  Pause post-hoc factor-code tuning on the reused texts after
  METH-32/33/34/48/49's ranking failures. METH-48's per-expert int8 bank
  projects to 94.17 GB and METH-49's row-scale bank to 117.86 GB at
  E273,547; both exclude core/router/index and both fail ranking at E128.
  Any quality-valid factor candidate needs new documents, generation and
  tasks before C integration. The target remains approximately 10× as
  many **distinct learned** experts from the 10B to 100B geometry when
  user RAM allows, with selected work per token bounded; RAM alone does
  not guarantee router accuracy or throughput.
  Only **distinct trained** expert expansion can validate quality as E
  grows. METH-63 now allocates independent E1280 factors (10× E128)
  and verifies exact top-four product-key routes on 768 cases/scale;
  BF16 factor payload rises 88.080→880.804 MB while selected top-four
  payload remains 2.753 MB/token. Those additional slots are untrained,
  so the next scale step must jointly train E1280 against an E128
  control and independently audit donor-relative quality and route
  load. METH-64 supplies a CPU-resident sparse-gradient path: small
  dense/offloaded A/B gradients match exactly, and a 24-layer E1280
  synthetic backward peaks at 385 MB allocated GPU with 3.106 GB
  process RSS. METH-65 now runs one real full-donor E1280 update with
  sparse CPU Adam state: exact initial logits, nonzero B gradients in
  all layers, 244–471 changed slots/layer and 3.281 GB allocated GPU
  peak. This is an apparatus pass only. The next gate is a controlled
  E128/E1280 training ladder with independent quality and route-load
  audit; a single update cannot show useful distinct learned experts.
  METH-66 now completes a matched 16-update E128/E1280 comparison.
  E1280 improves new-prompt donor top-1 from 93.885% to 94.882% and
  changes at least 846 B slots/layer, but both fail the fixed ≥95%
  gate. E1280 worst max/mean route load is 71.06×, and only 538 of
  1,280 slots are selected in the least-covered layer on the viewed
  prompts. Both checkpoints are retained for diagnosis, not promoted;
  any continuation needs a new rule and disjoint prompts.
  No T4 job is planned.

Preserve unrelated working-tree changes in `docs/research/RESEARCH_INDEX.md`
and `benchmarks/donor_adaptation/density/build_document_holdout.py`. Update
this index by replacing current state, and put detailed evidence in the
experiment record. Documentation is in English; no model/assistant signatures.
Graphify is optional for targeted architecture questions. Its last complete
snapshot predates METH-48/49; the later whole-repository update was stopped
after AST extraction because routine refresh cost outweighed its use here.
