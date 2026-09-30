# Native expert scaling: research control index

**Date:** 30 September 2026. **Branch:** `research/native-expert-scaling`.
**Current decision:** The real GigaChat 10B expert-weight METH-180 screen
rejects direct rank-192 int8 SVD factors: median best-case weight energy is
47.243% across 27 sampled projections, against a frozen 95% gate. Even rank
768 has 94.154% median energy at twice an ideal Q4 projection payload.
METH-181 also rejects diagonal activation-weighted rank-192 factors:
disjoint-domain median output-energy proxy is 49.152%, and even the
test-domain diagonal oracle reaches only 52.297%. Neither screen measures
full-model quality. METH-182 tests the actual Qwen grouped-R8 FFN in C:
12.829 ms/token at six threads and 1.896% median output relative L2 error
miss its frozen 10 ms/1% component gates. METH-185 tests a different
pretrained-to-conditional geometry: exact-activation top-K donor FFN
channels. Even K=2,048/4,864 leaves 16.49% median output error, so
that direct channel-selection rule fails its frozen fidelity gate. The
stored METH-186 Q6 core with E1,280 bank costs 481.535 MB/token by
ideal addressed-byte accounting, but METH-187 loses 5.318 donor-top-1
points versus the BF16 E1,280 composition on viewed sources. A
trained compact-core correction needs to recover this gap before
fresh quality and native timing. METH-188 attributes 4.361/5.318
ranking points to the Q6 FFNs alone. The stored rank-64 FFN
correction METH-189/190 fits a 534.619 MB/token ideal addressed ledger
but regresses viewed-source donor top-1 to 73.075% and fails every
development gate. METH-191 rejects a constrained B-only retry on
train-corpus validation before development. METH-192 finds that the
optimal rank-94 reconstruction within the byte budget captures only
20.85% median Q6 FFN error energy. Direct grouped-Q8 FFNs in
METH-193/194 almost eliminate BPB loss but miss pooled and code
top-1 gates by 0.113 and 0.229 points respectively, while using
559.982 MB/token of the 560 MB ideal allotment. METH-195/196 lowers
Q8 weight MSE but worsens both rankings; that scale-refinement route
is stopped. The
tenfold E1,280→E12,800 METH-175 training
passes its matched update, balance and artifact gates, but the untouched
METH-176 quality gain fails the paired source bootstrap. METH-178 finds
only 2.318% of route-weighted candidate-minus-control B difference in
within-parent child variation, and METH-179 finds no functional advantage
for the exact trained content route over eight rotations on consumed
diagnostic inputs. METH-201 applies all nine trained content children
to identical real states and gates: pooled sibling output spread is
only 1.583% of the content-parent mean residual and 0.00396% of the
full MLP output; no layer meets its functional-diversity screen.
METH-202 keeps the exact trained route and scales only child-specific
deviations: λ=0/1/4 BPB is 1.067704/1.067669/1.067793 on viewed
documents. Quadrupling is worse than zero deviation and fails both
frozen directional gates. Change route/training coupling before
another long large-pool training run.
Merely continuing the same hash-route/shared-base
recipe is not a justified path to useful additional specialists. The
METH-184 normalized content-score/hash mixture also fails its frozen
training load/signal gates before external validation, with repeated
chat contexts concentrated even at the largest noise setting. The
first synthetic `engine.c` selected-LUT E128,000/E12,800 ratio was
1.309× in METH-177, but METH-197's same-binary run reversed it;
METH-198/199/200 paired, bound and one-thread controls fail their
repeatability gates. The tenfold CPU LUT-pool ratio is inconclusive
on this host, although every measured selected-factor component
median stayed below 0.7 ms/token. Prefetch regressed both pool sizes
and is rejected. A stronger specialist-learning route,
quality-valid compact factors and a same-artifact full native rate test
remain necessary. Useful 10B/100B transfer and >=50 accepted tok/s are
not established. No T4 has been used.
**Established smaller rung:** The centered BF16 E1,280 children retain
donor-relative quality on new sources (METH-121/123). Their exact
learned content route beats all nine child-ID rotations on the same
24 already consumed documents (METH-183), with +0.000330 mean-rotation
BPB disadvantage and +0.000160 paired-bootstrap fifth percentile.
This is a mechanism diagnostic, not independent new-source quality or
evidence that the E12,800 route is solved. Their exact
shared-A factor bank passes actual-state native CPU access (METH-125/126);
a full C FP32-core reference runs at 16.818 tok/s (METH-127), with an
exact selected-row head reaching 20.791 tok/s on its matched reference
(METH-129). Q7 and Q15 paired-LUT child-B banks pass storage/CPU component
checks but fail separate blind full-model grounding gates (METH-131/133).
The first untrained E12,800 third-tier router passed CPU cost (METH-135)
but failed route balance after training (METH-136/138). Conditional decile
routes also failed source-held-out load (METH-139/140/141), motivating
the later balanced hash-route line; its detailed sequence is below.
The quality-valid compact donor core, native same-artifact >=50
accepted-token/s, CPU LUT factor cost at E12800, and 10B/100B transfer
remain open. The earlier 539.955 MB BF16-attention/R8-head+FFN core
failed blind semantic conservation (METH-62), and the base-donor R8
attempt failed generation (METH-27); neither can be promoted silently.
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
- **Earlier compressed-core investigation.** METH-57
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
  any continuation needs a new rule and disjoint prompts. METH-67
  scores the frozen METH-55/56 E128 checkpoints on the viewed METH-66
  set: 96.299%/96.535% top-1. Thus the set alone does not explain
  METH-66's E128 failure. METH-66 also changed the training draw
  schedule/raw exclusion pool. Before another 10× training ladder,
  restore validated E128 initialization and dense-AdamW semantics in
  the CPU-offloaded path, and compare an update on identical samples.
  METH-68 now matches METH-55 initialization and exact routes, outputs
  and gradients over three BF16 synthetic steps; dense-semantics CPU
  AdamW keeps maximum post-update error below 5×10⁻¹⁰. Replay the
  complete 16-update METH-55 E128 donor schedule under this offload
  method and compare the frozen checkpoint before training E1280 again.
  METH-69 did that full donor replay: first three updates have exactly
  matching microbatch losses, but a divergence begins at update four
  and the strict final checkpoint/BPB/top-1-count gates fail. The
  replay still clears the old viewed 95% top-1 screen (96.038%),
  so CPU offload can retain useful E128 behavior without exact
  checkpoint identity. Investigate GPU-paged dense AdamW semantics
  as one route to stronger parity, then use new prompts for a
  controlled E128/E1280 quality gate. Do not promote the viewed replay.
  METH-70 then uses the original dense AdamW geometry and identical
  draws on newly frozen prompts. Both arms fail ≥95%: E128 93.888%,
  E1280 92.865%. The 10× bank changes ≥751 B slots/layer but selects
  only 474/1,280 in the least-covered layer and reaches 129.19× worst
  maximum/mean route load. This directly rejects the current scaling
  rule. METH-71 applies a precommitted stronger donor-retention and
  product-key axis-balance objective on new disjoint prompts. Both
  matched 64-update arms pass development gates: E128 reaches 97.535%
  donor top-1, E1280 98.348%. E1280 selects at least 1,076/1,280
  slots per layer on those prompts, with 19.64× worst max/mean load,
  and changes at least 1,219 B slots/layer. These are real independent
  trained factors, but broad use alone is insufficient for promotion.
  METH-72 tests E1280 on 24 source-disjoint documents and all 1,838
  PIQA items. Document BPB, top-1, generation and PIQA gates pass,
  but the fixed factor-pair permutation utility is +0.001743 BPB,
  below its +0.002 threshold; promotion stops. METH-73's eight
  diagnostic permutations all worsen BPB, with median +0.000945
  across external documents and +0.001737 on the raw four-window
  set. That confirms a small pairing signal without reversing the
  METH-72 failure. METH-74's arm-blind semantic diagnosis counts
  26 unsupported claims for E1280 versus 30 donor, but 8 severe
  unsupported claims versus 7 donor. The severe criterion also fails.
  Continue with a frozen stronger-training study and new prompts to
  improve functional route utility and semantic quality; do not
  infer CPU LUT speed or full-model throughput from this GPU test.
  METH-75 resumes both METH-71 checkpoints from update 64 to 256
  under the same rule and matched draws. New development prompts
  exclude every raw row in that entire draw stream. Both arms retain
  broad routing and improve raw BPB, but fail the fixed same-prompt
  non-inferiority gate: E128 falls 97.636%→96.585% and E1280
  98.398%→96.559% donor top-1. The E1280 terminal checkpoint selects
  at least 1,197/1,280 slots per layer, yet ends one matched position
  below E128. Stop this longer continuation; a new fixed rule must
  improve learned route utility without sacrificing donor behavior.
  METH-76 exports the METH-71 update-64 E128/E1280 learned component
  to BF16-factor/FP32-product-key C banks. Both pass 96/96 exact
  native top-four/gate/residual fixtures and negative controls.
  Single-thread 24-layer route medians are 0.978/1.005 ms per
  token-equivalent pass (1.027× at 10× experts); selected top-four
  factor payload stays 2.753 MB/token. This is a hot repeated-input
  component result. The E1280 bank is 886.8 MB, quality still fails
  METH-72/74, and varied-token DRAM traffic, LUT coding, full
  engine integration and accepted-token rate remain open.
  METH-77 samples 256 actual external-prompt positions in both
  matched models and benchmarks varied CPU factor access. E1280
  visits 421–485 unique experts per layer, an estimated 315 MB of
  distinct BF16 factors across the 24 layers, versus E128's 85 MB.
  Route medians remain 1.043/1.071 ms, and selected-factor
  residual medians 1.129/1.097 ms per token-equivalent pass.
  This finite repeated set stresses a larger working set, but
  neither proves cold DRAM latency nor supplies LUT coding or a
  quality-valid full-model throughput result.
  METH-78's frozen objective-gradient readout finds that 0.02× axis
  balance has 62–498× the CE product-key router gradient norm on
  raw/chat training-source examples at E128/E1280, while its factor
  gradient is small. This is a local mechanism diagnostic, not proof
  that balance caused METH-75's retention decline. METH-79 froze a
  24-prompt manifest disjoint from every prior development
  set through METH-75 and the entire matched raw draw stream. The
  preregistered controlled continuation removed balance only after
  update 64, with matched E128/E1280 arms and fixed METH-75
  update-256 comparators. E128 cessation reaches 97.377% new-prompt
  donor top-1 versus 96.931% balanced, with full route coverage.
  E1280 reaches 97.088% versus 96.721% balanced and selects at
  least 1,088/1,280 experts per layer, but the +0.367-point gain
  misses the fixed +0.5-point gate and its −1.548-point decline
  from update 64 exceeds the ≤1.0-point limit. Its external
  promotion stops. The simple balance cessation test improves
  retention modestly; it does not solve large-E quality.
  METH-80 adds a fixed teacher-decision margin to the same continuation.
  On a newly frozen 24-prompt set, E1280 improves 96.563%→97.088%
  versus its no-margin comparator and selects at least 1,126/1,280
  experts per layer. It still declines 1.443 points from its update-64
  parent and trails the E128 candidate by 0.603 point; both exceed
  the preregistered allowances. External promotion stops. The next
  priority for the end-to-end objective is a quality-valid compact
  donor core and native full-artifact path; retain the distinct
  learned large-E evidence as an unresolved parallel constraint.
  METH-81 verifies from donor tensor headers that grouped-Q4 FFN
  with BF16 tied head/attention would charge 535.740 MB ideal
  addressed bytes/token including E128 keys and selected factors.
  METH-82 exports that map to a 527.375 MB packed file with exact
  tensor and BF16 reconstruction readback. METH-83 tests the exact
  composed model on 24 new source-disjoint documents and rejects it:
  pooled ΔBPB +0.045108 and donor prompt top-1 only 68.469%, versus
  96.332% for BF16+E128. Q4 donor alone is 67.802%, so the
  conversion dominates the observed loss. METH-84 tests all 24
  single-FFN-layer BF16 exceptions within the 560 MB ideal allotment
  on the now-viewed prompts; the best reaches 73.803%.
  Stop this Q4 rule before native integration. METH-85 tests a
  group-64 scale correction to the closer BF16-attention/R8-head+FFN
  core. The stored artifact reloads exactly and charges 557.106 MB
  ideal addressed bytes/token with E128, within the 560 MB design
  allotment. METH-86 freezes 24 new source-disjoint documents and
  finds pooled ΔBPB −0.003699, but grouped-core+E128 reaches only
  94.579% donor prompt top-1 against the 95% floor and is 1.487
  points below BF16+E128 against the allowed 1.0 point. It fails
  two gates; stop native promotion. METH-87 posthoc diagnostics on
  viewed prompts find half-strength experts worse (94.219%) and
  an over-budget BF16-head restore still only 94.939% with E128.
  METH-88 adapts only the B output factors of the existing distinct
  E128 bank to this fixed stored core, using the BF16+E128 model as
  teacher. METH-89 passes six frozen development gates on 24 new
  sources: donor prompt top-1 rises from 94.410% with the original
  bank to 95.430% adapted, 0.951 point below BF16+E128, while
  pooled BPB is 0.004560 better than donor. The 557.106 MB ideal
  addressed payload is unchanged. METH-90 freezes another disjoint
  set and passes automatic document, generation and 1,838-item PIQA
  gates (−0.003774 BPB; 22/24 EOS versus donor 23; PIQA 1,287 versus
  donor 1,291). The arm-blind semantic verdict was committed before
  unblinding and fails: 43 unsupported student claims versus 37 donor;
  severe counts 15/15 and missing-detail 0/0 pass. Stop native
  promotion. Next: investigate a core/adaptation rule that preserves
  excerpt-grounded response fidelity on untouched sources, while
  retaining the separate E1280 learned-quality/routing gap. METH-91
  spends 1.519 MB of remaining ideal payload on group-128 FP16 head
  scales and verifies a 558.626 MB/token stored core. METH-93 repeats
  the frozen 64-update B-only adaptation on that core. METH-94's 24
  newly disjoint sources show the head improves expert-disabled donor
  top-1 by 0.533 point, but adapted composition falls to 93.384%
  versus 94.161% for the old-core adapted arm. It fails three of
  seven development gates. Stop this head-precision refinement before
  external promotion. [METH-95/98](METH_95_98_HIERARCHICAL_EXPANSION_RESULT_20260927.md)
  expands the quality-audited E128 bank to E1280 with four selected
  child experts and bitwise-identical step-zero logits. The 128-update
  child specialization changes 1,115–1,263 slots/layer and improves
  fresh-document BPB, but loses 1.109 points of donor prompt top-1
  against the one-point E128 noninferiority gate. [METH-99/101](METH_99_101_HIERARCHICAL_RETENTION_RESULT_20260927.md)
  resumes with stronger teacher retention and leaves 1,072–1,152
  siblings/layer meaningfully distinct; all five new-source
  development gates pass. [METH-103/105](METH_103_105_HIERARCHICAL_EXTERNAL_RESULT_20260927.md)
  passes external document, prompt, generation and 1,838-item PIQA
  automatic gates, but its committed blind semantic verdict finds one
  E1280 answer without a grounded specific detail versus zero for
  E128. Unsupported and severe claims improve (30/8 versus 35/11),
  but the frozen joint semantic gate fails. No quality promotion.
  [METH-104](METH_104_HIERARCHICAL_CPU_ROUTE_RESULT_20260927.md)
  verifies 96 native parent/child route fixtures exactly and measures
  1.477 ms per 24-layer E1280 route versus 0.979 ms E128 (1.508×),
  below its component limits. That checker excludes factor access,
  LUT arithmetic and the full model. [METH-106/109](METH_106_109_LONG_CHAT_RETENTION_RESULT_20260927.md)
  creates 256 up-to-128-token E128 teacher answers and makes 256 B-only
  child updates. The METH-107 checkpoint has 1,084–1,152 meaningfully
  distinct children/layer and passes six fresh METH-109 development
  gates. [METH-110/112](METH_110_112_LONG_CHAT_EXTERNAL_RESULT_20260927.md)
  passes all frozen automatic external gates: pooled BPB improves by
  0.000450, donor prompt top-1 loses 0.942 point, EOS is 19/24 in both
  E128 and E1280, and PIQA falls 1,292→1,287/1,838. The committed
  blind semantic verdict fails: E1280 has 28 versus 27 unsupported
  claims and 7 versus 5 severe claims; missing-detail counts tie 2–2.
  Do not quality-promote METH-107. Its parent/child router weights remain
  METH-99's, so METH-104 route-only measurements still apply.
  [METH-113/114](METH_113_114_CHILD_RESIDUAL_CALIBRATION_RESULT_20260927.md)
  selects alpha=0.75 on a new development set, retains 1,069–1,152
  distinct children/layer, and passes a matched no-child utility
  ablation plus precommitted blind semantics. [METH-115/117](METH_115_117_ALPHA075_EXTERNAL_RESULT_20260927.md)
  repeats the child-choice BPB gain on eight separate new PG19 books
  (0.000084 versus parent-mean children) and passes all automatic
  gates, including PIQA 1,288 versus 1,292/1,838 for E128. Its
  committed blind semantic verdict fails: E1280 has 40 versus 33
  unsupported claims and 15 versus 14 severe claims, with zero
  missing-detail answers in either arm. Do not promote alpha=0.75 or
  lower alpha on the viewed METH-113/METH-115 sequence.
  [METH-118/120](METH_118_120_ZERO_MEAN_CHILD_DEVELOPMENT_RESULT_20260928.md)
  removes the per-parent child B common mode while preserving the
  METH-56 E128 parent mean, retains 1,084–1,152 distinct children/layer,
  and passes automatic plus precommitted blind development gates on
  new PG19-shard-2 books and nonoverlapping code/technical fragments.
  [METH-121/123](METH_121_123_ZERO_MEAN_CHILD_EXTERNAL_RESULT_20260928.md)
  repeats on PG19-shard-3 books and fresh fragments. E1280 improves
  pooled BPB by 0.000350 versus E128, loses 0.690 point of donor top-1,
  gains one EOS (20 versus 19), and scores 1,291 versus 1,292/1,838
  on PIQA. Its committed blind external verdict improves unsupported
  claims 31→25 and severe claims 12→7, with missing-detail 0/0.
  [METH-124](METH_124_CENTERED_FACTOR_CPU_RESULT_20260928.md)
  exports the actual centered A/B bank and checks 96 native parent,
  child, gate and BF16 residual fixtures with zero error; a one-byte
  fixture mutation fails. Its hot 24-layer route+factor median is
  2.450 ms/token-equivalent. [METH-125](METH_125_VARIED_CENTERED_FACTOR_CPU_RESULT_20260928.md)
  captures 256 matched source positions per arm and measures E1280
  route+factor 2.510 ms versus E128 2.017 ms (1.245×). It visits
  252–402 E1280 slots/layer and an estimated 233.5 MB of distinct
  factor rows. This passes the frozen CPU component gate. The bank
  uses BF16 factors and FP32 routing, not compact LUT codes.
  [METH-126](METH_126_SHARED_A_FACTOR_BANK_RESULT_20260928.md) verifies
  all 27,648 sibling A byte comparisons and stores A once per parent,
  reducing the exact E1280 bank from 893.1 to 496.8 MB. Native fixture
  outputs and varied-position routes/checksum are unchanged; distinct
  factor bytes touched fall 233.5→151.6 MB. This is a storage and
  working-set result, not a proven token-rate improvement. Full
  [METH-127](METH_127_FULL_C_REFERENCE_RESULT_20260928.md) composes a
  pinned FP32 Qwen core and the METH-126 E1280 bank in the existing Qwen
  C runtime. Dense/composed top-1 parity is 8/8 and 64/64 on one bound
  prompt; composed 64-position worst relative logit L2 is 3.69e-4.
  Greedy 64-token decode runs at 16.818
  tok/s on six Ryzen 5 3600X threads versus 18.244 dense. The FFN and
  tied head dominate; the expert path adds about 4 ms/token. This closes
  the full-path reference integration question, not native quality or
  speed promotion. A quality-valid compact core, target `engine.c`,
  >=50 accepted tok/s, 10B/100B transfer and cold random DRAM remain
  open. Next: bind a low-traffic Instruct core to the exact METH-126
  bank, use new source-disjoint document/generation/grounding gates,
  and then measure native full-path rate on that same representation.
  Prior R8/grouped-Q4 failures require an explicit quality repair,
  not another untrained format swap.
  [METH-128](METH_128_EXACT_HEAD_RERANK_RESULT_20260928.md) finds a
  new head option on the viewed quality-gated E1280 hidden states:
  the original BF16 top-1 is in the stored per-row R8 head's top-16
  at all 4,494 positions, and exact BF16 candidate rescoring with
  lowest-ID tie breaking reproduces 4,494/4,494 choices. This is a
  representation screen; the predicted ~135 MB/token head traffic
  reduction needs a native C kernel, full-model quality on new
  sources and a compact-body combination.
  [METH-129](METH_129_NATIVE_EXACT_HEAD_RESULT_20260928.md) implements
  that native head. Its K=64 choices match the full FP32 head on all
  64 bound prompt positions and the entire 64-token greedy stream at
  one and six threads. Six-thread decode rises 17.141→20.791 tok/s
  (+21.3%); head time falls 14.491→4.179 ms/token. The corrupted-header
  control fails as required. The original FP32 head and 136.7 MB R8
  sidecar are both resident. Approximate tail logits cannot support
  full-head likelihood, and no fresh semantic quality is claimed.
  Next: bind the head to a quality-valid compact body and exact bank,
  then audit new disjoint documents, generation, tasks and grounding
  before native rate and RAM-scaled expert-count tests. No T4 job is
  planned.
  [METH-130](METH_130_Q7_FACTOR_LUT_RESULT_20260928.md) applies a
  seven-level pair-LUT code only to child B in the same E1280 bank.
  On 6,144 actual fixed-state residuals, pooled/p95 relative L2 is
  0.0465/0.0861 with exact routes and gates. The bank falls
  496.8→276.6 MB, and five paired CPU repetitions give 0.8825 versus
  1.0388 ms/token-equivalent for the factor computation. The invalid
  nibble control fails. It licensed a full-model quality test, not a
  claim that E12800 contains useful distinct learned experts.
  [METH-131](METH_131_Q7_FULL_MODEL_QUALITY_RESULT_20260928.md) bound
  the stored Q7 bank into BF16 E1280 inference and selected 24 new
  disjoint excerpts. Pooled BPB differs by +0.000085, greedy EOS is
  23/24 in both arms, and PIQA is 1,288 versus 1,291: all automatic
  gates pass. The committed blind verdict, unblinded afterward, gives
  Q7 34 versus 33 unsupported and 14 versus 12 severe claims, failing
  both frozen semantic gates. Do not integrate this Q7 bank as a
  quality-valid target. Next: test a higher-resolution nibble code or
  another B representation on component states, then use **new**
  documents and blind grounding before promotion. A compact donor core,
  native same-artifact accepted-token rate and a learned expert-count
  ladder remain required.
  [METH-132](METH_132_Q15_FACTOR_LUT_RESULT_20260928.md) fills the same
  four child-B nibbles with fifteen signed levels. The 276.6 MB bank
  passes the native component gate on 6,144 fixed states: zero route/gate
  mismatches, pooled/p95 residual relative L2 0.01894/0.03690, and
  five-pair median factor time 0.8738 versus 1.0511 ms/token-equivalent
  for exact BF16. The reserved-nibble negative control fails as required.
  This reduces component error without a larger bank, but **does not
  repair METH-131's Q7 semantic failure or establish Q15 full-model
  quality**. Next: preregister a Q15 audit on new source-disjoint
  documents and blind grounding, then check native full-model parity
  and rate only if it passes. The compact core and RAM-scaled learned
  expert ladder remain open.
  [METH-133](METH_133_Q15_FULL_MODEL_QUALITY_RESULT_20260928.md) ran
  that fresh Q15 audit on 24 source/fragment-disjoint excerpts and the
  repeated 1,838-item PIQA task. Automatic gates pass: pooled BPB
  delta +0.000170, donor prompt top-1 +0.258 point, greedy EOS 20/24
  versus 21/24, and PIQA 1,283 versus 1,291. The committed blind
  excerpt verdict fails after unblinding: Q15 has 50 versus 39
  unsupported and 24 versus 16 severe claims; missing detail is 0/0.
  Do not quality-promote Q15. The exact METH-126 BF16 factor bank is
  still the grounded reference. Any new precision method needs new
  source-disjoint data; do not tune against METH-131/133 responses.
  Prioritize a compact quality-valid core and the learned RAM-scaled
  expert ladder; CPU LUT/full-model rate claims remain unproven.
  [METH-134](METH_134_SPARSE_E12800_TRAINING_RESULT_20260928.md)
  checks an executable CPU-master sparse-gradient path for a third
  tier below actual centered E1280 children. On 256 real layer-0
  states, 12,800 cloned grandchildren exactly preserve FP32 factor
  output; a duplicate-ID gradient oracle and one-step update pass,
  with only 467 unique rows carrying a gradient. The 24-layer FP32
  CPU B master is projected at 8.81 GB, not measured. Next: cost the
  actual 10× third-tier CPU route and selected factors, then register
  a bounded full-model sparse training/quality experiment with new
  sources. This is trainability apparatus, not learned E12800 quality.
  [METH-135](METH_135_THIRD_TIER_CPU_ROUTE_RESULT_20260928.md)
  measures the untrained ten-way third tier on all 6,144 actual
  fixed states. The 42.1 MB sidecar passes byte readback, cloned
  factor residuals match exactly, and five paired single-thread
  CPU repetitions give 2.002 versus 1.483 ms/token-equivalent for
  E12800/E1280 route (1.350×, below 2× and 3 ms limits). The bad
  magic control fails. Next: freeze a bounded **learned** E12800
  sparse-training run, including all 24-layer memory/optimizer costs,
  route load and new-source quality/grounding. Do not infer cold DRAM,
  quality-valid LUT, full rate or 10B/100B transfer from this gate.
  [METH-136](METH_136_MATCHED_E12800_SPARSE_TRAINING_RESULT_20260928.md)
  completes matched 256-update E1280/E12800 training from exact BF16
  parity. The candidate has 7,996–11,488 selected and
  11,820–12,780 BF16-distinct rows per layer; its 4.40 GB bank is
  hashed. The frozen maximum/mean route-load gate <=50 fails after
  export. Its exact train-time candidate skew was not saved, so no
  quality promotion or METH-137 audit follows. The run used 27.1
  minutes, 33.9 GB RSS and 5.04 GB peak allocated GPU memory locally.
  [METH-138](METH_138_FAILED_ROUTE_LOAD_DIAGNOSTIC_RESULT_20260928.md)
  replays both final banks on the same training inputs: all 24
  candidate layers exceed 50×, with worst load 287.349× versus
  56.417× control. The hottest layer-17 grandchild gets 50.8% of its
  parent's selections. Candidate parent-aggregated skew is 56.516×,
  localizing most additional concentration to the seeded third tier.
  This replay is not the exact changing-weight training histogram.
  Next: register a traffic-aware third-tier allocation/selection rule,
  keep CPU routing affordable, then retrain and audit on new sources.
  [METH-139](METH_139_QUANTILE_THIRD_ROUTER_RESULT_20260928.md)
  replaces seeded grandchild argmax with child-specific decile
  thresholds fitted on the first 128 training draws and screened on
  the last 128. Exact cloned full-model BF16 parity and coverage pass;
  the 3.86 MB sidecar is read back exactly. Its worst candidate/control
  load ratio is 1.311× in one layer versus the frozen <=1.25× gate,
  and one other layer has 26.94% share versus <=25%. This screen fails;
  no new E12800 training or quality promotion follows. Next: freeze a
  broader training-only calibration and genuinely source-held-out
  routing screen before a new training rung. CPU cost remains open.
  [METH-140](METH_140_EXTERNAL_QUANTILE_ROUTE_RESULT_20260928.md)
  fits the same conditional deciles on all 256 training draws and
  checks METH-121 and METH-133 document routes separately. Initial
  cloned BF16 logits and >6,000 selected grandchildren per layer
  pass. Source-held-out balance fails broadly: worst candidate/control
  load ratio 3.214×; worst within-parent share 88.09%; almost every
  layer fails the relative skew gate and all fail the share gate on
  both manifests. The 3.86 MB sidecar is read back exactly but is
  rejected before native CPU cost or B training. Next: address long
  context/source distribution shift with training-only calibration or
  a routing mechanism that generalizes without scalar deciles.
  [METH-141](METH_141_CONTEXT_DOMAIN_ROUTE_DIAGNOSTIC_RESULT_20260928.md)
  fixes METH-140's rejected sidecar and compares H0 raw versus
  METH-121/133 documents at 128 and 512-token windows. Worst
  candidate/control load ratios are 1.670/2.023× for H0,
  2.798/3.198× for METH-121, and 2.823/3.214× for METH-133.
  Both longer context and source shift increase imbalance; the
  latter is visible even at 128 tokens. More samples from the same
  short training mixture are not a sufficient repair. No quality
  promotion or native CPU cost claim follows.
  [METH-142](METH_142_TOKEN_HASH_ROUTE_RESULT_20260928.md)
  uses a fixed 64-bit hash of token context, position, source child
  and layer to choose one of ten grandchildren, without fitted
  thresholds. All six previously viewed load cells pass: worst
  candidate/control skew 1.240×, worst hot-parent share 20.47%,
  minimum per-layer coverage 6,992. The source child and E1280
  teacher/control BF16 logits remain bound, but the hashed choice was
  computed offline; a hash-routed model, new source load test, native
  C cost, trained specialization and external quality are next.
  [METH-143](METH_143_FRESH_HASH_ROUTE_RESULT_20260928.md) applies the
  unchanged METH-142 hash to 24 newly selected source/fragment-disjoint
  documents, in 128- and 512-token windows. All frozen load gates pass:
  worst candidate/control skew 1.1915×, worst hot-parent share 18.60%,
  and minimum per-layer coverage 7,182. This is still an offline choice
  after the original E1280 route. Next: frozen hash-routed full-model
  clone parity and native C CPU cost, then matched sparse B training and
  untouched quality if those gates pass.
  [METH-144](METH_144_INTEGRATED_HASH_PARITY_CPU_RESULT_20260928.md)
  passes both gates: all logits/routes/gates match exactly on 16 bound
  and fresh prompts with every grandchild B cloned from its source,
  and C/Python agree on all 6,144 actual-state route records. Five
  paired native CPU repetitions yield 1.545 versus 1.594 ms/token-
  equivalent for hash E12800 versus E1280, within the <=2× and <=3 ms
  limits. This is a warm routing component result; next freeze matched
  sparse B training and then untouched quality/grounding. Useful
  learned E12800 capacity, compact LUT arithmetic, cold DRAM and
  full-model accepted-token rate remain unproven.
  [METH-145](METH_145_MATCHED_HASH_E12800_TRAINING_RESULT_20260928.md)
  reuses the frozen 256-update E1280 control and completes hash-routed
  B-only E12800 training with exact initial parity. Coverage and BF16
  distinctness pass, but 20/24 layers fail the <=1.25× relative load
  gate (worst 2.875×), and all 24 fail the <=25% hot-parent share gate
  (worst 100%). The 4.40 GB readback-verified candidate is rejected;
  do not run METH-146 fresh quality on it. All 256 training chat
  sequences have an identical 55-token prefix, a likely source of
  concentrated deterministic routes. Next: attribute load by input
  segment and repair the training/routing mechanism under a new gate.
  [METH-147](METH_147_TEMPLATE_ROUTE_ATTRIBUTION_RESULT_20260928.md)
  replays that rejected final bank on the same draws. Raw traffic has
  worst hot-parent share 17.73%, the identical 55-token chat prefix
  reaches 100%, and the nominally variable chat remainder still
  reaches 96.58%. Repeated assistant/prompt scaffolding remains after
  position 55. Next: separate prompt structure from answer continuation
  and design a structural/shared route or different training mixture;
  total CPU traffic must still be counted. METH-145 stays rejected.
  [METH-148](METH_148_CHAT_MASK_ROUTE_ATTRIBUTION_RESULT_20260928.md)
  uses the frozen next-token loss mask to separate prompt scaffold
  from answer context. The latter still reaches 99.20% hot-parent
  share (minimum across layers 32.20%); repeated response openings
  remain. A 55-token-prefix bypass alone cannot fix this. Next:
  isolate strongly recurring causal contexts and preregister a shared
  structural route plus balanced content specialists if supported.
  [METH-149](METH_149_RECURRENT_CONTEXT_ROUTE_RESULT_20260928.md)
  freezes 75 token/previous/position tuples recurring in at least 32
  of 256 chats before replay. They cover 16,629/54,095 chat input
  positions and alone reach 100% hot-parent share. The remaining
  37,466 chat positions have worst share 18.46% and worst slot-versus-
  own-parent max load 1.168×; raw windows are similarly balanced.
  This is postfailure diagnostic evidence for a shared structural
  route, not a pass for METH-145 or permission to open METH-146.
  Next: freeze and cost a new shared-structure/content-specialist
  router, then retrain from exact BF16 and test untouched quality.
  [METH-150](METH_150_SHARED_STRUCTURE_ROUTE_SCREEN_RESULT_20260928.md)
  freezes that 75-tuple table and a shared local slot 0, hashing all
  other contexts into nine content slots. Content-only gates pass on
  matched raw/chat draws and 24 newly selected documents at two widths:
  worst slot-versus-own-parent load 1.171×, worst hot-parent share
  19.69%, minimum coverage 6,516. The chat shared path takes 66,516
  selections/layer and full all-token skew reaches 374.81×. This is
  an offline screen; full-model clone parity, native table/hash cost,
  fresh chat prompts, trained B usefulness and quality remain open.
  [METH-151](METH_151_SHARED_ROUTE_MODEL_NATIVE_RESULT_20260928.md)
  puts the shared route inside all 24 expert forwards and passes exact
  cloned-B BF16 logits/routes/gates on 17 bound/new-source sequences.
  C/Python agree on every route in both all-hit and all-miss fixtures;
  paired native CPU route medians are 1.596 and 1.568 ms/token-
  equivalent, each within <=2× E1280 and <=3 ms limits. These are
  component results with instrumentation token contexts. Next:
  freeze matched structural/content B-only training and then untouched
  quality/grounding; cold factors, LUT quality and full accepted-token
  rate are still open.
  [METH-152](METH_152_MATCHED_SHARED_E12800_TRAINING_RESULT_20260928.md)
  completes the matched 256-update shared-route E12800 B-only run against
  the frozen continued E1280 control. All preregistered training gates
  pass: minimum content coverage 8,673, worst nine-way content load
  ratio 1.104, hot-parent share 19.17%, and minimum BF16-distinct rows
  11,820. The structural slot takes 66,516/346,428 selections per layer;
  all-token skew reaches 234.14× and is reported separately. The exact
  4.40 GB bank is local and must be preserved. Next: run the already
  frozen [METH-153](METH_153_SHARED_E12800_FRESH_QUALITY_PROTOCOL_20260928.md)
  source/fragment-disjoint quality audit. No useful-expert or native
  accepted-token claim yet.
  [METH-153](METH_153_SHARED_E12800_FRESH_PREDICTION_RESULT_20260928.md)
  freezes 24 new source/fragment-disjoint documents and fails its first
  quality gate: pooled candidate-minus-continued-control BPB is
  +0.000183 against the required <=-0.00005, and the paired bootstrap
  lower gain bound is negative. Donor-top1 retention and all three
  category regression limits pass. Stop before generation/PIQA/blind
  grounding; METH-152 establishes trainable balanced content experts,
  not useful added capacity. Do not reuse the METH-153 sources to
  select another candidate. Next: preregister a matched shared-base
  plus conditional-residual training/control experiment, then evaluate
  on a newly reserved disjoint quality set.
  [METH-154](METH_154_POSTFAILURE_BANK_DECOMPOSITION_RESULT_20260928.md)
  confirms from bound training artifacts that the E1280 control learned
  a common B shift (RMS 8.26–10.53e-5/layer), whereas the exported
  E12800 grandchild mean was restored to the original B (RMS at most
  1.97e-6) despite nonzero specialist differences. This supports a
  shared-base-plus-residual hypothesis, not a causal quality claim.
  [METH-155](METH_155_MATCHED_SHARED_BASE_RESIDUAL_TRAINING_RESULT_20260928.md)
  trains a shared B base and conditional residuals on the same 256 draws.
  A corrected run with exact combined-bank audits after every update
  passes all frozen training gates: minimum content coverage 8,673,
  worst content load 1.127, hot-parent share 17.94%, minimum changed
  base rows 1,182 and minimum BF16-distinct combined rows 11,820. The
  exported ten-child mean now has a learned shift similar to the E1280
  control. The structural slot still takes 19.20% of choices and
  all-token skew reaches 234.25. Next: apply only the already frozen
  [METH-156](METH_156_SHARED_BASE_FRESH_QUALITY_PROTOCOL_20260928.md)
  to new source/fragment-disjoint data. No useful-expert claim yet.
  [METH-156](METH_156_FACTORIZED_E12800_FRESH_PREDICTION_RESULT_20260928.md)
  fails the first external quality gate on 24 new shard-8/code/technical
  sources: candidate-minus-continued-control pooled BPB is +0.000128
  versus required <=-0.00005, with a negative bootstrap lower gain
  bound. Category regression caps and donor-top1 retention pass; the
  technical category has a small BPB gain, but pooled quality does not.
  Stop before generation/PIQA/blind grounding. The shared-base repair
  fixes the lost common shift but is insufficient to show useful 10×
  expert capacity. METH-153/156 external sets are consumed. Next:
  determine whether training signal per specialist is limiting, then
  freeze a data-scale or objective change and a new independent set.
  [METH-157](METH_157_POSTFAILURE_TRAINING_SUPPORT_RESULT_20260928.md)
  measures METH-155 changing-weight selection support. Active content
  rows have a median of only 7–13 selections/layer; 76.5–84.6% of all
  11,520 content rows/layer receive fewer than 32, including 538–2,847
  never selected. This makes data support a plausible limiter but is
  not a causal explanation. Next: acquire independent training
  contexts at a larger matched budget; do not repeat the same 256 chat
  continuations or reuse the consumed METH-153/156 quality sources.
  [METH-158](METH_158_INDEPENDENT_TEACHER_ACQUISITION_RESULT_20260928.md)
  completes 40 verified teacher shards: 2,560 new chat continuations
  and 2,560 raw windows from distinct H0 rows, bound by the merged SHA.
  [METH-159](METH_159_TENFOLD_ROUTE_SUPPORT_PREFLIGHT_RESULT_20260928.md)
  replays all 887,330 input tokens and **fails** the frozen active-row
  median and hot-parent share gates, despite 10,829 minimum content
  coverage and 1.038 worst global load. Do not start the proposed long
  matched training with this route. PG19 shard 9 remains unused for a
  later independent quality audit.
  [METH-160](METH_160_ACTUAL_B_BANK_CPU_COST_RESULT_20260928.md)
  passes the separate actual-B-bank native component gate in both
  structural-hit and content-miss fixtures: E12800 route plus BF16
  factors takes 3.964/3.918 ms versus E1280 3.939/3.955 ms per token-
  equivalent. Pages are warm and the E12800 bank is quality-rejected;
  no compact LUT or full accepted-token rate is established.
  [METH-161](METH_161_SOURCE_ROUTE_TRACE_RESULT_20260928.md) captured
  all 887,330 input positions' four E1280 source-child IDs in each of
  24 layers. All METH-159 all/content histograms replay exactly. The
  hashed arrays are recoverable from `meth161_source_route_trace.zip`.
  [METH-162](METH_162_STRUCTURAL_TABLE_DIAGNOSTIC_RESULT_20260928.md)
  attributes 146/186 selections in the original worst slot to one
  recurring causal tuple. Adding all 63 nonshared tuples seen >=32 times
  lowers the in-sample worst hot-parent share to 24.26%, but minimum
  active content median remains 39 <50. The 138-tuple table is a
  candidate only, not a passed training route.
  [METH-163](METH_163_PREFIX_HASH_DIAGNOSTIC_RESULT_20260928.md)
  spreads that specific collision but creates a 35.81% worst hot-parent
  share in another layer and leaves median support at 39. Reject this
  prefix-only candidate.
  [METH-164](METH_164_CROSS_SOURCE_ROUTE_VALIDATION_RESULT_20260928.md)
  passes every frozen table-load gate on 128 new PG19 shard-10 chat
  prompts, 128 raw windows and 24 documents segmented at 128/512.
  Candidate worst hot-parent share is <=20.07%; old table also passes,
  so this is independent nonregression evidence, not a replicated repair.
  PG19 shard 9 remains unused for external quality.
  [METH-165](METH_165_EXPANDED_SUPPORT_RESULT_20260928.md) completes
  all 20 BF16 teacher shards and the exact route replay on 1,330,096
  pooled input tokens. Coverage 10,999, minimum active median 56,
  maximum under-32 fraction 41.03% and global load ratio 1.026 pass;
  worst hot-parent share 32.32% fails. Layer 20/source child 656/local 4
  receives 85/263 content selections. The old 2,560 and new 1,280
  subsets fall below the >=250 hot-parent definition separately, so
  their passes cannot be composed. No matched long training follows.
  [METH-166](METH_166_NATIVE_TABLE_COST_RESULT_20260928.md) passes
  native actual-state lookup cost: 138-entry table route medians
  2.484–2.492 ms/token on three context cells, ratios 0.995–0.998×
  the paired 75-entry table (timing noise). This is warm CPU routing
  only; compact LUT factors and end-to-end speed remain unmeasured.
  [METH-167](METH_167_EXPANDED_SOURCE_TRACE_RESULT_20260928.md) captures
  the new 1,280 pairs' exact source children and reproduces every METH-165
  raw/chat/combined histogram in all 24 layers. Its five arrays are in a
  SHA-verified ZIP. [METH-168](METH_168_CHAT_BOUNDARY_ROUTE_DIAGNOSTIC_RESULT_20260928.md)
  replays one frozen tokenizer-delimiter shared-slot rule over all 3,840
  training pairs. Minimum active median remains 56 and worst pooled hot
  share falls to 23.113%; every in-sample gate passes. It was selected
  after METH-165 failed, so it needs a prospective test.
  [METH-169](METH_169_FRESH_SOURCE_BOUNDARY_ROUTE_PROTOCOL_20260928.md)
  froze 512 chat, 512 raw and 24 document inputs from previously unused
  PG19 shard 11 in manifest SHA
  `8b2e3a545f33dbd31283fd173b9044e2c87bd55463e168d6b2f8c513cde4e468`.
  Its local BF16 teacher completed all 512 responses. The
  [METH-169 result](METH_169_FRESH_SOURCE_BOUNDARY_ROUTE_RESULT_20260928.md)
  passes all four new-source cells and the raw+chat pool, but **fails**
  the train+new-source hot-parent gate: layer 3/child 215/local 3 has
  82/302 selections (27.15% >25%). The new and training subsets are
  each below the >=250 hot-parent definition, so their separate passes
  cannot license the long matched training. METH-170 native timing is
  conditional on a route pass and does not run. [METH-171](METH_171_COMPOSED_HOT_PARENT_TRACE_PROTOCOL_20260928.md)
  [METH-171](METH_171_COMPOSED_HOT_PARENT_TRACE_RESULT_20260928.md)
  captures the exact new-source children and reproduces all METH-169
  raw/chat/combined histograms. One chat excerpt-boundary tuple, comma
  after newline at position 56, accounts for 57/82 selections in the
  failed slot across old, expanded and prospective inputs. The 16
  prospective occurrences come from 16 distinct sources. This is
  diagnostic only; a new rule needs another untouched prospective route
  set. [METH-172](METH_172_LOW_RECURRENCE_ROUTE_RESULT_20260928.md)
  freezes a generic >=8-recurrence table of 1,685 tuples from only the
  original 2,560 pairs, without inserting the observed comma tuple by
  hand. Exact replay on training plus consumed shard-11 raw/chat passes
  every in-sample gate: worst composed hot share 19.49%, minimum active
  median 62 and maximum structural traffic 21.05%. This remains
  postfailure evidence. [METH-173](METH_173_FRESH_LOW_RECURRENCE_ROUTE_PROTOCOL_20260928.md)
  froze 512 chat, 512 raw and 24 document inputs on unused PG19 shard
  12, manifest SHA
  `6ebfc36fd452f2a20cfcc9e8210505e36b725ef95992949655c19f3c94eaee20`.
  The [METH-173 result](METH_173_FRESH_LOW_RECURRENCE_ROUTE_RESULT_20260928.md)
  passes all four new-source cells and the composed pool: worst hot
  share 20.32%, minimum content coverage 11,029, active median 61 and
  maximum structural traffic 21.05%. The conditional
  [METH-174 native result](METH_174_NATIVE_LOW_RECURRENCE_ROUTE_RESULT_20260928.md)
  passes exact Python/C fixture parity and all four warm actual-state
  CPU cost cells. Its 20,236-byte table gives 2.427–2.470 ms/token
  route medians, at most 1.002× the paired 1,672-byte baseline within
  measurement noise. These passes license freezing a matched long
  E1280/E12800 learning and untouched quality protocol, not a useful-
  expert, compact LUT or >=50 accepted-token/s claim. PG19 shard 9
  remains untouched for quality.
  [METH-175](METH_175_MATCHED_LONG_LOW_RECURRENCE_TRAINING_PROTOCOL_20260928.md)
  now freezes 3,840 paired updates from the disjoint METH-158/165
  training inputs, order SHA
  `9d4e06eced3cc1c51bac4e6219013f37f4495acd10c87827b567f0966f5c490c`.
  The [16-update pilot](METH_175_LONG_TRAINING_PILOT_RESULT_20260928.md)
  passes BF16 parity, route integrity, optimizer/combined-bank audits and
  exact bank readback within local RAM/GPU/time caps; its sparse support
  is not a final training result. The first full E1280 control attempt
  stopped at update 16 because an inherited projected-time guard counted
  both arms in one process. The [failure record](METH_175_CONTROL_FIRST_ATTEMPT_20260928.md)
  preserves that result and explains the single-arm guard correction.
  The restarted full E1280 control reached update 1,408 without a
  training error, then Windows shut down at 00:30 on September 29.
  Both process handles ended with no final bank. The
  [exact-resume amendment](METH_175_CHECKPOINT_AMENDMENT_20260929.md)
  adds bounded CPU optimizer/bank snapshots. The
  [exact-resume pilot](METH_175_EXACT_RESUME_PILOT_RESULT_20260929.md)
  reproduces all 16 update records and the uninterrupted pilot's exact
  4.404 GB BF16 bank SHA after a stop at update 8. The
  [full matched result](METH_175_MATCHED_LONG_TRAINING_RESULT_20260930.md)
  now completes both 3,840-update arms within the frozen time/RAM/GPU
  caps and passes all training/support/artifact gates. The E12,800
  candidate changes at least 12,720 BF16 rows per layer and has active
  content median 56, but in-sample loss does not prove useful capacity.
  Both final bank hashes were independently re-read from disk. The
  [METH-176](METH_176_LONG_TRAINING_FRESH_QUALITY_PROTOCOL_20260928.md)
  then froze 24 new source-disjoint quality items, including PG19 shard 9,
  only after both arms completed. The
  [fresh prediction result](METH_176_LONG_FRESH_PREDICTION_RESULT_20260930.md)
  improves pooled candidate-minus-control BPB by 0.000114, but its paired
  bootstrap 5th-percentile gain is -0.0000419 and fails the required
  positive bound. Category nonregression and donor-top-1 retention pass.
  The frozen protocol therefore stops before generation, PIQA and blind
  grounding. Shard 9 is consumed by this adjudication. Useful added
  experts, CPU LUT/full-model rate and 10B/100B transfer remain unproved.
  [METH-177](METH_177_LARGE_RAM_LUT_POOL_PROTOCOL_20260930.md) tests the
  existing synthetic `engine.c` ternary selected path with fully initialized
  5.505/55.050 GB pools at E12,800/E128,000. The
  [result](METH_177_LARGE_RAM_LUT_POOL_RESULT_20260930.md) measures
  429.945/562.925 µs/token at six threads: 1.309× exceeds the frozen
  1.25× relative limit while the 5 ms absolute component ceiling passes.
  These random synthetic codes and IDs do not establish useful specialists,
  full model speed or the larger E273,547/100B rung.
  [METH-178](METH_178_TRAINED_CHILD_DIVERSITY_PROTOCOL_20260930.md)
  streams the exact METH-175 banks and training selection counts. Its
  [result](METH_178_TRAINED_CHILD_DIVERSITY_RESULT_20260930.md) finds
  only 1.286% of unweighted and 2.318% of route-weighted candidate-minus-
  control squared content-B difference within the nine siblings; most
  shift is shared at parent level. This is a weight diagnostic, not a
  functional or held-out quality verdict.
  [METH-179](METH_179_CONTENT_ROUTE_FUNCTION_PROTOCOL_20260930.md)
  scores the exact trained bank and all eight content-child cyclic route
  rotations on consumed METH-173 documents. The
  [result](METH_179_CONTENT_ROUTE_FUNCTION_RESULT_20260930.md) finds
  no functional alignment: exact BPB is 1.067669, the mean rotation is
  0.000015 better, and the paired bootstrap lower bound for a true-route
  advantage is -0.000145. This diagnostic rules out merely asserting that
  changed child rows are useful; it is not untouched quality evidence.
  [METH-201](METH_201_CHILD_FUNCTION_SPREAD_PROTOCOL_20260930.md)
  applies all trained content siblings to matched real states and gates.
  Its [result](METH_201_CHILD_FUNCTION_SPREAD_RESULT_20260930.md)
  finds only 1.583% gate-weighted sibling/mean residual spread and
  0.00396% sibling/full-MLP spread; zero of 24 layers passes its
  functional-diversity gate. [METH-202](METH_202_CHILD_DEVIATION_GAIN_PROTOCOL_20260930.md)
  then scales only those deviations under the exact route. Its
  [result](METH_202_CHILD_DEVIATION_GAIN_RESULT_20260930.md) gives
  λ=0/1/4 BPB 1.067704/1.067669/1.067793 on viewed documents;
  λ=4 fails both directional gates. More fixed-route residual
  amplification is stopped. Neither diagnostic opens a new quality set.
  [METH-180](METH_180_GIGACHAT_EXPERT_RANK_SCREEN_PROTOCOL_20260930.md)
  samples 27 BF16 expert projections from the pinned pretrained GigaChat
  10B source, after reviewing the paused donor-adaptation source binding
  and quality baselines. Its [result](METH_180_GIGACHAT_EXPERT_RANK_RESULT_20260930.md)
  rejects a direct unweighted rank-192 int8 factor export: optimal
  Frobenius energy is 43.320–54.113%, median 47.243%, against frozen
  90% per-matrix and 95% median gates. Rank 768 reaches only 94.154%
  median at 2× ideal Q4 factor bytes. [Raw rows](meth180_gigachat_expert_rank_result.json)
  bind every tensor and source shard. The pre-SVD API failure is preserved
  separately; its correction did not select a rank. This is a weight-only
  screen, not full-model quality, native speed or 10× expert scaling.
  The frozen [donor-adaptation pause record](../donor_adaptation/PAUSE_20260925.md)
  keeps its 54/64-checkpoint C parity line and normalization diagnostic
  suspended; METH-180 did not resume or rerun that line. Prior METH-07/09
  organ and quantization ablations already show that lowering expert
  precision alone is insufficient and that MLA sensitivity is material.
  [METH-181](METH_181_GIGACHAT_DIAGONAL_ACTIVATION_RANK_PROTOCOL_20260930.md)
  fits rank-192 factors using separate English/code/technical BF16 input
  second moments and evaluates their diagonal output-error proxy on
  disjoint Cyrillic input moments. Its [result](METH_181_GIGACHAT_DIAGONAL_ACTIVATION_RANK_RESULT_20260930.md)
  rejects this direct activation-weighted variant: median test energy
  49.152%, minimum 42.433%, and worst comparator loss 4.227 points;
  all miss frozen gates. The [27 raw rows](meth181_gigachat_diagonal_rank_result.json)
  and source hashes are saved. Even a test-domain diagonal oracle has
  only 52.297% median rank-192 energy, so domain shift alone cannot
  explain this proxy failure. No full model was exported or scored.
  [METH-182](METH_182_GROUP64_R8_NATIVE_FFN_PROTOCOL_20260930.md)
  exports the existing METH-85 grouped-R8 FFN bytes exactly and measures
  a six-thread AVX2 C kernel over 256 actual METH-125 BF16 E1280 states
  per layer. Its [result](METH_182_GROUP64_R8_NATIVE_FFN_RESULT_20260930.md)
  misses both the 10 ms 24-layer component budget (12.829 ms median)
  and the 1% median relative output-error gate (1.896%). The existing
  grouped core had separately failed blind semantic quality at METH-90;
  this test does not rerun or overturn that verdict. Its inputs are real
  donor/expert states but not actual quantized-core trajectories; no
  end-to-end rate follows.
  [METH-183](METH_183_E1280_CHILD_ROUTE_ALIGNMENT_PROTOCOL_20260930.md)
  tests whether the quality-valid E1,280 child bank benefits from its
  learned content route. The [result](METH_183_E1280_CHILD_ROUTE_ALIGNMENT_RESULT_20260930.md)
  reproduces the original 24 document nats exactly and finds the exact
  route better than all nine within-parent cyclic shifts; the paired
  bootstrap lower fifth percentile of mean-shift disadvantage is
  +0.0001601 BPB. This differentiates the learned E1,280 bank from the
  hash-routed E12,800 METH-179 failure. The documents were already
  consumed by METH-121, and no third-tier learning, 10B transfer or
  native full-model timing follows. Resume with a content-coupled
  load-controlled third-tier protocol and source-held-out load screen;
  separately price and validate the whole pretrained-to-native path.
  An earlier forward-level shift let downstream parent routes change and
  is retained as invalid; the reported run replays exact parent IDs and
  gates in every causal window.
  [METH-184](METH_184_CONTENT_HASH_THIRD_ROUTE_PROTOCOL_20260930.md)
  freezes a ten-way normalized content-score plus deterministic hash-noise
  screen, fitting one of five noise strengths on training raw/chat only.
  Its [result](METH_184_CONTENT_HASH_THIRD_ROUTE_RESULT_20260930.md)
  finds no training-passing strength, so source-separated documents are
  not evaluated. At the largest strength, chat fails both load gates in all
  24 layers and several hot parents send 100% to one grandchild;
  stronger noise also drops raw content-argmax agreement below 15%.
  The stateless tuple hash repeats routes on recurrent contexts. Do not
  repeat a beta sweep or promote a new E12,800 bank from this result.
  Resume by changing the specialist-learning mechanism while accounting
  for recurrent/shared traffic, and advance the separate pretrained core
  representation and native same-artifact speed gap.
  [METH-185](METH_185_PRETRAINED_FFN_CHANNEL_SPARSITY_PROTOCOL_20260930.md)
  reads the pinned BF16 Qwen donor FFNs and the METH-125 actual E1,280
  pre-MLP states. Its [result](METH_185_PRETRAINED_FFN_CHANNEL_SPARSITY_RESULT_20260930.md)
  finds 16.49% median output relative L2 even at K=2,048 of 4,864
  activation-ranked channels, versus the frozen 1% limit; K=4,864
  exactly reproduces the donor component. This rules out the specified
  direct norm-ranked channel omission without retraining. The ranking
  is not an optimal subset proof, and no cheap gate, full quality, native
  speed or large-scale transfer was measured. Resume the compact-core
  line with a changed representation or joint adaptation, with explicit
  whole-path bytes and fresh model-quality gates.
  [METH-186/187](METH_186_187_Q6_CORE_E1280_PROTOCOL_20260930.md)
  tests a physical grouped-Q6 FFN/R8-head/BF16-attention core with
  the quality-valid centered E1,280 bank. The
  [result](METH_186_187_Q6_CORE_E1280_RESULT_20260930.md) verifies
  a 470,294,752-byte stored core and 481,534,976 ideal addressed
  bytes/token including route and selected factors. On the consumed
  METH-121 cohort, BPB degrades only +0.002600 versus BF16+E1,280,
  but donor-top-1 falls 94.504%→89.186%; prompt gates fail. Q6 donor
  is similarly low, locating the loss in the core assembly rather
  than a missing expert update. The uncorrected artifact is not a
  quality-valid or timed native candidate. Resume with a budgeted
  trained core correction and new-source quality adjudication;
  simultaneously retain the large-RAM expert-count problem.

Preserve unrelated working-tree changes in `docs/research/RESEARCH_INDEX.md`
and `benchmarks/donor_adaptation/density/build_document_holdout.py`. Update
this index by replacing current state, and put detailed evidence in the
experiment record. Documentation is in English; no model/assistant signatures.
Graphify is optional for explicit knowledge-graph work. Its last complete
snapshot predates METH-48/49; the later whole-repository update was stopped
after AST extraction because routine refresh cost outweighed its use here.

