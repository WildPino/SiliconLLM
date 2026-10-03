# METH-300 result: full-feature pair LUT does not fit the complete budget

**Decision: reject this fixed two-coefficient U8 proposal for candidate
training/export.** All three prospective addressed-weight gates fail.
Preserving every source feature avoids the specific rank/channel omission
mechanisms that failed 298/299, but it does not make this format cheap enough.
No palette was trained, no quality data inspected and no runtime port resumed.

## Reproduction, source controls and resources

The [protocol](METH_300_GIGACHAT_VECTOR_LUT_COST_PROTOCOL_20261003.md) and
[implementation](../../../benchmarks/native_expert_scaling/meth300_gigachat_vector_lut_cost.py)
were frozen at `f2a8e38` before observation. The exact protocol command exited
0 locally. All 414 source/Q4 tensor names and shapes, pinned 26-layer/64-expert/
top4 topology, block alignment, contiguous descriptor spans and complete file
sizes pass. Every organ's source/Q4 active and stored bytes reconciles exactly
with immutable 01/04; 1,628,078,080 active large-matrix coefficients match.
The older Q4/BF16 artifacts and fidelity work from frozen `donor-adaptation`
are reused for this new cost question. Its generic port stays paused.

The raw [result](meth300_gigachat_vector_lut_cost_result.json) is 282,260B,
SHA-256 `9458aec3f6baafbe00eaa266a3ac7da84c43040153cbfee3fc24ea9fd64dcd75`.
It retains all 414 descriptor/cost rows, group totals, all scenarios and n
formulas. An independent standard-library shape recount reproduced all
812,810,240 codes, 5,840,384 scale bytes, 123,667,200 table entries, stored
payload and three failures. This checks accounting, not native execution.

Only metadata/descriptors/alignment were read: BF16 6,085,376B, header SHA
`2e18041f5c90d897f4ab3f882887c63b797f736fec63b1147c8cd9e0d1bcbe08`;
Q4 6,102,912B, SHA
`3d96ca766891be60b055441fcaa6d127014423d90431f693e2e14ee4db0db0c6`.
Previously verified full artifact hashes are recorded as provenance, not
fresh whole-payload verification. No tensor coefficient, activation capture,
GPU/T4, download, model inference, optimization or native benchmark was used.

Controller arithmetic took **1.797s**, end RSS **23,588,864B**; tool wall
2.476s. The 120s/1GiB/16MiB-header/2MB-output limits pass. No scientific job
remains active. A preliminary import of the pruned external `gguf` package
was unavailable; the frozen screen instead uses its own bounded header
reader, reconciled against both existing ledgers. No older helper was edited.

## Complete cost, rather than routed-only savings

All sizes below are decimal MB. Weight bytes are addressed logical payload;
table writes/gathers are separate logical memory operations. Neither is
measured physical DRAM traffic, cache residence, latency or token throughput.

| Frozen scenario | Weight MB/token | Table writes MB/token | Lookup/adds per token | Weight gate <=560MB |
| --- | ---: | ---: | ---: | --- |
| Routed-only; other Q4 organs unchanged | 956.773 | 92.160 | 294,912,000 | FAIL |
| Every source projection, per-projection palettes | 829.378 | 494.669 | 812,810,240 | FAIL |
| Every projection, optimistic common palettes/input reuse | 829.150 | 424.166 | 812,810,240 | FAIL |

The complete case contains 812,810,240 U8 code bytes, 5,840,384 row scale
bytes, 511,200 palette bytes and 10,216,544 unchanged router/control/embedding
bytes/token. Its encoded stored tensor/palette payload is **5,388,446,688B**,
excluding file headers, alignment, indices and runtime allocations. Storage
reduction alone does not resolve active cost.

| Encoded organ | Code+row-scale MB/token | Query table entries/token |
| --- | ---: | ---: |
| MLA | 328.227328 | 86,860,800 |
| Four routed experts/layer | 296.550400 | 23,040,000 |
| Shared experts | 74.137600 | 12,240,000 |
| Dense first FFN | 20.721664 | 1,353,600 |
| Complete untied head | 99.013632 | 172,800 |

The source-specific MLA matters: K-B has 32 separate head queries and V-B
32 separate attention-result inputs. A shared weight palette does not make
those inputs equal. Likewise, each selected routed expert needs its own down
table. This graph interpretation is also checked against reusable donor C:
[`strat01_kb_pinned_batch`](../../../benchmarks/phase60/strat01_gguf_kb_q5q8_diag.h)
uses the head-strided query, and
[`strat01_r2c_cross_run_arm`](../../../benchmarks/phase60/strat01_gguf_rung2c_cross_input.h)
forms per-head attended512-vectors before V-B. No donor execution is rerun.
The complete per-projection format constructs 123,667,200 entries
(371,001,600 multiply/add operations) and performs **3,251,240,960B of logical
FP32 table gathers/token**; its largest single table is 4,032,000B. These
counts are not all compulsory DRAM bytes and cannot be summed into a DRAM
roofline without an actual layout/cache measurement.

The 40GB/s yardstick prices weight payload alone at **20.734ms/token**,
requiring 41.469GB/s at 50token/s, before table construction/gathers, routing,
KV, source nonlinearities and other work. This is arithmetic under the stated
yardstick, not a measured impossibility bound for arbitrary hardware/cache.
It misses the prospective 14ms streaming allotment by a wide margin.

Even an impossible **zero-head/zero-palette** diagnostic leaves **729,853,536B**
of weight payload/token. Improving only head selection cannot license this
format. At unchanged source features and the fixed row scales/controls, codes
would require **<=2.676851 bits/original coefficient**, before any palette
bytes, to fit560MB. The current U8/two-coefficient format uses4. This is a
necessary cost constraint, not a quality result for any lower-bit format.

## User priority: useful n and RAM must include router growth

The fixed-source case has 64 distinct pretrained experts/layer. n=640/6400
rows below are formulas only: no extra experts, training, duplicated bank,
quality measurement or native timing was instantiated. Keep top4 fixed.

| n/layer | Encoded payload in RAM, GB | Selected routed code+scale MB/token | Flat F32 router MB/token | Complete weight MB/token |
| --- | ---: | ---: | ---: | ---: |
| 64, actual source inventory | 5.388 | 296.550 | 9.8304 | 829.378 |
| 640, analytical topology only | 48.180 | 296.550 | 98.3040 | 917.910 |
| 6400, analytical topology only | 476.098 | 296.550 | 983.0400 | 1803.222 |

Per-query table construction remains independent of n under these palette
assumptions; encoded bank storage grows with n. The unchanged flat router
also grows linearly in reads and dot-product coefficients, from 2,457,600 to
24,576,000/245,760,000 coefficients/token. Thus more RAM alone does not
establish a constant-cost useful large-n engine. A cheaper source-preserving
coefficient representation and separately quality-validated hierarchical or
otherwise selective expert search are still required.

## Method decision and exact next question

Do not fit or export the unchanged pair proposal. Do not relax the weight
budget, assume cached tensors or reassign old Q4/IQ2/factor-LUT quality.
The reusable product is a complete source-shape cost tool, including MLA
head-bank inputs and the n-dependent flat router.

The next distinct LUT hypothesis is **four original coefficients per U8
non-Cartesian index** (2bits/coefficient), preserving full source channels
and rank potential. It is UNMEASURED: fewer indices lower code/gather counts,
but a small four-dimensional palette can increase approximation error.
Old IQ2 quality failures remain directly relevant and cannot be dismissed.
Before any training, separately freeze its complete-organ cost calculation,
query/code layout, actual C lookup/construction/control checks, resource
gates and source-bound quality calibration/control/error protocol. A naive
lower-bit Cartesian grid is only a matched control, never a retained donor
quality claim. Reuse 298 captures; do not repeat their49min collection.

No useful additional n, accepted50token/s artifact or cross-family method is
established. The complete goal remains active.
