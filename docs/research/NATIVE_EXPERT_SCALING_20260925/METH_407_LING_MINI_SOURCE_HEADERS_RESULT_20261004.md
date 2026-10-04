# M407: Ling-mini actual source applicability permits cost screening

Frozen6268585. Raw SHA256 `431d9e2978c46b1d938a1efb0fc67ea345b18d01e671baa47604f4b8ddf94a56`.

Source `inclusionAI/Ling-mini-2.0`, revision
`a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab`.
Config SHA256 `88388c4e5040292a19b40da6fa89a3d6a0bb6d6878f389b0bba55ac51c239aa3`.

ALL6 apparatus gates PASS. ALL4 safetensors shards/14,813 actual tensors, index/
Git-bound config/custom Python source/complete offsets/dtype extents/EOF exact.
19 banks layers1-19 with256 gate/up/down slots [512,2048]/[2048,512]; top8,
shared512, first dense5120,20 fused GQA QKV[3072,2048]/attention dense[2048,2048].
Untied embedding/head[157184,2048]. No MTP. Custom source stored, NOT executed.

| Complete source category | Parameter positions |
| --- | ---: |
| routed | 15,300,820,992 |
| embedding | 321,912,832 |
| core_matrices | 300,941,312 |
| router_f32 | 9,966,336 |
| head | 321,912,832 |
| other_controls_f32 | 89,088 |

Complete main/all16,255,643,392 positions; extras0. BF16 positions16,255,638,528,
F32 bias positions4,864. Source tensor value extent32,511,296,512B; physical
archive total32,513,136,008B. Header positions are not unique/function/finite
content verification; whole declared LFS SHA not freshly verified without values.

## SAME317 full-width hypothetical additive geometry

| Active addressed descriptor | Bytes |
| --- | ---: |
| encoded_two_indices_per8_code_bytes | 194,772,992 |
| row_scale_bytes | 2,560,000 |
| palette_bytes | 2,277,376 |
| head_Q6_K_bytes | 264,069,120 |
| all_flat_router_F32_bytes | 39,865,344 |
| other_control_F32_bytes | 356,352 |
| one_embedding_BF16_row_bytes | 4,096 |
| addressed_weight_descriptor_bytes | 503,905,280 |

779,091,968 coded active matrix coefficients include ALL core/shared/dense and
selected8 functions in19 banks. Total503,905,280B <=560,000,000B gate PASS.
This includes head264.069MB and router/control40.222MB, not expert-only cost.
Hypothetical complete stored target4,969,196,544B. No representation fitted/
exported; does not establish quality, active latency or preserved source capacity.

Main active parameter positions with ONE embedding lookup row =1,111,062,272.
Card-style counting of the complete embedding matrix as active would add
321,910,784 positions (full321,912,832 minus actual lookup2048). This distinction
explains approximately1.43B versus1.11B arithmetic; no original runtime measured.

Source BF16/F32 nominal complete footprints fit total80GiB RAM; available
67410214912B at inspection. Loading/allocator/activations/
KV/shard duplication overhead unqualified; nominal fit is not a qualified reference.

MAIN16.531s /maximum checkedRSS58007552B;
13HTTP requests /3261092response bytes; ZERO weight values.
No inference/weight acquisition/training/GPU/T4. Source/header assets in
`results/native_expert_scaling/meth407_ling_mini_source_headers`.

Decision: eligible ONLY for separately frozen source-sized native active-cost
and reference/operator protocols before full acquisition or generic port. Existing
Switch loader D<=1024/vocab<=65536 cannot load D2048/vocab157184. Qualify original
sigmoid/group/bias/top8 probability amplitude, GQA/QK norm/halfRoPE, SiLU/shared
FFN/head/tokenizer explicitly. Another-family applicability is not another-family
transfer quality/rate, useful>256/10x or physical DRAM. Card H20 rate is not hostCPU.

Next proposed408: bounded CPU cost of this exact20-layer779.092M coefficient
geometry/head/router/full representation before learned books or32.5GB acquisition.
Predeclare scalar/integer controls, complete addressed bytes, profile/repeats/
limits and exact scope. Synthetic cost can reject this format; it cannot prove
pretrained capacity/quality/rate. No408 controller/protocol/run yet.
