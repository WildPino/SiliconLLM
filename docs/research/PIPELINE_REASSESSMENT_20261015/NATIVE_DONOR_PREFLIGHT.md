# Granite-4.0-H-Tiny instruct: complete header-only active-byte preflight

15 October 2026. **Complete metadata arithmetic; no weights, model execution, training, or runtime benchmark.** This tests the native recurrent/MoE donor's byte budget before source-value acquisition. It does not select a conversion checkpoint or qualify quality, DRAM traffic, or 50 accepted C tokens/s.

The [official instruct checkpoint](https://huggingface.co/ibm-granite/granite-4.0-h-tiny) was resolved through its official API to immutable revision `791e0d3d28c86e106c9b6e0b4cecdee0375b6124`. The [source configuration](https://huggingface.co/ibm-granite/granite-4.0-h-tiny/blob/main/config.json) and all three safetensors headers were fetched from that revision. The complete index contains 586 names; every name matched exactly one header tensor and the declared shard. All dtype, shape, offset, nonoverlap/continuity, payload length, file length, config topology, and API parameter-count checks passed. Only 127,120 bytes of metadata were read. Requests for each safetensors header required HTTP 206 and an exact `Content-Range` **before reading the body**; a 200 or missing range fails without a partial ledger. Each file has a 2 MiB metadata cap and the run a 16 MiB cap. The largest fetched header was 25,056 bytes. [Tool](../../../benchmarks/native_expert_scaling/conditional_donor_preflight.py); [result and raw fetched metadata](../../../results/native_expert_scaling/conditional_donor_preflight_20261015/preflight.json).

The source has 6,939,037,248 distinct parameters, including 6,228,541,440 stored routed-plus-shared expert parameters. It addresses 1,465,470,528 coefficients per token with top-6 of 64 routed experts at each of 40 layers. Tied `model.embed_tokens.weight` is charged as the **full output head once**, even though no explicit `lm_head.weight` tensor exists; an input lookup of one row does not require charging a second full head read. These are coefficient counts, not useful-capacity or physical-memory measurements.

| Active organ | Coefficients/token | BF16 bytes/token | I4 scenario bytes/token |
| --- | ---: | ---: | ---: |
| Routed experts, top-6 | 566,231,040 | 1,132,462,080 | 300,810,240 |
| Shared expert | 188,743,680 | 377,487,360 | 100,270,080 |
| Mamba projections | 526,417,920 | 1,052,835,840 | 279,659,520 |
| Tied head/embedding | 154,140,672 | 308,281,344 | 81,887,232 |
| Attention projections | 25,165,824 | 50,331,648 | 13,369,344 |
| Routers | 3,932,160 | 7,864,320 | 2,088,960 |
| Protected norms, state, conv and biases | 839,232 | 1,678,464 | 1,678,464 |
| **Total** | **1,465,470,528** | **2,930,941,056** | **779,763,840** |

Each hypothetical integer format packs **each matrix row** at its stated bit width, pads that row to a byte, and adds one F32 scale per ceiling group of 128 columns. Protected norms, state, conv and biases retain source dtype; BF16 has no added scales. The format is a byte-accounting convention, not an implemented quantizer or quality result. Values below are decimal GB/s needed to stream those addressed bytes once within the time limit; the 40 GB/s column divides bytes by the repository's CPU bandwidth yardstick. Actual execution also pays routing, dequantization, matrix arithmetic, recurrent state, sampling and other overhead.

| Analytical matrix format | Active MB/token | GB/s for 14 ms stream | GB/s for 20 ms total | Ideal read ms at 40 GB/s |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 2,930.941 | 209.353 | 146.547 | 73.274 |
| I8 + group scales | 1,512.079 | 108.006 | 75.604 | 37.802 |
| I4 + group scales | 779.764 | 55.697 | 38.988 | 19.494 |
| I3 + group scales | 596.685 | 42.620 | 29.834 | 14.917 |
| I2 + group scales | 413.606 | 29.543 | 20.680 | 10.340 |

The stated 560 MB/14 ms streaming allotment requires at least **219.764 MB/token** less than all-I4, or **36.685 MB/token** less than all-I3, before any execution margin. All-I4 is technically under the 800 MB/20 ms *ideal-read-only* ceiling, but leaves only 0.506 ms at 40 GB/s for every other operation. It is therefore not a credible unchanged C artifact. All-I2 clears the byte screen but has no demonstrated quality or packed-kernel path; recurrent projections and routing masses are especially sensitive to operator changes. A mixed-precision choice must be source-derived and tested on held-out donor-relative chat quality and the same C artifact's accepted batch-1 throughput.

The pinned config fixes width 1536, 36 Mamba2 and four attention layers, 64 routed experts/top-6, 512 expert hidden, 1024 shared hidden, Mamba state 128 and expansion 2, vocabulary 100352, tied head, NoPE attention, and embedding/residual/logit multipliers 12/.22/6. The [reference implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py), `GraniteMoeHybridTopKRouter` and `GraniteMoeHybridExperts`, uses selected-logit softmax masses and a shared expert path. Any follow-on converter must preserve these, gate/up SiLU/down ordering, recurrent state, attention sites and the chat template. Header arithmetic does not show that the source model can be swapped into the current engine's SSM operator.

This instruct result supersedes no prior local measurement. [METH-03](../NATIVE_EXPERT_SCALING_20260925/METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md) priced a **base** GGUF's actual mixed Q4_K_M header at 913.314 MB/token, not this instruct safetensors model or the all-I4 scenario. METH-412/413 rejected a different Granite-3.1 row-I8 synthetic kernel profile; they do not decide this donor. The next gate is a declared mixed-format/structural candidate with less than 560 MB addressed bytes and margin for compute, followed by real-value donor-relative quality. Do not acquire multi-GB weights simply because a 2-bit analytical row exists.

Mechanical checks: the script completed twice against the pinned revision with identical counts. A local fake HTTP 200 and a missing-`Content-Range` response were both rejected before any body read. Recomputing the ledger from the saved complete headers matched all 586 index entries; removing one index name was rejected. These checks verify the metadata protocol and arithmetic, not the source model's outputs.

## Pinned repetition and mixed-policy extension

The revised tool accepts `--revision 791e0d3d28c86e106c9b6e0b4cecdee0375b6124`, checks that the official revision API returns exactly that revision, and uses an exclusive new output directory. A successful pinned observation is retained in [native_donor_metadata/preflight.json](native_donor_metadata/preflight.json), with exact raw header bytes, API/config/index metadata and matching SHA256 values. The initial local observation is preserved in the original results directory. The pinned repeat also read 127,120 metadata bytes. No tensor values were requested.

The offline [candidate frontier](native_donor_metadata/candidate_frontier.json) verifies every raw header/config/index digest before recomputing the same uniform and two mixed-format ledgers. The source top-6 count and BF16 router payload remain unchanged. I2 routed experts plus I3 shared/SSM, I4 head/attention and protected source controls costs **554,586,240 bytes/token**; I3 routed/shared plus I2 SSM costs **559,562,880**. Both are analytical hypotheses with `implemented_format=false` and `quality_known=false`. They expose an aggressive compression requirement rather than a validated recipe.

The same frontier separately charges logical state/cache work for an explicitly proposed F32-SSM/F32-conv-ring/F16-KV runtime. Persistent SSM state is 56,623,104 bytes. One read/write pass costs 113,246,208 logical bytes/token; full ring read/write adds 3,833,856. Four attention layers add 8,192 bytes per cached position and 8,192 bytes per appended token. At context2048 these sum to **133,865,472 additional logical bytes/token**; at8192,184,197,120. Cache residence and actual DRAM traffic are unmeasured. These terms are additional to the matrix ledger; activation scratch, table construction, sampling and intermediate traffic remain unpriced. A mixed policy barely clearing a weight-only14ms allowance is not a complete latency pass.
