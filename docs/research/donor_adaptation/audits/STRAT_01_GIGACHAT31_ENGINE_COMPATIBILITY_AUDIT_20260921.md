# STRAT-01 GigaChat3.1-10B-A1.8B base: `phase60/engine.c` compatibility audit

**Date:** 2026-09-21  
**Status:** bounded technical inventory; no implementation or strategy decision  
**Scope:** the accepted STRAT-01 **base** Q4 GGUF only. This audit neither adds an engine nor evaluates speed, quality, or the project >=50 tok/s target.

## Scope and evidence boundary

The accepted artifact is the base model. The source-binding protocol explicitly excludes the one-layer MTP extension; its `model.layers.26` tensors are not in the 414-tensor base GGUF and are not part of this compatibility inventory. The MTP HF bundle is useful only as source metadata corroboration, not as an implied execution target.

This document distinguishes three different things:

1. the current synthetic SSM/MoE benchmark in `benchmarks/phase60/engine.c`;
2. the separate Qwen-oriented Transformer runtime in `benchmarks/donor_adaptation/engine/donor_engine.c`; and
3. the pinned `llama.cpp` DeepSeek2 implementation that establishes source-architecture semantics and GGUF naming.

The presence of an operation in (2), or its implementation in (3), is not evidence that it executes in (1).

## Accepted artifact identity

| Item | Pinned value | Evidence |
|---|---|---|
| Accepted base Q4 artifact | `GigaChat3.1-10B-A1.8B-q4_K_M.gguf` | Q4 manifests under `benchmarks/donor_adaptation/density/results/strat01_gigachat_{fresh_quality_v2,rollout_v1}/` |
| Q4 SHA-256 | `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` | Same manifests; `docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_DOCUMENT_ROLLOUT_RESULT_20260921.md` |
| Bound base BF16 teacher SHA-256 | `e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e` | Same manifests and source-binding protocol |
| HF source snapshot | `ai-sage/GigaChat3.1-10B-A1.8B-bf16` at `189fff27a1dee68473960c3d5bca53e0e07a3191` | `.../STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md:7-10` |
| Clean converter/runtime binding | llama.cpp `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` | Source-binding protocol:7-10 |
| Base binding result | `PASS_SOURCE_BINDING`: 414/414 names, types, shapes, sizes matched; all 21,350,179,072 BF16 payload bytes matched; zero mismatches | Source-binding protocol:32-42 |
| Binding manifest digest | `c4d3e3dca919a6b24f8ca2a389d3f8624523a3c1acf19d685d910dd7693fd309` | Source-binding protocol:32-42 |
| GGUF architecture | `deepseek2`, 414 tensors, GGUF quantization version 2, file type 15 | Read from accepted GGUF with the pinned GGUF reader; DeepSeek2 model loader in `C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4/src/models/deepseek2.cpp:3-44` |
| Base-only boundary | `num_hidden_layers=26`; `num_nextn_predict_layers=1`, excluded from the base artifact | `.../strat01_gigachat_source_189fff27/config.json`; source-binding protocol:3 |

The repository index records the base artifact as quality, PIQA, and frozen document-rollout accepted while engine/rate remain open. That state is provenance only: it does not demonstrate any `engine.c` execution. See `docs/research/RESEARCH_INDEX.md` and `docs/research/STRATEGIC_10B_20260916/ROADMAP.md`.

## Exact base-model configuration

The following dimensions are from `benchmarks/donor_adaptation/density/results/strat01_gigachat_source_189fff27/config.json`, corroborated by the accepted GGUF metadata and tensor shapes.

| Organ/configuration | Value |
|---|---:|
| Vocabulary / hidden size / layers | 128,256 / 1,536 / 26 |
| Standard dense FFN width | 8,960 |
| Routed and shared expert width | 1,280 |
| Attention heads / source KV heads | 32 / 32 |
| Query output width | 6,144 = 32 x 192 |
| Query head composition | 128 non-RoPE + 64 RoPE = 192 |
| KV latent (LoRA) rank | 512 |
| KV-A output | 576 = 512 latent + 64 RoPE key |
| KV-B output | 10,240 = 32 x (128 non-RoPE key + 192 value) x 512 |
| Dense-first layers | 1 (block 0) |
| Routed experts / selected per token | 64 / 4 |
| Shared experts | 1 |
| Router scoring / selection metadata | `sigmoid`, `noaux_tc`, `n_group=1`, `topk_group=1`, `norm_topk_prob=true`, routing scale 1 |
| Activation / norm | SiLU; RMSNorm epsilon `1e-6` |
| RoPE | dimension 64; theta 100,000; YaRN factor 64, original context 4,096, beta-fast 32, beta-slow 1, mscale 1, mscale-all-dim 1; GGUF YaRN log multiplier 0.1 |
| Declared maximum context / cache use | 262,144 / `use_cache=true` |
| Embeddings / output head | `[128256,1536]` each in source; `tie_word_embeddings=false` |
| Token boundary | source BOS 1, EOS 2; source tokenizer JSON SHA-256 `b4b3d90c67830a4e566296ad8d0f6b5ac5a5cdbfd331b200d0ee8263aaaea1fe` |

`llama.cpp` canonically records MLA/MQA as one KV head with key length 576, value length 512, and MLA key/value lengths 192/192. That representation is an implementation convention, not a contradiction of the source configuration's 32 source KV heads. The converter writes these fields in `C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4/convert_hf_to_gguf.py` dependencies at `conversion/deepseek.py:319-386`.

## Materialized Q4 tensor format

The accepted GGUF contains 233 `Q4_K`, 26 `Q6_K`, 26 `Q5_0`, and 129 `F32` tensors. Examples include Q4_K token embeddings, direct-Q, KV-A, attention output, and expert gate/up tensors; Q6_K down tensors and output head; Q5_0 MLA K-B tensors; and F32 norms, router gate, and router correction-bias tensors.

This is not a generic "four bits per weight" row-scale format. In the pinned ggml definition, `Q4_K` uses 256-element superblocks, eight 32-element affine subblocks, FP16 `d` and `dmin`, packed scale/min fields, and packed nibbles (`ggml/src/ggml-common.h:89,323-338`; dequantization `ggml/src/ggml-quants.c:1529-1550`). Exact parity also requires the distinct Q6_K and Q5_0 layouts. `phase60/engine.c` and `donor_engine.c` currently have no GGUF Q4_K/Q6_K/Q5_0 loader or corresponding matvec implementation.

## Decode-path inventory and compatibility mapping

“Already” below means an operation exists in the indicated engine, not that its dimensions, format, or semantics match the accepted artifact. “Reusable” means a code pattern may inform an extension, not an execution claim.

| Required base decode organ | Accepted artifact semantics / tensors | `phase60/engine.c` today | `donor_engine.c` reusable support | Gap / uncertainty and evidence path |
|---|---|---|---|---|
| Token input, embedding lookup | Untied `token_embd.weight`; 128,256 x 1,536; Q4_K | Has a synthetic embedding lookup, fixed `V=1024,D=256` | Has flat-binary embedding lookup | Both loaders/formats and dimensions differ. Verify GGUF `token_embd.weight`, source ID mapping, and Q4_K matvec. `phase60/engine.c:52-76,189-258`; `donor_engine.c:1267-1282` |
| Tokenizer boundary | Source IDs; BOS 1/EOS 2; source tokenizer required for frozen 96-document path | No tokenizer or source-token interface | No tokenizer implementation; caller supplies IDs | Accepted scorer found llama text-to-ID mismatch on 95/96 documents, so text tokenization cannot be presumed equivalent. A source-token-ID injection route is evidence-backed only after ID/vocab/output decoding parity. Manifests above; `RESEARCH_INDEX.md` |
| Pre-attention RMSNorm | RMSNorm, epsilon 1e-6 per layer | RMSNorm primitive, fixed epsilon 1e-5 | RMSNorm transformer primitive | Epsilon and placement need exact parity. `phase60/engine.c:272-278`; `donor_engine.c:1267-1461`; `deepseek2.cpp:465-470` |
| Attention query projection | Direct `attn_q.weight`, 1,536 -> 6,144; no Q-LoRA | No generic Q projection; one synthetic SWA QKV | Separate standard Q matrix | Donor pattern is conventional attention only. Direct Q weight exists in pinned DeepSeek2 loader `deepseek2.cpp:98-109`; verify Q4_K dequant and orientation. |
| KV compression and expansion | `attn_kv_a_mqa.weight` 1,536 -> 576; `attn_kv_a_norm` 512; K-B and V-B derived from `attn_k_b` / `attn_v_b` | Absent | Separate full K and V projections only | MLA compressed KV path is absent from both local engines. Exact loader tensor names/shapes: `deepseek2.cpp:103-122`; converter split/transpose: `conversion/deepseek.py:404-450`. |
| RoPE / YaRN | RoPE only on 64-d key/query positional component; YaRN values listed above | Absent | Conventional RoPE table applied to full Q/K with fixed theta | Donor RoPE is a reusable numerical pattern, not MLA/YaRN parity. Need the exact DeepSeek2 split, YaRN scaling, and positional API. `phase60/engine.c:349-410`; `donor_engine.c:913-927,1267-1461`; `deepseek2.cpp:533-580` |
| KV cache | Compact cache: 512 latent KV plus 64 positional K; MLA attention reconstructs/absorbs per head | One fixed ring K/V cache only for synthetic SWA, `WIN=128` | FP32 conventional K/V cache `[L][sequence][NKV*HD]` | Neither layout is the accepted MLA cache. Cache precision, allocation, layout, and long-context behavior for a new engine remain to be specified and tested. `phase60/engine.c:60-76,349-410`; `donor_engine.c:1189-1208`; `deepseek2.cpp:517-580` |
| Attention scores, softmax, output | MLA Q/K construction, causal attention, value construction, `attn_output.weight` | Synthetic sliding-window attention only, one layer/head layout | Generic causal dense Transformer attention | Attention primitives are reusable concepts; phase60's SWA and donor's standard MHA/GQA do not establish MLA parity. Evidence: `phase60/engine.c:349-410`; `donor_engine.c:1267-1461`; `deepseek2.cpp:540-620` |
| Attention residual | Add attention result to residual | Present, synthetic | Present, Transformer | Placement must follow source block ordering. `deepseek2.cpp:621-642` |
| Post-attention RMSNorm | RMSNorm, epsilon 1e-6 | Has generic primitive | Has Transformer post-attention norm | Same epsilon/ordering caveat; `deepseek2.cpp:643-646` |
| Dense FFN, block 0 | gate/up/down, 8,960 width, SwiGLU with SiLU | Synthetic dense ReLU x ReLU, fixed 512 | Dense SwiGLU implementation/pattern | Donor dense FFN is structurally closer, but its flat Qwen matrices and load format do not match. `phase60/engine.c:279-300`; `donor_engine.c:1438-1454`; `deepseek2.cpp:126-129,647-652` |
| Router / score correction | F32 `ffn_gate_inp` 1,536 x 64 plus F32 correction bias; sigmoid scoring, `noaux_tc`; select 4 | Synthetic softmax router, top-8, renormalized | Carved path top-k after bare matvec | Both differ materially. With one group, group selection is degenerate, but the exact correction-bias placement and normalization sequence must be matched from the pinned implementation before claiming parity. `phase60/engine.c:302-337`; `donor_engine.c:1243-1265`; `deepseek2.cpp:131-143,655-680`; source config. |
| Routed MoE experts, blocks 1-25 | 64 experts/layer; gate/up/down, 1,280 width; top-4 selected | Synthetic 32 experts, 128 hidden, top-8, ReLU x ReLU | Optional carved expert execution pattern, not this structure | Need tensor-stack access, exact router weights, SwiGLU, and 64 x 25 expert coverage. Converter stacking is in `conversion/deepseek.py:404-450`; phase60 fixed constants are at `engine.c:52-76`. |
| Shared expert | One always-active SwiGLU expert, 1,280 width, in every MoE layer | Absent | Absent from normal forward path | New required organ. Pinned loader and forward show its distinct tensors/branch: `deepseek2.cpp:146-148,670-680`. |
| MoE residual | Add routed and shared FFN contribution to block residual | Synthetic residual only around its own mode | Generic FFN residual | Exact sum/order must match source. `deepseek2.cpp:655-689` |
| Final RMSNorm | `output_norm.weight`, F32 | Final norm primitive, synthetic epsilon | Final Transformer norm | Exact epsilon and weight type/layout needed. `phase60/engine.c:402-405`; `donor_engine.c:1456-1459`; `deepseek2.cpp:87-92,694-697` |
| LM head / logits | Untied `output.weight`, 1,536 x 128,256; Q6_K | Synthetic head `V=1024` | Flat fp32/ternary head only | Q6_K decoding, orientation, full vocabulary logits, and token-ID policy are required. `phase60/engine.c:404-410`; `donor_engine.c:1459-1461`; `deepseek2.cpp:87-92` |
| GGUF metadata and tensor loading | `deepseek2` metadata plus Q4_K/Q6_K/Q5_0/F32 tensors; tensor names above | Fixed 16-word E1M1/E4M1 synthetic binary header | `QWENDON1` flat Qwen binary | Neither local loader accepts GGUF. Direct GGUF loading needs metadata/name/type/layout support; deterministic export needs a complete source-to-new-format contract. `phase60/engine.c:211-258`; `donor_engine.c:952-1122`; `qwen_export.py:3-57,567-641` |

## What each existing engine actually establishes

`benchmarks/phase60/engine.c` is a fixed-shape synthetic performance benchmark: six 256-wide layers, mostly SSM scan, one sliding-window attention layer, and a synthetic top-8 MoE option. Its fixed header and model magic only select the synthetic dense/MoE fixture; they do not describe a generic Transformer, MLA, GGUF, or GigaChat loader (`benchmarks/phase60/engine.c:4,52-76,189-258,302-410`). It offers low-level C precedents for loops, matvecs, RMSNorm, residual addition, softmax, a small cache, embedding, and a head, but no accepted-artifact execution path.

`benchmarks/donor_adaptation/engine/donor_engine.c` is a distinct own-C Transformer runtime for Qwen-style flat exports. It establishes that this repository has a separate conventional decoder pattern with attention, ordinary RoPE, KV cache, RMSNorm, SwiGLU, and greedy logits. It does not load the accepted GGUF and does not implement DeepSeek2/GigaChat MLA, YaRN, the compact KV cache, sigmoid/correction-bias top-4 routing, or the always-on shared expert (`benchmarks/donor_adaptation/engine/donor_engine.c:1-24,108-164,913-927,1033-1122,1189-1208,1243-1461`).

The H2I export-gap audit is relevant only as a bounded precedent: a container may have matrix-type tags while its exporter does not encode a particular accepted composition. Its SCORE-ONLY closure is not an instruction to revive H2I or rerun a closed cell. See `docs/research/donor_adaptation/audits/H2I_ENGINE_EXPORT_GAP_AUDIT.md:39-53,64-85,116-123`.

## Loader and representation boundary: two plausible integration classes

This audit does not choose an integration approach. The evidence leaves two classes to evaluate later.

| Class | What it would have to cover | Existing evidence | Known open work |
|---|---|---|---|
| Direct GGUF loading | Parse `deepseek2` metadata; locate 414 tensors by GGUF names; honor F32/Q4_K/Q5_0/Q6_K type/layout/orientation; instantiate MLA, YaRN, MoE, shared-expert, and token boundaries | The pinned converter/model implementation supplies the authoritative names and metadata mapping; source binding proves the base tensor correspondence | Neither local engine parses GGUF or has the required quantized kernels/architecture. Direct compatibility must not be inferred merely from `llama.cpp` support. |
| Deterministic export into a new engine format | Define a lossless, versioned export contract for all base tensors and metadata; choose/store an exact representation for each tensor type or an explicitly validated conversion; load into an extended engine | `qwen_export.py` and `donor_engine.c` demonstrate a repository-local exporter/loader pairing; H2I documents why exporter coverage must be audited separately | The current exporter is Qwen-specific and names only standard Q/K/V and dense FFN tensors. It cannot export the accepted DeepSeek2/MoE/MLA graph as-is. Any conversion requires its own identity and numerical-parity evidence. |

The accepted Q4 quantization layouts make “deterministic export” ambiguous until the destination representation is specified: preserving the GGUF blocks and converting to another representation are different claims. The latter cannot inherit the base source-binding result without new evidence.

## Minimal parity ladder for a future implementation

The rungs are deliberately small and cumulative. They are evidence requirements, not a request to implement them now.

| Rung | Narrow check | Exact evidence required to advance |
|---|---|---|
| 0. Metadata and load identity | Read the accepted Q4 base only; verify architecture, 26 base blocks, dimensions, all 414 tensor names/shapes/types, BOS/EOS, and source/accepted hashes | Machine-readable loader inventory compared against the accepted GGUF manifest and the source-binding manifest; explicit MTP exclusion; no substituted file accepted silently |
| 1. One tensor | Load and multiply one named quantized tensor (for example `blk.0.attn_q.weight`) on fixed inputs | Exact type/layout/orientation evidence plus numerical comparison to the pinned GGUF/llama reference under a stated tolerance; repeat separately before generalizing to Q4_K, Q5_0, and Q6_K |
| 2. One organ and one layer | Execute block 0 attention and dense SwiGLU, then a single MoE block including compact MLA cache, YaRN, sigmoid/correction routing, routed top-4, and shared expert | Fixed source token IDs, positions, weights, intermediate tensors, and output/residual comparisons against the pinned reference. The expected cache layout and precision must be recorded. |
| 3. Full-base logits | Execute all 26 base blocks for fixed source-token prefixes | Full 128,256-logit comparison at defined positions, including argmax and a stated numerical acceptance rule against Q4 and/or bound BF16 references; demonstrate untied output head use. |
| 4. Greedy token path | Autoregress greedily from fixed source-token IDs | Per-step selected token IDs, tie policy, position/cache handling, and decoded token mapping match a frozen reference trace. This establishes decoding semantics, not text-tokenizer equivalence. |
| 5. Frozen 96-document source-token path | Run the accepted source-token 96-document protocol without changing corpus, IDs, rollout policy, or artifact | Per-document trace and scoring-compatible records showing exact corpus/manifest identity, source IDs, BOS/EOS/position policy, greedy rule, and output-token handling. Text-to-ID tokenizer parity remains a separately demonstrated condition because the existing llama text tokenizer differed on 95/96 documents. |

No rung above is a speed benchmark, a quality acceptance, or a >=50 tok/s claim. Any later rate claim would require its own same-artifact, same-execution-path measurement; the roadmap explicitly treats execution-path alignment as material.

## Explicit uncertainties and non-inferences

- The BF16 source binding, together with the pinned Q4 manifests and paired quality results, establishes the base lineage used here; it does not prove bitwise source identity for the quantized payload, a new C loader, a new numerical kernel, or a new decode graph.
- The accepted GGUF's Q4_K/Q5_0/Q6_K tensors cannot be treated as one common quantization scheme. Exact matvec behavior, accumulation precision, and tensor orientation need direct parity evidence.
- `noaux_tc` is configured with one group, so group selection is degenerate here. The precise score-correction-bias and probability-normalization sequence must still be taken from the pinned architecture implementation, not guessed from its name.
- `donor_engine.c` has conventional attention/RoPE/KV cache and carved-expert patterns, but its model structure lacks all MLA fields and a shared-expert forward branch. Reuse of a pattern is not semantic compatibility.
- The exact long-context memory allocation and cache precision for a prospective `phase60` extension are not specified by the accepted artifact alone. The source model declares 262,144 context and cache use; the required local C layout is an implementation question.
- The source tokenizer and GGUF vocabulary metadata establish IDs and special-token facts, but the repository's frozen scorer observed text-to-ID mismatch in the llama runtime on 95/96 documents. A native tokenizer or source-ID path must be independently evidenced.
- No conclusion is drawn about throughput, memory fit, perplexity, PIQA, document quality, or the >=50/100 tok/s objectives. This audit neither recommends reopening closed E/H cells nor rerunning llama.cpp quality gates.

## Primary local evidence consulted

- `docs/research/RESEARCH_INDEX.md`
- `docs/research/STRATEGIC_10B_20260916/ROADMAP.md`
- `docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md`
- `docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_DOCUMENT_ROLLOUT_RESULT_20260921.md`
- `docs/research/donor_adaptation/audits/H2I_ENGINE_EXPORT_GAP_AUDIT.md`
- `benchmarks/donor_adaptation/density/results/strat01_gigachat_source_189fff27/config.json` and tokenizer metadata
- Accepted Q4/BF16 manifests in `benchmarks/donor_adaptation/density/results/strat01_gigachat_fresh_quality_v2/` and `benchmarks/donor_adaptation/density/results/strat01_gigachat_rollout_v1/`
- `benchmarks/phase60/engine.c`
- `benchmarks/donor_adaptation/engine/donor_engine.c`
- `benchmarks/donor_adaptation/engine/qwen_export.py`
- `C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4/src/models/deepseek2.cpp`
- `C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4/ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c`
