### T1 — shape, licence and the parameter cross-check

| donor | family | D | L | V | tied | params computed | params advertised (safetensors) | delta | cross-check | licence |
|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | qwen2 | 1536 | 28 | 151936 | yes | 1.544B | 1.544B | +0.00% | AGREE | apache-2.0 |
| `Qwen/Qwen3-1.7B` | qwen3 | 2048 | 28 | 151936 | yes | 1.721B | 2.032B | -15.32% | AGREE_TIED_DUPLICATED | apache-2.0 |
| `HuggingFaceTB/SmolLM2-1.7B` | llama | 2048 | 24 | 49152 | yes | 1.711B | 1.711B | +0.00% | AGREE | apache-2.0 |
| `microsoft/Phi-3-mini-4k-instruct` | phi3 | 3072 | 32 | 32064 | no | 3.821B | 3.821B | +0.00% | AGREE | mit |
| `allenai/OLMo-2-1124-7B` | olmo2 | 4096 | 32 | 100352 | no | 7.298B | 7.299B | -0.00% | AGREE | apache-2.0 |
| `mistralai/Mistral-7B-v0.3` | mistral | 4096 | 32 | 32768 | no | 7.248B | 7.248B | +0.00% | AGREE | apache-2.0 |
| `Qwen/Qwen3-8B` | qwen3 | 4096 | 36 | 151936 | no | 8.191B | 8.191B | -0.00% | AGREE | apache-2.0 |
| `Qwen/Qwen3-30B-A3B` | qwen3_moe | 2048 | 48 | 151936 | no | 30.532B | 30.532B | -0.00% | AGREE | apache-2.0 |
| `mistralai/Mixtral-8x7B-v0.1` | mixtral | 4096 | 32 | 32000 | no | 46.703B | 46.703B | +0.00% | AGREE | apache-2.0 |
| `deepseek-ai/DeepSeek-V2-Lite` | deepseek_v2 | 2048 | 27 | 102400 | no | 15.706B | 15.706B | -0.00% | AGREE | other |
| `allenai/OLMoE-1B-7B-0924` | olmoe | 2048 | 16 | 50304 | no | 6.919B | 6.919B | -0.00% | AGREE | apache-2.0 |
| `openai/gpt-oss-20b` | gpt_oss | 2880 | 24 | 201088 | no | 20.908B | 21.512B | -2.81% | PACKED_STORAGE | apache-2.0 |
| `Zyphra/Zamba2-2.7B` | zamba2 | 2560 | 54 | 32000 | no | 7.031B | 2.662B | +164.11% | DISAGREE | apache-2.0 |
| `ibm-granite/granite-4.0-h-small` | granitemoehybrid | 4096 | 40 | 100352 | yes | 32.207B | 32.207B | -0.00% | AGREE | apache-2.0 |
| `nvidia/Nemotron-H-8B-Base-8K` | nemotron_h | 4096 | 52 | 131072 | no | 8.101B | 8.101B | -0.00% | AGREE | other |
| `tiiuae/Falcon-H1-7B-Base` | falcon_h1 | 3072 | 44 | 130048 | no | 7.585B | 7.586B | -0.00% | AGREE | other |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | qwen3_next | 2048 | 48 | 151936 | no | 79.574B | 81.325B | -2.15% | DISAGREE | apache-2.0 |
| `state-spaces/mamba2-2.7b` | mamba_ssm | 2560 | 64 | 50288 | yes | 2.702B | — | — | NO_METADATA | apache-2.0 |
| `meta-llama/Llama-3.2-1B` | — | | | | | | | | **UNAVAILABLE** | — |
| `ai21labs/AI21-Jamba-Mini-1.6` | — | | | | | | | | **UNAVAILABLE** | — |

### T2 — operator map (which layers carry attention), read from the config

| donor | attn layers | SSM layers | linear-attn layers | source key | attention fraction |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 28 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `Qwen/Qwen3-1.7B` | 28 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `HuggingFaceTB/SmolLM2-1.7B` | 24 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `microsoft/Phi-3-mini-4k-instruct` | 32 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `allenai/OLMo-2-1124-7B` | 32 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `mistralai/Mistral-7B-v0.3` | 32 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `Qwen/Qwen3-8B` | 36 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `Qwen/Qwen3-30B-A3B` | 48 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `mistralai/Mixtral-8x7B-v0.1` | 32 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `deepseek-ai/DeepSeek-V2-Lite` | 27 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `allenai/OLMoE-1B-7B-0924` | 16 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `openai/gpt-oss-20b` | 24 | 0 | 0 | `layer_types` | 100% |
| `Zyphra/Zamba2-2.7B` | 9 | 45 | 0 | `layers_block_type` | 17% |
| `ibm-granite/granite-4.0-h-small` | 4 | 36 | 0 | `layer_types` | 10% |
| `nvidia/Nemotron-H-8B-Base-8K` | 4 | 24 | 0 | `hybrid_override_pattern` | 8% |
| `tiiuae/Falcon-H1-7B-Base` | 44 | 44 | 0 | `attn_layer_indices=null -> parallel hybrid, attention on every layer (INFERRED)` | 100% |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 12 | 0 | 36 | `full_attention_interval` | 25% |
| `state-spaces/mamba2-2.7b` | 0 | 0 | 0 | `attn_layer_idx` | 0% |

### T3 — resident weight footprint, both precision maps

| donor | (a) all-ternary 0.5 B/w | of which scales | of which 32-pad | (b) P61/D9 map | SKU-A 16GB @32K, 4-bit KV, map (a) | SKU-B 64GB @128K, 4-bit KV, map (a) |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 741.8 MB | 3.6 MB | 2.1 MB | 1.98 GB | 1.94 GB FIT | 2.60 GB FIT |
| `Qwen/Qwen3-1.7B` | 825.4 MB | 3.2 MB | 1.7 MB | 2.97 GB | 2.68 GB FIT | 5.31 GB FIT |
| `HuggingFaceTB/SmolLM2-1.7B` | 820.5 MB | 3.0 MB | 1.5 MB | 2.44 GB | 3.30 GB FIT | 7.80 GB FIT |
| `microsoft/Phi-3-mini-4k-instruct` | 1.79 GB | 4.9 MB | 3.0 MB | 6.36 GB | 5.79 GB FIT | 14.79 GB FIT |
| `allenai/OLMo-2-1124-7B` | 3.41 GB | 7.0 MB | 3.9 MB | 13.08 GB | 8.41 GB FIT | 20.41 GB FIT |
| `mistralai/Mistral-7B-v0.3` | 3.39 GB | 6.5 MB | 3.9 MB | 8.63 GB | 5.39 GB FIT | 8.39 GB FIT |
| `Qwen/Qwen3-8B` | 3.83 GB | 7.6 MB | 4.4 MB | 12.80 GB | 5.95 GB FIT | 9.33 GB FIT |
| `Qwen/Qwen3-30B-A3B` | 14.31 GB | 87.3 MB | 2.9 MB | 19.32 GB | 16.06 GB **OVER** | 18.31 GB FIT |
| `mistralai/Mixtral-8x7B-v0.1` | 21.79 GB | 34.5 MB | 3.9 MB | 27.01 GB | 23.79 GB **OVER** | 26.79 GB FIT |
| `deepseek-ai/DeepSeek-V2-Lite` | 7.35 GB | 34.0 MB | 1.7 MB | 9.93 GB | 8.59 GB FIT | 9.30 GB FIT |
| `allenai/OLMoE-1B-7B-0924` | 3.24 GB | 17.1 MB | 1023.0 KB | 4.79 GB | 5.24 GB FIT | 8.24 GB FIT |
| `openai/gpt-oss-20b` | 9.77 GB | 28.6 MB | 3.9 MB | 15.62 GB | 10.96 GB FIT | 11.52 GB FIT |
| `Zyphra/Zamba2-2.7B` | 3.30 GB | 10.7 MB | 10.8 MB | 12.36 GB | 5.70 GB FIT | 9.92 GB FIT |
| `ibm-granite/granite-4.0-h-small` | 15.08 GB | 70.0 MB | 13.5 MB | 28.99 GB | 16.20 GB **OVER** | 16.58 GB FIT |
| `nvidia/Nemotron-H-8B-Base-8K` | 3.79 GB | 8.9 MB | 12.7 MB | 16.40 GB | 4.92 GB FIT | 5.29 GB FIT |
| `tiiuae/Falcon-H1-7B-Base` | 3.55 GB | 10.3 MB | 8.2 MB | 12.02 GB | 4.89 GB FIT | 5.93 GB FIT |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 37.35 GB | 294.0 MB | 6.9 MB | 44.22 GB | 38.54 GB **OVER** | 39.10 GB FIT |
| `state-spaces/mamba2-2.7b` | 1.28 GB | 6.7 MB | 12.1 MB | 10.07 GB | 2.28 GB FIT | 2.28 GB FIT |

### T4 — KV cache, attention layers only

| donor | KV elems/tok/layer | kind | 4K fp16 | 4K 4-bit | 32K fp16 | 32K 4-bit | 128K fp16 | 128K 4-bit |
|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 512 | gqa | 112.0 MB | 28.0 MB | 896.0 MB | 224.0 MB | 3.50 GB | 896.0 MB |
| `Qwen/Qwen3-1.7B` | 2048 | gqa | 448.0 MB | 112.0 MB | 3.50 GB | 896.0 MB | 14.00 GB | 3.50 GB |
| `HuggingFaceTB/SmolLM2-1.7B` | 4096 | gqa | 768.0 MB | 192.0 MB | 6.00 GB | 1.50 GB | 24.00 GB | 6.00 GB |
| `microsoft/Phi-3-mini-4k-instruct` | 6144 | gqa | 1.50 GB | 384.0 MB | 12.00 GB | 3.00 GB | 48.00 GB | 12.00 GB |
| `allenai/OLMo-2-1124-7B` | 8192 | gqa | 2.00 GB | 512.0 MB | 16.00 GB | 4.00 GB | 64.00 GB | 16.00 GB |
| `mistralai/Mistral-7B-v0.3` | 2048 | gqa | 512.0 MB | 128.0 MB | 4.00 GB | 1.00 GB | 16.00 GB | 4.00 GB |
| `Qwen/Qwen3-8B` | 2048 | gqa | 576.0 MB | 144.0 MB | 4.50 GB | 1.12 GB | 18.00 GB | 4.50 GB |
| `Qwen/Qwen3-30B-A3B` | 1024 | gqa | 384.0 MB | 96.0 MB | 3.00 GB | 768.0 MB | 12.00 GB | 3.00 GB |
| `mistralai/Mixtral-8x7B-v0.1` | 2048 | gqa | 512.0 MB | 128.0 MB | 4.00 GB | 1.00 GB | 16.00 GB | 4.00 GB |
| `deepseek-ai/DeepSeek-V2-Lite` | 576 | mla | 121.5 MB | 30.4 MB | 972.0 MB | 243.0 MB | 3.80 GB | 972.0 MB |
| `allenai/OLMoE-1B-7B-0924` | 4096 | gqa | 512.0 MB | 128.0 MB | 4.00 GB | 1.00 GB | 16.00 GB | 4.00 GB |
| `openai/gpt-oss-20b` | 1024 | gqa | 99.0 MB | 24.8 MB | 771.0 MB | 192.8 MB | 3.00 GB | 768.8 MB |
| `Zyphra/Zamba2-2.7B` | 10240 | gqa | 720.0 MB | 180.0 MB | 5.62 GB | 1.41 GB | 22.50 GB | 5.62 GB |
| `ibm-granite/granite-4.0-h-small` | 2048 | gqa | 64.0 MB | 16.0 MB | 512.0 MB | 128.0 MB | 2.00 GB | 512.0 MB |
| `nvidia/Nemotron-H-8B-Base-8K` | 2048 | gqa | 64.0 MB | 16.0 MB | 512.0 MB | 128.0 MB | 2.00 GB | 512.0 MB |
| `tiiuae/Falcon-H1-7B-Base` | 512 | gqa | 176.0 MB | 44.0 MB | 1.38 GB | 352.0 MB | 5.50 GB | 1.38 GB |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 1024 | gqa | 96.0 MB | 24.0 MB | 768.0 MB | 192.0 MB | 3.00 GB | 768.0 MB |
| `state-spaces/mamba2-2.7b` | None | none | 0.0 KB | 0.0 KB | 0.0 KB | 0.0 KB | 0.0 KB | 0.0 KB |

### T5 — per-token byte traffic and the three time columns (weights only, no KV)

| donor | active params/tok | proj+head fp32 bytes/tok | r(proj) GB/s | ternary MLP bytes/tok | ternary expert bytes/tok | expert calls/tok | tok/s PESS | tok/s OPT | tok/s **MEASURED-ONLY** |
|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 1544M | 1.44 GB | 39.87 | 553.3 MB | 0.0 KB | 0 | 2.75 | 8.23 | **4.83** |
| `Qwen/Qwen3-1.7B` | 1721M | 2.47 GB | 39.87 | 505.5 MB | 0.0 KB | 0 | 2.05 | 6.04 | **3.58** |
| `HuggingFaceTB/SmolLM2-1.7B` | 1711M | 1.88 GB | 39.87 | 577.7 MB | 0.0 KB | 0 | 2.39 | 7.03 | **4.16** |
| `microsoft/Phi-3-mini-4k-instruct` | 3723M | 4.87 GB | 39.87 | 1.13 GB | 0.0 KB | 0 | 1.15 | 3.18 | **1.96** |
| `allenai/OLMo-2-1124-7B` | 6887M | 9.53 GB | 39.87 | 2.02 GB | 0.0 KB | 0 | 0.78 | 1.90 | **1.27** |
| `mistralai/Mistral-7B-v0.3` | 7114M | 5.50 GB | 39.87 | 2.63 GB | 0.0 KB | 0 | 0.84 | 2.23 | **1.41** |
| `Qwen/Qwen3-8B` | 7568M | 7.94 GB | 39.87 | 2.54 GB | 0.0 KB | 0 | 0.73 | 1.89 | **1.21** |
| `Qwen/Qwen3-30B-A3B` | 3042M | 4.58 GB | 39.87 | 0.0 KB | 869.2 MB | 384 | 0.98 | 3.45 | **1.75** |
| `mistralai/Mixtral-8x7B-v0.1` | 12749M | 5.49 GB | 39.87 | 0.0 KB | 5.26 GB | 64 | 0.42 | 1.87 | **0.74** |
| `deepseek-ai/DeepSeek-V2-Lite` | 2451M | 2.18 GB | 39.87 | 32.2 MB | 861.7 MB | 182 | 1.52 | 5.95 | **2.73** |
| `allenai/OLMoE-1B-7B-0924` | 1179M | 1.39 GB | 39.87 | 0.0 KB | 386.0 MB | 128 | 2.78 | 10.34 | **4.97** |
| `openai/gpt-oss-20b` | 3607M | 4.54 GB | 39.87 | 0.0 KB | 1.12 GB | 96 | 1.13 | 3.87 | **1.94** |
| `Zyphra/Zamba2-2.7B` | 6949M | 10.07 GB | 39.87 | 1.98 GB | 0.0 KB | 0 | 1.79 | 2.51 | **2.22** |
| `ibm-granite/granite-4.0-h-small` | 8803M | 15.92 GB | 39.87 | 0.0 KB | 2.12 GB | 440 | 0.89 | 1.85 | **1.25** |
| `nvidia/Nemotron-H-8B-Base-8K` | 7564M | 12.43 GB | 39.87 | 1.97 GB | 0.0 KB | 0 | 1.71 | 2.20 | **2.02** |
| `tiiuae/Falcon-H1-7B-Base` | 7186M | 8.21 GB | 39.87 | 2.32 GB | 0.0 KB | 0 | 0.78 | 1.96 | **1.28** |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 3463M | 6.71 GB | 39.87 | 0.0 KB | 798.2 MB | 528 | 1.77 | 4.01 | **2.62** |
| `state-spaces/mamba2-2.7b` | 2702M | 10.07 GB | 39.87 | 0.0 KB | 0.0 KB | 0 | 3.34 | 3.67 | **3.62** |

Projection rate is no longer a curve lookup. Every donor is priced on the **measured donor stream, 37.74 ± 0.18 GB/s** (`DONOR_PROJ_RATE.md` §5.2) — the whole per-token proj+head traffic timed in layer order, which is the same object this term models. The retired 34.0 floor was never measured and is 11% below it.


### T6 — tok/s including KV-cache read traffic, vs the sealed ≥10 tok/s gate

Reported as **central [interval]**. The single-corner "MEASURED-ONLY" column is **withdrawn** (audit F1): it took the pessimistic edge of every bracket at once and that is what produced "zero donors pass". `nat` = the donor's own `max_position_embeddings`; a context beyond it is marked `>nat` and is **not natively servable** without RoPE extension or the recall tier.

| donor | nat ctx | 32K central [lo..hi] | 128K central [lo..hi] | UNCONVERTED | **CONVERTED** central [lo..hi] | gate | KV-free |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 131072 | 4.70 [2.71..7.88] | 4.36 [2.58..7.00] | 4.70 | 11.27 [10.06..11.85] | **PASS** | 11.64 |
| `Qwen/Qwen3-1.7B` | 40960 | 3.31 [1.95..5.35] | 2.71 [1.72..3.98] `>nat` | 3.31 | 7.67 [6.86..8.02] | **FAIL** | 8.38 |
| `HuggingFaceTB/SmolLM2-1.7B` | 8192 | 3.59 [2.18..5.59] `>nat` | 2.54 [1.72..3.46] `>nat` | 7.76 | 9.75 [8.70..10.25] | FAIL (KV-only) | 10.23 |
| `microsoft/Phi-3-mini-4k-instruct` | 4096 | 1.70 [1.05..2.58] `>nat` | 1.22 [0.84..1.65] `>nat` | 4.20 | 4.16 [3.74..4.34] | **FAIL** | 4.24 |
| `allenai/OLMo-2-1124-7B` | 4096 | 1.12 [0.72..1.60] `>nat` | 0.84 [0.59..1.09] `>nat` | 2.35 | 2.26 [2.06..2.34] | **FAIL** | 2.30 |
| `mistralai/Mistral-7B-v0.3` | 32768 | 1.36 [0.82..2.12] | 1.23 [0.77..1.83] `>nat` | 1.36 | 2.43 [2.22..2.53] | **FAIL** | 2.51 |
| `Qwen/Qwen3-8B` | 40960 | 1.17 [0.72..1.79] | 1.06 [0.67..1.56] `>nat` | 1.17 | 2.05 [1.87..2.12] | **FAIL** | 2.11 |
| `Qwen/Qwen3-30B-A3B` | 40960 | 1.69 [0.96..3.25] | 1.54 [0.91..2.76] `>nat` | 1.69 | 3.42 [2.47..5.14] | **FAIL** | 3.54 |
| `mistralai/Mixtral-8x7B-v0.1` | 32768 | 0.72 [0.41..1.79] | 0.69 [0.40..1.58] `>nat` | 0.72 | 0.95 [0.60..2.08] | **FAIL** | 0.96 |
| `deepseek-ai/DeepSeek-V2-Lite` | 163840 | 2.68 [1.51..5.75] | 2.56 [1.47..5.23] | 2.68 | 4.67 [3.16..8.34] | **FAIL** | 4.74 |
| `allenai/OLMoE-1B-7B-0924` | 4096 | 4.41 [2.58..8.26] `>nat` | 3.29 [2.14..5.15] `>nat` | 9.05 | 9.27 [6.43..15.33] | **FAIL** | 9.41 |
| `openai/gpt-oss-20b` | 131072 | 1.93 [1.12..3.80] | 1.87 [1.10..3.61] | 1.93 | 3.03 [2.13..4.88] | **FAIL** | 3.05 |
| `Zyphra/Zamba2-2.7B` | 4096 | 2.06 [1.68..2.31] `>nat` | 1.68 [1.41..1.87] `>nat` | 2.57 | 2.47 [2.24..2.57] | **FAIL** | 2.56 |
| `ibm-granite/granite-4.0-h-small` | 131072 | 1.24 [0.89..1.84] | 1.23 [0.88..1.81] | 1.24 | 1.28 [0.94..1.84] | **FAIL** | 1.30 |
| `nvidia/Nemotron-H-8B-Base-8K` | 8192 | 2.01 [1.70..2.19] `>nat` | 1.97 [1.67..2.15] `>nat` | 2.20 | 2.13 [1.90..2.22] | **FAIL** | 2.15 |
| `tiiuae/Falcon-H1-7B-Base` | 262144 | 1.27 [0.78..1.93] | 1.22 [0.76..1.84] | 1.27 | 2.14 [1.95..2.21] | **FAIL** | 2.16 |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 262144 | 2.59 [1.76..3.94] | 2.50 [1.71..3.73] | 2.59 | 3.10 [2.32..4.31] | **FAIL** | 3.20 |
| `state-spaces/mamba2-2.7b` | ? | 3.62 [3.34..3.67] | 3.62 [3.34..3.67] | 3.62 | 3.40 [2.98..3.55] | **FAIL** | 3.40 |

KV priced at the measured aggregate DRAM stream [40–44 GB/s], central 42; 4-bit KV assumed, which **does not exist in this engine** (audit F8) — see the fp16 column in the arithmetic doc.

**`gate @SKU-A` reads PASS only when the LOWER BOUND clears 10 tok/s.** A central estimate on the right side of a gate whose interval straddles it is reported as straddling, not as a pass.

**KV-free central** isolates the owner's directive: a donor may **not** be eliminated solely because full-native KV does not fit or stream — that is what the recall tier absorbs. Any donor failing the gate on KV but clearing it KV-free is a *recall-tier* question, not an elimination.


### T7 — head alone (mandate §5's named first-class problem)

| donor | V | head bytes fp32 | ms/token fp32 | head bytes ternary | ms/token ternary (LUT bracket) | head share of the fp32 per-token stream |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 151936 | 890.2 MB | 23.0–25.9 | 111.9 MB | 6.9–27.9 | 60% |
| `Qwen/Qwen3-1.7B` | 151936 | 1.16 GB | 30.7–34.6 | 149.0 MB | 9.2–37.2 | 47% |
| `HuggingFaceTB/SmolLM2-1.7B` | 49152 | 384.0 MB | 9.9–11.2 | 48.2 MB | 3.0–12.0 | 20% |
| `microsoft/Phi-3-mini-4k-instruct` | 32064 | 375.8 MB | 9.7–10.9 | 47.1 MB | 2.9–11.8 | 8% |
| `allenai/OLMo-2-1124-7B` | 100352 | 1.53 GB | 40.6–45.7 | 196.4 MB | 12.1–49.0 | 16% |
| `mistralai/Mistral-7B-v0.3` | 32768 | 512.0 MB | 13.2–14.9 | 64.1 MB | 4.0–16.0 | 9% |
| `Qwen/Qwen3-8B` | 151936 | 2.32 GB | 61.4–69.1 | 297.3 MB | 18.3–74.2 | 29% |
| `Qwen/Qwen3-30B-A3B` | 151936 | 1.16 GB | 30.7–34.6 | 149.0 MB | 9.2–37.2 | 25% |
| `mistralai/Mixtral-8x7B-v0.1` | 32000 | 500.0 MB | 12.9–14.6 | 62.6 MB | 3.9–15.6 | 9% |
| `deepseek-ai/DeepSeek-V2-Lite` | 102400 | 800.0 MB | 20.7–23.3 | 100.4 MB | 6.2–25.1 | 36% |
| `allenai/OLMoE-1B-7B-0924` | 50304 | 393.0 MB | 10.2–11.4 | 49.3 MB | 3.0–12.3 | 28% |
| `openai/gpt-oss-20b` | 201088 | 2.16 GB | 57.1–64.3 | 276.9 MB | 17.1–69.1 | 48% |
| `Zyphra/Zamba2-2.7B` | 32000 | 312.5 MB | 8.1–9.1 | 39.2 MB | 2.4–9.8 | 3% |
| `ibm-granite/granite-4.0-h-small` | 100352 | 1.53 GB | 40.6–45.7 | 196.4 MB | 12.1–49.0 | 10% |
| `nvidia/Nemotron-H-8B-Base-8K` | 131072 | 2.00 GB | 53.0–59.7 | 256.5 MB | 15.8–64.0 | 16% |
| `tiiuae/Falcon-H1-7B-Base` | 130048 | 1.49 GB | 39.4–44.4 | 191.0 MB | 11.8–47.7 | 18% |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 151936 | 1.16 GB | 30.7–34.6 | 149.0 MB | 9.2–37.2 | 17% |
| `state-spaces/mamba2-2.7b` | 50288 | 491.1 MB | 12.7–14.3 | 61.6 MB | 3.8–15.4 | 5% |

### T8 — MoE granularity vs the ρ-safe 48 KB chunk

| donor | experts | top-k | expert width | expert chunk (ternary+scales) | ≥48 KB? |
|---|---|---|---|---|---|
| `Qwen/Qwen3-30B-A3B` | 128 | 8 | 768 | 2318 KB | yes |
| `mistralai/Mixtral-8x7B-v0.1` | 8 | 2 | 14336 | 86144 KB | yes |
| `deepseek-ai/DeepSeek-V2-Lite` | 64 | 6 | 1408 | 4243 KB | yes |
| `allenai/OLMoE-1B-7B-0924` | 64 | 8 | 1024 | 3088 KB | yes |
| `openai/gpt-oss-20b` | 32 | 4 | 2880 | 12184 KB | yes |
| `ibm-granite/granite-4.0-h-small` | 72 | 10 | 768 | 4630 KB | yes |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 512 | 10 | 512 | 1548 KB | yes |

### T9 — reproducibility manifest

| donor | resolved revision SHA | config.json sha256 | bytes | status |
|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | `0e8c8aa86468aba09c9d32157ff4bc23…` | 684 | OK |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `1ddb5b89ebc90dcb417a45c213d81857…` | 726 | OK |
| `HuggingFaceTB/SmolLM2-1.7B` | `effd688a12921b4cc83e3312b6feb579f70f9c71` | `33397f6af8090f7290f27ce6fb1cd23a…` | 635 | OK |
| `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` | `072d4df63228ef806a6b2b2f02a93f1d…` | 967 | OK |
| `allenai/OLMo-2-1124-7B` | `7df9a82518afdecae4e8c026b27adccc8c1f0032` | `f1e210de0ba704c579768c154bf57e36…` | 623 | OK |
| `mistralai/Mistral-7B-v0.3` | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | `affafc6478ec0fd07a32f0ca57aa2fc5…` | 601 | OK |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | `f7c4eadfbbf522470667b797a3c89be2…` | 728 | OK |
| `meta-llama/Llama-3.2-1B` | — | — | — | **UNAVAILABLE** — GATED: 401 Client Error. (Request ID: Root=1-6a86ddaa-4855c3b137f4eab75158ba36;67ffb929-ef |
| `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | `2850ddb3bf7aecad20b611e2d44f3077…` | 963 | OK |
| `mistralai/Mixtral-8x7B-v0.1` | `fc7ac94680e38d7348cfa806e51218e6273104b0` | `9d56d04b36d0fd12ff54ae4c5bac769c…` | 720 | OK |
| `deepseek-ai/DeepSeek-V2-Lite` | `604d5664dddd88a0433dbae533b7fe9472482de0` | `f346286b0f1c8b044252fd54cb4fa78b…` | 1522 | OK |
| `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | `3643aa880d2f1c9b418156269ae791c7…` | 759 | OK |
| `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | `3a2a26ded679375b7928ddeca59764df…` | 1806 | OK |
| `ai21labs/AI21-Jamba-Mini-1.6` | — | — | — | **UNAVAILABLE** — GATED: 401 Client Error. (Request ID: Root=1-6a86ddac-27f03ef621376c0713e7c5ed;dd59388d-35 |
| `Zyphra/Zamba2-2.7B` | `31afeeac4c66b4851a54290ba57c995a68c87861` | `0497ca3001366ed8eaef1030c84f856f…` | 2020 | OK |
| `ibm-granite/granite-4.0-h-small` | `b8c0982bab7fde4eb48110f5a069527c008fab39` | `8616e9f0b30e6fac9696f7c1e1dbd08f…` | 1799 | OK |
| `nvidia/Nemotron-H-8B-Base-8K` | `94ea861e008c2dfced3e8e1302094024077aa04e` | `81e822f85d6471312bcc0ccd34cc6278…` | 1504 | OK |
| `tiiuae/Falcon-H1-7B-Base` | `c9a4cbb95c01b1ede39f69eda083d03d8903b8f0` | `a0ed9a14b6a71ed4701b72549eb07e17…` | 1637 | OK |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7` | `2d483c7cabad7c8704478ed4038fa7e7…` | 1154 | OK |
| `state-spaces/mamba2-2.7b` | `99b226cc377d131cccc610ed4346db564f381f1e` | `0955f133a3eeb902ed11e1a65d0be293…` | 331 | OK |
