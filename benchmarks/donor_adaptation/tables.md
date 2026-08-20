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
| `Qwen/Qwen2.5-1.5B` | 1544M | 1.44 GB | 34–36* | 553.3 MB | 0.0 KB | 0 | 5.44 | 12.95 | **10.36** |
| `Qwen/Qwen3-1.7B` | 1721M | 2.47 GB | 34–36* | 505.5 MB | 0.0 KB | 0 | 4.89 | 9.53 | **8.02** |
| `HuggingFaceTB/SmolLM2-1.7B` | 1711M | 1.88 GB | 34–36* | 577.7 MB | 0.0 KB | 0 | 4.91 | 10.92 | **8.90** |
| `microsoft/Phi-3-mini-4k-instruct` | 3723M | 4.87 GB | 34–36* | 1.13 GB | 0.0 KB | 0 | 2.26 | 4.62 | **3.85** |
| `allenai/OLMo-2-1124-7B` | 6887M | 9.53 GB | 34–36* | 2.02 GB | 0.0 KB | 0 | 1.22 | 2.43 | **2.04** |
| `mistralai/Mistral-7B-v0.3` | 7114M | 5.50 GB | 34–36* | 2.63 GB | 0.0 KB | 0 | 1.18 | 3.03 | **2.37** |
| `Qwen/Qwen3-8B` | 7568M | 7.94 GB | 34–36* | 2.54 GB | 0.0 KB | 0 | 1.11 | 2.52 | **2.04** |
| `Qwen/Qwen3-30B-A3B` | 3042M | 4.58 GB | 34–36* | 0.0 KB | 869.2 MB | 384 | 2.74 | 5.17 | **2.76** |
| `mistralai/Mixtral-8x7B-v0.1` | 12749M | 5.49 GB | 34–36* | 0.0 KB | 5.26 GB | 64 | 0.66 | 2.01 | **0.66** |
| `deepseek-ai/DeepSeek-V2-Lite` | 2451M | 2.18 GB | 34–36* | 32.2 MB | 861.7 MB | 182 | 3.41 | 8.22 | **3.49** |
| `allenai/OLMoE-1B-7B-0924` | 1179M | 1.39 GB | 34–36* | 0.0 KB | 386.0 MB | 128 | 7.07 | 15.06 | **7.13** |
| `openai/gpt-oss-20b` | 3607M | 4.54 GB | 34–36* | 0.0 KB | 1.12 GB | 96 | 2.33 | 4.84 | **2.33** |
| `Zyphra/Zamba2-2.7B` | 6949M | 10.07 GB | 34–36* | 1.98 GB | 0.0 KB | 0 | 1.21 | 2.35 | **1.98** |
| `ibm-granite/granite-4.0-h-small` | 8803M | 15.92 GB | 34–36* | 0.0 KB | 2.12 GB | 440 | 0.95 | 1.63 | **0.96** |
| `nvidia/Nemotron-H-8B-Base-8K` | 7564M | 12.43 GB | 34–36* | 1.97 GB | 0.0 KB | 0 | 1.12 | 2.02 | **1.73** |
| `tiiuae/Falcon-H1-7B-Base` | 7186M | 8.21 GB | 34–36* | 2.32 GB | 0.0 KB | 0 | 1.17 | 2.55 | **2.09** |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 3463M | 6.71 GB | 34–36* | 0.0 KB | 798.2 MB | 528 | 2.41 | 3.94 | **2.43** |
| `state-spaces/mamba2-2.7b` | 2702M | 10.07 GB | 34–36* | 0.0 KB | 0.0 KB | 0 | 3.14 | 3.33 | **3.14** |

`*` = above the last measured point of the §1b curve (96 MB): the 34–36 GB/s asymptote, an **extrapolation**.


### T6 — tok/s including KV-cache read traffic, vs the sealed ≥10 tok/s gate

| donor | 4K pess..opt (meas-only) | 32K pess..opt (meas-only) | 128K pess..opt (meas-only) | gate @32K | gate @128K |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 5.42..12.84 (10.28) | 5.27..12.11 (9.76) | 4.82..10.14 (8.33) | **FAIL** | **FAIL** |
| `Qwen/Qwen3-1.7B` | 4.83..9.29 (7.84) | 4.39..7.92 (6.75) | 3.35..5.25 (4.58) | **FAIL** | **FAIL** |
| `HuggingFaceTB/SmolLM2-1.7B` | 4.80..10.40 (8.52) | 4.10..7.80 (6.55) | 2.74..4.20 (3.66) | **FAIL** | **FAIL** |
| `microsoft/Phi-3-mini-4k-instruct` | 2.21..4.43 (3.70) | 1.91..3.45 (2.94) | 1.31..1.96 (1.72) | **FAIL** | **FAIL** |
| `allenai/OLMo-2-1124-7B` | 1.20..2.36 (1.98) | 1.08..1.96 (1.67) | 0.80..1.25 (1.09) | **FAIL** | **FAIL** |
| `mistralai/Mistral-7B-v0.3` | 1.18..3.00 (2.35) | 1.15..2.82 (2.23) | 1.05..2.34 (1.89) | **FAIL** | **FAIL** |
| `Qwen/Qwen3-8B` | 1.11..2.50 (2.03) | 1.08..2.36 (1.92) | 0.98..1.97 (1.64) | **FAIL** | **FAIL** |
| `Qwen/Qwen3-30B-A3B` | 2.72..5.11 (2.75) | 2.60..4.72 (2.62) | 2.24..3.75 (2.26) | **FAIL** | **FAIL** |
| `mistralai/Mixtral-8x7B-v0.1` | 0.66..2.00 (0.66) | 0.65..1.92 (0.65) | 0.62..1.68 (0.62) | **FAIL** | **FAIL** |
| `deepseek-ai/DeepSeek-V2-Lite` | 3.40..8.17 (3.48) | 3.33..7.85 (3.41) | 3.13..6.90 (3.20) | **FAIL** | **FAIL** |
| `allenai/OLMoE-1B-7B-0924` | 6.91..14.40 (6.96) | 5.94..11.01 (5.98) | 4.02..6.10 (4.04) | **FAIL** | **FAIL** |
| `openai/gpt-oss-20b` | 2.33..4.82 (2.33) | 2.30..4.73 (2.31) | 2.22..4.44 (2.23) | **FAIL** | **FAIL** |
| `Zyphra/Zamba2-2.7B` | 1.21..2.33 (1.96) | 1.16..2.17 (1.84) | 1.02..1.78 (1.52) | **FAIL** | **FAIL** |
| `ibm-granite/granite-4.0-h-small` | 0.95..1.63 (0.96) | 0.95..1.62 (0.95) | 0.94..1.60 (0.95) | **FAIL** | **FAIL** |
| `nvidia/Nemotron-H-8B-Base-8K` | 1.11..2.02 (1.73) | 1.11..2.01 (1.72) | 1.10..1.97 (1.69) | **FAIL** | **FAIL** |
| `tiiuae/Falcon-H1-7B-Base` | 1.17..2.55 (2.09) | 1.16..2.50 (2.05) | 1.12..2.35 (1.94) | **FAIL** | **FAIL** |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 2.40..3.93 (2.43) | 2.38..3.87 (2.40) | 2.29..3.67 (2.32) | **FAIL** | **FAIL** |
| `state-spaces/mamba2-2.7b` | 3.14..3.33 (3.14) | 3.14..3.33 (3.14) | 3.14..3.33 (3.14) | **FAIL** | **FAIL** |

KV priced at the measured aggregate DRAM stream [40–44 GB/s]; 4-bit KV assumed, which is the most favourable of the two KV precisions asked for.


### T7 — head alone (mandate §5's named first-class problem)

| donor | V | head bytes fp32 | ms/token fp32 | head bytes ternary | ms/token ternary (LUT bracket) | head share of the fp32 per-token stream |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 151936 | 890.2 MB | 25.9–27.5* | 111.9 MB | 6.9–27.9 | 60% |
| `Qwen/Qwen3-1.7B` | 151936 | 1.16 GB | 34.6–36.6* | 149.0 MB | 9.2–37.2 | 47% |
| `HuggingFaceTB/SmolLM2-1.7B` | 49152 | 384.0 MB | 11.2–11.8* | 48.2 MB | 3.0–12.0 | 20% |
| `microsoft/Phi-3-mini-4k-instruct` | 32064 | 375.8 MB | 10.9–11.6* | 47.1 MB | 2.9–11.8 | 8% |
| `allenai/OLMo-2-1124-7B` | 100352 | 1.53 GB | 45.7–48.4* | 196.4 MB | 12.1–49.0 | 16% |
| `mistralai/Mistral-7B-v0.3` | 32768 | 512.0 MB | 14.9–15.8* | 64.1 MB | 4.0–16.0 | 9% |
| `Qwen/Qwen3-8B` | 151936 | 2.32 GB | 69.1–73.2* | 297.3 MB | 18.3–74.2 | 29% |
| `Qwen/Qwen3-30B-A3B` | 151936 | 1.16 GB | 34.6–36.6* | 149.0 MB | 9.2–37.2 | 25% |
| `mistralai/Mixtral-8x7B-v0.1` | 32000 | 500.0 MB | 14.6–15.4* | 62.6 MB | 3.9–15.6 | 9% |
| `deepseek-ai/DeepSeek-V2-Lite` | 102400 | 800.0 MB | 23.3–24.7* | 100.4 MB | 6.2–25.1 | 36% |
| `allenai/OLMoE-1B-7B-0924` | 50304 | 393.0 MB | 11.4–12.1* | 49.3 MB | 3.0–12.3 | 28% |
| `openai/gpt-oss-20b` | 201088 | 2.16 GB | 64.3–68.1* | 276.9 MB | 17.1–69.1 | 48% |
| `Zyphra/Zamba2-2.7B` | 32000 | 312.5 MB | 9.1–9.6* | 39.2 MB | 2.4–9.8 | 3% |
| `ibm-granite/granite-4.0-h-small` | 100352 | 1.53 GB | 45.7–48.4* | 196.4 MB | 12.1–49.0 | 10% |
| `nvidia/Nemotron-H-8B-Base-8K` | 131072 | 2.00 GB | 59.7–63.2* | 256.5 MB | 15.8–64.0 | 16% |
| `tiiuae/Falcon-H1-7B-Base` | 130048 | 1.49 GB | 44.4–47.0* | 191.0 MB | 11.8–47.7 | 18% |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 151936 | 1.16 GB | 34.6–36.6* | 149.0 MB | 9.2–37.2 | 17% |
| `state-spaces/mamba2-2.7b` | 50288 | 491.1 MB | 14.3–15.1* | 61.6 MB | 3.8–15.4 | 5% |

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
