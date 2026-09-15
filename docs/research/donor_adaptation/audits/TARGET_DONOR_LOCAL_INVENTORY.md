# Target donor local inventory

Status: factual inventory only.  This file records repository evidence available at inspection time; it is not a recommendation and does not infer missing model metadata.

## Scope and evidence rules

The project candidate list is `benchmarks/donor_adaptation/donor_inventory.py` lines 35–59.  This inventory includes every listed candidate with a nominal total or active size at/above 7B, plus the separately documented real Qwen2.5-Coder-7B donor.  The analyzer used for the config-derived fields is `donor_inventory.py` lines 919–996; its own warnings/partial states are retained.  “Real” means a pretrained weight artifact was actually used; “synthetic” means a shape-correct noise export; “derived” means arithmetic/model-fit output rather than a candidate run; “literature” is not used here because this inventory does not import external metadata.

`D` = hidden width, `F` = intermediate/FFN width, `L` = layer count, `H/KV` = attention and key/value head counts, `V` = vocabulary, and `tied` = the config’s embedding/output tying flag.  Active weights are analyzer-computed active parameters per token, not an externally remembered “active” count.  The manifest’s safetensors total is retained as the recorded parameter total where available.

## Candidate rows

### Qwen/Qwen2.5-Coder-7B

- Exact model/revision: `Qwen/Qwen2.5-Coder-7B`; revision `0396a76181e127dfc13e5c5ec48a8cee09938b02` is recorded in the E66 result (`benchmarks/donor_adaptation/engine/results/e66/e66_one_byte_at_7b_run2.json` lines 19–20, 127–128, 149–150).  E7 records 7.617B parameters and approximately 7.072B active/token (`probes/E7_REAL_LARGE_DONOR.md` §1, lines 12–15).
- Shape: `D=3584`, `F=18944`, `L=28`, `H=28`, `KV=4`, `HD=128`, `V=152064`, `tied=false`; E7 records this in §1, lines 12–15, and the engine parity result records the same values as `D`, `F`, `L`, `heads`, `hd`, `V`, and `tied=0` (`benchmarks/donor_adaptation/engine/results/e7/parity.json` lines 5–12).  E18 independently constructs the same shape in `e18_ladder_bandwidth.py` line 130.
- Local status: real pretrained fp32 export `qwen25-coder7b_f32.bin`, 30,462,466,100 bytes, and packed export `qwen25-coder7b_p.bin`, 5,722,636,340 bytes, are recorded in E7 §1, lines 12–15.  The repository itself does not contain those weight binaries.  E7 says nothing larger than this real donor exists offline (`INDEX.md` lines 70–72 and 295–297).
- Exporter/engine evidence: real artifact loaded and matched PyTorch on the full E7 parity path; E7 records 160/160 greedy-token identity and top-1 1.0000 (`INDEX.md` lines 137 and 295–297).  The packed R0 conversion used `--quant packed --rule R0 --head-ternary --fold layers` (E7 lines 14–15); E15 records that this packed artifact did not predict the donor (`INDEX.md` line 169).  E66 records later one-byte/int8 engine paths with `attn=avx4` controls (`probes/E66_ONE_BYTE_AT_SEVEN_BILLION.md` §§1, 3, 5, lines 31–39, 51–82).
- Measured quality/rate: fp32 donor 4.460 tok/s at context 300, 0.45% spread, 31.5G weights/s; the packed R0 artifact has a measured 6.79 tok/s standing rate but does not predict (`benchmarks/donor_adaptation/engine/e18_ladder_bandwidth.py` lines 34, 129–139; `probes/E7_REAL_LARGE_DONOR.md` lines 441–443).  Its packed quality is BPB 5.299200075 versus chance 4.070106 and 0/160 rank in E15 (`probes/E7_REAL_LARGE_DONOR.md` lines 428–433; `INDEX.md` line 169).  E66’s one-byte/int8 score arms provide quality only: +0.000378 BPB from fp32 and 137/160 rank, with no rate (`probes/E66_ONE_BYTE_AT_SEVEN_BILLION.md` lines 56–63, 79–109).
- Evidence class: real for the pretrained fp32 artifact and its measured parity/rate; real-but-converted for the packed R0 rate and quality; E66 one-byte/int8 evidence is quality-only.  Speed exists only for the non-predicting packed artifact; no rate exists for a working one-byte/int8 conversion.
- Blocking gaps: no rate exists for a working one-byte/int8 conversion; no 10B-scale pretrained artifact.
- References: `benchmarks/donor_adaptation/engine/results/e7/parity.json` lines 2, 5–13; `benchmarks/donor_adaptation/engine/e18_ladder_bandwidth.py` lines 34, 129–139; `probes/E7_REAL_LARGE_DONOR.md` §§1, 4, 5, 14; `probes/E15_DOES_THE_7B_PREDICT.md` §4; `probes/E66_ONE_BYTE_AT_SEVEN_BILLION.md` §§1, 3, 5; `audits/AXIS_COVERAGE_AND_NO_DUPLICATION_MAP.md` §candidate coverage, line 81.

### Qwen/Qwen3-8B

- Exact model/revision: repo id `Qwen/Qwen3-8B`; manifest revision `b968826d9c46dd6066d109eabc6255188de91218` (`configs/_manifest.json`, JSON member for this repo).
- Config-derived shape: `D=4096`, `F=12288`, `L=36`, `H=32`, `KV=8`, `V=151936`, `tied=false`, dense Qwen3 architecture; config lines 1–30, with `hidden_size` line 11, `intermediate_size` 13, heads 17 and 19, layers 18, and `vocab_size` 29.  Analyzer: total 8.190726144B computed / 8.190735360B manifest; active/token 7.568396288B.
- Local status: config downloaded (`OK` in manifest); no local pretrained weights or weight download is evidenced.  `INDEX.md` lines 70–72 calls the local Qwen3-8B object a 1MB config stub.
- Exporter/engine compatibility: uniform dense attention shape is the family represented by the synthetic exporter/engine, but no real Qwen3-8B export/load is recorded.  The engine-scale target uses synthetic `T10`, not this donor (`E3` §1.1, lines 85–107).
- Measured quality/rate: no candidate-specific real quality or rate.  Derived engine budget table gives head 622M, fitted `f=7.928ms`, and 152% of the 50-tok/s budget at context 300 (`INDEX.md` lines 535–537); the older 300/800 table is also derived from shape pricing, not a donor run.
- Evidence class: config-derived and engine-fit derived; no real-weight quality evidence.
- Blocking gaps: pretrained weights, exporter/load proof for this exact revision, and candidate-specific quality/rate.
- References: `configs/Qwen__Qwen3-8B.json` JSON root and lines 9, 11, 13, 17–19, 24, 29; `configs/_manifest.json`; `probes/E3_ENGINE_AT_TARGET_SCALE.md` §§1.1, 4.4–4.8; `INDEX.md` lines 70–72, 535–537.

### allenai/OLMo-2-1124-7B

- Exact model/revision: repo id `allenai/OLMo-2-1124-7B`; manifest revision `7df9a82518afdecae4e8c026b27adccc8c1f0032`.
- Config-derived shape: `D=4096`, `F=11008`, `L=32`, `H=32`, `KV=32`, `V=100352`, `tied=false`, `Olmo2ForCausalLM`; config lines 1–25, key lines 9, 11, 14–16, 25.  Analyzer total 7.298355200B / manifest 7.298617344B; active/token 6.887313408B.
- Local status: config `OK`; no pretrained weights evidenced.  No candidate-specific local binary is named in the tracked donor-adaptation results.
- Exporter/engine compatibility: dense uniform attention shape is representable by the shape engine, but no real export/load run for OLMo-2 is recorded.
- Measured quality/rate: no real candidate quality/rate.  Derived tables report `f=7.047ms`, 94% of the derived 50-tok/s budget at context 300 (`INDEX.md` lines 535–537); the same source labels the fit as not a real donor run.
- Evidence class: config-derived and derived pricing only.
- Blocking gaps: pretrained weights, exact exporter proof, quality, and candidate-specific rate.
- References: `configs/allenai__OLMo-2-1124-7B.json` JSON root and lines 9, 11, 14–16, 21, 25; `configs/_manifest.json`; `INDEX.md` lines 535–537, 556–560.

### mistralai/Mistral-7B-v0.3

- Exact model/revision: repo id `mistralai/Mistral-7B-v0.3`; manifest revision `caa1feb0e54d415e2df31207e5f4e273e33509b1`.
- Config-derived shape: `D=4096`, `F=14336`, `L=32`, `H=32`, `KV=8`, `V=32768`, `tied=false`, `MistralForCausalLM`; config lines 1–24, key lines 9, 11, 14, 16, 20, 24.  Analyzer total/manifest 7.248023552B; active/token 7.113805824B.
- Local status: config `OK`; no pretrained weight file is evidenced in the tracked project.
- Exporter/engine compatibility: dense uniform attention shape is representable; no real export/load run for this donor is recorded.
- Measured quality/rate: no real candidate quality/rate.  Derived table gives `f=7.047ms`, 31% of the context-300 budget and 141% at context 800 (`INDEX.md` lines 535–553).
- Evidence class: config-derived and derived pricing only.
- Blocking gaps: weights, exact exporter proof, quality, and candidate-specific rate.
- References: `configs/mistralai__Mistral-7B-v0.3.json` JSON root and lines 9, 11, 14, 16, 20, 24; `configs/_manifest.json`; `INDEX.md` lines 535–553.

### nvidia/Nemotron-H-8B-Base-8K

- Exact model/revision: repo id `nvidia/Nemotron-H-8B-Base-8K`; manifest revision `94ea861e008c2dfced3e8e1302094024077aa04e`.
- Config-derived shape: `D=4096`, `F=21504`, `L=52`, `H=32`, `KV=8`, `V=131072`, `tied=false`, architecture `NemotronHForCausalLM`; hybrid override pattern is `M-M-M-M*-...` and config has `attention_head_dim=128`, `mamba_head_dim=64`; config lines 1–57, key lines 18–19, 30–34, 42, 57.  Analyzer total 8.100819968B / manifest 8.100852736B; active/token 7.563949056B; native context 8192.
- Local status: config `OK`; no pretrained weights evidenced.
- Exporter/engine compatibility: hybrid/SSM architecture is outside the uniform dense-attention engine price path.  `INDEX.md` lines 556–560 explicitly says Nemotron-H is not priced by the fit; the old 97% figure was analogy only.
- Measured quality/rate: no candidate-specific quality or rate; old budget percentage is explicitly derived/analogy, not measured.
- Evidence class: config-derived; compatibility gap and old rate are documented derived/analogy evidence.
- Blocking gaps: hybrid exporter and engine implementation, pretrained weights, candidate-specific quality/rate.
- References: `configs/nvidia__Nemotron-H-8B-Base-8K.json` JSON root and lines 18–19, 30–34, 42, 57; `configs/_manifest.json`; `INDEX.md` lines 556–560.

### tiiuae/Falcon-H1-7B-Base

- Exact model/revision: repo id `tiiuae/Falcon-H1-7B-Base`; manifest revision `c9a4cbb95c01b1ede39f69eda083d03d8903b8f0`.
- Config-derived shape: `D=3072`, `F=12288`, `L=44`, `H=12`, `KV=2`, `head_dim=128`, `V=130048`, `tied=false`, `FalconH1ForCausalLM`; config lines 1–54, key lines 18, 24, 28–30, 38, 54.  The config contains Mamba fields (`mamba_d_state=256`, `mamba_n_heads=24`) and `attn_layer_indices=null`.  Analyzer total 7.585491040B / manifest 7.585648736B; active/token 7.185983584B; analyzer marks the row partial because the attention interpretation is inferred.
- Local status: config `OK`; no pretrained weights evidenced.
- Exporter/engine compatibility: hybrid attention+SSM; no uniform dense export path or candidate run is recorded.  The project’s donor table groups Falcon-H1 with hybrid/SSM candidates whose target-shape `f` was not measured (`INDEX.md` lines 556–559).
- Measured quality/rate: none candidate-specific.
- Evidence class: config-derived with analyzer-inferred hybrid interpretation; no measured candidate evidence.
- Blocking gaps: resolve architecture interpretation, implement/export hybrid format, weights, quality, and rate.
- References: `configs/tiiuae__Falcon-H1-7B-Base.json` JSON root and lines 18, 24, 28–30, 38, 54; `configs/_manifest.json`; `INDEX.md` lines 556–559.

### allenai/OLMoE-1B-7B-0924

- Exact model/revision: repo id `allenai/OLMoE-1B-7B-0924`; manifest revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
- Config-derived shape: `D=2048`, `F=1024`, `L=16`, `H=16`, `KV=16`, `V=50304`, `tied=false`, `num_experts=64`, `num_experts_per_tok=8`, architecture `OlmoeForCausalLM`; config lines 1–30, key lines 10, 12, 16–20, 26, 30.  Analyzer total 6.919096320B / manifest 6.919161856B; active/token 1.178929152B.
- Local status: config `OK`; no pretrained weights evidenced.
- Exporter/engine compatibility: MoE/non-uniform, so not priced by the dense fit; no real export/load run recorded (`INDEX.md` lines 556–560).
- Measured quality/rate: none candidate-specific; any dense-fit number would be inapplicable.
- Evidence class: config-derived only.
- Blocking gaps: MoE exporter/engine support, weights, quality, rate.
- References: `configs/allenai__OLMoE-1B-7B-0924.json` JSON root and lines 10, 12, 16–20, 26, 30; `configs/_manifest.json`; `INDEX.md` lines 556–560.

### Qwen/Qwen3-30B-A3B

- Exact model/revision: repo id `Qwen/Qwen3-30B-A3B`; manifest revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- Config-derived shape: `D=2048`, `F=6144`, `L=48`, `H=32`, `KV=4`, `head_dim=128`, `V=151936`, `tied=false`, `num_experts=128`, `num_experts_per_tok=8`, `moe_intermediate_size=768`, architecture `Qwen3MoeForCausalLM`; config lines 1–37, key lines 10, 12, 21–25, 32, 37.  Analyzer total 30.532110336B / manifest 30.532122624B; active/token 3.041855488B; analyzer status `PARTIAL` due its attention-per-layer sanity warning.
- Local status: config `OK`; E7 states the Qwen3-30B-A3B and Qwen3-Next-80B-A3B objects are 1MB config stubs, not weights (`E7` lines 24–30).
- Exporter/engine compatibility: MoE/non-uniform; no exact exporter/load path or candidate run.  Dense fit does not price this architecture (`INDEX.md` lines 556–560).
- Measured quality/rate: none.
- Evidence class: config-derived only; local stub status is direct project documentation.
- Blocking gaps: pretrained weights, MoE exporter/engine support, quality, rate, and resolve analyzer partial warning.
- References: `configs/Qwen__Qwen3-30B-A3B.json` JSON root and lines 10, 12, 21–25, 32, 37; `configs/_manifest.json`; `probes/E7_REAL_LARGE_DONOR.md` lines 24–30; `INDEX.md` lines 556–560.

### deepseek-ai/DeepSeek-V2-Lite

- Exact model/revision: repo id `deepseek-ai/DeepSeek-V2-Lite`; manifest revision `604d5664dddd88a0433dbae533b7fe9472482de0`.
- Config-derived shape: `D=2048`, `F=10944`, `L=27`, `H=16`, `KV=16`, `V=102400`, `tied=false`, `DeepseekV2ForCausalLM`; MLA fields `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `kv_lora_rank=512`, `v_head_dim=128`; MoE `n_routed_experts=64`, top-k 6, `moe_intermediate_size=1408`; config lines 1–58, key lines 17, 19, 22, 29–32, 51, 58.  Analyzer total 15.706470400B / manifest 15.706484224B; active/token 2.451421184B; native context 163840.
- Local status: config `OK`; no pretrained weights evidenced.
- Exporter/engine compatibility: MLA plus MoE is outside the uniform dense exporter/engine path; no exact export/load run.  Project list marks DeepSeek-V2-Lite as non-uniform and unpriced (`INDEX.md` lines 556–560).
- Measured quality/rate: none.
- Evidence class: config-derived only.
- Blocking gaps: MLA/MoE format support, weights, quality, rate.
- References: `configs/deepseek-ai__DeepSeek-V2-Lite.json` JSON root and lines 17, 19, 22, 29–32, 51, 58; `configs/_manifest.json`; `INDEX.md` lines 556–560.

### mistralai/Mixtral-8x7B-v0.1

- Exact model/revision: repo id `mistralai/Mixtral-8x7B-v0.1`; manifest revision `fc7ac94680e38d7348cfa806e51218e6273104b0`.
- Config-derived shape: `D=4096`, `F=14336`, `L=32`, `H=32`, `KV=8`, `V=32000`, `tied=false`, `num_local_experts=8`, `num_experts_per_tok=2`, `MixtralForCausalLM`; config lines 1–28, key lines 9, 11, 13–18, 24, 28.  Analyzer total/manifest 46.702792704B; active/token 12.748853248B.
- Local status: config `OK`; no pretrained weights evidenced.
- Exporter/engine compatibility: MoE/non-uniform; no candidate export/load.  Dense fit excludes Mixtral (`INDEX.md` lines 556–560).
- Measured quality/rate: none.
- Evidence class: config-derived only.
- Blocking gaps: MoE exporter/engine support, weights, quality, rate.
- References: `configs/mistralai__Mixtral-8x7B-v0.1.json` JSON root and lines 9, 11, 13–18, 24, 28; `configs/_manifest.json`; `INDEX.md` lines 556–560.

### ibm-granite/granite-4.0-h-small

- Exact model/revision: repo id `ibm-granite/granite-4.0-h-small`; manifest revision `b8c0982bab7fde4eb48110f5a069527c008fab39`.
- Config-derived shape: `D=4096`, `F=768`, `L=40`, `H=32`, `KV=8`, `V=100352`, `tied=true`, `GraniteMoeHybridForCausalLM`; 72 local experts, top-k 10, shared intermediate 1536; layer types contain 36 mamba and 4 attention entries; config lines 1–88, key lines 2, 12, 14–15, 68–74, 84, 88.  Analyzer total 32.207033856B / manifest 32.207337984B; active/token 8.802817536B.
- Local status: config `OK`; no pretrained weights evidenced.
- Exporter/engine compatibility: hybrid Mamba/MoE; no uniform dense path, no candidate export/load, and no dense-fit price (`INDEX.md` lines 556–560).
- Measured quality/rate: none.
- Evidence class: config-derived only.
- Blocking gaps: hybrid/MoE exporter/engine support, weights, quality, rate.
- References: `configs/ibm-granite__granite-4.0-h-small.json` JSON root and lines 2, 12, 14–15, 68–74, 84, 88; `configs/_manifest.json`; `INDEX.md` lines 556–560.

### openai/gpt-oss-20b

- Exact model/revision: repo id `openai/gpt-oss-20b`; manifest revision `6cee5e81ee83917806bbde320786a8fb61efebee`.
- Config-derived shape: `D=2880`, `F=2880`, `L=24`, `H=64`, `KV=8`, `head_dim=64`, `V=201088`, `tied=false`, `GptOssForCausalLM`; 32 local experts, top-k 4, alternating sliding/full attention; config lines 1–75, key lines 2, 9, 11, 15, 42–47, 72, 75.  Analyzer total 20.908050240B / manifest 21.511953984B; active/token 3.607406400B.  The analyzer records a total cross-check difference because the config is pre-quantized MXFP4 (`quantization_config` in config lines 51–66).
- Local status: config `OK`; no pretrained weights evidenced.  The manifest total is storage metadata for the shipped quantized artifact, not interchangeable with the analyzer’s logical formula.
- Exporter/engine compatibility: MoE plus mixed sliding/full attention and MXFP4 quantization; no exact export/load path.  Dense-fit pricing explicitly excludes gpt-oss-20b (`INDEX.md` lines 556–560); the old 105% table value is analogy/derived only (`INDEX.md` lines 576–584).
- Measured quality/rate: none candidate-specific; old 105% is not a measured candidate result.
- Evidence class: config-derived; quantized-storage cross-check; old budget figure derived/analogy.
- Blocking gaps: MXFP4/MoE/mixed-attention exporter/engine support, weights, quality, rate.
- References: `configs/openai__gpt-oss-20b.json` JSON root and lines 2, 9, 11, 15, 42–47, 51–66, 72, 75; `configs/_manifest.json`; `INDEX.md` lines 556–560, 576–584.

### Qwen/Qwen3-Next-80B-A3B-Instruct

- Exact model/revision: repo id `Qwen/Qwen3-Next-80B-A3B-Instruct`; manifest revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`.
- Config-derived shape: `D=2048`, `F=5120`, `L=48`, `H=16`, `KV=2`, `head_dim=256`, `V=151936`, `tied=false`, `Qwen3NextForCausalLM`; 512 experts, top-k 10, 12 full-attention layers and 36 linear-attention layers (`full_attention_interval=4`), linear key/value heads 16/32; config lines 1–43, key lines 2, 9–12, 14, 17–18, 22–29, 37, 42.  Analyzer total 79.573616640B / manifest 81.324862720B; active/token 3.462989824B; analyzer status `FORMULA_DISAGREEMENT`.
- Local status: config `OK`; E7 says this is a 1MB config stub, not weights (`E7` lines 24–30).
- Exporter/engine compatibility: linear attention plus MoE is outside the uniform dense path; no exact export/load run.  It is explicitly unpriced in the project’s non-uniform list (`INDEX.md` lines 556–560).
- Measured quality/rate: none.
- Evidence class: config-derived only; formula-disagreement preserved.
- Blocking gaps: linear-attention/MoE exporter and engine support, weights, quality, rate, and resolve analyzer disagreement.
- References: `configs/Qwen__Qwen3-Next-80B-A3B-Instruct.json` JSON root and lines 2, 9–12, 14, 17–18, 22–29, 37, 42; `configs/_manifest.json`; `probes/E7_REAL_LARGE_DONOR.md` lines 24–30; `INDEX.md` lines 556–560.

## Synthetic target-shape controls (not pretrained donors)

`T10` is a synthetic 10.603B shape file with noise weights, not a pretrained candidate.  E3 says timing uses shape and format only and that the files are written by `synth_export.py` with noise weights and no BPB (`probes/E3_ENGINE_AT_TARGET_SCALE.md` §1.1, lines 85–88, 248).  Its measured 3.090 tok/s at context 300 and 2.960 at 800, plus 32.8G weights/s, are therefore synthetic engine evidence only (`E3` lines 486–500).  E36 likewise labels the probe speed-only and reports `T10` 10.74B as a control (`probes/E36_TEN_BILLION_AT_FIFTY.md` lines 1–4, 42–52).  These rows must not be counted as a real 10B donor or as quality evidence for any candidate above.

## Cross-candidate evidence and unknowns

- The only tracked pretrained ≥7B weight evidence is Qwen2.5-Coder-7B.  The config manifest’s `OK` status means the small `config.json` metadata fetch succeeded; it does not mean weights are local.  This distinction is explicit in `donor_inventory.py` lines 284–352 and E7 lines 24–30.
- Dense candidate pricing exists only as a derived engine-fit table for Qwen3-8B, OLMo-2-7B, and Mistral-7B-v0.3.  The project explicitly excludes MoE and hybrid/SSM candidates from that fit (`INDEX.md` lines 535–560).
- No model revision is recorded for Qwen2.5-Coder-7B in the cited E7 record.  No candidate-specific exporter/engine run is recorded for the metadata-only rows.  No candidate-specific quality result exists for those rows.  No external/literature metadata was consulted.
- Unknown for every metadata-only row: actual local weight presence, weight-file revision correspondence, tokenizer artifact identity, exporter success on that exact weight set, engine load success, measured quality, and measured rate.
- Unknown for the hybrid/linear-attention rows: whether the project’s intended future format will represent their recurrent state, mixed layer schedule, and attention-specific tensors; the config alone does not answer this.
- Unknown for MoE rows: whether “active/token” as computed by the project’s analyzer matches the intended routing semantics for a future converted artifact; the values above are analyzer outputs, not claims about an implementation that does not exist.
- Unknown for `gpt-oss-20b`: a logical-vs-packed parameter reconciliation beyond the analyzer’s recorded MXFP4 cross-check warning.

## Explicit no-duplication warnings

- Do not count `Qwen/Qwen3-30B-A3B` and `Qwen/Qwen3-Next-80B-A3B-Instruct` as one model: they are distinct repo ids, revisions, architectures, and config rows.
- Do not count `Qwen/Qwen2.5-Coder-7B` as a second Qwen2.5-1.5B or as a config-manifest row.  It is the separately documented real 7B donor; the 1.5B controls are below this inventory’s size scope.
- Do not count `T10`, `S05`, `M7`, `Q8`, or other shape labels as pretrained donor candidates.  They are synthetic shape/control arms; `E3` explicitly says the weights are noise.
- Do not count old table percentages for `gpt-oss-20b` or `Nemotron-H-8B` as measured rates.  The project labels those values analogy/derived because their architectures were not priced by the dense fit (`INDEX.md` lines 556–560).
- Do not duplicate a candidate once by manifest parameter total and once by analyzer active/token total.  Those are two fields of one row; active/token is not another model.
- Do not interpret similarly named Mistral rows as duplicates: `mistralai/Mistral-7B-v0.3` and `mistralai/Mixtral-8x7B-v0.1` are separate dense and MoE repo ids.

## Files inspected

Project instruction: `AGENTS.md`.

Required graph context: `graphify-out/graph.json` query output for `real pretrained 10B donor candidates model shape vocabulary head GQA export engine compatibility`.

Core inventory/config sources: `benchmarks/donor_adaptation/donor_inventory.py`; `benchmarks/donor_adaptation/configs/_manifest.json`; all candidate config JSON files named in the rows above.

Research/audit sources: `docs/research/donor_adaptation/INDEX.md`; `docs/research/donor_adaptation/audits/AXIS_COVERAGE_AND_NO_DUPLICATION_MAP.md`; `docs/research/donor_adaptation/audits/H2I_ENGINE_EXPORT_GAP_AUDIT.md`; `docs/research/donor_adaptation/probes/E3_ENGINE_AT_TARGET_SCALE.md`; `E7_REAL_LARGE_DONOR.md`; `E15_DOES_THE_7B_PREDICT.md`; `E36_TEN_BILLION_AT_FIFTY.md`; `E39_THE_SAME_TEN_BILLION_PUT_SOMEWHERE_ELSE.md`; `E66_ONE_BYTE_AT_SEVEN_BILLION.md`.

Relevant engine/results sources sampled for compatibility and evidence: `benchmarks/donor_adaptation/engine/donor_engine.c`; `engine/e3_analyse.py`; `engine/e36_ten_billion.py`; `engine/e39_same_ten_billion.py`; `engine/e43_vocabulary.py`; `engine/e66_one_byte_at_7b.py`; `engine/results/e7/parity.json`; `engine/results/e7/generate.json`; `engine/results/e66/e66_one_byte_at_7b_run2.json`; `benchmarks/donor_adaptation/results/e3/budget_ctx300.txt`; `benchmarks/donor_adaptation/results/e3/budget_ctx800.txt`.
