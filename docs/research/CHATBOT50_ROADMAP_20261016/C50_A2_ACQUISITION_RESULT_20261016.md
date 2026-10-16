# C50-A2 result: Ling-mini-2.0 source, BF16 GGUF, FIT imatrix and the Q4_K_M P0 artifact

16 October 2026. Cell A2 of the [CHATBOT50 roadmap](ROADMAP.md) §5, run under the
[A2/A2b protocol](C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md) and its addenda 1-2.
A2 has no gate. It records what the later cells consume. The only stop condition
was a vocabulary mismatch, and it did not occur.

## Chain of artifacts

| Step | Artifact | Identity | Record |
| --- | --- | --- | --- |
| Source | `inclusionAI/Ling-mini-2.0` @ `a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab`, 14 files | every file hash-verified, verdict PASS | `benchmarks/chatbot50/results/c50_a2_ling_mini_source_manifest.json` |
| BF16 GGUF | `ling-mini-2.0-a810f64-bf16.gguf`, 32,537,895,808 B, 278 tensors | sha256 `2b1c8f72ff25d152…` | `benchmarks/chatbot50/results/c50_a2_gguf_check_bf16.json` |
| imatrix input | FIT spans 1-24 of each category (72 spans, addendum 1), 161,790 source tokens | sha256 `e290c7a490e5fba8…` | protocol addendum 1 |
| imatrix | `ling-mini-2.0-a810f64-imatrix-fit24.gguf`, 90,609,856 B, 176 entries | sha256 `77d9a409508c8e82a9402a2ccac637861d9653f46289ee8a81778ec2fa86aee5` | `gguf/imatrix_fit24.log` (campaign directory) |
| P0 artifact | `ling-mini-2.0-a810f64-q4_k_m.gguf`, 9,912,067,296 B, 278 tensors, file type 15 (Q4_K_M) | sha256 `44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4` | `benchmarks/chatbot50/results/c50_a2_gguf_check_q4_k_m.json`, `gguf/quantize_q4_k_m.log` |

All GGUF files are in
`E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf/`, outside the repository.
The tools are llama.cpp `5b335f4` (`build-avx2`, WinLibs GCC 15.2.0).

## Measurements

**Vocabulary identity (the stop condition).** The Q4_K_M GGUF matches
`tokenizer.json` on all 157,153 source IDs: zero mismatches. The 31 GGUF-only IDs
(157,184 rows) are padding that the source tokenizer never emits. The BF16 GGUF
gives the same result. The runtimes can therefore be fed source-tokenizer IDs.

**imatrix run.**
- Settings: `llama-imatrix`, BF16 model, 6 threads, `-c 512`; 302 chunks of
  512 tokens (`n_seq=4`, batch 2048).
- Projection after the first pass: 44.17 s per pass, ETA 55.6 min. This was
  under the one-hour rule, so the run went ahead.
- Actual wall time: 40 min 34 s, exit 0.
- Coverage: the final save warned of partial data in two layers. In layer 10,
  255 of 256 experts received activations (99.61%); in layer 13, 254 of 256
  (99.22%). Every other expert tensor was covered. This was 302 chunks with
  top-8 of 256 experts per token, so a few experts saw no token in the sample.
  `llama-quantize` gives an expert with a zero count a uniform importance of 1
  (`tools/quantize/quantize.cpp:210-213` at the pin). Those three experts are
  therefore quantized without imatrix weighting, as a no-imatrix Q4_K_M would
  be. Their quality effect is not measured separately; it falls inside A4's
  P0-vs-BF16 measurement.
- Perplexity: llama-imatrix printed PPL 8.0099 ± 0.102 on the FIT sample (BF16,
  context 512). This is a by-product of the tool, on FIT text, with llama.cpp's
  tokenization. It is not a quality number and enters no decision.

**Q4_K_M quantization.**
- Command: `llama-quantize --imatrix <imatrix-fit24> <bf16> <q4_k_m> Q4_K_M 6`
  (the trailing 6 is the thread count). Default options otherwise.
- The tool loaded 176 imatrix entries. It reported no imatrix for
  `token_embd.weight` and `output.weight`, which is expected because neither is
  a matrix product input that the imatrix observes.
- Run: 532.96 s, exit 0. Size: 31,024.36 MiB (16.01 bits per weight) →
  9,446.69 MiB (4.87 bits per weight).

**Tensor-type mix (read from the quantize log, counted per tensor):**

| Tensor | Q4_K | Q6_K |
| --- | ---: | ---: |
| `attn_qkv` | 10 layers | 10 layers |
| `attn_output` | 20 | 0 |
| `ffn_gate_exps`, `ffn_up_exps`, `ffn_gate_shexp`, `ffn_up_shexp` | 19 each | 0 |
| `ffn_down_exps`, `ffn_down_shexp` | 10 each | 9 each |
| dense layer 0 `ffn_gate`, `ffn_up` / `ffn_down` | 1 / 0 | 0 / 1 |
| `token_embd` | 1 | 0 |
| `output` (head) | 0 | 1 |

Norms, router (`ffn_gate_inp`) and expert bias stay F32. Bytes by type: Q4_K
7.552 GB, Q6_K 2.314 GB, F32 0.040 GB.

**Correction to a previous log row.** The ROADMAP §12 row for the A3 engine
said the Q4_K_M mix "needs Q5_K (`attn_qkv`)". The actual mix contains **no
Q5_K tensor**. llama.cpp `5b335f4` places the fused `attn_qkv` in Q6_K on 10
layers and Q4_K on the other 10. That row was a prediction from a reading of
`llama-quant.cpp`, made before the artifact existed, and it was wrong. The engine
dispatches on the type recorded in the GGUF, and Q4_K and Q6_K are both
self-tested, so A3 is unaffected. The Q5_K kernel stays tested but is unused
on P0.

## Per-token bytes against the roadmap ledger

The ledger (`benchmarks/chatbot50/results/byte_budget_ledger_20261016.txt`) priced P0
from headers at 742.2 MB of weights per token plus 83.9 MB of F16 KV at context
2048 (826.0 MB). The measured artifact, in MB per token:

| Organ | Ledger P0 | Artifact | Difference |
| --- | ---: | ---: | ---: |
| attention | 118.0 | 134.4 | +16.4 (half of `attn_qkv` in Q6_K) |
| dense FFN (layer 0) | 17.7 | 20.4 | +2.7 (`ffn_down` in Q6_K) |
| shared expert | 33.6 | 36.1 | +2.5 (9 `ffn_down_shexp` in Q6_K) |
| routed experts (8/256) | 269.0 | 288.4 | +19.4 (9 layers of `ffn_down_exps` in Q6_K) |
| router | 39.8 | 39.9 | 0 |
| head | 264.1 | 264.1 | 0 |
| embedding row, small tensors | — | 0.2 | +0.2 |
| **weights** | **742.2** | **783.4** | **+41.2 (+5.5%)** |

The ledger priced every matrix except the head at Q4_K. Producer Q4_K_M promotes
about half of `attn_qkv` and of `ffn_down` to Q6_K. With the KV, P0 streams
**867.3 MB per token at context 2048**, 5.0% above the ledger row.

- The policies P2-P4 inherit the same +41 MB unless A5 changes those organs'
  types. A5 has to price that explicitly.
- The ledger's "within 4%" check on the GigaChat producer GGUF was the same kind
  of measurement, and this one lands at 5%.

## What this result cannot say

- Nothing about quality: whether P0 is close to BF16 is A4's measurement.
- Nothing about speed: the byte count is an input to the desk model, not a rate.
- The imatrix sample is FIT text (24 per category). DEV and RESERVED were not
  read.
