# Recipient core triage after the complete int8 cost failure

15 October2026. Analytical comparison only. Qwen's actual test is in the
[baseline result](RECIPIENT_INT8_BASELINE_RESULT_20261015.md). Other model
configurations/cards were inspected as public primary sources, not acquired
as weights or admitted to the engine. Mutable URLs below must be pinned before
any selected conversion. No new source download or model execution here.

## Accounted source work

Each model has biasless SwiGLU and tied source embeddings. Count all query,
key, value, output, gate, up, down and full deployment-head matrices. W8 uses
one byte/coefficient plus four bytes/output-row. W4G32 uses18 bytes/32 input
coefficients including the FP16 group scale; attention/head stay W8. The source
embedding is a separate BF16 input lookup in these scenarios, not a full scan.
No router is present. These are proposed formats, not source quality evidence.

| Recipient | L / D / FFN | Q / KV heads | Vocab | All-W8 matrix MB/token | W4-FFN/W8-rest MB/token | Distinct F32 KV MB at2048 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 24 /896 /4864 | 14 /2 | 151936 | 495.785 | 357.484 | 50.332 |
| SmolLM2-360M-Instruct | 32 /960 /2560 | 15 /5 | 49152 | 363.061 | 259.064 | 167.772 |
| SmolLM2-135M-Instruct | 30 /576 /1536 | 9 /3 | 49152 | 135.299 | 100.024 | 94.372 |

The corresponding naive K/V read accounting per query head is352.322,
503.316 and283.116MB at2048. It includes repeated access to shared KV heads,
so it is neither distinct payload nor measured DRAM. Attention math, cache
reuse, norms, nonlinearities, scales, lookup/fusion and frontend still need
actual complete timing. In particular,360M's matrix savings do not guarantee
long-context savings. Retain full context in comparisons.

Primary configurations:
[135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/config.json),
[360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/config.json).
Both are conventional Llama-style, biasless attention/SwiGLU models with
RoPE theta100000 and RMS epsilon1e-5, unlike Qwen's q/k/v biases and constants.
Existing dense C operators are reusable; dimensions, serialization, tokenizer,
EOS and source parity still require explicit adaptation.

## Quality and training tradeoff

The [135M card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
reports MMLU-cloze29.3 and GSM8K5-shot1.4. A cheap initial core can therefore
leave a very large knowledge/reasoning gap. The
[360M card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
does not establish dominance over Qwen on chat/math. Source-reported benchmark
numbers are not our own donor-relative evaluation or a selection gate.

Conventional16-byte/parameter mixed-precision Adam state would be roughly
2.15GB for135M and5.79GB for360M before activations. That is compatible with
investigating joint recovery on a16GiB T4, not proof of a workable schedule.
Two frozen3M-row2048-wide BF16 donor memories add24.576GB host/file values,
not T4-resident trainable state; cloud host capacity/transfer must be measured.

## Next action

Do not select a weak small endpoint simply because its arithmetic fits. The
goal remains useful capacity transferred from a very-large donor into an
efficient conditional artifact. The recipient is only its online core.

The unchanged Qwen all-W8 profile is closed. Its worst measured28.02ms would
need more than1.55x improvement to meet the18ms working allowance. The prepared
mixed W4/integer profile reduces matrix payload only1.387x, so any adequate
improvement must also come from actual kernel/compute changes, not a byte-ratio
promise. It is the next bounded physical comparison using already retained
source weights, before acquiring a different recipient. If it fails the full
envelope,135M is a cost-motivated alternative for a donor-memory **transfer
falsifier**, with explicit absolute-usefulness and donor-gap gates; it is not
an admitted final solution. No new thread grid or long recovery follows from
these metadata alone.
