# METH-04: GigaChat organ-selective low-bit preflight result

**Decision:** the frozen [precision map](METH_04_GIGACHAT_LOWBITS_PROTOCOL_20260925.md)
passes its *descriptor traffic* gate at **534.025 MB/token**, 25.975 MB
below the 560 MB design allotment. It is a candidate conversion map, not
converted weights or a quality/speed result. The selected IQ2_XS type
requires a donor-specific importance matrix for 254 tensors; none is
currently bound. Actual quantization and paired quality must come next.

## Identity, method and cost

The [planner](../../../benchmarks/native_expert_scaling/plan_gigachat_lowbit.py)
stream-hashed the exact BF16 GigaChat source GGUF, 21,356,264,448 B,
SHA-256 `fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`.
It verified the 414-tensor DeepSeek2/GigaChat 26-layer inventory, 64 routed
experts and top-4, then calculated each target tensor's bytes from the
pinned llama.cpp GGUF block size/type size table. Each tensor's GGUF
innermost dimension must divide its target block size. The source header
offset plus its tensor bytes reconciles exactly with the file size. The
planner emitted an anchored, explicit
[414-tensor override file](meth04_gigachat_tensor_types.txt) and the
[machine-readable ledger](meth04_gigachat_lowbit_preflight.json).

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\plan_gigachat_lowbit.py --model benchmarks\donor_adaptation\density\results\strat01_gigachat_source_binding_v1\GigaChat3.1-10B-A1.8B-source-bf16.gguf --gguf-py C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4\gguf-py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth04_gigachat_lowbit_preflight.json --tensor-types-out docs\research\NATIVE_EXPERT_SCALING_20260925\meth04_gigachat_tensor_types.txt
```

The read-only command took about 3–4 minutes, dominated by hashing the
21.36 GB source. No new model file, GPU job or T4 run. The pinned local
llama.cpp quantizer source supports `--tensor-type-file`; its source also
explicitly requires an importance matrix for IQ2_XS non-head tensors.
Neither a quantizer executable nor a suitable matrix is currently bound
for this cell. The override file is prepared for a later, separately gated
conversion from BF16, never requantization of the existing Q4 control.

## Exact addressed-payload result

| Organ | Target type | Planned MB/token |
|---|---|---:|
| MLA, 104 tensors | IQ2_XS | 187.906 |
| MLA `attn_k_b`, 26 tensors | Q4_0 | 14.909 |
| Routed experts, 75 tensors | IQ2_XS | 170.496 |
| Shared experts, 75 tensors | IQ2_XS | 42.624 |
| Dense first FFN, 3 tensors | Q4_K | 23.224 |
| Untied output head | Q3_K | 84.649 |
| Router, norms/biases, one embedding row | F32 controls, Q4_K embedding | 10.217 |
| **Total** | | **534.025** |

The 26 `attn_k_b` tensors have a GGUF innermost dimension of 128, so a
256-element IQ2_XS block is invalid there; Q4_0's 32-element block is
compatible. The planned stored tensor payload is **3,202,277,120 B**,
excluding GGUF header/alignment. This is distinct from the active payload.
The quality-valid producer Q4 control addresses **1,016,194,144 B/token**
([METH-01](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md)), so the planned
reduction is 482.169 MB/token, about 47.4%. The bound BF16 input itself
addresses 3,261.460 MB/token; it is the source of the proposed quantization,
not the performance control.

At the 40 GB/s streaming yardstick, the planned payload alone costs
**13.351 ms/token**. A 50 tok/s end-to-end budget is 20 ms; only about
6.649 ms remain for decode, routing, kernels, cache/KV/state traffic and
other work. This is a favorable addressed-byte model, not measured DRAM
traffic or an `engine.c` rate. The 25.975 MB margin to the 14 ms streaming
allotment is thin. A converter can also choose fallbacks that change actual
types; the final GGUF must be inspected again against this ledger before
any performance claim.

## Next transfer gate

The next step is to bind a representative, non-held-out donor calibration
corpus and produce an importance matrix with the pinned runtime, with an
explicit wall-time/RAM budget. Then quantize the **BF16** source using the
override file, verify all 414 output tensor types and actual active bytes,
and score the same source BF16/Q4 and new format on paired held-out BPB,
generation and tasks. Freeze quality thresholds and a stop rule before
running the conversion. If step-zero quality is inadequate, only a costed
adaptation/distillation path can rescue this map; a cheaper file alone is
not a transferred model. Supporting these donor operators in `engine.c`
and proving ≥50 accepted tok/s on that same artifact remain separate gates.

An existing candidate is `benchmarks/donor_adaptation/density/corpus/
strat01_gigachat_fresh_v2/calib.jsonl`, SHA-256
`68d9327a823328b1104c843afe986efeefcdd88c911c02895d577f65bffd8f81`:
48 documents and 196,560 source-span bytes, 16 each in prose, technical
text and code. Its 96-document held-out counterpart shares no item or
source-content SHA-256 with calibration. A read-only character census found
**zero Cyrillic characters** in calibration, so using it alone would support
at most an initial pilot on those domains, not broad donor-language quality.
The source-bound Q4 GGUF could collect an importance matrix with lower RAM
than BF16, but its activations are an approximation; the calibration model,
dataset coverage and resource ceiling must be frozen before that run.

## Pinned quantizer dry run (September 25)

The pinned `llama.cpp` commit `5b335f413` was built locally with
`cmake --build .../build-cpu --target llama-quantize -j 6`; binary SHA-256
`71f518bc3323e2e887374f8d59adf197dc7e321b529eeb6d6cf2c1f54d65c260`.
The command, run from the repository root, was:

```powershell
& C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4\build-cpu\bin\llama-quantize.exe `
  --dry-run --tensor-type-file docs\research\NATIVE_EXPERT_SCALING_20260925\meth04_gigachat_tensor_types.txt `
  benchmarks\donor_adaptation\density\results\strat01_gigachat_source_binding_v1\GigaChat3.1-10B-A1.8B-source-bf16.gguf `
  results\native_expert_scaling\meth04_gigachat_lowbit_dryrun.gguf IQ2_XS 6
```

The [complete stdout/stderr log](meth04_gigachat_quantize_dryrun.log),
normalized to LF with trailing spaces removed, SHA-256
`92e3d03a21a16b4b49560df8c5887b39af2a967a154e30293593644688cf0bab`, records exit 0,
414/414 tensors, final types **129 F32, 254 IQ2_XS, 26 Q4_0, 4 Q4_K,
1 Q3_K**, and `3053.93 MiB` projected quantized payload. No tensor fallback
warning appears. The value agrees with the planner's 3,202,277,120 tensor
bytes to the log's two-decimal MiB precision. The quantizer explicitly warns
that an importance matrix is required for the actual conversion. `--dry-run`
produced no output GGUF and performed no quality or performance test.

This closes the uncertainty about whether the pinned quantizer would silently
change the type map *at dry-run metadata stage*. It does not verify the real
per-tensor imatrix coverage or numerical conversion. Before writing weights,
freeze a non-held-out calibration corpus with relevant language coverage,
source-tokenizer handling or a measured approximation error, the imatrix
runtime/RAM budget, and the paired BF16/Q4/mixed-format BPB, PIQA and rollout
stop gates. After conversion, compare its actual tensor types/bytes with this
map and reject any fallback or missing imatrix entry before scoring.
