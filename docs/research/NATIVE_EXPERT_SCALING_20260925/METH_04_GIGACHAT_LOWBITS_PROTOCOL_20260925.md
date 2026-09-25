# METH-04: GigaChat organ-selective low-bit preflight

**Status:** target map and decision gates frozen before descriptor result.
This is a donor-to-target *format* candidate and preflight, not a converted
model or native engine result. It responds to [METH-01](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md):
the quality-valid Q4 donor addresses 1,016.194 MB/token, above the 560 MB
streaming design allotment at 40 GB/s.

## One changed proposal

Use the source-bound **BF16** GigaChat 3.1 Lightning 10B-A1.8B base GGUF
(SHA-256 `fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`),
not a requantization of Q4. Map all tensor names explicitly:

| Organ | Proposed type | Reason / risk |
|---|---|---|
| MLA except `attn_k_b` | IQ2_XS | Largest shared active organ; high quality risk |
| MLA `attn_k_b` | Q4_0 | GGUF innermost dimension 128 cannot hold a 256-element IQ2 block |
| Routed and shared experts | IQ2_XS | Reduce conditional and always-active FFN traffic; high quality risk |
| Dense first FFN | Q4_K | Preserve more precision in the initial dense transformation |
| Untied output head | Q3_K | Reduce a large always-read organ with less severe precision than IQ2 |
| Embedding | Q4_K | Keep its stored footprint modest; only one row addressed/token |
| Router, norms and biases | F32 | Preserve decision/control precision |

The pinned llama.cpp quantizer supports `--tensor-type-file` and exact
per-tensor overrides. Its GGUF type table provides block sizes and bytes;
IQ2_XS requires a donor-specific importance matrix for these non-head
tensors. No such matrix is currently bound locally. This preflight must
therefore **not** attempt a quality claim or silently fabricate calibration.

The [planner](../../../benchmarks/native_expert_scaling/plan_gigachat_lowbit.py)
must verify the 414-tensor inventory, source bytes and SHA-256, GGUF tensor
block compatibility, exact organ-level active/stored bytes and a complete
override file. Advance to a bounded conversion/step-zero quality protocol
only if the target addresses **≤540 MB/token**, leaving ≥20 MB below the
560 MB design allotment, and every tensor can use its specified type. A
descriptor result that fails this gate requires a different map before any
quantization. This is a traffic gate only: even a pass does not establish
actual DRAM traffic, CPU rate or donor-relative quality.

The read-only plan costs one stream hash of the local 21.356 GB BF16 GGUF
and GGUF header/tensor arithmetic (under five minutes expected), no GPU/T4
or new weight file. If admitted, the next stage must budget importance-matrix
generation and quantization, freeze source/Q4/proposed format controls on
held-out BPB, generation and tasks, and stop on a quality loss that cannot
plausibly be recovered within stated resources. Native `engine.c` operator
support and accepted-token speed remain separate downstream gates.
