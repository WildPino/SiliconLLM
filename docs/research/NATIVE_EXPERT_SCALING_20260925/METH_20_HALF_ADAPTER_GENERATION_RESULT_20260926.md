# METH-20 result: exported adapter preserves relative repetition, both arms loop often

**Decision:** the exact factor-0.50 adapter passes the prospective
*relative* generation screen, but neither arm shows convincing
absolute free-generation quality on these document continuations.
Proceed to a labelled task test; do not treat this screen as a
usefulness pass. The [frozen protocol](METH_20_HALF_ADAPTER_GENERATION_PROTOCOL_20260926.md),
[prompt manifest](meth20_half_adapter_generation_manifest.json),
[raw paired IDs and decoded text](meth20_half_adapter_generation_result.json),
and [runner](../../../benchmarks/donor_adaptation/s1/meth20_half_adapter_generation.py)
bind the run.

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth20_half_adapter_generation.py --manifest docs/research/NATIVE_EXPERT_SCALING_20260925/meth20_half_adapter_generation_manifest.json --manifest-sha256 0e8641615a3e29fba7d9a04eb568268719cc4ff036fac1f0b7ed5cf7d5240284 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth20_half_adapter_generation_result.json
```

The scorer reconstructed its eight-code/eight-prose/eight-technical
selection from METH-19's bound documents, verified the donor and
tokenizer, and loaded **all 72 tensors directly from** the local
adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
Each arm used the same 256-token prefix and greedy generation of up
to 128 new tokens. The run took **299.360 s** on RTX 3060, peaked
at **1.256 GB allocated GPU memory**, and ended at **2.926 GB RSS**.

| Category | Prompts | Donor triple-8gram loops | Half adapter loops | First-token agreement |
|---|---:|---:|---:|---:|
| Code | 8 | 6 | **7** | 8/8 |
| Prose | 8 | 6 | **6** | 7/8 |
| General technical | 8 | 5 | **3** | 7/8 |
| **Pooled** | **24** | **17** | **16** | **22/24** |

Neither arm produced a premature non-EOS output under 16 tokens.
The donor generated 3,040 tokens and ended one continuation by EOS;
the adapter generated all 3,072 possible tokens and no EOS. Every
category stays within the frozen allowance of one extra repeated
8-gram failure, so the relative gate passes. One code and one prose
prompt gain a loop under the adapter; another prose and two technical
prompts lose a loop. The mean distinct-bigram fraction is 0.4204
donor versus 0.4560 adapter, but this coarse diversity measure is
not a correctness or fluency judgement.

These prompts begin at deterministic spans within source documents,
not user tasks. The high absolute loop counts (**16/24** for the
candidate, **7/8** on code) and saved examples prevent a claim of
useful open-ended generation. The result does verify that the
persistent adapter file can be loaded and run with the pinned donor;
it neither proves C parity nor fixes the dense-core active cost.
