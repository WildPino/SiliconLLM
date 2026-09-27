# METH-59: stored Instruct R8 core fails prompt ranking with product-key E128

**Decision:** the fixed R8 row quantizer produces a 496,122,304-byte
packed-only Instruct core and passes the reused external document-loss gate,
but its actual composition with the METH-56 E128 adapter fails the frozen
≥95% prompt top-1 gate at **3,887/4,125 = 94.230%** versus the BF16
Instruct donor. The run stopped before generation and PIQA as registered.
This format is not promoted to native integration.

The [protocol](METH_59_INSTRUCT_R8_CORE_PROTOCOL_20260927.md) binds the
Instruct source, METH-56 checkpoint and METH-57 manifest. The
[exporter](../../../benchmarks/donor_adaptation/s1/meth59_export_instruct_r8_core.py)
applies the METH-24/T2 R8 per-row rule to the tied head, FFN and attention
matrices and preserves FP32 controls. The stored artifact at
`results/native_expert_scaling/meth59_qwen05b_instruct_r8_core.safetensors`
has SHA-256
`5b6ace4a027d127a6eb49ca99197edd4a1b810ebe681d8e3b6ab7edc088ca9bd`.
Its 169 matrices, 121 controls and metadata passed exact stored readback;
the [export ledger](meth59_instruct_r8_core_export.json) records organ errors
and 15.266 s local RTX 3060 export time, 2.863 GB peak allocated GPU memory
and 3.591 GB end RSS. The large packed core remains a generated local file,
not a Git object.

The [composition evaluator](../../../benchmarks/donor_adaptation/s1/meth59_instruct_r8_composition.py)
read the stored codes/scales and reconstructed BF16 effective matrices.
Its [paired result](meth59_instruct_r8_composition_result.json), SHA-256
`53fe562172952113f82be9bd19eb2d76ac69d69e5b3a70f8dab0c8a79ad69bc9`,
uses METH-57's bound donor rows. This is a **viewed-set diagnostic**, not
an independent quality audit.

| Metric | All-R8 donor | All-R8 + product-key E128 |
|---|---:|---:|
| Pooled ΔBPB vs BF16 donor | +0.000192 | **−0.005086** |
| Code / prose / technical student ΔBPB | — | −0.002984 / −0.007754 / −0.004521 |
| Prompt top-1 versus BF16 donor | 3,888/4,125 = 94.255% | **3,887/4,125 = 94.230%** |

The document gate passes, while pooled prompt top-1 fails. The similar
R8-donor and R8-student ranking losses point toward core quantization;
they do not prove which organ causes the errors. The diagnostic took
49.344 s, peaked at 2.443 GB allocated GPU memory and ended at 3.389 GB
RSS. Generation, PIQA, semantic quality, C execution and accepted-token
rate were not evaluated after the early stop.

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth59_export_instruct_r8_core.py --out results/native_expert_scaling/meth59_qwen05b_instruct_r8_core.safetensors --report docs/research/NATIVE_EXPERT_SCALING_20260925/meth59_instruct_r8_core_export.json
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth59_instruct_r8_composition.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth59_instruct_r8_composition_result.json
```

The evaluator's first execution stopped at the same gate but lacked a
terminal resource ledger; a bookkeeping-only change was committed at
`a248121`, then the bound run was repeated to retain that ledger.
METH-60 isolates which organ's precision changes top-1 and payload.
