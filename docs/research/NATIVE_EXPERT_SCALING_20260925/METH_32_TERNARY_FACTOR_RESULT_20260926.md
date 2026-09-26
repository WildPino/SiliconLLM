# METH-32: real packed ternary factors fail the top-1 fidelity screen

The [prospective protocol](METH_32_TERNARY_FACTOR_PROTOCOL_20260926.md)
binds the stored Qwen2.5-0.5B R8 core and trained factor-0.50 E128
adapter, then applies a deterministic per-row squared-error-optimal
ternary transform to their rank-8 A/B factors. The router stays fp32.
The exported local
`results/native_expert_scaling/meth32_qwen05b_ternary_lut_adapter.safetensors`
is 77,179,624 bytes, SHA-256
`a99542407aea7f1ae9eb7ac0401f9406e7c0cb1516129955c427538950cf7eee`.
Its 120 tensors passed exact reload and code decode checks before
quality scoring. The [export ledger](meth32_ternary_factor_export.json),
SHA-256 `6f6522b7baaab6a86cb352b8296e3b0ee4853f88f3ed70d5e364c252f3a85474`,
records 55,050,240 LUT code bytes, 11,108,352 fp32 scale bytes
and 11,010,048 unchanged router bytes. Per-token selected factors
address 1,720,320 code plus 347,136 scale bytes at top-4/L24;
the exhaustive fp32 router is separate. Relative factor squared
error is **18.87% A** and **13.16% B**, not a language metric.

The intact R8+fp32 adapter reproduced all 48 METH-25 document
nats within 0.01 and all 24 top-1 streams exactly. The audit then
reloaded and decoded only the stored ternary artifact, keeping the
R8 core, router and scoring setup fixed. The METH-25 documents and
prompts were already observed, so these are diagnostic comparisons.

| Reused category | Intact BPB | Ternary BPB | Ternary minus intact | Ternary minus original BF16 donor |
|---|---:|---:|---:|---:|
| Code, 24 docs | 0.907641 | 0.905940 | −0.001701 | +0.003605 |
| Prose, 16 docs | 1.095427 | 1.095178 | −0.000249 | −0.002655 |
| Technical/general, 8 docs | 0.591005 | 0.593029 | +0.002024 | −0.022770 |
| Pooled, 48 docs | 0.917465 | 0.916869 | **−0.000596** | −0.002878 |

The category-stratified 20,000-draw one-sided 95% upper bound for
the ternary-minus-intact BPB penalty is −0.000227 on these reused
documents. Both the fixed BPB-penalty and donor-relative screens
pass. Yet next-token top-1 matches at only **5,899/6,144 = 96.012%**,
below the fixed ≥99% gate (245 mismatches). The unchanged fine
router follows changed hidden states: only **127,302/147,456 =
86.332%** of input-layer cases retain the same unordered top-4
expert set. This route divergence is a trajectory diagnostic, not
an approximate-router recall score.

The [full audit](meth32_ternary_factor_audit.json), SHA-256
`c378cf06e26934144c603fd9cd2ffe4b12e305fd6af272c5bb9da4bae33ad960`,
retains every document's paired nats, bootstrap and route counts.
Export command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth32_export_ternary_factors.py --out results/native_expert_scaling/meth32_qwen05b_ternary_lut_adapter.safetensors --report docs/research/NATIVE_EXPERT_SCALING_20260925/meth32_ternary_factor_export.json`.
Audit command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth32_ternary_factor_audit.py --artifact-sha256 a99542407aea7f1ae9eb7ac0401f9406e7c0cb1516129955c427538950cf7eee --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth32_ternary_factor_audit.json`.
Export took 2.203 s CPU and ended at 1.539 GB RSS. The audit took
48.641 s on the local RTX 3060, peaked at 2.439 GB allocated GPU
memory and ended at 3.455 GB RSS. These are PyTorch quality scores
after decoding the stored codes into factors, not native C timing.

**Decision:** this exact all-ternary A/B transform is rejected by
its top-1 fidelity gate despite favorable reused-document BPB.
The result does not imply that factor compression is impossible or
that a less exact top-1 match always harms generation. A next
diagnostic can isolate A versus B quantization with a predeclared
mixed-precision cost/fidelity tradeoff; a candidate must still pass
new independent documents, generation and tasks before native
promotion. METH-31's code-only synthetic LUT rate cannot be assigned
to this quality-failed artifact or to its scale/router overhead.
