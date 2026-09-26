# METH-19 result: fixed half-amplitude bank passes a new document gate

**Decision:** the factor-0.50 candidate passes the prospective
*document* gate on 56 newly selected documents. It may proceed to
generation/task testing and an exact, quality-preserving export
attempt. This is not a full transfer or native-speed pass. The
[protocol](METH_19_HALF_RESIDUAL_INDEPENDENT_PROTOCOL_20260926.md),
[manifest](meth19_half_residual_independent_manifest.json),
[per-document result](meth19_half_residual_independent_result.json),
and [runner](../../../benchmarks/donor_adaptation/s1/meth19_half_residual_independent_audit.py)
retain the bound inputs, scores and gates.

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth19_half_residual_independent_audit.py --manifest docs/research/NATIVE_EXPERT_SCALING_20260925/meth19_half_residual_independent_manifest.json --manifest-sha256 1eb4193e521df4d25d6367a84d8478d4ea733841fe9a77d550de01796ddf3d70 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth19_half_residual_independent_result.json
```

The run reconstructed the manifest from pre-adapter Git sources and
the bound external JSONL, verified the manifest SHA-256, source model,
tokenizer and METH-16 checkpoint hashes, then scored every token in
**56 documents / 229,316 UTF-8 bytes / 63,067 text tokens**. The
METH-17 selected source IDs and tested 256-byte fragments were
excluded, as were tested fragments from both Qwen corpus files. The
overlap screen remains partial and the external source was already
used in a separate sparse-donor research line. No METH-19 text was
used to choose factor 0.50 or train the adapter. The RTX 3060 run
took **89.266 s**, peaked at **2.549 GB allocated GPU memory**, and
ended at **3.189 GB process RSS**.

| Category | Docs / bytes | Donor BPB | Factor 0.50 BPB | 0.50 delta | Factor 1.00 delta |
|---|---:|---:|---:|---:|---:|
| Code | 24 / 98,280 | 0.995616 | 1.001183 | **+0.005567** | +0.030737 |
| Prose | 24 / 98,280 | 1.091711 | 1.088467 | −0.003244 | +0.005040 |
| General technical | 8 / 32,756 | 0.715441 | 0.703310 | −0.012131 | +0.002245 |
| **Pooled** | **56 / 229,316** | **0.996779** | **0.996042** | **−0.000737** | **+0.015654** |

The 20,000-draw category-stratified paired bootstrap's one-sided
95% upper bound for the half-minus-donor pooled delta is
**+0.000133 BPB**, under the frozen +0.02 bound. Pooled point delta
is under +0.01, code under +0.01, and prose/technical each under
+0.03. The trained factor-0.50 route beats the row-permuted null
by **+0.007235 pooled BPB**, above the required +0.002. All joint
document conditions pass. The factor-1.00 control again worsens
pooled BPB, consistent with the earlier amplitude diagnosis.

The code exception is important: **23/24 code documents still lose**
against donor at factor 0.50, though the pooled code loss is within
the preregistered limit; factor 1.00 loses on 24/24. Route utility
on code is only **+0.000759 BPB** versus +0.010649 on prose and
+0.016422 on technical. Thus the new result supports bounded
donor-relative document quality, not a claim that code transfer has
improved. A domain-sensitive repair remains an open research path.

The intact BF16 donor core still dominates active traffic; the
E128 router is exhaustive, and this adapter adds only 44M factors.
This run measures neither task usefulness, free generation, native
C fidelity/speed, nor growth to distinct E at 10B/100B. The
factor-0.50 arm was obtained by scaling checkpoint output factors
in memory. The [adapter exporter](../../../benchmarks/donor_adaptation/s1/meth19_export_half_adapter.py)
then wrote the same 72 factor/router tensors to local
`results/native_expert_scaling/meth19_half_residual_adapter.safetensors`:
**187,177,472 bytes**, SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
Its exact reload comparison passed for all 46,792,704 fp32 values.
This is an adapter-only artifact paired to the pinned BF16 donor, not
a complete low-cost model or an engine C export.
