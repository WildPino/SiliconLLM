# METH-36: stored one-byte router sketch on independent route inputs

METH-35's sole passing E128 arm uses an fp32 rank-64 SVD sketch
and 64 exact candidates on reused prompts. This cell tests whether
the same index survives one-byte row quantization and source-document
shift. Neither the rank nor candidate budget will be changed based
on this result. The flat trained router and factor weights stay fixed.

Bind the Qwen2.5-0.5B source SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
R8 core SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245b926fc3a6ee2572afa27`,
E128 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`,
METH-17 selection manifest SHA-256
`7e0593d6c56c28398e3440f9b80db15d506a31131a21480a87578c23b0a043a8`,
and METH-35 result SHA-256
`f701f1968987db0d107fe4383166837e07be985771754ee1bce36de6615710d7`.
The METH-17 documents were selected before METH-35 and are disjoint
from METH-25/27 by source ID. From each METH-17 category, sort source
IDs by SHA-256 of `meth36-36036|source_id` and take eight. Use the
first 256 tokenizer IDs as each prompt; record selected source IDs,
text/prompt hashes and the manifest hash **before** route scoring.
This yields 24 prompts, 6,144 positions and 147,456 input-layer cases.
The selected [prompt manifest](meth36_route_prompt_manifest.json) was
written before route scoring, SHA-256
`4bd2bcd666418bfc1b558a4ed6e57650dc31972b29f9a3f6671b99e573dad5ab`.

For each frozen fp32 fine router `W` compute its uncentered SVD as
in METH-35, take `V_64`, and store the basis fp32. Quantize each
row of `W V_64` independently: `scale=max(abs(row))/127`,
`code=round_to_even(row/scale)` clipped to `[-127,127]`, with
all-zero rows assigned all-zero codes and zero scale. Save signed
one-byte codes, fp32 row scales and fp32 bases in a safetensors
artifact; reload and verify every tensor before auditing.
The exported 5,720,160-byte artifact
`results/native_expert_scaling/meth36_e128_router_r64_int8.safetensors`
was reload-verified before the audit, SHA-256
`133287eb87cf993d9498fd18c1efafbf703bf10a29cc757a5a5ec73cadbb674d`.
The 64 exact candidates are rescored by the unchanged fp32 fine
router from the bound adapter. The fp32 rank-64 sketch is a paired
control on the same new inputs; the stored int8 sketch is the
candidate under decision.

Keep the model on its exact original top-4 path. At every layer,
compare candidate inclusion and exact-rescored top-4 to the
exhaustive fine router before any route replacement. Retain
per-layer/category counts, missed gate mass and stream hashes.
The *int8* arm passes only if pooled selected-ID inclusion ≥99.9%,
pooled complete-set match ≥99%, and each category's inclusion ≥99%.
Also report whether its full-set match drops by >0.5 percentage
points versus fp32 on these prompts; this paired difference is a
diagnostic, not an additional gate. A pass permits a route-replaced
language-quality test and native CPU kernel/timing work, not a
100B conclusion. A fail rejects this exact quantizer or rank/budget
combination.

Export on local CPU, audit on local RTX 3060. Each stage stops at
20 minutes wall time and 20 GiB RSS; the audit also stops at
10.5 GiB allocated GPU memory. No T4 job is planned.
