# METH-51: BF16-effective trained factors reload with exact logits

The [frozen protocol](METH_51_BF16_EFFECTIVE_FACTOR_PROTOCOL_20260927.md)
binds METH-47 update 512, its Instruct donor, and the previously viewed
12-prompt external manifest. The [exporter and verifier](../../../benchmarks/donor_adaptation/s1/meth51_bf16_effective_factors.py)
stores the exact BF16 values to which the existing forward casts its
selected fp32 factors. All 48 factor tensors reload bit-identically;
restoring them into the unchanged wrappers preserves their BF16-effective
values. Donor and fp32 router are unchanged.

| Fixed parity check | Result |
|---|---:|
| Prompt logits, all vocabulary entries | **317,698,176/317,698,176 elements bit-identical** across 2,091 positions and 12 prompts |
| Original greedy streams versus saved METH-47 | 12/12 exact |
| Reloaded greedy streams versus original | **12/12 exact** |

This is an exact representation of the factors **under the current BF16
PyTorch forward**. It does not repair unsupported semantic claims already
seen in METH-47, and it is not a complete native model. The local
[factor artifact](../../../results/native_expert_scaling/meth51_e128_effective_bf16.safetensors)
is 88,084,744 bytes, SHA-256
`53f2849da55f0da7f91f7d097ab3357308abfd85174cdf8a82b7e226cd738473`.
Its tensor payload is 688,128 bytes per expert across 24 layers. With
unchanged L24/D896/rank8, E27,355 projects to 18,823,741,440 factor
bytes and E273,547 to 188,235,350,016 factor bytes. These are
**shape projections**, excluding donor, router/index, metadata and
runtime workspace; no expanded bank was trained. At top-4, selected
factor payload is 2,752,512 bytes/token independent of total E.

The [machine result](meth51_bf16_effective_factor_result.json), SHA-256
`ac1e56623a2216d5eb8eee5a808ae3f7b6f54fee5114c5dd8fd25c8cd8624d86`,
contains each prompt's bit-parity and generation readback. Command:

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth51_bf16_effective_factors.py --artifact results/native_expert_scaling/meth51_e128_effective_bf16.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth51_bf16_effective_factor_result.json
```

The local RTX 3060 run took 87.547 s, peaked at 1.260 GB allocated GPU
memory and 4.872 GB RSS; no T4 was used.

**Decision:** retain this BF16-effective bank as the first exact
trained-factor export anchor for METH-47. Any smaller CPU LUT format must
be judged against its logits/generation, not only weight error or BPB.
The factor bank by itself has no CPU kernel, bounded large-E router,
source-grounded semantic pass or ≥50 accepted tok/s native measurement.
