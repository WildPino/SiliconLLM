# METH-15 result: exact-donor E128 residual experts train end to end

**Decision:** the 16-update local apparatus gate passes on the intended
RTX 3060. The [prospective protocol](METH_15_ZERO_RESIDUAL_EXPERT_SMOKE_PROTOCOL_20260926.md)
fixes donor, calibration/heldout IDs, architecture, optimizer and stops.
The [runner](../../../benchmarks/donor_adaptation/s1/meth15_zero_residual_expert_smoke.py)
and [RTX 3060 machine record](meth15_zero_residual_expert_smoke_rtx3060.json)
preserve the control readings. This is a short training-path result,
not a native export, final quality or large-E proof.

## Apparatus repair and exact run

The first invocation used `cuda:1` on an assumption about CUDA ordering.
PyTorch enumerated `cuda:1` as the **GTX 1660**, although `nvidia-smi`
listed the RTX 3060 second. Its [machine record](meth15_zero_residual_expert_smoke.json)
shows a clean 16-update run and −0.007892 BPB on the small diagnostic,
but its hardware binding violates the predeclared RTX 3060 condition.
It is retained as an **invalid hardware-gate run**, not counted as a
protocol pass. Its local checkpoint SHA-256 is
`72f8a94d3b79e5660a67c1b8ff01b5b65ffbb1f3dac96261ec96533bb42887da`.
The runner was repaired to select the exact device **name**; no data,
architecture, optimizer, threshold or seed changed.

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth15_zero_residual_expert_smoke.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth15_zero_residual_expert_smoke_rtx3060.json --checkpoint results/native_expert_scaling/meth15_zero_residual_expert_smoke_rtx3060.pt
```

The intended run verified the pinned Qwen2.5-0.5B source safetensors
SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`, 16M-token calibration array
SHA-256 `9bb5229fad6542aa8b3fd7edfc3b8d1dff965a573d7aa8379385caacc9d614da`,
and 4×256 heldout ID SHA-256
`c5345782fc0b08ef3212403878a33960ae0da236c999272e79e418f29c277d3c`.
All 24 donor FFNs and other base weights were frozen. Each layer
received 128 trainable rank-8 residual experts with top-4 routing.
The output factors began at zero, so active and disabled-student
logits agreed **exactly** at step zero on the checked first 16 positions.

| Control on intended RTX 3060 | Measured | Frozen bar |
|---|---:|---:|
| Applied updates | 16 | 16 |
| First-update minimum expert-output gradient across 24 layers | 0.000810 | >0 |
| Second-update minimum router gradient | 0.0000335 | >0 |
| Changed expert output slots, minimum across layers | 125/128 | ≥16 |
| Initial donor and student BPB, 4,588 bytes | 0.954563 / 0.954563 | identity |
| Terminal student BPB | 0.946438 | donor +≤0.10 |
| Terminal paired change | **−0.008125 BPB** | ≤+0.10 |
| Elapsed / GPU peak allocated / process RSS at exit | 32.688 s / 2.331 GB / 3.132 GB | ≤20 min / 10.5 GiB / 20 GiB |

The checkpoint with adapter factors, router, Adam and RNG is local at
`results/native_expert_scaling/meth15_zero_residual_expert_smoke_rtx3060.pt`,
SHA-256 `cff4d2cbe4851d736f76d7f0838c42752a9d7bceec83054c34239d413d9a727e`.
The heldout windows come from a corpus used earlier for method
selection, so the BPB improvement is a development signal only.
No free-generation, task, precision conversion or C timing was part
of this 16-update apparatus gate. METH-16 freezes a longer continuation
and joint quality checks before using this checkpoint further.
